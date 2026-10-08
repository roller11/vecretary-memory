# SPDX-License-Identifier: MPL-2.0
"""Prepare a disposable installed-package workspace; never configure the live client."""

import argparse
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SERVER_NAME = 'instinct_acceptance'


def write_workspace(root, executable, instructions, shared_writer=False):
    root = Path(root)
    executable = Path(executable)
    if not root.is_absolute() or not executable.is_absolute():
        raise ValueError('Workspace root and executable must be absolute paths.')
    workspace = root / 'workspace'
    (workspace / '.vscode').mkdir(parents=True)
    (workspace / '.github').mkdir()
    (workspace / '.codex').mkdir()
    data = root / 'data'
    server = {
        'type': 'stdio', 'command': str(executable),
        'args': ['--data-dir', str(data)],
        'env': {
            'INSTINCT_DATA_DIR': str(data), 'CE_DATA_DIR': str(data),
            'CE_EMBEDDING_PROVIDERS': 'onnx', 'PYTHONDONTWRITEBYTECODE': '1',
            'INSTINCT_ALLOW_REMOTE_PROVIDERS': '0',
            'INSTINCT_ALLOW_INSECURE_REMOTE_HTTP': '0',
        },
    }
    if shared_writer:
        server['args'].append('--shared-writer')
    (workspace / '.vscode' / 'mcp.json').write_text(
        json.dumps({'servers': {SERVER_NAME: server}}, indent=2) + '\n', encoding='utf-8',
    )
    (workspace / '.mcp.json').write_text(
        json.dumps({'mcpServers': {SERVER_NAME: server}}, indent=2) + '\n', encoding='utf-8',
    )
    codex = [
        f'[mcp_servers.{SERVER_NAME}]',
        f'command = {json.dumps(server["command"])}',
        f'args = {json.dumps(server["args"])}',
        f'cwd = {json.dumps(str(workspace))}',
        'startup_timeout_sec = 180', 'tool_timeout_sec = 180',
        f'\n[mcp_servers.{SERVER_NAME}.env]',
    ]
    codex.extend(f'{key} = {json.dumps(value)}' for key, value in server['env'].items())
    (workspace / '.codex' / 'config.toml').write_text('\n'.join(codex) + '\n', encoding='utf-8')
    scoped = (
        '# Isolated acceptance workspace\n\n'
        'Only use the MCP server named `instinct_acceptance` for these tests. '
        'If it is unavailable, report that; do not substitute another memory server. '
        'Only synthetic test notes may be written. Never access development or RC stores.\n\n'
        + instructions
    )
    for destination in (workspace / '.github' / 'copilot-instructions.md', workspace / 'AGENTS.md'):
        destination.write_text(scoped, encoding='utf-8')
    return workspace, server


def create_root(requested=None):
    if requested is None:
        return Path(tempfile.mkdtemp(prefix='instinct-client-acceptance-'))
    root = Path(requested)
    if not root.is_absolute():
        raise ValueError('Acceptance root must be an absolute path.')
    if not root.parent.is_dir() or root.parent.resolve() != root.parent:
        raise ValueError('Acceptance parent must exist and contain no symlink components.')
    root.mkdir(mode=0o700)
    return root


def preflight_server(root, server):
    data = str(Path(root) / 'preflight-data')
    return {
        **server, 'args': ['--data-dir', data, *(['--shared-writer'] if '--shared-writer' in server['args'] else [])],
        'env': {**server['env'], 'INSTINCT_DATA_DIR': data, 'CE_DATA_DIR': data},
    }


def result_text(result):
    return '\n'.join(getattr(item, 'text', '') for item in result.content)


def require(condition, label):
    if not condition:
        raise RuntimeError('Client preflight failed: ' + label)


async def preflight(root, server, templates):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    environment = os.environ.copy()
    environment.update(server['env'])
    environment.pop('PYTHONPATH', None)
    parameters = StdioServerParameters(
        command=server['command'], args=server['args'], env=environment, cwd=str(root),
    )
    with (root / 'preflight-mcp.log').open('w', encoding='utf-8') as log:
        async with stdio_client(parameters, errlog=log) as streams:
            async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=180)) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                required = {'instinct_get_context', 'instinct_get_mandatory', 'instinct_save_memory', 'instinct_get', 'instinct_delete_memory'}
                require(len(names) == 7 and required <= names, 'seven transport tools')
                mandatory = result_text(await session.call_tool('instinct_get_mandatory', {}))
                require(all(memory.concept_title in mandatory for memory in templates), 'starter titles')
                context = result_text(await session.call_tool('instinct_get_context', {
                    'query': 'isolated client acceptance with synthetic notes', 'include_mandatory': True,
                }))
                require(all(memory.concept_title in context for memory in templates), 'mandatory context')
                require('VALIDATION_TOKEN:' in context and 'RESPONSE_REQUIREMENT:' in context, 'token requirement')
                renewed = result_text(await session.call_tool('instinct_get_context', {
                    'query': 'synthetic new chat on a shared MCP connection',
                    'include_mandatory': True, 'new_session': True,
                }))
                require(all(memory.concept_title in renewed for memory in templates), 'explicit new-chat renewal')
                saved = result_text(await session.call_tool('instinct_save_memory', {
                    'title': 'Synthetic client preflight note',
                    'content': 'This disposable note verifies the isolated acceptance connection.',
                    'importance': 70, 'segment_type': 'context',
                }))
                match = re.search(r'^ID: (\S+)$', saved, re.MULTILINE)
                require('Saved memory segment' in saved and match is not None, 'committed save')
                segment_id = match.group(1)
                fetched = result_text(await session.call_tool('instinct_get', {'segment_id': segment_id}))
                require('Synthetic client preflight note' in fetched, 'persisted note')
                for arguments in (
                    {'importance': 100}, {'importance': 70, 'segment_type': 'standing_instruction'},
                ):
                    denied = await session.call_tool('instinct_save_memory', {
                        'title': 'Synthetic forbidden note', 'content': 'Must not persist.', **arguments,
                    })
                    denial = result_text(denied)
                    schema_detail = 'maximum' if arguments['importance'] == 100 else 'not one of'
                    expected_schema = 'Input validation error' in denial and schema_detail in denial
                    expected_policy = 'BLOCKED' in denial and 'owner-only' in denial
                    require(expected_schema or expected_policy, 'owner-only rejection')
                deleted = result_text(await session.call_tool('instinct_delete_memory', {'segment_id': segment_id}))
                require('Deleted segment' in deleted, 'test-note cleanup')
                missing = result_text(await session.call_tool('instinct_get', {'segment_id': segment_id}))
                require('No segment found' in missing, 'missing-record failure')
        async with stdio_client(parameters, errlog=log) as streams:
            async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=180)) as session:
                await session.initialize()
                renewed = result_text(await session.call_tool('instinct_get_context', {
                    'query': 'synthetic immediate reconnect without idle wait', 'include_mandatory': True,
                }))
                require(all(memory.concept_title in renewed for memory in templates), 'connection renewal')
    return {
        'transport': 'passed', 'starter_retrieval': 'passed', 'token_delivery': 'passed',
        'save_fetch_delete': 'passed', 'owner_only_guards': 'passed', 'missing_record': 'passed',
        'explicit_new_chat_renewal': 'passed', 'connection_renewal': 'passed',
        'model_compliance': 'not_tested', 'client_instruction_loading': 'not_tested',
        'client_failure_behavior': 'not_tested', 'real_idle_session': 'not_tested',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-starter-base', action='store_true', help='Approve synthetic bootstrap in a newly created temporary store')
    parser.add_argument('--root', help='New absolute private acceptance root; must not already exist')
    parser.add_argument('--shared-writer', action='store_true', help='Linux/WSL multi-chat routing through one private writer')
    arguments = parser.parse_args(argv)
    if not arguments.confirm_starter_base:
        parser.error('Review the bundled starter base, then provide --confirm-starter-base')
    import instinct
    from instinct.seed import load_starter_memories

    if 'site-packages' not in Path(instinct.__file__).parts:
        parser.error('Run from outside the checkout with a wheel-installed interpreter')
    executable = Path(sys.executable).parent / ('instinct-mcp.exe' if os.name == 'nt' else 'instinct-mcp')
    seeder = executable.with_name('instinct-seed.exe' if os.name == 'nt' else 'instinct-seed')
    require(executable.is_file() and seeder.is_file(), 'installed entrypoints')
    templates = load_starter_memories()
    instructions = (Path(__file__).parent.parent / 'examples' / 'client-instructions.md').read_text(encoding='utf-8')
    root = create_root(arguments.root)
    workspace, server = write_workspace(root, executable, instructions, shared_writer=arguments.shared_writer)
    probe_server = preflight_server(root, server)
    environment = os.environ.copy()
    environment.update(probe_server['env'])
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run(
        [str(seeder), '--data-dir', probe_server['env']['INSTINCT_DATA_DIR'], '--apply', '--confirm-starter-base'],
        cwd=root, env=environment, capture_output=True, text=True, timeout=180,
    )
    if completed.returncode:
        raise RuntimeError(f'Synthetic bootstrap failed; inspect only {root}:\n{completed.stderr}')
    report = {
        'workspace': str(workspace), 'data_dir': str(root / 'data'), 'server': SERVER_NAME,
        'preflight_data_dir': str(root / 'preflight-data'),
    }
    report.update(asyncio.run(preflight(root, probe_server, templates)))
    environment.update(server['env'])
    completed = subprocess.run(
        [str(seeder), '--data-dir', server['env']['INSTINCT_DATA_DIR'], '--apply', '--confirm-starter-base'],
        cwd=root, env=environment, capture_output=True, text=True, timeout=180,
    )
    if completed.returncode:
        raise RuntimeError(f'Workspace bootstrap failed; inspect only {root}:\n{completed.stderr}')
    seeded = json.loads(completed.stdout)
    require(len(seeded.get('created', [])) == len(templates), 'pristine workspace starters')
    report['workspace_store'] = 'freshly_seeded_not_exercised'
    (root / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (workspace / 'ACCEPTANCE.md').write_text(
        (Path(__file__).parent.parent / 'docs' / 'CLIENT_ACCEPTANCE.md').read_text(encoding='utf-8'),
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
