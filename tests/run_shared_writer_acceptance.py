# SPDX-License-Identifier: MPL-2.0
"""Installed Linux multi-process acceptance against a fresh synthetic store."""

import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def text(result):
    return '\n'.join(item.text for item in result.content if hasattr(item, 'text'))


async def run(root, environment):
    from instinct.seed import load_starter_memories
    from instinct.shared_writer import runtime_directory
    templates = load_starter_memories()
    executable = Path(sys.executable).parent / 'instinct-mcp'
    data = root / 'data'
    commands = [asyncio.Queue() for unused in range(7)]
    ready = [asyncio.get_running_loop().create_future() for unused in range(7)]
    delivered_tokens = set()
    parameters = StdioServerParameters(
        command=str(executable), args=['--data-dir', str(data), '--shared-writer'],
        env=environment, cwd=str(root),
    )

    def check_challenge(output):
        match = re.search(r'^VALIDATION_TOKEN: ([0-9A-F]{12})$', output, re.MULTILINE)
        assert match, 'Missing current delivery challenge'
        token = match.group(1)
        assert token not in delivered_tokens, 'A previous delivery challenge was reused'
        assert f'End response with 「{token}」' in output
        delivered_tokens.add(token)

    async def client(index):
        try:
            with (root / f'client-{index}.log').open('w') as log:
                async with stdio_client(parameters, errlog=log) as streams:
                    async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=180)) as session:
                        await session.initialize()
                        assert len((await session.list_tools()).tools) == 7
                        first = text(await session.call_tool('instinct_get_context', {
                            'query': 'Synthetic simultaneous chat lifecycle check', 'include_mandatory': True,
                        }))
                        assert all(record.concept_title in first for record in templates), first
                        assert 'RESPONSE_REQUIREMENT:' in first and 'VALIDATION_TOKEN:' in first
                        check_challenge(first)
                        ready[index].set_result(True)
                        while True:
                            command = await commands[index].get()
                            if command is None:
                                return
                            name, arguments, result = command
                            try:
                                result.set_result(text(await session.call_tool(name, arguments)))
                            except Exception as error:
                                result.set_exception(error)
                                raise
        except BaseException as error:
            if not ready[index].done():
                ready[index].set_exception(error)
            raise

    async def call(index, name, arguments):
        result = asyncio.get_running_loop().create_future()
        await commands[index].put((name, arguments, result))
        return await asyncio.wait_for(result, 180)

    tasks = [asyncio.create_task(client(index)) for index in range(6)]
    report = {}
    try:
        await asyncio.wait_for(asyncio.gather(*ready[:6]), 180)
        holder = int((data / '.writer.lock').read_text().strip())
        lock_report = subprocess.run(['lslocks', '-n', '-o', 'PID,PATH'], capture_output=True, text=True, check=True).stdout
        assert any(line.split(maxsplit=1) == [str(holder), str(data / '.writer.lock')] for line in lock_report.splitlines())
        process_command = (Path('/proc') / str(holder) / 'cmdline').read_bytes().split(b'\x00')
        assert b'instinct.shared_writer' in process_command
        report['simultaneous_live_clients'] = 6
        report['single_writer_pid'] = holder
        report['automatic_per_connection_renewal'] = 'passed'
        renewed = await asyncio.gather(*(call(index, 'instinct_get_context', {
            'query': 'Synthetic explicit new chat event', 'include_mandatory': True, 'new_session': True,
        }) for index in range(6)))
        assert all(all(record.concept_title in output for record in templates) for output in renewed)
        for output in renewed:
            check_challenge(output)
        report['concurrent_explicit_renewals'] = 'passed'
        saved = await call(0, 'instinct_save_memory', {
            'title': 'Synthetic shared writer acceptance', 'content': 'Only a disposable cross-chat persistence test.',
            'importance': 70, 'segment_type': 'context',
        })
        match = re.search(r'^ID: (\S+)$', saved, re.MULTILINE)
        assert match and 'Saved memory segment' in saved, saved
        segment_id = match.group(1)
        fetched = await call(1, 'instinct_get', {'segment_id': segment_id})
        assert 'Only a disposable cross-chat persistence test.' in fetched and '70' in fetched
        deleted = await call(2, 'instinct_delete_memory', {'segment_id': segment_id})
        assert 'Deleted segment' in deleted
        missing = await call(3, 'instinct_get', {'segment_id': segment_id})
        assert 'No segment found' in missing
        report['cross_chat_save_fetch_delete_absence'] = 'passed'
        for arguments in ({'importance': 100}, {'importance': 70, 'segment_type': 'standing_instruction'}):
            blocked = await call(4, 'instinct_save_memory', {
                'title': 'Synthetic forbidden', 'content': 'Synthetic forbidden', **arguments,
            })
            assert 'blocked' in blocked.lower() or 'error' in blocked.lower(), blocked
        blocked = await call(5, 'instinct_update_score', {'segment_id': templates[0].id, 'new_score': 100})
        assert 'blocked' in blocked.lower() or 'error' in blocked.lower(), blocked
        report['owner_guards'] = 'passed'
        await commands[0].put(None)
        await asyncio.wait_for(tasks[0], 10)
        tasks.append(asyncio.create_task(client(6)))
        await asyncio.wait_for(ready[6], 180)
        assert int((data / '.writer.lock').read_text().strip()) == holder
        report['immediate_reconnect_with_live_peers'] = 'passed'
        report['fresh_delivery_challenges'] = len(delivered_tokens)
        report['challenge_uniqueness_and_response_consistency'] = 'passed'
    finally:
        for index in range(7):
            await commands[index].put(None)
        await asyncio.gather(*tasks, return_exceptions=True)
    directory = runtime_directory(data)
    for unused in range(40):
        if not (directory / 'writer.sock').exists():
            break
        await asyncio.sleep(1)
    assert not (directory / 'writer.sock').exists(), 'Backend did not exit after final client disconnected'
    from instinct.chroma_writer_lock import is_writer_lock_held
    assert not is_writer_lock_held(str(data)), 'Backend retained the writer lock after idle shutdown'
    report['idle_shutdown_and_writer_release'] = 'passed'
    return report


def main():
    import instinct
    assert sys.platform == 'linux'
    assert 'site-packages' in Path(instinct.__file__).parts, 'Must test installed wheel outside checkout'
    root = Path(tempfile.mkdtemp(prefix='instinct-shared-acceptance-'))
    environment = os.environ.copy()
    environment.update({
        'INSTINCT_DATA_DIR': str(root / 'data'), 'CE_DATA_DIR': str(root / 'data'),
        'CE_EMBEDDING_PROVIDERS': 'onnx', 'INSTINCT_ALLOW_REMOTE_PROVIDERS': '0',
        'INSTINCT_ALLOW_INSECURE_REMOTE_HTTP': '0', 'PYTHONDONTWRITEBYTECODE': '1',
    })
    environment.pop('PYTHONPATH', None)
    seeded = subprocess.run([
        str(Path(sys.executable).parent / 'instinct-seed'), '--data-dir', str(root / 'data'),
        '--apply', '--confirm-starter-base',
    ], cwd=root, env=environment, capture_output=True, text=True, timeout=180)
    assert seeded.returncode == 0, seeded.stderr
    os.environ.update({key: value for key, value in environment.items() if key.startswith(('CE_', 'INSTINCT_'))})
    report = {'synthetic_root': str(root), 'installed_module': str(instinct.__file__)}
    report.update(asyncio.run(run(root, environment)))
    (root / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
