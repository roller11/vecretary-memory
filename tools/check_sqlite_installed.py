# SPDX-License-Identifier: MPL-2.0
"""Isolated installed SQLite checks, using new synthetic storage only."""

import asyncio
from importlib.metadata import PackageNotFoundError, distribution, distributions
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen


async def check_stdio(root):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=sys.executable, args=[
        '-I', '-m', 'instinct.sqlite_service', '--data-dir', str(root),
        '--database-dir', str(root / 'sqlite'),
        '--identity', str(root / 'embedding-identity.json')])
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as connection:
            await connection.initialize()
            names = {tool.name for tool in (await connection.list_tools()).tools}
            assert names == {'instinct_get_context', 'instinct_get_mandatory',
                             'instinct_search', 'instinct_save_memory', 'instinct_get',
                             'instinct_update_score'}
            result = await connection.call_tool('instinct_get_mandatory', {})
            assert not result.isError
            assert 'Error:' not in result.content[0].text
            result = await connection.call_tool('instinct_save_memory', {
                'content': 'Synthetic unbound refusal', 'title': 'Synthetic refusal'})
            assert 'Error executing' in result.content[0].text


def main():
    assert sys.flags.isolated, 'Run with python -I'
    try:
        distribution('chromadb')
    except PackageNotFoundError:
        pass
    else:
        raise AssertionError('SQLite release gate requires an environment without Chroma')
    import instinct
    from instinct.sqlite_init import initialize
    from instinct.sqlite_records import EmbeddingIdentity
    from instinct.sqlite_service import SQLiteService
    from instinct.sqlite_store import SQLiteRecordStore

    assert Path(instinct.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    product = distribution('vecretary-memory')
    assert product.version == '0.2.0a1'
    assert product.metadata['License-Expression'] == 'MPL-2.0'
    license_files = [file for file in product.files or ()
                     if str(file).endswith('.dist-info/licenses/LICENSE')]
    assert len(license_files) == 1
    assert 'Mozilla Public License Version 2.0' in product.locate_file(license_files[0]).read_text()
    entrypoints = {entry.name: entry for entry in product.entry_points
                  if entry.group == 'console_scripts'}
    assert {name: entry.value for name, entry in entrypoints.items()} == {
        'vecretary-mcp': 'instinct.sqlite_service:cli',
        'vecretary-sqlite': 'instinct.sqlite_service:cli',
        'vecretary-init': 'instinct.sqlite_init:cli'}
    for entry in entrypoints.values():
        entry.load()
        command = Path(sys.executable).parent / entry.name
        if sys.platform == 'win32':
            command = command.with_suffix('.exe')
        result = subprocess.run([str(command), '--help'], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, entry.name

    identity = EmbeddingIdentity('ollama', 'synthetic-unprobed', 2, 'cosine')
    with tempfile.TemporaryDirectory(prefix='vecretary-sqlite-installed-') as temporary:
        root = initialize(Path(temporary) / 'runtime', identity)
        try:
            initialize(root, identity)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Initializer overwrote an existing root')
        with SQLiteService(root / 'sqlite', identity, persist_dir=root, read_only=False) as service:
            address = service.start_dashboard()
            with urlopen(address + '/api/health', timeout=10) as response:
                health = json.load(response)
            assert health['backend'] == 'sqlite'
            assert health['runtime_ready'] is False
            assert health['embedding_probed'] is False
        asyncio.run(asyncio.wait_for(check_stdio(root), timeout=30))
        with SQLiteRecordStore.open_read_only(root / 'sqlite', identity) as repository:
            assert repository.count() == 0
    assert not any(name.split('.')[0] in ('chromadb', 'chromadb_rust_bindings', 'wisdom', 'triage', 'tc_mcp')
                   for name in sys.modules)
    print(json.dumps({'installed_sqlite': True, 'chroma_installed': False,
                      'cli_help': True, 'fresh_root': True, 'no_overwrite': True,
                      'loopback_dashboard': True, 'stdio_mcp': True,
                      'unbound_write_refused': True, 'provider_probed': False,
                      'sqlite_version': __import__('sqlite3').sqlite_version,
                      'dependencies': sorted((item.metadata['Name'], item.version) for item in distributions())}))


if __name__ == '__main__':
    main()
