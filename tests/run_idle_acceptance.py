# SPDX-License-Identifier: MPL-2.0
"""Opt-in real ten-minute idle restoration checks on fresh installed synthetic data."""

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time


IDLE_SECONDS = 660
OWNER_RECYCLE = '''
import sys
from instinct.store import InstinctStore
from instinct.chroma_writer_lock import release_writer_lock
store = InstinctStore(persist_dir=sys.argv[1])
try:
    recycled = store.recycle(sys.argv[2], by='synthetic-owner-acceptance', reason='Disposable idle-exclusion test')
    if recycled is None:
        raise RuntimeError('Synthetic starter not found')
finally:
    release_writer_lock(store._writer_lock_handle)
'''


def require(condition, description):
    if not condition:
        raise RuntimeError(description)


def text(result):
    return '\n'.join(item.text for item in result.content if hasattr(item, 'text'))


def snapshot(data):
    with sqlite3.connect((data / 'chroma.sqlite3').as_uri() + '?mode=ro', uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        rows = connection.execute('''
            SELECT embeddings.embedding_id, metadata.key, metadata.int_value,
                   metadata.string_value, metadata.float_value, metadata.bool_value
            FROM embeddings JOIN embedding_metadata AS metadata ON embeddings.id = metadata.id
            WHERE metadata.key IN ('importance_score', 'concept_title', 'recycled_at')
        ''').fetchall()
    records = {}
    for segment_id, key, integer, string, floating, boolean in rows:
        records.setdefault(segment_id, {})[key] = next(
            (value for value in (integer, string, floating, boolean) if value is not None), None,
        )
    return records


def state(data, filename):
    return json.loads((data / filename).read_text(encoding='utf-8'))


def idle_active_seconds(before, after):
    added = after['accumulated_session_seconds'] - before['accumulated_session_seconds']
    require(0 <= added <= 0.25, 'Real idle gap was charged to active-session decay time')
    return added


def environment_for(data):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.update({'INSTINCT_DATA_DIR': str(data), 'CE_DATA_DIR': str(data),
                        'CE_EMBEDDING_PROVIDERS': 'onnx', 'PYTHONDONTWRITEBYTECODE': '1'})
    return environment


@asynccontextmanager
async def connection_for(directory, environment, filename):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=str(Path(sys.executable).parent / 'instinct-mcp'),
        args=['--data-dir', str(directory / 'data')], env=environment, cwd=str(directory),
    )
    with (directory / filename).open('w', encoding='utf-8') as log:
        async with stdio_client(parameters, errlog=log) as streams:
            async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=180)) as session:
                await session.initialize()
                yield session


async def context(session):
    result = await session.call_tool('instinct_get_context', {
        'query': 'Synthetic acceptance of mandatory memory workflow and safe recovery',
        'include_mandatory': True,
    })
    require(not result.isError, 'Context call returned protocol error')
    return text(result)


def seed(directory, environment):
    subprocess.run(
        [str(Path(sys.executable).parent / 'instinct-seed'), '--data-dir', str(directory / 'data'),
         '--apply', '--confirm-starter-base'],
        cwd=directory, env=environment, check=True, capture_output=True, text=True, timeout=180,
    )


async def run(root, report):
    from instinct.seed import load_starter_memories
    from instinct.dormancy import DormancyConfig

    require(DormancyConfig().session_idle_minutes == 10, 'Unexpected default idle interval')
    templates = load_starter_memories()
    expected_ids = {memory.id for memory in templates}
    recycled_id = templates[-1].id
    scenarios = []
    for name in ('all-active', 'one-recycled'):
        directory = root / name
        directory.mkdir()
        data = directory / 'data'
        environment = environment_for(data)
        seed(directory, environment)
        if name == 'one-recycled':
            async with connection_for(directory, environment, 'initial-mcp.log') as session:
                initial = await context(session)
                require(all(memory.concept_title in initial for memory in templates), 'Initial synthetic starters missing')
            subprocess.run([sys.executable, '-c', OWNER_RECYCLE, str(data), recycled_id],
                           cwd=directory, env=environment, check=True, capture_output=True, text=True, timeout=180)
        scenarios.append({'name': name, 'directory': directory, 'data': data, 'environment': environment})

    async with AsyncExitStack() as stack:
        for scenario in scenarios:
            session = await stack.enter_async_context(connection_for(
                scenario['directory'], scenario['environment'], 'idle-mcp.log',
            ))
            scenario['session'] = session
            initial = await context(session)
            for memory in templates:
                expected = scenario['name'] == 'all-active' or memory.id != recycled_id
                require((memory.concept_title in initial) == expected, 'Initial active/recycled mandatory delivery mismatch')
            before = snapshot(scenario['data'])
            require(set(before) == expected_ids, 'Unexpected synthetic record identities')
            require(all(record['importance_score'] == 79 for record in before.values()), 'Initial access suppression must be 79')
            mandatory = text(await session.call_tool('instinct_get_mandatory', {}))
            require(mandatory == 'No mandatory items found.', f'Same-session mandatory suppression: {mandatory!r}')
            scenario['before'] = snapshot(scenario['data'])
            scenario['clock_before'] = state(scenario['data'], 'session_state.json')
            scenario['nudge_before'] = state(scenario['data'], 'nudge_state.json')
            scenario['writer_pid'] = int((scenario['data'] / '.writer.lock').read_text().strip())
        report['phase'] = 'waiting_real_idle'
        report['sleep_seconds_requested'] = IDLE_SECONDS
        report['status'] = 'running'
        (root / 'idle.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('Both isolated MCP processes remain connected. Waiting 660 real seconds without tool calls or dashboard polling.', flush=True)
        started = time.monotonic()
        await asyncio.sleep(IDLE_SECONDS)
        report['sleep_seconds_observed'] = time.monotonic() - started
        require(report['sleep_seconds_observed'] >= IDLE_SECONDS, 'Real idle wait too short')
        report['scenarios'] = {}
        for scenario in scenarios:
            data = scenario['data']
            require(snapshot(data) == scenario['before'], 'Memory metadata changed during idle')
            require(state(data, 'session_state.json') == scenario['clock_before'], 'Session clock changed during idle')
            require(state(data, 'nudge_state.json') == scenario['nudge_before'], 'Nudge clock changed during idle')
            require(int((data / '.writer.lock').read_text().strip()) == scenario['writer_pid'], 'MCP process identity changed during idle')
            elapsed = (datetime.now() - datetime.fromisoformat(scenario['clock_before']['session_last_tick_at'])).total_seconds()
            require(elapsed > 600, 'Actual last-tick idle interval did not exceed ten minutes')
            result = await scenario['session'].call_tool('instinct_get_mandatory', {})
            require(not result.isError, 'Mandatory call returned protocol error')
            returned = text(result)
            restored = snapshot(data)
            active_ids = expected_ids - ({recycled_id} if scenario['name'] == 'one-recycled' else set())
            require(returned.startswith(f'Found {len(active_ids)} mandatory items:'), 'Restored mandatory count mismatch')
            require(all(restored[segment_id]['importance_score'] == 100 for segment_id in active_ids), 'Active starters not restored to 100')
            clock_after = state(data, 'session_state.json')
            active_seconds_added = idle_active_seconds(scenario['clock_before'], clock_after)
            nudge_after = state(data, 'nudge_state.json')
            require(clock_after['session_started_at'] != scenario['clock_before']['session_started_at'], 'No new engine session observed')
            require(nudge_after['session_started_at'] == clock_after['session_started_at'], 'Nudge restoration marker mismatch')
            require(nudge_after['current_turn'] > scenario['nudge_before']['current_turn'], 'Turn counter must remain monotonic')
            refreshed = await context(scenario['session'])
            require('RESPONSE_REQUIREMENT:' in refreshed and 'VALIDATION_TOKEN:' in refreshed, 'Current context acknowledgement requirement missing')
            for memory in templates:
                if memory.id in active_ids:
                    require(memory.concept_title in returned and memory.concept_title in refreshed, 'Active starter missing from retrieval')
                else:
                    require(memory.concept_title not in returned and memory.concept_title not in refreshed, 'Recycled starter leaked into retrieval')
                    require(restored[memory.id]['importance_score'] == 79 and restored[memory.id].get('recycled_at'), 'Recycled starter was restored or unmarked')
            after_access = snapshot(data)
            require(all(after_access[segment_id]['importance_score'] == 79 for segment_id in active_ids), 'Post-retrieval access suppression mismatch')
            report['scenarios'][scenario['name']] = {
                'status': 'passed', 'actual_idle_seconds': elapsed, 'active_starters': len(active_ids),
                'before': scenario['before'], 'after_restoration': restored, 'after_context_access': after_access,
                'session_before': scenario['clock_before'], 'session_after': clock_after,
                'active_seconds_added_at_idle_renewal': active_seconds_added,
                'real_idle_gap_excluded_from_active_clock': True,
                'turn_before': scenario['nudge_before']['current_turn'], 'turn_after': nudge_after['current_turn'],
                'same_mcp_process_during_idle': True, 'context_acknowledgement_requirement': 'present',
            }
    report.update(status='passed', phase='complete', test_owned_mcp_connections_closed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-synthetic-idle', action='store_true')
    arguments = parser.parse_args(argv)
    if not arguments.confirm_synthetic_idle:
        parser.error('Approve fresh synthetic bootstrap, owner recycling and an eleven-minute real wait')
    import instinct

    if os.name != 'posix' or 'site-packages' not in Path(instinct.__file__).parts:
        parser.error('Run outside the checkout with a wheel-installed Linux/WSL interpreter')
    root = Path(tempfile.mkdtemp(prefix='instinct-idle-acceptance-'))
    report = {'status': 'running', 'root': str(root), 'installed_package': instinct.__file__,
              'live_stores_accessed': False, 'clocks_or_idle_configuration_changed': False,
              'model_compliance': 'not_tested', 'watchdog_and_forced_mcp_crashes': 'not_tested'}
    print(f'Isolated idle evidence: {root}', flush=True)
    try:
        asyncio.run(run(root, report))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (root / 'idle.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
