# SPDX-License-Identifier: MPL-2.0
"""Explicit installed-package lifecycle acceptance using a fresh synthetic store."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


WATCHDOG_TIMEOUT_SECONDS = 30.0
HEARTBEAT_REFRESH_SECONDS = 20.0
WATCHDOG_SETTLE_SECONDS = 17.0


def require(condition, description):
    if not condition:
        raise RuntimeError(description)


def result_text(result):
    return '\n'.join(item.text for item in result.content if hasattr(item, 'text'))


def request_json(base, route, payload=None):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if payload is not None:
        with urlopen(base + '/api/session', timeout=75) as response:
            headers['X-Instinct-Capability'] = json.load(response)['capability']
    request = Request(base + route, data=body, headers=headers)
    try:
        with urlopen(request, timeout=75) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


async def http(base, route, payload=None):
    return await asyncio.to_thread(request_json, base, route, payload)


async def wait_dashboard(process, base):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        require(process.poll() is None, 'Dashboard exited before becoming reachable')
        try:
            status, health = await http(base, '/api/health')
            if status == 200:
                return health
        except (URLError, TimeoutError, ConnectionError):
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError('Dashboard startup timed out')


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


@asynccontextmanager
async def mcp_session(root, environment, log_name):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=str(Path(sys.executable).parent / 'instinct-mcp'),
        args=['--data-dir', str(root / 'data')], env=environment, cwd=str(root),
    )
    with (root / log_name).open('w', encoding='utf-8') as log:
        async with stdio_client(parameters, errlog=log) as streams:
            async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=180)) as session:
                await session.initialize()
                yield session


async def mcp_request(process, method, params, request_id):
    message = {'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params}
    process.stdin.write((json.dumps(message) + '\n').encode('utf-8'))
    await process.stdin.drain()
    async with asyncio.timeout(180):
        while True:
            line = await process.stdout.readline()
            require(bool(line), 'Owned MCP closed before replying')
            response = json.loads(line)
            if response.get('id') != request_id:
                continue
            require('error' not in response and 'result' in response, 'Owned MCP returned protocol error')
            return response['result']


async def mcp_tool(process, name, arguments, request_id):
    from mcp.types import CallToolResult

    result = CallToolResult.model_validate(await mcp_request(
        process, 'tools/call', {'name': name, 'arguments': arguments}, request_id,
    ))
    require(not result.isError, f'Owned MCP tool failed: {name}')
    return result_text(result)


@asynccontextmanager
async def owned_crash_mcp(root, environment, log_name):
    from mcp.types import LATEST_PROTOCOL_VERSION

    command = str(Path(sys.executable).parent / 'instinct-mcp')
    with (root / log_name).open('w', encoding='utf-8') as log:
        process = await asyncio.create_subprocess_exec(
            command, '--data-dir', str(root / 'data'), env=environment, cwd=root,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
        )
        try:
            await mcp_request(process, 'initialize', {
                'protocolVersion': LATEST_PROTOCOL_VERSION, 'capabilities': {},
                'clientInfo': {'name': 'synthetic-crash-acceptance', 'version': '1'},
            }, 1)
            process.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            await process.stdin.drain()
            yield process
        finally:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=15)
                except TimeoutError:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=15)


async def kill_owned_mcp(process):
    require(process.returncode is None, 'Owned MCP already exited before forced crash')
    process.kill()
    exit_code = await asyncio.wait_for(process.wait(), timeout=15)
    require(exit_code == -signal.SIGKILL, 'Owned MCP did not exit from SIGKILL')
    return exit_code


async def check_mcp_crashes(root, environment, base, dashboard_process, templates, report):
    from instinct.chroma_writer_lock import is_writer_lock_held

    evidence = {'status': 'running', 'cycles': [], 'crash_point': 'after acknowledged save and verified fetch',
                'in_flight_write_durability': 'not_tested'}
    report['mcp_crashes'] = evidence
    lock = root / 'data' / '.writer.lock'
    starter_ids = {memory.id for memory in templates}
    for cycle_index in range(2):
        cycle = {'status': 'running', 'cycle': cycle_index + 1}
        evidence['cycles'].append(cycle)
        async with owned_crash_mcp(root, environment, f'mcp-crash-{cycle_index + 1}.log') as process:
            title = f'Synthetic acknowledged crash note {cycle_index + 1}'
            saved = await mcp_tool(process, 'instinct_save_memory', {
                'title': title, 'content': f'Disposable committed crash test note {cycle_index + 1}.',
                'importance': 70, 'segment_type': 'context',
            }, 2)
            require(saved.startswith('✓ Saved memory segment'), 'Crash note was not committed to main store')
            note_ids = [line.removeprefix('ID: ') for line in saved.splitlines() if line.startswith('ID: ')]
            require(len(note_ids) == 1 and note_ids[0] not in starter_ids, 'Crash note ID missing or unsafe')
            note_id = note_ids[0]
            fetched = await mcp_tool(process, 'instinct_get', {'segment_id': note_id}, 3)
            require(title in fetched, 'Acknowledged crash note fetch failed')
            require(int(lock.read_text().strip()) == process.pid, 'Crash target is not synthetic store writer')
            inode = lock.stat().st_ino
            status, observed = await http(base, '/api/health')
            require(status == 200 and observed['read_only'], 'Dashboard must observe before MCP crash')
            cycle.update(crashed_pid=process.pid, note_id=note_id)
            cycle['exit_code'] = await kill_owned_mcp(process)
        require(lock.stat().st_ino == inode, 'MCP crash replaced lock sentinel')
        require(not is_writer_lock_held(str(root / 'data')), 'MCP crash failed to release actual OS lock')
        require(dashboard_process.poll() is None, 'MCP crash stopped dashboard observer')
        status, acquired = await http(base, '/api/mode/take_control', {})
        require(status == 200 and acquired['acquired'] and not acquired['mcp_stopped'], 'Dashboard failed to acquire crash-released lock')
        status, listing = await http(base, '/api/list')
        require(status == 200 and {item['id'] for item in listing['items']} == starter_ids | {note_id}, 'MCP crash lost or added persisted records')
        status, released = await http(base, '/api/mode/return_control', {})
        require(status == 200 and released['released'] and not released['mcp_started'], 'Dashboard crash-handoff release failed')
        async with owned_crash_mcp(root, environment, f'mcp-restart-{cycle_index + 1}.log') as restarted:
            require(restarted.pid != cycle['crashed_pid'], 'MCP restart reused crash process')
            fetched = await mcp_tool(restarted, 'instinct_get', {'segment_id': note_id}, 2)
            require(title in fetched, 'Restart did not retrieve acknowledged crash note')
            require(int(lock.read_text().strip()) == restarted.pid, 'Restart did not acquire writer lock')
            deleted = await mcp_tool(restarted, 'instinct_delete_memory', {'segment_id': note_id}, 3)
            require('Deleted segment' in deleted, 'Restart failed to delete only synthetic crash note')
            missing = await mcp_tool(restarted, 'instinct_get', {'segment_id': note_id}, 4)
            require('No segment found' in missing, 'Crash note remained after cleanup')
            mandatory = await mcp_tool(restarted, 'instinct_get_mandatory', {}, 5)
            require(all(memory.concept_title in mandatory for memory in templates), 'Restart lost mandatory starters')
            cycle['restarted_pid'] = restarted.pid
        require(not is_writer_lock_held(str(root / 'data')), 'Restart cleanup left writer lock held')
        cycle.update(status='passed', committed_note_preserved=True, dashboard_survived=True,
                     dashboard_acquired_released_lock=True, restarted_mcp_acquired_lock=True,
                     lock_inode_preserved=True, synthetic_note_deleted_and_absent=True)
    status, listing = await http(base, '/api/list')
    require(status == 200 and {item['id'] for item in listing['items']} == starter_ids, 'Crash cleanup changed starter identities')
    evidence.update(status='passed', starter_identities_preserved=True, test_owned_mcp_processes_exited=True)


async def check_watchdog(root, base, process, report):
    from instinct.chroma_writer_lock import is_writer_lock_held

    report['watchdog'] = {'status': 'running', 'clocks_or_timeout_changed': False}
    status, acquired = await http(base, '/api/mode/take_control', {})
    require(status == 200 and acquired['acquired'], 'Watchdog dashboard acquisition')
    require(acquired['heartbeat_timeout_s'] == WATCHDOG_TIMEOUT_SECONDS, 'Unexpected watchdog timeout')
    marker = root / 'data' / '.dashboard_recovery.json'
    lock = root / 'data' / '.writer.lock'
    require(marker.exists(), 'Watchdog write-session recovery marker missing')
    lock_inode = lock.stat().st_ino
    acquired_at = time.monotonic()
    status, heartbeat = await http(base, '/api/heartbeat', {})
    require(status == 200 and heartbeat['in_write_session'], 'Initial watchdog heartbeat')
    print('Watchdog: renewing heartbeat, then waiting for the real default timeout.', flush=True)
    await asyncio.sleep(HEARTBEAT_REFRESH_SECONDS)
    status, heartbeat = await http(base, '/api/heartbeat', {})
    require(status == 200 and heartbeat['in_write_session'], 'Renewed watchdog heartbeat')
    renewed_at = time.monotonic()
    await asyncio.sleep(HEARTBEAT_REFRESH_SECONDS)
    require(process.poll() is None, 'Dashboard exited during heartbeat renewal')
    status, renewed = await http(base, '/api/health')
    elapsed_since_acquisition = time.monotonic() - acquired_at
    require(elapsed_since_acquisition > WATCHDOG_TIMEOUT_SECONDS, 'Renewal did not cross original timeout')
    require(status == 200 and not renewed['read_only'] and renewed['writer_lock']['held_by_us'], 'Heartbeat failed to renew writer ownership')
    require(marker.exists(), 'Recovery marker cleared before renewed deadline')
    await asyncio.sleep(WATCHDOG_SETTLE_SECONDS)
    silence_seconds = time.monotonic() - renewed_at
    require(silence_seconds > WATCHDOG_TIMEOUT_SECONDS, 'Watchdog silence was too short')
    require(process.poll() is None, 'Dashboard exited instead of releasing ownership')
    status, expired = await http(base, '/api/health')
    require(status == 200 and expired['read_only'] and not expired['writer_lock']['held_by_us'], 'Watchdog did not return dashboard to observer mode')
    require(expired['pid'] == process.pid, 'Watchdog replaced dashboard process')
    require(not marker.exists(), 'Watchdog left write-session recovery marker')
    require(lock.stat().st_ino == lock_inode, 'Watchdog replaced writer lock sentinel')
    require(not is_writer_lock_held(str(root / 'data')), 'Watchdog did not release actual OS lock')
    log = (root / 'dashboard.log').read_text(encoding='utf-8')
    require('WATCHDOG: heartbeat timed out' in log and 'WATCHDOG: auto-return complete. mcp_started=False' in log, 'Watchdog completion evidence missing')
    status, blocked = await http(base, '/api/add', {
        'title': 'Synthetic forbidden post-watchdog note', 'content': 'Must not persist.', 'score': 70,
    })
    require(status == 409 and blocked['read_only'], 'Post-watchdog write was not rejected')
    report['watchdog'] = {
        'status': 'passed', 'timeout_seconds': WATCHDOG_TIMEOUT_SECONDS,
        'renewal_elapsed_since_acquisition_seconds': elapsed_since_acquisition,
        'silence_after_last_heartbeat_seconds': silence_seconds,
        'same_dashboard_process': True, 'writer_lock_released': True,
        'lock_inode_preserved': True, 'recovery_marker_cleared': True,
        'post_timeout_write_rejected': True, 'mcp_auto_started': False,
        'clocks_or_timeout_changed': False,
    }


async def run(root, environment, report):
    from instinct.seed import load_starter_memories

    templates = load_starter_memories()
    executable = Path(sys.executable).parent
    subprocess.run(
        [str(executable / 'instinct-seed'), '--data-dir', str(root / 'data'),
         '--apply', '--confirm-starter-base'],
        env=environment, cwd=root, check=True, capture_output=True, text=True, timeout=180,
    )
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    require(port != 8765, 'Refusing the live dashboard port')
    base = f'http://127.0.0.1:{port}'
    report['port'] = port
    command = [str(executable / 'instinct-dashboard'), '--data-dir', str(root / 'data'), '--port', str(port)]
    process = None
    with (root / 'dashboard.log').open('w', encoding='utf-8') as log:
        try:
            async with mcp_session(root, environment, 'mcp-first.log') as session:
                mandatory = result_text(await session.call_tool('instinct_get_mandatory', {}))
                require(all(memory.concept_title in mandatory for memory in templates), 'Initial MCP starters')
                process = subprocess.Popen(command, env=environment, cwd=root, stdout=log, stderr=log)
                health = await wait_dashboard(process, base)
                require(health['store_dir'] == str(root / 'data'), 'Wrong dashboard data directory')
                require(health['read_only'], 'Dashboard must observe the MCP writer read-only')
                require(not health['writer_lock']['held_by_us'], 'Read-only dashboard must not hold writer lock')
                report['writer_owner_pid_diagnostic'] = (
                    'failed: waiting dashboard overwrote MCP holder PID'
                    if health['writer_lock']['holder_pid'] == process.pid else 'passed'
                )
                report['mcp_writer_dashboard_observer'] = 'passed'
                status, listing = await http(base, '/api/list')
                require(status == 200 and listing['count'] == 3, 'Dashboard must list three starters')
                status, blocked = await http(base, '/api/add', {
                    'title': 'Synthetic forbidden dashboard note', 'content': 'Must not persist.', 'score': 70,
                })
                require(status == 409 and blocked['read_only'], 'Read-only dashboard must reject writes')
                report['dashboard_read_only_write_rejection'] = 'passed'
                status, busy = await http(base, '/api/mode/take_control', {})
                require(status == 200 and not busy['acquired'] and not busy['mcp_stopped'], 'Busy takeover must not stop MCP')
                fetched = result_text(await session.call_tool('instinct_get', {'segment_id': templates[0].id}))
                require(templates[0].concept_title in fetched, 'MCP must remain usable after refused takeover')
                report['busy_takeover_preserves_mcp'] = 'passed'
            status, acquired = await http(base, '/api/mode/take_control', {})
            require(status == 200 and acquired['acquired'] and acquired['holder_pid'] == process.pid, 'Dashboard lock acquisition after MCP exit')
            status, saved = await http(base, '/api/add', {
                'title': 'Synthetic lifecycle dashboard note',
                'content': 'This disposable record verifies dashboard-to-MCP handoff.',
                'score': 70, 'segment_type': 'context',
            })
            require(status == 200 and saved.get('id'), 'Dashboard synthetic write')
            note_id = saved['id']
            report['dashboard_write_after_mcp_exit'] = 'passed'
            status, released = await http(base, '/api/mode/return_control', {})
            require(status == 200 and released['released'] and not released['mcp_started'], 'Dashboard release must not launch MCP')
            async with mcp_session(root, environment, 'mcp-second.log') as session:
                fetched = result_text(await session.call_tool('instinct_get', {'segment_id': note_id}))
                require('Synthetic lifecycle dashboard note' in fetched, 'MCP must see committed dashboard note')
                status, health = await http(base, '/api/health')
                require(status == 200 and health['read_only'], 'Dashboard must remain observer after release')
                deleted = result_text(await session.call_tool('instinct_delete_memory', {'segment_id': note_id}))
                require('Deleted segment' in deleted, 'MCP must reacquire writer and delete only synthetic note')
                missing = result_text(await session.call_tool('instinct_get', {'segment_id': note_id}))
                require('No segment found' in missing, 'Synthetic note cleanup')
                report['dashboard_to_mcp_handoff_and_cleanup'] = 'passed'
            await check_watchdog(root, base, process, report)
            async with mcp_session(root, environment, 'mcp-after-watchdog.log') as session:
                mandatory = result_text(await session.call_tool('instinct_get_mandatory', {}))
                require(all(memory.concept_title in mandatory for memory in templates), 'MCP must reacquire after watchdog release')
                report['mcp_reacquisition_after_watchdog'] = 'passed'
            status, acquired = await http(base, '/api/mode/take_control', {})
            require(status == 200 and acquired['acquired'], 'Dashboard reacquisition before crash check')
            marker = root / 'data' / '.dashboard_recovery.json'
            require(marker.exists(), 'Dashboard write-session recovery marker')
            crashed_pid = process.pid
            process.kill()
            process.wait(timeout=15)
            require(marker.exists(), 'Crash must leave recovery evidence for startup')
            process = subprocess.Popen(command, env=environment, cwd=root, stdout=log, stderr=log)
            health = await wait_dashboard(process, base)
            require(not health['read_only'] and health['writer_lock']['holder_pid'] == process.pid, 'Restart must reacquire crash-released OS lock')
            require(not marker.exists(), 'Restart must clear stale dashboard recovery marker')
            report['owned_dashboard_crash_recovery'] = 'passed'
            report['crashed_owned_dashboard_pid'] = crashed_pid
            status, listing = await http(base, '/api/list')
            require(status == 200 and listing['count'] == 3, 'Only starters must remain after restart')
            require({item['id'] for item in listing['items']} == {memory.id for memory in templates}, 'Starter identities preserved')
            report['starter_preservation'] = 'passed'
            status, released = await http(base, '/api/mode/return_control', {})
            require(status == 200 and released['released'], 'Dashboard release before MCP crash checks')
            print('Testing two forced crashes of helper-owned MCP writers only.', flush=True)
            await check_mcp_crashes(root, environment, base, process, templates, report)
            report['status'] = (
                'passed' if report['writer_owner_pid_diagnostic'] == 'passed' else 'failed'
            )
        finally:
            stop_owned(process)
            report['owned_dashboard_stopped'] = process is None or process.poll() is not None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-synthetic-lifecycle', action='store_true')
    arguments = parser.parse_args(argv)
    if not arguments.confirm_synthetic_lifecycle:
        parser.error('Approve fresh synthetic bootstrap and owned-process crash testing explicitly')
    import instinct

    if os.name != 'posix' or 'site-packages' not in Path(instinct.__file__).parts:
        parser.error('Run outside the checkout with a wheel-installed Linux/WSL interpreter')
    root = Path(tempfile.mkdtemp(prefix='instinct-lifecycle-acceptance-'))
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.update({
        'INSTINCT_DATA_DIR': str(root / 'data'), 'CE_DATA_DIR': str(root / 'data'),
        'CE_EMBEDDING_PROVIDERS': 'onnx', 'PYTHONDONTWRITEBYTECODE': '1',
    })
    report = {'status': 'running', 'root': str(root), 'installed_package': instinct.__file__,
              'live_stores_accessed': False, 'editor_owned_processes_stopped': False,
              'real_idle_restoration': 'not_tested', 'recycled_exclusion': 'not_tested'}
    print(f'Isolated lifecycle evidence: {root}', flush=True)
    try:
        asyncio.run(run(root, environment, report))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (root / 'lifecycle.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
