# SPDX-License-Identifier: MPL-2.0
"""Explicit five-minute installed-package concurrency check with synthetic data."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

if __package__:
    from .run_idle_acceptance import environment_for, snapshot
    from .run_lifecycle_acceptance import http, mcp_session, require, result_text, stop_owned, wait_dashboard
else:
    from run_idle_acceptance import environment_for, snapshot
    from run_lifecycle_acceptance import http, mcp_session, require, result_text, stop_owned, wait_dashboard


LOAD_SECONDS = 300.0
PACE_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30.0
MIN_LANE_OPERATIONS = 30
QUERY = 'mandatory memory workflow and safe recovery'


async def tool(session, name, arguments):
    result = await session.call_tool(name, arguments)
    require(not result.isError, f'Load tool returned protocol error: {name}')
    return result_text(result)


async def measured(record, awaitable):
    started = time.monotonic()
    result = await asyncio.wait_for(awaitable, timeout=REQUEST_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started
    record['requests'] += 1
    record['max_request_seconds'] = max(record['max_request_seconds'], elapsed)
    record['total_request_seconds'] += elapsed
    return result


def validate_context_delivery(delivered, record):
    require(delivered.startswith('Found ') and ' context items:' in delivered, 'Load context retrieval failed')
    if '## MANDATORY CONTEXT (Must Follow)' in delivered:
        require('RESPONSE_REQUIREMENT:' in delivered and 'VALIDATION_TOKEN:' in delivered, 'Load mandatory acknowledgement requirement missing')
        record['mandatory_deliveries'] = record.get('mandatory_deliveries', 0) + 1
    else:
        require('## RELEVANT CONTEXT (Priority Order)' in delivered, 'Load suppressed context body missing')
        record['suppressed_deliveries'] = record.get('suppressed_deliveries', 0) + 1


async def load_lane(deadline, action, record):
    try:
        while time.monotonic() < deadline:
            await action(record)
            record['operations'] += 1
            await asyncio.sleep(PACE_SECONDS)
        require(record['operations'] >= MIN_LANE_OPERATIONS, 'Too few repeated operations in load lane')
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    except asyncio.CancelledError:
        record['status'] = 'cancelled'
        raise


async def exercise_load(session, base, process, note_id, expected_ids, evidence):
    async def writer(record):
        score = 70 + record['operations'] % 2
        updated = await measured(record, tool(session, 'instinct_update_score', {
            'segment_id': note_id, 'new_score': score, 'reason': 'Synthetic concurrent-load checkpoint',
        }))
        require(updated.startswith('Updated score') and f' -> {score}' in updated, 'Load score write was not acknowledged')
        fetched = await measured(record, tool(session, 'instinct_get', {'segment_id': note_id}))
        require(fetched.startswith(f'[{score}] ') and note_id in fetched, 'Load read-after-write mismatch')
        evidence['last_acknowledged_score'] = score

    async def context(record):
        delivered = await measured(record, tool(session, 'instinct_get_context', {
            'query': QUERY, 'include_mandatory': True,
        }))
        validate_context_delivery(delivered, record)

    async def observer(record):
        require(process.poll() is None, 'Dashboard exited during concurrent load')
        status, health = await measured(record, http(base, '/api/health'))
        require(status == 200 and health['read_only'] and not health['writer_lock']['held_by_us'], 'Load dashboard stopped observing read-only')
        require(health['pid'] == process.pid, 'Load replaced dashboard process')
        status, listing = await measured(record, http(base, '/api/list'))
        require(status == 200 and {item['id'] for item in listing['items']} == expected_ids, 'Load changed persisted record identities')

    async def search(record):
        status, found = await measured(record, http(
            base, '/api/search_with_pop?q=mandatory%20memory%20workflow%20and%20safe%20recovery&apply_pops=0&track_access=0',
        ))
        require(status == 200 and 'mandatory' in found and 'relevant' in found,
                f'Load dashboard search failed: HTTP {status}, fields {sorted(found)}')
        require(not found['applied_pops'] and not found['tracked_access'], 'Observer search enabled mutations')

    actions = {'mcp_writer': writer, 'mcp_context': context, 'dashboard_observer': observer, 'dashboard_search': search}
    records = {name: {'status': 'running', 'operations': 0, 'requests': 0,
                      'max_request_seconds': 0.0, 'total_request_seconds': 0.0} for name in actions}
    evidence.update(status='running', duration_seconds_requested=LOAD_SECONDS,
                    concurrent_workers=len(actions), lanes=records, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                    pace_seconds=PACE_SECONDS, minimum_operations_per_lane=MIN_LANE_OPERATIONS)
    started = time.monotonic()
    try:
        async with asyncio.TaskGroup() as tasks:
            for name, action in actions.items():
                tasks.create_task(load_lane(started + LOAD_SECONDS, action, records[name]), name=name)
    except Exception:
        evidence['status'] = 'failed'
        raise
    finally:
        evidence['elapsed_seconds'] = time.monotonic() - started
    require(evidence['elapsed_seconds'] >= LOAD_SECONDS, 'Concurrent load interval was shortened')
    require(all(record['status'] == 'passed' for record in records.values()), 'Concurrent load lane failed')
    evidence['workload_status'] = 'passed'


async def run(root, environment, report):
    from instinct.chroma_writer_lock import is_writer_lock_held
    from instinct.seed import load_starter_memories

    executable = Path(sys.executable).parent
    subprocess.run(
        [str(executable / 'instinct-seed'), '--data-dir', str(root / 'data'), '--apply', '--confirm-starter-base'],
        env=environment, cwd=root, check=True, capture_output=True, text=True, timeout=180,
    )
    starter_ids = {memory.id for memory in load_starter_memories()}
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    require(port != 8765, 'Refusing the live dashboard port')
    base = f'http://127.0.0.1:{port}'
    report['port'] = port
    process = None
    with (root / 'dashboard.log').open('w', encoding='utf-8') as log:
        try:
            async with mcp_session(root, environment, 'mcp-load.log') as session:
                mandatory = await tool(session, 'instinct_get_mandatory', {})
                require(mandatory.startswith('Found 3 mandatory items:'), 'Load initial starter count mismatch')
                process = subprocess.Popen(
                    [str(executable / 'instinct-dashboard'), '--data-dir', str(root / 'data'), '--port', str(port)],
                    env=environment, cwd=root, stdout=log, stderr=log,
                )
                health = await wait_dashboard(process, base)
                require(health['store_dir'] == str(root / 'data') and health['read_only'], 'Load dashboard isolation/mode mismatch')
                writer_pid = int((root / 'data' / '.writer.lock').read_text().strip())
                require(health['writer_lock']['holder_pid'] == writer_pid and writer_pid != process.pid, 'Load MCP writer ownership mismatch')
                saved = await tool(session, 'instinct_save_memory', {
                    'title': 'Synthetic concurrent-load checkpoint',
                    'content': 'Disposable benchmark telemetry: green prism checkpoints verify acknowledged score changes during concurrent MCP and dashboard traffic.',
                    'importance': 70, 'segment_type': 'context',
                })
                require(saved.startswith('✓ Saved memory segment'), 'Load note was not committed to main store')
                note_ids = [line.removeprefix('ID: ') for line in saved.splitlines() if line.startswith('ID: ')]
                require(len(note_ids) == 1 and note_ids[0] not in starter_ids, 'Load note ID missing or unsafe')
                note_id = note_ids[0]
                report['synthetic_note_id'] = note_id
                expected_ids = starter_ids | {note_id}
                require(set(snapshot(root / 'data')) == expected_ids, 'Load initial persisted identity mismatch')
                print('Running four concurrent workers for 300 real seconds; only one MCP process writes.', flush=True)
                evidence = report.setdefault('load', {})
                await exercise_load(session, base, process, note_id, expected_ids, evidence)
                require(evidence['lanes']['mcp_context'].get('mandatory_deliveries', 0) > 0, 'Load never delivered mandatory context')
                require(evidence['lanes']['mcp_context'].get('suppressed_deliveries', 0) > 0, 'Load never exercised normal access suppression')
                persisted = snapshot(root / 'data')
                require(set(persisted) == expected_ids, 'Load final persisted identity mismatch')
                require(persisted[note_id]['importance_score'] == evidence['last_acknowledged_score'], 'Load final acknowledged score was not persisted')
                require(int((root / 'data' / '.writer.lock').read_text().strip()) == writer_pid, 'Load changed MCP writer process')
                status, blocked = await http(base, '/api/add', {
                    'title': 'Synthetic forbidden load observer write', 'content': 'Must not persist.', 'score': 70,
                })
                require(status == 409 and blocked['read_only'], 'Load observer write was not rejected')
                deleted = await tool(session, 'instinct_delete_memory', {'segment_id': note_id})
                require(deleted.startswith('Deleted segment'), 'Load failed to delete only synthetic checkpoint')
                missing = await tool(session, 'instinct_get', {'segment_id': note_id})
                require('No segment found' in missing, 'Load checkpoint remained after deletion')
                require(set(snapshot(root / 'data')) == starter_ids, 'Load cleanup altered starter identities')
                evidence.update(status='passed', final_score_persisted=True, same_mcp_writer=True,
                                observer_write_rejected=True, synthetic_note_deleted_and_absent=True,
                                starter_identities_preserved=True)
            report['test_owned_mcp_connection_closed'] = True
            require(not is_writer_lock_held(str(root / 'data')), 'Load cleanup left OS writer lock held')
            report['status'] = 'passed'
        finally:
            stop_owned(process)
            report['test_owned_dashboard_stopped'] = process is None or process.poll() is not None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-synthetic-load', action='store_true')
    arguments = parser.parse_args(argv)
    if not arguments.confirm_synthetic_load:
        parser.error('Approve fresh synthetic bootstrap and a five-minute concurrent workload explicitly')
    import instinct

    if os.name != 'posix' or 'site-packages' not in Path(instinct.__file__).parts:
        parser.error('Run outside the checkout with a wheel-installed Linux/WSL interpreter')
    root = Path(tempfile.mkdtemp(prefix='instinct-load-acceptance-'))
    report = {'status': 'running', 'root': str(root), 'installed_package': instinct.__file__,
              'live_stores_accessed': False, 'editor_owned_processes_stopped': False,
              'clocks_or_timeout_configuration_changed': False, 'model_compliance': 'not_tested',
              'capacity_or_long_soak_certification': 'not_tested'}
    print(f'Isolated load evidence: {root}', flush=True)
    try:
        asyncio.run(run(root, environment_for(root / 'data'), report))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (root / 'load.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
