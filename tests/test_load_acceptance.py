# SPDX-License-Identifier: MPL-2.0
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests import run_load_acceptance as helper


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None, 'write_ack', 'write_fetch', 'context', 'observer', 'identities', 'search', 'mutations'])
async def test_load_workers_validate_endpoint_contracts_and_failures(monkeypatch, failure):
    clock = [0.0]
    routes = []
    expected_ids = {'synthetic-starter', 'synthetic-note'}

    async def one_operation(deadline, action, record):
        try:
            await action(record)
            record.update(operations=1, status='passed')
            clock[0] = 300.0
        except Exception:
            record['status'] = 'failed'
            raise
        except asyncio.CancelledError:
            record['status'] = 'cancelled'
            raise

    async def synthetic_tool(name, arguments):
        if name == 'instinct_update_score':
            content = 'Queued fallback' if failure == 'write_ack' else 'Updated score\nScore: 71 -> 70'
        elif name == 'instinct_get':
            content = '[71] Wrong score' if failure == 'write_fetch' else '[70] Synthetic note\nID: synthetic-note'
        elif name == 'instinct_get_context':
            content = ('Found 1 context items:\n## MANDATORY CONTEXT (Must Follow)\n'
                       + ('No token' if failure == 'context' else 'RESPONSE_REQUIREMENT: VALIDATION_TOKEN: synthetic'))
        else:
            pytest.fail(f'Unexpected tool: {name}')
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=content)])

    async def synthetic_http(base, route):
        routes.append(route)
        if route == '/api/health':
            return 200, {'read_only': failure != 'observer', 'writer_lock': {'held_by_us': False}, 'pid': 123}
        if route == '/api/list':
            return 200, {'items': [{'id': segment_id} for segment_id in (set() if failure == 'identities' else expected_ids)]}
        assert route.startswith('/api/search_with_pop?')
        assert 'apply_pops=0' in route and 'track_access=0' in route
        if failure == 'search':
            return 200, {'count': 0, 'items': []}
        return 200, {'mandatory': [], 'relevant': [], 'applied_pops': failure == 'mutations', 'tracked_access': False}

    monkeypatch.setattr(helper, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(helper, 'load_lane', one_operation)
    monkeypatch.setattr(helper, 'http', synthetic_http)
    evidence = {}
    session = SimpleNamespace(call_tool=AsyncMock(side_effect=synthetic_tool))
    process = Mock(pid=123)
    process.poll.return_value = None
    if failure:
        with pytest.raises(ExceptionGroup):
            await helper.exercise_load(session, 'synthetic-loopback', process, 'synthetic-note', expected_ids, evidence)
        assert evidence['status'] == 'failed'
        assert evidence.get('workload_status') != 'passed'
    else:
        await helper.exercise_load(session, 'synthetic-loopback', process, 'synthetic-note', expected_ids, evidence)
        assert evidence['workload_status'] == 'passed'
        assert evidence['elapsed_seconds'] == 300.0
        assert evidence['concurrent_workers'] == 4
        assert all(record['status'] == 'passed' for record in evidence['lanes'].values())
        assert any(route.startswith('/api/search_with_pop?') for route in routes)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


@pytest.mark.parametrize('mandatory', [True, False])
def test_load_context_accepts_mandatory_tokens_or_suppressed_relevance(mandatory):
    record = {}
    body = ('## MANDATORY CONTEXT (Must Follow)\nRESPONSE_REQUIREMENT: VALIDATION_TOKEN: synthetic'
            if mandatory else '## RELEVANT CONTEXT (Priority Order)\nSynthetic relevant memory')
    helper.validate_context_delivery('Found 1 context items:\n' + body, record)
    assert record == {('mandatory_deliveries' if mandatory else 'suppressed_deliveries'): 1}


@pytest.mark.parametrize('content', ['Error: synthetic failure', 'No context found.', 'Found 1 context items:\nNo body'])
def test_load_context_rejects_error_empty_or_missing_body(content):
    with pytest.raises(RuntimeError):
        helper.validate_context_delivery(content, {})


def test_load_requires_confirmation_before_creating_data(monkeypatch):
    monkeypatch.setattr(helper.tempfile, 'mkdtemp', lambda **kwargs: pytest.fail('Created unapproved data'))
    with pytest.raises(SystemExit) as error:
        helper.main([])
    assert error.value.code == 2


def test_load_rejects_checkout_before_creating_data(monkeypatch, tmp_path):
    import instinct

    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'instinct' / '__init__.py'))
    monkeypatch.setattr(helper.tempfile, 'mkdtemp', lambda **kwargs: pytest.fail('Created data from checkout'))
    with pytest.raises(SystemExit) as error:
        helper.main(['--confirm-synthetic-load'])
    assert error.value.code == 2


def test_load_keeps_bounded_real_interval_and_pacing():
    assert helper.LOAD_SECONDS == 300.0
    assert helper.PACE_SECONDS == 1.0
    assert helper.REQUEST_TIMEOUT_SECONDS == 30.0
    assert helper.MIN_LANE_OPERATIONS == 30


@pytest.mark.asyncio
async def test_load_tool_rejects_protocol_error():
    session = SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(isError=True)))
    with pytest.raises(RuntimeError, match='protocol error'):
        await helper.tool(session, 'synthetic/read', {})


@pytest.mark.asyncio
async def test_measured_requests_count_only_success():
    record = {'requests': 0, 'max_request_seconds': 0.0, 'total_request_seconds': 0.0}
    assert await helper.measured(record, AsyncMock(return_value='synthetic')()) == 'synthetic'
    assert record['requests'] == 1
    with pytest.raises(RuntimeError):
        await helper.measured(record, AsyncMock(side_effect=RuntimeError('synthetic failure'))())
    assert record['requests'] == 1


@pytest.mark.asyncio
async def test_load_lane_repeats_until_deadline(monkeypatch):
    clock = [0.0]
    action = AsyncMock()
    record = {'operations': 0, 'status': 'running'}

    async def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(helper.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(helper.asyncio, 'sleep', advance)
    await helper.load_lane(30.0, action, record)
    assert action.await_count == 30
    assert record == {'operations': 30, 'status': 'passed'}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['operation', 'insufficient', 'cancelled'])
async def test_load_lane_never_marks_failure_passed(failure):
    record = {'operations': 0, 'status': 'running'}
    error_type = asyncio.CancelledError if failure == 'cancelled' else RuntimeError
    action = AsyncMock(side_effect=error_type('synthetic failure'))
    deadline = 0.0 if failure == 'insufficient' else helper.time.monotonic() + 10.0
    with pytest.raises(error_type):
        await helper.load_lane(deadline, action, record)
    assert record['status'] != 'passed'


@pytest.mark.parametrize('status', ['passed', 'failed'])
def test_load_report_exit_status_and_isolated_environment(monkeypatch, tmp_path, status):
    import instinct

    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'site-packages' / 'instinct' / '__init__.py'))
    monkeypatch.setattr(helper, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setenv('INSTINCT_DATA_DIR', 'live')
    monkeypatch.setenv('CE_DATA_DIR', 'rc')
    monkeypatch.setenv('PYTHONPATH', 'checkout')
    monkeypatch.setattr(helper.tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path))

    async def synthetic_run(root, environment, report):
        assert Path(environment['INSTINCT_DATA_DIR']) == tmp_path / 'data'
        assert Path(environment['CE_DATA_DIR']) == tmp_path / 'data'
        assert 'PYTHONPATH' not in environment
        report['status'] = status

    monkeypatch.setattr(helper, 'run', synthetic_run)
    if status == 'failed':
        with pytest.raises(SystemExit) as error:
            helper.main(['--confirm-synthetic-load'])
        assert error.value.code == 1
    else:
        helper.main(['--confirm-synthetic-load'])
    assert json.loads((tmp_path / 'load.json').read_text())['status'] == status
