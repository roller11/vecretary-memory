# SPDX-License-Identifier: MPL-2.0
import asyncio
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.run_lifecycle_acceptance import main, require, stop_owned
from tests import run_lifecycle_acceptance


@pytest.mark.asyncio
async def test_crash_rpc_matches_response_id_and_skips_notifications():
    stdout = asyncio.StreamReader()
    stdout.feed_data(b'{"jsonrpc":"2.0","method":"notifications/message"}\n')
    stdout.feed_data(b'{"jsonrpc":"2.0","id":7,"result":{"confirmed":true}}\n')
    process = Mock(stdout=stdout, stdin=Mock(drain=AsyncMock()))
    result = await run_lifecycle_acceptance.mcp_request(process, 'synthetic/read', {}, 7)
    assert result == {'confirmed': True}
    message = json.loads(process.stdin.write.call_args.args[0])
    assert message == {'jsonrpc': '2.0', 'id': 7, 'method': 'synthetic/read', 'params': {}}
    process.stdin.drain.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [b'', b'{"id":7,"error":{"code":-1}}\n', b'{"id":7}\n'])
async def test_crash_rpc_rejects_eof_or_protocol_error(response):
    stdout = asyncio.StreamReader()
    stdout.feed_data(response)
    stdout.feed_eof()
    process = Mock(stdout=stdout, stdin=Mock(drain=AsyncMock()))
    with pytest.raises(RuntimeError):
        await run_lifecycle_acceptance.mcp_request(process, 'synthetic/read', {}, 7)
    process.kill.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_code', [-9, -15])
async def test_crash_kills_only_supplied_process_and_requires_sigkill(monkeypatch, exit_code):
    monkeypatch.setattr(run_lifecycle_acceptance, 'signal', SimpleNamespace(SIGKILL=9))
    process = Mock(returncode=None, wait=AsyncMock(return_value=exit_code))
    if exit_code == -9:
        assert await run_lifecycle_acceptance.kill_owned_mcp(process) == -9
    else:
        with pytest.raises(RuntimeError, match='SIGKILL'):
            await run_lifecycle_acceptance.kill_owned_mcp(process)
    process.kill.assert_called_once_with()
    process.wait.assert_awaited_once()
    process.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_crash_refuses_already_exited_process():
    process = Mock(returncode=0)
    with pytest.raises(RuntimeError, match='already exited'):
        await run_lifecycle_acceptance.kill_owned_mcp(process)
    process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_crash_client_initialization_failure_cleans_up_owned_process(monkeypatch, tmp_path):
    process = Mock(returncode=None, wait=AsyncMock(return_value=-15))
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(run_lifecycle_acceptance.asyncio, 'create_subprocess_exec', create)
    monkeypatch.setattr(run_lifecycle_acceptance, 'mcp_request', AsyncMock(side_effect=RuntimeError('synthetic initialization failure')))
    with pytest.raises(RuntimeError, match='synthetic initialization failure'):
        async with run_lifecycle_acceptance.owned_crash_mcp(tmp_path, {'INSTINCT_DATA_DIR': str(tmp_path / 'data')}, 'owned.log'):
            pytest.fail('Yielded an uninitialized process')
    process.terminate.assert_called_once_with()
    process.wait.assert_awaited_once()
    process.kill.assert_not_called()
    assert create.call_args.kwargs['cwd'] == tmp_path
    assert create.call_args.args[1:] == ('--data-dir', str(tmp_path / 'data'))


@pytest.mark.asyncio
async def test_crash_tool_rejects_tool_error(monkeypatch):
    monkeypatch.setattr(run_lifecycle_acceptance, 'mcp_request', AsyncMock(return_value={
        'isError': True, 'content': [{'type': 'text', 'text': 'synthetic failure'}],
    }))
    with pytest.raises(RuntimeError, match='tool failed'):
        await run_lifecycle_acceptance.mcp_tool(Mock(), 'synthetic/read', {}, 7)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None, 'write_mode', 'marker', 'os_lock', 'log', 'write_allowed', 'exited'])
async def test_watchdog_requires_renewal_timeout_and_safe_release(monkeypatch, tmp_path, failure):
    data = tmp_path / 'data'
    data.mkdir()
    marker = data / '.dashboard_recovery.json'
    marker.write_text('synthetic write session')
    (data / '.writer.lock').write_text('123')
    log = tmp_path / 'dashboard.log'
    log.write_text('')
    clock = [0.0]
    waits = []
    calls = []
    process = Mock(pid=123)
    process.poll.return_value = 1 if failure == 'exited' else None

    async def synthetic_sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
        if len(waits) == 3:
            if failure != 'marker':
                marker.unlink()
            if failure != 'log':
                log.write_text('WATCHDOG: heartbeat timed out\nWATCHDOG: auto-return complete. mcp_started=False')

    async def synthetic_http(base, route, payload=None):
        calls.append(route)
        if route == '/api/mode/take_control':
            return 200, {'acquired': True, 'heartbeat_timeout_s': 30.0}
        if route == '/api/heartbeat':
            return 200, {'in_write_session': True}
        if route == '/api/health':
            expired = len(waits) == 3 and failure != 'write_mode'
            return 200, {'read_only': expired, 'writer_lock': {'held_by_us': not expired}, 'pid': 123}
        if route == '/api/add':
            return (200 if failure == 'write_allowed' else 409), {'read_only': True}
        pytest.fail(f'Unexpected request: {route}')

    monkeypatch.setattr(run_lifecycle_acceptance, 'http', synthetic_http)
    monkeypatch.setattr(run_lifecycle_acceptance.asyncio, 'sleep', synthetic_sleep)
    monkeypatch.setattr(run_lifecycle_acceptance.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr('instinct.chroma_writer_lock.is_writer_lock_held', lambda path: failure == 'os_lock')
    report = {}
    if failure:
        with pytest.raises(RuntimeError):
            await run_lifecycle_acceptance.check_watchdog(tmp_path, 'synthetic-loopback', process, report)
        assert report['watchdog']['status'] != 'passed'
    else:
        await run_lifecycle_acceptance.check_watchdog(tmp_path, 'synthetic-loopback', process, report)
        assert report['watchdog']['status'] == 'passed'
        assert report['watchdog']['silence_after_last_heartbeat_seconds'] == 37.0
        assert waits == [20.0, 20.0, 17.0]
        assert calls.count('/api/heartbeat') == 2
        assert '/api/mode/return_control' not in calls
        assert '/api/mode/release' not in calls
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_watchdog_acceptance_keeps_real_default_intervals():
    from instinct import dashboard

    assert run_lifecycle_acceptance.WATCHDOG_TIMEOUT_SECONDS == dashboard.HEARTBEAT_TIMEOUT_S == 30.0
    assert run_lifecycle_acceptance.HEARTBEAT_REFRESH_SECONDS < dashboard.HEARTBEAT_TIMEOUT_S
    assert (run_lifecycle_acceptance.HEARTBEAT_REFRESH_SECONDS + run_lifecycle_acceptance.WATCHDOG_SETTLE_SECONDS
            > dashboard.HEARTBEAT_TIMEOUT_S + dashboard.HEARTBEAT_CHECK_INTERVAL_S)


def test_lifecycle_requires_confirmation_before_creating_data(monkeypatch):
    monkeypatch.setattr('tests.run_lifecycle_acceptance.tempfile.mkdtemp',
                        lambda **kwargs: pytest.fail('Created data without approval'))
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_lifecycle_rejects_checkout_before_creating_data(monkeypatch, tmp_path):
    import instinct

    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'instinct' / '__init__.py'))
    monkeypatch.setattr('tests.run_lifecycle_acceptance.tempfile.mkdtemp',
                        lambda **kwargs: pytest.fail('Created data with checkout imports'))
    with pytest.raises(SystemExit) as error:
        main(['--confirm-synthetic-lifecycle'])
    assert error.value.code == 2


def test_owned_cleanup_ignores_already_exited_process():
    process = Mock()
    process.poll.return_value = 0
    stop_owned(process)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_owned_cleanup_terminates_only_supplied_process():
    process = Mock()
    process.poll.return_value = None
    stop_owned(process)
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=15)
    process.kill.assert_not_called()


def test_owned_cleanup_escalates_only_after_timeout():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired('synthetic-owned-process', 15), 0]
    stop_owned(process)
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2


def test_lifecycle_failure_cannot_be_marked_passed():
    with pytest.raises(RuntimeError, match='synthetic failure'):
        require(False, 'synthetic failure')


@pytest.mark.parametrize('status', ['passed', 'failed'])
def test_lifecycle_report_controls_exit_and_isolates_environment(monkeypatch, tmp_path, status):
    import instinct

    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'site-packages' / 'instinct' / '__init__.py'))
    monkeypatch.setattr(run_lifecycle_acceptance, 'os', SimpleNamespace(
        name='posix', environ={'INSTINCT_DATA_DIR': 'live', 'CE_DATA_DIR': 'rc', 'PYTHONPATH': 'checkout'},
    ))
    monkeypatch.setattr(run_lifecycle_acceptance.tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path))

    async def synthetic_run(root, environment, report):
        assert environment['INSTINCT_DATA_DIR'] == str(tmp_path / 'data')
        assert environment['CE_DATA_DIR'] == str(tmp_path / 'data')
        assert 'PYTHONPATH' not in environment
        report['status'] = status

    monkeypatch.setattr(run_lifecycle_acceptance, 'run', synthetic_run)
    if status == 'failed':
        with pytest.raises(SystemExit) as error:
            main(['--confirm-synthetic-lifecycle'])
        assert error.value.code == 1
    else:
        main(['--confirm-synthetic-lifecycle'])
    assert json.loads((tmp_path / 'lifecycle.json').read_text())['status'] == status
