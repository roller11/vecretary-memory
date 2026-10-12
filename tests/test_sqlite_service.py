# SPDX-License-Identifier: MPL-2.0
"""Isolated synthetic MCP transport and real loopback HTTP integration."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

import instinct
from instinct import fallback_store, mcp_server, nudge, session_timer
from instinct.mandatory_lifecycle import MandatoryLifecycle
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.service_binding import bound_store
from instinct.sqlite_queries import BoundQueryEmbedder
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_runtime import SQLiteSegmentAdapter
from instinct.sqlite_service import SQLiteDashboardHandler, SQLiteDashboardServer, SQLiteService
from instinct.sqlite_store import SQLiteRecordStore
from instinct.sqlite_writes import BoundDocumentEmbedder


IDENTITY = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')


def test_sqlite_entrypoint_import_does_not_load_legacy_chroma_backend():
    code = (
        'import sys; import instinct; import instinct.sqlite_service; '
        'import instinct.dashboard; import instinct.mcp_server; '
        'assert "instinct.store" not in sys.modules; '
        'assert "instinct.chroma_writer_lock" not in sys.modules; '
        'assert "chromadb" not in sys.modules'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_store_is_not_exposed_by_the_sqlite_package():
    code = (
        'import sys; import instinct; import instinct.sqlite_service; '
        'assert "instinct.store" not in sys.modules; '
        'assert not hasattr(instinct, "InstinctStore")'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


class SyntheticProvider:
    def name(self):
        return 'synthetic'

    def get_config(self):
        return {'provider': 'synthetic', 'model_name': 'fixed'}

    def supported_spaces(self):
        return ['cosine']

    def embed_query(self, queries):
        return [[1, 0] for query in queries]

    def embed_documents(self, documents):
        return [[1, 0.1] for document in documents]


def bindings():
    provider = SyntheticProvider()
    return {'query_embedder': BoundQueryEmbedder(IDENTITY, provider),
            'document_embedder': BoundDocumentEmbedder(IDENTITY, provider)}


def seed(root):
    root.chmod(0o700)
    session_timer.reset_session(root)
    with SQLiteRecordStore.create(root / 'sqlite', IDENTITY) as repository:
        adapter = SQLiteSegmentAdapter(repository, persist_dir=root, **bindings())
        adapter.add(ScoredSegment(segment=Segment(content='Synthetic vector recall activity timer test note',
            stored_id='ordinary', segment_type=SegmentType.CONTEXT), importance_score=80,
            concept_title='Synthetic ordinary memory',
            memory_identifiers=['vector recall', 'activity timer', 'sqlite']))
        adapter.add(ScoredSegment(segment=Segment(content='Synthetic test instruction only',
            stored_id='mandatory', segment_type=SegmentType.STANDING_INSTRUCTION),
            importance_score=100, concept_title='Synthetic instruction'))


@pytest.fixture
def service(tmp_path, monkeypatch):
    seed(tmp_path)
    monkeypatch.setattr(instinct, '_default_persist_dir', lambda: pytest.fail('Default storage path used'))
    with SQLiteService(tmp_path / 'sqlite', IDENTITY, persist_dir=tmp_path,
                       read_only=False, **bindings()) as instance:
        yield instance


def request(service, path, *, method='GET', body=None, headers=None):
    if service._http is None:
        service.start_dashboard()
    port = service._http.server_port
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    fields = {'Host': f'127.0.0.1:{port}'}
    if method == 'POST':
        fields |= {'X-Instinct-Capability': service._http.capability_token,
                   'Content-Type': 'application/json'}
    fields.update(headers or {})
    payload = json.dumps(body if body is not None else {}) if method == 'POST' else None
    try:
        connection.request(method, path, body=payload, headers=fields)
        response = connection.getresponse()
        raw = response.read()
        result = raw.decode() if response.getheader('Content-Type', '').startswith('text/html') else json.loads(raw)
        return response.status, result, dict(response.getheaders())
    finally:
        connection.close()


async def tool(service, name, arguments):
    return '\n'.join(item.text for item in await service.call(name, arguments))


@pytest.mark.parametrize('route, asset_name', [
    ('/assets/vecretary-logo.png', 'vecretary-logo-512.png'),
    ('/assets/vecretary-logo-256.png', 'vecretary-logo-256.png'),
])
def test_sqlite_dashboard_serves_only_the_packaged_logo_asset(service, route, asset_name):
    from importlib import resources
    service.start_dashboard()
    port = service._http.server_port
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    try:
        connection.request('GET', route)
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader('Content-Type') == 'image/png'
        expected = resources.files('instinct').joinpath('assets', asset_name).read_bytes()
        assert response.read() == expected
    finally:
        connection.close()
    assert request(service, '/assets/unknown.png')[0] == 501
    assert request(service, '/assets/../sqlite/memory.sqlite3')[0] == 501
    assert request(service, route, headers={'Origin': 'https://foreign.invalid'})[0] == 403


@pytest.mark.asyncio
async def test_mcp_dispatch_reads_saves_scores_and_checks_authority(service):
    response = await tool(service, 'instinct_get_context', {'query': 'vector recall', 'new_session': True})
    assert 'Found' in response
    assert 'Error' not in response
    response = await tool(service, 'instinct_save_memory', {
        'title': 'New synthetic note', 'content': 'Distinct synthetic service save', 'importance': 75})
    assert 'Saved memory segment' in response
    saved_id = response.split('ID: ')[1].splitlines()[0]
    assert 'Distinct synthetic' in await tool(service, 'instinct_get', {'segment_id': saved_id})
    assert 'Updated score' in await tool(service, 'instinct_update_score', {'segment_id': saved_id, 'new_score': 60})
    assert service.inspect(lambda store: store.get(saved_id).importance_score) == 60
    assert 'Blocked' in await tool(service, 'instinct_update_score', {'segment_id': saved_id, 'new_score': 93})
    assert 'BLOCKED' in await tool(service, 'instinct_update_score', {'segment_id': 'mandatory', 'new_score': 60})
    recycled = await tool(service, 'instinct_delete_memory', {'segment_id': saved_id})
    assert 'recoverable recycle bin' in recycled
    assert service.inspect(lambda store: store.is_recycled(saved_id))
    assert 'BLOCKED' in await tool(service, 'instinct_delete_memory', {'segment_id': 'mandatory'})
    assert not service.inspect(lambda store: store.is_recycled('mandatory'))
    assert service.inspect(lambda store: store.restore(saved_id)) is not None
    assert not service.inspect(lambda store: store.is_recycled(saved_id))
    assert 'Found' in await tool(service, 'instinct_search', {'query': 'synthetic', 'min_similarity': 0})


@pytest.mark.asyncio
async def test_invalid_mcp_input_does_not_tick_or_mutate(service):
    before = service.inspect(lambda store: nudge.get_current_turn(store))
    assert 'invalid' in await tool(service, 'instinct_save_memory', {'content': 'fixture', 'unknown': True})
    assert 'invalid' in await tool(service, 'absent_tool', {})
    assert service.inspect(lambda store: nudge.get_current_turn(store)) == before


@pytest.mark.asyncio
async def test_legacy_queue_is_never_examined_by_isolated_dispatch(service, monkeypatch):
    monkeypatch.setattr(fallback_store, 'pending_count', lambda: pytest.fail('Legacy pending queue inspected'))
    monkeypatch.setattr(fallback_store, 'sync_pending', lambda *args, **kwargs: pytest.fail('Legacy replay invoked'))
    response = await tool(service, 'instinct_get', {'segment_id': 'ordinary'})
    assert 'Synthetic ordinary memory' in response


@pytest.mark.asyncio
async def test_each_mcp_connection_restores_its_own_pending_mandatory(service):
    first, second = MandatoryLifecycle(), MandatoryLifecycle()
    await service.call('instinct_get', {'segment_id': 'ordinary'}, lifecycle=first)
    service.inspect(lambda store: store.update_score('mandatory', 79))
    await service.call('instinct_get', {'segment_id': 'ordinary'}, lifecycle=first)
    assert service.inspect(lambda store: store.get('mandatory').importance_score) == 79
    await service.call('instinct_get', {'segment_id': 'ordinary'}, lifecycle=second)
    assert service.inspect(lambda store: store.get('mandatory').importance_score) == 99


@pytest.mark.asyncio
async def test_timer_failure_aborts_sqlite_save_without_embedding(service, monkeypatch):
    def fail_tick(*args, **kwargs):
        raise RuntimeError('Synthetic failed clock commit')

    monkeypatch.setattr(nudge, 'tick_nudge_session', fail_tick)
    count = service.inspect(lambda store: store.count())
    response = await tool(service, 'instinct_save_memory', {'content': 'fixture', 'title': 'Failed clock fixture'})
    assert 'Error executing' in response
    assert service.inspect(lambda store: store.count()) == count


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_cancel_confirmable_inflight_write(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = mcp_server._handle_update_score

    async def delayed(store, arguments):
        entered.set()
        assert release.wait(5)
        return await original(store, arguments)

    monkeypatch.setattr(mcp_server, '_handle_update_score', delayed)
    task = asyncio.create_task(service.call('instinct_update_score', {'segment_id': 'ordinary', 'new_score': 61}))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    assert service.inspect(lambda store: store.get('ordinary').importance_score) == 61


def test_request_backlog_is_bounded(service):
    entered, release = threading.Event(), threading.Event()

    def blocked(store):
        entered.set()
        assert release.wait(5)

    futures = [service._submit(blocked)]
    try:
        assert entered.wait(5)
        futures.extend(service._submit(lambda store: store.count()) for index in range(15))
        with pytest.raises(RuntimeError, match='capacity'):
            service._submit(lambda store: store.count())
    finally:
        release.set()
    for future in futures:
        future.result(timeout=5)


def test_failed_owner_startup_does_not_change_existing_service(service):
    root = service.inspect(lambda store: store.persist_dir)
    with pytest.raises(RuntimeError):
        SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root, read_only=False, **bindings())
    assert service.inspect(lambda store: store.count()) == 2


def test_closed_service_and_invalid_port_fail_without_default_fallback(service):
    with pytest.raises(ValueError):
        service.start_dashboard(True)
    service.close()
    with pytest.raises(RuntimeError, match='closed'):
        service.inspect(lambda store: store.count())


def test_dashboard_is_opt_in_and_server_rejects_non_loopback_bind(service):
    assert service._http is None
    with pytest.raises(ValueError, match='127.0.0.1 only'):
        SQLiteDashboardServer(('0.0.0.0', 0), SQLiteDashboardHandler)
    service.start_dashboard(0)
    assert service._http.server_address[0] == '127.0.0.1'


def test_http_dashboard_reads_html_and_readiness_without_chroma(service):
    status, html, headers = request(service, '/')
    assert status == 200
    assert 'Experimental SQLite integration' in html
    assert 'handoff-to-dashboard' in html and 'handoff-to-mcp' in html
    assert 'Use the controls above to transfer write access.' in html
    assert 'Separate writer handoff is not available.' not in html
    assert '.mode-banner .status { font-size: 11px; color: inherit; }' in html
    assert 'id="recycle-link" aria-label="Recycle bin"' in html
    assert 'border-radius: 10px' in html
    assert '--bg: #d9dcda' in html
    assert '--accent: #176331' in html
    assert 'Content-Security-Policy' in headers
    status, ready, _ = request(service, '/api/ready')
    assert status == 200 and ready['ok']
    assert ready['checks']['backend'] == 'sqlite'
    assert ready['checks']['runtime_ready'] is False
    assert request(service, '/api/health')[1]['embedding_probed'] is False
    assert request(service, '/api/stats')[1]['mandatory_count'] == 1
    assert request(service, '/api/list')[1]['count'] == 2
    mode = request(service, '/api/mode')[1]
    assert mode['handoff_supported'] is True and mode['dashboard_control'] is True
    assert request(service, '/api/decay/config')[0] == 200
    assert request(service, '/api/turn')[0] == 200
    assert request(service, '/api/mode?unknown=1')[0] == 400
    assert request(service, '/api/search?q=synthetic')[0] == 200


@pytest.mark.parametrize('route', ['/', '/decay', '/score', '/modifier'])
def test_handoff_controls_render_on_every_dashboard_page(service, route):
    status, html, _ = request(service, route)
    assert status == 200
    assert 'handoff-to-dashboard' in html and 'handoff-to-mcp' in html
    assert '/assets/vecretary-logo-256.png' in html
    assert '--bg: #d9dcda' in html
    assert '--accent: #176331' in html
    assert 'dashboardApplySharedTheme' in html
    assert 'standalone-theme-toggle' not in html
    if route != '/':
        assert 'body header.secondary-app-bar .primary-nav { gap: 5px; }' in html
        assert 'body header.secondary-app-bar .primary-nav a {' in html
        assert "header.classList.add('app-bar', 'secondary-app-bar')" in html
        assert "nav.className = 'primary-nav'" in html
        assert 'dashboardToggleSharedTheme' in html


def test_http_writes_recycle_restore_config_and_pop(service):
    status, result, _ = request(service, '/api/add', method='POST', body={
        'title': 'Dashboard synthetic', 'content': 'Dashboard service vector recall fixture', 'score': 80})
    assert status == 200
    record_id = result['id']
    assert request(service, '/api/update_score', method='POST', body={'id': record_id, 'score': 70})[0] == 200
    assert request(service, '/api/recycle', method='POST', body={'id': record_id})[0] == 200
    assert request(service, '/api/recycled')[1]['count'] == 1
    assert request(service, '/api/restore', method='POST', body={'id': record_id})[0] == 200
    assert request(service, '/api/recycled')[1]['count'] == 0
    assert request(service, '/api/heartbeat', method='POST')[0] == 200
    assert request(service, '/api/decay/config', method='POST', body={'session_idle_minutes': 12})[0] == 200
    assert request(service, '/api/decay/config')[1]['session_idle_minutes'] == 12
    assert request(service, '/api/access', method='POST', body={'ids': ['ordinary']})[0] == 200
    assert service.inspect(lambda store: store.get('ordinary').access_count) >= 1
    assert request(service, '/api/search_with_pop', method='POST', body={
        'q': 'vector recall activity timer sqlite', 'n': 10, 'apply_pops': True, 'track_access': True})[0] == 200
    assert service.inspect(lambda store: store.get('ordinary').keyword_hits_total) >= 3


def test_mandatory_nudge_reappears_at_99_after_interval_and_access_drops_to_79(service):
    reminder = ScoredSegment(
        segment=Segment(content='Synthetic recurring mandatory reminder', stored_id='nudge-reminder',
                        segment_type=SegmentType.STANDING_INSTRUCTION),
        importance_score=99,
        concept_title='Synthetic recurring reminder',
    )
    nudge.set_nudge_interval(reminder, 2)
    service.inspect(lambda store: store.add(reminder))

    first = asyncio.run(tool(service, 'instinct_get_mandatory', {}))
    assert 'Synthetic recurring reminder' in first
    assert service.inspect(lambda store: store.get('nudge-reminder').importance_score) == 79

    asyncio.run(tool(service, 'instinct_get', {'segment_id': 'ordinary'}))
    second = asyncio.run(tool(service, 'instinct_get_mandatory', {}))
    assert 'Synthetic recurring reminder' in second
    assert service.inspect(lambda store: store.get('nudge-reminder').importance_score) == 79


def test_dashboard_mcp_handoff_pauses_and_resumes_tool_calls(service):
    service.start_dashboard()
    with service._state_lock:
        service._stdio_active = True
        service._mode = 'mcp'

    mode = request(service, '/api/mode')[1]
    assert mode['mcp_available'] is True and mode['dashboard_control'] is False

    status, taken, _ = request(service, '/api/mode/take_control', method='POST')
    assert status == 200 and taken['acquired'] and taken['mcp_paused']
    assert service.dashboard_control is True
    paused = asyncio.run(tool(service, 'instinct_get_mandatory', {}))
    assert 'paused' in paused.lower()
    paused_write = asyncio.run(tool(service, 'instinct_update_score', {
        'segment_id': 'ordinary', 'new_score': 1,
    }))
    assert 'paused' in paused_write.lower()
    assert service.inspect(lambda store: store.get('ordinary').importance_score) == 80

    status, updated, _ = request(service, '/api/update_score', method='POST',
                                 body={'id': 'ordinary', 'score': 70})
    assert status == 200 and updated['score'] == 70

    status, returned, _ = request(service, '/api/mode/return_control', method='POST')
    assert status == 200 and returned['released'] and returned['mcp_resumed']
    assert 'mcp_started' not in returned
    assert service.dashboard_control is False
    resumed = asyncio.run(tool(service, 'instinct_get', {'segment_id': 'ordinary'}))
    assert 'Synthetic ordinary memory' in resumed


def test_dashboard_handoff_requires_connected_mcp(tmp_path, monkeypatch):
    seed(tmp_path)
    monkeypatch.setattr(instinct, '_default_persist_dir', lambda: pytest.fail('Default storage path used'))
    with SQLiteService(tmp_path / 'sqlite', IDENTITY, persist_dir=tmp_path,
                       read_only=True, **bindings()) as observer:
        observer.start_dashboard()
        status, result, _ = request(observer, '/api/mode/take_control', method='POST')
        assert status == 409 and not result['acquired']
        assert observer.inspect(lambda store: store._read_only)


def test_observer_queue_capture_disables_dashboard_handoff(tmp_path, monkeypatch):
    seed(tmp_path)
    monkeypatch.setattr(instinct, '_default_persist_dir', lambda: pytest.fail('Default storage path used'))
    with SQLiteService(tmp_path / 'sqlite', IDENTITY, persist_dir=tmp_path,
                       read_only=True, queue_writes=True, **bindings()) as observer:
        observer.start_dashboard()
        with observer._state_lock:
            observer._stdio_active = True
            observer._mode = 'mcp'
        mode = request(observer, '/api/mode')[1]
        assert mode['handoff_supported'] is False
        status, result, _ = request(observer, '/api/mode/take_control', method='POST')
        assert status == 409 and not result['acquired']
        assert observer.inspect(lambda store: store._read_only)


@pytest.mark.asyncio
async def test_stdio_lifecycle_exposes_and_clears_mcp_handoff_receiver(service, monkeypatch):
    from contextlib import asynccontextmanager
    import mcp.server
    import mcp.server.stdio

    class FakeServer:
        def __init__(self, name, *, lifespan):
            self.request_context = None

        def list_tools(self):
            return lambda handler: handler

        def call_tool(self):
            return lambda handler: handler

        def create_initialization_options(self):
            return None

        async def run(self, reader, writer, options):
            assert service._stdio_active
            assert service._mode == 'mcp'

    @asynccontextmanager
    async def fake_stdio_server():
        yield object(), object()

    monkeypatch.setattr(mcp.server, 'Server', FakeServer)
    monkeypatch.setattr(mcp.server.stdio, 'stdio_server', fake_stdio_server)
    await service.run_stdio()
    assert not service._stdio_active
    assert service._mode == 'dashboard'


def test_dashboard_handoff_watchdog_returns_control_to_mcp(service):
    service.start_dashboard()
    with service._state_lock:
        service._stdio_active = True
        service._mode = 'mcp'
    assert request(service, '/api/mode/take_control', method='POST')[0] == 200
    deadline = service._dashboard_last_heartbeat + 31
    assert service._expire_dashboard_lease(now=deadline)
    assert not service.dashboard_control
    assert request(service, '/api/mode')[1]['mode'] == 'mcp'


def test_handoff_rejects_nonempty_body_without_changing_control(service):
    service.start_dashboard()
    with service._state_lock:
        service._stdio_active = True
        service._mode = 'mcp'
    status, _, _ = request(service, '/api/mode/take_control', method='POST', body={'id': 'ordinary'})
    assert status == 400
    assert request(service, '/api/mode')[1]['mode'] == 'mcp'


def test_cross_origin_cannot_return_dashboard_control(service):
    service.start_dashboard()
    with service._state_lock:
        service._stdio_active = True
        service._mode = 'mcp'
    assert request(service, '/api/mode/take_control', method='POST')[0] == 200
    assert request(service, '/api/mode/return_control', method='POST',
                   headers={'Origin': 'https://attacker.invalid'})[0] == 403
    assert service.dashboard_control
    assert request(service, '/api/mode/return_control', method='POST')[0] == 200


def test_read_only_mcp_session_promotes_only_after_exclusive_lock_is_acquired(tmp_path, monkeypatch):
    seed(tmp_path)
    monkeypatch.setattr(instinct, '_default_persist_dir', lambda: pytest.fail('Default storage path used'))
    with SQLiteService(tmp_path / 'sqlite', IDENTITY, persist_dir=tmp_path,
                       read_only=True, **bindings()) as observer:
        observer.start_dashboard()
        with observer._state_lock:
            observer._stdio_active = True
            observer._mode = 'mcp'
        assert observer.inspect(lambda store: store._read_only)
        status, result, _ = request(observer, '/api/mode/take_control', method='POST')
        assert status == 200 and result['acquired']
        assert not observer.inspect(lambda store: store._read_only)
        assert request(observer, '/api/update_score', method='POST',
                       body={'id': 'ordinary', 'score': 70})[0] == 200
        status, result, _ = request(observer, '/api/mode/return_control', method='POST')
        assert status == 200 and result['mcp_resumed']


def test_handoff_refuses_competing_sqlite_writer_without_touching_it(tmp_path, monkeypatch):
    seed(tmp_path)
    monkeypatch.setattr(instinct, '_default_persist_dir', lambda: pytest.fail('Default storage path used'))
    with SQLiteRecordStore.open_writable(tmp_path / 'sqlite', IDENTITY) as external_writer:
        with SQLiteService(tmp_path / 'sqlite', IDENTITY, persist_dir=tmp_path,
                           read_only=True, **bindings()) as observer:
            observer.start_dashboard()
            with observer._state_lock:
                observer._stdio_active = True
                observer._mode = 'mcp'
            status, result, _ = request(observer, '/api/mode/take_control', method='POST')
            assert status == 409 and not result['acquired']
            assert observer.inspect(lambda store: store._read_only)
            assert external_writer.count() == 2


def test_stats_tracks_recycled_identifiers_access_and_pop_without_embeddings(service):
    assert service.inspect(lambda store: store.stats())['total_segments'] == 2
    service.inspect(lambda store: store.record_access('ordinary'))
    assert service.inspect(lambda store: store.stats())['accessed_count'] == 1
    service.inspect(lambda store: store.recycle('ordinary'))
    stats = service.inspect(lambda store: store.stats())
    assert stats['total_segments'] == 2
    assert stats['high_priority_count'] == 1
    assert stats['identifier_terms'] == service.inspect(lambda store: len(set(store.get('mandatory').memory_identifiers)))


@pytest.mark.parametrize('path', ['/api/delete', '/api/purge', '/api/admin/reset', '/api/mode/edit'])
def test_unsupported_http_operations_are_fail_closed(service, path):
    before = service.inspect(lambda store: store.count())
    assert request(service, path, method='POST', body={'id': 'ordinary'})[0] == 501
    assert service.inspect(lambda store: store.count()) == before


@pytest.mark.parametrize('headers', [
    {'Host': 'attacker.invalid'}, {'Origin': 'https://attacker.invalid'},
    {'Sec-Fetch-Site': 'cross-site'}, {'X-Instinct-Capability': 'invalid'},
])
@pytest.mark.parametrize('path', [
    '/api/update_score', '/api/mode/take_control', '/api/mode/return_control',
])
def test_http_mutation_boundary_rejects_unauthorized_requests(service, headers, path):
    assert request(service, path, method='POST',
                   body={'id': 'ordinary', 'score': 1}, headers=headers)[0] == 403
    assert service.inspect(lambda store: store.get('ordinary').importance_score) == 80


@pytest.mark.asyncio
async def test_concurrent_mcp_and_http_share_one_owner_thread(service):
    service.start_dashboard()
    response, http_result = await asyncio.gather(
        tool(service, 'instinct_get', {'segment_id': 'ordinary'}),
        asyncio.to_thread(request, service, '/api/heartbeat', method='POST'))
    assert 'Synthetic ordinary memory' in response
    assert http_result[0] == 200
    with ThreadPoolExecutor(max_workers=8) as pool:
        threads = list(pool.map(lambda index: service.inspect(lambda store: threading.get_ident()), range(24)))
    assert len(set(threads)) == 1
    assert threads[0] != threading.get_ident()
    assert bound_store() is None


def test_owner_restart_preserves_clock_access_vectors_and_pop(service):
    root = service.inspect(lambda store: store.persist_dir)
    now = datetime(2026, 10, 7, 10)
    service.inspect(lambda store: session_timer.tick_session(root, now=now))
    service.inspect(lambda store: session_timer.tick_session(root, now=now + timedelta(seconds=20)))
    service.inspect(lambda store: store.record_access('ordinary', now=now + timedelta(seconds=20)))
    saved = service.inspect(lambda store: store.get('ordinary').to_storage_dict())
    seconds = service.inspect(lambda store: session_timer.state().accumulated_session_seconds)
    service.close()
    with SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root, read_only=False, **bindings()) as reopened:
        assert reopened.inspect(lambda store: store.get('ordinary').to_storage_dict()) == saved
        assert reopened.inspect(lambda store: session_timer.state().accumulated_session_seconds) == seconds
        reopened.inspect(lambda store: session_timer.tick_session(root, now=now + timedelta(days=30)))
        assert reopened.inspect(lambda store: session_timer.state().accumulated_session_seconds) == seconds


@pytest.mark.asyncio
async def test_observer_never_queues_writes_or_changes_sidecars(tmp_path):
    seed(tmp_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    with SQLiteService(tmp_path / 'sqlite', IDENTITY, persist_dir=tmp_path,
                       read_only=True, **bindings()) as observer:
        assert 'read-only' in await tool(observer, 'instinct_save_memory', {'content': 'fixture', 'title': 'Observer fixture'})
        assert 'read-only' in await tool(observer, 'instinct_update_score', {'segment_id': 'ordinary', 'new_score': 70})
        assert 'Mandatory renewal is pending' in await tool(observer, 'instinct_get_mandatory', {})
        assert request(observer, '/api/heartbeat', method='POST')[0] == 409
        assert request(observer, '/api/decay/config', method='POST', body={'session_idle_minutes': 12})[0] == 409
        assert request(observer, '/api/list')[0] == 200
        assert request(observer, '/api/decay/config')[0] == 200
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert after == before


@pytest.mark.asyncio
async def test_observer_queues_recycle_and_owner_replay_keeps_it_recoverable(service, tmp_path):
    with SQLiteService(tmp_path / 'sqlite', IDENTITY, persist_dir=tmp_path,
                       read_only=True, queue_writes=True, **bindings()) as observer:
        result = await tool(observer, 'instinct_delete_memory', {'segment_id': 'ordinary'})
        assert 'Queued to SQLite FALLBACK' in result
        assert 'BLOCKED' in await tool(observer, 'instinct_delete_memory', {'segment_id': 'mandatory'})
        pending = fallback_store.list_pending(str(tmp_path))
        assert len(pending) == 1 and pending[0]['operation'] == 'delete_memory'
        assert not service.inspect(lambda store: store.is_recycled('ordinary'))

    from instinct.sqlite_pending import replay
    replayed = service.inspect(replay)
    assert replayed['synced'] == 1 and not replayed['pending']
    assert service.inspect(lambda store: store.is_recycled('ordinary'))
    record = service.inspect(lambda store: store.get('ordinary'))
    assert record is not None and record.concept_title == 'Synthetic ordinary memory'
    assert record.segment.metadata['recycled_by'] == 'agent'
    receipts = record.segment.metadata['sqlite_pending_receipts']
    assert len(receipts) == 1 and len(receipts[0]) == 64
    assert service.inspect(lambda store: store.restore('ordinary')) is not None
    assert not service.inspect(lambda store: store.is_recycled('ordinary'))
    assert service.inspect(lambda store: store.get('ordinary')).segment.metadata['sqlite_pending_receipts'] == receipts
    assert fallback_store.list_pending(str(tmp_path)) == []


def run_synthetic_stdio(root):
    with SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root, read_only=False, **bindings()) as instance:
        asyncio.run(instance.run_stdio())


@pytest.mark.asyncio
async def test_real_stdio_mcp_transport_initializes_lists_and_saves(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    seed(tmp_path)
    code = 'import runpy; from pathlib import Path; runpy.run_path(' + repr(str(Path(__file__).resolve())) + ")[\"run_synthetic_stdio\"](Path(" + repr(str(tmp_path)) + '))'
    parameters = StdioServerParameters(command=sys.executable, args=['-c', code])
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as connection:
            initialized = await connection.initialize()
            assert initialized.serverInfo.name == 'vecretary-sqlite-experimental'
            tools = await connection.list_tools()
            assert 'instinct_delete_memory' in {item.name for item in tools.tools}
            result = await connection.call_tool('instinct_get_context', {'query': 'vector recall', 'new_session': True})
            assert not result.isError
            assert 'Found' in result.content[0].text
            saved = await connection.call_tool('instinct_save_memory', {
                'content': 'Transport synthetic note', 'title': 'Transport unique', 'importance': 75})
            assert 'Saved memory segment' in saved.content[0].text
            saved_id = saved.content[0].text.split('ID: ')[1].splitlines()[0]
            read = await connection.call_tool('instinct_get', {'segment_id': saved_id})
            assert 'Transport synthetic note' in read.content[0].text


@pytest.mark.asyncio
async def test_explicit_cli_stdio_without_bindings_reads_but_refuses_new_text(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    seed(tmp_path)
    identity_path = tmp_path / 'embedding-identity.json'
    identity_path.write_text(json.dumps(IDENTITY.to_dict()), encoding='utf-8')
    parameters = StdioServerParameters(command=sys.executable, args=[
        '-m', 'instinct.sqlite_service', '--database-dir', str(tmp_path / 'sqlite'),
        '--data-dir', str(tmp_path), '--identity', str(identity_path)])
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as connection:
            await connection.initialize()
            read = await connection.call_tool('instinct_get', {'segment_id': 'ordinary'})
            assert 'Synthetic ordinary memory' in read.content[0].text
            saved = await connection.call_tool('instinct_save_memory', {
                'title': 'Unavailable embedder', 'content': 'No unbound embedding fallback'})
            assert 'Error executing' in saved.content[0].text
    with SQLiteRecordStore.open_read_only(tmp_path / 'sqlite', IDENTITY) as reader:
        assert reader.count() == 2
