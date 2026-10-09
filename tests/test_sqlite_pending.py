# SPDX-License-Identifier: MPL-2.0
"""Synthetic opt-in SQLite fallback, receipt and recovery acceptance."""

import asyncio
from datetime import datetime
import json
import http.client
import os
from pathlib import Path
import subprocess
import sys

import pytest

from instinct import fallback_store, mcp_server, session_timer, sqlite_pending
from instinct.active_time import elapsed_hours
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.private_storage import private_file_lock, write_private_text
from instinct.sqlite_backup import restore_new, snapshot_offline
from instinct.sqlite_queries import BoundQueryEmbedder
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_runtime import SQLiteSegmentAdapter
from instinct.sqlite_service import SQLiteService
from instinct.sqlite_store import SQLiteRecordStore
from instinct.sqlite_writes import BoundDocumentEmbedder


IDENTITY = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')


class Provider:
    fail = False

    def name(self):
        return 'synthetic'

    def get_config(self):
        return {'provider': 'synthetic', 'model_name': 'fixed'}

    def supported_spaces(self):
        return ['cosine']

    def embed_query(self, queries):
        return [[1, 0] for query in queries]

    def embed_documents(self, documents):
        if self.fail:
            raise RuntimeError('synthetic provider failure')
        return [[1, 0] for document in documents]


def bindings(provider=None):
    provider = provider or Provider()
    return {'query_embedder': BoundQueryEmbedder(IDENTITY, provider),
            'document_embedder': BoundDocumentEmbedder(IDENTITY, provider)}


@pytest.fixture
def root(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    monkeypatch.setattr(mcp_server, '_highpri_write_timestamps', [])
    session_timer.reset_session(tmp_path)
    with SQLiteRecordStore.create(tmp_path / 'sqlite', IDENTITY) as repository:
        adapter = SQLiteSegmentAdapter(repository, persist_dir=tmp_path, **bindings())
        adapter.add(ScoredSegment(segment=Segment(content='Synthetic existing ordinary memory',
            segment_type=SegmentType.CONTEXT, stored_id='ordinary'), importance_score=70,
            concept_title='Existing ordinary'))
        adapter.add(ScoredSegment(segment=Segment(content='Synthetic mandatory test only',
            segment_type=SegmentType.STANDING_INSTRUCTION, stored_id='mandatory'), importance_score=100,
            concept_title='Existing mandatory'))
    return tmp_path


def owner(root, **options):
    return SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root, read_only=False, **options)


def observer(root):
    return SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root, read_only=True, queue_writes=True)


def call(service, name, arguments):
    return '\n'.join(item.text for item in asyncio.run(service.call(name, arguments)))


def capture(root, *, content='Synthetic queued test memory', title='Queued synthetic', importance=75):
    with observer(root) as service:
        result = call(service, 'instinct_save_memory', {'content': content, 'title': title, 'importance': importance})
        assert 'Queued to SQLite FALLBACK' in result, result
    return fallback_store.list_pending(str(root))[0]


def queued_id(record):
    return 'queued_' + sqlite_pending._receipt(record)


def test_capture_needs_no_provider_and_only_changes_queue(root):
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    record = capture(root)
    assert record['schema_version'] == 3
    assert record['sqlite_identity'] == IDENTITY.to_dict()
    assert record['active_clock']['clock_id'] == session_timer.state(root).clock_id
    for relative, payload in before.items():
        assert (root / relative).read_bytes() == payload
    with SQLiteRecordStore.open_read_only(root / 'sqlite', IDENTITY) as repository:
        assert repository.count() == 2


def test_owner_startup_replay_retains_timestamp_vector_and_activity_anchor(root):
    record = capture(root)
    with owner(root, replay_pending=True, **bindings()) as service:
        assert service.last_replay['synced'] == 1
        segment = service.inspect(lambda adapter: adapter.get(queued_id(record)))
        assert segment.segment.timestamp == datetime.fromisoformat(record['created_at'])
        assert segment.active_time_anchors['created']['seconds'] == record['active_clock']['seconds']
        assert segment.embedding == [1, 0]
        assert segment.initial_score == 75 and segment.access_count == 0
    assert not Path(record['_path']).exists()


def test_existing_owner_replays_on_next_mcp_call(root):
    with owner(root, replay_pending=True, **bindings()) as service:
        record = capture(root)
        assert service.inspect(lambda adapter: adapter.count()) == 2
        assert 'Existing ordinary' in call(service, 'instinct_get', {'segment_id': 'ordinary'})
        assert service.inspect(lambda adapter: adapter.count()) == 3
    assert not Path(record['_path']).exists()


def test_observer_capture_does_not_reset_owner_monotonic_clock(root):
    with owner(root, **bindings()) as service:
        call(service, 'instinct_get', {'segment_id': 'ordinary'})
        timer = session_timer._store_for(root)
        before = timer._last_monotonic
        assert before is not None
        capture(root)
        assert timer._last_monotonic == before


def test_score_capture_and_replay_preserve_initial_score_and_pop_state(root):
    with owner(root, **bindings()) as service:
        service.inspect(lambda adapter: adapter.record_access('ordinary'))
        before = service.inspect(lambda adapter: adapter.get('ordinary'))
    with observer(root) as service:
        assert 'Queued' in call(service, 'instinct_update_score', {
            'segment_id': 'ordinary', 'new_score': 65, 'reason': 'Synthetic delayed update'})
    record = fallback_store.list_pending(str(root))[0]
    with owner(root, replay_pending=True) as service:
        after = service.inspect(lambda adapter: adapter.get('ordinary'))
        assert after.importance_score == 65 and after.initial_score == before.initial_score
        assert after.access_count == before.access_count and after.popped_at == before.popped_at
        assert after.promotions_applied == before.promotions_applied
        assert after.pop_anchor_at == datetime.fromisoformat(record['created_at'])
        assert after.active_time_anchors['pop_anchor_at']['seconds'] == record['active_clock']['seconds']


@pytest.mark.parametrize('arguments', [
    {'content': 'synthetic', 'title': 'restricted', 'importance': 93},
    {'content': 'synthetic', 'title': 'restricted', 'segment_type': 'standing_instruction'},
    {'content': 'short', 'title': 'restricted', 'importance': 85},
    {'content': 'synthetic', 'title': 'Existing ordinary'},
    {'content': chr(233) * 40000, 'title': 'Too large', 'importance': 75},
])
def test_capture_authority_and_save_guards(root, arguments):
    with observer(root) as service:
        assert 'BLOCKED' in call(service, 'instinct_save_memory', arguments)
    assert not (root / 'fallback_pending').exists()


@pytest.mark.parametrize('target', ['mandatory', 'missing'])
def test_score_capture_protects_targets(root, target):
    with observer(root) as service:
        assert 'BLOCKED' in call(service, 'instinct_update_score', {'segment_id': target, 'new_score': 70})
    assert not (root / 'fallback_pending').exists()


def test_legacy_and_deletion_queues_retained_without_replay(root):
    original = fallback_store.write_pending('delete_memory', str(root), delete_id='ordinary')
    capture(root)
    with owner(root, replay_pending=True, **bindings()) as service:
        assert service.last_replay['synced'] == 1 and service.last_replay['skipped'] == 1
        assert service.inspect(lambda adapter: adapter.get('ordinary')) is not None
        assert 'recoverable recycle bin' in call(service, 'instinct_delete_memory', {'segment_id': 'ordinary'})
        assert service.inspect(lambda adapter: adapter.is_recycled('ordinary'))
    assert Path(original).exists()


def test_legacy_dispatch_cannot_replay_sqlite_envelope(root):
    record = capture(root)
    class Legacy:
        persist_dir = root
        _read_only = False
        def add(self, segment):
            pytest.fail('SQLite queue sent to legacy storage')
    assert fallback_store.sync_pending(Legacy())['failed'] == 1
    assert Path(record['_path']).exists()


def test_provider_failure_blocks_later_queue_and_new_writes(root):
    first = capture(root)
    with observer(root) as service:
        assert 'Queued' in call(service, 'instinct_update_score', {'segment_id': 'ordinary', 'new_score': 60})
    provider = Provider()
    provider.fail = True
    with owner(root, replay_pending=True, **bindings(provider)) as service:
        assert service.last_replay['failed'] == 1 and service.last_replay['pending']
        assert service.inspect(lambda adapter: adapter.get('ordinary')).importance_score == 70
        assert 'not drained' in call(service, 'instinct_save_memory', {'content': 'synthetic', 'title': 'Later write'})
        provider.fail = False
        call(service, 'instinct_get', {'segment_id': 'ordinary'})
        assert service.last_replay['synced'] == 2 and not service.last_replay['pending']
        assert service.inspect(lambda adapter: adapter.get('ordinary')).importance_score == 60
    assert not Path(first['_path']).exists()


@pytest.mark.parametrize('field', ['identity', 'clock', 'future_seconds', 'authority'])
def test_unmatched_or_forged_queue_retained(root, field):
    record = capture(root)
    payload = {key: value for key, value in record.items() if not key.startswith('_')}
    if field == 'identity':
        payload['sqlite_identity']['model'] = 'different'
    elif field == 'clock':
        payload['active_clock']['clock_id'] = 'different'
    elif field == 'future_seconds':
        payload['active_clock']['seconds'] = 999999
    else:
        payload['authority'] = 'owner'
    write_private_text(record['_path'], json.dumps(payload))
    with owner(root, replay_pending=True, **bindings()) as service:
        assert service.inspect(lambda adapter: adapter.count()) == 2
        assert service.last_replay['failed'] + service.last_replay['skipped'] == 1
    assert Path(record['_path']).exists()


def test_commit_before_ack_failure_does_not_duplicate_save_or_reset_access(root, monkeypatch):
    record = capture(root)
    original = sqlite_pending.os.unlink
    def refuse(path):
        if str(path) == record['_path']:
            raise OSError('synthetic acknowledgement failure')
        return original(path)
    monkeypatch.setattr(sqlite_pending.os, 'unlink', refuse)
    with owner(root, replay_pending=True, **bindings()) as service:
        assert service.last_replay['failed'] == 1
        service.inspect(lambda adapter: adapter.record_access(queued_id(record)))
        changed = service.inspect(lambda adapter: adapter.repository.get_record(queued_id(record))).to_dict()
    monkeypatch.setattr(sqlite_pending.os, 'unlink', original)
    with owner(root, replay_pending=True) as service:
        assert service.last_replay['synced'] == 1
        assert service.inspect(lambda adapter: adapter.count()) == 3
        assert service.inspect(lambda adapter: adapter.repository.get_record(queued_id(record))).to_dict() == changed
    assert not Path(record['_path']).exists()


def test_score_receipt_prevents_reapplying_old_score_after_owner_edit(root, monkeypatch):
    with observer(root) as service:
        call(service, 'instinct_update_score', {'segment_id': 'ordinary', 'new_score': 60})
    record = fallback_store.list_pending(str(root))[0]
    original = sqlite_pending.os.unlink
    def refuse(path):
        if str(path) == record['_path']:
            raise OSError('synthetic acknowledgement failure')
        return original(path)
    monkeypatch.setattr(sqlite_pending.os, 'unlink', refuse)
    with owner(root, replay_pending=True) as service:
        service.inspect(lambda adapter: adapter.update_score('ordinary', 95))
        expected = service.inspect(lambda adapter: adapter.repository.get_record('ordinary')).to_dict()
    monkeypatch.setattr(sqlite_pending.os, 'unlink', original)
    with owner(root, replay_pending=True) as service:
        assert service.last_replay['synced'] == 1
        assert service.inspect(lambda adapter: adapter.repository.get_record('ordinary')).to_dict() == expected


def test_replay_rechecks_target_authority(root):
    with observer(root) as service:
        call(service, 'instinct_update_score', {'segment_id': 'ordinary', 'new_score': 60})
    with owner(root) as service:
        service.inspect(lambda adapter: adapter.update_score('ordinary', 95))
    with owner(root, replay_pending=True) as service:
        assert service.last_replay['failed'] == 1
        assert service.inspect(lambda adapter: adapter.get('ordinary')).importance_score == 95


def test_suppressed_owner_initial_score_is_protected_at_capture(root):
    with owner(root) as service:
        service.inspect(lambda adapter: adapter.repository.patch_metadata('ordinary', {
            'initial_score': 99, 'importance_score': 65}))
    with observer(root) as service:
        assert 'BLOCKED' in call(service, 'instinct_update_score', {'segment_id': 'ordinary', 'new_score': 60})
    assert not (root / 'fallback_pending').exists()


def exit_before_ack(root):
    sqlite_pending.os.unlink = lambda path: os._exit(9)
    with owner(root, replay_pending=True, **bindings()):
        raise AssertionError('Expected abrupt exit')


def test_abrupt_process_exit_after_commit_before_ack(root):
    record = capture(root)
    script = 'import runpy; from pathlib import Path; runpy.run_path(' + repr(str(Path(__file__).resolve())) + ')["exit_before_ack"](Path(' + repr(str(root)) + '))'
    result = subprocess.run([sys.executable, '-c', script], timeout=30, capture_output=True)
    assert result.returncode == 9, result.stderr.decode()
    assert Path(record['_path']).exists()
    with owner(root, replay_pending=True) as service:
        assert service.last_replay['synced'] == 1
        assert service.inspect(lambda adapter: adapter.count()) == 3
    assert not Path(record['_path']).exists()


def test_backup_restore_pending_and_committed_receipt(root, tmp_path):
    record = capture(root)
    parent = root.parent
    snapshot = snapshot_offline(root, parent / (root.name + '-snapshot'), IDENTITY)
    restored = restore_new(snapshot, parent / (root.name + '-restored'), IDENTITY)
    with owner(restored, replay_pending=True, **bindings()) as service:
        assert service.last_replay['synced'] == 1
    write_private_text(restored / 'fallback_pending/reappeared.json', json.dumps({
        key: value for key, value in record.items() if not key.startswith('_')}))
    with owner(restored, replay_pending=True) as service:
        assert service.last_replay['synced'] == 1
        assert service.inspect(lambda adapter: adapter.count()) == 3


def test_activity_elapsed_in_queue_is_not_reset_at_replay(root):
    record = capture(root, content='Synthetic important memory with sufficient content for high priority capture.', importance=85)
    state = session_timer.state(root)
    state.accumulated_session_seconds += 7200
    session_timer._store_for(root).save(state)
    with owner(root, replay_pending=True, **bindings()) as service:
        elapsed = service.inspect(lambda adapter: elapsed_hours(adapter.get(queued_id(record)), 'created', adapter.persist_dir))
        assert elapsed == 2
    assert len(mcp_server._highpri_write_timestamps) == 1


def test_queue_lock_contention_retains_record_and_no_write(root):
    record = capture(root)
    with private_file_lock(root / 'fallback_pending/.capture.lock', timeout=0):
        with owner(root, replay_pending=True, **bindings()) as service:
            assert service.last_replay['pending']
            assert service.inspect(lambda adapter: adapter.count()) == 2
    assert Path(record['_path']).exists()


@pytest.mark.parametrize('options', [
    {'read_only': False, 'queue_writes': True}, {'read_only': True, 'replay_pending': True},
    {'read_only': False, 'replay_pending': 1}, {'read_only': True, 'queue_writes': 'yes'},
])
def test_explicit_role_policy_validation(root, options):
    with pytest.raises(ValueError):
        SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root, **options)


def test_replay_batches_are_bounded_and_drain_before_later_writes(root):
    clock = {'clock_id': session_timer.state(root).clock_id, 'seconds': 0}
    for index in range(sqlite_pending.MAX_REPLAY_BATCH + 3):
        fallback_store.write_pending('save_memory', str(root), content='Synthetic batch note ' + str(index),
            title='Synthetic batch title ' + str(index), importance_score=65,
            sqlite_identity=IDENTITY.to_dict(), active_clock=clock, sqlite_database='sqlite')
    with owner(root, replay_pending=True, **bindings()) as service:
        assert service.last_replay['synced'] == sqlite_pending.MAX_REPLAY_BATCH
        assert service.last_replay['pending']
        call(service, 'instinct_get', {'segment_id': 'ordinary'})
        assert service.last_replay['synced'] == 3 and not service.last_replay['pending']
        assert service.inspect(lambda adapter: adapter.count()) == sqlite_pending.MAX_REPLAY_BATCH + 5


def test_receipts_are_bounded_and_later_update_remains_pending(root, monkeypatch):
    monkeypatch.setattr(sqlite_pending, 'MAX_RECEIPTS', 1)
    with observer(root) as service:
        call(service, 'instinct_update_score', {'segment_id': 'ordinary', 'new_score': 60})
        call(service, 'instinct_update_score', {'segment_id': 'ordinary', 'new_score': 65})
    with owner(root, replay_pending=True) as service:
        assert service.last_replay['synced'] == 1 and service.last_replay['failed'] == 1
        assert service.inspect(lambda adapter: adapter.get('ordinary')).importance_score == 60
    assert len(fallback_store.list_pending(str(root))) == 1


def test_capture_quota_failure_does_not_replace_existing_pending(root, monkeypatch):
    record = capture(root)
    before = Path(record['_path']).read_bytes()
    monkeypatch.setattr(fallback_store, 'MAX_PENDING_RECORDS', 1)
    with observer(root) as service:
        assert 'capture unconfirmed' in call(service, 'instinct_save_memory', {'content': 'synthetic', 'title': 'Second queue'})
    assert Path(record['_path']).read_bytes() == before
    assert len(list((root / 'fallback_pending').glob('*.json'))) == 1


def run_owner_transport(root):
    with owner(root, replay_pending=True, **bindings()) as service:
        asyncio.run(service.run_stdio())


@pytest.mark.asyncio
async def test_actual_observer_cli_capture_and_owner_stdio_replay(root):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    identity_path = root.parent / (root.name + '-identity.json')
    write_private_text(identity_path, json.dumps(IDENTITY.to_dict()))
    parameters = StdioServerParameters(command=sys.executable, args=[
        '-m', 'instinct.sqlite_service', '--database-dir', str(root / 'sqlite'),
        '--data-dir', str(root), '--identity', str(identity_path), '--read-only', '--queue-writes'])
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as client:
            await client.initialize()
            result = await client.call_tool('instinct_save_memory', {
                'content': 'Synthetic real transport queue note', 'title': 'Transport queued', 'importance': 75})
            assert not result.isError and 'Queued to SQLite FALLBACK' in result.content[0].text
    record = fallback_store.list_pending(str(root))[0]
    script = 'import runpy; from pathlib import Path; runpy.run_path(' + repr(str(Path(__file__).resolve())) + ')["run_owner_transport"](Path(' + repr(str(root)) + '))'
    parameters = StdioServerParameters(command=sys.executable, args=['-c', script])
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as client:
            await client.initialize()
            tools = await client.list_tools()
            assert 'instinct_delete_memory' in {tool.name for tool in tools.tools}
            result = await client.call_tool('instinct_get', {'segment_id': queued_id(record)})
            assert not result.isError and 'Transport queued' in result.content[0].text
    assert not Path(record['_path']).exists()


def test_dashboard_security_boundary_precedes_queue_replay(root):
    with owner(root, replay_pending=True, **bindings()) as service:
        service.start_dashboard(0)
        record = capture(root)
        def heartbeat(token):
            connection = http.client.HTTPConnection('127.0.0.1', service._http.server_port, timeout=10)
            try:
                connection.request('POST', '/api/heartbeat', '{}', headers={
                    'Content-Type': 'application/json', 'X-Instinct-Capability': token})
                response = connection.getresponse()
                response.read()
                return response.status
            finally:
                connection.close()
        assert heartbeat('invalid') == 403
        assert Path(record['_path']).exists()
        assert service.inspect(lambda adapter: adapter.count()) == 2
        assert heartbeat(service._http.capability_token) == 200
        assert not Path(record['_path']).exists()
        assert service.inspect(lambda adapter: adapter.count()) == 3


def test_queue_cannot_cross_databases_with_same_root_clock_and_model(root):
    with SQLiteRecordStore.create(root / 'second', IDENTITY):
        pass
    record = capture(root)
    assert record['sqlite_database'] == 'sqlite'
    with SQLiteService(root / 'second', IDENTITY, persist_dir=root,
                       read_only=False, replay_pending=True, **bindings()) as service:
        assert service.last_replay['failed'] == 1
        assert service.inspect(lambda adapter: adapter.count()) == 0
    assert Path(record['_path']).exists()
    with owner(root, replay_pending=True, **bindings()) as service:
        assert service.last_replay['synced'] == 1
