# SPDX-License-Identifier: MPL-2.0
"""Synthetic-only acceptance for explicitly enabled single-record owner purge."""

import asyncio
import http.client
import json
import sqlite3
import subprocess
import sys

import pytest

from instinct import fallback_store, session_timer
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.private_storage import private_file_lock, write_private_text
from instinct.sqlite_backup import restore_new, snapshot_offline
from instinct.sqlite_pending import capture_call
from instinct.sqlite_purge import purge_recycled
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_runtime import SQLiteSegmentAdapter
from instinct.sqlite_service import SQLiteService
from instinct.sqlite_store import SQLiteRecordStore


IDENTITY = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')


@pytest.fixture
def root(tmp_path):
    tmp_path.chmod(0o700)
    session_timer.reset_session(tmp_path)
    with SQLiteRecordStore.create(tmp_path / 'sqlite', IDENTITY) as repository:
        for record_id, score, segment_type in (
                ('ordinary', 80, SegmentType.CONTEXT),
                ('mandatory', 100, SegmentType.STANDING_INSTRUCTION)):
            segment = ScoredSegment(segment=Segment(content='Synthetic purge fixture ' + record_id,
                stored_id=record_id, segment_type=segment_type),
                importance_score=score, concept_title='Synthetic ' + record_id)
            payload = segment.to_storage_dict()
            payload.pop('content', None)
            repository.insert_records([SQLiteRecord.from_dict({
                'id': record_id, 'content': segment.segment.content,
                'metadata': payload, 'embedding': [1, 0]}, IDENTITY)])
    return tmp_path


def owner(root, **options):
    return SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root,
                         read_only=False, **options)


def request(service, path='/api/purge', *, body=None, headers=None, method='POST'):
    if service._http is None:
        service.start_dashboard()
    port = service._http.server_port
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    fields = {'Host': f'127.0.0.1:{port}', 'Content-Type': 'application/json',
              'X-Instinct-Capability': service._http.capability_token}
    fields.update(headers or {})
    try:
        connection.request(method, path, body=json.dumps(body or {}) if method == 'POST' else None,
                           headers=fields)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def purge_body(record_id='ordinary'):
    return {'id': record_id, 'confirm_id': record_id}


def test_purge_is_disabled_by_default(root):
    with owner(root) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        assert request(service, body=purge_body())[0] == 501
        assert request(service, '/api/health', method='GET')[1]['purge_enabled'] is False
        assert service.inspect(lambda store: store.get('ordinary')) is not None


@pytest.mark.parametrize('policy', [1, 'yes', None])
def test_policy_requires_boolean(root, policy):
    with pytest.raises(ValueError, match='boolean'):
        owner(root, enable_purge=policy)
    with owner(root) as service:
        assert service.inspect(lambda store: store.count()) == 2


def test_observer_cannot_enable_purge(root):
    with pytest.raises(ValueError, match='owner-only'):
        SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root,
                      read_only=True, enable_purge=True)


@pytest.mark.parametrize('body', [
    {}, {'id': 'ordinary'}, {'id': 'ordinary', 'confirm_id': 'other'},
    {'id': '', 'confirm_id': ''}, {'id': 1, 'confirm_id': 1},
    {'id': 'ordinary', 'confirm_id': 'ordinary', 'all': True},
    {'id': 'x' * 257, 'confirm_id': 'x' * 257},
])
def test_purge_requires_exact_bounded_confirmation(root, body):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        before = service.inspect(lambda store: store.repository.get_record('ordinary'))
        assert request(service, body=body)[0] == 400
        assert service.inspect(lambda store: store.repository.get_record('ordinary')) == before


@pytest.mark.parametrize('headers', [
    {'X-Instinct-Capability': 'invalid'}, {'Origin': 'https://untrusted.invalid'},
    {'Sec-Fetch-Site': 'cross-site'}, {'Host': 'untrusted.invalid'},
])
def test_purge_checks_http_boundary_before_mutation(root, headers):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        assert request(service, body=purge_body(), headers=headers)[0] == 403
        assert service.inspect(lambda store: store.get('ordinary')) is not None


def test_active_record_must_be_recycled(root):
    with owner(root, enable_purge=True) as service:
        assert request(service, body=purge_body())[0] == 400
        assert service.inspect(lambda store: store.count()) == 2


def test_purge_preserves_other_records_clock_and_restart(root):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        remaining = service.inspect(lambda store: store.repository.get_record('mandatory'))
        sidecars = {path.name: path.read_bytes() for path in root.glob('*.json')}
        status, result = request(service, body=purge_body())
        assert status == 200 and result['purged'] is True
        assert service.inspect(lambda store: store.get('ordinary')) is None
        assert service.inspect(lambda store: store.list_recycled()) == []
        assert service.inspect(lambda store: store.repository.get_record('mandatory')) == remaining
        assert {path.name: path.read_bytes() for path in root.glob('*.json')} == sidecars
        assert request(service, body=purge_body())[1]['purged'] is False
    with owner(root) as restarted:
        assert restarted.inspect(lambda store: store.count()) == 1
        assert restarted.inspect(lambda store: store.get('ordinary')) is None


def test_owner_can_explicitly_purge_recycled_mandatory_record(root):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('mandatory'))
        assert request(service, body=purge_body('mandatory'))[1]['purged'] is True
        assert service.inspect(lambda store: store.get('ordinary')) is not None


def test_agent_delete_and_bulk_reset_remain_disabled(root):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        response = asyncio.run(service.call('instinct_delete_memory', {'segment_id': 'ordinary'}))
        assert 'disabled' in response[0].text
        assert request(service, '/api/admin/reset', body={'confirm': True})[0] == 501
        assert request(service, '/api/delete', body={'id': 'ordinary'})[0] == 501
        assert service.inspect(lambda store: store.count()) == 2


@pytest.mark.parametrize('kind', ['legacy', 'invalid', 'sqlite'])
def test_any_pending_file_blocks_purge_without_replay(root, kind):
    with owner(root, enable_purge=True, replay_pending=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        if kind == 'legacy':
            fallback_store.write_pending('delete_memory', str(root), delete_id='ordinary')
        elif kind == 'invalid':
            write_private_text(root / 'fallback_pending' / 'invalid.json', '{broken')
        else:
            with SQLiteService(root / 'sqlite', IDENTITY, persist_dir=root, read_only=True,
                               queue_writes=True) as observer:
                result = observer.inspect(lambda store: capture_call(store, 'instinct_update_score',
                    {'segment_id': 'ordinary', 'new_score': 70}))
                assert 'Queued' in result
        before = {path.name: path.read_bytes() for path in (root / 'fallback_pending').glob('*.json')}
        assert request(service, body=purge_body())[0] == 409
        assert {path.name: path.read_bytes() for path in (root / 'fallback_pending').glob('*.json')} == before
        assert service.inspect(lambda store: store.get('ordinary')) is not None
        assert service.last_replay['synced'] == 0


def test_capture_guard_contention_refuses_purge(root):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        with private_file_lock(root / 'fallback_pending' / '.capture.lock', timeout=0):
            assert request(service, body=purge_body())[0] == 409
        assert service.inspect(lambda store: store.get('ordinary')) is not None


def test_record_transaction_rolls_back_failed_delete(root):
    with SQLiteRecordStore.open_writable(root / 'sqlite', IDENTITY) as repository:
        repository.recycle('ordinary')
        before = repository.get_record('ordinary')
        repository._connection.set_authorizer(lambda action, *unused:
            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DELETE else sqlite3.SQLITE_OK)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                repository.purge_record('ordinary')
        finally:
            repository._connection.set_authorizer(None)
        assert not repository._connection.in_transaction
        assert repository.get_record('ordinary') == before
        assert repository.purge_record('ordinary') is True


def test_read_only_repository_and_adapter_refuse_purge(root):
    before = (root / 'sqlite' / 'memory.sqlite3').read_bytes()
    with SQLiteRecordStore.open_read_only(root / 'sqlite', IDENTITY) as repository:
        with pytest.raises(PermissionError):
            repository.purge_record('ordinary')
        adapter = SQLiteSegmentAdapter(repository, persist_dir=root)
        with pytest.raises(PermissionError):
            purge_recycled(adapter, 'ordinary', 'ordinary')
    assert (root / 'sqlite' / 'memory.sqlite3').read_bytes() == before
    assert not (root / 'fallback_pending').exists()


def test_offline_restore_retains_pre_purge_recovery_copy(root):
    with owner(root) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
    snapshot = snapshot_offline(root, root.parent / (root.name + '-snapshot'), IDENTITY)
    with owner(root, enable_purge=True) as service:
        assert request(service, body=purge_body())[1]['purged'] is True
    restored = restore_new(snapshot, root.parent / (root.name + '-restored'), IDENTITY)
    with owner(restored) as service:
        assert service.inspect(lambda store: store.is_recycled('ordinary')) is True
    with owner(root) as service:
        assert service.inspect(lambda store: store.get('ordinary')) is None


def test_http_database_failure_rolls_back_and_reports_unconfirmed(root):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        service.inspect(lambda store: store.repository._connection.set_authorizer(
            lambda action, *unused: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DELETE else sqlite3.SQLITE_OK))
        try:
            status, result = request(service, body=purge_body())
            assert status == 503
            assert 'inspect committed records' in result['error']
        finally:
            service.inspect(lambda store: store.repository._connection.set_authorizer(None))
        assert service.inspect(lambda store: store.get('ordinary')) is not None
        assert request(service, body=purge_body())[1]['purged'] is True


def test_bound_record_id_is_not_sql(root):
    record_id = "ordinary' OR 1=1 --"
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        status, result = request(service, body=purge_body(record_id))
        assert status == 200 and result['purged'] is False
        assert service.inspect(lambda store: store.count()) == 2


def test_unsupported_get_purge_does_not_mutate(root):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        assert request(service, '/api/purge?id=ordinary', method='GET')[0] == 501
        assert service.inspect(lambda store: store.get('ordinary')) is not None


def test_cli_rejects_observer_purge_before_identity_or_storage_access(root):
    result = subprocess.run([sys.executable, '-m', 'instinct.sqlite_service',
        '--database-dir', str(root / 'sqlite'), '--data-dir', str(root),
        '--identity', str(root / 'missing.json'), '--read-only', '--enable-purge'],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert 'Purge requires owner mode' in result.stderr
    assert not (root / 'fallback_pending').exists()


def test_post_purge_backup_does_not_reintroduce_deleted_record(root):
    with owner(root, enable_purge=True) as service:
        service.inspect(lambda store: store.recycle('ordinary'))
        assert request(service, body=purge_body())[1]['purged'] is True
    snapshot = snapshot_offline(root, root.parent / (root.name + '-post-snapshot'), IDENTITY)
    restored = restore_new(snapshot, root.parent / (root.name + '-post-restored'), IDENTITY)
    with owner(restored) as service:
        assert service.inspect(lambda store: store.count()) == 1
        assert service.inspect(lambda store: store.get('ordinary')) is None
