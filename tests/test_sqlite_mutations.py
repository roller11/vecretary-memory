# SPDX-License-Identifier: MPL-2.0
"""Synthetic transaction and lifecycle checks for the SQLite record scaffold."""

from datetime import datetime
import sqlite3

import pytest

from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_store import SQLiteRecordStore


@pytest.fixture
def record_store(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    record = SQLiteRecord.from_dict({
        'id': 'note', 'content': 'Synthetic lifecycle decision', 'embedding': [1, 0],
        'metadata': {'id': 'note', 'importance_score': 84, 'initial_score': 84,
                     'access_count': 7, 'active_time_anchors': '{"age":12.5}',
                     'owner_extra': {'retain': [True, 2]}, 'decay_profile': 'standard'},
    }, identity)
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([record])
        yield store, record, tmp_path / 'records'


def test_metadata_patch_preserves_content_vector_and_unmentioned_fields(record_store):
    store, record, _ = record_store
    original_blob = store._connection.execute('SELECT embedding FROM records').fetchone()[0]
    updated = store.patch_metadata('note', {'importance_score': 80, 'owner_added': 'kept'})
    expected = record.to_dict()['metadata'] | {'importance_score': 80, 'owner_added': 'kept'}
    assert updated.to_dict()['metadata'] == expected
    assert updated.content == record.content
    assert updated.embedding == record.embedding
    assert store._connection.execute('SELECT embedding FROM records').fetchone()[0] == original_blob
    assert store.get_record('note') == updated


def test_explicit_metadata_removal_does_not_remove_other_fields(record_store):
    store, record, _ = record_store
    updated = store.patch_metadata('note', {}, remove_keys=['owner_extra', 'absent'])
    expected = record.to_dict()['metadata']
    del expected['owner_extra']
    assert updated.to_dict()['metadata'] == expected


@pytest.mark.parametrize('changes,remove_keys', [
    ({'importance_score': 101}, ()), ({'initial_score': True}, ()),
    ({'id': 'different'}, ()), ({'content': 'different'}, ()),
    ({'value': float('nan')}, ()), ({'value': object()}, ()),
    ({1: 'invalid'}, ()), ([], ()), ({}, 'owner_extra'), ({}, [None]),
    ({'owner_extra': {}}, ['owner_extra']),
])
def test_invalid_patch_leaves_record_unchanged(record_store, changes, remove_keys):
    store, record, _ = record_store
    with pytest.raises(ValueError):
        store.patch_metadata('note', changes, remove_keys=remove_keys)
    assert store.get_record('note') == record
    assert not store._connection.in_transaction


def test_sql_update_failure_rolls_back_and_connection_remains_usable(record_store):
    store, record, _ = record_store
    store._connection.execute("CREATE TRIGGER reject_patch BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='synthetic failure'):
        store.patch_metadata('note', {'importance_score': 60})
    assert store.get_record('note') == record
    assert not store._connection.in_transaction
    store._connection.execute('DROP TRIGGER reject_patch')
    assert store.patch_metadata('note', {'importance_score': 60}) is not None


def test_recycle_and_restore_preserve_lifecycle_and_search_visibility(record_store):
    store, record, _ = record_store
    timestamp = datetime(2026, 1, 2, 3, 4, 5)
    recycled = store.recycle('note', by='synthetic-owner', reason='fixture', now=timestamp)
    assert store.is_recycled('note')
    assert recycled.to_dict()['metadata'] == record.to_dict()['metadata'] | {
        'recycled_at': '2026-01-02T03:04:05', 'recycled_by': 'synthetic-owner',
        'recycled_reason': 'fixture',
    }
    assert store.search_vectors([1, 0]) == []
    assert store.search_vectors([1, 0], include_recycled=True)[0].record == recycled
    assert store.restore('note') == record
    assert not store.is_recycled('note')
    assert store.search_vectors([1, 0])[0].record == record
    assert store.restore('note') == record


def test_recycle_empty_reason_retains_existing_audit_reason(record_store):
    store, _, _ = record_store
    store.recycle('note', reason='first reason')
    again = store.recycle('note', by='another-owner')
    assert again.to_dict()['metadata']['recycled_reason'] == 'first reason'


@pytest.mark.parametrize('options', [{'by': ''}, {'by': 2}, {'reason': None},
                                   {'reason': '\x00'}, {'reason': 'x' * 4097}, {'now': 'invalid'}])
def test_invalid_recycle_audit_fields_leave_record_unchanged(record_store, options):
    store, record, _ = record_store
    with pytest.raises(ValueError):
        store.recycle('note', **options)
    assert store.get_record('note') == record


def test_missing_records_do_not_get_created_by_lifecycle_mutations(record_store):
    store, _, _ = record_store
    assert store.patch_metadata('missing', {'importance_score': 80}) is None
    assert store.recycle('missing') is None
    assert store.restore('missing') is None
    assert not store.is_recycled('missing')
    assert store.count() == 1


def test_reopen_retains_restored_metadata_and_read_only_rejects_all_writes(record_store):
    store, record, destination = record_store
    store.recycle('note', reason='fixture')
    store.restore('note')
    with SQLiteRecordStore.open_read_only(destination, store.identity) as reader:
        assert reader.get_record('note') == record
        for operation in (
                lambda: reader.patch_metadata('note', {'importance_score': 60}),
                lambda: reader.upsert_records([record]),
                lambda: reader.recycle('note'), lambda: reader.restore('note')):
            with pytest.raises(PermissionError, match='read-only'):
                operation()
        assert reader.get_record('note') == record


def test_upsert_is_explicit_full_replacement_and_idempotent(record_store):
    store, record, _ = record_store
    payload = record.to_dict()
    payload.update(content='Explicit new content', embedding=[0, 1])
    payload['metadata'] = {'importance_score': 70}
    replacement = SQLiteRecord.from_dict(payload, store.identity)
    assert store.upsert_records([replacement]) == 1
    assert store.upsert_records([replacement]) == 1
    assert store.count() == 1
    assert store.get_record('note') == replacement


def test_upsert_batch_database_failure_rolls_back_existing_and_new_records(record_store):
    store, record, _ = record_store
    replacement = SQLiteRecord.from_dict(record.to_dict() | {'metadata': {'importance_score': 60}}, store.identity)
    added = SQLiteRecord.from_dict(record.to_dict() | {'id': 'new', 'metadata': {}}, store.identity)
    store._connection.execute("CREATE TRIGGER reject_new BEFORE INSERT ON records WHEN NEW.id = 'new' BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_records([replacement, added])
    assert store.get_record('note') == record
    assert store.get_record('new') is None
    assert store.count() == 1
    assert not store._connection.in_transaction


def test_upsert_limit_counts_only_new_ids_and_rolls_back_overflow(record_store, monkeypatch):
    store, record, _ = record_store
    monkeypatch.setattr('instinct.sqlite_store.MAX_RECORDS', 1)
    replacement = SQLiteRecord.from_dict(record.to_dict() | {'metadata': {'importance_score': 60}}, store.identity)
    assert store.upsert_records([replacement]) == 1
    added = SQLiteRecord.from_dict(record.to_dict() | {'id': 'new', 'metadata': {}}, store.identity)
    with pytest.raises(ValueError, match='count'):
        store.upsert_records([record, added])
    assert store.get_record('note') == replacement
    assert store.count() == 1


def test_duplicate_batch_ids_are_rejected_before_any_change(record_store):
    store, record, _ = record_store
    with pytest.raises(ValueError, match='Duplicate'):
        store.upsert_records([record, record])
    assert store.get_record('note') == record


def test_bound_id_handles_sql_shaped_text_without_affecting_other_records(record_store):
    store, record, _ = record_store
    identifier = "x'; UPDATE records SET metadata = '{}'; --"
    added = SQLiteRecord.from_dict(record.to_dict() | {'id': identifier, 'metadata': {}}, store.identity)
    store.upsert_records([added])
    assert store.patch_metadata(identifier, {'owner_extra': 'safe'}) is not None
    assert store.get_record('note') == record


def test_competing_writer_lock_fails_without_partial_metadata_change(record_store):
    store, record, destination = record_store
    store._connection.execute('PRAGMA busy_timeout = 1')
    with sqlite3.connect(destination / 'memory.sqlite3', isolation_level=None) as competitor:
        competitor.execute('BEGIN IMMEDIATE')
        try:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                store.patch_metadata('note', {'importance_score': 60})
        finally:
            competitor.rollback()
    assert store.get_record('note') == record
    assert store.patch_metadata('note', {'importance_score': 60}) is not None


@pytest.mark.parametrize('score,profile', [(84, 'standard'), (90, 'standard'),
                                         (100, 'standard'), (70, 'ephemeral'),
                                         (92, 'short_term_critical')])
def test_model_lifecycle_fields_survive_recycle_restore(tmp_path, score, profile):
    from instinct.models import ScoredSegment, Segment, SegmentType

    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    memory = ScoredSegment(segment=Segment(content='Synthetic lifecycle', segment_type=SegmentType.CONTEXT),
                           importance_score=score, concept_title='Synthetic decision',
                           decay_profile=profile, access_count=7, bump_count=2,
                           pop_recent_8h_active=1, promotions_applied=2,
                           active_time_anchors={'age': {'active_hours': 12.5}})
    record = SQLiteRecord.from_dict({'id': memory.id, 'content': memory.segment.content,
                                    'metadata': memory.to_storage_dict(), 'embedding': [1, 0]}, identity)
    with SQLiteRecordStore.create(tmp_path / 'lifecycle', identity) as store:
        store.upsert_records([record])
        store.recycle(record.record_id, reason='fixture')
        assert store.restore(record.record_id) == record


def test_recycled_manifest_roundtrip_retains_audit_and_not_runtime_ready(record_store, tmp_path):
    from instinct.sqlite_migration import import_manifest, write_manifest

    store, _, _ = record_store
    recycled = store.recycle('note', reason='fixture')
    manifest = tmp_path / 'recycled.json'
    write_manifest(manifest, store.identity, [recycled])
    destination = tmp_path / 'imported'
    report = import_manifest(manifest, destination, store.identity, apply=True)
    assert report['verified'] and not report['runtime_ready']
    with SQLiteRecordStore.open_read_only(destination, store.identity) as reader:
        assert reader.get_record('note') == recycled
        assert reader.is_recycled('note')
        assert reader.search_vectors([1, 0]) == []
