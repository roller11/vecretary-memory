# SPDX-License-Identifier: MPL-2.0
import hashlib
import json
import os
import sqlite3

import pytest

from instinct.private_storage import PrivateStorageError, write_private_text
from instinct.sqlite_migration import import_manifest, read_manifest, write_manifest
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord, canonical_json
from instinct.sqlite_store import SQLiteRecordStore


@pytest.fixture
def identity():
    return EmbeddingIdentity('synthetic', 'fixed-vectors-v1', 3, 'cosine')


@pytest.fixture
def record(identity):
    return SQLiteRecord.from_dict({
        'id': 'synthetic-note', 'content': 'Fictional Cedar Kite decision: caf\u00e9.',
        'metadata': {'id': 'synthetic-note', 'concept_title': 'Synthetic decision',
                     'importance_score': 84, 'initial_score': 84,
                     'segment_type': 'architectural_decision', 'decay_profile': 'standard',
                     'active_time_anchors': '{"age":12.5}', 'access_count': 4,
                     'keyword_hits_log': '[1.0,2.0]', 'promotions_applied': 1,
                     'recycled_at': '2026-01-01T00:00:00', 'owner_extra': {'keep': True}},
        'embedding': [0.25, -0.5, 1.0],
    }, identity)


def replace_record(record, identity, **updates):
    return SQLiteRecord.from_dict({**record.to_dict(), **updates}, identity)


def rewrite_manifest(path, change):
    envelope = json.loads(path.read_text(encoding='utf-8'))
    change(envelope['payload'])
    envelope['sha256'] = hashlib.sha256(canonical_json(envelope['payload']).encode('ascii')).hexdigest()
    write_private_text(path, canonical_json(envelope))


def test_records_roundtrip_preserves_lifecycle_and_vectors(tmp_path, identity, record):
    destination = tmp_path / 'new-store'
    with SQLiteRecordStore.create(destination, identity) as store:
        assert store.insert_records([record]) == 1
        assert store.count() == 1
        assert store.get_record(record.record_id) == record
        assert store.get_record('missing') is None
    with SQLiteRecordStore.open_read_only(destination, identity) as store:
        assert store.get_record(record.record_id).to_dict() == record.to_dict()
        with pytest.raises(PermissionError, match='read-only'):
            store.insert_records([record])
    if os.name == 'posix':
        assert destination.stat().st_mode & 0o777 == 0o700
        assert (destination / 'memory.sqlite3').stat().st_mode & 0o777 == 0o600


def test_failed_batch_rolls_back_all_new_records(tmp_path, identity, record):
    second = replace_record(record, identity, id='another-note', metadata={'importance_score': 55})
    with SQLiteRecordStore.create(tmp_path / 'store', identity) as store:
        store.insert_records([record])
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_records([second, record])
        assert store.count() == 1
        assert store.get_record(second.record_id) is None
        assert store.get_record(record.record_id) == record


def test_sql_values_are_bound_not_interpreted(tmp_path, identity, record):
    special = replace_record(record, identity, id="x'); DROP TABLE records; --", metadata={})
    with SQLiteRecordStore.create(tmp_path / 'store', identity) as store:
        store.insert_records([special])
        assert store.get_record(special.record_id) == special
        assert store.count() == 1


@pytest.mark.parametrize('embedding', [[1.0], [True, 0, 1], [float('nan'), 0, 1],
                                        [float('inf'), 0, 1], ['code', 0, 1],
                                        [10 ** 1000, 0, 1]])
def test_invalid_vectors_are_rejected(identity, record, embedding):
    with pytest.raises(ValueError):
        replace_record(record, identity, embedding=embedding)


@pytest.mark.parametrize('metadata', [{'importance_score': 101}, {'initial_score': True},
                                     {'id': 'different'}, {'content': 'different'},
                                     {'unsafe': float('nan')}])
def test_invalid_metadata_is_rejected(identity, record, metadata):
    with pytest.raises(ValueError):
        replace_record(record, identity, metadata=metadata)


def test_duplicate_records_rejected_before_writes(tmp_path, identity, record):
    with SQLiteRecordStore.create(tmp_path / 'store', identity) as store:
        with pytest.raises(ValueError, match='Duplicate'):
            store.insert_records([record, record])
        assert store.count() == 0


def test_existing_directory_is_never_adopted(tmp_path, identity):
    destination = tmp_path / 'existing'
    destination.mkdir()
    marker = destination / 'owner-note'
    marker.write_text('Keep this', encoding='ascii')
    with pytest.raises(FileExistsError):
        SQLiteRecordStore.create(destination, identity)
    assert marker.read_text() == 'Keep this'
    assert not (destination / 'memory.sqlite3').exists()


def test_missing_read_only_store_does_not_create_directory(tmp_path, identity):
    destination = tmp_path / 'absent'
    with pytest.raises(FileNotFoundError):
        SQLiteRecordStore.open_read_only(destination, identity)
    assert not destination.exists()


def test_reopen_rejects_provider_mismatch(tmp_path, identity):
    destination = tmp_path / 'store'
    with SQLiteRecordStore.create(destination, identity):
        pass
    changed = EmbeddingIdentity('synthetic', 'different-model', 3, 'cosine')
    with pytest.raises(ValueError, match='identity mismatch'):
        SQLiteRecordStore.open_read_only(destination, changed)


@pytest.mark.parametrize('pragma,value,match', [('user_version', 99, 'schema version'),
                                              ('application_id', 0, 'Foreign')])
def test_foreign_or_future_schema_is_refused(tmp_path, identity, pragma, value, match):
    destination = tmp_path / 'store'
    with SQLiteRecordStore.create(destination, identity):
        pass
    with sqlite3.connect(destination / 'memory.sqlite3') as connection:
        connection.execute(f'PRAGMA {pragma} = {value}')
    with pytest.raises(ValueError, match=match):
        SQLiteRecordStore.open_read_only(destination, identity)


def test_manifest_dry_run_does_not_create_destination(tmp_path, identity, record):
    manifest = tmp_path / 'records.json'
    destination = tmp_path / 'store'
    write_manifest(manifest, identity, [record])
    report = import_manifest(manifest, destination, identity)
    assert report == {'scope': 'records_only', 'records': 1, 'applied': False,
                      'runtime_ready': False, 'verified': False}
    assert not destination.exists()


def test_explicit_import_verifies_after_reopen_and_never_overwrites(tmp_path, identity, record):
    manifest = tmp_path / 'records.json'
    destination = tmp_path / 'store'
    write_manifest(manifest, identity, [record])
    report = import_manifest(manifest, destination, identity, apply=True)
    assert report['applied'] and report['verified'] and not report['runtime_ready']
    with pytest.raises(FileExistsError):
        import_manifest(manifest, destination, identity, apply=True)
    with SQLiteRecordStore.open_read_only(destination, identity) as store:
        assert store.get_record(record.record_id) == record


def test_existing_manifest_is_not_replaced(tmp_path, identity, record):
    manifest = tmp_path / 'records.json'
    write_manifest(manifest, identity, [record])
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        write_manifest(manifest, identity, [])
    assert manifest.read_bytes() == original


def test_tampered_manifest_is_refused_without_destination(tmp_path, identity, record):
    manifest = tmp_path / 'records.json'
    write_manifest(manifest, identity, [record])
    write_private_text(manifest, manifest.read_text().replace('Cedar Kite', 'Changed It'))
    with pytest.raises(ValueError, match='checksum'):
        import_manifest(manifest, tmp_path / 'store', identity, apply=True)
    assert not (tmp_path / 'store').exists()


@pytest.mark.parametrize('field,value', [('version', 2), ('version', True),
                                       ('scope', 'full_store'), ('format', 'foreign')])
def test_unknown_formats_are_refused(tmp_path, identity, record, field, value):
    manifest = tmp_path / 'records.json'
    write_manifest(manifest, identity, [record])
    rewrite_manifest(manifest, lambda payload: payload.update({field: value}))
    with pytest.raises(ValueError, match='Unsupported migration'):
        read_manifest(manifest, identity)


def test_duplicate_json_keys_are_refused(tmp_path, identity):
    manifest = tmp_path / 'records.json'
    write_private_text(manifest, '{"payload":{},"payload":{},"sha256":"x"}')
    with pytest.raises(ValueError, match='Duplicate JSON'):
        read_manifest(manifest, identity)


def test_symlink_manifest_is_refused(tmp_path, identity, record):
    manifest = tmp_path / 'records.json'
    link = tmp_path / 'linked.json'
    write_manifest(manifest, identity, [record])
    try:
        link.symlink_to(manifest)
    except OSError:
        pytest.skip('Symlink creation unavailable')
    with pytest.raises(PrivateStorageError):
        read_manifest(link, identity)


def test_runtime_api_is_sqlite_specific():
    import instinct
    assert not hasattr(instinct, 'InstinctStore')
    assert SQLiteRecordStore.__module__ == 'instinct.sqlite_store'


@pytest.mark.parametrize('score,profile', [(84, 'standard'), (90, 'standard'),
                                         (100, 'standard'), (70, 'ephemeral'),
                                         (92, 'short_term_critical')])
def test_actual_memory_storage_metadata_is_retained(tmp_path, identity, score, profile):
    from instinct.models import ScoredSegment, Segment, SegmentType

    memory = ScoredSegment(segment=Segment(content='Synthetic lifecycle fixture', segment_type=SegmentType.CONTEXT),
                           importance_score=score, concept_title='Synthetic lifecycle',
                           decay_profile=profile, access_count=7, bump_count=2,
                           pop_recent_8h_active=1, promotions_applied=2,
                           active_time_anchors={'age': {'active_hours': 12.5}})
    metadata = memory.to_storage_dict()
    record = SQLiteRecord.from_dict({'id': memory.id, 'content': memory.segment.content,
                                    'metadata': metadata, 'embedding': [0.25, -0.5, 1.0]}, identity)
    manifest = tmp_path / 'records.json'
    destination = tmp_path / 'store'
    write_manifest(manifest, identity, [record])
    import_manifest(manifest, destination, identity, apply=True)
    with SQLiteRecordStore.open_read_only(destination, identity) as store:
        assert store.get_record(memory.id).to_dict()['metadata'] == metadata


def test_sqlite_controls_are_explicit(tmp_path, identity):
    with SQLiteRecordStore.create(tmp_path / 'store', identity) as store:
        assert store._connection.execute('PRAGMA journal_mode').fetchone() == ('delete',)
        assert store._connection.execute('PRAGMA synchronous').fetchone() == (2,)
        assert store._connection.execute('PRAGMA trusted_schema').fetchone() == (0,)
        with pytest.raises(sqlite3.OperationalError, match='not authorized'):
            store._connection.execute("SELECT load_extension('not-a-real-extension')")


def test_manifest_size_limit_is_checked_before_application(tmp_path, identity, record, monkeypatch):
    from instinct import sqlite_migration

    manifest = tmp_path / 'records.json'
    write_manifest(manifest, identity, [record])
    monkeypatch.setattr(sqlite_migration, 'MAX_MANIFEST_BYTES', 32)
    with pytest.raises(ValueError, match='size limit'):
        import_manifest(manifest, tmp_path / 'store', identity, apply=True)
    assert not (tmp_path / 'store').exists()


def test_manifest_provider_mismatch_does_not_create_store(tmp_path, identity, record):
    manifest = tmp_path / 'records.json'
    write_manifest(manifest, identity, [record])
    changed = EmbeddingIdentity('synthetic', 'changed', 3, 'cosine')
    with pytest.raises(ValueError, match='identity mismatch'):
        import_manifest(manifest, tmp_path / 'store', changed, apply=True)
    assert not (tmp_path / 'store').exists()


def test_empty_record_import_is_explicitly_not_a_live_backend(tmp_path, identity):
    manifest = tmp_path / 'records.json'
    write_manifest(manifest, identity, [])
    report = import_manifest(manifest, tmp_path / 'store', identity, apply=True)
    assert report['records'] == 0 and report['verified'] and not report['runtime_ready']


def test_apply_does_not_accept_truthy_nonboolean(tmp_path, identity, record):
    manifest = tmp_path / 'records.json'
    write_manifest(manifest, identity, [record])
    with pytest.raises(ValueError, match='boolean'):
        import_manifest(manifest, tmp_path / 'store', identity, apply='yes')
    assert not (tmp_path / 'store').exists()


@pytest.mark.parametrize('metadata', [{'nested': {1: 'not-a-JSON-key'}},
                                     {'nested': ('not', 'a', 'JSON', 'array')},
                                     {'nested': {'value': float('inf')}}])
def test_metadata_is_not_silently_coerced(identity, record, metadata):
    with pytest.raises(ValueError):
        replace_record(record, identity, metadata=metadata)


def test_deep_metadata_is_rejected(identity, record):
    metadata = {}
    for depth in range(40):
        metadata = {'nested': metadata}
    with pytest.raises(ValueError, match='nesting'):
        replace_record(record, identity, metadata=metadata)


def test_changed_schema_is_refused_even_with_matching_version(tmp_path, identity):
    destination = tmp_path / 'store'
    with SQLiteRecordStore.create(destination, identity):
        pass
    with sqlite3.connect(destination / 'memory.sqlite3') as connection:
        connection.execute('CREATE VIEW extra AS SELECT 1')
    with pytest.raises(ValueError, match='record schema'):
        SQLiteRecordStore.open_read_only(destination, identity)
