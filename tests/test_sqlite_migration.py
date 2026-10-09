# SPDX-License-Identifier: MPL-2.0

import json
import hashlib
from pathlib import Path
from datetime import datetime

import pytest

from instinct.sqlite_migration import (
    export_collection_manifest, export_runtime_bundle, import_manifest,
    import_runtime_bundle, read_manifest, write_manifest,
)
from instinct.private_storage import write_private_text
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord, canonical_json
from instinct.sqlite_session import SQLiteRuntimeSession
from instinct.sqlite_store import SQLiteRecordStore


IDENTITY = EmbeddingIdentity('ollama', 'nomic-embed-text', 3, 'cosine')


class SyntheticCollection:
    def __init__(self, records):
        self.records = records
        self.count_calls = 0

    def count(self):
        self.count_calls += 1
        return len(self.records)

    def get(self, *, limit, offset, include):
        assert include == ['documents', 'metadatas', 'embeddings']
        selected = self.records[offset:offset + limit]
        return {
            'ids': [record['id'] for record in selected],
            'documents': [record['content'] for record in selected],
            'metadatas': [record['metadata'] for record in selected],
            'embeddings': [record['embedding'] for record in selected],
        }


def sample_record():
    from instinct.models import ScoredSegment, Segment, SegmentType
    content = 'Preserve activity timer and pop state during migration.'
    segment = Segment(content=content, segment_type=SegmentType.CONTEXT,
                      timestamp=datetime(2026, 10, 8, 10), source='synthetic',
                      stored_id='legacy-memory-1', metadata={
                          'mandatory_tags': ['sqlite', 'migration'],
                          'nudge_interval_turns': 6,
                          'nudged_at_turn': 9,
                          'turns_since_access': 2,
                          'nudge_active': True,
                          'future_module_data': {'labels': ['keep'], 'revision': 4},
                      })
    scored = ScoredSegment(segment=segment, importance_score=85,
                           concept_title='Synthetic timer/pop fixture',
                           reasoning='Synthetic migration test',
                           memory_identifiers=['activity timer', 'pop state'])
    scored.initial_score = 96
    scored.access_count = 4
    scored.last_accessed = datetime(2026, 10, 8, 10, 10)
    scored.popped_at = datetime(2026, 10, 8, 9, 45)
    scored.accesses_since_pop = 3
    scored.last_bumped_at = datetime(2026, 10, 8, 9, 15)
    scored.bump_count = 2
    scored.decay_profile = 'short_term_critical'
    scored.ephemeral_phase = 'settled'
    scored.ephemeral_turns_left = 4
    scored.initial_decay_path = 'synthetic-profile-v2'
    scored.initial_semantic_score = 0.765
    scored.keyword_hits_total = 5
    scored.keyword_hits_log = [{'ts': '2026-10-08T10:10:00', 'count': 2, 'query_id': 'synthetic-q'}]
    scored.last_popped_at = datetime(2026, 10, 8, 9, 45)
    scored.pop_anchor_at = datetime(2026, 10, 8, 9, 45)
    scored.pop_recent_8h_active = 3
    scored.promotions_applied = 2
    scored.active_time_anchors = {
        'created': {'clock_id': 'synthetic-clock', 'seconds': 1200.0,
                    'timestamp': '2026-10-08T10:00:00', 'score': 85},
        'pop_anchor_at': {'clock_id': 'synthetic-clock', 'seconds': 900.0,
                          'timestamp': '2026-10-08T09:45:00', 'score': 85},
    }
    metadata = scored.to_storage_dict()
    metadata['score_extensions'] = {
        'org.rawwire.relevance': {
            'confidence': 0.875,
            'signals': ['semantic', 'recency'],
            'parameters': {'half_life_hours': 48, 'enabled': True},
        },
    }
    return {
        'id': 'legacy-memory-1',
        'content': content,
        'metadata': metadata,
        'embedding': [0.25, 0.5, 0.75],
    }


def test_collection_export_is_paged_and_preserves_vectors_and_lifecycle_state(tmp_path):
    source = tmp_path / 'export.json'
    expected = [sample_record(), sample_record()]
    expected[1]['id'] = 'legacy-memory-2'
    expected[1]['content'] = 'Second synthetic persisted memory.'
    expected[1]['metadata']['id'] = expected[1]['id']
    expected[1]['metadata']['content'] = expected[1]['content']
    expected[1]['embedding'] = [0.75, 0.5, 0.25]
    collection = SyntheticCollection(expected)

    result = export_collection_manifest(source, IDENTITY, collection, batch_size=1)

    assert result['records'] == 2
    assert collection.count_calls == 2
    assert read_manifest(source, IDENTITY) == [SQLiteRecord.from_dict(record, IDENTITY) for record in expected]


@pytest.mark.parametrize('batch_size', [0, -1, 4097, True])
def test_collection_export_rejects_invalid_batch_size_without_output(tmp_path, batch_size):
    source = tmp_path / 'export.json'
    with pytest.raises(ValueError, match='batch size'):
        export_collection_manifest(source, IDENTITY, SyntheticCollection([]), batch_size=batch_size)
    assert not source.exists()


def test_collection_export_refuses_incomplete_batch_without_output(tmp_path):
    source = tmp_path / 'export.json'

    class IncompleteCollection(SyntheticCollection):
        def get(self, **kwargs):
            result = super().get(**kwargs)
            result['embeddings'] = []
            return result

    with pytest.raises(ValueError, match='incomplete batch'):
        export_collection_manifest(source, IDENTITY, IncompleteCollection([sample_record()]))
    assert not source.exists()


def test_collection_export_detects_count_change_without_output(tmp_path):
    source = tmp_path / 'export.json'

    class ChangingCollection(SyntheticCollection):
        def count(self):
            self.count_calls += 1
            return 1 if self.count_calls == 1 else 2

    with pytest.raises(ValueError, match='changed during export'):
        export_collection_manifest(source, IDENTITY, ChangingCollection([sample_record()]))
    assert not source.exists()


def test_real_embedded_vector_collection_export_parity_when_optional_dependency_exists(tmp_path):
    chromadb = pytest.importorskip('chromadb')
    from chromadb.config import Settings

    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        'synthetic_migration_fixture', embedding_function=None,
        metadata={'hnsw:space': 'cosine'},
    )
    expected = sample_record()
    expected['metadata']['memory_identifiers'] = 'activity timer|pop state'
    expected['metadata'] = {
        key: ('' if value is None else value if type(value) in (str, int, float, bool) else str(value))
        for key, value in expected['metadata'].items()
    }
    collection.add(
        ids=[expected['id']], documents=[expected['content']],
        metadatas=[expected['metadata']], embeddings=[expected['embedding']],
    )
    source = tmp_path / 'export.json'

    result = export_collection_manifest(source, IDENTITY, collection, batch_size=1)

    assert result['records'] == 1
    restored = read_manifest(source, IDENTITY)
    assert len(restored) == 1
    assert restored[0].to_dict() == expected
    assert restored[0].embedding == tuple(expected['embedding'])


def test_import_manifest_preserves_vector_and_lifecycle_metadata(tmp_path):
    source = tmp_path / 'export.json'
    database_dir = tmp_path / 'new-runtime' / 'sqlite'
    expected = sample_record()
    write_manifest(source, IDENTITY, [expected])

    dry_run = import_manifest(source, database_dir, IDENTITY)
    assert dry_run == {
        'scope': 'records_only', 'records': 1, 'applied': False,
        'runtime_ready': False, 'verified': False,
    }
    assert not database_dir.exists()

    result = import_manifest(source, database_dir, IDENTITY, apply=True)
    assert result['applied'] is True
    assert result['verified'] is True
    assert result['records'] == 1

    with SQLiteRecordStore.open_read_only(database_dir, IDENTITY) as store:
        saved = store.get_record(expected['id'])
        assert saved is not None
        assert saved.to_dict() == expected
        assert saved.embedding == tuple(expected['embedding'])


def test_import_manifest_rejects_checksum_tampering_before_creation(tmp_path):
    source = tmp_path / 'export.json'
    database_dir = tmp_path / 'new-runtime' / 'sqlite'
    write_manifest(source, IDENTITY, [sample_record()])
    envelope = json.loads(source.read_text(encoding='utf-8'))
    envelope['payload']['records'][0]['content'] = 'tampered content'
    source.write_text(json.dumps(envelope), encoding='utf-8')

    with pytest.raises(ValueError, match='checksum'):
        import_manifest(source, database_dir, IDENTITY, apply=True)
    assert not database_dir.exists()


def test_import_manifest_rejects_embedding_identity_mismatch_before_creation(tmp_path):
    source = tmp_path / 'export.json'
    database_dir = tmp_path / 'new-runtime' / 'sqlite'
    write_manifest(source, IDENTITY, [sample_record()])
    wrong_identity = EmbeddingIdentity('ollama', 'different-model', 3, 'cosine')

    with pytest.raises(ValueError, match='identity mismatch'):
        import_manifest(source, database_dir, wrong_identity, apply=True)
    assert not database_dir.exists()


def _write_runtime_fixture(source):
    source.mkdir(mode=0o700)
    write_private_text(source / 'session_state.json', json.dumps({
        'session_started_at': '2026-10-08T10:00:00',
        'session_last_tick_at': '2026-10-08T10:05:00',
        'accumulated_session_seconds': 3600.0,
        'clock_id': 'synthetic-clock',
        'baseline_session_seconds': 3000.0,
        'last_tick_monotonic': 125.0,
        'monotonic_identity': 'synthetic-boot',
    }))
    write_private_text(source / 'nudge_state.json', json.dumps({
        'current_turn': 17, 'last_tick_at': '2026-10-08T10:05:00',
        'session_started_at': '2026-10-08T10:00:00',
    }))
    write_private_text(source / 'dormancy_config.json', json.dumps({
        'session_idle_minutes': 7.5, 'dormant_floor': 87,
    }))
    from instinct.fallback_store import write_pending
    queued = write_pending('save_memory', str(source), content='Synthetic queued migration note',
                           title='Synthetic queue note', importance_score=50,
                           segment_type='context')
    return {name: (source / name).read_bytes() for name in (
        'session_state.json', 'nudge_state.json', 'dormancy_config.json')}, Path(queued)


def _tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def _write_sqlite_pending(source, name, *, identity=IDENTITY, clock=None):
    from instinct import fallback_store
    from instinct.active_time import clock_snapshot

    queued = Path(fallback_store.write_pending(
        'update_score', str(source), segment_id='legacy-memory-1', new_score=50,
        sqlite_identity=identity.to_dict(), active_clock=clock or clock_snapshot(source),
        sqlite_database='sqlite',
    ))
    target = queued.with_name(name)
    queued.rename(target)
    return target


def test_runtime_bundle_stages_records_clock_config_and_queue_without_replay(tmp_path):
    source = tmp_path / 'legacy-copy'
    sidecars, queued = _write_runtime_fixture(source)
    queued_bytes = queued.read_bytes()
    bundle = tmp_path / 'bundle'
    legacy = sample_record()
    legacy['metadata']['segment_type'] = 'CONTEXT'
    export = export_runtime_bundle(bundle, IDENTITY, SyntheticCollection([legacy]), source)

    assert export == {
        'records': 1,
        'sidecars': ['dormancy_config.json', 'nudge_state.json', 'session_state.json'],
        'pending_files': 1, 'runtime_ready': False,
        'queue_policy': 'preserve-no-replay',
    }
    destination = tmp_path / 'sqlite-runtime'
    preview = import_runtime_bundle(bundle, destination, IDENTITY)
    assert preview['applied'] is False and preview['verified'] is False
    assert preview['runtime_ready'] is False
    assert preview['pending_review'][0]['classification'] == 'legacy-pending-review-required'
    assert not destination.exists()

    applied = import_runtime_bundle(bundle, destination, IDENTITY, apply=True)
    assert applied['applied'] and applied['verified']
    assert applied['runtime_ready'] is False
    identity_bytes = (canonical_json(IDENTITY.to_dict()) + '\n').encode('ascii')
    assert (destination / 'embedding-identity.json').read_bytes() == identity_bytes
    assert {name: (destination / name).read_bytes() for name in sidecars} == sidecars
    migrated_queue = next((destination / 'fallback_pending').glob('*.json'))
    assert migrated_queue.read_bytes() == queued_bytes
    with SQLiteRecordStore.open_read_only(destination / 'sqlite', IDENTITY) as store:
        assert store.count() == 1
        assert store.get_record('legacy-memory-1').to_dict() == sample_record()

    from instinct import fallback_store, session_timer
    before_clock = session_timer.state(destination)
    result = fallback_store.list_pending(str(destination))
    assert len(result) == 1
    assert migrated_queue.exists()
    with SQLiteRuntimeSession.open(destination / 'sqlite', IDENTITY,
                                   persist_dir=destination, read_only=False) as session:
        session.tick_activity()
        with session.operation() as adapter:
            restored = adapter.get('legacy-memory-1')
            assert restored.accesses_since_pop == 3
            assert restored.popped_at == datetime(2026, 10, 8, 9, 45)
            assert restored.access_count == 4
    after_clock = session_timer.state(destination)
    assert after_clock.accumulated_session_seconds >= before_clock.accumulated_session_seconds
    assert migrated_queue.read_bytes() == queued_bytes


def test_runtime_bundle_preserves_recycle_state_and_future_score_parameters(tmp_path):
    source = tmp_path / 'legacy-copy'
    _write_runtime_fixture(source)
    record = sample_record()
    record['metadata'].update({
        'recycled_at': '2026-10-08T10:04:00',
        'recycled_by': 'legacy-owner',
        'recycled_reason': 'Synthetic preservation fixture',
        'org.example.future-module': {
            'weights': [0.25, 0.5],
            'policy': {'recall_present': True, 'penalty': 0},
        },
    })
    source_before = _tree_bytes(source)
    bundle = tmp_path / 'bundle'
    export_runtime_bundle(bundle, IDENTITY, SyntheticCollection([record]), source)
    destination = tmp_path / 'sqlite-runtime'

    imported = import_runtime_bundle(bundle, destination, IDENTITY, apply=True)
    assert imported['applied'] and imported['verified']
    with SQLiteRecordStore.open_read_only(destination / 'sqlite', IDENTITY) as repository:
        assert repository.get_record('legacy-memory-1').to_dict() == record

    with SQLiteRuntimeSession.open(destination / 'sqlite', IDENTITY,
                                   persist_dir=destination, read_only=False) as session:
        with session.operation() as adapter:
            recycled = adapter.get('legacy-memory-1')
            assert recycled.segment.metadata['recycled_by'] == 'legacy-owner'
            assert recycled.segment.metadata['org.example.future-module'] == {
                'weights': [0.25, 0.5],
                'policy': {'recall_present': True, 'penalty': 0},
            }
            assert recycled.segment.metadata['score_extensions']['org.rawwire.relevance']['confidence'] == 0.875
            restored = adapter.restore('legacy-memory-1')
            assert restored is not None
            assert not adapter.is_recycled('legacy-memory-1')
            assert restored.segment.metadata['score_extensions']['org.rawwire.relevance']['parameters'] == {
                'half_life_hours': 48, 'enabled': True,
            }
            updated = adapter.patch_score_extension('legacy-memory-1', 'org.rawwire.relevance', {
                'confidence': 0.91,
            })
            assert updated.segment.metadata['score_extensions']['org.rawwire.relevance']['signals'] == [
                'semantic', 'recency',
            ]
            assert updated.segment.metadata['org.example.future-module']['policy']['recall_present'] is True
            updated_metadata = updated.to_storage_dict()
    with SQLiteRuntimeSession.open(destination / 'sqlite', IDENTITY,
                                   persist_dir=destination, read_only=True) as session:
        with session.operation() as adapter:
            reopened = adapter.get('legacy-memory-1')
            assert reopened.to_storage_dict() == updated_metadata
            assert not adapter.is_recycled('legacy-memory-1')
    assert _tree_bytes(source) == source_before


@pytest.mark.parametrize(('field', 'value'), [
    ('keyword_hits_log', '{broken'),
    ('keyword_hits_log', '{}'),
    ('keyword_hits_log', '[12]'),
    ('keyword_hits_log', '[NaN]'),
    ('keyword_hits_log', '[{"count": 1, "count": 2}]'),
    ('active_time_anchors', '{broken'),
    ('active_time_anchors', '[]'),
    ('active_time_anchors', '{"created": 12}'),
    ('active_time_anchors', {'created': {'seconds': 1200}}),
])
def test_runtime_export_refuses_lossy_lifecycle_metadata(tmp_path, field, value):
    source = tmp_path / 'legacy-copy'
    _write_runtime_fixture(source)
    source_before = _tree_bytes(source)
    record = sample_record()
    record['metadata'][field] = value
    bundle = tmp_path / 'bundle'

    with pytest.raises(ValueError, match='metadata'):
        export_runtime_bundle(bundle, IDENTITY, SyntheticCollection([record]), source)

    assert not bundle.exists()
    assert _tree_bytes(source) == source_before


def test_migration_preserves_review_dispositions_without_replaying_any_queue(tmp_path, monkeypatch):
    from instinct import fallback_store, sqlite_pending
    from instinct.sqlite_backup import snapshot_offline
    from instinct.sqlite_queue_review import inventory_snapshot

    source = tmp_path / 'legacy-copy'
    _, legacy = _write_runtime_fixture(source)
    legacy.unlink()
    valid = _write_sqlite_pending(source, 'valid-sqlite.json')
    identity_mismatch = _write_sqlite_pending(
        source, 'identity-mismatch.json',
        identity=EmbeddingIdentity('ollama', 'different-model', 3, 'cosine'),
    )
    clock_mismatch = _write_sqlite_pending(
        source, 'clock-mismatch.json',
        clock={'clock_id': 'different-clock', 'seconds': 0},
    )
    legacy = Path(fallback_store.write_pending(
        'update_score', str(source), segment_id='legacy-memory-1', new_score=50,
    ))
    legacy.rename(legacy.with_name('legacy-pending.json'))
    legacy = source / 'fallback_pending' / 'legacy-pending.json'
    malformed = source / 'fallback_pending' / 'malformed.json'
    write_private_text(malformed, '{broken')
    queue_bytes = {path.name: path.read_bytes()
                   for path in (valid, identity_mismatch, clock_mismatch, legacy, malformed)}

    bundle = tmp_path / 'bundle'
    export_runtime_bundle(bundle, IDENTITY, SyntheticCollection([sample_record()]), source)
    bundle_bytes = _tree_bytes(bundle)
    destination = tmp_path / 'sqlite-runtime'

    def forbidden_replay(*_args, **_kwargs):
        pytest.fail('Migration must never replay pending operations')

    monkeypatch.setattr(sqlite_pending, 'replay', forbidden_replay)
    report = import_runtime_bundle(bundle, destination, IDENTITY, apply=True)

    assert report['applied'] and report['verified'] and not report['runtime_ready']
    assert report['queue_policy'] == 'preserve-no-replay'
    assert {path.name: path.read_bytes() for path in (destination / 'fallback_pending').glob('*.json')} == queue_bytes
    assert _tree_bytes(bundle) == bundle_bytes
    assert {path.name: path.read_bytes() for path in (source / 'fallback_pending').glob('*.json')} == queue_bytes
    with SQLiteRecordStore.open_read_only(destination / 'sqlite', IDENTITY) as store:
        assert store.get_record('legacy-memory-1').to_dict()['metadata']['importance_score'] == 85

    snapshot = snapshot_offline(destination, tmp_path / 'review-snapshot', IDENTITY)
    review = inventory_snapshot(snapshot, IDENTITY)
    dispositions = {entry['file'].split('/')[-1]: entry['classification']
                    for entry in review['files']}
    assert dispositions == {
        'valid-sqlite.json': 'sqlite_pending',
        'identity-mismatch.json': 'sqlite_binding_mismatch',
        'clock-mismatch.json': 'sqlite_binding_mismatch',
        'legacy-pending.json': 'legacy_pending',
        'malformed.json': 'invalid_pending',
    }
    assert all(entry['action'] == 'retain' for entry in review['files'])
    assert {path.name: path.read_bytes() for path in (destination / 'fallback_pending').glob('*.json')} == queue_bytes


def test_runtime_bundle_restores_mandatory_decay_and_pop_lifecycle_across_restart(tmp_path):
    from instinct import dormancy, session_timer
    from instinct.models import ScoredSegment, Segment, SegmentType

    source = tmp_path / 'legacy-copy'
    _write_runtime_fixture(source)
    legacy_memory = sample_record()
    legacy_memory['metadata']['active_time_anchors'] = json.dumps({
        'pop_anchor_at': {
            'clock_id': 'synthetic-clock', 'seconds': 3000.0,
            'timestamp': '2026-10-08T09:45:00', 'score': 85,
        },
    })
    mandatory_segment = ScoredSegment(
        segment=Segment(content='Synthetic mandatory renewal fixture',
                        segment_type=SegmentType.STANDING_INSTRUCTION,
                        timestamp=datetime(2026, 10, 8, 9, 30),
                        stored_id='mandatory-memory'),
        importance_score=79, concept_title='Synthetic mandatory fixture',
    )
    mandatory_record = {
        'id': 'mandatory-memory',
        'content': mandatory_segment.segment.content,
        'metadata': mandatory_segment.to_storage_dict(),
        'embedding': [0.1, 0.2, 0.3],
    }
    bundle = tmp_path / 'bundle'
    export_runtime_bundle(bundle, IDENTITY,
                          SyntheticCollection([legacy_memory, mandatory_record]), source)
    destination = tmp_path / 'sqlite-runtime'
    import_runtime_bundle(bundle, destination, IDENTITY, apply=True)
    config = dormancy.config_for(destination)

    with SQLiteRuntimeSession.open(destination / 'sqlite', IDENTITY,
                                   persist_dir=destination, read_only=False) as session:
        first = session.tick_activity(idle_minutes=7.5, now=datetime(2026, 10, 8, 10, 20))
        assert first.session.new_session_started
        assert first.session.state.accumulated_session_seconds == 3600
        with session.operation() as adapter:
            renewed = adapter.get('mandatory-memory')
            assert renewed.importance_score == 100
            context = adapter.get('legacy-memory-1')
            before_decay = dormancy.current_effective_score(
                context, datetime(2026, 10, 8, 10, 20), config)
            assert context.accesses_since_pop == 3
            assert context.popped_at == datetime(2026, 10, 8, 9, 45)

        active_tick = session.tick_activity(
            idle_minutes=7.5, now=datetime(2026, 10, 8, 10, 21))
        assert active_tick.session.state.accumulated_session_seconds == 3660
        with session.operation() as adapter:
            assert adapter.record_access('legacy-memory-1', now=datetime(2026, 10, 8, 10, 21))
            after_access = adapter.get('legacy-memory-1')
            after_decay = dormancy.current_effective_score(
                after_access, datetime(2026, 10, 8, 10, 21), config)
            assert after_decay < before_decay

    prior_clock = session_timer.state(destination)
    with SQLiteRuntimeSession.open(destination / 'sqlite', IDENTITY,
                                   persist_dir=destination, read_only=False) as restarted:
        resumed = restarted.tick_activity(
            idle_minutes=7.5, now=datetime(2026, 10, 8, 10, 40))
        assert resumed.session.new_session_started
        assert resumed.session.state.accumulated_session_seconds == prior_clock.accumulated_session_seconds
        with restarted.operation() as adapter:
            renewed = adapter.get('mandatory-memory')
            restored = adapter.get('legacy-memory-1')
            assert renewed.importance_score == 100
            assert restored.access_count == 5
            assert restored.accesses_since_pop == 4
            assert restored.popped_at == datetime(2026, 10, 8, 9, 45)
            assert restored.last_accessed == datetime(2026, 10, 8, 10, 21)


@pytest.mark.parametrize('failure', ['database-build', 'verification', 'publication'])
def test_failed_runtime_import_rolls_back_destination_and_preserves_bundle(
        tmp_path, monkeypatch, failure):
    from instinct import sqlite_migration

    source = tmp_path / 'legacy-copy'
    _write_runtime_fixture(source)
    bundle = tmp_path / 'bundle'
    export_runtime_bundle(bundle, IDENTITY, SyntheticCollection([sample_record()]), source)
    bundle_before = _tree_bytes(bundle)
    source_before = _tree_bytes(source)
    destination = tmp_path / 'sqlite-runtime'

    if failure == 'database-build':
        def interrupt_build(*_args, **_kwargs):
            raise OSError('synthetic interrupted database build')
        monkeypatch.setattr(SQLiteRecordStore, 'insert_records', interrupt_build)
        expected_error = 'interrupted database build'
    elif failure == 'verification':
        monkeypatch.setattr(SQLiteRecordStore, 'get_record', lambda *_args, **_kwargs: None)
        expected_error = 'verification failed'
    else:
        def fail_publication(*_args, **_kwargs):
            raise OSError('synthetic atomic publication failure')
        monkeypatch.setattr(sqlite_migration.os, 'rename', fail_publication)
        expected_error = 'publication failure'

    with pytest.raises((OSError, ValueError), match=expected_error):
        import_runtime_bundle(bundle, destination, IDENTITY, apply=True)

    assert not destination.exists()
    assert _tree_bytes(bundle) == bundle_before
    assert _tree_bytes(source) == source_before
    assert not list(tmp_path.glob('.sqlite-runtime-import-*'))


def test_runtime_bundle_refuses_invalid_clock_and_keeps_source_unchanged(tmp_path):
    source = tmp_path / 'legacy-copy'
    source.mkdir(mode=0o700)
    invalid = b'{"session_started_at": "bad", "session_last_tick_at": null, "accumulated_session_seconds": -1}'
    write_private_text(source / 'session_state.json', invalid.decode())
    destination = tmp_path / 'bundle'

    with pytest.raises(ValueError, match='Invalid session'):
        export_runtime_bundle(destination, IDENTITY, SyntheticCollection([]), source)

    assert (source / 'session_state.json').read_bytes() == invalid
    assert not destination.exists()


def test_runtime_bundle_rejects_tampering_before_creating_destination(tmp_path):
    source = tmp_path / 'legacy-copy'
    _write_runtime_fixture(source)
    bundle = tmp_path / 'bundle'
    export_runtime_bundle(bundle, IDENTITY, SyntheticCollection([sample_record()]), source)
    target_file = next((bundle / 'runtime' / 'fallback_pending').glob('*.json'))
    target_file.write_bytes(target_file.read_bytes() + b' ')
    destination = tmp_path / 'sqlite-runtime'

    with pytest.raises(ValueError, match='modified'):
        import_runtime_bundle(bundle, destination, IDENTITY, apply=True)

    assert not destination.exists()


@pytest.mark.parametrize(('field', 'value'), [
    ('segment_type', 'not-a-segment-type'),
    ('keyword_hits_log', '{broken'),
    ('keyword_hits_log', '{}'),
    ('keyword_hits_log', '[12]'),
    ('keyword_hits_log', '[NaN]'),
    ('keyword_hits_log', '[{"count": 1, "count": 2}]'),
    ('active_time_anchors', '{broken'),
    ('active_time_anchors', '[]'),
    ('active_time_anchors', '{"created": 12}'),
    ('active_time_anchors', {'created': {'seconds': 1200}}),
])
def test_runtime_bundle_rejects_self_consistent_but_unreadable_records(tmp_path, field, value):
    from instinct import sqlite_migration
    from instinct.sqlite_records import canonical_json

    source = tmp_path / 'legacy-copy'
    _write_runtime_fixture(source)
    bundle = tmp_path / 'bundle'
    export_runtime_bundle(bundle, IDENTITY, SyntheticCollection([sample_record()]), source)
    record_path = bundle / 'records.json'
    envelope = json.loads(record_path.read_text(encoding='utf-8'))
    envelope['payload']['records'][0]['metadata'][field] = value
    envelope['sha256'] = sqlite_migration._checksum(envelope['payload'])
    record_bytes = (canonical_json(envelope) + '\n').encode('ascii')
    record_path.write_bytes(record_bytes)
    manifest_path = bundle / 'bundle.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files']['records.json'] = {
        'bytes': len(record_bytes), 'sha256': hashlib.sha256(record_bytes).hexdigest(),
    }
    write_private_text(manifest_path, canonical_json(manifest) + '\n')
    destination = tmp_path / 'sqlite-runtime'

    with pytest.raises(ValueError, match='runtime-incompatible'):
        import_runtime_bundle(bundle, destination, IDENTITY, apply=True)

    assert not destination.exists()


def test_runtime_bundle_refuses_unknown_pending_files(tmp_path):
    source = tmp_path / 'legacy-copy'
    source.mkdir(mode=0o700)
    pending = source / 'fallback_pending'
    pending.mkdir(mode=0o700)
    write_private_text(pending / 'unexpected.tmp', 'synthetic')

    with pytest.raises(ValueError, match='Unknown pending'):
        export_runtime_bundle(tmp_path / 'bundle', IDENTITY, SyntheticCollection([]), source)

    assert not (tmp_path / 'bundle').exists()
