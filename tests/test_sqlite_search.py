# SPDX-License-Identifier: MPL-2.0
"""Synthetic exact-search and Chroma distance compatibility checks."""

import pytest
import struct
from unittest.mock import patch

from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_search import vector_distance, prepare_cosine_vector, prepared_cosine_distance
from instinct.sqlite_store import SQLiteRecordStore


def make_record(identity, identifier, vector, **metadata):
    return SQLiteRecord.from_dict({
        'id': identifier, 'content': 'Synthetic vector fixture',
        'metadata': metadata, 'embedding': vector,
    }, identity)


@pytest.fixture
def search_store(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with SQLiteRecordStore.create(tmp_path / 'vectors', identity) as store:
        store.insert_records([
            make_record(identity, 'best', [1, 0], importance_score=40, segment_type='context'),
            make_record(identity, 'next', [1, 1], importance_score=80, segment_type='context'),
            make_record(identity, 'other', [0, 1], importance_score=90, segment_type='decision'),
            make_record(identity, 'recycled', [2, 0], importance_score=100,
                        segment_type='context', recycled_at='2026-01-01'),
        ])
        yield store


def test_search_filters_before_ranking_and_threshold_after_limit(search_store):
    assert [hit.record.record_id for hit in search_store.search_vectors([1, 0])] == ['best', 'next', 'other']
    hits = search_store.search_vectors([1, 0], limit=1, min_importance=50, segment_types=['context'])
    assert [hit.record.record_id for hit in hits] == ['next']
    assert hits[0].distance == pytest.approx(1 - 2 ** -0.5)
    assert search_store.search_vectors([1, 0], min_importance=50, min_similarity=0.8) == []
    assert [hit.record.record_id for hit in search_store.search_vectors([1, 0], include_recycled=True)][:2] == ['best', 'recycled']


def test_read_only_search_has_no_score_or_access_side_effects(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    record = make_record(identity, 'note', [1, 0], importance_score=84, access_count=7)
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([record])
    with SQLiteRecordStore.open_read_only(tmp_path / 'records', identity) as store:
        assert store.search_vectors([1, 0])[0].record == record
        assert store.get_record('note') == record


@pytest.mark.parametrize('options', [
    {'limit': True}, {'limit': 0}, {'limit': 10001},
    {'min_importance': True}, {'min_importance': 101},
    {'include_recycled': 1}, {'segment_types': 'context'},
    {'segment_types': [None]}, {'min_similarity': True},
    {'min_similarity': float('nan')}, {'min_similarity': 1.1},
])
def test_invalid_search_options_fail(search_store, options):
    with pytest.raises(ValueError):
        search_store.search_vectors([1, 0], **options)


@pytest.mark.parametrize('vector', [[1], [float('inf'), 0], [True, 1]])
def test_invalid_query_vector_fails(search_store, vector):
    with pytest.raises(ValueError):
        search_store.search_vectors(vector)


@pytest.mark.parametrize('metric,expected', [('cosine', 1.0), ('l2', 13.0), ('ip', 1.0)])
def test_distance_conventions(metric, expected):
    assert vector_distance((2.0, 0.0), (0.0, 3.0), metric) == expected


def test_cosine_handles_large_tiny_and_zero_vectors():
    for magnitude in (1e308, 1e-308):
        assert vector_distance((magnitude, magnitude), (magnitude, magnitude), 'cosine') == pytest.approx(0)
    assert vector_distance((0, 0), (1, 0), 'cosine') == 1
    assert vector_distance((1, 0), (-1, 0), 'cosine') == 2


@pytest.mark.parametrize('metric', ['l2', 'ip'])
def test_nonfinite_distance_fails_closed(metric):
    with pytest.raises(ValueError, match='finite'):
        vector_distance((1e308, 1e308), (-1e308, -1e308), metric)


def test_empty_search_and_noncosine_similarity_rejection(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'l2')
    with SQLiteRecordStore.create(tmp_path / 'empty', identity) as store:
        assert store.search_vectors([1, 0]) == []
        with pytest.raises(ValueError, match='cosine'):
            store.search_vectors([1, 0], min_similarity=0)


def test_total_record_limit_rolls_back_batch(search_store, monkeypatch):
    monkeypatch.setattr('instinct.sqlite_store.MAX_RECORDS', 4)
    with pytest.raises(ValueError, match='count'):
        search_store.insert_records([make_record(search_store.identity, 'excess', [1, 0])])
    assert search_store.count() == 4
    assert search_store.get_record('excess') is None


@pytest.mark.parametrize('metric', ['cosine', 'l2', 'ip'])
def test_chroma_synthetic_distance_and_order_parity(tmp_path, metric):
    chromadb = pytest.importorskip('chromadb')
    from chromadb.config import Settings

    identity = EmbeddingIdentity('synthetic', 'fixed', 2, metric)
    records = [make_record(identity, identifier, vector) for identifier, vector in
               [('first', [1, 0]), ('second', [0.5, 1]), ('third', [-1, 0])]]
    client = chromadb.PersistentClient(path=str(tmp_path / 'synthetic-chroma'),
                                      settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection('synthetic-parity', embedding_function=None,
                                          configuration={'hnsw': {'space': metric}})
    collection.add(ids=[record.record_id for record in records],
                   embeddings=[list(record.embedding) for record in records])
    expected = collection.query(query_embeddings=[[1, 0]], n_results=3, include=['distances'])
    with SQLiteRecordStore.create(tmp_path / 'synthetic-sqlite', identity) as store:
        store.insert_records(records)
        hits = store.search_vectors([1, 0], limit=3)
        assert [hit.record.record_id for hit in hits] == expected['ids'][0]
        assert [hit.distance for hit in hits] == pytest.approx(expected['distances'][0], abs=1e-6)


def test_warm_search_reuses_only_unchanged_validated_records(search_store):
    cold = search_store.search_vectors([1, 0], include_recycled=True)
    with patch.object(SQLiteRecord, 'from_dict', wraps=SQLiteRecord.from_dict) as validate:
        warm = search_store.search_vectors([1, 0], include_recycled=True)
        assert validate.call_count == 1
    assert warm == cold


def test_cached_search_observes_vector_content_and_score_changes(search_store):
    search_store.search_vectors([1, 0], include_recycled=True)
    replacement = make_record(search_store.identity, 'best', [-1, 0], importance_score=85,
                              segment_type='decision')
    search_store.upsert_records([replacement])
    hits = search_store.search_vectors([-1, 0], min_importance=80, segment_types=['decision'])
    assert hits[0].record == replacement
    search_store.patch_metadata('best', {'importance_score': 20})
    assert 'best' not in [hit.record.record_id for hit in search_store.search_vectors([-1, 0], min_importance=80)]


def test_cached_search_observes_recycle_restore_and_purge(search_store):
    search_store.search_vectors([1, 0])
    search_store.recycle('best')
    assert 'best' not in [hit.record.record_id for hit in search_store.search_vectors([1, 0])]
    search_store.restore('best')
    assert search_store.search_vectors([1, 0])[0].record.record_id == 'best'
    search_store.recycle('best')
    assert search_store.purge_record('best')
    assert 'best' not in [hit.record.record_id for hit in search_store.search_vectors([1, 0], include_recycled=True)]


@pytest.mark.parametrize('column,value,message', [
    ('embedding', b'bad', 'dimension'),
    ('embedding', struct.pack('<2d', float('nan'), 0), 'finite'),
    ('content', 'invalid\x00body', 'content'),
    ('metadata', '{"importance_score":101}', 'importance_score'),
    ('metadata', '{"importance_score":100,"extra":NaN}', 'finite'),
])
def test_changed_nonwinning_filtered_row_still_fails_closed(search_store, column, value, message):
    search_store.search_vectors([1, 0])
    search_store._connection.execute(f'UPDATE records SET {column} = ? WHERE id = ?', (value, 'recycled'))
    with pytest.raises(ValueError, match=message):
        search_store.search_vectors([1, 0], limit=1, min_importance=80)


def test_cache_tracks_rollback_and_does_not_return_uncommitted_metadata(search_store):
    baseline = search_store.search_vectors([1, 0])
    search_store._connection.execute('BEGIN IMMEDIATE')
    try:
        search_store._connection.execute('UPDATE records SET metadata = ? WHERE id = ?',
                                        ('{"importance_score":99}', 'best'))
        assert search_store.search_vectors([1, 0])[0].record.to_dict()['metadata']['importance_score'] == 99
    finally:
        search_store._connection.rollback()
    assert search_store.search_vectors([1, 0]) == baseline


def test_observer_cache_sees_committed_changes_from_owner(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    root = tmp_path / 'records'
    with SQLiteRecordStore.create(root, identity) as owner:
        owner.insert_records([make_record(identity, 'note', [1, 0], importance_score=80)])
        with SQLiteRecordStore.open_read_only(root, identity) as observer:
            assert observer.search_vectors([1, 0])[0].distance == 0
            replacement = make_record(identity, 'note', [-1, 0], importance_score=70)
            owner.upsert_records([replacement])
            hit = observer.search_vectors([1, 0])[0]
            assert hit.distance == 2 and hit.record == replacement


@pytest.mark.parametrize('metric', ['cosine', 'l2', 'ip'])
def test_cache_eviction_and_oversize_bypass_preserve_results(tmp_path, monkeypatch, metric):
    from instinct import sqlite_search_cache
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, metric)
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([make_record(identity, str(index), [index + 1, 1]) for index in range(4)])
        baseline = store.search_vectors([1, 0])
        store._search_cache.clear()
        monkeypatch.setattr(sqlite_search_cache, 'MAX_CACHE_RECORDS', 2)
        assert store.search_vectors([1, 0]) == baseline
        assert len(store._search_cache._entries) == 2
        assert store._search_cache.retained_bytes <= sqlite_search_cache.MAX_CACHE_BYTES
        store._search_cache.clear()
        monkeypatch.setattr(sqlite_search_cache, 'MAX_CACHE_BYTES', 1)
        assert store.search_vectors([1, 0]) == baseline
        assert store._search_cache.retained_bytes == 0
        assert not store._search_cache._entries


def test_cache_releases_private_record_state_on_close(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([make_record(identity, 'note', [1, 0])])
        store.search_vectors([1, 0])
        assert store._search_cache.retained_bytes > 0
    assert not store._search_cache._entries
    assert store._search_cache.retained_bytes == 0


def test_cache_byte_budget_evicts_even_below_record_limit(search_store, monkeypatch):
    from instinct import sqlite_search_cache
    baseline = search_store.search_vectors([1, 0], include_recycled=True)
    largest = max(candidate.retained_bytes for candidate in search_store._search_cache._entries.values())
    search_store._search_cache.clear()
    monkeypatch.setattr(sqlite_search_cache, 'MAX_CACHE_BYTES', largest + 16)
    assert search_store.search_vectors([1, 0], include_recycled=True) == baseline
    assert len(search_store._search_cache._entries) == 1
    assert search_store._search_cache.retained_bytes <= largest + 16


def test_cache_embedding_identity_change_requires_fresh_validation(search_store):
    from instinct.sqlite_search_cache import SearchRecordCache
    row = search_store._connection.execute('SELECT id, content, metadata, embedding FROM records WHERE id = ?', ('best',)).fetchone()
    cache = SearchRecordCache()
    original = cache.candidate(row, search_store.identity)
    changed = EmbeddingIdentity('synthetic', 'another-model', 2, 'cosine')
    assert cache.candidate(row, changed) is not original
    assert len(cache._entries) == 1
    with pytest.raises(ValueError, match='dimension'):
        cache.candidate(row, EmbeddingIdentity('synthetic', 'another-model', 3, 'cosine'))
    assert not cache._entries and cache.retained_bytes == 0


@pytest.mark.parametrize('query,candidate', [
    ((1.0, 0.0), (1.0, 1.0)), ((0.0, 0.0), (1.0, 0.0)),
    ((1e308, 1e308), (-1e308, 1e308)), ((1e-308, 1e-308), (1e-308, 1e-308)),
])
def test_prepared_cosine_keeps_exact_distance_convention(query, candidate):
    assert prepared_cosine_distance(prepare_cosine_vector(query), prepare_cosine_vector(candidate)) == vector_distance(query, candidate, 'cosine')
