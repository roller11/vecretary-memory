# SPDX-License-Identifier: MPL-2.0
"""Parity checks for bounded scan admission and conservative record-size checks."""

import json
import math
import random
import struct
from unittest.mock import patch

import pytest

from instinct import sqlite_records, sqlite_search_cache
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord, canonical_json
from instinct.sqlite_search import prepare_cosine_vector, prepared_cosine_distance
from instinct.sqlite_store import SQLiteRecordStore


def make_record(identity, record_id, embedding, metadata=None, content='Synthetic fixture'):
    return SQLiteRecord.from_dict({'id': record_id, 'content': content,
                                  'metadata': metadata or {}, 'embedding': embedding}, identity)


@pytest.mark.parametrize('metric', ['cosine', 'l2', 'ip'])
def test_over_capacity_scans_keep_reusing_admitted_rows(tmp_path, monkeypatch, metric):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, metric)
    monkeypatch.setattr(sqlite_search_cache, 'MAX_CACHE_RECORDS', 2)
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([make_record(identity, str(index), [index + 1, 1]) for index in range(4)])
        baseline = store.search_vectors([1, 0])
        admitted = dict(store._search_cache._entries)
        assert len(admitted) == 2
        with patch.object(SQLiteRecord, 'from_dict', wraps=SQLiteRecord.from_dict) as validate:
            for _ in range(3):
                assert store.search_vectors([1, 0]) == baseline
            assert validate.call_count == 9
        assert store._search_cache._entries == admitted
        assert store._search_cache.retained_bytes <= sqlite_search_cache.MAX_CACHE_BYTES


def test_complete_scan_reclaims_deleted_cache_entries(tmp_path, monkeypatch):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    monkeypatch.setattr(sqlite_search_cache, 'MAX_CACHE_RECORDS', 2)
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([make_record(identity, str(index), [index + 1, 1]) for index in range(4)])
        store.search_vectors([1, 0])
        deleted_id = next(iter(store._search_cache._entries))
        store.recycle(deleted_id)
        assert store.purge_record(deleted_id)
        hits = store.search_vectors([1, 0])
        assert deleted_id not in store._search_cache._entries
        assert len(store._search_cache._entries) == 1
        assert store.search_vectors([1, 0]) == hits
        assert len(store._search_cache._entries) == 2
        assert store._search_cache.retained_bytes == sum(
            candidate.retained_bytes for candidate in store._search_cache._entries.values())


def test_incomplete_scan_does_not_reclaim_unvisited_valid_entries(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([make_record(identity, str(index), [index + 1, 1]) for index in range(4)])
        store.search_vectors([1, 0])
        store._connection.execute('UPDATE records SET metadata = ? WHERE id = ?',
                                  ('{"importance_score":101}', '0'))
        with pytest.raises(ValueError, match='importance_score'):
            store.search_vectors([1, 0])
        assert set(store._search_cache._entries) == {'1', '2', '3'}
        assert not store._search_cache._unseen


def test_byte_capacity_does_not_thrash_admitted_entries(tmp_path, monkeypatch):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([make_record(identity, str(index), [index + 1, 1]) for index in range(4)])
        expected = store.search_vectors([1, 0])
        largest = max(candidate.retained_bytes for candidate in store._search_cache._entries.values())
        store._search_cache.clear()
        monkeypatch.setattr(sqlite_search_cache, 'MAX_CACHE_BYTES', largest * 2)
        assert store.search_vectors([1, 0]) == expected
        admitted = dict(store._search_cache._entries)
        assert len(admitted) == 2
        with patch.object(SQLiteRecord, 'from_dict', wraps=SQLiteRecord.from_dict) as validate:
            assert store.search_vectors([1, 0]) == expected
            assert validate.call_count == 3
        assert store._search_cache._entries == admitted


def test_changed_cached_row_growing_over_budget_is_revalidated_and_bypassed(tmp_path, monkeypatch):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        store.insert_records([make_record(identity, 'note', [1, 0])])
        store.search_vectors([1, 0])
        budget = store._search_cache.retained_bytes + 16
        monkeypatch.setattr(sqlite_search_cache, 'MAX_CACHE_BYTES', budget)
        replacement = make_record(identity, 'note', [-1, 0], content='Changed ' * 1000)
        store.upsert_records([replacement])
        assert store.search_vectors([1, 0])[0].record == replacement
        assert not store._search_cache._entries
        assert store._search_cache.retained_bytes == 0


@pytest.mark.parametrize('content', ['Synthetic plain text', '\u2603' * 20, '\\"\n' * 20])
@pytest.mark.parametrize('difference', [-1, 0, 1])
def test_size_boundary_matches_exact_canonical_bytes(monkeypatch, content, difference):
    vector = [1.7976931348623157e308, -5e-324, -0.0, 1e-308, -1e308]
    identity = EmbeddingIdentity('synthetic', 'fixed', len(vector), 'cosine')
    payload = {'id': 'note', 'content': content, 'metadata': {'label': '\u2603'}, 'embedding': vector}
    size = len(canonical_json(payload).encode('ascii'))
    monkeypatch.setattr(sqlite_records, 'MAX_RECORD_BYTES', size + difference)
    if difference < 0:
        with pytest.raises(ValueError, match='size limit'):
            SQLiteRecord.from_dict(payload, identity)
    else:
        assert SQLiteRecord.from_dict(payload, identity).to_dict() == payload


def test_normal_size_fast_path_never_serializes_vector(monkeypatch):
    identity = EmbeddingIdentity('synthetic', 'fixed', 768, 'cosine')
    original = sqlite_records.canonical_json
    def check(value):
        if isinstance(value, dict) and 'embedding' in value:
            assert value['embedding'] == []
        return original(value)
    monkeypatch.setattr(sqlite_records, 'canonical_json', check)
    assert make_record(identity, 'note', [0.25] * 768).embedding == (0.25,) * 768


def test_float_json_bound_and_cosine_arithmetic_parity():
    generator = random.Random(1729)
    vectors = [[generator.uniform(-1, 1) for _ in range(768)] for _ in range(10)]
    vectors.extend([[1e308, -1e308], [1e-308, -1e-308], [0.0, -0.0]])
    for vector in vectors:
        assert all(len(json.dumps(number, allow_nan=False)) + 1 <= sqlite_records.MAX_FLOAT_JSON_BYTES
                   for number in vector)
        prepared = prepare_cosine_vector(vector)
        scale = max(abs(number) for number in vector)
        components = tuple(number / scale for number in vector) if scale else ()
        assert prepared.components == components
        assert prepared.norm == math.hypot(*components)
        if prepared.norm:
            baseline = 1.0 - max(-1.0, min(1.0, math.fsum(
                left * right for left, right in zip(components, components)) / prepared.norm ** 2))
            assert prepared_cosine_distance(prepared, prepared) == baseline
        else:
            assert prepared_cosine_distance(prepared, prepared) == 1.0


def test_float_size_bound_covers_deterministic_binary64_patterns():
    generator = random.Random(1729)
    for _ in range(2000):
        number = struct.unpack('<d', generator.getrandbits(64).to_bytes(8, 'little'))[0]
        if math.isfinite(number):
            assert len(json.dumps(number, allow_nan=False)) + 1 <= sqlite_records.MAX_FLOAT_JSON_BYTES


def test_fast_size_check_preserves_nested_metadata_depth_limit():
    nested = 0
    for _ in range(31):
        nested = {'child': nested}
    metadata = {'nested': nested}
    canonical_json(metadata)
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with pytest.raises(ValueError, match='JSON nesting'):
        make_record(identity, 'note', [1, 0], metadata=metadata)
