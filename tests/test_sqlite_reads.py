# SPDX-License-Identifier: MPL-2.0
"""Synthetic score, recycle and ordered batch-read checks."""

import pytest

from instinct.sqlite_records import MAX_RECORDS, EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_store import SQLiteRecordStore


@pytest.fixture
def record_store(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with SQLiteRecordStore.create(tmp_path / 'records', identity) as store:
        payloads = [
            ('mandatory', {'importance_score': 100}),
            ('retired', {'importance_score': 100, 'recycled_at': 'fixture'}),
            ('alpha', {'importance_score': 80, 'owner_extra': {'keep': True}}),
            ('beta', {'importance_score': 80, 'recycled_at': ''}),
            ('low', {'importance_score': 0}),
            ('unscored', {}),
        ]
        store.insert_records([SQLiteRecord.from_dict({
            'id': record_id, 'content': record_id, 'metadata': metadata,
            'embedding': [1, 0],
        }, identity) for record_id, metadata in payloads])
        yield store, tmp_path / 'records'


def ids(records):
    return [record.record_id for record in records]


def test_batch_preserves_order_duplicates_missing_and_recycled(record_store):
    store, _ = record_store
    assert ids(store.get_records(['beta', 'missing', 'retired', 'beta', 'alpha'])) == [
        'beta', 'retired', 'beta', 'alpha']
    assert store.get_records([]) == []
    assert not store._connection.in_transaction


def test_batch_spans_sql_chunks_and_retains_fields(record_store):
    store, _ = record_store
    requested = [f'missing-{index}' for index in range(600)] + ['alpha']
    assert store.get_records(requested) == [store.get_record('alpha')]
    assert store.get_records(['alpha'])[0].to_dict()['metadata']['owner_extra'] == {'keep': True}


@pytest.mark.parametrize('requested', ['alpha', None, [''], [False], ['x\x00'],
                                        ['x' * 257], ['alpha'] * (MAX_RECORDS + 1)])
def test_invalid_batch_rejected(record_store, requested):
    store, _ = record_store
    with pytest.raises(ValueError):
        store.get_records(requested)
    assert not store._connection.in_transaction


def test_list_scores_sorted_and_filtered_before_limit(record_store):
    store, _ = record_store
    assert ids(store.list_records()) == ['mandatory', 'alpha', 'beta', 'low']
    assert ids(store.list_records(min_score=80, max_score=80, limit=1)) == ['alpha']
    assert ids(store.list_records(min_score=100, max_score=100)) == ['mandatory']
    assert ids(store.list_records(recycled='recycled', limit=1)) == ['retired']
    assert ids(store.list_records(recycled='all', limit=2)) == ['mandatory', 'retired']
    assert store.list_records(min_score=81, max_score=99) == []


@pytest.mark.parametrize('options', [
    {'min_score': -1}, {'max_score': 101}, {'min_score': True}, {'max_score': 80.0},
    {'min_score': 90, 'max_score': 80}, {'limit': 0}, {'limit': True},
    {'limit': MAX_RECORDS + 1}, {'recycled': True}, {'recycled': 'invalid'},
])
def test_invalid_listing_rejected(record_store, options):
    store, _ = record_store
    with pytest.raises(ValueError):
        store.list_records(**options)


def test_reads_preserve_records_and_work_after_reopen(record_store):
    store, destination = record_store
    before = store.get_records(['alpha', 'mandatory', 'retired'])
    store.list_records(recycled='all')
    assert store.get_records(['alpha', 'mandatory', 'retired']) == before
    with SQLiteRecordStore.open_read_only(destination, store.identity) as reader:
        assert reader.get_records(['alpha', 'mandatory', 'retired']) == before
        assert ids(reader.list_records()) == ['mandatory', 'alpha', 'beta', 'low']


def test_corrupt_vector_fails_closed_and_releases_batch_snapshot(record_store):
    store, _ = record_store
    store._connection.execute('UPDATE records SET embedding = ? WHERE id = ?', (b'bad', 'alpha'))
    with pytest.raises(ValueError, match='dimension'):
        store.get_records(['alpha'])
    assert not store._connection.in_transaction
    with pytest.raises(ValueError, match='dimension'):
        store.list_records()
    assert store.get_records(['mandatory'])[0].record_id == 'mandatory'


def test_batch_does_not_commit_callers_transaction(record_store):
    store, _ = record_store
    store._connection.execute('BEGIN IMMEDIATE')
    store._connection.execute('UPDATE records SET content = ? WHERE id = ?', ('pending', 'alpha'))
    assert store.get_records(['alpha'])[0].content == 'pending'
    assert store._connection.in_transaction
    store._connection.rollback()
    assert store.get_record('alpha').content == 'alpha'


def test_batch_failure_does_not_release_callers_transaction(record_store):
    store, _ = record_store
    store._connection.execute('BEGIN IMMEDIATE')
    store._connection.execute('UPDATE records SET embedding = ? WHERE id = ?', (b'bad', 'alpha'))
    with pytest.raises(ValueError, match='dimension'):
        store.get_records(['alpha'])
    assert store._connection.in_transaction
    store._connection.rollback()
    assert store.get_record('alpha').embedding == (1.0, 0.0)


def test_batch_binds_ids_instead_of_interpreting_sql(record_store):
    store, _ = record_store
    assert store.get_records(["alpha' OR 1=1 --"]) == []
    assert store.count() == 6
