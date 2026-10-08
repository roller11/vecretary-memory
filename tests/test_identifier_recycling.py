# SPDX-License-Identifier: MPL-2.0
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from instinct.models import RECYCLED_AT_KEY, ScoredSegment, Segment, SegmentType
from instinct.store import InstinctStore


class SyntheticCollection:
    def __init__(self, persist_dir):
        self.persist_dir = str(persist_dir)
        self.records = {}

    def count(self):
        return len(self.records)

    def upsert(self, ids, documents, metadatas):
        for segment_id, document, metadata in zip(ids, documents, metadatas):
            previous = self.records.get(segment_id, {}).get('metadata', {})
            self.records[segment_id] = {
                'document': document, 'metadata': {**previous, **deepcopy(metadata)},
            }

    def get(self, ids=None, limit=None, offset=0, include=None):
        selected = list(self.records) if ids is None else [
            segment_id for segment_id in ids if segment_id in self.records
        ]
        selected = selected[offset:][:limit]
        return {
            'ids': selected,
            'documents': [self.records[segment_id]['document'] for segment_id in selected],
            'metadatas': [deepcopy(self.records[segment_id]['metadata']) for segment_id in selected],
        }


def make_store(collection):
    store = InstinctStore.__new__(InstinctStore)
    store.persist_dir = collection.persist_dir
    store._collection = collection
    store._read_only = False
    store._identifier_index = defaultdict(set)
    store._rebuild_identifier_index()
    return store


def make_memory(index=0, score=70, identifiers=None, **metadata):
    return ScoredSegment(
        segment=Segment(
            content=f'Synthetic identifier regression {index}',
            segment_type=SegmentType.CONTEXT,
            timestamp=datetime(2000, 1, 1) + timedelta(seconds=index),
            metadata=metadata,
        ),
        importance_score=score,
        concept_title=f'Synthetic note {index}',
        memory_identifiers=identifiers or ['safe recovery'],
    )


@pytest.fixture
def store(tmp_path):
    return make_store(SyntheticCollection(tmp_path))


def test_rebuilt_index_excludes_recycled_but_preserves_owner_fetch(store):
    active = make_memory()
    retired = make_memory(1)
    store.add_batch([active, retired])
    store.recycle(retired.id, by='synthetic-owner')
    restarted = make_store(store._collection)
    assert restarted._identifier_index['safe recovery'] == {active.id}
    assert [hit[0].id for hit in restarted.search_by_identifiers(['safe recovery'])] == [active.id]
    assert restarted.get(retired.id).segment.metadata[RECYCLED_AT_KEY]
    assert [memory.id for memory in restarted.get_many([retired.id])] == [retired.id]


@pytest.mark.parametrize('batch', [False, True])
def test_already_recycled_records_are_not_indexed(store, batch):
    retired = make_memory(recycled_at='synthetic-owner-retired')
    if batch:
        store.add_batch([retired])
    else:
        store.add(retired)
    assert not store.search_by_identifiers(['safe recovery'])
    assert retired.id not in store._identifier_index.get('safe recovery', set())


def test_stale_observer_index_rechecks_current_metadata(store):
    memory = make_memory()
    store.add(memory)
    observer = make_store(store._collection)
    store.recycle(memory.id, by='synthetic-owner')
    assert memory.id in observer._identifier_index['safe recovery']
    assert not observer.search_by_identifiers(['safe recovery'])
    assert observer.get(memory.id).segment.metadata[RECYCLED_AT_KEY]


def test_stale_recycled_candidates_do_not_consume_limit_across_batches(store):
    memories = [make_memory(index) for index in range(102)]
    store.add_batch(memories)
    observer = make_store(store._collection)
    for memory in memories[:-1]:
        store._collection.records[memory.id]['metadata'][RECYCLED_AT_KEY] = 'synthetic-retired'
    hits = observer.search_by_identifiers(['safe recovery'], limit=1)
    assert [hit[0].id for hit in hits] == [memories[-1].id]


def test_restored_record_is_searchable_before_and_after_rebuild(store):
    memory = make_memory()
    store.add(memory)
    store.recycle(memory.id, by='synthetic-owner')
    assert not store.search_by_identifiers(['safe recovery'])
    store.restore(memory.id)
    assert store.get(memory.id).segment.metadata[RECYCLED_AT_KEY] == ''
    for current in [store, make_store(store._collection)]:
        assert [hit[0].id for hit in current.search_by_identifiers(['safe recovery'])] == [memory.id]


@pytest.mark.parametrize('marker', [None, ''])
def test_falsey_recycle_marker_remains_active(store, marker):
    memory = make_memory(recycled_at=marker)
    store.add(memory)
    restarted = make_store(store._collection)
    assert restarted.search_by_identifiers(['safe recovery'])[0][0].id == memory.id


def test_active_hit_order_counts_and_identifier_normalization_are_unchanged(store):
    low = make_memory(0, score=60, identifiers=['safe recovery', 'workflow'])
    high = make_memory(1, score=90, identifiers=['safe recovery', 'workflow'])
    single = make_memory(2, score=95)
    store.add_batch([single, low, high])
    hits = store.search_by_identifiers(['SAFE recovery', 'workflow', 'safe recovery'])
    assert [(hit[0].id, hit[1], hit[2]) for hit in hits] == [
        (high.id, 2, ['safe recovery', 'workflow']),
        (low.id, 2, ['safe recovery', 'workflow']),
        (single.id, 1, ['safe recovery']),
    ]
    assert [hit[0].id for hit in store.search_by_identifiers(['workflow'], limit=1)] == [low.id]


@pytest.mark.parametrize('limit', [0, -1])
def test_nonpositive_identifier_limit_returns_no_results(store, limit):
    store.add(make_memory())
    assert store.search_by_identifiers(['safe recovery'], limit=limit) == []
