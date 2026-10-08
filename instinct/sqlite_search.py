# SPDX-License-Identifier: MPL-2.0
"""Exact vector primitives for the experimental SQLite record repository."""

from dataclasses import dataclass
import math
import operator
from typing import Protocol

from .sqlite_records import EmbeddingIdentity, SQLiteRecord


@dataclass(frozen=True)
class VectorHit:
    record: SQLiteRecord
    distance: float


class RecordRepository(Protocol):
    identity: EmbeddingIdentity

    def get_record(self, record_id: str) -> SQLiteRecord | None: ...

    def get_records(self, record_ids: list[str]) -> list[SQLiteRecord]: ...

    def list_records(self, *, min_score=0, max_score=100, limit=100,
                     recycled='active') -> list[SQLiteRecord]: ...

    def count(self) -> int: ...

    def search_vectors(self, embedding, *, limit=10, min_similarity=None,
                       min_importance=None, segment_types=None,
                       include_recycled=False) -> list[VectorHit]: ...


class WritableRecordRepository(RecordRepository, Protocol):
    def insert_records(self, records) -> int: ...

    def upsert_records(self, records) -> int: ...

    def patch_metadata(self, record_id, changes, *, remove_keys=()) -> SQLiteRecord | None: ...

    def recycle(self, record_id, by='dashboard', reason='', *, now=None) -> SQLiteRecord | None: ...

    def restore(self, record_id) -> SQLiteRecord | None: ...

    def is_recycled(self, record_id) -> bool: ...


def validate_vector(embedding, identity):
    return SQLiteRecord.from_dict({
        'id': 'query', 'content': 'query', 'metadata': {},
        'embedding': embedding,
    }, identity).embedding


@dataclass(frozen=True)
class PreparedCosineVector:
    components: tuple[float, ...]
    norm: float


def prepare_cosine_vector(vector):
    scale = max(map(abs, vector))
    if scale == 0:
        return PreparedCosineVector((), 0.0)
    components = tuple(value / scale for value in vector)
    return PreparedCosineVector(components, math.hypot(*components))


def prepared_cosine_distance(query, candidate):
    if query.norm == 0 or candidate.norm == 0:
        return 1.0
    similarity = math.fsum(map(operator.mul, query.components, candidate.components))
    similarity /= query.norm * candidate.norm
    return 1.0 - max(-1.0, min(1.0, similarity))


def vector_distance(query, candidate, metric):
    try:
        if metric == 'cosine':
            distance = prepared_cosine_distance(prepare_cosine_vector(query), prepare_cosine_vector(candidate))
        elif metric == 'l2':
            distance = math.fsum((left - right) ** 2 for left, right in zip(query, candidate))
        elif metric == 'ip':
            distance = 1.0 - math.fsum(left * right for left, right in zip(query, candidate))
        else:
            raise ValueError('Unsupported distance metric')
    except (OverflowError, ValueError) as error:
        raise ValueError('Vector distance is not finite') from error
    if not math.isfinite(distance):
        raise ValueError('Vector distance is not finite')
    return distance
