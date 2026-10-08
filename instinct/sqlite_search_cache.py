# SPDX-License-Identifier: MPL-2.0
"""Bounded connection-local search cache, checked against every current SQL row."""

from dataclasses import dataclass
import json
import struct
import sys

from .sqlite_records import SQLiteRecord
from .sqlite_search import prepare_cosine_vector


MAX_CACHE_RECORDS = 512
MAX_CACHE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class SearchCandidate:
    row: tuple
    record: SQLiteRecord
    cosine: object
    retained_bytes: int


class SearchRecordCache:
    """Keep admitted rows across scans; bypass capacity misses instead of churning."""

    def __init__(self):
        self._identity = None
        self._entries = {}
        self._unseen = set()
        self.retained_bytes = 0

    def clear(self):
        self._entries.clear()
        self._unseen.clear()
        self.retained_bytes = 0
        self._identity = None

    def begin_scan(self):
        self._unseen = set(self._entries)

    def end_scan(self, *, completed):
        if completed:
            for record_id in self._unseen:
                removed = self._entries.pop(record_id, None)
                if removed is not None:
                    self.retained_bytes -= removed.retained_bytes
        self._unseen.clear()

    def candidate(self, row, identity):
        if identity != self._identity:
            self.clear()
            self._identity = identity
        self._unseen.discard(row[0])
        previous = self._entries.get(row[0])
        if previous is not None:
            if row == previous.row:
                return previous
            del self._entries[row[0]]
            self.retained_bytes -= previous.retained_bytes
        if len(row[3]) != 8 * identity.dimension:
            raise ValueError('Stored embedding dimension mismatch')
        record = SQLiteRecord.from_dict({
            'id': row[0], 'content': row[1], 'metadata': json.loads(row[2]),
            'embedding': struct.unpack(f'<{identity.dimension}d', row[3]),
        }, identity)
        cosine = prepare_cosine_vector(record.embedding) if identity.metric == 'cosine' else None
        retained_bytes = (512 + sys.getsizeof(row) + sum(sys.getsizeof(value) for value in row)
                          + sys.getsizeof(record) + sys.getsizeof(record.__dict__)
                          + sys.getsizeof(record.metadata_json) + sys.getsizeof(record.embedding)
                          + len(record.embedding) * sys.getsizeof(0.0))
        if cosine is not None:
            retained_bytes += (sys.getsizeof(cosine) + sys.getsizeof(cosine.__dict__)
                               + sys.getsizeof(cosine.components) + sys.getsizeof(cosine.norm)
                               + len(cosine.components) * sys.getsizeof(0.0))
        candidate = SearchCandidate(row, record, cosine, retained_bytes)
        if (len(self._entries) < MAX_CACHE_RECORDS
                and self.retained_bytes + retained_bytes <= MAX_CACHE_BYTES):
            self._entries[row[0]] = candidate
            self.retained_bytes += retained_bytes
        return candidate
