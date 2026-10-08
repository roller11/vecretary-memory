# SPDX-License-Identifier: MPL-2.0
"""Bounded record reads without runtime scoring or access side effects."""

import heapq
import json
import struct

from .sqlite_records import MAX_RECORDS, SQLiteRecord, require_text


class SQLiteRecordReads:
    def _record_from_row(self, row):
        if len(row[3]) != 8 * self.identity.dimension:
            raise ValueError('Stored embedding dimension mismatch')
        return SQLiteRecord.from_dict({
            'id': row[0], 'content': row[1], 'metadata': json.loads(row[2]),
            'embedding': struct.unpack(f'<{self.identity.dimension}d', row[3]),
        }, self.identity)

    def get_records(self, record_ids):
        if not isinstance(record_ids, (list, tuple)) or len(record_ids) > MAX_RECORDS:
            raise ValueError('Invalid record IDs')
        for record_id in record_ids:
            require_text(record_id, 'record ID', 256)
        if not record_ids:
            return []
        records = {}
        self._connection.execute('SAVEPOINT batch_read')
        try:
            unique_ids = list(dict.fromkeys(record_ids))
            for start in range(0, len(unique_ids), 256):
                batch = unique_ids[start:start + 256]
                placeholders = ','.join('?' for _ in batch)
                cursor = self._connection.execute(
                    'SELECT id, content, metadata, embedding FROM records '
                    f'WHERE id IN ({placeholders})', batch)
                try:
                    for row in cursor:
                        record = self._record_from_row(row)
                        records[record.record_id] = record
                finally:
                    cursor.close()
        finally:
            self._connection.execute('RELEASE batch_read')
        return [records[record_id] for record_id in record_ids if record_id in records]

    def list_records(self, *, min_score=0, max_score=100, limit=100,
                     recycled='active'):
        if (type(min_score) is not int or type(max_score) is not int
                or not 0 <= min_score <= max_score <= 100):
            raise ValueError('Invalid score range')
        if type(limit) is not int or not 1 <= limit <= MAX_RECORDS:
            raise ValueError('Invalid record limit')
        if recycled not in ('active', 'recycled', 'all'):
            raise ValueError('Invalid recycled selection')

        def candidates():
            for index, row in enumerate(cursor):
                if index >= MAX_RECORDS:
                    raise ValueError('Record count exceeds scaffold limit')
                record = self._record_from_row(row)
                metadata = record.to_dict()['metadata']
                score = metadata.get('importance_score')
                if score is None or not min_score <= score <= max_score:
                    continue
                is_recycled = bool(metadata.get('recycled_at'))
                if recycled == 'active' and is_recycled:
                    continue
                if recycled == 'recycled' and not is_recycled:
                    continue
                yield score, record

        cursor = self._connection.execute('SELECT id, content, metadata, embedding FROM records')
        try:
            selected = heapq.nsmallest(limit, candidates(),
                                      key=lambda item: (-item[0], item[1].record_id))
        finally:
            cursor.close()
        return [record for score, record in selected]
