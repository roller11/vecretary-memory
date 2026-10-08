# SPDX-License-Identifier: MPL-2.0
"""Transactional record metadata mutations, without runtime lifecycle side effects."""

from datetime import datetime
import json

from .sqlite_records import SQLiteRecord, canonical_json, require_text


RECYCLE_KEYS = ('recycled_at', 'recycled_by', 'recycled_reason')


class SQLiteRecordMutations:
    def purge_record(self, record_id):
        if self.read_only:
            raise PermissionError('SQLite record store is read-only')
        require_text(record_id, 'record ID', 256)
        self._connection.execute('BEGIN IMMEDIATE')
        try:
            record = self.get_record(record_id)
            if record is not None and not record.to_dict()['metadata'].get('recycled_at'):
                raise ValueError('Record must be recycled before permanent purge')
            if record is not None:
                self._connection.execute('DELETE FROM records WHERE id = ?', (record_id,))
            self._connection.commit()
            return record is not None
        except BaseException:
            self._connection.rollback()
            raise

    def _mutate_metadata(self, record_id, transform):
        if self.read_only:
            raise PermissionError('SQLite record store is read-only')
        require_text(record_id, 'record ID', 256)
        self._connection.execute('BEGIN IMMEDIATE')
        try:
            record = self.get_record(record_id)
            updated = None
            if record is not None:
                payload = record.to_dict()
                payload['metadata'] = transform(payload['metadata'])
                updated = SQLiteRecord.from_dict(payload, self.identity)
                self._connection.execute('UPDATE records SET metadata = ? WHERE id = ?',
                                         (updated.metadata_json, record_id))
            self._connection.commit()
            return updated
        except BaseException:
            self._connection.rollback()
            raise

    def patch_metadata(self, record_id, changes, *, remove_keys=()):
        if self.read_only:
            raise PermissionError('SQLite record store is read-only')
        if not isinstance(changes, dict):
            raise ValueError('Metadata changes must be an object')
        changes = json.loads(canonical_json(changes))
        if (not isinstance(remove_keys, (list, tuple)) or len(remove_keys) > 1000
                or any(not isinstance(key, str) for key in remove_keys)):
            raise ValueError('Invalid metadata removal keys')
        if set(remove_keys).intersection(changes):
            raise ValueError('Metadata keys cannot be changed and removed together')

        def transform(metadata):
            for key in remove_keys:
                metadata.pop(key, None)
            metadata.update(changes)
            return metadata

        return self._mutate_metadata(record_id, transform)

    def recycle(self, record_id, by='dashboard', reason='', *, now=None):
        require_text(by, 'recycle actor', 256)
        if not isinstance(reason, str) or '\x00' in reason or len(reason.encode('utf-8')) > 4096:
            raise ValueError('Invalid recycle reason')
        if now is None:
            now = datetime.now()
        if not isinstance(now, datetime):
            raise ValueError('Recycle time must be a datetime')
        changes = {'recycled_at': now.isoformat(timespec='seconds'), 'recycled_by': by}
        if reason:
            changes['recycled_reason'] = reason
        return self.patch_metadata(record_id, changes)

    def restore(self, record_id):
        return self.patch_metadata(record_id, {}, remove_keys=RECYCLE_KEYS)

    def is_recycled(self, record_id):
        require_text(record_id, 'record ID', 256)
        record = self.get_record(record_id)
        return record is not None and bool(record.to_dict()['metadata'].get('recycled_at'))
