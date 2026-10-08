# SPDX-License-Identifier: MPL-2.0
"""Experimental existing-record model/access adapter, not a runtime backend."""

from datetime import datetime
import logging
from pathlib import Path

from .active_time import capture_anchors, set_anchor
from .decay import PopModifier
from .models import ScoredSegment
from .sqlite_records import MAX_RECORDS
from .sqlite_queries import BoundQueryEmbedder, SQLiteSegmentQueries
from .sqlite_writes import BoundDocumentEmbedder, SQLiteSegmentWrites


logger = logging.getLogger(__name__)


class SQLiteSegmentAdapter(SQLiteSegmentWrites, SQLiteSegmentQueries):
    runtime_ready = False
    pending_replay_ready = False

    def __init__(self, repository, *, persist_dir, query_embedder=None, document_embedder=None):
        if persist_dir is None or not isinstance(persist_dir, (str, Path)):
            raise ValueError('An explicit isolated clock directory is required')
        clock_dir = Path(persist_dir)
        if not clock_dir.is_absolute() or not clock_dir.is_dir():
            raise ValueError('An explicit existing absolute clock directory is required')
        if query_embedder is not None:
            if not isinstance(query_embedder, BoundQueryEmbedder) or query_embedder.identity != repository.identity:
                raise ValueError('Query embedding identity mismatch')
        if document_embedder is not None:
            if not isinstance(document_embedder, BoundDocumentEmbedder) or document_embedder.identity != repository.identity:
                raise ValueError('Document embedding identity mismatch')
        self.repository = repository
        self.persist_dir = clock_dir
        self.query_embedder = query_embedder
        self.document_embedder = document_embedder

    @property
    def _read_only(self):
        return self.repository.read_only

    @staticmethod
    def _segment(record):
        if record is None:
            return None
        payload = record.to_dict()
        metadata = payload['metadata'] | {'id': record.record_id, 'content': record.content}
        return ScoredSegment.from_storage_dict(metadata, embedding=payload['embedding'])

    def get(self, segment_id):
        return self._segment(self.repository.get_record(segment_id))

    def get_many(self, segment_ids):
        return [self._segment(record) for record in self.repository.get_records(segment_ids)]

    def get_by_score(self, min_score=0, max_score=100, limit=100, include_recycled=False):
        if type(include_recycled) is not bool:
            raise ValueError('Invalid recycled selection')
        records = self.repository.list_records(
            min_score=min_score, max_score=max_score, limit=limit,
            recycled='all' if include_recycled else 'active')
        return [self._segment(record) for record in records]

    def get_mandatory(self, include_recycled=False):
        return self.get_by_score(100, 100, include_recycled=include_recycled)

    def count(self):
        return self.repository.count()

    def stats(self):
        segments = [self._segment(record) for record in
                    self.repository.list_records(limit=MAX_RECORDS, recycled='all')]
        from .models import RECYCLED_AT_KEY
        active = [segment for segment in segments if not segment.segment.metadata.get(RECYCLED_AT_KEY)]
        return {
            'total_segments': len(segments),
            'mandatory_count': sum(segment.importance_score == 100 for segment in active),
            'high_priority_count': sum(segment.importance_score >= 70 for segment in active),
            'accessed_count': sum(segment.access_count > 0 for segment in segments),
            'popped_count': sum(segment.popped_at is not None for segment in segments),
            'identifier_terms': len({identifier for segment in active for identifier in segment.memory_identifiers}),
            'collection_name': 'sqlite', 'persist_dir': str(self.persist_dir),
        }

    @staticmethod
    def _changes(segment):
        return {key: value for key, value in segment.to_storage_dict().items()
                if key != 'content'}

    def update_metadata_only(self, segment_or_id):
        if self._read_only:
            return False
        segment = self.get(segment_or_id) if isinstance(segment_or_id, str) else segment_or_id
        if segment is None:
            return False
        if not isinstance(segment, ScoredSegment):
            raise ValueError('Expected a scored segment or ID')
        record = self.repository.get_record(segment.id)
        if record is None:
            return False
        if segment.segment.content != record.content:
            raise ValueError('Metadata-only writes cannot change content')
        capture_anchors(segment, self.persist_dir)
        return self.repository.patch_metadata(segment.id, self._changes(segment)) is not None

    def record_access(self, segment_id, now=None):
        if self._read_only:
            return False
        now = now or datetime.now()

        def transform(metadata):
            segment = self._segment(self.repository.get_record(segment_id))
            modifier = PopModifier(persist_dir=self.persist_dir)
            if modifier.should_pop(segment, now):
                segment.popped_at = now
                segment.accesses_since_pop = 1
            elif segment.popped_at:
                segment.accesses_since_pop += 1
            segment.last_accessed = now
            segment.access_count += 1
            set_anchor(segment, 'last_accessed', self.persist_dir)
            if segment.popped_at == now:
                set_anchor(segment, 'popped_at', self.persist_dir)
            capture_anchors(segment, self.persist_dir)
            try:
                from .nudge import apply_access_drop, get_current_turn
                if apply_access_drop(segment, get_current_turn(self)):
                    segment.pop_anchor_at = now
                    set_anchor(segment, 'pop_anchor_at', self.persist_dir)
            except Exception:
                logger.exception('nudge.apply_access_drop failed (non-fatal)')
            set_anchor(segment, 'last_accessed', self.persist_dir)
            return metadata | self._changes(segment)

        return self.repository._mutate_metadata(segment_id, transform) is not None

    def record_access_batch(self, segment_ids, now=None):
        now = now or datetime.now()
        return sum(self.record_access(segment_id, now) for segment_id in segment_ids)

    def clear_expired_pops(self, now=None):
        if self._read_only:
            return 0
        now = now or datetime.now()
        modifier = PopModifier(persist_dir=self.persist_dir)
        cleared = 0
        records = self.repository.list_records(limit=MAX_RECORDS, recycled='all')
        for record in records:
            if not self._segment(record).popped_at:
                continue
            did_clear = False

            def transform(metadata):
                nonlocal did_clear
                segment = self._segment(self.repository.get_record(record.record_id))
                if modifier.should_clear_pop(segment, now):
                    segment.popped_at = None
                    segment.accesses_since_pop = 0
                    capture_anchors(segment, self.persist_dir)
                    did_clear = True
                    return metadata | self._changes(segment)
                return metadata

            updated = self.repository._mutate_metadata(record.record_id, transform)
            cleared += int(updated is not None and did_clear)
        return cleared
