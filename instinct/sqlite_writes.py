# SPDX-License-Identifier: MPL-2.0
"""Explicit document embedding and bounded experimental model writes."""

from copy import deepcopy
from datetime import datetime
import json
import re

from .active_time import capture_anchors, set_anchor
from .models import ScoredSegment
from .sqlite_queries import BoundQueryEmbedder
from .sqlite_records import MAX_RECORD_BYTES, MAX_RECORDS, SQLiteRecord, canonical_json, require_text, validate_records
from .sqlite_search import validate_vector


MAX_WRITE_BATCH = 128
MAX_DOCUMENT_BYTES = 64 * 1024
MAX_SCORE_EXTENSION_BYTES = 64 * 1024
_SCORE_EXTENSION_NAMESPACE = re.compile(r'[A-Za-z][A-Za-z0-9_.-]{0,127}\Z')


class BoundDocumentEmbedder(BoundQueryEmbedder):
    def embed_documents(self, documents):
        if not isinstance(documents, (list, tuple)) or not 1 <= len(documents) <= MAX_WRITE_BATCH:
            raise ValueError('Invalid document embedding batch')
        for document in documents:
            require_text(document, 'embedding document', MAX_DOCUMENT_BYTES)
        self._validate_provider()
        try:
            batch = self.provider.embed_documents(list(documents))
        except Exception:
            raise RuntimeError('Document embedding failed; provider details withheld') from None
        self._validate_provider()
        if not isinstance(batch, (list, tuple)) or len(batch) != len(documents):
            raise ValueError('Document embedding count mismatch')
        return [validate_vector(vector, self.identity) for vector in batch]


class SQLiteSegmentWrites:
    def _require_writer(self):
        if self._read_only:
            raise RuntimeError('SQLite segment adapter is read-only')

    def add(self, segment):
        return self.add_batch([segment])[0]

    def add_batch(self, segments):
        self._require_writer()
        if not isinstance(segments, (list, tuple)) or len(segments) > MAX_WRITE_BATCH:
            raise ValueError('Invalid model write batch')
        if not segments:
            return []
        prepared = []
        records = []
        changed = []
        new_count = 0
        for segment in segments:
            if not isinstance(segment, ScoredSegment):
                raise ValueError('Expected a scored segment')
            require_text(segment.id, 'record ID', 256)
            require_text(segment.segment.content, 'record content', MAX_RECORD_BYTES)
            candidate = deepcopy(segment)
            existing = self.repository.get_record(candidate.id)
            needs_embedding = existing is None or existing.content != candidate.segment.content
            if needs_embedding:
                require_text(candidate.segment.content, 'embedding document', MAX_DOCUMENT_BYTES)
                changed.append(len(prepared))
            new_count += int(existing is None)
            capture_anchors(candidate, self.persist_dir, new=existing is None)
            metadata = existing.to_dict()['metadata'] if existing is not None else {}
            metadata = metadata | self._changes(candidate)
            if 'content' in metadata:
                metadata['content'] = candidate.segment.content
            vector = existing.embedding if existing is not None else [0.0] * self.repository.identity.dimension
            records.append(SQLiteRecord.from_dict({
                'id': candidate.id, 'content': candidate.segment.content,
                'metadata': metadata, 'embedding': vector}, self.repository.identity))
            prepared.append(candidate)
        records = validate_records(records, self.repository.identity)
        if self.repository.count() + new_count > MAX_RECORDS:
            raise ValueError('Record count exceeds scaffold limit')
        if changed:
            if self.document_embedder is None:
                raise RuntimeError('SQLite adapter supports existing-record metadata writes only without an explicit document embedder')
            if self.document_embedder.identity != self.repository.identity:
                raise ValueError('Document embedding identity mismatch')
            vectors = self.document_embedder.embed_documents([
                prepared[index].segment.content for index in changed])
            for index, vector in zip(changed, vectors):
                payload = records[index].to_dict()
                payload['embedding'] = vector
                records[index] = SQLiteRecord.from_dict(payload, self.repository.identity)
        records = validate_records(records, self.repository.identity)
        self.repository.upsert_records(records)
        for original, candidate, record in zip(segments, prepared, records):
            original.active_time_anchors = deepcopy(candidate.active_time_anchors)
            original.embedding = list(record.embedding)
        return [record.record_id for record in records]

    def update_score(self, segment_id, new_score, reasoning=''):
        self._require_writer()
        require_text(segment_id, 'record ID', 256)
        if type(new_score) is not int or not 0 <= new_score <= 100:
            raise ValueError('Importance must be an integer from 0 to 100')
        if not isinstance(reasoning, str) or '\x00' in reasoning or len(reasoning.encode('utf-8')) > 4096:
            raise ValueError('Invalid score reasoning')

        def transform(metadata):
            segment = self._segment(self.repository.get_record(segment_id))
            segment.importance_score = new_score
            segment.pop_anchor_at = datetime.now()
            set_anchor(segment, 'pop_anchor_at', self.persist_dir)
            if reasoning:
                segment.reasoning = reasoning
            capture_anchors(segment, self.persist_dir)
            return metadata | self._changes(segment)

        return self.repository._mutate_metadata(segment_id, transform) is not None

    def patch_score_extension(self, segment_id, namespace, changes):
        self._require_writer()
        require_text(segment_id, 'record ID', 256)
        if not isinstance(namespace, str) or not _SCORE_EXTENSION_NAMESPACE.fullmatch(namespace):
            raise ValueError('Invalid score extension namespace')
        if not isinstance(changes, dict):
            raise ValueError('Score extension changes must be an object')
        encoded = canonical_json(changes)
        if len(encoded.encode('ascii')) > MAX_SCORE_EXTENSION_BYTES:
            raise ValueError('Score extension changes exceed the size limit')
        normalized = json.loads(encoded)

        def transform(metadata):
            extensions = metadata.get('score_extensions', {})
            if not isinstance(extensions, dict):
                raise ValueError('Existing score extensions require owner review')
            current = extensions.get(namespace, {})
            if not isinstance(current, dict):
                raise ValueError('Existing module score extension requires owner review')
            return metadata | {
                'score_extensions': extensions | {namespace: current | normalized},
            }

        updated = self.repository._mutate_metadata(segment_id, transform)
        return self._segment(updated)

    def recycle(self, segment_id, by='dashboard', reason=''):
        self._require_writer()
        return self._segment(self.repository.recycle(segment_id, by, reason))

    def restore(self, segment_id):
        self._require_writer()
        return self._segment(self.repository.restore(segment_id))

    def is_recycled(self, segment_id):
        return self.repository.is_recycled(segment_id)

    def list_recycled(self, limit=200):
        return [self._segment(record) for record in
                self.repository.list_records(limit=limit, recycled='recycled')]
