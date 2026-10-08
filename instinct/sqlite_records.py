# SPDX-License-Identifier: MPL-2.0
"""Bounded neutral records for the experimental, records-only SQLite scaffold."""

from dataclasses import dataclass
import json
import math


MAX_RECORDS = 10_000
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024
MAX_DIMENSION = 4096
MAX_FLOAT_JSON_BYTES = 64


def _validate_json(value, depth=0):
    if depth > 32:
        raise ValueError('JSON nesting exceeds limit')
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('JSON keys must be strings')
            _validate_json(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item, depth + 1)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('Unsupported JSON value')
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('JSON numbers must be finite')


def canonical_json(value):
    _validate_json(value)
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True,
                          separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('Expected bounded, finite JSON data') from error


def require_text(value, field, maximum):
    if not isinstance(value, str) or not value or '\x00' in value or len(value.encode('utf-8')) > maximum:
        raise ValueError(f'Invalid {field}')
    return value


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model: str
    dimension: int
    metric: str

    def __post_init__(self):
        require_text(self.provider, 'embedding provider', 128)
        require_text(self.model, 'embedding model', 512)
        if type(self.dimension) is not int or not 1 <= self.dimension <= MAX_DIMENSION:
            raise ValueError('Invalid embedding dimension')
        if self.metric not in ('cosine', 'l2', 'ip'):
            raise ValueError('Unsupported distance metric')

    def to_dict(self):
        return {'provider': self.provider, 'model': self.model,
                'dimension': self.dimension, 'metric': self.metric}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'provider', 'model', 'dimension', 'metric'}:
            raise ValueError('Invalid embedding identity')
        return cls(**value)


@dataclass(frozen=True)
class SQLiteRecord:
    record_id: str
    content: str
    metadata_json: str
    embedding: tuple[float, ...]

    @classmethod
    def from_dict(cls, value, identity):
        """Validate all fields; use a conservative finite-float size bound when safe."""
        if not isinstance(value, dict) or set(value) != {'id', 'content', 'metadata', 'embedding'}:
            raise ValueError('Invalid record fields')
        record_id = require_text(value['id'], 'record ID', 256)
        content = require_text(value['content'], 'record content', MAX_RECORD_BYTES)
        metadata = value['metadata']
        if not isinstance(metadata, dict):
            raise ValueError('Record metadata must be an object')
        if any(not isinstance(key, str) for key in metadata):
            raise ValueError('Metadata keys must be strings')
        if 'id' in metadata and metadata['id'] != record_id:
            raise ValueError('Metadata ID does not match record ID')
        if 'content' in metadata and metadata['content'] != content:
            raise ValueError('Metadata content does not match record content')
        for field in ('importance_score', 'initial_score'):
            if field in metadata and (type(metadata[field]) is not int or not 0 <= metadata[field] <= 100):
                raise ValueError(f'Invalid {field}')
        embedding = value['embedding']
        if not isinstance(embedding, (list, tuple)) or len(embedding) != identity.dimension:
            raise ValueError('Embedding dimension mismatch')
        try:
            if any(type(number) not in (int, float) or not math.isfinite(number) for number in embedding):
                raise ValueError('Embedding must contain finite numbers')
        except OverflowError as error:
            raise ValueError('Embedding must contain finite numbers') from error
        metadata_json = canonical_json(metadata)
        record = cls(record_id, content, metadata_json, tuple(float(number) for number in embedding))
        bounded_fields = {'id': record_id, 'content': content, 'metadata': metadata, 'embedding': []}
        upper_bytes = len(canonical_json(bounded_fields)) + len(record.embedding) * MAX_FLOAT_JSON_BYTES
        if upper_bytes > MAX_RECORD_BYTES and len(canonical_json(record.to_dict())) > MAX_RECORD_BYTES:
            raise ValueError('Record exceeds size limit')
        return record

    def to_dict(self):
        return {'id': self.record_id, 'content': self.content,
                'metadata': json.loads(self.metadata_json), 'embedding': list(self.embedding)}


def validate_records(records, identity):
    validated = []
    identifiers = set()
    total_bytes = 0
    for record in records:
        if len(validated) >= MAX_RECORDS:
            raise ValueError('Too many records')
        candidate = SQLiteRecord.from_dict(record.to_dict() if isinstance(record, SQLiteRecord) else record, identity)
        if candidate.record_id in identifiers:
            raise ValueError('Duplicate record ID')
        identifiers.add(candidate.record_id)
        total_bytes += len(canonical_json(candidate.to_dict()).encode('ascii'))
        if total_bytes > MAX_MANIFEST_BYTES:
            raise ValueError('Records exceed manifest size limit')
        validated.append(candidate)
    return validated
