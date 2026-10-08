# SPDX-License-Identifier: MPL-2.0
"""Explicit query embedding and identifier scans for the experimental adapter."""

from dataclasses import dataclass
import heapq
import ipaddress
import math
from urllib.parse import urlsplit

from .models import SegmentType, normalize_identifier
from .provider_privacy import validated_provider_url
from .sqlite_records import MAX_RECORDS, EmbeddingIdentity, require_text
from .sqlite_search import validate_vector


@dataclass(frozen=True)
class BoundQueryEmbedder:
    identity: EmbeddingIdentity
    provider: object

    def __post_init__(self):
        EmbeddingIdentity.from_dict(self.identity.to_dict())
        self._validate_provider()

    def _validate_provider(self):
        config = self.provider.get_config()
        if (not isinstance(config, dict) or self.provider.name() != self.identity.provider
                or config.get('provider') != self.identity.provider
                or config.get('model_name') != self.identity.model):
            raise ValueError('Query provider/model identity mismatch')
        if self.identity.metric not in self.provider.supported_spaces():
            raise ValueError('Query provider does not support the stored metric')
        if 'base_url' in config:
            url = validated_provider_url(config['base_url'])
            hostname = urlsplit(url).hostname
            try:
                local = hostname.lower() == 'localhost' or ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                local = False
            if not local:
                raise ValueError('SQLite query provider must use a loopback endpoint')

    def embed_query(self, query):
        require_text(query, 'query', 64 * 1024)
        self._validate_provider()
        try:
            batch = self.provider.embed_query([query])
        except Exception:
            raise RuntimeError('Query embedding failed; provider details withheld') from None
        if not isinstance(batch, (list, tuple)) or len(batch) != 1:
            raise ValueError('Expected one query embedding')
        return validate_vector(batch[0], self.identity)


class SQLiteSegmentQueries:
    def search(self, query, limit=10, min_similarity=0.0, min_importance=None,
               segment_types=None, include_recycled=False):
        require_text(query, 'query', 64 * 1024)
        if type(limit) is not int or not 1 <= limit <= MAX_RECORDS:
            raise ValueError('Invalid search limit')
        if (type(min_similarity) not in (int, float) or not math.isfinite(min_similarity)
                or not -1 <= min_similarity <= 1):
            raise ValueError('Invalid similarity threshold')
        if min_importance is not None and (type(min_importance) is not int or not 0 <= min_importance <= 100):
            raise ValueError('Invalid importance threshold')
        if type(include_recycled) is not bool:
            raise ValueError('Invalid recycled selection')
        if segment_types is not None and (not isinstance(segment_types, (list, tuple))
                or len(segment_types) > 100 or any(not isinstance(value, SegmentType) for value in segment_types)):
            raise ValueError('Invalid segment types')
        if self.repository.identity.metric != 'cosine':
            raise ValueError('Model text search currently requires cosine distance')
        if self.query_embedder is None:
            raise RuntimeError('An explicit query embedder is required; no fallback is enabled')
        if self.query_embedder.identity != self.repository.identity:
            raise ValueError('Query embedding identity mismatch')
        vector = self.query_embedder.embed_query(query)
        hits = self.repository.search_vectors(
            vector, limit=limit, min_similarity=min_similarity, min_importance=min_importance,
            segment_types=[value.value for value in segment_types] if segment_types else None,
            include_recycled=include_recycled)
        return [self._segment(hit.record) for hit in hits]

    def search_by_identifiers(self, identifiers, limit=100):
        if not isinstance(identifiers, (list, tuple)) or len(identifiers) > 1000:
            raise ValueError('Invalid query identifiers')
        if type(limit) is not int or not 0 <= limit <= MAX_RECORDS:
            raise ValueError('Invalid identifier limit')
        normalized = []
        for identifier in identifiers:
            if not isinstance(identifier, str) or '\x00' in identifier or len(identifier.encode('utf-8')) > 512:
                raise ValueError('Invalid query identifier')
            value = normalize_identifier(identifier)
            if value and value not in normalized:
                normalized.append(value)
        if not normalized or limit == 0:
            return []

        def candidates():
            for record in self.repository.list_records(limit=MAX_RECORDS):
                segment = self._segment(record)
                matched = [value for value in normalized if value in segment.memory_identifiers]
                if matched:
                    yield segment, len(matched), matched

        selected = heapq.nsmallest(limit, candidates(), key=lambda item: (-item[1], item[0].id))
        selected.sort(key=lambda item: (-item[1], -item[0].importance_score, item[0].concept_title))
        return selected
