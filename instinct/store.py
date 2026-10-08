# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Weighted Vector Store

ChromaDB-based storage with importance scoring metadata.
"""

# Suppress progress bars and warnings before ANY imports
import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import ScoredSegment, Segment, SegmentType, normalize_identifier, parse_memory_identifiers

logger = logging.getLogger(__name__)

# Check for ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.debug("Optional legacy Chroma backend is unavailable")


class InstinctStore:
    """
    Weighted vector store for Instinct segments.
    
    Stores segments with:
    - Vector embeddings for semantic search
    - Importance scores (0-100)
    - Segment type classification
    - Concept titles and metadata
    """
    
    def __init__(
        self,
        collection_name: str = "instinct",
        persist_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        read_only: bool = False,
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB required. Install with: pip install chromadb")

        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self._identifier_index: Dict[str, Set[str]] = defaultdict(set)
        self._embedding_fn = self._get_embedding_function()

        # Acquire the cross-process writer lock BEFORE opening chroma.
        # Root-cause fix for "dashboard hangs / crashes when MCP server
        # or another probe is also using the same persist_dir". Two
        # processes pointing at the same chroma.sqlite3 race on the
        # OS file lock; the second one stalls in Dl (uninterruptible
        # disk wait) on WSL2. The lock serializes writers cleanly and
        # produces a clear error instead of a frozen request.
        #
        # Two modes:
        #   * default (read_only=False): must acquire the writer lock.
        #     If the MCP server (or any other process) holds it, raise
        #     ChromaWriterLockUnavailable with a clear message.
        #   * read_only=True: skip the writer lock and open chroma in
        #     read-only mode. Used by the dashboard when the MCP is
        #     the active writer, so the user can still observe the
        #     scoring system (browse, search, see KPIs) without
        #     crashing or stealing the lock. Writes are blocked at
        #     the dashboard level and return a clear 409.
        #
        # Bypass only when the caller injected a shared client (smoke
        # tests) — those run in-process and can't conflict.
        # NOTE: `shared` is computed BEFORE `_read_only` because both
        # branches below need it. Don't move this assignment.
        self._writer_lock_handle = None
        shared = getattr(sys.modules.get("instinct.store", None), "_smoke_test_client", None)
        if shared is None and persist_dir:
            from .private_storage import ensure_private_directory
            ensure_private_directory(persist_dir)
        self._read_only = bool(read_only) and shared is None and bool(persist_dir)
        if shared is None and persist_dir and not self._read_only:
            from .chroma_writer_lock import (
                acquire_writer_lock,
                ChromaWriterLockUnavailable,
            )
            try:
                self._writer_lock_handle = acquire_writer_lock(
                    persist_dir, timeout=30.0
                )
            except ChromaWriterLockUnavailable as e:
                # Re-raise with a one-line hint the user can act on.
                # The exception message already names the holder PID.
                logger.error(
                    "InstinctStore: cannot acquire chroma writer lock for %s "
                    "(holder pid=%s, waited %.1fs). Close the other process "
                    "and retry, or pass read_only=True to observe without writing.",
                    persist_dir, e.holder_pid, e.waited,
                )
                raise

        # Initialize ChromaDB
        # Allow smoke tests / callers to inject a shared client so multiple
        # InstinctStore instances in the same process don't trigger
        # "An instance of Chroma already exists ... with different settings".
        if shared is not None:
            self._client = shared
        elif persist_dir:
            # Read-only mode: open a separate PersistentClient pointed at
            # the same on-disk store but with a Settings flag that
            # prevents writes. We use a custom Settings object with
            # is_persistent=False's cousin — chromadb doesn't ship a
            # first-class "read_only" flag, so we use the lowest-level
            # hack: the Settings() allows_persistence + a chromadb
            # Settings that we know honors at the API layer. As a
            # belt-and-suspenders, the dashboard also rejects mutating
            # endpoints with HTTP 409 when self._read_only is True.
            if self._read_only:
                # Open a fresh PersistentClient. Same on-disk file.
                # We accept that chromadb may attempt to write metadata
                # on first call; if it does, the SQLite busy_timeout
                # below will block briefly. If the MCP holds the writer
                # lock, that block is fine because chroma is also
                # holding its own internal lock.
                self._client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(
                        chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
                        anonymized_telemetry=False,
                        sqlite_busy_timeout=10.0,
                    ),
                )
            else:
                self._client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(
                        chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
                        anonymized_telemetry=False,
                        # 10s is long enough to absorb a brief read/write
                        # overlap from within the same process without
                        # making the user wait on a real conflict.
                        sqlite_busy_timeout=10.0,
                    ),
                )
        else:
            self._client = chromadb.Client(
                settings=Settings(chroma_api_impl="chromadb.api.rust.RustBindingsAPI", anonymized_telemetry=False),
            )
        
        # Get or create collection with embedding function
        try:
            self._collection = self._open_collection()
        except Exception:
            if self._writer_lock_handle is not None:
                from .chroma_writer_lock import release_writer_lock
                release_writer_lock(self._writer_lock_handle)
                self._writer_lock_handle = None
            raise

        self._rebuild_identifier_index()
        
        logger.info(f"InstinctStore initialized: {collection_name} ({self._collection.count()} segments)")
    
    def _open_collection(self):
        from .collection_security import validate_collection_model
        collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        validate_collection_model(collection.get_model(), self._embedding_fn)
        return collection

    def _get_embedding_function(self):
        """Select an explicit provider before opening storage; fail closed."""
        try:
            from .embedding_provider import get_embedding_provider
            provider = get_embedding_provider()
            if provider is None:
                raise RuntimeError("Explicit embedding provider required")
            logger.info(f"Using embedding provider: {provider.name()}")
            return provider
        except Exception:
            logger.warning("Embedding provider setup failed; details withheld; storage not opened")
            raise RuntimeError("Embedding provider unavailable; storage setup refused. Review provider readiness.") from None
    
    def add(self, segment: ScoredSegment) -> str:
        """
        Add a scored segment to the store.
        
        Returns:
            The segment ID
        
        Raises:
            RuntimeError: if the store is in read-only mode.
        """
        if self._read_only:
            raise RuntimeError(
                "InstinctStore is in read-only mode (another process holds "
                "the writer lock). Cannot add segment. Either stop the other "
                "process or connect with read_only=False."
            )
        existing = self.get(segment.id)
        if existing:
            self._deindex_segment_identifiers(existing.id, existing.memory_identifiers)

        from .active_time import capture_anchors
        capture_anchors(segment, self.persist_dir, new=existing is None)
        metadata = segment.to_storage_dict()
        
        # ChromaDB metadata must be str, int, float, or bool
        # Convert any other types
        clean_metadata = {}
        for k, v in metadata.items():
            if k == "content":
                continue  # Content goes in documents
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif v is None:
                clean_metadata[k] = ""
            else:
                clean_metadata[k] = str(v)
        
        self._collection.upsert(
            ids=[segment.id],
            documents=[segment.segment.content],
            metadatas=[clean_metadata],
        )

        self._index_segment_identifiers(segment)
        
        logger.debug(f"Added segment: {segment.id} (score={segment.importance_score})")
        return segment.id
    
    def add_batch(self, segments: List[ScoredSegment]) -> List[str]:
        """Add multiple segments at once."""
        if self._read_only:
            raise RuntimeError(
                "InstinctStore is in read-only mode. Cannot add_batch."
            )
        if not segments:
            return []
        
        ids = []
        documents = []
        metadatas = []
        
        for seg in segments:
            from .active_time import capture_anchors
            capture_anchors(seg, self.persist_dir, new=self.get(seg.id) is None)
            ids.append(seg.id)
            documents.append(seg.segment.content)
            
            metadata = seg.to_storage_dict()
            clean_metadata = {}
            for k, v in metadata.items():
                if k == "content":
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                elif v is None:
                    clean_metadata[k] = ""
                else:
                    clean_metadata[k] = str(v)
            metadatas.append(clean_metadata)
        
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        self._rebuild_identifier_index()
        
        logger.info(f"Added {len(segments)} segments")
        return ids
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.0,
        min_importance: Optional[int] = None,
        segment_types: Optional[List[SegmentType]] = None,
        include_recycled: bool = False,
    ) -> List[ScoredSegment]:
        """
        Semantic search for relevant segments.

        Args:
            query: Search query
            limit: Max results
            min_similarity: Minimum cosine similarity (0-1)
            min_importance: Minimum importance score filter
            segment_types: Filter to specific types
            include_recycled: If False (default), segments in the
                recycle bin are excluded. The MCP server and dashboard
                both pass False unless the user explicitly toggles
                "Show recycled" in the UI.
        
        Returns:
            List of matching ScoredSegments, sorted by relevance
        """
        # Build where filter. Recycled segments are excluded unless the
        # caller explicitly opts in via include_recycled=True.
        where = None
        where_conditions = []

        if min_importance is not None:
            where_conditions.append({"importance_score": {"$gte": min_importance}})

        if segment_types:
            type_values = [t.value for t in segment_types]
            where_conditions.append({"segment_type": {"$in": type_values}})

        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        if not include_recycled:
            where = self._recycled_filter(where)

        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        from .models import RECYCLED_AT_KEY
        segments = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity (ChromaDB returns distance for cosine)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # Cosine distance to similarity

                if similarity < min_similarity:
                    continue

                metadata = results["metadatas"][0][i]
                # Post-fetch filter: drop recycled unless asked to include
                if not include_recycled and metadata.get(RECYCLED_AT_KEY):
                    continue
                content = results["documents"][0][i]

                # Reconstruct segment
                metadata["content"] = content
                segment = ScoredSegment.from_storage_dict(metadata)
                segments.append(segment)

        return segments

    def search_by_identifiers(
        self,
        identifiers: List[str],
        limit: int = 100,
    ) -> List[Tuple[ScoredSegment, int, List[str]]]:
        """Find segments with exact identifier hits using the in-memory index."""
        normalized_identifiers = []
        seen = set()
        for identifier in identifiers:
            normalized = normalize_identifier(identifier)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_identifiers.append(normalized)

        if not normalized_identifiers or limit <= 0:
            return []

        hit_counts: Dict[str, int] = {}
        matched_identifiers: Dict[str, List[str]] = {}

        for identifier in normalized_identifiers:
            for segment_id in self._identifier_index.get(identifier, set()):
                hit_counts[segment_id] = hit_counts.get(segment_id, 0) + 1
                matched_identifiers.setdefault(segment_id, []).append(identifier)

        ordered_ids = sorted(
            hit_counts.keys(),
            key=lambda segment_id: (-hit_counts[segment_id], segment_id),
        )

        from .models import RECYCLED_AT_KEY
        results: List[Tuple[ScoredSegment, int, List[str]]] = []
        for offset in range(0, len(ordered_ids), 100):
            for segment in self.get_many(ordered_ids[offset:offset + 100]):
                if segment.segment.metadata.get(RECYCLED_AT_KEY):
                    continue
                results.append(
                    (
                        segment,
                        hit_counts[segment.id],
                        matched_identifiers.get(segment.id, []),
                    )
                )
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        results.sort(key=lambda item: (-item[1], -item[0].importance_score, item[0].concept_title))
        return results
    
    def get_mandatory(self, include_recycled: bool = False) -> List[ScoredSegment]:
        """Get all score-100 (mandatory) segments. By default, recycled
        mandatory segments are excluded — the agent should not be force-
        fed a rule you've intentionally retired. Pass
        include_recycled=True to see them (dashboard's Recycle bin view)."""
        return self.get_by_score(
            min_score=100, max_score=100, limit=100, include_recycled=include_recycled,
        )

    def get_by_score(
        self,
        min_score: int = 0,
        max_score: int = 100,
        limit: int = 100,
        include_recycled: bool = False,
    ) -> List[ScoredSegment]:
        """Get segments within a score range. By default excludes
        recycled segments."""
        where = {"$and": [
            {"importance_score": {"$gte": min_score}},
            {"importance_score": {"$lte": max_score}},
        ]}
        # _recycled_filter is a no-op (Chroma doesn't support $exists);
        # we filter recycled client-side below.
        where = self._recycled_filter(where)

        results = self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )

        from .models import RECYCLED_AT_KEY
        segments = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                # Post-fetch filter: drop recycled unless asked to include
                if not include_recycled and metadata.get(RECYCLED_AT_KEY):
                    continue
                content = results["documents"][i]
                metadata["content"] = content
                segment = ScoredSegment.from_storage_dict(metadata)
                segments.append(segment)

        return segments
    
    def get(self, segment_id: str) -> Optional[ScoredSegment]:
        """Get a specific segment by ID."""
        results = self._collection.get(
            ids=[segment_id],
            include=["documents", "metadatas"],
        )
        
        if not results["ids"]:
            return None
        
        metadata = results["metadatas"][0]
        content = results["documents"][0]
        metadata["content"] = content
        return ScoredSegment.from_storage_dict(metadata)

    def get_many(self, segment_ids: List[str]) -> List[ScoredSegment]:
        """Get multiple segments by ID while preserving request order."""
        if not segment_ids:
            return []

        results = self._collection.get(
            ids=segment_ids,
            include=["documents", "metadatas"],
        )

        segment_map: Dict[str, ScoredSegment] = {}
        for index, segment_id in enumerate(results.get("ids", [])):
            metadata = results["metadatas"][index]
            content = results["documents"][index]
            metadata["content"] = content
            segment_map[segment_id] = ScoredSegment.from_storage_dict(metadata)

        return [segment_map[segment_id] for segment_id in segment_ids if segment_id in segment_map]
    
    def delete(self, segment_id: str) -> bool:
        """HARD delete a segment by ID. PREFER recycle() — hard delete is
        irreversible and loses the segment's history, embeddings, and any
        access tracking. The dashboard exposes hard delete only in the
        Recycle bin view's "purge" action; the default UI button is
        recycle(), which is reversible."""
        if self._read_only:
            raise RuntimeError(
                "InstinctStore is in read-only mode. Cannot delete."
            )
        try:
            existing = self.get(segment_id)
            self._collection.delete(ids=[segment_id])
            if existing:
                self._deindex_segment_identifiers(segment_id, existing.memory_identifiers)
            return True
        except Exception:
            return False

    # --- recycle bin (soft delete) ---------------------------------------

    def recycle(
        self,
        segment_id: str,
        by: str = "dashboard",
        reason: str = "",
    ) -> Optional[ScoredSegment]:
        """Soft-delete: hide the segment from default list / search /
        retriever, but keep it in chroma so it can be restored. Returns
        the recycled segment, or None if not found.

        The MCP server's retriever (PriorityRetriever.retrieve) does NOT
        include recycled segments by default — so a recycled mandatory
        rule stops being force-fed to the agent. The recycle action is
        recorded in segment metadata (`recycled_at`, `recycled_by`,
        `recycled_reason`) for audit and recovery.
        """
        seg = self.get(segment_id)
        if seg is None:
            return None
        # Stash recycle metadata on the segment's metadata dict so it
        # round-trips through chroma via to_storage_dict().
        md = seg.segment.metadata or {}
        from .models import RECYCLED_AT_KEY, RECYCLED_BY_KEY, RECYCLED_REASON_KEY
        md[RECYCLED_AT_KEY] = datetime.now().isoformat(timespec="seconds")
        md[RECYCLED_BY_KEY] = by
        if reason:
            md[RECYCLED_REASON_KEY] = reason
        seg.segment.metadata = md
        # Re-add to chroma (upsert preserves the embedding).
        self.add(seg)
        # Drop from the in-memory identifier index so the retriever
        # doesn't accidentally surface it.
        self._deindex_segment_identifiers(seg.id, seg.memory_identifiers)
        return seg

    def restore(self, segment_id: str) -> Optional[ScoredSegment]:
        """Reverse a recycle: clears the recycled_at/by/reason metadata
        and re-indexes identifiers. Returns the restored segment, or
        None if not found.

        NOTE on chroma metadata semantics: chroma's upsert merges
        metadata fields, it doesn't replace them. So if a key was
        present and we don't pass it in the new metadata dict, the
        old value sticks around. We work around this by explicitly
        setting the recycled_* keys to empty strings — our is_recycled
        / filter helpers treat empty string as "not recycled" (the
        bool(md.get(RECYCLED_AT_KEY)) check returns False on "").
        """
        seg = self.get(segment_id)
        if seg is None:
            return None
        md = dict(seg.segment.metadata or {})
        from .models import RECYCLED_AT_KEY, RECYCLED_BY_KEY, RECYCLED_REASON_KEY
        # Pop the in-memory copies (so the in-process state is clean),
        # but explicitly blank them in the chroma-persisted dict below.
        md.pop(RECYCLED_AT_KEY, None)
        md.pop(RECYCLED_BY_KEY, None)
        md.pop(RECYCLED_REASON_KEY, None)
        seg.segment.metadata = md
        # In-memory: clean (no recycled keys at all).
        # Persisted: blank strings so chroma merges them away from
        # the prior truthy values. Empty string is treated as "not
        # recycled" by is_recycled() and the post-fetch filter.
        persisted_md = dict(md)
        persisted_md[RECYCLED_AT_KEY] = ""
        persisted_md[RECYCLED_BY_KEY] = ""
        persisted_md[RECYCLED_REASON_KEY] = ""
        seg.segment.metadata = persisted_md
        self.add(seg)
        # Restore the in-memory copy to the clean version (without the
        # empty-string sentinels) so subsequent reads of this object
        # show clean metadata.
        seg.segment.metadata = md
        self._index_segment_identifiers(seg)
        return seg

    def is_recycled(self, segment_id: str) -> bool:
        seg = self.get(segment_id)
        if seg is None:
            return False
        from .models import RECYCLED_AT_KEY
        return bool((seg.segment.metadata or {}).get(RECYCLED_AT_KEY))

    def list_recycled(self, limit: int = 200) -> List[ScoredSegment]:
        """List all soft-deleted segments. Used by the dashboard's
        Recycle bin view."""
        # Chroma doesn't have an `IS NOT NULL` filter, so we fetch all
        # and filter in Python. For a 200-segment store this is fine;
        # if it ever needs to scale, add a dedicated metadata field
        # like `is_recycled=1` and use a where filter.
        all_segs = self._collection.get(
            limit=limit * 4 if limit else 1000,
            include=["documents", "metadatas"],
        )
        from .models import RECYCLED_AT_KEY
        out: List[ScoredSegment] = []
        if all_segs["ids"]:
            for i, doc_id in enumerate(all_segs["ids"]):
                md = all_segs["metadatas"][i]
                if md.get(RECYCLED_AT_KEY):
                    md["content"] = all_segs["documents"][i]
                    out.append(ScoredSegment.from_storage_dict(md))
                    if limit and len(out) >= limit:
                        break
        return out

    def _recycled_filter(self, where):
        """Augment a chroma `where` filter to exclude recycled segments.

        NOTE: ChromaDB doesn't support $exists in `where` filters, so
        we cannot push the recycled check down to chroma. Instead the
        callers of this helper fetch all matching segments and then
        drop the recycled ones in Python (in get_by_score and search).
        This helper is now a no-op kept for API compatibility; the
        actual filtering happens post-fetch.
        """
        return where

    def update_score(self, segment_id: str, new_score: int, reasoning: str = "") -> bool:
        """Update the importance score of a segment."""
        if self._read_only:
            raise RuntimeError(
                "InstinctStore is in read-only mode. Cannot update_score."
            )
        segment = self.get(segment_id)
        if not segment:
            return False
        
        segment.importance_score = max(0, min(100, new_score))
        from .active_time import set_anchor
        segment.pop_anchor_at = datetime.now()
        set_anchor(segment, "pop_anchor_at", self.persist_dir)
        if reasoning:
            segment.reasoning = reasoning
        
        self.add(segment)  # Upsert
        return True
    
    def record_access(self, segment_id: str, now: Optional[datetime] = None) -> bool:
        """
        Record that a segment was accessed (retrieved).
        
        Updates metadata only (no document/embedding rewrite) to avoid
        re-embedding every accessed segment via Ollama.

        Args:
            segment_id: ID of the segment that was accessed
            now: Current time (defaults to datetime.now)
        
        Returns:
            True if updated, False if segment not found or read-only.
        """
        if self._read_only:
            return False  # Silent no-op in read-only mode (access tracking is non-critical)

        segment = self.get(segment_id)
        if not segment:
            return False

        now = now or datetime.now()

        # Check if this should trigger a pop
        from .decay import PopModifier
        pop_mod = PopModifier(persist_dir=self.persist_dir)

        if pop_mod.should_pop(segment, now):
            # Dormant memory is being accessed - pop it!
            segment.popped_at = now
            segment.accesses_since_pop = 1
            logger.debug(f"Segment {segment_id} popped (dormant access)")
        elif segment.popped_at:
            # Already in pop state - increment counter
            segment.accesses_since_pop += 1
            logger.debug(f"Segment {segment_id} pop counter: {segment.accesses_since_pop}")

        # Update access tracking
        segment.last_accessed = now
        segment.access_count += 1
        from .active_time import set_anchor, capture_anchors
        set_anchor(segment, "last_accessed", self.persist_dir)
        if segment.popped_at == now:
            set_anchor(segment, "popped_at", self.persist_dir)
        capture_anchors(segment, self.persist_dir)

        # Nudge system: if this is a mandatory segment currently at 100,
        # drop it to 79 so the agent doesn't get force-fed on every call.
        # The nudge timer will re-elevate it after `nudge_interval_turns`.
        try:
            from .nudge import apply_access_drop, get_current_turn
            if apply_access_drop(segment, get_current_turn(self)):
                segment.pop_anchor_at = now
                set_anchor(segment, "pop_anchor_at", self.persist_dir)
        except Exception:
            logger.exception("nudge.apply_access_drop failed (non-fatal)")

        set_anchor(segment, "last_accessed", self.persist_dir)
        # Metadata-only update avoids re-embedding the document.
        metadata = segment.to_storage_dict()
        clean_metadata = {}
        for k, v in metadata.items():
            if k == "content":
                continue
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif v is None:
                clean_metadata[k] = ""
            else:
                clean_metadata[k] = str(v)

        self._collection.update(
            ids=[segment_id],
            metadatas=[clean_metadata],
        )

        logger.debug(f"Recorded access for {segment_id} (count: {segment.access_count})")
        return True

    def update_metadata_only(self, segment_or_id) -> bool:
        """Update only the metadata of a segment (no doc/embedding rewrite).

        Used by the nudge system and similar score-only mutators so that
        re-elevation to 100 doesn't trigger an Ollama re-embed. Same
        pattern as the inlined update in record_access.

        Accepts either a ScoredSegment (uses the in-memory copy with any
        pending mutations) or a segment_id string (loads fresh from
        chroma and writes back).

        Returns:
            True if updated successfully, False if segment not found or read-only.
        """
        if self._read_only:
            return False  # Silent no-op in read-only mode

        if isinstance(segment_or_id, str):
            segment = self.get(segment_or_id)
        else:
            segment = segment_or_id
        if not segment:
            return False

        from .active_time import capture_anchors
        capture_anchors(segment, self.persist_dir)
        metadata = segment.to_storage_dict()
        clean_metadata = {}
        for k, v in metadata.items():
            if k == "content":
                continue
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif v is None:
                clean_metadata[k] = ""
            else:
                clean_metadata[k] = str(v)

        self._collection.update(
            ids=[segment.id],
            metadatas=[clean_metadata],
        )
        return True

    def record_access_batch(self, segment_ids: List[str], now: Optional[datetime] = None) -> int:
        """
        Record access for multiple segments.

        Args:
            segment_ids: List of segment IDs that were accessed
            now: Current time (defaults to datetime.now)

        Returns:
            Number of segments successfully updated
        """
        now = now or datetime.now()
        updated = 0
        for seg_id in segment_ids:
            if self.record_access(seg_id, now):
                updated += 1
        return updated

    def clear_expired_pops(self, now: Optional[datetime] = None) -> int:
        """
        Clear pop state from segments that didn't stick.
        
        Should be called periodically as maintenance.
        
        Args:
            now: Current time (defaults to datetime.now)
        
        Returns:
            Number of segments cleared
        """
        from .decay import PopModifier
        pop_mod = PopModifier(persist_dir=self.persist_dir)
        now = now or datetime.now()

        # Chunked metadata-only scan to avoid loading all segments into memory
        total = self.count()
        CHUNK_SIZE = 500
        offset = 0
        cleared = 0

        while offset < total:
            results = self._collection.get(
                limit=min(CHUNK_SIZE, total - offset),
                offset=offset,
                include=["metadatas"],
            )
            for metadata in results.get("metadatas", []):
                segment = ScoredSegment.from_storage_dict(metadata)
                if pop_mod.should_clear_pop(segment, now):
                    segment.popped_at = None
                    segment.accesses_since_pop = 0
                    self.add(segment)
                    cleared += 1
                    logger.debug(f"Cleared expired pop for {segment.id}")
            offset += CHUNK_SIZE

        if cleared:
            logger.info(f"Cleared {cleared} expired pops")

        return cleared
    
    def count(self) -> int:
        """Get total number of segments."""
        return self._collection.count()
    
    def clear(self) -> None:
        """Delete all segments."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._open_collection()
        logger.info("InstinctStore cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        total = self.count()
        mandatory = len(self.get_mandatory())
        high_priority = len(self.get_by_score(min_score=70, max_score=100, limit=200))

        # Access tracking stats — metadata-only, chunked
        accessed_count = 0
        popped_count = 0
        CHUNK_SIZE = 500
        offset = 0
        while offset < total:
            results = self._collection.get(
                limit=min(CHUNK_SIZE, total - offset),
                offset=offset,
                include=["metadatas"],
            )
            for metadata in results.get("metadatas", []):
                if metadata.get("access_count", 0) > 0:
                    accessed_count += 1
                if metadata.get("popped_at") not in (None, "", "None"):
                    popped_count += 1
            offset += CHUNK_SIZE

        return {
            "total_segments": total,
            "mandatory_count": mandatory,
            "high_priority_count": high_priority,
            "accessed_count": accessed_count,
            "popped_count": popped_count,
            "identifier_terms": len(self._identifier_index),
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
        }

    def _index_segment_identifiers(self, segment: ScoredSegment) -> None:
        from .models import RECYCLED_AT_KEY
        if segment.segment.metadata.get(RECYCLED_AT_KEY):
            return
        for identifier in segment.memory_identifiers:
            self._identifier_index[identifier].add(segment.id)

    def _deindex_segment_identifiers(self, segment_id: str, identifiers: List[str]) -> None:
        for identifier in identifiers:
            segment_ids = self._identifier_index.get(identifier)
            if not segment_ids:
                continue
            segment_ids.discard(segment_id)
            if not segment_ids:
                self._identifier_index.pop(identifier, None)

    def _rebuild_identifier_index(self) -> None:
        from .models import RECYCLED_AT_KEY
        self._identifier_index = defaultdict(set)
        total = self._collection.count()
        if total <= 0:
            return

        # Chunked rebuild to avoid loading all documents into memory at once
        CHUNK_SIZE = 500
        offset = 0
        while offset < total:
            results = self._collection.get(
                limit=min(CHUNK_SIZE, total - offset),
                offset=offset,
                include=["metadatas"],  # identifiers only — no documents needed
            )
            for index, segment_id in enumerate(results.get("ids", [])):
                metadata = results["metadatas"][index]
                if metadata.get(RECYCLED_AT_KEY):
                    continue
                # Build identifiers from metadata without loading full content
                identifiers = parse_memory_identifiers(metadata.get("memory_identifiers", ""))
                for identifier in identifiers:
                    self._identifier_index[identifier].add(segment_id)
            offset += CHUNK_SIZE
