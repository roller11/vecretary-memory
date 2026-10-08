# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Priority Retriever

Retrieves context from the weighted vector store with mandatory
score-100 items always included first.

Now includes modifier pipeline for time-based decay and pop mechanics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from .models import ScoredSegment, RetrievalResult, SegmentType, PopResultRecord
from .modifiers import ModifierPipeline, QueryContext, ScoringResult, create_default_pipeline
from .pop import apply_pop, PopResult
from .dormancy import DormancyConfig

if TYPE_CHECKING:
    from .store import InstinctStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for context retrieval."""
    
    # Always retrieve all score-100 items
    always_include_mandatory: bool = True
    
    # Max items to retrieve (excluding mandatory)
    max_relevant_items: int = 10

    # Extra identifier candidates to consider before scoring
    identifier_candidate_limit: int = 50
    
    # Minimum similarity score for relevant items
    min_similarity: float = 0.5
    
    # Minimum importance score for relevant items (before decay)
    min_importance: int = 30
    
    # Minimum effective score after decay (set lower than min_importance)
    min_effective_score: float = 20.0
    
    # Token budget for context
    max_tokens: int = 8000
    
    # Record access when retrieving (for decay/pop tracking)
    track_access: bool = True

    # Apply pop rules during retrieval (uses identifier_hits from modifier pipeline
    # to decide whether to halve decay, jump to plateau, frequency-reset, or
    # promotion). Pop mutations are persisted via store.add().
    apply_pops_on_retrieve: bool = True

    # Optional dormancy configuration for pop rules. If None, the function
    # apply_pop() will load its own defaults from dormancy.DormancyConfig().
    pop_dormancy_config: Optional[DormancyConfig] = None

    # Boost recent items (deprecated - use modifier pipeline)
    recency_boost: bool = False
    recency_days: int = 7
    recency_boost_amount: float = 0.1


class PriorityRetriever:
    """
    Retrieves context with priority ordering.
    
    1. Always retrieves all score-100 (mandatory) items
    2. Retrieves query-relevant items sorted by effective score
    3. Respects token budget, mandatory items first
    4. Applies modifier pipeline (decay, pop, etc.)
    5. Records access for future decay calculations
    """
    
    def __init__(
        self,
        store: "InstinctStore",
        config: Optional[RetrievalConfig] = None,
        pipeline: Optional[ModifierPipeline] = None,
    ):
        self.store = store
        self.config = config or RetrievalConfig()
        self.pipeline = pipeline or create_default_pipeline()
        
        # Cache for scoring results (cleared on each retrieve)
        self._scoring_cache: Dict[str, ScoringResult] = {}

    def _should_include_mandatory(self, segment: ScoredSegment, context: QueryContext) -> bool:
        """Decide whether a mandatory segment applies to the current query."""
        scope = segment.mandatory_scope
        if scope == "always":
            return True
        if scope == "conditional":
            tags = set(segment.mandatory_tags)
            if not tags:
                return False
            return bool(tags & context.task_tags)
        return True
    
    def retrieve(
        self,
        query: str,
        max_items: Optional[int] = None,
        min_importance: Optional[int] = None,
        include_types: Optional[List[SegmentType]] = None,
        exclude_types: Optional[List[SegmentType]] = None,
        track_access: Optional[bool] = None,
        apply_pops: Optional[bool] = None,
    ) -> RetrievalResult:
        """
        Retrieve context for a query.

        Args:
            query: The user's query/prompt
                   Prefix with "long-term:" to use original scores (no decay)
            max_items: Override max relevant items
            min_importance: Override minimum importance score
            include_types: Only include these segment types
            exclude_types: Exclude these segment types
            track_access: Override config.track_access
            apply_pops: Override config.apply_pops_on_retrieve

        Returns:
            RetrievalResult with mandatory and relevant segments
        """
        max_items = max_items or self.config.max_relevant_items
        min_importance = min_importance or self.config.min_importance
        track_access = track_access if track_access is not None else self.config.track_access
        apply_pops = apply_pops if apply_pops is not None else self.config.apply_pops_on_retrieve

        # Clear scoring cache
        self._scoring_cache.clear()
        
        # Build query context for modifier pipeline
        context = QueryContext.from_query(query)
        context.persist_dir = self.store.persist_dir
        
        result = RetrievalResult(query=query)
        accessed_ids = []
        
        # 1. Always get mandatory items (score 100)
        if self.config.always_include_mandatory:
            mandatory = self.store.get_mandatory()
            mandatory = [m for m in mandatory if self._should_include_mandatory(m, context)]
            
            # Filter by type if specified
            if include_types:
                mandatory = [m for m in mandatory if m.segment.segment_type in include_types]
            if exclude_types:
                mandatory = [m for m in mandatory if m.segment.segment_type not in exclude_types]
            
            result.mandatory = mandatory
            accessed_ids.extend(m.id for m in mandatory)
            logger.debug(f"Retrieved {len(mandatory)} mandatory items")
        
        # 2. Query for relevant items
        semantic_relevant = self.store.search(
            query=context.query,  # Use cleaned query (without long-term: prefix)
            limit=max_items * 2,  # Get extra to filter
            min_similarity=self.config.min_similarity,
        )

        identifier_relevant = self.store.search_by_identifiers(
            identifiers=list(context.query_identifiers),
            limit=self.config.identifier_candidate_limit,
        )

        candidate_segments: Dict[str, ScoredSegment] = {}
        for seg in semantic_relevant:
            candidate_segments[seg.id] = seg
        for seg, _, _ in identifier_relevant:
            candidate_segments.setdefault(seg.id, seg)
        
        # Apply modifier pipeline and filter
        filtered_with_scores = []
        mandatory_ids = {m.id for m in result.mandatory}
        
        for seg in candidate_segments.values():
            # Skip if already in mandatory
            if seg.id in mandatory_ids:
                continue
            
            # Filter by type
            if include_types and seg.segment.segment_type not in include_types:
                continue
            if exclude_types and seg.segment.segment_type in exclude_types:
                continue
            
            # Apply modifier pipeline
            scoring_result = self.pipeline.apply(seg, context)
            self._scoring_cache[seg.id] = scoring_result

            # Skip if below original threshold and there was no exact identifier rescue
            if seg.importance_score < min_importance and scoring_result.identifier_hits <= 0:
                continue
            
            # Skip if effective score too low
            if scoring_result.effective_score < self.config.min_effective_score and not scoring_result.force_include:
                logger.debug(f"Filtered {seg.id}: effective score {scoring_result.effective_score:.1f} < {self.config.min_effective_score}")
                continue
            
            filtered_with_scores.append((seg, scoring_result))
        
        # Sort by force-include, exact hits, then effective score (highest first)
        filtered_with_scores.sort(
            key=lambda item: (
                item[1].force_include,
                item[1].identifier_hits,
                item[1].effective_score,
            ),
            reverse=True,
        )
        
        # Limit to max_items
        result.relevant = [seg for seg, _ in filtered_with_scores[:max_items]]
        accessed_ids.extend(seg.id for seg, _ in filtered_with_scores[:max_items])
        logger.debug(f"Retrieved {len(result.relevant)} relevant items")

        # 2.5. Apply pop rules to any segment that was retrieved with
        # identifier_hits >= 2 (i.e., the query explicitly hit the memory).
        # This is what makes "search → memory pops" automatic.
        if apply_pops:
            from .dormancy import config_for
            pop_cfg = self.config.pop_dormancy_config or config_for(self.store.persist_dir)
            pop_cfg.persist_dir = self.store.persist_dir
            result.pop_results = self._apply_pops_to_results(
                result.mandatory + result.relevant,
                context,
                pop_cfg,
            )
            result.relevant.sort(
                key=lambda segment: (
                    self._scoring_cache[segment.id].force_include,
                    self._scoring_cache[segment.id].identifier_hits,
                    self._scoring_cache[segment.id].effective_score,
                ),
                reverse=True,
            )

        # 3. Record access for retrieved segments
        # FAILSAFE: access tracking is non-critical; if it fails (read-only
        # mode, write error), log and continue — don't crash the retrieval.
        if track_access and accessed_ids:
            now = context.now
            try:
                self.store.record_access_batch(accessed_ids, now)
                logger.debug(f"Recorded access for {len(accessed_ids)} segments")
            except Exception as e:
                logger.debug("Access tracking skipped; details withheld")

        return result

    def _apply_pops_to_results(
        self,
        segments: List[ScoredSegment],
        context: QueryContext,
        pop_cfg: DormancyConfig,
    ) -> List[PopResultRecord]:
        """
        Apply pop rules to retrieved segments based on identifier hits.

        For each segment in the result, look up the scoring result from the
        cache (or re-apply the pipeline for mandatory items that may not have
        been scored). If identifier_hits >= 2, call apply_pop, persist the
        mutation, and return a PopResultRecord for the audit trail.

        Returns a list of PopResultRecord (empty if no pops fired).
        """
        records: List[PopResultRecord] = []

        for seg in segments:
            # Re-resolve scoring result. For relevant segments, the cache has
            # the post-decay view. For mandatory items, the modifier pipeline
            # may not have run (we filter mandatory at the store level), so
            # re-apply on demand.
            scoring = self._scoring_cache.get(seg.id)
            if scoring is None:
                scoring = self.pipeline.apply(seg, context)
                self._scoring_cache[seg.id] = scoring

            matched = scoring.identifier_hits
            if matched < 2:
                continue

            # Snapshot the "before" state for the audit record
            effective_before = scoring.effective_score
            initial_before = seg.initial_score
            was_mandatory = seg.is_mandatory

            # Apply pop (mutates seg in place)
            pop_result: PopResult = apply_pop(
                seg,
                matched_keywords=matched,
                cfg=pop_cfg,
                now=context.now,
            )

            # If apply_pop reported no rule, skip persistence (shouldn't
            # happen since matched >= 2, but be defensive)
            if pop_result.rule_applied == "none":
                continue

            # Persist the mutation. store.add() is upsert semantics for
            # Chroma (it updates metadata for the existing ID).
            # FAILSAFE: if the store is read-only or the write fails, log
            # the warning and skip persistence — the in-memory mutation
            # still affects this retrieval result, it just won't survive
            # to the next process. This prevents a write failure from
            # crashing the entire retrieval stack.
            try:
                self.store.add(seg)
            except RuntimeError as e:
                # Read-only mode — pop fired in-memory but can't persist
                logger.debug(
                    f"Pop fired on '{seg.concept_title}' but store is "
                    f"read-only; mutation not persisted: {e}"
                )
            except Exception as e:
                # Any other write failure — don't crash the retrieval
                logger.warning(
                    f"Pop fired on '{seg.concept_title}' but persist failed: {e}. "
                    f"Mutation applied in-memory only."
                )

            refreshed = self.pipeline.apply(seg, context)
            self._scoring_cache[seg.id] = refreshed
            effective_after = refreshed.effective_score

            records.append(
                PopResultRecord(
                    segment_id=seg.id,
                    segment_title=seg.concept_title,
                    matched_keywords=matched,
                    rule_applied=pop_result.rule_applied,
                    initial_score_before=initial_before,
                    initial_score_after=seg.initial_score,
                    effective_score_before=round(effective_before, 1),
                    effective_score_after=round(effective_after, 1),
                    anchor_reset=pop_result.anchor_reset,
                    message=pop_result.message,
                    was_mandatory=was_mandatory,
                )
            )
            logger.info(
                f"Pop fired on '{seg.concept_title}' "
                f"(matched={matched}, rule={pop_result.rule_applied})"
            )

        return records
    
    def get_scoring_result(self, segment_id: str) -> Optional[ScoringResult]:
        """
        Get the scoring result for a segment from the last retrieve.
        
        Useful for diagnostics and understanding why a score was adjusted.
        """
        return self._scoring_cache.get(segment_id)
    
    def retrieve_by_score(
        self,
        min_score: int = 70,
        max_items: int = 50,
    ) -> List[ScoredSegment]:
        """
        Retrieve all items above a score threshold.
        
        Useful for getting all high-priority context regardless of query.
        """
        return self.store.get_by_score(min_score=min_score, limit=max_items)
    
    def retrieve_mandatory(self) -> List[ScoredSegment]:
        """Retrieve only mandatory (score 100) items."""
        return self.store.get_mandatory()
    
    def get_context_string(
        self,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Convenience method to get formatted context string.
        
        Args:
            query: The user's query
            max_tokens: Token budget
        
        Returns:
            Formatted context string ready for injection
        """
        max_tokens = max_tokens or self.config.max_tokens
        result = self.retrieve(query)
        return result.to_context_string(max_tokens=max_tokens)
    
    def get_score_diagnostics(
        self,
        segment_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """
        Get detailed scoring diagnostics for a segment.
        
        Args:
            segment_id: ID of the segment to diagnose
            now: Time to use for calculations (defaults to now)
        
        Returns:
            Dict with scoring details, or None if segment not found
        """
        segment = self.store.get(segment_id)
        if not segment:
            return None
        
        context = QueryContext(now=now or datetime.now())
        context.persist_dir = self.store.persist_dir
        scoring_result = self.pipeline.apply(segment, context)
        from .dormancy import _active_age, config_for
        
        return {
            "segment_id": segment_id,
            "concept_title": segment.concept_title,
            "original_score": segment.importance_score,
            "effective_score": round(scoring_result.effective_score, 1),
            "total_adjustment": round(scoring_result.total_adjustment, 1),
            "is_immortal": scoring_result.is_immortal,
            "identifier_hits": scoring_result.identifier_hits,
            "force_include": scoring_result.force_include,
            "matched_identifiers": scoring_result.matched_identifiers,
            "audit_trail": [
                {
                    "modifier": e.modifier_name,
                    "from": round(e.original_score, 1),
                    "to": round(e.adjusted_score, 1),
                    "delta": round(e.delta, 1),
                    "reason": e.reason,
                }
                for e in scoring_result.audit_trail
            ],
            "access_tracking": {
                "last_accessed": segment.last_accessed.isoformat() if segment.last_accessed else None,
                "access_count": segment.access_count,
                "popped_at": segment.popped_at.isoformat() if segment.popped_at else None,
                "accesses_since_pop": segment.accesses_since_pop,
            },
            "age_days": (context.now - segment.segment.timestamp).days,
            "active_age_hours": _active_age(segment, context.now, config_for(self.store.persist_dir)),
            "created_at": segment.segment.timestamp.isoformat(),
        }


# Global retriever instance
_retriever: Optional[PriorityRetriever] = None


def get_retriever() -> Optional[PriorityRetriever]:
    """Get the global retriever instance."""
    return _retriever


def set_retriever(retriever: PriorityRetriever) -> None:
    """Set the global retriever instance."""
    global _retriever
    _retriever = retriever


def get_context(
    query: str,
    max_tokens: int = 8000,
) -> str:
    """
    Convenience function to get context for a query.
    
    Args:
        query: The user's query/prompt
        max_tokens: Token budget
    
    Returns:
        Formatted context string, or empty string if retriever not initialized
    """
    if _retriever is None:
        logger.warning("Instinct retriever not initialized")
        return ""
    
    return _retriever.get_context_string(query, max_tokens)
