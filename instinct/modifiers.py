# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Score Modifier Pipeline

Extensible system for adjusting importance scores based on various factors.
Each modifier in the pipeline can adjust scores and leave an audit trail.

Usage:
    pipeline = ModifierPipeline()
    pipeline.register(DecayModifier())
    pipeline.register(KeywordBoostModifier())
    
    result = pipeline.apply(segment, query_context)
    print(result.effective_score)  # Final score after all modifiers
    print(result.audit_trail)      # List of adjustments made
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .models import extract_query_identifiers


def infer_query_tags(query: str) -> Set[str]:
    """Infer lightweight applicability tags from a query."""
    normalized = query.lower()
    tags: Set[str] = set()

    tag_rules = {
        "coding": ["code", "function", "class", "refactor", "bug", "test", "python", "php", "typescript", "javascript"],
        "security": ["security", "auth", "token", "credential", "secret", "nonce", "permission", "capability"],
        "memory": ["memory", "instinct", "context", "mandatory", "retrieval", "identifier", "decay"],
        "git": ["git", "commit", "branch", "diff", "merge", "rebase"],
        "frontend": ["ui", "frontend", "css", "html", "react", "component"],
        "ops": ["deploy", "server", "api", "database", "docker", "wsl", "terminal"],
    }

    for tag, needles in tag_rules.items():
        if any(needle in normalized for needle in needles):
            tags.add(tag)

    return tags

if TYPE_CHECKING:
    from .models import ScoredSegment


@dataclass
class QueryContext:
    """
    Context passed through the modifier pipeline.
    
    Provides information about the current query that modifiers
    can use to make scoring decisions.
    """
    query: str = ""
    keywords: Set[str] = field(default_factory=set)  # Extracted keywords
    query_identifiers: Set[str] = field(default_factory=set)  # Exact-match identifiers
    task_tags: Set[str] = field(default_factory=set)  # Applicability tags for mandatory routing
    syntax_flags: Dict[str, Any] = field(default_factory=dict)  # e.g., {"exact_match": True}
    now: datetime = field(default_factory=datetime.now)  # Current time for decay
    long_term_mode: bool = False  # If True, use original scores (no decay)
    persist_dir: Optional[str] = None
    
    @classmethod
    def empty(cls) -> "QueryContext":
        """Create empty context for simple retrievals."""
        return cls()
    
    @classmethod
    def from_query(cls, query: str, now: Optional[datetime] = None) -> "QueryContext":
        """Create context from a query string, extracting keywords."""
        keywords = set()
        syntax_flags = {}
        
        # Extract quoted exact phrases
        import re
        exact_matches = re.findall(r'"([^"]+)"', query)
        if exact_matches:
            syntax_flags["exact_phrases"] = exact_matches
        
        # Check for special prefixes
        if query.startswith("long-term:"):
            return cls(
                query=query[10:].strip(),
                keywords=keywords,
                syntax_flags=syntax_flags,
                now=now or datetime.now(),
                long_term_mode=True,
            )
        
        # Extract simple keywords (words 3+ chars, excluding stopwords)
        stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out"}
        words = re.findall(r'\b\w{3,}\b', query.lower())
        keywords = {w for w in words if w not in stopwords}
        
        return cls(
            query=query,
            keywords=keywords,
            query_identifiers=set(extract_query_identifiers(query)),
            task_tags=infer_query_tags(query),
            syntax_flags=syntax_flags,
            now=now or datetime.now(),
            long_term_mode=False,
        )


@dataclass
class AuditEntry:
    """Single entry in the scoring audit trail."""
    modifier_name: str
    original_score: float
    adjusted_score: float
    delta: float
    reason: str
    
    def __str__(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return f"{self.modifier_name}: {self.original_score:.1f} → {self.adjusted_score:.1f} ({sign}{self.delta:.1f}) - {self.reason}"


@dataclass
class ScoringResult:
    """
    Result of applying the modifier pipeline to a segment.
    
    Preserves original score and tracks all adjustments.
    """
    segment_id: str
    original_score: int  # The stored importance_score
    effective_score: float  # Final score after all modifiers
    identifier_hits: int = 0
    force_include: bool = False
    matched_identifiers: List[str] = field(default_factory=list)
    audit_trail: List[AuditEntry] = field(default_factory=list)
    
    @property
    def total_adjustment(self) -> float:
        """Net change from original score."""
        return self.effective_score - self.original_score
    
    @property
    def is_immortal(self) -> bool:
        """Scores 99-100 are immortal (no decay)."""
        return self.original_score >= 99
    
    def add_adjustment(self, modifier_name: str, new_score: float, reason: str) -> None:
        """Record an adjustment to the score."""
        current = self.effective_score
        delta = new_score - current
        self.audit_trail.append(AuditEntry(
            modifier_name=modifier_name,
            original_score=current,
            adjusted_score=new_score,
            delta=delta,
            reason=reason,
        ))
        self.effective_score = new_score
    
    def format_audit(self) -> str:
        """Format audit trail for display."""
        lines = [f"Score: {self.original_score} → {self.effective_score:.1f}"]
        for entry in self.audit_trail:
            lines.append(f"  {entry}")
        return "\n".join(lines)


class ScoreModifier(ABC):
    """
    Base class for score modifiers.
    
    Each modifier can adjust scores based on its own logic.
    Modifiers are applied in priority order (lower priority first).
    """
    
    def __init__(self, name: str, priority: int = 50, enabled: bool = True):
        """
        Initialize modifier.
        
        Args:
            name: Human-readable name for audit trail
            priority: Execution order (0-100, lower runs first)
            enabled: Whether this modifier is active
        """
        self.name = name
        self.priority = priority
        self.enabled = enabled
    
    @abstractmethod
    def modify(
        self, 
        segment: "ScoredSegment", 
        context: QueryContext, 
        result: ScoringResult
    ) -> ScoringResult:
        """
        Apply modification to the score.
        
        Args:
            segment: The segment being scored
            context: Query context with keywords, time, etc.
            result: Current scoring result (mutate effective_score)
        
        Returns:
            The same result object with any adjustments applied
        """
        pass
    
    def should_skip(self, segment: "ScoredSegment", context: QueryContext) -> bool:
        """
        Check if this modifier should be skipped.
        
        Override for custom skip logic. Base implementation
        checks if modifier is disabled.
        """
        return not self.enabled


class ModifierPipeline:
    """
    Registry and executor for score modifiers.
    
    Modifiers are registered and then applied in priority order
    to produce effective scores with full audit trails.
    """
    
    def __init__(self):
        self._modifiers: List[ScoreModifier] = []
        self._sorted = True
    
    def register(self, modifier: ScoreModifier) -> "ModifierPipeline":
        """
        Add a modifier to the pipeline.
        
        Args:
            modifier: The modifier to register
        
        Returns:
            Self for chaining
        """
        self._modifiers.append(modifier)
        self._sorted = False
        return self
    
    def unregister(self, name: str) -> bool:
        """
        Remove a modifier by name.
        
        Args:
            name: Name of the modifier to remove
        
        Returns:
            True if found and removed
        """
        original_len = len(self._modifiers)
        self._modifiers = [m for m in self._modifiers if m.name != name]
        return len(self._modifiers) < original_len
    
    def get(self, name: str) -> Optional[ScoreModifier]:
        """Get a modifier by name."""
        for m in self._modifiers:
            if m.name == name:
                return m
        return None
    
    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a modifier by name."""
        modifier = self.get(name)
        if modifier:
            modifier.enabled = enabled
            return True
        return False
    
    def _ensure_sorted(self) -> None:
        """Sort modifiers by priority if needed."""
        if not self._sorted:
            self._modifiers.sort(key=lambda m: m.priority)
            self._sorted = True
    
    @property
    def modifiers(self) -> List[ScoreModifier]:
        """Get sorted list of modifiers."""
        self._ensure_sorted()
        return self._modifiers.copy()
    
    def apply(
        self, 
        segment: "ScoredSegment", 
        context: Optional[QueryContext] = None
    ) -> ScoringResult:
        """
        Apply all enabled modifiers to a segment.
        
        Args:
            segment: The segment to score
            context: Optional query context (empty context if not provided)
        
        Returns:
            ScoringResult with effective score and audit trail
        """
        context = context or QueryContext.empty()
        
        result = ScoringResult(
            segment_id=segment.id,
            original_score=segment.importance_score,
            effective_score=float(segment.importance_score),
        )
        
        self._ensure_sorted()
        
        for modifier in self._modifiers:
            if modifier.should_skip(segment, context):
                continue
            result = modifier.modify(segment, context, result)
        
        return result
    
    def apply_batch(
        self,
        segments: List["ScoredSegment"],
        context: Optional[QueryContext] = None
    ) -> Dict[str, ScoringResult]:
        """
        Apply modifiers to multiple segments.
        
        Args:
            segments: List of segments to score
            context: Optional query context
        
        Returns:
            Dict mapping segment ID to ScoringResult
        """
        return {seg.id: self.apply(seg, context) for seg in segments}
    
    def describe(self) -> str:
        """Describe the pipeline configuration."""
        self._ensure_sorted()
        lines = ["Modifier Pipeline:"]
        for m in self._modifiers:
            status = "✓" if m.enabled else "✗"
            lines.append(f"  [{status}] {m.name} (priority: {m.priority})")
        return "\n".join(lines)


class ExactIdentifierModifier(ScoreModifier):
    """Boost scores when query identifiers exactly match memory identifiers."""

    ONE_HIT_SCORE = 80.0
    TWO_HIT_SCORE = 88.0
    THREE_HIT_SCORE = 100.0
    FORCE_INCLUDE_HITS = 3

    def __init__(self, enabled: bool = True):
        super().__init__(name="exact_identifier", priority=30, enabled=enabled)

    def should_skip(self, segment: "ScoredSegment", context: QueryContext) -> bool:
        if not self.enabled:
            return True
        return not context.query_identifiers

    def modify(
        self,
        segment: "ScoredSegment",
        context: QueryContext,
        result: ScoringResult,
    ) -> ScoringResult:
        matched = [
            identifier
            for identifier in segment.memory_identifiers
            if identifier in context.query_identifiers
        ]

        hit_count = len(matched)
        result.identifier_hits = hit_count
        result.matched_identifiers = matched

        if hit_count <= 0:
            return result

        if hit_count >= self.FORCE_INCLUDE_HITS:
            result.force_include = True
            boosted = self.THREE_HIT_SCORE
            reason = f"{hit_count} exact identifier hits: force include ({', '.join(matched[:3])})"
        elif hit_count == 2:
            boosted = max(result.effective_score, self.TWO_HIT_SCORE)
            reason = f"2 exact identifier hits: boosted floor to 88 ({', '.join(matched)})"
        else:
            boosted = max(result.effective_score, self.ONE_HIT_SCORE)
            reason = f"1 exact identifier hit: boosted floor to 80 ({matched[0]})"

        result.add_adjustment(self.name, boosted, reason)
        return result


# ============================================================
# Default Pipeline Factory
# ============================================================

def create_default_pipeline() -> ModifierPipeline:
    """
    Create pipeline with standard modifiers.
    
    Default order:
    1. DecayModifier (priority 10) - Time-based score decay
    2. PopModifier (priority 20) - Dormant memory resurrection
    3. ExactIdentifierModifier (priority 30) - Exact-match identifier boost
    4. KeywordModifier (priority 50) - Coming soon
    5. RelevancyModifier (priority 60) - Coming soon
    
    Returns:
        Configured ModifierPipeline
    """
    from .decay import DecayModifier, PopModifier
    
    pipeline = ModifierPipeline()
    pipeline.register(DecayModifier())
    pipeline.register(PopModifier())
    pipeline.register(ExactIdentifierModifier())
    # Future: pipeline.register(KeywordModifier())
    # Future: pipeline.register(RelevancyModifier())
    
    return pipeline
