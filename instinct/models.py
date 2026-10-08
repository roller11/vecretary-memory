# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Data Models

Core data structures for the priority context system.
"""

import hashlib
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


IDENTIFIER_MIN_COUNT = 3
IDENTIFIER_MAX_COUNT = 6
IDENTIFIER_DELIMITER = "||"
TAG_DELIMITER = "|"
IDENTIFIER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "with",
}

# Recycle bin: soft-delete marker. Segments with a `recycled_at` ISO
# timestamp are hidden from the default list / search / retriever, but
# remain in chroma and can be restored via the dashboard's "Recycle
# bin" view. This is the safe alternative to hard `delete()` — every
# accidental wipe is reversible. The MCP server does not see recycled
# segments unless explicitly opted in (the dashboard's `include_recycled=1`
# query param is a UI-only escape hatch; the MCP's `instinct_get_context`
# never includes them by default).
# Nudge metadata keys (turn-based gentle reminder system).
# Kept here as constants so storage layouts stay in sync between producers
# (dormancy/nudge modules) and consumers (dashboard, mcp_server, retriever).
NUDGED_AT_TURN_KEY = "nudged_at_turn"
NUDGE_INTERVAL_TURNS_KEY = "nudge_interval_turns"
TURNS_SINCE_ACCESS_KEY = "turns_since_access"
NUDGE_ACTIVE_KEY = "nudge_active"

# Recycle bin keys (soft-delete pattern). When any of these is set to a
# truthy value in metadata, the segment is treated as soft-deleted and
# filtered out of all default reads.
RECYCLED_AT_KEY = "recycled_at"
RECYCLED_BY_KEY = "recycled_by"
RECYCLED_REASON_KEY = "recycled_reason"


def normalize_identifier(identifier: str) -> str:
    """Normalize identifier terms for exact matching."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", identifier.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def serialize_memory_identifiers(identifiers: List[str]) -> str:
    """Serialize identifiers for Chroma metadata storage."""
    return IDENTIFIER_DELIMITER.join(identifiers)


def serialize_tag_list(tags: List[str]) -> str:
    """Serialize applicability tags for storage."""
    return TAG_DELIMITER.join(tags)


def parse_memory_identifiers(value: Any) -> List[str]:
    """Parse stored identifier metadata into a normalized list."""
    if isinstance(value, list):
        raw_identifiers = value
    elif isinstance(value, str) and value:
        normalized_value = value.replace("\r", "\n")
        if IDENTIFIER_DELIMITER in normalized_value:
            raw_identifiers = normalized_value.split(IDENTIFIER_DELIMITER)
        else:
            raw_identifiers = re.split(r"[,\n]", normalized_value)
    else:
        raw_identifiers = []

    identifiers: List[str] = []
    seen = set()
    for item in raw_identifiers:
        normalized = normalize_identifier(str(item))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        identifiers.append(normalized)
    return identifiers


def parse_tag_list(value: Any) -> List[str]:
    """Parse applicability tags into a normalized unique list."""
    if isinstance(value, list):
        raw_tags = value
    elif isinstance(value, str) and value:
        normalized_value = value.replace("\r", "\n")
        if TAG_DELIMITER in normalized_value:
            raw_tags = normalized_value.split(TAG_DELIMITER)
        else:
            raw_tags = re.split(r"[,\n]", normalized_value)
    else:
        raw_tags = []

    tags: List[str] = []
    seen = set()
    for item in raw_tags:
        normalized = normalize_identifier(str(item))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        tags.append(normalized)
    return tags


def extract_query_identifiers(query: str) -> List[str]:
    """Extract exact-match identifier candidates from a query."""
    normalized_query = normalize_identifier(query)
    if not normalized_query:
        return []

    candidates: List[str] = []
    seen = set()

    def add_candidate(value: str) -> None:
        normalized = normalize_identifier(value)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    add_candidate(normalized_query)

    words = [
        word for word in normalized_query.split()
        if len(word) >= 3 or word.isdigit()
    ]
    for word in words:
        add_candidate(word)

    max_ngram = min(8, len(words))
    for size in range(max_ngram, 1, -1):
        for index in range(len(words) - size + 1):
            phrase_words = words[index:index + size]
            if all(word not in IDENTIFIER_STOPWORDS for word in phrase_words):
                add_candidate(" ".join(phrase_words))

    return candidates


def build_memory_identifiers(
    concept_title: str,
    content: str,
    provided_identifiers: Optional[List[str]] = None,
) -> List[str]:
    """Build a stable 3-6 identifier pack for a memory."""
    ordered: List[str] = []
    seen = set()

    def add_identifier(value: str) -> None:
        normalized = normalize_identifier(value)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        ordered.append(normalized)

    for identifier in provided_identifiers or []:
        add_identifier(identifier)

    title_normalized = normalize_identifier(concept_title)
    if title_normalized:
        add_identifier(title_normalized)

    for phrase in re.findall(r'"([^"]+)"', f"{concept_title}\n{content[:1000]}"):
        add_identifier(phrase)

    title_words = [
        word for word in normalize_identifier(concept_title).split()
        if len(word) >= 3 and word not in IDENTIFIER_STOPWORDS
    ]
    content_words = [
        word for word in normalize_identifier(content[:1200]).split()
        if len(word) >= 4 and word not in IDENTIFIER_STOPWORDS
    ]

    for size in (3, 2):
        for index in range(len(title_words) - size + 1):
            add_identifier(" ".join(title_words[index:index + size]))

    for word in title_words:
        add_identifier(word)

    for word in content_words:
        add_identifier(word)
        if len(ordered) >= IDENTIFIER_MAX_COUNT:
            break

    if not ordered:
        add_identifier("memory")

    if len(ordered) < IDENTIFIER_MIN_COUNT:
        fallback_words = [
            word for word in normalize_identifier(f"{concept_title} {content[:400]}").split()
            if len(word) >= 3
        ]
        for word in fallback_words:
            add_identifier(word)
            if len(ordered) >= IDENTIFIER_MIN_COUNT:
                break

    return ordered[:IDENTIFIER_MAX_COUNT]


def build_validation_token(
    mandatory_fingerprint: str = "",
) -> str:
    """Build a fresh acknowledgement challenge, independent of calendar time."""
    fingerprint = mandatory_fingerprint or "none"
    token_seed = f"instinct:{fingerprint}:{secrets.token_hex(16)}"
    return hashlib.sha256(token_seed.encode("utf-8")).hexdigest()[:12].upper()


class SegmentType(Enum):
    """Type of conversation segment."""
    
    STANDING_INSTRUCTION = "standing_instruction"  # Always score 100
    ARCHITECTURAL_DECISION = "architectural_decision"
    CRITICAL_CONSTRAINT = "critical_constraint"  # Non-negotiable rules
    IMPLEMENTATION_DETAIL = "implementation_detail"
    EXPLORATION = "exploration"  # Brainstorming, ideas
    CONTEXT = "context"  # Background information
    QUESTION = "question"
    ANSWER = "answer"
    CODE = "code"
    ERROR_RESOLUTION = "error_resolution"
    USER_PREFERENCE = "user_preference"


# Segment types that automatically get score 100
MANDATORY_TYPES = {
    SegmentType.STANDING_INSTRUCTION,
    SegmentType.CRITICAL_CONSTRAINT,
}


@dataclass
class Segment:
    """A logical segment of conversation."""
    
    content: str
    segment_type: SegmentType
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # e.g., "conversation", "manual", "import"
    metadata: Dict[str, Any] = field(default_factory=dict)
    stored_id: Optional[str] = field(default=None, repr=False, compare=False)
    
    @property
    def id(self) -> str:
        """Generate unique ID from content hash."""
        if self.stored_id is not None:
            return self.stored_id
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        ts = self.timestamp.strftime("%Y%m%d%H%M%S")
        return f"{ts}_{content_hash}"
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class ScoredSegment:
    """A segment with importance scoring."""

    segment: Segment
    importance_score: int  # 0-100, current decayed score
    concept_title: str  # Brief title for the concept
    reasoning: str = ""  # Why this score was assigned
    memory_identifiers: List[str] = field(default_factory=list)

    # Vector embedding (populated when stored)
    embedding: Optional[List[float]] = None

    # Access tracking for decay system
    last_accessed: Optional[datetime] = None  # When last retrieved
    access_count: int = 0  # Total retrieval count

    # Pop state machine for dormant memory resurrection
    popped_at: Optional[datetime] = None  # When memory "popped" back
    accesses_since_pop: int = 0  # Accesses after pop (need 2 in a week to "stick")

    # Dormancy system (v1)
    last_bumped_at: Optional[datetime] = None  # When last promoted back to surface
    bump_count: int = 0  # Number of times this segment has been bumped
    decay_profile: str = "standard"  # "standard" | "short_term_critical" | "ephemeral"

    # Ephemeral state machine (only used when decay_profile == "ephemeral")
    ephemeral_phase: str = "surface"   # "surface" | "fading" | "expired"
    ephemeral_turns_left: int = 0      # turns remaining in current phase

    # KPI tracking (v3 dashboard, 2026-06-15)
    # initial_score is the immutable anchor: it never decays.
    # The "current" score visible in the UI is the average of
    # initial_score and importance_score.
    initial_score: int = 0  # 0-100; set in __post_init__ from importance_score
    initial_decay_path: str = ""  # e.g., "plateau", "tier 94", "STC 92", "ephemeral"
    initial_semantic_score: float = 0.0  # Cosine similarity at creation, 0..1

    # Keyword hit tracking
    keyword_hits_total: int = 0
    keyword_hits_log: List[Dict[str, Any]] = field(default_factory=list)  # [{ts, count, query_id?}]

    # Pop rule tracking (separate from popped_at / accesses_since_pop which
    # are the legacy v1 fields)
    last_popped_at: Optional[datetime] = None
    pop_anchor_at: Optional[datetime] = None  # For "effective time since pop"
    pop_recent_8h_active: int = 0  # Count of pops in last 8 active hours
    promotions_applied: int = 0  # How many times the +10 score promotion has fired
    active_time_anchors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Enforce score 100 for mandatory types
        if self.segment.segment_type in MANDATORY_TYPES:
            self.importance_score = 100

        # Clamp score to valid range
        self.importance_score = max(0, min(100, self.importance_score))

        # If initial_score wasn't explicitly set (default 0), anchor it to the
        # current importance_score. This makes the existing code paths that
        # construct ScoredSegment(..., importance_score=N) automatically get
        # initial_score=N — so "overall" = average = N too.
        if self.initial_score == 0 and self.importance_score > 0:
            self.initial_score = self.importance_score
        self.initial_score = max(0, min(100, self.initial_score))

        # Only rebuild identifiers if not already populated.
        # When reconstructing from storage (via from_storage_dict), identifiers
        # are already parsed and passed in — rebuilding them here causes OOM
        # on bulk loads because build_memory_identifiers tokenizes full content.
        if not self.memory_identifiers:
            self.memory_identifiers = build_memory_identifiers(
                self.concept_title,
                self.segment.content,
                self.memory_identifiers,
            )
    
    @property
    def id(self) -> str:
        return self.segment.id
    
    @property
    def is_mandatory(self) -> bool:
        """Score 100 items must always be retrieved."""
        return self.importance_score == 100

    @property
    def is_mandatory_memory(self) -> bool:
        """Recognize a mandatory record even while its delivery score is suppressed."""
        return self.is_mandatory or self.initial_score == 100 or self.segment.segment_type in MANDATORY_TYPES
    
    @property
    def is_high_priority(self) -> bool:
        """Score 70+ items are high priority."""
        return self.importance_score >= 70

    @property
    def mandatory_scope(self) -> str:
        """Mandatory scope: always or conditional."""
        if not self.is_mandatory_memory:
            return "standard"
        scope = normalize_identifier(str(self.segment.metadata.get("mandatory_scope", "always")))
        if scope in {"conditional", "conditional mandatory"}:
            return "conditional"
        return "always"

    @property
    def mandatory_tags(self) -> List[str]:
        """Applicability tags for conditional mandatory items."""
        return parse_tag_list(self.segment.metadata.get("mandatory_tags", []))
    
    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dict for vector store metadata."""
        return {
            **self.segment.metadata,
            "id": self.id,
            "content": self.segment.content,
            "segment_type": self.segment.segment_type.value,
            "importance_score": self.importance_score,
            "concept_title": self.concept_title,
            "reasoning": self.reasoning,
            "memory_identifiers": serialize_memory_identifiers(self.memory_identifiers),
            "identifier_count": len(self.memory_identifiers),
            "mandatory_scope": self.mandatory_scope,
            "mandatory_tags": serialize_tag_list(self.mandatory_tags),
            "timestamp": self.segment.timestamp.isoformat(),
            "source": self.segment.source,
            "char_count": self.segment.char_count,
            "word_count": self.segment.word_count,
            # Access tracking
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "popped_at": self.popped_at.isoformat() if self.popped_at else None,
            "accesses_since_pop": self.accesses_since_pop,
            "last_bumped_at": self.last_bumped_at.isoformat() if self.last_bumped_at else None,
            "bump_count": self.bump_count,
            "decay_profile": self.decay_profile,
            "ephemeral_phase": self.ephemeral_phase,
            "ephemeral_turns_left": self.ephemeral_turns_left,
            # KPI fields
            "initial_score": self.initial_score,
            "initial_decay_path": self.initial_decay_path,
            "initial_semantic_score": self.initial_semantic_score,
            "keyword_hits_total": self.keyword_hits_total,
            "keyword_hits_log": json.dumps(self.keyword_hits_log),
            "last_popped_at": self.last_popped_at.isoformat() if self.last_popped_at else None,
            "pop_anchor_at": self.pop_anchor_at.isoformat() if self.pop_anchor_at else None,
            "pop_recent_8h_active": self.pop_recent_8h_active,
            "promotions_applied": self.promotions_applied,
            "active_time_anchors": json.dumps(self.active_time_anchors),
        }
    
    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any], embedding: Optional[List[float]] = None) -> "ScoredSegment":
        """Reconstruct from vector store metadata."""
        # Fields to exclude from metadata passthrough
        reserved_keys = {
            "id", "content", "segment_type", "importance_score", 
            "concept_title", "reasoning", "timestamp", "source",
            "memory_identifiers", "identifier_count",
            "mandatory_scope", "mandatory_tags",
            "char_count", "word_count",
            # Access tracking fields
            "last_accessed", "access_count", "popped_at", "accesses_since_pop",
            "last_bumped_at", "bump_count", "decay_profile", "ephemeral_phase", "ephemeral_turns_left",
            "initial_score", "initial_decay_path", "initial_semantic_score", "keyword_hits_total",
            "keyword_hits_log", "last_popped_at", "pop_anchor_at", "pop_recent_8h_active", "promotions_applied",
            "active_time_anchors",
            # NOTE: nudge keys (nudge_interval_turns, nudged_at_turn,
            # turns_since_access, nudge_active) are intentionally NOT
            # reserved so they round-trip through segment.metadata and
            # can be read back by get_nudge_interval() / get_last_nudged_turn().
        }
        
        segment = Segment(
            content=data.get("content", ""),
            stored_id=data.get("id"),
            segment_type=SegmentType(data["segment_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", ""),
            metadata={
                **{k: v for k, v in data.items() if k not in reserved_keys},
                "mandatory_scope": data.get("mandatory_scope", "always" if data["importance_score"] == 100 else "standard"),
                "mandatory_tags": parse_tag_list(data.get("mandatory_tags", "")),
            },
        )
        
        # Parse access tracking fields (handle missing for backward compatibility)
        last_accessed = None
        if data.get("last_accessed"):
            last_accessed = datetime.fromisoformat(data["last_accessed"])

        popped_at = None
        if data.get("popped_at"):
            popped_at = datetime.fromisoformat(data["popped_at"])

        last_popped_at = None
        if data.get("last_popped_at"):
            last_popped_at = datetime.fromisoformat(data["last_popped_at"])

        pop_anchor_at = None
        if data.get("pop_anchor_at"):
            pop_anchor_at = datetime.fromisoformat(data["pop_anchor_at"])

        # Parse keyword_hits_log JSON
        khl_raw = data.get("keyword_hits_log", "[]")
        try:
            khl = json.loads(khl_raw) if isinstance(khl_raw, str) else (khl_raw or [])
        except (ValueError, TypeError):
            khl = []

        # Parse identifiers from storage (may be empty for legacy segments)
        stored_identifiers = parse_memory_identifiers(data.get("memory_identifiers", ""))

        # Backfill initial_score for legacy segments that don't have one
        # persisted yet. We treat the current importance_score as the anchor
        # unless the segment already has a higher one (from prior promotion).
        importance = max(0, min(100, data["importance_score"]))
        initial = data.get("initial_score", 0)
        if initial == 0 and importance > 0:
            initial = importance
        initial = max(0, min(100, initial))

        # Bypass __post_init__ to avoid expensive identifier rebuild on bulk loads.
        # When identifiers are missing in storage, we accept the empty list rather
        # than re-tokenizing full content — this prevents OOM during store init.
        instance = object.__new__(cls)
        instance.segment = segment
        instance.importance_score = importance
        instance.concept_title = data["concept_title"]
        instance.reasoning = data.get("reasoning", "")
        instance.memory_identifiers = stored_identifiers if stored_identifiers else []
        instance.embedding = embedding
        instance.last_accessed = last_accessed
        instance.access_count = data.get("access_count", 0)
        instance.popped_at = popped_at
        instance.accesses_since_pop = data.get("accesses_since_pop", 0)
        instance.last_bumped_at = datetime.fromisoformat(data["last_bumped_at"]) if data.get("last_bumped_at") else None
        instance.bump_count = data.get("bump_count", 0)
        instance.decay_profile = data.get("decay_profile") or "standard"
        instance.ephemeral_phase = data.get("ephemeral_phase") or "surface"
        instance.ephemeral_turns_left = data.get("ephemeral_turns_left", 0)
        # KPI fields (backfilled for legacy)
        instance.initial_score = initial
        instance.initial_decay_path = data.get("initial_decay_path", "")
        instance.initial_semantic_score = data.get("initial_semantic_score", 0.0)
        instance.keyword_hits_total = data.get("keyword_hits_total", 0)
        instance.keyword_hits_log = khl
        instance.last_popped_at = last_popped_at
        instance.pop_anchor_at = pop_anchor_at
        instance.pop_recent_8h_active = data.get("pop_recent_8h_active", 0)
        instance.promotions_applied = data.get("promotions_applied", 0)
        try:
            anchors = json.loads(data.get("active_time_anchors") or "{}")
            instance.active_time_anchors = {
                name: snapshot for name, snapshot in anchors.items()
                if isinstance(snapshot, dict)
            } if isinstance(anchors, dict) else {}
        except (ValueError, TypeError):
            instance.active_time_anchors = {}
        return instance


@dataclass
class ProjectIdentity:
    """
    The mission, vision, and goals for a project.
    Used by the scorer to evaluate segment importance.
    """
    
    name: str
    mission: str  # What we're trying to accomplish
    vision: str   # The ideal end state
    goals: List[str] = field(default_factory=list)  # Specific objectives
    
    # Standing instructions - always score 100
    standing_instructions: List[str] = field(default_factory=list)
    
    # Non-negotiable constraints - always score 100
    critical_constraints: List[str] = field(default_factory=list)
    
    # Keywords that indicate importance
    priority_keywords: List[str] = field(default_factory=list)
    
    def to_scoring_context(self) -> str:
        """Format project identity for the scoring model."""
        parts = [
            f"# Project: {self.name}",
            f"\n## Mission\n{self.mission}",
            f"\n## Vision\n{self.vision}",
        ]
        
        if self.goals:
            parts.append("\n## Goals")
            for i, goal in enumerate(self.goals, 1):
                parts.append(f"{i}. {goal}")
        
        if self.standing_instructions:
            parts.append("\n## Standing Instructions (Score 100)")
            for inst in self.standing_instructions:
                parts.append(f"- {inst}")
        
        if self.critical_constraints:
            parts.append("\n## Critical Constraints (Score 100)")
            for const in self.critical_constraints:
                parts.append(f"- {const}")
        
        return "\n".join(parts)


@dataclass
class PopResultRecord:
    """Lightweight pop audit record stored on RetrievalResult.

    Distinct from instinct.pop.PopResult (which mutates segments). This is
    the small, JSON-friendly summary that goes into the retrieval response
    so callers can show "this segment got popped with rule X" in the UI.
    """
    segment_id: str
    segment_title: str
    matched_keywords: int
    rule_applied: str  # "mandatory_bump" | "halve" | "jump_to_plateau" | "frequency_reset" | "none"
    initial_score_before: int
    initial_score_after: int
    effective_score_before: float
    effective_score_after: float
    anchor_reset: bool
    message: str
    was_mandatory: bool = False


@dataclass
class RetrievalResult:
    """Result from priority context retrieval."""

    # Mandatory items (score 100) - always included
    mandatory: List[ScoredSegment] = field(default_factory=list)

    # Query-relevant items (sorted by score descending)
    relevant: List[ScoredSegment] = field(default_factory=list)

    # Pop audit trail: which segments got their decay modified this retrieval
    pop_results: List[PopResultRecord] = field(default_factory=list)

    # Stats
    total_tokens_estimate: int = 0
    query: str = ""

    _validation_token: str = field(default_factory=build_validation_token, init=False, repr=False, compare=False)
    
    @property
    def all_segments(self) -> List[ScoredSegment]:
        """All segments, mandatory first, then relevant by score."""
        return self.mandatory + sorted(
            self.relevant, 
            key=lambda s: s.importance_score, 
            reverse=True
        )

    @property
    def acknowledgement_memories(self) -> List[ScoredSegment]:
        """Mandatory records actually delivered, never additional forced context."""
        delivered = self.mandatory + [segment for segment in self.relevant if segment.is_mandatory_memory]
        return list({segment.id: segment for segment in delivered}.values())

    @property
    def mandatory_titles(self) -> List[str]:
        """Human-readable list of mandatory titles in retrieval order."""
        return [seg.concept_title for seg in self.acknowledgement_memories]

    @property
    def mandatory_fingerprint(self) -> str:
        """Stable short fingerprint of the mandatory set included in context."""
        if not self.acknowledgement_memories:
            return "none"
        joined = "||".join(f"{seg.id}:{seg.concept_title}" for seg in self.acknowledgement_memories)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]

    @property
    def mandatory_verification(self) -> Dict[str, Any]:
        """Compact verification payload for the active mandatory set."""
        always_on = sum(1 for seg in self.mandatory if seg.mandatory_scope == "always")
        conditional = sum(1 for seg in self.mandatory if seg.mandatory_scope == "conditional")
        return {
            "fingerprint": self.mandatory_fingerprint,
            "validation_token": self.validation_token if self.acknowledgement_memories else None,
            "titles": self.mandatory_titles,
            "always_on_count": always_on,
            "conditional_count": conditional,
        }

    @property
    def validation_token(self) -> str:
        """Fresh per-retrieval challenge, stable within this response."""
        return self._validation_token

    def acknowledgement_context(self) -> str:
        """Render the challenge only when mandatory memories were delivered."""
        if not self.acknowledgement_memories:
            return ""
        return "\n".join((
            f"MANDATORY_FINGERPRINT: {self.mandatory_fingerprint}",
            f"VALIDATION_TOKEN: {self.validation_token}",
            f"RESPONSE_REQUIREMENT: End response with 「{self.validation_token}」",
        ))

    def _summarize_mandatory_content(self, seg: ScoredSegment, max_points: int = 4) -> List[str]:
        """Compress a mandatory segment into a few high-signal bullets."""
        bullets: List[str] = []
        token_bullets: List[str] = []
        seen = set()

        def add_bullet(text: str) -> None:
            cleaned = " ".join(text.strip().split())
            if not cleaned:
                return
            lowered = cleaned.lower().rstrip(" .;:")
            if lowered in seen:
                return
            seen.add(lowered)
            bullets.append(cleaned)

        for raw_line in seg.segment.content.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("VALIDATION_TOKEN:"):
                cleaned = " ".join(stripped.strip().split())
                lowered = cleaned.lower().rstrip(" .;:")
                if lowered not in seen:
                    seen.add(lowered)
                    token_bullets.append(cleaned)
                continue
            if stripped.startswith(("-", "*")):
                if len(bullets) < max_points:
                    add_bullet(stripped[1:].strip())
                continue
            if len(stripped) <= 180 and stripped.endswith(":"):
                continue
            if len(bullets) < max_points:
                add_bullet(stripped)

        if len(bullets) < max_points:
            content = " ".join(seg.segment.content.split())
            sentences = [piece.strip() for piece in content.split(". ") if piece.strip()]
            for sentence in sentences:
                add_bullet(sentence.rstrip("."))
                if len(bullets) >= max_points:
                    break

        trimmed = bullets[:max_points]
        for token_line in token_bullets:
            if token_line not in trimmed:
                trimmed.append(token_line)

        return trimmed

    def _logicify_directive(self, text: str) -> str:
        """Convert a directive fragment into compact machine-oriented wording."""
        cleaned = " ".join(text.strip().split())
        cleaned = re.sub(r"^[0-9]+[.)]\s*", "", cleaned)
        cleaned = cleaned.replace("must always", "ALWAYS")
        cleaned = cleaned.replace("must never", "NEVER")
        cleaned = cleaned.replace("always ", "ALWAYS ")
        cleaned = cleaned.replace("never ", "NEVER ")
        cleaned = cleaned.replace("do not ", "AVOID ")
        cleaned = cleaned.replace("don't ", "AVOID ")
        cleaned = cleaned.replace("should ", "PREFER ")
        cleaned = cleaned.replace("should not ", "AVOID ")
        cleaned = cleaned.replace("required", "REQUIRED")
        cleaned = cleaned.replace("must be", "REQUIRE")
        cleaned = cleaned.replace("needs to", "REQUIRE")
        cleaned = cleaned.replace("need to", "REQUIRE")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;")
        return cleaned

    def _render_mandatory_entry(self, seg: ScoredSegment) -> str:
        """Render one mandatory item in a compact directive-oriented form."""
        summary_lines = self._summarize_mandatory_content(seg, max_points=3)
        logic_lines = [self._logicify_directive(line) for line in summary_lines if line.strip()]

        parts = [f"### {seg.concept_title}"]

        scope = "ALWAYS" if seg.mandatory_scope == "always" else f"WHEN[{','.join(seg.mandatory_tags)}]"
        parts.append(f"SCOPE: {scope}")

        if seg.memory_identifiers:
            parts.append(f"KEYS: {', '.join(seg.memory_identifiers[:3])}")

        token_lines = [line for line in logic_lines if "VALIDATION_TOKEN:" in line]
        directive_lines = [line for line in logic_lines if "VALIDATION_TOKEN:" not in line]

        if directive_lines:
            parts.append(f"RULES: {' | '.join(directive_lines)}")
        for token_line in token_lines:
            parts.append(token_line)

        return "\n".join(parts)
    
    def to_context_string(self, max_tokens: int = 8000) -> str:
        """
        Format as context string for injection.

        Mandatory items always included first, then relevant items
        up to token budget.
        """
        parts = []
        token_estimate = 0
        HARD_CAP = min(max_tokens, 12000)  # Absolute ceiling regardless of caller request

        if not self.mandatory and self.acknowledgement_memories:
            parts.append(self.acknowledgement_context())

        # Always include mandatory (score 100) — but respect the budget
        if self.mandatory:
            parts.append("## MANDATORY CONTEXT (Must Follow)")
            parts.append("")
            parts.append(self.acknowledgement_context())
            parts.append(
                f"MANDATORY_ACTIVE: {self.mandatory_verification['always_on_count']} always-on | {self.mandatory_verification['conditional_count']} conditional"
            )
            parts.append(f"MANDATORY_CHECKLIST: {' | '.join(self.mandatory_titles)}")
            parts.append("")
            for seg in self.mandatory:
                entry = self._render_mandatory_entry(seg)
                entry_tokens = max(10, len(entry.split()) * 1.1)
                if token_estimate + entry_tokens > HARD_CAP:
                    parts.append(f"*[{len(self.mandatory) - len([p for p in parts if p.startswith('### ')])} more mandatory items truncated]*")
                    break
                parts.append(entry)
                parts.append("")
                token_estimate += entry_tokens

        # Add relevant up to budget
        if self.relevant:
            parts.append("## RELEVANT CONTEXT (Priority Order)")
            parts.append("")
            relevant_shown = 0
            for seg in sorted(self.relevant, key=lambda s: s.importance_score, reverse=True):
                entry_tokens = seg.segment.word_count * 1.3
                if token_estimate + entry_tokens > HARD_CAP:
                    remaining = len(self.relevant) - relevant_shown
                    parts.append(f"*[{remaining} more relevant items truncated]*")
                    break

                score_indicator = "🔴" if seg.importance_score >= 90 else "🟡" if seg.importance_score >= 70 else "⚪"
                entry = f"### {score_indicator} [{seg.importance_score}] {seg.concept_title}\n{seg.segment.content}"
                parts.append(entry)
                parts.append("")
                token_estimate += entry_tokens
                relevant_shown += 1

        self.total_tokens_estimate = int(min(token_estimate, HARD_CAP))
        return "\n".join(parts)
