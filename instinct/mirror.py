# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Conversation Mirror

Real-time conversation capture and segmentation with logical
breakpoints and overlap.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple

from .models import Segment, SegmentType, ScoredSegment, ProjectIdentity
from .scorer import SegmentScorer, ScoringConfig

logger = logging.getLogger(__name__)


@dataclass
class MirrorConfig:
    """Configuration for conversation mirroring."""
    
    # Minimum segment size (chars)
    min_segment_size: int = 100
    
    # Maximum segment size (chars)
    max_segment_size: int = 2000
    
    # Overlap between segments (chars)
    segment_overlap: int = 200
    
    # Auto-detect segment type from content
    auto_detect_type: bool = True
    
    # Score segments immediately on capture
    auto_score: bool = True
    
    # Store segments immediately on capture
    auto_store: bool = True
    
    # Callback when segment is captured
    on_segment: Optional[Callable[[ScoredSegment], None]] = None


class ConversationMirror:
    """
    Mirrors conversations in real-time, segmenting and scoring.
    
    Captures:
    - User messages
    - Assistant responses
    - Code blocks
    - Decisions and architectural notes
    
    Segments at logical breakpoints:
    - Topic changes
    - Code block boundaries
    - Decision points
    - Natural conversation breaks
    """
    
    def __init__(
        self,
        project: ProjectIdentity,
        store: Optional[Any] = None,
        config: Optional[MirrorConfig] = None,
        scorer_config: Optional[ScoringConfig] = None,
    ):
        self.project = project
        self.store = store
        self.config = config or MirrorConfig()
        self.scorer = SegmentScorer(project, scorer_config)
        
        # Conversation buffer
        self._buffer: List[Tuple[str, str, datetime]] = []  # (role, content, timestamp)
        self._segments: List[ScoredSegment] = []
    
    def add_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
    ) -> List[ScoredSegment]:
        """
        Add a message to the conversation mirror.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            timestamp: When the message occurred
        
        Returns:
            List of new segments created (may be empty if buffering)
        """
        timestamp = timestamp or datetime.now()
        self._buffer.append((role, content, timestamp))
        
        # Process buffer into segments
        new_segments = self._process_buffer()
        
        if new_segments:
            self._segments.extend(new_segments)
            
            # Auto-store if configured
            if self.config.auto_store and self.store:
                self.store.add_batch(new_segments)
            
            # Callback
            if self.config.on_segment:
                for seg in new_segments:
                    self.config.on_segment(seg)
        
        return new_segments
    
    def add_standing_instruction(self, instruction: str) -> ScoredSegment:
        """
        Add a standing instruction (automatically score 100).
        
        These are project-level directives that must always be followed.
        """
        segment = Segment(
            content=instruction,
            segment_type=SegmentType.STANDING_INSTRUCTION,
            source="manual",
            metadata={"mandatory_scope": "always", "mandatory_tags": []},
        )
        
        scored = ScoredSegment(
            segment=segment,
            importance_score=100,
            concept_title=f"Standing: {instruction[:50]}...",
            reasoning="Standing instruction - mandatory",
        )
        
        self._segments.append(scored)
        
        if self.store:
            self.store.add(scored)
        
        return scored
    
    def add_critical_constraint(self, constraint: str) -> ScoredSegment:
        """
        Add a critical constraint (automatically score 100).
        
        These are non-negotiable rules.
        """
        segment = Segment(
            content=constraint,
            segment_type=SegmentType.CRITICAL_CONSTRAINT,
            source="manual",
            metadata={"mandatory_scope": "always", "mandatory_tags": []},
        )
        
        scored = ScoredSegment(
            segment=segment,
            importance_score=100,
            concept_title=f"Constraint: {constraint[:50]}...",
            reasoning="Critical constraint - mandatory",
        )
        
        self._segments.append(scored)
        
        if self.store:
            self.store.add(scored)
        
        return scored
    
    def _process_buffer(self) -> List[ScoredSegment]:
        """Process the message buffer into segments."""
        if not self._buffer:
            return []
        
        new_segments = []
        
        # Combine recent messages
        combined = ""
        for role, content, ts in self._buffer:
            combined += f"\n\n**{role.upper()}:** {content}"
        
        # Check if we have enough content
        if len(combined) < self.config.min_segment_size:
            return []  # Keep buffering
        
        # Segment the combined content
        raw_segments = self._segment_content(combined.strip())
        
        for content, seg_type in raw_segments:
            segment = Segment(
                content=content,
                segment_type=seg_type,
                timestamp=self._buffer[-1][2],  # Use last message timestamp
                source="conversation",
            )
            
            if self.config.auto_score:
                scored = self.scorer.score(segment)
            else:
                scored = ScoredSegment(
                    segment=segment,
                    importance_score=50,  # Default score
                    concept_title=content[:60],
                )
            
            new_segments.append(scored)
        
        # Clear buffer (keep last message for overlap)
        if len(self._buffer) > 1:
            self._buffer = self._buffer[-1:]
        else:
            self._buffer = []
        
        return new_segments
    
    def _segment_content(self, content: str) -> List[Tuple[str, SegmentType]]:
        """
        Split content into logical segments.
        
        Returns list of (content, segment_type) tuples.
        """
        segments = []
        
        # Extract code blocks as separate segments
        code_pattern = r"```[\w]*\n(.*?)```"
        code_blocks = re.findall(code_pattern, content, re.DOTALL)
        
        # Remove code blocks from content for other segmentation
        remaining = re.sub(code_pattern, "\n[CODE_BLOCK]\n", content, flags=re.DOTALL)
        
        # Split remaining content at logical breakpoints
        breakpoints = [
            r"\n## ",  # Markdown headers
            r"\n### ",
            r"\n\*\*USER:\*\*",  # Role changes
            r"\n\*\*ASSISTANT:\*\*",
            r"\n---\n",  # Horizontal rules
            r"\n\n\n",  # Triple newlines
        ]
        
        # Simple split for now - can be enhanced
        parts = [remaining]
        for bp in breakpoints:
            new_parts = []
            for part in parts:
                split = re.split(bp, part)
                new_parts.extend(split)
            parts = new_parts
        
        # Process non-code segments
        code_idx = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if "[CODE_BLOCK]" in part:
                # Replace placeholder with actual code
                if code_idx < len(code_blocks):
                    code_content = code_blocks[code_idx].strip()
                    if len(code_content) >= 50:  # Only keep substantial code
                        segments.append((code_content, SegmentType.CODE))
                    code_idx += 1
                part = part.replace("[CODE_BLOCK]", "").strip()
            
            if part and len(part) >= self.config.min_segment_size:
                seg_type = self._detect_type(part) if self.config.auto_detect_type else SegmentType.CONTEXT
                
                # Split if too long
                if len(part) > self.config.max_segment_size:
                    for chunk in self._chunk_content(part):
                        segments.append((chunk, seg_type))
                else:
                    segments.append((part, seg_type))
        
        return segments
    
    def _detect_type(self, content: str) -> SegmentType:
        """Auto-detect segment type from content."""
        content_lower = content.lower()
        
        # Check for decision indicators
        decision_words = ["decided", "decision", "we will", "i'll implement", "approach:", "solution:"]
        if any(word in content_lower for word in decision_words):
            return SegmentType.ARCHITECTURAL_DECISION
        
        # Check for error/fix patterns
        error_words = ["error", "fix", "bug", "issue", "problem", "solved", "resolution"]
        if any(word in content_lower for word in error_words):
            return SegmentType.ERROR_RESOLUTION
        
        # Check for questions
        if content.strip().endswith("?"):
            return SegmentType.QUESTION
        
        # Check for implementation details
        impl_words = ["implement", "code", "function", "class", "method", "add", "create"]
        if any(word in content_lower for word in impl_words):
            return SegmentType.IMPLEMENTATION_DETAIL
        
        # Check for exploration
        explore_words = ["maybe", "could", "might", "idea", "thought", "wonder", "experiment"]
        if any(word in content_lower for word in explore_words):
            return SegmentType.EXPLORATION
        
        return SegmentType.CONTEXT
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split large content into overlapping chunks."""
        chunks = []
        max_size = self.config.max_segment_size
        overlap = self.config.segment_overlap
        
        start = 0
        while start < len(content):
            end = start + max_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence end near boundary
                for sep in [". ", ".\n", "\n\n", "\n"]:
                    break_point = content.rfind(sep, start + max_size - 200, end)
                    if break_point > start:
                        end = break_point + len(sep)
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def flush(self) -> List[ScoredSegment]:
        """Force processing of any buffered content."""
        if not self._buffer:
            return []
        
        # Temporarily lower minimum size
        old_min = self.config.min_segment_size
        self.config.min_segment_size = 1
        
        segments = self._process_buffer()
        
        self.config.min_segment_size = old_min
        self._buffer = []
        
        return segments
    
    @property
    def segments(self) -> List[ScoredSegment]:
        """Get all captured segments."""
        return self._segments.copy()
    
    def get_stats(self) -> dict:
        """Get mirror statistics."""
        scores = [s.importance_score for s in self._segments]
        return {
            "total_segments": len(self._segments),
            "mandatory_count": sum(1 for s in self._segments if s.is_mandatory),
            "high_priority_count": sum(1 for s in self._segments if s.is_high_priority),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "buffer_size": len(self._buffer),
        }
