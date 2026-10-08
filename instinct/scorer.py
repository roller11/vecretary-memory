# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Segment Scorer

Scores conversation segments based on importance relative to
project mission, vision, goals, and standing instructions.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .models import (
    Segment,
    SegmentType,
    ScoredSegment,
    ProjectIdentity,
    MANDATORY_TYPES,
)

logger = logging.getLogger(__name__)


# Scoring prompt template for LLM-based scoring
SCORING_PROMPT = '''You are a context importance scorer for the Instinct system.

## Project Identity
{project_context}

## Task
Score the following conversation segment on a scale of 0-100 based on:
1. Alignment with mission, vision, and goals
2. Architectural significance
3. Decision importance
4. Future reference value

Score Guidelines:
- 90-100: Critical decisions, architectural foundations, must-remember
- 70-89: Important context, significant decisions, high reference value
- 50-69: Useful context, moderate importance
- 30-49: Background information, low priority
- 0-29: Transient, exploration, minimal future value

## Segment to Score
Type: {segment_type}
Content:
{content}

## Response Format
Respond with ONLY a JSON object:
{{"score": <0-100>, "title": "<brief concept title>", "reasoning": "<one sentence why>"}}
'''


@dataclass
class ScoringConfig:
    """Configuration for the segment scorer."""
    
    # Use LLM for scoring (vs rule-based fallback)
    use_llm: bool = True
    
    # Model to use for scoring (should be fast/cheap)
    scoring_model: str = "llama3.2"
    
    # Boost score for segments containing priority keywords
    keyword_boost: int = 15
    
    # Minimum score for any segment
    min_score: int = 10
    
    # Score thresholds
    mandatory_threshold: int = 100
    high_priority_threshold: int = 70


class SegmentScorer:
    """
    Scores conversation segments by importance.
    
    Uses a combination of:
    1. Rule-based scoring for obvious cases
    2. LLM-based scoring for nuanced evaluation
    """
    
    def __init__(
        self,
        project: ProjectIdentity,
        config: Optional[ScoringConfig] = None,
    ):
        self.project = project
        self.config = config or ScoringConfig()
        self._llm_available = False
        self._check_llm()
    
    def _check_llm(self) -> None:
        """Check if LLM is available for scoring."""
        if not self.config.use_llm:
            return
        
        try:
            import ollama
            from .provider_privacy import validated_provider_url
            self._llm_client = ollama.Client(host=validated_provider_url(os.environ.get('OLLAMA_HOST', 'http://localhost:11434')), trust_env=False, follow_redirects=False)
            # Quick check if model exists
            self._llm_client.show(self.config.scoring_model)
            self._llm_available = True
            logger.info(f"LLM scoring enabled with {self.config.scoring_model}")
        except Exception as e:
            logger.warning("LLM unavailable for scoring; details withheld, using rules")
            self._llm_available = False
    
    def score(self, segment: Segment) -> ScoredSegment:
        """
        Score a segment and return ScoredSegment.
        
        Mandatory types (standing instructions, critical constraints)
        automatically get score 100.
        """
        # Mandatory types always score 100
        if segment.segment_type in MANDATORY_TYPES:
            return ScoredSegment(
                segment=segment,
                importance_score=100,
                concept_title=self._generate_title(segment),
                reasoning="Mandatory segment type",
            )
        
        # Try LLM scoring
        if self._llm_available and self.config.use_llm:
            try:
                return self._score_with_llm(segment)
            except Exception as e:
                logger.warning("LLM scoring failed; details withheld, using rules")
        
        # Rule-based fallback
        return self._score_with_rules(segment)
    
    def score_batch(self, segments: List[Segment]) -> List[ScoredSegment]:
        """Score multiple segments."""
        return [self.score(seg) for seg in segments]
    
    def _score_with_llm(self, segment: Segment) -> ScoredSegment:
        """Use LLM to score segment."""
        import ollama
        import json
        
        prompt = SCORING_PROMPT.format(
            project_context=self.project.to_scoring_context(),
            segment_type=segment.segment_type.value,
            content=segment.content[:2000],  # Limit content length
        )
        
        response = self._llm_client.generate(
            model=self.config.scoring_model,
            prompt=prompt,
            options={"temperature": 0.1},  # Low temp for consistency
        )
        
        # Parse JSON response
        text = response["response"].strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        
        data = json.loads(text)
        
        score = int(data.get("score", 50))
        title = data.get("title", self._generate_title(segment))
        reasoning = data.get("reasoning", "")
        
        # Apply keyword boost
        score = self._apply_keyword_boost(segment, score)
        
        return ScoredSegment(
            segment=segment,
            importance_score=score,
            concept_title=title,
            reasoning=reasoning,
        )
    
    def _score_with_rules(self, segment: Segment) -> ScoredSegment:
        """Rule-based scoring fallback."""
        score = self._calculate_rule_score(segment)
        score = self._apply_keyword_boost(segment, score)
        
        return ScoredSegment(
            segment=segment,
            importance_score=score,
            concept_title=self._generate_title(segment),
            reasoning="Rule-based scoring",
        )
    
    def _calculate_rule_score(self, segment: Segment) -> int:
        """Calculate base score using rules."""
        base_scores = {
            SegmentType.STANDING_INSTRUCTION: 100,
            SegmentType.CRITICAL_CONSTRAINT: 100,
            SegmentType.ARCHITECTURAL_DECISION: 85,
            SegmentType.ERROR_RESOLUTION: 75,
            SegmentType.IMPLEMENTATION_DETAIL: 60,
            SegmentType.USER_PREFERENCE: 70,
            SegmentType.CODE: 55,
            SegmentType.ANSWER: 50,
            SegmentType.QUESTION: 40,
            SegmentType.CONTEXT: 45,
            SegmentType.EXPLORATION: 30,
        }
        
        score = base_scores.get(segment.segment_type, 50)
        
        # Adjust based on content signals
        content_lower = segment.content.lower()
        
        # Boost for decision language
        decision_words = ["decided", "decision", "we will", "we must", "important", "critical", "always", "never"]
        for word in decision_words:
            if word in content_lower:
                score += 5
        
        # Boost for architectural terms
        arch_words = ["architecture", "pattern", "design", "structure", "interface", "protocol", "system"]
        for word in arch_words:
            if word in content_lower:
                score += 3
        
        # Reduce for hedging language
        hedge_words = ["maybe", "perhaps", "might", "could try", "not sure", "experiment"]
        for word in hedge_words:
            if word in content_lower:
                score -= 5
        
        return max(self.config.min_score, min(99, score))  # Cap at 99, not 100
    
    def _apply_keyword_boost(self, segment: Segment, score: int) -> int:
        """Boost score if priority keywords are present."""
        if not self.project.priority_keywords:
            return score
        
        content_lower = segment.content.lower()
        for keyword in self.project.priority_keywords:
            if keyword.lower() in content_lower:
                score += self.config.keyword_boost
                break  # Only apply boost once
        
        return min(99, score)  # Don't boost to 100
    
    def _generate_title(self, segment: Segment) -> str:
        """Generate a brief title for the segment."""
        # Use first line or first N chars
        first_line = segment.content.split("\n")[0].strip()
        if len(first_line) <= 60:
            return first_line
        
        # Truncate and clean
        title = first_line[:57].rsplit(" ", 1)[0] + "..."
        return title


def score_segment(
    content: str,
    segment_type: SegmentType,
    project: ProjectIdentity,
    config: Optional[ScoringConfig] = None,
) -> ScoredSegment:
    """
    Convenience function to score a single segment.
    
    Args:
        content: The segment text content
        segment_type: Type classification
        project: Project identity for scoring context
        config: Optional scoring configuration
    
    Returns:
        ScoredSegment with importance score
    """
    segment = Segment(content=content, segment_type=segment_type)
    scorer = SegmentScorer(project, config)
    return scorer.score(segment)
