# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Decay and Pop Modifiers

Retrieval uses the same configured active-session curves as the dashboard.
Mandatory retrieval remains protected; nonmandatory memories age only from
their own persisted accumulator anchors. Wall-clock timestamps are audit data.

Legacy pop windows are measured in active-session days, not calendar days.
- Otherwise drops back to decayed value
"""

from datetime import datetime
from typing import TYPE_CHECKING

from .modifiers import ScoreModifier, QueryContext, ScoringResult

if TYPE_CHECKING:
    from .models import ScoredSegment


class DecayModifier(ScoreModifier):
    """
    Time-based decay modifier.
    
    Adjusts effective scores based on time since creation,
    using score-dependent decay curves.
    """
    
    def __init__(self, enabled: bool = True):
        super().__init__(name="decay", priority=10, enabled=enabled)
    
    def should_skip(self, segment: "ScoredSegment", context: QueryContext) -> bool:
        """Skip if disabled or in long-term mode."""
        if not self.enabled:
            return True
        if context.long_term_mode:
            return True
        return False
    
    def modify(
        self, 
        segment: "ScoredSegment", 
        context: QueryContext, 
        result: ScoringResult
    ) -> ScoringResult:
        """Apply time-based decay to the score."""
        
        # Immortal scores don't decay
        if result.is_immortal:
            return result
        
        from .dormancy import config_for, current_effective_score
        cfg = config_for(context.persist_dir)
        decayed_score = current_effective_score(segment, context.now, cfg)
        
        if decayed_score != result.effective_score:
            result.add_adjustment(
                self.name,
                decayed_score,
                "Canonical active-session decay"
            )
        
        return result
    


class PopModifier(ScoreModifier):
    """
    Pop modifier for dormant memory resurrection.
    
    When a memory that hasn't been accessed in >3 months gets retrieved,
    it "pops" back to a higher effective score. To make the pop permanent,
    the memory must be accessed 2 times within a week.
    """
    
    # Configuration
    DORMANT_THRESHOLD_DAYS = 90  # 3 months
    POP_BOOST = 30  # How much to boost when popping
    STICK_WINDOW_DAYS = 7  # Must access 2x within this window
    STICK_ACCESSES = 2  # Accesses needed to stick
    
    def __init__(self, enabled: bool = True, persist_dir=None):
        super().__init__(name="pop", priority=20, enabled=enabled)
        self.persist_dir = persist_dir
    
    def should_skip(self, segment: "ScoredSegment", context: QueryContext) -> bool:
        """Skip if disabled or in long-term mode."""
        if not self.enabled:
            return True
        if context.long_term_mode:
            return True
        return False
    
    def modify(
        self, 
        segment: "ScoredSegment", 
        context: QueryContext, 
        result: ScoringResult
    ) -> ScoringResult:
        """
        Apply pop mechanic.
        
        States:
        1. Memory is dormant (no access >3mo) - gets a boost when accessed
        2. Memory has popped - checking if it will stick
        3. Memory stuck - stays at boosted level
        4. Memory fell back - pop expired, back to decayed value
        """
        
        # Immortal memories don't need pop
        if result.is_immortal:
            return result
        
        now = context.now
        last_accessed = segment.last_accessed
        popped_at = segment.popped_at
        accesses_since_pop = segment.accesses_since_pop
        
        # Check if memory is currently in a popped state
        if popped_at:
            from .active_time import elapsed_hours
            days_since_pop = elapsed_hours(segment, "popped_at", context.persist_dir or self.persist_dir) / 24.0
            
            if days_since_pop <= self.STICK_WINDOW_DAYS:
                # Within stick window - check if we have enough accesses
                if accesses_since_pop >= self.STICK_ACCESSES:
                    # It stuck! Boost the score
                    boosted = min(100, result.effective_score + self.POP_BOOST)
                    result.add_adjustment(
                        self.name,
                        boosted,
                        f"Memory stuck! {accesses_since_pop} accesses in {days_since_pop}d"
                    )
                else:
                    # Still trying to stick - temporary boost
                    boosted = min(100, result.effective_score + (self.POP_BOOST * 0.5))
                    result.add_adjustment(
                        self.name,
                        boosted,
                        f"Popped, needs {self.STICK_ACCESSES - accesses_since_pop} more access in {self.STICK_WINDOW_DAYS - days_since_pop}d"
                    )
            else:
                # Pop window expired - did it stick?
                if accesses_since_pop >= self.STICK_ACCESSES:
                    # It stuck before window closed
                    boosted = min(100, result.effective_score + self.POP_BOOST)
                    result.add_adjustment(
                        self.name,
                        boosted,
                        f"Memory stuck (confirmed)"
                    )
                else:
                    # Fell back - no adjustment (decayed value stands)
                    result.add_adjustment(
                        self.name,
                        result.effective_score,
                        f"Pop expired, fell back (only {accesses_since_pop}/{self.STICK_ACCESSES} accesses)"
                    )
        
        elif last_accessed:
            # Has been accessed before - check if dormant
            from .active_time import elapsed_hours
            days_since_access = elapsed_hours(segment, "last_accessed", context.persist_dir or self.persist_dir) / 24.0
            
            if days_since_access >= self.DORMANT_THRESHOLD_DAYS:
                # Dormant memory being accessed - it pops!
                # The actual pop state is set when access is recorded
                boosted = min(100, result.effective_score + (self.POP_BOOST * 0.5))
                result.add_adjustment(
                    self.name,
                    boosted,
                    f"Dormant memory popped! ({days_since_access}d since last access)"
                )
        
        # If never accessed, no pop mechanics apply yet
        
        return result
    
    def should_pop(self, segment: "ScoredSegment", now: datetime) -> bool:
        """
        Check if accessing this segment should trigger a pop.
        
        Called by store when recording access.
        """
        if segment.importance_score >= 99:
            return False
        
        if not segment.last_accessed:
            return False
        
        from .active_time import elapsed_hours
        days_since_access = elapsed_hours(segment, "last_accessed", self.persist_dir) / 24.0
        return days_since_access >= self.DORMANT_THRESHOLD_DAYS
    
    def should_clear_pop(self, segment: "ScoredSegment", now: datetime) -> bool:
        """
        Check if pop state should be cleared (window expired without sticking).
        
        Called by store during maintenance.
        """
        if not segment.popped_at:
            return False
        
        from .active_time import elapsed_hours
        days_since_pop = elapsed_hours(segment, "popped_at", self.persist_dir) / 24.0
        if days_since_pop > self.STICK_WINDOW_DAYS:
            return segment.accesses_since_pop < self.STICK_ACCESSES
        
        return False


# ============================================================
# Utility Functions
# ============================================================

def calculate_effective_score(
    segment: "ScoredSegment", 
    now: datetime | None = None,
    long_term: bool = False,
    persist_dir=None,
) -> float:
    """
    Quick utility to calculate effective score for a segment.
    
    Args:
        segment: The segment to score
        now: Current time (defaults to now)
        long_term: If True, return original score (no decay)
    
    Returns:
        Effective score as float
    """
    if long_term:
        return float(segment.importance_score)
    
    context = QueryContext(
        now=now or datetime.now(),
        long_term_mode=long_term,
        persist_dir=persist_dir,
    )
    
    decay = DecayModifier()
    pop = PopModifier()
    
    result = ScoringResult(
        segment_id=segment.id,
        original_score=segment.importance_score,
        effective_score=float(segment.importance_score),
    )
    
    result = decay.modify(segment, context, result)
    result = pop.modify(segment, context, result)
    
    return result.effective_score


def describe_decay_curve(score: int) -> str:
    """
    Describe the decay curve for a given score.
    
    Useful for diagnostics and understanding how a score will decay.
    """
    from .dormancy import describe_decay
    return describe_decay(score)
