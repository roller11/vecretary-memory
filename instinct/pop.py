# SPDX-License-Identifier: MPL-2.0
"""
Vecretary - Pop rules (v3, 2026-06-15)

The core priority lifecycle combines dormancy (`instinct.dormancy`), this pop
engine and dashboard (`dashboard.py`). This is not the separate premium Wisdom
product. The package name remains `instinct`; you can also refer to
`PopResult` as `WisdomPop` and `SegmentKPIs` as `WisdomKPIs` via the
aliases in `instinct/__init__.py`.

Resurrects dormant memories when retrieval hits them with multiple
keyword matches. The user's rule set:

  2 keyword matches  -> pop AND halve the decay clock
                        (active age / 2, so it freshens up along its curve)
  3 keyword matches  -> pop AND jump to the post-initial-drop point
                        (band-specific plateau/mid/tail)
  Frequency rule:     if a memory is popped 2+ times in an 8h active window,
                        the memory restarts at its original (initial_score)
                        (full reset), repeatedly as long as the cadence holds.
  Promotion rule:     if a memory with initial_score < 70 is accessed 4+ times
                        in a 24h active window, initial_score += 10, capped at
                        78 (for initial_score < 50) or 88 (for 50-69).

The "effective time elapsed" surfaced on the dashboard is the active time
since the most recent pop_anchor_at (or fallback to last_bumped_at, then
last_accessed, then segment.timestamp).
"""

import logging
from .active_time import clock_snapshot, elapsed_hours, selected_anchor, set_anchor, snapshot_elapsed_hours
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .models import ScoredSegment
    from .dormancy import DormancyConfig

logger = logging.getLogger(__name__)


# ============================================================
# Tunables (exposed for /api/decay/config and dashboard)
# ============================================================

# Keyword match count thresholds
POP_HALVE_THRESHOLD = 2   # >= this many matches -> halve decay clock
POP_JUMP_THRESHOLD = 3    # >= this many matches -> jump to post-drop plateau

# Frequency rule: full reset
FREQUENCY_POP_THRESHOLD = 2   # >= this many pops in 8h active window -> reset
FREQUENCY_WINDOW_ACTIVE_HOURS = 8.0

# Promotion rule: +10 to initial_score
PROMOTION_ACCESS_THRESHOLD = 4   # >= this many accesses in 24h active window
PROMOTION_WINDOW_ACTIVE_HOURS = 24.0
PROMOTION_AMOUNT = 10
# Caps: < 50 initial -> 78, 50-69 -> 88
PROMOTION_CAP_LOW = 78    # cap for memories that started below 50
PROMOTION_CAP_HIGH = 88   # cap for memories that started 50-69


@dataclass
class PopResult:
    """What happened during a pop attempt."""
    popped: bool
    rule_applied: str  # "none" | "halve" | "jump_to_plateau" | "frequency_reset" | "promoted"
    matched_keywords: int
    initial_score_before: int
    initial_score_after: int
    effective_score_before: float
    effective_score_after: float
    anchor_reset: bool
    message: str


def _now() -> datetime:
    return datetime.now()




def _recent_count_in_active_window(seg: "ScoredSegment", window_h: float,
                                    cfg: "DormancyConfig", now: datetime,
                                    log_key: str = "pop") -> int:
    """Count entries in seg.keyword_hits_log whose age < window_h active hours.

    Each log entry is {"ts": iso, "count": int, "kind": "pop"|"access"|...}.
    """
    count = 0
    for entry in seg.keyword_hits_log or []:
        if entry.get("kind") != log_key:
            continue
        ts_raw = entry.get("ts")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except (ValueError, TypeError):
            continue
        snapshot = entry.get("active_clock")
        if not isinstance(snapshot, dict):
            continue
        if snapshot.get("clock_id") != clock_snapshot(getattr(cfg, "persist_dir", None))["clock_id"]:
            continue
        if snapshot_elapsed_hours(snapshot, getattr(cfg, "persist_dir", None)) <= window_h:
            count += 1
    return count


def _log_event(seg: "ScoredSegment", kind: str, count: int, now: datetime,
                cap: int = 100, persist_dir=None) -> None:
    """Append an event to the keyword_hits_log, capped at the most recent N."""
    seg.keyword_hits_log.append({
        "ts": now.isoformat(),
        "kind": kind,
        "count": count,
        "active_clock": clock_snapshot(persist_dir),
    })
    # Cap the log size to avoid unbounded growth
    if len(seg.keyword_hits_log) > cap:
        seg.keyword_hits_log = seg.keyword_hits_log[-cap:]


def _band_for_score(score: int) -> str:
    if score >= 99:
        return "mandatory"
    if score >= 80:
        return "plateau"
    if score >= 50:
        return "mid"
    return "tail"


def _post_drop_score(score: int, cfg: "DormancyConfig") -> float:
    from .dormancy import post_drop_score
    return post_drop_score(score, cfg)


# ============================================================
# Pop rules
# ============================================================

def apply_pop(seg: "ScoredSegment", matched_keywords: int,
              cfg: "DormancyConfig", now: Optional[datetime] = None,
              query_id: Optional[str] = None) -> PopResult:
    """Apply pop rules based on how many keyword matches the retrieval got.

    Mutates `seg` in place and returns a PopResult describing what happened.
    """
    from . import dormancy as _dorm  # local to dodge circular import

    if now is None:
        now = _now()
    initial_before = seg.initial_score
    persist_dir = getattr(cfg, "persist_dir", None)
    eff_before = _dorm.current_effective_score(seg, now, cfg)

    # No pop on 0 or 1 keyword matches
    if matched_keywords < POP_HALVE_THRESHOLD:
        # Still log the access
        if matched_keywords > 0:
            _log_event(seg, "access", matched_keywords, now, persist_dir=persist_dir)
            seg.keyword_hits_total += matched_keywords
        return PopResult(
            popped=False, rule_applied="none", matched_keywords=matched_keywords,
            initial_score_before=initial_before, initial_score_after=initial_before,
            effective_score_before=eff_before, effective_score_after=eff_before,
            anchor_reset=False,
            message=f"Need {POP_HALVE_THRESHOLD}+ keyword matches to pop; got {matched_keywords}.",
        )

    # --- We have a pop event ---

    # Log the access (always)
    _log_event(seg, "access", matched_keywords, now, cap=200, persist_dir=persist_dir)
    seg.keyword_hits_total += matched_keywords

    # Mandatory segments (initial_score=100): the "pop" is just a bump.
    # Reset the bump anchor to now so the segment surfaces at full 100 for
    # the grace window. Never demote a mandatory.
    if seg.initial_score >= 99 or seg.importance_score >= 99:
        seg.importance_score = 100
        seg.last_bumped_at = now
        seg.bump_count += 1
        seg.last_popped_at = now
        seg.pop_anchor_at = now
        prior_pops_in_window = _recent_count_in_active_window(
            seg, FREQUENCY_WINDOW_ACTIVE_HOURS, cfg, now, log_key="pop"
        )
        seg.pop_recent_8h_active = prior_pops_in_window + 1
        set_anchor(seg, "last_bumped_at", persist_dir)
        set_anchor(seg, "last_popped_at", persist_dir)
        set_anchor(seg, "pop_anchor_at", persist_dir)
        _log_event(seg, "pop", matched_keywords, now, cap=200, persist_dir=persist_dir)
        eff_after = _dorm.current_effective_score(seg, now, cfg)
        return PopResult(
            popped=True, rule_applied="mandatory_bump",
            matched_keywords=matched_keywords,
            initial_score_before=initial_before,
            initial_score_after=seg.initial_score,
            effective_score_before=eff_before, effective_score_after=eff_after,
            anchor_reset=True,
            message=(f"Mandatory bump: reset to 100 (was {eff_before:.1f}). "
                     f"Surfaces every turn for the next grace window."),
        )

    # Promotion rule (applies alongside any pop rule).
    # Run it first so a subsequent frequency-reset uses the new initial_score.
    accesses_in_24h = _recent_count_in_active_window(
        seg, PROMOTION_WINDOW_ACTIVE_HOURS, cfg, now, log_key="access"
    )
    promotion_msg = ""
    if (accesses_in_24h >= PROMOTION_ACCESS_THRESHOLD
            and seg.initial_score < 70):
        if initial_before < 50:
            cap = PROMOTION_CAP_LOW
        else:
            cap = PROMOTION_CAP_HIGH
        old = seg.initial_score
        seg.initial_score = min(cap, seg.initial_score + PROMOTION_AMOUNT)
        if seg.initial_score != old:
            seg.promotions_applied += 1
            promotion_msg = f" Promotion: initial_score {initial_before} -> {seg.initial_score}."

    # Priority 1 (DOMINANT): 3+ keyword matches -> jump to post-initial-drop
    # point. Wins over the frequency rule per the user's spec.
    if matched_keywords >= POP_JUMP_THRESHOLD:
        # Jump to post-initial-drop point of the band
        band_name = _band_for_score(seg.initial_score)
        target = _post_drop_score(seg.initial_score, cfg)
        standard = seg.decay_profile == _dorm.DecayProfile.STANDARD.value
        if not standard:
            target = float(seg.importance_score)
        seg.importance_score = int(target)
        seg.last_bumped_at = now
        seg.bump_count += 1
        seg.last_popped_at = now
        seg.pop_anchor_at = now
        prior_pops_in_window = _recent_count_in_active_window(
            seg, FREQUENCY_WINDOW_ACTIVE_HOURS, cfg, now, log_key="pop"
        )
        seg.pop_recent_8h_active = prior_pops_in_window + 1
        set_anchor(seg, "last_bumped_at", persist_dir)
        set_anchor(seg, "last_popped_at", persist_dir)
        set_anchor(seg, "pop_anchor_at", persist_dir)
        if standard:
            for anchor_name in ("last_bumped_at", "pop_anchor_at"):
                set_anchor(seg, anchor_name, persist_dir, score=seg.initial_score)
                seg.active_time_anchors[anchor_name]["post_drop"] = True
        _log_event(seg, "pop", matched_keywords, now, cap=200, persist_dir=persist_dir)
        eff_after = _dorm.current_effective_score(seg, now, cfg)
        return PopResult(
            popped=True, rule_applied="jump_to_plateau",
            matched_keywords=matched_keywords,
            initial_score_before=initial_before,
            initial_score_after=seg.initial_score,
            effective_score_before=eff_before, effective_score_after=eff_after,
            anchor_reset=True,
            message=(f"3+ keyword pop: jumped to {target:.0f} "
                     f"(band={band_name})." + promotion_msg),
        )

    # Priority 2: 2-keyword pop. If already popped once in the last 8h active
    # window, the frequency rule says: full reset to initial_score.
    prior_pops_in_window = _recent_count_in_active_window(
        seg, FREQUENCY_WINDOW_ACTIVE_HOURS, cfg, now, log_key="pop"
    )
    if prior_pops_in_window >= 1:
        seg.importance_score = seg.initial_score
        seg.last_bumped_at = now
        seg.bump_count += 1
        seg.last_popped_at = now
        seg.pop_anchor_at = now
        seg.pop_recent_8h_active = prior_pops_in_window + 1
        set_anchor(seg, "last_bumped_at", persist_dir)
        set_anchor(seg, "last_popped_at", persist_dir)
        set_anchor(seg, "pop_anchor_at", persist_dir)
        _log_event(seg, "pop", matched_keywords, now, cap=200, persist_dir=persist_dir)
        eff_after = _dorm.current_effective_score(seg, now, cfg)
        return PopResult(
            popped=True, rule_applied="frequency_reset",
            matched_keywords=matched_keywords,
            initial_score_before=initial_before,
            initial_score_after=seg.initial_score,
            effective_score_before=eff_before, effective_score_after=eff_after,
            anchor_reset=True,
            message=(f"Frequency rule: popped {seg.pop_recent_8h_active}x in "
                     f"{FREQUENCY_WINDOW_ACTIVE_HOURS}h active. Reset to "
                     f"initial_score={seg.initial_score}." + promotion_msg),
        )

    # Priority 3: plain 2-keyword pop -> halve the decay clock.
    age_hours = _dorm._active_age(seg, now, cfg)
    basis_name = selected_anchor(seg, ("pop_anchor_at", "last_bumped_at", "last_accessed", "created"))
    basis = seg.active_time_anchors.get(basis_name, {})
    basis_score = basis.get("score", seg.importance_score)
    post_drop = basis.get("post_drop", False)
    halfway_seconds = clock_snapshot(persist_dir)["seconds"] - age_hours * 1800.0
    seg.last_bumped_at = now
    seg.bump_count += 1
    seg.last_popped_at = now
    seg.pop_anchor_at = now
    seg.pop_recent_8h_active = prior_pops_in_window + 1
    set_anchor(seg, "last_bumped_at", persist_dir, halfway_seconds, basis_score)
    set_anchor(seg, "last_popped_at", persist_dir)
    set_anchor(seg, "pop_anchor_at", persist_dir, halfway_seconds, basis_score)
    if post_drop:
        for anchor_name in ("last_bumped_at", "pop_anchor_at"):
            seg.active_time_anchors[anchor_name]["post_drop"] = True
    _log_event(seg, "pop", matched_keywords, now, cap=200, persist_dir=persist_dir)
    eff_after = _dorm.current_effective_score(seg, now, cfg)
    return PopResult(
        popped=True, rule_applied="halve",
        matched_keywords=matched_keywords,
        initial_score_before=initial_before,
        initial_score_after=seg.initial_score,
        effective_score_before=eff_before, effective_score_after=eff_after,
        anchor_reset=True,
        message=(f"2-keyword pop: decay clock halved. "
                 f"New bump anchor: {seg.last_bumped_at.isoformat()}."
                 + promotion_msg),
    )


# ============================================================
# KPIs (for the dashboard column)
# ============================================================

@dataclass
class SegmentKPIs:
    """All the live KPIs the dashboard wants to surface for one segment."""
    initial_score: int
    current_score: int
    overall_score: float
    initial_decay_path: str
    initial_semantic_score: float
    active_hours_elapsed: float
    effective_hours_since_pop: float
    effective_score: float
    decay_path_description: str
    keyword_hits_total: int
    keyword_hits_24h_active: int
    accesses_last_24h_active: int
    access_count_lifetime: int
    last_bumped_at: Optional[str]
    last_popped_at: Optional[str]
    pop_anchor_at: Optional[str]
    pop_recent_8h_active: int
    promotions_applied: int
    is_mandatory: bool
    decay_profile: str


def compute_kpis(seg: "ScoredSegment", cfg: "DormancyConfig",
                 now: Optional[datetime] = None) -> SegmentKPIs:
    """Compute all the KPIs the dashboard wants to display."""
    from . import dormancy as _dorm

    if now is None:
        now = _now()

    # Initial semantic score fallback: if we never recorded one, estimate
    # from the cosine distance stored in the embedding (1 - distance).
    # We don't have a stored distance here, so we use 0.0 unless set.
    initial_semantic = seg.initial_semantic_score

    # Active hours elapsed: time since the last meaningful action,
    # measured in SESSION-hours (the canon). Idle wall-clock time is
    # excluded; see instinct.session_timer.
    anchor = seg.last_bumped_at or seg.last_accessed or seg.segment.timestamp
    if anchor:
        name = selected_anchor(seg, ("last_bumped_at", "last_accessed", "created"))
        active_h = elapsed_hours(seg, name, getattr(cfg, "persist_dir", None))
    else:
        active_h = 0.0

    # Effective hours since pop: time since pop_anchor_at (or last_bumped_at, or now)
    pop_anchor = seg.pop_anchor_at or seg.last_bumped_at
    if pop_anchor:
        name = selected_anchor(seg, ("pop_anchor_at", "last_bumped_at", "created"))
        eff_h = elapsed_hours(seg, name, getattr(cfg, "persist_dir", None))
    else:
        eff_h = active_h  # No pop has ever happened; mirror active age

    eff_score = _dorm.current_effective_score(seg, now, cfg)

    # Keyword hits in 24h active window
    kw_24h = _recent_count_in_active_window(
        seg, 24.0, cfg, now, log_key="access"
    )
    accesses_24h = _recent_count_in_active_window(
        seg, 24.0, cfg, now, log_key="access"
    )

    decay_path = describe_decay_path(seg, cfg)

    return SegmentKPIs(
        initial_score=seg.initial_score,
        current_score=seg.importance_score,
        overall_score=(seg.initial_score + seg.importance_score) / 2.0,
        initial_decay_path=seg.initial_decay_path or decay_path,
        initial_semantic_score=initial_semantic,
        active_hours_elapsed=round(active_h, 2),
        effective_hours_since_pop=round(eff_h, 2),
        effective_score=round(eff_score, 2),
        decay_path_description=decay_path,
        keyword_hits_total=seg.keyword_hits_total,
        keyword_hits_24h_active=kw_24h,
        accesses_last_24h_active=accesses_24h,
        access_count_lifetime=seg.access_count,
        last_bumped_at=seg.last_bumped_at.isoformat() if seg.last_bumped_at else "",
        last_popped_at=seg.last_popped_at.isoformat() if seg.last_popped_at else "",
        pop_anchor_at=pop_anchor.isoformat() if pop_anchor else "",
        pop_recent_8h_active=seg.pop_recent_8h_active,
        promotions_applied=seg.promotions_applied,
        is_mandatory=seg.is_mandatory,
        decay_profile=seg.decay_profile,
    )


def describe_decay_path(seg: "ScoredSegment", cfg: "DormancyConfig") -> str:
    """Short human-readable description of which curve the segment is on."""
    s = seg.importance_score
    profile = seg.decay_profile
    if profile == "ephemeral":
        return f"ephemeral ({seg.ephemeral_phase}, {seg.ephemeral_turns_left} turns left)"
    if profile == "short_term_critical":
        hours = cfg.stc_hours_by_score.get(s, cfg.stc_default_active_hours)
        return f"STC {s} (fades in {hours}h active)"
    if s == 100:
        return "mandatory (100, surfaces every turn)"
    if s == 95:
        return f"tier 95 (drop to {cfg.tier_95_drop_to}, floor {cfg.tier_95_floor} over {cfg.tier_95_total_active_hours}h active)"
    if s == 94:
        return f"tier 94 (drop to {cfg.tier_94_drop_to}, floor {cfg.tier_94_floor} over {cfg.tier_94_total_active_hours}h active)"
    if s == 90:
        return f"tier 90 (drop to {cfg.tier_90_drop_to}, floor {cfg.tier_90_floor} over {cfg.tier_90_total_active_hours}h active)"
    if s in (91, 92, 93):
        return f"STC {s} (fades in {cfg.stc_hours_by_score.get(s, '?')}h active)"
    if s >= 80:
        return f"plateau (band floor {cfg.plateau.floor})"
    if s >= 50:
        return f"mid (band floor {cfg.mid.floor})"
    return f"tail (band floor {cfg.tail.floor})"
