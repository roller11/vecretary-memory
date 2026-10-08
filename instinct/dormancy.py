# SPDX-License-Identifier: MPL-2.0
"""
Instinct - Dormancy System (v2)  --  the "Wisdom" decay engine

Score-driven decay + bump-back-to-surface, gated by SESSION time (not wall clock,
not the legacy "real_hours * 9/24" conversion). The session clock is
`instinct.session_timer`: a real timer that starts on the first tool call
after a long idle, stays alive as long as tool calls keep coming, and
expires after `session_idle_minutes` (default 10) of monotonic inactivity.
Idle periods are discarded -- they do not age memories.

This module is part of the Wisdom system: dormancy (this file) + pop
(`instinct.pop`) + dashboard (`dashboard.py`) together form the wisdom
module. The package name remains `instinct`; you can also refer to
`DormancyConfig` as `WisdomConfig` via the alias in `instinct/__init__.py`.

Time model
==========
The system tracks "session hours" -- time elapsed during active sessions.
A session begins on the first MCP tool call after a long idle and
expires after `session_idle_minutes` of silence (default 10). The
canonical clock lives in `instinct.session_timer`; all timed effects in
this module use the per-memory snapshots in `instinct.active_time` and the
DormancyConfig knob `session_idle_minutes`. There is intentionally no
wall-clock-to-session-hours conversion factor: the legacy
`active_hours_per_real_day` field has been removed.

Score-band model
================
4 default bands. Each has 5 tunables. All live in DormancyConfig so the
dashboard can edit them at runtime.

  mandatory  (99-100)  surface -> dormant_floor on idle; bump resets
  plateau   (80-89)   quick drop, then slow drift to plateau_floor
  mid       (50-79)   quick drop, then exponential half-life to mid_floor
  tail      (0-49)    pure exponential half-life to tail_floor

Scores 90/94/95 retain their tier overrides; other standard scores 91-98
use the plateau fallback. Short-term-critical and ephemeral profiles remain
separate overrides.

Per-band tunables (all in active hours, all in pts):
  initial_drop_pts            - quick drop amount
  initial_drop_active_hours   - time to complete the quick drop
  decay_function              - "linear" | "exponential" | "half_life"
  floor                       - minimum score in this band
  bin_retention_active_hours  - time at floor before fully retired (score = 0)

Short-term critical profile
===========================
decay_profile = "short_term_critical" - same shape, but after
`short_term_critical_active_hours` of no access, drops to `stc_floor`.

Persistence
===========
DormancyConfig persists to dormancy_config.json. The active accumulator and
clock identity persist separately to session_state.json; per-memory snapshots
live in memory metadata. Configuration and clock writes are atomic.
"""
from __future__ import annotations

import json
import math
import os
from .private_storage import private_binary_reader, write_private_text, PrivateStorageError
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ScoredSegment

# Session clock is the canonical time source for all timed effects in
# this module. Importing at module level would create a cycle through
# __init__ in some load orders, so we import functions locally where
# they are used and rely on the explicit function calls below.
from . import session_timer as _st
from .active_time import capture_anchors, elapsed_hours, selected_anchor


CONFIG_FILENAME = "dormancy_config.json"


# ============================================================
# Decay function shape
# ============================================================

class DecayFunction(str, Enum):
    LINEAR = "linear"          # lose (pts - floor) / (active_hours_to_floor)
    EXPONENTIAL = "exponential"  # half-life decay toward floor
    HALF_LIFE = "half_life"    # alias for exponential, kept for clarity


# ============================================================
# Per-band configuration
# ============================================================

@dataclass
class BandConfig:
    """The 5 tunables for one decay band. Time is in active hours."""
    initial_drop_pts: float
    initial_drop_active_hours: float
    decay_function: str            # DecayFunction value
    floor: int
    bin_retention_active_hours: float
    # For exponential: half-life in active hours (after initial drop)
    post_drop_half_life_active_hours: float = 0.0
    # For linear: total time in active hours to reach the floor
    post_drop_to_floor_active_hours: float = 0.0


# ============================================================
# Top-level config
# ============================================================

@dataclass
class DormancyConfig:
    """Runtime-mutable dormancy configuration."""

    # --- session clock ---
    # The session clock is owned by `instinct.session_timer` and lives
    # in `<persist_dir>/session_state.json`. This field is the idle
    # timeout that decides when a session expires (in minutes of
    # wall-clock silence with no MCP tool call). The session clock
    # itself is the canon for all timed effects in the engine; this
    # config knob is the only user-tunable part of it.
    session_idle_minutes: float = 10.0       # minutes of silence that close a session
    # NOTE: the legacy `active_hours_per_real_day` field has been
    # removed. There is no longer any wall-clock-to-session conversion;
    # idle time simply does not contribute to aging.

    # --- bump (mandatory) tunables ---
    bump_grace_active_hours: float = 0.05      # ~3 minutes active
    bump_base_active_hours: float = 0.5        # 30min active for score 100
    bump_decay_per_10: float = 1.6
    bump_min_score: int = 60

    # --- mandatory band (99-100) ---
    mandatory: BandConfig = field(default_factory=lambda: BandConfig(
        initial_drop_pts=0.0,                   # mandatory drops to dormant_floor
        initial_drop_active_hours=0.0,          # on first idle moment
        decay_function=DecayFunction.LINEAR.value,
        floor=78,                               # mandatory_floor
        bin_retention_active_hours=24.0,        # 24h active at 78 -> retired
        post_drop_to_floor_active_hours=24.0,
    ))
    # special: mandatory dormancy is "dormant_floor" (default 89), not the
    # band floor. The band floor is the bottom of the recycle bin.

    dormant_floor: int = 89                    # mandatory score while idle

    # --- plateau band (80-89; upper standard-score fallback) ---
    plateau: BandConfig = field(default_factory=lambda: BandConfig(
        initial_drop_pts=10.0,
        initial_drop_active_hours=14.0,         # ~1.5 work-days
        decay_function=DecayFunction.LINEAR.value,
        floor=60,
        bin_retention_active_hours=240.0,      # 240h active at 70 -> 0 (~27 work-days)
        post_drop_to_floor_active_hours=240.0,
    ))

    # --- mid band (50-79) ---
    mid: BandConfig = field(default_factory=lambda: BandConfig(
        initial_drop_pts=10.0,
        initial_drop_active_hours=4.0,
        decay_function=DecayFunction.EXPONENTIAL.value,
        floor=20,
        bin_retention_active_hours=80.0,
        post_drop_half_life_active_hours=20.0,
    ))

    # --- tail band (0-49) ---
    tail: BandConfig = field(default_factory=lambda: BandConfig(
        initial_drop_pts=0.0,
        initial_drop_active_hours=0.0,
        decay_function=DecayFunction.HALF_LIFE.value,
        floor=0,
        bin_retention_active_hours=0.0,
        post_drop_half_life_active_hours=8.0,
    ))

    # --- short-term critical profile (legacy, used when STC flag is set) ---
    short_term_critical_active_hours: float = 3.0   # default; overridden by stc_hours_by_score
    short_term_critical_fade_floor: int = 49

    # --- STC hours-by-score (91=session, 92=next-session, 93=multi-day) ---
    stc_hours_by_score: Dict[int, float] = field(default_factory=lambda: {
        91: 3.0,
        92: 24.0,
        93: 72.0,
    })
    stc_default_active_hours: float = 3.0     # used when score isn't 91/92/93

    # --- long-term tiers (90, 94, 95) ---
    tier_90_drop_to: int = 89                  # quick drop target
    tier_90_floor: int = 60
    tier_90_total_active_hours: float = 200.0  # total time to reach floor

    tier_94_drop_to: int = 88
    tier_94_floor: int = 29
    tier_94_total_active_hours: float = 500.0

    tier_95_drop_to: int = 88
    tier_95_floor: int = 60
    tier_95_total_active_hours: float = 1000.0

    tier_90_drop_active_hours: float = 4.0
    tier_94_drop_active_hours: float = 4.0
    tier_95_drop_active_hours: float = 4.0

    # --- ephemeral (turn-based recall) ---
    ephemeral_surface_turns: int = 6
    ephemeral_fade_turns: int = 6

    # --- retriever gate ---
    min_keyword_hits_for_faded_stc: int = 3   # faded STC needs 3+ kw hits to surface

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # `persist_dir` is a runtime-only handle for the on-disk
        # session-state file; it is not part of the saved config and
        # would be misleading to serialize.
        d.pop("persist_dir", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DormancyConfig":
        """Rebuild from JSON, dropping legacy clock fields silently.

        Pre-v2 configs carried `active_hours_per_real_day` and a
        `last_active_at` string. Both are now obsolete -- the canonical
        clock is the session timer. We drop them on load without error
        so existing `dormancy_config.json` files migrate cleanly.
        """
        # Legacy fields to ignore during migration.
        legacy_ignore = {
            "active_hours_per_real_day",
            "last_active_at",
            "persist_dir",  # runtime-only; not a dataclass field
        }
        # Bands need to be reconstructed as BandConfig, not dict
        valid = {f for f in cls.__dataclass_fields__.keys()}
        kwargs = {}
        for k, v in d.items():
            if k in legacy_ignore:
                continue
            if k not in valid:
                continue
            if k in ("mandatory", "plateau", "mid", "tail"):
                kwargs[k] = BandConfig(**v) if isinstance(v, dict) else v
            elif k == "stc_hours_by_score":
                kwargs[k] = {int(score): hours for score, hours in v.items()}
            else:
                kwargs[k] = v
        return cls(**kwargs)


# ============================================================
# Persistence
# ============================================================

class _ConfigStore:
    def __init__(self, persist_dir: str | os.PathLike):
        self.path = Path(persist_dir) / CONFIG_FILENAME
        self.persist_dir = str(persist_dir)
        self._lock = RLock()
        self._cached: Optional[DormancyConfig] = None

    def load(self) -> DormancyConfig:
        with self._lock:
            if self._cached is not None:
                return self._cached
            if not self.path.exists() and not self.path.is_symlink():
                self._cached = DormancyConfig()
                self._cached.persist_dir = self.persist_dir
                return self._cached
            try:
                with private_binary_reader(self.path) as f:
                    data = json.load(f)
                self._cached = DormancyConfig.from_dict(data)
            except PrivateStorageError:
                raise
            except Exception:
                self._cached = DormancyConfig()
            self._cached.persist_dir = self.persist_dir
            return self._cached

    def save(self, cfg: DormancyConfig) -> None:
        with self._lock:
            write_private_text(self.path, json.dumps(cfg.to_dict(), indent=2))
            self._cached = cfg

    def update(self, **changes) -> DormancyConfig:
        with self._lock:
            cfg = self.load()
            for k, v in changes.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            self.save(cfg)
            return cfg

    def update_band(self, band_name: str, **changes) -> DormancyConfig:
        with self._lock:
            cfg = self.load()
            band = getattr(cfg, band_name, None)
            if band is None:
                raise ValueError(f"unknown band: {band_name}")
            for k, v in changes.items():
                if hasattr(band, k):
                    setattr(band, k, v)
            self.save(cfg)
            return cfg


_store: Optional[_ConfigStore] = None


def configure(persist_dir: str | os.PathLike) -> DormancyConfig:
    global _store
    _store = _ConfigStore(persist_dir)
    return _store.load()


def _cfg() -> DormancyConfig:
    from .service_binding import bound_config
    isolated = bound_config()
    if isolated is not None:
        return isolated.load()
    global _store
    if _store is None:
        from . import _default_persist_dir  # type: ignore
        configure(_default_persist_dir())
    assert _store is not None
    return _store.load()


def config_for(persist_dir=None) -> DormancyConfig:
    return _cfg() if persist_dir is None else _ConfigStore(persist_dir).load()


# ============================================================
# Session-time clock (delegated to instinct.session_timer)
# ============================================================

def now_active(cfg: DormancyConfig) -> datetime:
    """Return the system 'now' in session time.

    The session clock lives in `instinct.session_timer`. "Now" in
    session time is represented by the session anchor plus accumulated
    session-hours. This compatibility datetime must not be subtracted from
    real memory timestamps; elapsed active time requires accumulator snapshots.

    The legacy implementation multiplied wall-clock hours by a
    conversion factor; the session timer does not. Idle periods are
    simply absent from this clock.
    """
    return _st.session_now(getattr(cfg, "persist_dir", None))


def tick_active_clock(cfg: DormancyConfig) -> DormancyConfig:
    """Advance the persisted session clock. Call this from a tick.

    This is a thin shim over `session_timer.tick_session()` that reads
    `cfg.session_idle_minutes` so the dashboard can tune the idle
    window without code changes. The session state itself is owned by
    `instinct.session_timer`, not by DormancyConfig; this function does
    not modify DormancyConfig.
    """
    _st.tick_session(getattr(cfg, "persist_dir", None), idle_minutes=cfg.session_idle_minutes)
    return cfg


# ============================================================
# Profile enum
# ============================================================

class DecayProfile(str, Enum):
    STANDARD = "standard"
    SHORT_TERM_CRITICAL = "short_term_critical"
    EPHEMERAL = "ephemeral"


# ============================================================
# Active-hours delta helper
# ============================================================

def _active_age(seg: "ScoredSegment", now: datetime, cfg: DormancyConfig) -> float:
    """Age of the segment in session-hours.

    Reads from `instinct.session_timer` (the canon). The session clock
    only advances while a session is alive, so the result is bounded by
    the total session time accumulated since the engine started using
    the new clock -- idle wall-clock hours contribute zero.

    Anchor selection (in order): `pop_anchor_at`, `last_bumped_at`,
    `last_accessed`, `segment.timestamp`. Each anchor carries its own
    accumulator snapshot; calendar timestamp subtraction is never used.
    """
    name = selected_anchor(seg, ("pop_anchor_at", "last_bumped_at", "last_accessed", "created"))
    return elapsed_hours(seg, name, getattr(cfg, "persist_dir", None))


# ============================================================
# Band selection
# ============================================================

def _band_for(score: int) -> str:
    if score >= 99: return "mandatory"
    if score >= 80: return "plateau"
    if score >= 50: return "mid"
    return "tail"


def _tier_for(score: int) -> Optional[str]:
    if score in (90, 94, 95):
        return str(score)
    return None


def post_drop_score(score: int, cfg: DormancyConfig) -> float:
    if score >= 99:
        return 100.0
    tier = _tier_for(score)
    if tier:
        return float(getattr(cfg, f"tier_{tier}_drop_to"))
    band = getattr(cfg, _band_for(score))
    drop = band.initial_drop_pts if band.initial_drop_active_hours > 0 else 0.0
    return min(float(score), max(0.0, float(band.floor), score - drop))


# ============================================================
# Tier decay (90, 94, 95)
# ============================================================

def _tier_decay(seg: "ScoredSegment", original: int, age_active: float,
                cfg: DormancyConfig, tier: str) -> float:
    if tier == "90":
        drop_to = cfg.tier_90_drop_to
        floor = cfg.tier_90_floor
        total = cfg.tier_90_total_active_hours
        drop_h = cfg.tier_90_drop_active_hours
    elif tier == "94":
        drop_to = cfg.tier_94_drop_to
        floor = cfg.tier_94_floor
        total = cfg.tier_94_total_active_hours
        drop_h = cfg.tier_94_drop_active_hours
    else:  # 95
        drop_to = cfg.tier_95_drop_to
        floor = cfg.tier_95_floor
        total = cfg.tier_95_total_active_hours
        drop_h = cfg.tier_95_drop_active_hours

    # Phase 1: linear drop from original to drop_to over drop_h hours
    if age_active <= drop_h:
        if drop_h <= 0:
            return float(drop_to)
        progress = age_active / drop_h
        return float(original) - (float(original - drop_to) * progress)

    # Phase 2: slow linear to floor over (total - drop_h) hours
    post_age = age_active - drop_h
    span = max(0.1, total - drop_h)
    rate = (float(drop_to) - float(floor)) / span
    v = float(drop_to) - (rate * post_age)
    return max(float(floor), v)


# ============================================================
# Ephemeral state machine
# ============================================================

def _ephemeral_effective_score(seg: "ScoredSegment", now: datetime) -> float:
    """Turn-based state: surface (high) -> fading (low) -> expired (0).

    The phase transitions are driven by tick_ephemeral_turn(), which is
    called by the retriever on every turn. Here we just read the phase.
    """
    phase = getattr(seg, "ephemeral_phase", "surface")
    if phase == "surface":
        return float(seg.importance_score)
    if phase == "fading":
        # Sit at the dormant_floor
        return float(_cfg().dormant_floor)
    return 0.0


def tick_ephemeral_turn(seg: "ScoredSegment", now: datetime) -> "ScoredSegment":
    """Advance the ephemeral state machine by one turn. Idempotent if the
    turn counter has already been decremented elsewhere."""
    cfg = _cfg()
    if seg.decay_profile != DecayProfile.EPHEMERAL.value:
        return seg

    if seg.ephemeral_phase == "surface":
        if seg.ephemeral_turns_left <= 0:
            seg.ephemeral_phase = "fading"
            seg.ephemeral_turns_left = cfg.ephemeral_fade_turns
        else:
            seg.ephemeral_turns_left -= 1
            if seg.ephemeral_turns_left <= 0:
                seg.ephemeral_phase = "fading"
                seg.ephemeral_turns_left = cfg.ephemeral_fade_turns
    elif seg.ephemeral_phase == "fading":
        if seg.ephemeral_turns_left <= 0:
            seg.ephemeral_phase = "expired"
            seg.ephemeral_turns_left = 0
        else:
            seg.ephemeral_turns_left -= 1
            if seg.ephemeral_turns_left <= 0:
                seg.ephemeral_phase = "expired"
                seg.ephemeral_turns_left = 0
    return seg


def start_ephemeral(seg: "ScoredSegment", cfg: DormancyConfig) -> "ScoredSegment":
    """Initialize the ephemeral state on a fresh segment."""
    seg.decay_profile = DecayProfile.EPHEMERAL.value
    seg.ephemeral_phase = "surface"
    seg.ephemeral_turns_left = cfg.ephemeral_surface_turns
    seg.importance_score = max(seg.importance_score, 50)  # keep visible
    return seg


# ============================================================
# Core: current effective score (in active time)
# ============================================================

def current_effective_score(seg: "ScoredSegment", now: datetime,
                            cfg: Optional[DormancyConfig] = None) -> float:
    """The single source of truth: what score should this segment have
    right now, given active time elapsed since last action.

    Combines: bump grace (mandatory), per-band decay, and STC profile.
    """
    cfg = cfg if cfg is not None else _cfg()
    capture_anchors(seg, getattr(cfg, "persist_dir", None))
    original = seg.importance_score

    # Bump grace: a recently-bumped mandatory stays at 100
    if original >= 99:
        last_bump = seg.last_bumped_at
        if last_bump and elapsed_hours(seg, "last_bumped_at", getattr(cfg, "persist_dir", None)) <= cfg.bump_grace_active_hours:
            return float(original)

    # Short-term critical: drop to fade floor if unused long enough
    profile = getattr(seg, "decay_profile", DecayProfile.STANDARD.value)
    if profile == DecayProfile.SHORT_TERM_CRITICAL.value:
        # Hours map by score: 91=session, 92=next-session, 93=multi-day
        hours_for_score = cfg.stc_hours_by_score.get(seg.importance_score)
        if hours_for_score is None:
            hours_for_score = cfg.stc_default_active_hours
        last_use = seg.last_accessed or seg.last_bumped_at or seg.segment.timestamp
        if last_use:
            # Session-hours since last use; idle wall-clock time is excluded.
            name = selected_anchor(seg, ("last_accessed", "last_bumped_at", "created"))
            active_hours_unused = elapsed_hours(seg, name, getattr(cfg, "persist_dir", None))
            if active_hours_unused >= hours_for_score:
                # Faded: return 0 so the retriever gate kicks in (retriever
                # checks the 3-keyword minimum before surfacing).
                return 0.0
        # Still in window: behave like a normal band-decay segment
        return _band_decay(seg, seg.importance_score, now, cfg)

    # Ephemeral: turn-based surface -> fade -> expired state machine
    if profile == DecayProfile.EPHEMERAL.value:
        return _ephemeral_effective_score(seg, now)

    # Standard band decay
    return _band_decay(seg, original, now, cfg)


def _band_decay(seg: "ScoredSegment", original: int, now: datetime,
                cfg: DormancyConfig, projected_hours: Optional[float] = None) -> float:
    age_active = _active_age(seg, now, cfg) if projected_hours is None else projected_hours
    name = selected_anchor(seg, ("pop_anchor_at", "last_bumped_at", "last_accessed", "created"))
    anchor = seg.active_time_anchors.get(name, {})
    original = anchor.get("score", original)
    post_drop = anchor.get("post_drop", False)
    band_name = _band_for(original)
    band: BandConfig = getattr(cfg, band_name)

    # Long-term tier overrides (90, 94, 95)
    tier = _tier_for(original)
    if tier:
        if post_drop:
            age_active += getattr(cfg, f"tier_{tier}_drop_active_hours")
        return _tier_decay(seg, original, age_active, cfg, tier)

    # Special case for mandatory: drop straight to dormant_floor after
    # the bump grace expires (this is the "mandatory is mandatory for
    # 3 minutes, then 89" behavior).
    if band_name == "mandatory":
        # Active hours since last bump
        last_bump = seg.last_bumped_at or seg.last_accessed or seg.segment.timestamp
        name = selected_anchor(seg, ("last_bumped_at", "last_accessed", "created"))
        bump_age = elapsed_hours(seg, name, getattr(cfg, "persist_dir", None)) if projected_hours is None else projected_hours
        if last_bump and bump_age > cfg.bump_grace_active_hours:
            # Drop to dormant_floor; from there, follow mandatory band
            # (slow linear drift down to mandatory floor over 24h active,
            # then recycle bin retention at 78 for 24h active)
            dormant_age = max(0.0, age_active - cfg.bump_grace_active_hours)
            return _decay_in_band(
                start=float(cfg.dormant_floor),
                band=band,
                age_active=dormant_age,
            )
        return float(original)

    # Plateau / mid / tail: standard per-band decay
    return _decay_in_band(
        start=post_drop_score(original, cfg) if post_drop else float(original),
        band=band,
        age_active=age_active,
        skip_initial_drop=post_drop,
    )


def _decay_in_band(start: float, band: BandConfig, age_active: float,
                   skip_initial_drop: bool = False) -> float:
    """Apply a band's curve to a starting score given active-time age."""
    floor = float(band.floor)

    # Phase 1: quick initial drop
    if not skip_initial_drop and band.initial_drop_active_hours > 0 and age_active <= band.initial_drop_active_hours:
        progress = age_active / band.initial_drop_active_hours
        return start - (band.initial_drop_pts * progress)

    if not skip_initial_drop and band.initial_drop_active_hours > 0:
        # Past the quick-drop phase
        start -= band.initial_drop_pts
        post_age = age_active - band.initial_drop_active_hours
    else:
        post_age = age_active

    if start <= floor:
        # Already at or below floor; check recycle bin
        if band.bin_retention_active_hours > 0 and post_age > band.bin_retention_active_hours:
            return 0.0
        return max(0.0, start)

    # Phase 2: post-drop curve
    if band.decay_function == DecayFunction.LINEAR.value:
        to_floor_h = max(0.1, band.post_drop_to_floor_active_hours)
        rate = (start - floor) / to_floor_h
        v = start - (rate * post_age)
        return max(floor, v)

    if band.decay_function in (DecayFunction.EXPONENTIAL.value, DecayFunction.HALF_LIFE.value):
        hl = max(0.1, band.post_drop_half_life_active_hours)
        # Decay toward floor with half-life hl
        # v = floor + (start - floor) * 0.5^(post_age / hl)
        v = floor + (start - floor) * (0.5 ** (post_age / hl))
        return max(floor, v)

    return max(floor, start)


# ============================================================
# Bump interval (active hours)
# ============================================================

def bump_interval_active_hours_for(score: int) -> float:
    cfg = _cfg()
    if score < cfg.bump_min_score:
        return math.inf
    delta = max(0, 100 - score)
    steps = delta / 10.0
    return cfg.bump_base_active_hours * (cfg.bump_decay_per_10 ** steps)


def should_bump_to_surface(seg: "ScoredSegment", now: datetime) -> bool:
    if seg.importance_score < _cfg().bump_min_score:
        return False
    last = seg.last_bumped_at or seg.last_accessed or seg.segment.timestamp
    if last is None:
        return False
    # Session-hours since the last bump/access (idle time excluded).
    name = selected_anchor(seg, ("last_bumped_at", "last_accessed", "created"))
    active_hours = elapsed_hours(seg, name, getattr(_cfg(), "persist_dir", None))
    interval = bump_interval_active_hours_for(seg.importance_score)
    if interval == math.inf:
        return False
    return active_hours >= interval


# ============================================================
# Apply decay to a stored segment
# ============================================================

def apply_decay_to_segment(seg: "ScoredSegment", now: datetime) -> "ScoredSegment":
    seg.importance_score = int(round(current_effective_score(seg, now)))
    return seg


# ============================================================
# Description (human-readable)
# ============================================================

def describe_band(band: BandConfig) -> str:
    floor = int(band.floor)
    drop = band.initial_drop_pts
    drop_h = band.initial_drop_active_hours
    fn = band.decay_function
    if drop_h > 0:
        head = f"drops {drop:g} pts in {drop_h:g}h active"
    else:
        head = "no initial drop"
    if fn == DecayFunction.LINEAR.value:
        tail = f"then linear to {floor} over {band.post_drop_to_floor_active_hours:g}h active"
    elif fn in (DecayFunction.EXPONENTIAL.value, DecayFunction.HALF_LIFE.value):
        tail = f"then half-life {band.post_drop_half_life_active_hours:g}h active toward {floor}"
    else:
        tail = f"floor {floor}"
    bin_h = band.bin_retention_active_hours
    bin_str = f", recycle after {bin_h:g}h at floor" if bin_h > 0 else ""
    return f"{head}, {tail}{bin_str}"


def describe_decay(score: int, profile: str = "standard") -> str:
    cfg = _cfg()
    if score >= 99:
        return (f"Score {score}: mandatory. Bump grace {cfg.bump_grace_active_hours:g}h active, "
                f"then drops to dormant_floor {cfg.dormant_floor}, "
                f"bump interval {bump_interval_active_hours_for(score):.2f}h active.")
    band_name = _band_for(score)
    band: BandConfig = getattr(cfg, band_name)
    p = " [short-term-critical]" if profile == DecayProfile.SHORT_TERM_CRITICAL.value else ""
    return f"Score {score} {band_name} band: {describe_band(band)}{p}"


# ============================================================
# Series generators
# ============================================================

def decay_series(score: int, profile: str,
                 days: int = 30, points: int = 60) -> List[Dict[str, Any]]:
    """Generate a decay series over `days` real-time days, expressed
    as {t_min, score}.

    The session clock is the canon for timed effects, but for charting
    a *future* decay path we must project as-if the user kept ticking.
    We therefore use wall-clock minutes since `now` as the projection
    axis: a 7-day chart spans 7*1440 real minutes. The semantics are
    "if you kept working continuously, this is the path". This is the
    honest framing for the dashboard chart.

    Under the new session-clock model the projection is no longer
    scaled by an arbitrary conversion factor; each real minute is one
    minute of session time, provided the user keeps ticking.
    """
    from .models import ScoredSegment, Segment, SegmentType

    now = datetime.now()
    seg = ScoredSegment(
        segment=Segment(content="", segment_type=SegmentType.CONTEXT, timestamp=now),
        importance_score=score,
        concept_title=f"synthetic-{score}",
        decay_profile=profile,
    )
    # Real minutes per point
    real_min_total = days * 1440
    step_min = real_min_total / points
    cfg = _cfg()
    out: List[Dict[str, Any]] = []
    for i in range(points + 1):
        active_hours = step_min * i / 60.0
        projected = _band_decay(seg, score, now, cfg, projected_hours=active_hours)
        if profile == DecayProfile.SHORT_TERM_CRITICAL.value:
            lifetime = cfg.stc_hours_by_score.get(score, cfg.stc_default_active_hours)
            if active_hours >= lifetime:
                projected = float(cfg.short_term_critical_fade_floor)
        elif profile == DecayProfile.EPHEMERAL.value:
            projected = float(score)
        out.append({
            "t_min": int(step_min * i),
            "score": round(projected, 2),
        })
    return out


def bump_series(score: int, days: int = 1, points: int = 60) -> List[Dict[str, Any]]:
    """Sawtooth: surface (100) -> dormant_floor -> surface over real time.

    Under the session-clock model, 1 real minute = 1 session-minute
    (provided the user keeps ticking). No conversion factor.
    """
    cfg = _cfg()
    interval_active = bump_interval_active_hours_for(score)
    if interval_active == math.inf or days * 24.0 < interval_active:
        return [{"t_min": 0, "score": 100.0 if score >= 99 else float(score)}]
    # interval_active is in session-hours; project to real minutes.
    interval_real_min = interval_active * 60.0
    out: List[Dict[str, Any]] = [{"t_min": 0, "score": 100.0}]
    t = 0
    while t < days * 1440:
        t += 1
        out.append({"t_min": t, "score": float(cfg.dormant_floor)})
        t += int(interval_real_min)
        out.append({"t_min": t, "score": 100.0})
    return out


# ============================================================
# Public helpers
# ============================================================

def get_config() -> DormancyConfig:
    return _cfg()


def update_config(**changes) -> DormancyConfig:
    from .service_binding import bound_config, bound_store
    if bound_config() is not None:
        if bound_store()._read_only:
            raise RuntimeError('SQLite observer is read-only')
        return bound_config().update(**changes)
    assert _store is not None
    return _store.update(**changes)


def update_band(band_name: str, **changes) -> DormancyConfig:
    from .service_binding import bound_config, bound_store
    if bound_config() is not None:
        if bound_store()._read_only:
            raise RuntimeError('SQLite observer is read-only')
        return bound_config().update_band(band_name, **changes)
    assert _store is not None
    return _store.update_band(band_name, **changes)


def tick_active_clock_public() -> DormancyConfig:
    """Public wrapper for the tick to advance the active clock."""
    return tick_active_clock(_cfg())
