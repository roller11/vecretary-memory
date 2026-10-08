# SPDX-License-Identifier: MPL-2.0
"""
Tests for instinct.session_timer -- the session-based active clock.

The session timer is the single canon for all timed effects in the
engine. These tests pin its invariants:

  1. Fresh state is dead (no session, zero accumulator).
  2. First tick starts a session and contributes zero seconds.
  3. Subsequent ticks within the idle window add to the accumulator.
  4. A tick after idle_minutes of silence opens a NEW session and the
     gap is discarded (the old session's time stays in the accumulator).
  5. State persists across process boundaries (atomic write).
  6. Legacy DormancyConfig JSON with `active_hours_per_real_day` loads
     cleanly; the field is dropped silently.
  7. `now_active` and `current_effective_score` no longer multiply by a
     wall-clock conversion factor.

Run: pytest tests/test_session_timer.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def persist_dir(tmp_path):
    """Each test gets its own persist_dir so session_state.json is isolated."""
    d = tmp_path / "instinct_data"
    d.mkdir(mode=0o700)
    return str(d)


# ---------------------------------------------------------------------------
# 1-4: tick semantics
# ---------------------------------------------------------------------------

def test_fresh_state_is_dead(persist_dir):
    from instinct.session_timer import (
        accumulated_session_hours, session_alive, reset_session,
    )
    reset_session(persist_dir)
    assert accumulated_session_hours(persist_dir) == 0.0
    assert not session_alive(persist_dir)


def test_first_tick_starts_session_and_adds_zero(persist_dir):
    from instinct.session_timer import tick_session
    t0 = datetime(2026, 9, 29, 10, 0, 0)
    r = tick_session(persist_dir, idle_minutes=10, now=t0)
    assert r.new_session_started is True
    assert r.session_alive is True
    assert r.seconds_added_to_accumulator == 0.0


def test_subsequent_ticks_within_idle_window_accumulate(persist_dir):
    from instinct.session_timer import (
        tick_session, accumulated_session_hours,
    )
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 5, 0))
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 7, 0))
    # 5 + 2 = 7 minutes = 420 seconds
    assert abs(accumulated_session_hours(persist_dir) - 7 / 60.0) < 1e-9


def test_tick_after_idle_window_starts_new_session_and_discards_gap(persist_dir):
    """The whole point: idle wall-clock time does NOT contribute to aging."""
    from instinct.session_timer import (
        tick_session, accumulated_session_hours,
    )
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 5, 0))
    assert abs(accumulated_session_hours(persist_dir) - 5 / 60.0) < 1e-9

    # 30-minute idle gap (well past the 10-minute window) -> new session,
    # the gap is discarded.
    r = tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 35, 0))
    assert r.new_session_started is True
    assert r.seconds_added_to_accumulator == 0.0
    # The 5 minutes from the closed session remain in the accumulator.
    assert abs(accumulated_session_hours(persist_dir) - 5 / 60.0) < 1e-9


def test_tick_at_exact_idle_boundary_does_not_close(persist_dir):
    """Boundary: exactly idle_minutes is still alive (use > idle_window_s)."""
    from instinct.session_timer import tick_session
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    # Exactly 10 minutes (600s == 10 * 60); comparison is `>` so this is
    # still inside the window.
    r = tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 10, 0))
    assert r.new_session_started is False
    assert abs(r.seconds_added_to_accumulator - 600.0) < 1e-6


def test_idle_minutes_is_respected(persist_dir):
    """Custom idle_minutes value is honored."""
    from instinct.session_timer import tick_session
    tick_session(persist_dir, idle_minutes=2, now=datetime(2026, 9, 29, 10, 0, 0))
    # 3 minutes idle with a 2-minute window -> new session.
    r = tick_session(persist_dir, idle_minutes=2, now=datetime(2026, 9, 29, 10, 3, 0))
    assert r.new_session_started is True


# ---------------------------------------------------------------------------
# 5: persistence
# ---------------------------------------------------------------------------

def test_state_persists_across_process_simulations(persist_dir):
    """Re-importing the module does not lose state; on-disk file is the source of truth."""
    from instinct.session_timer import (
        tick_session, accumulated_session_hours, _store_for,
    )
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 5, 0))

    # Simulate a new process: drop the in-process cache and re-read from disk.
    store = _store_for(persist_dir)
    store.invalidate()
    assert abs(accumulated_session_hours(persist_dir) - 5 / 60.0) < 1e-9


def test_atomic_write_does_not_corrupt_on_partial_failure(persist_dir):
    """If a write is interrupted, the previous state must survive."""
    from instinct.session_timer import (
        tick_session, accumulated_session_hours, _store_for,
    )
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 5, 0))

    # Drop a half-written tmp file alongside the real file (simulate crash).
    tmp_file = Path(persist_dir) / ".session_garbage.tmp"
    tmp_file.write_text("{ partial json")
    assert tmp_file.exists()

    store = _store_for(persist_dir)
    store.invalidate()
    # Real file should still load correctly.
    assert abs(accumulated_session_hours(persist_dir) - 5 / 60.0) < 1e-9


def test_reset_clears_everything(persist_dir):
    from instinct.session_timer import (
        tick_session, reset_session, accumulated_session_hours, session_alive,
    )
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 5, 0))
    assert accumulated_session_hours(persist_dir) > 0

    reset_session(persist_dir)
    assert accumulated_session_hours(persist_dir) == 0.0
    assert not session_alive(persist_dir)


# ---------------------------------------------------------------------------
# 6: legacy config migration
# ---------------------------------------------------------------------------

def test_legacy_dormancy_config_loads_without_active_hours_per_real_day(persist_dir):
    """A pre-v2 dormancy_config.json with the legacy fields must load cleanly."""
    from instinct.dormancy import DormancyConfig
    legacy = {
        "active_hours_per_real_day": 9.0,
        "last_active_at": "2026-09-29T10:00:00",
        "bump_grace_active_hours": 0.05,
        "bump_min_score": 60,
        "dormant_floor": 89,
        "mandatory": {
            "initial_drop_pts": 0.0, "initial_drop_active_hours": 0.0,
            "decay_function": "linear", "floor": 78,
            "bin_retention_active_hours": 24.0,
            "post_drop_to_floor_active_hours": 24.0,
            "post_drop_half_life_active_hours": 0.0,
        },
        "plateau": {
            "initial_drop_pts": 10.0, "initial_drop_active_hours": 14.0,
            "decay_function": "linear", "floor": 70,
            "bin_retention_active_hours": 240.0,
            "post_drop_to_floor_active_hours": 240.0,
            "post_drop_half_life_active_hours": 0.0,
        },
        "mid": {
            "initial_drop_pts": 10.0, "initial_drop_active_hours": 4.0,
            "decay_function": "exponential", "floor": 20,
            "bin_retention_active_hours": 80.0,
            "post_drop_to_floor_active_hours": 0.0,
            "post_drop_half_life_active_hours": 20.0,
        },
        "tail": {
            "initial_drop_pts": 0.0, "initial_drop_active_hours": 0.0,
            "decay_function": "half_life", "floor": 0,
            "bin_retention_active_hours": 0.0,
            "post_drop_to_floor_active_hours": 0.0,
            "post_drop_half_life_active_hours": 8.0,
        },
        "short_term_critical_active_hours": 3.0,
        "short_term_critical_fade_floor": 49,
        "stc_hours_by_score": {"91": 3.0, "92": 24.0, "93": 72.0},
        "stc_default_active_hours": 3.0,
        "tier_90_drop_to": 89, "tier_90_floor": 60, "tier_90_total_active_hours": 200.0,
        "tier_94_drop_to": 88, "tier_94_floor": 29, "tier_94_total_active_hours": 500.0,
        "tier_95_drop_to": 88, "tier_95_floor": 60, "tier_95_total_active_hours": 1000.0,
        "tier_90_drop_active_hours": 4.0,
        "tier_94_drop_active_hours": 4.0,
        "tier_95_drop_active_hours": 4.0,
        "ephemeral_surface_turns": 6, "ephemeral_fade_turns": 6,
        "min_keyword_hits_for_faded_stc": 3,
    }
    cfg = DormancyConfig.from_dict(legacy)
    # Default for the new field is 10.0 minutes.
    assert cfg.session_idle_minutes == 10.0
    # Round-trip: to_dict drops the legacy fields.
    d = cfg.to_dict()
    assert "active_hours_per_real_day" not in d
    assert "last_active_at" not in d
    assert "session_idle_minutes" in d


def test_session_idle_minutes_round_trips_via_disk(persist_dir):
    """The dashboard-editable field survives a save/load cycle."""
    from instinct.dormancy import configure, DormancyConfig, update_config, get_config
    configure(persist_dir)
    cfg = DormancyConfig()
    cfg.session_idle_minutes = 7.5
    update_config(session_idle_minutes=7.5)

    # Reload via the public API.
    loaded = get_config()
    assert loaded.session_idle_minutes == 7.5


# ---------------------------------------------------------------------------
# 7: end-to-end -- the session clock drives dormancy math
# ---------------------------------------------------------------------------

def test_current_effective_score_unchanged_when_no_session(persist_dir):
    """No ticks yet -> no aging. A fresh segment stays at its initial score.

    Reset session state in the persist_dir so the dormancy code sees a
    fresh accumulator (the dashboard's long-running ticks against the
    default persist dir don't pollute this test's session clock).
    """
    from instinct.dormancy import configure, current_effective_score
    from instinct.models import ScoredSegment, Segment, SegmentType
    from instinct import session_timer as st

    configure(persist_dir)
    st.reset_session(persist_dir)
    ts = datetime.now()
    seg = ScoredSegment(
        segment=Segment(content="", segment_type=SegmentType.CONTEXT, timestamp=ts),
        importance_score=70,  # mid band
        concept_title="synth-70",
    )
    # No ticks have happened, so accumulator is 0 and age_active is 0.
    assert abs(current_effective_score(seg, ts) - 70.0) < 0.01


def test_dormancy_no_longer_multiplies_by_active_hours_per_real_day(persist_dir):
    """A long wall-clock gap with no ticks does NOT age memories."""
    from instinct.dormancy import configure, current_effective_score
    from instinct.models import ScoredSegment, Segment, SegmentType
    from instinct import session_timer as st

    configure(persist_dir)
    st.reset_session(persist_dir)

    tick_time = datetime(2026, 9, 29, 10, 0, 0)
    seg = ScoredSegment(
        segment=Segment(content="", segment_type=SegmentType.CONTEXT, timestamp=tick_time),
        importance_score=70,
        concept_title="synth-70",
    )

    # One tick to start a session.
    st.tick_session(persist_dir, idle_minutes=10, now=tick_time)
    # Score queried at the same moment as the tick -> age_active ~ 0.
    s_after_first_tick = current_effective_score(seg, tick_time)
    assert abs(s_after_first_tick - 70.0) < 0.01

    # Now ask "what if 100 hours of wall-clock passed?" with no ticks.
    # Under the legacy math: real_hours * 9/24 = 37.5 active hours would have aged
    # the segment. Under the new clock, no session activity -> no aging.
    future = tick_time + timedelta(hours=100)
    s_after_long_idle = current_effective_score(seg, future)
    assert abs(s_after_long_idle - 70.0) < 0.01, (
        "Idle wall-clock time must not age memories; got %s" % s_after_long_idle
    )


def test_session_alive_predicate_uses_session_clock(persist_dir):
    from instinct.session_timer import tick_session, session_alive
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    assert session_alive(persist_dir, now=datetime(2026, 9, 29, 10, 9, 0)) is True
    assert session_alive(persist_dir, now=datetime(2026, 9, 29, 10, 11, 0)) is False


def test_session_now_is_an_anchor_for_subtraction(persist_dir):
    """session_now() returns a datetime that callers can subtract from segment.timestamp."""
    from instinct.session_timer import tick_session, session_now
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 5, 0))
    sn = session_now(persist_dir)
    # The accumulator should be 5 minutes = 300 seconds, so session_now is
    # session_started_at + 300s. session_started_at was the first tick at 10:00:00.
    assert sn == datetime(2026, 9, 29, 10, 5, 0)


def test_session_elapsed_hours_handles_none_anchor(persist_dir):
    from instinct.session_timer import session_elapsed_hours, tick_session
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    assert session_elapsed_hours(None, persist_dir) == 0.0


def test_session_elapsed_hours_bounded_by_accumulator(persist_dir):
    """The session clock cannot lie: it never reports more hours than were accumulated."""
    from instinct.session_timer import (
        tick_session, session_elapsed_hours, accumulated_session_hours,
    )
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 0, 0))
    tick_session(persist_dir, idle_minutes=10, now=datetime(2026, 9, 29, 10, 5, 0))
    acc = accumulated_session_hours(persist_dir)
    # Ask for elapsed-hours since an anchor 100 hours ago -- the answer
    # must be capped at the actual accumulator.
    early = datetime(2026, 9, 28, 10, 0, 0)  # 24h ago
    v = session_elapsed_hours(early, persist_dir)
    assert v <= acc + 1e-9
