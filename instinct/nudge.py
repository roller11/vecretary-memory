# SPDX-License-Identifier: MPL-2.0
"""
instinct.nudge - Turn-based gentle nudge system for mandatory memories.

A 100-score mandatory memory is "elevated" (visible to every agent call).
To avoid force-feeding the same rule on every call, this module:

  1. On access: drops an elevated mandatory memory's score by 21 (100 -> 79).
     This is a true score mutation, like the other modifiers.
  2. On turn: increments a global turn counter. Each tool call = 1 turn.
  3. On check: for every mandatory segment that has been at 79 (or anywhere
     below 100) for >= `nudge_interval_turns` turns, re-elevate it to 100
     and reset its `last_nudged_turn` to the current turn.

Per-segment `nudge_interval_turns` controls how aggressive the reminder
is. Default is 5 turns. Set to 0 to disable nudging for a segment.

The nudge is *gentle* — it only changes scores, not the stored content.
After re-elevation to 100, the next access will drop it back to 79 and
the cycle repeats. This is the desired behavior: occasional reminders
without spam.

Public API:
    tick_nudge_session(store) -> TickResult - tick session and restore mandatory
    tick_turn(store) -> int              - increment global turn counter
    apply_access_drop(seg, turn) -> bool - drop 100->79 on access (mutates)
    apply_nudge_check(store) -> dict     - re-elevate eligible segments
    get_current_turn(store) -> int       - read current turn without ticking
    get_nudge_state(seg) -> dict          - per-segment nudge metadata
    nudge_summary(store) -> dict          - aggregate nudge state
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .models import (
    NUDGE_ACTIVE_KEY,
    NUDGE_INTERVAL_TURNS_KEY,
    NUDGED_AT_TURN_KEY,
    TURNS_SINCE_ACCESS_KEY,
)
from .session_timer import DEFAULT_SESSION_IDLE_MINUTES, TickResult
from .private_storage import private_binary_reader, write_private_text, PrivateStorageError

logger = logging.getLogger("instinct.nudge")

# Drop semantics: on access, an elevated (100) mandatory memory is dropped
# to 79 and held there until the nudge timer re-elevates it. 100 - 21 = 79.
DROP_AMOUNT = 21
ELEVATED_SCORE = 100
DEFAULT_NUDGE_INTERVAL = 5  # turns


# Per-type defaults: standing instructions reappear faster than critical
# constraints (which are more general/evergreen).
_TYPE_DEFAULTS = {
    "standing_instruction": 5,
    "critical_constraint": 10,
    "architectural_decision": 20,
    "implementation_detail": 30,
    "context": 30,
    "user_preference": 15,
}


# =============================================================================
# Global turn counter (singleton, persisted in instinct_data)
# =============================================================================

def _turn_state_path(persist_dir: str) -> Path:
    return Path(persist_dir) / "nudge_state.json"


def _load_turn_state(persist_dir: str) -> Dict[str, Any]:
    path = _turn_state_path(persist_dir)
    if not path.exists() and not path.is_symlink():
        return {"current_turn": 0, "last_tick_at": None}
    try:
        with private_binary_reader(path) as source:
            return json.load(source)
    except PrivateStorageError:
        raise
    except (json.JSONDecodeError, OSError):
        return {"current_turn": 0, "last_tick_at": None}


def _save_turn_state(persist_dir: str, state: Dict[str, Any]) -> None:
    path = _turn_state_path(persist_dir)
    write_private_text(path, json.dumps(state, indent=2))


def get_current_turn(store) -> int:
    """Read the current turn counter without ticking.

    Args:
        store: An InstinctStore (or anything with a `persist_dir` attribute).

    Returns:
        Current global turn number (0 if never ticked).
    """
    return int(_load_turn_state(store.persist_dir).get("current_turn", 0))


def tick_turn(store) -> int:
    """Increment the global turn counter by 1. Persists to disk.

    Called once per MCP tool call (or once per `get_context` call from the
    dashboard's nudge-check endpoint).

    Returns:
        The new current turn number.
    """
    state = _load_turn_state(store.persist_dir)
    new_turn = int(state.get("current_turn", 0)) + 1
    state["current_turn"] = new_turn
    state["last_tick_at"] = datetime.now().isoformat(timespec="seconds")
    _save_turn_state(store.persist_dir, state)
    return new_turn


def reset_turn_counter(store) -> int:
    """Reset the global turn counter to 0. Test/debug helper."""
    state = {"current_turn": 0, "last_tick_at": datetime.now().isoformat(timespec="seconds")}
    _save_turn_state(store.persist_dir, state)
    return 0


def tick_nudge_session(
    store,
    idle_minutes: float = DEFAULT_SESSION_IDLE_MINUTES,
    now: Optional[datetime] = None,
) -> TickResult:
    """Tick the session clock and restore mandatory memories once per session.

    Read-only observers cannot advance the clock. Failed updates leave the
    marker pending so the next writable tick retries. Turn numbers stay
    monotonic across sessions.
    """
    from .session_timer import tick_session

    if getattr(store, "_read_only", False):
        from .session_timer import state, session_alive
        return TickResult(
            state=state(store.persist_dir), new_session_started=False,
            session_alive=session_alive(store.persist_dir, idle_minutes=idle_minutes, now=now),
            seconds_added_to_accumulator=0.0, idle_seconds_before_this_tick=0.0,
        )
    result = tick_session(store.persist_dir, idle_minutes=idle_minutes, now=now)

    turn_state = _load_turn_state(store.persist_dir)
    session_started_at = result.state.session_started_at
    if turn_state.get("session_started_at") == session_started_at:
        return result

    current_turn = int(turn_state.get("current_turn", 0))
    renew_mandatory(store, current_turn=current_turn)

    turn_state["session_started_at"] = session_started_at
    _save_turn_state(store.persist_dir, turn_state)
    return result


def renew_mandatory(store, current_turn: Optional[int] = None) -> int:
    """Renew existing active mandatory records without embedding or reseeding."""
    from .models import MANDATORY_TYPES

    if getattr(store, "_read_only", False):
        raise RuntimeError("Mandatory renewal requires a writable store.")
    if current_turn is None:
        current_turn = get_current_turn(store)
    renewed = 0
    candidates = store.get_by_score(
        min_score=0, max_score=99, limit=max(1, store.count()),
    )
    for segment in candidates:
        if segment.segment.segment_type not in MANDATORY_TYPES:
            continue
        segment.importance_score = ELEVATED_SCORE
        _reset_active_anchor(segment, store)
        _set_last_nudged_turn(segment, current_turn)
        _set_last_accessed_turn(segment, 0)
        if not store.update_metadata_only(segment):
            raise RuntimeError("Could not restore mandatory memory; renewal remains pending.")
        renewed += 1
    return renewed


def _reset_active_anchor(segment, store) -> None:
    from .active_time import set_anchor
    segment.last_bumped_at = datetime.now()
    segment.pop_anchor_at = segment.last_bumped_at
    set_anchor(segment, "last_bumped_at", store.persist_dir)
    set_anchor(segment, "pop_anchor_at", store.persist_dir)


# =============================================================================
# Per-segment nudge configuration
# =============================================================================

def default_interval_for_segment(seg) -> int:
    """Default nudge interval based on segment type."""
    try:
        stype = seg.segment.segment_type.value
    except AttributeError:
        stype = "context"
    return _TYPE_DEFAULTS.get(stype, DEFAULT_NUDGE_INTERVAL)


def get_nudge_interval(seg) -> int:
    """Read the nudge interval for a segment. Returns default if unset."""
    val = (seg.segment.metadata or {}).get(NUDGE_INTERVAL_TURNS_KEY)
    if val is None or val == "":
        return default_interval_for_segment(seg)
    try:
        return max(0, int(val))
    except (ValueError, TypeError):
        return default_interval_for_segment(seg)


def set_nudge_interval(seg, turns: int) -> None:
    """Set the per-segment nudge interval. Persisted in metadata."""
    if not isinstance(turns, int) or turns < 0:
        raise ValueError(f"nudge_interval_turns must be a non-negative int, got {turns!r}")
    md = dict(seg.segment.metadata or {})
    md[NUDGE_INTERVAL_TURNS_KEY] = int(turns)
    seg.segment.metadata = md


def nudge_active(seg) -> bool:
    """True if nudge is enabled for this segment. Default: True."""
    val = (seg.segment.metadata or {}).get(NUDGE_ACTIVE_KEY)
    if val is None or val == "":
        return True
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("1", "true", "yes")


def set_nudge_active(seg, active: bool) -> None:
    md = dict(seg.segment.metadata or {})
    md[NUDGE_ACTIVE_KEY] = bool(active)
    seg.segment.metadata = md


def get_turns_since_access(seg) -> int:
    """Read the turn number of the segment's last access (counter form)."""
    val = (seg.segment.metadata or {}).get(TURNS_SINCE_ACCESS_KEY)
    if val is None or val == "":
        return 0
    try:
        return max(0, int(val))
    except (ValueError, TypeError):
        return 0


def _set_last_accessed_turn(seg, turn: int) -> None:
    md = dict(seg.segment.metadata or {})
    md[TURNS_SINCE_ACCESS_KEY] = int(turn)
    seg.segment.metadata = md


def get_last_nudged_turn(seg) -> int:
    """Return the turn number at which the segment was last re-elevated to 100."""
    val = (seg.segment.metadata or {}).get(NUDGED_AT_TURN_KEY)
    if val is None or val == "":
        return 0
    try:
        return max(0, int(val))
    except (ValueError, TypeError):
        return 0


def _set_last_nudged_turn(seg, turn: int) -> None:
    md = dict(seg.segment.metadata or {})
    md[NUDGED_AT_TURN_KEY] = int(turn)
    seg.segment.metadata = md


# =============================================================================
# Mutation entry points
# =============================================================================

def apply_access_drop(seg, current_turn: int) -> bool:
    """On access of a mandatory segment, drop its score 100 -> 79.

    Only acts if the segment is currently at 100 (elevated state). Once
    dropped, the segment stays at 79 until the nudge timer fires and
    re-elevates it. This means the agent receives the rule once, then
    it's quiet for N turns, then the rule reappears.

    Returns:
        True if the drop was applied, False otherwise.
    """
    if seg.importance_score != ELEVATED_SCORE:
        return False
    if not nudge_active(seg):
        return False
    from .models import MANDATORY_TYPES
    if seg.segment.segment_type not in MANDATORY_TYPES:
        return False

    new_score = max(0, ELEVATED_SCORE - DROP_AMOUNT)
    seg.importance_score = new_score
    _set_last_accessed_turn(seg, current_turn)
    logger.debug(
        "nudge: dropped %s from 100 to %d (will re-elevate in %d turns)",
        seg.id[:12], new_score, get_nudge_interval(seg),
    )
    return True


def apply_nudge_check(store) -> Dict[str, int]:
    """Re-elevate mandatory segments whose nudge interval has elapsed.

    Iterates over all mandatory segments, computes `current_turn -
    max(last_nudged_turn, last_accessed_turn)` and re-elevates any
    segment where the delta >= `nudge_interval_turns`.

    Also walks previously-elevated segments that have been dropped
    (score 79). If their `turns_since_access` has reached the interval,
    they get re-elevated to 100 and the cycle resumes.

    Returns:
        Stats dict with `scanned`, `elevated`, `skipped_disabled`, `skipped_recent`.
    """
    from .models import MANDATORY_TYPES

    current_turn = get_current_turn(store)
    elevated = 0
    skipped_disabled = 0
    skipped_recent = 0
    scanned = 0

    # Walk all mandatory segments. Mandatory is a small set (26 typical)
    # so we can fetch them all and update in-place.
    mandatories = store.get_mandatory()
    for seg in mandatories:
        scanned += 1

        # Skip if nudge disabled for this segment
        if not nudge_active(seg):
            skipped_disabled += 1
            continue

        # Skip if interval is 0 (disabled) or negative
        interval = get_nudge_interval(seg)
        if interval <= 0:
            skipped_disabled += 1
            continue

        # Skip if not currently mandatory type (defensive)
        if seg.segment.segment_type not in MANDATORY_TYPES:
            continue

        # Compute baseline: last turn at which we either nudged or accessed.
        # If both are 0, the segment is brand-new and should be elevated now.
        last_nudged = get_last_nudged_turn(seg)
        last_accessed_turn = get_turns_since_access(seg)
        baseline = max(last_nudged, last_accessed_turn)

        # If baseline is 0, this segment has never been touched by the nudge
        # system. Treat it as eligible for the first nudge.
        if baseline == 0:
            # First nudge ever - elevate now to seed the baseline
            seg.importance_score = ELEVATED_SCORE
            _reset_active_anchor(seg, store)
            _set_last_nudged_turn(seg, current_turn)
            # Metadata-only update avoids triggering an Ollama re-embed.
            store.update_metadata_only(seg)
            elevated += 1
            logger.info(
                "nudge: first-elevated %s (%s) to 100 at turn %d",
                seg.id[:12], seg.concept_title[:30], current_turn,
            )
            continue

        # Otherwise, only re-elevate if the interval has elapsed
        if current_turn - baseline < interval:
            skipped_recent += 1
            continue

        # Re-elevate
        seg.importance_score = ELEVATED_SCORE
        _reset_active_anchor(seg, store)
        _set_last_nudged_turn(seg, current_turn)
        store.update_metadata_only(seg)
        elevated += 1
        logger.info(
            "nudge: re-elevated %s (%s) to 100 at turn %d (interval=%d)",
            seg.id[:12], seg.concept_title[:30], current_turn, interval,
        )

    # -------------------------------------------------------------------------
    # Second pass: dropped mandatory segments (score 79, with a recent access
    # recorded). get_mandatory() no longer returns them, so we have to look
    # them up by walking all segments of mandatory types. This set is small
    # (only recently dropped ones) so the full walk is acceptable.
    # -------------------------------------------------------------------------
    for stype in MANDATORY_TYPES:
        # _query_by_type is internal but exposed via duck-typing.
        # Fall back gracefully if not available.
        try:
            dropped = _find_dropped_of_type(store, stype)
        except Exception:
            continue
        for seg in dropped:
            scanned += 1
            if not nudge_active(seg):
                skipped_disabled += 1
                continue
            interval = get_nudge_interval(seg)
            if interval <= 0:
                skipped_disabled += 1
                continue
            last_accessed_turn = get_turns_since_access(seg)
            if last_accessed_turn == 0:
                # Never accessed but somehow not at 100 — treat as fresh.
                continue
            if current_turn - last_accessed_turn < interval:
                skipped_recent += 1
                continue
            # Re-elevate
            seg.importance_score = ELEVATED_SCORE
            _reset_active_anchor(seg, store)
            _set_last_nudged_turn(seg, current_turn)
            # Clear the access marker so the next access sets it again.
            _set_last_accessed_turn(seg, 0)
            store.update_metadata_only(seg)
            elevated += 1
            logger.info(
                "nudge: re-elevated dropped %s (%s) to 100 at turn %d "
                "(turns_since_access=%d, interval=%d)",
                seg.id[:12], seg.concept_title[:30], current_turn,
                last_accessed_turn, interval,
            )

    return {
        "scanned": scanned,
        "elevated": elevated,
        "skipped_disabled": skipped_disabled,
        "skipped_recent": skipped_recent,
        "current_turn": current_turn,
    }


def _find_dropped_of_type(store, stype) -> list:
    """Find dropped mandatory segments of a given type.

    A "dropped" segment is one that was previously at score 100 but
    has been dropped to 79 by an access. We identify candidates by:
      1. segment_type matches the given type
      2. importance_score is in the dropped range (60..99)
      3. metadata has a non-zero `turns_since_access` (was accessed)
      4. `nudge_active` is True in metadata

    This is intentionally cheap: dropped mandatory is a small set
    (typically zero, occasionally a handful during a busy session),
    so we just filter in Python.
    """
    from .models import SegmentType
    try:
        type_value = stype.value if isinstance(stype, SegmentType) else stype
    except Exception:
        return []
    # Range: 60..99. The canonical dropped score is 79, but other
    # modifiers could shift it. The lower bound 60 keeps us clear of
    # tail-band content.
    try:
        candidates = store.get_by_score(
            min_score=60,
            max_score=99,
            limit=200,
        )
    except Exception:
        return []
    dropped = []
    for seg in candidates:
        if seg.segment.segment_type.value != type_value:
            continue
        if get_turns_since_access(seg) <= 0:
            continue
        if not nudge_active(seg):
            continue
        dropped.append(seg)
    return dropped


# =============================================================================
# Reporting
# =============================================================================

def get_nudge_state(seg) -> Dict[str, Any]:
    """Return a dict describing the segment's nudge state for UI/reporting."""
    from .models import MANDATORY_TYPES

    is_mand = seg.segment.segment_type in MANDATORY_TYPES
    interval = get_nudge_interval(seg)
    active = nudge_active(seg)
    last_nudged = get_last_nudged_turn(seg)
    last_accessed_turn = get_turns_since_access(seg)
    baseline = max(last_nudged, last_accessed_turn)

    return {
        "id": seg.id,
        "concept_title": seg.concept_title,
        "segment_type": seg.segment.segment_type.value,
        "is_mandatory_type": is_mand,
        "importance_score": seg.importance_score,
        "is_elevated": seg.importance_score == ELEVATED_SCORE,
        "nudge_active": active,
        "nudge_interval_turns": interval,
        "last_nudged_turn": last_nudged,
        "last_accessed_turn": last_accessed_turn,
        "baseline_turn": baseline,
        "default_interval": default_interval_for_segment(seg),
    }


def nudge_summary(store) -> Dict[str, Any]:
    """Aggregate nudge state across all mandatory segments."""
    current_turn = get_current_turn(store)
    mandatories = store.get_mandatory()
    elevated = 0
    dropped = 0
    disabled = 0
    due_for_nudge = 0

    for seg in mandatories:
        if not nudge_active(seg):
            disabled += 1
            continue
        interval = get_nudge_interval(seg)
        if interval <= 0:
            disabled += 1
            continue
        if seg.importance_score == ELEVATED_SCORE:
            elevated += 1
        else:
            dropped += 1
        # Is this segment due for a nudge?
        baseline = max(get_last_nudged_turn(seg), get_turns_since_access(seg))
        if baseline == 0 or current_turn - baseline >= interval:
            due_for_nudge += 1

    return {
        "current_turn": current_turn,
        "mandatory_total": len(mandatories),
        "elevated": elevated,
        "dropped": dropped,
        "nudge_disabled": disabled,
        "due_for_nudge": due_for_nudge,
    }


# --- self-test ---
if __name__ == "__main__":
    print("nudge self-test")
    print(f"  DROP_AMOUNT = {DROP_AMOUNT}")
    print(f"  ELEVATED_SCORE = {ELEVATED_SCORE}")
    print(f"  DEFAULT_NUDGE_INTERVAL = {DEFAULT_NUDGE_INTERVAL}")
    print(f"  _TYPE_DEFAULTS = {_TYPE_DEFAULTS}")
