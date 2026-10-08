# SPDX-License-Identifier: MPL-2.0
"""Persistent per-memory anchors on the canonical active-session clock."""

from __future__ import annotations

import math
from typing import Optional

from . import session_timer


ANCHOR_FIELDS = (
    "created", "last_accessed", "last_bumped_at", "popped_at",
    "last_popped_at", "pop_anchor_at",
)


def clock_snapshot(persist_dir=None) -> dict:
    clock = session_timer.state(persist_dir)
    return {"clock_id": clock.clock_id, "seconds": clock.accumulated_session_seconds}


def anchor_timestamp(segment, name):
    return segment.segment.timestamp if name == "created" else getattr(segment, name)


def set_anchor(segment, name: str, persist_dir=None, seconds: Optional[float] = None,
               score: Optional[int] = None) -> None:
    timestamp = anchor_timestamp(segment, name)
    if timestamp is None:
        segment.active_time_anchors.pop(name, None)
        return
    snapshot = clock_snapshot(persist_dir)
    if seconds is not None:
        snapshot["seconds"] = max(0.0, min(snapshot["seconds"], seconds))
    snapshot["timestamp"] = timestamp.isoformat()
    snapshot["score"] = segment.importance_score if score is None else score
    segment.active_time_anchors[name] = snapshot


def capture_anchors(segment, persist_dir=None, new: bool = False) -> None:
    clock = session_timer.state(persist_dir)
    for name in ANCHOR_FIELDS:
        timestamp = anchor_timestamp(segment, name)
        if timestamp is None:
            segment.active_time_anchors.pop(name, None)
            continue
        previous = segment.active_time_anchors.get(name)
        if previous and previous.get("clock_id") == clock.clock_id and previous.get("timestamp") == timestamp.isoformat():
            continue
        baseline = None if (new or (previous and previous.get("clock_id"))) else clock.baseline_session_seconds
        set_anchor(segment, name, persist_dir, baseline)


def elapsed_hours(segment, name: str, persist_dir=None) -> float:
    timestamp = anchor_timestamp(segment, name)
    if timestamp is None:
        return 0.0
    clock = session_timer.state(persist_dir)
    snapshot = segment.active_time_anchors.get(name)
    if snapshot is None:
        return max(0.0, clock.accumulated_session_seconds - clock.baseline_session_seconds) / 3600.0
    if snapshot.get("timestamp") != timestamp.isoformat():
        return 0.0
    return snapshot_elapsed_hours(snapshot, persist_dir)


def snapshot_elapsed_hours(snapshot: dict, persist_dir=None) -> float:
    clock = session_timer.state(persist_dir)
    if not clock.clock_id or snapshot.get("clock_id") != clock.clock_id:
        return 0.0
    seconds = snapshot.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return 0.0
    return max(0.0, clock.accumulated_session_seconds - max(0.0, seconds)) / 3600.0


def selected_anchor(segment, names=ANCHOR_FIELDS[::-1]) -> str:
    return next((name for name in names if anchor_timestamp(segment, name) is not None), "created")
