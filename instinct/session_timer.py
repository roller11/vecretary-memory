# SPDX-License-Identifier: MPL-2.0
"""
instinct.session_timer - Session-based active clock (the canon for all timed effects).

Why this exists
---------------
The legacy "active time" math (real_hours * active_hours_per_real_day / 24)
treated idle wall-clock hours as if they were work hours. That meant a
weekend away, or a week spent on a different project, *aged* every
memory in the store as if the user had been working on them. That is
the opposite of what a "session" should mean.

This module is the **single canon** for "active time" across the
engine. It implements a real session-based clock:

  * A **session** begins on the first tool call after a long idle.
  * A session **stays alive** as long as tool calls keep coming.
  * A session **expires** when no tool call has arrived for
    `session_idle_minutes` (default 10) of wall-clock time.
  * Runtime deltas use a monotonic clock, not system calendar time.
  * The accumulator `accumulated_session_seconds` grows only while a
    session is alive. Any time spent outside a session is **discarded**
    -- it does not contribute to aging. A new writer process contributes zero
    on its first tick; offline/restart gaps cannot be reconstructed as work.

Other modules ask this one, and only this one, "how much session time
has elapsed?" -- never multiplying wall-clock hours by an arbitrary
conversion factor again.

Persistence
-----------
State is stored in `<persist_dir>/session_state.json`. Writes are atomic
(tempfile + os.replace). The file holds:

  {
    "session_started_at": "...ISO..." | null,
    "session_last_tick_at": "...ISO..." | null,
    "accumulated_session_seconds": 0.0
  }

Public API
----------
* `tick_session(persist_dir, idle_minutes=10) -> TickResult`
* `accumulated_session_hours(persist_dir) -> float`
* `session_now(persist_dir) -> datetime`
* `state(persist_dir) -> SessionState`    (read-only snapshot)
* `reset_session(persist_dir) -> SessionState`  (test/debug helper)
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional
from .private_storage import private_binary_reader, write_private_text, PrivateStorageError

# Default idle window: 10 minutes. User-tunable via DormancyConfig.session_idle_minutes.
DEFAULT_SESSION_IDLE_MINUTES = 10.0

SESSION_STATE_FILENAME = "session_state.json"


@lru_cache(maxsize=1)
def _monotonic_identity() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return f"process:{os.getpid()}"


@dataclass
class SessionState:
    """The canonical session-clock state, persisted to disk."""
    session_started_at: Optional[str]       # ISO datetime of session start, or None
    session_last_tick_at: Optional[str]     # ISO datetime of last tick, or None
    accumulated_session_seconds: float      # sum across all completed + active sessions
    clock_id: str = ""
    baseline_session_seconds: float = 0.0
    last_tick_monotonic: Optional[float] = None
    monotonic_identity: str = ""


@dataclass
class TickResult:
    """Returned from tick_session() so callers can log/audit."""
    state: SessionState
    new_session_started: bool               # True if this tick opened a new session
    session_alive: bool                     # True if session is still considered active
    seconds_added_to_accumulator: float     # how much this tick contributed
    idle_seconds_before_this_tick: float    # how long since the previous tick (or 0 if new session)


# ---------------------------------------------------------------------------
# Persistence (file-based, atomic)
# ---------------------------------------------------------------------------

class _SessionStore:
    """One-process-at-a-time writer for session_state.json.

    The session clock is single-writer per machine: the dashboard, the
    MCP server, and any probe scripts all share the same on-disk state.
    A small in-process RLock prevents torn writes inside one process.
    Cross-process ordering is handled by the chroma_writer_lock for the
    InstinctStore; the session clock is independent and fast (no
    embedding, no SQL), so we don't share that lock to keep ticks
    cheap.
    """
    def __init__(self, persist_dir: str | os.PathLike):
        self.path = Path(persist_dir) / SESSION_STATE_FILENAME
        self._lock = threading.RLock()
        self._cached: Optional[SessionState] = None
        self._last_monotonic: Optional[float] = None
        self._cached_signature = None

    def _signature(self):
        try:
            stat = self.path.lstat()
            return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size, stat.st_mode
        except FileNotFoundError:
            return None

    def load(self) -> SessionState:
        with self._lock:
            signature = self._signature()
            if self._cached is not None and signature == self._cached_signature:
                return self._cached
            self._cached_signature = signature
            self._last_monotonic = None
            if not self.path.exists() and not self.path.is_symlink():
                self._cached = SessionState(None, None, 0.0)
                return self._cached
            try:
                with private_binary_reader(self.path) as f:
                    raw = json.load(f)
                self._cached = SessionState(
                    session_started_at=raw.get("session_started_at"),
                    session_last_tick_at=raw.get("session_last_tick_at"),
                    accumulated_session_seconds=float(
                        raw.get("accumulated_session_seconds", 0.0)
                    ),
                    clock_id=str(raw.get("clock_id", "")),
                    baseline_session_seconds=float(raw.get(
                        "baseline_session_seconds", raw.get("accumulated_session_seconds", 0.0)
                    )),
                    last_tick_monotonic=raw.get("last_tick_monotonic"),
                    monotonic_identity=str(raw.get("monotonic_identity", "")),
                )
            except PrivateStorageError:
                raise
            except Exception:
                # Corrupt or unreadable -- start fresh, do not raise.
                self._cached = SessionState(None, None, 0.0)
            return self._cached

    def save(self, state: SessionState) -> None:
        with self._lock:
            write_private_text(self.path, json.dumps(asdict(state), indent=2))
            self._cached = state
            self._cached_signature = self._signature()

    def invalidate(self) -> None:
        """Force re-read on next load() (used after external writes)."""
        with self._lock:
            self._cached = None
            self._last_monotonic = None


# ---------------------------------------------------------------------------
# Module-level store singleton (lazy, persist_dir-bound)
# ---------------------------------------------------------------------------

_stores: Dict[str, _SessionStore] = {}
_stores_lock = threading.Lock()


def _store_for(persist_dir: str | os.PathLike) -> _SessionStore:
    from .private_storage import ensure_private_directory
    selected = Path(persist_dir)
    if selected.exists() or selected.is_symlink():
        ensure_private_directory(selected)
    key = str(Path(persist_dir).resolve())
    with _stores_lock:
        s = _stores.get(key)
        if s is None:
            s = _SessionStore(persist_dir)
            _stores[key] = s
        return s


def _resolve_persist_dir(persist_dir: Optional[str | os.PathLike]) -> str:
    """Resolve a persist dir, falling back to the instinct_data default."""
    if persist_dir is not None and str(persist_dir) != "":
        return str(persist_dir)
    from .service_binding import bound_store
    isolated = bound_store()
    if isolated is not None:
        return str(isolated.persist_dir)
    from . import _default_persist_dir  # local import to avoid cycle
    return _default_persist_dir()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def state(persist_dir: Optional[str | os.PathLike] = None) -> SessionState:
    """Return a snapshot of the current session state (no side effects)."""
    pd = _resolve_persist_dir(persist_dir)
    return _store_for(pd).load()


def accumulated_session_hours(
    persist_dir: Optional[str | os.PathLike] = None,
) -> float:
    """Total session-hours accumulated so far (the only number timed
    effects should ever look at)."""
    return float(state(persist_dir).accumulated_session_seconds) / 3600.0


def session_now(persist_dir: Optional[str | os.PathLike] = None) -> datetime:
    """Return 'now' in the session clock: the moment in real time
    represented by the accumulator plus the start of the active session.

    Conceptually, the session clock is a duration counter. It does not
    have its own 'date' the way wall clock does -- it has a duration.
Returning a datetime anchored to `session_started_at` is a compatibility
    representation only. Do not subtract real memory timestamps from it;
    use per-memory accumulator snapshots for elapsed active time."""
    s = state(persist_dir)
    if s.session_started_at is None:
        # No session has ever started. The session clock is effectively
        # the moment of creation. Return a real-time now so callers that
        # diff against segment timestamps still get a sensible (zero or
        # small) age.
        return datetime.now()
    try:
        started = datetime.fromisoformat(s.session_started_at)
    except Exception:
        return datetime.now()
    return started + timedelta(seconds=float(s.accumulated_session_seconds))


def reset_session(persist_dir: Optional[str | os.PathLike] = None) -> SessionState:
    """Test/debug helper: zero the session clock entirely."""
    pd = _resolve_persist_dir(persist_dir)
    fresh = SessionState(None, None, 0.0, uuid.uuid4().hex)
    _store_for(pd)._last_monotonic = None
    _store_for(pd).save(fresh)
    return fresh


def tick_session(
    persist_dir: Optional[str | os.PathLike] = None,
    idle_minutes: float = DEFAULT_SESSION_IDLE_MINUTES,
    now: Optional[datetime] = None,
) -> TickResult:
    """Commit activity using monotonic time; an explicit now is a test clock."""
    pd = _resolve_persist_dir(persist_dir)
    store = _store_for(pd)
    with store._lock:
        return _tick_session_locked(store, idle_minutes, now)


def _tick_session_locked(
    store: _SessionStore,
    idle_minutes: float,
    now: Optional[datetime],
) -> TickResult:
    """Advance the session clock by exactly one tick.

    Semantics:

    * Load current state.
    * If there is no active session (None last_tick_at, or the previous
      session is older than `idle_minutes` of wall-clock time), start a
      new one. The previous session's contribution is already in the
      accumulator; nothing else to do.
    * If a session is active, add (now - last_tick_at) seconds to the
      accumulator -- but only if that delta is <= idle_minutes worth
      of seconds. A delta larger than the idle window means the session
      should have already been closed; we treat it as if the user
      returned after a break (start a new session, add zero for this
      tick).
    * Persist atomically. Return a TickResult so callers (the MCP
      server, the dashboard) can log what happened.

    `now` is injectable so tests can drive time deterministically.
    """
    s = store.load()
    real_now = now if now is not None else datetime.now()
    idle_window_s = max(0.0, float(idle_minutes) * 60.0)
    clock_id = s.clock_id or uuid.uuid4().hex
    baseline = s.baseline_session_seconds if s.clock_id else s.accumulated_session_seconds
    monotonic_now = time.monotonic() if now is None else None
    resumed = now is None and store._last_monotonic is None

    # Closed session: previous session is finished, no active timer.
    if s.session_last_tick_at is None or s.session_started_at is None or resumed:
        new_state = SessionState(
            session_started_at=real_now.isoformat(),
            session_last_tick_at=real_now.isoformat(),
            accumulated_session_seconds=float(s.accumulated_session_seconds),
            clock_id=clock_id,
            baseline_session_seconds=baseline,
            last_tick_monotonic=monotonic_now,
            monotonic_identity=_monotonic_identity(),
        )
        store.save(new_state)
        store._last_monotonic = monotonic_now
        return TickResult(
            state=new_state,
            new_session_started=True,
            session_alive=True,
            seconds_added_to_accumulator=0.0,
            idle_seconds_before_this_tick=0.0,
        )

    # Existing session -- measure the gap since last tick.
    try:
        last_tick = datetime.fromisoformat(s.session_last_tick_at)
    except Exception:
        last_tick = real_now
    delta_s = max(0.0, (real_now - last_tick).total_seconds())
    if monotonic_now is not None:
        delta_s = max(0.0, monotonic_now - store._last_monotonic)

    if delta_s > idle_window_s:
        # Session expired between ticks. Start fresh; the old session's
        # time is already locked into the accumulator.
        new_state = SessionState(
            session_started_at=real_now.isoformat(),
            session_last_tick_at=real_now.isoformat(),
            accumulated_session_seconds=float(s.accumulated_session_seconds),
            clock_id=clock_id,
            baseline_session_seconds=baseline,
            last_tick_monotonic=monotonic_now,
            monotonic_identity=_monotonic_identity(),
        )
        store.save(new_state)
        store._last_monotonic = monotonic_now
        return TickResult(
            state=new_state,
            new_session_started=True,
            session_alive=True,
            seconds_added_to_accumulator=0.0,
            idle_seconds_before_this_tick=float(delta_s),
        )

    # Session is alive -- extend it.
    new_acc = float(s.accumulated_session_seconds) + delta_s
    new_state = SessionState(
        session_started_at=s.session_started_at,
        session_last_tick_at=real_now.isoformat(),
        accumulated_session_seconds=new_acc,
        clock_id=clock_id,
        baseline_session_seconds=baseline,
        last_tick_monotonic=monotonic_now,
        monotonic_identity=_monotonic_identity(),
    )
    store.save(new_state)
    store._last_monotonic = monotonic_now
    return TickResult(
        state=new_state,
        new_session_started=False,
        session_alive=True,
        seconds_added_to_accumulator=float(delta_s),
        idle_seconds_before_this_tick=float(delta_s),
    )


def session_alive(
    persist_dir: Optional[str | os.PathLike] = None,
    idle_minutes: float = DEFAULT_SESSION_IDLE_MINUTES,
    now: Optional[datetime] = None,
) -> bool:
    """Read-only predicate: is a session currently active?"""
    s = state(persist_dir)
    if s.session_last_tick_at is None:
        return False
    try:
        last = datetime.fromisoformat(s.session_last_tick_at)
    except Exception:
        return False
    idle_s = max(0.0, float(idle_minutes) * 60.0)
    if now is None:
        if s.monotonic_identity != _monotonic_identity() or s.last_tick_monotonic is None:
            return False
        gap = time.monotonic() - s.last_tick_monotonic
        return 0.0 <= gap <= idle_s
    return (now - last).total_seconds() <= idle_s


def session_elapsed_hours(
    anchor: Optional[datetime],
    persist_dir: Optional[str | os.PathLike] = None,
    *,
    anchor_seconds: Optional[float] = None,
    clock_id: Optional[str] = None,
) -> float:
    """Elapsed committed active time from a same-clock accumulator snapshot.

    A wall timestamp alone cannot reconstruct historical active time. Missing
    or foreign snapshots return zero rather than charging unrelated activity.
    """
    if anchor is None or anchor_seconds is None:
        return 0.0
    s = state(persist_dir)
    if not s.clock_id or clock_id != s.clock_id:
        return 0.0
    return max(0.0, s.accumulated_session_seconds - max(0.0, anchor_seconds)) / 3600.0
