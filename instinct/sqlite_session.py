# SPDX-License-Identifier: MPL-2.0
"""Experimental ownership of an isolated SQLite database and runtime sidecars."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import threading

from . import nudge, session_timer
from .private_storage import ensure_private_directory, private_file_lock
from .sqlite_guard import SQLiteWriterUnavailable
from .sqlite_runtime import SQLiteSegmentAdapter
from .sqlite_store import SQLiteRecordStore


RUNTIME_LOCK_FILENAME = '.sqlite.runtime.lock'


@dataclass(frozen=True)
class SQLiteActivityTick:
    session: session_timer.TickResult
    current_turn: int


class SQLiteRuntimeSession:
    """Thread-confined owner/observer, without service or backend selection."""

    runtime_ready = False

    def __init__(self, adapter, guard):
        self._adapter = adapter
        self._guard = guard
        self._thread_id = threading.get_ident()
        self._closed = False
        self._operation_active = False

    @classmethod
    def open(cls, database_dir, identity, *, persist_dir, read_only,
             query_embedder=None, document_embedder=None, writer_timeout=0):
        if type(read_only) is not bool:
            raise ValueError('An explicit owner or observer role is required')
        if type(writer_timeout) not in (int, float) or not math.isfinite(writer_timeout) or not 0 <= writer_timeout <= 5:
            raise ValueError('Writer timeout must be a finite value from 0 to 5 seconds')
        root = cls._existing_directory(persist_dir)
        database = cls._existing_directory(database_dir)
        if not database.is_relative_to(root):
            raise ValueError('SQLite database must be inside its isolated runtime directory')
        guard = None
        repository = None
        try:
            if not read_only:
                guard = private_file_lock(root / RUNTIME_LOCK_FILENAME, timeout=writer_timeout)
                try:
                    guard.__enter__()
                except TimeoutError:
                    guard = None
                    raise SQLiteWriterUnavailable('SQLite runtime sidecars are owned by another session') from None
                repository = SQLiteRecordStore.open_writable(database, identity, writer_timeout=writer_timeout)
            else:
                repository = SQLiteRecordStore.open_read_only(database, identity)
            adapter = SQLiteSegmentAdapter(repository, persist_dir=root,
                query_embedder=query_embedder, document_embedder=document_embedder)
            adapter.database_dir = database
            if not read_only:
                session_timer._store_for(root).invalidate()
            return cls(adapter, guard)
        except BaseException:
            try:
                if repository is not None:
                    repository.close()
            finally:
                if guard is not None:
                    guard.__exit__(None, None, None)
            raise

    @staticmethod
    def _existing_directory(value):
        if not isinstance(value, (str, Path)):
            raise ValueError('An explicit existing absolute runtime directory is required')
        path = Path(value)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError('An explicit existing absolute runtime directory is required')
        return ensure_private_directory(path).resolve(strict=True)

    def _check_open(self):
        if threading.get_ident() != self._thread_id:
            raise RuntimeError('SQLite runtime sessions are thread-confined')
        if self._closed:
            raise RuntimeError('SQLite runtime session is closed')

    @property
    def read_only(self):
        self._check_open()
        return self._adapter._read_only

    @contextmanager
    def operation(self):
        """Keep complete adapter work inside the owning synchronous thread."""
        self._check_open()
        if self._operation_active:
            raise RuntimeError('Nested SQLite runtime operations are not supported')
        self._operation_active = True
        try:
            yield self._adapter
        finally:
            self._operation_active = False

    def tick_activity(self, *, idle_minutes=session_timer.DEFAULT_SESSION_IDLE_MINUTES, now=None):
        if type(idle_minutes) not in (int, float) or not math.isfinite(idle_minutes) or idle_minutes < 0:
            raise ValueError('Idle window must be finite and nonnegative')
        if now is not None and (not isinstance(now, datetime) or now.tzinfo is not None):
            raise ValueError('Injected activity time must be a naive datetime')
        with self.operation() as adapter:
            result = nudge.tick_nudge_session(adapter, idle_minutes=idle_minutes, now=now)
            turn = nudge.get_current_turn(adapter) if adapter._read_only else nudge.tick_turn(adapter)
            return SQLiteActivityTick(result, turn)

    def replay_pending(self):
        self._check_open()
        raise NotImplementedError('SQLite queued writes require a reviewed replay protocol; no queue was read or changed')

    def close(self):
        if self._closed:
            return
        self._check_open()
        if self._operation_active:
            raise RuntimeError('Cannot close an active SQLite runtime operation')
        self._closed = True
        try:
            self._adapter.repository.close()
        finally:
            if self._guard is not None:
                self._guard.__exit__(None, None, None)
                self._guard = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
