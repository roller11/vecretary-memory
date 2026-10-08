# SPDX-License-Identifier: MPL-2.0
"""
Chroma writer lock — single-writer enforcement for the Instinct ChromaDB store.

Root cause this fixes
---------------------
ChromaDB's PersistentClient opens ``chroma.sqlite3`` directly. SQLite's
default locking is "processes must serialize". When two Python processes
(dashboard, MCP server, probe script) all point at the same
``instinct_data/`` directory, the second opener blocks on the OS file
lock. On WSL2 that blocking surfaces as the second process pinned in
``Dl`` (uninterruptible disk wait) in ``ps`` and a frozen ``/api/...``
request from the user's perspective.

This module gates ``InstinctStore.__init__`` (and any other entry point
that opens chroma) behind a POSIX advisory file lock
(``fcntl.flock``) on ``<persist_dir>/.writer.lock``. The first process
gets the lock; the second one waits up to a timeout, then either
acquires it (the first process exited) or raises
``ChromaWriterLockUnavailable`` with a clear, actionable message.

Design notes
------------
* POSIX ``flock`` is tied to an open file description. Same-process
  acquisitions share a registered descriptor, with registry operations
  serialized so competing threads cannot open separate writer handles.
* The lock is released automatically when the process exits or the
  file descriptor is closed, including abrupt kills — ``flock`` is
  released by the kernel, not by user-space cleanup. So a ``kill -9``
  of the holder does not strand the lock; the next opener gets it
  immediately.
* We hold the lock for the lifetime of the process (kept in a module-
  level global ``_HELD_LOCKS``) — that matches the "one writer process
  at a time" rule and also lets /api/health report "you are the
  primary writer" without a second SQLite call.
* We deliberately do **not** put the lock inside the chromadb library
  itself; this is a host-process concern, not a library concern. The
  dashboard, MCP server, and probe scripts are all host processes.
* The lock file lives next to ``chroma.sqlite3`` so a single
  ``persist_dir`` is self-contained — moving the directory moves the
  lock with it.
* The lock file is never removed for stale PID metadata. Only the OS
  lock determines ownership; replacing a locked inode could admit a
  second writer. Holder metadata is published only after acquisition.

Public API
----------
* ``acquire_writer_lock(persist_dir, timeout=30.0) -> WriterLockHandle``
* ``release_writer_lock(handle)``
* ``is_writer_lock_held(persist_dir) -> bool``
* ``ChromaWriterLockUnavailable`` exception
"""
from __future__ import annotations

import errno
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# --- Cross-platform advisory file lock ----------------------------------
# POSIX (Linux/macOS/WSL):  fcntl.flock(LOCK_EX|LOCK_NB|LOCK_UN)
# Windows native Python:    msvcrt.locking(LK_NBLCK|LK_UNLCK) on a single
#                           byte. Limited (only one byte range per fd) but
#                           sufficient for a tiny sentinel file we control.
#
# Fallback rule: if fcntl isn't importable (Windows native Python), use
# msvcrt. If neither is available (e.g., exotic embedded Python), raise
# a clear error at lock-acquisition time rather than crashing the import.
try:
    import fcntl as _fcntl  # POSIX
    _LOCK_BACKEND = "posix"

    def _lock_ex_nb(fd: int) -> None:
        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)

    def _lock_sh_nb(fd: int) -> None:
        _fcntl.flock(fd, _fcntl.LOCK_SH | _fcntl.LOCK_NB)

    def _lock_un(fd: int) -> None:
        _fcntl.flock(fd, _fcntl.LOCK_UN)
except ImportError:
    try:
        import msvcrt as _msvcrt  # Windows native
        _LOCK_BACKEND = "msvcrt"

        def _lock_ex_nb(fd: int) -> None:
            # Seek to byte 0 and try to lock 1 byte exclusively.
            os.lseek(fd, 0, os.SEEK_SET)
            _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)

        def _lock_sh_nb(fd: int) -> None:
            # msvcrt has no shared lock; treat as advisory-only and try
            # to acquire exclusive briefly. We only use SH_NB for the
            # cheap probe in is_writer_lock_held(), so close-enough.
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
                return
            except OSError as e:
                # errno 13 (permission denied) => held by another process
                # errno 36 (resource deadlock) => we already hold it
                if getattr(e, "errno", None) in (13, 36):
                    raise
                raise

        def _lock_un(fd: int) -> None:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    except ImportError:
        _LOCK_BACKEND = None  # set in _require_lock_backend()

        def _lock_ex_nb(fd: int) -> None:
            _require_lock_backend()

        def _lock_sh_nb(fd: int) -> None:
            _require_lock_backend()

        def _lock_un(fd: int) -> None:
            pass


def _require_lock_backend() -> None:
    if _LOCK_BACKEND is None:
        raise RuntimeError(
            "Instinct writer lock requires either fcntl (POSIX) or "
            "msvcrt (Windows). Neither module is available on this "
            "platform. Use read_only=True to observe without writing."
        )


def current_lock_backend() -> str:
    """Diagnostic: which advisory-lock backend is active."""
    return _LOCK_BACKEND or "unavailable"

# Module-level registry of held locks. Keyed by absolute persist_dir so a
# process that opens multiple persist_dirs (tests, etc.) doesn't trip on
# itself. Values are the WriterLockHandle; the OS file descriptor inside
# is what actually holds the flock.
_HELD_LOCKS: Dict[str, "WriterLockHandle"] = {}
_REGISTRY_LOCK = threading.RLock()


class ChromaWriterLockUnavailable(RuntimeError):
    """Raised when another process holds the chroma writer lock and the
    timeout elapses before it is released.

    The error message includes the PID of the holder (best effort) so the
    user can decide whether to wait, kill the holder, or accept that
    only one writer is allowed.
    """

    def __init__(self, persist_dir: str, holder_pid: Optional[int], waited: float):
        self.persist_dir = persist_dir
        self.holder_pid = holder_pid
        self.waited = waited
        msg = (
            f"Another Instinct writer holds the chroma lock for "
            f"{persist_dir} (holder pid={holder_pid or 'unknown'}; "
            f"waited {waited:.1f}s). Close the other process "
            f"(dashboard, MCP server, or probe) and retry, or pass a "
            f"larger timeout to acquire_writer_lock()."
        )
        super().__init__(msg)


@dataclass
class WriterLockHandle:
    """Opaque handle returned by acquire_writer_lock(). Hold for the
    process lifetime; release explicitly only if you need to drop the
    lock without exiting, such as a controlled dashboard handoff."""

    persist_dir: str
    lock_path: str
    fd: int
    pid: int

    def release(self) -> None:
        release_writer_lock(self)


# --- core ----------------------------------------------------------------

def _lock_path(persist_dir: str) -> str:
    """Path of the sentinel lock file. Created on demand, never deleted
    (the OS releases the flock on fd close, not file unlink)."""
    p = Path(persist_dir)
    p.mkdir(parents=True, exist_ok=True)
    return str(p / ".writer.lock")


def _is_pid_alive(pid: int) -> bool:
    """Best-effort: is ``pid`` currently a running process?

    On POSIX: signal 0 succeeds if the process exists.
    On Windows: OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION.
    Returns False if the PID is None, 0, or doesn't exist.
    """
    if not pid or pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            # Permission denied => process exists but we can't signal it
            return True
        except OSError:
            return False
    # Windows native: use OpenProcess
    try:
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid)
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def _holder_pid_from_lockfile(lock_path: str) -> Optional[int]:
    """Best-effort read of the holder PID stashed in the lock file.

    The first 8 bytes of the file are the ASCII PID of the process that
    currently holds the lock. This is purely informational; the actual
    lock state is in the kernel's flock table, not in the file
    contents. The file's content is only useful for the error message
    so the user knows which process is in the way.
    """
    try:
        with open(lock_path, "r") as f:
            head = f.read(16).strip()
        if head.isdigit():
            return int(head)
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return None


def acquire_writer_lock(persist_dir: str, timeout: float = 30.0, *, require_new: bool = False) -> WriterLockHandle:
    """Block (up to ``timeout`` seconds) until this process can become
    the sole writer for ``persist_dir``, then return a handle.

    Idempotent: calling twice from the same process with the same
    ``persist_dir`` returns the existing handle without re-acquiring.
    PID metadata is diagnostic only and cannot override the OS lock.
    ``require_new`` refuses an existing same-process owner atomically.
    """
    with _REGISTRY_LOCK:
        if require_new and str(Path(persist_dir).resolve()) in _HELD_LOCKS:
            raise ChromaWriterLockUnavailable(persist_dir, os.getpid(), 0.0)
        return _acquire_writer_lock(persist_dir, timeout)


def _acquire_writer_lock(persist_dir: str, timeout: float) -> WriterLockHandle:
    abs_dir = str(Path(persist_dir).resolve())
    existing = _HELD_LOCKS.get(abs_dir)
    if existing is not None:
        return existing

    lock_path = _lock_path(abs_dir)

    # Open the lock file, retrying briefly if the filesystem is mid-
    # transition (DrvFs/WSL2 occasionally serves a "no such file" for
    # a deleted-but-still-open inode on a path we just created).
    # Special case: dot-prefixed files (".writer.lock") sometimes
    # refuse to be created via os.open(O_CREAT) on DrvFS — the path
    # resolves to "file not found" because the dotfile is treated as
    # hidden. Pre-touching via Path.touch() bypasses that quirk.
    # Up to 5 attempts over ~0.5s, then raise the original error.
    fd = None
    last_err = None
    for attempt in range(5):
        try:
            # Ensure parent exists (idempotent).
            Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
            # Pre-touch the dotfile. On DrvFS this is the only reliable
            # way to create a dot-prefixed file before os.open(O_CREAT).
            if not os.path.exists(lock_path):
                try:
                    Path(lock_path).touch(exist_ok=True)
                except OSError as e:
                    # If touch itself fails, fall through to os.open
                    # and let it raise so we surface the real error.
                    pass
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            break
        except FileNotFoundError as e:
            last_err = e
            time.sleep(0.1 * (attempt + 1))
        except OSError as e:
            # Any other OS-level failure (permission, transient
            # DrvFS weirdness). Treat like FileNotFoundError and retry.
            last_err = e
            time.sleep(0.1 * (attempt + 1))
    if fd is None:
        # Wrap in a typed exception so callers can distinguish
        # "writer lock not acquirable" from generic OS errors. The
        # dashboard's get_store() catches ChromaWriterLockUnavailable
        # and falls back to read-only mode; raw FileNotFoundError
        # would crash it.
        raise ChromaWriterLockUnavailable(
            f"could not open writer lock at {lock_path} after retries: {last_err}",
            holder_pid=None, waited=0.0,
        ) from last_err

    deadline = time.monotonic() + timeout
    poll = 0.05  # 50ms — fast enough to feel snappy, slow enough not to spin
    waited = 0.0
    while True:
        try:
            _lock_ex_nb(fd)
            break  # we hold it
        except OSError as e:
            err_no = getattr(e, "errno", None)
            # Treat EWOULDBLOCK / EAGAIN (POSIX) and EBUSY/EACCES/EDEADLK
            # (Windows msvcrt) as "someone else holds it". Anything else
            # is a real error.
            if _LOCK_BACKEND == "posix":
                if err_no not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    os.close(fd)
                    raise
            else:
                # msvcrt surfaces "already locked" as PermissionError
                # (errno 13) or OSError(errno 36, 'Resource deadlock').
                # Both mean "someone else holds it".
                if err_no not in (errno.EAGAIN, errno.EBUSY,
                                  errno.EACCES, 13, 36):
                    os.close(fd)
                    raise
            # Someone else holds it. Wait, then retry.
            if time.monotonic() >= deadline:
                holder = _holder_pid_from_lockfile(lock_path)
                os.close(fd)
                raise ChromaWriterLockUnavailable(abs_dir, holder, waited)
            time.sleep(poll)
            waited += poll
            # Linear backoff capped at 0.5s — keeps the polling rate
            # reasonable over a 30s wait.
            poll = min(poll * 1.2, 0.5)

    try:
        payload = f"{os.getpid()}\n".encode("ascii")
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.ftruncate(fd, len(payload))
    except OSError:
        pass

    handle = WriterLockHandle(
        persist_dir=abs_dir,
        lock_path=lock_path,
        fd=fd,
        pid=os.getpid(),
    )
    _HELD_LOCKS[abs_dir] = handle
    return handle


def release_writer_lock(handle: WriterLockHandle) -> None:
    """Release a held writer lock. Safe to call on an already-released
    handle; an old handle cannot release a replacement writer. Process
    exit also releases the OS lock without deleting its sentinel."""
    with _REGISTRY_LOCK:
        if _HELD_LOCKS.get(handle.persist_dir) is not handle:
            return
        _HELD_LOCKS.pop(handle.persist_dir)
        try:
            _lock_un(handle.fd)
        except OSError:
            pass
        try:
            os.close(handle.fd)
        except OSError:
            pass


def is_writer_lock_held(persist_dir: str) -> bool:
    """True if *any* process (this one or another) currently holds the
    writer lock for ``persist_dir``. Cheap probe — opens the lock
    file, attempts a non-blocking shared lock, releases it. Does not
    block."""
    abs_dir = str(Path(persist_dir).resolve())
    if abs_dir in _HELD_LOCKS:
        return True
    lock_path = _lock_path(abs_dir)
    if not os.path.exists(lock_path):
        return False
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        try:
            _lock_sh_nb(fd)
            held = False
        except OSError:
            held = True
        try:
            _lock_un(fd)
        except OSError:
            pass
    except OSError as e:
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            held = True
        else:
            held = False
    finally:
        os.close(fd)
    return held


@contextmanager
def writer_lock(persist_dir: str, timeout: float = 30.0):
    """Context-manager form for one-off uses. Most callers should use
    acquire_writer_lock() and hold for the process lifetime."""
    handle = acquire_writer_lock(persist_dir, timeout=timeout)
    try:
        yield handle
    finally:
        # We do NOT release on context exit if this process is the
        # registered holder for the dir — releasing would let the next
        # opener in (defeating the lifetime guarantee). Use
        # acquire_writer_lock + handle.release() if you really need a
        # scoped lock.
        pass


# --- diagnostics --------------------------------------------------------

def describe_lock(persist_dir: str) -> dict:
    """Return a dict suitable for /api/health or logs. Never raises."""
    abs_dir = str(Path(persist_dir).resolve())
    lock_path = _lock_path(abs_dir)
    info = {
        "persist_dir": abs_dir,
        "lock_path": lock_path,
        "lock_file_exists": os.path.exists(lock_path),
        "held_by_us": abs_dir in _HELD_LOCKS,
        "holder_pid": _holder_pid_from_lockfile(lock_path),
        "platform": sys.platform,
    }
    return info
