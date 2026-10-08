# SPDX-License-Identifier: MPL-2.0
"""Private local paths without silently repairing existing owner data."""

from contextlib import contextmanager
import os
from pathlib import Path
import stat
import tempfile
import time


class PrivateStorageError(PermissionError):
    pass


def _check_directory(path, *, private=False):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise PrivateStorageError('Storage paths must be directories, not symlinks.')
    if os.name == 'posix':
        if info.st_uid not in (os.geteuid(), 0):
            raise PrivateStorageError('Storage ancestor is owned by another user.')
        if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            raise PrivateStorageError('Storage ancestor is writable by other users.')
        if private and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
            raise PrivateStorageError('Existing storage directory is not owner-private; review permissions explicitly or choose a new private directory.')


def ensure_private_directory(path):
    """Create missing directories as 0700; reject unsafe existing paths unchanged."""
    expanded = Path(os.path.expanduser(os.fspath(path)))
    if '..' in expanded.parts:
        raise PrivateStorageError('Storage paths must not contain parent traversal.')
    target = expanded.absolute()
    for component in [*reversed(target.parents), target]:
        try:
            _check_directory(component, private=component == target)
        except FileNotFoundError:
            try:
                component.mkdir(mode=0o700)
            except FileExistsError:
                pass
            _check_directory(component, private=True)
    return target


def check_private_file(path, *, confidential=True):
    """Reject links, foreign ownership and exposed application-owned files."""
    target = Path(path)
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PrivateStorageError('Storage files must be regular unlinked files, not symlinks or hardlinks.')
    if os.name == 'posix' and (info.st_uid != os.geteuid() or confidential and info.st_mode & 0o077):
        raise PrivateStorageError('Existing storage file is not owner-private; review permissions explicitly.')
    return info


@contextmanager
def private_binary_reader(path, *, confidential=True):
    target = Path(path)
    ensure_private_directory(target.parent)
    before = check_private_file(target, confidential=confidential)
    descriptor = os.open(target, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    with os.fdopen(descriptor, 'rb') as source:
        actual = os.fstat(source.fileno())
        if (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino):
            raise PrivateStorageError('Storage file changed during open.')
        yield source


@contextmanager
def atomic_private_writer(path, *, replace=True):
    """Publish a flushed 0600 file; do not follow or chmod existing targets."""
    target = Path(path)
    parent = ensure_private_directory(target.parent)
    if target.exists() or target.is_symlink():
        check_private_file(target)
        if not replace:
            raise FileExistsError('Refusing to overwrite an existing private record.')
    descriptor, temporary = tempfile.mkstemp(dir=parent, prefix='.instinct-', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'wb') as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        if replace:
            if target.exists() or target.is_symlink():
                check_private_file(target)
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            os.unlink(temporary)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_private_text(path, text, *, replace=True):
    with atomic_private_writer(path, replace=replace) as output:
        output.write(text.encode('utf-8'))


@contextmanager
def private_file_lock(path, timeout=2):
    target = Path(path)
    ensure_private_directory(target.parent)
    descriptor = os.open(target, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0), 0o600)
    acquired = False
    try:
        before = check_private_file(target)
        actual = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (actual.st_dev, actual.st_ino):
            raise PrivateStorageError('Lock file changed during open.')
        deadline = time.monotonic() + timeout
        while not acquired:
            try:
                if os.name == 'nt':
                    import msvcrt
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Private queue is busy; capture not accepted.') from None
                time.sleep(0.01)
        yield
    finally:
        try:
            if acquired:
                if os.name == 'nt':
                    import msvcrt
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
