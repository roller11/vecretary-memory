# SPDX-License-Identifier: MPL-2.0
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os
import threading
import time

import pytest

from instinct import chroma_writer_lock as locks


def hold_worker(directory, connection, diagnostic_pid):
    handle = None
    try:
        handle = locks.acquire_writer_lock(directory, timeout=1)
        if diagnostic_pid is not None:
            payload = f'{diagnostic_pid}\n'.encode('ascii')
            os.lseek(handle.fd, 0, os.SEEK_SET)
            os.write(handle.fd, payload)
            os.ftruncate(handle.fd, len(payload))
        connection.send({'pid': os.getpid(), 'inode': os.fstat(handle.fd).st_ino})
        connection.recv()
    except Exception as error:
        connection.send({'error': repr(error)})
    finally:
        if handle is not None:
            locks.release_writer_lock(handle)
        connection.close()


def probe_worker(directory, connection):
    handle = None
    try:
        handle = locks.acquire_writer_lock(directory, timeout=0.1)
        connection.send({'acquired': True, 'pid': os.getpid(), 'inode': os.fstat(handle.fd).st_ino})
    except locks.ChromaWriterLockUnavailable as error:
        connection.send({'acquired': False, 'holder_pid': error.holder_pid})
    except Exception as error:
        connection.send({'error': repr(error)})
    finally:
        if handle is not None:
            locks.release_writer_lock(handle)
        connection.close()


def stop_child(process):
    if process.is_alive():
        process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)
    assert not process.is_alive()


@contextmanager
def holder(directory, diagnostic_pid=None):
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=hold_worker, args=(str(directory), child, diagnostic_pid))
    process.start()
    child.close()
    try:
        assert parent.poll(30), 'Holder startup timed out'
        ready = parent.recv()
        assert 'error' not in ready, ready
        yield process, parent, ready
    finally:
        if process.is_alive():
            parent.send('release')
            process.join(timeout=10)
        stop_child(process)
        parent.close()


def probe(directory):
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=probe_worker, args=(str(directory), child))
    process.start()
    child.close()
    try:
        assert parent.poll(30), 'Contender timed out'
        result = parent.recv()
        assert 'error' not in result, result
        process.join(timeout=10)
        assert process.exitcode == 0
        return result
    finally:
        stop_child(process)
        parent.close()


def test_contender_preserves_real_holder_metadata(tmp_path):
    with holder(tmp_path) as (_, _, ready):
        result = probe(tmp_path)
        assert not result['acquired']
        if os.name == 'posix':
            assert result['holder_pid'] == ready['pid']
            assert locks.describe_lock(str(tmp_path))['holder_pid'] == ready['pid']
        else:
            assert result['holder_pid'] in (None, ready['pid'])
        assert locks.is_writer_lock_held(str(tmp_path))
    assert locks.describe_lock(str(tmp_path))['holder_pid'] == ready['pid']


@pytest.mark.skipif(os.name != 'posix', reason='POSIX inode and readable locked-file semantics')
def test_dead_diagnostic_pid_cannot_replace_live_locked_inode(tmp_path):
    with holder(tmp_path, diagnostic_pid=999999999999) as (_, _, ready):
        lock_path = tmp_path / '.writer.lock'
        original = lock_path.read_bytes()
        started = time.monotonic()
        result = probe(tmp_path)
        assert not result['acquired']
        assert result['holder_pid'] == 999999999999
        assert time.monotonic() - started < 30
        assert lock_path.stat().st_ino == ready['inode']
        assert lock_path.read_bytes() == original


def test_abrupt_holder_exit_releases_lock_without_replacing_file(tmp_path):
    with holder(tmp_path) as (process, _, ready):
        process.kill()
        process.join(timeout=10)
        result = probe(tmp_path)
        assert result['acquired']
        assert result['inode'] == ready['inode']
        assert (tmp_path / '.writer.lock').stat().st_ino == ready['inode']


@pytest.mark.skipif(os.name != 'posix', reason='POSIX open-file-description lock race')
def test_preopened_waiter_cannot_bypass_recovered_writer(tmp_path):
    with holder(tmp_path) as (process, _, ready):
        descriptor = os.open(tmp_path / '.writer.lock', os.O_RDWR)
        try:
            process.kill()
            process.join(timeout=10)
            with holder(tmp_path) as (_, _, recovered):
                assert recovered['inode'] == ready['inode']
                assert os.fstat(descriptor).st_ino == recovered['inode']
                with pytest.raises(OSError):
                    locks._lock_ex_nb(descriptor)
        finally:
            locks._lock_un(descriptor)
            os.close(descriptor)


def test_same_process_threads_share_one_writer_handle(tmp_path):
    barrier = threading.Barrier(6)

    def acquire():
        barrier.wait(timeout=10)
        return locks.acquire_writer_lock(str(tmp_path), timeout=0.1)

    handles = []
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            handles = list(executor.map(lambda index: acquire(), range(6)))
        assert all(handle is handles[0] for handle in handles)
        assert not probe(tmp_path)['acquired']
    finally:
        for handle in handles:
            locks.release_writer_lock(handle)
        registered = locks._HELD_LOCKS.get(str(tmp_path.resolve()))
        if registered is not None:
            locks.release_writer_lock(registered)


def test_old_handle_release_cannot_close_replacement_writer(tmp_path):
    first = locks.acquire_writer_lock(str(tmp_path), timeout=0)
    first.release()
    replacement = locks.acquire_writer_lock(str(tmp_path), timeout=0)
    try:
        first.release()
        assert locks._HELD_LOCKS[str(tmp_path.resolve())] is replacement
        assert locks.is_writer_lock_held(str(tmp_path))
        assert not probe(tmp_path)['acquired']
    finally:
        replacement.release()
        replacement.release()


@pytest.mark.parametrize('old_metadata', ['999999999999\n', 'invalid\n', ''])
def test_unlocked_stale_metadata_is_replaced_only_after_acquisition(tmp_path, old_metadata):
    lock_path = tmp_path / '.writer.lock'
    lock_path.write_text(old_metadata)
    inode = lock_path.stat().st_ino
    handle = locks.acquire_writer_lock(str(tmp_path), timeout=0)
    try:
        if os.name == 'posix':
            assert locks.describe_lock(str(tmp_path))['holder_pid'] == os.getpid()
        assert os.fstat(handle.fd).st_ino == inode
    finally:
        handle.release()
    assert locks.describe_lock(str(tmp_path))['holder_pid'] == os.getpid()


def test_lock_probe_never_rewrites_metadata(tmp_path):
    with holder(tmp_path) as (_, _, ready):
        assert locks.is_writer_lock_held(str(tmp_path))
        if os.name == 'posix':
            assert locks.describe_lock(str(tmp_path))['holder_pid'] == ready['pid']
    assert locks.describe_lock(str(tmp_path))['holder_pid'] == ready['pid']
