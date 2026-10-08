# SPDX-License-Identifier: MPL-2.0
"""Synthetic writable reopen, writer exclusion and process-crash checks."""

import os
import queue
import sqlite3
import subprocess
import sys
import threading

import pytest

from instinct.private_storage import private_file_lock, write_private_text
from instinct.sqlite_guard import WRITER_LOCK_FILENAME, SQLiteWriterUnavailable
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_store import SQLiteRecordStore


@pytest.fixture
def records(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    root = tmp_path / 'sqlite'
    record = SQLiteRecord.from_dict({
        'id': 'note', 'content': 'Synthetic writer fixture', 'embedding': [1, 0],
        'metadata': {'importance_score': 80, 'owner_extra': {'keep': True}},
    }, identity)
    with SQLiteRecordStore.create(root, identity) as writer:
        writer.insert_records([record])
    return root, identity, record


def test_reopen_commits_and_preserves_vector_and_unmentioned_metadata(records):
    root, identity, record = records
    with SQLiteRecordStore.open_writable(root, identity) as writer:
        assert not writer.read_only
        assert writer._connection.execute('PRAGMA synchronous').fetchone() == (2,)
        assert writer._connection.execute('PRAGMA journal_mode').fetchone() == ('delete',)
        writer.patch_metadata('note', {'importance_score': 79})
    with SQLiteRecordStore.open_read_only(root, identity) as reader:
        restored = reader.get_record('note')
        assert restored.embedding == record.embedding
        assert restored.content == record.content
        assert restored.to_dict()['metadata'] == {'importance_score': 79, 'owner_extra': {'keep': True}}


def test_only_one_writer_but_readers_can_observe_committed_records(records):
    root, identity, record = records
    with SQLiteRecordStore.open_writable(root, identity) as writer:
        with pytest.raises(SQLiteWriterUnavailable):
            SQLiteRecordStore.open_writable(root, identity)
        with SQLiteRecordStore.open_read_only(root, identity) as reader:
            assert reader.get_record('note') == record
            writer.patch_metadata('note', {'importance_score': 75})
            assert reader.get_record('note').to_dict()['metadata']['importance_score'] == 75


def test_new_store_creator_owns_writer_guard(tmp_path):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    root = tmp_path / 'new'
    with SQLiteRecordStore.create(root, identity):
        with pytest.raises(SQLiteWriterUnavailable):
            SQLiteRecordStore.open_writable(root, identity)
    with SQLiteRecordStore.open_writable(root, identity) as writer:
        assert writer.count() == 0


def test_close_is_idempotent_and_never_replaces_or_deletes_lock(records):
    root, identity, _ = records
    lock = root / WRITER_LOCK_FILENAME
    before = lock.stat()
    writer = SQLiteRecordStore.open_writable(root, identity)
    writer.close()
    writer.close()
    with SQLiteRecordStore.open_writable(root, identity):
        after = lock.stat()
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert lock.exists()


def test_wrong_identity_releases_guard_and_does_not_change_database(records):
    root, identity, _ = records
    path = root / 'memory.sqlite3'
    before = path.read_bytes()
    wrong = EmbeddingIdentity('synthetic', 'wrong', 2, 'cosine')
    with pytest.raises(ValueError, match='identity'):
        SQLiteRecordStore.open_writable(root, wrong)
    assert path.read_bytes() == before
    with SQLiteRecordStore.open_writable(root, identity) as writer:
        assert writer.count() == 1


def test_foreign_database_is_rejected_before_writable_connection(records, tmp_path):
    _, identity, _ = records
    root = tmp_path / 'foreign'
    root.mkdir(mode=0o700)
    path = root / 'memory.sqlite3'
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE foreign_table (value TEXT)')
    connection.commit()
    connection.close()
    path.chmod(0o600)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Foreign'):
        SQLiteRecordStore.open_writable(root, identity)
    assert path.read_bytes() == before
    assert not (root / WRITER_LOCK_FILENAME).exists()


def test_unknown_schema_is_rejected_without_schema_repair(records):
    root, identity, _ = records
    with SQLiteRecordStore.open_writable(root, identity) as writer:
        writer._connection.execute('CREATE TABLE unexpected (value TEXT)')
    path = root / 'memory.sqlite3'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='schema'):
        SQLiteRecordStore.open_writable(root, identity)
    assert path.read_bytes() == before
    with private_file_lock(root / WRITER_LOCK_FILENAME, timeout=0):
        pass


def test_missing_store_is_not_created(records):
    root, identity, _ = records
    missing = root / 'missing'
    with pytest.raises(FileNotFoundError):
        SQLiteRecordStore.open_writable(missing, identity)
    assert not missing.exists()


@pytest.mark.parametrize('timeout', [-1, True, float('nan'), float('inf'), 6, '1'])
def test_invalid_writer_timeout_does_not_acquire_guard(records, timeout):
    root, identity, _ = records
    with pytest.raises(ValueError, match='timeout'):
        SQLiteRecordStore.open_writable(root, identity, writer_timeout=timeout)
    with SQLiteRecordStore.open_writable(root, identity):
        pass


def test_wal_sidecars_are_rejected_not_removed(records):
    root, identity, _ = records
    sidecar = root / 'memory.sqlite3-wal'
    write_private_text(sidecar, 'Synthetic unsupported sidecar')
    with pytest.raises(ValueError, match='WAL'):
        SQLiteRecordStore.open_writable(root, identity)
    assert sidecar.read_text() == 'Synthetic unsupported sidecar'


def test_wal_header_is_rejected_without_converting_journal_mode(records):
    root, identity, _ = records
    connection = sqlite3.connect(root / 'memory.sqlite3')
    assert connection.execute('PRAGMA journal_mode=WAL').fetchone() == ('wal',)
    connection.close()
    before = (root / 'memory.sqlite3').read_bytes()
    with pytest.raises(ValueError, match='rollback'):
        SQLiteRecordStore.open_writable(root, identity)
    assert (root / 'memory.sqlite3').read_bytes() == before


def test_failed_creation_releases_lifetime_guard(tmp_path, monkeypatch):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    root = tmp_path / 'failed'

    def fail(*arguments, **options):
        raise RuntimeError('Synthetic connect failure')

    monkeypatch.setattr(SQLiteRecordStore, '_connect', staticmethod(fail))
    with pytest.raises(RuntimeError, match='Synthetic'):
        SQLiteRecordStore.create(root, identity)
    with private_file_lock(root / WRITER_LOCK_FILENAME, timeout=0):
        pass


def test_failed_schema_initialization_releases_guard_even_for_closed_connection(tmp_path, monkeypatch):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    root = tmp_path / 'failed-schema'
    connection = sqlite3.connect(':memory:')
    connection.close()
    monkeypatch.setattr(SQLiteRecordStore, '_connect', staticmethod(lambda *args, **kwargs: connection))
    with pytest.raises(sqlite3.ProgrammingError):
        SQLiteRecordStore.create(root, identity)
    with private_file_lock(root / WRITER_LOCK_FILENAME, timeout=0):
        pass


CHILD_IDENTITY = "identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')"


def test_competing_process_fails_fast_without_changing_records(records):
    root, identity, record = records
    code = '''
from pathlib import Path
import sys
from instinct.sqlite_guard import SQLiteWriterUnavailable
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_store import SQLiteRecordStore
''' + CHILD_IDENTITY + '''
try:
    with SQLiteRecordStore.open_writable(Path(sys.argv[1]), identity):
        raise RuntimeError('Unexpected writer acquisition')
except SQLiteWriterUnavailable:
    print('blocked')
'''
    with SQLiteRecordStore.open_writable(root, identity) as writer:
        child = subprocess.run([sys.executable, '-c', code, str(root)], capture_output=True,
                               text=True, check=True, timeout=15)
        assert child.stdout.strip() == 'blocked'
        assert writer.get_record('note') == record


def test_crashed_writer_releases_os_lock_and_rolls_back_hot_journal(records):
    root, identity, record = records
    with SQLiteRecordStore.open_writable(root, identity) as writer:
        writer.insert_records([SQLiteRecord.from_dict({
            'id': f'fixture-{index}', 'content': 'original ' * 100,
            'metadata': {'importance_score': 70}, 'embedding': [1, 0],
        }, identity) for index in range(200)])
    lock_before = (root / WRITER_LOCK_FILENAME).stat()
    code = '''
from pathlib import Path
import sys
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_store import SQLiteRecordStore
''' + CHILD_IDENTITY + '''
with SQLiteRecordStore.open_writable(Path(sys.argv[1]), identity) as writer:
    writer.patch_metadata('note', {'committed_before_crash': True})
    writer._connection.execute('PRAGMA cache_size=1')
    writer._connection.execute('BEGIN IMMEDIATE')
    writer._connection.execute('UPDATE records SET content=?', ('uncommitted ' * 1000,))
    print('ready', flush=True)
    sys.stdin.read()
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(root)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        lines = queue.Queue()
        thread = threading.Thread(target=lambda: lines.put(child.stdout.readline()), daemon=True)
        thread.start()
        assert lines.get(timeout=15).strip() == 'ready'
        assert (root / 'memory.sqlite3-journal').exists()
        child.kill()
        child.communicate(timeout=15)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=15)
    with SQLiteRecordStore.open_writable(root, identity) as recovered:
        restored = recovered.get_record('note')
        assert restored.content == record.content
        assert restored.embedding == record.embedding
        assert restored.to_dict()['metadata']['committed_before_crash'] is True
        assert recovered.count() == 201
        assert recovered._connection.execute('PRAGMA integrity_check').fetchone() == ('ok',)
    lock_after = (root / WRITER_LOCK_FILENAME).stat()
    assert (lock_after.st_dev, lock_after.st_ino) == (lock_before.st_dev, lock_before.st_ino)
    assert not (root / 'memory.sqlite3-journal').exists()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permission semantics')
def test_writer_lock_is_owner_private(records):
    root, identity, _ = records
    with SQLiteRecordStore.open_writable(root, identity):
        assert (root / WRITER_LOCK_FILENAME).stat().st_mode & 0o777 == 0o600
        assert root.stat().st_mode & 0o777 == 0o700
