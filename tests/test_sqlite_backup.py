# SPDX-License-Identifier: MPL-2.0
"""Synthetic offline backup safety and lifecycle persistence tests."""

from datetime import datetime, timedelta
import json
import subprocess
import sys

import pytest

from instinct.private_storage import private_file_lock, write_private_text
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.sqlite_backup import restore_new, snapshot_offline
from instinct.sqlite_guard import SQLiteWriterUnavailable
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord, canonical_json
from instinct.sqlite_session import SQLiteRuntimeSession
from instinct.sqlite_store import SQLiteRecordStore


def _create_runtime(tmp_path):
    tmp_path.chmod(0o700)
    root = tmp_path / 'runtime'
    root.mkdir(mode=0o700)
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    model = ScoredSegment(segment=Segment(content='synthetic only', stored_id='note',
        segment_type=SegmentType.CONTEXT), importance_score=85, concept_title='Synthetic note')
    record = SQLiteRecord.from_dict({'id': 'note', 'content': 'synthetic only',
        'metadata': {key: value for key, value in model.to_storage_dict().items() if key != 'content'},
        'embedding': [1, 0]}, identity)
    with SQLiteRecordStore.create(root / 'sqlite', identity) as repository:
        repository.insert_records([record])
    return root, identity


@pytest.fixture
def runtime(tmp_path):
    return _create_runtime(tmp_path)


def backup(runtime, tmp_path):
    root, identity = runtime
    return snapshot_offline(root, tmp_path / 'backup', identity)


def test_cli_embedding_identity_is_preserved_by_offline_snapshot(tmp_path):
    from instinct.sqlite_init import initialize

    identity = EmbeddingIdentity('ollama', 'nomic-embed-text:latest', 2, 'cosine')
    root = initialize(tmp_path / 'runtime', identity)
    original = (root / 'embedding-identity.json').read_bytes()
    snapshot = snapshot_offline(root, tmp_path / 'backup', identity)
    assert (snapshot / 'embedding-identity.json').read_bytes() == original
    restored = restore_new(snapshot, tmp_path / 'restored', identity)
    assert (restored / 'embedding-identity.json').read_bytes() == original


def test_identity_sidecar_mismatch_refuses_backup(tmp_path):
    root, identity = _create_runtime(tmp_path)
    other = EmbeddingIdentity('synthetic', 'different', 2, 'cosine')
    write_private_text(root / 'embedding-identity.json',
                       canonical_json(other.to_dict()) + '\n')
    with pytest.raises(ValueError, match='Runtime embedding identity mismatch'):
        snapshot_offline(root, tmp_path / 'backup', identity)
    assert not (tmp_path / 'backup').exists()


def test_malformed_identity_sidecar_refuses_backup(tmp_path):
    root, identity = _create_runtime(tmp_path)
    write_private_text(root / 'embedding-identity.json',
                       '{"provider":"synthetic","provider":"ollama"}')
    with pytest.raises(ValueError, match='Invalid runtime embedding identity'):
        snapshot_offline(root, tmp_path / 'backup', identity)
    assert not (tmp_path / 'backup').exists()


def test_roundtrip_preserves_clock_pop_vectors_and_queue(runtime, tmp_path):
    root, identity = runtime
    with SQLiteRuntimeSession.open(root / 'sqlite', identity, persist_dir=root, read_only=False) as session:
        session.tick_activity(now=datetime(2026, 10, 7, 10))
        session.tick_activity(now=datetime(2026, 10, 7, 10) + timedelta(seconds=20))
        with session.operation() as adapter:
            adapter.record_access('note')
            expected = adapter.repository.get_record('note').to_dict()
    write_private_text(root / 'dormancy_config.json', '{"custom": 60}')
    write_private_text(root / 'fallback_pending' / 'pending.json', '{"unreplayed": true}')
    sidecars = {path.name: path.read_bytes() for path in root.glob('*.json')}
    snapshot = backup(runtime, tmp_path)
    restored = restore_new(snapshot, tmp_path / 'restored', identity)
    assert {path.name: path.read_bytes() for path in restored.glob('*.json')} == sidecars
    assert (restored / 'fallback_pending/pending.json').read_bytes() == b'{"unreplayed": true}'
    with SQLiteRuntimeSession.open(restored / 'sqlite', identity, persist_dir=restored, read_only=False) as session:
        with session.operation() as adapter:
            assert adapter.repository.get_record('note').to_dict() == expected
        assert session.tick_activity(now=datetime(2026, 10, 7, 10, 0, 21)).current_turn == 3


def test_active_runtime_refuses_backup(runtime, tmp_path):
    root, identity = runtime
    with SQLiteRuntimeSession.open(root / 'sqlite', identity, persist_dir=root, read_only=False):
        with pytest.raises(TimeoutError):
            backup(runtime, tmp_path)
    assert not (tmp_path / 'backup').exists()


def test_record_writer_refuses_backup(runtime, tmp_path):
    root, identity = runtime
    with SQLiteRecordStore.open_writable(root / 'sqlite', identity):
        with pytest.raises(SQLiteWriterUnavailable):
            backup(runtime, tmp_path)


@pytest.mark.parametrize('name', ['unknown.json', 'sqlite/memory.sqlite3-journal', 'fallback_pending/unknown.tmp'])
def test_unknown_state_fails_closed(runtime, tmp_path, name):
    root, _ = runtime
    write_private_text(root / name, '{}')
    with pytest.raises(ValueError):
        backup(runtime, tmp_path)
    assert not (tmp_path / 'backup').exists()


@pytest.mark.parametrize('payload', ['[]', '{', '{"value": NaN}'])
def test_invalid_sidecar_refused(runtime, tmp_path, payload):
    root, _ = runtime
    write_private_text(root / 'session_state.json', payload)
    with pytest.raises(ValueError):
        backup(runtime, tmp_path)


@pytest.mark.parametrize('name', ['sqlite/memory.sqlite3', 'session_state.json', 'fallback_pending/pending.json'])
def test_tampering_refused_without_destination(runtime, tmp_path, name):
    root, identity = runtime
    write_private_text(root / 'session_state.json', '{}')
    write_private_text(root / 'fallback_pending/pending.json', '{}')
    snapshot = backup(runtime, tmp_path)
    with (snapshot / name).open('ab') as output:
        output.write(b' ')
    with pytest.raises(ValueError, match='integrity'):
        restore_new(snapshot, tmp_path / 'restored', identity)
    assert not (tmp_path / 'restored').exists()


def test_identity_and_overwrite_refused(runtime, tmp_path):
    _, identity = runtime
    snapshot = backup(runtime, tmp_path)
    other = EmbeddingIdentity('synthetic', 'different', 2, 'cosine')
    with pytest.raises(ValueError, match='identity'):
        restore_new(snapshot, tmp_path / 'restored', other)
    with pytest.raises(ValueError, match='new absolute'):
        restore_new(snapshot, runtime[0], identity)
    with pytest.raises(ValueError, match='outside'):
        snapshot_offline(runtime[0], runtime[0] / 'nested', identity)


def test_symlink_refused(runtime, tmp_path):
    root, _ = runtime
    external = tmp_path / 'external.json'
    write_private_text(external, '{}')
    try:
        (root / 'session_state.json').symlink_to(external)
    except OSError:
        pytest.skip('Symlink creation unavailable')
    with pytest.raises(PermissionError):
        backup(runtime, tmp_path)


def test_incomplete_manifest_refused(runtime, tmp_path):
    _, identity = runtime
    snapshot = backup(runtime, tmp_path)
    manifest = json.loads((snapshot / 'snapshot.json').read_text())
    manifest['files'] = {}
    write_private_text(snapshot / 'snapshot.json', json.dumps(manifest))
    with pytest.raises(ValueError, match='integrity'):
        restore_new(snapshot, tmp_path / 'restored', identity)


def test_publication_failure_does_not_create_final_snapshot(runtime, tmp_path, monkeypatch):
    from instinct import sqlite_backup
    def fail(*arguments):
        raise OSError('injected publish failure')
    monkeypatch.setattr(sqlite_backup.os, 'rename', fail)
    with pytest.raises(OSError, match='injected'):
        backup(runtime, tmp_path)
    assert not (tmp_path / 'backup').exists()
    assert len(list(tmp_path.glob('.sqlite-snapshot-*'))) == 1


def test_process_exit_releases_guards_and_preserves_completed_tick_access(runtime, tmp_path):
    root, identity = runtime
    script = '''
from datetime import datetime
import os
from pathlib import Path
import sys
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_session import SQLiteRuntimeSession
root = Path(sys.argv[1])
identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
with SQLiteRuntimeSession.open(root / 'sqlite', identity, persist_dir=root, read_only=False) as session:
    session.tick_activity(now=datetime(2026, 10, 7, 10))
    with session.operation() as adapter:
        assert adapter.record_access('note')
    os._exit(9)
'''
    completed = subprocess.run([sys.executable, '-c', script, str(root)], timeout=30, capture_output=True)
    assert completed.returncode == 9, completed.stderr.decode()
    clock = (root / 'session_state.json').read_bytes()
    restored = restore_new(backup(runtime, tmp_path), tmp_path / 'restored', identity)
    assert (restored / 'session_state.json').read_bytes() == clock
    with SQLiteRuntimeSession.open(restored / 'sqlite', identity, persist_dir=restored, read_only=False) as session:
        with session.operation() as adapter:
            assert adapter.get('note').access_count == 1


def test_queue_capture_contention_refuses_snapshot(runtime, tmp_path):
    root, _ = runtime
    write_private_text(root / 'fallback_pending/pending.json', '{}')
    with private_file_lock(root / 'fallback_pending/.capture.lock', timeout=0):
        with pytest.raises(TimeoutError):
            backup(runtime, tmp_path)
    assert (root / 'fallback_pending/pending.json').read_bytes() == b'{}'


def test_changed_inventory_refuses_snapshot(runtime, tmp_path, monkeypatch):
    from instinct import sqlite_backup
    original = sqlite_backup._payloads
    calls = []
    def changed(root):
        payloads = original(root)
        calls.append(True)
        if len(calls) > 1:
            payloads['session_state.json'] = b'{}'
        return payloads
    monkeypatch.setattr(sqlite_backup, '_payloads', changed)
    with pytest.raises(RuntimeError, match='changed'):
        backup(runtime, tmp_path)
    assert not (tmp_path / 'backup').exists()


def test_bounded_inventory_refused(runtime, tmp_path, monkeypatch):
    from instinct import sqlite_backup
    monkeypatch.setattr(sqlite_backup, 'MAX_TOTAL_BYTES', 16)
    with pytest.raises(ValueError, match='total size'):
        backup(runtime, tmp_path)


def test_snapshot_traversal_manifest_never_used_as_a_path(runtime, tmp_path):
    _, identity = runtime
    snapshot = backup(runtime, tmp_path)
    manifest = json.loads((snapshot / 'snapshot.json').read_text())
    manifest['files']['../../foreign.json'] = {'bytes': 0, 'sha256': ''}
    write_private_text(snapshot / 'snapshot.json', json.dumps(manifest))
    with pytest.raises(ValueError, match='integrity'):
        restore_new(snapshot, tmp_path / 'restored', identity)
    with pytest.raises(ValueError, match='outside'):
        restore_new(snapshot, snapshot / 'restored', identity)
