# SPDX-License-Identifier: MPL-2.0
import json
import os
from pathlib import Path
import select
import subprocess
import sys

import pytest

from instinct import backup
from instinct.chroma_writer_lock import ChromaWriterLockUnavailable, acquire_writer_lock, release_writer_lock
from instinct.private_storage import PrivateStorageError, write_private_text


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir(mode=0o700)
    (root / 'chroma.sqlite3').write_bytes(b'Synthetic database bytes')
    index = root / 'vector-index'
    index.mkdir(mode=0o755)
    (index / 'vectors.bin').write_bytes(b'Synthetic vectors')
    (root / 'empty-state').mkdir(mode=0o700)
    write_private_text(root / 'session_state.json', '{"synthetic":true}')
    return root


def test_full_roundtrip_preserves_bytes_and_empty_directories(source, tmp_path):
    destination, restored = tmp_path / 'backup', tmp_path / 'restored'
    assert backup.create_backup(source, destination)['ok']
    manifest = backup.verify(destination)
    assert set(manifest['files']) == {'chroma.sqlite3', 'vector-index/vectors.bin', 'session_state.json'}
    assert '.writer.lock' not in manifest['files']
    assert backup.restore_backup(destination, restored)['ok']
    assert (restored / 'empty-state').is_dir()
    for name in manifest['files']:
        assert (source / name).read_bytes() == (restored / name).read_bytes()
        if os.name == 'posix':
            assert (restored / name).stat().st_mode & 0o777 == 0o600


def test_busy_store_refused_without_publishing_backup(source, tmp_path):
    handle = acquire_writer_lock(str(source), timeout=0)
    try:
        with pytest.raises(ValueError, match='busy'):
            backup.create_backup(source, tmp_path / 'backup')
        assert not (tmp_path / 'backup').exists()
    finally:
        release_writer_lock(handle)


@pytest.mark.parametrize('operation', ['backup', 'restore'])
def test_existing_destination_never_replaced(source, tmp_path, operation):
    destination = tmp_path / 'backup'
    backup.create_backup(source, destination)
    existing = tmp_path / 'existing'
    existing.mkdir(mode=0o700)
    (existing / 'owner-note').write_bytes(b'Keep this')
    with pytest.raises(ValueError, match='already exists'):
        if operation == 'backup':
            backup.create_backup(source, existing)
        else:
            backup.restore_backup(destination, existing)
    assert (existing / 'owner-note').read_bytes() == b'Keep this'


def test_nested_destination_rejected_before_creation(source):
    with pytest.raises(ValueError, match='separate'):
        backup.create_backup(source, source / 'backup')
    assert not (source / 'backup').exists()


@pytest.mark.parametrize('corruption', ['changed', 'missing', 'extra', 'version', 'traversal'])
def test_corrupt_backup_refused_before_restore_creation(source, tmp_path, corruption):
    destination = tmp_path / 'backup'
    backup.create_backup(source, destination)
    if corruption == 'changed':
        (destination / 'data/chroma.sqlite3').write_bytes(b'Corrupted')
    elif corruption == 'missing':
        (destination / 'data/chroma.sqlite3').unlink()
    elif corruption == 'extra':
        (destination / 'data/extra').write_bytes(b'Unexpected')
    else:
        manifest = json.loads((destination / 'manifest.json').read_text())
        if corruption == 'version':
            manifest['versions'] = {'chromadb': 'incompatible'}
        else:
            manifest['files']['../escape'] = manifest['files']['chroma.sqlite3']
        write_private_text(destination / 'manifest.json', json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        backup.restore_backup(destination, tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX symlink/permission boundary')
@pytest.mark.parametrize('kind', ['file', 'directory', 'lock', 'hardlink'])
def test_linked_inputs_refused(source, tmp_path, kind):
    if kind == 'directory':
        (source / 'linked').symlink_to(tmp_path, target_is_directory=True)
    elif kind == 'hardlink':
        os.link(source / 'chroma.sqlite3', source / 'linked')
    elif kind == 'lock':
        (source / '.writer.lock').symlink_to(source / 'chroma.sqlite3')
    else:
        (source / 'linked').symlink_to(source / 'chroma.sqlite3')
    with pytest.raises((ValueError, PrivateStorageError)):
        backup.create_backup(source, tmp_path / 'backup')
    assert not (tmp_path / 'backup').exists()


def test_limits_fail_before_backup_creation(source, tmp_path, monkeypatch):
    monkeypatch.setattr(backup, 'MAX_BYTES', 1)
    with pytest.raises(ValueError, match='limit'):
        backup.create_backup(source, tmp_path / 'backup')
    assert not (tmp_path / 'backup').exists()


def test_cli_requires_owner_offline_confirmation(source, tmp_path, monkeypatch):
    monkeypatch.setattr(backup.sys, 'platform', 'linux')
    with pytest.raises(SystemExit) as error:
        backup.main(['backup', '--source', str(source), '--destination', str(tmp_path / 'backup')])
    assert error.value.code == 2
    assert not (tmp_path / 'backup').exists()


def test_failed_copy_leaves_no_completed_manifest_and_releases_writer(source, tmp_path, monkeypatch):
    def fail(*arguments):
        raise OSError('Synthetic disk failure')

    monkeypatch.setattr(backup, 'stream_file', fail)
    with pytest.raises(OSError):
        backup.create_backup(source, tmp_path / 'backup')
    assert not (tmp_path / 'backup/manifest.json').exists()
    assert not backup.is_writer_lock_held(str(source))
    with pytest.raises(FileNotFoundError):
        backup.restore_backup(tmp_path / 'backup', tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX subprocess pipe readiness')
def test_cross_process_writer_refuses_backup(source, tmp_path):
    code = (
        'import sys; from instinct.chroma_writer_lock import acquire_writer_lock, release_writer_lock; '
        'handle = acquire_writer_lock(sys.argv[1], timeout=0); '
        'print("ready", flush=True); sys.stdin.readline(); release_writer_lock(handle)'
    )
    with subprocess.Popen([sys.executable, '-c', code, str(source)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as owner:
        try:
            assert select.select([owner.stdout], [], [], 10)[0], 'Writer did not report readiness'
            assert owner.stdout.readline().strip() == 'ready'
            sentinel = (source / '.writer.lock').read_bytes()
            with pytest.raises((ValueError, ChromaWriterLockUnavailable)):
                backup.create_backup(source, tmp_path / 'backup')
            assert not (tmp_path / 'backup').exists()
            assert (source / '.writer.lock').read_bytes() == sentinel
        finally:
            owner.communicate('\n', timeout=10)
        assert owner.returncode == 0
    assert backup.create_backup(source, tmp_path / 'backup')['ok']


def test_same_process_owner_between_probe_and_acquire_is_not_released(source, tmp_path, monkeypatch):
    handle = acquire_writer_lock(str(source), timeout=0)
    monkeypatch.setattr(backup, 'is_writer_lock_held', lambda directory: False)
    try:
        with pytest.raises(ChromaWriterLockUnavailable):
            backup.create_backup(source, tmp_path / 'backup')
        assert acquire_writer_lock(str(source), timeout=0) is handle
        assert not (tmp_path / 'backup').exists()
    finally:
        release_writer_lock(handle)


def test_directory_read_error_aborts_before_destination_creation(source, tmp_path, monkeypatch):
    def fail_walk(root, **options):
        options['onerror'](PermissionError('Synthetic unreadable directory'))
        return iter(())

    monkeypatch.setattr(backup.os, 'walk', fail_walk)
    with pytest.raises(PermissionError, match='unreadable'):
        backup.create_backup(source, tmp_path / 'backup')
    assert not (tmp_path / 'backup').exists()
    assert not backup.is_writer_lock_held(str(source))


def test_source_mutation_with_restored_mtime_refuses_manifest(source, tmp_path, monkeypatch):
    original = backup.stream_file

    def mutate_after_copy(selected, destination=None):
        result = original(selected, destination)
        if destination is not None and selected.name == 'chroma.sqlite3':
            before = selected.stat()
            selected.write_bytes(b'X' * before.st_size)
            os.utime(selected, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(backup, 'stream_file', mutate_after_copy)
    with pytest.raises(ValueError, match='changed'):
        backup.create_backup(source, tmp_path / 'backup')
    assert not (tmp_path / 'backup/manifest.json').exists()
    assert not backup.is_writer_lock_held(str(source))


def test_interrupted_restore_releases_lock_and_refuses_destination_reuse(source, tmp_path, monkeypatch):
    snapshot, restored = tmp_path / 'backup', tmp_path / 'restored'
    backup.create_backup(source, snapshot)
    original = backup.stream_file
    copied = []

    def fail_second_copy(selected, destination=None):
        if destination is not None:
            if copied:
                raise OSError('Synthetic interrupted restore')
            copied.append(destination)
        return original(selected, destination)

    with monkeypatch.context() as patch:
        patch.setattr(backup, 'stream_file', fail_second_copy)
        with pytest.raises(OSError, match='interrupted'):
            backup.restore_backup(snapshot, restored)
    assert copied[0].is_file()
    assert not backup.is_writer_lock_held(str(restored))
    with pytest.raises(ValueError, match='already exists'):
        backup.restore_backup(snapshot, restored)
    assert backup.verify(snapshot)


@pytest.mark.parametrize('corruption', ['bool-size', 'negative-size', 'digest', 'metadata',
                                       'duplicate-directory', 'null-path', 'oversized'])
def test_untrusted_manifest_refused_before_destination_creation(source, tmp_path, corruption):
    snapshot, restored = tmp_path / 'backup', tmp_path / 'restored'
    backup.create_backup(source, snapshot)
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    entry = manifest['files']['chroma.sqlite3']
    if corruption == 'bool-size':
        entry['size'] = True
    elif corruption == 'negative-size':
        entry['size'] = -1
    elif corruption == 'digest':
        entry['sha256'] = 'G' * 64
    elif corruption == 'metadata':
        entry['unexpected'] = True
    elif corruption == 'duplicate-directory':
        manifest['directories'].append(manifest['directories'][0])
    elif corruption == 'null-path':
        manifest['files']['bad\x00path'] = entry
    else:
        entry['size'] = backup.MAX_BYTES + 1
    write_private_text(snapshot / 'manifest.json', json.dumps(manifest))
    with pytest.raises(ValueError):
        backup.restore_backup(snapshot, restored)
    assert not restored.exists()


def test_duplicate_manifest_keys_refused(source, tmp_path):
    snapshot = tmp_path / 'backup'
    backup.create_backup(source, snapshot)
    raw = (snapshot / 'manifest.json').read_text()
    write_private_text(snapshot / 'manifest.json', raw.replace('{', '{"format":"ambiguous",', 1))
    with pytest.raises(ValueError, match='Duplicate'):
        backup.restore_backup(snapshot, tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()
