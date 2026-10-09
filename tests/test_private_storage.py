# SPDX-License-Identifier: MPL-2.0
import os
from pathlib import Path
import stat
from unittest.mock import Mock

import pytest

from instinct import dashboard
from instinct.private_storage import ensure_private_directory, private_binary_reader, write_private_text


def test_new_nested_paths_and_atomic_records(tmp_path):
    root = ensure_private_directory(tmp_path / 'new' / 'data')
    path = root / 'record.json'
    write_private_text(path, '{"synthetic":true}')
    with private_binary_reader(path) as source:
        assert source.read() == b'{"synthetic":true}'
    if os.name == 'posix':
        assert root.stat().st_mode & 0o777 == 0o700
        assert root.parent.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
    assert list(root.iterdir()) == [path]
    with pytest.raises(FileExistsError):
        write_private_text(path, 'replacement', replace=False)
    assert 'synthetic' in path.read_text()


def test_no_parent_traversal(tmp_path):
    with pytest.raises(PermissionError):
        ensure_private_directory(tmp_path / 'child' / '..' / 'data')
    assert not (tmp_path / 'child').exists()


def test_application_state_files_use_private_storage(tmp_path):
    from instinct import dormancy, nudge, session_timer

    root = ensure_private_directory(tmp_path / 'data')
    nudge._save_turn_state(str(root), {'current_turn': 2, 'last_tick_at': None})
    configuration = dormancy._ConfigStore(root)
    configuration.save(dormancy.DormancyConfig())
    timer = session_timer._SessionStore(root)
    timer.save(session_timer.SessionState(None, None, 0.0))
    assert nudge._load_turn_state(str(root))['current_turn'] == 2
    assert dormancy._ConfigStore(root).load().mandatory == configuration.load().mandatory
    assert session_timer._SessionStore(root).load() == timer.load()
    assert len(list(root.iterdir())) == 3
    if os.name == 'posix':
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in root.iterdir())


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permissions')
@pytest.mark.parametrize('kind', ['nudge', 'dormancy', 'session'])
def test_exposed_state_is_not_silently_reset(tmp_path, kind):
    from instinct import dormancy, nudge, session_timer

    root = ensure_private_directory(tmp_path / 'data')
    filenames = {'nudge': 'nudge_state.json', 'dormancy': dormancy.CONFIG_FILENAME,
                 'session': session_timer.SESSION_STATE_FILENAME}
    path = root / filenames[kind]
    path.write_text('{"synthetic":true}')
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        if kind == 'nudge':
            nudge._load_turn_state(str(root))
        elif kind == 'dormancy':
            dormancy._ConfigStore(root).load()
        else:
            session_timer._SessionStore(root).load()
    assert path.read_text() == '{"synthetic":true}'
    assert path.stat().st_mode & 0o777 == 0o644


@pytest.mark.skipif(os.name != 'posix', reason='POSIX symlinks')
def test_runtime_preserves_symlink_for_storage_rejection(tmp_path, monkeypatch):
    from instinct import runtime

    root = ensure_private_directory(tmp_path / 'data')
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv('INSTINCT_DATA_DIR', str(alias))
    assert Path(runtime.default_data_dir()) == alias
    with pytest.raises(PermissionError):
        ensure_private_directory(runtime.default_data_dir())
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permissions')
@pytest.mark.parametrize('mode', [0o755, 0o770, 0o777])
def test_existing_insecure_directories_refused_unchanged(tmp_path, mode):
    root = tmp_path / 'data'
    root.mkdir(mode=mode)
    root.chmod(mode)
    with pytest.raises(PermissionError):
        ensure_private_directory(root)
    assert root.stat().st_mode & 0o777 == mode
    assert not list(root.iterdir())


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permissions')
def test_existing_insecure_file_not_overwritten(tmp_path):
    path = tmp_path / 'owner.json'
    path.write_text('owner content')
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        write_private_text(path, 'changed')
    assert path.read_text() == 'owner content'
    assert path.stat().st_mode & 0o777 == 0o644


@pytest.mark.skipif(os.name != 'posix', reason='POSIX links')
def test_symlink_ancestors_files_and_hardlinks_are_refused(tmp_path):
    outside = ensure_private_directory(tmp_path / 'outside')
    link = tmp_path / 'link'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError):
        ensure_private_directory(link / 'data')
    assert not (outside / 'data').exists()
    original = outside / 'owner.json'
    write_private_text(original, 'untouched')
    redirected = tmp_path / 'redirect.json'
    redirected.symlink_to(original)
    with pytest.raises(PermissionError):
        write_private_text(redirected, 'changed')
    hardlink = tmp_path / 'hardlink.json'
    os.link(original, hardlink)
    with pytest.raises(PermissionError):
        with private_binary_reader(hardlink):
            pass
    assert original.read_text() == 'untouched'


def test_dashboard_reset_refusal_never_touches_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'PERSIST_DIR', str(tmp_path))
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob('*') if path.is_file()}
    result = dashboard.admin_reset(dry_run=True)
    assert result['code'] == 501
    assert result['dry_run'] is True
    assert {path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob('*') if path.is_file()} == before


def test_dashboard_reset_refuses_legacy_database_without_copying_it(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'PERSIST_DIR', str(tmp_path))
    legacy_database = tmp_path / 'legacy-vector.sqlite'
    legacy_database.write_bytes(b'Synthetic legacy database fixture')
    first = dashboard.admin_reset()
    second = dashboard.admin_reset()
    assert first['code'] == second['code'] == 501
    assert legacy_database.read_bytes() == b'Synthetic legacy database fixture'
    assert not (tmp_path / 'backups').exists()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX ownership')
def test_foreign_ownership_is_refused_without_chown(tmp_path, monkeypatch):
    from instinct import private_storage
    path = tmp_path / 'record.json'
    write_private_text(path, 'owner content')
    original_uid = path.stat().st_uid
    monkeypatch.setattr(private_storage.os, 'geteuid', lambda: original_uid + 1000)
    with pytest.raises(PermissionError):
        private_storage.check_private_file(path)
    with pytest.raises(PermissionError):
        ensure_private_directory(tmp_path)
    assert path.stat().st_uid == original_uid
    assert path.read_text() == 'owner content'


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permissions')
def test_sqlite_session_refuses_exposed_directory_before_database_or_lock(tmp_path):
    from instinct.sqlite_records import EmbeddingIdentity
    from instinct.sqlite_session import SQLiteRuntimeSession

    root = tmp_path / 'data'
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(PermissionError):
        SQLiteRuntimeSession.open(root / 'sqlite', EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine'),
                                  persist_dir=root, read_only=False)
    assert not (root / '.sqlite.runtime.lock').exists()
    assert not (root / 'sqlite').exists()
    assert root.stat().st_mode & 0o777 == 0o755
