# SPDX-License-Identifier: MPL-2.0
import json
import os
import sqlite3

import pytest

from tests.run_idle_acceptance import IDLE_SECONDS, environment_for, idle_active_seconds, main, require, snapshot, state


def test_idle_renewal_excludes_real_gap_from_active_clock():
    before = {'accumulated_session_seconds': 12.0}
    assert idle_active_seconds(before, {'accumulated_session_seconds': 12.0}) == 0
    assert idle_active_seconds(before, {'accumulated_session_seconds': 12.1}) == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match='charged'):
        idle_active_seconds(before, {'accumulated_session_seconds': 672.0})
    with pytest.raises(RuntimeError, match='charged'):
        idle_active_seconds(before, {'accumulated_session_seconds': 0})


def test_idle_requires_explicit_confirmation_before_creation(monkeypatch):
    monkeypatch.setattr('tests.run_idle_acceptance.tempfile.mkdtemp',
                        lambda **kwargs: pytest.fail('Created a store without approval'))
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_idle_rejects_checkout_before_creation(monkeypatch, tmp_path):
    import instinct

    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'instinct' / '__init__.py'))
    monkeypatch.setattr('tests.run_idle_acceptance.tempfile.mkdtemp',
                        lambda **kwargs: pytest.fail('Created a store with checkout imports'))
    with pytest.raises(SystemExit) as error:
        main(['--confirm-synthetic-idle'])
    assert error.value.code == 2


def test_idle_uses_real_default_interval_with_margin():
    assert IDLE_SECONDS == 660


def test_idle_environment_does_not_reuse_live_paths(monkeypatch, tmp_path):
    monkeypatch.setenv('INSTINCT_DATA_DIR', 'live-data')
    monkeypatch.setenv('CE_DATA_DIR', 'rc-data')
    monkeypatch.setenv('PYTHONPATH', 'checkout')
    environment = environment_for(tmp_path)
    assert environment['INSTINCT_DATA_DIR'] == str(tmp_path)
    assert environment['CE_DATA_DIR'] == str(tmp_path)
    assert 'PYTHONPATH' not in environment
    assert os.environ['INSTINCT_DATA_DIR'] == 'live-data'
    assert os.environ['CE_DATA_DIR'] == 'rc-data'
    assert os.environ['PYTHONPATH'] == 'checkout'


def test_idle_snapshot_is_read_only_and_preserves_typed_metadata(tmp_path):
    database = tmp_path / 'chroma.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE embeddings (id INTEGER, embedding_id TEXT)')
        connection.execute('''CREATE TABLE embedding_metadata (
            id INTEGER, key TEXT, int_value INTEGER, string_value TEXT,
            float_value REAL, bool_value INTEGER)''')
        connection.execute("INSERT INTO embeddings VALUES (1, 'synthetic-starter')")
        connection.executemany('INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)', [
            (1, 'importance_score', 79, None, None, None),
            (1, 'concept_title', None, 'Synthetic starter', None, None),
            (1, 'recycled_at', None, 'synthetic-recycle-marker', None, None),
            (1, 'irrelevant', None, 'not requested', None, None),
        ])
    original = database.read_bytes()
    assert snapshot(tmp_path) == {'synthetic-starter': {
        'importance_score': 79, 'concept_title': 'Synthetic starter',
        'recycled_at': 'synthetic-recycle-marker',
    }}
    assert database.read_bytes() == original


def test_idle_snapshot_never_creates_missing_database(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        snapshot(tmp_path)
    assert not (tmp_path / 'chroma.sqlite3').exists()


def test_idle_state_read_does_not_change_clock(tmp_path):
    filename = tmp_path / 'session_state.json'
    payload = {'session_started_at': 'synthetic-marker', 'accumulated_session_seconds': 1}
    filename.write_text(json.dumps(payload))
    original = filename.read_bytes()
    assert state(tmp_path, filename.name) == payload
    assert filename.read_bytes() == original


def test_idle_failure_cannot_be_reported_as_passed():
    with pytest.raises(RuntimeError, match='synthetic failure'):
        require(False, 'synthetic failure')
