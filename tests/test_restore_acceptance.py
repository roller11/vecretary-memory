# SPDX-License-Identifier: MPL-2.0
import os

import pytest

from tests import run_restore_acceptance


def test_restore_confirmation_required_before_store_creation(monkeypatch):
    monkeypatch.setattr(run_restore_acceptance.tempfile, 'mkdtemp',
                        lambda **kwargs: pytest.fail('Created without confirmation'))
    with pytest.raises(SystemExit) as error:
        run_restore_acceptance.main([])
    assert error.value.code == 2


def test_restore_rejects_checkout_before_store_creation(monkeypatch, tmp_path):
    import instinct

    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'instinct' / '__init__.py'))
    monkeypatch.setattr(run_restore_acceptance.tempfile, 'mkdtemp',
                        lambda **kwargs: pytest.fail('Created with checkout runtime'))
    with pytest.raises(SystemExit) as error:
        run_restore_acceptance.main(['--confirm-synthetic-restore'])
    assert error.value.code == 2


def test_restore_environment_never_reuses_live_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv('INSTINCT_DATA_DIR', 'live')
    monkeypatch.setenv('CE_DATA_DIR', 'rc')
    monkeypatch.setenv('CE_EMBEDDING_PROVIDERS', 'remote')
    monkeypatch.setenv('PYTHONPATH', 'checkout')
    environment = run_restore_acceptance.environment_for(tmp_path)
    assert environment['INSTINCT_DATA_DIR'] == str(tmp_path)
    assert environment['CE_DATA_DIR'] == str(tmp_path)
    assert environment['CE_EMBEDDING_PROVIDERS'] == 'onnx'
    assert environment['INSTINCT_ALLOW_REMOTE_PROVIDERS'] == '0'
    assert 'PYTHONPATH' not in environment
    assert os.environ['INSTINCT_DATA_DIR'] == 'live'
