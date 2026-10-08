# SPDX-License-Identifier: MPL-2.0
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import run_acceptance, run_decay_acceptance


def test_confirmation_required_before_import_or_store_creation(monkeypatch):
    monkeypatch.setattr(run_acceptance, 'require_installed', lambda: pytest.fail('Imported before approval'))
    monkeypatch.setattr(run_acceptance.tempfile, 'mkdtemp', lambda **kwargs: pytest.fail('Created before approval'))
    with pytest.raises(SystemExit) as error:
        run_acceptance.main([])
    assert error.value.code == 2


def test_checkout_is_rejected_before_creation(monkeypatch, tmp_path):
    import instinct

    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'instinct' / '__init__.py'))
    monkeypatch.setattr(run_acceptance.tempfile, 'mkdtemp', lambda **kwargs: pytest.fail('Created despite checkout imports'))
    with pytest.raises(SystemExit) as error:
        run_acceptance.main(['--confirm-synthetic'])
    assert error.value.code == 2


def test_environment_removes_inherited_live_settings(monkeypatch, tmp_path):
    monkeypatch.setenv('INSTINCT_DATA_DIR', 'live')
    monkeypatch.setenv('CE_DATA_DIR', 'rc')
    monkeypatch.setenv('CE_EMBEDDING_PROVIDERS', 'remote')
    monkeypatch.setenv('CE_DASHBOARD_PORT', '8765')
    monkeypatch.setenv('PYTHONPATH', 'checkout')
    monkeypatch.setenv('PYTHONOPTIMIZE', '1')
    environment = run_acceptance.isolated_environment(tmp_path)
    assert environment['INSTINCT_DATA_DIR'] == str(tmp_path / 'unused-data')
    assert environment['CE_DATA_DIR'] == str(tmp_path / 'unused-data')
    assert environment['TMPDIR'] == str(tmp_path)
    assert environment['CE_EMBEDDING_PROVIDERS'] == 'onnx'
    assert environment['INSTINCT_ALLOW_REMOTE_PROVIDERS'] == '0'
    assert environment['INSTINCT_ALLOW_INSECURE_REMOTE_HTTP'] == '0'
    assert 'CE_DASHBOARD_PORT' not in environment
    assert 'PYTHONPATH' not in environment
    assert 'PYTHONOPTIMIZE' not in environment
    assert os.environ['INSTINCT_DATA_DIR'] == 'live'


def setup_run(monkeypatch, tmp_path):
    monkeypatch.setattr(run_acceptance, 'require_installed', lambda: '/synthetic/site-packages/instinct/__init__.py')
    monkeypatch.setattr(run_acceptance.tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path))


def test_quick_success_leaves_soak_and_editor_gates_open(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    called = []

    def stage(name, *arguments):
        called.append(name)
        return {'status': 'passed', 'exit_code': 0}

    monkeypatch.setattr(run_acceptance, 'run_stage', stage)
    assert run_acceptance.main(['--confirm-synthetic']) == 0
    report = json.loads((tmp_path / 'acceptance-report.json').read_text())
    assert report['status'] == 'quick_passed_soak_pending'
    assert report['stages']['real_idle']['status'] == 'not_run'
    assert report['actual_editor_model_behavior'] == 'manual_gate_not_certified'
    assert report['public_release'] == 'not_certified'
    assert called == ['installed_smoke', 'active_decay', 'cold_restore', 'shared_writer', 'lifecycle']
    assert not (tmp_path / 'acceptance-report.json.tmp').exists()


def test_soak_failure_returns_nonzero_and_retains_other_results(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    called = []

    def stage(name, *arguments):
        called.append(name)
        if name == 'shared_writer':
            raise RuntimeError('Synthetic failure')
        return {'status': 'passed', 'exit_code': 0}

    monkeypatch.setattr(run_acceptance, 'run_stage', stage)
    assert run_acceptance.main(['--confirm-synthetic', '--soak']) == 1
    report = json.loads((tmp_path / 'acceptance-report.json').read_text())
    assert report['status'] == 'failed'
    assert report['stages']['shared_writer']['error'] == 'RuntimeError: Synthetic failure'
    assert report['stages']['real_idle']['status'] == 'passed'
    assert len(called) == 6


def test_all_selected_stages_pass(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    monkeypatch.setattr(run_acceptance, 'run_stage', lambda *arguments: {'status': 'passed', 'exit_code': 0})
    assert run_acceptance.main(['--confirm-synthetic', '--soak']) == 0
    assert json.loads((tmp_path / 'acceptance-report.json').read_text())['status'] == 'passed'


def test_product_backup_option_is_forwarded(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    selected = {}

    def stage(name, script, arguments, *remaining):
        selected[name] = arguments
        return {'status': 'passed', 'exit_code': 0}

    monkeypatch.setattr(run_acceptance, 'run_stage', stage)
    assert run_acceptance.main(['--confirm-synthetic', '--product-backup']) == 0
    assert selected['cold_restore'] == ('--confirm-synthetic-restore', '--product-backup')
    assert json.loads((tmp_path / 'acceptance-report.json').read_text())['product_backup_requested']


def test_interruption_is_not_reported_as_success(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)

    def interrupt(*arguments):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_acceptance, 'run_stage', interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_acceptance.main(['--confirm-synthetic'])
    report = json.loads((tmp_path / 'acceptance-report.json').read_text())
    assert report['status'] == 'interrupted'
    assert 'finished_at' in report


def test_stage_uses_isolated_copy_and_captures_failure(monkeypatch, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    script = source / 'synthetic.py'
    script.write_text('synthetic runner')
    root = tmp_path / 'evidence'
    root.mkdir()
    environment = run_acceptance.isolated_environment(root)

    def execute(command, **options):
        assert command[1] == '-I'
        assert Path(command[2]) == root / script.name
        assert command[3:] == ['--synthetic']
        assert options['cwd'] == root
        assert options['env'] == environment
        options['stderr'].write('Synthetic error')
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(run_acceptance.subprocess, 'run', execute)
    result = run_acceptance.run_stage('synthetic', script, ('--synthetic',), root, environment)
    assert result['status'] == 'failed'
    assert result['exit_code'] == 7
    assert (root / 'synthetic.stderr.log').read_text() == 'Synthetic error'


def test_installed_decay_probe_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(run_decay_acceptance.time, 'sleep', lambda seconds: None)
    real_monotonic = iter([100.0, 101.1])
    monkeypatch.setattr(run_decay_acceptance.time, 'monotonic', lambda: next(real_monotonic))
    monkeypatch.setattr(run_decay_acceptance.subprocess, 'run', lambda *arguments, **options:
                        SimpleNamespace(stdout='{"first_tick_added_seconds": 0}'))
    report = run_decay_acceptance.run(tmp_path)
    assert report['status'] == 'passed'
    assert len(report['controlled_clock']['scores']) == 3
    assert report['controlled_clock']['idle_week_excluded']
    assert report['real_process_restart']['first_tick_added_seconds'] == 0
    assert report['real_ten_minute_idle'] == 'not_tested_here_use_soak_stage'
