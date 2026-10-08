# SPDX-License-Identifier: MPL-2.0
"""One-command installed Linux/WSL acceptance on disposable synthetic stores."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


STAGES = (
    ('installed_smoke', 'installed_smoke.py', (), 0),
    ('active_decay', 'run_decay_acceptance.py', (), 0),
    ('cold_restore', 'run_restore_acceptance.py', ('--confirm-synthetic-restore',), 0),
    ('shared_writer', 'run_shared_writer_acceptance.py', (), 0),
    ('lifecycle', 'run_lifecycle_acceptance.py', ('--confirm-synthetic-lifecycle',), 0),
    ('real_idle', 'run_idle_acceptance.py', ('--confirm-synthetic-idle',), 1),
)


def isolated_environment(root):
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(('CE_', 'INSTINCT_', 'PYTHON'))
    }
    environment.update({
        'INSTINCT_DATA_DIR': str(root / 'unused-data'),
        'CE_DATA_DIR': str(root / 'unused-data'),
        'CE_EMBEDDING_PROVIDERS': 'onnx',
        'INSTINCT_ALLOW_REMOTE_PROVIDERS': '0',
        'INSTINCT_ALLOW_INSECURE_REMOTE_HTTP': '0',
        'PYTHONDONTWRITEBYTECODE': '1',
        'TMPDIR': str(root), 'TMP': str(root), 'TEMP': str(root),
    })
    return environment


def require_installed():
    import instinct

    if sys.platform != 'linux' or 'site-packages' not in Path(instinct.__file__).resolve().parts:
        raise RuntimeError('Use a wheel-installed Linux/WSL interpreter, not checkout imports')
    if not __debug__:
        raise RuntimeError('Assertions must be enabled for the existing acceptance runners')
    return str(Path(instinct.__file__).resolve())


def run_stage(name, script, arguments, root, environment):
    started = time.monotonic()
    destination = root / script.name
    shutil.copyfile(script, destination)
    stdout_path = root / f'{name}.stdout.log'
    stderr_path = root / f'{name}.stderr.log'
    with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
        completed = subprocess.run(
            [sys.executable, '-I', str(destination), *arguments],
            cwd=root, env=environment, stdout=stdout, stderr=stderr, check=False,
        )
    return {
        'status': 'passed' if completed.returncode == 0 else 'failed',
        'exit_code': completed.returncode,
        'duration_seconds': round(time.monotonic() - started, 3),
        'stdout_log': str(stdout_path), 'stderr_log': str(stderr_path),
    }


def write_report(root, report):
    temporary = root / 'acceptance-report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(root / 'acceptance-report.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-synthetic', action='store_true',
                        help='Authorize disposable starter seeding, synthetic writes and owned-process crash checks')
    parser.add_argument('--soak', action='store_true',
                        help='Also run the real eleven-minute idle check without shortening defaults')
    parser.add_argument('--product-backup', action='store_true',
                        help='Test the owner backup/restore implementation; requires a build containing instinct.backup')
    arguments = parser.parse_args(argv)
    if not arguments.confirm_synthetic:
        parser.error('--confirm-synthetic is required; no live data directory is accepted')
    try:
        installed_module = require_installed()
    except RuntimeError as error:
        parser.error(str(error))
    scripts = Path(__file__).resolve().parent
    for name, filename, options, soak_only in STAGES:
        if not soak_only or arguments.soak:
            if not (scripts / filename).is_file():
                parser.error(f'Missing runner: {filename}')
    previous_umask = os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix='instinct-acceptance-suite-'))
    report = {
        'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'synthetic_root': str(root), 'installed_module': installed_module,
        'mode': 'real_idle_included' if arguments.soak else 'quick',
        'product_backup_requested': arguments.product_backup,
        'live_stores_accessed': False, 'editor_services_changed': False,
        'actual_editor_model_behavior': 'manual_gate_not_certified',
        'public_release': 'not_certified', 'stages': {},
    }
    try:
        write_report(root, report)
        print(f'Acceptance evidence: {root / "acceptance-report.json"}', flush=True)
        environment = isolated_environment(root)
        for name, filename, options, soak_only in STAGES:
            if name == 'cold_restore' and arguments.product_backup:
                options = (*options, '--product-backup')
            if soak_only and not arguments.soak:
                report['stages'][name] = {'status': 'not_run', 'reason': 'Requires --soak; real eleven-minute wait'}
            else:
                print(f'Running {name}...', flush=True)
                try:
                    report['stages'][name] = run_stage(name, scripts / filename, options, root, environment)
                except Exception as error:
                    report['stages'][name] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
                print(f'{name}: {report["stages"][name]["status"]}', flush=True)
            write_report(root, report)
        report['status'] = 'failed' if any(
            stage['status'] == 'failed' for stage in report['stages'].values()
        ) else ('passed' if arguments.soak else 'quick_passed_soak_pending')
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_report(root, report)
        os.umask(previous_umask)
    print(json.dumps(report, indent=2), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
