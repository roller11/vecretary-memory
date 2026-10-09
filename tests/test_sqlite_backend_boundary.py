# SPDX-License-Identifier: MPL-2.0
import os
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_package_does_not_install_chroma_or_select_its_backend():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert not any('chroma' in dependency.lower() for dependency in project['dependencies'])
    assert project['scripts']['vecretary-mcp'] == 'instinct.sqlite_service:cli'
    assert project['scripts']['vecretary-init'] == 'instinct.sqlite_init:cli'


def test_sqlite_service_import_does_not_load_legacy_backend_or_touch_data(tmp_path):
    data = tmp_path / 'unused'
    environment = os.environ.copy()
    environment['INSTINCT_DATA_DIR'] = str(data)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    program = (
        'import sys; import instinct.sqlite_service; '
        'assert "chromadb" not in sys.modules; '
        'assert "instinct.store" not in sys.modules; '
        'assert "instinct.chroma_writer_lock" not in sys.modules'
    )
    result = subprocess.run(
        [sys.executable, '-c', program], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not data.exists()
