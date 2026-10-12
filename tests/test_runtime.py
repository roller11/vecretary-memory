# SPDX-License-Identifier: MPL-2.0
import ast
from pathlib import Path
import subprocess
import sys
import tomllib
from unittest.mock import Mock

import pytest

from instinct import _default_persist_dir
from instinct import dashboard, mcp_server, runtime
from instinct.fallback_store import _pending_dir


@pytest.mark.parametrize('variable', ['INSTINCT_DATA_DIR', 'CE_DATA_DIR'])
def test_explicit_data_directory(monkeypatch, tmp_path, variable):
    monkeypatch.delenv('INSTINCT_DATA_DIR', raising=False)
    monkeypatch.delenv('CE_DATA_DIR', raising=False)
    monkeypatch.setenv(variable, str(tmp_path / 'memory'))
    expected = str(tmp_path / 'memory')
    assert runtime.default_data_dir() == expected
    assert _default_persist_dir() == expected
    assert Path(_pending_dir(None)) == tmp_path / 'memory' / 'fallback_pending'


def test_new_data_variable_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv('INSTINCT_DATA_DIR', str(tmp_path / 'new'))
    monkeypatch.setenv('CE_DATA_DIR', str(tmp_path / 'legacy'))
    assert Path(runtime.default_data_dir()) == tmp_path / 'new'


def test_default_data_is_user_owned(monkeypatch, tmp_path):
    monkeypatch.delenv('INSTINCT_DATA_DIR', raising=False)
    monkeypatch.delenv('CE_DATA_DIR', raising=False)
    monkeypatch.delenv('XDG_DATA_HOME', raising=False)
    monkeypatch.setattr(runtime.Path, 'home', lambda: tmp_path)
    assert Path(runtime.default_data_dir()) == tmp_path / '.local' / 'share' / 'instinct'


def test_xdg_data_directory(monkeypatch, tmp_path):
    monkeypatch.delenv('INSTINCT_DATA_DIR', raising=False)
    monkeypatch.delenv('CE_DATA_DIR', raising=False)
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg'))
    assert Path(runtime.default_data_dir()) == tmp_path / 'xdg' / 'instinct'


def test_explicit_fallback_store_path_wins(monkeypatch, tmp_path):
    monkeypatch.setenv('INSTINCT_DATA_DIR', str(tmp_path / 'default'))
    assert Path(_pending_dir(str(tmp_path / 'other'))) == tmp_path / 'other' / 'fallback_pending'


def test_packaging_excludes_private_modules():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    assert config['tool']['setuptools']['packages']['find']['include'] == ['instinct', 'instinct.*']
    assert config['project']['scripts'] == {
        'vecretary-mcp': 'instinct.sqlite_service:cli',
        'vecretary-init': 'instinct.sqlite_init:cli',
        'vecretary-sqlite': 'instinct.sqlite_service:cli',
    }


@pytest.mark.parametrize('module', ['instinct.mcp_server', 'instinct.dashboard'])
def test_module_help_does_not_start_service(module, tmp_path):
    import os
    environment = os.environ.copy()
    environment['INSTINCT_DATA_DIR'] = str(tmp_path / 'unused')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run(
        [sys.executable, '-m', module, '--help'], env=environment,
        text=True, capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert '--data-dir' in completed.stdout
    assert not (tmp_path / 'unused').exists()


@pytest.mark.asyncio
async def test_standalone_mcp_preserves_tool_surface():
    tools = await mcp_server.list_tools()
    assert {tool.name for tool in tools} == {
        'instinct_get_context', 'instinct_get_mandatory', 'instinct_save_memory',
        'instinct_search', 'instinct_get', 'instinct_delete_memory', 'instinct_update_score',
    }
    assert all(tool.inputSchema['type'] == 'object' for tool in tools)


def test_mcp_write_guard_still_blocks_owner_only_scores():
    assert mcp_server.MCP_SCORE_CAP == 92
    assert 'owner-only' in mcp_server._check_write_guards('Synthetic memory content', 'Synthetic title', 100)
    assert mcp_server._check_write_guards('Synthetic memory content', 'Synthetic title', 70) is None


@pytest.mark.parametrize('module', [mcp_server, dashboard])
def test_runtime_has_no_legacy_premium_or_private_controller_dependency(module):
    text = Path(module.__file__).read_text(encoding='utf-8')
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith(('tc_mcp', 'core', 'providers', 'wisdom', 'triage'))
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith(('tc_mcp', 'core', 'providers', 'wisdom', 'triage', 'instinct.cot')) for alias in node.names)
    assert 'mcp_control.py' not in text
    assert '/api/cot/' not in text
