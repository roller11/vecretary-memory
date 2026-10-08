# SPDX-License-Identifier: MPL-2.0
import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.prepare_client_acceptance import create_root, main, preflight_server, require, result_text, write_workspace


def test_workspace_is_scoped_to_fresh_synthetic_data(tmp_path, monkeypatch):
    monkeypatch.setenv('INSTINCT_DATA_DIR', 'untouched-live-path')
    executable = tmp_path / 'bin with spaces' / 'instinct-mcp'
    workspace, server = write_workspace(tmp_path, executable, 'Reviewed workflow\n')
    config = json.loads((workspace / '.vscode' / 'mcp.json').read_text(encoding='utf-8'))
    assert config == {'servers': {'instinct_acceptance': server}}
    agent_config = json.loads((workspace / '.mcp.json').read_text(encoding='utf-8'))
    assert agent_config == {'mcpServers': {'instinct_acceptance': server}}
    assert agent_config['mcpServers'] == config['servers']
    assert server['command'] == str(executable)
    assert server['args'] == ['--data-dir', str(tmp_path / 'data')]
    assert server['env']['INSTINCT_DATA_DIR'] == str(tmp_path / 'data')
    assert server['env']['CE_DATA_DIR'] == str(tmp_path / 'data')
    assert server['env']['CE_EMBEDDING_PROVIDERS'] == 'onnx'
    assert server['env']['INSTINCT_ALLOW_REMOTE_PROVIDERS'] == '0'
    assert server['env']['INSTINCT_ALLOW_INSECURE_REMOTE_HTTP'] == '0'
    codex = tomllib.loads((workspace / '.codex' / 'config.toml').read_text(encoding='utf-8'))
    configured = codex['mcp_servers']['instinct_acceptance']
    assert configured['command'] == server['command']
    assert configured['args'] == server['args']
    assert configured['env'] == server['env']
    assert configured['cwd'] == str(workspace)
    assert not (tmp_path / 'data').exists()
    assert os.environ['INSTINCT_DATA_DIR'] == 'untouched-live-path'


def test_instructions_are_identical_and_scoped(tmp_path):
    workspace, _ = write_workspace(tmp_path, tmp_path / 'instinct-mcp', 'Reviewed workflow\n')
    copilot = (workspace / '.github' / 'copilot-instructions.md').read_text(encoding='utf-8')
    assert copilot == (workspace / 'AGENTS.md').read_text(encoding='utf-8')
    assert 'do not substitute another memory server' in copilot
    assert copilot.endswith('Reviewed workflow\n')


def test_shared_writer_is_explicit_and_preserved_in_preflight(tmp_path):
    workspace, server = write_workspace(tmp_path, tmp_path / 'instinct-mcp', 'Reviewed\n', shared_writer=True)
    assert server['args'][-1] == '--shared-writer'
    probe = preflight_server(tmp_path, server)
    assert probe['args'] == ['--data-dir', str(tmp_path / 'preflight-data'), '--shared-writer']
    for destination, key in [('.vscode/mcp.json', 'servers'), ('.mcp.json', 'mcpServers')]:
        assert json.loads((workspace / destination).read_text())[key]['instinct_acceptance']['args'] == server['args']
    configured = tomllib.loads((workspace / '.codex/config.toml').read_text())
    assert configured['mcp_servers']['instinct_acceptance']['args'] == server['args']


def test_existing_workspace_is_never_overwritten(tmp_path):
    workspace, _ = write_workspace(tmp_path, tmp_path / 'instinct-mcp', 'Original\n')
    config = workspace / '.vscode' / 'mcp.json'
    original = config.read_bytes()
    agent_config = workspace / '.mcp.json'
    original_agent_config = agent_config.read_bytes()
    with pytest.raises(FileExistsError):
        write_workspace(tmp_path, tmp_path / 'different-mcp', 'Replacement\n')
    assert config.read_bytes() == original
    assert agent_config.read_bytes() == original_agent_config
    assert (workspace / 'AGENTS.md').read_text(encoding='utf-8').endswith('Original\n')


@pytest.mark.parametrize('relative_part', ['root', 'executable'])
def test_relative_paths_rejected_before_writing(tmp_path, relative_part):
    root = Path('relative-root') if relative_part == 'root' else tmp_path
    executable = Path('instinct-mcp') if relative_part == 'executable' else tmp_path / 'instinct-mcp'
    with pytest.raises(ValueError, match='absolute paths'):
        write_workspace(root, executable, 'Reviewed\n')
    assert not (tmp_path / 'workspace').exists()


def test_missing_confirmation_exits_without_creating_workspace(monkeypatch):
    monkeypatch.setattr('tests.prepare_client_acceptance.tempfile.mkdtemp', lambda **kwargs: pytest.fail('Created a root without approval'))
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_checkout_interpreter_rejected_before_creating_workspace(monkeypatch, tmp_path):
    import instinct
    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'instinct' / '__init__.py'))
    monkeypatch.setattr('tests.prepare_client_acceptance.tempfile.mkdtemp', lambda **kwargs: pytest.fail('Created a root with checkout imports'))
    with pytest.raises(SystemExit) as error:
        main(['--confirm-starter-base'])
    assert error.value.code == 2


def test_result_text_ignores_nontext_content():
    result = SimpleNamespace(content=[SimpleNamespace(text='first'), SimpleNamespace(), SimpleNamespace(text='last')])
    assert result_text(result) == 'first\n\nlast'


def test_failed_requirement_cannot_be_recorded_as_pass():
    with pytest.raises(RuntimeError, match='Client preflight failed: synthetic failure'):
        require(False, 'synthetic failure')


def test_successful_requirement():
    assert require(True, 'synthetic success') is None


def test_explicit_root_is_new_and_private(tmp_path):
    root = create_root(str(tmp_path / 'private-alpha'))
    assert root.is_dir()
    if os.name != 'nt':
        assert root.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        create_root(str(root))


def test_relative_root_rejected():
    with pytest.raises(ValueError, match='absolute path'):
        create_root('relative-alpha')


def test_preflight_cannot_consume_workspace_starter_state(tmp_path):
    _, server = write_workspace(tmp_path, tmp_path / 'instinct-mcp', 'Reviewed\n')
    original = json.dumps(server, sort_keys=True)
    probe = preflight_server(tmp_path, server)
    assert probe['args'] == ['--data-dir', str(tmp_path / 'preflight-data')]
    assert probe['env']['INSTINCT_DATA_DIR'] == str(tmp_path / 'preflight-data')
    assert probe['env']['CE_DATA_DIR'] == str(tmp_path / 'preflight-data')
    assert json.dumps(server, sort_keys=True) == original
    assert probe['env'] is not server['env']


@pytest.mark.parametrize('probe_ok', [False, True])
def test_main_seeds_pristine_workspace_only_after_preflight(tmp_path, monkeypatch, probe_ok):
    import instinct
    import instinct.seed
    from tests import prepare_client_acceptance as preparation

    root = tmp_path / 'alpha'
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    suffix = '.exe' if os.name == 'nt' else ''
    for entrypoint in ('instinct-mcp', 'instinct-seed'):
        (binaries / (entrypoint + suffix)).touch()
    monkeypatch.setattr(instinct, '__file__', str(tmp_path / 'site-packages' / 'instinct' / '__init__.py'))
    monkeypatch.setattr(preparation.sys, 'executable', str(binaries / ('python' + suffix)))
    monkeypatch.setattr(instinct.seed, 'load_starter_memories', lambda: ['first', 'second', 'third'])
    events = []

    def bootstrap(command, **kwargs):
        data = command[command.index('--data-dir') + 1]
        events.append(data)
        assert kwargs['env']['INSTINCT_DATA_DIR'] == data
        return SimpleNamespace(returncode=0, stdout=json.dumps({'created': ['first', 'second', 'third']}), stderr='')

    async def probe(actual_root, server, templates):
        assert actual_root == root
        assert server['env']['INSTINCT_DATA_DIR'] == str(root / 'preflight-data')
        assert len(templates) == 3
        events.append('preflight')
        if not probe_ok:
            raise RuntimeError('Synthetic preflight failure')
        return {'transport': 'passed'}

    monkeypatch.setattr(preparation.subprocess, 'run', bootstrap)
    monkeypatch.setattr(preparation, 'preflight', probe)
    arguments = ['--confirm-starter-base', '--root', str(root)]
    if probe_ok:
        main(arguments)
        assert events == [str(root / 'preflight-data'), 'preflight', str(root / 'data')]
        report = json.loads((root / 'preflight.json').read_text())
        assert report['workspace_store'] == 'freshly_seeded_not_exercised'
    else:
        with pytest.raises(RuntimeError, match='Synthetic preflight failure'):
            main(arguments)
        assert events == [str(root / 'preflight-data'), 'preflight']
        assert not (root / 'preflight.json').exists()


@pytest.mark.skipif(os.name == 'nt', reason='Linux/WSL path isolation')
def test_symlink_parent_rejected(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        create_root(str(alias / 'alpha'))
    assert not (real / 'alpha').exists()
