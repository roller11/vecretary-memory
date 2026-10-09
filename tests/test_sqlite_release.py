# SPDX-License-Identifier: MPL-2.0
"""SQLite-only export and explicit fresh-runtime initialization regressions."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tomllib

import pytest

from instinct.sqlite_init import initialize
from instinct.sqlite_init import cli
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_store import SQLiteRecordStore


IDENTITY = EmbeddingIdentity('ollama', 'synthetic-unprobed', 2, 'cosine')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sqlite_release_export', ROOT / 'tools/export_public_draft.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_fresh_empty_runtime_has_exact_identity_and_reopens(tmp_path):
    root = initialize(tmp_path / 'new-runtime', IDENTITY)
    assert json.loads((root / 'embedding-identity.json').read_text()) == IDENTITY.to_dict()
    with SQLiteRecordStore.open_read_only(root / 'sqlite', IDENTITY) as repository:
        assert repository.count() == 0
    if os.name == 'posix':
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE((root / 'embedding-identity.json').stat().st_mode) == 0o600


def test_existing_root_is_unchanged(tmp_path):
    root = initialize(tmp_path / 'runtime', IDENTITY)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        initialize(root, IDENTITY)
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()} == before


@pytest.mark.parametrize('identity', [EmbeddingIdentity('onnx', 'test', 2, 'cosine'),
                                    EmbeddingIdentity('ollama', 'test', 2, 'l2')])
def test_unsupported_cli_identity_has_no_output(tmp_path, identity):
    root = tmp_path / 'runtime'
    with pytest.raises(ValueError, match='Ollama/cosine'):
        initialize(root, identity)
    assert not root.exists()


def test_parent_traversal_is_refused(tmp_path):
    with pytest.raises(ValueError, match='traversal'):
        initialize(tmp_path / 'child' / '..' / 'runtime', IDENTITY)
    assert not (tmp_path / 'child').exists()


def test_initializer_cli_reports_no_provider_probe(tmp_path, monkeypatch, capsys):
    identity_path = tmp_path / 'identity.json'
    identity_path.write_text(json.dumps(IDENTITY.to_dict()))
    root = tmp_path / 'runtime'
    monkeypatch.setattr(sys, 'argv', ['vecretary-init', '--data-dir', str(root),
                                     '--identity', str(identity_path)])
    cli()
    report = json.loads(capsys.readouterr().out)
    assert report['records'] == 0
    assert report['provider_probed'] is False
    assert report['runtime_ready'] is False


@pytest.mark.parametrize('payload', [b'x' * 4097, b'{invalid', b'{}', b'\xff'])
def test_initializer_cli_invalid_identity_leaves_no_root(tmp_path, monkeypatch, payload):
    identity_path = tmp_path / 'identity.json'
    identity_path.write_bytes(payload)
    root = tmp_path / 'runtime'
    monkeypatch.setattr(sys, 'argv', ['vecretary-init', '--data-dir', str(root),
                                     '--identity', str(identity_path)])
    with pytest.raises(SystemExit) as error:
        cli()
    assert error.value.code == 2
    assert not root.exists()


def test_unknown_export_profile_has_no_output(tmp_path):
    with pytest.raises(ValueError, match='profile'):
        export.prepare(ROOT, tmp_path / 'draft', profile='unknown')
    assert not (tmp_path / 'draft').exists()


def test_sqlite_profile_is_chroma_free_and_hash_accounted(tmp_path):
    draft = tmp_path / 'draft'
    report = export.prepare(ROOT, draft, profile='sqlite')
    project = tomllib.loads((draft / 'pyproject.toml').read_text())['project']
    assert project['version'] == '0.2.0a2'
    assert not any('chroma' in item.lower() for item in project['dependencies'])
    assert project['scripts']['vecretary-mcp'] == 'instinct.sqlite_service:cli'
    assert project['scripts']['vecretary-init'] == 'instinct.sqlite_init:cli'
    assert 'instinct-dashboard' not in project['scripts']
    assert not (draft / 'tools/build_chroma_security_candidate.py').exists()
    assert 'build_chroma_security_candidate' not in (draft / 'MANIFEST.in').read_text()
    assert 'SQLite alpha' in (draft / 'README.md').read_text()
    assert 'Chroma loader backport' not in (draft / '.github/workflows/ci.yml').read_text()
    assert report['profile'] == 'sqlite'
    assert report['status'] == 'DRAFT_ONLY_NOT_CLEARED_FOR_PUBLICATION'
    for record in report['files']:
        assert hashlib.sha256((draft / record['path']).read_bytes()).hexdigest() == record['sha256']


def test_sqlite_profile_privacy_failure_precedes_output(tmp_path):
    source = tmp_path / 'source'
    for name in set(export.FIXED) | set(export.SQLITE_OVERRIDES):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('Synthetic content')
    (source / 'tests').mkdir(exist_ok=True)
    (source / 'docs/sqlite-release/README.md').write_text('/home/' + 'private-owner/store')
    with pytest.raises(ValueError, match='review required'):
        export.prepare(source, tmp_path / 'draft', profile='sqlite')
    assert not (tmp_path / 'draft').exists()


@pytest.mark.asyncio
async def test_installed_stdio_checker_accepts_the_runtime_tool_catalog(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import mcp
    import mcp.client.stdio
    from instinct import mcp_server

    checker_spec = importlib.util.spec_from_file_location(
        'sqlite_installed_checker', ROOT / 'tools/check_sqlite_installed.py')
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.list_tools.return_value = SimpleNamespace(tools=await mcp_server.list_tools())
    session.call_tool.side_effect = [
        SimpleNamespace(isError=False, content=[SimpleNamespace(text='No mandatory context')]),
        SimpleNamespace(content=[SimpleNamespace(text='Error executing: embedding provider unbound')]),
    ]

    @asynccontextmanager
    async def isolated_stdio(parameters):
        assert parameters.command == sys.executable
        assert parameters.args[:3] == ['-I', '-m', 'instinct.sqlite_service']
        yield object(), object()

    monkeypatch.setattr(mcp, 'ClientSession', lambda reader, writer: session)
    monkeypatch.setattr(mcp.client.stdio, 'stdio_client', isolated_stdio)
    await checker.check_stdio(tmp_path)
    session.initialize.assert_awaited_once()
    assert session.call_tool.await_count == 2
