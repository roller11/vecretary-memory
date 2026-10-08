# SPDX-License-Identifier: MPL-2.0
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest


@pytest.mark.parametrize('statement', [
    'import chromadb.server',
    'import chromadb.server.fastapi',
    'from chromadb.cli.cli import app; app()',
    "import chromadb_rust_bindings; chromadb_rust_bindings.cli(['chroma', 'run'])",
    "from chromadb_rust_bindings import chromadb_rust_bindings; chromadb_rust_bindings.cli(['chroma', 'run'])",
    "import importlib; native = importlib.import_module('chromadb_rust_bindings.chromadb_rust_bindings'); native.cli(['chroma', 'run'])",
])
def test_server_entry_points_fail_closed_in_fresh_process(statement, tmp_path):
    program = (
        'import socket\n'
        'def reject_bind(*args, **kwargs):\n'
        '    raise AssertionError("Server attempted socket binding")\n'
        'socket.socket.bind = reject_bind\n'
        'try:\n'
        f'    {statement}\n'
        'except RuntimeError as error:\n'
        '    assert "HTTP servers" in str(error) and "disabled" in str(error)\n'
        'else:\n'
        '    raise AssertionError("Server entry point succeeded")\n'
    )
    result = subprocess.run([sys.executable, '-I', '-c', program], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_installed_chroma_console_command_is_disabled(tmp_path):
    command = Path(sys.executable).parent / ('chroma.exe' if sys.platform == 'win32' else 'chroma')
    result = subprocess.run([str(command), 'run', '--path', str(tmp_path / 'must-not-create'),
                             '--host', '127.0.0.1', '--port', '0'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'HTTP servers' in result.stderr and 'disabled' in result.stderr
    assert not (tmp_path / 'must-not-create').exists()


def test_server_handlers_are_absent_from_installed_python_source():
    package = distribution('chromadb')
    assert package.version == '1.5.9+instinct.security2'
    for name in ('chromadb/server/__init__.py', 'chromadb/server/fastapi/__init__.py'):
        source = package.locate_file(name).read_text(encoding='utf-8')
        assert source.startswith('raise RuntimeError(')
        assert len(source.splitlines()) == 1


def test_embedded_storage_survives_server_disable(tmp_path):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(tmp_path / 'synthetic'),
                                       settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection('synthetic_server_disable', embedding_function=None)
    collection.add(ids=['synthetic'], documents=['Synthetic local record'],
                   embeddings=[[1.0, 0.0, 0.0]])
    assert collection.get(ids=['synthetic'])['documents'] == ['Synthetic local record']
    assert collection.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)['ids'] == [['synthetic']]
    collection.delete(ids=['synthetic'])
    assert collection.count() == 0
