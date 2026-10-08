# SPDX-License-Identifier: MPL-2.0
"""Offline native runtime evidence checks; no application database access."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('sqlite_runtime_collection',
    Path(__file__).resolve().parents[1] / 'tools/collect_sqlite_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_file_hash_matches_contents(tmp_path):
    target = tmp_path / 'synthetic-binary'
    target.write_bytes(b'Synthetic')
    assert runtime.file_hash(target) == hashlib.sha256(b'Synthetic').hexdigest()


def test_linkage_uses_argument_list_and_target(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    def run(arguments, **options):
        calls.append((arguments, options))
        return SimpleNamespace(returncode=0, stdout='Synthetic linkage', stderr='')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    target = tmp_path / 'synthetic binary'
    assert runtime.inspect_linkage(target)['status'] == 'COLLECTED'
    assert calls[0][0] == ['ldd', str(target)]
    assert calls[0][1]['timeout'] == 15


def test_missing_linker_is_incomplete(monkeypatch):
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    def unavailable(*arguments, **options):
        raise FileNotFoundError()
    monkeypatch.setattr(runtime.subprocess, 'run', unavailable)
    assert runtime.inspect_linkage('synthetic')['status'] == 'INCOMPLETE'


def test_nonlinux_linkage_is_not_claimed(monkeypatch):
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Windows')
    assert runtime.inspect_linkage('synthetic')['status'] == 'NOT_COLLECTED_ON_THIS_PLATFORM'


def test_builtin_binding_inspects_interpreter(monkeypatch):
    monkeypatch.setattr(runtime, '_sqlite3', SimpleNamespace())
    monkeypatch.setattr(runtime, 'inspect_linkage', lambda target: {'status': 'COLLECTED'})
    report = runtime.collect()
    assert report['binding_kind'] == 'builtin'
    assert report['linkage_target'] == report['python_executable']
    assert report['status'] == 'EVIDENCE_REQUIRES_NATIVE_REVIEW'


def test_shared_binding_inspects_extension(monkeypatch, tmp_path):
    target = tmp_path / 'synthetic-extension'
    target.write_bytes(b'Synthetic extension')
    monkeypatch.setattr(runtime, '_sqlite3', SimpleNamespace(__file__=str(target)))
    monkeypatch.setattr(runtime, 'inspect_linkage', lambda target: {'status': 'COLLECTED'})
    report = runtime.collect()
    assert report['binding_kind'] == 'shared_extension'
    assert report['linkage_target'] == str(target.resolve())


def test_existing_evidence_is_not_overwritten(monkeypatch, tmp_path):
    output = tmp_path / 'evidence.json'
    output.write_text('Reviewed evidence')
    monkeypatch.setattr(runtime.sys, 'argv', ['collector', '--output', str(output)])
    with pytest.raises(ValueError, match='new file'):
        runtime.main()
    assert output.read_text() == 'Reviewed evidence'


def test_native_library_hash_and_package_identity_are_collected(monkeypatch, tmp_path):
    library = tmp_path / 'libsqlite3.so.0'
    library.write_bytes(b'Synthetic SQLite')
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(runtime, 'loaded_sqlite_paths', lambda: [library])
    monkeypatch.setattr(runtime, 'package_provenance', lambda path: {'version': 'synthetic'})
    original_read_text = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path: 'Synthetic OS' if path == Path('/etc/os-release')
                        else original_read_text(path))
    report = runtime.inspect_native_sqlite()
    assert report['status'] == 'COLLECTED_REQUIRES_ADVISORY_REVIEW'
    assert report['loaded_library_sha256'] == hashlib.sha256(b'Synthetic SQLite').hexdigest()
    assert report['package']['version'] == 'synthetic'


def test_unresolved_native_library_is_explicitly_incomplete(monkeypatch):
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    def unresolved():
        raise ValueError('Unresolved native library')
    monkeypatch.setattr(runtime, 'loaded_sqlite_paths', unresolved)
    assert runtime.inspect_native_sqlite()['status'] == 'INCOMPLETE'


def test_nonlinux_native_provenance_is_not_claimed(monkeypatch):
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Windows')
    assert runtime.inspect_native_sqlite()['status'] == 'NOT_COLLECTED_ON_THIS_PLATFORM'


def test_package_checksum_and_architecture_bind_loaded_bytes(monkeypatch, tmp_path):
    library = tmp_path / 'libsqlite3.so.0'
    library.write_bytes(b'Synthetic SQLite')
    checksum = hashlib.md5(b'Synthetic SQLite', usedforsecurity=False).hexdigest()
    outputs = [f'libsqlite3-0:amd64: {library}\n',
               'libsqlite3-0:amd64\t3.45.1-1ubuntu2.8\tsqlite3\t3.45.1-1ubuntu2.8\tamd64\tinstall ok installed\n',
               f'{checksum}  {library.as_posix().lstrip("/")}\n']
    calls = []
    def run(arguments, **options):
        calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout=outputs.pop(0), stderr='')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    if runtime.platform.system() != 'Linux':
        pytest.skip('POSIX package-file names are Linux evidence')
    report = runtime.package_provenance(library)
    assert report['package_checksum_matches']
    assert report['architecture'] == 'amd64'
    assert calls[0] == ['dpkg-query', '-S', str(library)]
    assert calls[2] == ['dpkg-query', '--control-show', 'libsqlite3-0:amd64', 'md5sums']


def test_package_checksum_mismatch_is_rejected(monkeypatch, tmp_path):
    if runtime.platform.system() != 'Linux':
        pytest.skip('POSIX package-file names are Linux evidence')
    library = tmp_path / 'libsqlite3.so.0'
    library.write_bytes(b'Synthetic SQLite')
    outputs = [f'libsqlite3-0:amd64: {library}\n',
               'libsqlite3-0:amd64\t3.45.1-1ubuntu2.8\tsqlite3\t3.45.1-1ubuntu2.8\tamd64\tinstall ok installed\n',
               f'{"0" * 32}  {library.as_posix().lstrip("/")}\n']
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *arguments, **options:
                        SimpleNamespace(returncode=0, stdout=outputs.pop(0), stderr=''))
    with pytest.raises(ValueError, match='differs'):
        runtime.package_provenance(library)


def test_incomplete_native_evidence_fails_collector_exit(monkeypatch, tmp_path):
    output = tmp_path / 'evidence.json'
    monkeypatch.setattr(runtime.sys, 'argv', ['collector', '--output', str(output)])
    monkeypatch.setattr(runtime, 'collect', lambda: {'status': 'EVIDENCE_REQUIRES_NATIVE_REVIEW',
        'python_version': 'synthetic', 'sqlite_version': 'synthetic', 'binding_kind': 'shared_extension',
        'linkage': {'status': 'COLLECTED'}, 'native_sqlite': {'status': 'INCOMPLETE'}})
    assert runtime.main() == 2
    assert output.exists()
