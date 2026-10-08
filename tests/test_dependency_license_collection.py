# SPDX-License-Identifier: MPL-2.0
"""Offline checks for bounded installed license evidence collection."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('license_collection',
    Path(__file__).resolve().parents[1] / 'tools/collect_dependency_licenses.py')
licenses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(licenses)


def installed(root, files):
    return SimpleNamespace(metadata={'Name': 'synthetic', 'License-Expression': 'MIT'},
        version='1.0', files=files, locate_file=lambda path: root / path)


def test_text_hash_and_declared_license_are_collected(tmp_path):
    content = b'Synthetic license evidence\n'
    (tmp_path / 'LICENSE').write_bytes(content)
    report = licenses.collect([installed(tmp_path, ['LICENSE'])], tmp_path)
    assert report['status'] == 'EVIDENCE_COLLECTED_REQUIRES_REVIEW'
    package = report['packages'][0]
    assert package['license_expression'] == 'MIT'
    assert package['license_files'][0]['sha256'] == hashlib.sha256(content).hexdigest()
    assert package['license_files'][0]['text'] == content.decode()


def test_license_named_python_modules_are_not_license_evidence(tmp_path):
    report = licenses.collect([installed(tmp_path, ['licenses.py', 'licenses.pyc'])], tmp_path)
    assert report['status'] == 'INCOMPLETE'
    assert report['missing_license_files'] == ['synthetic']


def test_license_path_escape_is_refused(tmp_path):
    prefix = tmp_path / 'environment'
    prefix.mkdir()
    (tmp_path / 'LICENSE').write_text('Outside installed environment')
    with pytest.raises(ValueError, match='outside'):
        licenses.collect([installed(prefix, ['../LICENSE'])], prefix)


def test_oversized_license_is_refused(tmp_path, monkeypatch):
    (tmp_path / 'LICENSE').write_bytes(b'12345')
    monkeypatch.setattr(licenses, 'MAX_LICENSE_BYTES', 4)
    with pytest.raises(ValueError, match='limit'):
        licenses.collect([installed(tmp_path, ['LICENSE'])], tmp_path)


def test_empty_inventory_is_refused(tmp_path):
    with pytest.raises(ValueError, match='No installed'):
        licenses.collect([], tmp_path)


def test_missing_license_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        licenses.collect([installed(tmp_path, ['LICENSE'])], tmp_path)


def test_invalid_license_encoding_is_refused(tmp_path):
    (tmp_path / 'LICENSE').write_bytes(b'\xff')
    with pytest.raises(UnicodeDecodeError):
        licenses.collect([installed(tmp_path, ['LICENSE'])], tmp_path)


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / 'evidence.json'
    output.write_text('Existing reviewed evidence')
    monkeypatch.setattr(licenses.sys, 'argv', ['collector', '--output', str(output)])
    with pytest.raises(ValueError, match='new file'):
        licenses.main()
    assert output.read_text() == 'Existing reviewed evidence'
