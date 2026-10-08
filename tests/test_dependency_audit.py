# SPDX-License-Identifier: MPL-2.0
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'audit_dependencies', Path(__file__).resolve().parents[1] / 'tools/audit_dependencies.py',
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_requirements_audits_external_packages_only_without_private_install_paths():
    result = audit.requirements([
        {'name': 'instinct_context', 'version': '0.1.0.dev0'},
        {'name': 'vecretary_memory', 'version': '0.1.0a1'},
        {'name': 'chromadb', 'version': '1.5.9'},
        {'name': 'httpx', 'version': '0.28.1'},
    ])
    assert result == 'chromadb==1.5.9\nhttpx==0.28.1\n'


@pytest.mark.parametrize('name,version', [('bad\npackage', '1.0'), ('package', '1.0\n--option')])
def test_requirement_injection_is_refused(name, version):
    with pytest.raises(ValueError):
        audit.requirements([{'name': name, 'version': version}])


@pytest.mark.parametrize('report,returncode,expected', [
    ({'dependencies': [{'name': 'safe', 'vulns': []}]}, 0, 0),
    ({'dependencies': [{'name': 'unsafe', 'vulns': [{'id': 'synthetic-advisory'}]}]}, 1, 1),
    ({'dependencies': [{'name': 'unsafe', 'vulns': [{'id': 'synthetic-advisory'}]}]}, 0, 1),
    ({'dependencies': [{'name': 'unknown', 'skip_reason': 'Not found'}]}, 0, 2),
    ({'dependencies': []}, 0, 2),
    ({}, 0, 2),
    ({'dependencies': [{'name': 'safe', 'vulns': []}]}, 1, 2),
    ({'dependencies': [{'name': 'safe', 'vulns': []}]}, 2, 2),
])
def test_advisories_and_incomplete_scans_block_release(report, returncode, expected):
    assert audit.verdict(report, returncode) == expected


def test_missing_dependency_cannot_silently_pass():
    report = {'dependencies': [{'name': 'safe', 'version': '1.0', 'vulns': []}]}
    assert audit.verdict(report, 0, {('safe', '1.0'), ('missing', '2.0')}) == 2
    assert audit.verdict(report, 0, {('safe', '1.0')}) == 0


@pytest.mark.parametrize('version', ['1.5.9+instinct.security1', '1.5.9+instinct.security2'])
def test_custom_chroma_version_does_not_escape_base_version_advisories(version):
    assert audit.requirements([{'name': 'chromadb', 'version': version}]) == 'chromadb==1.5.9\n'
