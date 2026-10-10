# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import hashlib
import json
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_project_license_dependencies_and_entrypoints_match_sqlite_profile():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['name'] == 'vecretary-memory'
    assert project['license'] == 'MPL-2.0'
    assert project['license-files'] == ['LICENSE']
    assert project['requires-python'] == '>=3.11'
    assert not any('chroma' in dependency.lower() for dependency in project['dependencies'])
    assert not any('wisdom' in dependency.lower() for dependency in project['dependencies'])
    assert project['scripts'] == {
        'vecretary-mcp': 'instinct.sqlite_service:cli',
        'vecretary-init': 'instinct.sqlite_init:cli',
        'vecretary-sqlite': 'instinct.sqlite_service:cli',
    }


def test_readme_describes_current_features_and_limitations():
    readme = ' '.join((ROOT / 'README.md').read_text(encoding='utf-8').split())
    for section in ('## Benefits and proposed uses', '## Implemented features',
                    '## Fresh isolated setup', '## Performance and release limitations',
                    '## Future importance feedback', '## License, Wisdom and support'):
        assert section in readme
    for phrase in ('agent/vendor agnostic', 'exact linear cosine scans',
                   'both temporary-pop mechanisms', 'Active-session timer',
                   'pending-save/score capture', 'single-record recycled purge',
                   'not added by this release', 'no alpha SLA is promised'):
        assert phrase in readme
    assert 'chromadb' not in readme.lower()
    assert 'instinct_delete_memory' not in readme


def test_public_export_manifest_matches_every_selected_payload():
    manifest = json.loads((ROOT / 'DRAFT_EXPORT.json').read_text(encoding='utf-8'))
    records = manifest['files']
    paths = [record['path'] for record in records]
    assert len(paths) == len(set(paths))
    assert 'tests/test_chroma_candidate_builder.py' not in paths
    assert not (ROOT / 'tests/test_chroma_candidate_builder.py').exists()
    for record in records:
        payload = ROOT / record['path']
        content = payload.read_bytes()
        assert len(content) == record['bytes'], record['path']
        assert hashlib.sha256(content).hexdigest() == record['sha256'], record['path']


def test_premium_boundary_does_not_select_wisdom_license_or_require_purchase():
    boundary = (ROOT / 'docs/WISDOM_BOUNDARY.md').read_text(encoding='utf-8')
    assert 'license and distribution terms remain to be specified by the owner' in boundary
    assert 'Neither Wisdom nor a support purchase is required' in boundary
    assert 'Keeping Wisdom in another repository' in boundary


def test_rebranded_commands_preserve_legacy_entrypoints():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert 'vecretary-mcp' in project['scripts']
    assert 'vecretary-init' in project['scripts']
    assert 'vecretary-sqlite' in project['scripts']
    assert not any(name.startswith('instinct-') for name in project['scripts'])
    example = (ROOT / 'examples/mcp.json.example').read_text(encoding='utf-8')
    assert 'vecretary-mcp' in example


def test_dashboard_branding_retains_capability_protocol():
    source = (ROOT / 'instinct/dashboard.py').read_text(encoding='utf-8')
    assert '<title>Vecretary Memory Dashboard</title>' in source
    assert '<title>Instinct' not in source
    assert 'X-Instinct-Capability' in source


def test_canonical_license_and_intake_policy_are_shipped():
    license_text = (ROOT / 'LICENSE').read_text(encoding='utf-8')
    assert license_text.startswith('Mozilla Public License Version 2.0')
    assert 'Exhibit A' in license_text
    assert 'Exhibit B' in license_text
    policy = (ROOT / 'CONTRIBUTING.md').read_text(encoding='utf-8')
    assert 'External code contributions and pull requests are not' in policy
    assert 'not an additional license condition' in policy


def test_security_policy_matches_local_alpha_boundary():
    policy = (ROOT / 'SECURITY.md').read_text(encoding='utf-8').lower()
    assert 'no login or https is implemented' in policy
    assert 'single-owner local use' in policy
    assert 'security@raw-wire.com' in policy
    assert 'no alpha\nresponse sla is promised' in policy
    assert 'all installed dependencies must' in policy
    assert 'guarantee this software is free of vulnerabilities' in policy


def test_reporting_and_provenance_remain_bounded():
    policy = (ROOT / 'SECURITY.md').read_text(encoding='utf-8')
    assert 'security@raw-wire.com' in policy
    assert 'successful inbox' in policy
    assert 'No alpha\nresponse SLA is promised' in policy
    gates = ' '.join((ROOT / 'docs/RELEASE_GATES.md').read_text(encoding='utf-8').split())
    assert 'personally held rights' in gates
    assert 'not a legal certification or third-party licensing clearance' in gates
    assert 'Dependency/model licensing inventory' in gates
    assert 'Explicit final publication approval' in gates


def test_ci_bootstraps_reviewed_installer_and_keeps_audit_blocking():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
    assert 'python -m pip install --upgrade pip==26.2.1' in workflow
    assert 'setuptools==83.0.0' in workflow
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert 'setuptools>=83' in config['build-system']['requires']
    patterns = config['tool']['pytest']['ini_options']['python_files']
    assert 'test_release_policy.py' in patterns
    assert 'test_runtime.py' in patterns
    assert 'test_sqlite*.py' in patterns
    assert 'test_chroma_candidate_builder.py' not in patterns
    assert 'python tools/audit_dependencies.py' in workflow
    assert 'python -m pip install pip-audit' in workflow
    assert "python: ['3.11', '3.12']" in workflow
    assert 'tests/test_release_policy.py' in workflow
    assert 'continue-on-error' not in workflow
    assert 'if: always()' in workflow
    assert 'build/dependency-audit.json' in workflow
