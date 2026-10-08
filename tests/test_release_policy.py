# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_project_license_and_dependency_are_explicit():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['name'] == 'vecretary-memory'
    assert project['license'] == 'MPL-2.0'
    assert project['license-files'] == ['LICENSE']
    assert 'chromadb==1.5.9+instinct.security2' in project['dependencies']
    assert not any('wisdom' in dependency.lower() for dependency in project['dependencies'])


def test_readmes_cover_product_features_and_separate_premium_offerings():
    readme_paths = [ROOT / 'README.md']
    public_template = ROOT / 'docs/PUBLIC_README.md'
    if public_template.is_file():
        readme_paths.append(public_template)
    for readme_path in readme_paths:
        readme = readme_path.read_text(encoding='utf-8')
        for section in ('## What it does', '### Benefits and proposed use cases',
                        '### Agent-agnostic functionality',
                        '### Importance alongside vector recall',
                        '## Functions and features',
                        '## License, premium modules and support'):
            assert section in readme
        for tool in ('instinct_get_context', 'instinct_get_mandatory',
                     'instinct_search', 'instinct_get', 'instinct_save_memory',
                     'instinct_update_score', 'instinct_delete_memory'):
            assert f'| `{tool}` |' in readme
        assert 'not\nimplemented or promised in this release' in readme
        assert 'whether Wisdom\'s source will be public' in readme
        assert '**Premium support** is planned as a separate offering' in readme
        assert 'There is no premium restriction added to the core license' in readme


def test_premium_boundary_does_not_select_wisdom_license_or_require_purchase():
    boundary = (ROOT / 'docs/WISDOM_BOUNDARY.md').read_text(encoding='utf-8')
    assert 'license and distribution terms remain to be specified by the owner' in boundary
    assert 'Neither Wisdom nor a support purchase is required' in boundary
    assert 'Keeping Wisdom in another repository' in boundary


def test_rebranded_commands_preserve_legacy_entrypoints():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    for command in ('mcp', 'dashboard', 'seed', 'backup'):
        assert project['scripts'][f'vecretary-{command}'] == project['scripts'][f'instinct-{command}']
    example = (ROOT / 'examples/mcp.json.example').read_text()
    assert '/bin/vecretary-mcp' in example


def test_dashboard_branding_retains_capability_protocol():
    source = (ROOT / 'instinct/dashboard.py').read_text()
    assert '<title>Vecretary Memory Dashboard</title>' in source
    assert '<title>Instinct' not in source
    assert 'X-Instinct-Capability' in source


def test_canonical_license_and_intake_policy_are_shipped():
    license_text = (ROOT / 'LICENSE').read_text()
    assert license_text.startswith('Mozilla Public License Version 2.0')
    assert 'Exhibit A' in license_text
    assert 'Exhibit B' in license_text
    policy = (ROOT / 'CONTRIBUTING.md').read_text()
    assert 'External code contributions and pull requests are not' in policy
    assert 'not an additional license condition' in policy


def test_security_policy_keeps_all_advisories_visible():
    policy = (ROOT / 'SECURITY.md').read_text()
    for advisory in ('CVE-2026-45829', 'CVE-2026-45833',
                     'CVE-2026-45830', 'CVE-2026-45831'):
        assert advisory in policy
    assert 'Public release-risk review is pending' in policy


def test_reporting_and_provenance_remain_bounded():
    policy = (ROOT / 'SECURITY.md').read_text()
    assert 'security@raw-wire.com' in policy
    assert 'delivery has not been' in policy
    assert 'There is no promised response SLA' in policy
    gates = (ROOT / 'docs/RELEASE_GATES.md').read_text()
    assert 'personally holding the project rights' in gates
    assert 'owner attestation, not independent legal certification' in gates
    assert 'upstream dependencies retain their own rights' in gates
    assert 'Owner-approved SPDX MPL-2.0 notices' in gates
    assert 'Known advisory findings remain blocking' in gates


def test_ci_bootstraps_reviewed_installer_and_keeps_audit_blocking():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    assert 'python -m pip install --upgrade pip==26.2.1' in workflow
    assert 'setuptools==83.0.0' in workflow
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert 'setuptools>=83' in config['build-system']['requires']
    assert 'python tools/build_chroma_security_candidate.py' in workflow
    assert '--find-links build/chroma-patched' in workflow
    assert 'python tools/audit_dependencies.py' in workflow
    assert 'continue-on-error' not in workflow
    assert 'if: always()' in workflow
    assert 'path: build/dependency-audit.json' in workflow
