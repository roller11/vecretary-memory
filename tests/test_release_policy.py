# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_project_license_and_dependency_are_explicit():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['name'] == 'vecretary-memory'
    assert project['license'] == 'MPL-2.0'
    assert project['license-files'] == ['LICENSE']
    assert {'mcp>=1.27,<2', 'httpx>=0.27,<1'} <= set(project['dependencies'])
    assert not any('chroma' in dependency.lower() for dependency in project['dependencies'])
    assert not any('wisdom' in dependency.lower() for dependency in project['dependencies'])


def test_readmes_cover_product_features_and_separate_premium_offerings():
    readme_paths = [ROOT / 'README.md']
    public_template = ROOT / 'docs/PUBLIC_README.md'
    if public_template.is_file():
        readme_paths.append(public_template)
    for readme_path in readme_paths:
        readme = readme_path.read_text(encoding='utf-8')
        for section in ('## Benefits and proposed uses', '## Implemented features',
                        '## MCP tools', '## Future importance feedback',
                        '## License, Wisdom and support'):
            assert section in readme
        for tool in ('instinct_get_context', 'instinct_get_mandatory',
                     'instinct_search', 'instinct_get', 'instinct_save_memory',
                     'instinct_delete_memory', 'instinct_update_score'):
            assert f'| `{tool}` |' in readme
        assert 'not added by this release' in readme
        assert 'agent/vendor agnostic' in readme
        assert 'vector recall' in readme
        assert 'Wisdom is a separate private/premium API module' in readme
        assert 'Premium support is offered separately' in readme
        assert 'recoverable recycle bin' in readme


def test_premium_boundary_does_not_select_wisdom_license_or_require_purchase():
    boundary = (ROOT / 'docs/WISDOM_BOUNDARY.md').read_text(encoding='utf-8')
    assert 'license and distribution terms remain to be specified by the owner' in boundary
    assert 'Neither Wisdom nor a support purchase is required' in boundary
    assert 'Keeping Wisdom in another repository' in boundary


def test_rebranded_commands_select_the_sqlite_service():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['scripts'] == {
        'vecretary-mcp': 'instinct.sqlite_service:cli',
        'vecretary-init': 'instinct.sqlite_init:cli',
        'vecretary-sqlite': 'instinct.sqlite_service:cli',
    }
    example = (ROOT / 'examples/mcp.json.example').read_text(encoding='utf-8')
    assert '"command": "vecretary-mcp"' in example


def test_dashboard_branding_retains_capability_protocol():
    source = (ROOT / 'instinct/dashboard.py').read_text(encoding='utf-8')
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
    policy = ' '.join((ROOT / 'SECURITY.md').read_text().split())
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert not any('chroma' in dependency.lower() for dependency in project['dependencies'])
    assert 'No external vector-store library' in policy
    assert 'does not repair an older installation' in policy
    assert 'or guarantee this software is free of vulnerabilities' in policy


def test_reporting_and_provenance_remain_bounded():
    policy = (ROOT / 'SECURITY.md').read_text()
    assert 'security@raw-wire.com' in policy
    assert 'successful inbox delivery/forwarding' in ' '.join(policy.split())
    assert 'No alpha response SLA is promised' in ' '.join(policy.split())
    gates = (ROOT / 'docs/RELEASE_GATES.md').read_text()
    assert 'owner attested in-house development and personally held rights' in ' '.join(gates.split())
    assert 'not a legal certification or third-party licensing clearance' in ' '.join(gates.split())
    notices = (ROOT / 'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')
    notices = ' '.join(notices.split())
    assert 'Dependencies retain their own licenses' in notices
    assert 'not automatic legal clearance' in notices
    assert 'Full sidecar/auxiliary-state migration' in gates
    assert 'Explicit final publication approval' in gates


def test_ci_bootstraps_reviewed_installer_and_keeps_audit_blocking():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    assert 'python -m pip install --upgrade pip==26.2.1 setuptools==83.0.0' in workflow
    assert 'setuptools==83.0.0' in workflow
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert 'setuptools>=83' in config['build-system']['requires']
    assert 'python -I tools/check_sqlite_installed.py' in workflow
    assert 'Chroma' not in workflow
    assert 'python tools/audit_dependencies.py' in workflow
    assert 'continue-on-error' not in workflow
    assert 'if: always()' in workflow
    assert 'build/dependency-audit.json' in workflow
