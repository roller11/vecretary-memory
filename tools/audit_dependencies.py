# SPDX-License-Identifier: MPL-2.0
"""Fail closed on dependency advisories or incomplete advisory checks."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def requirements(packages):
    selected = []
    for package in packages:
        name, version = package['name'], package['version']
        if re.sub(r'[-_.]+', '-', name).lower() in {'instinct-context', 'vecretary-memory'}:
            continue
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or not re.fullmatch(
            r'[A-Za-z0-9][A-Za-z0-9.+!_-]*', version,
        ):
            raise ValueError('Invalid installed dependency metadata')
        selected.append(f'{name}=={version}')
    if not selected:
        raise ValueError('No external dependencies selected')
    return '\n'.join(sorted(selected)) + '\n'


def verdict(report, returncode, expected=None):
    dependencies = report.get('dependencies')
    if not isinstance(dependencies, list) or not dependencies:
        return 2
    if any('skip_reason' in dependency or not isinstance(dependency.get('vulns'), list)
           for dependency in dependencies):
        return 2
    if expected is not None:
        observed = {(re.sub(r'[-_.]+', '-', dependency['name']).lower(), dependency['version'])
                    for dependency in dependencies}
        if observed != expected or len(dependencies) != len(expected):
            return 2
    vulnerable = any(dependency['vulns'] for dependency in dependencies)
    if returncode not in (0, 1) or (returncode == 1 and not vulnerable):
        return 2
    return 1 if vulnerable else 0


def audit(audit_python, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Audit output must be a new file')
    installed = subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--format=json'],
        capture_output=True, text=True, check=True, timeout=30,
    )
    packages = json.loads(installed.stdout)
    text = requirements(packages)
    with tempfile.TemporaryDirectory(prefix='instinct-dependency-audit-') as temporary:
        requirement_file = Path(temporary) / 'requirements.txt'
        requirement_file.write_text(text, encoding='utf-8')
        raw = Path(temporary) / 'audit.json'
        result = subprocess.run(
            [str(audit_python), '-m', 'pip_audit', '-r', str(requirement_file),
             '--no-deps', '--disable-pip', '--strict', '--format', 'json', '--output', str(raw)],
            capture_output=True, text=True, timeout=120,
        )
        if not raw.is_file():
            raise ValueError('Scanner produced no report; security clearance refused')
        report = json.loads(raw.read_text(encoding='utf-8'))
    expected = {(re.sub(r'[-_.]+', '-', name).lower(), version)
                for name, version in (line.split('==') for line in text.splitlines())}
    status = verdict(report, result.returncode, expected)
    findings = {vulnerability['id'] for dependency in report.get('dependencies', [])
                for vulnerability in dependency.get('vulns', [])}
    envelope = {
        'checked_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'status': ('PASS', 'VULNERABLE', 'INCOMPLETE')[status],
        'local_product_not_in_advisory_index': 'vecretary-memory (legacy instinct-context); separate source review required',
        'requirements': text.splitlines(),
        'unique_findings': sorted(findings),
        'scanner': report,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as output:
        json.dump(envelope, output, indent=2)
    print(json.dumps({'status': envelope['status'], 'unique_findings': sorted(findings)}))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-python', default=sys.executable)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        return audit(arguments.audit_python, arguments.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print('Dependency audit failed or is incomplete; no security clearance.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
