# SPDX-License-Identifier: MPL-2.0
"""Collect installed license evidence without granting legal clearance."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import sys


MAX_LICENSE_BYTES = 2 * 1024 * 1024
LICENSE_NAMES = {'copying', 'copying.txt', 'notice', 'notice.txt', 'authors', 'authors.rst'}


def is_license_file(path):
    name = Path(path).name.lower()
    return name in LICENSE_NAMES or (
        name.startswith('license') and Path(path).suffix.lower() not in {'.py', '.pyc'})


def collect_package(distribution, prefix):
    prefix = Path(prefix).resolve(strict=True)
    evidence = []
    for relative in sorted(distribution.files or [], key=str):
        if not is_license_file(relative):
            continue
        path = Path(distribution.locate_file(relative)).resolve(strict=True)
        if not path.is_relative_to(prefix) or not path.is_file():
            raise ValueError('License file is outside the installed environment or not a file')
        with path.open('rb') as source:
            content = source.read(MAX_LICENSE_BYTES + 1)
        if len(content) > MAX_LICENSE_BYTES:
            raise ValueError('License file exceeds evidence limit')
        evidence.append({'path': str(relative), 'sha256': hashlib.sha256(content).hexdigest(),
                         'bytes': len(content), 'text': content.decode('utf-8', errors='strict')})
    metadata = distribution.metadata
    return {'name': metadata['Name'], 'version': distribution.version,
            'license_expression': metadata.get('License-Expression'),
            'license_metadata': metadata.get('License'),
            'license_files': evidence, 'license_evidence_present': bool(evidence)}


def collect(installed, prefix):
    packages = [collect_package(package, prefix) for package in
                sorted(installed, key=lambda package: package.metadata['Name'].lower())]
    if not packages:
        raise ValueError('No installed packages found')
    missing = [package['name'] for package in packages if not package['license_evidence_present']]
    return {'status': 'INCOMPLETE' if missing else 'EVIDENCE_COLLECTED_REQUIRES_REVIEW',
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'packages': packages, 'missing_license_files': missing,
            'limits': ['Installed license texts and hashes, not legal certification',
                       'Native bundled-library, model and source provenance require separate review']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError('Evidence output must be a new file')
    report = collect(distributions(), sys.prefix)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open('x', encoding='ascii') as output:
        json.dump(report, output, indent=2, ensure_ascii=True)
    print(json.dumps({'status': report['status'], 'packages': len(report['packages']),
                      'missing_license_files': report['missing_license_files']}))
    return 2 if report['missing_license_files'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
