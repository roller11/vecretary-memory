# SPDX-License-Identifier: MPL-2.0
"""Prepare a local draft tree, never publish or copy runtime data/history."""

import argparse
import hashlib
import json
from pathlib import Path
import re


FIXED = {
    '.github/FUNDING.yml': '.github/FUNDING.yml',
    'LICENSE': 'LICENSE',
    'CONTRIBUTING.md': 'CONTRIBUTING.md',
    'SECURITY.md': 'SECURITY.md',
    'THIRD_PARTY_NOTICES.md': 'THIRD_PARTY_NOTICES.md',
    'docs/WISDOM_BOUNDARY.md': 'docs/WISDOM_BOUNDARY.md',
    'docs/SQLITE_SERVICE_INTEGRATION.md': 'docs/SQLITE_SERVICE_INTEGRATION.md',
    'docs/DASHBOARD_AUTH_SECURITY.md': 'docs/DASHBOARD_AUTH_SECURITY.md',
    'docs/SQLITE_OFFLINE_BACKUP.md': 'docs/SQLITE_OFFLINE_BACKUP.md',
    'docs/SQLITE_PENDING_WRITES.md': 'docs/SQLITE_PENDING_WRITES.md',
    'docs/SQLITE_PURGE.md': 'docs/SQLITE_PURGE.md',
    'docs/SQLITE_QUEUE_REVIEW.md': 'docs/SQLITE_QUEUE_REVIEW.md',
    'docs/RC_ALPHA_NOTES.md': 'docs/RC_ALPHA_NOTES.md',
    'docs/RELEASE_GATES.md': 'docs/RELEASE_GATES.md',
    'docs/PUBLIC_README.md': 'README.md',
    'docs/PUBLIC_CLIENT_ACCEPTANCE.md': 'docs/CLIENT_ACCEPTANCE.md',
    'pyproject.toml': 'pyproject.toml',
    'MANIFEST.in': 'MANIFEST.in',
    '.gitignore': '.gitignore',
    'instinct/starter_memories.json': 'instinct/starter_memories.json',
    'examples/client-instructions.md': 'examples/client-instructions.md',
    'examples/mcp.json.example': 'examples/mcp.json.example',
    'tools/export_public_draft.py': 'tools/export_public_draft.py',
    'tools/check_installed.py': 'tools/check_installed.py',
    'tools/audit_dependencies.py': 'tools/audit_dependencies.py',
    'tools/check_sqlite_provider.py': 'tools/check_sqlite_provider.py',
    'tools/collect_dependency_licenses.py': 'tools/collect_dependency_licenses.py',
    'tools/collect_sqlite_runtime.py': 'tools/collect_sqlite_runtime.py',
    'tests/dashboard_capability_recovery.mjs': 'tests/dashboard_capability_recovery.mjs',
    'tools/build_chroma_security_candidate.py': 'tools/build_chroma_security_candidate.py',
    '.github/workflows/ci.yml': '.github/workflows/ci.yml',
}
PATTERNS = (
    re.compile(r'(?i)[a-z]:[\\/](?:dev|users)[\\/]'),
    re.compile(r'/home/(?!me/|user/|username/)[\w.-]+/'),
    re.compile(r'/mnt/[a-z]/(?:dev|Users)/', re.IGNORECASE),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\b(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,})\b'),
)

SQLITE_OVERRIDES = {
    'docs/sqlite-release/RELEASE_NOTES.md': 'RELEASE_NOTES.md',
    'docs/sqlite-release/pyproject.toml': 'pyproject.toml',
    'docs/sqlite-release/README.md': 'README.md',
    'docs/sqlite-release/SECURITY.md': 'SECURITY.md',
    'docs/sqlite-release/THIRD_PARTY_NOTICES.md': 'THIRD_PARTY_NOTICES.md',
    'docs/sqlite-release/ci.yml': '.github/workflows/ci.yml',
    'docs/sqlite-release/mcp.json.example': 'examples/mcp.json.example',
    'docs/sqlite-release/RC_ALPHA_NOTES.md': 'docs/RC_ALPHA_NOTES.md',
    'docs/sqlite-release/RELEASE_GATES.md': 'docs/RELEASE_GATES.md',
    'docs/sqlite-release/CLIENT_ACCEPTANCE.md': 'docs/CLIENT_ACCEPTANCE.md',
    'docs/sqlite-release/requirements-linux.txt': 'requirements-linux.txt',
    'tools/check_sqlite_installed.py': 'tools/check_sqlite_installed.py',
}


def prepare(source, destination, *, profile='legacy'):
    if profile not in ('legacy', 'sqlite'):
        raise ValueError('Unknown release profile')
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.exists() or source == destination or source.is_relative_to(destination):
        raise ValueError('Destination must be new and must not contain the source')
    selected = dict(FIXED)
    if profile == 'sqlite':
        replaced = set(SQLITE_OVERRIDES.values())
        selected = {original: target for original, target in selected.items()
                    if target not in replaced and original not in (
                        'tools/check_installed.py', 'tools/build_chroma_security_candidate.py')}
        selected.update(SQLITE_OVERRIDES)
    for directory in ('instinct', 'tests'):
        root = source / directory
        if root.is_symlink() or root.resolve().parent != source:
            raise ValueError('Unsafe source directory')
        for file in sorted(root.glob('*.py')):
            name = file.relative_to(source).as_posix()
            selected[name] = name
    payloads = {}
    inventory = []
    for original, target in sorted(selected.items()):
        file = source / original
        if file.is_symlink() or not file.resolve(strict=True).is_relative_to(source):
            raise ValueError(f'Unsafe source file: {original}')
        content = file.read_bytes()
        if profile == 'sqlite' and target == 'MANIFEST.in':
            content = content.replace(b'include tools/build_chroma_security_candidate.py\n', b'')
            content = content.replace(b'include tools/build_chroma_security_candidate.py\r\n', b'')
        text = content.decode('utf-8')
        if any(pattern.search(text) for pattern in PATTERNS):
            raise ValueError(f'Privacy/secret review required: {original}')
        payloads[target] = content
        inventory.append({'source': original, 'path': target,
                          'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
        if profile == 'sqlite' and original.startswith('docs/sqlite-release/'):
            payloads[original] = content
            inventory.append({'source': original, 'path': original,
                              'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
    destination.mkdir(parents=True, mode=0o700)
    for target, content in payloads.items():
        file = destination / target
        file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with file.open('xb') as output:
            output.write(content)
    report = {'status': 'DRAFT_ONLY_NOT_CLEARED_FOR_PUBLICATION', 'profile': profile, 'files': inventory,
              'limits': ['Targeted patterns are not a complete secret scan or manual review',
                         'MPL-2.0 selected; final provenance and distribution review still required',
                         'Private operational docs, history, stores and build output are excluded']}
    with (destination / 'DRAFT_EXPORT.json').open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--profile', choices=('legacy', 'sqlite'), default='legacy')
    arguments = parser.parse_args()
    report = prepare(arguments.source, arguments.destination, profile=arguments.profile)
    print(json.dumps({'status': report['status'], 'files': len(report['files'])}))


if __name__ == '__main__':
    main()
