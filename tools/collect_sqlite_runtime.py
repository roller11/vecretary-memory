# SPDX-License-Identifier: MPL-2.0
"""Record the effective SQLite native runtime; never grant advisory clearance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import _sqlite3
import subprocess
import sys


def file_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def inspect_linkage(target):
    if platform.system() != 'Linux':
        return {'status': 'NOT_COLLECTED_ON_THIS_PLATFORM'}
    try:
        result = subprocess.run(['ldd', str(target)], capture_output=True, text=True,
                                check=False, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        return {'status': 'INCOMPLETE', 'error_type': type(error).__name__}
    return {'status': 'COLLECTED' if result.returncode == 0 else 'INCOMPLETE',
            'returncode': result.returncode, 'output': result.stdout,
            'stderr': result.stderr}


def loaded_sqlite_paths():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith('libsqlite3.so'):
            paths.add(Path(fields[5]).resolve(strict=True))
    if len(paths) != 1:
        raise ValueError('Exactly one loaded SQLite shared library is required')
    return sorted(paths)


def package_provenance(library):
    ownership = subprocess.run(['dpkg-query', '-S', str(library)], capture_output=True,
                               text=True, check=False, timeout=15)
    owners = [line.rsplit(': ', 1)[0] for line in ownership.stdout.splitlines()
              if ': ' in line and Path(line.rsplit(': ', 1)[1]).resolve() == library]
    if ownership.returncode != 0 or len(owners) != 1:
        raise ValueError('Loaded SQLite library package ownership is unresolved')
    package = owners[0]
    format_string = '${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\t${Architecture}\t${Status}\n'
    metadata = subprocess.run(['dpkg-query', '-W', '-f', format_string, package],
                              capture_output=True, text=True, check=False, timeout=15)
    fields = metadata.stdout.strip().split('\t')
    if metadata.returncode != 0 or len(fields) != 6 or fields[5] != 'install ok installed':
        raise ValueError('SQLite package metadata is incomplete')
    sums = subprocess.run(['dpkg-query', '--control-show', package, 'md5sums'],
                          capture_output=True, text=True, check=False, timeout=15)
    checksums = []
    for line in sums.stdout.splitlines():
        checksum, separator, relative = line.partition('  ')
        if separator and Path('/' + relative).resolve() == library:
            checksums.append(checksum)
    if sums.returncode != 0 or len(checksums) != 1:
        raise ValueError('SQLite package file checksum is unresolved')
    with library.open('rb') as source:
        actual = hashlib.file_digest(source, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
    if actual != checksums[0]:
        raise ValueError('Loaded SQLite library differs from its package checksum')
    return {'binary_package': fields[0], 'version': fields[1], 'source_package': fields[2],
            'source_version': fields[3], 'architecture': fields[4],
            'package_database_md5': checksums[0], 'package_checksum_matches': True,
            'limits': ['Local package database is not independent archive-signature verification']}


def inspect_native_sqlite():
    if platform.system() != 'Linux':
        return {'status': 'NOT_COLLECTED_ON_THIS_PLATFORM'}
    report = {'status': 'INCOMPLETE'}
    try:
        library, = loaded_sqlite_paths()
        report.update({'loaded_library': str(library), 'loaded_library_sha256': file_hash(library),
                       'os_release': Path('/etc/os-release').read_text()})
        report['package'] = package_provenance(library)
        report['status'] = 'COLLECTED_REQUIRES_ADVISORY_REVIEW'
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report.update({'error_type': type(error).__name__, 'reason': str(error)})
    return report


def collect():
    connection = sqlite3.connect(':memory:')
    try:
        source_id = connection.execute('SELECT sqlite_source_id()').fetchone()[0]
        options = sorted(row[0] for row in connection.execute('PRAGMA compile_options'))
    finally:
        connection.close()
    binding = getattr(_sqlite3, '__file__', None)
    executable = Path(sys.executable).resolve(strict=True)
    target = Path(binding).resolve(strict=True) if binding else executable
    return {'status': 'EVIDENCE_REQUIRES_NATIVE_REVIEW',
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'python_version': platform.python_version(), 'platform': platform.platform(),
            'sqlite_version': sqlite3.sqlite_version, 'sqlite_source_id': source_id,
            'compile_options': options, 'binding_kind': 'shared_extension' if binding else 'builtin',
            'python_executable': str(executable), 'python_executable_sha256': file_hash(executable),
            'linkage_target': str(target), 'linkage_target_sha256': file_hash(target),
            'linkage': inspect_linkage(target),
            'native_sqlite': inspect_native_sqlite(),
            'limits': ['In-memory inspection only; no application databases opened',
                       'Version, hashes and linkage are not complete native build provenance',
                       'Distro backports apply only to the corresponding effective native build',
                       'Pip dependency scans do not clear embedded or system SQLite advisories']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError('Runtime evidence output must be a new file')
    report = collect()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open('x', encoding='ascii') as output:
        json.dump(report, output, indent=2, ensure_ascii=True)
    print(json.dumps({'status': report['status'], 'python_version': report['python_version'],
                      'sqlite_version': report['sqlite_version'], 'binding_kind': report['binding_kind']}))
    return 2 if 'INCOMPLETE' in (report['linkage']['status'], report['native_sqlite']['status']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
