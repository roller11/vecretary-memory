# SPDX-License-Identifier: MPL-2.0
"""Owner-operated offline directory backups; never replace an existing store."""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

from .chroma_writer_lock import acquire_writer_lock, is_writer_lock_held, release_writer_lock
from .private_storage import (
    check_private_file, ensure_private_directory, atomic_private_writer,
    private_binary_reader, write_private_text,
)


FORMAT = 'instinct-offline-backup-v1'
MAX_FILES = 100000
MAX_BYTES = 10 * 1024 ** 3
MAX_MANIFEST_BYTES = 16 * 1024 ** 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_versions():
    result = {}
    for package in ('instinct-context', 'chromadb'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = 'uninstalled'
    return result


def relative_name(name):
    require(isinstance(name, str) and bool(name), 'Invalid manifest path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and not any(part in ('.', '..') for part in path.parts)
            and path.as_posix() == name and '\\' not in name and ':' not in name
            and '\x00' not in name and name != '.' and path.parts[0] != '.writer.lock', 'Unsafe manifest path')
    return name


def existing_directory(path):
    selected = Path(path).expanduser().absolute()
    require(selected.exists(), 'Source directory does not exist')
    return ensure_private_directory(selected)


def destination_for(source, destination):
    selected = Path(destination).expanduser().absolute()
    require(not selected.exists() and not selected.is_symlink(), 'Destination already exists; nothing overwritten')
    require(source != selected and source not in selected.parents and selected not in source.parents,
            'Source and destination must be separate directory trees')
    ensure_private_directory(selected.parent)
    return selected


def inventory(root, exclude_writer=False):
    directories, files = [], {}
    total = 0

    def refuse_walk_error(error):
        raise error

    for parent, children, names in os.walk(root, followlinks=False, onerror=refuse_walk_error):
        for child in sorted(children):
            selected = Path(parent) / child
            info = selected.lstat()
            require(stat.S_ISDIR(info.st_mode), 'Symlink or special directory refused')
            if os.name == 'posix':
                require(info.st_uid == os.geteuid() and not info.st_mode & 0o022, 'Unsafe source directory')
            directories.append(relative_name(selected.relative_to(root).as_posix()))
        for name in sorted(names):
            selected = Path(parent) / name
            if exclude_writer and selected == root / '.writer.lock':
                check_private_file(selected, confidential=False)
                continue
            relative = relative_name(selected.relative_to(root).as_posix())
            info = check_private_file(selected, confidential=False)
            total += info.st_size
            require(total <= MAX_BYTES and len(files) < MAX_FILES, 'Backup exceeds resource limits')
            files[relative] = info
        require(len(directories) <= MAX_FILES, 'Backup exceeds directory limit')
    return sorted(directories), files


def file_identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def stream_file(source, destination=None):
    before = check_private_file(source, confidential=False)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    digest, size = hashlib.sha256(), 0
    with os.fdopen(descriptor, 'rb') as incoming:
        opened = os.fstat(incoming.fileno())
        require(file_identity(before) == file_identity(opened), 'Source changed during open')

        def consume(output=None):
            nonlocal size
            while chunk := incoming.read(1024 * 1024):
                size += len(chunk)
                require(size <= MAX_BYTES, 'File exceeds resource limit')
                digest.update(chunk)
                if output is not None:
                    output.write(chunk)

        if destination is None:
            consume()
        else:
            with atomic_private_writer(destination, replace=False) as output:
                consume(output)
        after = os.fstat(incoming.fileno())
        require(file_identity(opened) == file_identity(after)
                and size == opened.st_size, 'Source changed while copying')
    return {'size': size, 'sha256': digest.hexdigest()}


@contextmanager
def offline_writer(directory):
    lock = directory / '.writer.lock'
    if lock.exists() or lock.is_symlink():
        check_private_file(lock, confidential=False)
    require(not is_writer_lock_held(str(directory)), 'Store is busy; close all clients before retrying')
    handle = acquire_writer_lock(str(directory), timeout=0, require_new=True)
    try:
        yield
    finally:
        release_writer_lock(handle)


def manifest_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate manifest key')
        result[key] = value
    return result


def verify(backup):
    root = existing_directory(backup)
    with private_binary_reader(root / 'manifest.json') as source:
        raw = source.read(MAX_MANIFEST_BYTES + 1)
    require(len(raw) <= MAX_MANIFEST_BYTES, 'Manifest exceeds resource limit')
    manifest = json.loads(raw, object_pairs_hook=manifest_object)
    require(isinstance(manifest, dict) and set(manifest) == {'format', 'versions', 'directories', 'files'}
            and manifest['format'] == FORMAT, 'Unknown backup format')
    require(manifest['versions'] == runtime_versions(), 'Version mismatch; cross-version restore is not supported')
    require(isinstance(manifest['files'], dict) and len(manifest['files']) <= MAX_FILES,
            'Invalid file inventory')
    require(isinstance(manifest['directories'], list) and len(manifest['directories']) <= MAX_FILES,
            'Invalid directory inventory')
    for name in [*manifest['directories'], *manifest['files']]:
        relative_name(name)
    require(manifest['directories'] == sorted(set(manifest['directories'])), 'Invalid directory inventory')
    total = 0
    for entry in manifest['files'].values():
        require(isinstance(entry, dict) and set(entry) == {'size', 'sha256'}, 'Invalid file metadata')
        require(type(entry['size']) is int and entry['size'] >= 0, 'Invalid file size')
        digest = entry['sha256']
        require(isinstance(digest, str) and len(digest) == 64
                and all(character in '0123456789abcdef' for character in digest), 'Invalid file digest')
        total += entry['size']
        require(total <= MAX_BYTES, 'Backup exceeds resource limits')
    require('chroma.sqlite3' in manifest['files'], 'Backup has no Chroma database')
    data = existing_directory(root / 'data')
    directories, files = inventory(data)
    require(directories == manifest['directories'] and set(files) == set(manifest['files']),
            'Missing or unexpected backup entries')
    for name in files:
        require(stream_file(data / name) == manifest['files'][name], 'Backup integrity check failed')
    return manifest


def create_backup(source, destination):
    source = existing_directory(source)
    require((source / 'chroma.sqlite3').is_file(), 'Source has no Chroma database')
    destination = destination_for(source, destination)
    with offline_writer(source):
        directories, files = inventory(source, exclude_writer=True)
        destination.mkdir(mode=0o700)
        data = ensure_private_directory(destination / 'data')
        for name in directories:
            ensure_private_directory(data / name)
        copied = {name: stream_file(source / name, data / name) for name in files}
        after_directories, after_files = inventory(source, exclude_writer=True)
        require(directories == after_directories and set(files) == set(after_files)
                and all(file_identity(info) == file_identity(after_files[name])
                        for name, info in files.items()), 'Source tree changed during backup')
        manifest = {'format': FORMAT, 'versions': runtime_versions(), 'directories': directories, 'files': copied}
        encoded = json.dumps(manifest, indent=2)
        require(len(encoded.encode()) <= MAX_MANIFEST_BYTES, 'Manifest exceeds resource limit')
        write_private_text(destination / 'manifest.json', encoded, replace=False)
        verify(destination)
    return {'ok': True, 'backup': str(destination), 'files': len(files)}


def restore_backup(backup, destination):
    backup = existing_directory(backup)
    destination = destination_for(backup, destination)
    manifest = verify(backup)
    destination.mkdir(mode=0o700)
    with offline_writer(destination):
        for name in manifest['directories']:
            ensure_private_directory(destination / name)
        for name, expected in manifest['files'].items():
            require(stream_file(backup / 'data' / name, destination / name) == expected,
                    'Backup changed during restore; destination is incomplete')
        directories, files = inventory(destination, exclude_writer=True)
        require(directories == manifest['directories'] and set(files) == set(manifest['files']),
                'Restored inventory differs')
        for name, expected in manifest['files'].items():
            require(stream_file(destination / name) == expected, 'Restored integrity check failed')
    return {'ok': True, 'restored': str(destination), 'files': len(manifest['files'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('backup', 'verify', 'restore'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination')
    parser.add_argument('--confirm-offline', action='store_true', help='Confirm all clients using these paths are closed')
    arguments = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Offline backup/restore currently supports Linux/WSL only')
    if arguments.operation != 'verify' and (not arguments.confirm_offline or not arguments.destination):
        parser.error('Mutating operations require --confirm-offline and a new --destination')
    try:
        if arguments.operation == 'backup':
            result = create_backup(arguments.source, arguments.destination)
        elif arguments.operation == 'restore':
            result = restore_backup(arguments.source, arguments.destination)
        else:
            manifest = verify(arguments.source)
            result = {'ok': True, 'files': len(manifest['files'])}
        print(json.dumps(result))
        return 0
    except Exception:
        print(json.dumps({'ok': False, 'error': 'Backup/recovery refused or failed. No existing destination was overwritten. Any new destination may be incomplete; do not start clients against it.'}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
