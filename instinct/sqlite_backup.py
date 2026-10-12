# SPDX-License-Identifier: MPL-2.0
"""Offline, single-collection SQLite snapshots; never a live migration tool."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .private_storage import ensure_private_directory, private_binary_reader, private_file_lock, write_private_text, atomic_private_writer, check_private_file
from .sqlite_guard import acquire_writer_guard, SCHEMA_VERSION, WRITER_LOCK_FILENAME
from .sqlite_records import EmbeddingIdentity, MAX_RECORDS, canonical_json
from .sqlite_session import RUNTIME_LOCK_FILENAME
from .sqlite_store import SQLiteRecordStore


SIDECARS = frozenset({'session_state.json', 'nudge_state.json', 'dormancy_config.json'})
IDENTITY_FILE = 'embedding-identity.json'
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 1024
MANIFEST = 'snapshot.json'


def _json_object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError('Non-finite JSON constant')


def _validate_identity_payload(payload, expected_identity):
    if payload is None:
        return
    try:
        value = json.loads(payload.decode('utf-8'),
                           object_pairs_hook=_json_object_without_duplicates,
                           parse_constant=_reject_json_constant)
        actual = EmbeddingIdentity.from_dict(value)
        canonical_json(value)
    except (UnicodeError, TypeError, ValueError, RecursionError) as error:
        raise ValueError('Invalid runtime embedding identity') from error
    if actual != expected_identity:
        raise ValueError('Runtime embedding identity mismatch')


def _directory(value):
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError('An existing absolute private directory is required')
    return ensure_private_directory(path)


def _read(path, limit=MAX_FILE_BYTES):
    with private_binary_reader(path) as source:
        payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError('Snapshot file exceeds size limit')
    return payload


def _payloads(root):
    result = {}
    allowed = SIDECARS | {IDENTITY_FILE, 'sqlite', 'fallback_pending',
                          RUNTIME_LOCK_FILENAME, MANIFEST}
    for path in root.iterdir():
        if path.name not in allowed:
            raise ValueError('Unknown runtime state; snapshot refused')
        if path.name == IDENTITY_FILE:
            result[path.name] = _read(path, 4096)
            try:
                value = json.loads(result[path.name].decode('utf-8'),
                                   object_pairs_hook=_json_object_without_duplicates,
                                   parse_constant=_reject_json_constant)
                EmbeddingIdentity.from_dict(value)
                canonical_json(value)
            except (UnicodeError, TypeError, ValueError, RecursionError) as error:
                raise ValueError('Invalid runtime embedding identity') from error
        elif path.name in SIDECARS:
            result[path.name] = _read(path, 1024 * 1024)
            state = json.loads(result[path.name])
            if not isinstance(state, dict):
                raise ValueError('Runtime sidecar must be a JSON object')
            canonical_json(state)
        elif path.name in (RUNTIME_LOCK_FILENAME, MANIFEST):
            check_private_file(path)
    database = _directory(root / 'sqlite')
    for path in database.iterdir():
        if path.name == 'memory.sqlite3':
            result['sqlite/memory.sqlite3'] = _read(path)
        elif path.name == WRITER_LOCK_FILENAME:
            check_private_file(path)
        else:
            raise ValueError('Unknown database state or journal; snapshot refused')
    if 'sqlite/memory.sqlite3' not in result:
        raise ValueError('Snapshot database missing')
    queue = root / 'fallback_pending'
    if queue.exists() or queue.is_symlink():
        _directory(queue)
        for path in queue.iterdir():
            if path.name == '.capture.lock':
                check_private_file(path)
            elif path.suffix == '.json' and not path.name.startswith('.'):
                result['fallback_pending/' + path.name] = _read(path, 1024 * 1024)
            else:
                raise ValueError('Unknown queued state; snapshot refused')
            if len(result) > MAX_FILES:
                raise ValueError('Snapshot file count exceeds limit')
    if sum(map(len, result.values())) > MAX_TOTAL_BYTES:
        raise ValueError('Snapshot exceeds total size limit')
    return result


def _inventory(payloads):
    return {name: {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            for name, payload in sorted(payloads.items())}


def _publish(destination, payloads, manifest=None):
    destination = Path(destination)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        raise ValueError('Destination must be a new absolute directory')
    parent = _directory(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix='.sqlite-snapshot-', dir=parent))
    for name, payload in payloads.items():
        with atomic_private_writer(staging / name, replace=False) as output:
            output.write(payload)
    if manifest is not None:
        write_private_text(staging / MANIFEST, json.dumps(manifest, sort_keys=True), replace=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Snapshot destination was created concurrently')
    os.rename(staging, destination)
    return destination


def snapshot_offline(root, destination, identity):
    """Capture a stopped cooperative runtime; preserve queues without replay."""
    root = _directory(root)
    destination = Path(destination)
    if destination.absolute().is_relative_to(root):
        raise ValueError('Snapshot destination must be outside the runtime')
    identity = EmbeddingIdentity.from_dict(identity.to_dict())
    with ExitStack() as stack:
        stack.enter_context(private_file_lock(root / RUNTIME_LOCK_FILENAME, timeout=0))
        guard = acquire_writer_guard(_directory(root / 'sqlite'))
        stack.callback(guard.__exit__, None, None, None)
        queue = root / 'fallback_pending'
        if queue.exists() or queue.is_symlink():
            stack.enter_context(private_file_lock(_directory(queue) / '.capture.lock', timeout=0))
        payloads = _payloads(root)
        _validate_identity_payload(payloads.get(IDENTITY_FILE), identity)
        with SQLiteRecordStore.open_read_only(root / 'sqlite', identity) as repository:
            count = repository.count()
            repository.list_records(limit=MAX_RECORDS, recycled='all')
        if _inventory(payloads) != _inventory(_payloads(root)):
            raise RuntimeError('Runtime state changed during offline snapshot')
        manifest = {'format': 'vecretary-sqlite-offline-1', 'schema': SCHEMA_VERSION,
                    'identity': identity.to_dict(), 'records': count,
                    'queue_policy': 'preserve-no-replay', 'files': _inventory(payloads)}
        return _publish(destination, payloads, manifest)


def _validated_snapshot(snapshot, identity):
    snapshot = _directory(snapshot)
    identity = EmbeddingIdentity.from_dict(identity.to_dict())
    manifest_bytes = _read(snapshot / MANIFEST, 1024 * 1024)
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest, dict) or set(manifest) !=
            {'format', 'schema', 'identity', 'records', 'queue_policy', 'files'} or
            manifest['format'] != 'vecretary-sqlite-offline-1' or
            type(manifest['schema']) is not int or manifest['schema'] != SCHEMA_VERSION or
            manifest['identity'] != identity.to_dict() or
            manifest['queue_policy'] != 'preserve-no-replay'):
        raise ValueError('Unsupported snapshot or embedding identity')
    payloads = _payloads(snapshot)
    _validate_identity_payload(payloads.get(IDENTITY_FILE), identity)
    if manifest['files'] != _inventory(payloads):
        raise ValueError('Snapshot integrity mismatch')
    with SQLiteRecordStore.open_read_only(snapshot / 'sqlite', identity) as repository:
        if type(manifest['records']) is not int or repository.count() != manifest['records']:
            raise ValueError('Snapshot record count mismatch')
        repository.list_records(limit=MAX_RECORDS, recycled='all')
    if manifest_bytes != _read(snapshot / MANIFEST, 1024 * 1024) or _inventory(payloads) != _inventory(_payloads(snapshot)):
        raise ValueError('Snapshot changed during validation')
    return payloads, manifest


def restore_new(snapshot, destination, identity):
    """Validate first, then restore only into a new isolated directory."""
    if Path(destination).absolute().is_relative_to(Path(snapshot).absolute()):
        raise ValueError('Restore destination must be outside the snapshot')
    payloads, manifest = _validated_snapshot(snapshot, identity)
    return _publish(destination, payloads)
