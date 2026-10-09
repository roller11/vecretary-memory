# SPDX-License-Identifier: MPL-2.0
"""Opt-in neutral record migration; never opens a legacy backend or switches a live one."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .private_storage import (
    atomic_private_writer, check_private_file, ensure_private_directory,
    private_binary_reader, write_private_text,
)
from .sqlite_records import MAX_MANIFEST_BYTES, MAX_RECORDS, EmbeddingIdentity, canonical_json, validate_records
from .sqlite_store import SQLiteRecordStore


FORMAT = 'vecretary-record-migration'
VERSION = 1
SCOPE = 'records_only'
BUNDLE_FORMAT = 'vecretary-sqlite-runtime-migration'
BUNDLE_VERSION = 1
BUNDLE_MANIFEST = 'bundle.json'
RUNTIME_SIDECARS = frozenset({'session_state.json', 'nudge_state.json', 'dormancy_config.json'})
MAX_STATE_BYTES = 1024 * 1024


def _checksum(value):
    return hashlib.sha256(canonical_json(value).encode('ascii')).hexdigest()


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def _read_json_object(raw, description):
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_object_without_duplicates,
                           parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError(f'Invalid {description} JSON') from error
    if not isinstance(value, dict):
        raise ValueError(f'{description} must be a JSON object')
    canonical_json(value)
    return value


def _validate_lifecycle_metadata(metadata):
    for field, default, expected_type in (
        ('keyword_hits_log', '[]', list),
        ('active_time_anchors', '{}', dict),
    ):
        raw = metadata.get(field, default)
        if field == 'active_time_anchors' and raw in (None, ''):
            raw = default
        if field == 'active_time_anchors' and not isinstance(raw, str):
            raise ValueError(f'Unsupported {field} metadata representation')
        try:
            value = json.loads(raw, object_pairs_hook=_object_without_duplicates,
                               parse_constant=_reject_constant) if isinstance(raw, str) else raw
            canonical_json(value)
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError(f'Invalid {field} metadata') from error
        if not isinstance(value, expected_type):
            raise ValueError(f'Invalid {field} metadata shape')
        entries = value.values() if isinstance(value, dict) else value
        if any(not isinstance(entry, dict) for entry in entries):
            raise ValueError(f'Invalid {field} metadata entries')


def _valid_datetime(value, field):
    if value is None:
        return
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError(f'Invalid {field}')
    try:
        from datetime import datetime
        datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f'Invalid {field}') from error


def _validate_runtime_state(name, raw):
    value = _read_json_object(raw, name)
    if name == 'session_state.json':
        required = {'session_started_at', 'session_last_tick_at', 'accumulated_session_seconds'}
        allowed = required | {'clock_id', 'baseline_session_seconds', 'last_tick_monotonic', 'monotonic_identity'}
        if not required <= value.keys() or value.keys() - allowed:
            raise ValueError('Unsupported session clock schema')
        _valid_datetime(value['session_started_at'], 'session start')
        _valid_datetime(value['session_last_tick_at'], 'session tick')
        for field in ('accumulated_session_seconds', 'baseline_session_seconds'):
            number = value.get(field, value.get('accumulated_session_seconds'))
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError(f'Invalid {field}')
        monotonic = value.get('last_tick_monotonic')
        if monotonic is not None and (type(monotonic) not in (int, float) or not math.isfinite(monotonic) or monotonic < 0):
            raise ValueError('Invalid monotonic clock value')
        for field in ('clock_id', 'monotonic_identity'):
            if field in value and (not isinstance(value[field], str) or len(value[field]) > 256):
                raise ValueError(f'Invalid {field}')
    elif name == 'nudge_state.json':
        turn = value.get('current_turn')
        if type(turn) is not int or turn < 0:
            raise ValueError('Invalid nudge turn counter')
        _valid_datetime(value.get('last_tick_at'), 'nudge tick')
        _valid_datetime(value.get('session_started_at'), 'nudge session start')
    elif name == 'dormancy_config.json':
        from .dormancy import DormancyConfig
        try:
            DormancyConfig.from_dict(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('Invalid dormancy configuration') from error
    else:
        raise ValueError('Unsupported runtime sidecar')
    return value


def _read_private_payload(path, limit):
    with private_binary_reader(path) as source:
        info = os.fstat(source.fileno())
        if info.st_size > limit:
            raise ValueError('Migration input exceeds size limit')
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Migration input exceeds size limit')
    return raw


def _safe_queue_name(name):
    return (isinstance(name, str) and name not in ('', '.', '..') and
            '/' not in name and '\\' not in name and ':' not in name and
            Path(name).name == name and name.endswith('.json'))


def _queue_classification(raw):
    try:
        from . import fallback_store
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=fallback_store._json_object,
                           parse_constant=fallback_store._reject_constant)
        record = fallback_store._validate_record(value)
    except Exception:
        return 'invalid-retained'
    if record['schema_version'] == 3:
        return 'sqlite-pending-review-required'
    return 'legacy-pending-review-required'


def _runtime_payloads(source_dir):
    root = Path(source_dir)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError('An existing absolute stopped runtime copy is required')
    ensure_private_directory(root)
    payloads = {}
    for name in sorted(RUNTIME_SIDECARS):
        path = root / name
        if path.exists() or path.is_symlink():
            raw = _read_private_payload(path, MAX_STATE_BYTES)
            _validate_runtime_state(name, raw)
            payloads['runtime/' + name] = raw
    pending = root / 'fallback_pending'
    if pending.exists() or pending.is_symlink():
        if pending.is_symlink() or not pending.is_dir():
            raise ValueError('Pending queue must be a private directory')
        ensure_private_directory(pending)
        names = []
        for entry in pending.iterdir():
            if entry.name == '.capture.lock':
                check_private_file(entry)
            elif _safe_queue_name(entry.name):
                names.append(entry.name)
            else:
                raise ValueError('Unknown pending-queue state; migration refused')
        from . import fallback_store
        if len(names) > fallback_store.MAX_PENDING_RECORDS:
            raise ValueError('Pending queue exceeds migration count limit')
        total = 0
        for name in sorted(names):
            raw = _read_private_payload(pending / name, fallback_store.MAX_PENDING_BYTES)
            total += len(raw)
            if total > fallback_store.MAX_PENDING_TOTAL_BYTES:
                raise ValueError('Pending queue exceeds migration size limit')
            payloads['runtime/fallback_pending/' + name] = raw
    return payloads


def _inventory(payloads):
    return {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            for name, raw in sorted(payloads.items())}


def _publish_directory(destination, payloads):
    destination = Path(destination)
    if not destination.is_absolute() or '..' in destination.parts or destination.exists() or destination.is_symlink():
        raise ValueError('Migration destination must be a new absolute directory')
    parent = ensure_private_directory(destination.parent)
    with tempfile.TemporaryDirectory(prefix='.sqlite-migration-', dir=parent) as temporary:
        staging = Path(temporary)
        for name, raw in payloads.items():
            target = staging.joinpath(*Path(name).parts)
            with atomic_private_writer(target, replace=False) as output:
                output.write(raw)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Migration destination was created concurrently')
        os.rename(staging, destination)
    return destination


def write_manifest(destination, identity, records):
    """Publish checksummed neutral records to a new owner-private file."""
    validated = validate_records(records, identity)
    payload = {'format': FORMAT, 'version': VERSION, 'scope': SCOPE,
               'embedding_identity': identity.to_dict(),
               'records': [record.to_dict() for record in validated]}
    envelope = {'payload': payload, 'sha256': _checksum(payload)}
    text = canonical_json(envelope) + '\n'
    if len(text.encode('ascii')) > MAX_MANIFEST_BYTES:
        raise ValueError('Manifest exceeds size limit')
    write_private_text(destination, text, replace=False)
    return {'scope': SCOPE, 'records': len(validated), 'sha256': envelope['sha256']}


def _collect_collection_records(identity, collection, batch_size, *, runtime_compatible=False):
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError('Invalid migration batch size')
    try:
        expected_count = collection.count()
    except Exception:
        raise ValueError('Legacy collection count could not be read') from None
    if type(expected_count) is not int or not 0 <= expected_count <= MAX_RECORDS:
        raise ValueError('Legacy collection count is outside migration limits')

    records = []
    for offset in range(0, expected_count, batch_size):
        limit = min(batch_size, expected_count - offset)
        try:
            batch = collection.get(
                limit=limit, offset=offset,
                include=['documents', 'metadatas', 'embeddings'],
            )
        except Exception:
            raise ValueError('Legacy collection batch could not be read') from None
        if not isinstance(batch, dict):
            raise ValueError('Legacy collection returned an invalid batch')

        values = []
        for field in ('ids', 'documents', 'metadatas', 'embeddings'):
            value = batch.get(field)
            if hasattr(value, 'tolist'):
                value = value.tolist()
            if not isinstance(value, (list, tuple)):
                raise ValueError(f'Legacy collection omitted {field}')
            values.append(value)
        if any(len(value) != limit for value in values):
            raise ValueError('Legacy collection returned an incomplete batch')

        identifiers, documents, metadatas, embeddings = values
        for record_id, content, metadata, embedding in zip(
            identifiers, documents, metadatas, embeddings, strict=True,
        ):
            if not isinstance(metadata, dict):
                raise ValueError('Legacy record metadata must be an object')
            metadata = dict(metadata)
            if 'id' in metadata and metadata['id'] != record_id:
                raise ValueError('Legacy metadata ID does not match record ID')
            if 'content' in metadata and metadata['content'] != content:
                raise ValueError('Legacy metadata content does not match document')
            metadata.setdefault('id', record_id)
            metadata.setdefault('content', content)
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            records.append({
                'id': record_id,
                'content': content,
                'metadata': metadata,
                'embedding': embedding,
            })

    try:
        final_count = collection.count()
    except Exception:
        raise ValueError('Legacy collection final count could not be read') from None
    if type(final_count) is not int or final_count != expected_count:
        raise ValueError('Legacy collection changed during export')
    validated = validate_records(records, identity)
    if not runtime_compatible:
        return validated
    from .models import ScoredSegment, SegmentType
    converted = []
    for record in validated:
        payload = record.to_dict()
        metadata = dict(payload['metadata'])
        segment_type = metadata.get('segment_type')
        if isinstance(segment_type, str) and segment_type in SegmentType.__members__:
            metadata['segment_type'] = SegmentType[segment_type].value
        metadata.setdefault('id', record.record_id)
        metadata.setdefault('content', record.content)
        try:
            _validate_lifecycle_metadata(metadata)
            segment = ScoredSegment.from_storage_dict(metadata, embedding=payload['embedding'])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError('Legacy record metadata cannot be opened by the SQLite runtime') from error
        converted.append({'id': segment.id, 'content': segment.segment.content,
                          'metadata': segment.to_storage_dict(), 'embedding': payload['embedding']})
    return validate_records(converted, identity)


def _manifest_bytes(identity, records):
    payload = {'format': FORMAT, 'version': VERSION, 'scope': SCOPE,
               'embedding_identity': identity.to_dict(),
               'records': [record.to_dict() for record in records]}
    envelope = {'payload': payload, 'sha256': _checksum(payload)}
    text = canonical_json(envelope) + '\n'
    if len(text.encode('ascii')) > MAX_MANIFEST_BYTES:
        raise ValueError('Manifest exceeds size limit')
    return text.encode('ascii')


def export_collection_manifest(destination, identity, collection, *, batch_size=256):
    """Export a stable, stopped vector collection through its ``get`` API.

    This deliberately accepts an already-open collection rather than importing
    or constructing a database client. The caller must select an isolated,
    stopped source copy and provide the reviewed embedding identity explicitly.
    """
    identity = EmbeddingIdentity.from_dict(identity.to_dict())
    records = _collect_collection_records(identity, collection, batch_size)
    write_private_text(destination, _manifest_bytes(identity, records).decode('ascii'), replace=False)
    return {'scope': SCOPE, 'records': len(records),
            'sha256': json.loads(_manifest_bytes(identity, records))['sha256']}


def export_runtime_bundle(destination, identity, collection, runtime_dir, *, batch_size=256):
    """Export records and known runtime sidecars from an isolated, stopped copy.

    Pending JSON is copied byte-for-byte and never acknowledged or replayed.
    The bundle is a migration input, not a ready-to-run service directory.
    """
    identity = EmbeddingIdentity.from_dict(identity.to_dict())
    source = Path(runtime_dir)
    destination = Path(destination)
    if (not source.is_absolute() or not destination.is_absolute() or
            source == destination or source in destination.parents or destination in source.parents):
        raise ValueError('Source and migration bundle must be separate absolute paths')
    records = _collect_collection_records(identity, collection, batch_size, runtime_compatible=True)
    payloads = {'records.json': _manifest_bytes(identity, records)}
    state_before = _runtime_payloads(source)
    payloads.update(state_before)
    manifest = {'format': BUNDLE_FORMAT, 'version': BUNDLE_VERSION,
                'embedding_identity': identity.to_dict(), 'records': len(records),
                'record_transform': 'instinct-storage-compatible-v1',
                'queue_policy': 'preserve-no-replay', 'runtime_ready': False,
                'files': _inventory(payloads)}
    payloads[BUNDLE_MANIFEST] = (canonical_json(manifest) + '\n').encode('ascii')
    if sum(map(len, payloads.values())) > MAX_MANIFEST_BYTES + 16 * 1024 * 1024:
        raise ValueError('Runtime migration bundle exceeds size limit')
    if state_before != _runtime_payloads(source):
        raise ValueError('Runtime sidecars or pending queue changed during export')
    _publish_directory(destination, payloads)
    return {'records': len(records), 'sidecars': sorted(name.removeprefix('runtime/')
            for name in state_before if name.startswith('runtime/') and '/fallback_pending/' not in name),
            'pending_files': sum('/fallback_pending/' in name for name in state_before),
            'runtime_ready': False, 'queue_policy': 'preserve-no-replay'}


def _bundle_payloads(bundle):
    root = Path(bundle)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError('An existing absolute migration bundle is required')
    ensure_private_directory(root)
    payloads = {}
    for entry in root.iterdir():
        if entry.name == 'records.json':
            payloads['records.json'] = _read_private_payload(entry, MAX_MANIFEST_BYTES)
        elif entry.name == 'runtime':
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError('Invalid runtime bundle directory')
            ensure_private_directory(entry)
            for runtime_entry in entry.iterdir():
                if runtime_entry.name in RUNTIME_SIDECARS:
                    raw = _read_private_payload(runtime_entry, MAX_STATE_BYTES)
                    _validate_runtime_state(runtime_entry.name, raw)
                    payloads['runtime/' + runtime_entry.name] = raw
                elif runtime_entry.name == 'fallback_pending':
                    if runtime_entry.is_symlink() or not runtime_entry.is_dir():
                        raise ValueError('Invalid pending queue directory')
                    ensure_private_directory(runtime_entry)
                    from . import fallback_store
                    names = []
                    for queued in runtime_entry.iterdir():
                        if not _safe_queue_name(queued.name):
                            raise ValueError('Unknown queued state in migration bundle')
                        names.append(queued.name)
                    if len(names) > fallback_store.MAX_PENDING_RECORDS:
                        raise ValueError('Pending queue exceeds migration count limit')
                    total = 0
                    for name in sorted(names):
                        raw = _read_private_payload(runtime_entry / name, fallback_store.MAX_PENDING_BYTES)
                        total += len(raw)
                        if total > fallback_store.MAX_PENDING_TOTAL_BYTES:
                            raise ValueError('Pending queue exceeds migration size limit')
                        payloads['runtime/fallback_pending/' + name] = raw
                else:
                    raise ValueError('Unknown runtime migration state')
        elif entry.name != BUNDLE_MANIFEST:
            raise ValueError('Unknown migration bundle state')
    if 'records.json' not in payloads:
        raise ValueError('Migration bundle is missing records')
    return payloads


def _validated_runtime_bundle(bundle, expected_identity):
    payloads = _bundle_payloads(bundle)
    manifest_path = Path(bundle) / BUNDLE_MANIFEST
    raw_manifest = _read_private_payload(manifest_path, MAX_STATE_BYTES)
    manifest = _read_json_object(raw_manifest, 'migration bundle manifest')
    required = {'format', 'version', 'embedding_identity', 'records', 'record_transform',
                'queue_policy', 'runtime_ready', 'files'}
    if (set(manifest) != required or manifest['format'] != BUNDLE_FORMAT or
            type(manifest['version']) is not int or manifest['version'] != BUNDLE_VERSION or
            manifest['embedding_identity'] != expected_identity.to_dict() or
            type(manifest['records']) is not int or manifest['records'] < 0 or
            manifest['record_transform'] != 'instinct-storage-compatible-v1' or
            manifest['queue_policy'] != 'preserve-no-replay' or manifest['runtime_ready'] is not False or
            manifest['files'] != _inventory(payloads)):
        raise ValueError('Unsupported or modified runtime migration bundle')
    records = read_manifest(Path(bundle) / 'records.json', expected_identity)
    if len(records) != manifest['records']:
        raise ValueError('Migration record count mismatch')
    from .models import ScoredSegment
    for record in records:
        payload = record.to_dict()
        metadata = dict(payload['metadata'])
        metadata.setdefault('id', record.record_id)
        metadata.setdefault('content', record.content)
        try:
            _validate_lifecycle_metadata(metadata)
            ScoredSegment.from_storage_dict(metadata, embedding=payload['embedding'])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError('Migration bundle contains runtime-incompatible record metadata') from error
    if payloads != _bundle_payloads(bundle):
        raise ValueError('Migration bundle changed during validation')
    return payloads, manifest, records


def import_runtime_bundle(bundle, destination, expected_identity, *, apply=False):
    """Validate a runtime bundle and optionally import to a new SQLite root.

    Application uses a sibling staging directory and atomic rename. Queues are
    retained exactly as captured; this function never replays or deletes them.
    """
    if type(apply) is not bool:
        raise ValueError('apply must be a boolean')
    expected_identity = EmbeddingIdentity.from_dict(expected_identity.to_dict())
    bundle = Path(bundle)
    destination = Path(destination)
    if (not destination.is_absolute() or '..' in destination.parts or
            destination.exists() or destination.is_symlink() or
            destination == bundle or destination in bundle.parents or bundle in destination.parents):
        raise ValueError('Runtime destination must be new and outside the bundle')
    payloads, manifest, records = _validated_runtime_bundle(bundle, expected_identity)
    pending = sorted(name.removeprefix('runtime/fallback_pending/') for name in payloads
                     if name.startswith('runtime/fallback_pending/'))
    queue_report = [{'file': name, 'classification': _queue_classification(
        payloads['runtime/fallback_pending/' + name])} for name in pending]
    report = {'records': len(records), 'sidecars': sorted(name.removeprefix('runtime/')
            for name in payloads if name.startswith('runtime/') and '/fallback_pending/' not in name),
              'pending_files': len(pending), 'pending_review': queue_report,
              'applied': False, 'verified': False, 'runtime_ready': False,
              'queue_policy': 'preserve-no-replay'}
    if not apply:
        return report

    from .sqlite_store import SQLiteRecordStore
    parent = ensure_private_directory(destination.parent)
    with tempfile.TemporaryDirectory(prefix='.sqlite-runtime-import-', dir=parent) as temporary:
        staging = Path(temporary)
        with SQLiteRecordStore.create(staging / 'sqlite', expected_identity) as store:
            store.insert_records(records)
        for name, raw in payloads.items():
            if name == 'records.json':
                continue
            relative = name.removeprefix('runtime/')
            target = staging.joinpath(*Path(relative).parts)
            with atomic_private_writer(target, replace=False) as output:
                output.write(raw)
        with SQLiteRecordStore.open_read_only(staging / 'sqlite', expected_identity) as store:
            if store.count() != len(records):
                raise ValueError('Imported record count mismatch')
            for record in records:
                if store.get_record(record.record_id) != record:
                    raise ValueError('Imported record verification failed')
        for name, raw in payloads.items():
            if name.startswith('runtime/'):
                restored = _read_private_payload(staging.joinpath(*Path(name.removeprefix('runtime/')).parts),
                                                 MAX_STATE_BYTES if name.count('/') == 1 else 1024 * 1024)
                if restored != raw:
                    raise ValueError('Imported runtime state verification failed')
        final_payloads, final_manifest, final_records = _validated_runtime_bundle(bundle, expected_identity)
        if payloads != final_payloads or manifest != final_manifest or records != final_records:
            raise ValueError('Migration bundle changed during import')
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Runtime destination was created concurrently')
        os.rename(staging, destination)
    return {**report, 'applied': True, 'verified': True}


def read_manifest(source, expected_identity):
    """Validate a bounded, trusted owner export without deserializing code."""
    with private_binary_reader(source) as stream:
        content = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(content) > MAX_MANIFEST_BYTES:
        raise ValueError('Manifest exceeds size limit')
    try:
        envelope = json.loads(content, object_pairs_hook=_object_without_duplicates,
                              parse_constant=_reject_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('Invalid manifest JSON') from error
    if not isinstance(envelope, dict) or set(envelope) != {'payload', 'sha256'}:
        raise ValueError('Invalid manifest envelope')
    payload = envelope['payload']
    if not isinstance(payload, dict) or set(payload) != {'format', 'version', 'scope', 'embedding_identity', 'records'}:
        raise ValueError('Invalid manifest fields')
    if payload['format'] != FORMAT or type(payload['version']) is not int or payload['version'] != VERSION or payload['scope'] != SCOPE:
        raise ValueError('Unsupported migration format, version or scope')
    if envelope['sha256'] != _checksum(payload):
        raise ValueError('Manifest checksum mismatch')
    identity = EmbeddingIdentity.from_dict(payload['embedding_identity'])
    if identity != expected_identity:
        raise ValueError('Embedding identity mismatch')
    if not isinstance(payload['records'], list):
        raise ValueError('Manifest records must be a list')
    return validate_records(payload['records'], identity)


def import_manifest(source, destination, expected_identity, *, apply=False):
    """Dry-run by default; apply only to a new isolated record-store directory."""
    if type(apply) is not bool:
        raise ValueError('apply must be a boolean')
    records = read_manifest(source, expected_identity)
    report = {'scope': SCOPE, 'records': len(records), 'applied': False,
              'runtime_ready': False, 'verified': False}
    if not apply:
        return report
    with SQLiteRecordStore.create(destination, expected_identity) as store:
        store.insert_records(records)
    with SQLiteRecordStore.open_read_only(destination, expected_identity) as store:
        if store.count() != len(records):
            raise ValueError('Imported record count mismatch')
        for record in records:
            restored = store.get_record(record.record_id)
            if restored != record:
                raise ValueError('Imported record verification failed')
    return {**report, 'applied': True, 'verified': True}
