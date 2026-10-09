# SPDX-License-Identifier: MPL-2.0
"""Offline owner-reviewed queue disposition; preserve originals and never auto-replay."""

import hashlib
import json
import math
from pathlib import Path

from . import fallback_store
from .sqlite_backup import _inventory, _publish, _read, _validated_snapshot
from .sqlite_records import EmbeddingIdentity, canonical_json, require_text


FORMAT = 'vecretary-sqlite-queue-review-2'


def _decode(raw):
    return json.loads(raw.decode('utf-8'), object_pairs_hook=fallback_store._json_object,
                      parse_constant=fallback_store._reject_constant)


def _classification(raw, identity, clock):
    try:
        record = fallback_store._validate_record(_decode(raw))
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        return 'invalid_pending'
    if record['schema_version'] != 3:
        return 'legacy_pending'
    captured = record['active_clock']
    seconds = clock.get('accumulated_session_seconds')
    if (record['sqlite_identity'] != identity.to_dict() or record['sqlite_database'] != 'sqlite' or
            clock.get('clock_id') != captured['clock_id'] or type(seconds) not in (int, float)):
        return 'sqlite_binding_mismatch'
    try:
        if not math.isfinite(seconds) or seconds < captured['seconds']:
            return 'sqlite_binding_mismatch'
    except OverflowError:
        return 'sqlite_binding_mismatch'
    return 'sqlite_pending'


def _review(payloads, manifest, identity):
    queued = {name: raw for name, raw in payloads.items() if name.startswith('fallback_pending/')}
    if (len(queued) > fallback_store.MAX_PENDING_RECORDS or
            sum(map(len, queued.values())) > fallback_store.MAX_PENDING_TOTAL_BYTES):
        raise ValueError('Pending queue exceeds review limits')
    clock = {}
    if 'session_state.json' in payloads:
        clock = _decode(payloads['session_state.json'])
        if not isinstance(clock, dict):
            raise ValueError('Invalid snapshot clock')
    entries = []
    for name, raw in sorted(queued.items()):
        filename = name.removeprefix('fallback_pending/')
        require_text(filename, 'pending filename', 256)
        if any(character in filename for character in ('/', '\\', ':')):
            raise ValueError('Unsafe pending filename')
        if len(raw) > fallback_store.MAX_PENDING_BYTES:
            raise ValueError('Pending record exceeds review limit')
        classification = _classification(raw, identity, clock)
        legacy_operation = None
        if classification == 'legacy_pending':
            legacy_operation = fallback_store._validate_record(_decode(raw))['operation']
        entries.append({'file': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'classification': classification, 'legacy_operation': legacy_operation,
                        'action': 'retain'})
    fingerprint = hashlib.sha256(canonical_json({
        'identity': manifest['identity'], 'records': manifest['records'],
        'files': _inventory(payloads)}).encode('ascii')).hexdigest()
    return {'format': FORMAT, 'snapshot_sha256': fingerprint, 'files': entries}


def _convert_legacy_operation(raw, source_name, action, snapshot, payloads, identity):
    record = fallback_store._validate_record(_decode(raw))
    action_operations = {
        'convert_save': 'save_memory',
        'convert_score_update': 'update_score',
        'convert_soft_recycle': 'delete_memory',
    }
    expected_operation = action_operations.get(action)
    if (record['schema_version'] not in (1, 2) or expected_operation is None or
            record['operation'] != expected_operation):
        raise ValueError('Reviewed conversion action does not match the legacy operation')
    from datetime import datetime
    timestamp = datetime.fromisoformat(record['created_at'])
    if timestamp.tzinfo is not None:
        raise ValueError('Legacy timestamp must be timezone-naive for SQLite replay')
    clock_raw = payloads.get('session_state.json')
    if clock_raw is None:
        raise ValueError('A validated SQLite session clock is required for conversion')
    clock = _decode(clock_raw)
    clock_id = clock.get('clock_id')
    seconds = clock.get('accumulated_session_seconds')
    try:
        valid_seconds = type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0
    except OverflowError:
        valid_seconds = False
    if (not isinstance(clock_id, str) or not 1 <= len(clock_id) <= 256 or not valid_seconds):
        raise ValueError('Snapshot session clock cannot bind the legacy operation')

    payload = record['payload']
    if record['operation'] == 'save_memory':
        from .mcp_server import _check_write_guards
        from .write_policy import agent_write_error
        error = agent_write_error(payload.get('importance_score'), payload.get('segment_type'))
        error = error or _check_write_guards(
            payload['content'], payload['title'], payload['importance_score'], track_rate=False)
        if error:
            raise ValueError(error)
        operation_payload = {key: payload[key] for key in (
            'content', 'title', 'importance_score', 'segment_type', 'reasoning') if key in payload}
    else:
        from .models import ScoredSegment
        from .sqlite_store import SQLiteRecordStore
        from .write_policy import agent_target_error, agent_write_error
        if record['operation'] == 'update_score':
            target_id = require_text(payload.get('segment_id', ''), 'record ID', 256)
            error = agent_write_error(payload.get('new_score'))
            operation_payload = {key: payload[key] for key in (
                'segment_id', 'new_score', 'score_reason') if key in payload}
        else:
            target_id = require_text(payload.get('delete_id', ''), 'record ID', 256)
            error = None
            operation_payload = {'delete_id': target_id}
        if error:
            raise ValueError(error)
        with SQLiteRecordStore.open_read_only(Path(snapshot) / 'sqlite', identity) as repository:
            target = repository.get_record(target_id)
        if target is None:
            raise ValueError('Legacy operation target is absent from the SQLite snapshot')
        target_payload = target.to_dict()
        target_metadata = target_payload['metadata'] | {'id': target.record_id, 'content': target.content}
        try:
            segment = ScoredSegment.from_storage_dict(target_metadata, embedding=target_payload['embedding'])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError('Legacy operation target metadata requires owner review') from error
        authority_error = agent_target_error(segment)
        if authority_error:
            raise ValueError(authority_error)
        if record['operation'] == 'delete_memory' and target_metadata.get('recycled_at'):
            raise ValueError('Legacy recycle target is already in the recycle bin')

    converted = {
        'schema_version': 3,
        'authority': 'agent',
        'operation': record['operation'],
        'created_at': record['created_at'],
        'sync_state': 'pending',
        'synced_instinct_id': None,
        'payload': operation_payload,
        'sqlite_identity': identity.to_dict(),
        'active_clock': {'clock_id': clock_id, 'seconds': seconds},
        'sqlite_database': 'sqlite',
    }
    fallback_store._validate_record(converted)
    operation_digest = hashlib.sha256(source_name.encode('utf-8') + b'\x00' + raw).hexdigest()
    filename = 'converted-' + record['operation'] + '-' + operation_digest + '.json'
    return 'fallback_pending/' + filename, (canonical_json(converted) + '\n').encode('ascii')


def inventory_snapshot(snapshot, identity):
    """Return a private, payload-free plan; default actions retain every file."""
    payloads, manifest = _validated_snapshot(snapshot, identity)
    return _review(payloads, manifest, identity)


def prepare_reviewed_restore(snapshot, destination, identity, review, *, apply=False):
    if type(apply) is not bool:
        raise ValueError('Apply must be an explicit boolean')
    snapshot = Path(snapshot)
    destination = Path(destination)
    if (not snapshot.is_absolute() or not destination.is_absolute() or '..' in destination.parts or
            destination.exists() or destination.is_symlink() or
            destination.is_relative_to(snapshot) or snapshot.is_relative_to(destination)):
        raise ValueError('Review destination must be new, absolute and outside the snapshot')
    payloads, manifest = _validated_snapshot(snapshot, identity)
    expected = _review(payloads, manifest, identity)
    if (not isinstance(review, dict) or set(review) != set(expected) or
            review['format'] != FORMAT or review['snapshot_sha256'] != expected['snapshot_sha256'] or
            not isinstance(review['files'], list) or len(review['files']) != len(expected['files'])):
        raise ValueError('Review does not match the validated snapshot')
    actions = {}
    for selected, current in zip(review['files'], expected['files']):
        if (not isinstance(selected, dict) or set(selected) != set(current) or
                any(selected[key] != current[key] for key in
                    ('file', 'sha256', 'classification', 'legacy_operation')) or
                selected['action'] not in ('retain', 'archive', 'convert_save',
                                           'convert_score_update', 'convert_soft_recycle') or
                selected['action'].startswith('convert_') and
                current['classification'] != 'legacy_pending'):
            raise ValueError('Every queued file requires an exact digest-bound disposition')
        actions[current['file']] = selected['action']
    converted = {}
    for name, action in actions.items():
        if action.startswith('convert_'):
            converted_name, converted_raw = _convert_legacy_operation(
                payloads[name], name, action, snapshot, payloads, identity)
            if converted_name in payloads or converted_name in converted:
                raise ValueError('Converted queue filename conflicts with snapshot state')
            converted[converted_name] = converted_raw
    retained_queue = [name for name, action in actions.items() if action == 'retain']
    retained_bytes = sum(len(payloads[name]) for name in retained_queue)
    if (len(retained_queue) + len(converted) > fallback_store.MAX_PENDING_RECORDS or
            retained_bytes + sum(map(len, converted.values())) > fallback_store.MAX_PENDING_TOTAL_BYTES):
        raise ValueError('Converted pending queue exceeds runtime queue limits')
    archived = sum(action == 'archive' or action.startswith('convert_') for action in actions.values())
    converted_count = sum(action.startswith('convert_') for action in actions.values())
    retained = sum(action == 'retain' for action in actions.values())
    blocked = sum(entry['classification'] != 'sqlite_pending' and actions[entry['file']] == 'retain'
                  for entry in expected['files'])
    report = {'format': FORMAT, 'applied': apply, 'runtime_ready': False,
              'archived_files': archived, 'converted_files': converted_count,
              'retained_files': retained,
              'retained_blocked_files': blocked, 'review': review}
    if not apply:
        return report
    copied = {}
    for name, raw in payloads.items():
        action = actions.get(name, 'retain') if name.startswith('fallback_pending/') else 'retain'
        target_root = 'archive/' if action == 'archive' or action.startswith('convert_') else 'runtime/'
        copied[target_root + name] = raw
    copied.update({'runtime/' + name: raw for name, raw in converted.items()})
    copied['QUEUE_REVIEW.json'] = canonical_json(report).encode('ascii')
    _publish(destination, copied)
    return report


def cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--apply', action='store_true')
    arguments = parser.parse_args()
    if not arguments.snapshot.is_absolute():
        parser.error('Snapshot must be explicit and absolute')
    if arguments.apply and (arguments.review is None or arguments.destination is None):
        parser.error('Apply requires an owner-reviewed plan and a new destination')
    if (arguments.review is None) != (arguments.destination is None):
        parser.error('Review and destination must be supplied together')
    identity = EmbeddingIdentity.from_dict(_decode(_read(arguments.identity, 4096)))
    if arguments.review is None:
        result = inventory_snapshot(arguments.snapshot, identity)
    else:
        review = _decode(_read(arguments.review, 1024 * 1024))
        result = prepare_reviewed_restore(arguments.snapshot, arguments.destination, identity,
                                         review, apply=arguments.apply)
    print(canonical_json(result))


if __name__ == '__main__':
    cli()
