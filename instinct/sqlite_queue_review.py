# SPDX-License-Identifier: MPL-2.0
"""Offline owner-reviewed queue disposition; never convert, replay or erase data."""

import hashlib
import json
import math
from pathlib import Path

from . import fallback_store
from .sqlite_backup import _inventory, _publish, _read, _validated_snapshot
from .sqlite_records import EmbeddingIdentity, canonical_json, require_text


FORMAT = 'vecretary-sqlite-queue-review-1'


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
        entries.append({'file': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'classification': _classification(raw, identity, clock), 'action': 'retain'})
    fingerprint = hashlib.sha256(canonical_json({
        'identity': manifest['identity'], 'records': manifest['records'],
        'files': _inventory(payloads)}).encode('ascii')).hexdigest()
    return {'format': FORMAT, 'snapshot_sha256': fingerprint, 'files': entries}


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
                any(selected[key] != current[key] for key in ('file', 'sha256', 'classification')) or
                selected['action'] not in ('retain', 'archive')):
            raise ValueError('Every queued file requires an exact digest-bound disposition')
        actions[current['file']] = selected['action']
    archived = sum(action == 'archive' for action in actions.values())
    blocked = sum(entry['classification'] != 'sqlite_pending' and actions[entry['file']] == 'retain'
                  for entry in expected['files'])
    report = {'format': FORMAT, 'applied': apply, 'runtime_ready': False,
              'archived_files': archived, 'retained_files': len(actions) - archived,
              'retained_blocked_files': blocked, 'review': review}
    if not apply:
        return report
    copied = {('archive/' if actions.get(name) == 'archive' else 'runtime/') + name: raw
              for name, raw in payloads.items()}
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
