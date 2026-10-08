# SPDX-License-Identifier: MPL-2.0
"""Explicit SQLite agent fallback with transactional per-record replay receipts."""

from datetime import datetime
import hashlib
import os
from pathlib import Path

from . import fallback_store
from .active_time import capture_anchors, clock_snapshot
from .models import ScoredSegment, Segment, SegmentType
from .private_storage import ensure_private_directory, private_file_lock
from .sqlite_records import MAX_RECORDS, canonical_json, require_text
from .sqlite_writes import MAX_DOCUMENT_BYTES
from .write_policy import agent_target_error, agent_write_error


RECEIPTS_KEY = 'sqlite_pending_receipts'
MAX_RECEIPTS = 1024
MAX_REPLAY_BATCH = 32


def _duplicate_title(store, title):
    return any(segment.concept_title.lower().strip() == title.lower().strip()
               for segment in store.get_by_score(limit=MAX_RECORDS))


def _receipt(record):
    payload = {key: value for key, value in record.items() if not key.startswith('_')}
    return hashlib.sha256(canonical_json(payload).encode('ascii')).hexdigest()


def _receipts(metadata):
    receipts = metadata.get(RECEIPTS_KEY, [])
    if (not isinstance(receipts, list) or len(receipts) > MAX_RECEIPTS or
            any(not isinstance(value, str) or len(value) != 64 or
                any(character not in '0123456789abcdef' for character in value) for value in receipts)):
        raise ValueError('Invalid SQLite queue receipts; owner review required')
    return receipts


def _append_receipt(metadata, receipt):
    receipts = _receipts(metadata)
    if len(receipts) >= MAX_RECEIPTS:
        raise ValueError('SQLite queue receipt capacity reached; owner review required')
    return metadata | {RECEIPTS_KEY: receipts + [receipt]}


def capture_call(store, name, arguments):
    if not store._read_only or not getattr(store, 'queue_writes', False):
        raise RuntimeError('SQLite queue capture requires an explicitly enabled observer')
    from .mcp_server import _check_write_guards
    if name == 'instinct_save_memory':
        payload = {'content': arguments.get('content', ''), 'title': arguments.get('title', 'Untitled'),
                   'importance_score': arguments.get('importance', 70),
                   'segment_type': arguments.get('segment_type', 'context'), 'reasoning': 'Saved via MCP tool'}
        error = agent_write_error(payload['importance_score'], payload['segment_type'])
        if not error:
            try:
                require_text(payload['content'], 'embedding document', MAX_DOCUMENT_BYTES)
            except ValueError as failure:
                error = str(failure)
        error = error or _check_write_guards(payload['content'], payload['title'], payload['importance_score'])
        if not error and _duplicate_title(store, payload['title']):
            error = 'Duplicate queued memory title already exists in the database'
        operation = 'save_memory'
    elif name == 'instinct_update_score':
        payload = {'segment_id': arguments.get('segment_id', ''), 'new_score': arguments.get('new_score', -1),
                   'score_reason': arguments.get('reason', 'Score updated via MCP tool')}
        error = agent_write_error(payload['new_score'])
        segment = store.get(payload['segment_id'])
        error = error or ('Queued score target is missing' if segment is None else agent_target_error(segment))
        reason = payload['score_reason']
        if not isinstance(reason, str) or '\x00' in reason or len(reason.encode('utf-8')) > 4096:
            error = 'Invalid queued score reasoning'
        operation = 'update_score'
    else:
        raise ValueError('Only SQLite agent saves and score updates may be queued')
    if error:
        return 'BLOCKED: ' + error
    path = fallback_store.write_pending(operation, persist_dir=str(store.persist_dir),
        sqlite_identity=store.repository.identity.to_dict(), active_clock=clock_snapshot(store.persist_dir),
        sqlite_database=store.database_dir.relative_to(store.persist_dir).as_posix(), **payload)
    return ('Queued to SQLite FALLBACK; not yet committed to the memory database.\n'
            'Pending file: ' + Path(path).name + '\n'
            'An owner started with replay enabled will apply it. Do not retry an unconfirmed capture without checking this file.')


def _apply(store, record):
    fallback_store._validate_record({key: value for key, value in record.items() if not key.startswith('_')})
    if (record['schema_version'] != 3 or record['sqlite_identity'] != store.repository.identity.to_dict() or
            record['sqlite_database'] != store.database_dir.relative_to(store.persist_dir).as_posix()):
        raise ValueError('SQLite queue identity mismatch')
    clock = clock_snapshot(store.persist_dir)
    captured = record['active_clock']
    if clock['clock_id'] != captured['clock_id'] or captured['seconds'] > clock['seconds']:
        raise ValueError('SQLite queued clock does not match the current runtime')
    receipt = _receipt(record)
    payload = record['payload']
    timestamp = datetime.fromisoformat(record['created_at'])
    if record['operation'] == 'save_memory':
        segment_id = 'queued_' + receipt
        existing = store.repository.get_record(segment_id)
        if existing is not None:
            if receipt in _receipts(existing.to_dict()['metadata']):
                return
            raise ValueError('Queued save conflicts with an existing record')
        from .mcp_server import _check_write_guards
        error = _check_write_guards(payload['content'], payload['title'], payload['importance_score'], track_rate=False)
        if not error and _duplicate_title(store, payload['title']):
            error = 'Queued memory title conflicts with a committed record'
        if error:
            raise ValueError(error)
        segment = ScoredSegment(segment=Segment(content=payload['content'],
            segment_type=SegmentType(payload['segment_type']), timestamp=timestamp,
            source='mcp-sqlite-fallback', stored_id=segment_id,
            metadata={RECEIPTS_KEY: [receipt]}), importance_score=payload['importance_score'],
            concept_title=payload['title'], reasoning=payload.get('reasoning', 'Saved via MCP fallback'))
        segment.active_time_anchors['created'] = captured | {
            'timestamp': timestamp.isoformat(), 'score': segment.importance_score}
        store.add(segment)
        return
    if record['operation'] != 'update_score':
        raise ValueError('SQLite queued deletion is disabled')
    reason = payload.get('score_reason', '')
    if not isinstance(reason, str) or '\x00' in reason or len(reason.encode('utf-8')) > 4096:
        raise ValueError('Invalid queued score reasoning')

    def transform(metadata):
        if receipt in _receipts(metadata):
            return metadata
        segment = store.get(payload['segment_id'])
        error = agent_target_error(segment)
        if error:
            raise ValueError(error)
        segment.importance_score = payload['new_score']
        segment.pop_anchor_at = timestamp
        if reason:
            segment.reasoning = reason
        segment.active_time_anchors['pop_anchor_at'] = captured | {
            'timestamp': timestamp.isoformat(), 'score': segment.importance_score}
        capture_anchors(segment, store.persist_dir)
        return _append_receipt(metadata | store._changes(segment), receipt)

    if store.repository._mutate_metadata(payload['segment_id'], transform) is None:
        raise ValueError('Queued score target is missing')


def replay(store):
    if store._read_only:
        raise RuntimeError('SQLite observer cannot replay pending writes')
    stats = {'synced': 0, 'failed': 0, 'skipped': 0, 'pending': False}
    pending = Path(store.persist_dir) / fallback_store.PENDING_DIR_NAME
    if not pending.exists() and not pending.is_symlink():
        return stats
    ensure_private_directory(pending)
    with private_file_lock(pending / '.capture.lock', timeout=0):
        records = fallback_store.list_pending(str(store.persist_dir), _stats=stats)
        selected = [record for record in records if record['schema_version'] == 3]
        stats['skipped'] += len(records) - len(selected)
        for record in selected[:MAX_REPLAY_BATCH]:
            try:
                fallback_store._check_identity(record)
                _apply(store, record)
                fallback_store._check_identity(record)
                os.unlink(record['_path'])
                stats['synced'] += 1
            except Exception:
                stats['failed'] += 1
                break
        stats['pending'] = stats['synced'] < len(selected)
    return stats
