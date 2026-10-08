# SPDX-License-Identifier: MPL-2.0
"""Bounded, owner-private agent outage capture and validated replay.

Pending files are plaintext and retained until a confirmed replay succeeds.
Invalid or legacy-privileged records remain untouched for explicit owner review.
A same-user file is not proof of agent authorization: every replay is validated
as an ordinary agent operation, never as an owner bootstrap.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional
import uuid

from .private_storage import ensure_private_directory, private_binary_reader, write_private_text, check_private_file, private_file_lock
from .write_policy import agent_write_error, agent_target_error

logger = logging.getLogger('instinct.fallback_store')
PENDING_DIR_NAME = 'fallback_pending'
SCHEMA_VERSION = 2
MAX_PENDING_BYTES = 1024 * 1024
MAX_PENDING_RECORDS = 1000
MAX_PENDING_TOTAL_BYTES = 8 * 1024 * 1024
PAYLOAD_FIELDS = {
    'content', 'title', 'importance_score', 'segment_type', 'reasoning',
    'segment_id', 'new_score', 'score_reason', 'delete_id',
}


def _pending_dir(persist_dir: Optional[str]) -> str:
    from .runtime import default_data_dir
    return os.path.join(persist_dir or default_data_dir(), PENDING_DIR_NAME)


def _validate_payload(operation, payload):
    if not isinstance(payload, dict) or set(payload) - PAYLOAD_FIELDS:
        raise ValueError('Invalid queued payload schema.')
    for field in ('content', 'title', 'segment_type', 'reasoning', 'segment_id', 'score_reason', 'delete_id'):
        if field in payload and not isinstance(payload[field], str):
            raise ValueError('Queued text fields must be strings.')
    if operation == 'save_memory':
        if not payload.get('content', '').strip() or not payload.get('title', '').strip():
            raise ValueError('Queued save requires title and content.')
        error = agent_write_error(payload.get('importance_score'), payload.get('segment_type'))
        if error:
            raise ValueError(error)
    elif operation == 'update_score':
        if not payload.get('segment_id', '').strip():
            raise ValueError('Queued update requires a segment ID.')
        error = agent_write_error(payload.get('new_score'))
        if error:
            raise ValueError(error)
    elif operation == 'delete_memory':
        if not payload.get('delete_id', '').strip():
            raise ValueError('Queued delete requires a segment ID.')
    else:
        raise ValueError('Unknown queued operation.')


def _validate_record(record):
    if not isinstance(record, dict):
        raise ValueError('Queued record must be an object.')
    base_fields = {'schema_version', 'operation', 'created_at', 'sync_state', 'synced_instinct_id', 'payload'}
    version = record.get('schema_version')
    if type(version) is not int or version not in (1, SCHEMA_VERSION, 3):
        raise ValueError('Unsupported queued schema version.')
    fields = base_fields | ({'authority'} if version >= 2 else set())
    if version == 3:
        fields |= {'sqlite_identity', 'active_clock', 'sqlite_database'}
    if set(record) != fields or version >= 2 and record['authority'] != 'agent':
        raise ValueError('Queued records cannot claim owner authority.')
    if record.get('sync_state') != 'pending' or record.get('synced_instinct_id') is not None:
        raise ValueError('Invalid queued replay state.')
    timestamp = record.get('created_at')
    if not isinstance(timestamp, str) or len(timestamp) > 64:
        raise ValueError('Queued record requires a timestamp.')
    parsed = datetime.fromisoformat(timestamp)
    _validate_payload(record.get('operation'), record.get('payload'))
    if version == 3:
        from .sqlite_records import EmbeddingIdentity
        EmbeddingIdentity.from_dict(record['sqlite_identity'])
        database = record['sqlite_database']
        if (not isinstance(database, str) or not 1 <= len(database) <= 1024 or
                any(character in database for character in ('\x00', '\\', ':')) or
                PurePosixPath(database).is_absolute() or '..' in PurePosixPath(database).parts or
                PurePosixPath(database).as_posix() != database):
            raise ValueError('Invalid SQLite queued database selection.')
        clock = record['active_clock']
        if (record['operation'] not in ('save_memory', 'update_score') or parsed.tzinfo is not None or
                not isinstance(clock, dict) or set(clock) != {'clock_id', 'seconds'} or
                not isinstance(clock['clock_id'], str) or not 1 <= len(clock['clock_id']) <= 256 or
                type(clock['seconds']) not in (int, float) or not math.isfinite(clock['seconds']) or clock['seconds'] < 0):
            raise ValueError('Invalid SQLite queued clock or operation.')
    return record


def write_pending(
    operation: str, persist_dir: Optional[str] = None, *, content: str = '',
    title: str = '', importance_score: int = 50, segment_type: str = 'context',
    reasoning: str = '', segment_id: str = '', new_score: Optional[int] = None,
    score_reason: str = '', delete_id: str = '',
    sqlite_identity=None, active_clock=None, sqlite_database=None,
) -> str:
    now = datetime.now()
    record = {
        'schema_version': SCHEMA_VERSION, 'authority': 'agent', 'operation': operation,
        'created_at': now.isoformat(), 'sync_state': 'pending', 'synced_instinct_id': None,
        'payload': {
            'content': content, 'title': title, 'importance_score': importance_score,
            'segment_type': segment_type, 'reasoning': reasoning, 'segment_id': segment_id,
            'new_score': new_score, 'score_reason': score_reason, 'delete_id': delete_id,
        },
    }
    if sqlite_identity is not None or active_clock is not None or sqlite_database is not None:
        record.update(schema_version=3, sqlite_identity=sqlite_identity, active_clock=active_clock,
                      sqlite_database=sqlite_database)
    _validate_record(record)
    text = json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False)
    if len(text.encode('utf-8')) > MAX_PENDING_BYTES:
        raise ValueError('Queued record exceeds size limit.')
    base = ensure_private_directory(persist_dir or Path(_pending_dir(None)).parent)
    pending = ensure_private_directory(base / PENDING_DIR_NAME)
    path = pending / f'{now.strftime("%Y%m%d_%H%M%S_%f")}_{uuid.uuid4().hex}_{operation}.json'
    with private_file_lock(pending / '.capture.lock'):
        count = 0
        total_bytes = len(text.encode('utf-8'))
        with os.scandir(pending) as entries:
            for entry in entries:
                if entry.name.endswith('.json'):
                    count += 1
                    total_bytes += entry.stat(follow_symlinks=False).st_size
                    if count >= MAX_PENDING_RECORDS or total_bytes > MAX_PENDING_TOTAL_BYTES:
                        raise ValueError('Pending JSON quota reached; nothing captured. Owner review required.')
        if total_bytes > MAX_PENDING_TOTAL_BYTES:
            raise ValueError('Pending JSON quota reached; nothing captured. Owner review required.')
        write_private_text(path, text, replace=False)
    logger.info('Agent operation captured to private pending queue')
    return str(path)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate queued JSON field.')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Invalid queued JSON constant.')


def _read_record(path):
    with private_binary_reader(path) as source:
        info = os.fstat(source.fileno())
        if info.st_size > MAX_PENDING_BYTES:
            raise ValueError('Queued record exceeds size limit.')
        raw = source.read(MAX_PENDING_BYTES + 1)
        if len(raw) > MAX_PENDING_BYTES:
            raise ValueError('Queued record exceeds size limit.')
        record = _validate_record(json.loads(raw.decode('utf-8'), object_pairs_hook=_json_object, parse_constant=_reject_constant))
    record['_path'] = str(path)
    record['_identity'] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    return record


def list_pending(persist_dir: Optional[str] = None, *, _stats=None) -> List[Dict[str, Any]]:
    pending = Path(_pending_dir(persist_dir))
    if not pending.exists() and not pending.is_symlink():
        return []
    ensure_private_directory(pending.parent)
    ensure_private_directory(pending)
    records = []
    total_bytes = 0
    with os.scandir(pending) as entries:
        names = []
        for entry in entries:
            if entry.name.endswith('.json'):
                names.append(entry.name)
                if len(names) > MAX_PENDING_RECORDS:
                    raise ValueError('Pending queue exceeds automatic replay limit; owner review required.')
    for name in sorted(names):
        try:
            record = _read_record(pending / name)
            total_bytes += record['_identity'][2]
            if total_bytes > MAX_PENDING_TOTAL_BYTES:
                raise OverflowError('Pending queue exceeds automatic byte limit; owner review required.')
            records.append(record)
        except (ValueError, OSError, RecursionError):
            logger.warning('Invalid or unsafe pending record retained for owner review')
            if _stats is not None:
                _stats['skipped'] += 1
    return records


def pending_count(persist_dir: Optional[str] = None) -> int:
    return len(list_pending(persist_dir))


def _check_identity(record):
    info = check_private_file(record['_path'])
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != record['_identity']:
        raise ValueError('Pending file changed during replay; retained for owner review.')


def sync_pending(store, persist_dir: Optional[str] = None) -> Dict[str, int]:
    stats = {'synced': 0, 'failed': 0, 'skipped': 0}
    if getattr(store, '_read_only', False):
        return stats
    if getattr(store, 'pending_replay_ready', None) is False:
        raise RuntimeError('Pending replay is unsupported by this storage adapter; no queue files were read or changed.')
    target = persist_dir or getattr(store, 'persist_dir', None)
    if not target:
        raise ValueError('Queued replay requires an explicit store directory.')
    actual = getattr(store, 'persist_dir', None)
    if not actual or os.path.abspath(os.fspath(actual)) != os.path.abspath(os.fspath(target)):
        raise ValueError('Pending queue and target store must be the same directory.')
    for record in list_pending(str(target), _stats=stats):
        try:
            _check_identity(record)
            if record['schema_version'] == 3:
                raise ValueError('SQLite pending records require the SQLite replay protocol.')
            payload = record['payload']
            operation = record['operation']
            if operation == 'save_memory':
                _sync_save(store, payload, record['created_at'])
            elif operation == 'update_score':
                _sync_update_score(store, payload)
            else:
                _sync_delete(store, payload)
            _check_identity(record)
            os.unlink(record['_path'])
            stats['synced'] += 1
        except Exception:
            logger.warning('Pending replay failed; record retained without exposing contents')
            stats['failed'] += 1
    return stats


def _check_agent_target(segment):
    error = agent_target_error(segment)
    if error:
        raise ValueError(error)


def _sync_save(store, payload: Dict[str, Any], created_at: Optional[str] = None) -> None:
    from .models import Segment, ScoredSegment, SegmentType
    _validate_payload('save_memory', payload)
    if not created_at:
        raise ValueError('Queued saves require the original timestamp.')
    segment = Segment(content=payload['content'], segment_type=SegmentType(payload['segment_type']), timestamp=datetime.fromisoformat(created_at), source='mcp-fallback')
    scored = ScoredSegment(segment=segment, importance_score=payload['importance_score'], concept_title=payload['title'], reasoning=payload.get('reasoning', 'Saved via MCP fallback'))
    existing = store.get(scored.id)
    _check_agent_target(existing)
    if existing is None:
        if not store.add(scored):
            raise RuntimeError('Queued save was not confirmed.')


def _sync_update_score(store, payload: Dict[str, Any]) -> None:
    _validate_payload('update_score', payload)
    existing = store.get(payload['segment_id'])
    _check_agent_target(existing)
    if existing is not None and not store.update_score(payload['segment_id'], payload['new_score'], payload.get('score_reason', '')):
        raise RuntimeError('Queued score update was not confirmed.')


def _sync_delete(store, payload: Dict[str, Any]) -> None:
    _validate_payload('delete_memory', payload)
    existing = store.get(payload['delete_id'])
    _check_agent_target(existing)
    if existing is not None and not store.delete(payload['delete_id']):
        raise RuntimeError('Queued deletion was not confirmed.')
