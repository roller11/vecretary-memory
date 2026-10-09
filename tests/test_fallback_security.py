# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from instinct import fallback_store as fallback
from instinct.models import Segment, ScoredSegment, SegmentType
from instinct.private_storage import ensure_private_directory, write_private_text


@pytest.fixture
def queue_store(tmp_path):
    root = ensure_private_directory(tmp_path / 'data')
    store = Mock(persist_dir=str(root), _read_only=False)
    store.get.return_value = None
    store.add.side_effect = lambda memory: memory.id
    return root, store


def queue_save(root, **overrides):
    payload = {'title': 'Synthetic queued title', 'content': 'Synthetic durable queue test.', 'importance_score': 70, 'segment_type': 'context'}
    payload.update(overrides)
    return Path(fallback.write_pending('save_memory', str(root), **payload))


def rewrite(path, mutate):
    record = json.loads(path.read_text())
    mutate(record)
    write_private_text(path, json.dumps(record))


def test_valid_queue_replays_original_timestamp_and_removes_only_success(queue_store):
    root, store = queue_store
    path = queue_save(root)
    record = json.loads(path.read_text())
    result = fallback.sync_pending(store)
    assert result == {'synced': 1, 'failed': 0, 'skipped': 0}
    saved = store.add.call_args.args[0]
    assert saved.segment.timestamp == datetime.fromisoformat(record['created_at'])
    assert saved.importance_score == 70 and saved.concept_title == 'Synthetic queued title'
    assert not path.exists()


@pytest.mark.parametrize(('score', 'segment_type'), [(100, 'context'), (93, 'context'), (70, 'standing_instruction'), (70, 'critical_constraint'), (True, 'context'), ('70', 'context'), (-1, 'context'), (70, 'unknown')])
def test_capture_cannot_queue_owner_authority(queue_store, score, segment_type):
    root, store = queue_store
    with pytest.raises(ValueError):
        queue_save(root, importance_score=score, segment_type=segment_type)
    assert not (root / fallback.PENDING_DIR_NAME).exists()
    store.add.assert_not_called()


@pytest.mark.parametrize('change', [
    lambda record: record.update(authority='owner'),
    lambda record: record.update(schema_version=True),
    lambda record: record.update(schema_version=99),
    lambda record: record.update(created_at='not-a-date'),
    lambda record: record.update(sync_state='synced'),
    lambda record: record.update(_path='outside.json'),
    lambda record: record['payload'].update(importance_score=100),
    lambda record: record['payload'].update(segment_type='standing_instruction'),
    lambda record: record['payload'].update(_created_at='2026-01-01'),
    lambda record: record['payload'].update(content=[]),
])
def test_tampered_queue_is_retained_without_store_calls(queue_store, change):
    root, store = queue_store
    path = queue_save(root)
    rewrite(path, change)
    before = path.read_bytes()
    assert fallback.sync_pending(store) == {'synced': 0, 'failed': 0, 'skipped': 1}
    assert path.read_bytes() == before
    store.get.assert_not_called()
    store.add.assert_not_called()


def test_legacy_queue_is_revalidated_as_agent_and_never_promoted(queue_store):
    root, store = queue_store
    valid = queue_save(root)
    invalid = queue_save(root, content='Different synthetic legacy content')
    def legacy(record):
        record['schema_version'] = 1
        record.pop('authority')
    rewrite(valid, legacy)
    rewrite(invalid, legacy)
    rewrite(invalid, lambda record: record['payload'].update(importance_score=100))
    assert fallback.sync_pending(store) == {'synced': 1, 'failed': 0, 'skipped': 1}
    assert not valid.exists() and invalid.exists()


@pytest.mark.parametrize('body', ['[]', '{broken', '{"schema_version":1,"schema_version":2}', '{"value":NaN}'])
def test_invalid_json_retained(queue_store, body):
    root, store = queue_store
    path = queue_save(root)
    write_private_text(path, body)
    assert fallback.sync_pending(store)['skipped'] == 1
    assert path.read_text() == body
    store.get.assert_not_called()


def test_queue_size_bound_and_no_partial_capture(queue_store, monkeypatch):
    root, store = queue_store
    monkeypatch.setattr(fallback, 'MAX_PENDING_BYTES', 30)
    with pytest.raises(ValueError, match='size limit'):
        queue_save(root)
    assert not (root / fallback.PENDING_DIR_NAME).exists()


@pytest.mark.parametrize('limit', ['MAX_PENDING_RECORDS', 'MAX_PENDING_TOTAL_BYTES'])
def test_queue_aggregate_limits_stop_replay_without_modification(queue_store, monkeypatch, limit):
    root, store = queue_store
    paths = [queue_save(root), queue_save(root, content='Second synthetic record')]
    monkeypatch.setattr(fallback, limit, 1)
    with pytest.raises((ValueError, OverflowError)):
        fallback.sync_pending(store)
    assert all(path.exists() for path in paths)
    store.add.assert_not_called()


def test_cross_store_and_read_only_replay_do_not_open_queue(queue_store, tmp_path):
    root, store = queue_store
    path = queue_save(root)
    with pytest.raises(ValueError, match='same directory'):
        fallback.sync_pending(store, str(tmp_path / 'other'))
    store._read_only = True
    assert fallback.sync_pending(store) == {'synced': 0, 'failed': 0, 'skipped': 0}
    assert path.exists()
    store.get.assert_not_called()


def test_failed_replay_keeps_note_and_successful_retry_is_idempotent(queue_store):
    root, store = queue_store
    path = queue_save(root)
    store.add.side_effect = RuntimeError('Synthetic store failure')
    assert fallback.sync_pending(store)['failed'] == 1
    assert path.exists()
    store.add.side_effect = lambda memory: memory.id
    assert fallback.sync_pending(store)['synced'] == 1
    assert not path.exists()
    assert fallback.sync_pending(store)['synced'] == 0


def test_existing_save_is_not_overwritten_on_retry(queue_store):
    root, store = queue_store
    path = queue_save(root)
    store.get.return_value = ScoredSegment(Segment('Synthetic durable queue test.', SegmentType.CONTEXT), importance_score=70, concept_title='Synthetic queued title')
    assert fallback.sync_pending(store)['synced'] == 1
    store.add.assert_not_called()
    assert not path.exists()


@pytest.mark.parametrize('operation', ['update_score', 'delete_memory'])
@pytest.mark.parametrize('owner_state', ['mandatory_type', 'high_score', 'dropped_owner'])
def test_queued_updates_and_deletes_preserve_owner_records(queue_store, operation, owner_state):
    root, store = queue_store
    memory = ScoredSegment(Segment('Synthetic owner record.', SegmentType.STANDING_INSTRUCTION if owner_state == 'mandatory_type' else SegmentType.CONTEXT), importance_score=100 if owner_state != 'mandatory_type' else 70, concept_title='Synthetic owner title')
    if owner_state == 'dropped_owner':
        memory.importance_score = 79
    store.get.return_value = memory
    kwargs = {'segment_id': memory.id, 'new_score': 71} if operation == 'update_score' else {'delete_id': memory.id}
    path = Path(fallback.write_pending(operation, str(root), **kwargs))
    assert fallback.sync_pending(store)['failed'] == 1
    assert path.exists()
    store.update_score.assert_not_called()
    store.delete.assert_not_called()


@pytest.mark.parametrize('operation', ['update_score', 'delete_memory'])
def test_ordinary_targets_replay_and_failed_mutations_remain_pending(queue_store, operation):
    root, store = queue_store
    memory = ScoredSegment(Segment('Synthetic normal note.', SegmentType.CONTEXT), importance_score=70, concept_title='Synthetic normal title')
    store.get.return_value = memory
    kwargs = {'segment_id': memory.id, 'new_score': 71} if operation == 'update_score' else {'delete_id': memory.id}
    path = Path(fallback.write_pending(operation, str(root), **kwargs))
    method = store.update_score if operation == 'update_score' else store.delete
    method.return_value = False
    assert fallback.sync_pending(store)['failed'] == 1 and path.exists()
    method.return_value = True
    assert fallback.sync_pending(store)['synced'] == 1 and not path.exists()


def test_file_replaced_after_replay_is_not_deleted(queue_store):
    root, store = queue_store
    path = queue_save(root)
    def replace_during_save(memory):
        write_private_text(path, '{"synthetic":"replacement"}')
        return memory.id
    store.add.side_effect = replace_during_save
    assert fallback.sync_pending(store)['failed'] == 1
    assert path.exists() and 'replacement' in path.read_text()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX ownership and modes')
def test_unsafe_files_and_directory_are_not_repaired_or_replayed(queue_store):
    root, store = queue_store
    path = queue_save(root)
    path.chmod(0o644)
    before = path.read_bytes()
    assert fallback.sync_pending(store)['skipped'] == 1
    assert path.read_bytes() == before and path.stat().st_mode & 0o777 == 0o644
    path.parent.chmod(0o755)
    with pytest.raises(PermissionError):
        fallback.sync_pending(store)
    assert path.parent.stat().st_mode & 0o777 == 0o755
    store.add.assert_not_called()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX links')
def test_linked_pending_records_are_retained_without_following(queue_store):
    root, store = queue_store
    path = queue_save(root)
    outside = root / 'outside.json'
    path.rename(outside)
    path.symlink_to(outside)
    before = outside.read_bytes()
    assert fallback.sync_pending(store)['skipped'] == 1
    assert path.is_symlink() and outside.read_bytes() == before
    store.get.assert_not_called()
    path.unlink()
    os.link(outside, path)
    assert fallback.sync_pending(store)['skipped'] == 1
    assert outside.read_bytes() == before


def test_oversized_existing_record_is_not_read_or_replayed(queue_store, monkeypatch):
    root, store = queue_store
    path = queue_save(root)
    monkeypatch.setattr(fallback, 'MAX_PENDING_BYTES', 10)
    assert fallback.sync_pending(store)['skipped'] == 1
    assert path.exists()
    store.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('score', [100, 79])
async def test_direct_agent_score_updates_preserve_owner_records(queue_store, score):
    from instinct import mcp_server
    _, store = queue_store
    memory = ScoredSegment(Segment('Synthetic owner memory.', SegmentType.CONTEXT), importance_score=100, concept_title='Owner-only synthetic title')
    memory.importance_score = score
    store.get.return_value = memory
    result = await mcp_server._handle_update_score(store, {'segment_id': memory.id, 'new_score': 70})
    assert 'Owner-only' in result[0].text
    store.delete.assert_not_called()
    store.update_score.assert_not_called()
