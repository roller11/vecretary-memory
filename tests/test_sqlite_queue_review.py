# SPDX-License-Identifier: MPL-2.0
"""Synthetic-only offline queue review, immutable evidence and staged restores."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import pytest

from instinct import fallback_store, session_timer, sqlite_backup
from instinct.active_time import clock_snapshot
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.private_storage import write_private_text
from instinct.sqlite_backup import snapshot_offline
from instinct.sqlite_queue_review import inventory_snapshot, prepare_reviewed_restore
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_service import SQLiteService
from instinct.sqlite_store import SQLiteRecordStore


IDENTITY = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')


@pytest.fixture
def source(tmp_path):
    tmp_path.chmod(0o700)
    root = tmp_path / 'source'
    root.mkdir(mode=0o700)
    session_timer.reset_session(root)
    session_timer.tick_session(root, now=datetime(2026, 10, 7, 10))
    segment = ScoredSegment(segment=Segment(content='Synthetic review fixture',
        stored_id='ordinary', segment_type=SegmentType.CONTEXT), importance_score=80,
        concept_title='Synthetic fixture')
    metadata = segment.to_storage_dict()
    metadata.pop('content', None)
    with SQLiteRecordStore.create(root / 'sqlite', IDENTITY) as repository:
        repository.insert_records([SQLiteRecord.from_dict({
            'id': 'ordinary', 'content': segment.segment.content,
            'metadata': metadata, 'embedding': [1, 0]}, IDENTITY)])
    return root


def write_legacy(root, *, version=2, operation='save_memory'):
    path = Path(fallback_store.write_pending(operation, str(root),
        title='Private synthetic title', content='Private synthetic content',
        importance_score=80, segment_id='ordinary', new_score=70, delete_id='ordinary'))
    if version == 1:
        record = json.loads(path.read_bytes())
        record['schema_version'] = 1
        record.pop('authority')
        write_private_text(path, json.dumps(record))
    return path


def write_sqlite(root, *, model='fixed', database='sqlite', clock=None):
    return Path(fallback_store.write_pending('update_score', str(root),
        segment_id='ordinary', new_score=70,
        sqlite_identity=EmbeddingIdentity('synthetic', model, 2, 'cosine').to_dict(),
        active_clock=clock or clock_snapshot(root), sqlite_database=database))


def snapshot(source):
    return snapshot_offline(source, source.parent / 'snapshot', IDENTITY)


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def archive_plan(plan):
    selected = deepcopy(plan)
    for entry in selected['files']:
        if entry['classification'] != 'sqlite_pending':
            entry['action'] = 'archive'
    return selected


@pytest.mark.parametrize('version,operation', [
    (1, 'save_memory'), (1, 'update_score'), (1, 'delete_memory'),
    (2, 'save_memory'), (2, 'update_score'), (2, 'delete_memory'),
])
def test_legacy_inventory_preserves_all_bytes_without_payload_output(source, version, operation):
    write_legacy(source, version=version, operation=operation)
    saved = snapshot(source)
    before = tree_bytes(saved)
    plan = inventory_snapshot(saved, IDENTITY)
    assert plan['files'][0]['classification'] == 'legacy_pending'
    assert plan['files'][0]['action'] == 'retain'
    encoded = json.dumps(plan)
    assert 'Private synthetic title' not in encoded
    assert 'Private synthetic content' not in encoded
    assert tree_bytes(saved) == before


def test_default_review_is_dry_run_and_retains_legacy(source):
    write_legacy(source)
    saved = snapshot(source)
    destination = source.parent / 'reviewed'
    report = prepare_reviewed_restore(saved, destination, IDENTITY, inventory_snapshot(saved, IDENTITY))
    assert report['applied'] is False and report['runtime_ready'] is False
    assert report['retained_blocked_files'] == 1
    assert not destination.exists()


def test_reviewed_restore_archives_legacy_and_preserves_runtime_state(source):
    legacy = write_legacy(source, operation='delete_memory')
    queued = write_sqlite(source)
    write_private_text(source / 'nudge_state.json', '{"synthetic":true}')
    write_private_text(source / 'dormancy_config.json', '{"custom":60}')
    saved = snapshot(source)
    before = tree_bytes(saved)
    live_before = tree_bytes(source)
    destination = source.parent / 'reviewed'
    report = prepare_reviewed_restore(saved, destination, IDENTITY,
        archive_plan(inventory_snapshot(saved, IDENTITY)), apply=True)
    assert report['applied'] is True and report['runtime_ready'] is False
    assert report['archived_files'] == 1 and report['retained_files'] == 1
    assert report['retained_blocked_files'] == 0
    assert (destination / 'archive/fallback_pending' / legacy.name).read_bytes() == legacy.read_bytes()
    assert not (destination / 'runtime/fallback_pending' / legacy.name).exists()
    assert (destination / 'runtime/fallback_pending' / queued.name).read_bytes() == queued.read_bytes()
    for name in ('session_state.json', 'nudge_state.json', 'dormancy_config.json', 'sqlite/memory.sqlite3'):
        assert (destination / 'runtime' / name).read_bytes() == before[name]
    assert tree_bytes(saved) == before and tree_bytes(source) == live_before
    assert json.loads((destination / 'QUEUE_REVIEW.json').read_bytes()) == report
    runtime = destination / 'runtime'
    with SQLiteService(runtime / 'sqlite', IDENTITY, persist_dir=runtime,
                       read_only=False, replay_pending=True) as service:
        assert service.last_replay['synced'] == 1
        assert service.inspect(lambda store: store.get('ordinary').importance_score) == 70
        assert service.inspect(lambda store: store.count()) == 1
    assert legacy.exists() and queued.exists()
    assert (destination / 'archive/fallback_pending' / legacy.name).read_bytes() == legacy.read_bytes()


@pytest.mark.parametrize('raw', [
    b'{broken', b'[]', b'{"schema_version":2,"schema_version":1}',
    b'{"value":NaN}', b'\xff',
])
def test_malformed_queue_requires_explicit_archive_and_retains_exact_bytes(source, raw):
    write_private_text(source / 'fallback_pending/malformed.json', '{}')
    (source / 'fallback_pending/malformed.json').write_bytes(raw)
    saved = snapshot(source)
    plan = inventory_snapshot(saved, IDENTITY)
    assert plan['files'][0]['classification'] == 'invalid_pending'
    destination = source.parent / 'reviewed'
    prepare_reviewed_restore(saved, destination, IDENTITY, archive_plan(plan), apply=True)
    assert (destination / 'archive/fallback_pending/malformed.json').read_bytes() == raw
    assert (saved / 'fallback_pending/malformed.json').read_bytes() == raw


@pytest.mark.parametrize('binding', ['model', 'database', 'clock', 'future', 'missing_clock'])
def test_incompatible_sqlite_binding_is_not_silently_repaired(source, binding):
    options = {}
    if binding == 'model':
        options['model'] = 'other'
    elif binding == 'database':
        options['database'] = 'other'
    elif binding == 'clock':
        options['clock'] = {'clock_id': 'other', 'seconds': 0}
    elif binding == 'future':
        options['clock'] = clock_snapshot(source) | {'seconds': 1000}
    write_sqlite(source, **options)
    if binding == 'missing_clock':
        write_private_text(source / 'session_state.json', '{}')
    plan = inventory_snapshot(snapshot(source), IDENTITY)
    assert plan['files'][0]['classification'] == 'sqlite_binding_mismatch'
    assert plan['files'][0]['action'] == 'retain'


@pytest.mark.parametrize('mutation', ['fingerprint', 'digest', 'file', 'classification',
    'action', 'missing', 'duplicate', 'extra', 'schema'])
def test_stale_incomplete_or_forged_review_refused_before_publication(source, mutation):
    write_legacy(source)
    saved = snapshot(source)
    plan = inventory_snapshot(saved, IDENTITY)
    if mutation == 'fingerprint':
        plan['snapshot_sha256'] = '0' * 64
    elif mutation in ('digest', 'file', 'classification', 'action'):
        key = 'sha256' if mutation == 'digest' else mutation
        plan['files'][0][key] = 'unapproved'
    elif mutation == 'missing':
        plan['files'] = []
    elif mutation == 'duplicate':
        plan['files'].append(deepcopy(plan['files'][0]))
    elif mutation == 'extra':
        plan['owner'] = True
    else:
        plan['format'] = 'other'
    destination = source.parent / 'reviewed'
    with pytest.raises(ValueError):
        prepare_reviewed_restore(saved, destination, IDENTITY, plan, apply=True)
    assert not destination.exists()


def test_plan_is_bound_to_entire_snapshot_not_only_pending_bytes(source):
    write_legacy(source)
    saved = snapshot(source)
    plan = inventory_snapshot(saved, IDENTITY)
    write_private_text(source / 'dormancy_config.json', '{"changed":true}')
    different = snapshot_offline(source, source.parent / 'different-snapshot', IDENTITY)
    with pytest.raises(ValueError, match='snapshot'):
        prepare_reviewed_restore(different, source.parent / 'reviewed', IDENTITY, plan, apply=True)


@pytest.mark.parametrize('policy', [1, None, 'yes'])
def test_apply_requires_exact_boolean(source, policy):
    saved = snapshot(source)
    with pytest.raises(ValueError, match='boolean'):
        prepare_reviewed_restore(saved, source.parent / 'reviewed', IDENTITY,
                                 inventory_snapshot(saved, IDENTITY), apply=policy)


def test_empty_queue_restore_is_still_not_runtime_clearance(source):
    saved = snapshot(source)
    report = prepare_reviewed_restore(saved, source.parent / 'reviewed', IDENTITY,
                                     inventory_snapshot(saved, IDENTITY), apply=True)
    assert report['archived_files'] == report['retained_files'] == 0
    assert report['runtime_ready'] is False
    assert not (source.parent / 'reviewed/archive').exists()


def test_retain_legacy_is_allowed_but_still_reports_blocked(source):
    path = write_legacy(source)
    saved = snapshot(source)
    destination = source.parent / 'reviewed'
    report = prepare_reviewed_restore(saved, destination, IDENTITY,
                                     inventory_snapshot(saved, IDENTITY), apply=True)
    assert report['retained_blocked_files'] == 1
    assert (destination / 'runtime/fallback_pending' / path.name).read_bytes() == path.read_bytes()


def test_source_tamper_refused(source):
    path = write_legacy(source)
    saved = snapshot(source)
    plan = inventory_snapshot(saved, IDENTITY)
    with (saved / 'fallback_pending' / path.name).open('ab') as output:
        output.write(b' ')
    with pytest.raises(ValueError, match='integrity'):
        prepare_reviewed_restore(saved, source.parent / 'reviewed', IDENTITY, plan, apply=True)
    assert not (source.parent / 'reviewed').exists()


def test_changing_snapshot_refused_during_validation(source, monkeypatch):
    saved = snapshot(source)
    original = sqlite_backup._payloads
    calls = []
    def changed(root):
        payloads = original(root)
        calls.append(True)
        if len(calls) > 1:
            payloads['session_state.json'] = b'{}'
        return payloads
    monkeypatch.setattr(sqlite_backup, '_payloads', changed)
    with pytest.raises(ValueError, match='changed'):
        inventory_snapshot(saved, IDENTITY)


def test_nested_existing_and_relative_destinations_refused(source):
    saved = snapshot(source)
    plan = inventory_snapshot(saved, IDENTITY)
    for destination in (saved / 'nested', source, saved.parent, Path('relative')):
        with pytest.raises(ValueError, match='destination'):
            prepare_reviewed_restore(saved, destination, IDENTITY, plan, apply=True)
    assert not (saved / 'nested').exists()


def test_failed_publication_leaves_no_final_candidate(source, monkeypatch):
    path = write_legacy(source)
    saved = snapshot(source)
    before = tree_bytes(saved)
    plan = archive_plan(inventory_snapshot(saved, IDENTITY))
    def fail(*unused):
        raise OSError('synthetic publication failure')
    monkeypatch.setattr(sqlite_backup.os, 'rename', fail)
    destination = source.parent / 'reviewed'
    with pytest.raises(OSError, match='synthetic'):
        prepare_reviewed_restore(saved, destination, IDENTITY, plan, apply=True)
    assert not destination.exists()
    assert tree_bytes(saved) == before and path.exists()


@pytest.mark.parametrize('limit', ['count', 'bytes'])
def test_queue_limits_refuse_unbounded_review(source, monkeypatch, limit):
    write_legacy(source)
    saved = snapshot(source)
    monkeypatch.setattr(fallback_store,
        'MAX_PENDING_RECORDS' if limit == 'count' else 'MAX_PENDING_TOTAL_BYTES', 0)
    with pytest.raises(ValueError, match='limits'):
        inventory_snapshot(saved, IDENTITY)


def test_actual_cli_inventory_and_apply_require_explicit_plan(source):
    write_legacy(source)
    saved = snapshot(source)
    identity_file = source.parent / 'identity.json'
    write_private_text(identity_file, json.dumps(IDENTITY.to_dict()))
    command = [sys.executable, '-m', 'instinct.sqlite_queue_review',
               '--snapshot', str(saved), '--identity', str(identity_file)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    plan = archive_plan(json.loads(result.stdout))
    review_file = source.parent / 'review.json'
    write_private_text(review_file, json.dumps(plan))
    destination = source.parent / 'reviewed'
    result = subprocess.run(command + ['--review', str(review_file), '--destination', str(destination)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and json.loads(result.stdout)['applied'] is False
    assert not destination.exists()
    result = subprocess.run(command + ['--apply'], capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    result = subprocess.run(command + ['--review', str(review_file), '--destination', str(destination), '--apply'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['archived_files'] == 1
    assert destination.is_dir()


def test_explicit_owner_archive_of_supported_operation_does_not_apply_it(source):
    queued = write_sqlite(source)
    saved = snapshot(source)
    plan = inventory_snapshot(saved, IDENTITY)
    assert plan['files'][0]['classification'] == 'sqlite_pending'
    plan['files'][0]['action'] = 'archive'
    destination = source.parent / 'reviewed'
    prepare_reviewed_restore(saved, destination, IDENTITY, plan, apply=True)
    runtime = destination / 'runtime'
    with SQLiteService(runtime / 'sqlite', IDENTITY, persist_dir=runtime,
                       read_only=False, replay_pending=True) as service:
        assert service.last_replay['synced'] == 0
        assert service.inspect(lambda store: store.get('ordinary').importance_score) == 80
    assert (destination / 'archive/fallback_pending' / queued.name).read_bytes() == queued.read_bytes()


def test_review_package_has_separate_runtime_backup_boundary(source):
    legacy = write_legacy(source)
    saved = snapshot(source)
    destination = source.parent / 'reviewed'
    prepare_reviewed_restore(saved, destination, IDENTITY,
                             archive_plan(inventory_snapshot(saved, IDENTITY)), apply=True)
    with pytest.raises(ValueError):
        snapshot_offline(destination, source.parent / 'wrong-backup', IDENTITY)
    copied = snapshot_offline(destination / 'runtime', source.parent / 'runtime-backup', IDENTITY)
    assert not (copied / 'fallback_pending' / legacy.name).exists()
    assert (destination / 'archive/fallback_pending' / legacy.name).exists()


def test_privileged_legacy_payload_is_never_promoted_to_owner_authority(source):
    legacy = write_legacy(source)
    record = json.loads(legacy.read_bytes())
    record['payload']['importance_score'] = 100
    record['payload']['segment_type'] = 'standing_instruction'
    write_private_text(legacy, json.dumps(record))
    saved = snapshot(source)
    plan = inventory_snapshot(saved, IDENTITY)
    assert plan['files'][0]['classification'] == 'invalid_pending'
    destination = source.parent / 'reviewed'
    prepare_reviewed_restore(saved, destination, IDENTITY, archive_plan(plan), apply=True)
    assert (destination / 'archive/fallback_pending' / legacy.name).read_bytes() == legacy.read_bytes()
    with SQLiteRecordStore.open_read_only(destination / 'runtime/sqlite', IDENTITY) as repository:
        assert repository.count() == 1


def test_snapshot_identity_mismatch_refuses_review(source):
    saved = snapshot(source)
    other = EmbeddingIdentity('synthetic', 'other', 2, 'cosine')
    with pytest.raises(ValueError, match='identity'):
        inventory_snapshot(saved, other)


def test_symlink_snapshot_is_refused_without_repair(source):
    saved = snapshot(source)
    alias = source.parent / 'snapshot-alias'
    try:
        alias.symlink_to(saved, target_is_directory=True)
    except OSError:
        pytest.skip('Symlink creation unavailable')
    with pytest.raises(PermissionError):
        inventory_snapshot(alias, IDENTITY)
    assert alias.is_symlink()
