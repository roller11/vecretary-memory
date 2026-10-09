# SPDX-License-Identifier: MPL-2.0
"""Synthetic runtime-root ownership and activity-clock continuity checks."""

from datetime import datetime, timedelta
import json
import os
import queue
import subprocess
import sys
import threading

import pytest

from instinct import dormancy, session_timer
from instinct.active_time import capture_anchors, elapsed_hours
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.pop import apply_pop
from instinct.private_storage import write_private_text
from instinct.sqlite_guard import SQLiteWriterUnavailable
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_session import RUNTIME_LOCK_FILENAME, SQLiteRuntimeSession
from instinct.sqlite_store import SQLiteRecordStore


@pytest.fixture
def runtime(tmp_path):
    tmp_path.chmod(0o700)
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    database = tmp_path / 'sqlite'
    with SQLiteRecordStore.create(database, identity):
        pass
    return tmp_path, database, identity


def open_session(runtime, **options):
    root, database, identity = runtime
    return SQLiteRuntimeSession.open(database, identity, persist_dir=root, **options)


def contents(root):
    return {path.name: path.read_bytes() for path in root.iterdir()
            if path.is_file() and not path.name.endswith('.lock')}


def test_observer_does_not_create_clock_turn_or_lock(runtime):
    root, _, _ = runtime
    before = contents(root)
    with open_session(runtime, read_only=True) as observer:
        result = observer.tick_activity(now=datetime(2026, 10, 7, 10))
        assert result.current_turn == 0
        assert result.session.seconds_added_to_accumulator == 0
        assert observer.read_only
        with observer.operation() as adapter:
            assert adapter.count() == 0
    assert contents(root) == before
    assert not (root / RUNTIME_LOCK_FILENAME).exists()


def test_owner_excludes_second_database_with_same_sidecars(runtime):
    root, _, identity = runtime
    other = root / 'second'
    with SQLiteRecordStore.create(other, identity):
        pass
    with open_session(runtime, read_only=False):
        with pytest.raises(SQLiteWriterUnavailable, match='sidecars'):
            SQLiteRuntimeSession.open(other, identity, persist_dir=root, read_only=False)
        with open_session(runtime, read_only=True):
            pass


def test_owner_excludes_subprocess_and_releases_without_replacing_lock(runtime):
    root, database, identity = runtime
    script = '''
import sys
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_session import SQLiteRuntimeSession
from instinct.sqlite_guard import SQLiteWriterUnavailable
try:
    SQLiteRuntimeSession.open(sys.argv[2], EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine'), persist_dir=sys.argv[1], read_only=False)
except SQLiteWriterUnavailable:
    print('excluded')
else:
    raise RuntimeError('unexpected ownership')
'''
    with open_session(runtime, read_only=False):
        lock = root / RUNTIME_LOCK_FILENAME
        before = lock.stat()
        result = subprocess.run([sys.executable, '-c', script, str(root), str(database)],
                                capture_output=True, text=True, timeout=20, check=True)
        assert result.stdout.strip() == 'excluded'
    with open_session(runtime, read_only=False):
        after = lock.stat()
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def test_activity_turns_pop_and_anchors_survive_owner_handoff(runtime):
    root, _, _ = runtime
    start = datetime(2026, 10, 7, 10)
    with open_session(runtime, read_only=False) as owner:
        assert owner.tick_activity(now=start).current_turn == 1
        with owner.operation() as adapter:
            segment = ScoredSegment(segment=Segment(content='Synthetic active memory',
                segment_type=SegmentType.CONTEXT, timestamp=start), importance_score=80,
                concept_title='Synthetic session fixture')
            segment.popped_at = start
            segment.accesses_since_pop = 1
            capture_anchors(segment, root, new=True)
            payload = segment.to_storage_dict()
            content = payload.pop('content')
            adapter.repository.insert_records([SQLiteRecord.from_dict({
                'id': segment.id, 'content': content, 'metadata': payload,
                'embedding': [1, 0]}, adapter.repository.identity)])
        second = owner.tick_activity(now=start + timedelta(minutes=1))
        assert second.current_turn == 2
        assert second.session.seconds_added_to_accumulator == 60
        with owner.operation() as adapter:
            popped = adapter.get(segment.id)
            assert apply_pop(popped, 3, dormancy.config_for(root),
                             now=start + timedelta(minutes=1)).popped
            adapter.add(popped)
            assert adapter.record_access(segment.id, start + timedelta(minutes=1))
            preserved = adapter.repository.get_record(segment.id)
        clock_id = second.session.state.clock_id
        before = contents(root)
        with open_session(runtime, read_only=True) as observer:
            observed = observer.tick_activity(now=start + timedelta(minutes=2))
            assert observed.current_turn == 2
            assert observed.session.seconds_added_to_accumulator == 0
        assert contents(root) == before
    with open_session(runtime, read_only=False) as owner:
        tick = owner.tick_activity(now=start + timedelta(minutes=2))
        assert tick.current_turn == 3
        assert tick.session.state.clock_id == clock_id
        assert tick.session.state.accumulated_session_seconds == 120
        with owner.operation() as adapter:
            restored = adapter.get(segment.id)
            assert restored.access_count == 1
            assert restored.popped_at == start
            assert restored.accesses_since_pop == 2
            assert restored.last_popped_at == start + timedelta(minutes=1)
            assert elapsed_hours(restored, 'created', root) == pytest.approx(120 / 3600)
            assert adapter.repository.get_record(segment.id) == preserved
        idle = owner.tick_activity(now=start + timedelta(days=5))
        assert idle.current_turn == 4
        assert idle.session.new_session_started
        assert idle.session.state.accumulated_session_seconds == 120


def test_failed_adapter_binding_releases_both_guards(runtime):
    with pytest.raises(ValueError, match='embedding'):
        open_session(runtime, read_only=False, query_embedder=object())
    with open_session(runtime, read_only=False):
        pass


def test_database_lock_failure_releases_runtime_guard(runtime):
    _, database, identity = runtime
    with SQLiteRecordStore.open_writable(database, identity):
        with pytest.raises(SQLiteWriterUnavailable):
            open_session(runtime, read_only=False)
    with open_session(runtime, read_only=False):
        pass


def test_replay_refuses_without_touching_legacy_queue(runtime):
    root, _, _ = runtime
    pending = root / 'fallback_pending' / 'synthetic.json'
    write_private_text(pending, 'synthetic unreviewed queue')
    with open_session(runtime, read_only=False) as owner:
        before = contents(root)
        with pytest.raises(NotImplementedError, match='no queue'):
            owner.replay_pending()
        assert contents(root) == before
        assert pending.read_text() == 'synthetic unreviewed queue'


@pytest.mark.parametrize('options', [
    {'read_only': None}, {'read_only': 1},
    {'read_only': False, 'writer_timeout': True},
    {'read_only': False, 'writer_timeout': -1},
    {'read_only': False, 'writer_timeout': float('nan')},
    {'read_only': False, 'writer_timeout': 6},
])
def test_invalid_open_options_do_not_create_locks(runtime, options):
    with pytest.raises(ValueError):
        open_session(runtime, **options)
    assert not (runtime[0] / RUNTIME_LOCK_FILENAME).exists()


def test_database_outside_runtime_root_rejected(runtime, tmp_path):
    other = tmp_path / 'other'
    other.mkdir(mode=0o700)
    with pytest.raises(ValueError, match='inside'):
        SQLiteRuntimeSession.open(runtime[1], runtime[2], persist_dir=other, read_only=False)
    assert not (other / RUNTIME_LOCK_FILENAME).exists()


def test_nested_operations_close_and_cross_thread_rejected(runtime):
    with open_session(runtime, read_only=False) as owner:
        with owner.operation():
            with pytest.raises(RuntimeError, match='Nested'):
                with owner.operation():
                    pass
            with pytest.raises(RuntimeError, match='active'):
                owner.close()
        results = queue.Queue()

        def other_thread():
            try:
                owner.tick_activity()
            except RuntimeError as error:
                results.put(str(error))

        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert 'thread-confined' in results.get_nowait()
    owner.close()
    with pytest.raises(RuntimeError, match='closed'):
        owner.tick_activity()


@pytest.mark.parametrize('idle', [True, -1, float('inf'), float('nan')])
def test_invalid_ticks_do_not_write_sidecars(runtime, idle):
    with open_session(runtime, read_only=False) as owner:
        before = contents(runtime[0])
        with pytest.raises(ValueError):
            owner.tick_activity(idle_minutes=idle)
        assert contents(runtime[0]) == before


def test_real_clock_owner_handoff_does_not_charge_stopped_interval(runtime, monkeypatch):
    monotonic = [100.0]
    monkeypatch.setattr(session_timer.time, 'monotonic', lambda: monotonic[0])
    with open_session(runtime, read_only=False) as owner:
        first = owner.tick_activity()
        monotonic[0] += 10
        second = owner.tick_activity()
        assert second.session.state.accumulated_session_seconds == 10
        with open_session(runtime, read_only=True):
            pass
        monotonic[0] += 10
        assert owner.tick_activity().session.state.accumulated_session_seconds == 20
    monotonic[0] += 200
    with open_session(runtime, read_only=False) as owner:
        restarted = owner.tick_activity()
        assert restarted.session.new_session_started
        assert restarted.session.state.clock_id == first.session.state.clock_id
        assert restarted.session.state.accumulated_session_seconds == 20


def test_fresh_process_preserves_clock_and_turn_without_charging_offline_gap(runtime):
    root, database, _ = runtime
    start = datetime(2026, 10, 7, 10)
    with open_session(runtime, read_only=False) as owner:
        owner.tick_activity(now=start)
        previous = owner.tick_activity(now=start + timedelta(minutes=2))
    script = '''
import json, sys
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_session import SQLiteRuntimeSession
with SQLiteRuntimeSession.open(sys.argv[2], EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine'), persist_dir=sys.argv[1], read_only=False) as owner:
    tick = owner.tick_activity()
    print(json.dumps({'clock': tick.session.state.clock_id, 'seconds': tick.session.state.accumulated_session_seconds, 'turn': tick.current_turn, 'new': tick.session.new_session_started}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(root), str(database)],
                            capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(result.stdout) == {
        'clock': previous.session.state.clock_id, 'seconds': 120,
        'turn': 3, 'new': True}


def test_mandatory_renewal_failure_leaves_turn_and_marker_pending(runtime, monkeypatch):
    root, _, _ = runtime
    start = datetime(2026, 10, 7, 10)
    with open_session(runtime, read_only=False) as owner:
        with owner.operation() as adapter:
            segment = ScoredSegment(segment=Segment(content='Synthetic mandatory',
                segment_type=SegmentType.STANDING_INSTRUCTION, timestamp=start),
                importance_score=79, concept_title='Synthetic renewal fixture')
            segment.importance_score = 79
            payload = segment.to_storage_dict()
            content = payload.pop('content')
            adapter.repository.insert_records([SQLiteRecord.from_dict({
                'id': segment.id, 'content': content, 'metadata': payload,
                'embedding': [1, 0]}, adapter.repository.identity)])
            original = adapter.update_metadata_only
            monkeypatch.setattr(adapter, 'update_metadata_only', lambda segment: False)
        with pytest.raises(RuntimeError, match='renewal remains pending'):
            owner.tick_activity(now=start)
        assert not (root / 'nudge_state.json').exists()
        with owner.operation() as adapter:
            assert adapter.get(segment.id).importance_score == 79
            monkeypatch.setattr(adapter, 'update_metadata_only', original)
        retried = owner.tick_activity(now=start + timedelta(minutes=1))
        assert retried.current_turn == 1
        with owner.operation() as adapter:
            assert adapter.get(segment.id).importance_score == 100


def test_operation_exception_does_not_leak_active_operation(runtime):
    with open_session(runtime, read_only=False) as owner:
        with pytest.raises(ValueError):
            with owner.operation():
                raise ValueError('synthetic dispatch failure')
        assert owner.tick_activity().current_turn == 1


def test_killed_owner_releases_runtime_guard_and_preserves_committed_turn(runtime):
    root, database, _ = runtime
    script = '''
import sys
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_session import SQLiteRuntimeSession
with SQLiteRuntimeSession.open(sys.argv[2], EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine'), persist_dir=sys.argv[1], read_only=False) as owner:
    owner.tick_activity()
    print('owned', flush=True)
    sys.stdin.read()
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(root), str(database)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    ready = queue.Queue()
    waiter = threading.Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True)
    waiter.start()
    try:
        assert ready.get(timeout=20).strip() == 'owned'
        before = (root / RUNTIME_LOCK_FILENAME).stat()
        with pytest.raises(SQLiteWriterUnavailable):
            open_session(runtime, read_only=False)
        child.kill()
        child.wait(timeout=20)
        with open_session(runtime, read_only=False, writer_timeout=2) as owner:
            tick = owner.tick_activity()
            assert tick.current_turn == 2
            assert tick.session.seconds_added_to_accumulator == 0
        after = (root / RUNTIME_LOCK_FILENAME).stat()
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=20)
        waiter.join(timeout=5)


@pytest.mark.skipif(os.name != 'posix', reason='POSIX owner permissions')
def test_runtime_lock_is_private_and_symlink_clock_directory_is_rejected(runtime, tmp_path):
    root, database, identity = runtime
    with open_session(runtime, read_only=False):
        assert (root / RUNTIME_LOCK_FILENAME).stat().st_mode & 0o777 == 0o600
    link = tmp_path / 'clock-link'
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(PermissionError, match='symlinks'):
        SQLiteRuntimeSession.open(database, identity, persist_dir=link, read_only=False)
