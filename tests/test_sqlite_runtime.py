# SPDX-License-Identifier: MPL-2.0
"""Isolated SQLite model persistence using the real active-time/pop engines."""

from datetime import datetime, timedelta
import json
import sqlite3
import subprocess
import sys

import pytest

from instinct import dormancy, session_timer
from instinct.active_time import capture_anchors, elapsed_hours, set_anchor
from instinct.decay import DecayModifier, PopModifier
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.modifiers import ModifierPipeline, QueryContext, ScoreModifier
from instinct.pop import apply_pop
from instinct.retriever import PriorityRetriever
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_runtime import SQLiteSegmentAdapter
from instinct.sqlite_store import SQLiteRecordStore


@pytest.fixture
def runtime(tmp_path):
    tmp_path.chmod(0o700)
    session_timer.reset_session(tmp_path)
    session_timer.tick_session(tmp_path, now=datetime(2026, 10, 7, 10))
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with SQLiteRecordStore.create(tmp_path / 'sqlite', identity) as repository:
        yield SQLiteSegmentAdapter(repository, persist_dir=tmp_path)


def insert(adapter, score=80, segment_type=SegmentType.CONTEXT):
    segment = ScoredSegment(
        segment=Segment(content=f'Synthetic SQLite runtime {score}', segment_type=segment_type,
                        timestamp=datetime(2026, 10, 7, 10),
                        metadata={'owner_extra': {'keep': [True, 2]}}),
        importance_score=score, concept_title='Synthetic SQLite timer/pop note')
    capture_anchors(segment, adapter.persist_dir, new=True)
    metadata = segment.to_storage_dict()
    content = metadata.pop('content')
    adapter.repository.insert_records([SQLiteRecord.from_dict({
        'id': segment.id, 'content': content, 'metadata': metadata, 'embedding': [1, 0],
    }, adapter.repository.identity)])
    return segment


def advance(adapter, hours):
    clock = session_timer.state(adapter.persist_dir)
    clock.accumulated_session_seconds += hours * 3600
    session_timer._store_for(adapter.persist_dir).save(clock)


def test_model_reads_have_no_clock_or_record_side_effects(runtime):
    segment = insert(runtime)
    before = runtime.repository.get_record(segment.id)
    clock = session_timer.state(runtime.persist_dir)
    restored = runtime.get(segment.id)
    assert restored.to_storage_dict() == segment.to_storage_dict()
    assert restored.embedding == [1.0, 0.0]
    assert runtime.get_many([segment.id, 'missing', segment.id])[1].id == segment.id
    assert runtime.get('missing') is None
    assert runtime.get_mandatory() == []
    assert runtime.repository.get_record(segment.id) == before
    assert session_timer.state(runtime.persist_dir) == clock
    assert runtime.runtime_ready is False


def test_access_uses_active_clock_and_preserves_vector_and_unknown_metadata(runtime):
    segment = insert(runtime)
    advance(runtime, 2)
    before = runtime.repository.get_record(segment.id)
    assert runtime.record_access(segment.id, datetime(2036, 1, 1))
    restored = runtime.get(segment.id)
    assert restored.access_count == 1
    assert elapsed_hours(restored, 'created', runtime.persist_dir) == 2
    assert elapsed_hours(restored, 'last_accessed', runtime.persist_dir) == 0
    assert restored.active_time_anchors['last_accessed']['seconds'] == 7200
    after = runtime.repository.get_record(segment.id)
    assert after.embedding == before.embedding
    assert after.content == before.content
    assert after.to_dict()['metadata']['owner_extra'] == {'keep': [True, 2]}
    assert runtime.record_access_batch([segment.id, 'missing', segment.id]) == 2
    assert runtime.get(segment.id).access_count == 3


def test_idle_gap_and_reopen_do_not_age_memory(runtime):
    segment = insert(runtime)
    start = datetime(2026, 10, 7, 10)
    session_timer.tick_session(runtime.persist_dir, now=start + timedelta(minutes=5))
    session_timer.tick_session(runtime.persist_dir, now=start + timedelta(days=90))
    session_timer._store_for(runtime.persist_dir).invalidate()
    with SQLiteRecordStore.open_read_only(runtime.persist_dir / 'sqlite', runtime.repository.identity) as reader:
        adapter = SQLiteSegmentAdapter(reader, persist_dir=runtime.persist_dir)
        restored = adapter.get(segment.id)
        assert elapsed_hours(restored, 'created', adapter.persist_dir) == pytest.approx(5 / 60)
        assert not PopModifier(persist_dir=adapter.persist_dir).should_pop(restored, start + timedelta(days=90))


@pytest.mark.parametrize('score', [49, 50, 79, 80, 89, 90, 95, 100])
def test_keyword_pop_uses_existing_rules_and_survives_reopen(runtime, score):
    original = insert(runtime, score)
    cfg = dormancy.config_for(runtime.persist_dir)
    if 80 <= score <= 89:
        cfg.plateau.initial_drop_pts = 17
    expected = ScoredSegment.from_storage_dict(original.to_storage_dict())
    stored = runtime.get(original.id)
    now = datetime(2026, 10, 7, 10)
    expected_pop = apply_pop(expected, 3, cfg, now=now)
    actual_pop = apply_pop(stored, 3, cfg, now=now)
    assert actual_pop == expected_pop
    assert runtime.add(stored) == stored.id
    session_timer._store_for(runtime.persist_dir).invalidate()
    with SQLiteRecordStore.open_read_only(runtime.persist_dir / 'sqlite', runtime.repository.identity) as reader:
        restored = SQLiteSegmentAdapter(reader, persist_dir=runtime.persist_dir).get(stored.id)
        assert restored.to_storage_dict() == stored.to_storage_dict()
        assert dormancy.current_effective_score(restored, now, cfg) == actual_pop.effective_score_after
        assert restored.initial_score == original.initial_score


def test_keyword_frequency_window_counts_active_not_idle_time(runtime):
    segment = insert(runtime)
    cfg = dormancy.config_for(runtime.persist_dir)
    apply_pop(segment, 3, cfg, now=datetime(2026, 10, 7, 10))
    runtime.add(segment)
    session_timer.tick_session(runtime.persist_dir, now=datetime(2026, 12, 7, 10))
    restored = runtime.get(segment.id)
    result = apply_pop(restored, 2, cfg, now=datetime(2026, 12, 7, 10))
    assert result.rule_applied == 'frequency_reset'
    assert result.effective_score_after == 80
    runtime.add(restored)
    assert runtime.get(restored.id).pop_recent_8h_active == 2


def test_access_pop_expiration_uses_active_time(runtime):
    segment = insert(runtime)
    segment.last_accessed = segment.segment.timestamp
    set_anchor(segment, 'last_accessed', runtime.persist_dir)
    runtime.add(segment)
    advance(runtime, 90 * 24)
    assert runtime.record_access(segment.id, datetime(2026, 10, 7, 11))
    restored = runtime.get(segment.id)
    assert restored.popped_at is not None
    assert restored.accesses_since_pop == 1
    assert runtime.clear_expired_pops(datetime(2036, 1, 1)) == 0
    advance(runtime, 8 * 24)
    assert runtime.clear_expired_pops(datetime(2026, 10, 7, 12)) == 1
    restored = runtime.get(segment.id)
    assert restored.popped_at is None
    assert restored.accesses_since_pop == 0
    assert 'popped_at' not in restored.active_time_anchors


def test_stuck_access_pop_is_not_cleared(runtime):
    segment = insert(runtime)
    segment.popped_at = segment.segment.timestamp
    segment.accesses_since_pop = 2
    set_anchor(segment, 'popped_at', runtime.persist_dir)
    runtime.add(segment)
    advance(runtime, 8 * 24)
    assert runtime.clear_expired_pops() == 0
    assert runtime.get(segment.id).popped_at is not None


def test_mandatory_access_uses_existing_nudge_drop_and_clock_anchor(runtime):
    segment = insert(runtime, 100, SegmentType.STANDING_INSTRUCTION)
    assert runtime.get_mandatory()[0].id == segment.id
    advance(runtime, 1)
    assert runtime.record_access(segment.id)
    restored = runtime.get(segment.id)
    assert restored.importance_score == 79
    assert restored.initial_score == 100
    assert restored.active_time_anchors['pop_anchor_at']['seconds'] == 3600
    assert runtime.get_mandatory() == []


def test_score_99_remains_mandatory_until_access_drops_it_to_79(runtime):
    segment = insert(runtime, 100, SegmentType.STANDING_INSTRUCTION)
    assert runtime.update_score(segment.id, 99, 'synthetic nudge fixture')
    assert runtime.get_mandatory()[0].id == segment.id
    assert runtime.stats()['mandatory_count'] == 1
    assert runtime.record_access(segment.id)
    assert runtime.get(segment.id).importance_score == 79
    assert runtime.get_mandatory() == []


def test_read_only_adapter_does_not_mutate_record_or_clock(runtime):
    segment = insert(runtime)
    before = runtime.repository.get_record(segment.id)
    clock = session_timer.state(runtime.persist_dir)
    with SQLiteRecordStore.open_read_only(runtime.persist_dir / 'sqlite', runtime.repository.identity) as reader:
        adapter = SQLiteSegmentAdapter(reader, persist_dir=runtime.persist_dir)
        assert not adapter.record_access(segment.id)
        assert not adapter.update_metadata_only(segment.id)
        assert adapter.clear_expired_pops() == 0
        with pytest.raises(RuntimeError, match='read-only'):
            adapter.add(segment)
    assert runtime.repository.get_record(segment.id) == before
    assert session_timer.state(runtime.persist_dir) == clock


def test_metadata_writes_reject_new_records_and_content_changes(runtime):
    segment = insert(runtime)
    segment = runtime.get(segment.id)
    before = runtime.repository.get_record(segment.id)
    segment.segment.content = 'changed'
    with pytest.raises(ValueError, match='content'):
        runtime.update_metadata_only(segment)
    assert runtime.repository.get_record(segment.id) == before
    segment.segment.stored_id = 'new'
    with pytest.raises(RuntimeError, match='existing-record'):
        runtime.add(segment)
    assert runtime.repository.count() == 1


def test_access_sql_failure_rolls_back(runtime):
    segment = insert(runtime)
    before = runtime.repository.get_record(segment.id)
    runtime.repository._connection.execute(
        "CREATE TRIGGER reject_access BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'fixture'); END")
    with pytest.raises(sqlite3.IntegrityError):
        runtime.record_access(segment.id)
    assert runtime.repository.get_record(segment.id) == before
    assert not runtime.repository._connection.in_transaction


class SyntheticHitsModifier(ScoreModifier):
    def __init__(self):
        super().__init__('synthetic_hits', priority=30)

    def modify(self, segment, context, result):
        result.identifier_hits = 3
        return result


def test_real_retriever_pop_stage_persists_through_sqlite_adapter(runtime):
    segment = insert(runtime, 88)
    dormancy._ConfigStore(runtime.persist_dir).update_band('plateau', initial_drop_pts=17)
    cfg = dormancy.config_for(runtime.persist_dir)
    pipeline = ModifierPipeline()
    pipeline.register(DecayModifier())
    pipeline.register(SyntheticHitsModifier())
    retriever = PriorityRetriever(runtime, pipeline=pipeline)
    context = QueryContext.from_query('Synthetic SQLite timer pop')
    context.persist_dir = runtime.persist_dir
    records = retriever._apply_pops_to_results([runtime.get(segment.id)], context, cfg)
    assert records[0].effective_score_after == 71
    assert runtime.get(segment.id).importance_score == 71
    metadata = runtime.repository.get_record(segment.id).to_dict()['metadata']
    assert json.loads(metadata['active_time_anchors'])['pop_anchor_at']['post_drop'] is True


def test_clock_directory_is_explicit(runtime):
    with pytest.raises(ValueError, match='explicit'):
        SQLiteSegmentAdapter(runtime.repository, persist_dir=None)


@pytest.mark.parametrize('path', ['', '.', 'missing-directory', False])
def test_clock_directory_cannot_fall_back_to_default(runtime, path):
    with pytest.raises(ValueError, match='explicit'):
        SQLiteSegmentAdapter(runtime.repository, persist_dir=path)


def test_fresh_process_preserves_clock_pop_and_vector(runtime):
    segment = insert(runtime)
    cfg = dormancy.config_for(runtime.persist_dir)
    result = apply_pop(segment, 3, cfg, now=datetime(2026, 10, 7, 10))
    runtime.add(segment)
    advance(runtime, 2)
    expected = dormancy.current_effective_score(runtime.get(segment.id), datetime(2026, 10, 7, 12), cfg)
    code = '''
import json
from datetime import datetime
from pathlib import Path
import sys
from instinct import dormancy, session_timer
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_runtime import SQLiteSegmentAdapter
from instinct.sqlite_store import SQLiteRecordStore
root = Path(sys.argv[1])
identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
with SQLiteRecordStore.open_read_only(root / 'sqlite', identity) as repository:
    adapter = SQLiteSegmentAdapter(repository, persist_dir=root)
    segment = adapter.get(sys.argv[2])
    print(json.dumps({'clock_id': session_timer.state(root).clock_id,
                      'score': dormancy.current_effective_score(segment, datetime(2026, 10, 7, 12), dormancy.config_for(root)),
                      'anchors': segment.active_time_anchors,
                      'embedding': segment.embedding,
                      'initial_score': segment.initial_score}))
'''
    process = subprocess.run([sys.executable, '-c', code, str(runtime.persist_dir), segment.id],
                             capture_output=True, text=True, check=True, timeout=30)
    restored = json.loads(process.stdout)
    assert restored['clock_id'] == session_timer.state(runtime.persist_dir).clock_id
    assert restored['score'] == pytest.approx(expected)
    assert restored['anchors'] == runtime.get(segment.id).active_time_anchors
    assert restored['embedding'] == [1.0, 0.0]
    assert restored['initial_score'] == result.initial_score_after
