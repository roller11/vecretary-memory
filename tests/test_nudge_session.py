# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from instinct.models import (
    NUDGED_AT_TURN_KEY,
    NUDGE_ACTIVE_KEY,
    NUDGE_INTERVAL_TURNS_KEY,
    RECYCLED_AT_KEY,
    TURNS_SINCE_ACCESS_KEY,
    ScoredSegment,
    Segment,
    SegmentType,
)
from instinct.nudge import (
    apply_access_drop,
    apply_nudge_check,
    get_current_turn,
    get_last_nudged_turn,
    get_turns_since_access,
    tick_nudge_session,
    tick_turn,
)
from instinct.session_timer import tick_session


def make_memory(
    segment_type=SegmentType.STANDING_INSTRUCTION,
    score=79,
    content="Remember this rule",
    **metadata,
):
    memory = ScoredSegment(
        segment=Segment(
            content=content, segment_type=segment_type, metadata=metadata,
        ),
        importance_score=score,
        concept_title="Session rule",
    )
    memory.importance_score = score
    return memory


@pytest.fixture
def store(tmp_path):
    memory_store = Mock(persist_dir=str(tmp_path), _read_only=False)
    memory_store.memories = []
    memory_store.count.side_effect = lambda: len(memory_store.memories)

    def get_by_score(min_score=0, max_score=100, limit=100):
        return [
            deepcopy(memory) for memory in memory_store.memories
            if min_score <= memory.importance_score <= max_score
            and not memory.segment.metadata.get(RECYCLED_AT_KEY)
        ][:limit]

    def update_metadata_only(memory):
        for index, existing in enumerate(memory_store.memories):
            if existing.id == memory.id:
                memory_store.memories[index] = deepcopy(memory)
                return True
        return False

    memory_store.get_by_score.side_effect = get_by_score
    memory_store.get_mandatory.side_effect = lambda: get_by_score(100, 100)
    memory_store.update_metadata_only.side_effect = update_metadata_only
    return memory_store


@pytest.mark.parametrize("segment_type", [
    SegmentType.STANDING_INSTRUCTION, SegmentType.CRITICAL_CONSTRAINT,
])
@pytest.mark.parametrize("score", [0, 59, 79, 99])
def test_first_session_restores_mandatory_and_resets_baseline(store, segment_type, score):
    store.memories = [make_memory(segment_type, score, **{
        TURNS_SINCE_ACCESS_KEY: 42, NUDGED_AT_TURN_KEY: 40,
    })]
    tick_turn(store)
    result = tick_nudge_session(store, now=datetime(2026, 10, 2, 10))
    memory = store.memories[0]
    assert result.new_session_started
    assert memory.importance_score == 100
    assert get_turns_since_access(memory) == 0
    assert get_last_nudged_turn(memory) == get_current_turn(store) == 1
    store.update.assert_not_called()


def test_active_session_keeps_drop_but_idle_session_restores_it(store):
    store.memories = [make_memory()]
    started = datetime(2026, 10, 2, 10)
    tick_nudge_session(store, now=started)
    current_turn = tick_turn(store)
    assert apply_access_drop(store.memories[0], current_turn)
    store.update_metadata_only.reset_mock()
    result = tick_nudge_session(store, now=started + timedelta(minutes=10))
    assert not result.new_session_started
    assert store.memories[0].importance_score == 79
    store.update_metadata_only.assert_not_called()
    result = tick_nudge_session(store, now=started + timedelta(minutes=21))
    assert result.new_session_started
    assert store.memories[0].importance_score == 100
    assert apply_access_drop(store.memories[0], tick_turn(store))
    for _ in range(5):
        tick_turn(store)
    assert apply_nudge_check(store)["elevated"] == 1


def test_restore_ignores_interval_and_disabled_nudging(store):
    store.memories = [make_memory(**{
        NUDGE_ACTIVE_KEY: False, NUDGE_INTERVAL_TURNS_KEY: 0,
    })]
    tick_nudge_session(store)
    assert store.memories[0].importance_score == 100
    assert store.memories[0].segment.metadata[NUDGE_ACTIVE_KEY] is False
    assert store.memories[0].segment.metadata[NUDGE_INTERVAL_TURNS_KEY] == 0


def test_recycled_and_nonmandatory_memories_are_untouched(store):
    store.memories = [
        make_memory(SegmentType.CONTEXT),
        make_memory(**{RECYCLED_AT_KEY: "2026-10-01"}),
        make_memory(score=100),
    ]
    tick_nudge_session(store)
    assert [memory.importance_score for memory in store.memories] == [79, 79, 100]
    store.update_metadata_only.assert_not_called()


def test_restore_is_not_limited_to_200_memories(store):
    store.memories = [make_memory(content=f"Rule {index}") for index in range(205)]
    tick_nudge_session(store)
    assert all(memory.importance_score == 100 for memory in store.memories)
    assert store.update_metadata_only.call_count == 205


def test_readonly_session_is_restored_on_next_writable_tick(store):
    store.memories = [make_memory()]
    started = datetime(2026, 10, 2, 10)
    store._read_only = True
    tick_nudge_session(store, now=started)
    store.update_metadata_only.assert_not_called()
    store._read_only = False
    result = tick_nudge_session(store, now=started + timedelta(seconds=1))
    assert result.new_session_started
    assert store.memories[0].importance_score == 100


def test_external_session_tick_does_not_lose_reset(store):
    store.memories = [make_memory()]
    started = datetime(2026, 10, 2, 10)
    tick_session(store.persist_dir, now=started)
    result = tick_nudge_session(store, now=started + timedelta(seconds=1))
    assert not result.new_session_started
    assert store.memories[0].importance_score == 100


def test_failed_update_is_retried_within_same_session(store):
    store.memories = [make_memory()]
    started = datetime(2026, 10, 2, 10)
    store.update_metadata_only.return_value = False
    original_update = store.update_metadata_only.side_effect
    store.update_metadata_only.side_effect = None
    with pytest.raises(RuntimeError, match="Could not restore mandatory memory"):
        tick_nudge_session(store, now=started)
    assert store.memories[0].importance_score == 79
    store.update_metadata_only.side_effect = original_update
    tick_nudge_session(store, now=started + timedelta(seconds=1))
    assert store.memories[0].importance_score == 100


def test_session_marker_survives_restart_without_repeating_restore(store):
    from instinct.session_timer import _store_for

    store.memories = [make_memory()]
    started = datetime(2026, 10, 2, 10)
    tick_nudge_session(store, now=started)
    assert apply_access_drop(store.memories[0], tick_turn(store))
    _store_for(store.persist_dir).invalidate()
    store.update_metadata_only.reset_mock()
    tick_nudge_session(store, now=started + timedelta(seconds=1))
    assert store.memories[0].importance_score == 79
    store.update_metadata_only.assert_not_called()
