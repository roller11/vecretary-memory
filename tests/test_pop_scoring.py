# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from instinct import dormancy, session_timer
from instinct.active_time import capture_anchors
from instinct.decay import DecayModifier
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.modifiers import ModifierPipeline, ScoreModifier
from instinct.pop import apply_pop
from instinct.retriever import PriorityRetriever, RetrievalConfig


@pytest.fixture
def clock(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    session_timer.reset_session(tmp_path)
    session_timer.tick_session(tmp_path, now=datetime(2026, 10, 4, 10))
    monkeypatch.setattr(dormancy, '_store', dormancy._ConfigStore(tmp_path))
    return tmp_path


def note(clock, score=80, profile='standard'):
    memory = ScoredSegment(
        segment=Segment(content=f'Synthetic pop scoring test {score}', segment_type=SegmentType.CONTEXT),
        importance_score=score, concept_title=f'Synthetic pop score {score}',
        decay_profile=profile,
    )
    capture_anchors(memory, clock, new=True)
    return memory


def advance(clock, hours):
    start = datetime.fromisoformat(session_timer.state(clock).session_last_tick_at)
    for minutes in range(5, int(hours * 60) + 1, 5):
        session_timer.tick_session(clock, now=start + timedelta(minutes=minutes))
    return start + timedelta(hours=hours)


@pytest.mark.parametrize('score, drop, target', [(80, 10, 70), (89, 5, 84), (70, 7, 63), (30, 3, 27)])
def test_jump_uses_initial_score_and_configured_drop(clock, score, drop, target):
    cfg = dormancy.config_for(clock)
    band = getattr(cfg, dormancy._band_for(score))
    band.initial_drop_pts = drop
    band.initial_drop_active_hours = 1
    memory = note(clock, score)
    memory.importance_score = max(0, score - 20)
    result = apply_pop(memory, 3, cfg)
    assert memory.initial_score == score
    assert memory.importance_score == target
    assert result.effective_score_after == target
    assert memory.active_time_anchors['pop_anchor_at']['score'] == score


def test_jump_clamps_to_configured_floor(clock):
    cfg = dormancy.config_for(clock)
    cfg.plateau.initial_drop_pts = 40
    memory = note(clock)
    assert apply_pop(memory, 3, cfg).effective_score_after == 60


def test_post_drop_phase_survives_storage_and_keeps_original_band(clock):
    cfg = dormancy.config_for(clock)
    memory = note(clock)
    apply_pop(memory, 3, cfg)
    restored = ScoredSegment.from_storage_dict(memory.to_storage_dict())
    end = advance(clock, 2)
    expected = 70 - (70 - 60) * 2 / cfg.plateau.post_drop_to_floor_active_hours
    assert dormancy.current_effective_score(restored, end, cfg) == pytest.approx(expected)


@pytest.mark.parametrize('score, target', [(90, 89), (94, 88), (95, 88)])
def test_special_tier_pop_retains_its_post_drop_curve(clock, score, target):
    cfg = dormancy.config_for(clock)
    memory = note(clock, score)
    assert apply_pop(memory, 3, cfg).effective_score_after == target
    end = advance(clock, 1)
    floor = getattr(cfg, f'tier_{score}_floor')
    span = getattr(cfg, f'tier_{score}_total_active_hours') - getattr(cfg, f'tier_{score}_drop_active_hours')
    assert dormancy.current_effective_score(memory, end, cfg) == pytest.approx(target - (target - floor) / span)
    assert apply_pop(memory, 3, cfg, now=end).effective_score_after == target


@pytest.mark.parametrize('profile, score', [('short_term_critical', 92), ('ephemeral', 80)])
def test_profile_pop_does_not_apply_standard_band_target(clock, profile, score):
    cfg = dormancy.config_for(clock)
    memory = note(clock, score, profile)
    memory.ephemeral_turns_left = 4
    apply_pop(memory, 3, cfg)
    assert memory.importance_score == score
    assert memory.ephemeral_turns_left == 4
    assert not memory.active_time_anchors['pop_anchor_at'].get('post_drop', False)


def test_frequency_reset_returns_to_initial_phase(clock):
    cfg = dormancy.config_for(clock)
    memory = note(clock)
    apply_pop(memory, 3, cfg)
    end = advance(clock, 1)
    result = apply_pop(memory, 2, cfg, now=end)
    assert result.rule_applied == 'frequency_reset'
    assert result.effective_score_after == 80
    assert not memory.active_time_anchors['pop_anchor_at'].get('post_drop', False)


def test_halving_preserves_post_drop_phase_after_frequency_window(clock):
    cfg = dormancy.config_for(clock)
    memory = note(clock)
    apply_pop(memory, 3, cfg)
    end = advance(clock, 9)
    result = apply_pop(memory, 2, cfg, now=end)
    assert result.rule_applied == 'halve'
    assert result.effective_score_after == pytest.approx(70 - 10 * 4.5 / 240)
    assert memory.active_time_anchors['pop_anchor_at']['post_drop'] is True


def test_mandatory_pop_stays_at_one_hundred(clock):
    cfg = dormancy.config_for(clock)
    memory = note(clock, 100)
    memory.importance_score = 79
    result = apply_pop(memory, 3, cfg)
    assert result.rule_applied == 'mandatory_bump'
    assert result.effective_score_after == 100


class SyntheticHitsModifier(ScoreModifier):
    def __init__(self):
        super().__init__('synthetic_hits', priority=30)

    def modify(self, segment, context, result):
        result.identifier_hits = 3
        return result


def retriever_for(clock, memories, read_only=False):
    store = Mock(persist_dir=str(clock))
    store.search.return_value = memories
    store.search_by_identifiers.return_value = []
    if read_only:
        store.add.side_effect = RuntimeError('read-only synthetic store')
    pipeline = ModifierPipeline()
    pipeline.register(DecayModifier())
    pipeline.register(SyntheticHitsModifier())
    return PriorityRetriever(store, config=RetrievalConfig(always_include_mandatory=False), pipeline=pipeline)


@pytest.mark.parametrize('read_only', [False, True])
def test_retrieval_refreshes_cached_score_and_uses_saved_configuration(clock, read_only):
    dormancy._store.update_band('plateau', initial_drop_pts=17)
    memory = note(clock, 88)
    retriever = retriever_for(clock, [memory], read_only)
    result = retriever.retrieve('Synthetic pop scoring', track_access=False)
    assert retriever.get_scoring_result(memory.id).effective_score == 71
    assert result.pop_results[0].effective_score_before == 88
    assert result.pop_results[0].effective_score_after == 71
    assert result.relevant[0].importance_score == 71


def test_returned_order_reflects_refreshed_scores(clock):
    dormancy._store.update_band('plateau', initial_drop_pts=20)
    dormancy._store.update_band('mid', initial_drop_pts=1)
    higher = note(clock, 80)
    lower = note(clock, 79)
    retriever = retriever_for(clock, [higher, lower])
    result = retriever.retrieve('Synthetic pop scoring', track_access=False)
    assert [memory.id for memory in result.relevant] == [lower.id, higher.id]
    assert [retriever.get_scoring_result(memory.id).effective_score for memory in result.relevant] == [78, 60]
