# SPDX-License-Identifier: MPL-2.0
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from instinct import session_timer
from instinct.active_time import capture_anchors, elapsed_hours, set_anchor
from instinct.dormancy import config_for, current_effective_score
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_runtime import SQLiteSegmentAdapter
from instinct.sqlite_store import SQLiteRecordStore
from instinct.sqlite_writes import BoundDocumentEmbedder


@pytest.fixture
def clock(tmp_path):
    tmp_path.chmod(0o700)
    session_timer.reset_session(tmp_path)
    return tmp_path


def note(score=70):
    return ScoredSegment(
        segment=Segment(content="Synthetic active-clock test", segment_type=SegmentType.CONTEXT),
        importance_score=score, concept_title="Synthetic clock note",
    )


class FixedDocumentProvider:
    def name(self):
        return 'synthetic'

    def get_config(self):
        return {'provider': 'synthetic', 'model_name': 'fixed'}

    def supported_spaces(self):
        return ['cosine']

    def embed_documents(self, documents):
        return [[1, 0] for _ in documents]


@pytest.fixture
def sqlite_store(clock):
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    with SQLiteRecordStore.create(clock / 'sqlite', identity) as repository:
        yield SQLiteSegmentAdapter(
            repository, persist_dir=clock,
            document_embedder=BoundDocumentEmbedder(identity, FixedDocumentProvider()),
        )


def advance(clock, minutes, start=None):
    start = start or datetime(2026, 10, 3, 10)
    session_timer.tick_session(clock, now=start)
    for minute in range(1, minutes + 1):
        session_timer.tick_session(clock, now=start + timedelta(minutes=minute))
    return start + timedelta(minutes=minutes)


def test_new_memory_does_not_inherit_previous_activity(clock):
    end = advance(clock, 120)
    old = note()
    new = note()
    capture_anchors(old, clock)
    capture_anchors(new, clock, new=True)
    assert elapsed_hours(old, "created", clock) == 2.0
    assert elapsed_hours(new, "created", clock) == 0.0
    session_timer.tick_session(clock, now=end + timedelta(minutes=5))
    assert elapsed_hours(new, "created", clock) == pytest.approx(5 / 60)


def test_idle_and_reconnect_preserve_per_memory_age(clock):
    memory = note()
    capture_anchors(memory, clock, new=True)
    end = advance(clock, 30)
    before = elapsed_hours(memory, "created", clock)
    session_timer.tick_session(clock, now=end + timedelta(days=7))
    session_timer._store_for(clock).invalidate()
    restored = ScoredSegment.from_storage_dict(memory.to_storage_dict())
    assert elapsed_hours(restored, "created", clock) == before
    session_timer.tick_session(clock, now=end + timedelta(days=7, minutes=5))
    assert elapsed_hours(restored, "created", clock) == pytest.approx(before + 5 / 60)


def test_access_and_bump_reset_only_their_own_anchors(clock):
    memory = note()
    capture_anchors(memory, clock, new=True)
    end = advance(clock, 30)
    memory.last_accessed = end
    set_anchor(memory, "last_accessed", clock)
    assert elapsed_hours(memory, "last_accessed", clock) == 0
    assert elapsed_hours(memory, "created", clock) == 0.5
    session_timer.tick_session(clock, now=end + timedelta(minutes=5))
    memory.last_bumped_at = end + timedelta(minutes=5)
    set_anchor(memory, "last_bumped_at", clock)
    assert elapsed_hours(memory, "last_accessed", clock) == pytest.approx(5 / 60)
    assert elapsed_hours(memory, "last_bumped_at", clock) == 0


def test_legacy_clock_migrates_without_charging_past_activity(clock):
    from instinct.private_storage import write_private_text
    write_private_text(clock / "session_state.json", '{"accumulated_session_seconds": 360000}')
    session_timer._store_for(clock).invalidate()
    memory = note()
    end = advance(clock, 5)
    assert elapsed_hours(memory, "created", clock) == pytest.approx(5 / 60)
    capture_anchors(memory, clock)
    assert elapsed_hours(memory, "created", clock) == pytest.approx(5 / 60)
    assert current_effective_score(memory, end, config_for(clock)) > 69


def test_clock_reset_does_not_reuse_old_anchors(clock):
    memory = note()
    capture_anchors(memory, clock, new=True)
    advance(clock, 10)
    session_timer.reset_session(clock)
    advance(clock, 20)
    assert elapsed_hours(memory, "created", clock) == 0
    capture_anchors(memory, clock)
    assert elapsed_hours(memory, "created", clock) == 0


def test_real_ticks_use_monotonic_clock_despite_wall_clock_jump(clock, monkeypatch):
    fake_datetime = Mock()
    fake_datetime.now.side_effect = [datetime(2026, 10, 3), datetime(2036, 10, 3), datetime(2016, 10, 3)]
    fake_datetime.fromisoformat = datetime.fromisoformat
    monkeypatch.setattr(session_timer, "datetime", fake_datetime)
    monotonic = iter([100.0, 160.0, 220.0, 225.0])
    monkeypatch.setattr(session_timer.time, "monotonic", lambda: next(monotonic))
    session_timer.tick_session(clock)
    session_timer.tick_session(clock)
    session_timer.tick_session(clock)
    assert session_timer.state(clock).accumulated_session_seconds == 120.0
    assert session_timer.session_alive(clock)


def test_first_real_tick_after_process_restart_does_not_charge_offline_gap(clock, monkeypatch):
    advance(clock, 5)
    session_timer._store_for(clock).invalidate()
    monkeypatch.setattr(session_timer.time, "monotonic", lambda: 100.0)
    result = session_timer.tick_session(clock)
    assert result.new_session_started
    assert result.seconds_added_to_accumulator == 0
    assert result.state.accumulated_session_seconds == 300


@pytest.mark.parametrize("score", [10, 60, 70, 88, 90, 94, 95])
def test_retrieval_and_dashboard_use_identical_decay(clock, score):
    from instinct.decay import DecayModifier
    from instinct.modifiers import QueryContext, ScoringResult
    memory = note(score)
    capture_anchors(memory, clock, new=True)
    end = advance(clock, 300)
    expected = current_effective_score(memory, end, config_for(clock))
    result = ScoringResult(memory.id, score, float(score))
    result = DecayModifier().modify(memory, QueryContext(now=end, persist_dir=str(clock)), result)
    assert result.effective_score == expected
    assert expected < score


def test_repeated_decay_ticks_do_not_reapply_total_age(clock):
    memory = note()
    capture_anchors(memory, clock, new=True)
    end = advance(clock, 60)
    cfg = config_for(clock)
    expected = current_effective_score(memory, end, cfg)
    memory.importance_score = round(expected)
    assert current_effective_score(memory, end, cfg) == expected


def test_mandatory_grace_ignores_wall_clock_idle(clock):
    memory = note(100)
    memory.last_bumped_at = datetime(2026, 10, 3, 10)
    capture_anchors(memory, clock, new=True)
    cfg = config_for(clock)
    assert current_effective_score(memory, memory.last_bumped_at + timedelta(days=100), cfg) == 100
    end = advance(clock, 4)
    assert current_effective_score(memory, end, cfg) < 100


def test_pop_halves_active_age_across_idle_gap(clock):
    from instinct.pop import apply_pop
    from instinct.dormancy import _active_age
    memory = note()
    capture_anchors(memory, clock, new=True)
    end = advance(clock, 60)
    end += timedelta(days=7)
    session_timer.tick_session(clock, now=end)
    cfg = config_for(clock)
    result = apply_pop(memory, 2, cfg, now=end)
    assert result.rule_applied == "halve"
    assert _active_age(memory, end, cfg) == 0.5
    restored = ScoredSegment.from_storage_dict(memory.to_storage_dict())
    assert _active_age(restored, end, cfg) == 0.5


def test_pop_frequency_window_expires_in_active_time_only(clock):
    from instinct.pop import _log_event, _recent_count_in_active_window
    memory = note()
    end = advance(clock, 5)
    _log_event(memory, "pop", 2, end, persist_dir=clock)
    end += timedelta(days=30)
    session_timer.tick_session(clock, now=end)
    cfg = config_for(clock)
    assert _recent_count_in_active_window(memory, 8, cfg, end) == 1
    advance(clock, 481, start=end)
    assert _recent_count_in_active_window(memory, 8, cfg, end) == 0


def test_legacy_pop_does_not_trigger_from_wall_clock_age(clock):
    from instinct.decay import PopModifier
    memory = note()
    memory.last_accessed = datetime(2000, 1, 1)
    capture_anchors(memory, clock, new=True)
    advance(clock, 5)
    assert not PopModifier(persist_dir=clock).should_pop(memory, datetime(2026, 10, 3))


def test_persisted_score_updates_rebase_the_decay_curve(clock, sqlite_store):
    store = sqlite_store
    end = advance(clock, 60)
    memory = note()
    store.add(memory)
    assert elapsed_hours(store.get(memory.id), "created", clock) == 0
    session_timer.tick_session(clock, now=end + timedelta(minutes=5))
    store.update_score(memory.id, 40)
    restored = store.get(memory.id)
    assert current_effective_score(restored, end, config_for(clock)) == 40


def test_store_access_persists_its_own_snapshot(clock, sqlite_store):
    store = sqlite_store
    memory = note()
    store.add(memory)
    end = advance(clock, 60)
    store.record_access(memory.id, now=end)
    restored = store.get(memory.id)
    assert elapsed_hours(restored, "created", clock) == 1
    assert elapsed_hours(restored, "last_accessed", clock) == 0
    session_timer.tick_session(clock, now=end + timedelta(minutes=5))
    assert elapsed_hours(store.get(memory.id), "last_accessed", clock) == pytest.approx(5 / 60)


def test_cached_observer_sees_external_clock_updates(clock):
    observer = session_timer._SessionStore(clock)
    assert observer.load().accumulated_session_seconds == 0
    advance(clock, 5)
    assert observer.load().accumulated_session_seconds == 300


def test_readonly_connection_does_not_advance_clock(clock):
    from instinct.nudge import tick_nudge_session
    end = advance(clock, 5)
    before = session_timer.state(clock)
    observer = Mock(persist_dir=str(clock), _read_only=True)
    result = tick_nudge_session(observer, now=end + timedelta(minutes=5))
    assert result.state == before
    assert result.seconds_added_to_accumulator == 0


def test_old_boot_activity_is_not_reported_as_live(clock, monkeypatch):
    monkeypatch.setattr(session_timer.time, "monotonic", lambda: 100)
    session_timer.tick_session(clock)
    monkeypatch.setattr(session_timer, "_monotonic_identity", lambda: "another-boot")
    assert not session_timer.session_alive(clock)


def test_memories_saved_before_first_tick_age_only_after_clock_start(tmp_path):
    memory = note()
    capture_anchors(memory, tmp_path, new=True)
    end = advance(tmp_path, 60)
    assert current_effective_score(memory, end, config_for(tmp_path)) == pytest.approx(67.5)


def test_decay_basis_survives_crossing_a_score_band(clock):
    memory = note(50)
    capture_anchors(memory, clock, new=True)
    end = advance(clock, 60)
    cfg = config_for(clock)
    expected = current_effective_score(memory, end, cfg)
    memory.importance_score = round(expected)
    assert memory.importance_score < 50
    assert current_effective_score(memory, end, cfg) == expected


@pytest.mark.parametrize("score", [10, 60, 88, 90, 94, 95, 100])
def test_curve_preview_projects_active_time_without_ticking_store(clock, monkeypatch, score):
    from instinct import dormancy
    monkeypatch.setattr(dormancy, "_store", dormancy._ConfigStore(clock))
    before = session_timer.state(clock)
    series = dormancy.decay_series(score, "standard", days=2, points=48)
    assert series[0]["score"] == score
    assert series[-1]["score"] < score
    assert session_timer.state(clock) == before


def test_stc_preview_uses_active_hour_expiry(clock, monkeypatch):
    from instinct import dormancy
    monkeypatch.setattr(dormancy, "_store", dormancy._ConfigStore(clock))
    series = dormancy.decay_series(91, "short_term_critical", days=1, points=24)
    assert series[2]["score"] > 49
    assert series[3]["score"] == 49


def test_mandatory_renewal_and_access_drop_rebase_the_curve(clock, sqlite_store):
    from instinct.nudge import renew_mandatory
    store = sqlite_store
    memory = ScoredSegment(Segment("Synthetic rule", SegmentType.STANDING_INSTRUCTION), 100, "Synthetic rule")
    store.add(memory)
    end = advance(clock, 60)
    store.record_access(memory.id, now=end)
    dropped = store.get(memory.id)
    cfg = config_for(clock)
    assert dropped.importance_score == 79
    assert current_effective_score(dropped, end, cfg) == 79
    assert renew_mandatory(store) == 1
    renewed = store.get(memory.id)
    assert current_effective_score(renewed, end + timedelta(days=7), cfg) == 99
    assert elapsed_hours(renewed, "last_bumped_at", clock) == 0
