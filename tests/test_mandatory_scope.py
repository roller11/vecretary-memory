# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
from types import SimpleNamespace

import pytest

from instinct import nudge
from instinct.models import RetrievalResult, ScoredSegment, Segment, SegmentType
from instinct.modifiers import QueryContext
from instinct.retriever import PriorityRetriever
from tests.test_seed import MemoryStore


def make_memory(scope="always", score=100, segment_type=SegmentType.STANDING_INSTRUCTION):
    return ScoredSegment(
        Segment(
            "Synthetic scoped memory",
            segment_type,
            metadata={"mandatory_scope": scope, "mandatory_tags": ["security"]},
        ),
        importance_score=score,
        concept_title="Synthetic scope",
    )


@pytest.mark.parametrize("scope", ["standard", "", None, "always", "always-on", "always mandatory", "legacy unknown"])
def test_legacy_mandatory_scope_is_canonical_without_mutating_metadata(scope):
    memory = make_memory(scope)
    original = deepcopy(memory.segment.metadata)
    restored = ScoredSegment.from_storage_dict(memory.to_storage_dict())

    assert memory.mandatory_scope == restored.mandatory_scope == "always"
    assert memory.segment.metadata == original
    result = RetrievalResult(mandatory=[restored])
    assert result.mandatory_verification["always_on_count"] == 1
    assert result.mandatory_verification["conditional_count"] == 0
    assert "MANDATORY_ACTIVE: 1 always-on | 0 conditional" in result.to_context_string()
    assert "SCOPE: ALWAYS" in result.to_context_string()
    assert "WHEN[]" not in result.to_context_string()


@pytest.mark.parametrize("scope", ["always", "conditional", "conditional mandatory"])
def test_suppression_and_storage_round_trip_preserve_scope(scope):
    memory = make_memory(scope)
    assert nudge.apply_access_drop(memory, current_turn=1)
    assert memory.importance_score == 79
    canonical = "conditional" if scope.startswith("conditional") else "always"
    stored = memory.to_storage_dict()
    assert stored["mandatory_scope"] == canonical
    restored = ScoredSegment.from_storage_dict(stored)
    assert restored.mandatory_scope == canonical
    assert restored.importance_score == 79
    restored.importance_score = 100
    assert restored.mandatory_scope == canonical


@pytest.mark.parametrize("segment_type", [SegmentType.STANDING_INSTRUCTION, SegmentType.CRITICAL_CONSTRAINT, SegmentType.CONTEXT])
def test_original_mandatory_identity_preserves_conditional_scope(segment_type):
    memory = make_memory("conditional", segment_type=segment_type)
    memory.importance_score = 70
    assert memory.mandatory_scope == "conditional"
    assert ScoredSegment.from_storage_dict(memory.to_storage_dict()).mandatory_scope == "conditional"


@pytest.mark.parametrize("scope", ["always", "conditional", "standard"])
def test_ordinary_memories_remain_standard(scope):
    memory = make_memory(scope, score=70, segment_type=SegmentType.CONTEXT)
    assert not memory.is_mandatory_memory
    assert memory.mandatory_scope == "standard"
    assert memory.to_storage_dict()["mandatory_scope"] == "standard"


def test_conditional_scope_survives_session_renewal_and_controls_inclusion(tmp_path):
    memory = make_memory("conditional")
    nudge.apply_access_drop(memory, current_turn=1)
    store = MemoryStore(tmp_path / "data")
    store.add(ScoredSegment.from_storage_dict(memory.to_storage_dict()))

    assert nudge.renew_mandatory(store) == 1
    restored = store.get(memory.id)
    assert restored.importance_score == 99
    assert restored.mandatory_scope == "conditional"
    retriever = PriorityRetriever(store=SimpleNamespace())
    assert not retriever._should_include_mandatory(restored, QueryContext(task_tags={"garden"}))
    assert retriever._should_include_mandatory(restored, QueryContext(task_tags={"security"}))
    result = RetrievalResult(mandatory=[restored])
    assert result.mandatory_verification["conditional_count"] == 1
    assert "MANDATORY_ACTIVE: 0 always-on | 1 conditional" in result.to_context_string()
    assert "SCOPE: WHEN[security]" in result.to_context_string()
