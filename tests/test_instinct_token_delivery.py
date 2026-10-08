# SPDX-License-Identifier: MPL-2.0
from datetime import datetime
import asyncio
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from instinct import mcp_server
from instinct.models import RetrievalResult, ScoredSegment, Segment, SegmentType, build_validation_token


def make_mandatory_segment(title: str, content: str) -> ScoredSegment:
    return ScoredSegment(
        segment=Segment(
            content=content,
            segment_type=SegmentType.STANDING_INSTRUCTION,
            timestamp=datetime(2026, 4, 14, 12, 0, 0),
            source="test",
        ),
        importance_score=100,
        concept_title=title,
        reasoning="test",
    )


def test_validation_token_is_fresh_for_same_fingerprint():
    fingerprint = "abc123finger"
    token_one = build_validation_token(fingerprint)
    token_two = build_validation_token(fingerprint)

    assert token_one != token_two
    assert len(token_one) == 12
    assert token_one.upper() == token_one


def test_validation_token_changes_when_mandatory_set_changes():
    token_one = build_validation_token("fingerprint-a")
    token_two = build_validation_token("fingerprint-b")

    assert token_one != token_two


def test_retrieval_result_surfaces_validation_token_everywhere():
    mandatory = [
        make_mandatory_segment("Validation Rules", "Call get_mandatory before each response."),
        make_mandatory_segment("Formatting", "End the response with the token."),
    ]
    result = RetrievalResult(mandatory=mandatory)

    assert result.validation_token == result.mandatory_verification["validation_token"]

    context = result.to_context_string(max_tokens=2000)
    assert f"VALIDATION_TOKEN: {result.validation_token}" in context
    assert f"RESPONSE_REQUIREMENT: End response with 「{result.validation_token}」" in context

    assert context == result.to_context_string(max_tokens=2000)


def test_same_memories_get_new_challenge_each_retrieval():
    mandatory = [make_mandatory_segment("Rules", "Use only authorized memory tools.")]
    results = [RetrievalResult(mandatory=mandatory) for unused in range(100)]

    assert len({result.validation_token for result in results}) == 100
    assert len({result.mandatory_fingerprint for result in results}) == 1
    assert all(re.fullmatch(r"[0-9A-F]{12}", result.validation_token) for result in results)


def test_suppressed_or_empty_context_does_not_emit_challenge():
    memory = make_mandatory_segment("Rules", "Use only authorized memory tools.")
    memory.importance_score = 79
    ordinary = ScoredSegment(Segment("Synthetic note", SegmentType.CONTEXT), importance_score=70, concept_title="Synthetic note")
    for result in (RetrievalResult(), RetrievalResult(relevant=[ordinary])):
        context = result.to_context_string()
        assert "VALIDATION_TOKEN:" not in context
        assert "RESPONSE_REQUIREMENT:" not in context
        assert result.acknowledgement_context() == ""
        assert result.mandatory_verification["validation_token"] is None
    assert memory.importance_score == 79


def test_suppressed_mandatory_returned_as_relevant_gets_fresh_challenge():
    memory = make_mandatory_segment("Rules", "Use only authorized memory tools.")
    memory.importance_score = 79
    first = RetrievalResult(relevant=[memory])
    second = RetrievalResult(relevant=[memory])
    assert first.validation_token != second.validation_token
    assert first.mandatory_fingerprint == second.mandatory_fingerprint != "none"
    assert f"VALIDATION_TOKEN: {first.validation_token}" in first.to_context_string()
    assert "## MANDATORY CONTEXT" not in first.to_context_string()
    assert first.mandatory == []
    assert memory.importance_score == 79


@pytest.mark.asyncio
async def test_each_mandatory_tool_delivery_has_a_fresh_challenge():
    memory = make_mandatory_segment("Rules", "Use only authorized memory tools.")
    store = SimpleNamespace(get_mandatory=lambda: [memory])
    responses = await asyncio.gather(*(
        mcp_server._handle_get_mandatory(store, {}) for unused in range(20)
    ))
    tokens = [re.search(r"^VALIDATION_TOKEN: ([0-9A-F]{12})$", response[0].text, re.MULTILINE).group(1) for response in responses]
    assert len(set(tokens)) == 20
    assert all(f"End response with 「{token}」" in response[0].text for token, response in zip(tokens, responses))


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [100, 79])
async def test_explicit_mandatory_get_rotates_without_renewing(score):
    memory = make_mandatory_segment("Rules", "Use only authorized memory tools.")
    memory.importance_score = score
    store = SimpleNamespace(get=lambda segment_id: memory)
    first = (await mcp_server._handle_get(store, {"segment_id": memory.id}))[0].text
    second = (await mcp_server._handle_get(store, {"segment_id": memory.id}))[0].text
    first_token = re.search(r"^VALIDATION_TOKEN: (\w+)$", first, re.MULTILINE).group(1)
    second_token = re.search(r"^VALIDATION_TOKEN: (\w+)$", second, re.MULTILINE).group(1)
    assert first_token != second_token
    assert memory.importance_score == score


@pytest.mark.asyncio
async def test_explicit_search_challenges_only_returned_mandatories(tmp_path, monkeypatch):
    from instinct import modifiers
    memory = make_mandatory_segment("Rules", "Use only authorized memory tools.")
    store = SimpleNamespace(persist_dir=str(tmp_path), search=Mock(return_value=[memory]))
    pipeline = SimpleNamespace(apply=lambda segment, context: SimpleNamespace(effective_score=segment.importance_score))
    monkeypatch.setattr(modifiers, "create_default_pipeline", lambda: pipeline)
    responses = [(await mcp_server._handle_search(store, {"query": "rules"}))[0].text for unused in range(2)]
    tokens = [re.search(r"^VALIDATION_TOKEN: (\w+)$", text, re.MULTILINE).group(1) for text in responses]
    assert tokens[0] != tokens[1]
    store.search.return_value = []
    assert "VALIDATION_TOKEN:" not in (await mcp_server._handle_search(store, {"query": "missing"}))[0].text


@pytest.mark.asyncio
async def test_ordinary_get_and_missing_get_do_not_emit_challenge():
    memory = ScoredSegment(Segment("Synthetic note", SegmentType.CONTEXT), importance_score=70, concept_title="Synthetic note")
    store = SimpleNamespace(get=Mock(return_value=memory))
    assert "VALIDATION_TOKEN:" not in (await mcp_server._handle_get(store, {"segment_id": memory.id}))[0].text
    store.get.return_value = None
    assert "VALIDATION_TOKEN:" not in (await mcp_server._handle_get(store, {"segment_id": "missing"}))[0].text


@pytest.mark.asyncio
async def test_each_context_delivery_rotates_challenge(monkeypatch):
    from instinct import retriever
    memory = make_mandatory_segment("Rules", "Use only authorized memory tools.")
    fake = SimpleNamespace(retrieve=lambda query: RetrievalResult(mandatory=[memory]))
    monkeypatch.setattr(retriever, "PriorityRetriever", lambda **kwargs: fake)
    responses = [(await mcp_server._handle_get_context(None, {"query": "rules"}))[0].text for unused in range(2)]
    tokens = [re.search(r"^VALIDATION_TOKEN: (\w+)$", text, re.MULTILINE).group(1) for text in responses]
    assert tokens[0] != tokens[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["instinct_get_context", "instinct_get_mandatory"])
async def test_library_delivery_rotates_and_preserves_json_consistency(name, monkeypatch):
    from instinct import mcp_tools, retriever
    memory = make_mandatory_segment("Rules", "Use only authorized memory tools.")
    fake = SimpleNamespace(
        retrieve=lambda **kwargs: RetrievalResult(mandatory=[memory]),
        retrieve_mandatory=lambda: [memory],
    )
    monkeypatch.setattr(retriever, "get_retriever", lambda: fake)
    tool = next(tool for tool in mcp_tools.get_instinct_tools() if tool.name == name)
    arguments = {"query": "rules"} if name == "instinct_get_context" else {}
    first = await tool.execute(**arguments)
    second = await tool.execute(**arguments)
    assert first.data["validation_token"] != second.data["validation_token"]
    assert first.data["mandatory_fingerprint"] == second.data["mandatory_fingerprint"]
    for result in (first, second):
        assert result.data["validation_token"] == result.data["mandatory_verification"]["validation_token"]
        if "context" in result.data:
            assert f'End response with 「{result.data["validation_token"]}」' in result.data["context"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["instinct_get_context", "instinct_get_mandatory"])
async def test_library_empty_delivery_has_no_challenge(name, monkeypatch):
    from instinct import mcp_tools, retriever
    fake = SimpleNamespace(retrieve=lambda **kwargs: RetrievalResult(), retrieve_mandatory=lambda: [])
    monkeypatch.setattr(retriever, "get_retriever", lambda: fake)
    tool = next(tool for tool in mcp_tools.get_instinct_tools() if tool.name == name)
    arguments = {"query": "missing"} if name == "instinct_get_context" else {}
    result = await tool.execute(**arguments)
    assert result.data["validation_token"] is None
    assert result.data["mandatory_verification"]["validation_token"] is None
