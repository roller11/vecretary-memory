# SPDX-License-Identifier: MPL-2.0
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from instinct import mcp_server, mcp_tools, nudge
from instinct.mandatory_lifecycle import MandatoryLifecycle
from instinct.models import RECYCLED_AT_KEY, ScoredSegment, Segment, SegmentType
from instinct.service_binding import bind_store
from instinct.seed import load_starter_memories, seed_memories
from instinct.tool_contracts import ResultStatus
from tests.test_seed import MemoryStore


@pytest.fixture
def store(tmp_path):
    memory = MemoryStore(tmp_path / 'data')
    seed_memories(memory, load_starter_memories())
    nudge.tick_nudge_session(memory, now=datetime(2026, 10, 3, 10))
    turn = nudge.tick_turn(memory)
    for record in memory.records.values():
        nudge.apply_access_drop(record, turn)
    assert all(record.importance_score == 79 for record in memory.records.values())
    return memory


def test_renewal_changes_only_existing_active_mandatory_metadata(store):
    records = list(store.records.values())
    records[0].segment.stored_id = records[0].id
    records[0].concept_title = 'Owner edited title'
    records[0].segment.content = 'Owner edited content'
    records[1].segment.metadata[RECYCLED_AT_KEY] = '2026-10-03'
    ordinary = ScoredSegment(Segment('Synthetic ordinary', SegmentType.CONTEXT), importance_score=70, concept_title='Synthetic ordinary')
    store.add(ordinary)
    before = deepcopy(store.records)
    added = store.added
    assert nudge.renew_mandatory(store) == 2
    assert store.added == added
    assert store.records[records[1].id].importance_score == 79
    assert store.records[ordinary.id].importance_score == 70
    assert set(store.records) == set(before)
    for segment_id, record in store.records.items():
        assert record.segment.content == before[segment_id].segment.content
        assert record.concept_title == before[segment_id].concept_title
        assert record.segment.segment_type == before[segment_id].segment.segment_type


def test_connection_renewal_is_once_and_explicit_event_rearms(store):
    lifecycle = MandatoryLifecycle()
    assert lifecycle.restore_if_pending(store)
    turn = nudge.tick_turn(store)
    for record in store.records.values():
        nudge.apply_access_drop(record, turn)
    assert lifecycle.restore_if_pending(store)
    assert all(record.importance_score == 79 for record in store.records.values())
    lifecycle.request()
    assert lifecycle.restore_if_pending(store)
    assert all(record.importance_score == 99 for record in store.records.values())


def test_readonly_renewal_remains_pending_without_writes(store, monkeypatch):
    lifecycle = MandatoryLifecycle()
    store._read_only = True
    update = Mock(wraps=store.update_metadata_only)
    monkeypatch.setattr(store, 'update_metadata_only', update)
    assert not lifecycle.restore_if_pending(store)
    update.assert_not_called()
    with pytest.raises(RuntimeError, match='writable'):
        nudge.renew_mandatory(store)
    store._read_only = False
    assert lifecycle.restore_if_pending(store)
    assert update.call_count == 3


def test_failed_renewal_retries_without_reseeding(store, monkeypatch):
    lifecycle = MandatoryLifecycle()
    update = store.update_metadata_only
    monkeypatch.setattr(store, 'update_metadata_only', Mock(return_value=False))
    with pytest.raises(RuntimeError, match='Could not restore mandatory memory'):
        lifecycle.restore_if_pending(store)
    monkeypatch.setattr(store, 'update_metadata_only', update)
    assert lifecycle.restore_if_pending(store)
    assert all(record.importance_score == 99 for record in store.records.values())


def test_renewal_does_not_resurrect_deleted_starters(store):
    removed = next(iter(store.records))
    store.records.pop(removed)
    assert nudge.renew_mandatory(store) == 2
    assert removed not in store.records


def test_concurrent_first_calls_renew_once(store, monkeypatch):
    lifecycle = MandatoryLifecycle()
    update = Mock(wraps=store.update_metadata_only)
    monkeypatch.setattr(store, 'update_metadata_only', update)
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(workers.map(lambda unused: lifecycle.restore_if_pending(store), range(8)))
    assert update.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('name, arguments', [
    ('instinct_get_context', {'query': 'synthetic', 'new_session': True}),
    ('instinct_get_mandatory', {'new_session': True}),
])
async def test_transport_explicit_event_restores_before_handler(store, monkeypatch, name, arguments):
    monkeypatch.setattr(nudge, 'tick_nudge_session', lambda *args, **kwargs: None)

    async def check(actual, supplied):
        assert actual is store
        assert len(store.get_mandatory()) == 3
        return []

    monkeypatch.setattr(mcp_server, '_handle_get_context', check)
    monkeypatch.setattr(mcp_server, '_handle_get_mandatory', check)
    lifecycle = MandatoryLifecycle()
    with bind_store(store):
        token = mcp_server._backend_lifecycle.set(lifecycle)
        try:
            assert await mcp_server.call_tool(name, arguments) == []
        finally:
            mcp_server._backend_lifecycle.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', ['true', 1, None])
async def test_invalid_lifecycle_flag_rejected_before_store(monkeypatch, flag):
    factory = Mock()
    monkeypatch.setattr(mcp_server, 'get_store', factory)
    result = await mcp_server.call_tool('instinct_get_context', {'query': 'synthetic', 'new_session': flag})
    assert 'Error' in result[0].text
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_transport_readonly_reports_pending_and_retries_after_handoff(store, monkeypatch):
    lifecycle = MandatoryLifecycle()
    store._read_only = True
    monkeypatch.setattr(mcp_server, '_connection_lifecycle', lambda: lifecycle)
    monkeypatch.setattr(nudge, 'tick_nudge_session', lambda *args, **kwargs: None)
    with bind_store(store):
        token = mcp_server._backend_lifecycle.set(lifecycle)
        try:
            pending = await mcp_server.call_tool('instinct_get_mandatory', {})
            assert 'Mandatory renewal is pending' in pending[0].text
            assert all(record.importance_score == 79 for record in store.records.values())
            store._read_only = False
            completed = await mcp_server.call_tool('instinct_get_mandatory', {})
            assert 'Found 3 mandatory items' in completed[0].text
        finally:
            mcp_server._backend_lifecycle.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['instinct_get_context', 'instinct_get_mandatory'])
async def test_library_explicit_event_renews_existing_mandatories(store, monkeypatch, name):
    retriever = SimpleNamespace(store=store)
    monkeypatch.setattr('instinct.retriever.get_retriever', lambda: retriever)
    retriever.retrieve_mandatory = store.get_mandatory
    from instinct.models import RetrievalResult
    retriever.retrieve = lambda **kwargs: RetrievalResult(mandatory=store.get_mandatory())
    tool = next(tool for tool in mcp_tools.get_instinct_tools() if tool.name == name)
    parameters = {'query': 'synthetic'} if name == 'instinct_get_context' else {}
    result = await tool.execute(**parameters, new_session=True)
    assert result.status == ResultStatus.SUCCESS
    assert len(store.get_mandatory()) == 3
