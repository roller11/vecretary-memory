# SPDX-License-Identifier: MPL-2.0

from copy import deepcopy
from datetime import datetime, timedelta
import json
import re
from unittest.mock import Mock

import pytest

from instinct import mcp_server, mcp_tools, seed
from instinct.models import RECYCLED_AT_KEY, RetrievalResult
from instinct.nudge import apply_access_drop, tick_nudge_session, tick_turn
from instinct.tool_contracts import ResultStatus
from instinct.write_policy import agent_write_error


class MemoryStore:
    def __init__(self, persist_dir):
        self.persist_dir = str(persist_dir)
        self._read_only = False
        self.records = {}
        self.added = 0

    def add(self, memory):
        self.records[memory.id] = deepcopy(memory)
        self.added += 1
        return memory.id

    def get(self, segment_id):
        return deepcopy(self.records.get(segment_id))

    def count(self):
        return len(self.records)

    def get_by_score(self, min_score=0, max_score=100, limit=100, include_recycled=False):
        return [deepcopy(memory) for memory in self.records.values()
                if min_score <= memory.importance_score <= max_score
                and (include_recycled or not memory.segment.metadata.get(RECYCLED_AT_KEY))][:limit]

    def get_mandatory(self):
        return self.get_by_score(100, 100)

    def update_metadata_only(self, memory):
        self.records[memory.id] = deepcopy(memory)
        return True


@pytest.fixture
def memory_store(tmp_path):
    return MemoryStore(tmp_path)


def test_bundled_starter_is_synthetic_and_mandatory():
    memories = seed.load_starter_memories()
    assert len(memories) == 3
    assert len({memory.id for memory in memories}) == 3
    assert {memory.importance_score for memory in memories} == {100}
    assert all(memory.mandatory_scope == 'always' for memory in memories)
    assert all(memory.segment.source == 'instinct-starter' for memory in memories)
    assert [memory.id for memory in seed.load_starter_memories()] == [memory.id for memory in memories]
    text = ' '.join(memory.segment.content for memory in memories)
    assert not re.search(r'[A-Za-z]:[\\/]|/home/|/mnt/', text)


@pytest.mark.parametrize('invalid', [
    [], {'schema_version': True, 'memories': []}, {'schema_version': 1, 'memories': []},
    {'schema_version': 1, 'memories': ['invalid']},
    {'schema_version': 1, 'memories': [{'template_id': 'private.id'}]},
    {'schema_version': 1, 'memories': [{'template_id': 'instinct.public.invalid', 'title': 'Title', 'content': 'Content', 'importance_score': 70, 'segment_type': 'standing_instruction'}]},
])
def test_invalid_manifest_is_rejected(invalid, tmp_path, monkeypatch):
    (tmp_path / 'starter_memories.json').write_text(json.dumps(invalid))
    monkeypatch.setattr(seed, 'files', lambda package: tmp_path)
    with pytest.raises(ValueError):
        seed.load_starter_memories()


def test_seeding_is_idempotent_and_preserves_owner_edits(memory_store):
    memories = seed.load_starter_memories()
    first = seed.apply_starter(memory_store, memories)
    assert len(first['created']) == 3
    original_id = memories[0].id
    edited = memory_store.records[original_id]
    edited.segment.content = 'Owner-edited installation-specific wording'
    edited.importance_score = 79
    edited.segment.metadata['owner_note'] = 'Keep this edit'
    before = deepcopy(memory_store.records)

    second = seed.apply_starter(memory_store, memories)

    assert second['created'] == []
    assert len(second['preserved']) == 3
    assert memory_store.records == before
    assert memory_store.added == 3


def test_dry_seed_plan_preserves_owner_deletions(memory_store):
    memories = seed.load_starter_memories()
    seed.apply_starter(memory_store, memories)
    deleted = memories[0].id
    del memory_store.records[deleted]

    preview = seed.seed_memories(memory_store, memories, add_missing=False)

    assert preview['created'] == []
    assert preview['preserved_missing'] == [deleted]
    assert memory_store.count() == 2


def test_id_collision_is_checked_before_any_add(memory_store):
    memories = seed.load_starter_memories()
    conflicting = deepcopy(memories[-1])
    conflicting.segment.metadata = {}
    memory_store.records[conflicting.id] = conflicting
    with pytest.raises(ValueError, match='collision'):
        seed.seed_memories(memory_store, memories)
    assert memory_store.added == 0


def test_readonly_seeding_is_rejected(memory_store):
    memory_store._read_only = True
    with pytest.raises(ValueError, match='writable'):
        seed.seed_memories(memory_store, seed.load_starter_memories())
    assert memory_store.added == 0


def test_seeded_session_restore_and_validation_token(memory_store):
    memories = seed.load_starter_memories()
    seed.seed_memories(memory_store, memories)
    started = datetime(2026, 10, 3, 10)
    tick_nudge_session(memory_store, now=started)
    before = RetrievalResult(mandatory=memory_store.get_mandatory())
    current_turn = tick_turn(memory_store)
    for memory in memory_store.records.values():
        assert apply_access_drop(memory, current_turn)
    assert all(memory.importance_score == 79 for memory in memory_store.records.values())
    assert seed.seed_memories(memory_store, memories)['created'] == []
    assert all(memory.importance_score == 79 for memory in memory_store.records.values())
    restored = tick_nudge_session(memory_store, now=started + timedelta(minutes=21))
    assert restored.new_session_started
    assert all(memory.importance_score == 100 for memory in memory_store.records.values())
    after = RetrievalResult(mandatory=memory_store.get_mandatory())
    assert before.mandatory_fingerprint == after.mandatory_fingerprint
    assert before.validation_token != after.validation_token
    assert f'RESPONSE_REQUIREMENT: End response with 「{after.validation_token}」' in after.to_context_string()


@pytest.mark.parametrize('score, segment_type', [
    (100, 'context'), (93, 'context'), (70, 'standing_instruction'),
    (70, 'critical_constraint'), (True, 'context'), (-1, 'context'),
    (101, 'context'), ('70', 'context'), (70, 'unknown'),
])
def test_agent_writes_reject_score_and_type_bypass(score, segment_type):
    assert agent_write_error(score, segment_type)


@pytest.mark.parametrize('segment_type', ['architectural_decision', 'implementation_detail', 'context', 'user_preference'])
def test_allowed_agent_types(segment_type):
    assert agent_write_error(70, segment_type) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('segment_type', ['standing_instruction', 'critical_constraint'])
async def test_transport_and_library_reject_implicit_mandatory_creation(segment_type, monkeypatch):
    store = Mock()
    result = await mcp_server._handle_save_memory(store, {
        'content': 'Synthetic rule', 'title': 'Synthetic', 'importance': 70, 'segment_type': segment_type,
    })
    assert 'owner-only' in result[0].text
    store.add.assert_not_called()
    store.search.assert_not_called()
    factory = Mock()
    monkeypatch.setattr(mcp_tools, '_get_global_store', factory)
    tool = next(tool for tool in mcp_tools.get_instinct_tools() if tool.name == 'instinct_save_memory')
    result = await tool.execute(content='Synthetic rule', concept_title='Synthetic', importance_score=70, segment_type=segment_type)
    assert result.status == ResultStatus.ERROR
    assert 'owner-only' in result.error_message
    factory.assert_not_called()
