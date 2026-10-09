# SPDX-License-Identifier: MPL-2.0
"""Synthetic model writes, provider rejection and transactional lifecycle tests."""

from copy import deepcopy
from datetime import datetime, timedelta
import sqlite3

import pytest

from instinct import dormancy, session_timer
from instinct import fallback_store
from instinct.active_time import elapsed_hours
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.pop import apply_pop
from instinct.sqlite_queries import BoundQueryEmbedder
from instinct.sqlite_records import EmbeddingIdentity
from instinct.sqlite_runtime import SQLiteSegmentAdapter
from instinct.sqlite_session import SQLiteRuntimeSession
from instinct.sqlite_store import SQLiteRecordStore
from instinct.sqlite_writes import BoundDocumentEmbedder, MAX_DOCUMENT_BYTES, MAX_WRITE_BATCH
from instinct.write_policy import agent_target_error, agent_write_error


class DocumentProvider:
    def __init__(self):
        self.calls = []
        self.model = 'fixed'
        self.url = None
        self.response = None
        self.failure = False
        self.change_model = False

    def name(self):
        return 'synthetic'

    def get_config(self):
        result = {'provider': 'synthetic', 'model_name': self.model}
        if self.url is not None:
            result['base_url'] = self.url
        return result

    def supported_spaces(self):
        return ['cosine']

    def embed_documents(self, documents):
        self.calls.append(documents)
        if self.failure:
            raise RuntimeError('synthetic provider private details')
        if self.change_model:
            self.model = 'changed'
        return self.response if self.response is not None else [[len(document), 1] for document in documents]

    def embed_query(self, queries):
        return [[1, 0] for query in queries]


@pytest.fixture
def runtime(tmp_path):
    tmp_path.chmod(0o700)
    session_timer.reset_session(tmp_path)
    session_timer.tick_session(tmp_path, now=datetime(2026, 10, 7, 10))
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    provider = DocumentProvider()
    with SQLiteRecordStore.create(tmp_path / 'sqlite', identity) as repository:
        adapter = SQLiteSegmentAdapter(repository, persist_dir=tmp_path,
            query_embedder=BoundQueryEmbedder(identity, provider),
            document_embedder=BoundDocumentEmbedder(identity, provider))
        yield adapter, provider


def memory(record_id='note', content='Synthetic model write', score=80):
    return ScoredSegment(segment=Segment(content=content, stored_id=record_id,
        segment_type=SegmentType.CONTEXT, timestamp=datetime(2026, 10, 7, 10)),
        importance_score=score, concept_title='Synthetic write fixture',
        memory_identifiers=['sqlite', 'activity timer', 'vector recall'])


def test_new_model_batch_embeds_once_and_survives_read_only_reopen(runtime):
    adapter, provider = runtime
    first, second = memory('one'), memory('two', 'Second synthetic document')
    assert adapter.add_batch([first, second]) == ['one', 'two']
    assert provider.calls == [[first.segment.content, second.segment.content]]
    assert first.embedding == [len(first.segment.content), 1]
    assert first.active_time_anchors['created']['seconds'] == 0
    with SQLiteRecordStore.open_read_only(adapter.persist_dir / 'sqlite', adapter.repository.identity) as reader:
        reopened = SQLiteSegmentAdapter(reader, persist_dir=adapter.persist_dir)
        assert reopened.get('one').to_storage_dict() == first.to_storage_dict()
        assert reopened.get('two').embedding == second.embedding


def test_pop_and_metadata_writes_preserve_vectors_unknown_fields_and_clock(runtime):
    adapter, provider = runtime
    adapter.add(memory())
    adapter.repository.patch_metadata('note', {'opaque_extra': {'keep': [True, 2]}})
    before = adapter.repository.get_record('note')
    restored = adapter.get('note')
    assert apply_pop(restored, 3, dormancy.config_for(adapter.persist_dir),
                     now=datetime(2026, 10, 7, 10, 1)).popped
    restored.embedding = [float('nan')]
    provider.calls.clear()
    provider.model = 'unavailable-for-metadata-only'
    clock = session_timer.state(adapter.persist_dir)
    adapter.add(restored)
    after = adapter.repository.get_record('note')
    assert after.embedding == before.embedding
    assert after.content == before.content
    assert after.to_dict()['metadata']['opaque_extra'] == {'keep': [True, 2]}
    assert restored.embedding == list(before.embedding)
    assert adapter.get('note').last_popped_at == datetime(2026, 10, 7, 10, 1)
    assert provider.calls == []
    assert session_timer.state(adapter.persist_dir) == clock


def test_namespaced_score_extensions_survive_score_updates_and_reopen(runtime):
    adapter, _ = runtime
    adapter.add(memory())
    patch = {
        'confidence': 0.875,
        'signals': ['semantic', 'recency'],
        'parameters': {'half_life_hours': 48, 'enabled': True},
    }

    updated = adapter.patch_score_extension('note', 'org.rawwire.relevance', patch)
    assert updated.segment.metadata['score_extensions']['org.rawwire.relevance'] == patch
    assert adapter.update_score('note', 76, 'Synthetic module adjustment')
    persisted = adapter.repository.get_record('note').to_dict()['metadata']
    assert persisted['importance_score'] == 76
    assert persisted['score_extensions']['org.rawwire.relevance'] == patch
    with SQLiteRecordStore.open_read_only(adapter.persist_dir / 'sqlite', adapter.repository.identity) as reader:
        reopened = SQLiteSegmentAdapter(reader, persist_dir=adapter.persist_dir).get('note')
        assert reopened.segment.metadata['score_extensions']['org.rawwire.relevance'] == patch


def test_score_extension_rejects_invalid_namespace_and_non_object(runtime):
    adapter, _ = runtime
    adapter.add(memory())
    with pytest.raises(ValueError, match='namespace'):
        adapter.patch_score_extension('note', '../untrusted', {'value': 1})
    with pytest.raises(ValueError, match='object'):
        adapter.patch_score_extension('note', 'org.rawwire.relevance', ['not', 'an', 'object'])


def test_changed_text_is_reembedded_with_preserved_unknown_metadata(runtime):
    adapter, provider = runtime
    adapter.add(memory())
    adapter.repository.patch_metadata('note', {'opaque': {'keep': True},
        'content': 'Synthetic model write'})
    changed = adapter.get('note')
    before_anchor = deepcopy(changed.active_time_anchors['created'])
    changed.segment.content = 'A different synthetic document'
    changed.memory_identifiers = ['changed exact identifier']
    provider.calls.clear()
    adapter.add(changed)
    restored = adapter.get('note')
    assert provider.calls == [[changed.segment.content]]
    assert restored.embedding == [len(changed.segment.content), 1]
    assert restored.active_time_anchors['created'] == before_anchor
    assert restored.segment.metadata['opaque'] == {'keep': True}
    assert adapter.repository.get_record('note').to_dict()['metadata']['content'] == changed.segment.content
    assert adapter.search_by_identifiers(['changed exact identifier'])[0][0].id == 'note'
    assert adapter.search_by_identifiers(['vector recall']) == []


def test_without_document_binding_new_content_is_not_embedded(runtime):
    adapter, provider = runtime
    adapter.document_embedder = None
    with pytest.raises(RuntimeError, match='existing-record'):
        adapter.add(memory())
    assert adapter.count() == 0
    assert provider.calls == []


def test_mixed_batch_only_embeds_changed_or_new_documents(runtime):
    adapter, provider = runtime
    adapter.add_batch([memory('one'), memory('two')])
    first, second = adapter.get('one'), adapter.get('two')
    first.reasoning = 'Synthetic metadata update'
    second.segment.content = 'Changed second'
    third = memory('three')
    provider.calls.clear()
    assert adapter.add_batch([first, second, third]) == ['one', 'two', 'three']
    assert provider.calls == [[second.segment.content, third.segment.content]]


def test_metadata_only_batch_does_not_require_provider(runtime):
    adapter, provider = runtime
    adapter.add_batch([memory('one'), memory('two')])
    adapter.document_embedder = None
    provider.calls.clear()
    segments = adapter.get_many(['one', 'two'])
    for segment in segments:
        segment.importance_score = 79
    assert adapter.add_batch(segments) == ['one', 'two']
    assert provider.calls == []
    assert [item.importance_score for item in adapter.get_many(['one', 'two'])] == [79, 79]


@pytest.mark.parametrize('response', [[], [[1, 0], [1, 0]], [[1]],
    [[1, 0, 0]], [[True, 0]], [[float('nan'), 0]], [[float('inf'), 0]], 'invalid'])
def test_invalid_embedding_does_not_write_or_mutate_caller(runtime, response):
    adapter, provider = runtime
    segment = memory()
    before = deepcopy(segment.to_storage_dict())
    provider.response = response
    with pytest.raises(ValueError):
        adapter.add(segment)
    assert adapter.count() == 0
    assert segment.to_storage_dict() == before
    assert segment.embedding is None


def test_provider_errors_are_sanitized_and_leave_no_record(runtime):
    adapter, provider = runtime
    provider.failure = True
    with pytest.raises(RuntimeError, match='details withheld') as failure:
        adapter.add(memory())
    assert 'private details' not in str(failure.value)
    assert failure.value.__cause__ is None
    assert adapter.count() == 0


@pytest.mark.parametrize('during_call', [False, True])
def test_provider_identity_change_prevents_committing_documents(runtime, during_call):
    adapter, provider = runtime
    if during_call:
        provider.change_model = True
    else:
        provider.model = 'changed'
    with pytest.raises(ValueError, match='identity'):
        adapter.add(memory())
    assert adapter.count() == 0
    assert len(provider.calls) == int(during_call)


def test_remote_document_provider_rejected_before_embedding(runtime):
    adapter, provider = runtime
    provider.url = 'http://192.0.2.1:11434'
    with pytest.raises(ValueError):
        adapter.add(memory())
    assert provider.calls == []
    assert adapter.count() == 0


@pytest.mark.parametrize('invalid', ['score', 'initial', 'json', 'content', 'id'])
def test_invalid_model_rejected_before_provider_call(runtime, invalid):
    adapter, provider = runtime
    segment = memory()
    if invalid == 'score':
        segment.importance_score = True
    elif invalid == 'initial':
        segment.initial_score = 101
    elif invalid == 'json':
        segment.segment.metadata['bad'] = float('nan')
    elif invalid == 'content':
        segment.segment.content = 'x' * (MAX_DOCUMENT_BYTES + 1)
    else:
        segment.segment.stored_id = 'bad\x00id'
    with pytest.raises(ValueError):
        adapter.add(segment)
    assert provider.calls == []
    assert adapter.count() == 0


def test_duplicate_ids_and_oversized_batches_rejected_before_embedding(runtime):
    adapter, provider = runtime
    with pytest.raises(ValueError, match='Duplicate'):
        adapter.add_batch([memory(), memory()])
    with pytest.raises(ValueError, match='batch'):
        adapter.add_batch([memory()] * (MAX_WRITE_BATCH + 1))
    with pytest.raises(ValueError, match='batch'):
        adapter.add_batch(iter([memory()]))
    assert provider.calls == []
    assert adapter.add_batch([]) == []


def test_quota_preflight_prevents_unnecessary_embedding(runtime, monkeypatch):
    adapter, provider = runtime
    adapter.add(memory())
    provider.calls.clear()
    monkeypatch.setattr('instinct.sqlite_writes.MAX_RECORDS', 1)
    with pytest.raises(ValueError, match='count'):
        adapter.add(memory('new'))
    assert provider.calls == []


def test_upsert_batch_sql_failure_rolls_back_existing_and_new_records(runtime):
    adapter, provider = runtime
    adapter.add(memory())
    before = adapter.repository.get_record('note')
    existing, fresh = adapter.get('note'), memory('reject')
    existing.reasoning = 'Uncommitted reasoning'
    original = [deepcopy(item.to_storage_dict()) for item in (existing, fresh)]
    adapter.repository._connection.execute(
        "CREATE TRIGGER reject_new BEFORE INSERT ON records WHEN NEW.id = 'reject' BEGIN SELECT RAISE(ABORT, 'synthetic'); END")
    with pytest.raises(sqlite3.IntegrityError):
        adapter.add_batch([existing, fresh])
    assert adapter.repository.get_record('note') == before
    assert adapter.get('reject') is None
    assert not adapter.repository._connection.in_transaction
    assert [item.to_storage_dict() for item in (existing, fresh)] == original


def test_score_update_is_metadata_only_and_preserves_initial_pop_state(runtime):
    adapter, provider = runtime
    segment = memory()
    segment.popped_at = datetime(2026, 10, 7, 10)
    segment.accesses_since_pop = 2
    adapter.add(segment)
    before = adapter.repository.get_record('note')
    start = datetime(2026, 10, 7, 10)
    session_timer.tick_session(adapter.persist_dir, now=start + timedelta(minutes=2))
    provider.calls.clear()
    assert adapter.update_score('note', 60, 'Synthetic manual adjustment')
    restored = adapter.get('note')
    assert restored.initial_score == 80
    assert restored.importance_score == 60
    assert restored.reasoning == 'Synthetic manual adjustment'
    assert restored.popped_at == segment.popped_at
    assert restored.accesses_since_pop == 2
    assert elapsed_hours(restored, 'pop_anchor_at', adapter.persist_dir) == 0
    assert adapter.repository.get_record('note').embedding == before.embedding
    assert provider.calls == []
    assert not adapter.update_score('missing', 60)


@pytest.mark.parametrize('score', [True, -1, 101, 50.5, '50'])
def test_invalid_score_changes_leave_record_unchanged(runtime, score):
    adapter, _ = runtime
    adapter.add(memory())
    before = adapter.repository.get_record('note')
    with pytest.raises(ValueError):
        adapter.update_score('note', score)
    assert adapter.repository.get_record('note') == before


def test_score_sql_failure_rolls_back_anchor_changes(runtime):
    adapter, _ = runtime
    adapter.add(memory())
    before = adapter.repository.get_record('note')
    adapter.repository._connection.execute(
        "CREATE TRIGGER reject_score BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'synthetic'); END")
    with pytest.raises(sqlite3.IntegrityError):
        adapter.update_score('note', 60)
    assert adapter.repository.get_record('note') == before


def test_recycle_restore_preserve_content_vectors_pop_and_unknown_fields(runtime):
    adapter, provider = runtime
    adapter.add(memory())
    adapter.repository.patch_metadata('note', {'opaque': {'keep': True}})
    before = adapter.repository.get_record('note')
    provider.calls.clear()
    assert adapter.recycle('note', 'synthetic-owner', 'Synthetic recycle').id == 'note'
    assert adapter.is_recycled('note')
    assert adapter.get_by_score() == []
    assert adapter.search_by_identifiers(['vector recall']) == []
    assert [item.id for item in adapter.list_recycled()] == ['note']
    assert adapter.restore('note').id == 'note'
    assert not adapter.is_recycled('note')
    assert adapter.repository.get_record('note') == before
    assert adapter.list_recycled() == []
    assert provider.calls == []
    assert adapter.recycle('missing') is None
    assert adapter.restore('missing') is None


def test_observer_write_operations_refused_without_provider_calls(runtime):
    adapter, provider = runtime
    adapter.add(memory())
    provider.calls.clear()
    with SQLiteRecordStore.open_read_only(adapter.persist_dir / 'sqlite', adapter.repository.identity) as reader:
        observer = SQLiteSegmentAdapter(reader, persist_dir=adapter.persist_dir,
            document_embedder=adapter.document_embedder)
        operations = [lambda: observer.add(memory('new')), lambda: observer.add_batch([]),
            lambda: observer.update_score('note', 50), lambda: observer.recycle('note'),
            lambda: observer.restore('note')]
        for operation in operations:
            with pytest.raises(RuntimeError, match='read-only'):
                operation()
        assert observer.list_recycled() == []
    assert provider.calls == []


def test_document_binding_is_explicit_and_identity_checked(runtime):
    adapter, provider = runtime
    with pytest.raises(ValueError, match='Document embedding identity'):
        SQLiteSegmentAdapter(adapter.repository, persist_dir=adapter.persist_dir,
            document_embedder=BoundQueryEmbedder(adapter.repository.identity, provider))
    assert provider.calls == []


def test_agent_write_policy_still_protects_owner_records_during_score_suppression(runtime):
    adapter, _ = runtime
    owner = memory(score=95)
    adapter.add(owner)
    adapter.update_score('note', 79)
    assert agent_target_error(adapter.get('note')) is not None
    assert agent_write_error(93) is not None
    assert agent_write_error(92) is None


def test_runtime_session_document_binding_and_restart(runtime):
    adapter, provider = runtime
    database = adapter.persist_dir / 'sqlite'
    identity, root, embedder = adapter.repository.identity, adapter.persist_dir, adapter.document_embedder
    adapter.repository.close()
    with SQLiteRuntimeSession.open(database, identity, persist_dir=root, read_only=False,
        document_embedder=embedder) as owner:
        with owner.operation() as writer:
            assert writer.add(memory()) == 'note'
    with SQLiteRuntimeSession.open(database, identity, persist_dir=root, read_only=True) as observer:
        with observer.operation() as reader:
            assert reader.get('note').embedding == [len('Synthetic model write'), 1]
        with pytest.raises(NotImplementedError, match='no queue'):
            observer.replay_pending()


def test_invalid_model_in_batch_prevents_all_embedding_and_writes(runtime):
    adapter, provider = runtime
    first, second = memory('one'), memory('two')
    second.importance_score = 101
    with pytest.raises(ValueError):
        adapter.add_batch([first, second])
    assert provider.calls == []
    assert adapter.count() == 0
    assert first.active_time_anchors == {}


def test_failed_embedding_leaves_existing_metadata_and_new_memory_unchanged(runtime):
    adapter, provider = runtime
    adapter.add(memory())
    original = adapter.repository.get_record('note')
    changed = adapter.get('note')
    changed.reasoning = 'Uncommitted metadata'
    provider.failure = True
    with pytest.raises(RuntimeError):
        adapter.add_batch([changed, memory('new')])
    assert adapter.repository.get_record('note') == original
    assert adapter.get('new') is None


@pytest.mark.parametrize('reason', [None, 'invalid\x00reason', 'x' * 4097])
def test_invalid_score_reasoning_preserves_record(runtime, reason):
    adapter, _ = runtime
    adapter.add(memory())
    before = adapter.repository.get_record('note')
    with pytest.raises(ValueError, match='reasoning'):
        adapter.update_score('note', 60, reason)
    assert adapter.repository.get_record('note') == before


def test_bad_document_binding_releases_runtime_and_database_ownership(runtime):
    adapter, _ = runtime
    identity, root = adapter.repository.identity, adapter.persist_dir
    adapter.repository.close()
    with pytest.raises(ValueError, match='Document embedding'):
        SQLiteRuntimeSession.open(root / 'sqlite', identity, persist_dir=root,
            read_only=False, document_embedder=object())
    with SQLiteRuntimeSession.open(root / 'sqlite', identity, persist_dir=root,
        read_only=False) as owner:
        assert not owner.read_only


def test_legacy_queue_dispatch_refuses_sqlite_before_listing_files(runtime, monkeypatch):
    adapter, provider = runtime

    def forbidden_list(*args, **kwargs):
        raise AssertionError('Queue must not be inspected')

    monkeypatch.setattr(fallback_store, 'list_pending', forbidden_list)
    with pytest.raises(RuntimeError, match='no queue files'):
        fallback_store.sync_pending(adapter)
    assert adapter.count() == 0
    assert provider.calls == []


def test_legacy_queue_observer_remains_no_op_without_listing_files(runtime, monkeypatch):
    adapter, _ = runtime

    def forbidden_list(*args, **kwargs):
        raise AssertionError('Queue must not be inspected')

    monkeypatch.setattr(fallback_store, 'list_pending', forbidden_list)
    with SQLiteRecordStore.open_read_only(adapter.persist_dir / 'sqlite', adapter.repository.identity) as reader:
        observer = SQLiteSegmentAdapter(reader, persist_dir=adapter.persist_dir)
        assert fallback_store.sync_pending(observer) == {'synced': 0, 'failed': 0, 'skipped': 0}


def test_existing_ollama_document_interface_with_substituted_response(monkeypatch):
    from instinct.embedding_provider import OllamaProvider
    provider = OllamaProvider(model_name='explicit-fixture', base_url='http://127.0.0.1:11434')
    monkeypatch.setattr(provider, 'embed_documents', lambda documents: [[1, 0]])
    binding = BoundDocumentEmbedder(EmbeddingIdentity('ollama', 'explicit-fixture', 2, 'cosine'), provider)
    assert binding.embed_documents(['Synthetic local provider input']) == [(1.0, 0.0)]
