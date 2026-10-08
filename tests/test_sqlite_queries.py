# SPDX-License-Identifier: MPL-2.0
"""Synthetic text and exact-identifier retrieval with timer/pop integration."""

from datetime import datetime, timedelta

import pytest

from instinct import dormancy, session_timer
from instinct.active_time import capture_anchors
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.retriever import PriorityRetriever, RetrievalConfig
from instinct.sqlite_queries import BoundQueryEmbedder
from instinct.sqlite_records import EmbeddingIdentity, SQLiteRecord
from instinct.sqlite_runtime import SQLiteSegmentAdapter
from instinct.sqlite_store import SQLiteRecordStore


class SyntheticProvider:
    def __init__(self):
        self.model = 'fixed'
        self.calls = []
        self.batch = [[1, 0]]
        self.url = None

    def name(self):
        return 'synthetic'

    def get_config(self):
        config = {'provider': self.name(), 'model_name': self.model}
        if self.url is not None:
            config['base_url'] = self.url
        return config

    def supported_spaces(self):
        return ['cosine']

    def embed_query(self, queries):
        self.calls.append(queries)
        return self.batch


@pytest.fixture
def runtime(tmp_path):
    tmp_path.chmod(0o700)
    session_timer.reset_session(tmp_path)
    session_timer.tick_session(tmp_path, now=datetime(2026, 10, 7, 10))
    identity = EmbeddingIdentity('synthetic', 'fixed', 2, 'cosine')
    provider = SyntheticProvider()
    with SQLiteRecordStore.create(tmp_path / 'sqlite', identity) as repository:
        adapter = SQLiteSegmentAdapter(repository, persist_dir=tmp_path,
                                       query_embedder=BoundQueryEmbedder(identity, provider))
        yield adapter, provider


def insert(adapter, record_id, score=80, vector=(1, 0), identifiers=None,
           segment_type=SegmentType.CONTEXT):
    memory = ScoredSegment(
        segment=Segment(content=f'Synthetic query memory {record_id}', segment_type=segment_type,
                        stored_id=record_id, timestamp=datetime(2026, 10, 7, 10)),
        importance_score=score, concept_title=record_id,
        memory_identifiers=identifiers or ['sqlite', 'timer', 'pop'])
    capture_anchors(memory, adapter.persist_dir, new=True)
    metadata = memory.to_storage_dict()
    content = metadata.pop('content')
    adapter.repository.insert_records([SQLiteRecord.from_dict({
        'id': record_id, 'content': content, 'metadata': metadata, 'embedding': list(vector),
    }, adapter.repository.identity)])
    return memory


def test_text_search_uses_explicit_embedding_and_vector_filters(runtime):
    adapter, provider = runtime
    insert(adapter, 'high', 88)
    insert(adapter, 'low', 40)
    insert(adapter, 'negative', vector=(-1, 0))
    insert(adapter, 'retired', 99)
    adapter.repository.recycle('retired')
    assert [memory.id for memory in adapter.search('sqlite timer', min_importance=50)] == ['high']
    assert provider.calls == [['sqlite timer']]
    assert [memory.id for memory in adapter.search('query', include_recycled=True)] == ['high', 'low', 'retired']
    assert adapter.search('query', segment_types=[SegmentType.CODE]) == []
    assert [memory.id for memory in adapter.search('query', min_similarity=-1)] == ['high', 'low', 'negative']


def test_direct_queries_do_not_change_clock_or_access(runtime):
    adapter, _ = runtime
    memory = insert(adapter, 'note')
    before = adapter.repository.get_record(memory.id)
    clock = session_timer.state(adapter.persist_dir)
    adapter.search('sqlite timer pop')
    adapter.search_by_identifiers(['sqlite'])
    assert adapter.repository.get_record(memory.id) == before
    assert session_timer.state(adapter.persist_dir) == clock


def test_identifiers_normalize_deduplicate_and_exclude_recycled(runtime):
    adapter, provider = runtime
    insert(adapter, 'alpha', 70)
    insert(adapter, 'beta', 88)
    insert(adapter, 'retired', 99)
    adapter.repository.recycle('retired')
    matches = adapter.search_by_identifiers(['SQLITE!!', 'timer', 'sqlite', '???'])
    assert [(memory.id, hits, words) for memory, hits, words in matches] == [
        ('beta', 2, ['sqlite', 'timer']), ('alpha', 2, ['sqlite', 'timer'])]
    assert adapter.search_by_identifiers(['sqlite'], limit=1)[0][0].id == 'alpha'
    assert adapter.search_by_identifiers([]) == []
    assert adapter.search_by_identifiers(['sqlite'], limit=0) == []
    assert provider.calls == []


def test_identifier_scan_reflects_metadata_recycle_and_restore_without_stale_index(runtime):
    adapter, _ = runtime
    insert(adapter, 'note')
    adapter.repository.patch_metadata('note', {'memory_identifiers': 'new term||exact phrase||other'})
    assert adapter.search_by_identifiers(['sqlite']) == []
    assert adapter.search_by_identifiers(['NEW-TERM'])[0][2] == ['new term']
    adapter.repository.recycle('note')
    assert adapter.search_by_identifiers(['new term']) == []
    adapter.repository.restore('note')
    assert adapter.search_by_identifiers(['new term'])[0][0].id == 'note'


@pytest.mark.parametrize('options', [
    {'limit': 0}, {'limit': True}, {'min_similarity': float('nan')},
    {'min_similarity': 2}, {'min_similarity': True}, {'min_importance': -1},
    {'segment_types': ['context']}, {'include_recycled': 1},
])
def test_invalid_search_options_rejected_before_embedding(runtime, options):
    adapter, provider = runtime
    with pytest.raises(ValueError):
        adapter.search('private query fixture', **options)
    assert provider.calls == []


@pytest.mark.parametrize('query', ['', None, 'bad\x00query', 'x' * (64 * 1024 + 1)],
                         ids=['empty', 'none', 'nul', 'oversized'])
def test_invalid_query_rejected_before_embedding(runtime, query):
    adapter, provider = runtime
    with pytest.raises(ValueError):
        adapter.search(query)
    assert provider.calls == []


@pytest.mark.parametrize('identifiers', ['sqlite', [1], ['x\x00'], ['x' * 513], ['x'] * 1001],
                         ids=['scalar', 'nontext', 'nul', 'oversized-term', 'too-many'])
def test_invalid_identifiers_rejected(runtime, identifiers):
    adapter, provider = runtime
    with pytest.raises(ValueError):
        adapter.search_by_identifiers(identifiers)
    assert provider.calls == []


@pytest.mark.parametrize('batch', [[], [[1, 0], [0, 1]], [[1]], [[1, float('nan')]],
                                   [[True, 0]], 'invalid'])
def test_invalid_provider_vectors_fail_closed(runtime, batch):
    adapter, provider = runtime
    provider.batch = batch
    with pytest.raises(ValueError):
        adapter.search('query')


def test_provider_model_drift_is_rejected_before_query(runtime):
    adapter, provider = runtime
    provider.model = 'different'
    with pytest.raises(ValueError, match='identity'):
        adapter.search('private query')
    assert provider.calls == []


def test_remote_provider_stays_disallowed_even_with_global_override(runtime, monkeypatch):
    adapter, provider = runtime
    monkeypatch.setenv('INSTINCT_ALLOW_REMOTE_PROVIDERS', '1')
    provider.url = 'https://example.com'
    with pytest.raises(ValueError, match='loopback'):
        adapter.search('private query')
    assert provider.calls == []


def test_provider_failure_does_not_fallback_or_leak_query(runtime, monkeypatch):
    adapter, provider = runtime

    def fail(queries):
        raise RuntimeError('private fixture contents')

    monkeypatch.setattr(provider, 'embed_query', fail)
    with pytest.raises(RuntimeError, match='details withheld') as failure:
        adapter.search('private fixture contents')
    assert 'private fixture contents' not in str(failure.value)


def test_provider_identity_must_match_repository(runtime):
    adapter, provider = runtime
    mismatch = BoundQueryEmbedder(EmbeddingIdentity('synthetic', 'fixed', 3, 'cosine'), provider)
    with pytest.raises(ValueError, match='identity'):
        SQLiteSegmentAdapter(adapter.repository, persist_dir=adapter.persist_dir, query_embedder=mismatch)
    assert provider.calls == []


def test_search_without_embedder_does_not_enable_automatic_provider(runtime):
    adapter, provider = runtime
    unbound = SQLiteSegmentAdapter(adapter.repository, persist_dir=adapter.persist_dir)
    with pytest.raises(RuntimeError, match='no fallback'):
        unbound.search('query')
    assert provider.calls == []


def test_complete_retrieval_persists_keyword_pop_and_access_on_active_clock(runtime):
    adapter, provider = runtime
    insert(adapter, 'note', 88)
    dormancy._ConfigStore(adapter.persist_dir).update_band('plateau', initial_drop_pts=17)
    retriever = PriorityRetriever(adapter, config=RetrievalConfig(always_include_mandatory=False))
    result = retriever.retrieve('sqlite timer pop')
    assert [memory.id for memory in result.relevant] == ['note']
    assert result.pop_results[0].effective_score_after == 100
    assert retriever.get_scoring_result('note').force_include
    stored = adapter.get('note')
    assert stored.importance_score == 71
    assert stored.initial_score == 88
    assert stored.access_count == 1
    clock_id = session_timer.state(adapter.persist_dir).clock_id
    assert stored.active_time_anchors['pop_anchor_at']['clock_id'] == clock_id
    assert stored.active_time_anchors['last_accessed']['clock_id'] == clock_id
    assert stored.embedding == [1.0, 0.0]
    assert provider.calls == [['sqlite timer pop']]


def test_identifier_candidates_rescue_semantically_excluded_memory(runtime):
    adapter, _ = runtime
    insert(adapter, 'rescued', 79, vector=(-1, 0))
    assert adapter.search('sqlite timer pop') == []
    result = PriorityRetriever(adapter, config=RetrievalConfig(always_include_mandatory=False)).retrieve(
        'sqlite timer pop', track_access=False)
    assert result.relevant[0].id == 'rescued'
    assert result.pop_results
    assert adapter.get('rescued').access_count == 0


def test_read_only_full_retrieval_only_changes_in_memory_view(runtime):
    adapter, _ = runtime
    insert(adapter, 'note', 88)
    before = adapter.repository.get_record('note')
    with SQLiteRecordStore.open_read_only(adapter.persist_dir / 'sqlite', adapter.repository.identity) as reader:
        read_adapter = SQLiteSegmentAdapter(reader, persist_dir=adapter.persist_dir,
                                           query_embedder=adapter.query_embedder)
        result = PriorityRetriever(read_adapter, config=RetrievalConfig(always_include_mandatory=False)).retrieve(
            'sqlite timer pop')
        assert result.pop_results
        assert result.relevant[0].importance_score != 88
    assert adapter.repository.get_record('note') == before


def test_query_binding_accepts_existing_local_ollama_interface_without_network(runtime, monkeypatch):
    from instinct.embedding_provider import OllamaProvider
    provider = OllamaProvider(model_name='explicit-fixture', base_url='http://127.0.0.1:11434')
    monkeypatch.setattr(provider, 'embed_query', lambda queries: [[1, 0]])
    binding = BoundQueryEmbedder(EmbeddingIdentity('ollama', 'explicit-fixture', 2, 'cosine'), provider)
    assert binding.embed_query('synthetic fixture') == (1.0, 0.0)


def test_writable_restart_preserves_clock_and_keyword_pop_frequency(runtime):
    adapter, provider = runtime
    insert(adapter, 'note', 88)
    first = PriorityRetriever(adapter, config=RetrievalConfig(always_include_mandatory=False)).retrieve('sqlite timer pop')
    assert first.pop_results[0].rule_applied == 'jump_to_plateau'
    previous = adapter.get('note')
    identity = adapter.repository.identity
    clock = session_timer.state(adapter.persist_dir)
    adapter.repository.close()
    session_timer._store_for(adapter.persist_dir).invalidate()
    with SQLiteRecordStore.open_writable(adapter.persist_dir / 'sqlite', identity) as writer:
        reopened = SQLiteSegmentAdapter(writer, persist_dir=adapter.persist_dir,
                                        query_embedder=BoundQueryEmbedder(identity, provider))
        assert reopened.get('note').to_storage_dict() == previous.to_storage_dict()
        assert session_timer.state(adapter.persist_dir).clock_id == clock.clock_id
        session_timer.tick_session(adapter.persist_dir, now=datetime(2026, 10, 7, 10) + timedelta(minutes=5))
        second = PriorityRetriever(reopened, config=RetrievalConfig(always_include_mandatory=False)).retrieve('sqlite timer')
        assert second.pop_results[0].rule_applied == 'frequency_reset'
        restored = reopened.get('note')
        assert restored.importance_score == 88
        assert restored.initial_score == 88
        assert restored.access_count == 2
        assert restored.active_time_anchors['pop_anchor_at']['seconds'] == 300
        assert restored.embedding == [1.0, 0.0]
