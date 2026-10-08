# SPDX-License-Identifier: MPL-2.0
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from instinct.collection_security import CollectionConfigurationError, validate_collection_model
from instinct.embedding_provider import OnnxProvider
from instinct.store import InstinctStore


def model(configuration=None, schema=None):
    return SimpleNamespace(configuration_json=configuration or {}, serialized_schema=schema)


def selected_config(provider):
    return {'type': 'known', 'name': provider.name(), 'config': provider.get_config()}


@pytest.fixture
def provider(monkeypatch):
    from chromadb.utils.embedding_functions import known_embedding_functions
    selected = OnnxProvider()
    monkeypatch.setitem(known_embedding_functions, selected.name(), type(selected))
    return selected


@pytest.mark.parametrize('settings', [None, {'type': 'legacy'}])
def test_missing_and_inert_legacy_settings_are_accepted(provider, settings):
    validate_collection_model(model({'embedding_function': settings}), provider)


def test_only_exact_selected_provider_settings_are_accepted(provider):
    settings = selected_config(provider)
    schema = {'defaults': {'float_list': {'vector_index': {'config': {'embedding_function': settings}}}}}
    validate_collection_model(model({'embedding_function': settings}, schema), provider)


@pytest.mark.parametrize('settings', [
    {'type': 'known', 'name': 'sentence_transformer', 'config': {'trust_remote_code': True}},
    {'type': 'known', 'name': 'onnx', 'config': {'provider': 'onnx', 'model_name': 'unexpected'}},
    {'type': 'known', 'name': 'onnx', 'config': {'provider': 'onnx',
     'model_name': 'all-MiniLM-L6-v2', 'kwargs': {'trust_remote_code': True}}},
    {'type': 'legacy', 'config': {'unexpected': True}},
    {'type': 'unknown'}, 'invalid', [],
])
def test_foreign_or_malformed_saved_provider_settings_are_refused(provider, settings):
    with pytest.raises(CollectionConfigurationError):
        validate_collection_model(model({'embedding_function': settings}), provider)


def test_dense_schema_hidden_provider_is_refused_before_factory_execution(provider, monkeypatch):
    from chromadb.utils.embedding_functions import known_embedding_functions
    factory = Mock()
    monkeypatch.setitem(known_embedding_functions, 'synthetic-unsafe-factory', factory)
    settings = {'type': 'known', 'name': 'synthetic-unsafe-factory', 'config': {}}
    schema = {'keys': {'#embedding': {'float_list': {'vector_index': {
        'config': {'embedding_function': settings},
    }}}}}
    with pytest.raises(CollectionConfigurationError):
        validate_collection_model(model(schema=schema), provider)
    factory.build_from_config.assert_not_called()


def test_sparse_schema_cannot_select_a_separate_factory(provider):
    schema = {'keys': {'synthetic': {'sparse_vector': {'sparse_vector_index': {
        'config': {'embedding_function': selected_config(provider)},
    }}}}}
    with pytest.raises(CollectionConfigurationError):
        validate_collection_model(model(schema=schema), provider)


def test_inert_sparse_unknown_marker_is_accepted(provider):
    schema = {'defaults': {'sparse_vector': {'sparse_vector_index': {
        'config': {'embedding_function': {'type': 'unknown'}},
    }}}}
    validate_collection_model(model(schema=schema), provider)


@pytest.mark.parametrize('host', ['localhost', '127.0.0.1'])
@pytest.mark.parametrize('discriminator', [True, False])
def test_legacy_local_ollama_configuration_is_accepted(monkeypatch, host, discriminator):
    from chromadb.utils.embedding_functions import known_embedding_functions
    from instinct.embedding_provider import OllamaProvider
    selected = OllamaProvider(base_url='http://127.0.0.1:11434')
    monkeypatch.setitem(known_embedding_functions, selected.name(), type(selected))
    settings = selected_config(selected)
    settings['config']['base_url'] = f'http://{host}:11434'
    if not discriminator:
        settings['config'].pop('provider')
    schema = {'defaults': {'float_list': {'vector_index': {'config': {'embedding_function': settings}}}}}
    validate_collection_model(model({'embedding_function': settings}, schema), selected)


@pytest.mark.parametrize('changes', [
    {'model_name': 'other-model'}, {'timeout': 999}, {'provider': 'foreign'},
    {'trust_remote_code': True}, {'kwargs': {}},
    {'base_url': 'http://localhost:11435'},
    {'base_url': 'http://localhost:11434/api'},
    {'base_url': 'http://localhost:11434?extra=1'},
    {'base_url': 'http://localhost:11434#extra'},
    {'base_url': 'http://user@localhost:11434'},
    {'base_url': 'http://remote.example:11434'},
    {'base_url': 'https://localhost:11434'},
    {'base_url': 'http://[invalid'},
])
def test_legacy_ollama_compatibility_does_not_accept_other_settings(monkeypatch, changes):
    from chromadb.utils.embedding_functions import known_embedding_functions
    from instinct.embedding_provider import OllamaProvider
    selected = OllamaProvider(base_url='http://127.0.0.1:11434')
    monkeypatch.setitem(known_embedding_functions, selected.name(), type(selected))
    settings = selected_config(selected)
    settings['config'].pop('provider')
    settings['config'].update(changes)
    with pytest.raises(CollectionConfigurationError):
        validate_collection_model(model({'embedding_function': settings}), selected)


def test_registry_class_substitution_is_refused(provider, monkeypatch):
    from chromadb.utils.embedding_functions import known_embedding_functions
    replacement = Mock()
    monkeypatch.setitem(known_embedding_functions, provider.name(), replacement)
    with pytest.raises(CollectionConfigurationError):
        validate_collection_model(model({'embedding_function': selected_config(provider)}), provider)
    replacement.build_from_config.assert_not_called()


def test_recursive_settings_are_bounded(provider):
    schema = {}
    schema['recursive'] = schema
    with pytest.raises(CollectionConfigurationError, match='safety bounds'):
        validate_collection_model(model(schema=schema), provider)


def test_oversized_settings_are_bounded(provider):
    with pytest.raises(CollectionConfigurationError, match='safety bounds'):
        validate_collection_model(model(schema={'oversized': list(range(4097))}), provider)


@pytest.mark.parametrize('settings', [[], 'invalid'])
def test_invalid_raw_schema_is_refused(provider, settings):
    with pytest.raises(CollectionConfigurationError):
        validate_collection_model(model(schema=settings), provider)


def test_real_chroma_schema_is_checked_before_any_store_reads(provider, monkeypatch):
    from chromadb.api.models.Collection import Collection
    from chromadb.types import Collection as CollectionModel
    from chromadb.utils.embedding_functions import known_embedding_functions
    factory = Mock()
    monkeypatch.setitem(known_embedding_functions, 'synthetic-unsafe-factory', factory)
    settings = {'type': 'known', 'name': 'synthetic-unsafe-factory', 'config': {}}
    schema = {'defaults': {'float_list': {'vector_index': {
        'enabled': True, 'config': {'embedding_function': settings},
    }}}}
    backend = Mock()
    raw = CollectionModel(id=uuid4(), name='synthetic-guard-test', configuration_json={},
                          serialized_schema=schema, metadata=None, dimension=None,
                          tenant='default_tenant', database='default_database')
    collection = Collection(client=backend, model=raw, embedding_function=provider)
    client = Mock()
    client.get_or_create_collection.return_value = collection
    monkeypatch.setattr('instinct.store.chromadb.Client', Mock(return_value=client))
    monkeypatch.setattr(InstinctStore, '_get_embedding_function', lambda self: provider)
    with pytest.raises(CollectionConfigurationError):
        InstinctStore()
    backend.count.assert_not_called()
    backend.get.assert_not_called()
    factory.build_from_config.assert_not_called()


def test_unguarded_chroma_write_reconstructs_saved_dense_schema_with_inert_factory(provider, monkeypatch):
    from chromadb.api.models.Collection import Collection
    from chromadb.types import Collection as CollectionModel
    from chromadb.utils.embedding_functions import known_embedding_functions
    factory = Mock()
    factory.build_from_config.return_value = provider
    monkeypatch.setitem(known_embedding_functions, 'synthetic-inert-factory', factory)
    provider._fn = Mock(return_value=[[1.0, 2.0, 3.0]])
    settings = {'type': 'known', 'name': 'synthetic-inert-factory', 'config': {}}
    schema = {'defaults': {'float_list': {'vector_index': {
        'enabled': True, 'config': {'space': 'cosine', 'embedding_function': settings},
    }}}}
    raw = CollectionModel(id=uuid4(), name='synthetic-baseline-test', configuration_json={},
                          serialized_schema=schema, metadata=None, dimension=None,
                          tenant='default_tenant', database='default_database')
    collection = Collection(client=Mock(), model=raw, embedding_function=provider)
    collection.upsert(ids=['synthetic'], documents=['synthetic note'])
    factory.build_from_config.assert_called_with({})


def test_new_real_store_and_reopen_keep_selected_provider(provider, monkeypatch, tmp_path):
    from instinct.chroma_writer_lock import release_writer_lock
    provider._fn = Mock(return_value=[[1.0, 2.0, 3.0]])
    monkeypatch.setenv('CHROMA_API_IMPL', 'synthetic_forbidden_http_implementation')
    monkeypatch.setattr(InstinctStore, '_get_embedding_function', lambda self: provider)
    target = str(tmp_path / 'synthetic-store')
    first = InstinctStore(persist_dir=target)
    try:
        from instinct.models import ScoredSegment, Segment, SegmentType
        note = ScoredSegment(segment=Segment(content='synthetic note', segment_type=SegmentType.CONTEXT),
                             importance_score=70, concept_title='Synthetic security test')
        first.add(note)
        assert first._collection.query(query_texts=['synthetic'], n_results=1)['ids'] == [[note.id]]
    finally:
        release_writer_lock(first._writer_lock_handle)
        first._writer_lock_handle = None
    reopened = InstinctStore(persist_dir=target)
    try:
        assert reopened.count() == 1
        assert reopened.get(note.id).segment.content == 'synthetic note'
    finally:
        release_writer_lock(reopened._writer_lock_handle)
        reopened._writer_lock_handle = None


def test_ephemeral_client_cannot_be_redirected_by_chroma_api_environment(provider, monkeypatch):
    monkeypatch.setenv('CHROMA_API_IMPL', 'synthetic_forbidden_http_implementation')
    client = Mock()
    collection = client.get_or_create_collection.return_value
    collection.get_model.return_value = model()
    collection.count.return_value = 0
    constructor = Mock(return_value=client)
    monkeypatch.setattr('instinct.store.chromadb.Client', constructor)
    monkeypatch.setattr(InstinctStore, '_get_embedding_function', lambda self: provider)
    InstinctStore()
    assert constructor.call_args.kwargs['settings'].chroma_api_impl == 'chromadb.api.rust.RustBindingsAPI'


def test_rejected_settings_release_only_the_lock_acquired_by_startup(provider, monkeypatch, tmp_path):
    from instinct.chroma_writer_lock import is_writer_lock_held
    client = Mock()
    collection = client.get_or_create_collection.return_value
    collection.get_model.return_value = model({'embedding_function': {'type': 'unknown'}})
    monkeypatch.setattr('instinct.store.chromadb.PersistentClient', Mock(return_value=client))
    monkeypatch.setattr(InstinctStore, '_get_embedding_function', lambda self: provider)
    target = str(tmp_path / 'synthetic-rejected-store')
    with pytest.raises(CollectionConfigurationError):
        InstinctStore(persist_dir=target)
    assert not is_writer_lock_held(target)
    collection.count.assert_not_called()
    collection.get.assert_not_called()
