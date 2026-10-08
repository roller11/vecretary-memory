# SPDX-License-Identifier: MPL-2.0
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


validation = pytest.importorskip('chromadb.utils.embedding_functions.config_validation')


@pytest.mark.parametrize('kwargs', [
    {'trust_remote_code': True}, {'trust_remote_code': False},
    {'model_kwargs': {'trust_remote_code': True}},
    {'nested': [{'trust_remote_code': True}]},
])
@pytest.mark.parametrize('name', ['sentence_transformer', 'fastembed_sparse', 'huggingface_sparse'])
def test_upstream_guard_rejects_remote_code_keys_at_any_depth(name, kwargs):
    with pytest.raises(ValueError, match='trust_remote_code'):
        validation.validate_embedding_function_config_is_safe(name, {'kwargs': kwargs})


@pytest.mark.parametrize('loader_name', [
    'load_collection_configuration_from_json', 'load_create_collection_configuration_from_json',
    'load_update_collection_configuration_from_json',
])
def test_collection_loaders_reject_before_factory_execution(monkeypatch, loader_name):
    from chromadb.api import collection_configuration
    from chromadb.utils.embedding_functions import known_embedding_functions
    factory = Mock()
    monkeypatch.setitem(known_embedding_functions, 'sentence_transformer', factory)
    config = {'embedding_function': {'type': 'known', 'name': 'sentence_transformer',
              'config': {'kwargs': {'trust_remote_code': True}}}}
    with pytest.raises(ValueError, match='trust_remote_code'):
        getattr(collection_configuration, loader_name)(config)
    factory.build_from_config.assert_not_called()


def test_generic_registry_loader_rejects_before_factory_execution(monkeypatch):
    from chromadb.utils import embedding_functions
    factory = Mock()
    monkeypatch.setitem(embedding_functions.known_embedding_functions, 'sentence_transformer', factory)
    with pytest.raises(ValueError, match='trust_remote_code'):
        embedding_functions.config_to_embedding_function({'name': 'sentence_transformer',
            'config': {'kwargs': {'trust_remote_code': True}}})
    factory.build_from_config.assert_not_called()


@pytest.mark.parametrize('sparse', [False, True])
def test_schema_reconstruction_never_invokes_an_unsafe_factory(monkeypatch, sparse):
    from chromadb.api.types import Schema
    from chromadb.utils import embedding_functions
    factory = Mock()
    name = 'huggingface_sparse' if sparse else 'sentence_transformer'
    registry = (embedding_functions.sparse_known_embedding_functions if sparse
                else embedding_functions.known_embedding_functions)
    monkeypatch.setitem(registry, name, factory)
    config = {'type': 'known', 'name': name, 'config': {'kwargs': {'trust_remote_code': True}}}
    value_type = 'sparse_vector' if sparse else 'float_list'
    index_type = 'sparse_vector_index' if sparse else 'vector_index'
    schema = {'defaults': {value_type: {index_type: {'enabled': True,
        'config': {'embedding_function': config, 'space': 'cosine'} if not sparse
                  else {'embedding_function': config}}}}}
    with pytest.warns(UserWarning, match='trust_remote_code'):
        Schema.deserialize_from_json(schema)
    factory.build_from_config.assert_not_called()


@pytest.fixture
def model_constructors(monkeypatch):
    dense, sparse, fastembed = Mock(), Mock(), Mock()
    sentence_module = ModuleType('sentence_transformers')
    sentence_module.SentenceTransformer = dense
    sentence_module.SparseEncoder = sparse
    fastembed_module = ModuleType('fastembed')
    fastembed_module.SparseTextEmbedding = fastembed
    monkeypatch.setitem(sys.modules, 'sentence_transformers', sentence_module)
    monkeypatch.setitem(sys.modules, 'fastembed', fastembed_module)
    from chromadb.utils import embedding_functions
    monkeypatch.setattr(embedding_functions.SentenceTransformerEmbeddingFunction, 'models', {})
    monkeypatch.setattr(embedding_functions.HuggingFaceSparseEmbeddingFunction, 'models', {})
    return dense, sparse, fastembed


@pytest.mark.parametrize('class_name', [
    'SentenceTransformerEmbeddingFunction', 'HuggingFaceSparseEmbeddingFunction',
    'FastembedSparseEmbeddingFunction',
])
def test_direct_constructors_reject_before_model_construction(model_constructors, class_name):
    from chromadb.utils import embedding_functions
    arguments = {'model_name': 'synthetic-model', 'trust_remote_code': True}
    if class_name != 'FastembedSparseEmbeddingFunction':
        arguments['device'] = 'cpu'
    with pytest.raises(ValueError, match='trust_remote_code'):
        getattr(embedding_functions, class_name)(**arguments)
    for constructor in model_constructors:
        constructor.assert_not_called()


def test_instinct_optional_provider_works_with_actual_patched_constructor(model_constructors):
    from instinct.embedding_provider import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model_name='synthetic-safe-model')
    provider._ensure_loaded()
    model_constructors[0].assert_called_once_with(model_name_or_path='synthetic-safe-model', device='cpu')


@pytest.mark.parametrize('name', ['sentence_transformer', 'fastembed_sparse', 'huggingface_sparse'])
def test_config_schemas_also_reject_remote_code_kwargs(name):
    from jsonschema import ValidationError
    from chromadb.utils.embedding_functions.schemas import validate_config_schema
    config = {'model_name': 'synthetic-model', 'kwargs': {'trust_remote_code': True}}
    if name != 'fastembed_sparse':
        config['device'] = 'cpu'
    if name == 'sentence_transformer':
        config['normalize_embeddings'] = False
    validate_config_schema({**config, 'kwargs': {}}, name)
    with pytest.raises(ValidationError):
        validate_config_schema(config, name)
