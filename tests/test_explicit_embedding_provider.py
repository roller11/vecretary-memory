# SPDX-License-Identifier: MPL-2.0
from unittest.mock import Mock

import pytest

from instinct import embedding_provider
from instinct.store import InstinctStore


def test_default_alias_uses_explicit_onnx(monkeypatch):
    monkeypatch.setattr(embedding_provider.OnnxProvider, 'is_available', lambda self: True)
    provider = embedding_provider.get_embedding_provider({'providers': ['default']})
    assert isinstance(provider, embedding_provider.DefaultProvider)
    assert isinstance(provider, embedding_provider.OnnxProvider)
    assert provider.get_config()['provider'] == 'default'
    embed = Mock(return_value=[[0.1, 0.2]])
    provider._fn = embed
    assert provider(['synthetic']) == [[0.1, 0.2]]
    embed.assert_called_once_with(['synthetic'])


@pytest.mark.parametrize('names', [['onnx'], ['default'], ['unknown']])
def test_no_provider_fails_closed(monkeypatch, names):
    monkeypatch.setattr(embedding_provider.OnnxProvider, 'is_available', lambda self: False)
    with pytest.raises(RuntimeError, match='No configured embedding provider'):
        embedding_provider.get_embedding_provider({'providers': names})


@pytest.mark.parametrize('failure', [None, RuntimeError('SYNTHETIC-PRIVATE-DETAIL')])
def test_store_refuses_missing_or_failed_provider_before_opening(monkeypatch, tmp_path, failure):
    factory = Mock(return_value=failure) if failure is None else Mock(side_effect=failure)
    monkeypatch.setattr(embedding_provider, 'get_embedding_provider', factory)
    import instinct.store as store_module
    client = Mock()
    monkeypatch.setattr(store_module.chromadb, 'PersistentClient', client)
    target = tmp_path / 'must-not-open'
    with pytest.raises(RuntimeError, match='storage setup refused') as error:
        InstinctStore(persist_dir=str(target))
    assert 'SYNTHETIC-PRIVATE-DETAIL' not in str(error.value)
    client.assert_not_called()
    assert not target.exists()


def test_chroma_embed_uses_explicit_provider_without_persisted_configuration():
    from chromadb.api.models.CollectionCommon import CollectionCommon

    class PoisonedCollection:
        @property
        def configuration(self):
            pytest.fail('Persisted embedding configuration was consulted')

    collection = PoisonedCollection()
    collection._embedding_function = embedding_provider.OnnxProvider()
    collection._embedding_function._fn = Mock(return_value=[[0.1, 0.2]])
    assert CollectionCommon._embed(collection, ['synthetic']) == [[0.1, 0.2]]


def test_sentence_transformer_explicitly_disables_remote_model_code(monkeypatch):
    from chromadb.utils import embedding_functions

    monkeypatch.delattr(embedding_functions, 'validate_embedding_function_config_is_safe', raising=False)
    constructor = Mock()
    monkeypatch.setattr(embedding_functions, 'SentenceTransformerEmbeddingFunction', constructor)
    provider = embedding_provider.SentenceTransformerProvider(model_name='synthetic-model')
    provider._ensure_loaded()
    constructor.assert_called_once_with(model_name='synthetic-model', trust_remote_code=False)


def test_sentence_transformer_uses_upstream_guard_without_forbidden_kwargs(monkeypatch):
    from chromadb.utils import embedding_functions

    validation = Mock()
    constructor = Mock()
    monkeypatch.setattr(embedding_functions, 'validate_embedding_function_config_is_safe', validation, raising=False)
    monkeypatch.setattr(embedding_functions, 'SentenceTransformerEmbeddingFunction', constructor)
    provider = embedding_provider.SentenceTransformerProvider(model_name='synthetic-model')
    provider._ensure_loaded()
    validation.assert_called_once_with('sentence_transformer', {'kwargs': {}})
    constructor.assert_called_once_with(model_name='synthetic-model')
