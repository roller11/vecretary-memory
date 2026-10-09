# SPDX-License-Identifier: MPL-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from instinct import embedding_provider


def test_provider_registry_has_no_backend_owned_embedding_functions():
    assert set(embedding_provider._PROVIDERS) == {'ollama', 'sentence_transformers'}


def test_provider_selection_is_explicit_and_stable(monkeypatch):
    monkeypatch.setattr(embedding_provider.OllamaProvider, 'is_available', lambda self: True)
    provider = embedding_provider.get_embedding_provider({'providers': ['ollama']})
    assert isinstance(provider, embedding_provider.OllamaProvider)
    assert provider.get_config()['provider'] == 'ollama'


@pytest.mark.parametrize('names', [['unknown'], ['onnx'], []])
def test_unavailable_or_unsupported_provider_fails_closed(monkeypatch, names):
    monkeypatch.setattr(embedding_provider.OllamaProvider, 'is_available', lambda self: False)
    monkeypatch.setattr(embedding_provider.SentenceTransformerProvider, 'is_available', lambda self: False)
    with pytest.raises(RuntimeError, match='No configured embedding provider'):
        embedding_provider.get_embedding_provider({'providers': names})


def test_sentence_transformer_provider_disables_remote_code(monkeypatch):
    encoded = SimpleNamespace(tolist=lambda: [[0.1, 0.2]])
    model = SimpleNamespace(encode=Mock(return_value=encoded))
    constructor = Mock(return_value=model)
    monkeypatch.setitem(__import__('sys').modules, 'sentence_transformers',
                        SimpleNamespace(SentenceTransformer=constructor))
    provider = embedding_provider.SentenceTransformerProvider('synthetic-model')

    assert provider(['synthetic']) == [[0.1, 0.2]]
    constructor.assert_called_once_with('synthetic-model', trust_remote_code=False)
    model.encode.assert_called_once_with(['synthetic'], convert_to_numpy=True)
