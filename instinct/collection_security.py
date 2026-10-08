# SPDX-License-Identifier: MPL-2.0
"""Validate inert collection settings before Chroma reconstructs model providers."""


class CollectionConfigurationError(RuntimeError):
    pass


def _matches_selected_settings(settings, expected, provider):
    if settings == expected:
        return True
    from .embedding_provider import OllamaProvider
    if type(provider) is not OllamaProvider or not isinstance(settings, dict):
        return False
    if set(settings) != {'type', 'name', 'config'} or settings['type'] != 'known' or settings['name'] != 'ollama':
        return False
    configuration = settings['config']
    selected = expected['config']
    if not isinstance(configuration, dict):
        return False
    allowed_keys = set(selected) - {'provider'}
    if set(configuration) not in (allowed_keys, set(selected)):
        return False
    if configuration.get('provider', 'ollama') != 'ollama':
        return False
    from urllib.parse import urlsplit

    def normalized(configuration):
        result = dict(configuration)
        result.pop('provider', None)
        address = result.get('base_url')
        if not isinstance(address, str):
            return None
        try:
            parsed = urlsplit(address)
            if (parsed.scheme == 'http' and parsed.hostname in {'localhost', '127.0.0.1'}
                    and parsed.username is None and parsed.password is None
                    and parsed.netloc in {'localhost:11434', '127.0.0.1:11434'}
                    and not parsed.path and not parsed.query and not parsed.fragment):
                result['base_url'] = 'http://127.0.0.1:11434'
        except ValueError:
            return None
        return result

    normalized_selected = normalized(selected)
    return normalized_selected is not None and normalized(configuration) == normalized_selected


def validate_collection_model(model, provider):
    """Only the explicitly selected provider may appear in persisted settings."""
    from chromadb.utils.embedding_functions import known_embedding_functions

    configuration = model.configuration_json
    schema = model.serialized_schema
    if not isinstance(configuration, dict) or (schema is not None and not isinstance(schema, dict)):
        raise CollectionConfigurationError('Invalid stored collection settings; storage refused.')
    expected = {'type': 'known', 'name': provider.name(), 'config': provider.get_config()}
    pending = [(configuration, 0, False), (schema, 0, False)]
    visited = 0
    while pending:
        value, depth, sparse = pending.pop()
        visited += 1
        if visited > 4096 or depth > 32:
            raise CollectionConfigurationError('Stored collection settings exceed safety bounds; storage refused.')
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'embedding_function':
                    if child is None or child == {'type': 'legacy'}:
                        continue
                    if sparse and child == {'type': 'unknown'}:
                        continue
                    if (sparse or not _matches_selected_settings(child, expected, provider)
                            or known_embedding_functions.get(provider.name()) is not type(provider)):
                        raise CollectionConfigurationError(
                            'Stored embedding settings do not match the selected provider; storage refused. '
                            'Review provider compatibility without modifying the original store.'
                        )
                else:
                    pending.append((child, depth + 1, sparse or key == 'sparse_vector_index'))
        elif isinstance(value, list):
            pending.extend((child, depth + 1, sparse) for child in value)
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise CollectionConfigurationError('Invalid stored collection settings; storage refused.')
