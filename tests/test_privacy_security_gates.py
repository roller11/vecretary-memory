# SPDX-License-Identifier: MPL-2.0
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import re
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from instinct import dashboard, embedding_provider, fallback_store, mcp_server
from instinct.provider_privacy import validated_provider_url, valid_query_embedding
from instinct.mcp_tools import get_instinct_tools
from instinct.models import ProjectIdentity, Segment, SegmentType
from instinct.scorer import ScoringConfig, SegmentScorer
from tests.test_dashboard_security import secured_dashboard, request
from tests.test_dashboard_features import feature_dashboard

CANARY = 'SYNTHETIC-PRIVATE-PROVIDER-CANARY'


@pytest.mark.parametrize('vectors', [[], [[]], [1], [[float('nan')]], [[float('inf')]], [[True]], [['1']], [[[]]], [[1], [2]], 'invalid'])
def test_rejects_invalid_embedding_shape_and_values(vectors):
    assert not valid_query_embedding(vectors)


@pytest.mark.parametrize('url', ['http://localhost:11434', 'http://127.0.0.1:11434', 'http://[::1]:11434'])
def test_loopback_provider_is_allowed_without_opt_in(monkeypatch, url):
    monkeypatch.delenv('INSTINCT_ALLOW_REMOTE_PROVIDERS', raising=False)
    assert validated_provider_url(url) == url


@pytest.mark.parametrize('url', ['http://192.168.1.50:11434', 'https://synthetic.example', 'http://127.0.0.1.evil.example', 'https://localhost.evil.example'])
def test_nonloopback_provider_requires_reviewed_opt_in(monkeypatch, url):
    monkeypatch.delenv('INSTINCT_ALLOW_REMOTE_PROVIDERS', raising=False)
    with pytest.raises(ValueError, match='ALLOW_REMOTE'):
        validated_provider_url(url)


@pytest.mark.parametrize('url', ['file:///tmp/private', 'http://user:password@localhost', 'http://localhost?secret=synthetic', 'http://localhost#secret', 'http://localhost:invalid', 'http://local host'])
def test_unsafe_provider_url_is_rejected_even_with_opt_in(monkeypatch, url):
    monkeypatch.setenv('INSTINCT_ALLOW_REMOTE_PROVIDERS', '1')
    with pytest.raises(ValueError):
        validated_provider_url(url)


def test_remote_http_requires_separate_transport_opt_in(monkeypatch):
    monkeypatch.setenv('INSTINCT_ALLOW_REMOTE_PROVIDERS', '1')
    monkeypatch.delenv('INSTINCT_ALLOW_INSECURE_REMOTE_HTTP', raising=False)
    assert validated_provider_url('https://synthetic.example') == 'https://synthetic.example'
    with pytest.raises(ValueError, match='HTTPS'):
        validated_provider_url('http://synthetic.example')
    monkeypatch.setenv('INSTINCT_ALLOW_INSECURE_REMOTE_HTTP', '1')
    assert validated_provider_url('http://synthetic.example') == 'http://synthetic.example'


def test_provider_failure_never_logs_or_returns_prompt(monkeypatch, caplog):
    post = Mock(side_effect=RuntimeError(CANARY))
    monkeypatch.setattr('httpx.post', post)
    provider = embedding_provider.OllamaProvider()
    with pytest.raises(RuntimeError) as failure:
        provider([CANARY])
    assert CANARY not in str(failure.value)
    assert all(CANARY not in record.getMessage() for record in caplog.records)
    assert post.call_args.kwargs['trust_env'] is False


@pytest.mark.parametrize('route', ['/api/list?limit=0', '/api/list?limit=1001', '/api/search?q=synthetic&n=-1', '/api/search?q=synthetic&n=101', '/api/decay/series?score=101', '/api/decay/series?days=100000', '/api/decay/series?profile=unknown', '/api/list?limit=1&limit=2', '/api/list?unknown=1'])
def test_query_bounds_reject_before_store_access(secured_dashboard, monkeypatch, route):
    server, _, retriever = secured_dashboard
    store = Mock()
    monkeypatch.setattr(dashboard, 'get_store', store)
    status, _, _ = request(server, 'GET', route, body=None)
    assert status == 400
    store.assert_not_called()
    retriever.retrieve.assert_not_called()


@pytest.mark.parametrize('payload', [{'ids': ['x'] * 101}, {'ids': [None]}, {'matched_keywords': 1.5}, {'title': {}}, {'content': 'x' * 65537}, {'q': 'x' * 8193}, {'tags': ['x' * 257]}])
def test_post_bounds_reject_before_mutations(secured_dashboard, payload):
    server, reset, _ = secured_dashboard
    status, _, _ = request(server, body=json.dumps(payload).encode())
    assert status == 400
    reset.assert_not_called()


def test_csp_nonce_and_handler_hashes_preserve_only_reviewed_scripts(secured_dashboard):
    server, _, _ = secured_dashboard
    status, body, headers = request(server, 'GET', '/', body=None)
    assert status == 200
    policy = headers['Content-Security-Policy']
    assert "default-src 'none'" in policy and "frame-ancestors 'none'" in policy
    assert "script-src-attr 'unsafe-hashes'" in policy
    assert "unsafe-eval" not in policy and "script-src 'unsafe-inline'" not in policy
    nonces = re.findall(rb'<script nonce="([^"]+)">', body)
    assert len(nonces) >= 2 and len(set(nonces)) == 1
    assert headers['X-Content-Type-Options'] == 'nosniff'
    assert headers['Referrer-Policy'] == 'no-referrer'
    _, _, other_headers = request(server, 'GET', '/', body=None)
    assert other_headers['Content-Security-Policy'] != policy


def test_worker_saturation_rejects_without_spawning(secured_dashboard):
    server, reset, _ = secured_dashboard
    for _ in range(dashboard.MAX_HTTP_WORKERS):
        assert server._workers.acquire(blocking=False)
    try:
        status, _, _ = request(server)
        assert status == 503
        reset.assert_not_called()
    finally:
        for _ in range(dashboard.MAX_HTTP_WORKERS):
            server._workers.release()


def test_dashboard_runtime_error_does_not_expose_exception(secured_dashboard, monkeypatch, capsys):
    server, _, _ = secured_dashboard
    monkeypatch.setattr(dashboard, 'get_store', Mock(side_effect=RuntimeError(CANARY)))
    status, body, _ = request(server, 'GET', '/api/stats', body=None)
    assert status == 500 and CANARY.encode() not in body
    assert CANARY not in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize('arguments', [{'query': 'x' * 8193}, {'limit': 101}, {'limit': True}, {'max_results': -1}, {'max_tokens': 1000000}, {'include_mandatory': 'true'}, {'min_similarity': float('nan')}, {'min_similarity': 10**1000}, {'content': 'x' * 65537}])
async def test_mcp_bounds_reject_before_store_or_session(monkeypatch, arguments):
    store = Mock()
    monkeypatch.setattr(mcp_server, 'get_store', store)
    result = await mcp_server.call_tool('instinct_search', arguments)
    assert 'invalid tool arguments' in result[0].text
    store.assert_not_called()


def test_queue_quota_counts_retained_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(fallback_store, 'MAX_PENDING_RECORDS', 1)
    first = fallback_store.write_pending('save_memory', str(tmp_path / 'data'), title='Synthetic first', content='Synthetic note')
    path = Path(first)
    path.write_text('invalid synthetic retained JSON')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='quota'):
        fallback_store.write_pending('save_memory', str(tmp_path / 'data'), title='Synthetic second', content='Synthetic note')
    assert path.read_bytes() == before
    assert len(list(path.parent.glob('*.json'))) == 1


def test_queue_bytes_refuse_capture_even_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(fallback_store, 'MAX_PENDING_TOTAL_BYTES', 1)
    with pytest.raises(ValueError, match='quota'):
        fallback_store.write_pending('save_memory', str(tmp_path / 'data'), title='Synthetic quota', content='Synthetic note')
    assert not list((tmp_path / 'data' / fallback_store.PENDING_DIR_NAME).glob('*.json'))


def test_concurrent_queue_captures_cannot_exceed_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(fallback_store, 'MAX_PENDING_RECORDS', 2)
    def capture(index):
        try:
            fallback_store.write_pending('save_memory', str(tmp_path / 'data'), title=f'Synthetic concurrent {index}', content='Synthetic note')
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(capture, range(6)))
    assert sum(outcomes) == 2
    assert len(list((tmp_path / 'data' / fallback_store.PENDING_DIR_NAME).glob('*.json'))) == 2


def _process_capture(directory):
    fallback_store.MAX_PENDING_RECORDS = 2
    try:
        fallback_store.write_pending('save_memory', directory, title='Synthetic process capture', content='Synthetic note')
        return True
    except ValueError:
        return False


def test_cross_process_capture_respects_aggregate_quota(tmp_path):
    with ProcessPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(_process_capture, [str(tmp_path / 'data')] * 6))
    assert sum(outcomes) == 2
    assert len(list((tmp_path / 'data' / fallback_store.PENDING_DIR_NAME).glob('*.json'))) == 2


def test_private_queue_lock_preserves_empty_sentinel(tmp_path):
    from instinct.private_storage import private_file_lock
    sentinel = tmp_path / 'data' / '.capture.lock'
    with private_file_lock(sentinel):
        assert sentinel.stat().st_size == 0
    assert sentinel.stat().st_size == 0


@pytest.mark.asyncio
async def test_published_mcp_schema_bounds_and_unknown_arguments(monkeypatch):
    schemas = {tool.name: tool.inputSchema for tool in await mcp_server.list_tools()}
    assert all(schema['additionalProperties'] is False for schema in schemas.values())
    assert schemas['instinct_save_memory']['properties']['content']['maxLength'] == 65536
    assert schemas['instinct_search']['properties']['limit']['maximum'] == 100
    store = Mock()
    monkeypatch.setattr(mcp_server, 'get_store', store)
    result = await mcp_server.call_tool('instinct_search', {'query': 'synthetic', 'unexpected': CANARY})
    assert 'invalid tool arguments' in result[0].text and CANARY not in result[0].text
    store.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('kwargs', [{'query': 'x' * 8193}, {'query': 'synthetic', 'limit': True}, {'query': 'synthetic', 'limit': 101}, {'query': 'synthetic', 'min_score': float('nan')}, {'query': 'synthetic', 'unexpected': CANARY}])
async def test_library_tools_validate_before_retriever(monkeypatch, kwargs):
    retriever = Mock()
    monkeypatch.setattr('instinct.retriever.get_retriever', retriever)
    tool = next(tool for tool in get_instinct_tools() if tool.name == 'instinct_search')
    result = await tool.execute(**kwargs)
    assert result.error_message and CANARY not in result.error_message
    retriever.assert_not_called()


@pytest.mark.asyncio
async def test_library_tools_withhold_exception_content(monkeypatch, caplog):
    monkeypatch.setattr('instinct.retriever.get_retriever', Mock(side_effect=RuntimeError(CANARY)))
    tool = next(tool for tool in get_instinct_tools() if tool.name == 'instinct_search')
    result = await tool.execute(query='synthetic')
    assert CANARY not in json.dumps(result.to_dict())
    assert CANARY not in caplog.text


def test_optional_scorer_client_uses_reviewed_host_and_no_proxy(monkeypatch):
    client = Mock()
    client.generate.return_value = {'response': '{"score":70,"title":"Synthetic","reasoning":"Synthetic"}'}
    factory = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, 'ollama', SimpleNamespace(Client=factory))
    monkeypatch.setenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
    scorer = SegmentScorer(ProjectIdentity('Synthetic', 'Synthetic mission', 'Synthetic vision'))
    assert scorer._llm_available
    scorer.score(Segment(content='Synthetic scorer note', segment_type=SegmentType.CONTEXT))
    factory.assert_called_once_with(host='http://127.0.0.1:11434', trust_env=False, follow_redirects=False)
    client.generate.assert_called_once()


def test_optional_scorer_refuses_unapproved_remote_host(monkeypatch, caplog):
    factory = Mock()
    monkeypatch.setitem(sys.modules, 'ollama', SimpleNamespace(Client=factory))
    monkeypatch.setenv('OLLAMA_HOST', 'https://synthetic.example')
    monkeypatch.delenv('INSTINCT_ALLOW_REMOTE_PROVIDERS', raising=False)
    scorer = SegmentScorer(ProjectIdentity('Synthetic', 'Synthetic mission', 'Synthetic vision'), ScoringConfig())
    assert not scorer._llm_available
    factory.assert_not_called()
    assert 'synthetic.example' not in caplog.text


def test_dashboard_response_limit_is_enforced(secured_dashboard, monkeypatch):
    server, _, _ = secured_dashboard
    monkeypatch.setattr(dashboard, 'MAX_RESPONSE_BYTES', 40)
    status, body, _ = request(server, 'GET', '/api/health', body=None)
    assert status == 413 and b'response exceeds limit' in body


def test_config_resource_bound():
    from instinct.dormancy import DormancyConfig
    with pytest.raises(ValueError, match='resource bound'):
        dashboard._validated_config_updates({'bump_decay_per_10': 1e100}, DormancyConfig())


def test_provider_constructor_rejects_credentials_before_network(monkeypatch):
    request_provider = Mock()
    monkeypatch.setattr('httpx.get', request_provider)
    with pytest.raises(ValueError):
        embedding_provider.OllamaProvider(base_url='http://user:' + CANARY + '@localhost:11434')
    request_provider.assert_not_called()


def test_provider_url_parse_failure_withholds_values():
    with pytest.raises(ValueError) as failure:
        validated_provider_url('http://localhost:' + CANARY)
    assert CANARY not in str(failure.value)


@pytest.mark.parametrize('route', ['http://synthetic.example/api/stats', '//[invalid', '/api\\stats'])
def test_invalid_request_target_is_rejected(secured_dashboard, route):
    server, _, _ = secured_dashboard
    status, _, _ = request(server, 'GET', route, body=None)
    assert status == 400


@pytest.mark.parametrize('vector', [[float('nan')], [True], ['invalid'], []])
def test_ollama_rejects_invalid_vectors(monkeypatch, vector):
    response = Mock()
    response.json.return_value = {'embedding': vector, 'embeddings': [vector]}
    monkeypatch.setattr('httpx.post', Mock(return_value=response))
    with pytest.raises(RuntimeError):
        embedding_provider.OllamaProvider()(['Synthetic vector validation'])
