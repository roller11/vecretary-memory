# SPDX-License-Identifier: MPL-2.0
"""Check an installed SQLite alpha with a real local model and disposable notes."""

import argparse
import asyncio
import hashlib
import http.client
from importlib.metadata import PackageNotFoundError, distribution
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlsplit


CONTENT = 'Synthetic acceptance note about lunar basalt calibration. No private data.'


def http_request(address, path, *, body=None, headers=None):
    destination = urlsplit(address)
    connection = http.client.HTTPConnection(destination.hostname, destination.port, timeout=15)
    fields = dict(headers or {})
    payload = None
    if body is not None:
        payload = json.dumps(body)
        fields['Content-Type'] = 'application/json'
    try:
        connection.request('POST' if body is not None else 'GET', path, payload, fields)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


async def tool(service, name, arguments):
    result = await service.call(name, arguments)
    return '\n'.join(item.text for item in result)


def snapshot(service, segment_id):
    from instinct import nudge, session_timer

    def inspect(store):
        segment = store.get(segment_id)
        record = store.repository.get_record(segment_id)
        return {
            'records': store.count(),
            'score': segment.importance_score,
            'access_count': segment.access_count,
            'keyword_pop_recorded': segment.last_popped_at is not None,
            'turn': nudge.get_current_turn(store),
            'active_hours': session_timer.accumulated_session_hours(store.persist_dir),
            'record_sha256': hashlib.sha256(json.dumps(record.to_dict(), sort_keys=True).encode()).hexdigest(),
        }

    return service.inspect(inspect)


async def check_stdio(root, provider_config, segment_id):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=sys.executable, args=[
        '-I', '-m', 'instinct.sqlite_service', '--database-dir', str(root / 'sqlite'),
        '--data-dir', str(root), '--identity', str(root / 'embedding-identity.json'),
        '--provider-config', str(provider_config)])
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as connection:
            await connection.initialize()
            names = {entry.name for entry in (await connection.list_tools()).tools}
            assert 'instinct_delete_memory' not in names
            read = await connection.call_tool('instinct_get', {'segment_id': segment_id})
            assert not read.isError and CONTENT in read.content[0].text
            found = await connection.call_tool('instinct_search', {
                'query': 'lunar basalt calibration', 'min_similarity': 0})
            assert not found.isError and 'Synthetic Lunar Basalt Calibration' in found.content[0].text
            saved = await connection.call_tool('instinct_save_memory', {
                'content': 'Synthetic transport acceptance note on crater depth. No private data.',
                'title': 'Synthetic Crater Depth Transport', 'importance': 70})
            assert not saved.isError and 'Saved memory segment' in saved.content[0].text
            return saved.content[0].text.split('ID: ')[1].splitlines()[0]


def run(model, dimension, base_url):
    import httpx
    import instinct
    from instinct.embedding_provider import OllamaProvider
    from instinct.sqlite_init import initialize
    from instinct.sqlite_queries import BoundQueryEmbedder
    from instinct.sqlite_records import EmbeddingIdentity
    from instinct.sqlite_service import SQLiteService
    from instinct.sqlite_store import SQLiteRecordStore
    from instinct.sqlite_writes import BoundDocumentEmbedder

    assert sys.flags.isolated, 'Run the installed checker with python -I'
    assert Path(instinct.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    try:
        distribution('chromadb')
    except PackageNotFoundError:
        pass
    else:
        raise AssertionError('Use a SQLite-only installed environment')
    identity = EmbeddingIdentity('ollama', model, dimension, 'cosine')
    provider = OllamaProvider.build_from_config({'model_name': model, 'base_url': base_url, 'timeout': 120})
    bindings = {'query_embedder': BoundQueryEmbedder(identity, provider),
                'document_embedder': BoundDocumentEmbedder(identity, provider)}
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(provider.base_url + '/api/tags')
        response.raise_for_status()
        names = {entry.get('name') for entry in response.json().get('models', [])}
        if model not in names:
            raise ValueError('The explicitly selected model is not already installed; no download performed')
        response = client.get(provider.base_url + '/api/version')
        response.raise_for_status()
        provider_version = response.json()['version']
    bindings['query_embedder'].embed_query('Synthetic dimension acceptance probe')

    with tempfile.TemporaryDirectory(prefix='vecretary-provider-acceptance-') as temporary:
        root = initialize(Path(temporary) / 'runtime', identity)
        config_path = root / 'provider-config.json'
        descriptor = os.open(config_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, 'w', encoding='ascii') as output:
            json.dump({'providers': ['ollama'], 'ollama': provider.get_config()}, output)
        with SQLiteService(root / 'sqlite', identity, persist_dir=root, read_only=False, **bindings) as service:
            saved = asyncio.run(tool(service, 'instinct_save_memory', {
                'content': CONTENT, 'title': 'Synthetic Lunar Basalt Calibration', 'importance': 70}))
            assert 'Saved memory segment' in saved, 'Synthetic save was not confirmed'
            segment_id = saved.split('ID: ')[1].splitlines()[0]
            fetched = asyncio.run(tool(service, 'instinct_get', {'segment_id': segment_id}))
            assert CONTENT in fetched
            found = asyncio.run(tool(service, 'instinct_search', {
                'query': 'lunar basalt calibration', 'min_similarity': 0}))
            assert 'Synthetic Lunar Basalt Calibration' in found
            context = asyncio.run(tool(service, 'instinct_get_context', {
                'query': 'lunar basalt calibration', 'new_session': True}))
            assert 'Synthetic Lunar Basalt Calibration' in context
            updated = asyncio.run(tool(service, 'instinct_update_score', {
                'segment_id': segment_id, 'new_score': 65}))
            assert 'Updated score' in updated
            blocked = asyncio.run(tool(service, 'instinct_update_score', {
                'segment_id': segment_id, 'new_score': 93}))
            assert 'Blocked' in blocked
            address = service.start_dashboard()
            assert service._http.server_address[0] == '127.0.0.1'
            status, listing = http_request(address, '/api/list')
            assert status == 200 and listing['count'] == 1
            status, session = http_request(address, '/api/session')
            assert status == 200 and isinstance(session['capability'], str)
            denied, _ = http_request(address, '/api/update_score', body={'id': segment_id, 'score': 66})
            assert denied == 403
            host_denied, _ = http_request(address, '/api/list', headers={'Host': 'attacker.invalid'})
            assert host_denied == 403
            origin_denied, _ = http_request(address, '/api/list', headers={'Origin': 'https://attacker.invalid'})
            assert origin_denied == 403
            token = {'X-Instinct-Capability': session['capability']}
            access_status, _ = http_request(address, '/api/access',
                body={'ids': [segment_id]}, headers=token)
            assert access_status == 200
            pop_status, popped = http_request(address, '/api/segment/pop',
                body={'id': segment_id, 'matched_keywords': 3}, headers=token)
            assert pop_status == 200 and popped['popped']
            heartbeat_status, _ = http_request(address, '/api/heartbeat', body={}, headers=token)
            assert heartbeat_status == 200
            before = snapshot(service, segment_id)
            assert before['records'] == 1 and before['access_count'] > 0
            assert before['turn'] >= 5 and before['keyword_pop_recorded']
            assert before['active_hours'] > 0
        with SQLiteService(root / 'sqlite', identity, persist_dir=root, read_only=False, **bindings) as service:
            after = snapshot(service, segment_id)
            assert after == before, 'Record/timer/access/pop state changed on idle owner restart'
            results = service.inspect(lambda store: store.search('lunar basalt calibration', min_similarity=0))
            assert any(item.id == segment_id for item in results)
        transport_id = asyncio.run(asyncio.wait_for(check_stdio(root, config_path, segment_id), timeout=180))
        with SQLiteRecordStore.open_read_only(root / 'sqlite', identity) as repository:
            assert repository.count() == 2
            assert repository.get_record(transport_id).content == 'Synthetic transport acceptance note on crater depth. No private data.'

    return {'status': 'PASS', 'package_version': distribution('vecretary-memory').version,
            'identity': identity.to_dict(), 'provider_version': provider_version,
            'real_local_embeddings': True, 'synthetic_records_only': True,
            'mcp_handler_save_read_search_update': True, 'protected_score_refused': True,
            'real_stdio_cli_read_search_save': True,
            'dashboard_loopback': True, 'capability_mutations_checked': True,
            'cross_origin_and_host_refused': True, 'unauthenticated_local_reads': True,
            'unauthenticated_local_token_bootstrap': True, 'login_implemented': False,
            'https_implemented': False, 'state_preserved_on_restart': True,
            'keyword_pop_and_access_persisted': True, 'active_time_persisted': True,
            'snapshot': after,
            'limits': ['Actual service handlers, not a new observed agent/chat/reload acceptance',
                       'No crash-recovery, production-provider certification or live-store migration',
                       'Temporary-pop rule triggering remains separately covered by synthetic regressions']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dimension', type=int, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.model, arguments.dimension, arguments.base_url)
    with arguments.output.open('x', encoding='ascii') as output:
        json.dump(report, output, indent=2, ensure_ascii=True)
    print(json.dumps({'status': report['status'], 'synthetic_records_only': True,
                      'real_local_embeddings': True, 'state_preserved_on_restart': True}))


if __name__ == '__main__':
    main()
