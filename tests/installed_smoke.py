# SPDX-License-Identifier: MPL-2.0
"""Run installed-wheel acceptance outside the checkout with synthetic data."""

import asyncio
from datetime import datetime, timedelta
import importlib.abc
from importlib.metadata import distribution, version
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DENIED_IMPORTS = ('tc_mcp', 'core', 'providers', 'wisdom', 'triage', 'instinct.cot')


class DenyPrivateImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in DENIED_IMPORTS):
            raise AssertionError('Unexpected private import: ' + fullname)


def install_import_guard():
    sys.meta_path.insert(0, DenyPrivateImports())


def starter_owner_check(mode):
    from instinct.models import RECYCLED_AT_KEY, RetrievalResult
    from instinct.nudge import tick_nudge_session, tick_turn
    from instinct.seed import load_starter_memories
    from instinct.store import InstinctStore
    from unittest.mock import patch

    store = InstinctStore(persist_dir=os.environ['INSTINCT_DATA_DIR'])
    templates = load_starter_memories()
    if mode == '--starter-owner-edit-child':
        assert store.count() == 3
        quickstart = store.get(templates[0].id)
        quickstart.concept_title = 'Owner-edited starter quickstart'
        quickstart.importance_score = 79
        store.update_metadata_only(quickstart)
        recycled = store.get(templates[2].id)
        recycled.segment.metadata[RECYCLED_AT_KEY] = '2026-10-03'
        recycled.importance_score = 79
        store.update_metadata_only(recycled)
        print(json.dumps({'owner_title_and_score_edited': True, 'recycled': True}))
    elif mode == '--starter-delete-child':
        assert store.delete(templates[1].id)
        print(json.dumps({'deleted': templates[1].id}))
    else:
        assert store.get(templates[0].id).concept_title == 'Owner-edited starter quickstart'
        assert store.get(templates[0].id).importance_score == 79
        started = datetime(2026, 10, 3, 10)
        with patch.object(store._collection, 'upsert', side_effect=AssertionError('Session restoration must not re-embed')):
            assert tick_nudge_session(store, now=started).new_session_started
            active = store.get_mandatory()
            assert len(active) == 2
            before = RetrievalResult(mandatory=active)
            tick_turn(store)
            for memory in active:
                assert store.record_access(memory.id)
                assert store.get(memory.id).importance_score == 79
            assert tick_nudge_session(store, now=started + timedelta(minutes=21)).new_session_started
            restored = store.get_mandatory()
            assert len(restored) == 2
            after = RetrievalResult(mandatory=restored)
            assert before.mandatory_fingerprint == after.mandatory_fingerprint
            assert before.validation_token != after.validation_token
            assert f'RESPONSE_REQUIREMENT: End response with 「{after.validation_token}」' in after.to_context_string()
            assert store.get(templates[2].id).importance_score == 79
        print(json.dumps({'new_session_restores_100': True, 'metadata_only': True, 'recycled_excluded': True, 'validation_token_delivered': True}))


def fallback_check(mode):
    from instinct import fallback_store, mcp_server
    from instinct.models import Segment, SegmentType
    from instinct.private_storage import write_private_text
    from instinct.seed import MARKER_NAME, load_starter_memories
    from instinct.store import InstinctStore

    data = Path(os.environ['INSTINCT_DATA_DIR'])
    assert data.parent.name.startswith('instinct-installed-') and data.name == 'data'
    assert (data / MARKER_NAME).is_file()
    store = InstinctStore(persist_dir=str(data), read_only=mode == '--fallback-capture-child')
    if mode == '--fallback-capture-child':
        result = asyncio.run(mcp_server._handle_save_memory(store, {
            'title': 'Synthetic queue outage sapphire lantern',
            'content': 'Synthetic sapphire lantern queue fixture verifies outage capture and delayed durable replay with an original timestamp.',
            'importance': 70, 'segment_type': 'context',
        }))
        assert 'FALLBACK' in result[0].text, result[0].text
        records = fallback_store.list_pending(str(data))
        assert len(records) == 1
        print(json.dumps({'read_only_handler_capture': True, 'not_claimed_persisted': True}))
        return
    records = fallback_store.list_pending(str(data))
    assert len(records) == 1
    original = records[0]
    owner = store.get(load_starter_memories()[0].id)
    before = owner.to_storage_dict()
    privileged = {key: value for key, value in original.items() if not key.startswith('_')}
    privileged['payload'] = dict(original['payload'])
    privileged['schema_version'] = 1
    privileged.pop('authority')
    privileged['payload']['importance_score'] = 100
    invalid_path = data / 'fallback_pending' / 'synthetic-legacy-privileged.json'
    write_private_text(invalid_path, json.dumps(privileged), replace=False)
    result = fallback_store.sync_pending(store)
    assert result == {'synced': 1, 'failed': 0, 'skipped': 1}, result
    segment = Segment(original['payload']['content'], SegmentType.CONTEXT, timestamp=datetime.fromisoformat(original['created_at']))
    saved = store.get(segment.id)
    assert saved is not None and saved.importance_score == 70
    assert saved.segment.timestamp == segment.timestamp
    assert invalid_path.exists()
    owner_path = Path(fallback_store.write_pending('update_score', str(data), segment_id=owner.id, new_score=70))
    result = fallback_store.sync_pending(store)
    assert result['failed'] == 1 and owner_path.exists()
    assert store.get(owner.id).to_storage_dict() == before
    fallback_store.write_pending('update_score', str(data), segment_id=saved.id, new_score=71)
    assert fallback_store.sync_pending(store)['synced'] == 1
    assert store.get(saved.id).importance_score == 71
    fallback_store.write_pending('delete_memory', str(data), delete_id=saved.id)
    assert fallback_store.sync_pending(store)['synced'] == 1
    assert store.get(saved.id) is None
    if os.name == 'posix':
        assert data.stat().st_mode & 0o777 == 0o700
        assert invalid_path.parent.stat().st_mode & 0o777 == 0o700
        assert invalid_path.stat().st_mode & 0o777 == 0o600
        assert owner_path.stat().st_mode & 0o777 == 0o600
        exposed = data.parent / 'synthetic-exposed-data'
        exposed.mkdir(mode=0o755)
        exposed.chmod(0o755)
        try:
            InstinctStore(persist_dir=str(exposed))
        except PermissionError:
            pass
        else:
            raise AssertionError('Exposed existing directory was accepted')
        assert not list(exposed.iterdir()) and exposed.stat().st_mode & 0o777 == 0o755
    invalid_path.unlink()
    owner_path.unlink()
    print(json.dumps({'queue_committed_then_updated_and_deleted': True, 'legacy_privileged_record_rejected_and_retained': True, 'owner_target_rejected_and_retained': True, 'owner_record_unchanged': True, 'original_timestamp_preserved': True, 'private_posix_modes': os.name == 'posix', 'unsafe_existing_directory_refused_unchanged': os.name == 'posix', 'only_new_synthetic_queue_files_cleaned': True}))


def request_json(base, route, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {'Content-Type': 'application/json'}
    if payload is not None:
        with urlopen(base + '/api/session', timeout=90) as response:
            headers['X-Instinct-Capability'] = json.load(response)['capability']
    request = Request(base + route, data=data, headers=headers)
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def dashboard_smoke(root, environment):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    log_path = root / 'dashboard.log'
    command = [sys.executable, str(Path(__file__).resolve()), '--dashboard-child', '--port', str(port), '--data-dir', str(root / 'data')]
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError('Dashboard exited: ' + log_path.read_text())
                try:
                    stats = request_json(base, '/api/stats')
                    break
                except (URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise AssertionError('Dashboard did not become ready: ' + log_path.read_text())
            with urlopen(base, timeout=10) as response:
                assert 'Instinct' in response.read().decode()
            boundary = dashboard_security_smoke(base)
            saved = request_json(base, '/api/add', {
                'title': 'Synthetic acceptance guardrail',
                'content': 'Confirm the target workspace and preserve unrelated user changes.',
                'score': 100, 'segment_type': 'standing_instruction',
            })
            assert saved.get('id'), saved
            fetched = request_json(base, '/api/get?id=' + saved['id'])
            assert 'Synthetic acceptance guardrail' in json.dumps(fetched), fetched
            try:
                request_json(base, '/api/cot/list')
            except HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError('Deferred transcript route was exposed')
            return {'loopback': True, 'root_html': True, 'stats': True, 'request_boundary': boundary, 'synthetic_mandatory_memory_id': saved['id'], 'deferred_cot_status': 404}
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def dashboard_security_smoke(base):
    session = request_json(base, '/api/session')
    baseline = request_json(base, '/api/stats')
    token = session['capability']
    cases = [
        ('host', '/api/session', None, {'Host': 'attacker.invalid'}, 403),
        ('origin', '/api/session', None, {'Origin': 'https://attacker.invalid'}, 403),
        ('missing_token', '/api/admin/reset', b'{"dry_run":true}', {}, 403),
        ('wrong_token', '/api/admin/reset', b'{"dry_run":true}', {'X-Instinct-Capability': 'wrong'}, 403),
        ('media_type', '/api/admin/reset', b'{"dry_run":true}', {'X-Instinct-Capability': token, 'Content-Type': 'text/plain'}, 415),
        ('array', '/api/admin/reset', b'[]', {'X-Instinct-Capability': token}, 400),
        ('invalid_json', '/api/admin/reset', b'{invalid', {'X-Instinct-Capability': token}, 400),
        ('oversized', '/api/admin/reset', b'{}', {'X-Instinct-Capability': token, 'Content-Length': str(1024 * 1024 + 1)}, 413),
        ('reset_schema', '/api/admin/reset', b'{"dry_run":"true"}', {'X-Instinct-Capability': token}, 400),
        ('get_mutation', '/api/search_with_pop?q=synthetic&apply_pops=1', None, {}, 405),
    ]
    results = {}
    for name, route, body, overrides, expected in cases:
        headers = {'Content-Type': 'application/json', **overrides}
        request = Request(base + route, data=body, headers=headers)
        try:
            with urlopen(request, timeout=10) as response:
                status = response.status
                response.read()
        except HTTPError as error:
            status = error.code
            error.read()
        assert status == expected, (name, status, expected)
        results[name] = status
    assert request_json(base, '/api/stats') == baseline
    preview = request_json(base, '/api/admin/reset', {'dry_run': True})
    assert preview['dry_run'] is True
    results['authorized_dry_run'] = True
    return results


async def mcp_smoke(root, environment):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), '--mcp-child', '--data-dir', str(root / 'data')],
        env=environment, cwd=str(root),
    )
    with (root / 'mcp.log').open('w') as log:
        async with stdio_client(parameters, errlog=log) as streams:
            async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=180)) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == 'instinct'
                tools = await session.list_tools()
                assert len(tools.tools) == 7
                mandatory = await session.call_tool('instinct_get_mandatory', {})
                mandatory_text = '\n'.join(getattr(item, 'text', '') for item in mandatory.content)
                assert 'Synthetic acceptance guardrail' in mandatory_text, mandatory_text
                assert 'VALIDATION_TOKEN:' in mandatory_text and 'RESPONSE_REQUIREMENT:' in mandatory_text
                context = await session.call_tool('instinct_get_context', {'query': 'retrieve context safely', 'include_mandatory': True})
                context_text = '\n'.join(getattr(item, 'text', '') for item in context.content)
                assert 'Owner-edited starter quickstart' in context_text, context_text
                assert 'VALIDATION_TOKEN:' in context_text, context_text
                mandatory_token = re.search(r'^VALIDATION_TOKEN: ([0-9A-F]{12})$', mandatory_text, re.MULTILINE).group(1)
                context_token = re.search(r'^VALIDATION_TOKEN: ([0-9A-F]{12})$', context_text, re.MULTILINE).group(1)
                assert mandatory_token != context_token
                assert f'End response with 「{context_token}」' in context_text
                saved = await session.call_tool('instinct_save_memory', {
                    'title': 'Synthetic MCP acceptance note',
                    'content': 'A synthetic memory verifies standalone save and semantic retrieval.',
                    'importance': 70, 'segment_type': 'context',
                })
                saved_text = '\n'.join(getattr(item, 'text', '') for item in saved.content)
                assert 'Saved memory segment' in saved_text, saved_text
                search = await session.call_tool('instinct_search', {'query': 'standalone semantic retrieval', 'min_similarity': 0, 'limit': 5})
                search_text = '\n'.join(getattr(item, 'text', '') for item in search.content)
                assert 'Synthetic MCP acceptance note' in search_text, search_text
                blocked = await session.call_tool('instinct_save_memory', {'title': 'Synthetic blocked write', 'content': 'Synthetic content', 'importance': 100})
                blocked_text = '\n'.join(getattr(item, 'text', '') for item in blocked.content)
                assert ('BLOCKED' in blocked_text and 'owner-only' in blocked_text) or ('Input validation error' in blocked_text and 'maximum' in blocked_text), blocked_text
                implicit = await session.call_tool('instinct_save_memory', {'title': 'Synthetic type bypass', 'content': 'Synthetic content', 'importance': 70, 'segment_type': 'standing_instruction'})
                implicit_text = '\n'.join(getattr(item, 'text', '') for item in implicit.content)
                assert ('BLOCKED' in implicit_text and 'owner-only' in implicit_text) or ('Input validation error' in implicit_text and 'not one of' in implicit_text), implicit_text
                log_text = (root / 'mcp.log').read_text()
                for canary in ('Synthetic MCP acceptance note', 'synthetic memory verifies', 'standalone semantic retrieval', 'Synthetic blocked write', 'Synthetic type bypass'):
                    assert canary not in log_text, 'Private canary leaked to MCP stderr'
                assert 'Tool called: instinct_save_memory' in log_text
                return {'initialized': True, 'server_name': initialized.serverInfo.name, 'tools': 7, 'mandatory_retrieval': True, 'get_context': True, 'save': True, 'semantic_search': True, 'owner_only_score_guard': True, 'mandatory_type_guard': True, 'metadata_only_stderr': True}


def main():
    install_import_guard()
    import instinct

    if len(sys.argv) > 1 and sys.argv[1].startswith('--starter-'):
        starter_owner_check(sys.argv[1])
        return

    if len(sys.argv) > 1 and sys.argv[1] in ('--fallback-capture-child', '--fallback-replay-child'):
        fallback_check(sys.argv[1])
        return

    if len(sys.argv) > 1 and sys.argv[1] in ('--mcp-child', '--dashboard-child'):
        mode = sys.argv.pop(1)
        if mode == '--mcp-child':
            from instinct.mcp_server import cli
            cli()
        else:
            from instinct.dashboard import main as dashboard_main
            dashboard_main()
        return

    installed = Path(instinct.__file__).resolve()
    assert 'site-packages' in installed.parts, 'Acceptance must use an installed wheel, not the checkout: ' + str(installed)
    wheel_files = [str(path) for path in distribution('vecretary-memory').files]
    assert not any(path.startswith(('docs/', 'tests/', 'wisdom/', 'triage/', 'tc_mcp/', 'core/', 'providers/')) for path in wheel_files)
    root = Path(tempfile.mkdtemp(prefix='instinct-installed-'))
    environment = os.environ.copy()
    environment['INSTINCT_DATA_DIR'] = str(root / 'data')
    environment['CE_DATA_DIR'] = str(root / 'data')
    environment['CE_EMBEDDING_PROVIDERS'] = 'onnx'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment.pop('PYTHONPATH', None)
    environment['HOME'] = str(root / 'home')
    (root / 'home').mkdir()
    bin_dir = Path(sys.executable).parent
    for executable in ('instinct-mcp', 'instinct-dashboard', 'instinct-seed'):
        completed = subprocess.run([str(bin_dir / executable), '--help'], cwd=root, env=environment, text=True, capture_output=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        assert '--data-dir' in completed.stdout
    results = {
        'python': sys.version.split()[0], 'module': str(installed), 'test_root': str(root),
        'versions': {name: version(name) for name in ('vecretary-memory', 'mcp', 'chromadb', 'onnxruntime')},
        'wheel_files': len(wheel_files), 'console_help': True,
    }
    try:
        seed_command = [str(bin_dir / 'instinct-seed'), '--data-dir', str(root / 'data')]
        def run_json(command):
            completed = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True, timeout=180)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            return json.loads(completed.stdout)
        preview = run_json(seed_command)
        assert len(preview['preview']) == 3 and not (root / 'data').exists()
        first = run_json(seed_command + ['--apply', '--confirm-starter-base'])
        assert len(first['created']) == 3
        child_command = [sys.executable, str(Path(__file__).resolve())]
        run_json(child_command + ['--starter-owner-edit-child'])
        repeat = run_json(seed_command + ['--apply', '--confirm-starter-base'])
        assert repeat['created'] == [] and len(repeat['preserved']) == 3
        session_check = run_json(child_command + ['--starter-session-child'])
        deleted = run_json(child_command + ['--starter-delete-child'])
        repeated_after_delete = run_json(seed_command + ['--apply', '--confirm-starter-base'])
        assert repeated_after_delete['created'] == []
        assert repeated_after_delete['preserved_missing'] == [deleted['deleted']]
        results['starter'] = {'preview_without_writes': True, 'created': 3, 'repeat_preserves_owner_edits': True, 'owner_deletion_preserved': True, **session_check}
        results['dashboard'] = dashboard_smoke(root, environment)
        captured = run_json(child_command + ['--fallback-capture-child'])
        replayed = run_json(child_command + ['--fallback-replay-child'])
        results['private_queue'] = {**captured, **replayed}
        results['mcp'] = asyncio.run(mcp_smoke(root, environment))
    finally:
        (root / 'acceptance.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
