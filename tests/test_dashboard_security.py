# SPDX-License-Identifier: MPL-2.0
import http.client
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from instinct import dashboard


@pytest.fixture
def secured_dashboard(monkeypatch):
    reset = Mock(return_value={'ok': True})
    retriever = Mock()
    retriever.retrieve.return_value = SimpleNamespace(mandatory=[], relevant=[], pop_results=[])
    monkeypatch.setattr(dashboard, 'admin_reset', reset)
    monkeypatch.setattr(dashboard, 'get_retriever', lambda: retriever)
    monkeypatch.setattr(dashboard, 'get_store', Mock(side_effect=AssertionError('store access forbidden')))
    monkeypatch.setattr(dashboard, 'is_read_only', lambda: False)
    server = dashboard.DashboardHTTPServer(('127.0.0.1', 0), dashboard.Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    worker.start()
    try:
        yield server, reset, retriever
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()


def request(server, method='POST', route='/api/admin/reset', body=b'{}', headers=None):
    actual_headers = {
        'Host': f'127.0.0.1:{server.server_port}',
        'Content-Type': 'application/json',
        'Content-Length': str(len(body or b'')),
        'X-Instinct-Capability': server.capability_token,
    }
    actual_headers.update(headers or {})
    header_text = ''.join(f'{name}: {value}\r\n' for name, value in actual_headers.items())
    raw = f'{method} {route} HTTP/1.1\r\n{header_text}\r\n'.encode() + (body or b'')
    with socket.create_connection(('127.0.0.1', server.server_port), timeout=3) as connection:
        connection.sendall(raw)
        with http.client.HTTPResponse(connection) as response:
            response.begin()
            return response.status, response.read(), dict(response.getheaders())


@pytest.mark.parametrize('headers', [
    {'Host': 'attacker.example'},
    {'Host': '127.0.0.1:1'},
    {'Origin': 'https://attacker.example'},
    {'Origin': 'null'},
    {'Sec-Fetch-Site': 'cross-site'},
    {'Sec-Fetch-Site': 'same-site'},
    {'X-Instinct-Capability': ''},
    {'X-Instinct-Capability': 'wrong-token'},
])
def test_rejects_untrusted_requests_before_reset(secured_dashboard, headers):
    server, reset, _ = secured_dashboard
    status, body, _ = request(server, headers=headers)
    assert status == 403
    assert 'error' in json.loads(body)
    reset.assert_not_called()


@pytest.mark.parametrize('route', ['/api/admin/reset', '/api/mode/edit', '/api/heartbeat', '/api/add'])
def test_every_post_requires_capability(secured_dashboard, route):
    server, reset, _ = secured_dashboard
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
    try:
        connection.request('POST', route, headers={'Content-Type': 'application/json', 'Content-Length': '2'})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
    finally:
        connection.close()
    reset.assert_not_called()


@pytest.mark.parametrize('origin', [False, True])
def test_valid_browser_and_local_client_reset(secured_dashboard, origin):
    server, reset, _ = secured_dashboard
    headers = {'Origin': f'http://127.0.0.1:{server.server_port}'} if origin else {}
    status, body, _ = request(server, body=b'{"dry_run":true}', headers=headers)
    assert status == 200 and json.loads(body) == {'ok': True}
    reset.assert_called_once_with(include_recycle=False, dry_run=True)


def test_capability_bootstrap_and_html_are_same_origin_only(secured_dashboard):
    server, _, _ = secured_dashboard
    status, body, headers = request(server, 'GET', '/api/session', body=None)
    assert status == 200
    assert json.loads(body) == {'capability': server.capability_token}
    assert headers['Cache-Control'] == 'no-store'
    assert 'Access-Control-Allow-Origin' not in headers
    for route in ('/api/session', '/'):
        status, body, _ = request(server, 'GET', route, body=None, headers={'Origin': 'https://attacker.example'})
        assert status == 403
        assert server.capability_token.encode() not in body
    status, body, _ = request(server, 'GET', '/', body=None)
    assert status == 200
    assert server.capability_token.encode() in body
    assert body.index(b'async function dashboardFetch') < body.index(b'mode-banner')
    assert b'navigator.sendBeacon' not in body
    assert b'await fetch(' not in body


def test_capability_is_launch_specific(secured_dashboard):
    server, reset, _ = secured_dashboard
    other = dashboard.DashboardHTTPServer(('127.0.0.1', 0), dashboard.Handler)
    try:
        assert other.capability_token != server.capability_token
        status, _, _ = request(server, headers={'X-Instinct-Capability': other.capability_token})
        assert status == 403
    finally:
        other.server_close()
    reset.assert_not_called()


@pytest.mark.parametrize('scenario', [
    'recovery', 'edit_denied', 'origin_denied', 'cross_origin', 'read_only',
    'bootstrap_denied', 'invalid_capability', 'retry_denied', 'concurrent', 'preference_toggle',
    'success', 'get_denied',
])
def test_browser_capability_recovery(secured_dashboard, scenario):
    executable = shutil.which('node')
    if executable is None:
        pytest.skip('Node is required to execute the dashboard browser helper')
    server, reset, _ = secured_dashboard
    status, body, _ = request(server, 'GET', '/', body=None)
    assert status == 200
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', body.decode(), re.DOTALL)
    bootstrap = next(script for script in scripts if 'async function dashboardFetch' in script)
    harness = Path(__file__).with_name('dashboard_capability_recovery.mjs')
    result = subprocess.run(
        [executable, str(harness)], input=json.dumps({'script': bootstrap, 'scenario': scenario}),
        text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reset.assert_not_called()


@pytest.mark.parametrize(('body', 'headers', 'expected'), [
    (b'{}', {'Content-Type': 'text/plain'}, 415),
    (b'[]', {}, 400),
    (b'null', {}, 400),
    (b'"synthetic-secret"', {}, 400),
    (b'{invalid', {}, 400),
    (b'\xff', {}, 400),
    (b'{"value":NaN}', {}, 400),
    (b'{"value":Infinity}', {}, 400),
    (b'{"value":1e309}', {}, 400),
    (b'{' + b'"value":' + b'[' * 1100 + b'0' + b']' * 1100 + b'}', {}, 400),
    (b'', {}, 400),
    (b'{}', {'Content-Length': '-1'}, 400),
    (b'{}', {'Content-Length': 'not-a-number'}, 400),
    (b'{}', {'Content-Length': str(dashboard.MAX_JSON_BYTES + 1)}, 413),
    (b'{}', {'Content-Length': '9' * 100}, 413),
    (b'{}', {'Transfer-Encoding': 'chunked'}, 400),
])
def test_invalid_bodies_never_reach_reset(secured_dashboard, body, headers, expected):
    server, reset, _ = secured_dashboard
    status, response, _ = request(server, body=body, headers=headers)
    assert status == expected
    assert b'synthetic-secret' not in response
    reset.assert_not_called()


@pytest.mark.parametrize('extra_header', [
    'Host: attacker.example\r\n',
    'Content-Length: 2\r\n',
    'Content-Type: application/json\r\n',
    'X-Instinct-Capability: extra-token\r\n',
])
def test_duplicate_boundary_headers_are_rejected(secured_dashboard, extra_header):
    server, reset, _ = secured_dashboard
    raw = (
        f'POST /api/admin/reset HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n'
        f'X-Instinct-Capability: {server.capability_token}\r\n'
        'Content-Type: application/json\r\nContent-Length: 2\r\n'
        f'{extra_header}Connection: close\r\n\r\n{{}}'
    ).encode()
    with socket.create_connection(('127.0.0.1', server.server_port), timeout=3) as connection:
        connection.sendall(raw)
        response = connection.makefile('rb').read()
    assert response.startswith((b'HTTP/1.0 400', b'HTTP/1.0 403', b'HTTP/1.0 415'))
    reset.assert_not_called()


def test_stalled_body_has_deadline(secured_dashboard, monkeypatch):
    server, reset, _ = secured_dashboard
    monkeypatch.setattr(dashboard, 'JSON_READ_TIMEOUT_SECONDS', 0.15)
    raw = (
        f'POST /api/admin/reset HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n'
        f'X-Instinct-Capability: {server.capability_token}\r\n'
        'Content-Type: application/json\r\nContent-Length: 20\r\n\r\n{'
    ).encode()
    with socket.create_connection(('127.0.0.1', server.server_port), timeout=3) as connection:
        connection.sendall(raw)
        response = connection.makefile('rb').read()
    assert response.startswith(b'HTTP/1.0 408')
    reset.assert_not_called()


def test_truncated_body_is_rejected(secured_dashboard):
    server, reset, _ = secured_dashboard
    raw = (
        f'POST /api/admin/reset HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n'
        f'X-Instinct-Capability: {server.capability_token}\r\n'
        'Content-Type: application/json\r\nContent-Length: 20\r\n\r\n{}'
    ).encode()
    with socket.create_connection(('127.0.0.1', server.server_port), timeout=3) as connection:
        connection.sendall(raw)
        connection.shutdown(socket.SHUT_WR)
        response = connection.makefile('rb').read()
    assert response.startswith(b'HTTP/1.0 400')
    reset.assert_not_called()


def test_get_search_is_read_only_and_post_search_is_authorized(secured_dashboard):
    server, _, retriever = secured_dashboard
    for query in ('apply_pops=1', 'track_access=1'):
        status, _, _ = request(server, 'GET', f'/api/search_with_pop?q=synthetic&{query}', body=None)
        assert status == 405
    retriever.retrieve.assert_not_called()
    status, body, _ = request(server, 'GET', '/api/search_with_pop?q=synthetic', body=None)
    assert status == 200 and json.loads(body)['applied_pops'] is False
    retriever.retrieve.assert_called_once_with('synthetic', max_items=10, apply_pops=False, track_access=False)
    retriever.reset_mock()
    status, body, _ = request(server, route='/api/search_with_pop', body=b'{"q":"synthetic","n":20}')
    assert status == 200 and json.loads(body)['applied_pops'] is True
    retriever.retrieve.assert_called_once_with('synthetic', max_items=20, apply_pops=True, track_access=False)


def test_post_search_respects_read_only_mode(secured_dashboard, monkeypatch):
    server, _, retriever = secured_dashboard
    monkeypatch.setattr(dashboard, 'is_read_only', lambda: True)
    status, _, _ = request(server, route='/api/search_with_pop', body=b'{"q":"synthetic"}')
    assert status == 409
    retriever.retrieve.assert_not_called()


@pytest.mark.parametrize('body', [b'{"dry_run":"false"}', b'{"include_recycle":1}', b'{"dry_run":null}'])
def test_reset_flags_are_strict_booleans(secured_dashboard, body):
    server, reset, _ = secured_dashboard
    status, _, _ = request(server, body=body)
    assert status == 400
    reset.assert_not_called()


@pytest.mark.parametrize('body', [b'{"max_wait_seconds":true}', b'{"max_wait_seconds":-1}', b'{"max_wait_seconds":61}', b'{"max_wait_seconds":"60"}'])
def test_writer_handoff_is_disabled_for_sqlite(secured_dashboard, body):
    server, _, _ = secured_dashboard
    status, _, _ = request(server, route='/api/mode/auto_acquire', body=body)
    assert status == 501
