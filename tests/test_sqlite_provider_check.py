# SPDX-License-Identifier: MPL-2.0
"""Offline safety checks for the explicit real-provider acceptance helper."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('sqlite_provider_check',
    Path(__file__).resolve().parents[1] / 'tools/check_sqlite_provider.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_http_checker_closes_connection_on_success(monkeypatch):
    calls = []
    connection = SimpleNamespace(
        request=lambda *args: calls.append(args),
        getresponse=lambda: SimpleNamespace(status=200, read=lambda: b'{"ok":true}'),
        close=lambda: calls.append('closed'))
    monkeypatch.setattr(check.http.client, 'HTTPConnection', lambda *args, **kwargs: connection)
    headers = {'X-Instinct-Capability': 'synthetic'}
    assert check.http_request('http://127.0.0.1:8767', '/api/heartbeat', body={}, headers=headers) == (200, {'ok': True})
    assert calls[0][0] == 'POST'
    assert calls[0][3]['Content-Type'] == 'application/json'
    assert headers == {'X-Instinct-Capability': 'synthetic'}
    assert calls[-1] == 'closed'


def test_http_checker_closes_connection_on_failure(monkeypatch):
    closed = []
    connection = SimpleNamespace(request=lambda *args: None,
        getresponse=lambda: SimpleNamespace(status=200, read=lambda: b'invalid-json'),
        close=lambda: closed.append(True))
    monkeypatch.setattr(check.http.client, 'HTTPConnection', lambda *args, **kwargs: connection)
    with pytest.raises(ValueError):
        check.http_request('http://127.0.0.1:8767', '/api/list')
    assert closed == [True]


def test_snapshot_uses_owner_inspection_and_omits_memory_content(monkeypatch):
    from instinct import nudge, session_timer

    monkeypatch.setattr(nudge, 'get_current_turn', lambda store: 6)
    monkeypatch.setattr(session_timer, 'accumulated_session_hours', lambda root: 0.1)
    record = SimpleNamespace(to_dict=lambda: {'content': 'Synthetic private-ish text', 'embedding': [1, 0]})
    segment = SimpleNamespace(importance_score=60, access_count=2, last_popped_at=True)
    store = SimpleNamespace(get=lambda record_id: segment, count=lambda: 1, persist_dir='synthetic',
        repository=SimpleNamespace(get_record=lambda record_id: record))
    service = SimpleNamespace(inspect=lambda operation: operation(store))
    snapshot = check.snapshot(service, 'synthetic-id')
    assert snapshot['turn'] == 6 and snapshot['keyword_pop_recorded']
    assert len(snapshot['record_sha256']) == 64
    assert 'Synthetic private-ish text' not in str(snapshot)
    assert 'embedding' not in snapshot
