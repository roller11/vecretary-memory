# SPDX-License-Identifier: MPL-2.0
import logging
import io
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import instinct
from instinct import mcp_server, nudge


CANARY = 'synthetic-private-canary-42'


@pytest.fixture
def private_logs(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='instinct-mcp')
    store = Mock(_read_only=False)
    monkeypatch.setattr(mcp_server, 'get_store', lambda: store)
    monkeypatch.setattr(nudge, 'tick_nudge_session', lambda *args, **kwargs: None)
    monkeypatch.setattr(instinct, 'tick_turn', lambda *args: None)
    monkeypatch.setattr(mcp_server.call_tool, '_last_fb_sync_check', time.monotonic(), raising=False)
    yield store, caplog
    records = [record for record in caplog.get_records('call') if record.name == 'instinct-mcp']
    assert records
    assert all(CANARY not in record.getMessage() for record in records)
    assert all(record.exc_info is None and record.stack_info is None for record in records)


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'failure', 'unavailable', 'init_exception', 'unknown'])
async def test_dispatch_logs_no_arguments_or_exception_payload(private_logs, monkeypatch, outcome):
    store, caplog = private_logs
    handler = AsyncMock(return_value=[])
    monkeypatch.setattr(mcp_server, '_handle_search', handler)
    name = 'instinct_search'
    if outcome == 'failure':
        handler.side_effect = RuntimeError(CANARY)
    elif outcome == 'unavailable':
        monkeypatch.setattr(mcp_server, 'get_store', lambda: None)
    elif outcome == 'init_exception':
        monkeypatch.setattr(mcp_server, 'get_store', Mock(side_effect=RuntimeError(CANARY)))
    elif outcome == 'unknown':
        name = CANARY
    await mcp_server.call_tool(name, {'query': CANARY, 'title': CANARY, 'reason': CANARY})


@pytest.mark.asyncio
@pytest.mark.parametrize('denial', ['owner_only', 'guard', 'dedup'])
async def test_denied_save_does_not_log_content_or_rejected_input(private_logs, monkeypatch, denial):
    store, _ = private_logs
    monkeypatch.setattr(mcp_server, '_check_write_guards', lambda *args: CANARY if denial == 'guard' else None)
    monkeypatch.setattr(mcp_server, '_check_dedup', lambda *args: CANARY if denial == 'dedup' else None)
    await mcp_server.call_tool('instinct_save_memory', {
        'content': CANARY, 'title': CANARY,
        'importance': 100 if denial == 'owner_only' else 70, 'segment_type': 'context',
    })
    store.add.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['delete', 'update'])
async def test_mutation_logs_do_not_include_memory_metadata(private_logs, operation):
    store, _ = private_logs
    from instinct.models import ScoredSegment, Segment, SegmentType
    store.get.return_value = ScoredSegment(Segment('Synthetic note.', SegmentType.CONTEXT), importance_score=70, concept_title=CANARY)
    if operation == 'delete':
        await mcp_server.call_tool('instinct_delete_memory', {'segment_id': CANARY})
        store.delete.assert_called_once_with(CANARY)
    else:
        await mcp_server.call_tool('instinct_update_score', {'segment_id': CANARY, 'new_score': 71, 'reason': CANARY})
        store.update_score.assert_called_once_with(CANARY, 71, CANARY)


def test_no_exception_traceback_logging_remains():
    from pathlib import Path
    source = Path(mcp_server.__file__).read_text(encoding='utf-8')
    assert 'logger.exception(' not in source
    assert 'exc_info=' not in source


@pytest.mark.parametrize('component', ['instinct.retriever', 'instinct.embedding_provider', 'instinct.fallback_store', 'external-library'])
def test_stderr_formatter_withholds_dependency_payloads_and_tracebacks(component):
    from instinct.privacy_logging import MCPPrivacyFormatter
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(MCPPrivacyFormatter('%(name)s %(levelname)s %(message)s'))
    try:
        raise RuntimeError(CANARY)
    except RuntimeError:
        import sys
        record = logging.LogRecord(component, logging.ERROR, CANARY, 1, 'Failure: %s', (CANARY,), sys.exc_info())
    handler.handle(record)
    output = stream.getvalue()
    assert 'runtime ERROR Runtime event' in output
    assert CANARY not in output
    assert 'Traceback' not in output


def test_stderr_formatter_keeps_safe_mcp_events():
    from instinct.privacy_logging import MCPPrivacyFormatter
    record = logging.LogRecord('instinct-mcp', logging.INFO, '', 1, 'Tool called: %s', ('instinct_search',), None)
    assert MCPPrivacyFormatter('%(message)s').format(record) == 'Tool called: instinct_search'
