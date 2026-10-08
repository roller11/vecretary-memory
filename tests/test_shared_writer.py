# SPDX-License-Identifier: MPL-2.0
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import os
import sys
from unittest.mock import AsyncMock, Mock

import pytest

from instinct import mcp_server, nudge, shared_writer
from instinct.models import ScoredSegment, Segment, SegmentType
from instinct.seed import load_starter_memories, seed_memories
from tests.test_seed import MemoryStore


linux_only = pytest.mark.skipif(sys.platform != 'linux', reason='Private Linux socket transport')


def test_shared_mode_rejects_non_linux(monkeypatch):
    monkeypatch.setattr(shared_writer.sys, 'platform', 'win32')
    with pytest.raises(shared_writer.SharedWriterUnavailable, match='Linux/WSL'):
        shared_writer.require_linux()


@linux_only
def test_runtime_rejects_symlink_data_path(tmp_path):
    target = tmp_path / 'actual'
    target.mkdir(mode=0o700)
    link = tmp_path / 'linked'
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(PermissionError):
        shared_writer.runtime_directory(link)


@linux_only
def test_private_runtime_rejects_exposed_directory_unchanged(tmp_path):
    exposed = tmp_path / 'exposed'
    exposed.mkdir(mode=0o755)
    with pytest.raises(shared_writer.SharedWriterUnavailable, match='Unsafe'):
        shared_writer.private_directory(exposed)
    assert exposed.stat().st_mode & 0o777 == 0o755


@linux_only
def test_lock_rejects_symlink_and_hardlink(tmp_path):
    target = tmp_path / 'target'
    target.touch(mode=0o600)
    symlink = tmp_path / 'symlink'
    symlink.symlink_to(target)
    with pytest.raises(OSError):
        shared_writer.open_private_lock(symlink)
    hardlink = tmp_path / 'hardlink'
    os.link(target, hardlink)
    with pytest.raises(shared_writer.SharedWriterUnavailable):
        shared_writer.open_private_lock(hardlink)
    assert target.read_bytes() == b''


@linux_only
def test_backend_election_never_removes_existing_lock(tmp_path):
    with shared_writer.exclusive_backend_lock(tmp_path):
        inode = (tmp_path / 'backend.lock').stat().st_ino
        with pytest.raises(BlockingIOError):
            with shared_writer.exclusive_backend_lock(tmp_path):
                pytest.fail('Second writer admitted')
        assert (tmp_path / 'backend.lock').stat().st_ino == inode


def test_configuration_fingerprint_excludes_path_aliases_but_checks_consent(tmp_path, monkeypatch):
    monkeypatch.setenv('INSTINCT_DATA_DIR', 'ignored')
    before = shared_writer.configuration_fingerprint(tmp_path)
    monkeypatch.setenv('INSTINCT_DATA_DIR', 'also-ignored')
    assert shared_writer.configuration_fingerprint(tmp_path) == before
    monkeypatch.setenv('INSTINCT_ALLOW_REMOTE_PROVIDERS', '1')
    assert shared_writer.configuration_fingerprint(tmp_path) != before


def test_configuration_fingerprint_distinguishes_virtual_environments(tmp_path, monkeypatch):
    before = shared_writer.configuration_fingerprint(tmp_path)
    monkeypatch.setattr(shared_writer.sys, 'prefix', str(tmp_path / 'other-venv'))
    assert shared_writer.configuration_fingerprint(tmp_path) != before


@linux_only
@pytest.mark.asyncio
async def test_failed_startup_spawns_isolated_worker_once_without_sending_tool(tmp_path, monkeypatch):
    directory = tmp_path / 'broker'
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(shared_writer, 'runtime_directory', lambda unused: directory)
    process = Mock(poll=Mock(return_value=1))
    spawn = Mock(return_value=process)
    monkeypatch.setattr(shared_writer.subprocess, 'Popen', spawn)
    client = shared_writer.SharedWriterClient(tmp_path / 'data')
    monkeypatch.setattr(client, '_connect', AsyncMock(side_effect=FileNotFoundError))
    with pytest.raises(shared_writer.SharedWriterUnavailable, match='failed to start'):
        await client.call('instinct_save_memory', {'content': 'Synthetic'})
    assert spawn.call_count == 1
    assert spawn.call_args.args[0][1:4] == ['-I', '-m', 'instinct.shared_writer']
    assert not (tmp_path / 'data').exists()


@pytest.mark.asyncio
async def test_shared_proxy_does_not_open_store_or_run_local_ticks(monkeypatch):
    client = Mock(call=AsyncMock(return_value=[]))
    factory = Mock(side_effect=AssertionError('Proxy must not open Chroma'))
    monkeypatch.setattr(mcp_server, '_connection_client', lambda: client)
    monkeypatch.setattr(mcp_server, 'get_store', factory)
    assert await mcp_server.call_tool('instinct_get_mandatory', {'new_session': True}) == []
    client.call.assert_awaited_once_with('instinct_get_mandatory', {'new_session': True})
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_proxy_arguments_never_forward(monkeypatch):
    client = Mock(call=AsyncMock())
    monkeypatch.setattr(mcp_server, '_connection_client', lambda: client)
    result = await mcp_server.call_tool('instinct_get_context', {'query': 'synthetic', 'new_session': 1})
    assert 'nothing executed' in result[0].text
    client.call.assert_not_called()


@pytest.mark.asyncio
async def test_missing_shared_connection_cannot_fall_back_to_direct_store(monkeypatch):
    factory = Mock(side_effect=AssertionError('No direct writer allowed'))
    monkeypatch.setattr(mcp_server, '_shared_writer_enabled', True)
    monkeypatch.setattr(mcp_server, '_connection_client', lambda: None)
    monkeypatch.setattr(mcp_server, 'get_store', factory)
    result = await mcp_server.call_tool('instinct_get_mandatory', {})
    assert 'no local store was opened' in result[0].text
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_proxy_failure_is_sanitized_and_never_replays(monkeypatch):
    client = Mock(call=AsyncMock(side_effect=RuntimeError('private-note-secret')))
    monkeypatch.setattr(mcp_server, '_connection_client', lambda: client)
    result = await mcp_server.call_tool('instinct_save_memory', {
        'title': 'Synthetic', 'content': 'Synthetic', 'importance': 70,
    })
    assert 'unconfirmed' in result[0].text
    assert 'private-note-secret' not in result[0].text
    assert client.call.await_count == 1


@pytest.mark.asyncio
async def test_connection_lifespan_closes_its_proxy(monkeypatch):
    client = Mock(close=AsyncMock())
    monkeypatch.setattr(mcp_server, '_shared_writer_enabled', True)
    monkeypatch.setattr(shared_writer, 'SharedWriterClient', lambda unused: client)
    async with mcp_server._connection_lifespan(mcp_server.server) as context:
        assert context['shared_writer_client'] is client
    client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('payload', [b'[]\n', b'{invalid}\n', b'{}', b''])
async def test_invalid_frames_are_bounded(payload):
    reader = asyncio.StreamReader(limit=64)
    reader.feed_data(payload)
    reader.feed_eof()
    with pytest.raises(shared_writer.SharedWriterUnavailable):
        await shared_writer.read_frame(reader, 64)


@pytest.mark.asyncio
async def test_oversized_frame_is_rejected():
    reader = asyncio.StreamReader(limit=8)
    reader.feed_data(b'{"secret": "oversized"}\n')
    with pytest.raises(shared_writer.SharedWriterUnavailable):
        await shared_writer.read_frame(reader, 8)


@asynccontextmanager
async def socket_backend(tmp_path, monkeypatch, dispatch):
    directory = tmp_path / 'broker'
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(shared_writer, 'runtime_directory', lambda unused: directory)
    server = shared_writer.SharedWriterServer(tmp_path / 'data', dispatch)
    task = asyncio.create_task(server.run(idle_seconds=0.1))
    try:
        for unused in range(100):
            if (directory / 'writer.sock').exists():
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        assert (directory / 'writer.sock').exists()
        yield server
    finally:
        for writer in list(server.clients):
            writer.close()
        await asyncio.wait_for(task, 3)
        assert not (directory / 'writer.sock').exists()


@linux_only
@pytest.mark.asyncio
async def test_multiple_live_chats_renew_atomically_on_one_store(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / 'data')
    seed_memories(store, load_starter_memories())
    nudge.tick_nudge_session(store, now=datetime(2026, 10, 3, 10))
    ordinary = ScoredSegment(Segment('Synthetic ordinary', SegmentType.CONTEXT), importance_score=70, concept_title='Ordinary')
    store.add(ordinary)
    lifecycles = set()
    active = 0

    async def dispatch(name, arguments, lifecycle):
        nonlocal active
        lifecycles.add(lifecycle)
        active += 1
        assert active == 1
        token = mcp_server._backend_lifecycle.set(lifecycle)
        try:
            await asyncio.sleep(0.01)
            return await mcp_server.call_tool(name, arguments)
        finally:
            active -= 1
            mcp_server._backend_lifecycle.reset(token)

    async def mandatory_handler(actual, arguments):
        records = actual.get_mandatory()
        assert len(records) == 3
        content = await original_handler(actual, arguments)
        turn = nudge.get_current_turn(actual)
        for record in records:
            nudge.apply_access_drop(record, turn)
        return content

    original_handler = mcp_server._handle_get_mandatory
    monkeypatch.setattr(mcp_server, '_handle_get_mandatory', mandatory_handler)
    monkeypatch.setattr(mcp_server, 'get_store', lambda: store)
    monkeypatch.setattr(nudge, 'tick_nudge_session', lambda *args, **kwargs: None)
    monkeypatch.setattr(mcp_server.call_tool, '_last_fb_sync_check', shared_writer.time.monotonic(), raising=False)
    async with socket_backend(tmp_path, monkeypatch, dispatch) as server:
        clients = [shared_writer.SharedWriterClient(tmp_path / 'data') for unused in range(6)]
        try:
            results = await asyncio.gather(*(client.call('instinct_get_mandatory', {}) for client in clients))
            assert all('Found 3 mandatory items' in result[0].text for result in results)
            assert len(lifecycles) == 6
            assert len(server.clients) == 6
            renewed = await clients[0].call('instinct_get_mandatory', {'new_session': True})
            assert 'Found 3 mandatory items' in renewed[0].text
            assert store.get(ordinary.id).importance_score == 70
            denied = await clients[1].call('instinct_save_memory', {
                'title': 'Synthetic forbidden', 'content': 'Synthetic forbidden', 'importance': 100,
            })
            assert 'BLOCKED' in denied[0].text
            assert store.count() == 4
        finally:
            await asyncio.gather(*(client.close() for client in clients))


@linux_only
@pytest.mark.asyncio
async def test_configuration_mismatch_cannot_dispatch(tmp_path, monkeypatch):
    dispatch = AsyncMock()
    async with socket_backend(tmp_path, monkeypatch, dispatch):
        client = shared_writer.SharedWriterClient(tmp_path / 'data')
        client.profile = 'different-provider-consent'
        with pytest.raises(shared_writer.SharedWriterUnavailable, match='mismatch'):
            await client.call('instinct_get_mandatory', {})
        dispatch.assert_not_called()


@linux_only
@pytest.mark.asyncio
async def test_duplicate_backend_cannot_initialize_or_replace_socket(tmp_path, monkeypatch):
    async with socket_backend(tmp_path, monkeypatch, AsyncMock()) as first:
        inode = (first.directory / 'writer.sock').stat().st_ino
        second = shared_writer.SharedWriterServer(tmp_path / 'data', AsyncMock())
        initialize = AsyncMock()
        with pytest.raises(BlockingIOError):
            await second.run(initialize=initialize)
        initialize.assert_not_called()
        assert (first.directory / 'writer.sock').stat().st_ino == inode


@linux_only
@pytest.mark.asyncio
async def test_disconnection_after_dispatch_is_not_replayed(tmp_path, monkeypatch):
    completed = 0

    async def dispatch(name, arguments, lifecycle):
        nonlocal completed
        completed += 1
        for writer in list(server.clients):
            writer.close()
        return []

    async with socket_backend(tmp_path, monkeypatch, dispatch) as server:
        client = shared_writer.SharedWriterClient(tmp_path / 'data')
        with pytest.raises((shared_writer.SharedWriterUnavailable, ConnectionError)):
            await client.call('instinct_save_memory', {'title': 'Synthetic'})
        assert completed == 1
        assert client.writer is None


@linux_only
@pytest.mark.asyncio
async def test_unrecognized_envelope_cannot_dispatch(tmp_path, monkeypatch):
    dispatch = AsyncMock()
    async with socket_backend(tmp_path, monkeypatch, dispatch) as server:
        reader, writer = await asyncio.open_unix_connection(str(server.directory / 'writer.sock'))
        await shared_writer.write_frame(writer, {'protocol': 1, 'profile': server.profile}, 1024)
        assert (await shared_writer.read_frame(reader, 1024))['ready']
        await shared_writer.write_frame(writer, {'name': 'instinct_get_mandatory', 'arguments': {}, 'owner_override': True}, 1024)
        assert await reader.read() == b''
        writer.close()
        await writer.wait_closed()
        dispatch.assert_not_called()
