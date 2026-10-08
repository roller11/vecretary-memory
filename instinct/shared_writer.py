# SPDX-License-Identifier: MPL-2.0
"""Private Linux socket routing for multiple stdio clients and one store writer."""

import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import time


PROTOCOL_VERSION = 1
REQUEST_LIMIT = 512 * 1024
RESPONSE_LIMIT = 16 * 1024 * 1024
STARTUP_TIMEOUT = 180
CALL_TIMEOUT = 180
IDLE_SECONDS = 30
MAX_CLIENTS = 64


class SharedWriterUnavailable(RuntimeError):
    """A private backend could not be reached safely; writes are not replayed."""


def require_linux():
    if sys.platform != 'linux':
        raise SharedWriterUnavailable('Shared writer requires Linux/WSL.')


def private_directory(path):
    path.mkdir(mode=0o700, exist_ok=True)
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid() or details.st_mode & 0o077:
        raise SharedWriterUnavailable('Unsafe shared-writer directory; no permissions changed.')
    return path


def runtime_directory(data_dir):
    require_linux()
    from .private_storage import ensure_private_directory
    target = ensure_private_directory(data_dir)
    identity = hashlib.sha256(str(target.resolve()).encode()).hexdigest()[:32]
    parent = private_directory(Path('/tmp') / f'instinct-mcp-{os.getuid()}')
    return private_directory(parent / identity)


def configuration_fingerprint(data_dir):
    settings = {
        key: value for key, value in os.environ.items()
        if key.startswith(('CE_', 'INSTINCT_')) and key not in ('CE_DATA_DIR', 'INSTINCT_DATA_DIR')
    }
    settings.update(
        data_dir=str(Path(data_dir).resolve()), executable=str(Path(sys.executable).absolute()),
        environment=str(Path(sys.prefix).resolve()), protocol=PROTOCOL_VERSION,
    )
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def open_private_lock(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid() or details.st_mode & 0o077 or details.st_nlink != 1:
        os.close(descriptor)
        raise SharedWriterUnavailable('Unsafe shared-writer lock; no locks changed.')
    return descriptor


@contextmanager
def exclusive_backend_lock(directory):
    import fcntl
    descriptor = open_private_lock(directory / 'backend.lock')
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def check_socket(path):
    details = path.lstat()
    if not stat.S_ISSOCK(details.st_mode) or details.st_uid != os.getuid() or details.st_mode & 0o077:
        raise SharedWriterUnavailable('Unsafe shared-writer socket; no files changed.')


def check_peer(writer):
    connection = writer.get_extra_info('socket')
    credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i'))
    _, user_id, _ = struct.unpack('3i', credentials)
    if user_id != os.getuid():
        raise SharedWriterUnavailable('Shared-writer peer has a different owner.')


async def read_frame(reader, maximum):
    try:
        encoded = await reader.readline()
        if not encoded or len(encoded) > maximum or not encoded.endswith(b'\n'):
            raise ValueError('Invalid frame')
        frame = json.loads(encoded)
        if not isinstance(frame, dict):
            raise ValueError('Invalid object')
        return frame
    except (ValueError, UnicodeError, RecursionError) as error:
        raise SharedWriterUnavailable('Invalid or oversized shared-writer frame.') from error


async def write_frame(writer, frame, maximum):
    encoded = json.dumps(frame, ensure_ascii=True, allow_nan=False).encode() + b'\n'
    if len(encoded) > maximum:
        raise SharedWriterUnavailable('Shared-writer response exceeded its size bound; do not replay writes.')
    writer.write(encoded)
    await writer.drain()


class SharedWriterClient:
    """Keep one private connection per MCP session, without opening Chroma."""

    def __init__(self, data_dir):
        self.directory = runtime_directory(data_dir)
        self.data_dir = str(Path(data_dir).resolve())
        self.profile = configuration_fingerprint(self.data_dir)
        self.reader = None
        self.writer = None
        self._lock = asyncio.Lock()

    async def close(self):
        writer, self.writer = self.writer, None
        self.reader = None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _connect(self):
        path = self.directory / 'writer.sock'
        check_socket(path)
        reader, writer = await asyncio.open_unix_connection(str(path), limit=RESPONSE_LIMIT)
        try:
            check_peer(writer)
            await write_frame(writer, {'protocol': PROTOCOL_VERSION, 'profile': self.profile}, REQUEST_LIMIT)
            response = await asyncio.wait_for(read_frame(reader, RESPONSE_LIMIT), 10)
            if response != {'ready': True, 'protocol': PROTOCOL_VERSION}:
                raise SharedWriterUnavailable('Shared-writer configuration mismatch or backend unavailable.')
        except BaseException:
            writer.close()
            await writer.wait_closed()
            raise
        self.reader, self.writer = reader, writer

    async def _ensure_connection(self):
        import fcntl
        if self.writer is not None:
            return
        descriptor = open_private_lock(self.directory / 'startup.lock')
        acquired = False
        process = None
        deadline = time.monotonic() + STARTUP_TIMEOUT
        try:
            while time.monotonic() < deadline:
                try:
                    await self._connect()
                    return
                except (FileNotFoundError, ConnectionRefusedError):
                    pass
                if not acquired:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                    except BlockingIOError:
                        await asyncio.sleep(0.1)
                        continue
                if process is None:
                    process = subprocess.Popen(
                        [sys.executable, '-I', '-m', 'instinct.shared_writer', '--data-dir', self.data_dir],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        close_fds=True, start_new_session=True,
                    )
                if process.poll() is not None:
                    raise SharedWriterUnavailable('Shared writer failed to start. Check permissions, provider configuration and existing writer ownership; no locks were removed.')
                await asyncio.sleep(0.1)
            raise SharedWriterUnavailable('Shared-writer startup timed out; no tool was submitted.')
        finally:
            os.close(descriptor)

    async def call(self, name, arguments):
        from mcp.types import TextContent
        async with self._lock:
            try:
                await self._ensure_connection()
                await write_frame(self.writer, {'name': name, 'arguments': arguments}, REQUEST_LIMIT)
                response = await asyncio.wait_for(read_frame(self.reader, RESPONSE_LIMIT), CALL_TIMEOUT)
                if set(response) != {'content'} or not isinstance(response['content'], list):
                    raise SharedWriterUnavailable('Shared writer rejected the request.')
                return [TextContent.model_validate(item) for item in response['content']]
            except BaseException:
                await self.close()
                raise


class SharedWriterServer:
    """Serialize complete tool operations, with independent connection lifecycles."""

    def __init__(self, data_dir, dispatch):
        self.directory = runtime_directory(data_dir)
        self.profile = configuration_fingerprint(data_dir)
        self.dispatch = dispatch
        self.operation_lock = asyncio.Lock()
        self.clients = set()
        self.last_empty = time.monotonic()

    async def handle(self, reader, writer):
        from .mandatory_lifecycle import MandatoryLifecycle
        lifecycle = MandatoryLifecycle()
        if len(self.clients) >= MAX_CLIENTS:
            writer.close()
            await writer.wait_closed()
            return
        self.clients.add(writer)
        try:
            check_peer(writer)
            hello = await asyncio.wait_for(read_frame(reader, REQUEST_LIMIT), 10)
            if hello != {'protocol': PROTOCOL_VERSION, 'profile': self.profile}:
                await write_frame(writer, {'ready': False}, RESPONSE_LIMIT)
                return
            await write_frame(writer, {'ready': True, 'protocol': PROTOCOL_VERSION}, RESPONSE_LIMIT)
            while True:
                request = await read_frame(reader, REQUEST_LIMIT)
                if set(request) != {'name', 'arguments'} or not isinstance(request['name'], str) or not isinstance(request['arguments'], dict):
                    raise SharedWriterUnavailable('Invalid tool envelope.')
                async with self.operation_lock:
                    content = await self.dispatch(request['name'], request['arguments'], lifecycle)
                    response = {'content': [item.model_dump(mode='json') for item in content]}
                await asyncio.wait_for(write_frame(writer, response, RESPONSE_LIMIT), 10)
        except (SharedWriterUnavailable, ConnectionError, OSError, asyncio.TimeoutError):
            pass
        finally:
            self.clients.discard(writer)
            if not self.clients:
                self.last_empty = time.monotonic()
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def run(self, idle_seconds=IDLE_SECONDS, initialize=None):
        path = self.directory / 'writer.sock'
        with exclusive_backend_lock(self.directory):
            if initialize is not None:
                await initialize()
            if path.exists() or path.is_symlink():
                check_socket(path)
                path.unlink()
            previous_umask = os.umask(0o077)
            try:
                listener = await asyncio.start_unix_server(self.handle, str(path), limit=REQUEST_LIMIT)
            finally:
                os.umask(previous_umask)
            self.last_empty = time.monotonic()
            try:
                async with listener:
                    while self.clients or time.monotonic() - self.last_empty < idle_seconds:
                        await asyncio.sleep(min(1, idle_seconds))
            finally:
                path.unlink(missing_ok=True)


async def run_worker(data_dir):
    from . import mcp_server
    from .store import InstinctStore

    async def initialize():
        os.environ['INSTINCT_DATA_DIR'] = str(Path(data_dir).resolve())
        mcp_server._store = await asyncio.to_thread(InstinctStore, persist_dir=os.environ['INSTINCT_DATA_DIR'])

    async def dispatch(name, arguments, lifecycle):
        token = mcp_server._backend_lifecycle.set(lifecycle)
        try:
            return await mcp_server.call_tool(name, arguments)
        finally:
            mcp_server._backend_lifecycle.reset(token)

    await SharedWriterServer(data_dir, dispatch).run(initialize=initialize)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Private Instinct shared-writer worker')
    parser.add_argument('--data-dir', required=True)
    arguments = parser.parse_args()
    require_linux()
    asyncio.run(run_worker(arguments.data_dir))


if __name__ == '__main__':
    main()
