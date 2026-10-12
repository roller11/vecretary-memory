# SPDX-License-Identifier: MPL-2.0
"""Opt-in isolated SQLite stdio MCP and loopback dashboard integration."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import os
import socket
import sqlite3
import threading
import time
from urllib.parse import urlparse

from . import dashboard, mcp_server
from .mandatory_lifecycle import MandatoryLifecycle
from .service_binding import bind_store
from .sqlite_session import SQLiteRuntimeSession


GET_ROUTES = frozenset({
    '/', '/index.html', '/favicon.ico', '/assets/vecretary-logo.png', '/assets/vecretary-logo-256.png', '/modifier', '/decay', '/decay.html',
    '/score', '/score.html', '/api/session', '/api/ready', '/api/mode', '/api/health', '/api/stats',
    '/api/list', '/api/recycled', '/api/mandatory', '/api/get', '/api/turn',
    '/api/search', '/api/search_with_pop', '/api/segment/kpis',
    '/api/segment/modifier_window', '/api/decay/config', '/api/decay/series',
    '/api/decay/preview', '/api/nudge/summary', '/api/nudge/list',
})
POST_ROUTES = frozenset({
    '/api/ready', '/api/heartbeat', '/api/add', '/api/update', '/api/update_score',
    '/api/access', '/api/search_with_pop', '/api/segment/pop', '/api/recycle',
    '/api/restore', '/api/decay/config', '/api/decay/tick', '/api/score/recommend',
    '/api/score/ephemeral', '/api/nudge/run', '/api/nudge/interval',
    '/api/mode/take_control', '/api/mode/return_control',
})

DASHBOARD_HEARTBEAT_TIMEOUT_S = 30
DASHBOARD_WATCHDOG_INTERVAL_S = 2


class SQLiteService:
    """Serialize complete service requests on one connection-owning thread.

    Only existing isolated databases are accepted. Agent permanent deletion,
    legacy replay and backup/reset remain fail-closed. Dashboard write control
    pauses MCP tools while the shared service grants dashboard mutations.
    """

    runtime_ready = False

    def __init__(self, database_dir, identity, *, persist_dir, read_only,
                 query_embedder=None, document_embedder=None, queue_writes=False, replay_pending=False,
                 enable_purge=False):
        if type(enable_purge) is not bool or enable_purge and read_only:
            raise ValueError('Purge requires an explicit owner-only boolean policy')
        if type(queue_writes) is not bool or type(replay_pending) is not bool:
            raise ValueError('SQLite queue policies must be explicit booleans')
        if queue_writes and not read_only or replay_pending and read_only:
            raise ValueError('Queue capture requires an observer; replay requires an owner')
        self._capture_enabled = queue_writes
        self._replay_enabled = replay_pending
        self._purge_enabled = enable_purge
        self._session_options = {
            'database_dir': database_dir,
            'identity': identity,
            'persist_dir': persist_dir,
            'query_embedder': query_embedder,
            'document_embedder': document_embedder,
        }
        self.last_replay = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sqlite-owner')
        self._closed = False
        self._http = None
        self._http_thread = None
        self._lifecycle = MandatoryLifecycle()
        self._state_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(16)
        self._mode = 'observer' if read_only else 'dashboard'
        self._stdio_active = False
        self._stdio_previous_mode = self._mode
        self._dashboard_last_heartbeat = None
        self._watchdog_stop = threading.Event()
        try:
            self._session = self._executor.submit(
                SQLiteRuntimeSession.open, database_dir, identity,
                persist_dir=persist_dir, read_only=read_only,
                query_embedder=query_embedder, document_embedder=document_embedder,
            ).result()
            self.inspect(self._configure_queue)
            self._watchdog_thread = threading.Thread(
                target=self._watch_dashboard_lease,
                name='sqlite-dashboard-lease', daemon=True)
            self._watchdog_thread.start()
        except BaseException:
            try:
                if getattr(self, '_session', None) is not None:
                    self._executor.submit(self._session.close).result()
            finally:
                self._executor.shutdown(wait=True)
            raise

    def _configure_queue(self, store):
        store.queue_writes = self._capture_enabled
        return self._replay_store(store)

    def _replay_store(self, store):
        if not self._replay_enabled:
            return False
        from .sqlite_pending import replay
        try:
            self.last_replay = replay(store)
        except Exception:
            self.last_replay = {'synced': 0, 'failed': 1, 'skipped': 0, 'pending': True}
        return self.last_replay['pending']

    def _execute(self, operation):
        with self._session.operation() as store, bind_store(store):
            return operation(store)

    def _submit(self, operation):
        if not self._slots.acquire(blocking=False):
            raise RuntimeError('SQLite service request capacity exceeded')
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError('SQLite service is closed')
                future = self._executor.submit(self._execute, operation)
            future.add_done_callback(lambda completed: self._slots.release())
            return future
        except BaseException:
            self._slots.release()
            raise

    async def call(self, name, arguments, *, lifecycle=None):
        connection = lifecycle if lifecycle is not None else self._lifecycle

        def dispatch(store):
            with self._state_lock:
                mcp_paused = self._stdio_active and self._mode != 'mcp'
            if mcp_paused:
                from mcp import types
                return [types.TextContent(
                    type='text',
                    text='MCP tool access is paused while the dashboard owns write control. Return control to MCP before retrying.',
                )]
            pending = self._replay_store(store)
            if pending and name in ('instinct_save_memory', 'instinct_update_score', 'instinct_delete_memory'):
                from mcp import types
                return [types.TextContent(type='text', text='Error: SQLite pending writes have not drained; no new database write performed. Check provider, queue and replay status before retrying.')]
            token = mcp_server._backend_lifecycle.set(connection)
            try:
                return asyncio.run(mcp_server.call_tool(name, arguments))
            finally:
                mcp_server._backend_lifecycle.reset(token)

        future = self._submit(dispatch)
        return await asyncio.shield(asyncio.wrap_future(future))

    @property
    def dashboard_control(self):
        with self._state_lock:
            return self._mode == 'dashboard'

    def take_dashboard_control(self):
        with self._state_lock:
            if self._mode == 'dashboard':
                return {'acquired': True, 'mode': 'dashboard',
                        'message': 'Dashboard already owns write control.'}
            if not self._stdio_active or self._mode != 'mcp':
                return {'acquired': False, 'mode': self._mode,
                        'message': 'A connected MCP service must own control before dashboard handoff.'}
            if self._capture_enabled:
                return {'acquired': False, 'mode': self._mode,
                        'message': 'Dashboard handoff is unavailable while observer queue capture is enabled.'}
            self._mode = 'taking_control'
            self._dashboard_last_heartbeat = None

        if self._session.read_only:
            try:
                next_session = SQLiteRuntimeSession.open(
                    **self._session_options, read_only=False)
            except Exception:
                with self._state_lock:
                    self._mode = 'mcp' if self._stdio_active else self._stdio_previous_mode
                return {'acquired': False, 'mode': self._mode,
                        'message': 'Could not acquire SQLite write ownership. Another owner may still hold it; no lock was removed.'}
            current_session = self._session
            try:
                current_session.close_for_handoff()
            except Exception:
                next_session.close()
                with self._state_lock:
                    self._mode = 'mcp' if self._stdio_active else self._stdio_previous_mode
                return {'acquired': False, 'mode': self._mode,
                        'message': 'Could not safely complete the SQLite handoff; the previous session remains active.'}
            self._session = next_session
            next_session._adapter.queue_writes = False

        with self._state_lock:
            self._mode = 'dashboard'
            self._dashboard_last_heartbeat = time.monotonic()
        return {'acquired': True, 'mode': 'dashboard', 'mcp_paused': True,
                'holder_pid': os.getpid(),
                'heartbeat_timeout_s': DASHBOARD_HEARTBEAT_TIMEOUT_S,
                'message': 'Dashboard now owns write control. MCP tools are paused until control is returned.'}

    def return_control_to_mcp(self):
        with self._state_lock:
            if self._mode != 'dashboard':
                return {'released': False, 'mode': self._mode,
                        'message': 'Dashboard does not currently own write control.'}
            if not self._stdio_active:
                return {'released': False, 'mode': self._mode,
                        'message': 'No connected MCP service is available to receive write control.'}
            self._mode = 'returning_to_mcp'
            self._dashboard_last_heartbeat = None
            self._mode = 'mcp'
        return {'released': True, 'mode': 'mcp', 'mcp_resumed': True,
                'message': 'Write control returned to the connected MCP service.'}

    def heartbeat_dashboard(self):
        with self._state_lock:
            if self._mode != 'dashboard':
                return False
            self._dashboard_last_heartbeat = time.monotonic()
            return True

    def _watch_dashboard_lease(self):
        while not self._watchdog_stop.wait(DASHBOARD_WATCHDOG_INTERVAL_S):
            self._expire_dashboard_lease()

    def _expire_dashboard_lease(self, *, now=None):
        current_time = time.monotonic() if now is None else now
        with self._state_lock:
            expired = (
                self._mode == 'dashboard'
                and self._dashboard_last_heartbeat is not None
                and current_time - self._dashboard_last_heartbeat >= DASHBOARD_HEARTBEAT_TIMEOUT_S
            )
            if not expired:
                return False
            self._mode = 'mcp' if self._stdio_active else self._stdio_previous_mode
            self._dashboard_last_heartbeat = None
            return True

    def inspect(self, operation):
        """Trusted internal diagnostic callback; not exposed as an agent tool."""
        return self._submit(operation).result()

    def start_dashboard(self, port=0):
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError('Invalid loopback dashboard port')
        with self._state_lock:
            if self._closed or self._http is not None:
                raise RuntimeError('Dashboard cannot be started in this service state')
            self._http = SQLiteDashboardServer(('127.0.0.1', port), SQLiteDashboardHandler)
            self._http.service = self
            self._http_thread = threading.Thread(
                target=self._http.serve_forever, name='sqlite-dashboard', daemon=True)
            self._http_thread.start()
            return f'http://127.0.0.1:{self._http.server_port}'

    async def run_stdio(self):
        from mcp.server import Server
        from mcp.server.stdio import stdio_server

        @asynccontextmanager
        async def lifespan(server):
            yield {'mandatory_lifecycle': MandatoryLifecycle()}

        server = Server('vecretary-sqlite-experimental', lifespan=lifespan)

        @server.list_tools()
        async def tools():
            return await mcp_server.list_tools()

        @server.call_tool()
        async def dispatch(name, arguments):
            lifecycle = server.request_context.lifespan_context['mandatory_lifecycle']
            return await self.call(name, arguments, lifecycle=lifecycle)

        with self._state_lock:
            previous_mode = self._mode
            self._stdio_previous_mode = previous_mode
            self._stdio_active = True
            self._mode = 'mcp'
        try:
            async with stdio_server() as (reader, writer):
                await server.run(reader, writer, server.create_initialization_options())
        finally:
            with self._state_lock:
                self._stdio_active = False
                if self._mode == 'mcp':
                    self._mode = previous_mode

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._watchdog_stop.set()
        watchdog = getattr(self, '_watchdog_thread', None)
        if watchdog is not None:
            watchdog.join(timeout=3)
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http_thread.join()
        try:
            self._executor.submit(self._session.close).result()
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SQLiteDashboardServer(dashboard.DashboardHTTPServer):
    def __init__(self, server_address, handler_class):
        if not isinstance(server_address, tuple) or len(server_address) < 2 or server_address[0] != '127.0.0.1':
            raise ValueError('SQLite dashboard must bind to 127.0.0.1 only')
        super().__init__(server_address, handler_class)

    def shutdown_request(self, request):
        try:
            request.shutdown(socket.SHUT_WR)
            request.settimeout(0.05)
            request.recv(65536)
        except OSError:
            pass
        finally:
            super().shutdown_request(request)

    def process_request_thread(self, request, client_address):
        try:
            self.service._submit(
                lambda store: self.finish_request(request, client_address)).result()
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)
            self._workers.release()


class SQLiteDashboardHandler(dashboard.Handler):
    def _send_html(self, body, code=200):
        policy = 'Agent fallback capture enabled.' if self.server.service._capture_enabled else (
            'SQLite queue replay enabled.' if self.server.service._replay_enabled else 'Queue capture/replay disabled.')
        purge_policy = ('Single recycled-record purge enabled with explicit ID confirmation. '
                        if self.server.service._purge_enabled else 'Hard deletion and purge disabled. ')
        notice = '<p role="status">Experimental SQLite integration: dashboard and MCP write control can be handed off. Reset and backup remain disabled. ' + purge_policy + policy + '</p>'
        disabled = '<style>#rb-empty-bin{display:none!important}</style>'
        body = body.replace(
            'MCP and dashboard requests share the same SQLite service owner. Separate writer handoff is not available.',
            'MCP and dashboard requests share one SQLite service owner. Use the controls above to transfer write access.',
        )
        return super()._send_html(body.replace('</header>', '</header>' + notice, 1).replace('</head>', disabled + '</head>', 1), code)

    def _unsupported(self):
        self.close_connection = True
        return self._send_json({'error': 'Endpoint disabled in experimental SQLite service',
                                'runtime_ready': False}, 501)

    def do_GET(self):
        if not self._check_boundary():
            return
        url = urlparse(self.path)
        try:
            dashboard._validated_query(url.query)
        except dashboard.RequestBoundaryError as error:
            return self._send_json({'error': str(error)}, error.status)
        path = url.path
        if path not in GET_ROUTES:
            return self._unsupported()
        if path == '/api/mode':
            from .service_binding import bound_store
            store = bound_store()
            dashboard_control = self.server.service.dashboard_control
            with self.server.service._state_lock:
                mode = self.server.service._mode
                mcp_available = self.server.service._stdio_active
            return self._send_json({
                'backend': 'sqlite', 'read_only': not dashboard_control,
                'mode': mode, 'dashboard_control': dashboard_control,
                'mcp_available': mcp_available,
                'handoff_supported': not self.server.service._capture_enabled,
                'writer_lock': {'held_by_us': not store._read_only,
                                'holder_pid': os.getpid() if not store._read_only else None},
                'pid': os.getpid(), 'runtime_ready': False,
            })
        if path == '/api/ready':
            from .service_binding import bound_store
            store = bound_store()
            bound = store.query_embedder is not None and store.document_embedder is not None
            return self._send_json({
                'ok': True, 'ready': True, 'backend': 'sqlite', 'runtime_ready': False,
                'read_only': store._read_only,
                'checks': {'backend': 'sqlite', 'runtime_ready': False,
                           'embedding_bindings': bound},
                'degraded_reasons': [] if bound else ['Embedding bindings missing; text operations unavailable'],
            })
        if path == '/api/health':
            from .service_binding import bound_store
            store = bound_store()
            bound = store.query_embedder is not None and store.document_embedder is not None
            return self._send_json({
                'ok': True, 'ready': True, 'backend': 'sqlite',
                'degraded_reasons': [] if bound else ['Embedding bindings missing; text operations unavailable'],
                'store_dir': str(store.persist_dir), 'read_only': store._read_only,
                'writer_lock': {'held_by_us': not store._read_only,
                                'holder_pid': os.getpid() if not store._read_only else None},
                'pid': os.getpid(), 'runtime_ready': False, 'embedding_probed': False,
                'dashboard_control': self.server.service.dashboard_control,
                'mcp_available': self.server.service._stdio_active,
                'queue_capture_enabled': self.server.service._capture_enabled,
                'queue_replay_enabled': self.server.service._replay_enabled,
                'purge_enabled': self.server.service._purge_enabled,
                'last_queue_replay': self.server.service.last_replay,
                'embedding_model': store.repository.identity.model + (' (bound; not probed)' if bound else ' (unbound)'),
            })
        return super().do_GET()

    def do_POST(self):
        if not self._check_boundary(mutation=True):
            return
        path = urlparse(self.path).path
        if path not in POST_ROUTES and not (path == '/api/purge' and self.server.service._purge_enabled):
            return self._unsupported()
        if path in ('/api/mode/take_control', '/api/mode/return_control'):
            try:
                data = self._read_json()
                if data:
                    raise dashboard.RequestBoundaryError('Handoff accepts an empty object', 400)
            except dashboard.RequestBoundaryError as error:
                self.close_connection = True
                return self._send_json({'error': str(error)}, error.status)
            result = (self.server.service.take_dashboard_control()
                      if path.endswith('take_control')
                      else self.server.service.return_control_to_mcp())
            accepted = result.get('acquired', result.get('released', False))
            return self._send_json(result, 200 if accepted else 409)
        from .service_binding import bound_store
        if (not self.server.service.dashboard_control
                and path not in ('/api/ready', '/api/score/recommend', '/api/heartbeat')):
            self.close_connection = True
            return self._send_json({'error': 'Dashboard has no write control; hand off from MCP first'}, 409)
        if bound_store()._read_only and path not in ('/api/ready', '/api/score/recommend'):
            self.close_connection = True
            return self._send_json({'error': 'SQLite observer is read-only; no queued write created'}, 409)
        if path == '/api/purge':
            return self._purge_record()
        if self.server.service._replay_store(bound_store()) and path not in ('/api/ready', '/api/heartbeat'):
            self.close_connection = True
            return self._send_json({'error': 'SQLite pending writes have not drained; mutation refused'}, 503)
        if path == '/api/heartbeat':
            if not self.server.service.heartbeat_dashboard():
                return self._send_json({'error': 'Dashboard does not own write control'}, 409)
            try:
                data = self._read_json()
                if data:
                    raise dashboard.RequestBoundaryError('Heartbeat accepts an empty object', 400)
            except dashboard.RequestBoundaryError as error:
                self.close_connection = True
                return self._send_json({'error': str(error)}, error.status)
            from . import nudge, dormancy
            tick = nudge.tick_nudge_session(bound_store(), idle_minutes=dormancy._cfg().session_idle_minutes)
            return self._send_json({'ok': True, 'backend': 'sqlite', 'new_session': tick.new_session_started})
        return super().do_POST()

    def _purge_record(self):
        from .service_binding import bound_store
        from .sqlite_purge import purge_recycled
        try:
            data = self._read_json()
            if not isinstance(data, dict) or set(data) != {'id', 'confirm_id'}:
                raise dashboard.RequestBoundaryError('Purge requires only id and confirm_id', 400)
            purged = purge_recycled(bound_store(), data['id'], data['confirm_id'])
        except dashboard.RequestBoundaryError as error:
            self.close_connection = True
            return self._send_json({'error': str(error)}, error.status)
        except ValueError as error:
            return self._send_json({'error': str(error)}, 400)
        except (RuntimeError, PermissionError, OSError):
            return self._send_json({'error': 'Purge refused; review pending files, storage permissions and guards'}, 409)
        except sqlite3.Error:
            return self._send_json({'error': 'Database purge was not confirmed; inspect committed records before retrying'}, 503)
        return self._send_json({'id': data['id'], 'purged': purged, 'backend': 'sqlite'})


def cli():
    """Explicit existing-database opt-in; never migrate or fall back implicitly."""
    import argparse
    from pathlib import Path
    from .sqlite_records import EmbeddingIdentity
    from .sqlite_queries import BoundQueryEmbedder
    from .sqlite_writes import BoundDocumentEmbedder
    from .embedding_provider import _PROVIDERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True, help='EmbeddingIdentity JSON file')
    parser.add_argument('--provider-config', type=Path, help='Single matching provider JSON configuration')
    parser.add_argument('--read-only', action='store_true')
    parser.add_argument('--queue-writes', action='store_true', help='Explicit observer-only MCP save/score capture')
    parser.add_argument('--replay-pending', action='store_true', help='Explicit owner-only SQLite queue replay')
    parser.add_argument('--enable-purge', action='store_true', help='Explicit owner-only confirmed purge of recycled records')
    parser.add_argument('--dashboard-port', type=int)
    arguments = parser.parse_args()
    if arguments.queue_writes and not arguments.read_only or arguments.replay_pending and arguments.read_only:
        parser.error('Queue capture requires --read-only; replay requires owner mode')
    if arguments.enable_purge and arguments.read_only:
        parser.error('Purge requires owner mode')
    identity = EmbeddingIdentity.from_dict(json.loads(arguments.identity.read_text(encoding='utf-8')))
    query = document = None
    if arguments.provider_config is not None:
        config = json.loads(arguments.provider_config.read_text(encoding='utf-8'))
        if not isinstance(config, dict) or config.get('providers') != [identity.provider]:
            parser.error('Exactly the stored provider must be selected; fallback is disabled')
        provider_class = _PROVIDERS.get(identity.provider)
        if provider_class is None or identity.provider != 'ollama':
            parser.error('CLI currently accepts only an explicitly configured local Ollama provider; other providers require reviewed API injection')
        provider = provider_class.build_from_config(config.get(identity.provider, {}))
        query = BoundQueryEmbedder(identity, provider)
        document = BoundDocumentEmbedder(identity, provider)
    with SQLiteService(arguments.database_dir, identity, persist_dir=arguments.data_dir,
                       read_only=arguments.read_only, query_embedder=query,
                       document_embedder=document, queue_writes=arguments.queue_writes,
                       replay_pending=arguments.replay_pending, enable_purge=arguments.enable_purge) as service:
        if arguments.dashboard_port is not None:
            import sys
            print(service.start_dashboard(arguments.dashboard_port), file=sys.stderr)
        asyncio.run(service.run_stdio())


if __name__ == '__main__':
    cli()
