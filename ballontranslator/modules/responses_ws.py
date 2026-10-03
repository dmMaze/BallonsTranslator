"""Persistent Responses WebSocket transport shared by API and subscription callers."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
import json
import threading
import time
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Tuple, TYPE_CHECKING

from .exceptions import LLMRequestStopped
from ballontranslator.utils.logger import logger as LOGGER

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


def run_async(operation: Coroutine, stop_event: Optional[threading.Event], *,
              is_current: Callable[[], bool], loop: Optional[asyncio.AbstractEventLoop] = None) -> Any:
    """Run worker IO with cancellation and ownership checks, including after completion."""
    async def run() -> Any:
        task = asyncio.create_task(operation)
        try:
            while True:
                if (stop_event is not None and stop_event.is_set()) or not is_current():
                    raise LLMRequestStopped()
                done, _ = await asyncio.wait((task,), timeout=0.05)
                if done:
                    if (stop_event is not None and stop_event.is_set()) or not is_current():
                        raise LLMRequestStopped()
                    return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        return asyncio.run(run()) if loop is None else asyncio.run_coroutine_threadsafe(run(), loop).result()
    except (asyncio.CancelledError, FutureCancelledError):
        raise LLMRequestStopped() from None


class ResponsesWebSocket:
    """Own a job's socket, event loop and connection-scoped continuation.

    >>> ResponsesWebSocket('https://api.openai.com/v1/responses', 'job').cache_key
    'job'
    """

    def __init__(self, url: str, cache_key: str, proxy: str = '', *, websocket: bool = True) -> None:
        self.cache_key = cache_key
        self.proxy = proxy
        self.url = url
        self.websocket = websocket
        self.closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._idle_timer: Optional[threading.Timer] = None
        self._socket: Optional[ClientConnection] = None
        self._connected_at = 0.0
        self._last_used = 0.0
        self._previous: Optional[Tuple[Dict, Dict]] = None
        self._http_only = not websocket

    def run(self, operation: Coroutine, stop_event: Optional[threading.Event]) -> Any:
        with self._lock:
            if self.closed:
                operation.close()
                raise LLMRequestStopped()
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            if self._loop is None and not self._http_only:
                self._loop = asyncio.new_event_loop()
                # The socket must answer server pings while page workers are
                # idle or replaced. Its event loop lives with this session.
                self._thread = threading.Thread(target=self._loop.run_forever,
                                                name='Responses WebSocket', daemon=True)
                self._thread.start()
            try:
                return run_async(operation, stop_event, is_current=self._is_current, loop=self._loop)
            finally:
                if self.closed or (stop_event is not None and stop_event.is_set()) or not self._is_current():
                    self.closed = True
                    self._dispose()
                elif self._socket is None:
                    self._dispose()
                else:
                    self._last_used = time.monotonic()
                    self._idle_timer = threading.Timer(300.0, self._expire)
                    self._idle_timer.daemon = True
                    self._idle_timer.start()

    def _is_current(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        if self._lock.acquire(blocking=False):
            try:
                self._dispose()
            finally:
                self._lock.release()
        else:
            loop = self._loop
            if loop is not None and loop.is_running():
                try:
                    loop.call_soon_threadsafe(self._cancel)
                except RuntimeError:
                    pass  # The worker already disposed the loop.

    def _expire(self) -> None:
        if self._lock.acquire(blocking=False):
            try:
                if time.monotonic() - self._last_used >= 300.0:
                    self.closed = True
                    self._dispose()
            finally:
                self._lock.release()

    def _cancel(self) -> None:
        if self._loop is None:
            return  # Disposal owns cancellation once it detaches the loop.
        if self._socket is not None:
            self._socket.transport.abort()
        current = asyncio.current_task(self._loop)
        for task in asyncio.all_tasks(self._loop):
            if task is not current:
                task.cancel()

    def _dispose(self) -> None:
        if self._loop is None:
            return
        loop, thread, socket = self._loop, self._thread, self._socket
        # Repeated close calls must not cancel the cleanup coroutine itself.
        self._loop = None
        self._thread = None
        self._socket = None
        self._previous = None

        async def shutdown() -> None:
            if socket is not None:
                socket.transport.abort()
            pending = asyncio.all_tasks(loop) - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await loop.shutdown_asyncgens()

        asyncio.run_coroutine_threadsafe(shutdown(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()

    async def _disconnect(self) -> None:
        socket, self._socket = self._socket, None
        self._previous = None
        if socket is not None:
            await socket.close()

    async def request(self, payload: Dict, headers: Callable[[], Awaitable[Dict[str, str]]],
                      on_event: Callable[[Dict, List[Dict]], Optional[Dict]], *,
                      on_headers: Optional[Callable[[object], None]] = None,
                      client_metadata: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        if self._http_only:
            return None
        try:
            import websockets
            from websockets.asyncio.client import connect
            from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidProxy, InvalidStatus
            if int(websockets.__version__.split('.', 1)[0]) < 15:
                raise ImportError('WebSocket proxy support requires version 15.')
        except ImportError:
            self._http_only = True
            LOGGER.warning('Responses WebSocket requires Python>=3.9 and websockets>=15; using HTTP for this job.')
            return None
        started = False
        for attempt in range(2):
            reused = self._socket is not None
            try:
                if self._socket is not None and time.monotonic() - self._connected_at >= 55 * 60:
                    await self._disconnect()
                reused = self._socket is not None
                if self._socket is None:
                    connection = connect(
                        self.url.replace('https://', 'wss://', 1).replace('http://', 'ws://', 1),
                        additional_headers=await headers(), proxy=self.proxy or True,
                        open_timeout=15.0, close_timeout=1.0, ping_interval=None, max_size=None,
                    )
                    # Preserve HTTPX's fixed-endpoint policy for auth and routing headers.
                    connection.process_redirect = lambda error: error
                    self._socket = await connection
                    self._connected_at = time.monotonic()
                    if on_headers is not None:
                        on_headers(self._socket.response.headers)
                wire = payload
                if self._previous is not None:
                    previous, result = self._previous
                    baseline = previous['input'] + result.get('output', [])
                    settings = {key: value for key, value in payload.items() if key != 'input'}
                    old_settings = {key: value for key, value in previous.items() if key != 'input'}
                    if settings == old_settings and payload['input'][:len(baseline)] == baseline and result.get('id'):
                        wire = {**payload, 'previous_response_id': result['id'], 'input': payload['input'][len(baseline):]}
                    else:
                        self._previous = None
                LOGGER.debug('Responses WebSocket request: reused=%s, continuation=%s, input_items=%d',
                             reused, 'previous_response_id' in wire, len(wire['input']))
                frame = {'type': 'response.create', **wire}
                if client_metadata:
                    frame['client_metadata'] = dict(client_metadata)
                await self._socket.send(json.dumps(frame))
                completed_items = []
                while True:
                    event = json.loads(await asyncio.wait_for(self._socket.recv(), timeout=300.0))
                    # Public Responses errors may be flat; Codex wraps them.
                    error = event.get('error') or event
                    if (event.get('type') == 'error' and isinstance(error, dict)
                            and error.get('code') == 'websocket_connection_limit_reached' and not started):
                        await self._disconnect()
                        if attempt == 0:
                            break
                        self._http_only = True
                        return None
                    if (event.get('type') == 'error' and isinstance(error, dict)
                            and error.get('code') == 'previous_response_not_found'
                            and 'previous_response_id' in wire and attempt == 0):
                        await self._disconnect()
                        break
                    result = on_event(event, completed_items)
                    started = True
                    if result is not None:
                        self._previous = payload, result
                        return result
            except InvalidStatus as error:
                if on_headers is not None:
                    on_headers(error.response.headers)
                await self._disconnect()
                if (error.response.status_code not in (404, 405, 426, 500, 502, 503, 504)
                        and not 300 <= error.response.status_code < 400):
                    raise
                self._http_only = True
                LOGGER.debug('Responses WebSocket unavailable (HTTP %d); using HTTP.', error.response.status_code)
                return None
            except (OSError, asyncio.TimeoutError, ConnectionClosed, InvalidHandshake, InvalidProxy, ImportError) as error:
                await self._disconnect()
                if started:
                    raise
                if attempt == 0 and reused:
                    continue
                self._http_only = True
                LOGGER.debug('Responses WebSocket unavailable (%s); using HTTP.', type(error).__name__)
                return None
            except BaseException:
                await self._disconnect()
                raise
        raise RuntimeError('Responses could not recover the WebSocket continuation.')
