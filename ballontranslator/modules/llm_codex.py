"""Official Codex App Server transport; authentication stays with Codex."""

from __future__ import annotations

import os
import queue
import shutil
import sys
import subprocess
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager, suppress
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .exceptions import LLMRequestStopped, LLMUserActionRequiredError
from .llm_chat import LLMChatRequestError, LLMChatResult
from ballontranslator.utils.llm_profiles import (
    CODEX_MODEL_REASONING_EFFORTS, LLMProfile,
)


class CodexRequestError(LLMUserActionRequiredError):
    """Stop batching when Codex needs attention or completion is uncertain.

    >>> issubclass(CodexRequestError, LLMUserActionRequiredError)
    True
    """


class CodexBusyError(CodexRequestError):
    """A terminal capacity rejection; exhaustion still stops the outer batch.

    >>> isinstance(CodexBusyError('at capacity'), LLMUserActionRequiredError)
    True
    """

    def __init__(self, message: str, usage: Any = None) -> None:
        super().__init__(message)
        self.usage = usage


class CodexTimeoutError(CodexRequestError):
    """Retry a timed-out attempt only after its owned session is closed.

    >>> CodexTimeoutError('timed out').usage is None
    True
    """

    usage: Any = None


def _quota_exhausted(detail: str, info: Any = None) -> bool:
    """Recognize terminal quota errors before generic busy/retry handling.

    >>> _quota_exhausted('Your workspace is out of credits.')
    True
    """
    return info == 'usageLimitExceeded' or (info != 'contextWindowExceeded' and any(
        marker in detail.lower() for marker in (
            'out of credits', 'usage limit reached', 'insufficient_quota')))


class _CodexSession:
    """Adapt the official SDK to the cancellable translation worker.

    >>> _CodexSession.__name__
    '_CodexSession'
    """

    def __init__(self, executable: Optional[str], cwd: str, timeout: int,
                 stop_event: Optional[threading.Event]) -> None:
        if sys.version_info < (3, 10):
            raise CodexRequestError('The official Codex Python SDK requires Python 3.10 or newer.')
        try:
            from openai_codex.client import CodexClient, CodexConfig
            from pydantic import RootModel
        except ImportError as error:
            raise CodexRequestError(
                f'Install the official Python SDK into this Python environment: "{sys.executable}" '
                '-m pip install -r '
                f'"{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "requirements-codex.txt"))}" '
                '(Python 3.10 or newer).'
            ) from error
        self.stop_event = stop_event
        self.deadline = time.monotonic() + timeout
        self.incoming: queue.Queue = queue.Queue()
        self.outgoing: queue.Queue = queue.Queue()
        self.request_id = 0
        self.thread_id = ''
        self.turn_id = ''
        self.login_id = ''
        self.response_model = RootModel[Dict[str, Any]]
        overrides = {
            'model_provider': '"openai"', 'mcp_servers': '{}',
            'model_providers': '{}', 'project_doc_max_bytes': '0',
            'web_search': '"disabled"', 'notify': '[]',
        }
        for feature in ('shell_tool', 'apps', 'plugins', 'hooks', 'multi_agent',
                        'browser_use', 'computer_use', 'image_generation',
                        'code_mode', 'code_mode_host', 'memories', 'skill_search'):
            overrides[f'features.{feature}'] = 'false'
        # SDK environment overrides are merged with the parent environment.
        # Empty values keep inherited credentials from overriding the saved login.
        self.client = CodexClient(CodexConfig(
            codex_bin=executable, cwd=cwd,
            config_overrides=tuple(f'{key}={value}' for key, value in overrides.items()),
            env={key: '' for key in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL')},
        ), approval_handler=self._reject_tool)
        try:
            self.client.start()
        except Exception as error:
            self.client.close()
            raise CodexRequestError(f'Cannot start Codex SDK: {error}') from error
        self.reader: Optional[threading.Thread] = None
        self.writer = threading.Thread(target=self._requests, daemon=True)
        self.writer.start()
        # Turn/login events have dedicated SDK queues. Drain unrelated status
        # events so a client reused across a whole chapter stays bounded.
        self.status_reader = threading.Thread(target=self._drain_status, daemon=True)
        self.status_reader.start()

    def _drain_status(self) -> None:
        try:
            while True:
                self.client.next_notification()
        except Exception:
            return

    @staticmethod
    def _reject_tool(method: str, params: Any) -> Dict[str, Any]:
        raise CodexRequestError('Codex requested an interactive tool; translation stopped.')

    def _notifications(self, identifier: str, login: bool = False) -> None:
        from openai_codex.models import UnknownNotification
        try:
            while True:
                notification = (self.client.next_login_notification(identifier) if login
                                else self.client.next_turn_notification(identifier))
                payload = notification.payload
                params = (payload.params if isinstance(payload, UnknownNotification)
                          else payload.model_dump(mode='json', by_alias=True))
                self.incoming.put({'method': notification.method, 'params': params})
                if notification.method in ('turn/completed', 'account/login/completed'):
                    return
        except Exception as error:
            self.incoming.put(CodexRequestError(f'Codex SDK connection closed: {error}'))
        finally:
            if login:
                self.client.unregister_login_notifications(identifier)
            else:
                self.client.unregister_turn_notifications(identifier)

    def _requests(self) -> None:
        try:
            while True:
                message = self.outgoing.get()
                if message is None:
                    return
                if 'id' not in message:
                    self.client.notify(message['method'], message.get('params'))
                    continue
                if message['method'] == 'turn/start':
                    params = message['params']
                    result = self.client.turn_start(params['threadId'], params['input'], params)
                    self.turn_id = result.turn.id
                    self.incoming.put({'id': message['id'], 'result': result.model_dump(mode='json', by_alias=True)})
                    self.reader = threading.Thread(
                        target=self._notifications, args=(self.turn_id,), daemon=True,
                    )
                    self.reader.start()
                else:
                    result = self.client.request(
                        message['method'], message['params'], response_model=self.response_model,
                    )
                    self.incoming.put({'id': message['id'], 'result': result.root})
        except Exception as error:
            # The SDK raises RPC failures instead of returning raw envelopes.
            data = getattr(error, 'data', None)
            info = data.get('codexErrorInfo') if isinstance(data, dict) else None
            prefix = 'Codex quota exhausted' if _quota_exhausted(str(error), info) else 'Codex SDK request failed'
            self.incoming.put(CodexRequestError(f'{prefix}: {error}'))

    def send(self, method: str, params: Dict[str, Any]) -> int:
        self.request_id += 1
        self.outgoing.put({'id': self.request_id, 'method': method, 'params': params})
        return self.request_id

    def receive(self) -> Dict[str, Any]:
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise CodexTimeoutError(
                    'Codex request timed out. Retrying may consume additional tokens '
                    'if the service already processed the request.'
                )
            try:
                message = self.incoming.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if isinstance(message, Exception):
                raise message
            params = message.get('params') or {}
            error = message.get('error')
            if isinstance(params, dict):
                if message.get('method') == 'error':
                    error = params.get('error')
                elif message.get('method') == 'turn/completed':
                    turn = params.get('turn') or {}
                    if isinstance(turn, dict) and turn.get('status') == 'failed':
                        error = turn.get('error')
            if isinstance(error, dict):
                detail = str(error.get('message', ''))
                info = error.get('codexErrorInfo')
                if _quota_exhausted(detail, info):
                    # Quota is terminal even when wrapped as a retryable event or 503.
                    raise CodexRequestError(f'Codex quota exhausted: {detail}')
            return message

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request_id = self.send(method, params)
        while True:
            message = self.receive()
            if message.get('id') == request_id:
                return message['result']

    def close(self) -> None:
        if self.login_id:
            self.send('account/login/cancel', {'loginId': self.login_id})
        if self.thread_id and self.turn_id:
            self.send('turn/interrupt', {'threadId': self.thread_id, 'turnId': self.turn_id})
        self.outgoing.put(None)
        # Bound interruption even if the server never acknowledges the request.
        self.writer.join(timeout=0.2)
        # SDK 0.156.1 only terminates its direct process; Windows custom
        # launchers can own a native child. Keep retries from leaving it running.
        process = self.client._proc
        if (os.name == 'nt' and self.client.config.codex_bin is not None
                and process is not None and process.poll() is None):
            subprocess.run(
                ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=5,
            )
        # SDK 0.156.1 can raise while closing stdin after a server disconnect.
        with suppress(OSError):
            self.client.close()
        if process is not None:
            process.wait(timeout=5)
        if self.reader is not None:
            self.reader.join(timeout=1)
        self.writer.join(timeout=1)
        self.status_reader.join(timeout=1)
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def _start_codex_session(profile: LLMProfile, cwd: str,
                         stop_event: Optional[threading.Event]) -> _CodexSession:
    if stop_event is not None and stop_event.is_set():
        raise LLMRequestStopped()
    if type(profile.codex_timeout) is not int or not 1 <= profile.codex_timeout <= 86400:
        raise CodexRequestError('Codex Timeout must be between 1 and 86400 seconds.')
    configured_executable = profile.codex_executable.strip()
    executable = None
    if configured_executable not in ('', 'codex'):
        executable = shutil.which(configured_executable)
        if not executable:
            raise CodexRequestError('Codex Executable was not found. Use codex for the bundled SDK runtime.')
    session = _CodexSession(executable, cwd, profile.codex_timeout, stop_event)
    try:
        session.call('initialize', {
            'clientInfo': {'name': 'ballontranslator', 'version': '1.0'},
            'capabilities': {'experimentalApi': True},
        })
        session.outgoing.put({'method': 'initialized', 'params': {}})
        return session
    except Exception:
        session.close()
        raise


class CodexSessionPool:
    """Reuse exclusive SDK connections only inside an explicitly owned batch.

    >>> CodexSessionPool().users
    0
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0
        self.idle: List[Tuple[Tuple[str, Optional[str]], _CodexSession, tempfile.TemporaryDirectory]] = []

    @contextmanager
    def batch(self) -> Iterator[None]:
        with self.lock:
            self.users += 1
        try:
            yield
        finally:
            with self.lock:
                self.users -= 1
                idle = self.idle if not self.users else []
                if not self.users:
                    self.idle = []
            with ExitStack() as cleanup:
                for _, session, directory in idle:
                    cleanup.callback(directory.cleanup)
                    cleanup.callback(session.close)

    @contextmanager
    def session(self, profile: LLMProfile, stop_event: Optional[threading.Event]) -> Iterator[Tuple[_CodexSession, str]]:
        key = (profile.codex_executable, os.environ.get('CODEX_HOME'))
        with self.lock:
            index = next((i for i, item in enumerate(self.idle) if item[0] == key), None)
            cached = self.idle.pop(index) if index is not None else None
        if cached is not None and cached[1].client._proc.poll() is not None:
            _, session, directory = cached
            try:
                session.close()
            finally:
                directory.cleanup()
            cached = None
        if cached is None:
            directory = tempfile.TemporaryDirectory(prefix='ballontranslator-codex-')
            try:
                session = _start_codex_session(profile, directory.name, stop_event)
            except BaseException:
                directory.cleanup()
                raise
        else:
            _, session, directory = cached
            session.stop_event = stop_event
            session.deadline = time.monotonic() + profile.codex_timeout
        reusable = False
        try:
            yield session, directory.name
            # Release the server's loaded conversation without deleting saved
            # history. A failed cleanup discards the client, never the result.
            if self.users:
                try:
                    session.call('thread/unsubscribe', {'threadId': session.thread_id})
                    if session.reader is not None:
                        session.reader.join(timeout=1)
                    session.thread_id = ''
                    reusable = session.reader is None or not session.reader.is_alive()
                except (CodexRequestError, LLMRequestStopped):
                    pass
        finally:
            with self.lock:
                keep = reusable and self.users > 0
                if keep:
                    self.idle.append((key, session, directory))
            if not keep:
                try:
                    session.close()
                finally:
                    directory.cleanup()


def authenticate_codex(profile: LLMProfile, method: str = 'reuse', api_key: str = '',
                       on_challenge: Optional[Callable[[Dict[str, str]], None]] = None,
                       stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """Read or update SDK-owned authentication without storing secrets in profiles.

    >>> callable(authenticate_codex)
    True
    """
    if method not in ('reuse', 'chatgpt', 'chatgptDeviceCode', 'apiKey'):
        raise CodexRequestError('Unsupported Codex login method.')
    if method == 'apiKey' and not api_key.strip():
        raise CodexRequestError('Enter an API key.')
    with tempfile.TemporaryDirectory(prefix='ballontranslator-codex-login-') as cwd:
        session = _start_codex_session(profile, cwd, stop_event)
        try:
            if method != 'reuse':
                params = {'type': method}
                if method == 'apiKey':
                    params['apiKey'] = api_key.strip()
                started = session.call('account/login/start', params)
                if method != 'apiKey':
                    session.login_id = started['loginId']
                    session.client.register_login_notifications(session.login_id)
                    session.reader = threading.Thread(
                        target=session._notifications, args=(session.login_id, True), daemon=True,
                    )
                    session.reader.start()
                    if on_challenge is not None:
                        on_challenge({
                            'url': started.get('authUrl') or started['verificationUrl'],
                            'code': started.get('userCode', ''),
                        })
                    while True:
                        message = session.receive()
                        data = message.get('params', {})
                        if (message.get('method') == 'account/login/completed'
                                and data.get('loginId') == session.login_id):
                            session.login_id = ''
                            if not data.get('success'):
                                raise CodexRequestError(data.get('error') or 'Codex login failed.')
                            break
            account = session.call('account/read', {'refreshToken': False}).get('account')
            return {key: account.get(key) for key in ('type', 'email', 'planType')} if account else {}
        except (KeyError, TypeError, ValueError) as error:
            raise CodexRequestError('Invalid Codex login response.') from error
        except CodexRequestError as error:
            # Servers can echo submitted values; never expose a submitted key.
            detail = str(error)
            if api_key.strip():
                detail = detail.replace(api_key.strip(), '[redacted]')
            raise CodexRequestError(detail) from None
        finally:
            session.close()


def _input_parts(content: Any, *, history: bool = False,
                 assistant: bool = False) -> List[Dict[str, Any]]:
    """Convert chat content without flattening images or changing their order.

    >>> _input_parts('hello')
    [{'type': 'text', 'text': 'hello'}]
    """
    parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
    if not isinstance(parts, list):
        raise CodexRequestError('Codex requires text or image message content.')
    converted = []
    for part in parts:
        if part.get('type') == 'text':
            kind = ('output_text' if assistant else 'input_text') if history else 'text'
            converted.append({'type': kind, 'text': part['text']})
        elif part.get('type') == 'image_url' and not assistant:
            image = part['image_url']
            converted.append(
                {'type': 'input_image', 'image_url': image['url'], 'detail': image.get('detail', 'auto')}
                if history else {'type': 'image', 'url': image['url']}
            )
        else:
            raise CodexRequestError('Unsupported Codex message content.')
    return converted


def request_codex_completion(profile: LLMProfile, api_args: Dict[str, Any],
                             stop_event: Optional[threading.Event] = None,
                             pool: Optional[CodexSessionPool] = None) -> LLMChatResult:
    """Run a fresh turn with SDK-owned authentication and the caller's exact history.

    No account tokens are read by BallonsTranslator. Capacity rejection and
    timeout allow bounded retries after cleanup; other uncertain failures stop.

    >>> request_codex_completion.__name__
    'request_codex_completion'
    """
    if stop_event is not None and stop_event.is_set():
        raise LLMRequestStopped()
    if type(profile.codex_timeout) is not int or not 1 <= profile.codex_timeout <= 86400:
        raise CodexRequestError('Codex Timeout must be between 1 and 86400 seconds.')
    if type(profile.codex_save_sessions) is not bool:
        raise CodexRequestError('Save Codex Sessions must be enabled or disabled.')
    effort = api_args.get('reasoning_effort', profile.thinking_level)
    if effort not in CODEX_MODEL_REASONING_EFFORTS.get(api_args['model'], ()):
        raise CodexRequestError(
            f'Codex model {api_args["model"]} does not support thinking level {effort}. '
            'Select a supported Codex model and thinking level.'
        )
    messages = api_args['messages']
    if not messages or messages[-1].get('role') != 'user':
        raise CodexRequestError('Codex requires a final user message.')
    history = []
    for message in messages[:-1]:
        role = message['role']
        if role not in ('system', 'developer', 'user', 'assistant'):
            raise CodexRequestError(f'Unsupported Codex message role: {role}')
        history.append({'type': 'message', 'role': role, 'content': _input_parts(
            message['content'], history=True, assistant=role == 'assistant',
        )})
    inputs = _input_parts(messages[-1]['content'])
    # Always start fresh; saving conversation content requires explicit opt-in.
    with (pool or CodexSessionPool()).session(profile, stop_event) as (session, cwd):
        usage = None
        try:
            account = session.call('account/read', {'refreshToken': False}).get('account')
            if not account or account.get('type') not in ('chatgpt', 'apiKey'):
                raise CodexRequestError('Codex login is required. Open Codex Login in the LLM profile settings.')
            thread = session.call('thread/start', {
                'model': api_args['model'], 'modelProvider': 'openai',
                'cwd': cwd, 'ephemeral': not profile.codex_save_sessions, 'approvalPolicy': 'never',
                'sandbox': 'read-only', 'environments': [],
                'baseInstructions': 'Follow the supplied translation or OCR contract. Return only the requested result. Do not use tools.',
                'developerInstructions': '',
            })
            session.thread_id = thread['thread']['id']
            if history:
                session.call('thread/inject_items', {'threadId': session.thread_id, 'items': history})
            turn_params = {'threadId': session.thread_id, 'input': inputs, 'effort': effort}
            schema = api_args.get('response_format', {}).get('json_schema', {}).get('schema')
            if schema is not None:
                turn_params['outputSchema'] = schema
            # Consume turn/start and notifications in one loop: fast servers can
            # finish a turn before its request response reaches the pipe reader.
            turn_request = session.send('turn/start', turn_params)
            output = {}
            while True:
                message = session.receive()
                if message.get('id') == turn_request:
                    if 'error' in message:
                        raise CodexRequestError(f'Codex turn/start: {message["error"].get("message", "request failed")}')
                    session.turn_id = message['result']['turn']['id']
                params = message.get('params', {})
                if params.get('threadId') != session.thread_id:
                    continue
                method = message.get('method')
                if method == 'turn/started':
                    session.turn_id = params['turn']['id']
                elif method == 'item/completed':
                    item = params['item']
                    if item.get('type') == 'agentMessage' and item.get('phase') != 'commentary':
                        output[item['id']] = item['text']
                elif method == 'thread/tokenUsage/updated':
                    # Each request owns a fresh thread. Replace its cumulative
                    # snapshot; adding updates double-counts repeated reports.
                    tokens = params['tokenUsage']['total']
                    usage = SimpleNamespace(
                        total_tokens=tokens['totalTokens'], prompt_tokens=tokens['inputTokens'],
                        completion_tokens=tokens['outputTokens'],
                        prompt_tokens_details={
                            'cached_tokens': tokens['cachedInputTokens'],
                            'cache_write_tokens': tokens.get('cacheWriteInputTokens'),
                        },
                        completion_tokens_details={'reasoning_tokens': tokens['reasoningOutputTokens']},
                    )
                elif method == 'turn/completed':
                    turn = params['turn']
                    session.turn_id = ''
                    if turn['status'] == 'interrupted':
                        raise LLMRequestStopped()
                    if turn['status'] != 'completed':
                        error = turn.get('error') or {}
                        if error.get('codexErrorInfo') == 'contextWindowExceeded':
                            raise LLMChatRequestError(RuntimeError('maximum context length exceeded'))
                        detail = error.get('message', turn['status'])
                        info = error.get('codexErrorInfo')
                        http_error = (info.get('httpConnectionFailed') or {}) if isinstance(info, dict) else {}
                        # Wait for terminal failure, never retry an intermediate
                        # error event while Codex may still be recovering itself.
                        if ('selected model is at capacity' in detail.lower()
                                or http_error.get('httpStatusCode') == 503):
                            raise CodexBusyError(f'Codex turn failed: {detail}', usage)
                        raise CodexRequestError(f'Codex turn failed: {detail}')
                    return LLMChatResult(content='\n'.join(output.values()), usage=usage, finish_reason='stop')
        except CodexTimeoutError as error:
            error.usage = usage
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            # A protocol mismatch may occur after submission; never let the
            # feature owner's generic retry loop charge for the same turn again.
            raise CodexRequestError(f'Invalid Codex protocol response: {error}') from error
