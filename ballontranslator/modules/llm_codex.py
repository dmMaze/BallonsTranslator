"""Official Codex App Server transport; authentication stays with Codex."""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

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


class _CodexSession:
    """One disposable stdio session, with bounded and cancellable pipe IO.

    >>> _CodexSession.__name__
    '_CodexSession'
    """

    def __init__(self, executable: str, cwd: str, timeout: int,
                 stop_event: Optional[threading.Event]) -> None:
        self.stop_event = stop_event
        self.deadline = time.monotonic() + timeout
        self.incoming: queue.Queue = queue.Queue()
        self.outgoing: queue.Queue = queue.Queue()
        self.request_id = 0
        self.thread_id = ''
        self.turn_id = ''
        # Disable unrelated user integrations without editing their Codex config.
        overrides = {
            'model_provider': '"openai"', 'mcp_servers': '{}',
            'model_providers': '{}', 'project_doc_max_bytes': '0',
            'web_search': '"disabled"', 'notify': '[]',
        }
        for feature in ('shell_tool', 'apps', 'plugins', 'hooks', 'multi_agent',
                        'browser_use', 'computer_use', 'image_generation',
                        'code_mode', 'code_mode_host', 'memories', 'skill_search'):
            overrides[f'features.{feature}'] = 'false'
        command = [executable, 'app-server', '--listen', 'stdio://']
        for key, value in overrides.items():
            command.extend(['-c', f'{key}={value}'])
        env = os.environ.copy()
        for key in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL'):
            env.pop(key, None)
        self.stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.stderr, text=True,
                encoding='utf-8', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                start_new_session=os.name != 'nt',
            )
        except OSError as error:
            self.stderr.close()
            raise CodexRequestError(f'Cannot start Codex: {error}') from error
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.reader.start()
        self.writer.start()

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError('Expected a JSON-RPC object.')
                self.incoming.put(message)
        except (OSError, ValueError) as error:
            self.incoming.put(CodexRequestError(f'Invalid Codex response: {error}'))
        finally:
            self.incoming.put(None)

    def _write(self) -> None:
        try:
            while True:
                message = self.outgoing.get()
                if message is None:
                    break
                self.process.stdin.write(json.dumps(message, ensure_ascii=False) + '\n')
                self.process.stdin.flush()
        except (OSError, ValueError) as error:
            self.incoming.put(CodexRequestError(f'Codex connection closed: {error}'))
        finally:
            with suppress(OSError):
                self.process.stdin.close()

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
                raise CodexRequestError(
                    'Codex request timed out. The run was stopped without resubmitting; '
                    'check Codex usage or increase Codex Timeout before retrying.'
                )
            try:
                message = self.incoming.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if message is None:
                self.stderr.seek(0, os.SEEK_END)
                self.stderr.seek(max(0, self.stderr.tell() - 2000))
                detail = self.stderr.read().decode('utf-8', errors='replace').strip()
                raise CodexRequestError(f'Codex App Server exited before completion. {detail}')
            if isinstance(message, Exception):
                raise message
            if 'method' in message and 'id' in message:
                # Translation/OCR never approves tool execution or interactive input.
                self.outgoing.put({'id': message['id'], 'error': {
                    'code': -32601, 'message': 'Interactive tools are unavailable in translation/OCR.',
                }})
                raise CodexRequestError('Codex requested an interactive tool; translation stopped.')
            return message

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request_id = self.send(method, params)
        while True:
            message = self.receive()
            if message.get('id') != request_id:
                continue
            if 'error' in message:
                raise CodexRequestError(f'Codex {method}: {message["error"].get("message", "request failed")}')
            return message['result']

    def close(self) -> None:
        if self.thread_id and self.turn_id and self.process.poll() is None:
            self.send('turn/interrupt', {'threadId': self.thread_id, 'turnId': self.turn_id})
        # EOF normally shuts down Codex. npm launchers own a native child, so a
        # stalled launcher must be terminated together with its process tree.
        self.outgoing.put(None)
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(
                    ['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW, timeout=5,
                )
            else:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        self.reader.join(timeout=1)
        self.writer.join(timeout=1)
        self.process.stdout.close()
        self.stderr.close()


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
                             stop_event: Optional[threading.Event] = None) -> LLMChatResult:
    """Run a fresh turn with official ChatGPT auth and the caller's exact history.

    No account tokens are read by BallonsTranslator. A lost/failed turn stops the
    batch instead of triggering the caller's ordinary automatic retry loop.

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
    executable = shutil.which(profile.codex_executable.strip())
    if not executable:
        raise CodexRequestError('Install the official Codex CLI and set Codex Executable, then run codex login.')
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
    with tempfile.TemporaryDirectory(prefix='ballontranslator-codex-') as cwd:
        session = _CodexSession(executable, cwd, profile.codex_timeout, stop_event)
        try:
            session.call('initialize', {
                'clientInfo': {'name': 'ballontranslator', 'version': '1.0'},
                'capabilities': {'experimentalApi': True},
            })
            session.outgoing.put({'method': 'initialized', 'params': {}})
            account = session.call('account/read', {'refreshToken': False}).get('account')
            if not account or account.get('type') != 'chatgpt':
                raise CodexRequestError('Codex ChatGPT login is required. Run codex login, then retry.')
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
            usage = None
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
                        raise CodexRequestError(f'Codex turn failed: {error.get("message", turn["status"])}')
                    return LLMChatResult(content='\n'.join(output.values()), usage=usage, finish_reason='stop')
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            # A protocol mismatch may occur after submission; never let the
            # feature owner's generic retry loop charge for the same turn again.
            raise CodexRequestError(f'Invalid Codex protocol response: {error}') from error
        finally:
            session.close()
