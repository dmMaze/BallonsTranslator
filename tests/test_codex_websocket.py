import asyncio
import copy
import inspect
import json
import os
import sys
import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from websockets.protocol import State

from ballontranslator.modules import codex
from ballontranslator.modules.exceptions import LLMRequestStopped, LLMUserActionRequiredError
from ballontranslator.modules.ocr.ocr_llm import LLMOCR
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, sync_codex_profile
from test_codex import CATALOG, completion, sse, tokens


class ConnectAttempt:
    """Model connect's configurable awaitable, rather than a coroutine function."""

    def __init__(self, result) -> None:
        self.result = result

    def __await__(self):
        return self.result.__await__()


class Socket:
    def __init__(self) -> None:
        self.frames = []
        self.loops = []
        self.events = deque()
        self.closed = False
        self.state = State.OPEN
        self.transport = Mock()
        self.response = SimpleNamespace(headers=httpx.Headers())
        self.respond = self.complete

    def complete(self, body: dict) -> list:
        event = completion()
        result = event['response']
        result['id'] = f'resp_{len(self.frames)}'
        result['output'][0].update(id=f'msg_{len(self.frames)}', status='completed')
        result['output'].insert(0, {'type': 'reasoning', 'id': f'rs_{len(self.frames)}',
                                   'summary': [], 'encrypted_content': f'opaque-{len(self.frames)}'})
        return [{'type': 'response.created'}, event]

    async def send(self, data: str) -> None:
        body = json.loads(data)
        self.frames.append(body)
        self.loops.append(asyncio.get_running_loop())
        self.events.extend(self.respond(body))

    async def recv(self) -> str:
        if not self.events:
            await asyncio.Future()
        event = self.events.popleft()
        if isinstance(event, BaseException):
            raise event
        return json.dumps(event)

    async def close(self) -> None:
        self.closed = True


@unittest.skipIf(sys.version_info < (3, 9), 'WebSocket transport requires Python 3.9 or newer.')
class CodexWebSocketTest(unittest.TestCase):
    def setUp(self) -> None:
        params_patcher = patch.object(LLMTranslator, 'params', copy.deepcopy(LLMTranslator.params))
        params_patcher.start()
        self.addCleanup(params_patcher.stop)
        self.account = codex.CodexAccount()
        self.account._loaded = True
        self.account._credentials = tokens()
        self.profile = default_codex_profile()
        self.profile.model = 'vision-model'
        sync_codex_profile(self.profile, CATALOG)
        self.requester = LLMTranslator('日本語', 'English')
        self.requester.set_stop_event(threading.Event())
        self.requester._respect_delay = Mock()
        self.messages = []
        self.socket = Socket()
        self.http_requests = []
        self.connect = AsyncMock(return_value=self.socket)
        self.connect_patcher = patch('websockets.asyncio.client.connect',
                                     side_effect=lambda *args, **kwargs: ConnectAttempt(self.connect(*args, **kwargs)))
        for patcher in (
            patch.object(codex, 'account', self.account),
            patch.object(codex, 'CODEX_REPLAY_ENABLED', True),
            patch.object(codex, '_client_version_checked_at', time.monotonic()),
            patch.dict(pcfg.module.codex_models, copy.deepcopy(CATALOG), clear=True),
            self.connect_patcher,
            patch.object(codex, '_http_client', side_effect=lambda proxy='': httpx.AsyncClient(
                transport=httpx.MockTransport(self.http))),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.requester.set_stop_event, None)

    def http(self, request: httpx.Request) -> httpx.Response:
        self.http_requests.append(request)
        return httpx.Response(200, text=sse(completion()))

    def call(self, text: str = 'page', *, turn=None):
        self.messages.append({'role': 'user', 'content': text})
        result = self.requester.request_chat_completion(self.profile, {
            'model': self.profile.model,
            'messages': [{'role': 'system', 'content': 'Translate.'}, *self.messages],
        }, codex_turn=turn)
        self.messages.append({'role': 'assistant', 'content': result.content, 'codex_response': result})
        return result

    def test_reuses_connection_and_loop_and_sends_only_new_input(self) -> None:
        first = self.call('first page')
        self.call('second page')
        self.assertEqual(self.connect.await_count, 1)
        self.assertIs(self.socket.loops[0], self.socket.loops[1])
        first_body, second_body = self.socket.frames
        self.assertNotIn('previous_response_id', first_body)
        self.assertEqual(second_body['previous_response_id'], 'resp_1')
        self.assertEqual(second_body['input'], [{'type': 'message', 'role': 'user',
                                                'content': [{'type': 'input_text', 'text': 'second page'}]}])
        self.assertFalse(second_body['store'])
        self.assertNotIn('stream', first_body)
        self.assertNotIn('stream', second_body)
        headers = self.connect.call_args.kwargs['additional_headers']
        self.assertEqual(headers['OpenAI-Beta'], 'responses_websockets=2026-02-06')
        self.assertEqual(headers['version'], codex._client_version)
        self.assertEqual(headers['x-codex-routing-hint'], 'model=' + self.profile.model)
        self.assertIn(f"codex_cli_rs/{headers['version']}", headers['User-Agent'])
        self.assertEqual(headers['session-id'], first.codex_cache_key)
        self.assertEqual(headers['x-client-request-id'], first.codex_cache_key)
        self.assertEqual(headers['thread-id'], first.codex_cache_key)
        self.assertEqual(self.http_requests, [])

        with patch.object(codex, 'CODEX_REPLAY_ENABLED', False):
            self.call('third page')
            self.call('fourth page')
        self.assertEqual(self.connect.await_count, 1)
        for body in self.socket.frames[2:]:
            self.assertNotIn('previous_response_id', body)
            self.assertTrue(all(item['type'] == 'message' and 'id' not in item for item in body['input']))
            self.assertEqual(body['input'][1]['role'], 'assistant')
            self.assertEqual(body['input'][1]['content'][0]['text'], first.content)

    def test_ocr_uses_websocket_by_default(self) -> None:
        with patch.object(LLMOCR, 'params', copy.deepcopy(LLMOCR.params)):
            requester = LLMOCR()
            requester.set_stop_event(threading.Event())
            self.addCleanup(requester.set_stop_event, None)
            requester._respect_delay = Mock()
            result = requester.request_chat_completion(self.profile, {
                'model': self.profile.model, 'messages': [{'role': 'user', 'content': 'Read this page.'}],
            })
        self.assertEqual(result.content, '{"1":"hello"}')
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(self.http_requests, [])

    @patch.dict(os.environ, {'BALLOONTRANS_CODEX_WEBSOCKET': '0'})
    def test_diagnostic_transport_switch_keeps_job_identity(self) -> None:
        first = self.call('first page')
        self.assertEqual(self.connect.await_count, 0)
        self.assertEqual(len(self.http_requests), 1)
        for header in ('session-id', 'thread-id', 'x-client-request-id'):
            self.assertEqual(self.http_requests[0].headers[header], first.codex_cache_key)
        os.environ['BALLOONTRANS_CODEX_WEBSOCKET'] = '1'
        second = self.call('second page')
        self.assertEqual(second.codex_cache_key, first.codex_cache_key)
        self.assertEqual(self.connect.await_count, 1)
        self.assertNotIn('previous_response_id', self.socket.frames[0])
        os.environ['BALLOONTRANS_CODEX_WEBSOCKET'] = '0'
        third = self.call('third page')
        self.assertEqual(third.codex_cache_key, first.codex_cache_key)
        self.socket.transport.abort.assert_called_once()
        self.assertEqual(len(self.http_requests), 2)
        self.assertEqual(self.connect.await_count, 1)
        self.assertNotIn('previous_response_id', json.loads(self.http_requests[-1].content))

    @patch.dict(os.environ)
    def test_obsolete_saved_checkbox_is_discarded_and_websocket_is_automatic(self) -> None:
        os.environ.pop('BALLOONTRANS_CODEX_WEBSOCKET', None)
        for module_type in (LLMTranslator, LLMOCR):
            with self.subTest(module=module_type.__name__), \
                    patch.object(module_type, 'params', copy.deepcopy(module_type.params)), \
                    self.assertLogs(codex.LOGGER, level='WARNING') as captured:
                args = ('日本語', 'English') if module_type is LLMTranslator else ()
                requester = module_type(*args, **{'codex websocket': False})
            self.assertIn('codex websocket', '\n'.join(captured.output))
            self.assertNotIn('codex websocket', requester.params)
            requester.set_stop_event(threading.Event())
            self.addCleanup(requester.set_stop_event, None)
            requester._respect_delay = Mock()
            requester.request_chat_completion(self.profile, {
                'model': self.profile.model, 'messages': [{'role': 'user', 'content': 'Read.'}],
            })
        self.assertEqual(self.connect.await_count, 2)
        self.assertEqual(self.http_requests, [])

    def test_handshake_turn_state_replays_for_one_turn_without_breaking_continuation(self) -> None:
        self.socket.response.headers['X-Codex-Turn-State'] = 'handshake-route'
        turn = codex.CodexTurnState()
        self.call('first page', turn=turn)
        self.call('retry/continuation', turn=turn)
        self.call('next page')
        first, second, third = self.socket.frames
        self.assertEqual(first['client_metadata'], {'x-codex-turn-state': 'handshake-route'})
        self.assertEqual(second['client_metadata'], first['client_metadata'])
        self.assertEqual(second['previous_response_id'], 'resp_1')
        self.assertEqual(third['previous_response_id'], 'resp_2')
        self.assertNotIn('client_metadata', third)
        self.assertNotIn('x-codex-turn-state', self.connect.call_args.kwargs['additional_headers'])
        self.assertEqual(self.connect.await_count, 1)

    def test_metadata_turn_state_survives_reconnect_but_next_turn_does_not_inherit_it(self) -> None:
        def respond(body: dict) -> list:
            return [{'type': 'response.metadata', 'headers': {'X-Codex-Turn-State': ['metadata-route']}},
                    *self.socket.complete(body)]

        self.socket.respond = respond
        turn = codex.CodexTurnState()
        self.call(turn=turn)
        self.assertNotIn('client_metadata', self.socket.frames[0])
        self.socket.state = State.CLOSED
        replacement = Socket()
        replacement.response.headers['x-codex-turn-state'] = 'replacement-route'
        self.connect.return_value = replacement
        self.call(turn=turn)
        self.assertEqual(self.connect.call_args.kwargs['additional_headers']['x-codex-turn-state'], 'metadata-route')
        self.assertEqual(replacement.frames[0]['client_metadata'], {'x-codex-turn-state': 'metadata-route'})
        self.call()
        self.assertNotIn('client_metadata', replacement.frames[1])
        self.assertEqual(replacement.frames[1]['previous_response_id'], 'resp_1')

    def test_native_handshake_headers_allow_multiple_cookies_and_keep_first_routing_value(self) -> None:
        from websockets.datastructures import Headers
        for routes in ([], [('X-Codex-Turn-State', 'route-one'), ('X-Codex-Turn-State', 'route-two')]):
            with self.subTest(routing=bool(routes)):
                self.requester.set_stop_event(threading.Event())
                self.socket = Socket()
                self.socket.response.headers = Headers([('Set-Cookie', 'cookie-one'),
                                                        ('Set-Cookie', 'cookie-two'), *routes])
                self.connect.return_value = self.socket
                self.call()
                if routes:
                    self.assertEqual(self.socket.frames[0]['client_metadata'], {'x-codex-turn-state': 'route-one'})
                else:
                    self.assertNotIn('client_metadata', self.socket.frames[0])
                self.assertEqual(self.http_requests, [])

    def test_connection_limit_sse_fallback_replays_turn_state(self) -> None:
        self.socket.response.headers['x-codex-turn-state'] = 'fallback-route'
        self.socket.respond = lambda body: [
            {'type': 'response.metadata', 'headers': {}},
            {'type': 'error', 'error': {'code': 'websocket_connection_limit_reached'}},
        ]
        self.call()
        self.assertEqual(self.http_requests[0].headers['x-codex-turn-state'], 'fallback-route')
        self.call()
        self.assertNotIn('x-codex-turn-state', self.http_requests[1].headers)

    def test_eviction_rebuilds_full_input_on_the_same_connection(self) -> None:
        self.call('first page')
        self.call('second page')
        self.messages = self.messages[2:]
        self.call('third page')
        self.assertEqual(self.connect.await_count, 1)
        self.assertNotIn('previous_response_id', self.socket.frames[-1])
        self.assertEqual(self.socket.frames[-1]['input'][0]['content'][0]['text'], 'second page')
        self.call('fourth page')
        self.assertEqual(self.socket.frames[-1]['previous_response_id'], 'resp_3')

    def test_effort_change_requires_full_input(self) -> None:
        self.call()
        self.profile.thinking_level = 'high'
        self.call()
        self.assertNotIn('previous_response_id', self.socket.frames[-1])
        self.assertEqual(self.socket.frames[-1]['reasoning'], {'effort': 'high'})

    def test_missing_response_reconnects_with_full_history_once(self) -> None:
        self.call()
        self.socket.respond = lambda body: [{'type': 'error', 'error': {
            'code': 'previous_response_not_found', 'message': 'Response is unavailable.'}}]
        replacement = Socket()
        self.connect.return_value = replacement
        self.call('next page')
        self.assertTrue(self.socket.closed)
        self.assertEqual(self.connect.await_count, 2)
        self.assertNotIn('previous_response_id', replacement.frames[0])
        self.assertEqual(len(replacement.frames[0]['input']), 4)
        self.assertEqual(self.http_requests, [])

    def test_connect_failure_falls_back_once_per_job(self) -> None:
        self.connect.side_effect = OSError('unavailable')
        self.call()
        with patch('ballontranslator.modules.responses_ws.threading.Thread', wraps=threading.Thread) as worker:
            self.call()
        worker.assert_not_called()
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(len(self.http_requests), 2)
        bodies = [json.loads(request.content) for request in self.http_requests]
        self.assertEqual(bodies[0]['prompt_cache_key'], bodies[1]['prompt_cache_key'])
        self.assertNotIn('previous_response_id', bodies[1])

    @patch.dict(os.environ, {'BALLOONTRANS_CODEX_WEBSOCKET': '0'})
    def test_sse_retains_only_routing_cookies_and_honors_scope_rotation_and_deletion(self) -> None:
        headers = [
            [('set-cookie', '__oailb=route-one; Path=/; Secure'),
             ('set-cookie', 'account_session=private; Path=/; Secure'),
             ('set-cookie', '__cf_bm=other-path; Path=/auth; Secure'),
             ('set-cookie', '__cflb=other-host; Domain=example.com; Path=/; Secure')],
            [('set-cookie', '__oailb=route-two; Path=/; Secure')],
            [('set-cookie', '__oailb=deleted; Max-Age=0; Path=/; Secure')],
            [],
        ]

        def respond(request: httpx.Request) -> httpx.Response:
            index = len(self.http_requests)
            self.http_requests.append(request)
            return httpx.Response(200, headers=headers[index], text=sse(completion()))

        with patch.object(self, 'http', side_effect=respond):
            for _ in headers:
                self.call()
        self.assertEqual([request.headers.get('cookie') for request in self.http_requests],
                         [None, '__oailb=route-one', '__oailb=route-two', None])
        session = next(iter(self.requester._codex_sessions.values()))
        self.assertNotIn('account_session', session._cookies)

    def test_websocket_cookie_survives_reconnect_and_error_handshake_sse_fallback(self) -> None:
        from websockets.datastructures import Headers
        from websockets.exceptions import InvalidStatus
        from websockets.http11 import Response

        self.socket.response.headers = Headers([
            ('Set-Cookie', '__oailb=ws-route; Path=/; Secure'),
            ('Set-Cookie', 'account_session=private; Path=/; Secure'),
        ])
        self.call()
        self.socket.state = State.CLOSED
        self.connect.side_effect = InvalidStatus(Response(503, 'Unavailable', Headers([
            ('Set-Cookie', '__oailb=fallback-route; Path=/; Secure'),
        ])))
        self.call()
        self.assertEqual(self.connect.call_args.kwargs['additional_headers']['Cookie'], '__oailb=ws-route')
        self.assertEqual(self.http_requests[-1].headers['cookie'], '__oailb=fallback-route')
        self.call()
        self.assertEqual(self.http_requests[-1].headers['cookie'], '__oailb=fallback-route')
        self.assertEqual(self.connect.await_count, 2)

    @patch.dict(os.environ, {'BALLOONTRANS_CODEX_WEBSOCKET': '0'})
    def test_routing_cookies_do_not_cross_job_account_or_proxy_changes(self) -> None:
        for boundary in ('job', 'account', 'proxy'):
            with self.subTest(boundary=boundary):
                self.requester.set_stop_event(threading.Event())
                self.http_requests.clear()

                def respond(request: httpx.Request) -> httpx.Response:
                    self.http_requests.append(request)
                    return httpx.Response(200, headers={'set-cookie': '__oailb=private-route; Path=/; Secure'},
                                          text=sse(completion()))

                with patch.object(self, 'http', side_effect=respond):
                    self.call()
                    old_session = next(iter(self.requester._codex_sessions.values()))
                    self.assertIn('__oailb', old_session._cookies)
                    if boundary == 'job':
                        self.requester.set_stop_event(threading.Event())
                    elif boundary == 'account':
                        self.account.invalidate()
                    else:
                        self.requester.set_param_value('proxy', 'http://proxy.example:8080')
                    self.call()
                self.assertNotIn('cookie', self.http_requests[-1].headers)
                self.assertEqual(len(old_session._cookies), 0)
                old_session.capture_cookies(httpx.Headers({'set-cookie': '__oailb=late; Path=/; Secure'}))
                self.assertEqual(len(old_session._cookies), 0)

    def test_already_closed_connection_reconnects_before_sending(self) -> None:
        self.call()
        self.socket.state = State.CLOSED
        replacement = Socket()
        self.connect.return_value = replacement
        self.call()
        self.assertEqual(self.connect.await_count, 2)
        self.assertNotIn('previous_response_id', replacement.frames[0])
        self.assertEqual(self.http_requests, [])

    def test_disconnect_after_send_without_events_does_not_replay(self) -> None:
        self.socket.respond = lambda body: [OSError('disconnected')]
        with self.assertRaises(RuntimeError):
            self.call()
        self.assertEqual(len(self.socket.frames), 1)
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(self.http_requests, [])

    def test_refreshed_credentials_reconnect_before_sending(self) -> None:
        first = self.call()
        self.account._credentials['access_token'] = 'refreshed-access'
        replacement = Socket()
        self.connect.return_value = replacement
        second = self.call()
        self.assertEqual(self.connect.await_count, 2)
        self.assertTrue(self.socket.closed)
        self.assertEqual(self.connect.call_args.kwargs['additional_headers']['Authorization'], 'Bearer refreshed-access')
        self.assertEqual(second.codex_cache_key, first.codex_cache_key)
        self.assertNotIn('previous_response_id', replacement.frames[0])
        self.assertEqual(self.http_requests, [])

    def test_replacing_run_during_throttle_does_not_send_old_request(self) -> None:
        self.requester._respect_delay.side_effect = lambda: self.requester.set_stop_event(threading.Event())
        with self.assertRaises(LLMRequestStopped):
            self.call()
        self.connect.assert_not_called()
        self.assertEqual(self.http_requests, [])

    def test_idle_expiry_preserves_job_routing_cookies(self) -> None:
        self.socket.response.headers['set-cookie'] = '__oailb=job-route; Path=/; Secure'
        first = self.call()
        session = next(iter(self.requester._codex_sessions.values()))
        session._last_used -= 301
        session._expire()
        replacement = Socket()
        self.connect.return_value = replacement
        second = self.call()
        self.assertEqual(first.codex_cache_key, second.codex_cache_key)
        self.assertEqual(self.connect.call_args.kwargs['additional_headers'].get('Cookie'), '__oailb=job-route')
        self.assertNotIn('previous_response_id', replacement.frames[0])

    def test_failure_after_stream_start_never_replays_through_sse(self) -> None:
        self.socket.respond = lambda body: [{'type': 'response.created'}, OSError('private-request-url-with-token')]
        with self.assertLogs(codex.LOGGER, level='DEBUG') as captured, self.assertRaises(RuntimeError):
            self.call()
        self.assertIn('Codex request failure: exception=OSError', '\n'.join(captured.output))
        self.assertNotIn('private-request-url-with-token', '\n'.join(captured.output))
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(self.http_requests, [])
        self.assertTrue(self.socket.closed)

    def test_provider_rejection_does_not_fall_back(self) -> None:
        self.socket.respond = lambda body: [{'type': 'error', 'error': {
            'code': 'invalid_parameter', 'param': 'reasoning.context', 'message': 'Unsupported.'}}]
        with self.assertRaises(LLMUserActionRequiredError):
            self.call()
        self.assertEqual(self.http_requests, [])

    def test_invalid_handshake_uses_sse_without_repeating_setup(self) -> None:
        from websockets.exceptions import InvalidHandshake
        self.connect.side_effect = InvalidHandshake('unsupported upgrade')
        self.call()
        self.call()
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(len(self.http_requests), 2)

    def test_unauthorized_handshake_renews_once_with_same_identity(self) -> None:
        from websockets.datastructures import Headers
        from websockets.exceptions import InvalidStatus
        from websockets.http11 import Response
        self.connect.side_effect = [InvalidStatus(Response(401, 'Unauthorized', Headers())), self.socket]

        def renew(request: httpx.Request) -> httpx.Response:
            self.http_requests.append(request)
            return httpx.Response(200, json={'access_token': 'new-access', 'refresh_token': 'new-refresh',
                                            'expires_in': 3600})

        with patch.object(self, 'http', side_effect=renew), patch.object(self.account, '_save'):
            result = self.call()
        self.assertEqual(self.connect.await_count, 2)
        self.assertEqual(len(self.http_requests), 1)
        self.assertEqual(self.http_requests[0].url.path, '/oauth/token')
        first, second = [call.kwargs['additional_headers'] for call in self.connect.call_args_list]
        self.assertEqual(first['session-id'], second['session-id'])
        self.assertEqual(second['session-id'], result.codex_cache_key)
        self.assertEqual(second['Authorization'], 'Bearer new-access')

    def test_invalid_proxy_uses_sse_without_repeating_websocket_setup(self) -> None:
        from websockets.exceptions import InvalidProxy
        self.connect.side_effect = InvalidProxy('private-proxy-url', 'unsupported scheme')
        with self.assertLogs(codex.LOGGER, level='DEBUG') as captured:
            self.call()
            self.call()
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(len(self.http_requests), 2)
        self.assertNotIn('private-proxy-url', '\n'.join(captured.output))

    def test_status_only_websocket_auth_error_renews_token_and_reconnects(self) -> None:
        replacement = Socket()
        self.socket.respond = lambda body: [{'type': 'error', 'status': 401,
                                            'error': {'message': 'Request rejected.'}}]
        self.connect.side_effect = [self.socket, replacement]

        def renew(request: httpx.Request) -> httpx.Response:
            self.http_requests.append(request)
            self.assertEqual(request.url.path, '/oauth/token')
            self.assertNotIn('cookie', request.headers)
            self.assertNotIn('x-codex-routing-hint', request.headers)
            return httpx.Response(200, json={'access_token': 'renewed', 'expires_in': 3600})

        self.socket.response.headers['Set-Cookie'] = '__oailb=retry-route; Path=/; Secure'
        with patch.object(self, 'http', side_effect=renew), patch.object(self.account, '_save'):
            self.call()
        self.assertEqual(self.connect.await_count, 2)
        headers = self.connect.call_args.kwargs['additional_headers']
        self.assertEqual(headers['Authorization'], 'Bearer renewed')
        self.assertEqual(headers['Cookie'], '__oailb=retry-route')
        self.assertEqual(len(self.http_requests), 1)

    def test_websocket_status_code_alias_preserves_permission_failure(self) -> None:
        self.socket.respond = lambda body: [{'type': 'error', 'status_code': 403,
                                            'error': {'message': 'Request rejected.'}}]
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'permission'):
            self.call()
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(self.http_requests, [])

    def test_proxy_change_reconnects_without_changing_job_identity(self) -> None:
        first = self.call()
        replacement = Socket()
        self.connect.return_value = replacement
        with patch.object(self.requester, 'get_param_value', return_value='http://proxy.example:8080'):
            second = self.call()
        self.socket.transport.abort.assert_called_once()
        self.assertEqual(first.codex_cache_key, second.codex_cache_key)
        self.assertEqual(self.connect.call_args.kwargs['proxy'], 'http://proxy.example:8080')
        self.assertNotIn('previous_response_id', replacement.frames[0])

    def test_done_event_is_accepted_as_completion(self) -> None:
        def respond(body: dict) -> list:
            events = self.socket.complete(body)
            events[-1]['type'] = 'response.done'
            return events

        self.socket.respond = respond
        result = self.call()
        self.assertEqual(result.content, '{"1":"hello"}')

    def test_done_failure_preserves_provider_error_classification(self) -> None:
        self.socket.respond = lambda body: [{'type': 'response.done', 'response': {
            'status': 'failed', 'error': {'code': 'usage_limit_reached', 'message': 'quota'},
        }}]
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'usage limit'):
            self.call()
        self.assertEqual(self.http_requests, [])

    def test_filtered_incomplete_response_is_not_called_token_truncation(self) -> None:
        for kind in ('response.incomplete', 'response.done'):
            for details in ({'reason': 'content_filter'}, None, {}):
                with self.subTest(kind=kind, details=details):
                    self.socket.respond = lambda body: [{'type': kind, 'response': {
                        'status': 'incomplete', 'incomplete_details': details,
                    }}]
                    with self.assertRaisesRegex(LLMUserActionRequiredError, 'did not complete') as caught:
                        self.call()
                    self.assertNotIn('truncated', str(caught.exception))
        self.assertEqual(self.http_requests, [])
        self.assertEqual(self.http_requests, [])

    def test_connection_limit_retries_once_then_falls_back(self) -> None:
        self.socket.respond = lambda body: [{'type': 'error', 'error': {
            'code': 'websocket_connection_limit_reached', 'message': 'Limit reached.'}}]
        self.call()
        self.call()
        self.assertEqual(self.connect.await_count, 2)
        self.assertEqual(len(self.http_requests), 2)

    def test_handshake_quota_error_is_not_reported_as_a_connection_problem(self) -> None:
        from websockets.datastructures import Headers
        from websockets.exceptions import InvalidStatus
        from websockets.http11 import Response

        body = json.dumps({'error': {'code': 'insufficient_quota', 'message': 'Quota exhausted.'}}).encode()
        self.connect.side_effect = InvalidStatus(Response(429, 'Rejected', Headers(), body))
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'usage limit'):
            self.call()
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(self.http_requests, [])

    def test_new_job_closes_previous_connection_and_changes_identity(self) -> None:
        first = self.call()
        loop = self.socket.loops[0]
        self.requester.set_stop_event(threading.Event())
        self.assertTrue(loop.is_closed())
        self.socket.transport.abort.assert_called_once()
        replacement = Socket()
        self.connect.return_value = replacement
        second = self.call()
        self.assertNotEqual(first.codex_cache_key, second.codex_cache_key)
        self.assertNotIn('previous_response_id', replacement.frames[0])

    def test_account_change_closes_old_socket_and_drops_continuation(self) -> None:
        first = self.call()
        self.account.invalidate()
        replacement = Socket()
        self.connect.return_value = replacement
        second = self.call()
        self.assertTrue(self.socket.loops[0].is_closed())
        self.socket.transport.abort.assert_called_once()
        self.assertNotEqual(first.codex_cache_key, second.codex_cache_key)
        self.assertNotIn('previous_response_id', replacement.frames[0])
        self.assertEqual([item['type'] for item in replacement.frames[0]['input']],
                         ['message', 'message', 'message'])

    def test_account_change_before_transport_cannot_reuse_old_authenticated_socket(self) -> None:
        self.call()
        require_sign_in = self.account.require_sign_in

        def change_account(stop_event) -> None:
            self.account.invalidate()
            require_sign_in(stop_event)

        with patch.object(self.account, 'require_sign_in', side_effect=change_account):
            with self.assertRaises(LLMRequestStopped):
                self.call()
        self.assertEqual(len(self.socket.frames), 1)
        self.assertEqual(self.connect.await_count, 1)

    def test_real_socket_and_loop_survive_separate_worker_threads(self) -> None:
        from websockets.sync.server import serve
        from websockets.exceptions import ConnectionClosed
        self.connect_patcher.stop()
        frames = []
        connections = []
        errors = []
        finished = threading.Event()

        def handle(connection) -> None:
            connections.append(connection)
            try:
                for data in connection:
                    body = json.loads(data)
                    frames.append(body)
                    self.socket.frames.append(body)
                    for event in self.socket.complete(body):
                        connection.send(json.dumps(event))
            except ConnectionClosed:
                pass
            finally:
                finished.set()

        def run_page(page: str) -> None:
            try:
                self.call(page)
            except BaseException as error:
                errors.append(error)

        def response_headers(connection, request, response):
            response.headers['Set-Cookie'] = 'cookie-one'
            response.headers['Set-Cookie'] = 'cookie-two'
            response.headers['x-codex-turn-state'] = 'handshake-route'
            return response

        with serve(handle, '127.0.0.1', 0, ping_interval=None, process_response=response_headers) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.socket.getsockname()[1]
            with patch.object(codex, 'API_URL', f'http://127.0.0.1:{port}/codex'):
                try:
                    for page in ('first page', 'second page'):
                        worker = threading.Thread(target=run_page, args=(page,), daemon=True)
                        worker.start()
                        worker.join(3)
                        self.assertFalse(worker.is_alive())
                finally:
                    self.requester.set_stop_event(None)
            self.assertTrue(finished.wait(1))
            server.shutdown()
            thread.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(len(connections), 1)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]['client_metadata'], {'x-codex-turn-state': 'handshake-route'})
        self.assertNotIn('client_metadata', frames[1])
        self.assertEqual(frames[1]['previous_response_id'], 'resp_1')
        self.assertEqual(len(frames[1]['input']), 1)
        self.assertEqual(self.http_requests, [])

    def test_real_handshake_does_not_follow_redirects(self) -> None:
        from websockets.sync.server import serve
        self.connect_patcher.stop()
        paths = []

        def redirect(connection, request):
            paths.append(request.path)
            response = connection.respond(302, 'Moved.')
            response.headers['Location'] = '/redirect'
            return response

        with serve(lambda connection: None, '127.0.0.1', 0, process_request=redirect) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.socket.getsockname()[1]
            with patch.object(codex, 'API_URL', f'http://127.0.0.1:{port}/codex'):
                self.call()
            server.shutdown()
            thread.join(3)
        self.assertEqual(paths, ['/codex/responses'])
        self.assertEqual(len(self.http_requests), 1)

    def test_real_disconnect_after_receiving_request_does_not_replay(self) -> None:
        from websockets.sync.server import serve

        self.connect_patcher.stop()
        frames = []

        def handle(connection) -> None:
            frames.append(json.loads(connection.recv()))
            connection.close()

        with serve(handle, '127.0.0.1', 0, ping_interval=None) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.socket.getsockname()[1]
            try:
                with patch.object(codex, 'API_URL', f'http://127.0.0.1:{port}/codex'):
                    with self.assertRaises(RuntimeError):
                        self.call()
                session = next(iter(self.requester._codex_sessions.values()))
                self.assertIsNone(session._loop)
                self.assertIsNone(session._thread)
                self.assertEqual(len(frames), 1)
                self.assertEqual(self.http_requests, [])
            finally:
                self.requester.set_stop_event(None)
                server.shutdown()
                thread.join(3)

    def test_real_socket_answers_server_ping_between_page_workers(self) -> None:
        from websockets.sync.server import serve
        from websockets.exceptions import ConnectionClosed
        self.connect_patcher.stop()
        connections = []
        errors = []

        def handle(connection) -> None:
            connections.append(connection)
            try:
                for data in connection:
                    body = json.loads(data)
                    self.socket.frames.append(body)
                    for event in self.socket.complete(body):
                        connection.send(json.dumps(event))
            except ConnectionClosed:
                pass

        def run_page() -> None:
            try:
                self.call()
            except BaseException as error:
                errors.append(error)

        with serve(handle, '127.0.0.1', 0, ping_interval=None) as server:
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            port = server.socket.getsockname()[1]
            try:
                with patch.object(codex, 'API_URL', f'http://127.0.0.1:{port}/codex'):
                    worker = threading.Thread(target=run_page, daemon=True)
                    worker.start()
                    worker.join(3)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(errors, [])
                    session_thread = next(iter(self.requester._codex_sessions.values()))._thread
                    self.assertTrue(session_thread.is_alive())
                    # The page worker is gone. A healthy retained connection must
                    # still service control frames without another user request.
                    pong = connections[0].ping()
                    self.assertTrue(pong.wait(1), 'Idle Codex socket did not answer the server ping')
                    worker = threading.Thread(target=run_page, daemon=True)
                    worker.start()
                    worker.join(3)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(len(connections), 1)
                    self.assertEqual(self.socket.frames[-1]['previous_response_id'], 'resp_1')
            finally:
                self.requester.set_stop_event(None)
                server.shutdown()
                server_thread.join(3)
        self.assertFalse(session_thread.is_alive())

    def test_idle_and_age_expiry_preserve_job_identity(self) -> None:
        first = self.call()
        session = next(iter(self.requester._codex_sessions.values()))
        session._last_used -= 301
        session._expire()
        self.assertFalse(session.closed)
        replacement = Socket()
        self.connect.return_value = replacement
        second = self.call()
        self.assertEqual(first.codex_cache_key, second.codex_cache_key)
        self.assertNotIn('previous_response_id', replacement.frames[0])
        current_session = next(iter(self.requester._codex_sessions.values()))
        current_session._connected_at -= 55 * 60
        third_socket = Socket()
        self.connect.return_value = third_socket
        self.call()
        self.assertTrue(replacement.closed)
        self.assertNotIn('previous_response_id', third_socket.frames[0])

    def test_stopping_cancels_a_socket_wait_and_closes_the_loop(self) -> None:
        self.socket.respond = lambda body: [{'type': 'response.created'}]
        stop = self.requester.stop_event
        timer = threading.Timer(0.05, stop.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(LLMRequestStopped):
            self.call()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(self.socket.loops[0].is_closed())
        self.assertTrue(self.socket.closed)
        self.assertEqual(self.http_requests, [])

    def test_repeated_close_cannot_interrupt_session_cleanup(self) -> None:
        self.call()
        session = next(iter(self.requester._codex_sessions.values()))
        loop, loop_thread = session._loop, session._thread
        cleaning_up = threading.Event()
        errors = []

        async def pending_work() -> None:
            try:
                await asyncio.Future()
            finally:
                cleaning_up.set()
                await asyncio.sleep(0.05)

        def close() -> None:
            try:
                session.close()
            except BaseException as error:
                errors.append(error)

        # Keep the task alive until shutdown cancels and drains it.
        pending = asyncio.run_coroutine_threadsafe(pending_work(), loop)
        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        try:
            self.assertTrue(cleaning_up.wait(1))
            session.close()
            closer.join(2)
            self.assertFalse(closer.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(pending.cancelled())
            self.assertFalse(loop_thread.is_alive())
            self.assertTrue(loop.is_closed())
        finally:
            closer.join(2)
            session.close()

    def test_close_before_scheduled_request_starts_releases_its_coroutine(self) -> None:
        self.call()
        session = next(iter(self.requester._codex_sessions.values()))
        loop, loop_thread = session._loop, session._thread
        blocked, release, queued = threading.Event(), threading.Event(), threading.Event()
        errors = []
        operation = asyncio.sleep(0)
        submit = asyncio.run_coroutine_threadsafe

        def block_loop() -> None:
            blocked.set()
            release.wait(3)

        def enqueue(coroutine, target_loop):
            future = submit(coroutine, target_loop)
            queued.set()
            return future

        def request() -> None:
            try:
                session.run(operation, self.requester.stop_event)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=request, daemon=True)
        loop.call_soon_threadsafe(block_loop)
        try:
            self.assertTrue(blocked.wait(1))
            with patch('ballontranslator.modules.responses_ws.asyncio.run_coroutine_threadsafe', new=enqueue):
                worker.start()
                self.assertTrue(queued.wait(1))
                session.close()
                release.set()
                worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], LLMRequestStopped)
            self.assertEqual(inspect.getcoroutinestate(operation), inspect.CORO_CLOSED)
            self.assertTrue(loop.is_closed())
            self.assertFalse(loop_thread.is_alive())
        finally:
            release.set()
            if worker.ident is not None:
                worker.join(2)
            session.close()
            operation.close()

    def test_changing_runs_cancels_an_inflight_request(self) -> None:
        self.socket.respond = lambda body: [{'type': 'response.created'}]
        timer = threading.Timer(0.05, self.requester.set_stop_event, args=(threading.Event(),))
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(LLMRequestStopped):
            self.call()
        self.assertTrue(self.socket.loops[0].is_closed())
        self.assertEqual(self.http_requests, [])
