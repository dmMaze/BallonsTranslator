import copy
import os
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from ballontranslator.modules import codex
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMOutputLimitError, LLMRequestStopped
from ballontranslator.modules.llm_chat import LLMChatRequestError
from ballontranslator.modules.ocr.ocr_llm import LLMOCR
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils.llm_profiles import default_profile
from test_codex_websocket import ConnectAttempt, Socket


class OpenAIResponsesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = default_profile('OpenAI')
        self.profile.api_key = 'sk-test-only'
        self.profile.model = 'gpt-5.6'
        self.profile.thinking_level = 'medium'
        self.profile.json_schema_response_format = True
        self.http_requests = []
        self.socket = Socket()
        self.connect = AsyncMock(return_value=self.socket)
        self.params = patch.object(LLMTranslator, 'params', copy.deepcopy(LLMTranslator.params))
        self.params.start()
        self.addCleanup(self.params.stop)
        self.requester = LLMTranslator('日本語', 'English')
        self.requester.set_stop_event(threading.Event())
        self.requester._respect_delay = Mock()
        for patcher in (
            patch('websockets.asyncio.client.connect', side_effect=lambda *args, **kwargs: ConnectAttempt(self.connect(*args, **kwargs))),
            patch.object(self.requester, '_http_client', side_effect=lambda proxy: httpx.Client(transport=httpx.MockTransport(self.http))),
            patch.object(codex, '_latest_client_version', side_effect=AssertionError('No subscription identity on API calls')),
            patch.object(codex.account, 'require_sign_in', side_effect=AssertionError('No subscription account on API calls')),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.requester.set_stop_event, None)

    def http(self, request: httpx.Request) -> httpx.Response:
        self.http_requests.append(request)
        return httpx.Response(200, json={
            'id': 'chat-one', 'object': 'chat.completion', 'created': 0, 'model': self.profile.model,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'HTTP result'}, 'finish_reason': 'stop'}],
        })

    def call(self, messages=None, **overrides):
        messages = messages or [{'role': 'system', 'content': 'Return JSON.'}, {'role': 'user', 'content': 'Read.'}]
        args = self.requester._api_args(self.profile, messages)
        args.update(overrides)
        return self.requester.request_chat_completion(self.profile, args)

    def test_public_headers_payload_and_usage_preserve_contract_without_codex_state(self) -> None:
        messages = [{'role': 'system', 'content': 'Translate JSON.'}, {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Page'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA==', 'detail': 'high'}}]}]
        before = copy.deepcopy(messages)
        result = self.call(messages)
        self.assertEqual(messages, before)
        url = self.connect.call_args.args[0]
        headers = {k.lower(): v for k, v in self.connect.call_args.kwargs['additional_headers'].items()}
        self.assertEqual(url, 'wss://api.openai.com/v1/responses')
        self.assertEqual(headers['authorization'], 'Bearer sk-test-only')
        for key in ('chatgpt-account-id', 'cookie', 'originator', 'version', 'openai-beta', 'session-id', 'thread-id'):
            self.assertNotIn(key, headers)
        body = self.socket.frames[0]
        self.assertEqual(body['reasoning'], {'effort': 'medium'})
        self.assertEqual(body['max_output_tokens'], self.profile.max_tokens)
        self.assertEqual(body['text']['format']['type'], 'json_schema')
        self.assertEqual(body['prompt_cache_options'], {'mode': 'explicit', 'ttl': '30m'})
        self.assertEqual(body['input'][0]['content'][0]['prompt_cache_breakpoint'], {'mode': 'explicit'})
        self.assertEqual(body['input'][1]['content'][1], {'type': 'input_image', 'image_url': 'data:image/png;base64,AA==', 'detail': 'high'})
        self.assertFalse(body['store'])
        self.assertNotIn('stream', body)
        self.assertNotIn('client_metadata', body)
        self.assertEqual(result.usage.prompt_tokens, 10)
        self.assertEqual(result.usage.prompt_tokens_details, {'cached_tokens': 8})
        self.assertEqual(result.usage.completion_tokens_details, {'reasoning_tokens': 2})
        self.assertEqual(self.http_requests, [])

    def test_exact_history_continues_and_eviction_rebuilds_on_same_socket(self) -> None:
        self.profile.model = 'gpt-4.1'
        messages = [{'role': 'user', 'content': 'page one'}]
        first = self.call(messages)
        messages += [{'role': 'assistant', 'content': first.content}, {'role': 'user', 'content': 'page two'}]
        self.call(messages)
        self.assertEqual(self.socket.frames[-1]['previous_response_id'], 'resp_1')
        self.assertEqual(len(self.socket.frames[-1]['input']), 1)
        self.call([{'role': 'user', 'content': 'new window'}])
        self.assertNotIn('previous_response_id', self.socket.frames[-1])
        self.assertEqual(self.connect.await_count, 1)

    def test_setup_failure_uses_original_chat_request_once_per_run(self) -> None:
        self.connect.side_effect = OSError('offline')
        self.assertEqual(self.call().content, 'HTTP result')
        with patch('ballontranslator.modules.openai_responses.responses_payload',
                   side_effect=AssertionError('HTTP fallback must not rebuild a Responses payload')):
            self.assertEqual(self.call().content, 'HTTP result')
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(len(self.http_requests), 2)
        self.assertTrue(all(str(r.url) == 'https://api.openai.com/v1/chat/completions' for r in self.http_requests))

    def test_missing_public_continuation_recovers_flat_error_with_full_context(self) -> None:
        self.profile.model = 'gpt-4.1'
        messages = [{'role': 'user', 'content': 'page one'}]
        first = self.call(messages)
        messages += [{'role': 'assistant', 'content': first.content}, {'role': 'user', 'content': 'page two'}]
        self.socket.respond = lambda body: ([{'type': 'error', 'code': 'previous_response_not_found'}]
                                          if 'previous_response_id' in body else self.socket.complete(body))
        self.call(messages)
        self.assertEqual(self.connect.await_count, 2)
        self.assertNotIn('previous_response_id', self.socket.frames[-1])
        self.assertEqual(self.socket.frames[-1]['input'][0]['content'][0]['text'], 'page one')
        self.assertEqual(self.socket.frames[-1]['input'][-1]['content'][0]['text'], 'page two')
        self.assertEqual(self.http_requests, [])

    def test_invalid_completed_output_cannot_pollute_retry_history(self) -> None:
        self.profile.model = 'gpt-4.1'
        messages = [{'role': 'user', 'content': 'page one'}]
        first = self.call(messages)
        messages += [{'role': 'assistant', 'content': first.content}, {'role': 'user', 'content': 'page two'}]
        self.socket.respond = lambda body: [{'type': 'response.completed', 'response': {
            'status': 'completed', 'output': [{'type': 'reasoning', 'encrypted_content': 'opaque'}]}}]
        with self.assertRaises(RuntimeError):
            self.call(messages)
        self.socket.respond = self.socket.complete
        self.call(messages)
        self.assertEqual(len(self.socket.frames[-1]['input']), 3)
        self.assertNotIn('previous_response_id', self.socket.frames[-1])
        self.assertEqual(self.socket.frames[-1]['input'][-1]['content'][0]['text'], 'page two')

    def test_public_context_error_reaches_existing_context_recovery(self) -> None:
        from ballontranslator.modules.context.errors import is_context_length_error
        self.socket.respond = lambda body: [{'type': 'error', 'code': 'context_length_exceeded',
                                            'message': 'Input is too long.'}]
        with self.assertRaises(LLMChatRequestError) as caught:
            self.call()
        self.assertTrue(is_context_length_error(caught.exception.provider_error))
        self.assertEqual(self.http_requests, [])

    def test_midstream_failure_does_not_replay_via_http(self) -> None:
        self.socket.respond = lambda body: [{'type': 'response.created'}, OSError('secret-url')]
        with self.assertRaisesRegex(RuntimeError, 'OpenAI could not complete'):
            self.call()
        self.assertEqual(self.http_requests, [])
        self.assertTrue(self.socket.closed)

    def test_auth_and_permission_errors_remain_actionable_without_http_retry(self) -> None:
        from websockets.datastructures import Headers
        from websockets.exceptions import InvalidStatus
        from websockets.http11 import Response
        self.connect.side_effect = InvalidStatus(Response(401, 'Unauthorized', Headers()))
        with self.assertRaises(LLMApiKeyRequiredError):
            self.call()
        self.connect.side_effect = None
        self.socket.respond = lambda body: [{'type': 'error', 'status_code': 403, 'error': {'message': 'No access'}}]
        with self.assertRaises(LLMChatRequestError) as caught:
            self.call()
        self.assertEqual(caught.exception.provider_error.status_code, 403)
        self.assertEqual(self.http_requests, [])

    def test_output_limit_preserves_existing_actionable_error(self) -> None:
        self.socket.respond = lambda body: [{'type': 'response.incomplete', 'response': {
            'incomplete_details': {'reason': 'max_output_tokens'}}}]
        with self.assertRaises(LLMOutputLimitError):
            self.call()
        self.assertEqual(self.http_requests, [])

    def test_compatibility_endpoints_and_chat_only_controls_keep_http(self) -> None:
        for base_url, overrides in (
            ('https://api.deepseek.com', {}), ('https://openrouter.ai/api/v1', {}),
            ('https://api.openai.com.example/v1', {}), ('https://api.openai.com/v1', {'frequency_penalty': 0.5}),
        ):
            with self.subTest(base_url=base_url):
                self.profile.base_url = base_url
                self.assertEqual(self.call(**overrides).content, 'HTTP result')
        self.assertEqual(self.connect.await_count, 0)

    def test_empty_base_url_respects_sdk_environment_override(self) -> None:
        self.profile.base_url = ''
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://gateway.example/v1'}):
            self.assertEqual(self.call().content, 'HTTP result')
        self.assertEqual(self.connect.await_count, 0)
        self.assertEqual(self.http_requests[0].url.host, 'gateway.example')

    def test_credentials_profile_model_proxy_and_job_changes_close_previous_session(self) -> None:
        for boundary in ('key', 'profile', 'model', 'proxy', 'job', 'idle'):
            with self.subTest(boundary=boundary):
                self.call()
                old_session = self.requester._openai_session
                old_client = self.requester.client
                if boundary == 'key':
                    self.profile.api_key = 'sk-new-key'
                elif boundary == 'profile':
                    self.profile.id += '-new'
                elif boundary == 'model':
                    self.profile.model = 'gpt-4.1'
                elif boundary == 'proxy':
                    self.requester.set_param_value('proxy', 'http://proxy.example:8080')
                elif boundary == 'job':
                    self.requester.set_stop_event(threading.Event())
                else:
                    old_session._last_used -= 301
                    old_session._expire()
                self.call()
                self.assertTrue(old_session.closed)
                self.assertIsNot(self.requester._openai_session, old_session)
                self.assertNotIn('previous_response_id', self.socket.frames[-1])
                if boundary in ('key', 'proxy'):
                    self.assertTrue(old_client.is_closed())
                if boundary == 'idle':
                    self.assertEqual(self.requester._openai_session.cache_key, old_session.cache_key)
        self.assertEqual(self.connect.call_args.kwargs['additional_headers']['Authorization'], 'Bearer sk-new-key')

    def test_cancellation_stops_pending_public_request(self) -> None:
        self.socket.respond = lambda body: [{'type': 'response.created'}]
        timer = threading.Timer(0.05, self.requester.stop_event.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(LLMRequestStopped):
            self.call()
        self.assertTrue(self.requester._openai_session.closed)
        self.assertEqual(self.http_requests, [])

    def test_ocr_also_dispatches_through_public_websocket(self) -> None:
        with patch.object(LLMOCR, 'params', copy.deepcopy(LLMOCR.params)):
            ocr = LLMOCR()
            ocr.set_stop_event(threading.Event())
            ocr._respect_delay = Mock()
            self.addCleanup(ocr.set_stop_event, None)
            with patch.object(ocr, '_http_client', side_effect=lambda proxy: httpx.Client(transport=httpx.MockTransport(self.http))):
                result = ocr.request_chat_completion(self.profile, {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': 'Read.'}]})
        self.assertEqual(result.content, '{"1":"hello"}')
        self.assertEqual(self.connect.await_count, 1)
        self.assertEqual(self.http_requests, [])
