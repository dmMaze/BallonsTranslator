import asyncio
import base64
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import weakref
from typing import Callable, TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from ballontranslator.modules import codex, image_generation
from ballontranslator.modules.context.errors import ContextLengthError
from ballontranslator.modules.context.translation_context import PageSummary
from ballontranslator.modules.exceptions import LLMRequestStopped, LLMUserActionRequiredError
from ballontranslator.modules.llm_image import LLMImageRequester
from ballontranslator.modules.ocr.ocr_llm import LLMOCR
from ballontranslator.modules.translators.llm_translation_contract import TranslationPromptSpec
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils.config import ModuleConfig, ProgramConfig, json_dump_program_config, pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, default_profile, normalize_codex_models, profile_to_dict, sync_codex_profile

if TYPE_CHECKING:
    from ballontranslator.ui.codex_settings import CodexSettingsPanel

CATALOG = {'vision-model': {'modalities': ['text', 'image'], 'efforts': ['low', 'high']},
           'text-model': {'modalities': ['text'], 'efforts': ['none']}}
PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='
PNG_BYTES = base64.b64decode(PNG_BASE64)
PNG_URL = 'data:image/png;base64,' + PNG_BASE64


def jwt(claims: dict) -> str:
    return 'header.' + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip('=') + '.signature'


def tokens(access: str = 'access', refresh: str = 'refresh', expiry: float = None) -> dict:
    return {'access_token': access, 'refresh_token': refresh, 'account_id': 'account-one',
            'expires_at': time.time() + 3600 if expiry is None else expiry, 'email': 'demo@example.com'}


def completion(text: str = '{"1":"hello"}') -> dict:
    return {'type': 'response.completed', 'response': {
        'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
            'phase': 'final_answer', 'content': [{'type': 'output_text', 'text': text}]}],
        'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15,
                  'input_tokens_details': {'cached_tokens': 8}, 'output_tokens_details': {'reasoning_tokens': 2}},
    }}


def image_completion() -> dict:
    return {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
        {'type': 'image_generation_call', 'id': 'image-one', 'status': 'completed', 'result': PNG_BASE64},
    ]}}


def sse(*events: dict) -> str:
    return ': keepalive\n\n' + ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)


class CodexHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.account = codex.CodexAccount()
        self.account._credentials = tokens()
        self.account._loaded = True
        self.path = Path(self.directory.name) / 'http-auth.json'
        self.requests = []
        self.events = [completion()]
        self.responder = self.respond
        self.profile = default_codex_profile()
        self.profile.model = self.profile.vision_model = 'vision-model'
        sync_codex_profile(self.profile, CATALOG)
        for patcher in (
            patch.object(codex, 'account', self.account),
            patch.object(codex, 'CODEX_REPLAY_ENABLED', True),
            patch.object(codex, '_client_version_checked_at', time.monotonic()),
            patch.object(self.account, '_path', return_value=self.path),
            patch.object(codex, '_system_keyring', side_effect=ImportError),
            patch.object(codex, '_http_client', side_effect=lambda proxy='': httpx.AsyncClient(transport=httpx.MockTransport(self.responder))),
            patch.object(codex.CodexChatSession, 'request', new=AsyncMock(return_value=None)),
            patch.object(pcfg.module, 'codex_models', copy.deepcopy(CATALOG)),
            patch.object(LLMTranslator, 'params', copy.deepcopy(LLMTranslator.params)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, text=sse(*self.events), headers={'Content-Type': 'text/event-stream'})

    def args(self) -> dict:
        return {'model': 'vision-model', 'messages': [
            {'role': 'system', 'content': 'Translate numbered blocks.'},
            {'role': 'system', 'content': 'Known names.'},
            {'role': 'user', 'content': 'old source'},
            {'role': 'assistant', 'content': 'old translation'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'current page'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc', 'detail': 'high'}}]},
        ], 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'translations', 'strict': True, 'schema': {'type': 'object'}}}}

    def request(self, stop: threading.Event = None):
        return codex.request_chat_completion(self.profile, self.args(), stop, 'job-cache-key')

    def test_page_retries_replay_first_turn_state_and_identical_next_page_starts_fresh(self) -> None:
        for source in ('response-header', 'metadata-event'):
            with self.subTest(source=source):
                self.requests.clear()
                translator = LLMTranslator('日本語', 'English')
                translator.set_stop_event(threading.Event())
                self.addCleanup(translator.set_stop_event, None)
                translator.set_param_value('retry attempts', 2)
                spec = TranslationPromptSpec('Japanese', 'English', 'Translate.', False)

                def respond(request: httpx.Request) -> httpx.Response:
                    index = len(self.requests)
                    self.requests.append(request)
                    headers = {'x-codex-turn-state': f'private-route-{index}'} if source == 'response-header' else {}
                    events = [
                        {'type': 'response.metadata', 'headers': {'X-Codex-Turn-State': [f'private-metadata-{index}']}},
                        completion('invalid JSON' if index % 2 == 0 else '{"1":"translated"}'),
                    ]
                    return httpx.Response(200, headers=headers, text=sse(*events))

                self.responder = respond
                with patch.object(translator, '_respect_delay'), patch.object(translator, '_wait'), \
                        self.assertLogs(codex.LOGGER, level='DEBUG') as captured:
                    for _page in range(2):
                        self.assertEqual(translator._translate(['same source'], profile=self.profile,
                                                              prompt_spec=spec), ['translated'])
                prefix = 'private-route' if source == 'response-header' else 'private-metadata'
                self.assertEqual([request.headers.get('x-codex-turn-state') for request in self.requests],
                                 [None, f'{prefix}-0', None, f'{prefix}-2'])
                identities = {request.headers['session-id'] for request in self.requests}
                self.assertEqual(len(identities), 1)
                self.assertTrue(all(request.content == self.requests[0].content for request in self.requests))
                self.assertNotIn('private-route', '\n'.join(captured.output))
                self.assertNotIn('private-metadata', '\n'.join(captured.output))

    def test_turn_state_survives_stream_auth_retry_and_token_renewal(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path.endswith('/oauth/token'):
                return httpx.Response(200, json={'access_token': 'rotated', 'expires_in': 3600})
            if request.headers['Authorization'] == 'Bearer access':
                return httpx.Response(200, text=sse(
                    {'type': 'response.metadata', 'headers': {'x-codex-turn-state': 'route-one'}},
                    {'type': 'error', 'error': {'code': 'token_expired', 'message': 'Token expired.'}},
                ))
            return httpx.Response(200, text=sse(completion()), headers={'x-codex-turn-state': 'route-two'})

        self.responder = respond
        self.request()
        self.assertEqual(len(self.requests), 3)
        self.assertNotIn('x-codex-turn-state', self.requests[0].headers)
        self.assertEqual(self.requests[-1].headers['x-codex-turn-state'], 'route-one')
        self.assertEqual(self.requests[-1].headers['Authorization'], 'Bearer rotated')

    def test_turn_state_clears_on_job_or_account_change(self) -> None:
        self.events.insert(0, {'type': 'response.metadata', 'headers': {'x-codex-turn-state': 'route'}})
        turn = codex.CodexTurnState()
        for key, change_account in (('job-one', False), ('job-one', False),
                                    ('job-two', False), ('job-two', True)):
            if change_account:
                self.account.invalidate()
            codex.request_chat_completion(self.profile, self.args(), None, key, turn=turn)
        self.assertEqual([request.headers.get('x-codex-turn-state') for request in self.requests],
                         [None, 'route', None, None])

    def test_ocr_retries_keep_turn_state_and_next_request_starts_fresh(self) -> None:
        requester = LLMOCR()
        requester.set_stop_event(threading.Event())
        self.addCleanup(requester.set_stop_event, None)
        requester.set_param_value('retry attempts', 2)

        def respond(request: httpx.Request) -> httpx.Response:
            index = len(self.requests)
            self.requests.append(request)
            if index % 2 == 0:
                return httpx.Response(200, headers={'x-codex-turn-state': 'ocr-route'}, text=sse(
                    {'type': 'response.failed', 'response': {'error': {'code': 'server_error'}}},
                ))
            return httpx.Response(200, text=sse(completion('recognized text')))

        self.responder = respond
        with patch.object(requester, '_respect_delay'), patch.object(requester, '_wait'):
            for _page in range(2):
                self.assertEqual(requester._request_with_retries(
                    self.profile, [{'role': 'user', 'content': 'Read.'}], failure_label='OCR'), 'recognized text')
        self.assertEqual([request.headers.get('x-codex-turn-state') for request in self.requests],
                         [None, 'ocr-route', None, 'ocr-route'])

    def test_compaction_retries_keep_turn_state_and_next_batch_starts_fresh(self) -> None:
        translator = LLMTranslator('日本語', 'English')
        translator.set_stop_event(threading.Event())
        self.addCleanup(translator.set_stop_event, None)
        translator.set_param_value('retry attempts', 2)

        def respond(request: httpx.Request) -> httpx.Response:
            index = len(self.requests)
            self.requests.append(request)
            return httpx.Response(200, headers={'x-codex-turn-state': 'memory-route'},
                                  text=sse(completion('' if index % 2 == 0 else 'Compacted memory.')))

        self.responder = respond
        with patch.object(translator, '_respect_delay'), patch.object(translator, '_wait'):
            for _batch in range(2):
                memory = translator._compact_summary_batch(
                    previous=None, summaries=(PageSummary('001.png', 'A fact.'),),
                    profile=self.profile, model=self.profile.model, target_language='English',
                )
                self.assertEqual(memory.text, 'Compacted memory.')
        self.assertEqual([request.headers.get('x-codex-turn-state') for request in self.requests],
                         [None, 'memory-route', None, 'memory-route'])

    def test_stateless_payload_preserves_roles_images_schema_usage_and_omits_tools(self) -> None:
        self.profile.thinking_level = 'high'
        diagnostics = {'type': 'cache_miss', 'reason': 'input_changed',
                       'comparison_reusable_tokens': 2048, 'cache_missed_tokens': 1024}
        self.events[0]['response']['prompt_cache_diagnostics'] = diagnostics
        self.events.insert(0, {'type': 'response.output_text.delta', 'delta': 'partial garbage'})
        result = self.request()
        self.assertEqual(result.content, '{"1":"hello"}')
        self.assertEqual(result.usage.total_tokens, 15)
        self.assertEqual(result.usage.prompt_tokens_details['cached_tokens'], 8)
        self.assertEqual(result.prompt_cache_diagnostics, diagnostics)
        payload = json.loads(self.requests[0].content)
        self.assertEqual(payload['instructions'], 'Translate numbered blocks.\n\nKnown names.')
        self.assertEqual([item['role'] for item in payload['input']], ['user', 'assistant', 'user'])
        self.assertEqual(payload['input'][1]['content'][0], {'type': 'output_text', 'text': 'old translation'})
        self.assertEqual(payload['input'][-1]['content'][-1]['detail'], 'high')
        self.assertEqual(payload['text']['format']['schema'], {'type': 'object'})
        self.assertEqual(payload['reasoning'], {'effort': 'high'})
        self.assertEqual(payload['tools'], [])
        self.assertFalse(payload['store'])
        for header in ('session-id', 'thread-id', 'x-client-request-id'):
            self.assertEqual(payload['prompt_cache_key'], self.requests[0].headers[header])
        self.assertNotIn('session_id', self.requests[0].headers)
        self.assertEqual(self.requests[0].headers['Authorization'], 'Bearer access')
        self.assertEqual(self.requests[0].headers['OpenAI-Beta'], 'responses=experimental')
        self.assertEqual(payload['include'], ['reasoning.encrypted_content'])
        self.assertEqual(result.codex_response_items, tuple(self.events[-1]['response']['output']))
        self.assertNotIn('previous_response_id', payload)
        self.assertNotIn('prompt_cache_options', payload)

    def test_request_fingerprints_locate_changes_without_logging_content(self) -> None:
        args = self.args()
        snapshots = []
        cache_key = 'job-cache-key'
        self.events[-1]['response'].update(model=self.profile.model, reasoning={'context': 'all_turns'})
        with self.assertLogs(codex.LOGGER, level='DEBUG') as captured:
            for change in ('initial', 'repeat', 'text', 'image', 'instructions', 'schema', 'session'):
                if change == 'text':
                    args['messages'][-1]['content'][0]['text'] = 'next page'
                elif change == 'image':
                    args['messages'][-1]['content'][1]['image_url']['url'] = 'data:image/png;base64,def'
                elif change == 'instructions':
                    args['messages'][0]['content'] = 'Changed instructions.'
                elif change == 'schema':
                    args['response_format']['json_schema']['name'] = 'changed_schema'
                elif change == 'session':
                    cache_key = 'next-job-cache-key'
                snapshots.append(copy.deepcopy(args))
                codex.request_chat_completion(self.profile, args, None, cache_key)

        records = [record for record in captured.records if record.msg.startswith('Codex request fingerprints:')]
        fingerprints = [record.args for record in records]
        self.assertEqual(len(fingerprints), 7)
        response_contexts = [record.args for record in captured.records
                             if record.msg.startswith('Codex response context:')]
        self.assertEqual(response_contexts, [(self.profile.model, 'all_turns')] * 7)
        self.assertEqual(fingerprints[0], fingerprints[1])
        for index, component in ((2, 3), (3, 3), (4, 2), (5, 1), (6, 0)):
            previous, current = fingerprints[index - 1], fingerprints[index]
            for field in range(4):
                if field == component:
                    self.assertNotEqual(previous[field], current[field])
                else:
                    self.assertEqual(previous[field], current[field])
            if component == 3:
                self.assertEqual(previous[3][:-1], current[3][:-1])
                self.assertNotEqual(previous[3][-1], current[3][-1])
        logged = '\n'.join(record.getMessage() for record in captured.records)
        for private in ('Translate numbered blocks.', 'Known names.', 'old source', 'old translation',
                        'current page', 'next page', 'base64,', 'Changed instructions.',
                        'changed_schema', 'job-cache-key', 'Bearer access', 'account-one'):
            self.assertNotIn(private, logged)
        for request, original in zip(self.requests, snapshots):
            payload = json.loads(request.content)
            instructions, inputs = codex._request_messages(original['messages'])
            self.assertEqual(payload['instructions'], instructions)
            self.assertEqual(payload['input'], inputs)
            self.assertEqual(payload['text']['format'], {'type': 'json_schema', **original['response_format']['json_schema']})
            self.assertNotIn('fingerprints', payload)
            self.assertNotIn('fingerprints', request.headers)

    def test_gpt_subscription_translation_keeps_implicit_cache_and_fixed_schema(self) -> None:
        model = 'gpt-6-astra'
        self.profile.model = model
        self.profile.model_options = [model]
        translator = LLMTranslator('日本語', 'English')
        messages = self.args()['messages']
        messages.insert(2, {'role': 'system', 'content': 'Project memory.'})
        original = copy.deepcopy(messages)
        with patch.dict(pcfg.module.codex_models, {model: CATALOG['vision-model']}), \
                patch.object(translator, '_respect_delay'):
            for count in (1, 3):
                args = translator._api_args(self.profile, messages, expected_translations=count)
                translator.request_chat_completion(self.profile, args)
        self.assertEqual(messages, original)
        first, second = [json.loads(request.content) for request in self.requests]
        self.assertEqual(first, second)
        self.assertNotIn('prompt_cache_options', first)
        self.assertNotIn('prompt_cache_breakpoint', json.dumps(first))
        self.assertEqual(first['instructions'], 'Translate numbered blocks.\n\nKnown names.\n\nProject memory.')
        self.assertEqual([item['role'] for item in first['input']],
                         ['user', 'assistant', 'user'])
        self.assertEqual(first['input'][1]['content'][0],
                         {'type': 'output_text', 'text': 'old translation'})
        self.assertEqual(first['input'][-1]['content'][-1]['image_url'],
                         original[-1]['content'][-1]['image_url']['url'])
        self.assertEqual(first['text']['format']['schema']['required'], ['translations'])
        self.assertEqual(first['prompt_cache_key'], self.requests[0].headers['session-id'])
        with patch.dict(pcfg.module.__dict__, {'llm_profiles': [self.profile],
                                             'translator_llm_id': self.profile.id}):
            self.assertIn("prompt_cache='implicit'", translator.translation_run_description())
            self.assertIn("transport='websocket'", translator.translation_run_description())

    def test_gpt_56_omits_reasoning_context_without_changing_effort(self) -> None:
        for model in ('gpt-5.6', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'):
            with patch.dict(pcfg.module.codex_models, {model: {'modalities': ['text', 'image'],
                                                            'efforts': ['none', 'high']}}):
                for thinking, effort in ((codex.THINKING_AUTO, None), ('high', 'high'),
                                         (codex.THINKING_DISABLED, 'none')):
                    with self.subTest(model=model, thinking=thinking):
                        self.profile.thinking_level = thinking
                        args = self.args()
                        args['model'] = model
                        codex.request_chat_completion(self.profile, args, None, 'job-cache-key')
                        payload = json.loads(self.requests[-1].content)
                        if effort is None:
                            self.assertNotIn('reasoning', payload)
                        else:
                            self.assertEqual(payload['reasoning'], {'effort': effort})

    def test_parameter_rejection_is_not_misclassified_as_model_unavailable(self) -> None:
        for parameter in ('prompt_cache_options', 'prompt_cache_breakpoint', 'private input with spaces'):
            for streamed in (False, True):
                with self.subTest(parameter=parameter, streamed=streamed):
                    received = []

                    def respond(request: httpx.Request) -> httpx.Response:
                        received.append(request)
                        error = {'error': {
                            'message': f'{parameter} is not supported on this model: private server details',
                            'type': 'invalid_request_error', 'param': parameter, 'code': 'invalid_parameter',
                        }}
                        return (httpx.Response(200, text=sse({'type': 'response.failed', 'response': error}))
                                if streamed else httpx.Response(400, json=error))

                    self.responder = respond
                    with self.assertLogs(codex.LOGGER, level='DEBUG') as captured, \
                            self.assertRaisesRegex(LLMUserActionRequiredError, 'request parameter') as caught:
                        self.request()
                    http_errors = [record.getMessage() for record in captured.records
                                   if record.msg.startswith('Codex HTTP failure:')]
                    self.assertEqual(http_errors, [] if streamed else ['Codex HTTP failure: status=400'])
                    self.assertNotIn('private', '\n'.join(captured.output))
                    message = str(caught.exception)
                    if parameter.startswith('prompt_cache_'):
                        self.assertIn(parameter, message)
                    self.assertNotIn('model is unavailable', message)
                    self.assertNotIn('private', message)
                    self.assertEqual(len(received), 1)
                    self.assertFalse(self.account.auth_invalid)

    def test_image_edit_preserves_references_and_refreshes_once_with_same_payload(self) -> None:
        received = []

        def respond(request):
            received.append(request)
            if request.url.path.endswith('/oauth/token'):
                return httpx.Response(200, json={'access_token': 'rotated', 'expires_in': 3600})
            if request.headers['Authorization'] == 'Bearer access':
                return httpx.Response(401, json={'error': {'code': 'token_expired'}})
            return httpx.Response(200, json={'data': [{'b64_json': PNG_BASE64}]})

        self.responder = respond
        result = codex.request_image('gpt-image-2', 'Edit image 1 using mask image 2.',
                                     PNG_BYTES, PNG_BYTES, None, timeout=23.0)
        self.assertEqual(result, base64.b64decode(PNG_BASE64))
        self.assertEqual(received[0].content, received[2].content)
        image_request = received[2]
        self.assertEqual(str(image_request.url), codex.API_URL + '/images/edits')
        self.assertEqual(image_request.headers['Authorization'], 'Bearer rotated')
        self.assertEqual(image_request.headers['ChatGPT-Account-Id'], 'account-one')
        self.assertEqual(image_request.headers['Accept'], 'application/json')
        for header in ('session-id', 'thread-id', 'x-client-request-id'):
            self.assertNotIn(header, image_request.headers)
        self.assertNotIn('OpenAI-Beta', image_request.headers)
        self.assertEqual(image_request.extensions['timeout']['read'], 23.0)
        self.assertEqual(json.loads(image_request.content), {
            'model': 'gpt-image-2', 'prompt': 'Edit image 1 using mask image 2.',
            'images': [{'image_url': PNG_URL}, {'image_url': PNG_URL}],
            'n': 1, 'background': 'auto', 'quality': 'auto', 'size': 'auto',
        })

    def test_image_generation_uses_verified_sibling_endpoint(self) -> None:
        def respond(request):
            self.requests.append(request)
            return httpx.Response(200, json={'data': [{'b64_json': PNG_BASE64}]})

        self.responder = respond
        self.assertEqual(codex.request_image('gpt-image-2', 'A plain white background.', None, None, None),
                         base64.b64decode(PNG_BASE64))
        self.assertTrue(str(self.requests[0].url).endswith('/images/generations'))
        self.assertNotIn('images', json.loads(self.requests[0].content))

    def test_assisted_edit_forces_image_tool_and_preserves_references_on_auth_retry(self) -> None:
        received = []

        def respond(request):
            received.append(request)
            if request.url.path.endswith('/oauth/token'):
                return httpx.Response(200, json={'access_token': 'rotated', 'expires_in': 3600})
            if request.headers['Authorization'] == 'Bearer access':
                return httpx.Response(401, json={'error': {'code': 'token_expired'}})
            return httpx.Response(200, text=sse(image_completion()))

        with patch.object(codex, '_http_client', side_effect=lambda proxy: httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        )) as factory:
            result = codex.request_image('gpt-image-2', 'Keep every frame.\nUse mask image 2.',
                                         PNG_BYTES, PNG_BYTES, None, reasoning_model='vision-model',
                                         proxy='socks5://proxy.example:1080', timeout=23.0, cache_key='image-job-cache')
        self.assertEqual(result, PNG_BYTES)
        factory.assert_called_once_with('socks5://proxy.example:1080')
        self.assertEqual(len(received), 3)
        self.assertEqual(received[0].content, received[2].content)
        for header in ('session-id', 'thread-id', 'x-client-request-id'):
            self.assertEqual(received[0].headers[header], 'image-job-cache')
            self.assertEqual(received[2].headers[header], 'image-job-cache')
        request = received[2]
        self.assertEqual(str(request.url), codex.API_URL + '/responses')
        self.assertEqual(request.headers['Accept'], 'text/event-stream')
        self.assertEqual(request.headers['OpenAI-Beta'], 'responses=experimental')
        self.assertEqual(request.headers['Authorization'], 'Bearer rotated')
        self.assertEqual(request.extensions['timeout']['read'], 23.0)
        body = json.loads(request.content)
        self.assertEqual(body['model'], 'vision-model')
        self.assertEqual(body['tools'], [{'type': 'image_generation', 'model': 'gpt-image-2',
                                         'action': 'edit', 'quality': 'auto', 'size': 'auto'}])
        self.assertEqual(body['tool_choice'], {'type': 'image_generation'})
        self.assertEqual(body['prompt_cache_key'], 'image-job-cache')
        self.assertEqual(body['prompt_cache_key'], request.headers['session-id'])
        self.assertEqual(body['input'], [{'type': 'message', 'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'Keep every frame.\nUse mask image 2.'},
            {'type': 'input_image', 'image_url': PNG_URL, 'detail': 'high'},
            {'type': 'input_image', 'image_url': PNG_URL, 'detail': 'high'},
        ]}])
        self.assertFalse(body['store'])
        self.assertTrue(body['stream'])
        self.assertNotIn('reasoning', body)
        self.assertNotIn('previous_response_id', body)

    def test_assisted_generation_has_no_source_and_validates_cached_vision_model(self) -> None:
        self.events = [image_completion()]
        self.assertEqual(codex.request_image('gpt-image-2', 'A blank panel.', None, None, None,
                                             reasoning_model='vision-model'), PNG_BYTES)
        body = json.loads(self.requests[0].content)
        self.assertEqual(body['tools'][0]['action'], 'generate')
        self.assertEqual(body['input'][0]['content'], [{'type': 'input_text', 'text': 'A blank panel.'}])
        self.requests.clear()
        for model in ('missing-model', 'text-model'):
            with self.subTest(model=model), self.assertRaises(LLMUserActionRequiredError):
                codex.request_image('gpt-image-2', 'Generate.', None, None, None, reasoning_model=model)
        self.assertFalse(self.requests)

    def test_assisted_stream_accepts_completed_item_fallback_and_duplicate_final_image(self) -> None:
        for repeat_image in (False, True):
            with self.subTest(repeat_image=repeat_image):
                final = image_completion()
                item = final['response']['output'][0]
                if not repeat_image:
                    final['response']['output'] = []
                body = sse({'type': 'response.output_item.done', 'item': item}, final).replace('\n', '\r\n').encode()

                class Chunks(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        for index in range(0, len(body), 17):
                            yield body[index:index + 17]

                self.responder = lambda request: httpx.Response(200, stream=Chunks())
                # A single-image body budget must allow both SSE representations.
                with patch.object(image_generation, 'MAX_IMAGE_RESPONSE_BYTES', len(body) // 2 + 1):
                    self.assertEqual(codex.request_image('gpt-image-2', 'Generate.', None, None, None,
                                                         reasoning_model='vision-model'), PNG_BYTES)

    def test_assisted_output_requires_one_completed_inline_image_and_rejects_refusals(self) -> None:
        item = image_completion()['response']['output'][0]
        refusal = {'type': 'message', 'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'private details'}]}
        for output in (
            [], [None], completion()['response']['output'], [item, item], [refusal, item],
            [{**item, 'status': 'failed'}], [{**item, 'result': None}], [{**item, 'result': ''}],
            [{**item, 'result': '%%%'}], [{**item, 'result': 'https://untrusted.example/image.png'}],
        ):
            with self.subTest(output=output):
                self.requests.clear()
                self.events = [{'type': 'response.completed', 'response': {'status': 'completed', 'output': output}}]
                with self.assertRaises(LLMUserActionRequiredError) as caught:
                    codex.request_image('gpt-image-2', 'Generate.', None, None, None, reasoning_model='vision-model')
                self.assertNotIn('private details', str(caught.exception))
                self.assertEqual(len(self.requests), 1)
        self.events = [image_completion()]
        for limit in ('MAX_IMAGE_BASE64_BYTES', 'MAX_IMAGE_BYTES'):
            with self.subTest(limit=limit), patch.object(image_generation, limit, 1), self.assertRaises(LLMUserActionRequiredError):
                codex.request_image('gpt-image-2', 'Generate.', None, None, None, reasoning_model='vision-model')

    def test_assisted_stream_bounds_unterminated_lines_and_closes_response(self) -> None:
        closed = threading.Event()

        class Oversized(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: ' + b' ' * 128
                raise AssertionError('Oversized stream should already be rejected.')

            async def aclose(self):
                closed.set()

        self.responder = lambda request: httpx.Response(200, stream=Oversized())
        with patch.object(image_generation, 'MAX_IMAGE_RESPONSE_BYTES', 64), self.assertRaisesRegex(LLMUserActionRequiredError, 'oversized'):
            codex.request_image('gpt-image-2', 'Generate.', None, None, None, reasoning_model='vision-model')
        self.assertTrue(closed.is_set())

    def test_unsupported_assisted_tool_is_actionable_for_http_and_stream_errors(self) -> None:
        errors = (
            {'error': {'code': 'unsupported_tool', 'message': 'image_generation is not supported with this reasoning model: private details'}},
            {'error': {'code': 'invalid_value', 'message': "Invalid value: 'image_generation'. Supported values are private details"}},
        )
        for error, streamed in ((error, streamed) for error in errors for streamed in (False, True)):
            with self.subTest(error=error, streamed=streamed):
                received = []

                def respond(request):
                    received.append(request)
                    return (httpx.Response(200, text=sse({'type': 'response.failed', 'response': error}))
                            if streamed else httpx.Response(400, json=error))

                self.responder = respond
                with self.assertRaisesRegex(LLMUserActionRequiredError, 'assisted editing') as caught:
                    codex.request_image('gpt-image-2', 'Generate.', None, None, None, reasoning_model='vision-model')
                self.assertNotIn('private details', str(caught.exception))
                self.assertEqual(len(received), 1)
                self.assertFalse(self.account.auth_invalid)

    def test_invalid_image_inputs_never_reach_network(self) -> None:
        for model, image, mask, expected in (
            ('', PNG_BYTES, None, LLMUserActionRequiredError),
            (' \t', PNG_BYTES, None, LLMUserActionRequiredError),
            (None, PNG_BYTES, None, LLMUserActionRequiredError),
            (42, PNG_BYTES, None, LLMUserActionRequiredError),
            ('gpt-image-2', None, PNG_BYTES, ValueError),
            ('gpt-image-2', 'https://untrusted.example/image.png', None, ValueError),
            ('gpt-image-2', b'not a PNG', None, ValueError),
            ('gpt-image-2', b'', None, ValueError),
        ):
            with self.subTest(model=model, image=image), self.assertRaises(expected):
                codex.request_image(model, 'Edit.', image, mask, None)
        stopped = threading.Event()
        stopped.set()
        with self.assertRaises(LLMRequestStopped):
            codex.request_image('invalid', '', None, None, stopped)
        with patch.object(image_generation, 'MAX_IMAGE_BYTES', 16), self.assertRaises(ValueError):
            codex.request_image('gpt-image-2', 'Edit.', PNG_BYTES, None, None)
        self.assertEqual(self.requests, [])

    def test_custom_image_model_is_sent_unchanged_to_direct_and_assisted_routes(self) -> None:
        model = 'custom-image-model-2026'
        for reasoning_model, image in (('', None), ('', PNG_BYTES), ('vision-model', None), ('vision-model', PNG_BYTES)):
            with self.subTest(reasoning_model=reasoning_model, edit=image is not None):
                self.requests.clear()

                def respond(request):
                    self.requests.append(request)
                    return (httpx.Response(200, text=sse(image_completion())) if reasoning_model
                            else httpx.Response(200, json={'data': [{'b64_json': PNG_BASE64}]}))

                self.responder = respond
                self.assertEqual(codex.request_image(model, 'Generate or edit.', image, None, None,
                                                     reasoning_model=reasoning_model), PNG_BYTES)
                self.assertEqual(len(self.requests), 1)
                request = self.requests[0]
                body = json.loads(request.content)
                if reasoning_model:
                    self.assertEqual(request.url.path, '/backend-api/codex/responses')
                    self.assertEqual(body['tools'][0]['model'], model)
                else:
                    endpoint = 'edits' if image is not None else 'generations'
                    self.assertEqual(request.url.path, '/backend-api/codex/images/' + endpoint)
                    self.assertEqual(body['model'], model)

    def test_unavailable_custom_image_model_is_actionable_without_fallback(self) -> None:
        for reasoning_model, code in (('', 'model_not_found'), ('', 'invalid_model'),
                                      ('vision-model', 'model_not_supported')):
            with self.subTest(reasoning_model=reasoning_model, code=code):
                self.requests.clear()

                def respond(request):
                    self.requests.append(request)
                    error = {'error': {'code': code, 'message': 'image generation model is not available: private server details'}}
                    return (httpx.Response(200, text=sse({'type': 'response.failed', 'response': error}))
                            if reasoning_model else httpx.Response(400, json=error))

                self.responder = respond
                with self.assertRaisesRegex(LLMUserActionRequiredError, 'check the model ID') as caught:
                    codex.request_image('custom-image-model', 'Edit.', PNG_BYTES, None, None,
                                        reasoning_model=reasoning_model)
                self.assertNotIn('private server details', str(caught.exception))
                self.assertEqual(len(self.requests), 1)
                self.assertFalse(self.account.auth_invalid)

    def test_invalid_image_output_never_downloads_response_urls(self) -> None:
        for payload in (None, {}, {'data': []}, {'data': [{'url': 'https://untrusted.example/image.png'}]},
                        {'data': [{'b64_json': '%%%'}]}, {'data': [{'b64_json': ''}]},
                        {'data': [{'b64_json': PNG_BASE64}, {'b64_json': PNG_BASE64}]}):
            with self.subTest(payload=payload):
                self.requests = []

                def respond(request):
                    self.requests.append(request)
                    return httpx.Response(200, json=payload)

                self.responder = respond
                with self.assertRaises(RuntimeError):
                    codex.request_image('gpt-image-2', 'Edit.', PNG_BYTES, None, None)
                self.assertEqual(len(self.requests), 1)
                self.assertEqual(self.requests[0].url.host, 'chatgpt.com')

    def test_image_body_is_bounded_and_timeout_errors_are_sanitized(self) -> None:
        self.responder = lambda request: httpx.Response(200, content=b' ' * 128)
        with patch.object(image_generation, 'MAX_IMAGE_RESPONSE_BYTES', 64), self.assertRaisesRegex(LLMUserActionRequiredError, 'oversized'):
            codex.request_image('gpt-image-2', 'Edit.', PNG_BYTES, None, None)

        def timed_out(request):
            self.requests.append(request)
            raise httpx.ReadTimeout('sensitive response details', request=request)

        for reasoning_model in ('', 'vision-model'):
            with self.subTest(reasoning_model=reasoning_model):
                self.requests.clear()
                self.responder = timed_out
                with self.assertRaises(RuntimeError) as caught:
                    codex.request_image('gpt-image-2', 'Edit.', PNG_BYTES, None, None, reasoning_model=reasoning_model)
                self.assertNotIn('sensitive', str(caught.exception))
                self.assertEqual(len(self.requests), 1)
                self.responder = lambda request: httpx.Response(403, json={'error': {'code': 'usage_limit_reached'}})
                with self.assertRaisesRegex(LLMUserActionRequiredError, 'usage limit'):
                    codex.request_image('gpt-image-2', 'Edit.', PNG_BYTES, None, None, reasoning_model=reasoning_model)

    def test_image_cancellation_and_account_change_close_stalled_response(self) -> None:
        for reasoning_model, invalidate_account in (('', False), ('', True), ('vision-model', False), ('vision-model', True)):
            with self.subTest(reasoning_model=reasoning_model, invalidate_account=invalidate_account):
                stopped, entered, closed = threading.Event(), threading.Event(), threading.Event()

                class Stalled(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        entered.set()
                        await asyncio.Event().wait()
                        yield b''

                    async def aclose(self):
                        closed.set()

                self.responder = lambda request: httpx.Response(200, stream=Stalled())
                errors = []

                def run():
                    try:
                        codex.request_image('gpt-image-2', 'Edit.', PNG_BYTES, None, stopped,
                                             timeout=None, reasoning_model=reasoning_model)
                    except Exception as error:
                        errors.append(error)

                worker = threading.Thread(target=run)
                worker.start()
                self.assertTrue(entered.wait(2))
                self.account.invalidate() if invalidate_account else stopped.set()
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertTrue(closed.is_set())
                self.assertIsInstance(errors[0], LLMRequestStopped)

    def test_image_account_change_during_decoding_rejects_late_result(self) -> None:
        decode = base64.b64decode

        def invalidate_during_decode(value, **kwargs):
            self.account.invalidate()
            return decode(value, **kwargs)

        for reasoning_model in ('', 'vision-model'):
            self.responder = lambda request: (httpx.Response(200, text=sse(image_completion())) if reasoning_model
                                             else httpx.Response(200, json={'data': [{'b64_json': PNG_BASE64}]}))
            with self.subTest(reasoning_model=reasoning_model), \
                    patch.object(codex.base64, 'b64decode', side_effect=invalidate_during_decode), self.assertRaises(LLMRequestStopped):
                codex.request_image('gpt-image-2', 'Generate.', None, None, None, reasoning_model=reasoning_model)

    def test_completed_items_and_final_phase_are_used_without_committing_commentary(self) -> None:
        final = completion()['response']['output'][0]
        response = completion()
        response['response']['output'] = []
        self.events = [
            {'type': 'response.output_item.done', 'item': {'type': 'message', 'role': 'assistant', 'phase': 'commentary',
                'content': [{'type': 'output_text', 'text': 'thinking'}]}},
            {'type': 'response.output_item.done', 'item': final}, response,
        ]
        self.assertEqual(self.request().content, '{"1":"hello"}')

    def test_ocr_owner_consumes_usage_and_run_key_is_stable_until_new_job(self) -> None:
        ocr = LLMOCR()
        job = threading.Event()
        with patch.object(ocr, '_respect_delay'):
            for _ in range(2):
                ocr.set_stop_event(job)
                self.assertEqual(ocr._request_with_retries(self.profile, self.args()['messages'], failure_label='OCR'), '{"1":"hello"}')
            ocr.set_stop_event(threading.Event())
            ocr._request_with_retries(self.profile, self.args()['messages'], failure_label='OCR')
        keys = [json.loads(request.content)['prompt_cache_key'] for request in self.requests]
        for header in ('session-id', 'thread-id', 'x-client-request-id'):
            self.assertEqual([request.headers[header] for request in self.requests], keys)
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[1], keys[2])
        self.assertEqual((ocr.token_count, ocr.token_count_last), (45, 15))

    def test_optional_usage_cannot_retry_a_successful_ocr_response(self) -> None:
        ocr = LLMOCR()
        for usage in (None, {}, {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': None},
                      {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': '15'}):
            with self.subTest(usage=usage), patch.object(ocr, '_respect_delay'):
                self.events = [completion()]
                self.events[0]['response']['usage'] = usage
                count = len(self.requests)
                self.assertEqual(ocr._request_with_retries(self.profile, self.args()['messages'], failure_label='OCR'), '{"1":"hello"}')
                self.assertEqual(len(self.requests), count + 1)
                self.assertEqual(ocr.token_count_last, 0)

    def test_pre_stopped_request_does_not_validate_or_send(self) -> None:
        stop = threading.Event()
        stop.set()
        with self.assertRaises(LLMRequestStopped):
            codex.request_chat_completion(self.profile, {}, stop, 'job-key')
        self.assertEqual(self.requests, [])

    def test_transient_and_actionable_stream_errors_are_sanitized_and_typed(self) -> None:
        for event, expected in (
            ({'type': 'response.failed', 'response': {'error': {'code': 'context_length_exceeded', 'message': 'secret'}}}, ContextLengthError),
            ({'type': 'response.failed', 'response': {'error': {'code': 'usage_limit_reached', 'message': 'secret'}}}, LLMUserActionRequiredError),
            ({'type': 'response.incomplete', 'response': {'incomplete_details': {'reason': 'max_output_tokens'}}}, LLMUserActionRequiredError),
        ):
            with self.subTest(event=event):
                self.events = [event]
                with self.assertRaises(expected) as caught:
                    self.request()
                self.assertNotIn('secret', str(caught.exception))
        self.events = [{'type': 'response.output_text.delta', 'delta': 'partial'}]
        with self.assertRaisesRegex(RuntimeError, 'connection'):
            self.request()

    def test_cancellation_interrupts_a_stalled_stream_and_closes_it(self) -> None:
        stop, entered, closed = threading.Event(), threading.Event(), threading.Event()

        class Stalled(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b''

            async def aclose(self):
                closed.set()

        self.responder = lambda request: httpx.Response(200, stream=Stalled())
        errors = []

        def run():
            try:
                self.request(stop)
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(entered.wait(2))
        start = time.monotonic()
        stop.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - start, 1)
        self.assertTrue(closed.is_set())
        self.assertIsInstance(errors[0], LLMRequestStopped)

    def test_logout_invalidates_requests_and_removes_credentials(self) -> None:
        self.account._save()
        generation = self.account.generation
        self.account.logout(threading.Event())
        self.assertGreater(self.account.generation, generation)
        self.assertFalse(self.path.exists())
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'Sign in'):
            self.request()

    def test_concurrent_refresh_rotates_once_and_preserves_missing_token_fields(self) -> None:
        self.account._credentials = tokens(expiry=0)
        refreshes = []

        async def respond(request):
            refreshes.append(request)
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={'access_token': 'rotated-access', 'expires_in': 3600})

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                return await asyncio.gather(self.account.tokens(client), self.account.tokens(client))

        old_generation = self.account.generation
        first, second = asyncio.run(run())
        self.assertEqual(len(refreshes), 1)
        self.assertEqual(first, second)
        self.assertEqual(first['refresh_token'], 'refresh')
        self.assertEqual(first['access_token'], 'rotated-access')
        self.assertEqual(self.account.generation, old_generation)
        restarted = codex.CodexAccount()
        with patch.object(restarted, '_path', return_value=self.path):
            restarted._load()
        self.assertEqual(restarted._credentials, first)
        if os.name != 'nt':
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_refresh_persistence_failure_retains_rotated_token_and_retries_storage(self) -> None:
        self.account._credentials = tokens(expiry=0)
        refreshes = []

        def respond(request):
            refreshes.append(request)
            return httpx.Response(200, json={'access_token': 'new-access', 'refresh_token': 'new-refresh', 'expires_in': 3600})

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                with patch.object(codex.os, 'replace', side_effect=PermissionError()):
                    with self.assertRaisesRegex(LLMUserActionRequiredError, 'saved'):
                        await self.account.tokens(client)
                return await self.account.tokens(client)

        result = asyncio.run(run())
        self.assertEqual(len(refreshes), 1)
        self.assertEqual(result['refresh_token'], 'new-refresh')
        restarted = codex.CodexAccount()
        with patch.object(restarted, '_path', return_value=self.path):
            restarted._load()
        self.assertEqual(restarted._credentials['refresh_token'], 'new-refresh')
        self.assertEqual(list(self.path.parent.glob('.http-auth-*')), [])

    def test_unauthorized_request_refreshes_once_without_changing_payload_or_cache_key(self) -> None:
        received = []

        def respond(request):
            received.append(request)
            if request.url.path.endswith('/oauth/token'):
                return httpx.Response(200, json={'access_token': 'rotated', 'expires_in': 3600})
            if request.headers['Authorization'] == 'Bearer access':
                return httpx.Response(401, json={'error': {'code': 'token_expired'}})
            return httpx.Response(200, text=sse(completion()))

        self.responder = respond
        self.assertEqual(self.request().content, '{"1":"hello"}')
        self.assertEqual(received[0].content, received[2].content)
        for header in ('session-id', 'thread-id', 'x-client-request-id'):
            self.assertEqual(received[0].headers[header], 'job-cache-key')
            self.assertEqual(received[2].headers[header], 'job-cache-key')
        self.assertEqual(received[2].headers['Authorization'], 'Bearer rotated')

    def test_catalog_uses_subscription_visibility_and_explicit_modalities(self) -> None:
        def respond(request):
            self.assertEqual(request.headers['accept'], 'application/json')
            self.assertNotIn('OpenAI-Beta', request.headers)
            for header in ('session-id', 'thread-id', 'x-client-request-id'):
                self.assertNotIn(header, request.headers)
            return httpx.Response(200, json={'models': [
                None, {}, {'slug': 42},
                {'slug': 'vision-model', 'visibility': 'list', 'supported_in_api': False,
                 'input_modalities': ['text', 'image'], 'supported_reasoning_levels': [None, {'effort': 'high'}]},
                {'slug': 'text-model', 'visibility': 'list', 'supported_reasoning_levels': None},
                {'slug': 'hidden-model', 'visibility': 'hide'},
            ]})
        self.responder = respond
        models = self.account.catalog(threading.Event())
        self.assertEqual(self.account.cached_account_label, 'demo@example.com')
        self.assertEqual(models['vision-model']['modalities'], ['text', 'image'])
        self.assertEqual(models['text-model']['modalities'], ['text'])
        self.assertNotIn('hidden-model', models)

    def test_catalog_reasoning_levels_survive_refresh_save_reload_and_requests(self) -> None:
        efforts = ['low', 'high', 'max', 'ultra', 'future-level']
        self.responder = lambda request: httpx.Response(200, json={'models': [{
            'slug': 'vision-model', 'input_modalities': ['text', 'image'],
            'supported_reasoning_levels': [{'effort': effort} for effort in efforts],
        }]})
        with self.assertNoLogs('BallonTranslator', level='WARNING'):
            models = self.account.catalog(threading.Event())
            module = ModuleConfig(llm_profiles=[self.profile], codex_models=models)
            saved = json.loads(json_dump_program_config(ProgramConfig(module=module)))
            restored = ModuleConfig(**saved['module'])
        self.assertEqual(restored.codex_models['vision-model']['efforts'], efforts)
        self.profile = restored.llm_profiles[0]
        self.assertEqual(self.profile.thinking_level_options, ['Auto', *efforts])
        pcfg.module.codex_models = restored.codex_models
        self.responder = self.respond
        for effort in ('max', 'ultra', 'future-level'):
            with self.subTest(effort=effort):
                self.profile.thinking_level = effort
                self.request()
                self.assertEqual(json.loads(self.requests[-1].content)['reasoning'], {'effort': effort})
        self.profile.thinking_level = 'not-advertised'
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'does not support this thinking level'):
            self.request()

    def test_browser_login_checks_state_and_pkce_and_cancellation_preserves_credentials(self) -> None:
        responses, authorization, exchanged = [], {}, []

        class Server:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class Writer:
            def write(self, data):
                responses.append(data)

            async def drain(self):
                pass

            def close(self):
                pass

        async def start_server(handler, host, port, **kwargs):
            self.assertEqual((host, port), ('127.0.0.1', 1455))
            authorization['handler'] = handler
            return Server()

        async def callback(state):
            reader = asyncio.StreamReader()
            query = urlencode({'code': 'authorization-code', 'state': state})
            reader.feed_data(('GET /auth/callback?' + query + ' HTTP/1.1\r\n').encode())
            reader.feed_eof()
            await authorization['handler'](reader, Writer())

        async def browser():
            await callback('unrelated-state')
            self.assertIn(b'400 Bad Request', responses[-1])
            self.assertEqual(exchanged, [])
            await callback(authorization['query']['state'][0])
            self.assertIn(b'200 OK', responses[-1])

        def show_url(url):
            authorization['query'] = parse_qs(urlsplit(url).query)
            authorization['browser'] = asyncio.create_task(browser())

        def exchange(request):
            form = parse_qs(request.content.decode())
            exchanged.append(form)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(form['code_verifier'][0].encode()).digest()).decode().rstrip('=')
            self.assertEqual(challenge, authorization['query']['code_challenge'][0])
            self.assertEqual(form['code'], ['authorization-code'])
            self.assertEqual(form['redirect_uri'], authorization['query']['redirect_uri'])
            self.assertNotIn('authorization-code', str(request.url))
            return httpx.Response(200, json={
                'access_token': jwt({'exp': time.time() + 3600}), 'refresh_token': 'new-refresh',
                'id_token': jwt({'email': 'new@example.com', 'https://api.openai.com/auth': {'chatgpt_account_id': 'new-account'}}),
            })

        self.responder = exchange
        with patch.object(codex.asyncio, 'start_server', side_effect=start_server):
            self.account.login(threading.Event(), show_url)
            authorization['browser'].result()
            self.assertEqual(self.account._credentials['account_id'], 'new-account')
            self.assertEqual(len(exchanged), 1)
            saved = self.path.read_bytes()
            stop = threading.Event()
            with self.assertRaises(LLMRequestStopped):
                self.account.login(stop, lambda url: stop.set())
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(self.account._credentials['refresh_token'], 'new-refresh')
        self.assertFalse(self.account.changing)

    def test_translator_context_recovery_keeps_input_and_commits_only_valid_response(self) -> None:
        from ballontranslator.modules.context.history import HistoryPage, HistoryWindow, HistoryWindowKey
        from ballontranslator.modules.context.translation_context import RequestContext
        from ballontranslator.modules.translators.llm_translation_contract import InvalidNumTranslations, TranslationPromptSpec, render_history_page
        spec = TranslationPromptSpec('Japanese', 'English', 'Translate.', False, True, True)
        history = tuple(render_history_page(HistoryPage(str(i), ('source-' + str(i),), ('translated-' + str(i),)),
                                            self.profile.model, spec) for i in (1, 2))
        key = HistoryWindowKey(object(), ())
        context = RequestContext(history, history_budget=10000, window_key=key, request_page_key='current')
        for text, valid in (('{"translations":[{"id":1,"translation":"hello"}]}', True), ('{"translations":[]}', False)):
            with self.subTest(text=text):
                requests = []

                def respond(request):
                    requests.append(json.loads(request.content))
                    if len(requests) == 1:
                        return httpx.Response(400, json={'error': {'code': 'context_length_exceeded', 'message': 'oversized'}})
                    return httpx.Response(200, text=sse(completion(text)))

                self.responder = respond
                translator = LLMTranslator('日本語', 'English')
                translator.set_param_value('retry attempts', 1)
                translator.set_stop_event(threading.Event())
                previous = HistoryWindow(key, 'previous', history, sum(page.token_count for page in history))
                translator._history_window = previous
                with patch.object(translator, '_respect_delay'):
                    if valid:
                        self.assertEqual(translator._translate(['current-source'], prompt_spec=spec, profile=self.profile, request_context=context), ['hello'])
                        self.assertEqual(translator._history_window.history, history[1:])
                    else:
                        with self.assertRaises(InvalidNumTranslations):
                            translator._translate(['current-source'], prompt_spec=spec, profile=self.profile, request_context=context)
                        self.assertIs(translator._history_window, previous)
                self.assertEqual(requests[0]['prompt_cache_key'], requests[1]['prompt_cache_key'])
                self.assertEqual(requests[0]['input'][-1], requests[1]['input'][-1])
                self.assertEqual(requests[1]['input'], requests[0]['input'][2:])

    def test_translation_pages_share_fixed_schema_and_prefix_then_reject_duplicate_ids(self) -> None:
        from ballontranslator.modules.translators.llm_translation_contract import InvalidNumTranslations
        from ballontranslator.utils.config import LLMTranslateContext
        from ballontranslator.utils.proj_imgtrans import ProjImgTrans
        from ballontranslator.utils.textblock import TextBlock
        project = ProjImgTrans()
        project.pages = {str(page): [TextBlock(text=[f'source-{page}-{i}'], translation='previous')
                                    for i in range(count)] for page, count in enumerate((1, 3, 2, 1))}
        project._pagename2idx = {key: index for index, key in enumerate(project.pages)}
        project._image_info = {page: {'finish_code': 0} for page in project.pages}
        translator = LLMTranslator('日本語', 'English')
        self.profile.model = 'gpt-5.6-luna'
        translator.set_stop_event(threading.Event())
        translator.set_param_value('retry attempts', 1)
        received = []
        outputs = []

        def respond(request):
            body = json.loads(request.content)
            received.append(body)
            current = body['input'][-1]['content'][0]['text'].split('INPUT:\n', 1)[1]
            translations = [{'id': item['id'], 'translation': f'translated-{item["id"]}。'} for item in json.loads(current)]
            if len(received) == 4:
                translations.append(translations[0])
            event = completion(json.dumps({'translations': translations}, indent=2))
            output = event['response']['output']
            output[0].update(id=f'msg_{len(received)}', status='completed')
            output[0]['content'][0]['annotations'] = []
            output.insert(0, {'type': 'reasoning', 'id': f'rs_{len(received)}', 'summary': [],
                              'content': None,
                              'encrypted_content': f'opaque-reasoning-{len(received)}'})
            outputs.append(copy.deepcopy(output))
            return httpx.Response(200, text=sse(event))

        self.responder = respond
        settings = {'llm_profiles': [self.profile], 'translator_llm_id': self.profile.id,
                    'llm_translate_context': LLMTranslateContext.HISTORY, 'llm_prior_context_token_budget': 4096,
                    'llm_translate_vision': False, 'llm_translate_summary_memory': False, 'llm_glossary_path': ''}
        with patch.dict(pcfg.module.__dict__, settings), patch.object(translator, '_respect_delay'), \
                patch.dict(pcfg.module.codex_models, {self.profile.model: CATALOG['vision-model']}):
            for page in ('0', '1', '2'):
                translator.translate_textblk_lst(project.pages[page], project=project, page_key=page, full_page=True)
                project.mark_translation_finished(page, 'English')
                for block in project.pages[page]:
                    self.assertTrue(block.translation.endswith('.'))
                    block.translation = block.translation.replace('translated-', 'trans\nlated-')
            previous_window = translator._history_window
            with self.assertRaises(InvalidNumTranslations):
                translator.translate_textblk_lst(project.pages['3'], project=project, page_key='3', full_page=True)
        self.assertIs(translator._history_window, previous_window)
        self.assertIsNone(translator._translation_codex_response)
        self.assertEqual(project.pages['3'][0].translation, 'previous')
        self.assertEqual(len({body['prompt_cache_key'] for body in received}), 1)
        self.assertTrue(all('reasoning' not in body for body in received))
        for page_index, (previous, current) in enumerate(zip(received, received[1:])):
            self.assertEqual(previous['text']['format'], current['text']['format'])
            self.assertEqual(previous['instructions'], current['instructions'])
            self.assertEqual(previous['input'][:-1], current['input'][:len(previous['input']) - 1])
            baseline = previous['input'] + outputs[page_index]
            self.assertEqual(current['input'][:len(baseline)], baseline)
            history = json.loads(current['input'][-2]['content'][0]['text'])
            self.assertEqual(current['input'][-2]['role'], 'assistant')
            self.assertEqual(current['input'][-2]['content'][0]['type'], 'output_text')
            self.assertEqual(history['translations'], [
                {'id': i + 1, 'translation': f'translated-{i + 1}。'}
                for i in range((1, 3, 2)[page_index])
            ])
            self.assertNotIn('prompt_cache_options', current)
            self.assertNotIn('prompt_cache_breakpoint', json.dumps(current['input']))
        self.assertIn('"translations":[{"id":1,"translation":"Translated text"}]', received[0]['instructions'])

    def test_translation_replay_follows_page_edits_job_account_and_history_lifetime(self) -> None:
        from ballontranslator.utils.config import LLMTranslateContext
        from ballontranslator.utils.proj_imgtrans import ProjImgTrans
        from ballontranslator.utils.textblock import TextBlock

        for change in ('previous-edit', 'retained-edit', 'previous-source-edit', 'retained-source-edit',
                       'previous-incomplete', 'new-job', 'account', 'account-before-send', 'reload', 'model', 'page-mode', 'eviction'):
            with self.subTest(change=change):
                project = ProjImgTrans()
                project.pages = {str(i): [TextBlock(text=[f'source-{i}'])] for i in range(4)}
                project._pagename2idx = {key: i for i, key in enumerate(project.pages)}
                project._image_info = {key: {'finish_code': 0} for key in project.pages}
                profile = copy.deepcopy(self.profile)
                translator = LLMTranslator('日本語', 'English')
                translator.set_stop_event(threading.Event())
                received = []

                def respond(request):
                    received.append(json.loads(request.content))
                    event = completion('{"translations":[{"id":1,"translation":"translated"}]}')
                    event['response']['output'].insert(0, {
                        'type': 'reasoning', 'id': f'rs_{len(received)}', 'summary': [],
                        'encrypted_content': 'opaque' * 2000,
                    })
                    return httpx.Response(200, text=sse(event))

                self.responder = respond
                settings = {'llm_profiles': [profile], 'translator_llm_id': profile.id,
                            'llm_translate_context': LLMTranslateContext.HISTORY,
                            'llm_prior_context_token_budget': (8 if change == 'eviction' else
                                                             12 if change in ('new-job', 'account', 'account-before-send') else 4096),
                            'llm_translate_vision': False, 'llm_translate_summary_memory': False, 'llm_glossary_path': ''}
                with patch.dict(pcfg.module.__dict__, settings), patch.object(translator, '_respect_delay') as delay, \
                        patch('ballontranslator.modules.translators.llm_translation_contract.messages_token_count', return_value=3):
                    for page in ('0', '1', '2'):
                        translator.translate_textblk_lst(project.pages[page], project=project, page_key=page, full_page=True)
                        project.mark_translation_finished(page, 'English')
                    if change == 'previous-edit':
                        project.pages['2'][0].translation = 'edited'
                    elif change == 'retained-edit':
                        project.pages['0'][0].translation = 'edited'
                    elif change == 'previous-source-edit':
                        project.pages['2'][0].text = ['edited-source']
                    elif change == 'retained-source-edit':
                        project.pages['0'][0].text = ['edited-source']
                    elif change == 'previous-incomplete':
                        project.begin_full_page_translation('2')
                    elif change == 'new-job':
                        translator.set_stop_event(threading.Event())
                    elif change == 'account':
                        self.account.invalidate()
                    elif change == 'account-before-send':
                        delay.side_effect = self.account.invalidate
                    elif change == 'reload':
                        reloaded = ProjImgTrans()
                        reloaded.pages, reloaded._pagename2idx, reloaded._image_info = (
                            project.pages, project._pagename2idx, project._image_info,
                        )
                        project = reloaded
                    elif change == 'model':
                        profile.model = 'text-model'
                    elif change == 'page-mode':
                        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
                    translator.translate_textblk_lst(project.pages['3'], project=project, page_key='3', full_page=True)

                self.assertEqual({item['id'] for item in received[2]['input'] if item['type'] == 'reasoning'},
                                 {'rs_1', 'rs_2'})
                expected = {
                    'previous-edit': {'rs_1', 'rs_2', 'rs_3'},
                    'retained-edit': {'rs_1', 'rs_2', 'rs_3'},
                    'previous-source-edit': {'rs_1', 'rs_2'},
                    'retained-source-edit': {'rs_2', 'rs_3'},
                    'previous-incomplete': {'rs_1', 'rs_2'},
                    'eviction': {'rs_3'},
                }.get(change, set())
                self.assertEqual({item['id'] for item in received[3]['input'] if item['type'] == 'reasoning'}, expected)
                if change in ('previous-edit', 'retained-edit'):
                    self.assertNotIn('edited', json.dumps(received[3]['input']))
                if change in ('previous-source-edit', 'retained-source-edit'):
                    self.assertIn('edited-source', json.dumps(received[3]['input']))
                if change in ('new-job', 'account', 'account-before-send'):
                    # A replay boundary must not trim otherwise reusable text history.
                    self.assertEqual(len(received[3]['input']), 7)

    def test_account_change_while_preparing_request_cancels_before_http(self) -> None:
        fingerprint = codex._request_fingerprint
        references = image_generation.image_references

        def change_account(value):
            self.account.invalidate()
            return fingerprint(value)

        def change_image_account(*args):
            self.account.invalidate()
            return references(*args)

        for target in ('chat', 'image', 'assisted-image'):
            with self.subTest(target=target), \
                    patch.object(codex, '_request_fingerprint', side_effect=change_account), \
                    patch.object(image_generation, 'image_references', side_effect=change_image_account), \
                    self.assertRaises(LLMRequestStopped):
                if target == 'chat':
                    self.request()
                else:
                    codex.request_image('gpt-image-2', 'Edit.', PNG_BYTES, None, None,
                                        reasoning_model='vision-model' if target == 'assisted-image' else '')
        self.assertEqual(self.requests, [])

    def test_replay_skips_reasoning_without_encrypted_content(self) -> None:
        reasoning = {'type': 'reasoning', 'id': 'rs_unavailable', 'summary': []}
        self.events[0]['response']['output'].insert(0, reasoning)
        result = self.request()
        self.assertEqual(result.content, '{"1":"hello"}')
        self.assertNotIn(reasoning, result.codex_response_items)

    def test_summary_only_page_uses_same_codex_schema_and_persists_through_owner(self) -> None:
        import numpy as np
        from ballontranslator.utils.config import LLMTranslateContext
        from ballontranslator.utils.proj_imgtrans import ProjImgTrans
        from ballontranslator.utils.textblock import TextBlock
        project = ProjImgTrans()
        project.pages = {'0': [TextBlock(text=['source'])], '1': [], '2': [TextBlock(text=['next'])]}
        project._pagename2idx = {key: index for index, key in enumerate(project.pages)}
        project._image_info = {page: {'finish_code': 0} for page in project.pages}
        project.read_img = Mock(return_value=np.zeros((16, 16, 3), dtype=np.uint8))
        translator = LLMTranslator('日本語', 'English')
        translator.set_stop_event(threading.Event())
        received = []

        def respond(request):
            received.append(json.loads(request.content))
            page = str(len(received) - 1)
            result = {'page_summary': 'Scene ' + page, 'translations': [
                {'id': i + 1, 'translation': 'translated'} for i in range(len(project.pages[page]))
            ]}
            return httpx.Response(200, text=sse(completion(json.dumps(result))))

        self.responder = respond
        settings = {'llm_profiles': [self.profile], 'translator_llm_id': self.profile.id,
                    'llm_translate_context': LLMTranslateContext.HISTORY, 'llm_prior_context_token_budget': 4096,
                    'llm_translate_vision': True, 'llm_translate_summary_memory': True,
                    'llm_translate_overwrite_summary': False, 'llm_glossary_path': ''}
        with patch.dict(pcfg.module.__dict__, settings), patch.object(translator, '_respect_delay'), \
                patch.object(translator, '_compact_last_page_memory'):
            for page in project.pages:
                translator.translate_textblk_lst(project.pages[page], project=project, page_key=page, full_page=True)
                self.assertIsNone(project.get_llm_visual_summary(page))
                project.mark_translation_finished(page, 'English')
                translator.on_page_translation_finished(project, page)
                self.assertEqual(project.get_llm_visual_summary(page)['text'], 'Scene ' + page)
        for body in received:
            self.assertEqual(body['text']['format'], received[0]['text']['format'])
            self.assertEqual(list(body['text']['format']['schema']['properties']), ['page_summary', 'translations'])
        empty_history = json.loads(received[2]['input'][-2]['content'][0]['text'])
        self.assertEqual(empty_history, {'page_summary': 'Scene 1', 'translations': []})


class CodexConfigTest(unittest.TestCase):
    def test_codex_capabilities_and_saved_model_options_need_no_account_io(self) -> None:
        defaults = list(default_codex_profile().model_options)
        for model, vision_model in (('', ''), ('saved-text', ''), ('saved-text', 'saved-vision'),
                                    (defaults[0], defaults[-1])):
            with self.subTest(model=model, vision_model=vision_model), \
                    patch.object(codex.account, '_load', side_effect=AssertionError('Unexpected credential IO')), \
                    patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected HTTP')):
                module = ModuleConfig(llm_profiles=[{
                    'id': 'codex', 'backend': 'codex', 'model': model, 'vision_model': vision_model,
                    'support_text': False, 'support_vision': False, 'support_image': False,
                    'image_model': 'saved-image', 'image_model_options': [],
                }], codex_models={})
                profile = module.llm_profiles[0]
                self.assertTrue(profile.support_text)
                self.assertTrue(profile.support_vision)
                self.assertTrue(profile.support_image)
                text_options = defaults + ([model] if model and model not in defaults else [])
                vision_options = defaults + ([vision_model] if vision_model and vision_model not in defaults else [])
                self.assertEqual(profile.model_options, text_options)
                self.assertEqual(profile.vision_model_options, vision_options)
                self.assertEqual(profile.image_model_options, ['saved-image'])
                sync_codex_profile(profile, CATALOG)
                self.assertEqual(profile.model_options, list(CATALOG))
                self.assertEqual(profile.vision_model_options, ['vision-model'])
                self.assertEqual((profile.model, profile.vision_model), (model, vision_model))
                sync_codex_profile(profile, {})
                self.assertEqual(profile.model_options, text_options)
                self.assertEqual(profile.vision_model_options, vision_options)
                self.assertTrue(profile.support_image)
        self.assertEqual(default_codex_profile().model_options, defaults)
        self.assertEqual(default_codex_profile().vision_model_options, defaults)

    def test_explicit_proxy_routes_requests_even_with_environment_proxy(self) -> None:
        routed = []

        async def handle(transport, request):
            routed.append((transport._pool._proxy_url.host, request.url.host))
            return httpx.Response(200, json={'routed': True})

        async def request():
            async with codex._http_client('http://module-proxy.test:8080') as client:
                result = await client.get('http://backend.test/models')
                self.assertTrue(result.json()['routed'])

        with patch.dict(os.environ, {'HTTP_PROXY': 'http://environment-proxy.test:8080'}, clear=True), \
                patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle):
            asyncio.run(request())
        self.assertEqual(routed, [(b'module-proxy.test', 'backend.test')])

    def test_catalog_is_single_persistent_source_and_selections_survive_missing_catalog(self) -> None:
        profile = default_codex_profile()
        image_options = list(profile.image_model_options)
        profile.model = profile.vision_model = 'vision-model'
        profile.api_key = 'must-not-export'
        module = ModuleConfig(llm_profiles=[profile], codex_models=CATALOG)
        saved = json.loads(json_dump_program_config(ProgramConfig(module=module)))
        self.assertNotIn('model_options', saved['module']['llm_profiles'][0])
        self.assertNotIn('must-not-export', json.dumps(saved))
        self.assertEqual(saved['module']['codex_models'], CATALOG)
        self.assertEqual(profile_to_dict(profile)['api_key'], '')
        sync_codex_profile(profile, {})
        self.assertEqual((profile.model, profile.vision_model), ('vision-model', 'vision-model'))
        self.assertTrue(profile.support_image)
        self.assertEqual(profile.image_model_options, image_options)

    def test_malformed_catalog_discards_only_invalid_optional_data(self) -> None:
        with self.assertLogs('BallonTranslator', level='WARNING'):
            models = normalize_codex_models({
                'good': {'modalities': ['text', 'bad'], 'efforts': ['high', None, '', '  ', 3, {}]},
                'bad': None,
            })
        self.assertEqual(models, {'good': {'modalities': ['text'], 'efforts': ['high']}})

    def test_image_backend_never_falls_through_to_api(self) -> None:
        requester = LLMImageRequester()
        for backend in ('codex', 'unavailable'):
            profile = default_profile('OpenAI')
            profile.backend = backend
            with self.assertRaises(LLMUserActionRequiredError):
                requester._initialize_client(profile)


class CodexSettingsAccountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from qtpy.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from ballontranslator.ui import codex_account
        from qtpy.QtWidgets import QMessageBox
        profile = default_codex_profile()
        profile.image_model = 'gpt-image-2'
        profile.image_model_options = ['gpt-image-2']
        self.account = codex.CodexAccount()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for patcher in (
            patch.object(codex, 'account', self.account),
            patch.object(codex_account, 'account', self.account),
            patch.object(codex, '_client_version_checked_at', time.monotonic()),
            patch.object(self.account, '_path', return_value=Path(self.directory.name) / 'http-auth.json'),
            patch.object(codex, '_system_keyring', side_effect=ImportError),
            patch.object(pcfg.module, 'codex_models', copy.deepcopy(CATALOG)),
            patch.object(pcfg.module, 'llm_profiles', [profile]),
            patch('ballontranslator.ui.codex_settings.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes),
            patch('ballontranslator.ui.codex_account.QMessageBox.warning', return_value=QMessageBox.StandardButton.Ok),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.warning = QMessageBox.warning
        CodexAccountController = codex_account.CodexAccountController
        self.controller = CodexAccountController.instance()
        self.controller.account = ''

    def tearDown(self) -> None:
        from qtpy.QtCore import QCoreApplication, QEvent
        self.doCleanups()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        super().tearDown()

    def finish_worker(self) -> None:
        worker = self.controller.worker
        self.assertIsNotNone(worker)
        self.assertTrue(worker.wait(2000))
        self.app.processEvents()
        self.assertIsNone(self.controller.worker)

    def make_panel(self) -> 'CodexSettingsPanel':
        from ballontranslator.ui.codex_settings import CodexSettingsPanel
        panel = CodexSettingsPanel()
        self.addCleanup(panel.deleteLater)
        return panel

    def test_config_rendering_never_loads_credentials_or_starts_http(self) -> None:
        from ballontranslator.ui.codex_settings import CodexSettingsPanel
        profile = pcfg.module.llm_profiles[0]
        profile.model = 'vision-model'
        sync_codex_profile(profile, CATALOG)
        with patch.object(pcfg.module, 'codex_models', CATALOG), patch.object(codex.account, '_load') as load:
            panel = CodexSettingsPanel()
            self.addCleanup(panel.deleteLater)
            self.assertTrue(profile.support_image)
            self.assertEqual(profile.image_model_options, ['gpt-image-2'])
            self.assertEqual(profile.vision_model_options, ['vision-model'])
            panel.param_widgets['model'].setCurrentText('text-model')
            self.assertEqual(profile.thinking_level_options, ['Auto', 'Disabled'])
            self.assertIsNone(self.account.cached_account_label)
            self.assertFalse(panel.account_buttons['login'].isHidden())
            self.assertTrue(panel.account_buttons['logout'].isHidden())
            load.assert_not_called()

    def test_startup_restores_persisted_signin_and_renews_expired_tokens_without_browser_login(self) -> None:
        from qtpy.QtCore import QTimer

        panel = self.make_panel()
        for expired in (False, True):
            with self.subTest(expired=expired):
                saved = tokens(expiry=time.time() - 60 if expired else None)
                path = self.account._path()
                self.account._credentials = saved
                self.account._save()
                original = path.read_bytes()
                self.account._loaded = False
                self.account._credentials = None
                self.controller.account = ''
                self.controller.changed.emit()
                requests = []
                ui_thread = threading.get_ident()

                def respond(request: httpx.Request) -> httpx.Response:
                    self.assertNotEqual(threading.get_ident(), ui_thread)
                    requests.append(request)
                    if request.url.path.endswith('/oauth/token'):
                        self.assertIn(b'grant_type=refresh_token', request.content)
                        return httpx.Response(200, json={
                            'access_token': 'renewed-access', 'refresh_token': 'renewed-refresh', 'expires_in': 3600,
                        })
                    self.assertTrue(request.url.path.endswith('/models'))
                    return httpx.Response(200, json={'models': [{
                        'slug': 'available-after-restart', 'input_modalities': ['text', 'image'],
                        'supported_reasoning_levels': [{'effort': 'high'}],
                    }]})

                with patch.object(codex, '_http_client', side_effect=lambda proxy='': httpx.AsyncClient(
                    transport=httpx.MockTransport(respond)
                )), patch.object(self.account, 'login', side_effect=AssertionError('Browser login was requested')):
                    # MainWindow schedules this after its construction; opening
                    # the panel itself still performs no account or network IO.
                    QTimer.singleShot(0, self.controller.restoreSession)
                    self.app.processEvents()
                    self.finish_worker()

                self.assertEqual(self.controller.account, saved['email'])
                self.assertFalse(panel.account_buttons['logout'].isHidden())
                self.assertTrue(panel.account_buttons['login'].isHidden())
                self.assertEqual(list(pcfg.module.codex_models), ['available-after-restart'])
                self.assertEqual(requests[-1].headers['Authorization'], 'Bearer renewed-access' if expired else 'Bearer access')
                if expired:
                    restarted = codex.CodexAccount()
                    with patch.object(restarted, '_path', return_value=path):
                        restarted._load()
                    self.assertEqual(restarted._credentials['refresh_token'], 'renewed-refresh')
                else:
                    self.assertEqual(path.read_bytes(), original)

    def test_startup_without_saved_credentials_does_not_use_network(self) -> None:
        from qtpy.QtTest import QSignalSpy
        profile = pcfg.module.llm_profiles[0]
        profile.model = 'text-model'
        profile.vision_model = 'vision-model'
        sync_codex_profile(profile, CATALOG)
        published = QSignalSpy(self.controller.catalog_changed)
        panel = self.make_panel()
        with patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected network')):
            self.controller.restoreSession()
            self.finish_worker()
        self.assertFalse(panel.account_buttons['login'].isHidden())
        self.assertTrue(panel.account_buttons['logout'].isHidden())
        self.assertEqual(pcfg.module.codex_models, CATALOG)
        self.assertEqual(profile.model_options, list(CATALOG))
        self.assertEqual(profile.vision_model_options, ['vision-model'])
        self.assertEqual((profile.model, profile.vision_model), ('text-model', 'vision-model'))
        self.assertEqual(len(published), 0)

    def test_authenticated_empty_catalog_replaces_stale_options_without_hiding_codex(self) -> None:
        from qtpy.QtTest import QSignalSpy
        self.account._loaded = True
        self.account._credentials = tokens()
        profile = pcfg.module.llm_profiles[0]
        profile.model = 'saved-text'
        profile.vision_model = 'saved-vision'
        sync_codex_profile(profile, CATALOG)
        published = QSignalSpy(self.controller.catalog_changed)
        with patch.object(codex, '_http_client', side_effect=lambda proxy='': httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'models': []}))
        )):
            self.controller.restoreSession()
            self.finish_worker()
        self.assertEqual(self.controller.account, 'demo@example.com')
        self.assertEqual(pcfg.module.codex_models, {})
        self.assertEqual(len(published), 1)
        self.assertEqual(profile.model_options, default_codex_profile().model_options + ['saved-text'])
        self.assertEqual(profile.vision_model_options, default_codex_profile().vision_model_options + ['saved-vision'])
        self.assertEqual((profile.model, profile.vision_model), ('saved-text', 'saved-vision'))
        self.assertTrue(profile.support_text)
        self.assertTrue(profile.support_vision)
        self.assertTrue(profile.support_image)
        self.assertEqual(profile.image_model_options, ['gpt-image-2'])

    def test_startup_ignores_plaintext_and_shows_signed_out_without_error(self) -> None:
        path = self.account._path()
        path.write_text(json.dumps(tokens()))
        unchanged = path.read_bytes()
        panel = self.make_panel()
        with patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected network')), \
                patch.object(codex, '_system_keyring', side_effect=AssertionError('Unexpected vault access')):
            self.controller.restoreSession()
            self.finish_worker()
        self.assertEqual(self.controller.account, '')
        self.warning.assert_not_called()
        self.assertFalse(panel.account_buttons['login'].isHidden())
        self.assertTrue(panel.account_buttons['logout'].isHidden())
        self.assertTrue(panel.account_buttons['login'].isEnabled())
        self.assertEqual(pcfg.module.codex_models, CATALOG)
        self.assertEqual(path.read_bytes(), unchanged)

    def test_account_actions_switch_login_and_logout_without_showing_both(self) -> None:
        panel = self.make_panel()
        sign_in, sign_out = panel.account_buttons['login'], panel.account_buttons['logout']
        self.assertEqual(sign_in.text(), panel.tr('Sign in'))
        self.assertEqual(sign_out.text(), panel.tr('Sign out'))
        self.assertFalse(sign_in.icon().isNull())
        self.assertTrue(sign_out.icon().isNull())
        self.assertFalse(sign_out.isEnabled())

        def login(stop_event, show_url):
            self.account._loaded = True
            self.account._credentials = tokens()

        with patch.object(self.account, 'login', side_effect=login), patch.object(
            self.account, 'catalog', return_value=copy.deepcopy(CATALOG),
        ) as catalog:
            sign_in.click()
            self.finish_worker()
        self.assertEqual(catalog.call_count, 1)
        self.assertFalse(panel.account_buttons['logout'].isHidden())
        self.assertTrue(panel.account_buttons['login'].isHidden())
        self.assertFalse(sign_in.isEnabled())
        self.assertTrue(sign_out.isEnabled())
        sign_out.click()
        self.finish_worker()
        self.assertFalse(panel.account_buttons['login'].isHidden())
        self.assertTrue(panel.account_buttons['logout'].isHidden())
        self.assertEqual(self.controller.account, '')
        self.assertEqual(pcfg.module.codex_models, CATALOG)

    def test_login_commit_is_visible_even_when_catalog_fails_or_is_cancelled(self) -> None:
        panel = self.make_panel()
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                self.account._loaded = False
                self.account._credentials = None
                self.controller.account = ''
                self.controller.changed.emit()

                def login(stop_event, show_url):
                    self.account._loaded = True
                    self.account._credentials = tokens()

                def catalog(stop_event):
                    if cancelled:
                        stop_event.set()
                        raise LLMRequestStopped()
                    raise RuntimeError('Temporary model-list failure')

                with patch.object(self.account, 'login', side_effect=login), \
                        patch.object(self.account, 'catalog', side_effect=catalog):
                    panel.account_buttons['login'].click()
                    self.finish_worker()
                self.assertEqual(self.controller.account, 'demo@example.com')
                self.assertFalse(panel.account_buttons['logout'].isHidden())
                self.assertTrue(panel.account_buttons['login'].isHidden())
                self.assertEqual(pcfg.module.codex_models, CATALOG)

    def test_failed_logout_and_refresh_keep_signout_but_committed_cancel_clears_it(self) -> None:
        self.account._loaded = True
        self.account._credentials = tokens()
        self.controller.account = 'demo@example.com'
        panel = self.make_panel()
        with patch.object(self.account, 'logout', side_effect=LLMUserActionRequiredError('Could not remove credentials')):
            panel.account_buttons['logout'].click()
            self.finish_worker()
        self.assertFalse(panel.account_buttons['logout'].isHidden())
        self.assertTrue(panel.account_buttons['login'].isHidden())
        with patch.object(self.account, 'catalog', side_effect=RuntimeError('Temporary connection failure')):
            panel.account_buttons['refresh'].click()
            self.finish_worker()
        self.assertFalse(panel.account_buttons['logout'].isHidden())
        self.assertTrue(panel.account_buttons['login'].isHidden())
        self.assertEqual(pcfg.module.codex_models, CATALOG)

        def logout(stop_event):
            self.account._credentials = None
            stop_event.set()
            raise LLMRequestStopped()

        with patch.object(self.account, 'logout', side_effect=logout):
            panel.account_buttons['logout'].click()
            self.finish_worker()
        self.assertFalse(panel.account_buttons['login'].isHidden())
        self.assertTrue(panel.account_buttons['logout'].isHidden())
        self.assertEqual(pcfg.module.codex_models, CATALOG)

    def test_cancelled_unknown_refresh_keeps_previously_known_account(self) -> None:
        self.controller.account = 'previous@example.com'
        panel = self.make_panel()

        def cancelled(stop_event):
            stop_event.set()
            raise LLMRequestStopped()

        with patch.object(self.account, 'catalog', side_effect=cancelled):
            panel.account_buttons['refresh'].click()
            self.finish_worker()
        self.assertEqual(self.controller.account, 'previous@example.com')
        self.assertFalse(panel.account_buttons['logout'].isHidden())
        self.assertTrue(panel.account_buttons['login'].isHidden())
        self.assertEqual(pcfg.module.codex_models, CATALOG)

    def test_worker_errors_show_after_controls_restore_and_cancellation_suppresses_them(self) -> None:
        self.account._loaded = True
        self.account._credentials = tokens()
        self.controller.account = 'demo@example.com'
        panel = self.make_panel()
        warning_states = []
        self.warning.side_effect = lambda parent, title, message: warning_states.append((
            panel.refresh_button.isEnabled(), panel.account_buttons['logout'].isEnabled(),
            self.controller.worker is None,
        ))
        for failure, cancelled in (
            (RuntimeError('private provider URL'), False),
            (LLMUserActionRequiredError('Sign in again to refresh models.'), False),
            (RuntimeError('private provider URL'), True),
        ):
            with self.subTest(cancelled=cancelled, error=type(failure).__name__):
                self.warning.reset_mock()
                warning_states.clear()

                def catalog(stop_event: threading.Event) -> dict:
                    if cancelled:
                        stop_event.set()
                    raise failure

                with patch.object(self.account, 'catalog', side_effect=catalog):
                    self.controller.start('refresh')
                    self.finish_worker()
                self.assertTrue(panel.refresh_button.isEnabled())
                self.assertTrue(panel.account_buttons['logout'].isEnabled())
                self.assertEqual(self.controller.account, 'demo@example.com')
                self.assertEqual(pcfg.module.codex_models, CATALOG)
                if cancelled:
                    self.warning.assert_not_called()
                else:
                    self.warning.assert_called_once()
                    message = self.warning.call_args.args[2]
                    self.assertEqual(warning_states, [(True, True, True)])
                    self.assertNotIn('private provider URL', message)
                    if isinstance(failure, LLMUserActionRequiredError):
                        self.assertEqual(message, str(failure))

    def test_browser_open_failure_cancels_login_and_shows_only_actionable_browser_error(self) -> None:
        from ballontranslator.ui import codex_account

        panel = self.make_panel()
        submitted = threading.Event()
        cancelled = threading.Event()

        def login(stop_event: threading.Event, show_url: Callable[[str], None]) -> None:
            show_url('https://auth.openai.com/authorize?sensitive=private-value')
            submitted.set()
            if stop_event.wait(2):
                cancelled.set()
            raise RuntimeError('private provider failure after browser cancellation')

        with patch.object(codex_account.QDesktopServices, 'openUrl', return_value=False), \
                patch.object(self.account, 'login', side_effect=login), \
                patch.object(self.account, 'catalog', side_effect=AssertionError('Unexpected catalog request')):
            self.controller.start('login')
            self.assertTrue(submitted.wait(2))
            self.app.processEvents()
            if self.controller.worker is not None:
                self.finish_worker()
        self.assertTrue(cancelled.is_set())
        self.warning.assert_called_once()
        message = self.warning.call_args.args[2]
        self.assertIn('browser', message)
        self.assertIn('try signing in again', message)
        self.assertNotIn('private', message)
        self.assertEqual(self.controller.account, '')
        self.assertTrue(panel.account_buttons['login'].isEnabled())

    def test_browser_open_succeeds_and_ignores_url_after_cancellation(self) -> None:
        from ballontranslator.ui import codex_account

        worker = codex_account.CodexAccountWorker('login', self.controller)
        self.controller.worker = worker
        worker.login_url.connect(self.controller.openLoginUrl)
        try:
            with patch.object(codex_account.QDesktopServices, 'openUrl', return_value=True) as open_url:
                worker.login_url.emit('https://auth.openai.com/authorize')
                self.assertTrue(open_url.called)
                self.assertFalse(worker.stop_event.is_set())
                worker.stop_event.set()
                open_url.reset_mock()
                worker.login_url.emit('https://auth.openai.com/authorize')
                open_url.assert_not_called()
            self.warning.assert_not_called()
        finally:
            self.controller.worker = None
            worker.deleteLater()

    def test_login_refreshes_catalog_once_and_publishes_even_an_unchanged_list(self) -> None:
        from qtpy.QtCore import QCoreApplication, QEvent
        from qtpy.QtTest import QSignalSpy
        from ballontranslator.ui.codex_settings import CodexSettingsPanel

        for cached in ({}, CATALOG):
            with self.subTest(cached=bool(cached)), \
                    patch.object(pcfg.module, 'codex_models', copy.deepcopy(cached)), \
                    patch.object(pcfg.module, 'llm_profiles', [default_codex_profile()]):
                profile = pcfg.module.llm_profiles[0]
                profile.model = 'vision-model'
                panel = CodexSettingsPanel()
                published = QSignalSpy(self.controller.catalog_changed)
                ui_updated = QSignalSpy(panel.profile_ui_updated)
                operations = []
                ui_thread = threading.get_ident()

                def login(stop_event, show_url) -> None:
                    self.assertNotEqual(threading.get_ident(), ui_thread)
                    operations.append('login')
                    self.account._loaded = True
                    self.account._credentials = {**tokens(), 'email': 'signed-in@example.com'}

                def catalog(stop_event: threading.Event) -> dict:
                    self.assertNotEqual(threading.get_ident(), ui_thread)
                    self.assertEqual(operations, ['login'])
                    operations.append('catalog')
                    return copy.deepcopy(CATALOG)

                try:
                    with patch.object(codex.account, 'login', side_effect=login), \
                            patch.object(codex.account, 'catalog', side_effect=catalog):
                        self.controller.start('login')
                        worker = self.controller.worker
                        self.assertTrue(worker.wait(2000))
                        self.app.processEvents()
                    self.assertIsNone(self.controller.worker)
                    self.assertEqual(operations, ['login', 'catalog'])
                    self.assertEqual(len(published), 1)
                    self.assertEqual(len(ui_updated), 1)
                    self.assertEqual(self.controller.account, 'signed-in@example.com')
                    self.assertEqual(pcfg.module.codex_models, CATALOG)
                    self.assertEqual(profile.model, 'vision-model')
                    self.assertEqual(profile.model_options, list(CATALOG))
                    self.assertEqual(profile.vision_model_options, ['vision-model'])
                    self.assertTrue(profile.support_image)
                    self.assertEqual(panel.param_widgets['model'].currentText(), 'vision-model')
                finally:
                    panel.deleteLater()
                    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_failed_refresh_keeps_catalog_and_deleted_widget_does_not_own_worker(self) -> None:
        from qtpy.QtCore import QCoreApplication, QEvent
        from ballontranslator.ui.codex_settings import CodexSettingsPanel
        release = threading.Event()

        def delayed_failure(stop_event: threading.Event) -> None:
            release.wait(2)
            raise RuntimeError('Temporary connection failure.')

        with patch.object(self.account, 'catalog', side_effect=delayed_failure):
            widget = CodexSettingsPanel()
            ref = weakref.ref(widget)
            self.controller.account = 'previous@example.com'
            self.controller.start('refresh')
            worker = self.controller.worker
            worker_ref = weakref.ref(worker)
            try:
                widget.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                del widget
                gc.collect()
                self.assertIsNone(ref())
            finally:
                release.set()
                self.assertTrue(worker.wait(2000))
                self.app.processEvents()
            self.assertIsNone(self.controller.worker)
            self.assertEqual(pcfg.module.codex_models, CATALOG)
            self.assertEqual(self.controller.account, 'previous@example.com')
            del worker
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            gc.collect()
            self.assertIsNone(worker_ref())

    def test_new_pipeline_job_uses_a_new_event_without_reviving_old_cancellation(self) -> None:
        from ballontranslator.ui.module_manager import ImgtransThread, InpaintThread, OCRThread, TextDetectThread, TranslateThread
        from ballontranslator.utils.proj_imgtrans import ProjImgTrans
        workers = [TextDetectThread(), OCRThread(), TranslateThread(), InpaintThread()]
        with patch('ballontranslator.ui.module_manager.register_global_callback'):
            thread = ImgtransThread(*workers)
        self.addCleanup(thread.deleteLater)
        for worker in workers:
            self.addCleanup(worker.deleteLater)
        with patch.object(thread, 'start'):
            thread.runImgtransPipeline(ProjImgTrans())
            first = thread.stop_event
            thread.requestStop()
            thread.runBlktransPipeline([], 0, [])
            second = thread.stop_event
            self.assertIsNot(second, first)
            self.assertTrue(first.is_set())
            self.assertFalse(second.is_set())
            thread.runImgtransPipeline(ProjImgTrans())
            self.assertIsNot(thread.stop_event, second)


if __name__ == '__main__':
    unittest.main()
