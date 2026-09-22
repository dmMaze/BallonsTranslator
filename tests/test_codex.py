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
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from ballontranslator.modules import codex
from ballontranslator.modules.context.errors import ContextLengthError
from ballontranslator.modules.exceptions import LLMRequestStopped, LLMUserActionRequiredError
from ballontranslator.modules.llm_image import LLMImageRequester
from ballontranslator.modules.ocr.ocr_llm import LLMOCR
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils.config import ModuleConfig, ProgramConfig, json_dump_program_config, pcfg
from ballontranslator.utils.llm_profiles import default_profile, normalize_codex_models, profile_to_export_dict, sync_codex_profile

CATALOG = {'vision-model': {'modalities': ['text', 'image'], 'efforts': ['low', 'high']},
           'text-model': {'modalities': ['text'], 'efforts': ['none']}}


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
        self.profile = default_profile('Codex')
        self.profile.model = self.profile.vision_model = 'vision-model'
        sync_codex_profile(self.profile, CATALOG)
        for patcher in (
            patch.object(codex, 'account', self.account),
            patch.object(self.account, '_path', return_value=self.path),
            patch.object(codex, '_http_client', side_effect=lambda proxy='': httpx.AsyncClient(transport=httpx.MockTransport(self.responder))),
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

    def test_stateless_payload_preserves_roles_images_schema_usage_and_omits_tools(self) -> None:
        self.profile.thinking_level = 'high'
        self.events.insert(0, {'type': 'response.output_text.delta', 'delta': 'partial garbage'})
        result = self.request()
        self.assertEqual(result.content, '{"1":"hello"}')
        self.assertEqual(result.usage.total_tokens, 15)
        self.assertEqual(result.usage.prompt_tokens_details['cached_tokens'], 8)
        payload = json.loads(self.requests[0].content)
        self.assertEqual(payload['instructions'], 'Translate numbered blocks.\n\nKnown names.')
        self.assertEqual([item['role'] for item in payload['input']], ['user', 'assistant', 'user'])
        self.assertEqual(payload['input'][1]['content'][0], {'type': 'output_text', 'text': 'old translation'})
        self.assertEqual(payload['input'][-1]['content'][-1]['detail'], 'high')
        self.assertEqual(payload['text']['format']['schema'], {'type': 'object'})
        self.assertEqual(payload['reasoning'], {'effort': 'high'})
        self.assertEqual(payload['tools'], [])
        self.assertFalse(payload['store'])
        self.assertEqual(payload['prompt_cache_key'], self.requests[0].headers['session_id'])
        self.assertEqual(self.requests[0].headers['Authorization'], 'Bearer access')
        self.assertNotIn('previous_response_id', payload)

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

    def test_logout_invalidates_inflight_requests_and_only_removes_owned_credentials(self) -> None:
        legacy = self.path.with_name('auth.json')
        legacy.write_text('legacy-sdk-credentials')
        self.account._save()
        generation = self.account.generation
        self.account.logout(threading.Event())
        self.assertGreater(self.account.generation, generation)
        self.assertFalse(self.path.exists())
        self.assertEqual(legacy.read_text(), 'legacy-sdk-credentials')
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
        self.assertEqual(json.loads(self.path.read_text()), first)
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
        self.assertEqual(json.loads(self.path.read_text())['refresh_token'], 'new-refresh')
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
        self.assertEqual(received[0].headers['session_id'], received[2].headers['session_id'])
        self.assertEqual(received[2].headers['Authorization'], 'Bearer rotated')

    def test_catalog_uses_subscription_visibility_and_explicit_modalities(self) -> None:
        def respond(request):
            self.assertEqual(request.headers['accept'], 'application/json')
            return httpx.Response(200, json={'models': [
                None, {}, {'slug': 42},
                {'slug': 'vision-model', 'visibility': 'list', 'supported_in_api': False,
                 'input_modalities': ['text', 'image'], 'supported_reasoning_levels': [None, {'effort': 'high'}]},
                {'slug': 'text-model', 'visibility': 'list', 'supported_reasoning_levels': None},
                {'slug': 'hidden-model', 'visibility': 'hide'},
            ]})
        self.responder = respond
        label, models = self.account.catalog(threading.Event())
        self.assertEqual(label, 'demo@example.com')
        self.assertEqual(models['vision-model']['modalities'], ['text', 'image'])
        self.assertEqual(models['text-model']['modalities'], ['text'])
        self.assertNotIn('hidden-model', models)

    def test_browser_login_checks_state_and_pkce_and_cancellation_preserves_credentials(self) -> None:
        responses, authorization, exchanged = [], {}, []
        legacy = self.path.with_name('auth.json')
        legacy.write_text('legacy-sdk-credentials')

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
        self.assertEqual(legacy.read_text(), 'legacy-sdk-credentials')
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
        project._image_info = {page: {'finish_code': 0} for page in project.pages}
        translator = LLMTranslator('日本語', 'English')
        translator.set_stop_event(threading.Event())
        translator.set_param_value('retry attempts', 1)
        received = []

        def respond(request):
            body = json.loads(request.content)
            received.append(body)
            current = body['input'][-1]['content'][0]['text'].split('INPUT:\n', 1)[1]
            translations = [{'id': item['id'], 'translation': f'translated-{item["id"]}'} for item in json.loads(current)]
            if len(received) == 4:
                translations.append(translations[0])
            return httpx.Response(200, text=sse(completion(json.dumps({'translations': translations}))))

        self.responder = respond
        settings = {'llm_profiles': [self.profile], 'translator_llm_id': self.profile.id,
                    'llm_translate_context': LLMTranslateContext.HISTORY, 'llm_prior_context_token_budget': 4096,
                    'llm_translate_vision': False, 'llm_translate_summary_memory': False, 'llm_glossary_path': ''}
        with patch.dict(pcfg.module.__dict__, settings), patch.object(translator, '_respect_delay'):
            for page in ('0', '1', '2'):
                translator.translate_textblk_lst(project.pages[page], project=project, page_key=page, full_page=True)
                project.mark_translation_finished(page, 'English')
            previous_window = translator._history_window
            with self.assertRaises(InvalidNumTranslations):
                translator.translate_textblk_lst(project.pages['3'], project=project, page_key='3', full_page=True)
        self.assertIs(translator._history_window, previous_window)
        self.assertEqual(project.pages['3'][0].translation, 'previous')
        self.assertEqual(len({body['prompt_cache_key'] for body in received}), 1)
        for previous, current in zip(received, received[1:]):
            self.assertEqual(previous['text']['format'], current['text']['format'])
            self.assertEqual(previous['instructions'], current['instructions'])
            self.assertEqual(previous['input'], current['input'][:len(previous['input'])])
            history = json.loads(current['input'][-2]['content'][0]['text'])
            self.assertIsInstance(history['translations'], list)
        self.assertIn('"translations":[{"id":1,"translation":"Translated text"}]', received[0]['instructions'])

    def test_summary_only_page_uses_same_codex_schema_and_persists_through_owner(self) -> None:
        import numpy as np
        from ballontranslator.utils.config import LLMTranslateContext
        from ballontranslator.utils.proj_imgtrans import ProjImgTrans
        from ballontranslator.utils.textblock import TextBlock
        project = ProjImgTrans()
        project.pages = {'0': [TextBlock(text=['source'])], '1': [], '2': [TextBlock(text=['next'])]}
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

    def test_catalog_is_single_persistent_source_and_selections_survive_logout(self) -> None:
        profile = default_profile('Codex')
        profile.model = profile.vision_model = 'vision-model'
        profile.api_key = 'must-not-export'
        module = ModuleConfig(llm_profiles=[profile], codex_models=CATALOG)
        saved = json.loads(json_dump_program_config(ProgramConfig(module=module)))
        self.assertNotIn('model_options', saved['module']['llm_profiles'][0])
        self.assertNotIn('must-not-export', json.dumps(saved))
        self.assertEqual(saved['module']['codex_models'], CATALOG)
        self.assertEqual(profile_to_export_dict(profile)['api_key'], '')
        sync_codex_profile(profile, {})
        self.assertEqual((profile.model, profile.vision_model), ('vision-model', 'vision-model'))
        self.assertFalse(profile.support_image)

    def test_malformed_catalog_discards_only_invalid_optional_data(self) -> None:
        self.assertEqual(normalize_codex_models({'good': {'modalities': ['text', 'bad'], 'efforts': ['high', None]}, 'bad': None}),
                         {'good': {'modalities': ['text'], 'efforts': ['high']}})

    def test_image_backend_never_falls_through_to_api(self) -> None:
        requester = LLMImageRequester()
        for backend in ('codex', 'unavailable'):
            profile = default_profile('OpenAI')
            profile.backend = backend
            with self.assertRaises(LLMUserActionRequiredError):
                requester._initialize_client(profile)


class CodexAccountWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from qtpy.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from ballontranslator.ui.codex_account import CodexAccountController
        self.controller = CodexAccountController.instance()
        self.controller.account = ''

    def test_config_rendering_never_loads_credentials_or_starts_http(self) -> None:
        from ballontranslator.ui.llm_profile_widgets import ProfileCardWidget
        profile = default_profile('Codex')
        profile.model = 'vision-model'
        with patch.object(pcfg.module, 'codex_models', CATALOG), patch.object(codex.account, '_load') as load:
            card = ProfileCardWidget(profile)
            self.addCleanup(card.deleteLater)
            card.toggleImageSupport()
            self.assertFalse(profile.support_image)
            self.assertEqual(profile.vision_model_options, ['vision-model'])
            card.model_combo.setCurrentText('text-model')
            self.assertEqual(profile.thinking_level_options, ['Auto', 'Disabled'])
            load.assert_not_called()

    def test_failed_refresh_keeps_catalog_and_deleted_widget_does_not_own_worker(self) -> None:
        from qtpy.QtCore import QCoreApplication, QEvent
        from ballontranslator.ui.codex_account import CodexAccountWidget, CodexAccountWorker
        release = threading.Event()

        def delayed_failure(worker):
            release.wait(2)
            worker.error = 'Temporary connection failure.'

        with patch.object(CodexAccountWorker, 'run', delayed_failure), patch.object(pcfg.module, 'codex_models', CATALOG):
            widget = CodexAccountWidget(None)
            ref = weakref.ref(widget)
            self.controller.account = 'previous@example.com'
            self.controller.start('refresh')
            worker = self.controller.worker
            widget.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            del widget
            gc.collect()
            self.assertIsNone(ref())
            release.set()
            self.assertTrue(worker.wait(2000))
            self.app.processEvents()
            self.assertIsNone(self.controller.worker)
            self.assertEqual(pcfg.module.codex_models, CATALOG)
            self.assertEqual(self.controller.account, 'previous@example.com')

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
