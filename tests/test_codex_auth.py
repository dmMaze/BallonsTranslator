import asyncio
import base64
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from ballontranslator.modules import codex
from ballontranslator.modules.exceptions import (
    CodexSignInRequiredError, LLMRequestStopped, LLMUserActionRequiredError,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile


class CodexAuthenticationTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'http-auth.json'
        self.account = codex.CodexAccount()
        self.account._loaded = True
        self.account._credentials = {
            'access_token': 'test-access', 'refresh_token': 'test-refresh',
            'account_id': 'test-account', 'email': 'test@example.com',
            'expires_at': time.time() + 3600,
        }
        self.profile = default_codex_profile()
        self.profile.model = 'test-model'
        self.requests = []
        self.respond = self.success
        for patcher in (
            patch.object(codex, 'account', self.account),
            patch.object(self.account, '_path', return_value=self.path),
            patch.object(codex, '_system_keyring', side_effect=ImportError),
            patch.object(codex, '_http_client', side_effect=self.client),
            patch.object(pcfg.module, 'codex_models', {
                'test-model': {'modalities': ['text', 'image'], 'efforts': ['high']},
            }),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def client(self, proxy: str = '') -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def success(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/oauth/token'):
            return httpx.Response(200, json={'access_token': 'renewed-access', 'expires_in': 3600})
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'models': [{'slug': 'test-model', 'input_modalities': ['text']}]})
        if request.url.path.endswith('/responses'):
            if json.loads(request.content).get('tools'):
                event = {'type': 'response.completed', 'response': {
                    'status': 'completed', 'output': [{'type': 'image_generation_call',
                    'status': 'completed', 'result': base64.b64encode(b'image').decode()}],
                }}
                return httpx.Response(200, text='data: ' + json.dumps(event) + '\n\n')
            event = {'type': 'response.completed', 'response': {
                'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'translated'}]}],
            }}
            return httpx.Response(200, text='data: ' + json.dumps(event) + '\n\n')
        return httpx.Response(200, json={'data': [{'b64_json': base64.b64encode(b'image').decode()}]})

    def request(self, target: str) -> object:
        if target == 'catalog':
            return self.account.catalog(threading.Event())
        if target == 'chat':
            return codex.request_chat_completion(self.profile, {
                'model': self.profile.model, 'messages': [{'role': 'user', 'content': 'Translate.'}],
            }, None, 'job-key')
        if target == 'assisted-image':
            return codex.request_image('gpt-image-2', 'Clean the image.', None, None, None,
                                       reasoning_model='test-model', cache_key='image-job')
        return codex.request_image('gpt-image-2', 'Clean the image.', None, None, None)

    def test_missing_signin_precedes_model_validation_and_http(self) -> None:
        self.account._credentials = None
        with patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected HTTP')):
            calls = (
                lambda: self.account.catalog(threading.Event()),
                lambda: codex.request_chat_completion(self.profile, {}, None, 'job'),
                lambda: codex.request_image('', '', None, None, None),
            )
            for call in calls:
                with self.assertRaises(CodexSignInRequiredError) as caught:
                    call()
                self.assertFalse(caught.exception.invalid)
        self.assertFalse(self.account.auth_invalid)
        self.assertEqual(self.account.cached_account_label, '')

    def test_preflight_restores_credentials_without_http_or_rewriting_file(self) -> None:
        self.account._save()
        unchanged = self.path.read_bytes()
        self.account._loaded = False
        self.account._credentials = None
        with patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected HTTP')):
            self.account.require_sign_in()
        self.assertEqual(self.account.cached_account_label, 'test@example.com')
        self.assertEqual(self.path.read_bytes(), unchanged)

    def test_each_transport_renews_once_before_reporting_exhausted_401(self) -> None:
        for target in ('catalog', 'chat', 'image', 'assisted-image'):
            with self.subTest(target=target):
                self.account.auth_invalid = False
                self.account._credentials['access_token'] = 'test-access'
                self.account._save()
                self.requests.clear()
                generation = self.account.generation

                def respond(request: httpx.Request) -> httpx.Response:
                    if request.url.path.endswith('/oauth/token'):
                        return self.success(request)
                    return httpx.Response(401, json={'error': {'message': 'private-provider-detail'}})

                self.respond = respond
                with self.assertRaises(CodexSignInRequiredError) as caught:
                    self.request(target)
                self.assertTrue(caught.exception.invalid)
                self.assertNotIn('private', str(caught.exception))
                self.assertEqual(len(self.requests), 3)
                self.assertTrue(self.requests[1].url.path.endswith('/oauth/token'))
                self.assertEqual(self.requests[-1].headers['Authorization'], 'Bearer renewed-access')
                self.assertTrue(self.account.auth_invalid)
                self.assertEqual(self.account.generation, generation)
                self.assertEqual(self.account.cached_account_label, '')
                self.assertTrue(self.path.is_file())
                with patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected HTTP')):
                    with self.assertRaises(CodexSignInRequiredError) as cached:
                        self.account.require_sign_in()
                self.assertTrue(cached.exception.invalid)

    def test_each_transport_keeps_a_successful_401_recovery_signed_in(self) -> None:
        for target in ('catalog', 'chat', 'image', 'assisted-image'):
            with self.subTest(target=target):
                self.account._credentials['access_token'] = 'test-access'
                self.requests.clear()

                def respond(request: httpx.Request) -> httpx.Response:
                    if request.headers.get('Authorization') == 'Bearer test-access':
                        return httpx.Response(401, json={'error': {'code': 'token_expired'}})
                    return self.success(request)

                self.respond = respond
                self.request(target)
                self.assertEqual(len(self.requests), 3)
                self.assertFalse(self.account.auth_invalid)
                self.assertEqual(self.account.cached_account_label, 'test@example.com')

    def test_oauth_refresh_auth_failures_preserve_saved_credentials(self) -> None:
        for error in (
            {'error': 'invalid_grant', 'error_description': 'private expired token'},
            {'error': {'code': 'refresh_token_reused', 'message': 'private token'}},
            {'error': {'type': 'invalid_request_error', 'message': 'Refresh token is invalid: private'}},
        ):
            with self.subTest(error=error):
                self.account.auth_invalid = False
                self.account._credentials['expires_at'] = 0
                self.account._save()
                unchanged = self.path.read_bytes()
                self.requests.clear()
                self.respond = lambda request: httpx.Response(400, json=error)
                with self.assertRaises(CodexSignInRequiredError) as caught:
                    self.request('catalog')
                self.assertTrue(caught.exception.invalid)
                self.assertTrue(self.account.auth_invalid)
                self.assertEqual(len(self.requests), 1)
                self.assertTrue(self.requests[0].url.path.endswith('/oauth/token'))
                self.assertEqual(self.path.read_bytes(), unchanged)
                self.assertNotIn('private', str(caught.exception))

    def test_permission_quota_and_model_403_do_not_invalidate_signin(self) -> None:
        for code, message in (
            ('permission_denied', 'Authentication scope is insufficient.'),
            ('insufficient_scope', ''), ('usage_limit_reached', ''), ('model_not_found', ''),
            ('invalid_request_error', 'Refresh token usage limit reached.'),
        ):
            with self.subTest(code=code):
                self.requests.clear()
                self.respond = lambda request: httpx.Response(403, json={'error': {'code': code, 'message': message}})
                with self.assertRaises(LLMUserActionRequiredError) as caught:
                    self.request('chat')
                self.assertNotIsInstance(caught.exception, CodexSignInRequiredError)
                self.assertFalse(self.account.auth_invalid)
                self.assertEqual(self.account.cached_account_label, 'test@example.com')
                self.assertEqual(len(self.requests), 1)

    def test_structured_and_stream_auth_errors_have_safe_typed_failures(self) -> None:
        for event in (
            {'type': 'response.failed', 'response': {'error': {'code': 'token_expired', 'message': 'private'}}},
            {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'private'}},
            {'type': 'error', 'status_code': 401, 'error': {'message': 'private'}},
        ):
            with self.subTest(event=event):
                self.account.auth_invalid = False
                self.requests.clear()
                self.respond = lambda request: self.success(request) if request.url.path.endswith('/oauth/token') else httpx.Response(200, text='data: ' + json.dumps(event) + '\n\n')
                with self.assertRaises(CodexSignInRequiredError) as caught:
                    self.request('chat')
                self.assertTrue(caught.exception.invalid)
                self.assertTrue(self.account.auth_invalid)
                self.assertNotIn('private', str(caught.exception))
                self.assertEqual(len(self.requests), 3)
        for target in ('catalog', 'image'):
            with self.subTest(target=target):
                self.account.auth_invalid = False
                self.requests.clear()
                self.respond = lambda request: self.success(request) if request.url.path.endswith('/oauth/token') else httpx.Response(200, json={'error': {'code': 'invalid_token'}})
                with self.assertRaises(CodexSignInRequiredError):
                    self.request(target)
                self.assertTrue(self.account.auth_invalid)
                self.assertEqual(len(self.requests), 3)

    def test_obsolete_auth_failure_does_not_invalidate_new_account(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.account.invalidate()
            self.account._update_tokens({'access_token': 'new-account-access'}, self.account._credentials)
            return httpx.Response(400, json={'error': {'code': 'invalid_token'}})

        self.respond = respond
        with self.assertRaises(LLMRequestStopped):
            self.request('chat')
        self.assertFalse(self.account.auth_invalid)
        self.assertEqual(self.account.cached_account_label, 'test@example.com')

    def test_auth_failure_reuses_concurrent_renewal_before_or_after_rejection(self) -> None:
        for target in ('catalog', 'chat', 'image', 'assisted-image'):
            for renew_first in (False, True):
                with self.subTest(target=target, renew_first=renew_first):
                    self.account._credentials['access_token'] = 'test-access'
                    entered, renewal_entered, release_failure, release_renewal = (threading.Event() for _ in range(4))
                    rejected = threading.Event()
                    requests, failures, results = [], [], []
                    generation = self.account.generation

                    async def respond(request: httpx.Request) -> httpx.Response:
                        requests.append(request)
                        if request.url.path.endswith('/oauth/token'):
                            renewal_entered.set()
                            while not release_renewal.is_set():
                                await asyncio.sleep(0.005)
                            return self.success(request)
                        if request.headers.get('Authorization') != 'Bearer test-access':
                            return self.success(request)
                        entered.set()
                        while not release_failure.is_set():
                            await asyncio.sleep(0.005)
                        rejected.set()
                        payload = {'error': {'code': 'invalid_token'}}
                        if target in ('chat', 'assisted-image'):
                            return httpx.Response(200, text='data: ' + json.dumps({'type': 'error', **payload}) + '\n\n')
                        return httpx.Response(200, json=payload)

                    def request() -> None:
                        try:
                            results.append(self.request(target))
                        except Exception as error:
                            failures.append(error)

                    def renew() -> None:
                        async def run() -> None:
                            async with codex._http_client() as client:
                                await self.account.tokens(client, 'test-access')
                        try:
                            asyncio.run(run())
                        except Exception as error:
                            failures.append(error)

                    with patch.object(codex, '_http_client', side_effect=lambda proxy='': httpx.AsyncClient(
                        transport=httpx.MockTransport(respond)
                    )), patch.object(self.account, '_save'):
                        request_worker = threading.Thread(target=request)
                        renewal_worker = threading.Thread(target=renew)
                        request_worker.start()
                        try:
                            self.assertTrue(entered.wait(2))
                            renewal_worker.start()
                            self.assertTrue(renewal_entered.wait(2))
                            if not renew_first:
                                release_failure.set()
                                self.assertTrue(rejected.wait(2))
                            release_renewal.set()
                            renewal_worker.join(2)
                            self.assertFalse(renewal_worker.is_alive())
                            release_failure.set()
                        finally:
                            release_failure.set()
                            release_renewal.set()
                            request_worker.join(2)
                            if renewal_worker.ident is not None:
                                renewal_worker.join(2)
                    self.assertFalse(request_worker.is_alive())
                    self.assertEqual(failures, [])
                    self.assertEqual(len(results), 1)
                    self.assertEqual(len(requests), 3)
                    service_requests = [request for request in requests if not request.url.path.endswith('/oauth/token')]
                    self.assertEqual(service_requests[0].content, service_requests[1].content)
                    self.assertEqual(service_requests[0].headers.get('session_id'), service_requests[1].headers.get('session_id'))
                    self.assertFalse(self.account.auth_invalid)
                    self.assertEqual(self.account.generation, generation)

    def test_exhausted_stale_token_failure_remains_retryable(self) -> None:
        for target in ('catalog', 'chat', 'image', 'assisted-image'):
            with self.subTest(target=target), patch.object(self.account, '_save'):
                self.account._credentials['access_token'] = 'test-access'
                self.requests.clear()

                def respond(request: httpx.Request) -> httpx.Response:
                    if request.url.path.endswith('/oauth/token'):
                        return self.success(request)
                    if request.headers['Authorization'] == 'Bearer test-access':
                        return httpx.Response(401)
                    # A different request has committed a newer token before
                    # this last response rejects the token used for its replay.
                    self.account._update_tokens({'access_token': 'newer-access'}, self.account._credentials)
                    return httpx.Response(401)

                self.respond = respond
                with self.assertRaises(RuntimeError) as caught:
                    self.request(target)
                self.assertNotIsInstance(caught.exception, (CodexSignInRequiredError, LLMRequestStopped))
                self.assertFalse(self.account.auth_invalid)
                self.assertEqual(self.account._credentials['access_token'], 'newer-access')
                self.assertEqual(len(self.requests), 3)
                self.account.require_sign_in()

    def test_known_http401_does_not_depend_on_reading_its_body(self) -> None:
        closed = []

        class Unreadable(httpx.AsyncByteStream):
            async def __aiter__(self):
                raise AssertionError('A known 401 does not need its error body.')
                yield b''

            async def aclose(self) -> None:
                closed.append(True)

        self.respond = lambda request: self.success(request) if request.url.path.endswith('/oauth/token') else httpx.Response(401, stream=Unreadable())
        with self.assertRaises(CodexSignInRequiredError):
            self.request('chat')
        self.assertTrue(self.account.auth_invalid)
        self.assertEqual(len(closed), 2)

    def test_new_token_commit_clears_invalid_state_and_cancelled_preflight_does_no_io(self) -> None:
        self.account.auth_invalid = True
        self.account._update_tokens({'access_token': 'new-account-access'}, self.account._credentials)
        self.assertFalse(self.account.auth_invalid)
        self.account.require_sign_in()
        stopped = threading.Event()
        stopped.set()
        with patch.object(self.account, '_load', side_effect=AssertionError('Unexpected credential IO')):
            with self.assertRaises(LLMRequestStopped):
                self.account.require_sign_in(stopped)
        stopped.clear()
        with patch.object(self.account, '_load', side_effect=stopped.set):
            with self.assertRaises(LLMRequestStopped):
                self.account.require_sign_in(stopped)


if __name__ == '__main__':
    unittest.main()
