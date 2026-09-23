import base64
import copy
import io
import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import httpx
from PIL import Image

from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMBaseURLRequiredError,
    LLMModelRequiredError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.modules.inpaint.inpaint_llm import LLMInpaint
from ballontranslator.modules import codex, image_generation, llm_image
from ballontranslator.modules.llm_image import (
    LLMImageRequester,
    LLMImageRequestPolicy,
    _SharedLLMImageThrottle,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, default_profile, sync_codex_profile
from ballontranslator.utils.textblock import TextBlock


def _encoded_png() -> str:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf8')


def _png_bytes() -> bytes:
    return base64.b64decode(_encoded_png())


def _gemini_image_profile():
    profile = default_profile('Gemini')
    profile.api_key = 'gemini-key'
    profile.support_image = True
    profile.image_base_url = 'https://generativelanguage.googleapis.com/v1beta'
    profile.image_model = 'gemini-2.5-flash-image'
    profile.image_model_options = ['gemini-2.5-flash-image']
    return profile


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text='', content=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {'data': [{'b64_json': _encoded_png()}]}
        self.text = text
        self.content = content if content is not None else _png_bytes()
        self.reason_phrase = 'OK' if status_code < 400 else 'Error'

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text or f'HTTP {self.status_code}')


class FakeHTTPClient:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.calls = []
        self.get_calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append({'url': url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response

    def get(self, url):
        self.get_calls.append(url)
        return self.response

    def close(self):
        self.closed = True


class FakeInpaint(LLMInpaint):
    def __init__(self, response=None, error=None):
        super().__init__()
        self.http_client = FakeHTTPClient(response=response, error=error)

    def _initialize_client(self, profile):
        self._api_key_for_profile(profile)
        self._image_base_url(profile)
        return self.http_client

    def _respect_delay(self):
        pass


class LLMImageThrottleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.throttle = _SharedLLMImageThrottle()
        self.throttle_patch = patch.object(
            llm_image, '_LLM_IMAGE_THROTTLE', self.throttle
        )
        self.throttle_patch.start()

    def tearDown(self) -> None:
        self.throttle_patch.stop()

    @staticmethod
    def _requester(delay: float, rpm: int = 0) -> LLMImageRequester:
        return LLMImageRequester(image_request_policy=LLMImageRequestPolicy(
            delay=delay,
            max_requests_per_minute=rpm,
        ))

    def test_delay_is_shared_across_fresh_requesters(self):
        first = self._requester(0.04)
        second = self._requester(0.04)
        started = time.monotonic()
        first._respect_delay()
        second._respect_delay()

        self.assertGreaterEqual(time.monotonic() - started, 0.025)

    def test_concurrent_fresh_requesters_reserve_distinct_slots(self):
        requesters = (self._requester(0.04), self._requester(0.04))
        barrier = threading.Barrier(3)
        completed = []
        errors = []

        def reserve(requester) -> None:
            try:
                barrier.wait()
                requester._respect_delay()
                completed.append(time.monotonic())
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=reserve, args=(requester,))
            for requester in requesters
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(1.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(completed), 2)
        self.assertGreaterEqual(abs(completed[1] - completed[0]), 0.025)

    def test_rpm_is_shared_and_throttle_wait_is_stoppable(self):
        first = self._requester(0.0, rpm=1)
        second = self._requester(0.0, rpm=1)
        first._respect_delay()
        with self.throttle._condition:
            almost_expired = time.monotonic() - 59.9
            self.throttle._request_times.clear()
            self.throttle._request_times.append(almost_expired)
            self.throttle._last_request_time = almost_expired
            self.throttle._next_allowed_time = 0.0

        started = time.monotonic()
        second._respect_delay()
        self.assertGreaterEqual(time.monotonic() - started, 0.12)

        with self.throttle._condition:
            self.throttle._request_times.clear()
            self.throttle._last_request_time = None
            self.throttle._next_allowed_time = 0.0
        blocker = self._requester(10.0)
        waiter = self._requester(10.0)
        blocker._respect_delay()
        stop_event = threading.Event()
        waiter.set_stop_event(stop_event)
        stopped = []

        def wait_for_slot() -> None:
            try:
                waiter._respect_delay()
            except LLMRequestStopped:
                stopped.append(True)

        thread = threading.Thread(target=wait_for_slot)
        thread.start()
        time.sleep(0.02)
        stop_event.set()
        thread.join(0.5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(stopped, [True])

    def test_user_action_required_bypasses_image_retries(self):
        requester = LLMImageRequester(
            image_request_policy=LLMImageRequestPolicy(
                retry_attempts=3,
                retry_timeout=0,
            )
        )
        error = LLMUserActionRequiredError('update the profile')

        with patch.object(
            requester,
            'request_image',
            side_effect=error,
        ) as request:
            with self.assertRaises(LLMUserActionRequiredError):
                requester.request_image_with_retries(
                    default_profile('OpenRouter'),
                    None,
                    'prompt',
                    'model',
                )

        request.assert_called_once()


class APIImageResponseDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = default_profile('Infistar')
        self.profile.api_key = 'private-api-key'
        self.profile.image_base_url = 'https://infistar.cc/v1?tenant=private-query'
        self.profile.image_model = 'gpt-image-2'
        self.profile.image_model_options = ['gpt-image-2']
        self.prompt = 'Private edit instructions.'
        self.requester = LLMImageRequester(image_request_policy=LLMImageRequestPolicy(
            delay=0, max_requests_per_minute=0, retry_attempts=2, retry_timeout=0,
        ))
        self.requests = []
        self.response = httpx.Response(200, json={'data': [{'b64_json': _encoded_png()}]})

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self.response

        def client(proxy: str) -> httpx.Client:
            result = httpx.Client(transport=httpx.MockTransport(respond))
            self.addCleanup(result.close)
            return result

        factory = patch.object(self.requester, '_http_client', side_effect=client)
        factory.start()
        self.addCleanup(factory.stop)
        self.addCleanup(self.requester.close)
        warning = patch.object(llm_image.LOGGER, 'warning')
        self.warning = warning.start()
        self.addCleanup(warning.stop)

    def request(self) -> np.ndarray:
        return self.requester.request_image_with_retries(
            self.profile, np.zeros((2, 2, 3), np.uint8), self.prompt, self.profile.image_model,
        )

    def diagnostics(self) -> list[str]:
        return [call.args[0] % call.args[1:] for call in self.warning.call_args_list
                if call.args[0].startswith('LLM image response failed:')]

    def test_redirect_html_reports_actual_endpoint_and_location_without_following(self) -> None:
        self.response = httpx.Response(301, headers={
            'Content-Type': 'text/html',
            'Location': 'https://user:password@infistar.cc/v1/?signature=private-signature',
        }, text='<html><title>301 Moved Permanently</title><hr>nginx</html>')
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'HTTP 301 redirect to https://infistar.cc/v1/') as caught:
            self.request()
        self.assertEqual(len(self.requests), 1)
        message = self.diagnostics()[0]
        self.assertIn("profile_id='infistar'", message)
        self.assertIn("endpoint='https://infistar.cc/v1'", message)
        self.assertIn("location='https://infistar.cc/v1/'", message)
        self.assertIn('nginx', message)
        self.assertIn("content_type='text/html", message)
        for secret in ('private-query', 'private-signature', 'user:password'):
            self.assertNotIn(secret, message + str(caught.exception))

    def test_non_json_and_empty_successes_are_actionable_across_direct_providers(self) -> None:
        for provider, body in (('Infistar', b'<html>nginx returned a web page</html>'),
                               ('OpenRouter', b''), ('Gemini', b'<html>wrong endpoint</html>')):
            with self.subTest(provider=provider):
                self.profile = default_profile(provider)
                self.profile.api_key = 'private-api-key'
                self.profile.image_model = 'image-model'
                self.profile.image_model_options = ['image-model']
                self.response = httpx.Response(200, headers={'Content-Type': 'text/html'}, content=body)
                self.requests.clear()
                self.warning.reset_mock()
                with self.assertRaisesRegex(LLMUserActionRequiredError, 'empty or non-JSON response'):
                    self.request()
                self.assertEqual(len(self.requests), 1)
                self.assertIn('status=200', self.diagnostics()[0])
                self.assertIn(body.decode() if body else '<empty>', self.diagnostics()[0])

    def test_json_image_failures_log_provider_detail_and_keep_retries(self) -> None:
        for status, payload, expected in (
            (200, {'data': []}, 'no image data'),
            (200, {'error': {'code': 'model_rejected', 'message': 'This model cannot edit images.'}}, 'cannot edit'),
            (400, {'error': {'message': 'This model needs a different image size.'}}, 'different image size'),
            (400, {'message': 'Quota needs replenishing.'}, 'Quota needs replenishing'),
            (200, {'data': [{'b64_json': base64.b64encode(b'broken image').decode()}]}, 'cannot identify'),
        ):
            with self.subTest(status=status, payload=payload):
                self.response = httpx.Response(status, json=payload)
                self.requests.clear()
                self.warning.reset_mock()
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.request()
                self.assertEqual(len(self.requests), 2)
                messages = self.diagnostics()
                self.assertEqual(len(messages), 2)
                if payload.get('error'):
                    self.assertIn(payload['error']['message'], messages[0])
                if payload.get('data'):
                    self.assertNotIn(payload['data'][0]['b64_json'], messages[0])

    def test_failure_preview_is_bounded_and_omits_echoed_secrets_and_image_data(self) -> None:
        self.response = httpx.Response(400, json={
            'error': {'message': 'Rejected private-api-key while fetching https://user:password@cdn.example/image.png?signature=private-signature'},
            'prompt': self.prompt,
            'data': [{'b64_json': 'A' * 100_000}],
            'debug': self.prompt + ' data:image/png;base64,' + 'B' * 100_000,
            'detail': 'Long diagnostic. ' * 10_000,
        })
        with self.assertRaises(RuntimeError) as caught:
            self.request()
        message = self.diagnostics()[0]
        self.assertLess(len(message), 2500)
        self.assertIn('Rejected <redacted>', message)
        self.assertIn('https://cdn.example/image.png', message)
        for secret in ('private-api-key', 'private-signature', self.prompt, 'user:password', 'A' * 80, 'B' * 80):
            self.assertNotIn(secret, message + str(caught.exception))

    def test_html_echo_is_bounded_and_redacted(self) -> None:
        self.response = httpx.Response(200, headers={'Content-Type': 'text/html'}, text=(
            '<html>private-api-key ' + self.prompt + ' https://cdn.example/a?signature=private-signature '
            + 'data:image/png;base64,' + 'A' * 100_000 + ' diagnostic' * 10_000 + '</html>'
        ))
        with self.assertRaises(LLMUserActionRequiredError):
            self.request()
        message = self.diagnostics()[0]
        self.assertLess(len(message), 2500)
        for secret in ('private-api-key', self.prompt, 'private-signature', 'A' * 80):
            self.assertNotIn(secret, message)

    def test_structured_validation_error_never_echoes_multiline_input_in_retries(self) -> None:
        self.prompt = 'Private first line\nPrivate second line'
        self.response = httpx.Response(422, json={
            'detail': [{'msg': 'Invalid field', 'input': self.prompt}],
        })
        with self.assertRaisesRegex(RuntimeError, 'HTTP 422') as caught:
            self.request()
        all_warnings = '\n'.join(call.args[0] % call.args[1:] for call in self.warning.call_args_list)
        self.assertIn('Invalid field', self.diagnostics()[0])
        self.assertEqual(len(self.requests), 2)
        for text in ('Private first line', 'Private second line'):
            self.assertNotIn(text, all_warnings + str(caught.exception))

    def test_valid_image_and_authentication_preserve_existing_behavior(self) -> None:
        result = self.request()
        np.testing.assert_array_equal(result, np.full((2, 2, 3), [255, 0, 0], np.uint8))
        self.assertEqual(self.diagnostics(), [])
        self.response = httpx.Response(401, json={'error': {'message': 'Invalid token private-api-key'}})
        self.requests.clear()
        with self.assertRaises(LLMApiKeyRequiredError):
            self.request()
        self.assertEqual(len(self.requests), 1)
        self.assertIn('Invalid token <redacted>', self.diagnostics()[0])

    def test_direct_403_distinguishes_model_access_from_explicit_invalid_key(self) -> None:
        for code, is_auth in (('model_not_found', False), ('permission_denied', False), ('invalid_api_key', True)):
            with self.subTest(code=code):
                self.response = httpx.Response(403, json={'error': {
                    'code': code, 'message': 'The requested model is unavailable for this account.',
                }})
                self.requests.clear()
                with self.assertRaises(LLMUserActionRequiredError) as caught:
                    self.request()
                self.assertEqual(isinstance(caught.exception, LLMApiKeyRequiredError), is_auth)
                if not is_auth:
                    self.assertIn('requested model is unavailable', str(caught.exception))
                self.assertEqual(len(self.requests), 1)

    def test_logger_without_bound_response_request_preserves_original_error(self) -> None:
        response = httpx.Response(301, text='nginx')
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'HTTP 301 redirect'):
            self.requester._decode_api_image_response(self.profile, response, self.profile.image_base_url, self.prompt)
        self.assertIn("endpoint='https://infistar.cc/v1'", self.diagnostics()[0])

    def test_malformed_optional_prompt_and_failed_logger_do_not_replace_errors(self) -> None:
        self.profile.image_prompt = 42
        self.response = httpx.Response(301, headers={'Location': '/v1/'}, text='nginx')
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'HTTP 301 redirect'):
            self.request()
        self.assertIn('nginx', self.diagnostics()[0])
        self.warning.side_effect = RuntimeError('logging unavailable')
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'HTTP 301 redirect'):
            self.request()
        self.response = httpx.Response(401, json={'error': {'message': 'Invalid token'}})
        with self.assertRaises(LLMApiKeyRequiredError):
            self.request()
        original = RuntimeError('original image decoding error')
        self.response = httpx.Response(200, json={'data': []})
        with patch.object(self.requester, '_decode_response_image', side_effect=original), \
                self.assertRaises(RuntimeError) as caught:
            self.requester.request_image(self.profile, None, prompt=self.prompt, model=self.profile.image_model)
        self.assertIs(caught.exception, original)

    def test_stop_during_response_handling_is_not_logged_or_retried(self) -> None:
        original = LLMRequestStopped()
        with patch.object(self.requester, '_decode_response_image', side_effect=original), \
                self.assertRaises(LLMRequestStopped) as caught:
            self.request()
        self.assertIs(caught.exception, original)
        self.assertEqual(len(self.requests), 1)
        self.warning.assert_not_called()

    def test_download_failure_retries_without_exposing_signed_url(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.host == 'cdn.example':
                return httpx.Response(503, text='temporarily unavailable')
            return httpx.Response(200, json={'data': [{'url': 'https://cdn.example/image.png?signature=private-signature'}]})

        def client(proxy: str) -> httpx.Client:
            result = httpx.Client(transport=httpx.MockTransport(respond))
            self.addCleanup(result.close)
            return result

        with patch.object(self.requester, '_http_client', side_effect=client), self.assertRaises(RuntimeError) as caught:
            self.request()
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(self.diagnostics()), 2)
        all_warnings = '\n'.join(call.args[0] % call.args[1:] for call in self.warning.call_args_list)
        self.assertNotIn('private-signature', all_warnings + str(caught.exception))
        self.assertIn('503', str(caught.exception))


class AssistedAPIImageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = default_profile('OpenAI')
        self.profile.api_key = 'image-key'
        self.profile.image_base_url = 'https://images.example/custom/v2/images/edits?tenant=example'
        self.profile.image_model = 'gpt-reasoning → gpt-image-custom'
        self.profile.image_model_options = ['gpt-image-custom']
        self.requester = LLMImageRequester(image_request_policy=LLMImageRequestPolicy(
            delay=0, max_requests_per_minute=0, retry_attempts=2, retry_timeout=0,
            request_timeout=23, max_resolution=4, proxy='http://proxy.example:8080',
        ))
        self.requests = []
        self.response = {'status': 'completed', 'output': [
            {'type': 'image_generation_call', 'status': 'completed', 'result': _encoded_png()},
        ]}
        self.status = 200

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(self.status, json=self.response)

        self.client = httpx.Client(transport=httpx.MockTransport(respond), timeout=23)
        self.addCleanup(self.client.close)
        self.addCleanup(self.requester.close)
        factory = patch.object(self.requester, '_http_client', return_value=self.client)
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        auth = patch.object(codex.account, 'require_sign_in', side_effect=AssertionError('Codex auth leaked'))
        auth.start()
        self.addCleanup(auth.stop)

    def test_assisted_edit_uses_configured_service_and_aligned_mask_references(self) -> None:
        image = np.full((8, 12, 4), 255, np.uint8)
        mask = np.zeros((8, 12), np.uint8)
        mask[1, 1] = 255
        result = self.requester.request_image(
            self.profile, image, prompt='Keep all artwork.\nRemove text.', mask=mask,
            resize_to_input=True,
        )
        self.assertEqual(result.shape, (8, 12, 3))
        request = self.requests[0]
        self.assertEqual(str(request.url), 'https://images.example/custom/v2/responses?tenant=example')
        self.assertEqual(request.headers['Authorization'], 'Bearer image-key')
        self.assertEqual(request.extensions['timeout']['read'], 23)
        self.factory.assert_called_once_with('http://proxy.example:8080')
        body = json.loads(request.content)
        self.assertEqual(body['model'], 'gpt-reasoning')
        self.assertEqual(body['tools'], [{'type': 'image_generation', 'model': 'gpt-image-custom',
                                         'action': 'edit', 'quality': 'auto', 'size': 'auto'}])
        self.assertEqual(body['tool_choice'], {'type': 'image_generation'})
        self.assertFalse(body['store'])
        self.assertFalse(body['stream'])
        self.assertNotIn('previous_response_id', body)
        self.assertNotIn('reasoning', body)
        content = body['input'][0]['content']
        self.assertTrue(content[0]['text'].startswith('Keep all artwork.\nRemove text.'))
        self.assertIn('white marks the editable region', content[0]['text'])
        refs = [np.array(Image.open(io.BytesIO(base64.b64decode(item['image_url'].split(',')[1]))))
                for item in content[1:]]
        self.assertEqual(refs[0].shape[:2], (3, 4))
        self.assertEqual(refs[1].shape[:2], (3, 4))
        self.assertGreater(np.count_nonzero(refs[1]), 0)
        self.assertEqual(self.profile.image_model, 'gpt-reasoning → gpt-image-custom')

    def test_infistar_pair_uses_responses_and_direct_edit_keeps_configured_url(self) -> None:
        profile = default_profile('Infistar')
        profile.api_key = 'image-key'
        profile.image_base_url = 'https://infistar.cc/v1'
        profile.image_model = 'gpt-5.6-luna → gpt-image-2'
        image = np.zeros((2, 2, 3), np.uint8)
        self.requester.request_image(profile, image, prompt='Remove text.')
        request = self.requests[-1]
        self.assertEqual(str(request.url), 'https://infistar.cc/v1/responses')
        body = json.loads(request.content)
        self.assertEqual(body['model'], 'gpt-5.6-luna')
        self.assertEqual(body['tools'][0]['model'], 'gpt-image-2')
        self.assertEqual(body['tools'][0]['action'], 'edit')

        self.response = {'data': [{'b64_json': _encoded_png()}]}
        self.requester.request_image(profile, image, prompt='Remove text.', model='gpt-image-2')
        self.assertEqual(str(self.requests[-1].url), 'https://infistar.cc/v1')
        self.assertEqual(profile.image_base_url, 'https://infistar.cc/v1')

    def test_infistar_direct_success_then_missing_reasoning_model_keeps_same_key(self) -> None:
        profile = default_profile('Infistar')
        profile.api_key = 'same-valid-image-key'
        profile.image_base_url = 'https://infistar.cc/v1/images/edits'
        image = np.zeros((2, 2, 3), np.uint8)
        self.response = {'data': [{'b64_json': _encoded_png()}]}
        provider_message = '模型 gpt-6-luna 不在当前供应目录中'
        with patch.object(llm_image.LOGGER, 'warning') as warning:
            self.requester.request_image(profile, image, prompt='Remove text.', model='gpt-image-2')
            warning.assert_not_called()
            self.status = 403
            self.response = {'error': {'code': 'model_not_found', 'message': provider_message, 'type': 'new_api_error'}}
            with self.assertRaises(LLMUserActionRequiredError) as caught:
                self.requester.request_image_with_retries(profile, image, 'Remove text.', 'gpt-6-luna → gpt-image-2')
        self.assertNotIsInstance(caught.exception, LLMApiKeyRequiredError)
        self.assertIn(provider_message, str(caught.exception))
        self.assertEqual(len(self.requests), 2)
        self.assertEqual([request.headers['Authorization'] for request in self.requests], ['Bearer same-valid-image-key'] * 2)
        self.assertEqual([request.url.path for request in self.requests], ['/v1/images/edits', '/v1/responses'])
        diagnostic = warning.call_args.args[0] % warning.call_args.args[1:]
        self.assertIn(provider_message, diagnostic)
        self.assertIn('model_not_found', diagnostic)
        self.assertIn('status=403', diagnostic)
        self.assertNotIn(profile.api_key, diagnostic)

    def test_assisted_explicit_authentication_codes_keep_key_dialog(self) -> None:
        for status, payload in (
            (403, {'error': {'code': 'invalid_api_key', 'message': 'Invalid image-key'}}),
            (400, {'error': {'type': 'authentication_error', 'message': 'Invalid image-key'}}),
            (200, {'type': 'response.failed', 'response': {'error': {'code': 'invalid_token', 'message': 'Invalid image-key'}}}),
        ):
            with self.subTest(status=status, payload=payload):
                self.status, self.response = status, payload
                self.requests.clear()
                with patch.object(llm_image.LOGGER, 'warning') as warning, self.assertRaises(LLMApiKeyRequiredError):
                    self.requester.request_image_with_retries(self.profile, None, 'Draw.', self.profile.image_model)
                self.assertEqual(len(self.requests), 1)
                diagnostic = warning.call_args.args[0] % warning.call_args.args[1:]
                self.assertIn('Invalid <redacted>', diagnostic)
                self.assertNotIn('image-key', diagnostic)

    def test_failed_response_event_preserves_model_error_detail_without_key_dialog(self) -> None:
        self.response = {'type': 'response.failed', 'response': {'error': {
            'code': 'model_not_found', 'message': 'The selected model is unavailable.',
        }}}
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'selected model is unavailable') as caught:
            self.requester.request_image_with_retries(self.profile, None, 'Draw.', self.profile.image_model)
        self.assertNotIsInstance(caught.exception, LLMApiKeyRequiredError)
        self.assertEqual(len(self.requests), 1)

    def test_assisted_generation_uses_explicit_pair_and_rotates_job_cache(self) -> None:
        stop = threading.Event()
        self.requester.set_stop_event(stop)
        for _ in range(2):
            self.requester.request_image(self.profile, None, prompt='Draw paper.',
                                         model='gpt-other -> gpt-image-other')
        bodies = [json.loads(request.content) for request in self.requests]
        self.assertEqual(bodies[0]['prompt_cache_key'], bodies[1]['prompt_cache_key'])
        self.assertTrue(bodies[0]['prompt_cache_key'])
        self.assertEqual(bodies[0]['model'], 'gpt-other')
        self.assertEqual(bodies[0]['tools'][0]['model'], 'gpt-image-other')
        self.assertEqual(bodies[0]['tools'][0]['action'], 'generate')
        self.assertEqual(bodies[0]['input'][0]['content'], [{'type': 'input_text', 'text': 'Draw paper.'}])
        self.requester.set_stop_event(threading.Event())
        self.requester.request_image(self.profile, None, prompt='Draw paper.',
                                     model='gpt-other → gpt-image-other')
        self.assertNotEqual(bodies[0]['prompt_cache_key'], json.loads(self.requests[-1].content)['prompt_cache_key'])

    def test_assisted_extreme_aspect_edit_pads_references_and_restores_pixels(self) -> None:
        self.requester._image_request_policy = LLMImageRequestPolicy(
            delay=0, max_requests_per_minute=0, max_resolution=0,
        )
        image = np.arange(3 * 21 * 3, dtype=np.uint8).reshape(3, 21, 3)
        mask = np.zeros((3, 21), np.uint8)
        mask[1, 4] = 255
        padded = np.pad(image, ((2, 2), (0, 0), (0, 0)), mode='edge')
        buffer = io.BytesIO()
        Image.fromarray(padded).save(buffer, format='PNG')
        self.response['output'][0]['result'] = base64.b64encode(buffer.getvalue()).decode('ascii')

        result = self.requester.request_image(self.profile, image, prompt='Keep artwork.', mask=mask)

        np.testing.assert_array_equal(result, image)
        content = json.loads(self.requests[0].content)['input'][0]['content']
        references = [np.array(Image.open(io.BytesIO(base64.b64decode(item['image_url'].split(',')[1]))))
                      for item in content[1:]]
        np.testing.assert_array_equal(references[0], padded)
        np.testing.assert_array_equal(references[1][:, :, 0], np.pad(mask, ((2, 2), (0, 0))))

    def test_invalid_pair_prompt_or_endpoint_fails_before_client_or_network(self) -> None:
        for model, prompt, endpoint in (
            ('non-gpt → gpt-image-2', 'Draw.', self.profile.image_base_url),
            ('gpt-reason → other-image', 'Draw.', self.profile.image_base_url),
            (self.profile.image_model, ' \t', self.profile.image_base_url),
            (self.profile.image_model, 'Draw.', 'https://images.example/custom'),
            (self.profile.image_model, 'Draw.', 'https://openrouter.ai/api/v1/images/edits'),
            (self.profile.image_model, 'Draw.', 'https://generativelanguage.googleapis.com/v1beta/responses'),
        ):
            with self.subTest(model=model, endpoint=endpoint):
                self.profile.image_base_url = endpoint
                with self.assertRaises(LLMUserActionRequiredError):
                    self.requester.request_image_with_retries(self.profile, None, prompt, model)
        self.factory.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_assisted_response_failures_are_actionable_and_never_retried(self) -> None:
        image = self.response['output'][0]
        for payload in (
            {'status': 'failed', 'error': {'message': 'Model access unavailable.'}},
            {'status': 'completed', 'output': []},
            {'status': 'completed', 'output': [image, image]},
            {'status': 'completed', 'output': [image, {'type': 'message', 'content': [
                {'type': 'refusal', 'refusal': 'private details'},
            ]}]},
            {'status': 'completed', 'output': [{**image, 'result': '%%%'}]},
            {'status': 'completed', 'output': [{**image, 'result': 'https://untrusted.example/image.png'}]},
            {'status': 'completed', 'output': [{**image, 'result': 'YQ=='}]},
        ):
            with self.subTest(payload=payload):
                self.response = payload
                self.requests.clear()
                with self.assertRaises(LLMUserActionRequiredError) as caught:
                    self.requester.request_image_with_retries(self.profile, None, 'Draw.', self.profile.image_model)
                self.assertNotIn('private', str(caught.exception))
                if payload.get('error'):
                    self.assertIn('Model access unavailable.', str(caught.exception))
                self.assertEqual(len(self.requests), 1)

    def test_api_auth_and_unsupported_responses_errors_are_not_retried(self) -> None:
        for status, expected in ((401, LLMApiKeyRequiredError), (403, LLMUserActionRequiredError),
                                  (400, LLMUserActionRequiredError), (404, LLMUserActionRequiredError),
                                  (422, LLMUserActionRequiredError)):
            with self.subTest(status=status):
                self.status = status
                self.requests.clear()
                self.response = {'error': {'message': 'Model access unavailable.'}}
                with self.assertRaises(expected) as caught:
                    self.requester.request_image_with_retries(self.profile, None, 'Draw.', self.profile.image_model)
                if status != 401:
                    self.assertNotIsInstance(caught.exception, LLMApiKeyRequiredError)
                    self.assertIn('Model access unavailable.', str(caught.exception))
                self.assertEqual(len(self.requests), 1)

    def test_transient_errors_keep_existing_retry_policy_and_cache_key(self) -> None:
        for status in (408, 429, 503):
            with self.subTest(status=status):
                self.status = status
                self.requests.clear()
                with self.assertRaisesRegex(RuntimeError, f'HTTP {status}'):
                    self.requester.request_image_with_retries(self.profile, None, 'Draw.', self.profile.image_model)
                self.assertEqual(len(self.requests), 2)
                self.assertEqual(self.requests[0].content, self.requests[1].content)

    def test_error_body_cap_closes_response_and_omits_partial_multiline_input(self) -> None:
        prompt = '私密第一行\nPrivate second line'
        for content_type in ('application/json', 'text/html'):
            closed = threading.Event()
            prefix = b'{"padding":"' + b'x' * (16 * 1024 - 30) + b'","input":'
            encoded = json.dumps(prompt, ensure_ascii=False).encode('utf-8')
            body = prefix + encoded + b',"padding2":"' + b'x' * 20_000 + b'"}'
            requests = []

            class Chunks(httpx.SyncByteStream):
                def __iter__(self):
                    yield body
                    raise AssertionError('The error-body cap must stop reading here.')

                def close(self) -> None:
                    closed.set()

            def respond(request: httpx.Request) -> httpx.Response:
                requests.append(request)
                return httpx.Response(403, headers={'Content-Type': content_type}, stream=Chunks())

            client = httpx.Client(transport=httpx.MockTransport(respond))
            with self.subTest(content_type=content_type), client, \
                    patch.object(self.requester, '_initialize_client', return_value=client), \
                    patch.object(llm_image.LOGGER, 'warning') as warning, \
                    self.assertRaises(LLMUserActionRequiredError) as caught:
                self.requester.request_image_with_retries(self.profile, None, prompt, self.profile.image_model)
            self.assertNotIsInstance(caught.exception, LLMApiKeyRequiredError)
            self.assertEqual(len(requests), 1)
            self.assertTrue(closed.is_set())
            diagnostic = warning.call_args.args[0] % warning.call_args.args[1:]
            self.assertIn('incomplete response body omitted', diagnostic)
            self.assertLess(len(diagnostic), 2500)
            for private in ('私密', 'Private second line', '\\u79c1'):
                self.assertNotIn(private, diagnostic + str(caught.exception))

    def test_complete_html_forbidden_is_logged_without_key_dialog(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            403, headers={'Content-Type': 'text/html'},
            text='<html>Service access denied for image-key.</html>',
        )))
        with client, patch.object(self.requester, '_initialize_client', return_value=client), \
                patch.object(llm_image.LOGGER, 'warning') as warning, \
                self.assertRaises(LLMUserActionRequiredError) as caught:
            self.requester.request_image_with_retries(self.profile, None, 'Draw.', self.profile.image_model)
        self.assertNotIsInstance(caught.exception, LLMApiKeyRequiredError)
        self.assertIn('Service access denied', str(caught.exception))
        diagnostic = warning.call_args.args[0] % warning.call_args.args[1:]
        self.assertIn('Service access denied', diagnostic)
        self.assertNotIn('image-key', diagnostic + str(caught.exception))

    def test_error_body_stops_at_exact_cap_without_requesting_another_chunk(self) -> None:
        read_past_cap, closed = threading.Event(), threading.Event()

        class ExactCap(httpx.SyncByteStream):
            def __iter__(self):
                yield b'x' * (16 * 1024)
                read_past_cap.set()
                raise AssertionError('No more error-body bytes can be retained.')

            def close(self) -> None:
                closed.set()

        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403, stream=ExactCap())))
        with client, patch.object(self.requester, '_initialize_client', return_value=client), \
                self.assertRaises(LLMUserActionRequiredError) as caught:
            self.requester.request_image_with_retries(self.profile, None, 'Draw.', self.profile.image_model)
        self.assertNotIsInstance(caught.exception, LLMApiKeyRequiredError)
        self.assertFalse(read_past_cap.is_set())
        self.assertTrue(closed.is_set())

    def test_interrupted_error_body_preserves_status_and_stop_wins(self) -> None:
        for status, cancel, expected_attempts in ((401, False, 1), (403, False, 1), (503, False, 2), (403, True, 1)):
            stop = threading.Event()
            self.requester.set_stop_event(stop)
            closed, requests = [], []

            class Broken(httpx.SyncByteStream):
                def __iter__(self):
                    yield b'{"input":"partial private input'
                    if cancel:
                        stop.set()
                    raise httpx.ReadError('private transport details')

                def close(self) -> None:
                    closed.append(True)

            def respond(request: httpx.Request) -> httpx.Response:
                requests.append(request)
                return httpx.Response(status, headers={'Content-Type': 'application/json'}, stream=Broken())

            client = httpx.Client(transport=httpx.MockTransport(respond))
            expected = (LLMRequestStopped if cancel else LLMApiKeyRequiredError if status == 401
                        else RuntimeError if status == 503 else LLMUserActionRequiredError)
            with self.subTest(status=status, cancel=cancel), client, \
                    patch.object(self.requester, '_initialize_client', return_value=client), \
                    patch.object(llm_image.LOGGER, 'warning') as warning, self.assertRaises(expected) as caught:
                self.requester.request_image_with_retries(self.profile, None, 'Draw.', self.profile.image_model)
            self.assertEqual(len(requests), expected_attempts)
            self.assertEqual(len(closed), expected_attempts)
            if cancel:
                warning.assert_not_called()
            else:
                if status == 403:
                    self.assertNotIsInstance(caught.exception, LLMApiKeyRequiredError)
                all_warnings = '\n'.join(call.args[0] % call.args[1:] for call in warning.call_args_list)
                self.assertIn(f'status={status}', all_warnings)
                self.assertIn('incomplete response body omitted', all_warnings)
                self.assertNotIn('private', all_warnings + str(caught.exception))

    def test_stream_bound_and_cancellation_close_response(self) -> None:
        stop = threading.Event()
        self.requester.set_stop_event(stop)
        for cancel in (False, True):
            closed = threading.Event()

            class Chunks(httpx.SyncByteStream):
                def __iter__(self):
                    yield b'{'
                    if cancel:
                        stop.set()
                    yield b' ' * 64
                    raise AssertionError('The response should have stopped before this chunk.')

                def close(self) -> None:
                    closed.set()

            client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Chunks())))
            with self.subTest(cancel=cancel), client, patch.object(self.requester, '_initialize_client', return_value=client), \
                    patch.object(image_generation, 'MAX_IMAGE_RESPONSE_BYTES', 64), \
                    self.assertRaises(LLMRequestStopped if cancel else LLMUserActionRequiredError):
                self.requester.request_image(self.profile, None, prompt='Draw.')
            self.assertTrue(closed.is_set())

    def test_stop_during_image_decode_discards_result(self) -> None:
        stop = threading.Event()
        self.requester.set_stop_event(stop)

        def decode(raw: bytes) -> np.ndarray:
            stop.set()
            return np.zeros((2, 2, 3), np.uint8)

        with patch.object(self.requester, '_decode_image_bytes', side_effect=decode), self.assertRaises(LLMRequestStopped):
            self.requester.request_image(self.profile, None, prompt='Draw.')

    def test_malformed_json_is_actionable_and_response_is_closed(self) -> None:
        closed = threading.Event()

        class Malformed(httpx.SyncByteStream):
            def __iter__(self):
                yield b'not JSON'

            def close(self) -> None:
                closed.set()

        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Malformed())))
        with client, patch.object(self.requester, '_initialize_client', return_value=client), \
                self.assertRaisesRegex(LLMUserActionRequiredError, 'invalid Responses result'):
            self.requester.request_image(self.profile, None, prompt='Draw.')
        self.assertTrue(closed.is_set())

    def test_api_assisted_inpaint_preserves_unmasked_pixels_and_honors_mask_toggle(self) -> None:
        inpainter = LLMInpaint()
        inpainter.params = copy.deepcopy(inpainter.params)
        inpainter.set_param_value('delay', 0)
        inpainter.set_param_value('max requests per minute', 0)
        self.addCleanup(inpainter.close)
        image = np.full((3, 5, 4), 37, np.uint8)
        image[:, :, 3] = 123
        mask = np.zeros((3, 5), np.uint8)
        mask[1, 2] = 255
        for use_mask in (True, False):
            with self.subTest(use_mask=use_mask), patch.object(inpainter, '_http_client', return_value=self.client):
                result = inpainter.inpaint(image, mask.copy(), profile=self.profile, use_mask=use_mask)
            content = json.loads(self.requests[-1].content)['input'][0]['content']
            self.assertEqual(len(content), 3 if use_mask else 2)
            if use_mask:
                np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
                np.testing.assert_array_equal(result[mask > 0, :3], [[255, 0, 0]])
            else:
                np.testing.assert_array_equal(result[:, :, :3], np.full((3, 5, 3), [255, 0, 0], np.uint8))
            np.testing.assert_array_equal(result[:, :, 3], image[:, :, 3])


class LLMInpaintTest(unittest.TestCase):
    def setUp(self) -> None:
        account = codex.CodexAccount()
        account._loaded = True
        account._credentials = {
            'access_token': 'test-access', 'refresh_token': 'test-refresh',
            'account_id': 'test-account', 'expires_at': time.time() + 3600,
        }
        account_patch = patch.object(codex, 'account', account)
        account_patch.start()
        self.addCleanup(account_patch.stop)
        self._old_profiles = copy.deepcopy(pcfg.module.llm_profiles)
        self._old_inpaint_llm_id = pcfg.module.inpaint_llm_id
        profile = default_profile('OpenRouter')
        profile.api_key = 'sk-demo'
        profile.image_model = 'black-forest-labs/flux.2-klein-4b'
        profile.image_model_options = [profile.image_model]
        pcfg.module.llm_profiles = [profile]
        pcfg.module.inpaint_llm_id = 'openrouter'
        self.inpainter = FakeInpaint()

    def tearDown(self):
        pcfg.module.llm_profiles = self._old_profiles
        pcfg.module.inpaint_llm_id = self._old_inpaint_llm_id

    def test_missing_required_api_key_raises_profile_error(self):
        profile = default_profile('OpenRouter')
        profile.api_key = ''

        with self.assertRaises(LLMApiKeyRequiredError):
            self.inpainter._api_key_for_profile(profile)

    def test_blank_image_model_requires_model(self):
        profile = self.inpainter.profile
        profile.image_model = ''

        with self.assertRaises(LLMModelRequiredError) as caught:
            self.inpainter._api_args(profile, io.BytesIO(), prompt='x')

        self.assertEqual(caught.exception.target, 'image_model')

    def test_image_enabled_profile_requires_model_options(self):
        profile = default_profile('OpenRouter')
        profile.api_key = 'sk-demo'
        profile.image_model = 'stale-image-model'
        profile.image_model_options = []
        pcfg.module.llm_profiles = [profile]
        pcfg.module.inpaint_llm_id = profile.id

        with self.assertRaises(LLMModelRequiredError) as caught:
            _ = self.inpainter.profile

        self.assertEqual(caught.exception.target, 'image_model')

    def test_profile_rejects_a_non_image_capability(self):
        profile = default_profile('DeepSeek')
        pcfg.module.llm_profiles = [profile]
        pcfg.module.inpaint_llm_id = profile.id

        with self.assertRaisesRegex(
            RuntimeError,
            'does not have image cleanup enabled',
        ):
            _ = self.inpainter.profile

    def test_blank_image_base_url_requires_url(self):
        profile = self.inpainter.profile
        profile.image_base_url = ''

        class URLInpaint(FakeInpaint):
            def _initialize_client(self_inner, selected_profile):
                return LLMInpaint._initialize_client(self_inner, selected_profile)

            def _http_client(self_inner, proxy):
                return FakeHTTPClient()

        with self.assertRaises(LLMBaseURLRequiredError) as caught:
            URLInpaint()._initialize_client(profile)

        self.assertEqual(caught.exception.target, 'image_base_url')

    def test_request_timeout_defaults_high_and_can_be_disabled(self):
        self.assertEqual(self.inpainter._request_timeout(), 180.0)

        self.inpainter.set_param_value('request timeout', 0)

        self.assertIsNone(self.inpainter._request_timeout())

    def test_inpaint_by_block_defaults_true_and_updates_from_param(self):
        self.assertTrue(self.inpainter.inpaint_by_block)

        self.inpainter.updateParam('inpaint by block', False)

        self.assertFalse(self.inpainter.inpaint_by_block)

    def test_max_resolution_defaults_and_scales_long_side(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        scaled = self.inpainter._scale_image_for_request(img)

        self.assertEqual(scaled.shape[:2], (640, 1280))

    def test_zero_max_resolution_keeps_original_size(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        self.inpainter.set_param_value('max resolution', 0)

        scaled = self.inpainter._scale_image_for_request(img)

        self.assertIs(scaled, img)

    def test_request_sends_scaled_image_and_returns_original_size(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        result = self.inpainter._request_inpaint(self.inpainter.profile, img)

        self.assertEqual(result.shape, img.shape)
        call = self.inpainter.http_client.calls[0]
        image_url = call['json']['input_references'][0]['image_url']['url']
        encoded = image_url.split(',', 1)[1]
        sent = Image.open(io.BytesIO(base64.b64decode(encoded)))
        self.assertEqual(sent.size, (1280, 640))

    def test_openai_compatible_request_uses_image_base_url_not_text_base_url(self):
        profile = self.inpainter.profile
        profile.base_url = 'https://text.example/v1'
        profile.image_base_url = 'https://image.example/v1'

        result = self.inpainter._request_inpaint(profile, np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertEqual(result.shape, (2, 2, 3))
        call = self.inpainter.http_client.calls[0]
        self.assertEqual(call['url'], 'https://image.example/v1')
        self.assertEqual(call['data']['model'], 'black-forest-labs/flux.2-klein-4b')
        self.assertIn('image', call['files'])

    def test_openai_compatible_request_accepts_final_edit_endpoint(self):
        profile = self.inpainter.profile
        profile.image_base_url = 'https://image.example/v1/images/edits'

        self.inpainter._request_inpaint(profile, np.zeros((2, 2, 3), dtype=np.uint8))

        call = self.inpainter.http_client.calls[0]
        self.assertEqual(call['url'], 'https://image.example/v1/images/edits')

    def test_prompt_only_openai_request_uses_generation_endpoint_and_json(self):
        profile = self.inpainter.profile
        profile.image_base_url = 'https://image.example/v1/images/edits'

        result = self.inpainter.request_image(
            profile, None, prompt='Draw paper', model='image-v2'
        )

        self.assertEqual(result.shape, (2, 2, 3))
        call = self.inpainter.http_client.calls[0]
        self.assertEqual(
            call['url'], 'https://image.example/v1/images/generations'
        )
        self.assertEqual(
            call['json'], {'model': 'image-v2', 'prompt': 'Draw paper'}
        )
        self.assertNotIn('files', call)
        self.assertNotIn('data', call)

    def test_stop_after_throttle_reservation_skips_provider_dispatch(self):
        profile = self.inpainter.profile
        stop_event = threading.Event()
        self.inpainter.set_stop_event(stop_event)

        with patch.object(
            self.inpainter, '_respect_delay', side_effect=stop_event.set
        ), self.assertRaises(LLMRequestStopped):
            self.inpainter.request_image(
                profile,
                np.zeros((2, 2, 3), dtype=np.uint8),
                prompt='Do not dispatch',
            )

        self.assertEqual(self.inpainter.http_client.calls, [])

    def test_openrouter_request_accepts_final_images_endpoint(self):
        profile = self.inpainter.profile
        profile.image_base_url = 'https://openrouter.ai/api/v1/images'

        self.inpainter._request_inpaint(profile, np.zeros((2, 2, 3), dtype=np.uint8))

        call = self.inpainter.http_client.calls[0]
        self.assertEqual(call['url'], 'https://openrouter.ai/api/v1/images')

    def test_prompt_only_openrouter_request_omits_image_reference(self):
        profile = self.inpainter.profile

        self.inpainter.request_image(
            profile, None, prompt='Draw paper', model='image-v2'
        )

        payload = self.inpainter.http_client.calls[0]['json']
        self.assertEqual(payload['prompt'], 'Draw paper')
        self.assertNotIn('input_references', payload)

    def test_gemini_request_uses_generate_content_endpoint_and_x_goog_key(self):
        profile = _gemini_image_profile()
        inpainter = FakeInpaint(FakeResponse(json_data={
            'candidates': [
                {'content': {'parts': [{'inlineData': {'data': _encoded_png()}}]}},
            ],
        }))

        result = inpainter._request_inpaint(profile, np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertEqual(result.shape, (2, 2, 3))
        call = inpainter.http_client.calls[0]
        self.assertEqual(
            call['url'],
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent',
        )
        self.assertEqual(call['headers']['x-goog-api-key'], 'gemini-key')
        self.assertNotIn('Authorization', call['headers'])
        parts = call['json']['contents'][0]['parts']
        self.assertIn('text', parts[0])
        self.assertEqual(parts[1]['inline_data']['mime_type'], 'image/png')
        self.assertIn('data', parts[1]['inline_data'])
        self.assertEqual(call['json']['generationConfig']['responseModalities'], ['IMAGE'])

    def test_gemini_request_strips_openai_compat_suffix_from_base_url(self):
        profile = _gemini_image_profile()
        profile.image_base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
        inpainter = FakeInpaint(FakeResponse(json_data={'output_image': {'data': _encoded_png()}}))

        inpainter._request_inpaint(profile, np.zeros((2, 2, 3), dtype=np.uint8))

        call = inpainter.http_client.calls[0]
        self.assertEqual(
            call['url'],
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent',
        )

    def test_prompt_only_gemini_request_omits_inline_image(self):
        profile = _gemini_image_profile()
        inpainter = FakeInpaint(FakeResponse(json_data={
            'output_image': {'data': _encoded_png()},
        }))

        inpainter.request_image(
            profile, None, prompt='Draw paper', model=profile.image_model
        )

        parts = inpainter.http_client.calls[0]['json']['contents'][0]['parts']
        self.assertEqual(parts, [{'text': 'Draw paper'}])

    def test_gemini_response_decodes_inline_data_image(self):
        response = {
            'candidates': [
                {'content': {'parts': [{'inline_data': {'data': _encoded_png()}}]}},
            ],
        }

        result = self.inpainter._decode_gemini_response_image(response)

        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(result[0, 0].tolist(), [255, 0, 0])

    def test_gemini_response_decodes_steps_model_output_image(self):
        response = {
            'steps': [
                {'type': 'model_output', 'content': [{'type': 'image', 'data': _encoded_png()}]},
            ],
        }

        result = self.inpainter._decode_gemini_response_image(response)

        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(result[0, 0].tolist(), [255, 0, 0])

    def test_request_args_use_image_model_and_no_mask(self):
        profile = self.inpainter.profile
        image_file = self.inpainter._png_image_file(np.zeros((2, 2, 3), dtype=np.uint8))

        args = self.inpainter._api_args(profile, image_file)

        self.assertEqual(args['model'], 'black-forest-labs/flux.2-klein-4b')
        self.assertIn('image', args)
        self.assertNotIn('mask', args)
        self.assertIn('Remove all visible text elements', args['prompt'])
        image_file.close()

    def test_inpaint_decodes_response_image(self):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=np.uint8) * 255

        result = self.inpainter._inpaint(img, mask)

        self.assertEqual(result.shape, img.shape)
        self.assertEqual(result[0, 0].tolist(), [255, 0, 0])
        call = self.inpainter.http_client.calls[0]
        self.assertEqual(call['url'], 'https://openrouter.ai/api/v1/images')
        self.assertEqual(call['json']['model'], 'black-forest-labs/flux.2-klein-4b')
        self.assertEqual(call['json']['output_format'], 'png')
        self.assertEqual(call['json']['n'], 1)
        self.assertIn('input_references', call['json'])
        self.assertTrue(call['json']['input_references'][0]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertNotIn('mask', call['json'])

    def test_explicit_profile_applies_to_each_crop_without_changing_pipeline_profile(self) -> None:
        run_profile = self.inpainter.profile
        draw_profile = copy.deepcopy(run_profile)
        draw_profile.image_model = 'drawing-model'
        draw_profile.image_prompt = 'Drawing prompt.'
        image = np.zeros((12, 24, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[3:6, 3:6] = mask[3:6, 18:21] = 255
        blocks = [TextBlock(xyxy=[3, 3, 6, 6]), TextBlock(xyxy=[18, 3, 21, 6])]
        with (
            patch.object(pcfg.module, 'check_need_inpaint', False),
            patch.object(self.inpainter, '_request_inpaint',
                         side_effect=lambda profile, crop, **kwargs: np.full_like(crop, 99)) as request,
        ):
            result = self.inpainter.inpaint(image, mask.copy(), blocks, profile=draw_profile)
            self.assertEqual(request.call_count, 2)
            self.assertTrue(all(call.args[0] is draw_profile for call in request.call_args_list))
            np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
            np.testing.assert_array_equal(result[mask > 0, :3], 99)
            np.testing.assert_array_equal(result[:, :, 3], image[:, :, 3])
            self.inpainter.inpaint(image, mask.copy())
            pipeline_profile = request.call_args.args[0]
        self.assertEqual(pipeline_profile.image_model, run_profile.image_model)
        self.assertEqual(pipeline_profile.image_prompt, run_profile.image_prompt)
        self.assertEqual(self.inpainter.profile, run_profile)

    def test_explicit_profile_auth_uses_its_backend_instead_of_run_selection(self) -> None:
        profile = self.inpainter.profile
        pcfg.module.inpaint_llm_id = 'codex'
        with patch.object(codex.account, 'require_sign_in', side_effect=AssertionError('Run auth leaked')), \
                patch.object(self.inpainter, '_request_inpaint', side_effect=lambda profile, img, **kwargs: img):
            self.inpainter.inpaint(np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 255, np.uint8), profile=profile)

        pcfg.module.inpaint_llm_id = profile.id
        codex_profile = default_codex_profile()
        with patch.object(codex.account, 'require_sign_in', side_effect=LLMUserActionRequiredError('Sign in with ChatGPT')):
            with self.assertRaisesRegex(LLMUserActionRequiredError, 'Sign in with ChatGPT'):
                self.inpainter.inpaint(np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 255, np.uint8), profile=codex_profile)

    def test_codex_inpaint_sends_aligned_mask_and_preserves_unmasked_rgba_pixels(self) -> None:
        profile = default_codex_profile()
        profile.image_model = 'gpt-reasoning-model → gpt-image-2'
        sync_codex_profile(profile, {'text-model': {'modalities': ['text'], 'efforts': []}})
        pcfg.module.llm_profiles = [profile]
        pcfg.module.inpaint_llm_id = profile.id
        inpainter = LLMInpaint()
        inpainter.params = copy.deepcopy(inpainter.params)
        inpainter.set_param_value('delay', 0)
        inpainter.set_param_value('max requests per minute', 0)
        inpainter.set_param_value('max resolution', 4)
        image = np.arange(8 * 12 * 4, dtype=np.uint8).reshape(8, 12, 4)
        image[:, :, 3] = 255
        image[0, :, 3] = 100
        mask = np.zeros((8, 12), dtype=np.uint8)
        mask[2:6, 3:9] = 255
        original_mask = mask.copy()
        stop = threading.Event()
        inpainter.set_stop_event(stop)

        with patch('ballontranslator.modules.codex.request_image', return_value=_png_bytes()) as request, \
                patch.object(inpainter, '_http_client', side_effect=AssertionError('API fallback')):
            result = inpainter.inpaint(image, mask)

        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
        np.testing.assert_array_equal(result[mask > 0, :3], np.tile([255, 0, 0], (24, 1)))
        np.testing.assert_array_equal(result[:, :, 3], image[:, :, 3])
        np.testing.assert_array_equal(mask, original_mask)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(request.call_args.args[0], 'gpt-image-2')
        self.assertIn('white marks the editable region', request.call_args.args[1])
        sent_image = Image.open(io.BytesIO(request.call_args.args[2]))
        sent_mask = Image.open(io.BytesIO(request.call_args.args[3]))
        self.assertEqual(sent_image.size, (4, 3))
        self.assertEqual(sent_mask.size, sent_image.size)
        np.testing.assert_array_equal(np.array(sent_mask)[:, :, 0], [[0, 255, 255, 0], [0, 255, 255, 0], [0, 255, 255, 0]])
        self.assertIs(request.call_args.args[4], stop)
        self.assertEqual(request.call_args.kwargs['timeout'], 180.0)
        self.assertEqual(request.call_args.kwargs['reasoning_model'], 'gpt-reasoning-model')

    def test_codex_empty_masks_skip_requests_and_stop_wins(self) -> None:
        inpainter = LLMInpaint()
        image = np.zeros((3, 5, 3), np.uint8)
        with patch.object(inpainter, '_request_inpaint', side_effect=AssertionError('empty request')):
            np.testing.assert_array_equal(inpainter.inpaint(image, np.zeros((3, 5), np.uint8)), image)
            stop = threading.Event()
            stop.set()
            inpainter.set_stop_event(stop)
            with self.assertRaises(LLMRequestStopped):
                inpainter.inpaint(image, np.zeros((3, 5), np.uint8))

    def test_codex_unmasked_edit_omits_mask_and_keeps_the_entire_result(self) -> None:
        profile = default_codex_profile()
        profile.image_model = 'gpt-reasoning-model → gpt-image-2'
        inpainter = LLMInpaint()
        inpainter.params = copy.deepcopy(inpainter.params)
        inpainter.set_param_value('delay', 0)
        inpainter.set_param_value('max requests per minute', 0)
        image = np.full((3, 5, 3), 37, np.uint8)
        for marked in (False, True):
            mask = np.zeros((3, 5), np.uint8)
            if marked:
                mask[1, 2] = 255
            with self.subTest(marked=marked), patch.object(codex, 'request_image', return_value=_png_bytes()) as request:
                result = inpainter.inpaint(image, mask, profile=profile, use_mask=False)
            np.testing.assert_array_equal(result, np.full_like(image, [255, 0, 0]))
            self.assertIsNone(request.call_args.args[3])
            self.assertEqual(request.call_args.kwargs['reasoning_model'], 'gpt-reasoning-model')
            self.assertNotIn('white marks the editable region', request.call_args.args[1])
            np.testing.assert_array_equal(image, np.full_like(image, 37))

    def test_codex_downscaling_retains_single_pixel_mask(self) -> None:
        profile = default_codex_profile()
        sync_codex_profile(profile, {'text-model': {'modalities': ['text'], 'efforts': []}})
        requester = LLMImageRequester(image_request_policy=LLMImageRequestPolicy(
            delay=0, max_requests_per_minute=0, max_resolution=4,
        ))
        mask = np.zeros((8, 8), np.uint8)
        mask[1, 1] = 255
        with patch('ballontranslator.modules.codex.request_image', return_value=_png_bytes()) as request:
            requester.request_image(profile, np.zeros((8, 8, 3), np.uint8), mask=mask)
        with Image.open(io.BytesIO(request.call_args.args[3])) as sent:
            self.assertEqual(sent.size, (4, 4))
            self.assertEqual(sent.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(int(np.array(sent)[:, :, 0].sum()), 255)

    def test_codex_extreme_aspect_edits_unpad_without_moving_pixels(self) -> None:
        profile = default_codex_profile()
        sync_codex_profile(profile, {'text-model': {'modalities': ['text'], 'efforts': []}})
        requester = LLMImageRequester(image_request_policy=LLMImageRequestPolicy(
            delay=0, max_requests_per_minute=0, max_resolution=0,
        ))
        for height, width in ((6, 25), (25, 6)):
            with self.subTest(shape=(height, width)):
                source = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
                mask = np.zeros((height, width), np.uint8)
                mask[1, 1] = mask[-2, -2] = 255

                def edit(model, prompt, image_bytes, mask_bytes, stop, **kwargs) -> bytes:
                    with Image.open(io.BytesIO(image_bytes)) as image, Image.open(io.BytesIO(mask_bytes)) as sent_mask:
                        self.assertLessEqual(max(image.size) / min(image.size), 3)
                        pixels = np.array(image)
                        marked = np.array(sent_mask)[:, :, 0] > 127
                    self.assertEqual(int(marked.sum()), 2)
                    pixels[marked] = [17, 29, 43]
                    # Provider returns a larger canvas. Unpadding must happen
                    # in that canvas's coordinates, after restoring its size.
                    doubled = np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1)
                    with requester._png_image_file(doubled) as generated:
                        return generated.getvalue()

                with patch('ballontranslator.modules.codex.request_image', side_effect=edit):
                    result = requester._request_inpaint(profile, source, mask=mask)
                expected = source.copy()
                expected[mask > 0] = [17, 29, 43]
                np.testing.assert_array_equal(result, expected)

    def test_codex_logged_out_requires_sign_in_before_inpainting(self) -> None:
        pcfg.module.llm_profiles = [default_codex_profile()]
        pcfg.module.inpaint_llm_id = 'codex'
        account = codex.CodexAccount()
        account._loaded = True
        with patch.object(codex, 'account', account), \
                patch.object(llm_image, '_LLM_IMAGE_THROTTLE', _SharedLLMImageThrottle()), \
                patch.object(httpx.AsyncClient, 'send', side_effect=AssertionError('Unexpected HTTP')) as send:
            with self.assertRaisesRegex(LLMUserActionRequiredError, 'Sign in with ChatGPT'):
                LLMInpaint().inpaint(np.zeros((3, 5, 3), np.uint8), np.full((3, 5), 255, np.uint8))
            send.assert_not_called()

    def test_codex_image_card_requests_and_failures_use_existing_policy(self) -> None:
        profile = default_codex_profile()
        profile.image_model = 'gpt-reasoning-model → gpt-image-2'
        sync_codex_profile(profile, {'text-model': {'modalities': ['text'], 'efforts': []}})
        requester = LLMImageRequester(image_request_policy=LLMImageRequestPolicy(
            delay=0, max_requests_per_minute=0, retry_attempts=2, retry_timeout=0,
        ))
        with patch('ballontranslator.modules.codex.request_image', side_effect=[b'broken image', _png_bytes()]) as request:
            result = requester.request_image_with_retries(profile, None, 'Draw paper.', profile.image_model)
        self.assertEqual(result[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(request.call_args.args[1:4], ('Draw paper.', None, None))
        self.assertEqual(request.call_args.kwargs['reasoning_model'], 'gpt-reasoning-model')
        for error in (LLMUserActionRequiredError('Sign in.'), LLMRequestStopped()):
            with self.subTest(error=error), patch('ballontranslator.modules.codex.request_image', side_effect=error) as request:
                with self.assertRaises(type(error)):
                    requester.request_image_with_retries(profile, None, 'Draw paper.', profile.image_model)
                request.assert_called_once()

    def test_assisted_image_cache_key_survives_crops_and_rotates_with_job_model_and_account(self) -> None:
        profile = default_codex_profile()
        profile.image_model = 'gpt-reasoning-model → gpt-image-2'
        requester = LLMImageRequester(image_request_policy=LLMImageRequestPolicy(
            delay=0, max_requests_per_minute=0,
        ))
        stop = threading.Event()
        requester.set_stop_event(stop)
        with patch.object(codex, 'request_image', return_value=_png_bytes()) as request:
            for _ in range(2):
                requester.request_image(profile, None, prompt='Draw paper.')
                requester.set_stop_event(stop)
            first, second = [call.kwargs['cache_key'] for call in request.call_args_list]
            self.assertTrue(first)
            self.assertEqual(first, second)
            requester.set_stop_event(threading.Event())
            requester.request_image(profile, None, prompt='Draw paper.')
            new_job = request.call_args.kwargs['cache_key']
            profile.image_model = 'gpt-another-reasoning-model → gpt-image-2'
            requester.request_image(profile, None, prompt='Draw paper.')
            new_model = request.call_args.kwargs['cache_key']
            codex.account.invalidate()
            requester.request_image(profile, None, prompt='Draw paper.')
            new_account = request.call_args.kwargs['cache_key']
        self.assertEqual(len({first, new_job, new_model, new_account}), 4)

    def test_codex_mismatched_mask_fails_before_request(self) -> None:
        with patch('ballontranslator.modules.codex.request_image') as request:
            with self.assertRaisesRegex(ValueError, 'mask must match'):
                LLMImageRequester().request_image(
                    default_codex_profile(), np.zeros((3, 5, 3), np.uint8), mask=np.zeros((5, 3), np.uint8)
                )
            request.assert_not_called()

    def test_authentication_error_becomes_required_key_error(self):
        inpainter = FakeInpaint(FakeResponse(status_code=401, json_data={'error': {'message': 'bad key'}}))

        with self.assertRaises(LLMApiKeyRequiredError):
            inpainter._request_inpaint(inpainter.profile, np.zeros((2, 2, 3), dtype=np.uint8))

    def test_status_error_extracts_provider_message(self):
        inpainter = FakeInpaint(FakeResponse(status_code=400, json_data={'error': {'message': 'image provider says no'}}))

        with self.assertRaisesRegex(RuntimeError, 'image provider says no'):
            inpainter._request_inpaint(inpainter.profile, np.zeros((2, 2, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
