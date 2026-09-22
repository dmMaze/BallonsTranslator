import base64
import copy
import io
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
from ballontranslator.modules import codex, llm_image
from ballontranslator.modules.llm_image import (
    LLMImageRequester,
    LLMImageRequestPolicy,
    _SharedLLMImageThrottle,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_profile, sync_codex_profile
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


class LLMInpaintTest(unittest.TestCase):
    def setUp(self):
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
        codex_profile = default_profile('Codex')
        with patch.object(codex.account, 'require_sign_in', side_effect=LLMUserActionRequiredError('Sign in with ChatGPT')):
            with self.assertRaisesRegex(LLMUserActionRequiredError, 'Sign in with ChatGPT'):
                self.inpainter.inpaint(np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 255, np.uint8), profile=codex_profile)

    def test_codex_inpaint_sends_aligned_mask_and_preserves_unmasked_rgba_pixels(self) -> None:
        profile = default_profile('Codex')
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
        profile = default_profile('Codex')
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
            self.assertNotIn('white marks the editable region', request.call_args.args[1])
            np.testing.assert_array_equal(image, np.full_like(image, 37))

    def test_codex_downscaling_retains_single_pixel_mask(self) -> None:
        profile = default_profile('Codex')
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
        profile = default_profile('Codex')
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
        pcfg.module.llm_profiles = [default_profile('Codex')]
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
        profile = default_profile('Codex')
        sync_codex_profile(profile, {'text-model': {'modalities': ['text'], 'efforts': []}})
        requester = LLMImageRequester(image_request_policy=LLMImageRequestPolicy(
            delay=0, max_requests_per_minute=0, retry_attempts=2, retry_timeout=0,
        ))
        with patch('ballontranslator.modules.codex.request_image', side_effect=[b'broken image', _png_bytes()]) as request:
            result = requester.request_image_with_retries(profile, None, 'Draw paper.', 'gpt-image-2')
        self.assertEqual(result[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(request.call_args.args[1:4], ('Draw paper.', None, None))
        for error in (LLMUserActionRequiredError('Sign in.'), LLMRequestStopped()):
            with self.subTest(error=error), patch('ballontranslator.modules.codex.request_image', side_effect=error) as request:
                with self.assertRaises(type(error)):
                    requester.request_image_with_retries(profile, None, 'Draw paper.', 'gpt-image-2')
                request.assert_called_once()

    def test_codex_mismatched_mask_fails_before_request(self) -> None:
        with patch('ballontranslator.modules.codex.request_image') as request:
            with self.assertRaisesRegex(ValueError, 'mask must match'):
                LLMImageRequester().request_image(
                    default_profile('Codex'), np.zeros((3, 5, 3), np.uint8), mask=np.zeros((5, 3), np.uint8)
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
