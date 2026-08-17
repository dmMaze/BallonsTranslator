import base64
import copy
import io
import unittest

import numpy as np
from PIL import Image

from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMBaseURLRequiredError, LLMModelRequiredError
from ballontranslator.modules.inpaint.inpaint_llm import LLMInpaint
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_profile


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


class LLMInpaintTest(unittest.TestCase):
    def setUp(self):
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

    def test_openrouter_request_accepts_final_images_endpoint(self):
        profile = self.inpainter.profile
        profile.image_base_url = 'https://openrouter.ai/api/v1/images'

        self.inpainter._request_inpaint(profile, np.zeros((2, 2, 3), dtype=np.uint8))

        call = self.inpainter.http_client.calls[0]
        self.assertEqual(call['url'], 'https://openrouter.ai/api/v1/images')

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
