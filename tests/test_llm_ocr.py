import copy
import unittest
from types import SimpleNamespace

import numpy as np

from ballontranslator.modules.ocr.ocr_llm import LLMOCR
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_profile


class FakeAuthError(Exception):
    pass


class FakeStatusError(Exception):
    def __init__(self):
        self.response = SimpleNamespace(json=lambda: {'error': {'message': 'provider says no'}}, text='raw')
        super().__init__('status')


class FakeOpenAI:
    AuthenticationError = FakeAuthError
    APIStatusError = FakeStatusError


class FakeCompletions:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='hello\nworld'))],
            usage=SimpleNamespace(total_tokens=3),
        )


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class FakeOCR(LLMOCR):
    def __init__(self, error=None):
        super().__init__()
        self.completions = FakeCompletions(error)

    def _openai_module(self):
        return FakeOpenAI

    def _initialize_client(self, profile):
        self._api_key_for_profile(profile)
        return FakeClient(self.completions)

    def _respect_delay(self):
        pass


class LLMOCRTest(unittest.TestCase):
    def setUp(self):
        self._old_profiles = copy.deepcopy(pcfg.module.llm_profiles)
        self._old_ocr_llm_id = pcfg.module.ocr_llm_id
        profile = default_profile('OpenAI')
        profile.api_key = 'sk-demo'
        profile.vision_model = 'gpt-4o'
        profile.vision_detail_level = 'auto'
        pcfg.module.llm_profiles = [profile]
        pcfg.module.ocr_llm_id = 'openai'
        self.ocr = FakeOCR()

    def tearDown(self):
        pcfg.module.llm_profiles = self._old_profiles
        pcfg.module.ocr_llm_id = self._old_ocr_llm_id

    def test_missing_required_api_key_raises_profile_error(self):
        profile = default_profile('OpenAI')
        profile.api_key = ''

        with self.assertRaises(LLMApiKeyRequiredError):
            self.ocr._api_key_for_profile(profile)

    def test_request_args_use_vision_model_and_detail(self):
        profile = self.ocr.profile
        img = np.zeros((2, 2, 3), dtype=np.uint8)

        messages = self.ocr._messages(img, profile)
        args = self.ocr._api_args(profile, messages)

        self.assertEqual(args['model'], 'gpt-4o')
        self.assertNotEqual(args['model'], profile.model)
        image_part = messages[1]['content'][1]
        self.assertEqual(image_part['type'], 'image_url')
        self.assertIn('data:image/jpeg;base64,', image_part['image_url']['url'])
        self.assertEqual(image_part['image_url']['detail'], 'auto')

    def test_none_detail_omits_image_detail(self):
        profile = self.ocr.profile
        profile.vision_detail_level = 'None'

        image_part = self.ocr._image_content_part(np.zeros((2, 2, 3), dtype=np.uint8), profile)

        self.assertNotIn('detail', image_part['image_url'])

    def test_blank_vision_model_requires_model(self):
        profile = self.ocr.profile
        profile.vision_model = ''

        with self.assertRaises(LLMModelRequiredError):
            self.ocr._api_args(profile, [{'role': 'user', 'content': 'x'}])

    def test_vision_enabled_profile_requires_model(self):
        profile = default_profile('OpenAI')
        profile.api_key = 'sk-demo'
        profile.model = ''
        profile.vision_model = ''
        pcfg.module.llm_profiles = [profile]
        pcfg.module.ocr_llm_id = profile.id

        with self.assertRaises(LLMModelRequiredError):
            _ = self.ocr.profile
        with self.assertRaises(LLMModelRequiredError):
            self.ocr._api_args(profile, [{'role': 'user', 'content': 'x'}])

        profile.vision_model = 'stale-vision-model'
        profile.vision_model_options = []
        with self.assertRaises(LLMModelRequiredError):
            _ = self.ocr.profile
        with self.assertRaises(LLMModelRequiredError):
            self.ocr._api_args(profile, [{'role': 'user', 'content': 'x'}])

    def test_ocr_img_returns_raw_normalized_text(self):
        result = self.ocr.ocr_img(np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertEqual(result, 'hello world')
        self.assertEqual(self.ocr.completions.calls[0]['model'], 'gpt-4o')

    def test_authentication_error_becomes_required_key_error(self):
        ocr = FakeOCR(FakeAuthError('bad key'))

        with self.assertRaises(LLMApiKeyRequiredError):
            ocr._request_ocr(ocr.profile, [{'role': 'user', 'content': 'x'}])

    def test_status_error_extracts_provider_message(self):
        ocr = FakeOCR(FakeStatusError())

        with self.assertRaisesRegex(RuntimeError, 'provider says no'):
            ocr._request_ocr(ocr.profile, [{'role': 'user', 'content': 'x'}])


if __name__ == '__main__':
    unittest.main()
