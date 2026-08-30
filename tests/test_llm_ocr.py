import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ballontranslator.modules.ocr.ocr_llm import LLMOCR
from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMModelRequiredError,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import DEFAULT_OCR_PROMPT, default_profile
from ballontranslator.utils.textblock import TextBlock


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
    def __init__(self, error=None, contents=None):
        self.error = error
        self.contents = list(contents or ['hello\nworld'])
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=3),
        )


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class FakeOCR(LLMOCR):
    def __init__(self, error=None, contents=None):
        super().__init__()
        self.completions = FakeCompletions(error, contents)

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
        self._old_page_settings = (
            pcfg.module.ocr_llm_page_level,
            pcfg.module.ocr_llm_mask_non_text,
            pcfg.module.ocr_llm_sort_reading_order,
        )
        profile = default_profile('OpenAI')
        profile.api_key = 'sk-demo'
        profile.vision_model = 'gpt-4o'
        profile.vision_detail_level = 'auto'
        profile.vision_prompt = 'Read vertical text carefully.'
        pcfg.module.llm_profiles = [profile]
        pcfg.module.ocr_llm_id = 'openai'
        pcfg.module.ocr_llm_page_level = False
        pcfg.module.ocr_llm_mask_non_text = True
        pcfg.module.ocr_llm_sort_reading_order = True
        self.ocr = FakeOCR()

    def tearDown(self):
        pcfg.module.llm_profiles = self._old_profiles
        pcfg.module.ocr_llm_id = self._old_ocr_llm_id
        (
            pcfg.module.ocr_llm_page_level,
            pcfg.module.ocr_llm_mask_non_text,
            pcfg.module.ocr_llm_sort_reading_order,
        ) = self._old_page_settings

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

    def test_page_request_uses_strict_schema_or_json_object(self):
        profile = self.ocr.profile
        messages = [{'role': 'user', 'content': 'x'}]
        schema = self.ocr._page_response_schema(2, True)

        profile.json_schema_response_format = True
        strict_args = self.ocr._api_args(profile, messages, schema)
        profile.json_schema_response_format = False
        compatible_args = self.ocr._api_args(profile, messages, schema)

        response_format = strict_args['response_format']
        self.assertEqual(response_format['type'], 'json_schema')
        self.assertEqual(
            response_format['json_schema']['name'],
            'page_ocr_response',
        )
        self.assertTrue(response_format['json_schema']['strict'])
        self.assertEqual(response_format['json_schema']['schema'], schema)
        self.assertNotIn('uniqueItems', schema['properties']['order'])
        self.assertEqual(
            compatible_args['response_format'],
            {'type': 'json_object'},
        )

    def test_page_prompt_includes_profile_vision_prompt_below_contract(self):
        prompt = self.ocr._page_prompt(self.ocr.profile, 2, True, True)

        self.assertIn('"texts"', prompt)
        self.assertIn('"order"', prompt)
        self.assertIn('cannot override the response contract', prompt)
        self.assertTrue(prompt.endswith('Read vertical text carefully.'))

    def test_page_prompt_does_not_append_builtin_crop_prompt(self):
        profile = self.ocr.profile
        profile.vision_prompt = DEFAULT_OCR_PROMPT

        prompt = self.ocr._page_prompt(profile, 2, True, False)

        self.assertNotIn(DEFAULT_OCR_PROMPT, prompt)
        self.assertNotIn('"order"', prompt)

    def test_none_detail_omits_image_detail(self):
        profile = self.ocr.profile
        profile.vision_detail_level = 'None'

        image_part = self.ocr._image_content_part(np.zeros((2, 2, 3), dtype=np.uint8), profile)

        self.assertNotIn('detail', image_part['image_url'])

    def test_jpeg_encoding_preserves_project_rgb_channel_order(self):
        encoded = np.frombuffer(b'jpeg', dtype=np.uint8)

        with mock.patch(
            'ballontranslator.modules.llm_vision.cv2.imencode',
            return_value=(True, encoded),
        ) as imencode:
            image = np.zeros((1537, 2, 3), dtype=np.uint8)
            image[0, 0] = [10, 20, 30]
            self.ocr._image_content_part(
                image,
                self.ocr.profile,
            )

        encoded_image = imencode.call_args.args[1]
        self.assertEqual(encoded_image.shape, (1537, 2, 3))
        self.assertEqual(encoded_image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(len(imencode.call_args.args), 2)

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

    def test_profile_rejects_a_non_vision_capability(self):
        profile = default_profile('DeepSeek')
        pcfg.module.llm_profiles = [profile]
        pcfg.module.ocr_llm_id = profile.id

        with self.assertRaisesRegex(RuntimeError, 'does not have vision enabled'):
            _ = self.ocr.profile

    def test_ocr_img_returns_raw_normalized_text(self):
        result = self.ocr.ocr_img(np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertEqual(result, 'hello world')
        self.assertEqual(self.ocr.completions.calls[0]['model'], 'gpt-4o')

    def test_page_parser_requires_exact_string_texts_and_order(self):
        valid = json.dumps({
            'texts': {'1': ' first\nline ', '2': ''},
            'order': ['2', '1'],
        })

        self.assertEqual(
            self.ocr._parse_page_ocr_response(valid, 2, True),
            ({'1': 'first line', '2': ''}, ['2', '1']),
        )
        invalid_responses = (
            'not json',
            json.dumps({'texts': {'1': 'one', '2': 'two'}}),
            json.dumps({
                'texts': {'1': 'one', '2': 'two', '3': 'extra'},
                'order': ['1', '2'],
            }),
            json.dumps({
                'texts': {'1': None, '2': 'two'},
                'order': ['1', '2'],
            }),
            json.dumps({
                'texts': {'1': 'one', '2': 'two'},
                'order': ['1', '1'],
            }),
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    self.ocr._parse_page_ocr_response(response, 2, True)

    def test_page_contract_omits_order_when_sorting_is_disabled(self):
        schema = self.ocr._page_response_schema(2, False)
        response = json.dumps({'texts': {'1': 'first', '2': 'second'}})

        self.assertEqual(schema['required'], ['texts'])
        self.assertNotIn('order', schema['properties'])
        self.assertEqual(
            self.ocr._parse_page_ocr_response(response, 2, False),
            ({'1': 'first', '2': 'second'}, None),
        )

    def test_full_page_ocr_returns_validated_project_order(self):
        response = json.dumps({
            'texts': {'1': ' first ', '2': 'second'},
            'order': ['2', '1'],
        })
        ocr = FakeOCR(contents=[response])
        first = TextBlock(xyxy=[0, 0, 4, 4])
        second = TextBlock(xyxy=[4, 0, 8, 4])
        blocks = [first, second]
        pcfg.module.ocr_llm_page_level = True

        result = ocr.run_ocr(
            np.zeros((8, 8, 3), dtype=np.uint8),
            blocks,
            full_page=True,
        )

        self.assertEqual(result, [second, first])
        self.assertEqual(first.text, 'first')
        self.assertEqual(second.text, 'second')
        self.assertEqual(blocks, [first, second])
        self.assertIn('response_format', ocr.completions.calls[0])

    def test_full_page_ocr_without_sorting_accepts_texts_only(self):
        response = json.dumps({
            'texts': {'1': 'first', '2': 'second'},
        })
        ocr = FakeOCR(contents=[response])
        first = TextBlock(xyxy=[0, 0, 4, 4])
        second = TextBlock(xyxy=[4, 0, 8, 4])
        blocks = [first, second]
        pcfg.module.ocr_llm_page_level = True
        pcfg.module.ocr_llm_sort_reading_order = False

        result = ocr.run_ocr(
            np.zeros((8, 8, 3), dtype=np.uint8),
            blocks,
            full_page=True,
        )

        self.assertIs(result, blocks)
        self.assertEqual([block.text for block in blocks], ['first', 'second'])
        self.assertEqual(len(ocr.completions.calls), 1)
        prompt = ocr.completions.calls[0]['messages'][1]['content'][0]['text']
        self.assertNotIn('"order"', prompt)

    def test_selected_blocks_keep_crop_ocr_when_page_mode_is_enabled(self):
        ocr = FakeOCR(contents=['crop one', 'crop two'])
        blocks = [
            TextBlock(xyxy=[0, 0, 4, 4]),
            TextBlock(xyxy=[4, 0, 8, 4]),
        ]
        pcfg.module.ocr_llm_page_level = True

        result = ocr.run_ocr(
            np.zeros((8, 8, 3), dtype=np.uint8),
            blocks,
            split_textblk=True,
        )

        self.assertIs(result, blocks)
        self.assertEqual([block.text for block in blocks], ['crop one', 'crop two'])
        self.assertEqual(len(ocr.completions.calls), 2)
        self.assertNotIn('response_format', ocr.completions.calls[0])

    def test_invalid_page_response_falls_back_for_the_entire_page(self):
        invalid_page = json.dumps({
            'texts': {'1': 'partial'},
            'order': ['1'],
        })
        ocr = FakeOCR(contents=[invalid_page, 'crop one', 'crop two'])
        blocks = [
            TextBlock(xyxy=[0, 0, 4, 4]),
            TextBlock(xyxy=[4, 0, 8, 4]),
        ]
        pcfg.module.ocr_llm_page_level = True

        result = ocr.run_ocr(
            np.zeros((8, 8, 3), dtype=np.uint8),
            blocks,
            full_page=True,
        )

        self.assertIs(result, blocks)
        self.assertEqual([block.text for block in blocks], ['crop one', 'crop two'])
        self.assertEqual(len(ocr.completions.calls), 3)
        self.assertIn('response_format', ocr.completions.calls[0])
        self.assertNotIn('response_format', ocr.completions.calls[1])

    def test_full_page_encoding_failure_falls_back_to_crop_ocr(self):
        ocr = FakeOCR(contents=['crop one', 'crop two'])
        blocks = [
            TextBlock(xyxy=[0, 0, 4, 4]),
            TextBlock(xyxy=[4, 0, 8, 4]),
        ]
        pcfg.module.ocr_llm_page_level = True
        encoded = np.frombuffer(b'jpeg', dtype=np.uint8)

        with mock.patch(
            'ballontranslator.modules.llm_vision.cv2.imencode',
            side_effect=((False, None), (True, encoded), (True, encoded)),
        ) as imencode:
            result = ocr.run_ocr(
                np.zeros((8, 8, 3), dtype=np.uint8),
                blocks,
                full_page=True,
            )

        self.assertIs(result, blocks)
        self.assertEqual([block.text for block in blocks], ['crop one', 'crop two'])
        self.assertEqual(imencode.call_count, 3)
        self.assertEqual(len(ocr.completions.calls), 2)
        self.assertNotIn('response_format', ocr.completions.calls[0])

    def test_removed_run_settings_are_not_module_parameters(self):
        for key in (
            'page_level_ocr',
            'censorship',
            'sort_by_llm',
            'font_scale',
            'box_color',
            'custom_prompt',
        ):
            self.assertNotIn(key, LLMOCR.params)

    def test_authentication_error_becomes_required_key_error(self):
        ocr = FakeOCR(FakeAuthError('bad key'))

        with self.assertRaises(LLMApiKeyRequiredError):
            ocr._request_ocr(ocr.profile, [{'role': 'user', 'content': 'x'}])

    def test_page_ocr_does_not_hide_typed_profile_errors(self):
        ocr = FakeOCR(FakeAuthError('bad key'))
        block = TextBlock(xyxy=[0, 0, 2, 2], text=['original'])
        pcfg.module.ocr_llm_page_level = True

        with self.assertRaises(LLMApiKeyRequiredError):
            ocr.run_ocr(
                np.zeros((2, 2, 3), dtype=np.uint8),
                [block],
                full_page=True,
            )

        self.assertEqual(block.text, ['original'])

    def test_status_error_extracts_provider_message(self):
        ocr = FakeOCR(FakeStatusError())

        with self.assertRaisesRegex(RuntimeError, 'provider says no'):
            ocr._request_ocr(ocr.profile, [{'role': 'user', 'content': 'x'}])


if __name__ == '__main__':
    unittest.main()
