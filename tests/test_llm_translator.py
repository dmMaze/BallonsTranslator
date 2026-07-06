import threading
import unittest
from types import SimpleNamespace

from ballontranslator.modules.translators.exceptions import LLMApiKeyRequiredError, LLMTranslationStopped
from ballontranslator.modules.translators.trans_llm import LLMTranslator
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
    def __init__(self, error):
        self.error = error

    def create(self, **kwargs):
        raise self.error


class FakeClient:
    def __init__(self, error):
        self.chat = SimpleNamespace(completions=FakeCompletions(error))


class FakeTranslator(LLMTranslator):
    def __init__(self, error=None):
        self.fake_error = error
        super().__init__('日本語', '简体中文')

    def _openai_module(self):
        return FakeOpenAI

    def _initialize_client(self, profile):
        return FakeClient(self.fake_error)

    @property
    def profile(self):
        profile = default_profile('OpenAI')
        profile['api key'] = 'sk-demo'
        return profile


class LLMTranslatorTest(unittest.TestCase):
    def setUp(self):
        self.translator = LLMTranslator('日本語', '简体中文')

    def test_json_response_parser_accepts_schema(self):
        result = self.translator._parse_response(
            {},
            '{"translations": [{"id": 1, "translation": "心"}, {"id": 2, "translation": "精神"}]}',
            2,
        )

        self.assertEqual(result, ['心', '精神'])

    def test_legacy_response_parser_uses_delimiters(self):
        result = self.translator._parse_legacy_response(
            '<|1|>心\n<|2|>精神',
            2,
        )

        self.assertEqual(result, ['心', '精神'])

    def test_xml_response_parser_uses_ids(self):
        result = self.translator._parse_xml_response(
            '<root><element><id>1</id><dst>心</dst></element><element><id>2</id><dst>精神</dst></element></root>',
            2,
        )

        self.assertEqual(result, ['心', '精神'])

    def test_json_prompt_uses_system_prompt_without_formatting_json_braces(self):
        profile = default_profile('OpenAI')
        profile['system prompt'] = 'Keep JSON example {"x": 1}. Translate to {to_lang}.'

        messages, _, prompt = next(self.translator._assemble_batches(['心'], profile))

        self.assertEqual(messages[0]['content'], 'Keep JSON example {"x": 1}. Translate to Simplified Chinese.')
        self.assertIn('"source": "心"', prompt)

    def test_missing_required_api_key_raises_profile_error(self):
        profile = default_profile('OpenAI')
        profile['api key'] = ''

        with self.assertRaises(LLMApiKeyRequiredError):
            self.translator._api_key_for_profile(profile)

    def test_thinking_level_only_passed_when_not_none(self):
        profile = default_profile('OpenAI')
        profile['thinking level'] = 'None'
        args = self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])
        self.assertNotIn('reasoning_effort', args)
        self.assertEqual(args['response_format'], {'type': 'json_object'})

        profile['thinking level'] = 'none'
        args = self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])
        self.assertNotIn('reasoning_effort', args)

        profile['thinking level'] = 'low'
        args = self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])
        self.assertEqual(args['reasoning_effort'], 'low')

    def test_stop_event_interrupts_wait(self):
        event = threading.Event()
        event.set()
        self.translator.set_stop_event(event)

        with self.assertRaises(LLMTranslationStopped):
            self.translator._wait(5)

    def test_runtime_settings_live_on_translator_params(self):
        self.translator.set_param_value('delay', 0.8)
        self.translator.set_param_value('retry attempts', 2)
        self.translator.set_param_value('retry timeout', 3)
        self.translator.set_param_value('max requests per minute', 7)
        self.translator.set_param_value('proxy', 'http://127.0.0.1:7890')

        self.assertEqual(self.translator.delay(), 0.8)
        self.assertEqual(self.translator._setting_int('retry attempts'), 2)
        self.assertEqual(self.translator._setting_float('retry timeout'), 3.0)
        self.assertEqual(self.translator._setting_int('max requests per minute'), 7)
        self.assertEqual(self.translator._setting_str('proxy'), 'http://127.0.0.1:7890')

    def test_authentication_error_becomes_required_key_error(self):
        translator = FakeTranslator(FakeAuthError('bad key'))

        with self.assertRaises(LLMApiKeyRequiredError):
            translator._request_translation(translator.profile, [{'role': 'user', 'content': 'x'}])

    def test_status_error_extracts_provider_message(self):
        translator = FakeTranslator(FakeStatusError())

        with self.assertRaisesRegex(RuntimeError, 'provider says no'):
            translator._request_translation(translator.profile, [{'role': 'user', 'content': 'x'}])


if __name__ == '__main__':
    unittest.main()
