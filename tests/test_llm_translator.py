import threading
import unittest
from types import SimpleNamespace

from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped
from ballontranslator.modules.translators.token_usage import format_token_usage
from ballontranslator.modules.translators.trans_llm import InvalidNumTranslations, LLMTranslator
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
    def __init__(self, error):
        self.error = error

    def create(self, **kwargs):
        raise self.error


class FakeClient:
    def __init__(self, error):
        self.chat = SimpleNamespace(completions=FakeCompletions(error))


class FakeTranslator(LLMTranslator):
    def __init__(self, error=None, profile=None):
        self.fake_error = error
        self.profile_override = profile
        super().__init__('日本語', '简体中文')

    def _openai_module(self):
        return FakeOpenAI

    def _initialize_client(self, profile):
        return FakeClient(self.fake_error)

    @property
    def profile(self):
        if self.profile_override is not None:
            return self.profile_override
        profile = default_profile('OpenAI')
        profile.api_key = 'sk-demo'
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

    def test_json_prompt_wraps_profile_prompt_without_formatting_json_braces(self):
        profile = default_profile('OpenAI')
        profile.prompt = 'Keep JSON example {"x": 1}.'

        messages, _, prompt = next(self.translator._assemble_batches(['心'], profile))

        self.assertIn('Translate every source string into Simplified Chinese.', messages[0]['content'])
        self.assertIn('Additional translation instructions:\nKeep JSON example {"x": 1}.', messages[0]['content'])
        self.assertIn('"translations"', messages[0]['content'])
        self.assertIn('"source": "心"', prompt)

    def test_missing_required_api_key_raises_profile_error(self):
        profile = default_profile('OpenAI')
        profile.api_key = ''

        with self.assertRaises(LLMApiKeyRequiredError):
            self.translator._api_key_for_profile(profile)

    def test_text_disabled_profile_is_not_translator_usable(self):
        old_profiles = pcfg.module.llm_profiles
        old_translator_llm_id = pcfg.module.translator_llm_id
        profile = default_profile('OpenAI')
        profile.support_text = False
        try:
            pcfg.module.llm_profiles = [profile]
            pcfg.module.translator_llm_id = profile.id

            with self.assertRaisesRegex(RuntimeError, 'text translation'):
                _ = self.translator.profile
        finally:
            pcfg.module.llm_profiles = old_profiles
            pcfg.module.translator_llm_id = old_translator_llm_id

    def test_text_enabled_profile_requires_model(self):
        old_profiles = pcfg.module.llm_profiles
        old_translator_llm_id = pcfg.module.translator_llm_id
        profile = default_profile('OpenAI')
        profile.model = ''
        try:
            pcfg.module.llm_profiles = [profile]
            pcfg.module.translator_llm_id = profile.id

            with self.assertRaises(LLMModelRequiredError):
                _ = self.translator.profile
            with self.assertRaises(LLMModelRequiredError):
                self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])

            profile.model = 'stale-model'
            profile.model_options = []
            with self.assertRaises(LLMModelRequiredError):
                _ = self.translator.profile
            with self.assertRaises(LLMModelRequiredError):
                self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])
        finally:
            pcfg.module.llm_profiles = old_profiles
            pcfg.module.translator_llm_id = old_translator_llm_id

    def test_model_required_error_propagates_from_request_loop(self):
        profile = default_profile('OpenAI')
        profile.api_key = 'sk-demo'
        profile.model = ''
        translator = FakeTranslator(profile=profile)

        with self.assertRaises(LLMModelRequiredError):
            translator._translate(['hello'])

    def test_thinking_level_only_passed_when_not_none(self):
        profile = default_profile('OpenAI')
        profile.thinking_level = 'None'
        args = self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])
        self.assertNotIn('reasoning_effort', args)
        self.assertNotIn('frequency_penalty', args)
        self.assertNotIn('presence_penalty', args)
        self.assertEqual(args['response_format'], {'type': 'json_object'})
        self.assertEqual(args['max_tokens'], 8192)

        profile.thinking_level = 'none'
        args = self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])
        self.assertNotIn('reasoning_effort', args)

        profile.thinking_level = 'low'
        args = self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])
        self.assertEqual(args['reasoning_effort'], 'low')

    def test_json_schema_response_format_is_profile_controlled(self):
        profile = default_profile('OpenAI')
        profile.json_schema_response_format = True

        args = self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])

        self.assertEqual(args['response_format']['type'], 'json_schema')
        self.assertEqual(args['response_format']['json_schema']['name'], 'translation_response')
        self.assertTrue(args['response_format']['json_schema']['strict'])
        self.assertEqual(args['response_format']['json_schema']['schema'], self.translator._json_schema())

    def test_stop_event_interrupts_wait(self):
        event = threading.Event()
        event.set()
        self.translator.set_stop_event(event)

        with self.assertRaises(LLMRequestStopped):
            self.translator._wait(5)

    def test_runtime_settings_live_on_translator_params(self):
        self.translator.set_param_value('delay', 0.8)
        self.translator.set_param_value('retry attempts', 2)
        self.translator.set_param_value('retry timeout', 3)
        self.translator.set_param_value('max requests per minute', 7)
        self.translator.set_param_value('proxy', 'http://127.0.0.1:7890')

        self.assertEqual(self.translator.delay(), 0.8)
        self.assertEqual(self.translator.get_param_value('retry attempts'), 2)
        self.assertEqual(self.translator.get_param_value('retry timeout'), 3.0)
        self.assertEqual(self.translator.get_param_value('max requests per minute'), 7)
        self.assertEqual(self.translator.get_param_value('proxy'), 'http://127.0.0.1:7890')

    def test_authentication_error_becomes_required_key_error(self):
        translator = FakeTranslator(FakeAuthError('bad key'))

        with self.assertRaises(LLMApiKeyRequiredError):
            translator._request_translation(translator.profile, [{'role': 'user', 'content': 'x'}])

    def test_status_error_extracts_provider_message(self):
        translator = FakeTranslator(FakeStatusError())

        with self.assertRaisesRegex(RuntimeError, 'provider says no'):
            translator._request_translation(translator.profile, [{'role': 'user', 'content': 'x'}])

    def test_token_usage_supports_openai_and_deepseek_cache_fields(self):
        openai_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
        )
        deepseek_usage = {
            'prompt_tokens': 100,
            'completion_tokens': 20,
            'total_tokens': 120,
            'prompt_cache_hit_tokens': 70,
            'prompt_cache_miss_tokens': 30,
        }
        messages = []
        self.translator.logger = SimpleNamespace(info=messages.append)

        self.translator._log_token_usage(
            SimpleNamespace(usage=openai_usage),
        )
        self.assertEqual(
            messages,
            ['LLM token usage: prompt=100, completion=20, total=120, cache_hit=80'],
        )
        self.assertEqual(
            format_token_usage(deepseek_usage),
            'prompt=100, completion=20, total=120, cache_hit=70, cache_miss=30',
        )

    def test_token_usage_omits_missing_or_invalid_fields(self):
        class IncompleteUsage:
            total_tokens = 3

            @property
            def prompt_tokens(self):
                raise RuntimeError('not available')

        self.assertEqual(
            format_token_usage(IncompleteUsage()),
            'total=3',
        )
        self.assertEqual(format_token_usage(None), '')
        self.translator._log_token_usage(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
