import unittest
from types import SimpleNamespace
from unittest import mock

from ballontranslator.modules.exceptions import (
    LLMModelRequiredError,
    LLMUserActionRequiredError,
)
from ballontranslator.modules.context.errors import ContextLengthError
from ballontranslator.modules.context.token_usage import format_token_usage
from ballontranslator.modules.llm_chat import LLMChatRequestError
from ballontranslator.modules.translators.llm_translation_contract import (
    TranslationPromptSpec,
    translation_system_prompt,
)
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_profile


class FakeStatusError(Exception):
    def __init__(self, message='provider says no', status_code=400, code=''):
        self.status_code = status_code
        self.code = code
        self.response = SimpleNamespace(
            json=lambda: {'error': {'message': message, 'code': code}},
            text='raw',
            status_code=status_code,
        )
        super().__init__('status')


class LLMTranslatorTest(unittest.TestCase):
    def setUp(self):
        self.translator = LLMTranslator('日本語', '简体中文')

    def _prompt_spec(
        self,
        profile,
        *,
        summary_enabled: bool = False,
    ) -> TranslationPromptSpec:
        target_language = self.translator._translated_lang(
            self.translator.lang_target
        )
        return TranslationPromptSpec(
            source_language=self.translator._translated_lang(
                self.translator.lang_source
            ),
            target_language=target_language,
            system_prompt=translation_system_prompt(
                profile.prompt,
                target_language,
                history_enabled=False,
                summary_enabled=summary_enabled,
            ),
            summary_enabled=summary_enabled,
        )

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

    def test_dynamic_schema_stays_out_of_cacheable_message_prefix(self):
        profile = default_profile('LM Studio')
        messages = [{'role': 'system', 'content': 'stable prefix'}]

        one_item = self.translator._api_args(
            profile,
            messages,
            expected_translations=1,
        )
        three_items = self.translator._api_args(
            profile,
            messages,
            expected_translations=3,
        )

        self.assertIs(one_item['messages'], messages)
        self.assertIs(three_items['messages'], messages)
        self.assertEqual(
            one_item['response_format']['json_schema']['schema']['required'],
            ['1'],
        )
        self.assertEqual(
            three_items['response_format']['json_schema']['schema']['required'],
            ['1', '2', '3'],
        )

    def test_translate_passes_input_count_to_structured_request(self):
        profile = default_profile('LM Studio')
        with mock.patch.object(
            self.translator,
            '_request_translation',
            return_value='{"1":"甲","2":"乙","3":"丙"}',
        ) as request:
            result = self.translator._translate(
                ['a', 'b', 'c'],
                profile=profile,
                prompt_spec=self._prompt_spec(profile),
            )

        self.assertEqual(result, ['甲', '乙', '丙'])
        self.assertEqual(request.call_args.kwargs['expected_translations'], 3)

    def test_context_status_error_is_typed_and_preserves_provider_message(self):
        provider_message = (
            "This model's maximum context length is 4096 tokens, but the "
            'request used 5000 tokens.'
        )
        provider_error = FakeStatusError(provider_message)
        profile = default_profile('OpenAI')

        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            side_effect=LLMChatRequestError(provider_error),
        ), self.assertRaisesRegex(
            ContextLengthError, 'maximum context length'
        ) as caught:
            self.translator._request_translation(
                profile, [{'role': 'user', 'content': 'x'}]
            )
        self.assertIs(caught.exception.__cause__, provider_error)

    def test_context_error_code_is_recognized_but_unrelated_errors_are_not(self):
        profile = default_profile('OpenAI')
        coded_error = FakeStatusError(
            'input rejected',
            code='context_length_exceeded',
        )
        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            side_effect=LLMChatRequestError(coded_error),
        ), self.assertRaises(ContextLengthError):
            self.translator._request_translation(
                profile, [{'role': 'user', 'content': 'x'}]
            )

        unrelated_errors = (
            FakeStatusError('max_tokens must be less than 8192'),
            FakeStatusError('maximum context length exceeded', status_code=404),
            FakeStatusError('maximum context length exceeded', status_code=500),
            FakeStatusError('rate limit exceeded', status_code=429),
        )
        for error in unrelated_errors:
            with self.subTest(error=error.response.json()['error']['message']):
                with mock.patch.object(
                    self.translator,
                    'request_chat_completion',
                    side_effect=LLMChatRequestError(error),
                ), self.assertRaises(LLMChatRequestError) as caught:
                    self.translator._request_translation(
                        profile,
                        [{'role': 'user', 'content': 'x'}],
                    )
                self.assertNotIsInstance(caught.exception, ContextLengthError)

    def test_user_action_required_bypasses_ordinary_retries(self):
        profile = default_profile('OpenAI')
        self.translator.set_param_value('retry attempts', 5)
        self.translator.set_param_value('retry timeout', 0)

        with mock.patch.object(
            self.translator,
            '_request_translation',
            side_effect=LLMUserActionRequiredError('update the profile'),
        ) as request:
            with self.assertRaisesRegex(
                LLMUserActionRequiredError,
                'update the profile',
            ):
                self.translator._translate(
                    ['source'],
                    profile=profile,
                    prompt_spec=self._prompt_spec(profile),
                    page_key='003-0.png',
                )

        self.assertEqual(request.call_count, 1)

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
            'completion_tokens_details': {'reasoning_tokens': 18},
            'total_tokens': 120,
            'prompt_cache_hit_tokens': 70,
            'prompt_cache_miss_tokens': 30,
        }
        self.assertEqual(
            format_token_usage(openai_usage),
            'prompt=100, completion=20, total=120, cache_hit=80',
        )
        self.assertEqual(
            format_token_usage(deepseek_usage),
            'prompt=100, completion=20, reasoning=18, total=120, '
            'cache_hit=70, cache_miss=30',
        )

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


if __name__ == '__main__':
    unittest.main()
