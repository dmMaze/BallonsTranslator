import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMRequestStopped,
)
from ballontranslator.modules.llm_chat import (
    LLMChatRequester,
    LLMChatRequestError,
    openai_chat_completion_args,
)
from ballontranslator.utils.llm_profiles import default_profile


class FakeAuthError(Exception):
    pass


class FakeStatusError(Exception):
    def __init__(self, message: str = 'provider says no') -> None:
        self.response = SimpleNamespace(
            json=lambda: {'error': {'message': message}},
            text='raw',
        )
        super().__init__('status')


class FakeOpenAI:
    AuthenticationError = FakeAuthError
    APIStatusError = FakeStatusError


class RequesterHarness(LLMChatRequester):
    def __init__(self) -> None:
        super().__init__()
        self.values = {
            'delay': 0.0,
            'max requests per minute': 0,
            'proxy': '',
        }
        self.logs = []
        self.logger = SimpleNamespace(
            debug=self.logs.append,
            error=self.logs.append,
            warning=self.logs.append,
        )

    def get_param_value(self, name: str):
        return self.values[name]


class LLMChatRequesterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.requester = RequesterHarness()
        self.profile = default_profile('OpenAI')
        self.profile.api_key = 'sk-demo'

    def test_provider_args_preserve_native_openai_compatibility(self):
        self.assertEqual(
            openai_chat_completion_args(self.profile, 'gpt-5.5'),
            {'top_p': 1.0, 'max_completion_tokens': 8192},
        )
        self.assertEqual(
            openai_chat_completion_args(self.profile, 'gpt-4o'),
            {
                'top_p': 1.0,
                'temperature': 0.1,
                'max_tokens': 8192,
            },
        )

        self.profile.base_url = 'https://openrouter.ai/api/v1'
        self.assertEqual(
            openai_chat_completion_args(self.profile, 'openai/gpt-5.5'),
            {
                'top_p': 1.0,
                'temperature': 0.1,
                'max_tokens': 8192,
            },
        )

    def test_request_returns_normalized_content_and_usage(self):
        usage = SimpleNamespace(total_tokens=3)
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='hello'))],
            usage=usage,
        )
        completions = SimpleNamespace(create=mock.Mock(return_value=completion))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        api_args = {'model': 'gpt-5.5', 'messages': []}

        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=FakeOpenAI
            ),
            mock.patch.object(
                self.requester, '_initialize_client', return_value=client
            ),
            mock.patch.object(self.requester, '_respect_delay') as delay,
        ):
            result = self.requester.request_chat_completion(
                self.profile, api_args
            )

        delay.assert_called_once_with()
        completions.create.assert_called_once_with(**api_args)
        self.assertEqual(result.content, 'hello')
        self.assertIs(result.usage, usage)

    def test_request_normalizes_authentication_and_status_errors(self):
        def client_for(error: Exception):
            completions = SimpleNamespace(
                create=mock.Mock(side_effect=error)
            )
            return SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )

        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=FakeOpenAI
            ),
            mock.patch.object(self.requester, '_respect_delay'),
            mock.patch.object(
                self.requester,
                '_initialize_client',
                return_value=client_for(FakeAuthError('bad key')),
            ),
        ):
            with self.assertRaises(LLMApiKeyRequiredError):
                self.requester.request_chat_completion(self.profile, {})

        provider_error = FakeStatusError()
        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=FakeOpenAI
            ),
            mock.patch.object(self.requester, '_respect_delay'),
            mock.patch.object(
                self.requester,
                '_initialize_client',
                return_value=client_for(provider_error),
            ),
        ):
            with self.assertRaisesRegex(
                LLMChatRequestError, 'provider says no'
            ) as caught:
                self.requester.request_chat_completion(self.profile, {})
        self.assertIs(caught.exception.provider_error, provider_error)

    def test_client_is_reused_for_the_same_profile_and_proxy(self):
        client = object()
        openai = SimpleNamespace(OpenAI=mock.Mock(return_value=client))
        http_client = object()
        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=openai
            ),
            mock.patch.object(
                self.requester, '_http_client', return_value=http_client
            ),
        ):
            first = self.requester._initialize_client(self.profile)
            second = self.requester._initialize_client(self.profile)

        self.assertIs(first, client)
        self.assertIs(second, client)
        openai.OpenAI.assert_called_once_with(
            api_key='sk-demo',
            base_url='https://api.openai.com/v1',
            http_client=http_client,
        )

    def test_wait_remains_stop_aware(self):
        stop_event = threading.Event()
        stop_event.set()
        self.requester.set_stop_event(stop_event)

        with self.assertRaises(LLMRequestStopped):
            self.requester._wait(5)


if __name__ == '__main__':
    unittest.main()
