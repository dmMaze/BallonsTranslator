import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMOutputLimitError,
    LLMRequestStopped,
)
from ballontranslator.modules.llm_chat import (
    LLMChatRequester,
    LLMChatRequestError,
    openai_chat_completion_args,
    openai_json_response_format,
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

        self.profile.base_url = (
            'https://api.openai.com/v1/chat/completions'
        )
        self.assertEqual(
            openai_chat_completion_args(self.profile, 'gpt-5.5'),
            {'top_p': 1.0, 'max_completion_tokens': 8192},
        )

    def test_reasoning_control_maps_provider_specific_disable_requests(self):
        self.profile.thinking_level = 'None'
        automatic = openai_chat_completion_args(
            self.profile,
            'gpt-5.5',
        )
        self.assertNotIn('reasoning_effort', automatic)
        self.assertNotIn('extra_body', automatic)

        self.profile.thinking_level = 'Disabled'
        native_openai = openai_chat_completion_args(
            self.profile,
            'gpt-5.5',
        )
        self.assertEqual(native_openai['reasoning_effort'], 'none')

        self.profile.base_url = (
            'https://api.deepseek.com/v1/chat/completions'
        )
        deepseek = openai_chat_completion_args(
            self.profile,
            'deepseek-v4-flash-vision-exp',
        )
        self.assertEqual(
            deepseek['extra_body'],
            {'thinking': {'type': 'disabled'}},
        )
        self.assertNotIn('reasoning_effort', deepseek)

        self.profile.base_url = 'https://openrouter.ai/api/v1'
        openrouter = openai_chat_completion_args(
            self.profile,
            'deepseek/deepseek-v4-flash-vision-exp',
        )
        self.assertEqual(
            openrouter['extra_body'],
            {'reasoning': {'effort': 'none'}},
        )

        self.profile.thinking_level = 'low'
        explicit = openai_chat_completion_args(
            self.profile,
            'deepseek/deepseek-v4-flash-vision-exp',
        )
        self.assertEqual(explicit['reasoning_effort'], 'low')
        self.assertNotIn('extra_body', explicit)

    def test_json_response_format_preserves_strict_and_compatible_shapes(self):
        schema = {'type': 'object', 'properties': {}}
        self.profile.json_schema_response_format = True

        strict = openai_json_response_format(
            self.profile,
            'demo_response',
            schema,
        )
        self.profile.json_schema_response_format = False
        compatible = openai_json_response_format(
            self.profile,
            'demo_response',
            schema,
        )

        self.assertEqual(strict, {
            'type': 'json_schema',
            'json_schema': {
                'name': 'demo_response',
                'strict': True,
                'schema': schema,
            },
        })
        self.assertIs(strict['json_schema']['schema'], schema)
        self.assertEqual(compatible, {'type': 'json_object'})

    def test_request_returns_normalized_content_and_usage(self):
        usage = SimpleNamespace(total_tokens=3)
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='hello'),
                finish_reason='stop',
            )],
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
        self.assertEqual(result.finish_reason, 'stop')

    def test_request_raises_actionable_output_limit(self):
        self.profile.max_tokens = 1234
        self.profile.thinking_level = 'low'
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=1234,
            total_tokens=1244,
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"partial":'),
                finish_reason='length',
            )],
            usage=usage,
        )
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=mock.Mock(return_value=completion)
            )
        ))

        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=FakeOpenAI
            ),
            mock.patch.object(
                self.requester, '_initialize_client', return_value=client
            ),
            mock.patch.object(self.requester, '_respect_delay'),
        ):
            with self.assertRaisesRegex(
                LLMOutputLimitError,
                r'Max Tokens: 1234.*Thinking Level: low',
            ):
                self.requester.request_chat_completion(
                    self.profile,
                    {'model': 'demo-model', 'messages': []},
                )

        self.assertEqual(len(self.requester.logs), 1)
        self.assertIn('completion=1234', self.requester.logs[0])
        self.assertIn('content=\'{"partial":\'', self.requester.logs[0])

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

    def test_full_chat_endpoint_is_requested_once(self):
        requested_urls = []

        def respond(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            return httpx.Response(200, json={
                'id': 'chatcmpl-test',
                'object': 'chat.completion',
                'created': 0,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'ok'},
                    'finish_reason': 'stop',
                }],
            })

        self.profile.base_url = (
            'https://openrouter.ai/api/v1/chat/completions/'
        )
        with (
            httpx.Client(transport=httpx.MockTransport(respond)) as http_client,
            mock.patch.object(
                self.requester, '_http_client', return_value=http_client
            ),
            mock.patch.object(self.requester, '_respect_delay'),
        ):
            result = self.requester.request_chat_completion(
                self.profile,
                {'model': 'test-model', 'messages': []},
            )

        self.assertEqual(result.content, 'ok')
        self.assertEqual(
            requested_urls,
            ['https://openrouter.ai/api/v1/chat/completions'],
        )

    def test_wait_remains_stop_aware(self):
        stop_event = threading.Event()
        stop_event.set()
        self.requester.set_stop_event(stop_event)

        with self.assertRaises(LLMRequestStopped):
            self.requester._wait(5)


if __name__ == '__main__':
    unittest.main()
