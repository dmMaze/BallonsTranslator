"""Shared OpenAI-compatible Chat Completions request transport."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .context.errors import provider_error_message
from .exceptions import LLMApiKeyRequiredError, LLMRequestStopped
from ballontranslator.utils.llm_profiles import (
    LLMProfile,
    PROVIDER_DEFAULTS,
    resolve_api_key,
)


OPENAI_MAX_TOKENS_MODELS = frozenset({
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
})


def _normalized_base_url(url: str) -> str:
    return str(url or '').strip().rstrip('/')


def openai_chat_completion_args(
    profile: LLMProfile,
    model: str,
) -> Dict[str, Any]:
    """Map provider-neutral profile values to OpenAI chat API arguments.

    Native OpenAI models default to ``max_completion_tokens`` so future model
    names do not need version-pattern guesses. Only explicitly listed older
    models and compatibility endpoints retain ``max_tokens``. Native GPT-5.5+
    models use the API's fixed default temperature, so omit that argument.

    >>> profile = LLMProfile.from_provider('OpenAI')
    >>> openai_chat_completion_args(profile, 'gpt-5.5')
    {'top_p': 1.0, 'max_completion_tokens': 8192}
    >>> openai_chat_completion_args(profile, 'gpt-4o')['temperature']
    0.1
    """

    base_url = _normalized_base_url(profile.base_url)
    openai_base_url = _normalized_base_url(
        PROVIDER_DEFAULTS['OpenAI']['base_url']
    )
    model_name = str(model or '').rsplit('/', 1)[-1].lower()
    is_native_openai = not base_url or base_url == openai_base_url
    args: Dict[str, Any] = {'top_p': float(profile.top_p)}
    version_match = re.match(
        r'^gpt-(\d+)(?:\.(\d+))?(?:-|$)', model_name
    )
    gpt_version = (
        (int(version_match.group(1)), int(version_match.group(2) or 0))
        if version_match
        else None
    )
    if not is_native_openai or gpt_version is None or gpt_version < (5, 5):
        args['temperature'] = float(profile.temperature)
    token_limit_key = (
        'max_completion_tokens'
        if is_native_openai and model_name not in OPENAI_MAX_TOKENS_MODELS
        else 'max_tokens'
    )
    args[token_limit_key] = int(profile.max_tokens)
    return args


@dataclass(frozen=True)
class LLMChatResult:
    content: str
    usage: Any = None


class LLMChatRequestError(RuntimeError):
    """A normalized provider status error from Chat Completions."""

    def __init__(self, provider_error: Exception) -> None:
        self.provider_error = provider_error
        super().__init__(provider_error_message(provider_error))


class LLMChatRequester:
    """Issue one profile-backed OpenAI-compatible chat request.

    Prompt construction and retries stay with the owning Translator or OCR
    module; this boundary owns only transport and provider normalization.

    >>> LLMChatRequester().client is None
    True
    """

    dummy_api_key = 'dummy-key'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client: Any = None
        self.client_cache_key: Optional[
            Tuple[str, Optional[str], str]
        ] = None
        self.last_request_time = 0.0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.stop_event: Optional[threading.Event] = None

    def set_stop_event(
        self,
        stop_event: Optional[threading.Event],
    ) -> None:
        self.stop_event = stop_event

    def _wait(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self.stop_event is not None:
            if self.stop_event.wait(seconds):
                raise LLMRequestStopped()
            return
        time.sleep(seconds)

    @staticmethod
    def _openai_module() -> Any:
        import openai  # type: ignore

        return openai

    def _http_client(self, proxy: str) -> Any:
        import httpx  # type: ignore

        if not proxy:
            return httpx.Client()
        try:
            mounts = {
                'http://': httpx.HTTPTransport(proxy=proxy),
                'https://': httpx.HTTPTransport(proxy=proxy),
            }
            return httpx.Client(mounts=mounts)
        except Exception as error:
            self.logger.error(
                f"Failed to initialize proxy '{proxy}': {error}. "
                'Proceeding without proxy.'
            )
            return httpx.Client()

    @staticmethod
    def _api_key_for_profile(profile: LLMProfile) -> str:
        api_key = resolve_api_key(profile).strip()
        if profile.require_api_key and not api_key:
            raise LLMApiKeyRequiredError(profile.id, profile.name)
        return api_key

    def _client_api_key_for_profile(self, profile: LLMProfile) -> str:
        api_key = self._api_key_for_profile(profile)
        if not api_key:
            self.logger.debug(
                f'LLM profile "{profile.name or profile.id}" does not require '
                'an API key; using a dummy API key for OpenAI-compatible '
                'client initialization.'
            )
            return self.dummy_api_key
        return api_key

    def _initialize_client(self, profile: LLMProfile) -> Any:
        api_key = self._client_api_key_for_profile(profile)
        base_url = profile.base_url or None
        proxy = str(self.get_param_value('proxy') or '')
        cache_key = (api_key, base_url, proxy)
        if self.client is not None and self.client_cache_key == cache_key:
            return self.client

        openai = self._openai_module()
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self._http_client(proxy),
        )
        self.client_cache_key = cache_key
        return self.client

    def _respect_delay(self) -> None:
        current_time = time.time()
        rpm = self.get_param_value('max requests per minute')
        delay = self.get_param_value('delay')
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f'Global RPM limit ({rpm}) reached. Waiting '
                        f'{wait_time:.2f} seconds.'
                    )
                    self._wait(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            self._wait(delay - time_since_last_request)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    @staticmethod
    def _completion_content(completion: Any) -> str:
        for choice in getattr(completion, 'choices', ()):
            message = getattr(choice, 'message', None)
            content = getattr(message, 'content', None)
            if content is not None:
                return str(content)
            text = getattr(choice, 'text', None)
            if text is not None:
                return str(text)
        return ''

    def request_chat_completion(
        self,
        profile: LLMProfile,
        api_args: Dict[str, Any],
    ) -> LLMChatResult:
        """Perform one request; feature owners decide whether to retry it."""
        openai = self._openai_module()
        client = self._initialize_client(profile)
        self._respect_delay()
        try:
            completion = client.chat.completions.create(**api_args)
        except getattr(openai, 'AuthenticationError') as error:
            raise LLMApiKeyRequiredError(
                profile.id, profile.name
            ) from error
        except getattr(openai, 'APIStatusError') as error:
            raise LLMChatRequestError(error) from error

        return LLMChatResult(
            content=self._completion_content(completion),
            usage=getattr(completion, 'usage', None),
        )
