"""Public OpenAI Responses adapter using the shared persistent WebSocket transport."""

from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace
from typing import Dict, List, Optional, TYPE_CHECKING

from .exceptions import LLMAuthenticationError, LLMOutputLimitError, LLMRequestStopped, LLMUserActionRequiredError
from .responses_ws import ResponsesWebSocket
from ballontranslator.utils.llm_profiles import LLMProfile
from ballontranslator.utils.logger import logger as LOGGER

if TYPE_CHECKING:
    from .llm_chat import LLMChatResult


def responses_payload(api_args: Dict) -> Optional[Dict]:
    """Translate the app's chat arguments without dropping unsupported controls.

    Returning None keeps Chat Completions for requests that cannot be represented.

    >>> responses_payload({'model': 'gpt-4.1', 'messages': [{'role': 'user', 'content': 'Hi'}]})['input'][0]['content']
    [{'type': 'input_text', 'text': 'Hi'}]
    """
    supported = {'model', 'messages', 'temperature', 'top_p', 'max_tokens', 'max_completion_tokens',
                 'reasoning_effort', 'response_format', 'extra_body'}
    if set(api_args) - supported or not api_args.get('messages'):
        return None
    extra = api_args.get('extra_body', {})
    if set(extra) - {'prompt_cache_options'}:
        return None
    inputs = []
    for message in api_args['messages']:
        role = message['role']
        if role not in ('system', 'developer', 'user', 'assistant'):
            return None
        content = message.get('content', '')
        parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
        converted = []
        for part in parts:
            if part['type'] == 'text':
                converted.append({**part, 'type': 'output_text' if role == 'assistant' else 'input_text'})
                if 'prompt_cache_breakpoint' in part:
                    converted[-1]['prompt_cache_breakpoint'] = dict(part['prompt_cache_breakpoint'])
            elif part['type'] == 'image_url' and role == 'user':
                converted.append({'type': 'input_image', 'image_url': part['image_url']['url'],
                                  'detail': part['image_url'].get('detail', 'auto')})
            else:
                return None
        inputs.append({'role': role, 'content': converted})
    payload = {'model': api_args['model'], 'input': inputs, 'store': False,
               'include': ['reasoning.encrypted_content'], **copy.deepcopy(extra)}
    for name in ('temperature', 'top_p'):
        if name in api_args:
            payload[name] = api_args[name]
    limit = api_args.get('max_completion_tokens', api_args.get('max_tokens'))
    if limit is not None:
        payload['max_output_tokens'] = limit
    if 'reasoning_effort' in api_args:
        payload['reasoning'] = {'effort': api_args['reasoning_effort']}
    response_format = api_args.get('response_format')
    if response_format:
        kind = response_format['type']
        if kind not in ('json_schema', 'json_object', 'text'):
            return None
        payload['text'] = {'format': ({'type': kind, **response_format['json_schema']}
                                     if kind == 'json_schema' else {'type': kind})}
    return payload


class OpenAIResponsesSession(ResponsesWebSocket):
    """Own one API profile's connection and exact chat-to-Responses replay prefix.

    The SDK client supplies API authentication only; this adapter never reads the
    Codex account, subscription cookies, or release identity.
    """

    def __init__(self, url: str, cache_key: str, proxy: str, headers: Dict[str, str]) -> None:
        super().__init__(url, cache_key, proxy)
        self.headers = headers
        self._chat_prefix: List[Dict] = []

    def _dispose(self) -> None:
        self._chat_prefix = []
        super()._dispose()

    def request_chat(self, api_args: Dict, profile: LLMProfile,
                     stop_event: Optional[threading.Event]) -> Optional[LLMChatResult]:
        if self.closed or (stop_event is not None and stop_event.is_set()):
            raise LLMRequestStopped()
        if self._http_only:
            return None
        payload = responses_payload(api_args)
        if payload is None:
            self.close()
            return None
        from .llm_chat import LLMChatResult, LLMChatRequestError
        import httpx
        import openai

        def reject(body: Dict, status: Optional[int] = None) -> None:
            error = body.get('error') or body
            code = error.get('code') if isinstance(error, dict) else None
            if status == 401 or code in ('invalid_api_key', 'token_expired'):
                raise LLMAuthenticationError(profile.id, profile.name)
            if not isinstance(status, int):
                status = {'rate_limit_exceeded': 429, 'server_error': 500,
                          'permission_denied': 403}.get(code, 400)
            response = httpx.Response(status, request=httpx.Request('POST', self.url))
            message = error.get('message', 'OpenAI request failed.') if isinstance(error, dict) else str(error)
            raise LLMChatRequestError(openai.APIStatusError(message, response=response, body=body))

        def complete(event: Dict, items: List[Dict]) -> Optional[Dict]:
            kind = event.get('type')
            if kind == 'response.done':
                status = event['response'].get('status')
                if status in ('completed', 'failed', 'incomplete'):
                    kind = 'response.' + status
            if kind == 'response.output_item.done':
                items.append(event['item'])
            elif kind in ('response.failed', 'error'):
                reject(event.get('response', event), event.get('status', event.get('status_code')))
            elif kind == 'response.incomplete':
                response = event['response']
                if (response.get('incomplete_details') or {}).get('reason') == 'max_output_tokens':
                    raise LLMOutputLimitError(profile.id, profile.name, profile.max_tokens, profile.thinking_level)
                raise LLMUserActionRequiredError('OpenAI did not complete the response. Review the input and retry.')
            elif kind in ('response.completed', 'response.done'):
                response = event['response']
                if response.get('status') != 'completed':
                    reject(response)
                if not response.get('output'):
                    response['output'] = items
                return response
            return None

        async def headers() -> Dict[str, str]:
            return self.headers

        async def request() -> Optional[LLMChatResult]:
            payload['prompt_cache_key'] = self.cache_key
            original = payload['input']
            wire = payload
            # Restore provider output only when the caller supplies exactly the
            # previous conversation and answer. Edits/eviction use full input.
            if (self._previous is not None and self._chat_prefix
                    and original[:len(self._chat_prefix)] == self._chat_prefix):
                previous, result = self._previous
                wire = {**payload, 'input': previous['input'] + result['output'] + original[len(self._chat_prefix):]}
            result = await self.request(wire, headers, complete)
            if result is None:
                self._chat_prefix = []
                return None
            messages = [item for item in result.get('output', [])
                        if item.get('type') == 'message' and item.get('role') == 'assistant'
                        and item.get('phase') in (None, 'final_answer')]
            messages = [item for item in messages if item.get('phase') == 'final_answer'] or messages
            if not messages:
                raise RuntimeError('OpenAI response contained no assistant message.')
            parts = [part for item in messages for part in item.get('content', [])]
            if any(part.get('type') == 'refusal' for part in parts):
                raise LLMUserActionRequiredError('OpenAI declined the request. Review the input before retrying.')
            text = ''.join(part['text'] for part in parts if part.get('type') == 'output_text')
            # Conversion already owns these containers; retain that snapshot.
            self._chat_prefix = original + [
                {'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}]
            raw_usage = result.get('usage')
            usage = None if not isinstance(raw_usage, dict) else SimpleNamespace(
                prompt_tokens=raw_usage.get('input_tokens', 0), completion_tokens=raw_usage.get('output_tokens', 0),
                total_tokens=raw_usage.get('total_tokens', 0),
                prompt_tokens_details=raw_usage.get('input_tokens_details'),
                completion_tokens_details=raw_usage.get('output_tokens_details'),
            )
            return LLMChatResult(text, usage=usage, finish_reason='stop',
                                 prompt_cache_diagnostics=result.get('prompt_cache_diagnostics'))

        try:
            return self.run(request(), stop_event)
        except (LLMRequestStopped, LLMUserActionRequiredError, LLMChatRequestError):
            self._chat_prefix = []
            self._previous = None
            raise
        except Exception as error:
            self._chat_prefix = []
            self._previous = None
            from websockets.exceptions import InvalidStatus
            if isinstance(error, InvalidStatus):
                try:
                    body = json.loads(error.response.body)
                except (ValueError, TypeError):
                    body = None
                reject(body if isinstance(body, dict) else {'error': {
                    'message': f'OpenAI rejected the WebSocket connection (HTTP {error.response.status_code}).'
                }}, error.response.status_code)
            LOGGER.debug('OpenAI Responses failure: exception=%s', type(error).__name__)
            raise RuntimeError(f'OpenAI Responses WebSocket request failed ({type(error).__name__}).') from None
