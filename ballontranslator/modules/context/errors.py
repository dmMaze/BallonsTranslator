"""Provider-agnostic context-window error classification."""

import re


class ContextLengthError(RuntimeError):
    """A provider rejected a request because its input context is too long."""


def provider_error_message(error) -> str:
    response = getattr(error, 'response', None)
    if response is not None:
        try:
            data = response.json()
            if isinstance(data, dict):
                nested = data.get('error')
                if isinstance(nested, dict) and nested.get('message'):
                    return str(nested['message'])
                if data.get('message'):
                    return str(data['message'])
        except Exception:
            pass
        text = getattr(response, 'text', '')
        if text:
            return str(text)
    return str(error)


def provider_error_code(error) -> str:
    values = [getattr(error, 'code', '')]
    body = getattr(error, 'body', None)
    if isinstance(body, dict):
        values.extend((body.get('code', ''), body.get('type', '')))
        nested = body.get('error')
        if isinstance(nested, dict):
            values.extend((nested.get('code', ''), nested.get('type', '')))

    response = getattr(error, 'response', None)
    if response is not None:
        try:
            data = response.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            values.extend((data.get('code', ''), data.get('type', '')))
            nested = data.get('error')
            if isinstance(nested, dict):
                values.extend((nested.get('code', ''), nested.get('type', '')))
    return ' '.join(str(value).lower() for value in values if value)


def is_context_length_error(error) -> bool:
    """Recognize only provider errors that clearly describe oversized input.

    >>> is_context_length_error(RuntimeError('maximum context length exceeded'))
    True
    >>> is_context_length_error(RuntimeError('max_tokens is invalid'))
    False
    """

    response = getattr(error, 'response', None)
    status_code = getattr(error, 'status_code', None)
    if status_code is None and response is not None:
        status_code = getattr(response, 'status_code', None)
    if status_code is not None:
        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None
    if status_code is not None and status_code not in (400, 413, 422):
        return False

    normalized_code = re.sub(
        r'[^a-z0-9]+',
        '_',
        provider_error_code(error),
    ).strip('_')
    recognized_codes = (
        'context_length_exceeded',
        'context_window_exceeded',
        'input_too_long',
        'prompt_too_long',
        'too_many_input_tokens',
    )
    if any(code in normalized_code for code in recognized_codes):
        return True

    message = provider_error_message(error).lower()
    patterns = (
        r'\b(?:maximum|max)\s+context\s+(?:length|window)\b',
        r'\bcontext\s+(?:length|window)\b.{0,80}\b(?:exceed|overflow|too\s+long)',
        r'\b(?:exceed|overflow|too\s+long).{0,80}\bcontext\s+(?:length|window)\b',
        r'\b(?:maximum|max)\s+input\s+tokens?\b',
        r'\binput\s+tokens?\b.{0,80}\b(?:exceed|too\s+many)\b',
        r'\bprompt\s+(?:is\s+)?too\s+long\b',
    )
    return any(re.search(pattern, message) for pattern in patterns)
