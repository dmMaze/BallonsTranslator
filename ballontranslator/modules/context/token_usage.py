"""Prompt-token estimation and provider-usage formatting for LLM requests."""

from collections.abc import Mapping
from functools import lru_cache
from typing import Dict, List


MESSAGE_TOKEN_OVERHEAD = 4


@lru_cache(maxsize=64)
def _cached_token_encoding_for_model(model: str):
    import tiktoken  # type: ignore

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown model names are stable and safe to cache as fallbacks.
        return None


def _token_encoding_for_model(model: str):
    try:
        return _cached_token_encoding_for_model(model)
    except Exception:
        # Import/initialization failures may be transient; do not cache them.
        return None


def fallback_token_count(text: str) -> int:
    """Estimate tokens deterministically without assuming a model encoding.

    >>> fallback_token_count('abcdefgh你')
    3
    """
    total = 0
    ascii_run = 0
    for character in text:
        if ord(character) < 128:
            ascii_run += 1
            continue
        if ascii_run:
            total += (ascii_run + 3) // 4
            ascii_run = 0
        total += 1
    if ascii_run:
        total += (ascii_run + 3) // 4
    return total


def messages_token_count(messages: List[Dict], model: str) -> int:
    """Count message content with a deterministic fallback for unknown models.

    >>> messages_token_count([{'content': 'abcdefgh你'}], 'unknown-model')
    7
    """
    encoding = _token_encoding_for_model(model)
    total = 0
    for message in messages:
        content = str(message.get('content', ''))
        if encoding is None:
            content_tokens = fallback_token_count(content)
        else:
            try:
                content_tokens = len(encoding.encode(content))
            except Exception:
                content_tokens = fallback_token_count(content)
        total += MESSAGE_TOKEN_OVERHEAD + content_tokens
    return total


def _usage_member(container, *names):
    if container is None:
        return None
    for name in names:
        try:
            value = container.get(name) if isinstance(container, Mapping) else getattr(container, name, None)
        except Exception:
            continue
        if value is not None:
            return value
    return None


def _usage_count(container, *names):
    value = _usage_member(container, *names)
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except Exception:
        return None
    return count if count >= 0 else None


def format_token_usage(usage) -> str:
    """Format available provider token fields without requiring a fixed schema.

    >>> format_token_usage({
    ...     'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12,
    ...     'prompt_cache_hit_tokens': 8,
    ... })
    'prompt=10, completion=2, total=12, cache_hit=8'
    """
    prompt = _usage_count(usage, 'prompt_tokens', 'input_tokens')
    completion = _usage_count(usage, 'completion_tokens', 'output_tokens')
    total = _usage_count(usage, 'total_tokens')
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    prompt_details = _usage_member(
        usage,
        'prompt_tokens_details',
        'input_tokens_details',
    )
    cache_hit = _usage_count(
        usage,
        'prompt_cache_hit_tokens',
        'cached_tokens',
        'cache_read_tokens',
        'cache_read_input_tokens',
    )
    if cache_hit is None:
        cache_hit = _usage_count(
            prompt_details,
            'cached_tokens',
            'cache_read_tokens',
            'cache_read_input_tokens',
        )

    cache_miss = _usage_count(
        usage,
        'prompt_cache_miss_tokens',
        'cache_miss_tokens',
    )
    if cache_miss is None:
        cache_miss = _usage_count(prompt_details, 'cache_miss_tokens')

    cache_write = _usage_count(
        usage,
        'prompt_cache_write_tokens',
        'cache_write_tokens',
        'cache_creation_input_tokens',
    )
    if cache_write is None:
        cache_write = _usage_count(
            prompt_details,
            'cache_write_tokens',
            'cache_creation_input_tokens',
        )

    fields = (
        ('prompt', prompt),
        ('completion', completion),
        ('total', total),
        ('cache_hit', cache_hit),
        ('cache_miss', cache_miss),
        ('cache_write', cache_write),
    )
    return ', '.join(
        f'{name}={value}'
        for name, value in fields
        if value is not None
    )


def format_completion_token_usage(completion) -> str:
    return format_token_usage(_usage_member(completion, 'usage'))
