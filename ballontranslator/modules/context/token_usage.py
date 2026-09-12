"""Prompt-token estimation and provider-usage formatting for LLM requests."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Dict, List, Optional, Sequence


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


def token_usage_counts(usage) -> Dict[str, int]:
    """Normalize available provider counts, keeping missing fields absent.

    >>> token_usage_counts({'input_tokens': 10, 'output_tokens': 2})
    {'prompt': 10, 'completion': 2, 'total': 12}
    """
    prompt = _usage_count(usage, 'prompt_tokens', 'input_tokens')
    completion = _usage_count(usage, 'completion_tokens', 'output_tokens')
    total = _usage_count(usage, 'total_tokens')
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    completion_details = _usage_member(
        usage,
        'completion_tokens_details',
        'output_tokens_details',
    )
    reasoning = _usage_count(completion_details, 'reasoning_tokens')

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
        ('reasoning', reasoning),
        ('total', total),
        ('cache_hit', cache_hit),
        ('cache_miss', cache_miss),
        ('cache_write', cache_write),
    )
    return {name: value for name, value in fields if value is not None}


def format_token_usage(usage) -> str:
    return ', '.join(f'{name}={value}' for name, value in token_usage_counts(usage).items())


def format_completion_token_usage(completion) -> str:
    return format_token_usage(_usage_member(completion, 'usage'))


# USD per million tokens, verified 2026-09-11; Standard API equivalents, not
# subscription charges. https://developers.openai.com/api/docs/pricing
# GPT-5.5: https://developers.openai.com/api/docs/models/gpt-5.5
OPENAI_STANDARD_PRICES = {
    # Input, cached input, output. Cache writes for 5.6/Astra cost 1.25x input.
    'gpt-6-astra': ('10', '1', '50'),
    'gpt-5.6-sol': ('4', '0.4', '20'),
    'gpt-5.6-terra': ('2', '0.2', '12'),
    'gpt-5.6-luna': ('0.2', '0.02', '1.2'),
    'gpt-5.5': ('5', '0.5', '30'),
}


def estimated_token_cost(model: str, counts: Dict[str, int]) -> Optional[Decimal]:
    """Estimate Standard API value; reasoning/cache counts are already in totals.

    >>> estimated_token_cost('gpt-5.6-sol', {'prompt': 1000, 'completion': 100, 'cache_hit': 500})
    Decimal('0.0042')
    """
    prices = OPENAI_STANDARD_PRICES.get(model)
    prompt, output = counts.get('prompt'), counts.get('completion')
    cached, written = counts.get('cache_hit', 0), counts.get('cache_write', 0)
    if prices is None or prompt is None or output is None or cached + written > prompt:
        return None
    if model == 'gpt-5.5' and written:
        return None  # No verified cache-write price for this model.
    input_rate, cached_rate, output_rate = map(Decimal, prices)
    # Apply long-context rates per response, never to the whole batch total.
    if prompt > 272_000:
        input_rate *= 2
        cached_rate *= 2
        output_rate *= Decimal('1.5')
    return ((prompt - cached - written) * input_rate + cached * cached_rate
            + written * input_rate * Decimal('1.25') + output * output_rate) / 1_000_000


@dataclass
class LLMUsageTotals:
    """One requester's run counters; each module has a single worker writer.

    >>> totals = LLMUsageTotals(requests=1)
    >>> totals.add('gpt-5.6-sol', {'input_tokens': 1000, 'output_tokens': 100})
    >>> totals.total_tokens
    1100
    """

    requests: int = 0
    usage_reports: int = 0
    priced_requests: int = 0
    total_tokens: int = 0
    cost_usd: Decimal = Decimal(0)

    def add(self, model: str, usage) -> None:
        counts = token_usage_counts(usage)
        if 'total' in counts:
            self.usage_reports += 1
            self.total_tokens += counts['total']
        cost = estimated_token_cost(model, counts)
        if cost is not None:
            self.priced_requests += 1
            self.cost_usd += cost


def format_run_token_usage(totals: Sequence[LLMUsageTotals]) -> str:
    """Combine finished workers without inventing zero usage for missing reports.

    >>> 'total_tokens=12' in format_run_token_usage([LLMUsageTotals(total_tokens=12)])
    True
    """
    requests = sum(item.requests for item in totals)
    reported = sum(item.usage_reports for item in totals)
    priced = sum(item.priced_requests for item in totals)
    tokens = sum(item.total_tokens for item in totals)
    cost = sum((item.cost_usd for item in totals), Decimal(0))
    cost_text = f'{cost:.6f}' if priced == requests else 'unavailable'
    return (f'requests={requests}, total_tokens={tokens}, '
            f'missing_usage_requests={requests - reported}, '
            f'estimated_cost_usd={cost_text}, priced_subtotal_usd={cost:.6f}, '
            f'unpriced_requests={requests - priced}, '
            'price_basis=OpenAI Standard API equivalent (not a bill), rates_date=2026-09-11')
