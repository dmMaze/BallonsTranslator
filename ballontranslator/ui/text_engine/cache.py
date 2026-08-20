"""Small cache primitives shared by text layout and rendering."""

from __future__ import annotations

from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    TypeVar,
    ValuesView,
)


CacheKey = TypeVar('CacheKey', bound=Hashable)
CacheValue = TypeVar('CacheValue')


class KeyedLruCache(Generic[CacheKey, CacheValue]):
    """Cache explicit keys without retaining factory-only arguments.

    >>> cache = KeyedLruCache(2)
    >>> cache.get_or_create('a', str.upper, 'first')
    'FIRST'
    >>> cache.get_or_create('a', str.upper, 'ignored')
    'FIRST'
    >>> cache.get_or_create('b', str.upper, 'second')
    'SECOND'
    >>> cache.get_or_create('c', str.upper, 'third')
    'THIRD'
    >>> tuple(cache)
    ('b', 'c')
    """

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError('max_entries must be positive')
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[CacheKey, CacheValue] = OrderedDict()

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[CacheKey]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def values(self) -> ValuesView[CacheValue]:
        return self._entries.values()

    def get_or_create(
        self,
        key: CacheKey,
        factory: Callable[..., CacheValue],
        *factory_args: Any,
        **factory_kwargs: Any,
    ) -> CacheValue:
        """Return the keyed value, invoking but never storing miss arguments."""
        try:
            value = self._entries[key]
        except KeyError:
            value = factory(*factory_args, **factory_kwargs)
            self._entries[key] = value
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
        return value
