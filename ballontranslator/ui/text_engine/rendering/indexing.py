"""Qt UTF-16 and grapheme indexing used by transformed glyph layouts."""

from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from functools import lru_cache

from qtpy.QtCore import QTextBoundaryFinder


@lru_cache(maxsize=1024)
def _utf16_boundaries(text: str):
    """Return each Python character boundary in Qt's UTF-16 coordinates."""
    boundaries = [0]
    offset = 0
    for char in text:
        offset += 2 if ord(char) > 0xFFFF else 1
        boundaries.append(offset)
    return tuple(boundaries)


def _utf16_length(text: str) -> int:
    return _utf16_boundaries(text)[-1]


def _utf16_slice(text: str, start: int, length: int) -> str:
    """Slice without allowing a Qt offset inside a surrogate pair to split it."""
    boundaries = _utf16_boundaries(text)
    start = max(0, min(start, boundaries[-1]))
    end = max(start, min(start + length, boundaries[-1]))
    py_start = max(0, bisect_right(boundaries, start) - 1)
    py_end = min(len(text), bisect_left(boundaries, end))
    return text[py_start:py_end]


def _utf16_char_at(text: str, offset: int) -> str:
    if not text:
        return ''
    boundaries = _utf16_boundaries(text)
    offset = max(0, min(offset, boundaries[-1] - 1))
    return text[bisect_right(boundaries, offset) - 1]


_UTF16_ZWJ = '\u200d'.encode('utf-16-le')


def _iter_grapheme_ranges(text: str) -> Iterator[tuple[int, int]]:
    """Yield Qt UTF-16 ranges, merging pieces on either side of a joiner."""
    finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Grapheme, text)
    # Qt versions differ on ZWJ sequences. Look up neighboring UTF-16 units
    # only for strings containing a joiner. The final boundary always counts.
    utf16 = text.encode('utf-16-le', 'surrogatepass') if '\u200d' in text else b''
    start = 0
    while (end := finder.toNextBoundary()) != -1:
        if utf16:
            offset = end * 2
            if offset < len(utf16) and (
                utf16.startswith(_UTF16_ZWJ, offset - 2) or utf16.startswith(_UTF16_ZWJ, offset)
            ):
                continue
        yield start, end
        start = end


def _has_multiple_graphemes(text: str) -> bool:
    if len(text) < 2:
        return False
    ranges = _iter_grapheme_ranges(text)
    next(ranges, None)
    return next(ranges, None) is not None


@lru_cache(maxsize=1024)
def _grapheme_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return grapheme ranges in Qt UTF-16 coordinates.

    >>> _grapheme_ranges('A')
    ((0, 1),)
    """
    return tuple(_iter_grapheme_ranges(text))


def _grapheme_count(text: str) -> int:
    """Count Qt grapheme clusters for the vertical one-column layout."""
    return len(_grapheme_ranges(text))
