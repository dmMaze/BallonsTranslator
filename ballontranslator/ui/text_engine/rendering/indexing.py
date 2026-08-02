"""Qt UTF-16 and grapheme indexing used by transformed glyph layouts."""

from bisect import bisect_left, bisect_right
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


def _grapheme_count(text: str) -> int:
    """Count Qt grapheme clusters for the vertical one-column layout."""
    if not text:
        return 0
    finder = QTextBoundaryFinder(
        QTextBoundaryFinder.BoundaryType.Grapheme,
        text,
    )
    finder.toStart()
    previous = 0
    join_next = False
    count = 0
    while True:
        boundary = finder.toNextBoundary()
        if boundary == -1:
            break
        segment = _utf16_slice(text, previous, boundary - previous)
        if not join_next and not segment.startswith('\u200d'):
            count += 1
        join_next = segment.endswith('\u200d')
        previous = boundary
    return count
