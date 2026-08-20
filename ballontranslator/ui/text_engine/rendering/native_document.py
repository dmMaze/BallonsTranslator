"""Bounded native QTextDocument glyph sources for layout annotations."""

from __future__ import annotations

from typing import NamedTuple, Tuple

from qtpy.QtCore import QByteArray, QDataStream, QIODevice, QRectF, Qt
from qtpy.QtGui import (
    QBrush,
    QPainter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTransform,
)

from ..cache import KeyedLruCache


NATIVE_DOCUMENT_CACHE_MAX_ENTRIES = 128


class NativeTextDocument(NamedTuple):
    document: QTextDocument
    glyph_bounds: QRectF
    ink_bounds: QRectF


NATIVE_DOCUMENT_CACHE: KeyedLruCache[
    Tuple[str, bytes], NativeTextDocument
] = KeyedLruCache(NATIVE_DOCUMENT_CACHE_MAX_ENTRIES)


def _format_cache_key(char_format: QTextCharFormat) -> bytes:
    data = QByteArray()
    stream = QDataStream(data, QIODevice.OpenModeFlag.WriteOnly)
    stream << char_format
    return bytes(data)


def _glyph_bounds(document: QTextDocument) -> QRectF:
    line = document.firstBlock().layout().lineAt(0)
    bounds = QRectF()
    for run in line.glyphRuns():
        raw_font = run.rawFont()
        for glyph_index, position in zip(
            run.glyphIndexes(), run.positions()
        ):
            candidate = raw_font.boundingRect(glyph_index).translated(position)
            bounds = (
                QRectF(candidate)
                if bounds.isNull()
                else bounds.united(candidate)
            )
    return bounds


def _build_native_text_document(
    text: str,
    char_format: QTextCharFormat,
) -> NativeTextDocument:
    document = QTextDocument()
    document.setUndoRedoEnabled(False)
    document.setDocumentMargin(0.0)
    QTextCursor(document).insertText(text, char_format)
    document.adjustSize()
    glyph_bounds = _glyph_bounds(document)
    ink_bounds = QRectF(glyph_bounds)
    outline = char_format.textOutline()
    if outline.style() != Qt.PenStyle.NoPen and outline.widthF() > 0.0:
        radius = outline.widthF() / 2.0
        ink_bounds.adjust(-radius, -radius, radius, radius)
    return NativeTextDocument(
        document,
        glyph_bounds,
        ink_bounds,
    )


def native_text_document(
    text: str,
    char_format: QTextCharFormat,
) -> NativeTextDocument:
    """Return one cached zero-margin native document for exact paint inputs.

    >>> NATIVE_DOCUMENT_CACHE_MAX_ENTRIES > 0
    True
    """
    key = (text, _format_cache_key(char_format))
    return NATIVE_DOCUMENT_CACHE.get_or_create(
        key,
        _build_native_text_document,
        text,
        char_format,
    )


def draw_native_text_document(
    painter: QPainter,
    source: NativeTextDocument,
    transform: QTransform,
) -> None:
    """Draw one cached document while keeping gradients item-local.

    >>> callable(draw_native_text_document)
    True
    """
    document = source.document
    cursor = QTextCursor(document)
    cursor.setPosition(0)
    cursor.setPosition(
        max(0, document.characterCount() - 1),
        QTextCursor.MoveMode.KeepAnchor,
    )
    char_format = cursor.charFormat()
    foreground = char_format.foreground()
    restore_foreground = foreground.gradient() is not None
    if restore_foreground:
        inverse, invertible = transform.inverted()
        if invertible:
            compensated = QBrush(foreground)
            compensated.setTransform(foreground.transform() * inverse)
            override = QTextCharFormat()
            override.setForeground(compensated)
            cursor.mergeCharFormat(override)
        else:
            restore_foreground = False
    painter.save()
    try:
        painter.setTransform(transform, True)
        document.drawContents(painter)
    finally:
        painter.restore()
        if restore_foreground:
            restored = QTextCharFormat()
            restored.setForeground(foreground)
            cursor.mergeCharFormat(restored)
