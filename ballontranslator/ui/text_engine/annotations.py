"""Persistent rich-text annotations unsupported by Qt's HTML subset.

Qt remains the live editing model.  Annotation meaning is stored on character
formats, while a small versioned metadata record carries those properties
through the existing ``TextBlock.rich_text`` HTML boundary.
"""

from __future__ import annotations

from enum import IntEnum
from html import escape
from html.parser import HTMLParser
import json
from typing import Optional

from qtpy.QtCore import QByteArray, QMimeData
from qtpy.QtGui import (
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextDocumentFragment,
    QTextFormat,
)

from ballontranslator.utils.logger import logger as LOGGER


RICH_TEXT_METADATA_NAME = 'ballontranslator-rich-text'
RICH_TEXT_METADATA_VERSION = 1
RICH_TEXT_MIME_TYPE = 'application/x-ballonstranslator-rich-text'
MAX_RICH_TEXT_MIME_BYTES = 16 * 1024 * 1024


def _enum_value(value: object) -> int:
    return int(getattr(value, 'value', value))


class AnnotationProperty(IntEnum):
    """Stable ``QTextFormat.UserProperty`` IDs for inline annotations.

    >>> AnnotationProperty.EMPHASIS_STYLE != AnnotationProperty.EMPHASIS_POSITION
    True
    """

    # Keep this small range emphasis-only; later annotation kinds get their
    # own ranges so persisted formats never acquire colliding meanings.
    EMPHASIS_STYLE = _enum_value(QTextFormat.Property.UserProperty) + 1300
    EMPHASIS_POSITION = _enum_value(QTextFormat.Property.UserProperty) + 1301


EMPHASIS_STYLES = (
    'none',
    'filled dot',
    'open dot',
    'filled circle',
    'open circle',
    'filled double-circle',
    'open double-circle',
    'filled triangle',
    'open triangle',
    'filled sesame',
    'open sesame',
)
EMPHASIS_POSITIONS = (
    'over right',
    'under right',
    'over left',
    'under left',
)
DEFAULT_EMPHASIS_POSITION = 'over right'


class _RichTextMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.contents = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() != 'meta':
            return
        attributes = {str(name).lower(): value for name, value in attrs}
        if attributes.get('name') == RICH_TEXT_METADATA_NAME:
            self.contents.append(attributes.get('content', ''))

    handle_startendtag = handle_starttag


def _read_metadata(html: str) -> Optional[dict]:
    parser = _RichTextMetadataParser()
    try:
        parser.feed(html)
        parser.close()
    except (ValueError, TypeError) as error:
        LOGGER.warning('Unable to parse rich-text annotation metadata: %s', error)
        return None
    if not parser.contents:
        return None
    if len(parser.contents) > 1:
        LOGGER.warning('Ignoring duplicate rich-text annotation metadata')
    try:
        payload = json.loads(parser.contents[0])
    except (json.JSONDecodeError, TypeError) as error:
        LOGGER.warning('Ignoring malformed rich-text annotation metadata: %s', error)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning('Ignoring non-object rich-text annotation metadata')
        return None
    if payload.get('version') != RICH_TEXT_METADATA_VERSION:
        LOGGER.warning(
            'Ignoring unsupported rich-text annotation metadata version: %r',
            payload.get('version'),
        )
        return None
    return payload


def _valid_emphasis_entry(
    entry: object,
    document_end: int,
) -> Optional[tuple[int, int, str, str]]:
    if not isinstance(entry, dict) or entry.get('kind') != 'emphasis':
        return None
    start = entry.get('start')
    length = entry.get('length')
    style = entry.get('style')
    position = entry.get('position', DEFAULT_EMPHASIS_POSITION)
    if (
        type(start) is not int
        or type(length) is not int
        or start < 0
        or length <= 0
        or start + length > document_end
        or style not in EMPHASIS_STYLES
        or style == 'none'
        or position not in EMPHASIS_POSITIONS
    ):
        return None
    return start, length, style, position


def _restore_annotations(document: QTextDocument, payload: dict) -> None:
    entries = payload.get('annotations', [])
    if not isinstance(entries, list):
        LOGGER.warning('Ignoring invalid rich-text annotation list')
        return
    document_end = max(0, document.characterCount() - 1)
    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    try:
        for entry in entries:
            values = _valid_emphasis_entry(entry, document_end)
            if values is None:
                LOGGER.warning('Ignoring invalid rich-text annotation: %r', entry)
                continue
            start, length, style, position = values
            modifier = QTextCharFormat()
            modifier.setProperty(AnnotationProperty.EMPHASIS_STYLE, style)
            modifier.setProperty(AnnotationProperty.EMPHASIS_POSITION, position)
            cursor.setPosition(start)
            cursor.setPosition(start + length, QTextCursor.MoveMode.KeepAnchor)
            cursor.mergeCharFormat(modifier)
    finally:
        cursor.endEditBlock()


def load_rich_text_html(document: QTextDocument, html: str) -> None:
    """Load old Qt HTML or extended annotation HTML into ``document``.

    >>> isinstance(RICH_TEXT_METADATA_VERSION, int)
    True
    """
    payload = _read_metadata(html)
    undo_enabled = document.isUndoRedoEnabled()
    document.setUndoRedoEnabled(False)
    try:
        document.setHtml(html)
        if payload is not None:
            _restore_annotations(document, payload)
    finally:
        document.setUndoRedoEnabled(undo_enabled)


def _emphasis_entries(document: QTextDocument) -> list[dict]:
    entries = []
    block = document.firstBlock()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.length() > 0:
                char_format = fragment.charFormat()
                style = str(
                    char_format.property(AnnotationProperty.EMPHASIS_STYLE) or ''
                )
                position = str(
                    char_format.property(AnnotationProperty.EMPHASIS_POSITION)
                    or DEFAULT_EMPHASIS_POSITION
                )
                if (
                    style in EMPHASIS_STYLES
                    and style != 'none'
                    and position in EMPHASIS_POSITIONS
                ):
                    entry = {
                        'kind': 'emphasis',
                        'start': fragment.position(),
                        'length': fragment.length(),
                        'style': style,
                        'position': position,
                    }
                    if (
                        entries
                        and entries[-1]['start'] + entries[-1]['length']
                        == entry['start']
                        and entries[-1]['style'] == style
                        and entries[-1]['position'] == position
                    ):
                        entries[-1]['length'] += entry['length']
                    else:
                        entries.append(entry)
            iterator += 1
        block = block.next()
    return entries


def to_rich_text_html(
    document: QTextDocument,
    html: Optional[str] = None,
) -> str:
    """Add versioned annotation metadata to Qt's existing HTML output."""
    if html is None:
        html = document.toHtml()
    entries = _emphasis_entries(document)
    if not entries:
        return html
    payload = json.dumps(
        {
            'version': RICH_TEXT_METADATA_VERSION,
            'annotations': entries,
        },
        ensure_ascii=False,
        separators=(',', ':'),
    )
    metadata = (
        f'<meta name="{RICH_TEXT_METADATA_NAME}" '
        f'content="{escape(payload, quote=True)}" />'
    )
    head_end = html.lower().find('</head>')
    if head_end < 0:
        return metadata + html
    return html[:head_end] + metadata + html[head_end:]


def emphasis_values(char_format: QTextCharFormat) -> tuple[str, str]:
    """Return canonical emphasis values from one character format."""
    style = str(char_format.property(AnnotationProperty.EMPHASIS_STYLE) or 'none')
    position = str(
        char_format.property(AnnotationProperty.EMPHASIS_POSITION)
        or DEFAULT_EMPHASIS_POSITION
    )
    if style not in EMPHASIS_STYLES:
        style = 'none'
    if position not in EMPHASIS_POSITIONS:
        position = DEFAULT_EMPHASIS_POSITION
    return style, position


def apply_emphasis(cursor: QTextCursor, style: str, position: str) -> None:
    """Merge emphasis into a selection or the cursor's insertion format."""
    if style not in EMPHASIS_STYLES:
        raise ValueError(f'unsupported emphasis style: {style!r}')
    if position not in EMPHASIS_POSITIONS:
        raise ValueError(f'unsupported emphasis position: {position!r}')
    modifier = QTextCharFormat()
    modifier.setProperty(AnnotationProperty.EMPHASIS_STYLE, style)
    modifier.setProperty(AnnotationProperty.EMPHASIS_POSITION, position)
    cursor.mergeCharFormat(modifier)


def create_rich_text_mime(cursor: QTextCursor) -> QMimeData:
    """Create interoperable clipboard data plus exact annotation metadata."""
    mime = QMimeData()
    if not cursor.hasSelection():
        return mime
    document = QTextDocument()
    document.setUndoRedoEnabled(False)
    target = QTextCursor(document)
    target.insertFragment(QTextDocumentFragment(cursor))
    extended_html = to_rich_text_html(document)
    mime.setText(document.toPlainText())
    mime.setHtml(document.toHtml())
    mime.setData(
        RICH_TEXT_MIME_TYPE,
        QByteArray(extended_html.encode('utf-8')),
    )
    return mime


def insert_rich_text_mime(cursor: QTextCursor, mime: QMimeData) -> bool:
    """Insert the custom rich-text representation when it is valid."""
    if not mime.hasFormat(RICH_TEXT_MIME_TYPE):
        return False
    encoded = bytes(mime.data(RICH_TEXT_MIME_TYPE))
    if not encoded or len(encoded) > MAX_RICH_TEXT_MIME_BYTES:
        LOGGER.warning('Ignoring invalid rich-text clipboard payload size')
        return False
    try:
        html = encoded.decode('utf-8')
    except UnicodeDecodeError:
        LOGGER.warning('Ignoring non-UTF-8 rich-text clipboard payload')
        return False
    document = QTextDocument()
    load_rich_text_html(document, html)
    cursor.insertFragment(QTextDocumentFragment(document))
    return True
