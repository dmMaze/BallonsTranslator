"""Rich-text extensions that Qt cannot export by itself.

Qt remains the live editing model. Character-format properties carry live
meaning; one semantic inline HTML boundary stores emphasis, tate-chu-yoko,
letter spacing, and their exact application-owned values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from html import escape, unescape
from html.parser import HTMLParser
import math
from typing import Optional
from uuid import uuid4

from qtpy.QtCore import QByteArray, QMimeData
from qtpy.QtGui import (
    QTextBlock,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextDocumentFragment,
    QTextFormat,
    QFont,
)

from ballontranslator.utils.logger import logger as LOGGER


RICH_TEXT_MIME_TYPE = 'application/x-ballonstranslator-rich-text'
MAX_RICH_TEXT_MIME_BYTES = 16 * 1024 * 1024
LETTER_SPACING_ATTRIBUTE = 'data-btrans-letter-spacing'
TEXT_COMBINE_ID_ATTRIBUTE = 'data-btrans-text-combine-id'


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

    # Vertical-language features use a separate range from emphasis and the
    # future ruby range. The ID preserves adjacent combined-run boundaries.
    TEXT_COMBINE_UPRIGHT = _enum_value(QTextFormat.Property.UserProperty) + 1340
    TEXT_COMBINE_ID = _enum_value(QTextFormat.Property.UserProperty) + 1341

    # Qt does not round-trip QFont letter spacing through QTextDocument HTML.
    LETTER_SPACING = _enum_value(QTextFormat.Property.UserProperty) + 1380


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
TEXT_COMBINE_NONE = 'none'
TEXT_COMBINE_ALL = 'all'
MAX_ANNOTATION_ID_LENGTH = 128
MIN_LETTER_SPACING = 0.0
MAX_LETTER_SPACING = 10.0


@dataclass(frozen=True)
class _InlineExtension:
    """Formatting Qt drops when serializing HTML.

    >>> _InlineExtension().is_empty()
    True
    """

    emphasis_style: str = 'none'
    emphasis_position: str = DEFAULT_EMPHASIS_POSITION
    text_combine_id: str = ''
    letter_spacing: Optional[float] = None

    def is_empty(self) -> bool:
        return (
            self.emphasis_style == 'none'
            and not self.text_combine_id
            and self.letter_spacing is None
        )


def _canonical_letter_spacing(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if (
        not math.isfinite(value)
        or value < MIN_LETTER_SPACING
        or value > MAX_LETTER_SPACING
    ):
        return None
    return value


def _parse_letter_spacing_attribute(value: object) -> Optional[float]:
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return _canonical_letter_spacing(parsed)


def _style_declarations(value: object) -> dict[str, str]:
    if not isinstance(value, str):
        return {}
    declarations = {}
    for declaration in value.split(';'):
        name, separator, style_value = declaration.partition(':')
        if separator:
            declarations[name.strip().lower()] = style_value.strip().lower()
    return declarations


def _span_extension(
    inherited: _InlineExtension,
    attrs: list,
) -> _InlineExtension:
    attributes = {str(name).lower(): value for name, value in attrs}
    styles = _style_declarations(attributes.get('style'))
    extension = inherited

    if 'text-emphasis-style' in styles:
        style = styles['text-emphasis-style']
        if style not in EMPHASIS_STYLES:
            LOGGER.warning('Ignoring invalid text emphasis style: %r', style)
            style = 'none'
        position = styles.get(
            'text-emphasis-position',
            inherited.emphasis_position,
        )
        if position not in EMPHASIS_POSITIONS:
            LOGGER.warning(
                'Ignoring invalid text emphasis position: %r', position
            )
            position = DEFAULT_EMPHASIS_POSITION
        extension = replace(
            extension,
            emphasis_style=style,
            emphasis_position=position,
        )

    if 'text-combine-upright' in styles:
        value = styles['text-combine-upright']
        if value == TEXT_COMBINE_ALL:
            group_id = attributes.get(TEXT_COMBINE_ID_ATTRIBUTE)
            if not group_id:
                group_id = uuid4().hex
            elif len(group_id) > MAX_ANNOTATION_ID_LENGTH:
                LOGGER.warning('Ignoring overlong tate-chu-yoko group ID')
                group_id = uuid4().hex
            extension = replace(extension, text_combine_id=group_id)
        elif value == TEXT_COMBINE_NONE:
            extension = replace(extension, text_combine_id='')
        else:
            LOGGER.warning('Ignoring invalid text-combine-upright: %r', value)
            extension = replace(extension, text_combine_id='')

    if LETTER_SPACING_ATTRIBUTE in attributes:
        spacing = _parse_letter_spacing_attribute(
            attributes[LETTER_SPACING_ATTRIBUTE]
        )
        if spacing is None:
            LOGGER.warning(
                'Ignoring invalid inline letter spacing: %r',
                attributes[LETTER_SPACING_ATTRIBUTE],
            )
        extension = replace(extension, letter_spacing=spacing)
    return extension


def _letter_spacing_modifier(
    value: float,
    vertical: bool,
) -> QTextCharFormat:
    modifier = QTextCharFormat()
    modifier.setProperty(AnnotationProperty.LETTER_SPACING, value)
    modifier.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
    modifier.setFontLetterSpacing(100.0 if vertical else value * 100.0)
    return modifier


def _apply_document_letter_spacing(
    document: QTextDocument,
    value: float,
    vertical: bool,
) -> None:
    cursor = QTextCursor(document)
    cursor.select(QTextCursor.SelectionType.Document)
    modifier = _letter_spacing_modifier(value, vertical)
    cursor.mergeCharFormat(modifier)
    cursor.mergeBlockCharFormat(modifier)


def _utf16_length(text: str) -> int:
    return len(text.encode('utf-16-le')) // 2


def _document_blocks(document: QTextDocument) -> tuple[QTextBlock, ...]:
    blocks = []
    block = document.firstBlock()
    while block.isValid():
        blocks.append(block)
        block = block.next()
    return tuple(blocks)


class _InlineExtensionRangeParser(HTMLParser):
    """Read inline extensions using loaded Qt block positions.

    >>> _parse_letter_spacing_attribute('1.15')
    1.15
    """

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks = _document_blocks(document)
        self.block_index = -1
        self.block_offset = 0
        self.in_block = False
        self.extension = _InlineExtension()
        self.extension_stack: list[_InlineExtension] = []
        self.ranges: list[tuple[int, int, _InlineExtension]] = []

    def _start_block(self) -> None:
        self.block_index += 1
        self.block_offset = 0
        self.in_block = True

    def _current_block(self) -> Optional[QTextBlock]:
        if self.block_index < 0 and self.blocks:
            self._start_block()
        if 0 <= self.block_index < len(self.blocks):
            return self.blocks[self.block_index]
        return None

    def _advance_text(self, text: str) -> None:
        block = self._current_block()
        length = _utf16_length(text)
        if block is not None and not self.extension.is_empty() and length:
            start = block.position() + self.block_offset
            if (
                self.ranges
                and self.ranges[-1][0] + self.ranges[-1][1] == start
                and self.ranges[-1][2] == self.extension
            ):
                old_start, old_length, extension = self.ranges[-1]
                self.ranges[-1] = (
                    old_start,
                    old_length + length,
                    extension,
                )
            else:
                self.ranges.append((start, length, self.extension))
        self.block_offset += length

    def _advance_object(self) -> None:
        block = self._current_block()
        if block is not None and self.block_offset < _utf16_length(block.text()):
            self._advance_text('\ufffc')

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in ('p', 'li'):
            self._start_block()
        if tag == 'span':
            self.extension_stack.append(self.extension)
            self.extension = _span_extension(self.extension, attrs)
        elif tag == 'br':
            self._advance_object()
        elif tag == 'img':
            self._advance_object()

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in ('br', 'img'):
            self._advance_object()
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == 'span' and self.extension_stack:
            self.extension = self.extension_stack.pop()
        elif tag in ('p', 'li'):
            self.in_block = False

    def handle_data(self, data: str) -> None:
        if self.in_block or (
            self.block_index < 0 and not self.extension.is_empty()
        ):
            self._advance_text(data)

    def handle_entityref(self, name: str) -> None:
        if self.in_block or (
            self.block_index < 0 and not self.extension.is_empty()
        ):
            self._advance_text(unescape(f'&{name};'))

    def handle_charref(self, name: str) -> None:
        if self.in_block or (
            self.block_index < 0 and not self.extension.is_empty()
        ):
            self._advance_text(unescape(f'&#{name};'))


def _inline_extension_ranges_from_html(
    document: QTextDocument,
    html: str,
) -> tuple[tuple[int, int, _InlineExtension], ...]:
    parser = _InlineExtensionRangeParser(document)
    try:
        parser.feed(html)
        parser.close()
    except (ValueError, TypeError) as error:
        LOGGER.warning('Unable to parse rich-text extensions: %s', error)
        return ()
    return tuple(parser.ranges)


def _apply_inline_extension_ranges(
    document: QTextDocument,
    ranges: tuple[tuple[int, int, _InlineExtension], ...],
    vertical: bool,
) -> None:
    document_end = max(0, document.characterCount() - 1)
    cursor = QTextCursor(document)
    for start, length, extension in ranges:
        if start < 0 or length <= 0 or start + length > document_end:
            LOGGER.warning(
                'Ignoring out-of-range rich-text extension: %r',
                (start, length, extension),
            )
            continue
        modifier = QTextCharFormat()
        if extension.emphasis_style != 'none':
            modifier.setProperty(
                AnnotationProperty.EMPHASIS_STYLE,
                extension.emphasis_style,
            )
            modifier.setProperty(
                AnnotationProperty.EMPHASIS_POSITION,
                extension.emphasis_position,
            )
        if extension.text_combine_id:
            modifier.setProperty(
                AnnotationProperty.TEXT_COMBINE_UPRIGHT,
                TEXT_COMBINE_ALL,
            )
            modifier.setProperty(
                AnnotationProperty.TEXT_COMBINE_ID,
                extension.text_combine_id,
            )
        if extension.letter_spacing is not None:
            modifier.setProperty(
                AnnotationProperty.LETTER_SPACING,
                extension.letter_spacing,
            )
            modifier.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
            modifier.setFontLetterSpacing(
                100.0 if vertical else extension.letter_spacing * 100.0
            )
        cursor.setPosition(start)
        cursor.setPosition(start + length, QTextCursor.MoveMode.KeepAnchor)
        cursor.mergeCharFormat(modifier)


def load_rich_text_html(
    document: QTextDocument,
    html: str,
    *,
    letter_spacing_fallback: Optional[float] = None,
    vertical: bool = False,
) -> None:
    """Load old Qt HTML or semantic extension HTML into ``document``."""
    undo_enabled = document.isUndoRedoEnabled()
    document.setUndoRedoEnabled(False)
    try:
        document.setHtml(html)
        extension_ranges = _inline_extension_ranges_from_html(document, html)
        fallback = _canonical_letter_spacing(letter_spacing_fallback)
        if fallback is not None:
            # Old HTML has no inline spacing. Seed its item-wide value; the
            # next save writes explicit spans for every resulting range.
            _apply_document_letter_spacing(document, fallback, vertical)
        _apply_inline_extension_ranges(document, extension_ranges, vertical)
    finally:
        document.setUndoRedoEnabled(undo_enabled)


def text_combine_upright_values(
    char_format: QTextCharFormat,
) -> tuple[str, str]:
    """Return the canonical text-combine value and its run ID."""
    value = str(
        char_format.property(AnnotationProperty.TEXT_COMBINE_UPRIGHT) or ''
    )
    group_id = str(
        char_format.property(AnnotationProperty.TEXT_COMBINE_ID) or ''
    )
    if value != TEXT_COMBINE_ALL or not group_id:
        return TEXT_COMBINE_NONE, ''
    return value, group_id


def text_combine_upright_ranges(
    block: QTextBlock,
) -> tuple[tuple[int, int, str], ...]:
    """Return contiguous local UTF-16 ranges grouped by their stable ID."""
    ranges = []
    iterator = block.begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        if fragment.isValid() and fragment.length() > 0:
            value, group_id = text_combine_upright_values(
                fragment.charFormat()
            )
            if value == TEXT_COMBINE_ALL:
                start = fragment.position() - block.position()
                length = fragment.length()
                if (
                    ranges
                    and ranges[-1][0] + ranges[-1][1] == start
                    and ranges[-1][2] == group_id
                ):
                    old_start, old_length, old_id = ranges[-1]
                    ranges[-1] = (old_start, old_length + length, old_id)
                else:
                    ranges.append((start, length, group_id))
        iterator += 1
    return tuple(ranges)


def _inline_extension_ranges(
    document: QTextDocument,
) -> list[tuple[int, int, _InlineExtension]]:
    fragments = []
    has_explicit_spacing = False
    block = document.firstBlock()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.length() > 0:
                spacing = _canonical_letter_spacing(
                    fragment.charFormat().property(
                        AnnotationProperty.LETTER_SPACING
                    )
                )
                has_explicit_spacing = (
                    has_explicit_spacing or spacing is not None
                )
                fragments.append(
                    (
                        fragment.position(),
                        fragment.length(),
                        fragment.charFormat(),
                        spacing,
                    )
                )
            iterator += 1
        block = block.next()

    ranges = []
    for start, length, char_format, spacing in fragments:
        emphasis_style, emphasis_position = emphasis_values(char_format)
        _combine_value, text_combine_id = text_combine_upright_values(
            char_format
        )
        extension = _InlineExtension(
            emphasis_style=emphasis_style,
            emphasis_position=emphasis_position,
            text_combine_id=text_combine_id,
            letter_spacing=(
                1.0 if has_explicit_spacing and spacing is None else spacing
            ),
        )
        if extension.is_empty():
            continue
        if (
            ranges
            and ranges[-1][0] + ranges[-1][1] == start
            and ranges[-1][2] == extension
        ):
            old_start, old_length, old_extension = ranges[-1]
            ranges[-1] = (
                old_start,
                old_length + length,
                old_extension,
            )
        else:
            ranges.append((start, length, extension))
    return ranges


def _format_spacing_number(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return format(value, '.12g')


def _inline_extension_span(text: str, extension: _InlineExtension) -> str:
    styles = []
    attributes = []
    if extension.emphasis_style != 'none':
        styles.extend(
            (
                f'text-emphasis-style: {extension.emphasis_style}',
                f'text-emphasis-position: {extension.emphasis_position}',
            )
        )
    if extension.text_combine_id:
        styles.append('text-combine-upright: all')
        attributes.append(
            f'{TEXT_COMBINE_ID_ATTRIBUTE}="'
            f'{escape(extension.text_combine_id, quote=True)}"'
        )
    if extension.letter_spacing is not None:
        multiplier = _format_spacing_number(extension.letter_spacing)
        css_spacing = _format_spacing_number(
            extension.letter_spacing - 1.0
        )
        styles.append(f'letter-spacing: {css_spacing}em')
        attributes.append(f'{LETTER_SPACING_ATTRIBUTE}="{multiplier}"')
    style_attribute = f'style="{"; ".join(styles)};"'
    suffix = ' '.join((style_attribute, *attributes))
    return f'<span {suffix}>{text}</span>'


class _InlineExtensionHTMLExporter(HTMLParser):
    """Inject inline extensions without replacing Qt's HTML serializer.

    >>> _format_spacing_number(1.15)
    '1.15'
    """

    def __init__(
        self,
        document: QTextDocument,
        ranges: list[tuple[int, int, _InlineExtension]],
    ) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks = _document_blocks(document)
        self.ranges = ranges
        self.range_index = 0
        self.block_index = -1
        self.block_offset = 0
        self.in_block = False
        self.output: list[str] = []

    def _start_block(self) -> None:
        self.block_index += 1
        self.block_offset = 0
        self.in_block = True

    def _current_block(self) -> Optional[QTextBlock]:
        if 0 <= self.block_index < len(self.blocks):
            return self.blocks[self.block_index]
        return None

    def _extension_at(self, position: int) -> Optional[_InlineExtension]:
        while self.range_index < len(self.ranges):
            start, length, _value = self.ranges[self.range_index]
            if start + length > position:
                break
            self.range_index += 1
        if self.range_index >= len(self.ranges):
            return None
        start, length, extension = self.ranges[self.range_index]
        if start <= position < start + length:
            return extension
        return None

    def _append_text(self, raw: str, decoded: Optional[str] = None) -> None:
        if not self.in_block:
            self.output.append(raw)
            return
        block = self._current_block()
        if block is None:
            self.output.append(raw)
            return
        decoded = raw if decoded is None else decoded
        if raw != decoded:
            position = block.position() + self.block_offset
            extension = self._extension_at(position)
            self.output.append(
                raw
                if extension is None
                else _inline_extension_span(raw, extension)
            )
            self.block_offset += _utf16_length(decoded)
            return

        segments: list[tuple[Optional[_InlineExtension], str]] = []
        for character in raw:
            position = block.position() + self.block_offset
            extension = self._extension_at(position)
            if segments and segments[-1][0] == extension:
                old_extension, old_text = segments[-1]
                segments[-1] = (old_extension, old_text + character)
            else:
                segments.append((extension, character))
            self.block_offset += _utf16_length(character)
        for extension, text in segments:
            self.output.append(
                text
                if extension is None
                else _inline_extension_span(text, extension)
            )

    def _advance_object(self) -> None:
        block = self._current_block()
        if block is not None and self.block_offset < _utf16_length(block.text()):
            self.block_offset += 1

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in ('p', 'li'):
            self._start_block()
        self.output.append(self.get_starttag_text())
        if tag in ('br', 'img'):
            self._advance_object()

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self.output.append(self.get_starttag_text())
        if tag.lower() in ('br', 'img'):
            self._advance_object()

    def handle_endtag(self, tag: str) -> None:
        self.output.append(f'</{tag}>')
        if tag.lower() in ('p', 'li'):
            self.in_block = False

    def handle_data(self, data: str) -> None:
        self._append_text(data)

    def handle_entityref(self, name: str) -> None:
        raw = f'&{name};'
        self._append_text(raw, unescape(raw))

    def handle_charref(self, name: str) -> None:
        raw = f'&#{name};'
        self._append_text(raw, unescape(raw))

    def handle_comment(self, data: str) -> None:
        self.output.append(f'<!--{data}-->')

    def handle_decl(self, decl: str) -> None:
        if decl.lower().startswith('doctype html'):
            self.output.append('<!DOCTYPE html>')
        else:
            self.output.append(f'<!{decl}>')

    def handle_pi(self, data: str) -> None:
        self.output.append(f'<?{data}>')

    def unknown_decl(self, data: str) -> None:
        self.output.append(f'<![{data}]>')


def _add_inline_extensions(
    document: QTextDocument,
    html: str,
) -> str:
    ranges = _inline_extension_ranges(document)
    if not ranges:
        return html
    parser = _InlineExtensionHTMLExporter(document, ranges)
    try:
        parser.feed(html)
        parser.close()
    except (ValueError, TypeError) as error:
        LOGGER.warning('Unable to export rich-text extensions: %s', error)
        return html
    return ''.join(parser.output)


def to_rich_text_html(
    document: QTextDocument,
    html: Optional[str] = None,
) -> str:
    """Extend Qt's HTML with semantic inline formatting."""
    if html is None:
        html = document.toHtml()
    return _add_inline_extensions(document, html)


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


def letter_spacing_value(
    char_format: QTextCharFormat,
    fallback: float = 1.0,
) -> float:
    """Return semantic character spacing as a font-size multiplier."""
    value = _canonical_letter_spacing(
        char_format.property(AnnotationProperty.LETTER_SPACING)
    )
    if value is not None:
        return value
    fallback_value = _canonical_letter_spacing(fallback)
    return 1.0 if fallback_value is None else fallback_value


def apply_letter_spacing(
    cursor: QTextCursor,
    value: float,
    *,
    vertical: bool,
) -> None:
    """Merge character spacing into a selection or insertion format."""
    canonical_value = _canonical_letter_spacing(value)
    if canonical_value is None:
        raise ValueError(f'unsupported letter spacing: {value!r}')
    cursor.mergeCharFormat(
        _letter_spacing_modifier(canonical_value, vertical)
    )


def set_document_letter_spacing_writing_mode(
    document: QTextDocument,
    *,
    vertical: bool,
    fallback: float,
) -> None:
    """Preserve semantic spacing while changing its Qt shaping value."""
    fallback_value = _canonical_letter_spacing(fallback)
    fallback = 1.0 if fallback_value is None else fallback_value
    ranges = []
    block = document.firstBlock()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.length() > 0:
                ranges.append(
                    (
                        fragment.position(),
                        fragment.length(),
                        letter_spacing_value(fragment.charFormat(), fallback),
                    )
                )
            iterator += 1
        block = block.next()

    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    try:
        if not ranges:
            _apply_document_letter_spacing(document, fallback, vertical)
        for start, length, value in ranges:
            cursor.setPosition(start)
            cursor.setPosition(
                start + length,
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.mergeCharFormat(
                _letter_spacing_modifier(value, vertical)
            )
    finally:
        cursor.endEditBlock()


def apply_text_combine_upright(
    cursor: QTextCursor,
    enabled: bool,
) -> None:
    """Apply one combined-run ID to a selection or insertion format."""
    modifier = QTextCharFormat()
    modifier.setProperty(
        AnnotationProperty.TEXT_COMBINE_UPRIGHT,
        TEXT_COMBINE_ALL if enabled else TEXT_COMBINE_NONE,
    )
    modifier.setProperty(
        AnnotationProperty.TEXT_COMBINE_ID,
        uuid4().hex if enabled else '',
    )
    cursor.mergeCharFormat(modifier)


def _remap_text_combine_ids(document: QTextDocument) -> None:
    """Give every pasted combined group a document-local identity."""
    replacements = {}
    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    try:
        block = document.firstBlock()
        while block.isValid():
            for start, length, group_id in text_combine_upright_ranges(block):
                replacement = replacements.setdefault(group_id, uuid4().hex)
                modifier = QTextCharFormat()
                modifier.setProperty(
                    AnnotationProperty.TEXT_COMBINE_ID,
                    replacement,
                )
                cursor.setPosition(block.position() + start)
                cursor.setPosition(
                    block.position() + start + length,
                    QTextCursor.MoveMode.KeepAnchor,
                )
                cursor.mergeCharFormat(modifier)
            block = block.next()
    finally:
        cursor.endEditBlock()


def create_rich_text_mime(cursor: QTextCursor) -> QMimeData:
    """Create interoperable clipboard data with exact inline extensions."""
    mime = QMimeData()
    if not cursor.hasSelection():
        return mime
    document = QTextDocument()
    document.setUndoRedoEnabled(False)
    target = QTextCursor(document)
    target.insertFragment(QTextDocumentFragment(cursor))
    extended_html = to_rich_text_html(document)
    mime.setText(document.toPlainText())
    mime.setHtml(extended_html)
    mime.setData(
        RICH_TEXT_MIME_TYPE,
        QByteArray(extended_html.encode('utf-8')),
    )
    return mime


def insert_rich_text_mime(
    cursor: QTextCursor,
    mime: QMimeData,
    *,
    vertical: bool = False,
) -> bool:
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
    load_rich_text_html(document, html, vertical=vertical)
    _remap_text_combine_ids(document)
    cursor.insertFragment(QTextDocumentFragment(document))
    return True
