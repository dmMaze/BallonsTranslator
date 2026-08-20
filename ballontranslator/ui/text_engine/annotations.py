"""Rich-text extensions that Qt cannot export by itself.

Qt remains the live editing model. Character and block formats carry live
meaning; one semantic HTML boundary stores paragraph line spacing, emphasis,
tate-chu-yoko, letter spacing, font variants, and their exact
application-owned values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from html import escape, unescape
from html.parser import HTMLParser
import math
import re
from typing import AbstractSet, Callable, Optional
from uuid import uuid4

from qtpy import QT6
from qtpy.QtCore import QByteArray, QMimeData
from qtpy.QtGui import (
    QTextBlock,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextDocumentFragment,
    QTextFormat,
    QFont,
)

from ballontranslator.utils.fontformat import (
    LineSpacingType,
    export_font_weight_html,
    import_font_weight_html,
)
from ballontranslator.utils.logger import logger as LOGGER
from .font_family import (
    html_uses_project_font_family,
    normalize_document_font_families,
    restore_project_font_families_in_html,
)
from .rendering.indexing import _grapheme_ranges, _utf16_length, _utf16_slice


RICH_TEXT_MIME_TYPE = 'application/x-ballonstranslator-rich-text'
MAX_RICH_TEXT_MIME_BYTES = 16 * 1024 * 1024
LETTER_SPACING_ATTRIBUTE = 'data-btrans-letter-spacing'
LINE_DISTANCE_ATTRIBUTE = 'data-btrans-line-distance'
TEXT_COMBINE_ID_ATTRIBUTE = 'data-btrans-text-combine-id'
_RICH_TEXT_EXTENSION_MARKERS = (
    'text-emphasis-style',
    'text-combine-upright',
    'font-variant-ligatures',
    'font-variant-numeric',
    LETTER_SPACING_ATTRIBUTE,
    LINE_DISTANCE_ATTRIBUTE,
    '<ruby',
    'data-btrans-runtime-ruby-id',
)

RUBY_TYPES = ('group', 'mono')
RUBY_POSITIONS = ('over', 'under')
DEFAULT_RUBY_POSITION = 'over'
_RUNTIME_RUBY_ATTRIBUTES = (
    'data-btrans-runtime-ruby-id',
    'data-btrans-runtime-ruby-unit-id',
    'data-btrans-runtime-ruby-type',
    'data-btrans-runtime-ruby-text',
    'data-btrans-runtime-ruby-position',
)


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

    # Font-variant intent is semantic state. Native shaping is derived from it
    # together with tracking and writing mode for Qt 5/6 parity.
    FONT_VARIANT_LIGATURES = (
        _enum_value(QTextFormat.Property.UserProperty) + 1400
    )
    FONT_VARIANT_NUMERIC = (
        _enum_value(QTextFormat.Property.UserProperty) + 1401
    )

    # Ruby IDs are runtime-only. Semantic HTML stores only the relationship;
    # loading and in-app paste allocate fresh container and unit identities.
    RUBY_ID = _enum_value(QTextFormat.Property.UserProperty) + 1420
    RUBY_UNIT_ID = _enum_value(QTextFormat.Property.UserProperty) + 1421
    RUBY_TYPE = _enum_value(QTextFormat.Property.UserProperty) + 1422
    RUBY_TEXT = _enum_value(QTextFormat.Property.UserProperty) + 1423
    RUBY_POSITION = _enum_value(QTextFormat.Property.UserProperty) + 1424


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
EMPHASIS_GLYPHS = {
    'filled dot': '\u2022',
    'open dot': '\u25e6',
    'filled circle': '\u25cf',
    'open circle': '\u25cb',
    'filled double-circle': '\u25c9',
    'open double-circle': '\u25ce',
    'filled triangle': '\u25b2',
    'open triangle': '\u25b3',
    'filled sesame': '\ufe45',
    'open sesame': '\ufe46',
}
EMPHASIS_POSITIONS = (
    'over right',
    'under right',
    'over left',
    'under left',
)
DEFAULT_EMPHASIS_POSITION = 'over right'
TEXT_COMBINE_NONE = 'none'
TEXT_COMBINE_ALL = 'all'
FONT_VARIANT_LIGATURES_NORMAL = 'normal'
FONT_VARIANT_LIGATURES_NONE = 'none'
FONT_VARIANT_NUMERIC_NORMAL = 'normal'
FONT_VARIANT_NUMERIC_OLDSTYLE = 'oldstyle-nums'
FONT_VARIANT_NUMERIC_LINING = 'lining-nums'
OLDSTYLE_NUMS = 'oldstyle'
LIGATURE_COMMON = 'common'
LIGATURE_DISCRETIONARY = 'discretionary'
LIGATURE_HISTORICAL = 'historical'
LIGATURE_CONTEXTUAL = 'contextual'
LIGATURE_DEFAULT = 'default'
LIGATURE_ENABLED = 'enabled'
LIGATURE_DISABLED = 'disabled'
LIGATURE_AXIS_VALUES = (
    LIGATURE_DEFAULT,
    LIGATURE_ENABLED,
    LIGATURE_DISABLED,
)
_LIGATURE_AXIS_TOKENS = {
    LIGATURE_COMMON: ('common-ligatures', 'no-common-ligatures'),
    LIGATURE_DISCRETIONARY: (
        'discretionary-ligatures',
        'no-discretionary-ligatures',
    ),
    LIGATURE_HISTORICAL: (
        'historical-ligatures',
        'no-historical-ligatures',
    ),
    LIGATURE_CONTEXTUAL: (
        'contextual',
        'no-contextual',
    ),
}
_LIGATURE_FEATURE_TAGS = {
    LIGATURE_COMMON: ('liga', 'clig'),
    LIGATURE_DISCRETIONARY: ('dlig',),
    LIGATURE_HISTORICAL: ('hlig',),
    LIGATURE_CONTEXTUAL: ('calt',),
}
# QTextCharFormat's feature-map API is the Qt 6.11 support boundary.
FONT_FEATURES_AVAILABLE = bool(
    QT6 and hasattr(QTextCharFormat, 'setFontFeatures')
)
MAX_ANNOTATION_ID_LENGTH = 128
MAX_LETTER_SPACING = 10.0
MAX_LINE_SPACING = 100.0


def _canonical_spacing(value: object, maximum: float) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= maximum else None


def canonical_line_spacing(value: object) -> Optional[float]:
    """Return a supported line-spacing value, or ``None`` when invalid.

    >>> canonical_line_spacing(1.25)
    1.25
    >>> canonical_line_spacing(float('nan')) is None
    True
    """
    return _canonical_spacing(value, MAX_LINE_SPACING)


def canonical_line_spacing_type(
    value: object,
) -> Optional[LineSpacingType]:
    """Return a supported line-spacing type.

    >>> canonical_line_spacing_type(0) == LineSpacingType.Proportional
    True
    >>> canonical_line_spacing_type(True) is None
    True
    """
    if isinstance(value, bool):
        return None
    try:
        return LineSpacingType(value)
    except (TypeError, ValueError):
        return None


def validated_line_spacing(
    value: object,
    spacing_type: object,
) -> tuple[float, LineSpacingType]:
    canonical_value = canonical_line_spacing(value)
    canonical_type = canonical_line_spacing_type(spacing_type)
    if canonical_value is None or canonical_type is None:
        raise ValueError(f'unsupported line spacing: {(value, spacing_type)!r}')
    return canonical_value, canonical_type


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
    font_variant_ligatures: str = FONT_VARIANT_LIGATURES_NORMAL
    font_variant_numeric: str = FONT_VARIANT_NUMERIC_NORMAL
    ruby_id: str = ''
    ruby_unit_id: str = ''
    ruby_type: str = ''
    ruby_text: str = ''
    ruby_position: str = DEFAULT_RUBY_POSITION

    def is_empty(self) -> bool:
        return (
            self.emphasis_style == 'none'
            and not self.text_combine_id
            and self.letter_spacing is None
            and self.font_variant_ligatures
            == FONT_VARIANT_LIGATURES_NORMAL
            and self.font_variant_numeric == FONT_VARIANT_NUMERIC_NORMAL
            and not self.ruby_id
        )


def canonical_letter_spacing(value: object) -> Optional[float]:
    """Return a supported spacing multiplier, or ``None`` when invalid.

    >>> canonical_letter_spacing(1.25)
    1.25
    >>> canonical_letter_spacing(True) is None
    True
    """
    return _canonical_spacing(value, MAX_LETTER_SPACING)


def canonical_font_variant_ligatures(value: object) -> Optional[str]:
    """Return one canonical CSS ``font-variant-ligatures`` value.

    The four CSS axes are independent, while ``normal`` and ``none`` are
    exclusive aggregate values.

    >>> canonical_font_variant_ligatures('contextual discretionary-ligatures')
    'discretionary-ligatures contextual'
    >>> canonical_font_variant_ligatures('none')
    'none'
    >>> canonical_font_variant_ligatures('contextual no-contextual') is None
    True
    """
    if not isinstance(value, str):
        return None
    tokens = value.strip().lower().split()
    if len(tokens) == 1 and tokens[0] in {
        FONT_VARIANT_LIGATURES_NORMAL,
        FONT_VARIANT_LIGATURES_NONE,
    }:
        return tokens[0]
    if not tokens or any(
        token in {
            FONT_VARIANT_LIGATURES_NORMAL,
            FONT_VARIANT_LIGATURES_NONE,
        }
        for token in tokens
    ):
        return None

    selected = {}
    for token in tokens:
        axis = next(
            (
                candidate
                for candidate, values in _LIGATURE_AXIS_TOKENS.items()
                if token in values
            ),
            None,
        )
        if axis is None or axis in selected:
            return None
        selected[axis] = token
    return ' '.join(
        selected[axis]
        for axis in _LIGATURE_AXIS_TOKENS
        if axis in selected
    )


def canonical_font_variant_numeric(value: object) -> Optional[str]:
    """Return the supported CSS figure-style value.

    >>> canonical_font_variant_numeric('OLDSTYLE-NUMS')
    'oldstyle-nums'
    """
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value in {
        FONT_VARIANT_NUMERIC_NORMAL,
        FONT_VARIANT_NUMERIC_OLDSTYLE,
        FONT_VARIANT_NUMERIC_LINING,
    }:
        return value
    return None


def _ligature_axis_states(value: str) -> dict[str, str]:
    states = {
        axis: LIGATURE_DEFAULT for axis in _LIGATURE_AXIS_TOKENS
    }
    if value == FONT_VARIANT_LIGATURES_NONE:
        return {axis: LIGATURE_DISABLED for axis in states}
    if value == FONT_VARIANT_LIGATURES_NORMAL:
        return states
    tokens = value.split()
    for axis, (enabled, disabled) in _LIGATURE_AXIS_TOKENS.items():
        if enabled in tokens:
            states[axis] = LIGATURE_ENABLED
        elif disabled in tokens:
            states[axis] = LIGATURE_DISABLED
    return states


def _font_variant_ligatures_with_axis(
    value: str,
    axis: str,
    state: str,
) -> str:
    states = _ligature_axis_states(value)
    states[axis] = state
    if all(value == LIGATURE_DEFAULT for value in states.values()):
        return FONT_VARIANT_LIGATURES_NORMAL
    if all(value == LIGATURE_DISABLED for value in states.values()):
        return FONT_VARIANT_LIGATURES_NONE
    return ' '.join(
        _LIGATURE_AXIS_TOKENS[current_axis][
            0 if current_state == LIGATURE_ENABLED else 1
        ]
        for current_axis, current_state in states.items()
        if current_state != LIGATURE_DEFAULT
    )


def _parse_letter_spacing_attribute(value: object) -> Optional[float]:
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return canonical_letter_spacing(parsed)


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

    if 'font-variant-ligatures' in styles:
        value = canonical_font_variant_ligatures(
            styles['font-variant-ligatures']
        )
        if value is None:
            LOGGER.warning(
                'Ignoring unsupported font-variant-ligatures: %r',
                styles['font-variant-ligatures'],
            )
        else:
            extension = replace(
                extension,
                font_variant_ligatures=value,
            )

    if 'font-variant-numeric' in styles:
        value = canonical_font_variant_numeric(
            styles['font-variant-numeric']
        )
        if value is None:
            LOGGER.warning(
                'Ignoring unsupported font-variant-numeric: %r',
                styles['font-variant-numeric'],
            )
        else:
            extension = replace(extension, font_variant_numeric=value)

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

    if _RUNTIME_RUBY_ATTRIBUTES[0] in attributes:
        values = tuple(attributes.get(name, '') for name in _RUNTIME_RUBY_ATTRIBUTES)
        ruby_id, unit_id, ruby_type, ruby_text, position = values
        valid = (
            bool(ruby_id)
            and bool(unit_id)
            and ruby_type in RUBY_TYPES
            and bool(ruby_text)
            and position in RUBY_POSITIONS
            and len(ruby_id) <= MAX_ANNOTATION_ID_LENGTH
            and len(unit_id) <= MAX_ANNOTATION_ID_LENGTH
        )
        if valid:
            extension = replace(
                extension,
                ruby_id=ruby_id,
                ruby_unit_id=unit_id,
                ruby_type=ruby_type,
                ruby_text=ruby_text,
                ruby_position=position,
            )
        else:
            LOGGER.warning('Ignoring invalid runtime Ruby annotation')
    return extension


def font_variant_ligatures_value(char_format: QTextCharFormat) -> str:
    """Return the canonical ligature property from a character format."""
    value = canonical_font_variant_ligatures(
        char_format.property(AnnotationProperty.FONT_VARIANT_LIGATURES)
    )
    return FONT_VARIANT_LIGATURES_NORMAL if value is None else value


def font_variant_numeric_value(char_format: QTextCharFormat) -> str:
    """Return the canonical numeric figure style from a character format."""
    value = canonical_font_variant_numeric(
        char_format.property(AnnotationProperty.FONT_VARIANT_NUMERIC)
    )
    return FONT_VARIANT_NUMERIC_NORMAL if value is None else value


def oldstyle_nums_value(char_format: QTextCharFormat) -> str:
    """Return oldstyle figures as ``default``, ``enabled``, or ``disabled``."""
    value = font_variant_numeric_value(char_format)
    if value == FONT_VARIANT_NUMERIC_OLDSTYLE:
        return LIGATURE_ENABLED
    if value == FONT_VARIANT_NUMERIC_LINING:
        return LIGATURE_DISABLED
    return LIGATURE_DEFAULT


def ligature_axis_value(
    char_format: QTextCharFormat,
    axis: str,
) -> str:
    """Return one axis as ``default``, ``enabled``, or ``disabled``."""
    if axis not in _LIGATURE_AXIS_TOKENS:
        raise ValueError(f'unsupported ligature axis: {axis!r}')
    return _ligature_axis_states(
        font_variant_ligatures_value(char_format)
    )[axis]


def _sync_native_font_features(
    char_format: QTextCharFormat,
    feature_values: dict[str, Optional[int]],
) -> None:
    current = dict(char_format.fontFeatures())
    updated = dict(current)
    for name, value in feature_values.items():
        tag = QFont.Tag.fromString(name)
        if value is None:
            updated.pop(tag, None)
        else:
            updated[tag] = value
    if updated != current:
        char_format.setFontFeatures(updated)


def sync_native_oldstyle_nums(char_format: QTextCharFormat) -> None:
    """Apply the semantic figure style through Qt 6.11 font features."""
    if not FONT_FEATURES_AVAILABLE:
        return
    state = oldstyle_nums_value(char_format)
    if state == LIGATURE_ENABLED:
        values = {'onum': 1, 'lnum': 0}
    elif state == LIGATURE_DISABLED:
        values = {'onum': 0, 'lnum': 1}
    else:
        values = {'onum': None, 'lnum': None}
    _sync_native_font_features(char_format, values)


def sync_native_ligature_shaping(
    char_format: QTextCharFormat,
    *,
    vertical: bool,
    letter_spacing_fallback: float = 1.0,
) -> None:
    """Derive Qt 5/6 shaping properties from semantic inline state.

    Qt disables optional ligatures whenever native letter spacing is present,
    even at 100%. Identity spacing stays unset for horizontal runs that allow
    common ligatures; explicit spacing implements tracking and the Qt 5
    ``no-common-ligatures`` fallback. Qt 6.11 feature tags then restore
    explicitly enabled common and discretionary ligatures, including
    discretionary ligatures in ordinary vertical cells.

    >>> fmt = QTextCharFormat()
    >>> sync_native_ligature_shaping(fmt, vertical=False)
    >>> fmt.hasProperty(QTextFormat.Property.FontLetterSpacing)
    False
    """
    semantic_spacing = canonical_letter_spacing(
        char_format.property(AnnotationProperty.LETTER_SPACING)
    )
    fallback_spacing = canonical_letter_spacing(letter_spacing_fallback)
    spacing = semantic_spacing
    if spacing is None:
        spacing = 1.0 if fallback_spacing is None else fallback_spacing
    states = _ligature_axis_states(
        font_variant_ligatures_value(char_format)
    )
    combine_value, _combine_id = text_combine_upright_values(char_format)
    horizontal_run = not vertical or combine_value == TEXT_COMBINE_ALL
    preserve_native_spacing = (
        semantic_spacing is None
        and char_format.hasProperty(
            QTextFormat.Property.FontLetterSpacing
        )
    )
    if preserve_native_spacing:
        native_spacing = char_format.fontLetterSpacing()
        percentage_spacing = (
            char_format.fontLetterSpacingType()
            == QFont.SpacingType.PercentageSpacing
        )
        identity_spacing = math.isclose(
            native_spacing,
            100.0 if percentage_spacing else 0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    else:
        identity_spacing = math.isclose(
            spacing, 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    needs_explicit_spacing = (
        not horizontal_run
        or not identity_spacing
        or states[LIGATURE_COMMON] == LIGATURE_DISABLED
    )
    if needs_explicit_spacing and not preserve_native_spacing:
        char_format.setFontLetterSpacingType(
            QFont.SpacingType.PercentageSpacing
        )
        char_format.setFontLetterSpacing(
            100.0 if vertical else spacing * 100.0
        )
    elif not needs_explicit_spacing:
        char_format.clearProperty(QTextFormat.Property.FontLetterSpacing)
        char_format.clearProperty(QTextFormat.Property.FontLetterSpacingType)

    if not FONT_FEATURES_AVAILABLE:
        return

    feature_values = {}
    for axis, tags in _LIGATURE_FEATURE_TAGS.items():
        state = states[axis]
        override_tracking = (
            state == LIGATURE_ENABLED
            and axis in {LIGATURE_COMMON, LIGATURE_DISCRETIONARY}
        )
        if (
            axis == LIGATURE_COMMON and not horizontal_run
        ) or (
            axis in {
                LIGATURE_COMMON,
                LIGATURE_DISCRETIONARY,
                LIGATURE_HISTORICAL,
            }
            and not identity_spacing
            and not override_tracking
        ):
            # Native spacing already suppresses these optional features.
            native_value = None
        elif state == LIGATURE_DEFAULT:
            native_value = None
        else:
            native_value = 1 if state == LIGATURE_ENABLED else 0
        feature_values.update({tag: native_value for tag in tags})
    _sync_native_font_features(char_format, feature_values)


def _rewrite_cursor_char_formats(
    cursor: QTextCursor,
    rewrite: Callable[[QTextCharFormat], None],
) -> None:
    """Rewrite selected fragment and paragraph-boundary formats exactly.

    >>> callable(_rewrite_cursor_char_formats)
    True
    """
    def changed_format(
        source: QTextCharFormat,
    ) -> Optional[QTextCharFormat]:
        updated = QTextCharFormat(source)
        rewrite(updated)
        return None if updated == source else updated

    if not cursor.hasSelection():
        char_format = changed_format(cursor.charFormat())
        if char_format is not None:
            cursor.setCharFormat(char_format)
            if cursor.document().isEmpty():
                # Qt rebuilds an empty caret format from its block after text
                # is inserted and deleted, so keep that insertion owner equal.
                cursor.setBlockCharFormat(char_format)
        return

    document = cursor.document()
    start = cursor.selectionStart()
    end = cursor.selectionEnd()
    whole_document = (
        start == 0 and end == max(0, document.characterCount() - 1)
    )
    ranges = []
    block_formats = []
    block = document.findBlock(start)
    while block.isValid() and block.position() <= end:
        # Qt's document selection omits an empty first paragraph's insertion
        # format; an item-wide change must still apply to future text there.
        if start < block.position() or (
            whole_document and block.position() == 0
        ):
            char_format = changed_format(block.charFormat())
            if char_format is not None:
                block_formats.append((block.position(), char_format))
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                fragment_start = fragment.position()
                fragment_end = fragment_start + fragment.length()
                range_start = max(start, fragment_start)
                range_end = min(end, fragment_end)
                if range_start < range_end:
                    char_format = changed_format(fragment.charFormat())
                    if char_format is not None:
                        ranges.append((
                            range_start,
                            range_end,
                            char_format,
                        ))
            iterator += 1
        block = block.next()

    target = QTextCursor(document)
    for range_start, range_end, char_format in ranges:
        target.setPosition(range_start)
        target.setPosition(range_end, QTextCursor.MoveMode.KeepAnchor)
        target.setCharFormat(char_format)
    for position, char_format in block_formats:
        block = document.findBlock(position)
        QTextCursor(block).setBlockCharFormat(char_format)


def _set_semantic_letter_spacing(
    char_format: QTextCharFormat,
    value: float,
    vertical: bool,
) -> None:
    char_format.setProperty(AnnotationProperty.LETTER_SPACING, value)
    sync_native_ligature_shaping(char_format, vertical=vertical)


def _apply_document_letter_spacing(
    document: QTextDocument,
    value: float,
    vertical: bool,
) -> None:
    cursor = QTextCursor(document)
    if document.isEmpty():
        char_format = QTextCharFormat(cursor.blockCharFormat())
        _set_semantic_letter_spacing(char_format, value, vertical)
        cursor.setBlockCharFormat(char_format)
        return
    cursor.select(QTextCursor.SelectionType.Document)
    apply_letter_spacing(cursor, value, vertical=vertical)


def _native_line_spacing_values(
    block_format: QTextBlockFormat,
) -> Optional[tuple[float, LineSpacingType]]:
    line_height_type = _enum_value(block_format.lineHeightType())
    proportional_type = _enum_value(
        QTextBlockFormat.LineHeightTypes.ProportionalHeight
    )
    distance_type = _enum_value(
        QTextBlockFormat.LineHeightTypes.LineDistanceHeight
    )
    if line_height_type == proportional_type:
        value = canonical_line_spacing(block_format.lineHeight() / 100.0)
        spacing_type = LineSpacingType.Proportional
    elif line_height_type == distance_type:
        value = canonical_line_spacing(block_format.lineHeight() / 10.0)
        spacing_type = LineSpacingType.Distance
    else:
        return None
    return None if value is None else (value, spacing_type)


def line_spacing_values(
    block_format: QTextBlockFormat,
    fallback: float = 1.2,
    fallback_type: int = LineSpacingType.Proportional,
) -> tuple[float, LineSpacingType]:
    """Return the paragraph's semantic line-spacing value and type.

    >>> line_spacing_values(QTextBlockFormat(), 1.25)[0]
    1.25
    """
    values = _native_line_spacing_values(block_format)
    if values is not None:
        return values
    fallback_value = canonical_line_spacing(fallback)
    spacing_type = canonical_line_spacing_type(fallback_type)
    return (
        1.2 if fallback_value is None else fallback_value,
        LineSpacingType.Proportional
        if spacing_type is None else spacing_type,
    )


def _line_spacing_modifier(
    value: float,
    spacing_type: LineSpacingType,
) -> QTextBlockFormat:
    modifier = QTextBlockFormat()
    if spacing_type == LineSpacingType.Proportional:
        modifier.setLineHeight(
            value * 100.0,
            _enum_value(
                QTextBlockFormat.LineHeightTypes.ProportionalHeight
            ),
        )
    else:
        modifier.setLineHeight(
            value * 10.0,
            _enum_value(
                QTextBlockFormat.LineHeightTypes.LineDistanceHeight
            ),
        )
    return modifier


def apply_line_spacing(
    cursor: QTextCursor,
    value: float,
    spacing_type: int,
) -> None:
    """Apply one spacing pair to the end-exclusive selected paragraphs.

    A caret formats its current paragraph. A non-empty selection formats the
    paragraph containing its first character through the paragraph containing
    its last selected character, so ending at the next paragraph start does
    not include that paragraph.

    >>> callable(apply_line_spacing)
    True
    """
    canonical_value, canonical_type = validated_line_spacing(
        value, spacing_type
    )
    start = cursor.selectionStart()
    end = cursor.selectionEnd()
    target = QTextCursor(cursor)
    if end > start:
        target.setPosition(start)
        target.setPosition(end - 1, QTextCursor.MoveMode.KeepAnchor)
    target.mergeBlockFormat(
        _line_spacing_modifier(canonical_value, canonical_type)
    )


def _document_blocks(document: QTextDocument) -> tuple[QTextBlock, ...]:
    blocks = []
    block = document.firstBlock()
    while block.isValid():
        blocks.append(block)
        block = block.next()
    return tuple(blocks)


def _start_tag(tag: str, attrs: list) -> str:
    attributes = ''.join(
        f' {name}' if value is None else f' {name}="{escape(str(value), quote=True)}"'
        for name, value in attrs
    )
    return f'<{tag}{attributes}>'


class _RubyBaseOnlyParser(HTMLParser):
    """Remove Ruby annotations while retaining ordinary base markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.output: list[str] = []
        self.annotation_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in ('rt', 'rp'):
            self.annotation_depth += 1
        elif tag != 'ruby' and self.annotation_depth == 0:
            self.output.append(self.get_starttag_text() or _start_tag(tag, attrs))

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if self.annotation_depth == 0 and tag.lower() not in ('ruby', 'rt', 'rp'):
            self.output.append(self.get_starttag_text() or _start_tag(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ('rt', 'rp') and self.annotation_depth:
            self.annotation_depth -= 1
        elif tag != 'ruby' and self.annotation_depth == 0:
            self.output.append(f'</{tag}>')

    def handle_data(self, data: str) -> None:
        if self.annotation_depth == 0:
            self.output.append(data)

    def handle_entityref(self, name: str) -> None:
        if self.annotation_depth == 0:
            self.output.append(f'&{name};')

    def handle_charref(self, name: str) -> None:
        if self.annotation_depth == 0:
            self.output.append(f'&#{name};')


class _RubyContentParser(HTMLParser):
    """Split one non-nested Ruby element into direct base/reading pairs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.base_html: list[str] = []
        self.base_text: list[str] = []
        self.reading: list[str] = []
        self.pairs: list[tuple[str, str, str]] = []
        self.markup_stack: list[str] = []
        self.annotation_tag = ''
        self.annotation_stack: list[str] = []
        self.invalid = False

    def _append_base_start(self, tag: str, attrs: list) -> None:
        self.base_html.append(self.get_starttag_text() or _start_tag(tag, attrs))
        self.markup_stack.append(tag)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if self.annotation_tag:
            if tag in ('ruby', 'rtc', 'rt', 'rp'):
                self.invalid = True
            self.annotation_stack.append(tag)
            return
        if tag in ('rt', 'rp') and not self.markup_stack:
            self.annotation_tag = tag
            self.annotation_stack = [tag]
            if tag == 'rt' and not ''.join(self.base_text):
                self.invalid = True
            return
        if tag in ('ruby', 'rtc', 'br', 'p', 'div', 'li'):
            self.invalid = True
        self._append_base_start(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        if self.annotation_tag:
            if tag.lower() in ('ruby', 'rtc', 'rt', 'rp'):
                self.invalid = True
            return
        tag = tag.lower()
        if tag in ('rt', 'rp', 'ruby', 'rtc', 'br', 'p', 'div', 'li'):
            self.invalid = True
            return
        self.base_html.append(self.get_starttag_text() or _start_tag(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.annotation_tag:
            if not self.annotation_stack or self.annotation_stack[-1] != tag:
                self.invalid = True
                return
            self.annotation_stack.pop()
            if not self.annotation_stack:
                annotation_tag = self.annotation_tag
                self.annotation_tag = ''
                if annotation_tag == 'rt':
                    reading = ''.join(self.reading)
                    self.pairs.append((
                        ''.join(self.base_html),
                        ''.join(self.base_text),
                        reading,
                    ))
                    self.base_html.clear()
                    self.base_text.clear()
                    self.reading.clear()
            return
        if self.markup_stack and self.markup_stack[-1] == tag:
            self.base_html.append(f'</{tag}>')
            self.markup_stack.pop()
        else:
            self.invalid = True

    def handle_data(self, data: str) -> None:
        if self.annotation_tag == 'rt':
            self.reading.append(data)
        elif not self.annotation_tag:
            self.base_html.append(data)
            self.base_text.append(data)

    def handle_entityref(self, name: str) -> None:
        raw = f'&{name};'
        if self.annotation_tag == 'rt':
            self.reading.append(unescape(raw))
        elif not self.annotation_tag:
            self.base_html.append(raw)
            self.base_text.append(unescape(raw))

    def handle_charref(self, name: str) -> None:
        raw = f'&#{name};'
        if self.annotation_tag == 'rt':
            self.reading.append(unescape(raw))
        elif not self.annotation_tag:
            self.base_html.append(raw)
            self.base_text.append(unescape(raw))


def _runtime_ruby_span(
    base_html: str,
    ruby_id: str,
    unit_id: str,
    ruby_type: str,
    ruby_text: str,
    position: str,
) -> str:
    values = (ruby_id, unit_id, ruby_type, ruby_text, position)
    attributes = ' '.join(
        f'{name}="{escape(value, quote=True)}"'
        for name, value in zip(_RUNTIME_RUBY_ATTRIBUTES, values)
    )
    return f'<span {attributes}>{base_html}</span>'


def _sanitize_ruby_element(attrs: list, inner_html: str) -> str:
    parser = _RubyContentParser()
    try:
        parser.feed(inner_html)
        parser.close()
    except (TypeError, ValueError) as error:
        LOGGER.warning('Discarding malformed Ruby annotation: %s', error)
        parser.invalid = True

    attributes = {str(name).lower(): value for name, value in attrs}
    styles = _style_declarations(attributes.get('style'))
    merge = styles.get('ruby-merge', '')
    position = styles.get('ruby-position', DEFAULT_RUBY_POSITION)
    align = styles.get('ruby-align', '')
    overhang = styles.get('ruby-overhang', '')
    inferred_type = 'mono' if len(parser.pairs) > 1 else 'group'
    ruby_type = {
        'merge': 'group',
        'separate': 'mono',
        '': inferred_type,
    }.get(merge)
    trailing_text = ''.join(parser.base_text)
    valid = (
        not parser.invalid
        and not parser.markup_stack
        and not parser.annotation_stack
        and ruby_type in RUBY_TYPES
        and position in RUBY_POSITIONS
        and align in ('', 'space-around')
        and overhang in ('', 'none')
        and not trailing_text
        and bool(parser.pairs)
        and all(
            base_text and reading.strip()
            and not any(
                separator in base_text
                for separator in ('\n', '\r', '\u2028', '\u2029')
            )
            for _html, base_text, reading in parser.pairs
        )
    )
    if ruby_type == 'group':
        valid = valid and len(parser.pairs) == 1
    elif ruby_type == 'mono':
        valid = valid and all(
            len(_grapheme_ranges(base_text)) == 1
            and not any(character.isspace() for character in reading)
            for _html, base_text, reading in parser.pairs
        )
    if not valid:
        fallback = _RubyBaseOnlyParser()
        try:
            fallback.feed(inner_html)
            fallback.close()
        except (TypeError, ValueError):
            return unescape(inner_html)
        LOGGER.warning('Discarding unsupported or malformed Ruby annotation')
        return ''.join(fallback.output)

    ruby_id = uuid4().hex
    if ruby_type == 'group':
        base_html, _base_text, reading = parser.pairs[0]
        return _runtime_ruby_span(
            base_html,
            ruby_id,
            uuid4().hex,
            ruby_type,
            reading,
            position,
        )
    return ''.join(
        _runtime_ruby_span(
            base_html,
            ruby_id,
            uuid4().hex,
            ruby_type,
            reading,
            position,
        )
        for base_html, _base_text, reading in parser.pairs
    )


class _RubyHTMLPreprocessor(HTMLParser):
    """Replace semantic Ruby with base-only HTML plus transient metadata."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.output: list[str] = []
        self.ruby_attrs: list = []
        self.ruby_inner: list[str] = []
        self.ruby_depth = 0

    def _append(self, raw: str) -> None:
        (self.ruby_inner if self.ruby_depth else self.output).append(raw)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag == 'ruby':
            if self.ruby_depth:
                self.ruby_inner.append(self.get_starttag_text() or _start_tag(tag, attrs))
            else:
                self.ruby_attrs = attrs
            self.ruby_depth += 1
            return
        self._append(self.get_starttag_text() or _start_tag(tag, attrs))

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self._append(self.get_starttag_text() or _start_tag(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == 'ruby' and self.ruby_depth:
            self.ruby_depth -= 1
            if self.ruby_depth:
                self.ruby_inner.append('</ruby>')
            else:
                self.output.append(_sanitize_ruby_element(
                    self.ruby_attrs, ''.join(self.ruby_inner)
                ))
                self.ruby_attrs = []
                self.ruby_inner.clear()
            return
        self._append(f'</{tag}>')

    def handle_data(self, data: str) -> None:
        self._append(data)

    def handle_entityref(self, name: str) -> None:
        self._append(f'&{name};')

    def handle_charref(self, name: str) -> None:
        self._append(f'&#{name};')

    def handle_comment(self, data: str) -> None:
        self._append(f'<!--{data}-->')

    def handle_decl(self, decl: str) -> None:
        self._append(f'<!{decl}>')


def _preprocess_ruby_html(html: str) -> str:
    if '<ruby' not in html.lower():
        return html
    parser = _RubyHTMLPreprocessor()
    try:
        parser.feed(html)
        parser.close()
    except (TypeError, ValueError) as error:
        LOGGER.warning('Unable to preprocess Ruby HTML: %s', error)
        return html
    if parser.ruby_depth:
        LOGGER.warning('Discarding unterminated Ruby annotation')
        fallback = _RubyBaseOnlyParser()
        fallback.feed(''.join(parser.ruby_inner))
        parser.output.extend(fallback.output)
    return ''.join(parser.output)


def _line_distance_attribute(attrs: list) -> Optional[float]:
    attributes = {str(name).lower(): value for name, value in attrs}
    if LINE_DISTANCE_ATTRIBUTE not in attributes:
        return None
    raw_value = attributes[LINE_DISTANCE_ATTRIBUTE]
    try:
        value = canonical_line_spacing(float(raw_value))
    except (TypeError, ValueError):
        value = None
    if value is None:
        LOGGER.warning(
            'Ignoring invalid paragraph line distance: %r',
            raw_value,
        )
        return None
    return value


def _replace_style_declarations(
    style: object,
    replacements: tuple[str, ...],
) -> str:
    kept = []
    if isinstance(style, str):
        for declaration in style.split(';'):
            name, separator, _value = declaration.partition(':')
            if (
                separator
                and name.strip().lower()
                in {'line-height', '-qt-line-height-type'}
            ):
                continue
            declaration = declaration.strip()
            if declaration:
                kept.append(declaration)
    kept.extend(replacements)
    return '; '.join(kept) + (';' if kept else '')


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
        self.line_distances: list[tuple[int, float]] = []

    def _start_block(self, attrs: list = ()) -> None:
        self.block_index += 1
        self.block_offset = 0
        self.in_block = True
        distance = _line_distance_attribute(attrs)
        if distance is not None:
            self.line_distances.append((self.block_index, distance))

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
        if tag in ('p', 'li', 'div'):
            self._start_block(attrs)
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
        elif tag in ('p', 'li', 'div'):
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


def _rich_text_extensions_from_html(
    document: QTextDocument,
    html: str,
) -> tuple[
    tuple[tuple[int, int, _InlineExtension], ...],
    tuple[tuple[int, float], ...],
]:
    parser = _InlineExtensionRangeParser(document)
    try:
        parser.feed(html)
        parser.close()
    except (ValueError, TypeError) as error:
        LOGGER.warning('Unable to parse rich-text extensions: %s', error)
        return (), ()
    return tuple(parser.ranges), tuple(parser.line_distances)


def _apply_inline_extension_ranges(
    document: QTextDocument,
    ranges: tuple[tuple[int, int, _InlineExtension], ...],
    vertical: bool,
) -> None:
    def sync_shaping(char_format: QTextCharFormat) -> None:
        sync_native_ligature_shaping(char_format, vertical=vertical)
        sync_native_oldstyle_nums(char_format)

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
        if (
            extension.font_variant_ligatures
            != FONT_VARIANT_LIGATURES_NORMAL
        ):
            modifier.setProperty(
                AnnotationProperty.FONT_VARIANT_LIGATURES,
                extension.font_variant_ligatures,
            )
        if extension.font_variant_numeric != FONT_VARIANT_NUMERIC_NORMAL:
            modifier.setProperty(
                AnnotationProperty.FONT_VARIANT_NUMERIC,
                extension.font_variant_numeric,
            )
        if extension.ruby_id:
            modifier.setProperty(AnnotationProperty.RUBY_ID, extension.ruby_id)
            modifier.setProperty(
                AnnotationProperty.RUBY_UNIT_ID, extension.ruby_unit_id
            )
            modifier.setProperty(
                AnnotationProperty.RUBY_TYPE, extension.ruby_type
            )
            modifier.setProperty(
                AnnotationProperty.RUBY_TEXT, extension.ruby_text
            )
            modifier.setProperty(
                AnnotationProperty.RUBY_POSITION, extension.ruby_position
            )
        cursor.setPosition(start)
        cursor.setPosition(start + length, QTextCursor.MoveMode.KeepAnchor)
        cursor.mergeCharFormat(modifier)
        if (
            extension.letter_spacing is not None
            or extension.text_combine_id
            or extension.font_variant_ligatures
            != FONT_VARIANT_LIGATURES_NORMAL
            or extension.font_variant_numeric
            != FONT_VARIANT_NUMERIC_NORMAL
        ):
            _rewrite_cursor_char_formats(
                cursor,
                sync_shaping,
            )


def _apply_paragraph_line_distances(
    document: QTextDocument,
    distances: tuple[tuple[int, float], ...],
) -> None:
    for block_number, value in distances:
        block = document.findBlockByNumber(block_number)
        if not block.isValid():
            LOGGER.warning(
                'Ignoring out-of-range paragraph line distance: %d',
                block_number,
            )
            continue
        QTextCursor(block).mergeBlockFormat(
            _line_spacing_modifier(value, LineSpacingType.Distance)
        )


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
        qt_html = import_font_weight_html(
            _preprocess_ruby_html(html), qt6=QT6
        )
        document.setHtml(qt_html)
        if html_uses_project_font_family(qt_html):
            normalize_document_font_families(document)
        lowered_html = qt_html.lower()
        extension_ranges, paragraph_distances = (
            _rich_text_extensions_from_html(document, qt_html)
            if any(
                marker in lowered_html
                for marker in _RICH_TEXT_EXTENSION_MARKERS
            )
            else ((), ())
        )
        fallback = canonical_letter_spacing(letter_spacing_fallback)
        if fallback is not None:
            # Old HTML has no inline spacing. Seed its item-wide value; the
            # next save writes explicit spans for every resulting range.
            _apply_document_letter_spacing(document, fallback, vertical)
        _apply_paragraph_line_distances(document, paragraph_distances)
        _apply_inline_extension_ranges(document, extension_ranges, vertical)
        _discard_ruby_tate_overlaps(document)
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


@dataclass(frozen=True)
class RubyUnitRange:
    """One group base or mono pair in absolute Qt UTF-16 coordinates.

    >>> RubyUnitRange(0, 1, 'u', 'か').end
    1
    """

    start: int
    length: int
    unit_id: str
    text: str

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class RubyContainerRange:
    """One validated, contiguous semantic Ruby container.

    >>> unit = RubyUnitRange(0, 1, 'unit', 'か')
    >>> RubyContainerRange(0, 1, 'ruby', 'group', 'over', (unit,)).end
    1
    """

    start: int
    length: int
    container_id: str
    ruby_type: str
    position: str
    units: tuple[RubyUnitRange, ...]

    @property
    def end(self) -> int:
        return self.start + self.length


def ruby_values(
    char_format: QTextCharFormat,
) -> tuple[str, str, str, str, str]:
    """Return canonical runtime Ruby values, or five empty values."""
    values = tuple(str(char_format.property(prop) or '') for prop in (
        AnnotationProperty.RUBY_ID,
        AnnotationProperty.RUBY_UNIT_ID,
        AnnotationProperty.RUBY_TYPE,
        AnnotationProperty.RUBY_TEXT,
        AnnotationProperty.RUBY_POSITION,
    ))
    container_id, unit_id, ruby_type, text, position = values
    if (
        not container_id
        or not unit_id
        or ruby_type not in RUBY_TYPES
        or not text
        or position not in RUBY_POSITIONS
        or len(container_id) > MAX_ANNOTATION_ID_LENGTH
        or len(unit_id) > MAX_ANNOTATION_ID_LENGTH
    ):
        return '', '', '', '', ''
    return container_id, unit_id, ruby_type, text, position


def ruby_containers_in_block(
    block: QTextBlock,
) -> tuple[RubyContainerRange, ...]:
    """Normalize fragment-split Ruby properties into validated containers."""
    chunks = []
    iterator = block.begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        if fragment.isValid() and fragment.length() > 0:
            values = ruby_values(fragment.charFormat())
            if values[0]:
                start = fragment.position()
                length = fragment.length()
                if (
                    chunks
                    and chunks[-1][0] + chunks[-1][1] == start
                    and chunks[-1][2:] == values
                ):
                    previous = chunks[-1]
                    chunks[-1] = (previous[0], previous[1] + length, *values)
                else:
                    chunks.append((start, length, *values))
        iterator += 1

    containers = []
    index = 0
    while index < len(chunks):
        start, length, container_id, unit_id, ruby_type, text, position = chunks[index]
        unit_chunks = [(start, length, unit_id, text)]
        end = start + length
        index += 1
        while index < len(chunks):
            candidate = chunks[index]
            if (
                candidate[0] != end
                or candidate[2] != container_id
                or candidate[4] != ruby_type
                or candidate[6] != position
            ):
                break
            c_start, c_length, _rid, c_unit, _type, c_text, _position = candidate
            if unit_chunks[-1][2:] == (c_unit, c_text):
                old = unit_chunks[-1]
                unit_chunks[-1] = (old[0], old[1] + c_length, c_unit, c_text)
            else:
                unit_chunks.append((c_start, c_length, c_unit, c_text))
            end += c_length
            index += 1

        units = tuple(RubyUnitRange(*unit) for unit in unit_chunks)
        block_text = block.text()
        valid = (
            ruby_type == 'group' and len(units) == 1
            or ruby_type == 'mono' and all(
                len(_grapheme_ranges(_utf16_slice(
                    block_text,
                    unit.start - block.position(),
                    unit.length,
                ))) == 1
                for unit in units
            )
        )
        if valid:
            containers.append(RubyContainerRange(
                start,
                end - start,
                container_id,
                ruby_type,
                position,
                units,
            ))
    return tuple(containers)


def ruby_containers(document: QTextDocument) -> tuple[RubyContainerRange, ...]:
    """Return every valid document-local Ruby container in order."""
    containers = []
    block = document.firstBlock()
    while block.isValid():
        containers.extend(ruby_containers_in_block(block))
        block = block.next()
    return tuple(containers)


def _discard_ruby_tate_overlaps(document: QTextDocument) -> None:
    overlapping = [
        container
        for container in ruby_containers(document)
        if _range_has_text_combine(document, container.start, container.end)
    ]
    if not overlapping:
        return
    LOGGER.warning('Discarding Ruby annotation overlapping Tate-chu-yoko')
    cursor = QTextCursor(document)
    for container in overlapping:
        cursor.setPosition(container.start)
        cursor.setPosition(container.end, QTextCursor.MoveMode.KeepAnchor)
        cursor.mergeCharFormat(_clear_ruby_modifier())


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
                spacing = canonical_letter_spacing(
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
            font_variant_ligatures=font_variant_ligatures_value(
                char_format
            ),
            font_variant_numeric=font_variant_numeric_value(char_format),
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


def _css_line_height(
    value: float,
    spacing_type: LineSpacingType,
) -> str:
    if spacing_type == LineSpacingType.Proportional:
        return _format_spacing_number(value)
    distance = _format_spacing_number(value * 10.0)
    return f'calc(1em + {distance}px)'


def _line_spacing_start_tag(
    tag: str,
    attrs: list,
    values: Optional[tuple[float, LineSpacingType]],
) -> str:
    attributes = {str(name).lower(): raw for name, raw in attrs}
    declarations = ()
    if values is not None:
        value, spacing_type = values
        declarations = (
            f'line-height: {_css_line_height(value, spacing_type)}',
        )
    style = _replace_style_declarations(
        attributes.get('style'), declarations
    )
    replacements = {'style': style} if style else {}
    if values is not None and spacing_type == LineSpacingType.Distance:
        replacements[LINE_DISTANCE_ATTRIBUTE] = _format_spacing_number(value)
    rewritten = [
        (name, raw)
        for name, raw in attrs
        if str(name).lower() not in {'style', LINE_DISTANCE_ATTRIBUTE}
    ]
    rewritten.extend(replacements.items())
    return _start_tag(tag, rewritten)


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
    if (
        extension.font_variant_ligatures
        != FONT_VARIANT_LIGATURES_NORMAL
    ):
        styles.append(
            'font-variant-ligatures: '
            f'{extension.font_variant_ligatures}'
        )
    if extension.font_variant_numeric != FONT_VARIANT_NUMERIC_NORMAL:
        styles.append(
            'font-variant-numeric: '
            f'{extension.font_variant_numeric}'
        )
    style_attribute = f'style="{"; ".join(styles)};"'
    suffix = ' '.join((style_attribute, *attributes))
    return f'<span {suffix}>{text}</span>'


class _InlineExtensionHTMLExporter(HTMLParser):
    """Inject rich-text extensions without replacing Qt's HTML serializer.

    >>> _format_spacing_number(1.15)
    '1.15'
    """

    def __init__(
        self,
        blocks: tuple[QTextBlock, ...],
        ranges: list[tuple[int, int, _InlineExtension]],
        line_spacing_fallback: Optional[tuple[float, LineSpacingType]],
    ) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks = blocks
        self.ranges = ranges
        self.range_index = 0
        self.block_index = -1
        self.block_offset = 0
        self.in_block = False
        self.output: list[str] = []
        self.line_spacing_fallback = line_spacing_fallback

    def _start_block(self) -> None:
        self.block_index += 1
        self.block_offset = 0
        self.in_block = True

    def _current_block(self) -> Optional[QTextBlock]:
        if 0 <= self.block_index < len(self.blocks):
            return self.blocks[self.block_index]
        return None

    def _current_line_spacing(
        self,
    ) -> Optional[tuple[float, LineSpacingType]]:
        block = self._current_block()
        if block is None:
            return None
        return (
            _native_line_spacing_values(block.blockFormat())
            or self.line_spacing_fallback
        )

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
        raw = self.get_starttag_text() or _start_tag(tag, attrs)
        if tag in ('p', 'li'):
            self._start_block()
            values = self._current_line_spacing()
            if values is not None or any(
                str(name).lower() == LINE_DISTANCE_ATTRIBUTE
                for name, _value in attrs
            ):
                raw = _line_spacing_start_tag(tag, attrs, values)
        self.output.append(raw)
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
    line_spacing_fallback: Optional[float],
    line_spacing_type_fallback: int,
) -> str:
    ranges = _inline_extension_ranges(document)
    blocks = _document_blocks(document)
    fallback = canonical_line_spacing(line_spacing_fallback)
    fallback_type = canonical_line_spacing_type(
        line_spacing_type_fallback
    )
    fallback_pair = (
        (fallback, fallback_type)
        if fallback is not None and fallback_type is not None
        else None
    )
    needs_line_spacing = (
        fallback_pair is not None
        or LINE_DISTANCE_ATTRIBUTE in html.lower()
        or any(
            values is not None and values[1] == LineSpacingType.Distance
            for values in (
                _native_line_spacing_values(block.blockFormat())
                for block in blocks
            )
        )
    )
    if not ranges and not needs_line_spacing:
        return html
    parser = _InlineExtensionHTMLExporter(
        blocks,
        ranges,
        fallback_pair,
    )
    try:
        parser.feed(html)
        parser.close()
    except (ValueError, TypeError) as error:
        LOGGER.warning('Unable to export rich-text extensions: %s', error)
        return html
    return ''.join(parser.output)


_HTML_BLOCK_TAGS = {
    'html', 'head', 'body', 'meta', 'style', 'title', 'p', 'div', 'li',
    'ul', 'ol', 'table', 'tbody', 'thead', 'tfoot', 'tr', 'td', 'th',
}
_HTML_VOID_TAGS = {'br', 'img', 'meta', 'hr', 'input', 'link'}


class _RubySemanticHTMLExporter(HTMLParser):
    """Insert balanced semantic Ruby around Qt's ordinary inline markup."""

    def __init__(
        self,
        document: QTextDocument,
        containers: tuple[RubyContainerRange, ...],
    ) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks = _document_blocks(document)
        self.containers = containers
        self.container_index = 0
        self.active: Optional[RubyContainerRange] = None
        self.unit_index = 0
        self.block_index = -1
        self.block_offset = 0
        self.in_block = False
        self.inline_stack: list[tuple[str, str]] = []
        self.output: list[str] = []

    def _position(self) -> int:
        if 0 <= self.block_index < len(self.blocks):
            return self.blocks[self.block_index].position() + self.block_offset
        return -1

    def _close_inline(self) -> None:
        self.output.extend(f'</{tag}>' for tag, _raw in reversed(self.inline_stack))

    def _reopen_inline(self) -> None:
        self.output.extend(raw for _tag, raw in self.inline_stack)

    def _before_text(self) -> None:
        if self.active is not None or self.container_index >= len(self.containers):
            return
        candidate = self.containers[self.container_index]
        if self._position() != candidate.start:
            return
        self._close_inline()
        merge = 'merge' if candidate.ruby_type == 'group' else 'separate'
        self.output.append(
            '<ruby style="'
            f'ruby-position: {candidate.position}; '
            'ruby-align: space-around; '
            f'ruby-merge: {merge}; '
            'ruby-overhang: none;">'
        )
        self.output.append('<span>')
        self._reopen_inline()
        self.active = candidate
        self.unit_index = 0

    def _after_text(self) -> None:
        if self.active is None:
            return
        unit = self.active.units[self.unit_index]
        if self._position() != unit.end:
            return
        self._close_inline()
        self.output.append(f'</span><rt>{escape(unit.text)}</rt>')
        self.unit_index += 1
        if self._position() == self.active.end:
            self.output.append('</ruby>')
            self.active = None
            self.container_index += 1
        else:
            self.output.append('<span>')
        self._reopen_inline()

    def _append_character(self, raw: str, decoded: str) -> None:
        if not self.in_block:
            self.output.append(raw)
            return
        self._before_text()
        self.output.append(raw)
        self.block_offset += _utf16_length(decoded)
        self._after_text()

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        raw = self.get_starttag_text() or _start_tag(tag, attrs)
        if tag in ('p', 'li'):
            self.block_index += 1
            self.block_offset = 0
            self.in_block = True
            self.inline_stack.clear()
        self.output.append(raw)
        if (
            self.in_block
            and tag not in _HTML_BLOCK_TAGS
            and tag not in _HTML_VOID_TAGS
        ):
            self.inline_stack.append((tag, raw))
        if tag in ('br', 'img') and self.in_block:
            self.block_offset += 1

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _HTML_VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        self.output.append(f'</{tag}>')
        if self.inline_stack and self.inline_stack[-1][0] == tag:
            self.inline_stack.pop()
        if tag in ('p', 'li'):
            self.in_block = False
            self.inline_stack.clear()

    def handle_data(self, data: str) -> None:
        for character in data:
            self._append_character(character, character)

    def handle_entityref(self, name: str) -> None:
        raw = f'&{name};'
        self._append_character(raw, unescape(raw))

    def handle_charref(self, name: str) -> None:
        raw = f'&#{name};'
        self._append_character(raw, unescape(raw))

    def handle_comment(self, data: str) -> None:
        self.output.append(f'<!--{data}-->')

    def handle_decl(self, decl: str) -> None:
        self.output.append(
            '<!DOCTYPE html>'
            if decl.lower().startswith('doctype html')
            else f'<!{decl}>'
        )


def _add_semantic_ruby(document: QTextDocument, html: str) -> str:
    containers = ruby_containers(document)
    if not containers:
        return html
    parser = _RubySemanticHTMLExporter(document, containers)
    try:
        parser.feed(html)
        parser.close()
    except (IndexError, TypeError, ValueError) as error:
        LOGGER.warning('Unable to export Ruby annotations: %s', error)
        return html
    if parser.active is not None or parser.container_index != len(containers):
        LOGGER.warning('Unable to map all Ruby annotations during export')
        return html
    result = ''.join(parser.output)
    # Boundary balancing can reopen a Qt span immediately before its authored
    # close tag. Empty spans carry no document state, so omit that serializer
    # artifact from the canonical representation.
    return re.sub(r'<span\b[^>]*></span>', '', result, flags=re.IGNORECASE)


def to_rich_text_html(
    document: QTextDocument,
    html: Optional[str] = None,
    *,
    line_spacing_fallback: Optional[float] = None,
    line_spacing_type_fallback: int = LineSpacingType.Proportional,
) -> str:
    """Extend Qt HTML with semantic inline and paragraph formatting."""
    if html is None:
        html = document.toHtml()
    extended_html = _add_semantic_ruby(
        document,
        _add_inline_extensions(
            document,
            html,
            line_spacing_fallback,
            line_spacing_type_fallback,
        ),
    )
    return restore_project_font_families_in_html(
        export_font_weight_html(extended_html, qt6=QT6)
    )


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
    value = canonical_letter_spacing(
        char_format.property(AnnotationProperty.LETTER_SPACING)
    )
    if value is not None:
        return value
    fallback_value = canonical_letter_spacing(fallback)
    return 1.0 if fallback_value is None else fallback_value


def apply_letter_spacing(
    cursor: QTextCursor,
    value: float,
    *,
    vertical: bool,
) -> None:
    """Merge character spacing into a selection or insertion format."""
    canonical_value = canonical_letter_spacing(value)
    if canonical_value is None:
        raise ValueError(f'unsupported letter spacing: {value!r}')
    _rewrite_cursor_char_formats(
        cursor,
        lambda char_format: _set_semantic_letter_spacing(
            char_format,
            canonical_value,
            vertical,
        ),
    )


def apply_ligature_axis(
    cursor: QTextCursor,
    axis: str,
    state: str,
    *,
    vertical: bool,
) -> None:
    """Apply one CSS ligature axis to a selection or insertion format."""
    _rewrite_cursor_char_formats(
        cursor,
        lambda char_format: set_ligature_axes(
            char_format,
            {axis: state},
            vertical=vertical,
        ),
    )


def set_ligature_axes(
    char_format: QTextCharFormat,
    states: dict[str, str],
    *,
    vertical: bool,
) -> None:
    """Set semantic ligature axes on one character format.

    >>> fmt = QTextCharFormat()
    >>> set_ligature_axes(
    ...     fmt, {LIGATURE_DISCRETIONARY: LIGATURE_ENABLED}, vertical=False
    ... )
    >>> ligature_axis_value(fmt, LIGATURE_DISCRETIONARY)
    'enabled'
    """
    value = font_variant_ligatures_value(char_format)
    for axis, state in states.items():
        if axis not in _LIGATURE_AXIS_TOKENS:
            raise ValueError(f'unsupported ligature axis: {axis!r}')
        if state not in LIGATURE_AXIS_VALUES:
            raise ValueError(f'unsupported ligature axis value: {state!r}')
        value = _font_variant_ligatures_with_axis(value, axis, state)
    if value == FONT_VARIANT_LIGATURES_NORMAL:
        char_format.clearProperty(
            AnnotationProperty.FONT_VARIANT_LIGATURES
        )
    else:
        char_format.setProperty(
            AnnotationProperty.FONT_VARIANT_LIGATURES,
            value,
        )
    sync_native_ligature_shaping(char_format, vertical=vertical)


def set_oldstyle_nums(char_format: QTextCharFormat, state: str) -> None:
    """Set the semantic oldstyle-figure state on one character format.

    >>> fmt = QTextCharFormat()
    >>> set_oldstyle_nums(fmt, LIGATURE_ENABLED)
    >>> oldstyle_nums_value(fmt)
    'enabled'
    """
    if state not in LIGATURE_AXIS_VALUES:
        raise ValueError(f'unsupported oldstyle figure value: {state!r}')
    if state == LIGATURE_ENABLED:
        value = FONT_VARIANT_NUMERIC_OLDSTYLE
    elif state == LIGATURE_DISABLED:
        value = FONT_VARIANT_NUMERIC_LINING
    else:
        value = FONT_VARIANT_NUMERIC_NORMAL
    if value == FONT_VARIANT_NUMERIC_NORMAL:
        char_format.clearProperty(AnnotationProperty.FONT_VARIANT_NUMERIC)
    else:
        char_format.setProperty(
            AnnotationProperty.FONT_VARIANT_NUMERIC,
            value,
        )
    sync_native_oldstyle_nums(char_format)


def apply_oldstyle_nums(cursor: QTextCursor, state: str) -> None:
    """Apply oldstyle figures to a selection or insertion format."""
    _rewrite_cursor_char_formats(
        cursor,
        lambda char_format: set_oldstyle_nums(char_format, state),
    )


def set_document_letter_spacing_writing_mode(
    document: QTextDocument,
    *,
    vertical: bool,
    fallback: float,
) -> None:
    """Preserve semantic spacing while changing its Qt shaping value."""
    fallback_value = canonical_letter_spacing(fallback)
    fallback = 1.0 if fallback_value is None else fallback_value
    cursor = QTextCursor(document)
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.beginEditBlock()
    try:
        if document.isEmpty():
            _apply_document_letter_spacing(document, fallback, vertical)
        else:
            _rewrite_cursor_char_formats(
                cursor,
                lambda char_format: _set_semantic_letter_spacing(
                    char_format,
                    letter_spacing_value(char_format, fallback),
                    vertical,
                ),
            )
    finally:
        cursor.endEditBlock()


def _text_combine_modifier(enabled: bool) -> QTextCharFormat:
    modifier = QTextCharFormat()
    modifier.setProperty(
        AnnotationProperty.TEXT_COMBINE_UPRIGHT,
        TEXT_COMBINE_ALL if enabled else TEXT_COMBINE_NONE,
    )
    modifier.setProperty(
        AnnotationProperty.TEXT_COMBINE_ID,
        uuid4().hex if enabled else '',
    )
    return modifier


def apply_text_combine_upright(
    cursor: QTextCursor,
    enabled: bool,
    *,
    vertical: bool = True,
) -> None:
    """Apply one combined-run ID to a selection or insertion format."""
    if enabled:
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        if any(
            container.start < end and start < container.end
            if cursor.hasSelection()
            else container.start <= cursor.position() < container.end
            for container in ruby_containers(cursor.document())
        ):
            raise RubyValidationError('Tate-chu-yoko cannot overlap Ruby')
    modifier = _text_combine_modifier(enabled)
    cursor.mergeCharFormat(modifier)
    _rewrite_cursor_char_formats(
        cursor,
        lambda char_format: sync_native_ligature_shaping(
            char_format,
            vertical=vertical,
        ),
    )


def apply_auto_text_combine_upright(
    document: QTextDocument,
    allowed_characters: AbstractSet[str],
    max_length: int,
) -> bool:
    """Replace Tate-chu-yoko with qualified non-Ruby character runs.

    ``allowed_characters`` is compiled once by the pipeline and reused for
    every block on the page. Return whether the document changed.

    >>> document = QTextDocument('12')
    >>> apply_auto_text_combine_upright(document, frozenset('12'), 2)
    True
    >>> apply_auto_text_combine_upright(document, frozenset('12'), 2)
    True
    """
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 1
    ):
        raise ValueError(f'unsupported maximum run length: {max_length!r}')

    old_ranges = []
    block = document.firstBlock()
    while block.isValid():
        old_ranges.extend(
            (block.position() + start, length)
            for start, length, _group_id in text_combine_upright_ranges(block)
        )
        block = block.next()

    new_ranges = []
    if allowed_characters:
        ruby_ranges = ruby_containers(document)
        ruby_index = 0
        block = document.firstBlock()
        while block.isValid():
            run_start = -1
            run_length = 0
            utf16_offset = 0
            runs = []
            for character in block.text():
                if character in allowed_characters:
                    if run_start < 0:
                        run_start = utf16_offset
                    run_length += 1
                elif run_start >= 0:
                    runs.append((run_start, utf16_offset - run_start, run_length))
                    run_start = -1
                    run_length = 0
                utf16_offset += _utf16_length(character)
            if run_start >= 0:
                runs.append((run_start, utf16_offset - run_start, run_length))

            for local_start, utf16_length, character_count in runs:
                if character_count > max_length:
                    continue
                start = block.position() + local_start
                end = start + utf16_length
                while (
                    ruby_index < len(ruby_ranges)
                    and ruby_ranges[ruby_index].end <= start
                ):
                    ruby_index += 1
                if (
                    ruby_index < len(ruby_ranges)
                    and ruby_ranges[ruby_index].start < end
                ):
                    continue
                new_ranges.append((start, utf16_length))
            block = block.next()

    if not old_ranges and not new_ranges:
        return False

    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    try:
        clear_format = _text_combine_modifier(False)
        for start, length in old_ranges:
            cursor.setPosition(start)
            cursor.setPosition(
                start + length,
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.mergeCharFormat(clear_format)
        for start, length in new_ranges:
            cursor.setPosition(start)
            cursor.setPosition(
                start + length,
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.mergeCharFormat(_text_combine_modifier(True))
        for start, length in sorted({*old_ranges, *new_ranges}):
            cursor.setPosition(start)
            cursor.setPosition(
                start + length,
                QTextCursor.MoveMode.KeepAnchor,
            )
            _rewrite_cursor_char_formats(
                cursor,
                lambda char_format: sync_native_ligature_shaping(
                    char_format,
                    vertical=True,
                ),
            )
    finally:
        cursor.endEditBlock()
    return True


class RubyValidationError(ValueError):
    """A requested Ruby edit cannot be represented by the first version."""


def ruby_container_for_cursor(
    cursor: QTextCursor,
) -> Optional[RubyContainerRange]:
    """Return the single complete container identified by a caret/selection."""
    start = cursor.selectionStart()
    end = cursor.selectionEnd()
    for container in ruby_containers(cursor.document()):
        if cursor.hasSelection():
            if container.start <= start and end <= container.end:
                return container
        elif container.start <= cursor.position() < container.end:
            return container
    return None


def ruby_containers_intersecting_cursor(
    cursor: QTextCursor,
) -> tuple[RubyContainerRange, ...]:
    """Return every container touched by the selection or caret."""
    start = cursor.selectionStart()
    end = cursor.selectionEnd()
    if cursor.hasSelection():
        return tuple(
            container
            for container in ruby_containers(cursor.document())
            if container.start < end and container.end > start
        )
    return tuple(
        container
        for container in ruby_containers(cursor.document())
        if container.start <= cursor.position() < container.end
    )


def _ruby_readings(
    ruby_type: str,
    text: str,
    base_text: str,
) -> tuple[tuple[int, int, str], ...]:
    if ruby_type not in RUBY_TYPES:
        raise RubyValidationError(f'unsupported Ruby type: {ruby_type!r}')
    if not isinstance(text, str) or not text.strip():
        raise RubyValidationError('Ruby text cannot be empty')
    graphemes = _grapheme_ranges(base_text)
    if ruby_type == 'group':
        return ((0, _utf16_length(base_text), text.strip()),)
    readings = text.split()
    if len(readings) != len(graphemes):
        raise RubyValidationError(
            'Mono Ruby needs one whitespace-separated reading per base grapheme'
        )
    return tuple(
        (start, end - start, reading)
        for (start, end), reading in zip(graphemes, readings)
    )


def _range_has_text_combine(
    document: QTextDocument,
    start: int,
    end: int,
) -> bool:
    block = document.findBlock(start)
    while block.isValid() and block.position() < end:
        for local_start, length, _group_id in text_combine_upright_ranges(block):
            absolute_start = block.position() + local_start
            if absolute_start < end and absolute_start + length > start:
                return True
        block = block.next()
    return False


def _ruby_modifier(
    container_id: str,
    unit_id: str,
    ruby_type: str,
    text: str,
    position: str,
) -> QTextCharFormat:
    modifier = QTextCharFormat()
    for prop, value in (
        (AnnotationProperty.RUBY_ID, container_id),
        (AnnotationProperty.RUBY_UNIT_ID, unit_id),
        (AnnotationProperty.RUBY_TYPE, ruby_type),
        (AnnotationProperty.RUBY_TEXT, text),
        (AnnotationProperty.RUBY_POSITION, position),
    ):
        modifier.setProperty(prop, value)
    return modifier


def _clear_ruby_modifier() -> QTextCharFormat:
    return _ruby_modifier('', '', '', '', '')


def apply_ruby(
    cursor: QTextCursor,
    ruby_type: str,
    text: str,
    position: str = DEFAULT_RUBY_POSITION,
) -> RubyContainerRange:
    """Create or update one Ruby container as one native undo transaction.

    >>> issubclass(RubyValidationError, ValueError)
    True
    """
    if position not in RUBY_POSITIONS:
        raise RubyValidationError(f'unsupported Ruby position: {position!r}')
    document = cursor.document()
    existing = ruby_container_for_cursor(cursor)
    selection_start = cursor.selectionStart()
    selection_end = cursor.selectionEnd()
    if existing is None and not cursor.hasSelection():
        raise RubyValidationError('Select non-empty base text before applying Ruby')
    if existing is not None:
        start, end = existing.start, existing.end
    else:
        start, end = selection_start, selection_end

    start_block = document.findBlock(start)
    end_block = document.findBlock(max(start, end - 1))
    if (
        not start_block.isValid()
        or start_block != end_block
        or end <= start
    ):
        raise RubyValidationError(
            'Ruby base text cannot contain paragraph or forced line breaks'
        )
    base_text = _utf16_slice(
        start_block.text(), start - start_block.position(), end - start
    )
    if any(separator in base_text for separator in ('\n', '\r', '\u2028', '\u2029')):
        raise RubyValidationError(
            'Ruby base text cannot contain paragraph or forced line breaks'
        )
    readings = _ruby_readings(ruby_type, text, base_text)
    if _range_has_text_combine(document, start, end):
        raise RubyValidationError('Ruby cannot overlap Tate-chu-yoko')

    overlaps = tuple(
        container
        for container in ruby_containers(document)
        if container.start < end and container.end > start
    )
    if overlaps and (
        existing is None
        or len(overlaps) != 1
        or overlaps[0].container_id != existing.container_id
    ):
        raise RubyValidationError('Ruby cannot partially overlap an existing container')

    container_id = uuid4().hex
    work = QTextCursor(document)
    work.beginEditBlock()
    try:
        if existing is not None:
            work.setPosition(existing.start)
            work.setPosition(existing.end, QTextCursor.MoveMode.KeepAnchor)
            work.mergeCharFormat(_clear_ruby_modifier())
        for offset, length, reading in readings:
            work.setPosition(start + offset)
            work.setPosition(
                start + offset + length,
                QTextCursor.MoveMode.KeepAnchor,
            )
            work.mergeCharFormat(_ruby_modifier(
                container_id,
                uuid4().hex,
                ruby_type,
                reading,
                position,
            ))
    finally:
        work.endEditBlock()
    result = next(
        container
        for container in ruby_containers(document)
        if container.container_id == container_id
    )
    return result


def remove_ruby(cursor: QTextCursor) -> bool:
    """Remove every Ruby container intersecting ``cursor`` in one undo step."""
    containers = ruby_containers_intersecting_cursor(cursor)
    if not containers:
        return False
    work = QTextCursor(cursor.document())
    work.beginEditBlock()
    try:
        for container in containers:
            work.setPosition(container.start)
            work.setPosition(
                container.end, QTextCursor.MoveMode.KeepAnchor
            )
            work.mergeCharFormat(_clear_ruby_modifier())
    finally:
        work.endEditBlock()
    return True


def prepare_ruby_insertion(cursor: QTextCursor, text: str = '') -> None:
    """Prepare text insertion/replacement without creating invalid Ruby."""
    document = cursor.document()
    start = cursor.selectionStart()
    end = cursor.selectionEnd()
    has_break = any(
        separator in text for separator in ('\n', '\r', '\u2028', '\u2029')
    )
    clear_ranges = []
    for container in ruby_containers(document):
        overlaps = (
            container.start < end and start < container.end
            if cursor.hasSelection()
            else container.start < cursor.position() < container.end
        )
        if not overlaps:
            continue
        if has_break and container.ruby_type == 'group':
            clear_ranges.append((container.start, container.end))
        elif cursor.hasSelection() and container.ruby_type == 'mono':
            clear_ranges.extend(
                (unit.start, unit.end)
                for unit in container.units
                if unit.start < end and start < unit.end
            )
    if clear_ranges:
        work = QTextCursor(document)
        for range_start, range_end in clear_ranges:
            work.setPosition(range_start)
            work.setPosition(range_end, QTextCursor.MoveMode.KeepAnchor)
            work.mergeCharFormat(_clear_ruby_modifier())

    if cursor.hasSelection():
        return
    position = cursor.position()
    inherited = ruby_values(cursor.charFormat())
    if not inherited[0]:
        return
    keep = any(
        container.ruby_type == 'group'
        and container.start < position < container.end
        and not has_break
        for container in ruby_containers(document)
    )
    if keep:
        return
    char_format = QTextCharFormat(cursor.charFormat())
    for prop in (
        AnnotationProperty.RUBY_ID,
        AnnotationProperty.RUBY_UNIT_ID,
        AnnotationProperty.RUBY_TYPE,
        AnnotationProperty.RUBY_TEXT,
        AnnotationProperty.RUBY_POSITION,
    ):
        char_format.setProperty(prop, '')
    cursor.setCharFormat(char_format)


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


def _remap_ruby_ids(document: QTextDocument) -> None:
    """Give pasted Ruby containers and units fresh document-local IDs."""
    container_ids = {}
    unit_ids = {}
    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    try:
        for container in ruby_containers(document):
            replacement_container = container_ids.setdefault(
                container.container_id, uuid4().hex
            )
            for unit in container.units:
                replacement_unit = unit_ids.setdefault(
                    (container.container_id, unit.unit_id), uuid4().hex
                )
                modifier = QTextCharFormat()
                modifier.setProperty(
                    AnnotationProperty.RUBY_ID, replacement_container
                )
                modifier.setProperty(
                    AnnotationProperty.RUBY_UNIT_ID, replacement_unit
                )
                cursor.setPosition(unit.start)
                cursor.setPosition(unit.end, QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(modifier)
    finally:
        cursor.endEditBlock()


def create_rich_text_mime(
    cursor: QTextCursor,
    *,
    line_spacing_fallback: Optional[float] = None,
    line_spacing_type_fallback: int = LineSpacingType.Proportional,
) -> QMimeData:
    """Create interoperable clipboard data with exact inline extensions."""
    mime = QMimeData()
    if not cursor.hasSelection():
        return mime
    document = QTextDocument()
    document.setUndoRedoEnabled(False)
    target = QTextCursor(document)
    target.insertFragment(QTextDocumentFragment(cursor))
    extended_html = to_rich_text_html(
        document,
        line_spacing_fallback=line_spacing_fallback,
        line_spacing_type_fallback=line_spacing_type_fallback,
    )
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
    _remap_ruby_ids(document)
    pasted_text = document.toPlainText()
    pasted_has_ruby = bool(ruby_containers(document))
    pasted_has_tate = any(
        text_combine_upright_ranges(block)
        for block in _document_blocks(document)
    )
    target_start = cursor.selectionStart()
    target_end = cursor.selectionEnd()
    surrounding_group = None
    for container in ruby_containers(cursor.document()):
        if container.ruby_type != 'group':
            continue
        contained_selection = (
            cursor.hasSelection()
            and container.start <= target_start
            and target_end <= container.end
        )
        interior_caret = (
            not cursor.hasSelection()
            and container.start < cursor.position() < container.end
        )
        if contained_selection or interior_caret:
            surrounding_group = container
            break
    has_break = any(
        separator in pasted_text
        for separator in ('\n', '\r', '\u2028', '\u2029')
    )
    inherit_group = (
        surrounding_group is not None
        and not has_break
        and not pasted_has_ruby
        and not pasted_has_tate
    )

    cursor.beginEditBlock()
    try:
        if surrounding_group is not None and (pasted_has_ruby or pasted_has_tate):
            work = QTextCursor(cursor.document())
            work.setPosition(surrounding_group.start)
            work.setPosition(
                surrounding_group.end, QTextCursor.MoveMode.KeepAnchor
            )
            work.mergeCharFormat(_clear_ruby_modifier())
        prepare_ruby_insertion(cursor, pasted_text)
        insertion_start = cursor.selectionStart()
        cursor.insertFragment(QTextDocumentFragment(document))
        insertion_end = cursor.position()
        if inherit_group and insertion_end > insertion_start:
            unit = surrounding_group.units[0]
            work = QTextCursor(cursor.document())
            work.setPosition(insertion_start)
            work.setPosition(
                insertion_end, QTextCursor.MoveMode.KeepAnchor
            )
            work.mergeCharFormat(_ruby_modifier(
                surrounding_group.container_id,
                unit.unit_id,
                'group',
                unit.text,
                surrounding_group.position,
            ))
    finally:
        cursor.endEditBlock()
    return True
