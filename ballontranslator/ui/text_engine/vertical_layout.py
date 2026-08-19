import re
import unicodedata
from bisect import bisect_right
from functools import lru_cache
from typing import List, Optional, Tuple

from qtpy.QtCore import QPointF, QRectF, QSizeF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QBrush,
    QPainter,
    QTextBlock,
    QTextCharFormat,
    QTextDocument,
    QTextLine,
    QTextOption,
    QTransform,
)

from ballontranslator.utils import shared as C
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import FontFormat, TextAlignment
from .annotations import text_combine_upright_ranges
from .cache import KeyedLruCache
from .layout import (
    CharFontFormat,
    SceneTextLayout,
    _block_cursor_position,
    paint_context_without_selection_ranges,
)
from .rendering.emphasis import (
    draw_emphasis_marks,
    emphasis_ink_bounds,
    emphasis_margins,
)
from .rendering.glyph import draw_slanted_line, glyph_geometry
from .rendering.indexing import (
    _grapheme_count,
    _grapheme_ranges,
    _utf16_char_at,
    _utf16_length,
    _utf16_slice,
)
from .rendering.tate_chu_yoko import (
    tate_chu_yoko_ink_bounds,
    tate_chu_yoko_natural_bounds,
    tate_chu_yoko_transform,
)
from .rendering.ruby import (
    RubyBlockMetrics,
    RubyPlacement,
    RubyUnitMetrics,
    clear_horizontal_ruby_layout,
    draw_ruby_placement,
    ruby_placement,
    ruby_side_margins,
    vertical_ruby_metrics,
)

PUNSET_HALF = {chr(i) for i in range(0x21, 0x7F)}

# CLREQ Appendix A: pause/stop marks stay upright, while parenthetical
# punctuation, dashes, ellipses, connectors, and indicators rotate.
PUNSET_PAUSEORSTOP = {
    '。', '．', '，', '、', '：', '；', '！', '‼', '？', '⁇', '⁈', '⁉',
}
PUNSET_ALIGNCENTER = {'·', '・', '‧', '●', '•'}
# ‶ pairs with either 〟 or ″ as the closing mark.
PUNSET_BRACKETL = {'「', '『', '“', '‘', '‶', '（', '《', '〈', '【', '〖', '〔', '［', '｛', '('}
PUNSET_BRACKETR = {'」', '』', '”', '’', '〟', '″', '）', '》', '〉', '】', '〗', '〕', '］', '｝', ')'}
PUNSET_BRACKET = PUNSET_BRACKETL.union(PUNSET_BRACKETR)
PUNSET_COMPACT = PUNSET_PAUSEORSTOP.union(PUNSET_BRACKET)

PUNSET_INSEPARABLE_REPEAT = {'—', '―', '‥', '…', '⋯'}
PUNSET_NONBRACKET = {'⸺', '…', '⋯', '～', '-', '–', '—', '＿', '﹏', '~'}
PUNSET_VERNEEDROTATE = (
    PUNSET_NONBRACKET
    | PUNSET_BRACKET
    | PUNSET_HALF
    | PUNSET_INSEPARABLE_REPEAT
)
PUNSET_STANDARD_VERTICAL_ROMAN = (
    PUNSET_VERNEEDROTATE - PUNSET_HALF
) | PUNSET_NONBRACKET
_STANDARD_SHAPED_ROTATION_CHARS = ''.join(
    sorted(PUNSET_STANDARD_VERTICAL_ROMAN)
)
_SHAPED_ROTATION_CHARS = ''.join(sorted(PUNSET_VERNEEDROTATE))
# The shared glyph cache owns full paths at layout-specific offsets. Vertical
# settlement needs only normalized bounds reusable across documents and lines.
LINE_INK_BOUNDS_CACHE_MAX_ENTRIES = 2048
_LINE_INK_BOUNDS_CACHE: KeyedLruCache[tuple, QRectF] = KeyedLruCache(
    LINE_INK_BOUNDS_CACHE_MAX_ENTRIES
)

PUNSET_ROTATE_ALIGNL = {'」', '』', '”', '’', '〟', '″'}
PUNSET_ROTATE_ALIGNR = {'「', '『', '“', '‘', '‶'}

Dingbats_vertical_aligncenter = r'\u2700-\u275A\u2761-\u2767\u2776-\u27BF'
Miscellaneous_Symbols_Pattern = r'\u2600-\u26FF'  # align center in vertical mode

vertical_force_aligncentel_pattern = re.compile('[' + Dingbats_vertical_aligncenter + Miscellaneous_Symbols_Pattern + r'⁁⁂⁇⁈⁉⁊⁋⁎※⁑⁒⁕⁖⁘⁙⁛⁜‼‽]')


@lru_cache(maxsize=512)
def _is_non_fullwidth_roman(char: str) -> bool:
    """Return whether a glyph follows the item-wide Roman orientation.

    >>> _is_non_fullwidth_roman('A')
    True
    >>> _is_non_fullwidth_roman('Ａ')
    False
    """
    if char in PUNSET_HALF:
        return True
    if unicodedata.east_asian_width(char) in {'F', 'W'}:
        return False
    name = unicodedata.name(char, '')
    return name.startswith('LATIN ') or name.startswith('ROMAN NUMERAL ')


def _inseparable_punctuation_run(
    text: str,
    start: int,
) -> Optional[Tuple[int, int]]:
    """Return the UTF-16 start and columns of a repeated punctuation run.

    >>> _inseparable_punctuation_run(' ……', 0)
    (1, 2)
    >>> _inseparable_punctuation_run('—…', 0) is None
    True
    """
    text_length = _utf16_length(text)
    run_start = start
    while run_start < text_length:
        char = _utf16_char_at(text, run_start)
        if not char.isspace():
            break
        run_start += 2 if ord(char) > 0xFFFF else 1
    if run_start >= text_length:
        return None
    mark = _utf16_char_at(text, run_start)
    if mark not in PUNSET_INSEPARABLE_REPEAT:
        return None
    columns = 0
    offset = run_start
    while offset < text_length and _utf16_char_at(text, offset) == mark:
        columns += 1
        offset += 1
    if columns < 2:
        return None
    return run_start, columns


def _single_glyph_character(
    line: QTextLine,
    candidates: str,
) -> Optional[str]:
    """Return the encoded candidate matching a one-glyph shaped line.

    >>> _single_glyph_character(QTextLine(), '「') is None
    True
    """
    if not line.isValid():
        return None
    shaped = [
        (run, int(glyph_index))
        for run in line.glyphRuns()
        for glyph_index in run.glyphIndexes()
    ]
    if len(shaped) != 1 or shaped[0][1] == 0:
        return None
    run, glyph_index = shaped[0]
    raw_font = run.rawFont()
    if not raw_font.isValid():
        return None
    for char, candidate_index in zip(
        candidates,
        raw_font.glyphIndexesForString(candidates),
    ):
        if int(candidate_index) == glyph_index:
            return char
    return None


def _uncached_line_ink_bounds(
    line: QTextLine,
    space_shift: float = 0.0,
) -> QRectF:
    """Return shaped vector ink normalized to the line origin.

    >>> callable(_uncached_line_ink_bounds)
    True
    """
    return glyph_geometry(
        line,
        line.textStart(),
        line.textLength(),
        QPointF(-line.x() - space_shift, -line.y()),
        QTransform(),
        0.0,
    ).bounds


def _line_ink_cache_key(
    line: QTextLine,
    space_shift: float,
) -> Optional[tuple]:
    """Describe exact shaped ink independently of line placement.

    >>> _line_ink_cache_key(QTextLine(), 0.0) is None
    True
    """
    if not line.isValid():
        return None
    origin_x = line.x() + space_shift
    origin_y = line.y()
    signature = []
    for run in line.glyphRuns(line.textStart(), line.textLength()):
        raw_font = run.rawFont()
        font_key = (type(raw_font), raw_font)
        try:
            hash(font_key)
        except (RuntimeError, TypeError, ValueError):
            return None
        signature.append((
            font_key,
            tuple(int(index) for index in run.glyphIndexes()),
            tuple(
                (
                    position.x() - origin_x,
                    position.y() - origin_y,
                )
                for position in run.positions()
            ),
        ))
    return tuple(signature)


def _line_ink_bounds(
    line: QTextLine,
    space_shift: float = 0.0,
) -> QRectF:
    """Return exact normalized ink without retaining live Qt layouts.

    >>> LINE_INK_BOUNDS_CACHE_MAX_ENTRIES > 0
    True
    """
    cache_key = _line_ink_cache_key(line, space_shift)
    if cache_key is None:
        return _uncached_line_ink_bounds(line, space_shift)
    cached = _LINE_INK_BOUNDS_CACHE.get_or_create(
        cache_key,
        _uncached_line_ink_bounds,
        line,
        space_shift,
    )
    return QRectF(cached)


class VerticalTextDocumentLayout(SceneTextLayout):
    def __init__(self, doc: QTextDocument, fontformat: FontFormat):
        super().__init__(doc, fontformat)

        self.line_spaces_lst = []
        self.min_height = 0
        self.layout_left = 0
        self.has_selection = False

        self.need_ideal_width = True
        self.per_char_records = []
        self.text_combine_ranges = []
        self._ruby_metrics: List[RubyBlockMetrics] = []
        self._base_ink_bounds = QRectF()
        self._annotation_ink_bounds = QRectF()
        self._cursor_update_rect = QRectF()
        self._resize_layout_max_width = None
        self._resize_layout_available_height = None
        self._resize_layout_padding = None
        self._selection_geometry_cache = {}

    def needs_vertical_rotation(self, char: str) -> bool:
        rotation_chars = (
            PUNSET_STANDARD_VERTICAL_ROMAN
            if self.fontformat.standard_vertical_roman_alignment
            else PUNSET_VERNEEDROTATE
        )
        return (
            char in rotation_chars
            or (
                not self.fontformat.standard_vertical_roman_alignment
                and _is_non_fullwidth_roman(char)
            )
        )

    def _line_orientation_char(
        self,
        line: QTextLine,
        source_char: str,
        text: str,
    ) -> str:
        """Use an encoded substitute glyph's existing punctuation rule."""
        if line.textLength() <= 1:
            return source_char
        if (
            self.needs_vertical_rotation(source_char)
            or _grapheme_count(text.strip()) <= 1
        ):
            return source_char
        candidates = (
            _STANDARD_SHAPED_ROTATION_CHARS
            if self.fontformat.standard_vertical_roman_alignment
            else _SHAPED_ROTATION_CHARS
        )
        return _single_glyph_character(line, candidates) or source_char

    def centers_vertical_glyph(self, char: str) -> bool:
        if char in PUNSET_PAUSEORSTOP:
            return self.fontformat.standard_vertical_roman_alignment
        if (
            self.fontformat.standard_vertical_roman_alignment
            and _is_non_fullwidth_roman(char)
        ):
            return True
        return (
            char in PUNSET_ALIGNCENTER
            or vertical_force_aligncentel_pattern.match(char) is not None
        )

    @property
    def align_right(self):
        return False

    def _translate_columns(self, x_shift: float) -> float:
        """Translate every settled vertical-layout x coordinate together."""
        if abs(x_shift) <= 1e-9:
            return 0.0
        applied_shift: Optional[float] = None
        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            for line_number in range(layout.lineCount()):
                line = layout.lineAt(line_number)
                position = line.position()
                before = position.x()
                line_shift = (
                    x_shift
                    if applied_shift is None
                    else applied_shift
                )
                position.setX(before + line_shift)
                line.setPosition(position)
                if applied_shift is None:
                    # Keep every derived coordinate on QTextLine's actual
                    # 26.6 fixed-point movement, not the requested float.
                    applied_shift = line.position().x() - before
            block = block.next()
        if applied_shift is None:
            applied_shift = x_shift
        self.x_offset_lst = [
            x_offset + applied_shift for x_offset in self.x_offset_lst
        ]
        self.layout_left += applied_shift
        return applied_shift

    def _column_content_width(self) -> float:
        if not self.x_offset_lst:
            return 0.0
        return max(0.0, self.x_offset_lst[0] - self.layout_left)

    def _alignment_column_shift(self) -> float:
        slack = max(0.0, self.available_width - self._column_content_width())
        if self.fontformat.alignment == TextAlignment.Left:
            target_left = self._effect_padding
        elif self.fontformat.alignment == TextAlignment.Center:
            target_left = self._effect_padding + slack / 2
        else:
            target_left = self._effect_padding + slack
        return target_left - self.layout_left

    def apply_alignment(self) -> bool:
        """Translate settled columns without reshaping or resizing the box."""
        x_shift = self._alignment_column_shift()
        if abs(x_shift) <= 1e-9:
            return False
        self._begin_layout_generation()
        self._selection_geometry_cache.clear()
        self._translate_columns(x_shift)
        self._refresh_base_ink_bounds()
        self._refresh_annotation_ink_bounds()
        return True

    def spacing_change_height_growth(
        self,
        selection_start: int,
        selection_end: int,
        value: float,
    ) -> float:
        """Return height needed to keep a tight single column from wrapping.

        Multi-column blocks retain their fixed-area reflow behavior. This
        narrowly covers point-like vertical items whose current content was
        squeezed to exactly one column.

        >>> callable(VerticalTextDocumentLayout.spacing_change_height_growth)
        True
        """
        if selection_end <= selection_start:
            return 0.0

        column_x = None
        maximum_bottom = self._effect_padding
        spacing_delta = 0.0
        final_line_delta = 0.0
        final_positive_spacing = 0.0
        block = self.document().firstBlock()
        while block.isValid():
            block_number = block.blockNumber()
            text = block.text()
            text_length = _utf16_length(text)
            text_layout = block.layout()
            for line_number in range(text_layout.lineCount()):
                line = text_layout.lineAt(line_number)
                if not line.isValid() or line.textLength() <= 0:
                    continue
                if column_x is None:
                    column_x = line.x()
                elif abs(line.x() - column_x) > 1e-6:
                    return 0.0

                _trailing, leading, offsets, line_position = (
                    self.line_spaces_lst[block_number][line_number]
                )
                if offsets:
                    maximum_bottom = max(maximum_bottom, offsets[-1])
                char_position = min(
                    line_position + leading,
                    text_length - 1,
                )
                document_position = block.position() + char_position
                if char_position < 0:
                    continue

                char_format = self.get_char_fontfmt(
                    block_number, char_position
                )
                old_value = char_format.letter_spacing
                record = self._line_record(block, line_number)
                spacing_unit = record.get(
                    'text_combine_height',
                    char_format.tbr.height(),
                )
                final_line_delta = 0.0
                final_positive_spacing = max(
                    spacing_unit * (old_value - 1.0), 0.0
                )
                if not (
                    selection_start <= document_position < selection_end
                ):
                    continue

                line_delta = spacing_unit * (value - old_value)
                spacing_delta += line_delta
                final_line_delta = line_delta
            block = block.next()

        available_bottom = self.available_height + self._effect_padding
        # The final glyph's positive trailing advance never forces another
        # line, so it is existing slack rather than required fit height.
        unused_height = max(
            0.0,
            available_bottom - maximum_bottom + final_positive_spacing,
        )
        growth = max(
            0.0,
            spacing_delta - final_line_delta - unused_height,
        )
        # Match minSize()'s guard against Qt's fractional metric rounding.
        return 0.0 if growth <= 1e-6 else growth + 0.01

    def reLayout(self):
        self._begin_layout_generation()
        self._selection_geometry_cache.clear()
        self.min_height = 0
        self.layout_left = 0
        self.line_spaces_lst = []
        self.per_char_records = []
        self.text_combine_ranges = []
        self._ruby_metrics = []
        self.shrink_height = 0
        self.shrink_width = 0
        self.text_padding = 0
        doc = self.document()
        doc_margin = self._effect_padding
        block = doc.firstBlock()
        while block.isValid():
            self.layoutBlock(block)
            block = block.next()

        enlarged = False
        x_shift = 0
        if self.layout_left < doc_margin:
            x_shift = doc_margin - self.layout_left
            self.max_width += x_shift
            self.available_width = self.max_width - 2*doc_margin
            enlarged = True
        if self.min_height - doc_margin > self.available_height:
            self.available_height = self.min_height - doc_margin
            self.max_height = self.available_height + doc_margin * 2
            enlarged = True
        if enlarged:
            self._emit_size_enlarged()
        self._translate_columns(x_shift)
        self._translate_columns(self._alignment_column_shift())
        self.updateDrawOffsets()
        self._refresh_base_ink_bounds()
        self._refresh_annotation_ink_bounds()
        self._resize_layout_max_width = self.max_width
        self._resize_layout_available_height = self.available_height
        self._resize_layout_padding = self._effect_padding
        self.documentSizeChanged.emit(QSizeF(self.max_width, self.max_height))

    def reLayoutForResize(self):
        """Translate a width-only resize; height changes still reflow columns."""
        if (
            self._resize_layout_max_width is None
            or self.available_height
            != self._resize_layout_available_height
            or self._effect_padding != self._resize_layout_padding
        ):
            self.reLayout()
            return
        width_shift = self.max_width - self._resize_layout_max_width
        if width_shift == 0:
            self.documentSizeChanged.emit(
                QSizeF(self.max_width, self.max_height)
            )
            return
        if self.available_width + 1e-9 < self._column_content_width():
            # The normal path enforces the content's minimum column width.
            self.reLayout()
            return

        column_shift = self._alignment_column_shift()
        if abs(column_shift) > 1e-9:
            self._begin_layout_generation()
            self._translate_columns(column_shift)
            self._refresh_base_ink_bounds()
            self._refresh_annotation_ink_bounds()
        self._resize_layout_max_width = self.max_width
        self.documentSizeChanged.emit(QSizeF(self.max_width, self.max_height))

    def updateDrawOffsets(self):
        if self._is_painting_stroke and len(self._draw_offset) > 0:
            return
        self._draw_offset.clear()
        doc = self.document()
        block = doc.firstBlock()
        custom_rendering = self.render_delegate is not None

        while block.isValid():
            blk_no = block.blockNumber()
            _draw_offsets = []
            self._draw_offset.append(_draw_offsets)

            layout = block.layout()
            blk_text = block.text()
            has_text_combine = bool(self.text_combine_ranges[blk_no])
            utf16_indexing = (
                custom_rendering
                or has_text_combine
                or _utf16_length(blk_text) != len(blk_text)
            )
            blk_text_len = (
                _utf16_length(blk_text) if utf16_indexing else len(blk_text)
            )

            line_spaces_lst = self.line_spaces_lst[blk_no]
            char_records = self.per_char_records[blk_no]

            for ii in range(layout.lineCount()):
                xy_offsets = [0, 0]
                _draw_offsets.append(xy_offsets)

                line = layout.lineAt(ii)
                if line.textLength() == 0:
                    continue
                if self.is_tate_chu_yoko_line(block, ii):
                    # The run-level transform performs all cell alignment.
                    continue
                num_rspaces, num_lspaces, _, line_pos  = line_spaces_lst[ii]
                char_idx = min(line_pos + num_lspaces, blk_text_len - 1)
                if char_idx < 0:
                    continue

                char = (
                    _utf16_char_at(blk_text, char_idx)
                    if utf16_indexing
                    else blk_text[char_idx]
                )
                cfmt = self.get_char_fontfmt(blk_no, char_idx)

                line_width = -1
                if char_idx in char_records:
                    line_width = char_records[char_idx]['line_width']
                if line_width < 0:
                    line_width = cfmt.tbr.width()
                record = char_records.get(char_idx, {})
                char = record.get('orientation_char', char)
                base_width = record.get('base_width', line_width)
                left_margin = record.get('left_margin', 0.0)
                compact_advance = record.get('compact_punctuation_advance')
                compact_leading_trim = record.get(
                    'compact_punctuation_leading_trim', 0.0
                )

                space_shift = 0
                if num_lspaces > 0:
                    space_shift = num_lspaces * cfmt.space_width

                if self.needs_vertical_rotation(char):
                    char = (
                        _utf16_char_at(blk_text, char_idx)
                        if utf16_indexing
                        else blk_text[char_idx]
                    )
                    if char.isalpha():
                        xoff = 0
                        yoff = (
                            -line.ascent()
                            - (base_width - cfmt.font_metrics.capHeight()) / 2
                        )

                    else:   # () （）
                        non_bracket_br = _line_ink_bounds(
                            line, space_shift
                        )
                        yoff = (
                            -non_bracket_br.top()
                            - non_bracket_br.height()
                        )
                        if compact_leading_trim > 0:
                            xoff = -compact_leading_trim
                        elif char in PUNSET_BRACKETL:
                            if ii == 0:
                                xoff = -non_bracket_br.left()
                            else:
                                xoff = 0
                        else:
                            xoff = -non_bracket_br.left()

                        if char in PUNSET_ROTATE_ALIGNL:
                            yoff = yoff
                        elif char in PUNSET_ROTATE_ALIGNR:
                            yoff -= base_width - non_bracket_br.height()
                        else:
                            yoff -= (
                                base_width - non_bracket_br.height()
                            ) / 2
                    yoff -= left_margin

                else:
                    standard_roman = (
                        self.fontformat.standard_vertical_roman_alignment
                        and _is_non_fullwidth_roman(char)
                    )
                    if standard_roman:
                        tight_rect = _line_ink_bounds(line, space_shift)
                        xoff = (
                            -tight_rect.left()
                            + (base_width - tight_rect.width()) / 2
                        )
                        yoff = (
                            -tight_rect.top()
                            + (
                                (
                                    compact_advance
                                    if compact_advance is not None
                                    else cfmt.tbr.height()
                                )
                                - tight_rect.height()
                            ) / 2
                        )
                    else:
                        act_rect = _line_ink_bounds(line, space_shift)
                        if self.centers_vertical_glyph(char):
                            xoff = (
                                -act_rect.left()
                                + (base_width - act_rect.width()) / 2
                            )
                            yoff = (
                                -act_rect.top()
                                + (
                                    (
                                        compact_advance
                                        if compact_advance is not None
                                        else cfmt.tbr.height()
                                    )
                                    - act_rect.height()
                                ) / 2
                            )
                        elif char in PUNSET_PAUSEORSTOP:
                            # CLREQ's Mainland convention places stop marks at
                            # the upper-right of their full character frame.
                            xoff = (
                                -act_rect.left()
                                + base_width
                                - act_rect.width()
                            )
                            yoff = -act_rect.top()
                        else:
                            yoff = min(
                                cfmt.br.top() - cfmt.tbr.top(),
                                -cfmt.tbr.top() - line.ascent(),
                            )
                            xoff = (
                                -act_rect.left()
                                + (base_width - act_rect.width()) / 2
                            )

                    xoff += left_margin

                    if num_lspaces > 0:
                        xoff -= space_shift
                        yoff += space_shift

                xy_offsets[0], xy_offsets[1] = xoff, yoff
            block = block.next()

    def _line_record(
        self,
        block: QTextBlock,
        line_number: int,
    ) -> dict:
        layout = block.layout()
        line = layout.lineAt(line_number)
        if not line.isValid():
            return {}
        block_number = block.blockNumber()
        if not 0 <= block_number < len(self.per_char_records):
            return {}
        return self.per_char_records[block_number].get(line.textStart(), {})

    def is_tate_chu_yoko_line(
        self,
        block: QTextBlock,
        line_number: int,
    ) -> bool:
        """Return whether one vertical layout line is a combined run."""
        return 'text_combine_height' in self._line_record(block, line_number)

    @staticmethod
    def _line_cursor_x(line: QTextLine, position: int) -> float:
        value = line.cursorToX(position)
        if isinstance(value, (tuple, list)):
            value = value[0]
        return float(value)

    def _vertical_line_cells(
        self,
        block: QTextBlock,
        line_number: int,
    ) -> List[Tuple[int, int, float, float, bool]]:
        """Return logical cells using the line's settled vertical boundaries.

        Cells use block-local UTF-16 ranges. The final boolean distinguishes
        whitespace whose geometry Qt does not paint at its vertical position.

        >>> callable(VerticalTextDocumentLayout._vertical_line_cells)
        True
        """
        layout = block.layout()
        line = layout.lineAt(line_number)
        if not line.isValid() or line.textLength() <= 0:
            return []
        trailing, leading, offsets, line_position = self.line_spaces_lst[
            block.blockNumber()
        ][line_number]
        text_length = line.textLength()
        if line_position + text_length > _utf16_length(block.text()):
            return []

        # An all-whitespace QTextLine reports the same run as both leading
        # and trailing. Keep one cell per authored code unit.
        if leading >= text_length and trailing >= text_length:
            count = min(text_length, max(0, len(offsets) - 1))
            return [
                (
                    line_position + index,
                    line_position + index + 1,
                    offsets[index],
                    offsets[index + 1],
                    True,
                )
                for index in range(count)
            ]

        cells = []
        leading = min(leading, text_length)
        trailing = min(trailing, text_length - leading)
        required_boundaries = leading + trailing + 2
        if len(offsets) < required_boundaries:
            return []

        for index in range(leading):
            cells.append((
                line_position + index,
                line_position + index + 1,
                offsets[index],
                offsets[index + 1],
                True,
            ))

        content_start = line_position + leading
        content_length = text_length - leading - trailing
        content_top = offsets[leading]
        content_bottom = offsets[leading + 1]
        if content_length > 0:
            content = _utf16_slice(
                block.text(), content_start, content_length
            )
            graphemes = _grapheme_ranges(content)
            if graphemes:
                run_height = max(content_bottom - content_top, 0.0)
                cell_height = run_height / len(graphemes)
                char = _utf16_char_at(block.text(), content_start)
                char = self._line_record(block, line_number).get(
                    'orientation_char', char
                )
                if len(graphemes) > 1 and self.needs_vertical_rotation(char):
                    char_format = self.get_char_fontfmt(
                        block.blockNumber(), content_start
                    )
                    if char_format is not None:
                        natural_height = char_format.punc_rect(char)[0].width()
                        cell_height = min(cell_height, natural_height)
                for index, (start, end) in enumerate(graphemes):
                    bottom = (
                        content_bottom
                        if index == len(graphemes) - 1
                        else content_top + (index + 1) * cell_height
                    )
                    cells.append((
                        content_start + start,
                        content_start + end,
                        content_top + index * cell_height,
                        bottom,
                        False,
                    ))
            else:
                cells.append((
                    content_start,
                    content_start + content_length,
                    content_top,
                    content_bottom,
                    False,
                ))

        trailing_start = content_start + content_length
        boundary_index = leading + 1
        for index in range(trailing):
            cells.append((
                trailing_start + index,
                trailing_start + index + 1,
                offsets[boundary_index + index],
                offsets[boundary_index + index + 1],
                True,
            ))
        return cells

    def _vertical_line_width(
        self,
        block: QTextBlock,
        line_number: int,
    ) -> float:
        _trailing, leading, _offsets, line_position = self.line_spaces_lst[
            block.blockNumber()
        ][line_number]
        char_position = min(
            line_position + leading,
            max(0, _utf16_length(block.text()) - 1),
        )
        char_format = self.get_char_fontfmt(
            block.blockNumber(), char_position
        )
        record = self.per_char_records[block.blockNumber()].get(
            char_position, {}
        )
        if char_format is None:
            return max(0.0, self.block_ideal_width[block.blockNumber()])
        return max(
            0.0,
            record.get('line_width', char_format.tbr.width()),
        )

    @staticmethod
    def _line_has_selection(
        block: QTextBlock,
        line: QTextLine,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> bool:
        line_start = line.textStart()
        line_end = line_start + line.textLength()
        for selection in context.selections:
            if not selection.cursor.hasSelection():
                continue
            start = selection.cursor.selectionStart() - block.position()
            end = selection.cursor.selectionEnd() - block.position()
            if line_start < end and line_end > start:
                return True
        return False

    @staticmethod
    def _selection_foreground_context(
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> QAbstractTextDocumentLayout.PaintContext:
        copied = QAbstractTextDocumentLayout.PaintContext()
        copied.clip = QRectF(context.clip)
        copied.cursorPosition = context.cursorPosition
        copied.palette = context.palette
        copied.selections = []
        for selection in context.selections:
            foreground_selection = QAbstractTextDocumentLayout.Selection()
            foreground_selection.cursor = selection.cursor
            foreground_selection.format = QTextCharFormat(selection.format)
            foreground_selection.format.clearBackground()
            copied.selections.append(foreground_selection)
        return copied

    def _vertical_selection_backgrounds(
        self,
        block: QTextBlock,
        line_number: int,
        context: QAbstractTextDocumentLayout.PaintContext,
        cells: List[Tuple[int, int, float, float, bool]],
    ) -> List[Tuple[QRectF, QBrush]]:
        if not cells:
            return []
        line = block.layout().lineAt(line_number)
        width = self._vertical_line_width(block, line_number)
        backgrounds = []
        for selection in context.selections:
            if not selection.cursor.hasSelection():
                continue
            selection_start = selection.cursor.selectionStart()
            selection_end = selection.cursor.selectionEnd()
            brush = selection.format.background()
            if brush.style() == Qt.BrushStyle.NoBrush:
                continue
            for start, end, top, bottom, _is_space in cells:
                absolute_start = block.position() + start
                absolute_end = block.position() + end
                if (
                    selection_start < absolute_end
                    and selection_end > absolute_start
                ):
                    backgrounds.append((
                        QRectF(line.x(), top, width, bottom - top),
                        brush,
                    ))
        return backgrounds

    def tate_chu_yoko_cell_rect(
        self,
        block: QTextBlock,
        line_number: int,
    ) -> Optional[QRectF]:
        """Return the untransformed vertical cell reserved for a run."""
        record = self._line_record(block, line_number)
        cell_width = record.get('text_combine_width')
        cell_height = record.get('text_combine_height')
        if cell_width is None or cell_height is None:
            return None
        line = block.layout().lineAt(line_number)
        base_width = record.get('base_width', cell_width)
        left_margin = record.get('left_margin', 0.0)
        line_width = record.get('line_width', base_width)
        cell_left = (
            line.x() + left_margin + (base_width - cell_width) / 2
        )
        if cell_width <= line_width:
            # Use an occupied annotation-side margin before Tate ink overhangs.
            cell_left = max(
                line.x(),
                min(cell_left, line.x() + line_width - cell_width),
            )
        return QRectF(
            cell_left,
            line.y(),
            cell_width,
            cell_height,
        )

    def annotation_ink_bounds(self) -> QRectF:
        """Return cached Tate-chu-yoko, Ruby, and emphasis ink."""
        return QRectF(self._annotation_ink_bounds)

    def base_ink_bounds(self) -> QRectF:
        """Return cached ink for base lines with non-native orientation."""
        return QRectF(self._base_ink_bounds)

    def _refresh_base_ink_bounds(self) -> None:
        """Cache exact neutral ink for transformed vertical base lines.

        Ordinary upright lines remain covered by the logical text box. Rotated
        lines can overhang it because their horizontal glyph ink becomes
        vertical-layout x ink after placement.

        >>> callable(VerticalTextDocumentLayout._refresh_base_ink_bounds)
        True
        """
        bounds = QRectF()
        block = self.document().firstBlock()
        while block.isValid():
            text_layout = block.layout()
            for line_number in range(text_layout.lineCount()):
                placement = self.vertical_line_placement(block, line_number)
                if placement is None:
                    continue
                line, offset, orientation = placement
                if orientation.isIdentity():
                    continue
                candidate = glyph_geometry(
                    line,
                    line.textStart(),
                    line.textLength(),
                    offset,
                    orientation,
                    0.0,
                ).bounds
                if candidate.isEmpty():
                    continue
                bounds = (
                    QRectF(candidate)
                    if bounds.isEmpty()
                    else bounds.united(candidate)
                )
            block = block.next()
        self._base_ink_bounds = bounds

    def _vertical_ruby_base_cell(
        self,
        block: QTextBlock,
        metric: RubyUnitMetrics,
    ) -> QRectF:
        layout = block.layout()
        local_start = metric.unit.start - block.position()
        local_end = metric.unit.end - block.position()
        base_bounds = QRectF()
        first_line = layout.lineForTextPosition(local_start)
        last_line = layout.lineForTextPosition(max(local_start, local_end - 1))
        if not first_line.isValid() or not last_line.isValid():
            return base_bounds
        for line_number in range(
            first_line.lineNumber(), last_line.lineNumber() + 1
        ):
            line = layout.lineAt(line_number)
            line_start = line.textStart()
            line_end = line_start + line.textLength()
            if line_start >= local_end or line_end <= local_start:
                continue
            cells = self._vertical_line_cells(block, line_number)
            if not cells:
                continue
            top = min(cell[2] for cell in cells)
            bottom = max(cell[3] for cell in cells)
            char_format = self.get_char_fontfmt(
                block.blockNumber(), max(local_start, line_start)
            )
            base_width = (
                self._vertical_line_width(block, line_number)
                if char_format is None
                else char_format.tbr.width()
            )
            line_width = self._vertical_line_width(block, line_number)
            record = self._line_record(block, line_number)
            column_base_width = record.get('base_width', line_width)
            left_margin = record.get('left_margin', 0.0)
            rect = QRectF(
                line.x()
                + left_margin
                + (column_base_width - base_width) / 2,
                top,
                base_width,
                bottom - top,
            )
            base_bounds = (
                rect
                if base_bounds.isEmpty()
                else base_bounds.united(rect)
            )
        return base_bounds

    def _vertical_ruby_unit_cell(
        self,
        block: QTextBlock,
        metric: RubyUnitMetrics,
    ) -> QRectF:
        base_bounds = self._vertical_ruby_base_cell(block, metric)
        edge = metric.base_gap / 2
        return base_bounds.adjusted(
            0.0, -edge, 0.0, edge
        ) if not base_bounds.isEmpty() else base_bounds

    def _vertical_ruby_placements(
        self,
        block: QTextBlock,
        context: Optional[QAbstractTextDocumentLayout.PaintContext] = None,
    ) -> Tuple[RubyPlacement, ...]:
        if block.blockNumber() >= len(self._ruby_metrics):
            return ()
        angle = float(getattr(self.render_delegate, 'glyph_slant_angle', 0.0))
        placements = []
        block_metrics = self._ruby_metrics[block.blockNumber()]
        for metric in block_metrics:
            cell = self._vertical_ruby_unit_cell(block, metric)
            if cell.isEmpty():
                continue
            placements.append(ruby_placement(
                block,
                metric.container,
                metric.unit,
                cell,
                vertical=True,
                context=context,
                glyph_slant_angle=angle,
                format_index=block_metrics.format_index,
                inline_offset=metric.annotation_center_offset,
            ))
        return tuple(placements)

    def _refresh_annotation_ink_bounds(self) -> None:
        """Measure paint overflow after final line placement.

        >>> callable(VerticalTextDocumentLayout._refresh_annotation_ink_bounds)
        True
        """
        bounds = QRectF()
        block = self.document().firstBlock()
        while block.isValid():
            text_layout = block.layout()
            has_ruby = (
                block.blockNumber() < len(self._ruby_metrics)
                and bool(self._ruby_metrics[block.blockNumber()])
            )
            for line_number in range(text_layout.lineCount()):
                cell = self.tate_chu_yoko_cell_rect(block, line_number)
                if cell is None and not has_ruby:
                    continue
                placement = self.vertical_line_placement(
                    block, line_number
                )
                if placement is None:
                    continue
                line, offset, orientation = placement
                ruby_margins = ruby_side_margins(
                    block,
                    line,
                    self._ruby_metrics[block.blockNumber()],
                    vertical=True,
                )
                candidates = [
                    emphasis_ink_bounds(
                        block,
                        line,
                        vertical=True,
                        offset=offset,
                        orientation=orientation,
                        side_offsets=ruby_margins,
                    ),
                ]
                if cell is not None:
                    candidates.extend((
                        cell,
                        tate_chu_yoko_ink_bounds(line, cell),
                    ))
                for candidate in candidates:
                    if candidate.isEmpty():
                        continue
                    bounds = (
                        QRectF(candidate)
                        if bounds.isEmpty()
                        else bounds.united(candidate)
                    )
            for placement in self._vertical_ruby_placements(block):
                candidate = placement.ink_bounds
                if not candidate.isEmpty():
                    bounds = (
                        QRectF(candidate)
                        if bounds.isEmpty()
                        else bounds.united(candidate)
                    )
            block = block.next()
        self._annotation_ink_bounds = bounds

    def _ruby_hit_test(self, point: QPointF) -> Optional[int]:
        block = self.document().firstBlock()
        while block.isValid():
            if block.blockNumber() >= len(self._ruby_metrics):
                block = block.next()
                continue
            angle = float(getattr(
                self.render_delegate, 'glyph_slant_angle', 0.0
            ))
            block_metrics = self._ruby_metrics[block.blockNumber()]
            for metric in block_metrics:
                cell = self._vertical_ruby_unit_cell(block, metric)
                if cell.isEmpty():
                    continue
                placement = ruby_placement(
                    block,
                    metric.container,
                    metric.unit,
                    cell,
                    vertical=True,
                    glyph_slant_angle=angle,
                    format_index=block_metrics.format_index,
                    inline_offset=metric.annotation_center_offset,
                )
                base_cell = self._vertical_ruby_base_cell(block, metric)
                annotation_hit = placement.ink_bounds.contains(point)
                gap_hit = (
                    placement.cell.contains(point)
                    and not base_cell.contains(point)
                )
                if not annotation_hit and not gap_hit:
                    continue
                boundaries = []
                layout = block.layout()
                local_start = metric.unit.start - block.position()
                local_end = metric.unit.end - block.position()
                first_line = layout.lineForTextPosition(local_start)
                last_line = layout.lineForTextPosition(
                    max(local_start, local_end - 1)
                )
                for line_number in range(
                    first_line.lineNumber(), last_line.lineNumber() + 1
                ):
                    for start, end, top, bottom, _is_space in (
                        self._vertical_line_cells(block, line_number)
                    ):
                        if start < local_end and end > local_start:
                            boundaries.extend(((top, start), (bottom, end)))
                if boundaries:
                    _distance, local = min(
                        (abs(point.y() - y), position)
                        for y, position in boundaries
                    )
                    return block.position() + local
                return placement.unit.start
            block = block.next()
        return None

    def _paint_ruby_selection_backgrounds(
        self,
        painter: QPainter,
        block: QTextBlock,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> None:
        if block.blockNumber() >= len(self._ruby_metrics):
            return
        for selection in context.selections:
            if not selection.cursor.hasSelection():
                continue
            brush = selection.format.background()
            if brush.style() == Qt.BrushStyle.NoBrush:
                continue
            for metric in self._ruby_metrics[block.blockNumber()].contained(
                selection.cursor.selectionStart(),
                selection.cursor.selectionEnd(),
            ):
                cell = self._vertical_ruby_unit_cell(block, metric)
                if not cell.isEmpty():
                    painter.fillRect(cell, brush)

    def _tate_chu_yoko_hit_position(
        self,
        line: QTextLine,
        transform: QTransform,
        point: QPointF,
    ) -> int:
        """Map one horizontal cell back to its ordinary text cursor.

        >>> callable(VerticalTextDocumentLayout._tate_chu_yoko_hit_position)
        True
        """
        line_start = line.textStart()
        line_end = line_start + line.textLength()
        start_x = transform.map(
            QPointF(self._line_cursor_x(line, line_start), line.y())
        ).x()
        end_x = transform.map(
            QPointF(self._line_cursor_x(line, line_end), line.y())
        ).x()
        if start_x <= end_x:
            if point.x() <= start_x:
                return line_start
            if point.x() >= end_x:
                return line_end
        else:
            if point.x() >= start_x:
                return line_start
            if point.x() <= end_x:
                return line_end
        inverse, invertible = transform.inverted()
        if not invertible:
            return line_start
        return line.xToCursor(
            inverse.map(point).x(),
            QTextLine.CursorBetweenCharacters,
        )

    def _tate_chu_yoko_hit_test(
        self,
        point: QPointF,
    ) -> Optional[int]:
        """Hit-test visible overhang without widening its layout column."""
        block = self.document().firstBlock()
        while block.isValid():
            text_layout = block.layout()
            for line_number in range(text_layout.lineCount()):
                cell = self.tate_chu_yoko_cell_rect(block, line_number)
                if cell is None or not cell.contains(point):
                    continue
                placement = self.vertical_line_placement(
                    block, line_number
                )
                if placement is None:
                    continue
                line, _offset, transform = placement
                return block.position() + self._tate_chu_yoko_hit_position(
                    line, transform, point
                )
            block = block.next()
        return None

    def vertical_line_placement(
        self,
        block: QTextBlock,
        line_number: int,
    ) -> Optional[Tuple[QTextLine, QPointF, QTransform]]:
        """Return the established glyph placement for annotation painters."""
        text_layout = block.layout()
        line = text_layout.lineAt(line_number)
        if not line.isValid() or line.textLength() <= 0:
            return None
        text_combine_cell = self.tate_chu_yoko_cell_rect(block, line_number)
        if text_combine_cell is not None:
            return (
                line,
                QPointF(),
                tate_chu_yoko_transform(line, text_combine_cell),
            )
        block_number = block.blockNumber()
        block_text = block.text()
        block_text_length = _utf16_length(block_text)
        _, leading_spaces, _, line_position = self.line_spaces_lst[
            block_number
        ][line_number]
        char_offset = min(
            line_position + leading_spaces,
            block_text_length - 1,
        )
        if char_offset < 0:
            return line, QPointF(), QTransform()
        char = _utf16_char_at(block_text, char_offset)
        char = self._line_record(block, line_number).get(
            'orientation_char', char
        )
        x_offset, y_offset = self._draw_offset[block_number][line_number]
        orientation = QTransform()
        if self.needs_vertical_rotation(char):
            line_x, line_y = line.x(), line.y()
            orientation = QTransform(
                0,
                1,
                0,
                -1,
                0,
                0,
                line_y + line_x,
                line_y - line_x,
                1,
            )
        return line, QPointF(x_offset, y_offset), orientation

    def source_cursor_rect(self, cursor_position: int):
        """Return the caret owned by the vertical source layout."""
        block = self.document().firstBlock()
        while block.isValid():
            cpos = _block_cursor_position(block, cursor_position)
            if cpos >= 0:
                layout = block.layout()
                if block.blockNumber() < len(self._ruby_metrics):
                    metric = self._ruby_metrics[
                        block.blockNumber()
                    ].containing(cursor_position)
                    if metric is not None:
                        start = metric.unit.start - block.position()
                        end = metric.unit.end - block.position()
                        cell = self._vertical_ruby_unit_cell(block, metric)
                        if not cell.isEmpty() and cpos in (start, end):
                            y = cell.top() if cpos == start else cell.bottom()
                            return QRectF(cell.left(), y, cell.width(), 2.0)
                line = layout.lineForTextPosition(cpos)
                if not line.isValid():
                    return QRectF()
                line_number = line.lineNumber()
                text_combine_cell = self.tate_chu_yoko_cell_rect(
                    block, line_number
                )
                if text_combine_cell is not None:
                    placement = self.vertical_line_placement(
                        block, line_number
                    )
                    if placement is None:
                        return QRectF()
                    _line, _offset, transform = placement
                    cursor_x = self._line_cursor_x(line, cpos)
                    mapped_x = transform.map(
                        QPointF(cursor_x, line.y())
                    ).x()
                    mapped_x = min(
                        max(mapped_x, text_combine_cell.left()),
                        text_combine_cell.right(),
                    )
                    return QRectF(
                        mapped_x - 1.0,
                        text_combine_cell.top(),
                        2.0,
                        text_combine_cell.height(),
                    )
                position = line.position()
                cells = self._vertical_line_cells(block, line_number)
                y = position.y()
                if cells:
                    y = cells[-1][3]
                    for start, end, top, bottom, _is_space in cells:
                        if cpos <= start:
                            y = top
                            break
                        if cpos <= end:
                            y = bottom
                            break
                return QRectF(
                    position.x(),
                    y,
                    self._vertical_line_width(block, line_number),
                    2.0,
                )
            block = block.next()
        return QRectF()

    def adjacent_column_cursor_position(
        self,
        cursor_position: int,
        horizontal_direction: int,
        preferred_y: float,
    ) -> Optional[int]:
        """Return the caret position in the adjacent visual column.

        ``horizontal_direction`` is negative for the column to the left and
        positive for the column to the right. The caller keeps ``preferred_y``
        stable across repeated moves, matching Qt's horizontal line navigation.

        >>> callable(VerticalTextDocumentLayout.adjacent_column_cursor_position)
        True
        """
        if horizontal_direction not in (-1, 1):
            raise ValueError('horizontal_direction must be -1 or 1')

        block = self.document().findBlock(cursor_position)
        if not block.isValid():
            return None
        layout = block.layout()
        line = layout.lineForTextPosition(
            cursor_position - block.position()
        )
        if not line.isValid():
            return None
        current_x = line.x()
        line_number = line.lineNumber()
        step = 1 if horizontal_direction < 0 else -1

        target_block = block
        target_line_number = line_number + step
        target_line = QTextLine()
        while target_block.isValid():
            target_layout = target_block.layout()
            while 0 <= target_line_number < target_layout.lineCount():
                candidate = target_layout.lineAt(target_line_number)
                x_delta = candidate.x() - current_x
                if x_delta * horizontal_direction > 1e-6:
                    target_line = candidate
                    break
                target_line_number += step
            if target_line.isValid():
                break
            target_block = (
                target_block.next()
                if horizontal_direction < 0
                else target_block.previous()
            )
            if target_block.isValid():
                target_layout = target_block.layout()
                target_line_number = (
                    0
                    if horizontal_direction < 0
                    else target_layout.lineCount() - 1
                )

        if not target_line.isValid():
            return None
        target_center_x = (
            target_line.x()
            + self._vertical_line_width(
                target_block, target_line_number
            ) / 2
        )
        return self._source_hit_test(
            QPointF(target_center_x, preferred_y)
        )

    def draw(self, painter: QPainter, context: QAbstractTextDocumentLayout.PaintContext) -> None:
        doc = self.document()
        self.deferred_cursor_position = context.cursorPosition
        painter.save()
        block = doc.firstBlock()
        cursor_block = None
        context_sel = context.selections
        has_selection = False
        render_delegate = self.render_delegate
        custom_rendering = render_delegate is not None
        if len(context_sel) > 0:
            has_selection = True

        while block.isValid():
            blk_no = block.blockNumber()
            layout = block.layout()
            blk_text = block.text()
            utf16_indexing = (
                custom_rendering
                or bool(self.text_combine_ranges[blk_no])
                or _utf16_length(blk_text) != len(blk_text)
            )
            blk_text_len = (
                _utf16_length(blk_text) if utf16_indexing else len(blk_text)
            )
            line_spaces_lst = self.line_spaces_lst[blk_no]
            uniform_block_drawn = (
                custom_rendering
                and render_delegate.draw_uniform_block(
                    painter, block, context
                )
            )

            if _block_cursor_position(block, context.cursorPosition) >= 0:
                cursor_block = block

            self._paint_ruby_selection_backgrounds(
                painter, block, context
            )

            for ii in range(layout.lineCount()):
                line = layout.lineAt(ii)
                if line.textLength() == 0:
                    continue
                num_rspaces, num_lspaces, _, line_pos  = line_spaces_lst[ii]
                char_idx = min(line_pos + num_lspaces, blk_text_len - 1)
                if char_idx < 0:
                    if custom_rendering:
                        if not uniform_block_drawn:
                            render_delegate.draw_vertical_line(
                                painter, block, ii, context
                            )
                    else:
                        line.draw(painter, QPointF(0, 0))
                    continue

                xoff, yoff = self._draw_offset[blk_no][ii]

                if custom_rendering:
                    if not uniform_block_drawn:
                        render_delegate.draw_vertical_line(
                            painter, block, ii, context
                        )
                    placement = self.vertical_line_placement(block, ii)
                    if placement is not None:
                        placed_line, offset, orientation = placement
                        draw_emphasis_marks(
                            painter,
                            block,
                            placed_line,
                            context,
                            vertical=True,
                            offset=offset,
                            orientation=orientation,
                            side_offsets=ruby_side_margins(
                                block,
                                placed_line,
                                self._ruby_metrics[block.blockNumber()],
                                vertical=True,
                            ),
                        )
                    continue
                intersects = has_selection and self._line_has_selection(
                    block, line, context
                )
                line_context = context
                selection_backgrounds = ()
                if intersects and not self.is_tate_chu_yoko_line(block, ii):
                    cells = self._vertical_line_cells(block, ii)
                    selection_backgrounds = (
                        self._vertical_selection_backgrounds(
                            block, ii, context, cells
                        )
                    )
                    space_ranges = [
                        (start, end)
                        for start, end, _top, _bottom, is_space in cells
                        if is_space
                    ]
                    line_context = paint_context_without_selection_ranges(
                        self.document(),
                        block,
                        context,
                        space_ranges,
                    )
                    line_context = self._selection_foreground_context(
                        line_context
                    )
                placement = self.vertical_line_placement(block, ii)
                if self.is_tate_chu_yoko_line(block, ii):
                    if placement is not None:
                        placed_line, offset, orientation = placement
                        draw_slanted_line(
                            painter,
                            block,
                            placed_line,
                            offset,
                            orientation,
                            0.0,
                            context,
                            self._report_render_failure,
                        )
                elif (
                    placement is not None
                    and intersects
                ):
                    placed_line, offset, orientation = placement
                    draw_slanted_line(
                        painter,
                        block,
                        placed_line,
                        offset,
                        orientation,
                        0.0,
                        line_context,
                        self._report_render_failure,
                        self._selection_geometry_cache,
                        (self.layout_generation, blk_no, ii),
                        background_overlays=selection_backgrounds,
                    )
                elif placement is not None and not placement[2].isIdentity():
                    _placed_line, offset, orientation = placement
                    inverse, _invertible = orientation.inverted()
                    painter.setTransform(orientation, True)
                    line.draw(painter, offset)
                    painter.setTransform(inverse, True)
                else:
                    line.draw(painter, QPointF(xoff, yoff))

                if placement is not None:
                    placed_line, offset, orientation = placement
                    draw_emphasis_marks(
                        painter,
                        block,
                        placed_line,
                        context,
                        vertical=True,
                        offset=offset,
                        orientation=orientation,
                        side_offsets=ruby_side_margins(
                            block,
                            placed_line,
                            self._ruby_metrics[block.blockNumber()],
                            vertical=True,
                        ),
                    )

            for ruby_annotation in self._vertical_ruby_placements(
                block, context
            ):
                draw_ruby_placement(painter, ruby_annotation)
            block = block.next()

        if self.foreground_pixmap is not None:
            painter.drawPixmap(0, 0, self.foreground_pixmap)

        if not self.defer_cursor_paint:
            cursor_rect = QRectF()
            if cursor_block is not None:
                cursor_rect = self.source_cursor_rect(
                    context.cursorPosition
                )
            if not cursor_rect.isEmpty():
                painter.setCompositionMode(
                    QPainter.CompositionMode.RasterOp_NotDestination
                )
                painter.fillRect(cursor_rect, painter.pen().brush())
            if self.has_selection != has_selection:
                dirty_rect = QRectF(
                    0, 0, self.max_width, self.max_height
                )
            elif cursor_rect != self._cursor_update_rect:
                dirty_rect = cursor_rect.united(
                    self._cursor_update_rect
                )
            else:
                dirty_rect = QRectF()
            self._cursor_update_rect = QRectF(cursor_rect)
            if not dirty_rect.isEmpty():
                if C.USE_PYSIDE6:
                    self.update.emit()
                else:
                    self.update.emit(dirty_rect)
            self.has_selection = has_selection
        painter.restore()

    def _source_hit_test(self, point: QPointF) -> int:
        """Resolve a point already expressed in source-layout coordinates.

        >>> callable(VerticalTextDocumentLayout._source_hit_test)
        True
        """
        ruby_hit = self._ruby_hit_test(point)
        if ruby_hit is not None:
            return ruby_hit
        text_combine_hit = self._tate_chu_yoko_hit_test(point)
        if text_combine_hit is not None:
            return text_combine_hit
        blk = self.document().firstBlock()
        x, y = point.x(), point.y()
        off = 0
        while blk.isValid():
            blk_no = blk.blockNumber()
            rect_right = self.x_offset_lst[blk_no]
            rect_left = self.x_offset_lst[blk_no + 1]
            if rect_left <= x and rect_right >= x:
                layout = blk.layout()
                for line_number in range(layout.lineCount()):
                    line = layout.lineAt(line_number)
                    if line.x() > x:
                        continue
                    cells = self._vertical_line_cells(
                        blk, line_number
                    )
                    if not cells:
                        continue
                    line_top = min(cell[2] for cell in cells)
                    line_bottom = max(cell[3] for cell in cells)
                    if line_top > y:
                        off = min(off, cells[0][0])
                    elif line_bottom < y:
                        off = max(off, cells[-1][1])
                    else:
                        for start, end, top, bottom, _is_space in cells:
                            if top <= y <= bottom:
                                off = start if y - top < bottom - y else end
                                break
                        break
                break
            blk = blk.next()
        return blk.position() + off

    def hitTest(self, point: QPointF, accuracy: Qt.HitTestAccuracy) -> int:
        return self._source_hit_test(self.map_input_point(point))

    def layoutBlock(self, block: QTextBlock):
        doc = self.document()
        compact_punctuation = pcfg.compact_vertical_punctuation_spacing

        block.clearLayout()
        clear_horizontal_ruby_layout(block)
        doc_margin = self._effect_padding
        line_y_offset = doc_margin
        blk_char_yoffset = []
        blk_line_spaces = []

        block_no = block.blockNumber()
        blk_text = block.text()
        custom_rendering = self.render_delegate is not None
        text_combine_ranges = text_combine_upright_ranges(block)
        self.text_combine_ranges.append(text_combine_ranges)
        ruby_metrics = vertical_ruby_metrics(
            block,
            self.needs_vertical_rotation,
            self.letter_spacing,
        )
        self._ruby_metrics.append(ruby_metrics)
        ruby_starts = {
            metric.unit.start - block.position(): metric
            for metric in ruby_metrics
        }
        inline_unit_boundaries = None
        ruby_base_leading = {}
        ruby_base_trailing = {}
        for metric in ruby_metrics:
            if metric.extra <= 1e-6:
                continue
            unit_start = metric.unit.start - block.position()
            unit_end = metric.unit.end - block.position()
            half_gap = metric.base_gap / 2
            ruby_base_leading[unit_start] = half_gap
            ruby_base_trailing[unit_end] = half_gap
            for boundary in metric.base_opportunity_ends:
                local_boundary = unit_start + boundary
                ruby_base_leading[local_boundary] = half_gap
                ruby_base_trailing[local_boundary] = half_gap
        text_combine_lengths = {
            start: length for start, length, _group_id in text_combine_ranges
        }
        utf16_indexing = (
            custom_rendering
            or bool(text_combine_ranges)
            or _utf16_length(blk_text) != len(blk_text)
        )
        blk_text_len = (
            _utf16_length(blk_text) if utf16_indexing else len(blk_text)
        )
        if blk_text_len != 0:
            block_width = self.block_ideal_width[block_no]
        else:
            block_width = CharFontFormat(block.charFormat()).tbr.width()
        block_line_spacing, block_line_spacing_type = (
            self.block_line_spacing(block)
        )

        layout_first_block = block == doc.firstBlock()
        if layout_first_block:
            x_offset = self.max_width - doc_margin
            self.x_offset_lst = [self.max_width - doc_margin]
            self.y_offset_lst = []
        else:
            x_offset = self.x_offset_lst[-1]

        char_idx = 0
        tl = block.layout()
        tl.beginLayout()
        option = doc.defaultTextOption()
        option.setWrapMode(QTextOption.WrapAnywhere)
        tl.setTextOption(option)

        shrink_height = 0
        width_list = []
        line_not_set = []
        ypos_list = []
        is_first_line = block_no == 0
        char_records = {}
        line_char_ids = []
        active_ruby_metric = None

        while True:
            inseparable_run_range = None
            line = tl.createLine()
            if not line.isValid():
                break

            line.setLineWidth(block_width)
            text_combine_length = text_combine_lengths.get(char_idx)
            is_text_combine = text_combine_length is not None
            if is_text_combine:
                combined_text = _utf16_slice(
                    blk_text, char_idx, text_combine_length
                )
                line.setNumColumns(max(1, _grapheme_count(combined_text)))
            else:
                line.setNumColumns(1)
                punctuation_run = _inseparable_punctuation_run(
                    blk_text, char_idx
                )
                if punctuation_run is not None:
                    if inline_unit_boundaries is None:
                        boundaries = set()
                        for start, length, _group_id in text_combine_ranges:
                            boundaries.update((start, start + length))
                        for metric in ruby_metrics:
                            boundaries.update((
                                metric.unit.start - block.position(),
                                metric.unit.end - block.position(),
                            ))
                        inline_unit_boundaries = tuple(sorted(boundaries))
                    run_start, punctuation_columns = punctuation_run
                    run_end = run_start + punctuation_columns
                    # Never merge an inseparable run across another inline
                    # annotation's layout unit.
                    boundary_index = bisect_right(
                        inline_unit_boundaries, char_idx
                    )
                    if boundary_index < len(inline_unit_boundaries):
                        run_end = min(
                            run_end,
                            inline_unit_boundaries[boundary_index],
                        )
                    columns = 1
                    # Qt may already shape the full pair as one column.
                    while (
                        run_end - run_start > 1
                        and columns < run_end - line.textStart()
                        and line.textStart() + line.textLength() < run_end
                    ):
                        columns += 1
                        line.setNumColumns(columns)
                    if (
                        run_end - run_start > 1
                        and line.textStart() + line.textLength() >= run_end
                    ):
                        inseparable_run_range = (run_start, run_end)

            available_height = self.available_height + doc_margin
            text_len = line.textLength()
            end_char = char_idx + text_len >= blk_text_len
            if active_ruby_metric is None:
                active_ruby_metric = ruby_starts.get(char_idx)
            ruby_metric = active_ruby_metric
            ruby_unit_start = (
                -1
                if ruby_metric is None
                else ruby_metric.unit.start - block.position()
            )
            ruby_unit_end = (
                -1
                if ruby_metric is None
                else ruby_metric.unit.end - block.position()
            )
            ruby_leading = ruby_base_leading.get(char_idx, 0.0)
            ruby_trailing = ruby_base_trailing.get(
                char_idx + text_len, 0.0
            )
            group_ruby = (
                ruby_metric is not None
                and ruby_metric.container.ruby_type == 'group'
            )
            if ruby_metric is not None and ruby_metric.extent > self.available_height:
                self.min_height = max(
                    self.min_height, doc_margin + ruby_metric.extent
                )
            force_ruby_wrap = (
                ruby_metric is not None
                and char_idx == ruby_unit_start
                and line_y_offset > doc_margin + 1e-6
                and line_y_offset + ruby_metric.extent
                > self.available_height + doc_margin
            )

            is_first_lbracket = False
            # _lbracket_shift = 0

            if char_idx + text_len > blk_text_len:
                ypos = ypos_list[-1] if len(ypos_list) > 0 else 0
                blk_line_spaces.append([0, 0, [ypos], char_idx])
                line.setPosition(QPointF(x_offset - block_width, ypos))
                continue

            num_rspaces, num_lspaces = 0, 0
            if utf16_indexing:
                text = _utf16_slice(
                    blk_text, char_idx, text_len
                ).replace('\n', '')
                num_rspaces = _utf16_length(text[len(text.rstrip()):])
                num_lspaces = _utf16_length(
                    text[:len(text) - len(text.lstrip())]
                )
            else:
                text = blk_text[char_idx: char_idx + text_len].replace('\n', '')
                num_rspaces = text_len - len(text.rstrip())
                num_lspaces = text_len - len(text.lstrip())

            if is_text_combine:
                # Whitespace is part of the authored horizontal run, not
                # vertical column leading around it.
                num_rspaces = num_lspaces = 0

            tbr_h = space_w = spacing_advance = 0
            char_idx += num_lspaces
            single_char_h = None
            text_combine_line_metrics = None
            line_base_width = block_width

            if char_idx < blk_text_len:
                cfmt = self.get_char_fontfmt(block_no, char_idx)
                line_base_width = cfmt.tbr.width()
                if inseparable_run_range is not None:
                    line_base_width = max(
                        self.get_char_fontfmt(block_no, position).tbr.width()
                        for position in range(*inseparable_run_range)
                    )
                space_shift = 0
                if num_lspaces > 0:
                    space_shift = num_lspaces * cfmt.space_width
                line_char_ids.append(char_idx)
                space_w = cfmt.space_width
                spacing_advance = (
                    cfmt.tbr.height() * (cfmt.letter_spacing - 1)
                )

                tbr_h = cfmt.tbr.height() + spacing_advance
                source_char = (
                    _utf16_char_at(blk_text, char_idx)
                    if utf16_indexing
                    else blk_text[char_idx]
                )
                char = (
                    source_char
                    if is_text_combine
                    else self._line_orientation_char(
                        line, source_char, text
                    )
                )
                if char != source_char:
                    char_records.setdefault(char_idx, {})[
                        'orientation_char'
                    ] = char
                is_first_lbracket = (
                    not compact_punctuation
                    and char_idx - num_lspaces == 0
                    and char in PUNSET_BRACKETL
                    and self.needs_vertical_rotation(char)
                )
                if is_first_lbracket:
                    _lbracket_shift = -_line_ink_bounds(
                        line, space_shift
                    ).left()

                if is_text_combine:
                    # A grouped run has no separate container format. Its
                    # leading fragment supplies the normal cell while natural
                    # horizontal flow and every fragment style are retained.
                    right_margin, left_margin = emphasis_margins(
                        block, line, vertical=True
                    )
                    natural_bounds = tate_chu_yoko_natural_bounds(line)
                    text_combine_width = max(
                        cfmt.tbr.width(),
                        natural_bounds.width(),
                    )
                    text_combine_height = max(
                        cfmt.tbr.height(), natural_bounds.height()
                    )
                    spacing_advance = (
                        text_combine_height * (cfmt.letter_spacing - 1)
                    )
                    tbr_h = text_combine_height + spacing_advance
                    text_combine_line_metrics = (
                        cfmt.tbr.width(),
                        right_margin,
                        left_margin,
                    )
                    char_records[char_idx] = {
                        'text_combine_height': text_combine_height,
                        'text_combine_width': text_combine_width,
                    }
                elif self.needs_vertical_rotation(char):
                    tbr, br = cfmt.punc_rect(char)
                    single_char_h = tbr.width()
                    tbr_h = tbr.width() * (
                        _grapheme_count(text) if utf16_indexing else text_len
                    )
                    if char.isalpha():
                        cw2 = cfmt.punc_rect(char+char)[1].width()
                        tbr_h = br.width() - (br.width() * 2 - cw2)
                    else:
                        # Rotated punctuation keeps Qt's natural advance;
                        # joined runs subdivide that same occupied extent.
                        tbr_h = line.naturalTextWidth() - num_lspaces * space_w
                    tbr_h += spacing_advance

                # Ruby owns one max(base, annotation) unit; do not desync its
                # shared paint, cursor, and hit geometry from those metrics.
                if (
                    compact_punctuation
                    and not is_text_combine
                    and ruby_metric is None
                    and char in PUNSET_COMPACT
                    and _grapheme_count(text.strip()) == 1
                ):
                    full_advance = max(tbr_h - spacing_advance, 0.0)
                    ink_bounds = _line_ink_bounds(line, space_shift)
                    visible_ink_advance = (
                        ink_bounds.width()
                        if self.needs_vertical_rotation(char)
                        else ink_bounds.height()
                    )
                    compact_advance = min(
                        full_advance,
                        max(
                            cfmt.tbr.height() / 2,
                            visible_ink_advance,
                        ),
                    )
                    tbr_h = compact_advance + spacing_advance
                    record = char_records.setdefault(char_idx, {})
                    record['compact_punctuation_advance'] = compact_advance
                    if char in PUNSET_BRACKETL:
                        record['compact_punctuation_leading_trim'] = max(
                            full_advance - compact_advance,
                            0.0,
                        )
            elif char_idx - num_lspaces < blk_text_len:
                cfmt = self.get_char_fontfmt(block_no, char_idx - num_lspaces)
                line_base_width = cfmt.tbr.width()
                tbr_h = cfmt.tbr.height() + cfmt.font_metrics.descent()
                space_w = cfmt.space_width

            # Zero tracking may collapse a narrow rotated glyph completely,
            # but a logical cell must never advance backwards.
            tbr_h = max(tbr_h, 0.0)

            line_position_y = line_y_offset + ruby_leading
            char_yoffset_lst = [line_position_y]
            if is_first_lbracket:
                char_yoffset_lst[0] += _lbracket_shift
            for _ in range(num_lspaces):
                char_yoffset_lst.append(min(available_height - tbr_h, char_yoffset_lst[-1] + space_w))
            blk_line_spaces.append([num_rspaces, num_lspaces, char_yoffset_lst, char_idx - num_lspaces])

            char_bottom = char_yoffset_lst[-1] + tbr_h
            out_of_vspace = (
                force_ruby_wrap
                or (
                    not group_ruby
                    and char_bottom + ruby_trailing
                    - max(spacing_advance, 0) > available_height
                )
            )
            if out_of_vspace:
                # switch to next line
                if char_idx == 0 and layout_first_block:
                    self.min_height = doc_margin + tbr_h

                line_y_offset = doc_margin
                line_position_y = line_y_offset + ruby_leading
                char_yoffset_lst[-1] = line_position_y
                char_yoffset_lst.append(line_position_y + tbr_h)
                for _ in range(num_rspaces):
                    char_yoffset_lst.append(min(char_yoffset_lst[-1] + space_w, available_height))
                line_bottom = char_yoffset_lst[-1] + ruby_trailing
            else:
                cfmt = self.get_char_fontfmt(block_no, char_idx)
                if cfmt is not None:
                    if text_combine_line_metrics is None:
                        right_margin, left_margin = emphasis_margins(
                            block, line, vertical=True
                        )
                        ruby_right, ruby_left = ruby_side_margins(
                            block, line, ruby_metrics, vertical=True
                        )
                        right_margin += ruby_right
                        left_margin += ruby_left
                        current_line_metrics = (
                            line_base_width,
                            right_margin,
                            left_margin,
                        )
                    else:
                        current_line_metrics = text_combine_line_metrics
                    width_list.append(current_line_metrics)
                else:
                    width_list.append((-1.0, 0.0, 0.0))

                char_yoffset_lst.append(char_bottom)
                for _ in range(num_rspaces):
                    char_yoffset_lst.append(min(char_yoffset_lst[-1] + space_w, available_height))
                line_bottom = char_yoffset_lst[-1] + ruby_trailing
                shrink_height = max(shrink_height, line_bottom)

            ypos_list.append(line_position_y)
            line_not_set.append(line)
            if out_of_vspace or end_char:
                if is_first_line:
                    line_spacing = self.identity_linespacing(
                        block_line_spacing_type
                    )
                else:
                    line_spacing = block_line_spacing
                if len(width_list) == 0:
                    width_list = [(block_width, 0.0, 0.0)]
                end_line, end_ypos, end_metrics = (
                    line,
                    line_position_y,
                    width_list[-1],
                )
                if out_of_vspace and text_combine_line_metrics is not None:
                    # This line belongs to the next column and therefore did
                    # not enter the previous column's width list.
                    end_metrics = text_combine_line_metrics
                if out_of_vspace and end_char and len(width_list) > 1:
                    column_metrics = width_list[:-1]
                else:
                    column_metrics = width_list
                idea_base_width = max(
                    metrics[0] for metrics in column_metrics
                )
                if idea_base_width == -1:
                    idea_base_width = block_width
                idea_right_margin = max(
                    metrics[1] for metrics in column_metrics
                )
                idea_left_margin = max(
                    metrics[2] for metrics in column_metrics
                )
                idea_line_width = (
                    idea_base_width
                    + idea_right_margin
                    + idea_left_margin
                )

                if len(line_char_ids) == 0:
                    line_char_ids = [char_idx]
                end_char_id = line_char_ids[-1]
                for cidx in line_char_ids:
                    char_records.setdefault(cidx, {}).update({
                        'line_width': idea_line_width,
                        'base_width': idea_base_width,
                        'right_margin': idea_right_margin,
                        'left_margin': idea_left_margin,
                    })
                line_char_ids = []

                x_offset -= self.calculate_line_spacing(
                    idea_line_width,
                    line_spacing,
                    block_line_spacing_type,
                )

                for line, ypos in zip(line_not_set[:-1], ypos_list[:-1]):
                    line.setPosition(QPointF(x_offset, ypos))
                if out_of_vspace:
                    if end_char:
                        end_base_width = end_metrics[0]
                        if end_base_width == -1:
                            end_base_width = block_width
                        end_width = (
                            end_base_width
                            + end_metrics[1]
                            + end_metrics[2]
                        )
                        if not len(line_not_set) == 1:
                            x_offset -= self.calculate_line_spacing(
                                end_width,
                                block_line_spacing,
                                block_line_spacing_type,
                            )
                        end_line.setPosition(QPointF(x_offset, end_ypos))
                        char_records.setdefault(end_char_id, {}).update({
                            'line_width': end_width,
                            'base_width': end_base_width,
                            'right_margin': end_metrics[1],
                            'left_margin': end_metrics[2],
                        })
                    else:
                        line_not_set = [end_line]
                        ypos_list = [end_ypos]
                        width_list = [end_metrics]
                        line_char_ids = [end_char_id]
                else:
                    end_line.setPosition(QPointF(x_offset, end_ypos))

                if out_of_vspace:
                    is_first_line = False

            strip_space_textlen = (
                _grapheme_count(text.lstrip())
                if utf16_indexing
                else text_len - num_lspaces
            )
            if strip_space_textlen > 1 and single_char_h is not None:
                run_height = max(line_bottom - line_y_offset, 0.0)
                cell_height = min(
                    single_char_h,
                    run_height / strip_space_textlen,
                )
                for ii in range(strip_space_textlen - 1):
                    blk_char_yoffset.append([
                        line_y_offset + ii * cell_height,
                        line_y_offset + (ii + 1) * cell_height,
                    ])
                blk_char_yoffset.append([blk_char_yoffset[-1][1], line_bottom])
            else:
                blk_char_yoffset.append([line_y_offset, line_bottom])

            line_y_offset = max(line_bottom, doc_margin)
            char_idx += text_len - num_lspaces
            if ruby_metric is not None and char_idx >= ruby_unit_end:
                active_ruby_metric = None
        tl.endLayout()

        self.layout_left = x_offset
        self.shrink_width = max(self.max_width - self.layout_left - doc_margin + 0.01, self.shrink_width)
        self.shrink_height = max(shrink_height + 0.01 - doc_margin, self.shrink_height)
        self.x_offset_lst.append(x_offset)
        self.y_offset_lst.append(blk_char_yoffset)
        self.line_spaces_lst.append(blk_line_spaces)
        self.per_char_records.append(char_records)
