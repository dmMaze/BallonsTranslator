import re
import unicodedata

from qtpy.QtCore import Qt, QRectF, QPointF, Signal, QSizeF
from qtpy.QtGui import QTextCharFormat, QTextDocument, QPixmap, QImage, QTransform, QPalette, QPainter, QTextFrame, QTextBlock, QAbstractTextDocumentLayout, QTextLayout, QFont, QFontMetricsF, QTextOption, QTextLine, QTextFormat

import cv2
import numpy as np
from typing import List, Optional, Tuple
from functools import lru_cache, cached_property

from ..misc import pixmap2ndarray, LruIgnoreArg
from ballontranslator.utils import shared as C
from ballontranslator.utils.fontformat import (
    pt2px,
    FontFormat,
    LineSpacingType,
    TextAlignment,
)
from .annotations import letter_spacing_value, text_combine_upright_ranges
from .rendering.indexing import (
    _grapheme_count,
    _utf16_char_at,
    _utf16_length,
    _utf16_slice,
)
from .rendering.emphasis import (
    draw_emphasis_marks,
    emphasis_ink_bounds,
    emphasis_margins,
)
from .rendering.glyph import draw_slanted_line
from .rendering.tate_chu_yoko import (
    tate_chu_yoko_ink_bounds,
    tate_chu_yoko_natural_bounds,
    tate_chu_yoko_transform,
)

PUNSET_HALF = {chr(i) for i in range(0x21, 0x7F)}

# CLREQ Appendix A: pause/stop marks stay upright, while parenthetical
# punctuation, dashes, ellipses, connectors, and indicators rotate.
PUNSET_PAUSEORSTOP = {
    '。', '．', '，', '、', '：', '；', '！', '‼', '？', '⁇', '⁈', '⁉',
}
PUNSET_ALIGNCENTER = {'·', '・', '‧', '●', '•'}
PUNSET_BRACKETL = {'「', '『', '“', '‘', '（', '《', '〈', '【', '〖', '〔', '［', '｛', '('}
PUNSET_BRACKETR = {'」', '』', '”', '’', '）', '》', '〉', '】', '〗', '〕', '］', '｝', ')'}
PUNSET_BRACKET = PUNSET_BRACKETL.union(PUNSET_BRACKETR)

PUNSET_NONBRACKET = {'⸺', '…', '⋯', '～', '-', '–', '—', '＿', '﹏', '~'}
PUNSET_VERNEEDROTATE = PUNSET_NONBRACKET.union(PUNSET_BRACKET).union(PUNSET_HALF)
PUNSET_STANDARD_VERTICAL_ROMAN = PUNSET_VERNEEDROTATE.difference(PUNSET_HALF)

PUNSET_ROTATE_ALIGNL = {'」', '』', '”', '’'}
PUNSET_ROTATE_ALIGNR = {'「', '『', '“', '‘'}

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


@lru_cache(maxsize=512)
def _font_metrics(ffamily: str, size: float, weight: int, italic: bool) -> QFontMetricsF:
    font = QFont(ffamily, int(size), weight, italic)
    font.setPointSizeF(size)
    return QFontMetricsF(font)

@lru_cache(maxsize=2048)
def get_punc_rect(char: str, ffamily: str, size: float, weight: int, italic: bool) -> List[QRectF]:
    fm = _font_metrics(ffamily, size, weight, italic)
    br = [fm.tightBoundingRect(char), fm.boundingRect(char)]
    return br

@lru_cache(maxsize=2048)
def get_char_width(char: str, ffamily: str, size: float, weight: int, italic: bool) -> int:
    fm = _font_metrics(ffamily, size, weight, italic)
    return fm.horizontalAdvance(char)

def punc_actual_rect(line: QTextLine, family: str, size: float, weight: int, italic: bool, stroke_width: float, h: int = None, w: int = None, space_shift = 0) -> List[int]:
    if h is None:
        h = int(line.height())
    if w is None:
        w = int(line.naturalTextWidth())
    pixmap = QImage(w * 2, h * 2, QImage.Format.Format_ARGB32)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    line.draw(p, QPointF(-line.x() - space_shift, -line.y()))
    p.end()
    mask = pixmap2ndarray(pixmap, keep_alpha=True)
    if mask is None:
        print(f'invalid text line!')
        return [0, 0, 1, 1]
    mask = mask[..., -1]
    
    ar = cv2.boundingRect(cv2.findNonZero(mask))
    # if stroke_width != 0:
    ar = np.array(ar, dtype=np.float64)
    ar[[0, 1]] += stroke_width
    ar[[2, 3]] -= stroke_width * 2
    ar = ar.tolist()
    return ar

@lru_cache(maxsize=2048)
def punc_actual_rect_cached(cached_args: LruIgnoreArg, char: str, family: str, size: float, weight: int, italic: bool, stroke_width: float, h: int, w: int) -> List[int]:
    '''
    char is actually not used, but can be set as some cache flag
    '''
    # QtextLine line is invisibale to lru
    return punc_actual_rect(cached_args.line, family, size, weight, italic, stroke_width, h, w, cached_args.space_shift)


def _block_cursor_position(block: QTextBlock, cursor_position: int) -> int:
    layout = block.layout()
    if cursor_position < -1:
        if not layout.preeditAreaText():
            return -1
        # Qt encodes IME preedit cursor positions as -(preeditCursor + 2).
        return layout.preeditAreaPosition() - (cursor_position + 2)

    block_position = block.position()
    if block_position <= cursor_position < block_position + block.length():
        return cursor_position - block_position
    return -1


class CharFontFormat:
    def __init__(
        self,
        fcmt: QTextCharFormat,
        letter_spacing_fallback: float = 1.0,
    ) -> None:
        font = fcmt.font()
        self.font = font
        self.font_metrics = QFontMetricsF(font)
        self.letter_spacing = letter_spacing_value(
            fcmt,
            letter_spacing_fallback,
        )

    @cached_property
    def br(self) -> QRectF:
        # return get_punc_rect('啊', self.family, self.size, self.weight, self.font.italic())[1]
        _, br1 = get_punc_rect('啊', self.family, self.size, self.weight, self.font.italic())
        _, br2 = get_punc_rect('木', self.family, self.size, self.weight, self.font.italic())
        return QRectF(min(br1.left(), br2.left()), br2.top(), max(br1.right(), br2.right()) - min(br1.left(), br2.left()), br2.height())

    @cached_property
    def tbr(self) -> QRectF:
        # return get_punc_rect('啊', self.family, self.size, self.weight, self.font.italic())[0]
        br1, _ = get_punc_rect('啊', self.family, self.size, self.weight, self.font.italic())
        br2, _ = get_punc_rect('木', self.family, self.size, self.weight, self.font.italic())
        return QRectF(min(br1.left(), br2.left()), br2.top(), max(br1.right(), br2.right()) - min(br1.left(), br2.left()), br2.height())

    @cached_property
    def space_width(self) -> int:
        return get_char_width(' ', self.family, self.size, self.weight, self.font.italic())

    def punc_rect(self, punc: str, family: str = None) -> List[QRectF]:
        if family is None:
            family = self.family
        return get_punc_rect(punc, family, self.size, self.weight, self.font.italic())

    @property
    def family(self) -> str:
        return self.font.family()

    @property
    def weight(self) -> float:
        return self.font.weight()

    @property
    def size(self) -> float:
        return self.font.pointSizeF()

    def punc_actual_rect(self, line: QTextLine, char: str, cache=False, stroke_width=0, h=None, w=None, space_shift=0) -> List[int]:
        if cache:
            cached_args = LruIgnoreArg(line=line, space_shift=space_shift)
            ar = punc_actual_rect_cached(cached_args, char, self.family, self.size, self.weight, self.font.italic(), stroke_width, h, w)
        else:
            ar =  punc_actual_rect(line, self.family, self.size, self.weight, self.font.italic(), stroke_width, h, w, space_shift)
        return ar


def line_draw_qt6(painter: QPainter, line: QTextLine, x: float, y: float, selected: bool, selection: QAbstractTextDocumentLayout.Selection = None, char_fmt: CharFontFormat = None, char: str = None, line_width: int = None):
    # some how qt6 line.draw doesn't allow pass FormatRange
    if selected:    
        qimg = QImage(int(line.naturalTextWidth()), int(line.height()), QImage.Format.Format_ARGB32)
        qimg.fill(Qt.GlobalColor.transparent)
        p = QPainter(qimg)
        line.draw(p, QPointF(-line.x(), -line.y()))
        p.end()
        qimg = qimg.convertToFormat(QImage.Format.Format_Alpha8)
        qimg.reinterpretAsFormat(QImage.Format.Format_Grayscale8)
        if char_fmt is None:
            painter.drawImage(QPointF(line.x() + x, line.y() + y), qimg)
        else:
            act_rect = char_fmt.punc_actual_rect(line, char, cache=True)
            tbr = QRectF(0, act_rect[1], line_width, act_rect[3])
            tgt_rect = QRectF(line.x() + x, line.y() + y + tbr.y(), line_width, tbr.height())
            painter.drawImage(tgt_rect, qimg, tbr)
    else:
        line.draw(painter, QPointF(x, y))

def line_draw_qt5(painter: QPainter, line: QTextLine, x: float, y: float, selected: bool, selection: QAbstractTextDocumentLayout.Selection = None, char_fmt: CharFontFormat = None, char: str = None, line_width: int = None):
    o = None
    if selected:
        o = QTextLayout.FormatRange()
        o.start = line.textStart()
        o.length = line.textLength()
        o.format = selection.format
    line.draw(painter, QPointF(x, y), o)


class SceneTextLayout(QAbstractTextDocumentLayout):
    size_enlarged = Signal()
    def __init__(self, doc: QTextDocument, fontformat: FontFormat) -> None:
        super().__init__(doc)
        self.max_height = 0
        self.max_width = 0
        self.available_width = 0
        self.available_height = 0
        self.line_spacing = fontformat.line_spacing
        self.letter_spacing = fontformat.letter_spacing
        self.linespacing_type = fontformat.line_spacing_type
        self.fontformat = fontformat
        self.render_delegate = None
        self.layout_generation = 0
        self.render_failure_handler = None
        self.defer_cursor_paint = False
        self.deferred_cursor_position = -1
        self.publishing_size_enlargement = False
        # QWidgetTextControl routes its mouse and drag hit tests through this
        # layout.  Nonlinear visual effects can therefore restore source
        # coordinates here without replacing Qt's editing state machine.
        self.input_point_mapper = None

        self.x_offset_lst = []
        self.y_offset_lst = []

        self.block_charfmt_lst = []
        self.block_ideal_width = []
        self.need_ideal_width = False
        self.block_ideal_height = []
        self.need_ideal_height = False
        self._map_charidx2frag = []
        self._max_font_size = -1

        self.foreground_pixmap: QPixmap = None
        self.relayout_on_changed = True

        # Effect padding is derived layout state, not rich-text content.
        # QTextDocument margins create undo entries in supported Qt bindings.
        self._effect_padding = max(0.0, float(doc.documentMargin()))

        # relative bottom/right
        self.shrink_height = 0 
        self.shrink_width = 0

        # The upstream vertical-stroke renderer clones a document and reuses
        # the already-computed draw offsets.  Keep that neutral-state path
        # intact; feature rendering only replaces it while Glyph Slant is
        # active.
        self._is_painting_stroke = False
        self._draw_offset = []
        self.text_padding = 0

    def source_cursor_rect(self, cursor_position: int):
        """Return a layout-owned caret rectangle, or defer to Qt.

        Horizontal layout uses Qt's native cursor geometry. Vertical layout
        overrides this because its caret is horizontal.

        >>> SceneTextLayout.source_cursor_rect(None, 0) is None
        True
        """
        return None

    def annotation_ink_bounds(self) -> QRectF:
        """Return annotation paint overflow without changing layout bounds."""
        return QRectF()

    def _begin_layout_generation(self):
        self.layout_generation += 1

    def _emit_size_enlarged(self) -> None:
        """Publish a resize without exposing half-settled paint geometry."""
        self.publishing_size_enlargement = True
        try:
            self.size_enlarged.emit()
        finally:
            self.publishing_size_enlargement = False

    def map_input_point(self, point: QPointF) -> QPointF:
        mapper = self.input_point_mapper
        return QPointF(point) if mapper is None else mapper(QPointF(point))

    def _report_render_failure(self, error, effect_pass=False):
        handler = self.render_failure_handler
        if handler is not None:
            handler(error, effect_pass)

    def setMaxSize(self, max_width: int, max_height: int, relayout=True):
        self.max_height = max_height
        self.max_width = max_width
        doc_margin = self._effect_padding * 2
        self.available_width = max(max_width -  doc_margin, 0)
        self.available_height = max(max_height - doc_margin, 0)
        if relayout:
            self.reLayoutForResize()

    def reLayoutForResize(self):
        self.reLayout()

    def setLineSpacing(self, line_spacing: float):
        if self.line_spacing != line_spacing:
            self.line_spacing = line_spacing
            self.reLayout()

    def setLineSpacingType(self, linespacing_type: int):
        if self.linespacing_type != linespacing_type:
            self.linespacing_type = linespacing_type
            self.reLayout()

    def calculate_line_spacing(self, size: float, line_spacing: float = 1):
        if self.linespacing_type == LineSpacingType.Proportional:
            return line_spacing * size
        elif self.linespacing_type == LineSpacingType.Distance:
            return line_spacing * 10 + size
        else:
            raise Exception(f'Invalid line spacing type: {self.linespacing_type}')

    def identity_linespacing(self):
        if self.linespacing_type == LineSpacingType.Proportional:
            return 1.
        elif self.linespacing_type == LineSpacingType.Distance:
            return 0.
        else:
            raise Exception(f'Invalid line spacing type: {self.linespacing_type}')

    def blockBoundingRect(self, block: QTextBlock) -> QRectF:
        if not block.isValid():
            return QRectF()
        br = block.layout().boundingRect()
        rect = QRectF(0, 0, br.width(), br.height())
        return rect

    def setEffectPadding(self, padding):
        self._effect_padding = max(0.0, float(padding))
        doubled_margin = self._effect_padding * 2
        self.max_height = doubled_margin + self.available_height
        self.max_width = doubled_margin + self.available_width

    def effectPadding(self) -> float:
        return self._effect_padding

    def documentSize(self) -> QSizeF:
        return QSizeF(self.max_width, self.max_height)

    def frameBoundingRect(self, frame: QTextFrame) -> QRectF:
        return QRectF(
            0,
            0,
            max(self.document().pageSize().width(), self.max_width),
            2147483647,
        )

    def documentChanged(self, position: int, charsRemoved: int, charsAdded: int) -> None:
        if not self.relayout_on_changed:
            return
        self.reLayoutEverything()
        
    def reLayoutEverything(self):
        self._max_font_size = -1
        block = self.document().firstBlock()
        self.block_charfmt_lst = []
        self.block_ideal_width = []
        self.block_ideal_height = []
        self._map_charidx2frag = []
        while block.isValid():
            charfmt_lst, ideal_width, char_idx = [], -1, 0
            ideal_height = 0
            charidx_map = {}
            it = block.begin()
            frag_idx = 0
            while not it.atEnd():
                fragment = it.fragment()
                fcmt = fragment.charFormat()
                cfmt = CharFontFormat(fcmt, self.letter_spacing)
                charfmt_lst.append(cfmt)
                if cfmt.size > self._max_font_size:
                    self._max_font_size = cfmt.size

                if self.need_ideal_width:
                    w_ = cfmt.br.width()
                    if ideal_width < w_:
                        ideal_width = w_

                if self.need_ideal_height:
                    h_ = cfmt.punc_rect('木fg')[0].height()
                    if ideal_height < h_:
                        ideal_height = h_

                text_len = fragment.length()
                for _ in range(text_len):
                    charidx_map[char_idx] = frag_idx
                    char_idx += 1
                it += 1
                frag_idx += 1

            self.block_charfmt_lst.append(charfmt_lst)
            self.block_ideal_width.append(ideal_width)
            self.block_ideal_height.append(ideal_height)
            self._map_charidx2frag.append(charidx_map)
            block = block.next()
        self.reLayout()

    def max_font_size(self, to_px=False) -> float:
        fs = self._max_font_size if self._max_font_size > 0 else self.document().defaultFont().pointSizeF()
        if to_px:
            fs = pt2px(fs)
        return fs

    def minSize(self):
        return (self.shrink_height + self.text_padding, self.shrink_width + self.text_padding)
    
    def get_char_fontfmt(self, block_number: int, char_idx: int) -> CharFontFormat:
        charidx2frag_map = self._map_charidx2frag[block_number]
        if len(charidx2frag_map) == 0:
            return None
        if char_idx not in charidx2frag_map:    # caused by inputmethod
            char_idx = len(charidx2frag_map) - 1
        frag_idx = charidx2frag_map[char_idx]
        return self.block_charfmt_lst[block_number][frag_idx]
    

class VerticalTextDocumentLayout(SceneTextLayout):
    def __init__(self, doc: QTextDocument, fontformat: FontFormat):
        super().__init__(doc, fontformat)

        self.line_spaces_lst = []
        self.min_height = 0
        self.layout_left = 0
        self.has_selection = False
        self.draw_shifted = 0

        self.need_ideal_width = True
        self.line_draw = line_draw_qt6 if C.FLAG_QT6 else line_draw_qt5

        self.per_char_records = []
        self.text_combine_ranges = []
        self._annotation_ink_bounds = QRectF()
        self._cursor_update_rect = QRectF()
        self._resize_layout_max_width = None
        self._resize_layout_available_height = None
        self._resize_layout_padding = None
        self._alignment_x_shift = 0.0

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

    def _translate_columns(self, x_shift: float) -> None:
        """Translate every settled vertical-layout x coordinate together."""
        if abs(x_shift) <= 1e-9:
            return
        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            for line_number in range(layout.lineCount()):
                line = layout.lineAt(line_number)
                position = line.position()
                position.setX(position.x() + x_shift)
                line.setPosition(position)
            block = block.next()
        self.x_offset_lst = [
            x_offset + x_shift for x_offset in self.x_offset_lst
        ]
        self.layout_left += x_shift

    def _column_content_width(self) -> float:
        if not self.x_offset_lst:
            return 0.0
        return max(0.0, self.x_offset_lst[0] - self.layout_left)

    def _desired_alignment_x_shift(self) -> float:
        slack = max(0.0, self.available_width - self._column_content_width())
        if self.fontformat.alignment == TextAlignment.Left:
            return -slack
        if self.fontformat.alignment == TextAlignment.Center:
            return -slack / 2
        return 0.0

    def apply_alignment(self) -> bool:
        """Translate settled columns without reshaping or resizing the box."""
        desired = self._desired_alignment_x_shift()
        x_shift = desired - self._alignment_x_shift
        if abs(x_shift) <= 1e-9:
            return False
        self._begin_layout_generation()
        self._translate_columns(x_shift)
        self._alignment_x_shift = desired
        self._refresh_annotation_ink_bounds()
        return True

    def reLayout(self):
        self._begin_layout_generation()
        self.min_height = 0
        self.layout_left = 0
        self.line_spaces_lst = []
        self.per_char_records = []
        self.text_combine_ranges = []
        self.draw_shifted = 0
        self.shrink_height = 0
        self.shrink_width = 0
        self.text_padding = 0
        self._alignment_x_shift = 0.0
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
        self._alignment_x_shift = self._desired_alignment_x_shift()
        self._translate_columns(self._alignment_x_shift)
        self.updateDrawOffsets()
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

        previous_alignment_shift = self._alignment_x_shift
        desired_alignment_shift = self._desired_alignment_x_shift()
        column_shift = (
            width_shift
            + desired_alignment_shift
            - previous_alignment_shift
        )
        if abs(column_shift) > 1e-9:
            self._begin_layout_generation()
            self._translate_columns(column_shift)
            self._refresh_annotation_ink_bounds()
        self._alignment_x_shift = desired_alignment_shift
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
                        yoff = -line.ascent() - (line_width - cfmt.font_metrics.capHeight()) / 2

                    else:   # () （）
                        non_bracket_br = cfmt.punc_actual_rect(line, char, cache=True, space_shift=space_shift)
                        yoff = -non_bracket_br[1] - non_bracket_br[3]
                        if char in PUNSET_BRACKETL:
                            if ii == 0:
                                xoff = -non_bracket_br[0]
                            else:
                                xoff = 0
                        else:
                            xoff = -non_bracket_br[0]

                        if char in PUNSET_ROTATE_ALIGNL:
                            yoff = yoff
                        elif char in PUNSET_ROTATE_ALIGNR:
                            yoff = yoff - (line_width - non_bracket_br[3])
                        else:
                            yoff = yoff - (line_width - non_bracket_br[3]) / 2

                else:
                    standard_roman = (
                        self.fontformat.standard_vertical_roman_alignment
                        and _is_non_fullwidth_roman(char)
                    )
                    if standard_roman:
                        # Roman ink has ordinary baseline metrics. Center it
                        # without priming Qt's process-global glyph raster
                        # cache during layout.
                        tight_rect, _ = cfmt.punc_rect(char)
                        xoff = (
                            -tight_rect.left()
                            + (line_width - tight_rect.width()) / 2
                        )
                        yoff = (
                            -line.ascent()
                            - tight_rect.top()
                            + (cfmt.tbr.height() - tight_rect.height()) / 2
                        )
                    else:
                        act_rect = cfmt.punc_actual_rect(
                            line,
                            char,
                            cache=True,
                            space_shift=space_shift,
                        )
                        if self.centers_vertical_glyph(char):
                            xoff = (
                                -act_rect[0]
                                + (line_width - act_rect[2]) / 2
                            )
                            yoff = (
                                -act_rect[1]
                                + (cfmt.tbr.height() - act_rect[3]) / 2
                            )
                        elif char in PUNSET_PAUSEORSTOP:
                            # CLREQ's Mainland convention places stop marks at
                            # the upper-right of their full character frame.
                            xoff = -act_rect[0] + line_width - act_rect[2]
                            yoff = -act_rect[1]
                        else:
                            yoff = min(
                                cfmt.br.top() - cfmt.tbr.top(),
                                -cfmt.tbr.top() - line.ascent(),
                            )
                            xoff = (
                                -act_rect[0]
                                + (line_width - act_rect[2]) / 2
                            )
                    
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
        line_width = record.get('line_width', cell_width)
        return QRectF(
            line.x() + (line_width - cell_width) / 2,
            line.y(),
            cell_width,
            cell_height,
        )

    def annotation_ink_bounds(self) -> QRectF:
        """Return cached Tate-chu-yoko and attached emphasis ink."""
        return QRectF(self._annotation_ink_bounds)

    def _refresh_annotation_ink_bounds(self) -> None:
        """Measure paint overflow after final line placement.

        >>> callable(VerticalTextDocumentLayout._refresh_annotation_ink_bounds)
        True
        """
        bounds = QRectF()
        block = self.document().firstBlock()
        while block.isValid():
            text_layout = block.layout()
            for line_number in range(text_layout.lineCount()):
                cell = self.tate_chu_yoko_cell_rect(block, line_number)
                if cell is None:
                    continue
                placement = self.vertical_line_placement(
                    block, line_number
                )
                if placement is None:
                    continue
                line, offset, orientation = placement
                candidates = (
                    cell,
                    tate_chu_yoko_ink_bounds(line, cell),
                    emphasis_ink_bounds(
                        block,
                        line,
                        vertical=True,
                        offset=offset,
                        orientation=orientation,
                    ),
                )
                for candidate in candidates:
                    if candidate.isEmpty():
                        continue
                    bounds = (
                        QRectF(candidate)
                        if bounds.isEmpty()
                        else bounds.united(candidate)
                    )
            block = block.next()
        self._annotation_ink_bounds = bounds

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
                x, y = position.x(), position.y()
                cfmt = self.get_char_fontfmt(block.blockNumber(), cpos)
                metrics = (
                    cfmt.font_metrics
                    if cfmt is not None
                    else QFontMetricsF(block.charFormat().font())
                )
                line_spaces = self.line_spaces_lst[block.blockNumber()]
                if line.lineNumber() < len(line_spaces):
                    _right, _left, offsets, line_position = line_spaces[
                        line.lineNumber()
                    ]
                    offset_index = cpos - line_position
                    if 0 <= offset_index < len(offsets):
                        y = offsets[offset_index]
                return QRectF(x, y, metrics.height(), 2.0)
            block = block.next()
        return QRectF()

    def draw(self, painter: QPainter, context: QAbstractTextDocumentLayout.PaintContext) -> None:
        doc = self.document()
        self.deferred_cursor_position = context.cursorPosition
        painter.save()
        block = doc.firstBlock()
        cursor_block = None
        context_sel = context.selections
        has_selection = False
        selection = None
        render_delegate = self.render_delegate
        custom_rendering = render_delegate is not None
        if len(context_sel) > 0:
            has_selection = True
            selection = context_sel[0]

        while block.isValid():
            blk_no = block.blockNumber()
            blpos, bllen = block.position(), block.length()
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
            char_records = self.per_char_records[blk_no]
            
            line_spaces_lst = self.line_spaces_lst[blk_no]
            uniform_block_drawn = (
                custom_rendering
                and render_delegate.draw_uniform_block(
                    painter, block, context
                )
            )

            if _block_cursor_position(block, context.cursorPosition) >= 0:
                cursor_block = block

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

                char = (
                    _utf16_char_at(blk_text, char_idx)
                    if utf16_indexing
                    else blk_text[char_idx]
                )
                cfmt = self.get_char_fontfmt(blk_no, char_idx)
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
                        )
                    continue
                selected = False
                if has_selection:
                    sel_start = selection.cursor.selectionStart() - blpos 
                    sel_end = selection.cursor.selectionEnd() - blpos
                    line_start = line.textStart()
                    line_end = line_start + line.textLength()
                    if line_start < sel_end and line_end > sel_start:
                        selected = True

                line_width = -1
                if char_idx in char_records:
                    line_width = char_records[char_idx]['line_width']
                if line_width < 0:
                    line_width = cfmt.tbr.width()
                
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
                elif self.needs_vertical_rotation(char):
                    line_x, line_y = line.x(), line.y()
                    y_x = line_y - line_x
                    y_p_x = line_y + line_x
                    transform = QTransform(0, 1, 0, -1, 0, 0, y_p_x, y_x, 1)
                    inv_transform = QTransform(0, -1, 0, 1, 0, 0, -y_x, y_p_x, 1)
                    painter.setTransform(transform, True)
                    self.line_draw(painter, line, xoff,  yoff, selected, selection, char_fmt=None)
                    painter.setTransform(inv_transform, True)
                else:
                    self.line_draw(painter, line, xoff, yoff, selected, selection, char_fmt=cfmt, char=char, line_width=line_width)

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
                    )

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

    def hitTest(self, point: QPointF, accuracy: Qt.HitTestAccuracy) -> int:
        point = self.map_input_point(point)
        text_combine_hit = self._tate_chu_yoko_hit_test(point)
        if text_combine_hit is not None:
            return text_combine_hit
        blk = self.document().firstBlock()
        custom_rendering = self.render_delegate is not None
        x, y = point.x(), point.y()
        off = 0
        while blk.isValid():
            blk_no = blk.blockNumber()
            blk_text = blk.text()
            has_text_combine = bool(self.text_combine_ranges[blk_no])
            utf16_indexing = (
                custom_rendering
                or has_text_combine
                or _utf16_length(blk_text) != len(blk_text)
            )
            blk_char_yoffset = self.y_offset_lst[blk_no]
            nyoffset = len(blk_char_yoffset)
            rect = blk.layout().boundingRect()
            rect_left = rect.left()
            rect_right = rect.right()
            rect_right, rect_left = self.x_offset_lst[blk_no], self.x_offset_lst[blk_no+1]
            if rect_left <= x and rect_right >= x:
                layout = blk.layout()
                for ii in range(layout.lineCount()):
                    line_top, line_bottom = blk_char_yoffset[min(nyoffset - 1, ii)]
                    line = layout.lineAt(ii)
                    line_xy = line.position()
                    if not line_xy.x() <= x:
                        continue 
                    if line_top > y:
                        off = min(off, line.textStart())
                    elif line_bottom < y:
                        off = max(off, line.textStart() + line.textLength())
                    else:
                        num_rspaces, num_lspaces, char_yoffset_lst, line_pos = self.line_spaces_lst[blk_no][ii]
                        if num_rspaces > 0 or num_lspaces > 0:
                            for ii, (ytop, ybottom) in enumerate(zip(char_yoffset_lst[:-1], char_yoffset_lst[1:])):
                                dis_top, dis_bottom = y - ytop, ybottom - y
                                if dis_top >= 0 and dis_bottom >= 0:
                                    off = ii + line_pos if dis_top < dis_bottom else ii + 1 + line_pos
                                    break
                            break
                        else:
                            if self.is_tate_chu_yoko_line(blk, ii):
                                placement = self.vertical_line_placement(
                                    blk, ii
                                )
                                if placement is not None:
                                    _line, _offset, transform = placement
                                    off = self._tate_chu_yoko_hit_position(
                                        line, transform, point
                                    )
                                    break
                            ntr = line.naturalTextRect()
                            off = line.textStart()
                            if utf16_indexing:
                                # This path consumes Qt UTF-16 positions, so
                                # never return a caret inside one glyph run.
                                after = line_bottom - y < y - line_top
                                if line.textLength() > 1:
                                    after = after or (
                                        ntr.right() - x < x - ntr.left()
                                    )
                                if after:
                                    off += line.textLength()
                            else:
                                # Preserve the upstream neutral hit-test
                                # behavior.  General vertical grapheme fixes
                                # are deliberately outside this feature PR.
                                if line.textLength() != 1:
                                    if line_bottom - y < y - line_top:
                                        off += 2
                                    elif ntr.right() - x < x - ntr.left():
                                        off += 1
                                elif line_bottom - y < y - line_top:
                                    off += 1
                            break
                break
            blk = blk.next()
        return blk.position() + off
    
    def layoutBlock(self, block: QTextBlock):
        doc = self.document()

        block.clearLayout()
        doc_margin = self._effect_padding
        line_y_offset = doc_margin
        blk_char_yoffset = []
        blk_line_spaces = []

        block_no = block.blockNumber()
        is_final_block = block == doc.lastBlock()
        blk_text = block.text()
        custom_rendering = self.render_delegate is not None
        text_combine_ranges = text_combine_upright_ranges(block)
        self.text_combine_ranges.append(text_combine_ranges)
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

        while True:
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
            
            available_height = self.available_height + doc_margin
            text_len = line.textLength()
            end_char = char_idx + text_len >= blk_text_len

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

            tbr_h = space_w = let_sp_offset = 0
            char_idx += num_lspaces
            single_char_h = None
            text_combine_line_width = None

            if char_idx < blk_text_len:
                cfmt = self.get_char_fontfmt(block_no, char_idx)
                space_shift = 0
                if num_lspaces > 0:
                    space_shift = num_lspaces * cfmt.space_width
                line_char_ids.append(char_idx)
                space_w = cfmt.space_width
                let_sp_offset = (
                    cfmt.tbr.height() * (cfmt.letter_spacing - 1)
                )

                tbr_h = cfmt.tbr.height() + let_sp_offset
                char = (
                    _utf16_char_at(blk_text, char_idx)
                    if utf16_indexing
                    else blk_text[char_idx]
                )
                is_first_lbracket = (
                    char_idx - num_lspaces == 0
                    and char in PUNSET_BRACKETL
                    and self.needs_vertical_rotation(char)
                )
                if is_first_lbracket:
                    _lbracket_shift = -cfmt.punc_actual_rect(line, char, cache=True, space_shift=space_shift)[0]

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
                    let_sp_offset = (
                        text_combine_height * (cfmt.letter_spacing - 1)
                    )
                    tbr_h = text_combine_height + let_sp_offset
                    text_combine_line_width = (
                        cfmt.tbr.width()
                        + 2 * max(right_margin, left_margin)
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
                    elif char in {'…', '⋯', '—', '～'}:
                        tbr_h = line.naturalTextWidth() - num_lspaces * space_w
                        next_char_idx = char_idx + (
                            _utf16_length(char) if utf16_indexing else 1
                        )
                        if (
                            next_char_idx < blk_text_len
                            and (
                                _utf16_char_at(blk_text, next_char_idx)
                                if utf16_indexing
                                else blk_text[next_char_idx]
                            ) == char
                        ):
                            tbr_h -= let_sp_offset
                    else:
                        tbr_h = line.naturalTextWidth() - num_lspaces * space_w
                    tbr_h += let_sp_offset
            elif char_idx - num_lspaces < blk_text_len:
                cfmt = self.get_char_fontfmt(block_no, char_idx - num_lspaces)
                tbr_h = cfmt.tbr.height() + cfmt.font_metrics.descent()
                space_w = cfmt.space_width
            
            if num_lspaces == 0 and tbr_h != 0 and not is_text_combine:
                ntw = line.naturalTextWidth()
                shifted = ntw - cfmt.br.width()
                if is_final_block:
                    self.draw_shifted = max(self.draw_shifted, shifted)

            char_yoffset_lst = [line_y_offset]
            if is_first_lbracket:
                char_yoffset_lst[0] += _lbracket_shift
            for _ in range(num_lspaces):
                char_yoffset_lst.append(min(available_height - tbr_h, char_yoffset_lst[-1] + space_w))
            blk_line_spaces.append([num_rspaces, num_lspaces, char_yoffset_lst, char_idx - num_lspaces])
            
            char_bottom = char_yoffset_lst[-1] + tbr_h
            out_of_vspace = char_bottom - max(let_sp_offset, 0) > available_height
            if out_of_vspace:
                # switch to next line
                if char_idx == 0 and layout_first_block:
                    self.min_height = doc_margin + tbr_h
                    
                line_y_offset = doc_margin
                
                char_yoffset_lst[-1] = line_y_offset
                char_yoffset_lst.append(line_y_offset + tbr_h)
                for _ in range(num_rspaces):
                    char_yoffset_lst.append(min(char_yoffset_lst[-1] + space_w, available_height))
                line_bottom = char_yoffset_lst[-1]
            else:
                cfmt = self.get_char_fontfmt(block_no, char_idx)
                if cfmt is not None:
                    if text_combine_line_width is None:
                        right_margin, left_margin = emphasis_margins(
                            block, line, vertical=True
                        )
                        current_line_width = (
                            cfmt.tbr.width()
                            + 2 * max(right_margin, left_margin)
                        )
                    else:
                        current_line_width = text_combine_line_width
                    width_list.append(current_line_width)
                else:
                    width_list.append(-1)

                char_yoffset_lst.append(char_bottom)
                for _ in range(num_rspaces):
                    char_yoffset_lst.append(min(char_yoffset_lst[-1] + space_w, available_height))
                line_bottom = char_yoffset_lst[-1]
                shrink_height = max(shrink_height, line_bottom)
            
            ypos_list.append(line_y_offset)
            line_not_set.append(line)
            if out_of_vspace or end_char:
                if is_first_line:
                    line_spacing = self.identity_linespacing()
                else:
                    line_spacing = self.line_spacing
                if len(width_list) == 0:
                    width_list = [block_width]
                end_line, end_ypos, end_w = (
                    line,
                    line_y_offset,
                    width_list[-1],
                )
                if out_of_vspace and text_combine_line_width is not None:
                    # This line belongs to the next column and therefore did
                    # not enter the previous column's width list.
                    end_w = text_combine_line_width
                idea_line_width = -1
                if out_of_vspace and end_char and len(width_list) > 1:
                    idea_line_width = max(width_list[:-1])
                else:
                    idea_line_width = max(width_list)
                if idea_line_width == -1:
                    idea_line_width = block_width

                if len(line_char_ids) == 0:
                    line_char_ids = [char_idx]
                end_char_id = line_char_ids[-1]
                for cidx in line_char_ids:
                    char_records.setdefault(cidx, {})[
                        'line_width'
                    ] = idea_line_width
                line_char_ids = []

                x_offset -= self.calculate_line_spacing(
                    idea_line_width, line_spacing
                )
                
                for line, ypos in zip(line_not_set[:-1], ypos_list[:-1]):
                    line.setPosition(QPointF(x_offset, ypos))
                if out_of_vspace:
                    if end_char:
                        if not len(line_not_set) == 1:
                            x_offset -= self.calculate_line_spacing(
                                end_w, line_spacing
                            )
                        end_line.setPosition(QPointF(x_offset, end_ypos))
                        char_records.setdefault(end_char_id, {})[
                            'line_width'
                        ] = end_w
                    else:
                        line_not_set = [end_line]
                        ypos_list = [end_ypos]
                        width_list = [end_w]
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
                for ii in range(strip_space_textlen - 1):
                    blk_char_yoffset.append([line_y_offset + ii * single_char_h, line_y_offset + (ii + 1) * single_char_h])
                blk_char_yoffset.append([blk_char_yoffset[-1][1], line_bottom])
            else:
                blk_char_yoffset.append([line_y_offset, line_bottom])

            line_y_offset = max(line_bottom, doc_margin)
            char_idx += text_len - num_lspaces
        tl.endLayout()
            
        self.layout_left = x_offset - self.draw_shifted
        self.shrink_width = max(self.max_width - self.layout_left - doc_margin + 0.01, self.shrink_width)
        self.shrink_height = max(shrink_height + 0.01 - doc_margin, self.shrink_height)
        self.x_offset_lst.append(x_offset)
        self.y_offset_lst.append(blk_char_yoffset)
        self.line_spaces_lst.append(blk_line_spaces)
        self.per_char_records.append(char_records)

class HorizontalTextDocumentLayout(SceneTextLayout):

    def __init__(self, doc: QTextDocument, fontformat: FontFormat):
        super().__init__(doc, fontformat)
        self.need_ideal_height = True

    def reLayout(self):
        self._begin_layout_generation()
        doc = self.document()
        doc_margin = self._effect_padding
        self.text_padding = 0
        self.shrink_height = 0
        self.shrink_width = 0
        block = doc.firstBlock()
        while block.isValid():
            self.layoutBlock(block)
            block = block.next()
        
        if len(self.y_offset_lst) > 0:
            new_height = self.shrink_height
        else:
            new_height = doc_margin
        if new_height > self.available_height:
            self.max_height = new_height + doc_margin * 2
            self.available_height = new_height
            self._emit_size_enlarged()

        if doc.defaultTextOption().alignment() == Qt.AlignmentFlag.AlignCenter:
            block = doc.firstBlock()
            y_offset = (self.max_height - new_height) / 2 - doc_margin
            while block.isValid():
                tl = block.layout()
                for ii in range(tl.lineCount()):
                    line = tl.lineAt(ii)
                    line_pos = line.position()
                    line_pos.setY(y_offset + line_pos.y())
                    line.setPosition(line_pos)
                block = block.next()

        self.documentSizeChanged.emit(QSizeF(self.max_width, self.max_height))

    def hitTest(self, point: QPointF, accuracy: Qt.HitTestAccuracy) -> int:
        point = self.map_input_point(point)
        blk = self.document().firstBlock()
        x, y = point.x(), point.y()
        off = 0
        while blk.isValid():
            rect = blk.layout().boundingRect()
            if rect.top() <= y and rect.bottom() >= y:
                layout = blk.layout()
                for ii in range(layout.lineCount()):
                    line = layout.lineAt(ii)
                    ntr = line.naturalTextRect()
                    if ntr.top() < y and ntr.bottom() >= y:
                        off = line.xToCursor(point.x(), QTextLine.CursorBetweenCharacters)
                        break
                    elif ntr.left() > x:
                        off = min(off, line.textStart())
                    else:
                        off = max(off, line.textStart() + line.textLength())
                break
            blk = blk.next()
        return blk.position() + off

    def layoutBlock(self, block: QTextBlock):
        doc = self.document()
        block.clearLayout()
        tl = block.layout()
        
        option = doc.defaultTextOption()
        # maybe an option for it
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        tl.setTextOption(option)
        font = block.charFormat().font()
        
        # fm = QFontMetrics(font)
        doc_margin = self._effect_padding

        block_height = self.block_ideal_height[block.blockNumber()]
        if block_height == 0:
            tbr, br = get_punc_rect('木fg', font.family(), font.pointSizeF(), font.weight(), font.italic())
            block_height = tbr.height()
        if block == doc.firstBlock():
            self.x_offset_lst = []
            self.y_offset_lst = []
            # y_offset = -tbr.top() - fm.ascent() + doc_margin
            # y_offset = min(br.top() - tbr.top(), -tbr.top() - fm.ascent()) + doc_margin
            y_offset = doc_margin
        else:
            y_offset = self.y_offset_lst[-1]

        line_idx = 0
        tl.beginLayout()
        shrink_width = 0
        char_idx = 0
        blk_no = block.blockNumber()
        is_last_block = blk_no == self.document().blockCount() - 1
        is_first_block = blk_no == 0
        text_padding = 0
        is_first_line = False

        while True:
            line = tl.createLine()
            if not line.isValid():
                break
            # line.setLeadingIncluded(False)
            line.setLineWidth(self.available_width)
            nchar = line.textLength()

            dy = 0
            idea_height = -1
            if nchar > 0:
                tgt_cfmt = None
                tgt_size = -1
                for ii in range(nchar):
                    cfmt = self.get_char_fontfmt(blk_no, char_idx + ii)
                    if cfmt is None:
                        break
                    sz = cfmt.font.pointSizeF()
                    if sz > tgt_size:
                        tgt_size = sz
                        tgt_cfmt = cfmt
                if tgt_cfmt is not None:
                    font = tgt_cfmt.font
                    tbr, br = get_punc_rect('木fg', font.family(), font.pointSizeF(), font.weight(), font.italic())
                    dy = -tbr.top() - line.ascent()
                    idea_height = tbr.height()

            if idea_height == -1:
                idea_height = block_height

            over_margin, under_margin = emphasis_margins(
                block, line, vertical=False
            )
            y_offset += over_margin
            line.setPosition(QPointF(doc_margin, y_offset + dy))
            tw = line.naturalTextWidth()
            shrink_width = max(tw, shrink_width)
            self.shrink_height = max(
                idea_height + y_offset - doc_margin + under_margin,
                self.shrink_height,
            )
            y_offset += (
                self.calculate_line_spacing(idea_height, self.line_spacing)
                + under_margin
            )
            line_idx += 1
            char_idx += nchar
            if is_first_block and is_first_line:
                text_padding = max(
                    text_padding,
                    idea_height + over_margin + under_margin,
                )
            elif is_last_block:
                text_padding = idea_height + over_margin + under_margin
            is_first_line = False

        tl.endLayout()

        if is_first_block or is_last_block:
            self.text_padding = max(self.text_padding, text_padding / 2)
        self.y_offset_lst.append(y_offset)
        self.shrink_width = max(shrink_width, self.shrink_width)
        return 1

    def draw(self, painter: QPainter, context: QAbstractTextDocumentLayout.PaintContext) -> None:
        doc = self.document()
        self.deferred_cursor_position = context.cursorPosition
        painter.save()
        painter.setPen(context.palette.color(QPalette.ColorRole.Text))
        block = doc.firstBlock()
        cursor_block = None
        render_delegate = self.render_delegate
        while block.isValid():
            blpos = block.position()
            layout = block.layout()
            bllen = block.length()
            if _block_cursor_position(block, context.cursorPosition) >= 0:
                cursor_block = block
            if render_delegate is None:
                selections = []
                for sel in context.selections:
                    selStart = sel.cursor.selectionStart() - blpos
                    selEnd = sel.cursor.selectionEnd() - blpos
                    if selStart < bllen and selEnd > 0 and selEnd > selStart:
                        o = QTextLayout.FormatRange()
                        o.start = selStart
                        o.length = selEnd - selStart
                        o.format = sel.format
                        selections.append(o)
                    elif not sel.cursor.hasSelection() \
                        and sel.format.hasProperty(QTextFormat.FullWidthSelection) \
                        and block.contains(sel.cursor.position()):
                        o = QTextLayout.FormatRange()
                        l = layout.lineForTextPosition(sel.cursor.position() - blpos)
                        o.start = l.textStart()
                        o.length = l.textLength()
                        if o.start + o.length == bllen - 1:
                            ++o.length
                        o.format = sel.format
                        selections.append(o)
                clip = context.clip if context.clip.isValid() else QRectF()
                layout.draw(painter, QPointF(0, 0), selections, clip)
            else:
                if context.clip.isValid():
                    painter.save()
                    painter.setClipRect(context.clip, Qt.ClipOperation.IntersectClip)
                try:
                    render_delegate.draw_horizontal_block(
                        painter, block, context
                    )
                finally:
                    if context.clip.isValid():
                        painter.restore()
            for line_number in range(layout.lineCount()):
                line = layout.lineAt(line_number)
                if line.isValid() and line.textLength() > 0:
                    draw_emphasis_marks(
                        painter,
                        block,
                        line,
                        context,
                        vertical=False,
                    )
            block = block.next()
        
        if self.foreground_pixmap is not None:
            painter.drawPixmap(0, 0, self.foreground_pixmap)

        if cursor_block is not None and not self.defer_cursor_paint:
            block = cursor_block
            blpos = block.position()
            bllen = block.length()
            layout = block.layout()
            cpos = _block_cursor_position(block, context.cursorPosition)
            if cpos >= 0:
                layout.drawCursor(painter, QPointF(0, 0), cpos, 1)
        painter.restore()
