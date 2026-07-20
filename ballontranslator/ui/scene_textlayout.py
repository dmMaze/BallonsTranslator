import re
from bisect import bisect_left, bisect_right

from qtpy.QtCore import (
    Qt,
    QRectF,
    QPointF,
    Signal,
    QSizeF,
    QSize,
    QTextBoundaryFinder,
)
from qtpy.QtGui import QTextCharFormat, QTextDocument, QPixmap, QImage, QTransform, QPalette, QPainter, QTextFrame, QTextBlock, QAbstractTextDocumentLayout, QTextLayout, QFont, QFontMetricsF, QTextOption, QTextLine, QTextFormat

import cv2
import numpy as np
from typing import List
from functools import lru_cache, cached_property

from .misc import pixmap2ndarray, LruIgnoreArg
from ballontranslator.utils import shared as C
from ballontranslator.utils.fontformat import (
    FontFormat,
    LineSpacingType,
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    normalize_text_transform_value,
    pt2px,
)
from .text_glyph_renderer import (
    draw_slanted_glyph_mask,
    draw_slanted_line,
    slanted_line_ink_bounds,
)

def print_transform(tr: QTransform):
    print(f'[[{tr.m11(), tr.m12(), tr.m13()}]\n [{tr.m21(), tr.m22(), tr.m23()}]\n [{tr.m31(), tr.m32(), tr.m33()}]]')


PUNSET_HALF = {chr(i) for i in range(0x21, 0x7F)}

# https://www.w3.org/TR/2022/DNOTE-clreq-20220801/#tables_of_chinese_punctuation_marks
# https://www.w3.org/TR/2022/DNOTE-clreq-20220801/#glyphs_sizes_and_positions_in_character_faces_of_punctuation_marks
PUNSET_PAUSEORSTOP = {'。', '．', '，', '、', '·', '：', '；', '！', '？'}     # dont need to rotate, 
PUNSET_ALIGNCENTER = {'。', '．', '，', '、', '·'}
PUNSET_BRACKETL = {'「', '『', '“', '‘', '（', '《', '〈', '【', '〖', '〔', '［', '｛', '('}
PUNSET_BRACKETR = {'」', '』', '”', '’', '）', '》', '〉', '】', '〗', '〕', '］', '｝', ')'}
PUNSET_BRACKET = PUNSET_BRACKETL.union(PUNSET_BRACKETR)

PUNSET_NONBRACKET = {'⸺', '…', '⋯', '～', '-', '–', '—', '＿', '﹏', '●', '•', '~'}
PUNSET_VERNEEDROTATE = PUNSET_NONBRACKET.union(PUNSET_BRACKET).union(PUNSET_HALF)

PUNSET_ROTATE_ALIGNL = {'」', '』', '”', '’'}
PUNSET_ROTATE_ALIGNR = {'「', '『', '“', '‘'}

Dingbats_vertical_aligncenter = r'\u2700-\u275A\u2761-\u2767\u2776-\u27BF'
Miscellaneous_Symbols_Pattern = r'\u2600-\u26FF'  # align center in vertical mode

vertical_force_aligncentel_pattern = re.compile('[' + Dingbats_vertical_aligncenter + Miscellaneous_Symbols_Pattern + r'⁁⁂⁇⁈⁉⁊⁋⁎※⁑⁒⁕⁖⁘⁙⁛⁜‼‽]')


@lru_cache
def vertical_force_aligncentel(char: str) -> bool:
    return char in PUNSET_PAUSEORSTOP or vertical_force_aligncentel_pattern.match(char) is not None

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


class CharFontFormat:
    def __init__(self, fcmt: QTextCharFormat) -> None:
        font = fcmt.font()
        self.font = font
        self.stroke_width = fcmt.textOutline().widthF() / 2
        self.font_metrics = QFontMetricsF(font)

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
        self._glyph_slant_angle = normalize_text_transform_value(
            getattr(fontformat, 'glyph_slant_angle', 0.0),
            TEXT_TRANSFORM_GLYPH_SLANT_MIN,
            TEXT_TRANSFORM_GLYPH_SLANT_MAX,
        )
        self.glyph_raster_failure_handler = None
        self._layout_generation = 0
        self._glyph_bounds_cache = {}

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
        self.draw_foreground_only = False

        self.relayout_on_changed = True

        # Effect padding is view/layout state, not rich-text content. Keeping
        # it off QTextDocument prevents Glyph Slant preview bounds from adding
        # unrelated text-edit undo entries.
        self._document_margin = max(0.0, float(doc.documentMargin()))

        # relative bottom/right
        self.shrink_height = 0 
        self.shrink_width = 0

        self._doc_text: str = ''

        # The upstream vertical-stroke renderer clones a document and reuses
        # the already-computed draw offsets.  Keep that neutral-state path
        # intact; feature rendering only replaces it while Glyph Slant is
        # active.
        self._is_painting_stroke = False
        self._draw_offset = []
        self.text_padding = 0

    @property
    def glyph_slant_angle(self) -> float:
        return self._glyph_slant_angle

    @property
    def layout_generation(self) -> int:
        return self._layout_generation

    def setGlyphSlantAngle(self, angle: float) -> bool:
        """Set transient glyph ink slant without changing document geometry."""
        angle = normalize_text_transform_value(
            angle,
            TEXT_TRANSFORM_GLYPH_SLANT_MIN,
            TEXT_TRANSFORM_GLYPH_SLANT_MAX,
        )
        if angle == self._glyph_slant_angle:
            return False
        self._glyph_slant_angle = angle
        self._glyph_bounds_cache.clear()
        if C.USE_PYSIDE6:
            self.update.emit()
        else:
            self.update.emit(QRectF(0, 0, self.max_width, self.max_height))
        return True

    def _begin_layout_generation(self):
        self._layout_generation += 1
        self._glyph_bounds_cache.clear()

    def _report_glyph_raster_failure(self, error, effect_pass=False):
        handler = self.glyph_raster_failure_handler
        if handler is not None:
            handler(error, effect_pass)

    def _iter_glyph_line_placements(self):
        """Yield ``(line, offset, orientation)`` in item-local paint order."""
        raise NotImplementedError

    def glyphInkBounds(self) -> QRectF:
        """Return live shaped ink bounds for the effective glyph slant."""
        if self.document().isEmpty():
            return QRectF()
        key = (
            self.document().revision(),
            self._layout_generation,
            type(self),
            self._glyph_slant_angle,
        )
        cached = self._glyph_bounds_cache.get(key)
        if cached is not None:
            return QRectF(cached)
        bounds = QRectF()
        for line, offset, orientation in self._iter_glyph_line_placements():
            line_bounds = slanted_line_ink_bounds(
                line,
                offset,
                orientation,
                self._glyph_slant_angle,
            )
            if line_bounds.isEmpty():
                continue
            bounds = line_bounds if bounds.isNull() else bounds.united(line_bounds)
        self._glyph_bounds_cache = {key: QRectF(bounds)}
        return bounds

    def setMaxSize(self, max_width: int, max_height: int, relayout=True):
        self.max_height = max_height
        self.max_width = max_width
        doc_margin = self._document_margin * 2
        self.available_width = max(max_width -  doc_margin, 0)
        self.available_height = max(max_height - doc_margin, 0)
        if relayout:
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

    def updateDocumentMargin(self, margin):
        self._document_margin = max(0.0, float(margin))
        doubled_margin = self._document_margin * 2
        self.max_height = doubled_margin + self.available_height
        self.max_width = doubled_margin + self.available_width

    def documentMargin(self) -> float:
        return self._document_margin

    def documentSize(self) -> QSizeF:
        return QSizeF(self.max_width, self.max_height)

    def documentChanged(self, position: int, charsRemoved: int, charsAdded: int) -> None:
        if not self.relayout_on_changed:
            return
        self.reLayoutEverything()
        
    def reLayoutEverything(self):
        self._doc_text = self.document().toPlainText()
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
                cfmt = CharFontFormat(fcmt)
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
        self.force_single_char = True
        self.has_selection = False
        self.draw_shifted = 0

        self.need_ideal_width = True
        self.line_draw = line_draw_qt6 if C.FLAG_QT6 else line_draw_qt5

        self.per_char_records = []

    @property
    def align_right(self):
        return False

    def reLayout(self):
        self._begin_layout_generation()
        self.min_height = 0
        self.layout_left = 0
        self.line_spaces_lst = []
        self.per_char_records = []
        self.draw_shifted = 0
        self.shrink_height = 0
        self.shrink_width = 0
        self.text_padding = 0
        doc = self.document()
        doc_margin = self._document_margin
        block = doc.firstBlock()
        while block.isValid():
            self.layoutBlock(block)
            block = block.next()

        enlarged = False
        x_shift = 0
        if self.layout_left < doc_margin:
            x_shift  = doc_margin - self.layout_left
            self.max_width += x_shift
            self.available_width = self.max_width - 2*doc_margin
            enlarged = True
        if self.min_height - doc_margin > self.available_height:
            self.available_height = self.min_height - doc_margin
            self.max_height = self.available_height + doc_margin * 2
            enlarged = True
        if enlarged:
            self.size_enlarged.emit()
            if x_shift != 0:
                block = doc.firstBlock()
                while block.isValid():
                    tl = block.layout()
                    for ii in range(tl.lineCount()):
                        line = tl.lineAt(ii)
                        line_pos = line.position()
                        line_pos.setX(x_shift + line_pos.x())
                        line.setPosition(line_pos)
                    block = block.next()
                for ii, xoffset in enumerate(self.x_offset_lst):
                    self.x_offset_lst[ii] = xoffset + x_shift
        self.updateDrawOffsets()
        self.documentSizeChanged.emit(QSizeF(self.max_width, self.max_height))

    def updateDrawOffsets(self):
        if self._is_painting_stroke and len(self._draw_offset) > 0:
            return
        self._draw_offset.clear()
        doc = self.document()
        block = doc.firstBlock()

        while block.isValid():
            blk_no = block.blockNumber()
            _draw_offsets = []
            self._draw_offset.append(_draw_offsets)

            layout = block.layout()
            blk_text = block.text()
            glyph_slanted = self._glyph_slant_angle != 0.0
            blk_text_len = (
                _utf16_length(blk_text) if glyph_slanted else len(blk_text)
            )
            
            line_spaces_lst = self.line_spaces_lst[blk_no]
            char_records = self.per_char_records[blk_no]

            for ii in range(layout.lineCount()):
                xy_offsets = [0, 0]
                _draw_offsets.append(xy_offsets)

                line = layout.lineAt(ii)
                if line.textLength() == 0:
                    continue
                num_rspaces, num_lspaces, _, line_pos  = line_spaces_lst[ii]
                char_idx = min(line_pos + num_lspaces, blk_text_len - 1)
                if char_idx < 0:
                    continue

                char = (
                    _utf16_char_at(blk_text, char_idx)
                    if glyph_slanted
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

                if char in PUNSET_VERNEEDROTATE:
                    char = (
                        _utf16_char_at(blk_text, char_idx)
                        if glyph_slanted
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
                    # other characters will simply be aligned center for this line
                    act_rect = cfmt.punc_actual_rect(line, char, cache=True, space_shift=space_shift)
                    if vertical_force_aligncentel(char):
                        yoff = -act_rect[1]
                    else:
                        yoff = min(cfmt.br.top() - cfmt.tbr.top(), -cfmt.tbr.top() - line.ascent())
                    xoff = -act_rect[0] + (line_width - act_rect[2]) / 2
                    # if char in PUNSET_ALIGNTOP:
                    #     yoff = yoff + (cfmt.tbr.height() - act_rect[3]) / 2
                    
                    if num_lspaces > 0:
                        xoff -= space_shift
                        yoff += space_shift

                    if char in PUNSET_ALIGNCENTER:
                        tbr, br = cfmt.punc_rect(char)
                        yoff += (tbr.height() + cfmt.font_metrics.descent() - act_rect[3]) / 2

                xy_offsets[0], xy_offsets[1] = xoff, yoff
            block = block.next()

    def _vertical_line_placement(self, block: QTextBlock, line_number: int):
        layout = block.layout()
        line = layout.lineAt(line_number)
        if not line.isValid() or line.textLength() <= 0:
            return None
        block_number = block.blockNumber()
        block_text = block.text()
        block_text_length = _utf16_length(block_text)
        _, leading_spaces, _, line_position = self.line_spaces_lst[block_number][
            line_number
        ]
        char_offset = min(line_position + leading_spaces, block_text_length - 1)
        if char_offset < 0:
            return line, QPointF(), QTransform()
        char = _utf16_char_at(block_text, char_offset)
        x_offset, y_offset = self._draw_offset[block_number][line_number]
        orientation = QTransform()
        if char in PUNSET_VERNEEDROTATE:
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

    def _iter_glyph_line_placements(self):
        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            for line_number in range(layout.lineCount()):
                placement = self._vertical_line_placement(block, line_number)
                if placement is not None:
                    yield placement
            block = block.next()

    def draw_glyph_selection_mask(
        self,
        painter: QPainter,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> None:
        """Draw only glyphs named by temporary document-layout selections.

        This is the Qt 5/6 common path used to build vertical effect masks.
        It reads glyph runs from the attached QTextLayout objects, preserving
        their established positions and rotations without changing document
        formats. QText decorations are intentionally excluded from the mask.
        """
        painter.save()
        try:
            painter.setPen(Qt.GlobalColor.white)
            if context.clip.isValid():
                painter.setClipRect(context.clip)
            for selection in context.selections:
                selection_start = selection.cursor.selectionStart()
                selection_end = selection.cursor.selectionEnd()
                if selection_end <= selection_start:
                    continue
                self._draw_glyph_range(
                    painter, selection_start, selection_end
                )
        finally:
            painter.restore()

    def _draw_glyph_range(
        self, painter: QPainter, selection_start: int, selection_end: int
    ) -> None:
        block = self.document().firstBlock()
        while block.isValid():
            block_start = block.position()
            block_end = block_start + block.length() - 1
            if selection_start >= block_end:
                block = block.next()
                continue
            if selection_end <= block_start:
                break

            local_start = max(0, selection_start - block_start)
            local_end = min(block_end - block_start, selection_end - block_start)
            block_number = block.blockNumber()
            block_text = block.text()
            block_text_length = _utf16_length(block_text)
            layout = block.layout()
            line_spaces = self.line_spaces_lst[block_number]
            for line_number in range(layout.lineCount()):
                line = layout.lineAt(line_number)
                line_start = line.textStart()
                line_end = line_start + line.textLength()
                run_start = max(line_start, local_start)
                run_end = min(line_end, local_end)
                if run_end <= run_start:
                    continue

                glyph_runs = line.glyphRuns(run_start, run_end - run_start)
                if not glyph_runs:
                    continue
                _, leading_spaces, _, line_position = line_spaces[line_number]
                char_offset = min(
                    line_position + leading_spaces, block_text_length - 1
                )
                if char_offset < 0:
                    continue
                char = _utf16_char_at(block_text, char_offset)
                x_offset, y_offset = self._draw_offset[block_number][line_number]

                if self._glyph_slant_angle != 0.0:
                    placement = self._vertical_line_placement(block, line_number)
                    if placement is not None:
                        placed_line, offset, orientation = placement
                        draw_slanted_glyph_mask(
                            painter,
                            placed_line,
                            run_start,
                            run_end - run_start,
                            offset,
                            orientation,
                            self._glyph_slant_angle,
                            self._report_glyph_raster_failure,
                        )
                    continue

                painter.save()
                try:
                    if char in PUNSET_VERNEEDROTATE:
                        line_x, line_y = line.x(), line.y()
                        painter.setTransform(
                            QTransform(
                                0,
                                1,
                                0,
                                -1,
                                0,
                                0,
                                line_y + line_x,
                                line_y - line_x,
                                1,
                            ),
                            True,
                        )
                    for glyph_run in glyph_runs:
                        # QGlyphRun carries decoration flags independently of
                        # its glyph indexes. They belong to the normal text
                        # pass and must not be expanded into a thick outline.
                        glyph_run.setUnderline(False)
                        glyph_run.setOverline(False)
                        glyph_run.setStrikeOut(False)
                        painter.drawGlyphRun(
                            QPointF(x_offset, y_offset), glyph_run
                        )
                finally:
                    painter.restore()
            block = block.next()


    def draw(self, painter: QPainter, context: QAbstractTextDocumentLayout.PaintContext) -> None:
        doc = self.document()
        painter.save()
        block = doc.firstBlock()
        cursor_block = None
        context_sel = context.selections
        has_selection = False
        selection = None
        if len(context_sel) > 0:
            has_selection = True
            selection = context_sel[0]

        fm = None
        while block.isValid():
            blk_no = block.blockNumber()
            blpos, bllen = block.position(), block.length()
            layout = block.layout()
            blk_text = block.text()
            glyph_slanted = self._glyph_slant_angle != 0.0
            blk_text_len = (
                _utf16_length(blk_text) if glyph_slanted else len(blk_text)
            )
            char_records = self.per_char_records[blk_no]
            
            line_spaces_lst = self.line_spaces_lst[blk_no]

            if _block_cursor_position(block, context.cursorPosition) >= 0:
                cursor_block = block

            for ii in range(layout.lineCount()):
                line = layout.lineAt(ii)
                if line.textLength() == 0:
                    continue
                num_rspaces, num_lspaces, _, line_pos  = line_spaces_lst[ii]
                char_idx = min(line_pos + num_lspaces, blk_text_len - 1)
                if char_idx < 0:
                    if self._glyph_slant_angle != 0.0:
                        placement = self._vertical_line_placement(block, ii)
                        if placement is not None:
                            placed_line, offset, orientation = placement
                            draw_slanted_line(
                                painter,
                                block,
                                placed_line,
                                offset,
                                orientation,
                                self._glyph_slant_angle,
                                context,
                                self._report_glyph_raster_failure,
                            )
                    else:
                        line.draw(painter, QPointF(0, 0))
                    continue

                xoff, yoff = self._draw_offset[blk_no][ii]

                char = (
                    _utf16_char_at(blk_text, char_idx)
                    if glyph_slanted
                    else blk_text[char_idx]
                )
                cfmt = self.get_char_fontfmt(blk_no, char_idx)
                fm = cfmt.font_metrics
                if self._glyph_slant_angle != 0.0:
                    placement = self._vertical_line_placement(block, ii)
                    if placement is not None:
                        placed_line, offset, orientation = placement
                        draw_slanted_line(
                            painter,
                            block,
                            placed_line,
                            offset,
                            orientation,
                            self._glyph_slant_angle,
                            context,
                            self._report_glyph_raster_failure,
                        )
                    continue
                selected = False
                if has_selection:
                    sel_start = selection.cursor.selectionStart() - blpos 
                    sel_end = selection.cursor.selectionEnd() - blpos
                    if char_idx < sel_end and char_idx >= sel_start:
                        selected = True

                line_width = -1
                if char_idx in char_records:
                    line_width = char_records[char_idx]['line_width']
                if line_width < 0:
                    line_width = cfmt.tbr.width()
                
                if char in PUNSET_VERNEEDROTATE:
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

            block = block.next()

        if self.foreground_pixmap is not None:
            painter.drawPixmap(0, 0, self.foreground_pixmap)

        if cursor_block is not None:
            block = cursor_block
            blk_text = block.text()
            blpos = block.position()
            bllen = block.length()
            blk_no = block.blockNumber()
            layout = block.layout()
            cpos = _block_cursor_position(block, context.cursorPosition)

            if cpos >= 0:
                line = layout.lineForTextPosition(cpos)
                if line.isValid():

                    pos = line.position()
                    x, y = pos.x(), pos.y()
                    if line.textLength() == 0 or fm is None:
                        fm = QFontMetricsF(block.charFormat().font())
                    else:
                        num_rspaces, num_lspaces, char_yoffset_lst, line_pos = self.line_spaces_lst[blk_no][line.lineNumber()]
                        yidx = cpos - line_pos
                        if yidx >= 0 and yidx < len(char_yoffset_lst):
                            y = char_yoffset_lst[yidx]

                    painter.setCompositionMode(QPainter.CompositionMode.RasterOp_NotDestination)
                    painter.fillRect(QRectF(x, y, fm.height(), 2), painter.pen().brush())
                    if self.has_selection == has_selection:
                        if C.USE_PYSIDE6:
                            self.update.emit()
                        else:
                            self.update.emit(QRectF(x, y, fm.height(), 2))
                    else:
                        if C.USE_PYSIDE6:
                            self.update.emit()
                        else:
                            self.update.emit(QRectF(0, 0, self.max_width, self.max_height))
            self.has_selection = has_selection  # update this flag when drawing the cursor
        painter.restore()

    def hitTest(self, point: QPointF, accuracy: Qt.HitTestAccuracy) -> int:
        blk = self.document().firstBlock()
        x, y = point.x(), point.y()
        off = 0
        while blk.isValid():
            blk_no = blk.blockNumber()
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
                            ntr = line.naturalTextRect()
                            off = line.textStart()
                            if self._glyph_slant_angle != 0.0:
                                # The feature path consumes Qt UTF-16 glyph
                                # runs, so never return a caret inside a
                                # transformed grapheme.
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
        ls = self.letter_spacing

        block.clearLayout()
        doc_margin = self._document_margin
        line_y_offset = doc_margin
        blk_char_yoffset = []
        blk_line_spaces = []

        block_no = block.blockNumber()
        is_final_block = block == doc.lastBlock()
        blk_text = block.text()
        glyph_slanted = self._glyph_slant_angle != 0.0
        blk_text_len = (
            _utf16_length(blk_text) if glyph_slanted else len(blk_text)
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
            if glyph_slanted:
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

            tbr_h = space_w = let_sp_offset = 0
            char_idx += num_lspaces
            single_char_h = None

            if char_idx < blk_text_len:
                cfmt = self.get_char_fontfmt(block_no, char_idx)
                space_shift = 0
                if num_lspaces > 0:
                    space_shift = num_lspaces * cfmt.space_width
                line_char_ids.append(char_idx)
                space_w = cfmt.space_width
                let_sp_offset = cfmt.tbr.height() * (ls - 1)

                tbr_h = cfmt.tbr.height() + let_sp_offset
                char = (
                    _utf16_char_at(blk_text, char_idx)
                    if glyph_slanted
                    else blk_text[char_idx]
                )
                is_first_lbracket = char_idx - num_lspaces == 0 and char in PUNSET_BRACKETL
                if is_first_lbracket:
                    _lbracket_shift = -cfmt.punc_actual_rect(line, char, cache=True, space_shift=space_shift)[0]

                if char in PUNSET_VERNEEDROTATE:
                    tbr, br = cfmt.punc_rect(char)
                    single_char_h = tbr.width()
                    tbr_h = tbr.width() * (
                        _grapheme_count(text) if glyph_slanted else text_len
                    )
                    if char.isalpha():
                        cw2 = cfmt.punc_rect(char+char)[1].width()
                        tbr_h = br.width() - (br.width() * 2 - cw2)
                    elif char in {'…', '⋯', '—', '～'}:
                        tbr_h = line.naturalTextWidth() - num_lspaces * space_w
                        next_char_idx = char_idx + (
                            _utf16_length(char) if glyph_slanted else 1
                        )
                        if (
                            next_char_idx < blk_text_len
                            and (
                                _utf16_char_at(blk_text, next_char_idx)
                                if glyph_slanted
                                else blk_text[next_char_idx]
                            ) == char
                        ):
                            tbr_h -= let_sp_offset
                    else:
                        tbr_h = line.naturalTextWidth() - num_lspaces * space_w
                    tbr_h += let_sp_offset
                elif vertical_force_aligncentel(char):
                    if char not in PUNSET_ALIGNCENTER:
                        tbr_h = cfmt.punc_actual_rect(line, char, cache=True, space_shift=space_shift)[3]
                    else:
                        tbr, br = cfmt.punc_rect(char)
                        tbr_h = tbr.height() + cfmt.font_metrics.descent()
                    tbr_h += let_sp_offset
            elif char_idx - num_lspaces < blk_text_len:
                cfmt = self.get_char_fontfmt(block_no, char_idx - num_lspaces)
                tbr_h = cfmt.tbr.height() + cfmt.font_metrics.descent()
                space_w = cfmt.space_width
            
            if num_lspaces == 0 and tbr_h != 0:
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
                    width_list.append(cfmt.tbr.width())
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
                end_line, end_ypos, end_w = line, line_y_offset, width_list[-1]
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
                    char_records[cidx] = {'line_width': idea_line_width}
                line_char_ids = []

                x_offset = x_offset - self.calculate_line_spacing(idea_line_width, line_spacing)
                
                for line, ypos in zip(line_not_set[:-1], ypos_list[:-1]):
                    line.setPosition(QPointF(x_offset, ypos))
                if out_of_vspace:
                    if end_char:
                        if not len(line_not_set) == 1:
                            x_offset = x_offset - self.calculate_line_spacing(end_w, line_spacing)
                        end_line.setPosition(QPointF(x_offset, end_ypos))
                        char_records[end_char_id] = {'line_width': end_w}
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
                if glyph_slanted
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

    def frameBoundingRect(self, frame: QTextFrame):
        return QRectF(0, 0, max(self.document().pageSize().width(), self.max_width), 2147483647)

    def setLetterSpacing(self, letter_spacing: float):
        if self.letter_spacing != letter_spacing:
            self.letter_spacing = letter_spacing
            self.reLayout()



class HorizontalTextDocumentLayout(SceneTextLayout):

    def __init__(self, doc: QTextDocument, fontformat: FontFormat):
        super().__init__(doc, fontformat)
        self.need_ideal_height = True

    def reLayout(self):
        self._begin_layout_generation()
        doc = self.document()
        doc_margin = self._document_margin
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
            self.size_enlarged.emit()

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

    def frameBoundingRect(self, frame: QTextFrame):
        return QRectF(0, 0, max(self.document().pageSize().width(), self.max_width), 2147483647)

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
        doc_margin = self._document_margin

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
                
            line.setPosition(QPointF(doc_margin, y_offset + dy))
            tw = line.naturalTextWidth()
            shrink_width = max(tw, shrink_width)
            self.shrink_height = max(idea_height + y_offset - doc_margin, self.shrink_height)    #????
            y_offset += self.calculate_line_spacing(idea_height, self.line_spacing)
            line_idx += 1
            char_idx += nchar
            if is_first_block and is_first_line:
                text_padding = max(text_padding, idea_height)
            elif is_last_block:
                text_padding = idea_height
            is_first_line = False

        tl.endLayout()

        if is_first_block or is_last_block:
            self.text_padding = max(self.text_padding, text_padding / 2)
        self.y_offset_lst.append(y_offset)
        self.shrink_width = max(shrink_width, self.shrink_width)
        return 1

    def _iter_glyph_line_placements(self):
        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            for line_number in range(layout.lineCount()):
                line = layout.lineAt(line_number)
                if line.isValid() and line.textLength() > 0:
                    yield line, QPointF(), QTransform()
            block = block.next()

    def draw(self, painter: QPainter, context: QAbstractTextDocumentLayout.PaintContext) -> None:
        doc = self.document()
        painter.save()
        painter.setPen(context.palette.color(QPalette.ColorRole.Text))
        block = doc.firstBlock()
        cursor_block = None
        while block.isValid():
            blpos = block.position()
            layout = block.layout()
            bllen = block.length()
            if _block_cursor_position(block, context.cursorPosition) >= 0:
                cursor_block = block
            if self._glyph_slant_angle == 0.0:
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
                    for line_number in range(layout.lineCount()):
                        line = layout.lineAt(line_number)
                        if line.isValid() and line.textLength() > 0:
                            draw_slanted_line(
                                painter,
                                block,
                                line,
                                QPointF(),
                                QTransform(),
                                self._glyph_slant_angle,
                                context,
                                self._report_glyph_raster_failure,
                            )
                finally:
                    if context.clip.isValid():
                        painter.restore()
            block = block.next()
        
        if self.foreground_pixmap is not None:
            painter.drawPixmap(0, 0, self.foreground_pixmap)

        if cursor_block is not None:
            block = cursor_block
            blpos = block.position()
            bllen = block.length()
            layout = block.layout()
            cpos = _block_cursor_position(block, context.cursorPosition)
            if cpos >= 0:
                layout.drawCursor(painter, QPointF(0, 0), cpos, 1)
        painter.restore()
