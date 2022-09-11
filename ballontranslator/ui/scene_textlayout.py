from qtpy.QtCore import Qt, QRectF, QPointF, Signal, QSizeF, QSize
from qtpy.QtGui import QTextCharFormat, QTextDocument, QPixmap, QImage, QTransform, QPalette, QPainter, QTextFrame, QTextBlock, QAbstractTextDocumentLayout, QTextLayout, QFont, QFontMetrics, QTextOption, QTextLine, QTextFormat

import cv2
import numpy as np
from typing import List
from functools import lru_cache, cached_property

from .misc import pixmap2ndarray, pt2px, LruIgnoreArg
from . import constants as C

def print_transform(tr: QTransform):
    print(f'[[{tr.m11(), tr.m12(), tr.m13()}]\n [{tr.m21(), tr.m22(), tr.m23()}]\n [{tr.m31(), tr.m32(), tr.m33()}]]')


PUNSET_HALF = {chr(i) for i in range(0x21, 0x7F)}

# https://www.w3.org/TR/2022/DNOTE-clreq-20220801/#tables_of_chinese_punctuation_marks
# https://www.w3.org/TR/2022/DNOTE-clreq-20220801/#glyphs_sizes_and_positions_in_character_faces_of_punctuation_marks
PUNSET_PAUSEORSTOP = {'。', '．', '，', '、', '：', '；', '！', '‼', '？', '⁇', '⁉', '⁈'}     # dont need to rotate, 
PUNSET_BRACKETL = {'「', '『', '“', '‘', '（', '《', '〈', '【', '〖', '〔', '［', '｛'}
PUNSET_BRACKETR = {'」', '』', '”', '’', '）', '》', '〉', '】', '〗', '〕', '］', '｝'}
PUNSET_BRACKET = PUNSET_BRACKETL.union(PUNSET_BRACKETR)

PUNSET_NONBRACKET = {'⸺', '…', '⋯', '～', '-', '–', '—', '＿', '﹏', '●', '•', '~'}
PUNSET_VERNEEDROTATE = PUNSET_NONBRACKET.union(PUNSET_BRACKET).union(PUNSET_HALF)


@lru_cache(maxsize=256)
def _font_metrics(ffamily: str, size: float, weight: int, italic: bool) -> QFontMetrics:
    font = QFont(ffamily, size, weight, italic)
    return QFontMetrics(font)

@lru_cache(maxsize=2048)
def get_punc_rect(char: str, ffamily: str, size: float, weight: int, italic: bool) -> List[QRectF]:
    fm = _font_metrics(ffamily, size, weight, italic)
    br = [fm.tightBoundingRect(char), fm.boundingRect(char)]
    return br

@lru_cache(maxsize=2048)
def get_char_width(char: str, ffamily: str, size: float, weight: int, italic: bool) -> int:
    fm = _font_metrics(ffamily, size, weight, italic)
    return fm.width(char)

def punc_actual_rect(line: QTextLine, family: str, size: float, weight: int, italic: bool, stroke_width: float) -> List[int]:
    pixmap = QImage(line.naturalTextWidth(), line.height(), QImage.Format.Format_ARGB32)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    line.draw(p, QPointF(-line.x(), -line.y()))
    p.end()
    mask = pixmap2ndarray(pixmap, keep_alpha=True)[..., -1]
    
    ar = cv2.boundingRect(cv2.findNonZero(mask))
    # if stroke_width != 0:
    ar = np.array(ar, dtype=np.float64)
    ar[[0, 1]] += stroke_width
    ar[[2, 3]] -= stroke_width * 2
    ar = ar.tolist()
    return ar

@lru_cache(maxsize=2048)
def punc_actual_rect_cached(line: LruIgnoreArg, char: str, family: str, size: float, weight: int, italic: bool, stroke_width: float) -> List[int]:
    # QtextLine line is invisibale to lru
    return punc_actual_rect(line.line, family, size, weight, italic, stroke_width)

def line_draw_qt6(painter: QPainter, line: QTextLine, x: float, y: float, selected: bool, selection: QAbstractTextDocumentLayout.Selection = None):
    # some how qt6 line.draw doesn't allow pass FormatRange
    if selected:    
        qimg = QImage(line.naturalTextWidth(), line.height(), QImage.Format.Format_ARGB32)
        qimg.fill(Qt.GlobalColor.transparent)
        p = QPainter(qimg)
        line.draw(p, QPointF(-line.x(), -line.y()))
        p.end()
        qimg.invertPixels(QImage.InvertMode.InvertRgba)
        painter.drawImage(QPointF(line.x() + x, line.y() + y), qimg.createAlphaMask())
    else:
        line.draw(painter, QPointF(x, y))

def line_draw_qt5(painter: QPainter, line: QTextLine, x: float, y: float, selected: bool, selection: QAbstractTextDocumentLayout.Selection = None):
    o = None
    if selected:
        o = QTextLayout.FormatRange()
        o.start = line.textStart()
        o.length = line.textLength()
        o.format = selection.format
    line.draw(painter, QPointF(x, y), o)

class CharFontFormat:
    def __init__(self, fcmt: QTextCharFormat) -> None:
        font = fcmt.font()
        self.font = font
        self.stroke_width = fcmt.textOutline().widthF() / 2
        self.font_metrics = QFontMetrics(font)

    @cached_property
    def br(self) -> QRectF:
        return get_punc_rect('大', self.family, self.size, self.weight, self.font.italic())[1]

    @cached_property
    def tbr(self) -> QRectF:
        return get_punc_rect('大', self.family, self.size, self.weight, self.font.italic())[0]

    @cached_property
    def space_width(self) -> int:
        return get_char_width(' ', self.family, self.size, self.weight, self.font.italic())

    def punc_rect(self, punc: str) -> List[QRectF]:
        return get_punc_rect(punc, self.family, self.size, self.weight, self.font.italic())

    @property
    def family(self) -> str:
        return self.font.family()

    @property
    def weight(self) -> float:
        return self.font.weight()

    @property
    def size(self) -> float:
        return self.font.pointSizeF()

    def punc_actual_rect(self, line: QTextLine, char: str, cache=False) -> List[int]:
        if cache:
            line = LruIgnoreArg(line=line)
            ar = punc_actual_rect_cached(line, char, self.family, self.size, self.weight, self.font.italic(), self.stroke_width)
        else:
            ar =  punc_actual_rect(line, self.family, self.size, self.weight, self.font.italic(), self.stroke_width)
        return ar


class SceneTextLayout(QAbstractTextDocumentLayout):
    size_enlarged = Signal()
    def __init__(self, doc: QTextDocument) -> None:
        super().__init__(doc)
        self.max_height = 0
        self.max_width = 0
        self.available_width = 0
        self.available_height = 0
        self.line_spacing = 1.
        self.letter_spacing = 1.

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

    def setMaxSize(self, max_width: int, max_height: int, relayout=True):
        self.max_height = max_height
        self.max_width = max_width
        doc_margin = self.document().documentMargin() * 2
        self.available_width = max(max_width -  doc_margin, 0)
        self.available_height = max(max_height - doc_margin, 0)
        if relayout:
            self.reLayout()

    def setLineSpacing(self, line_spacing: float):
        if self.line_spacing != line_spacing:
            self.line_spacing = line_spacing
            self.reLayout()

    def blockBoundingRect(self, block: QTextBlock) -> QRectF:
        if not block.isValid():
            return QRectF()
        br = block.layout().boundingRect()
        rect = QRectF(0, 0, br.width(), br.height())
        return rect

    def updateDocumentMargin(self, margin):
        doc_margin = self.document().documentMargin()
        dm = margin - doc_margin
        doc_margin *= 2
        self.document().setDocumentMargin(margin)
        margin *= 2
        self.max_height = margin + self.available_height
        self.max_width = margin + self.available_width

    def documentSize(self) -> QSizeF:
        return QSizeF(self.max_width, self.max_height)

    def documentChanged(self, position: int, charsRemoved: int, charsAdded: int) -> None:
        if not self.relayout_on_changed:
            return
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
                    h_ = cfmt.punc_rect('fg')[0].height()
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

class VerticalTextDocumentLayout(SceneTextLayout):

    def __init__(self, doc: QTextDocument):
        super().__init__(doc)

        self.line_spaces_lst = []
        self.min_height = 0
        self.layout_left = 0
        self.force_single_char = True
        self.has_selection = False
        self.punc_align_center = True
        self.draw_shifted = 0

        self.need_ideal_width = True
        self.line_draw = line_draw_qt6 if C.FLAG_QT6 else line_draw_qt5

    @property
    def align_right(self):
        return False

    def reLayout(self):
        self.min_height = 0
        self.layout_left = 0
        self.line_spaces_lst = []
        self.draw_shifted = 0
        doc = self.document()
        doc_margin = doc.documentMargin()
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
        self.documentSizeChanged.emit(QSizeF(self.max_width, self.max_height))

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

        while block.isValid():
            blpos, bllen = block.position(), block.length()
            layout = block.layout()
            blk_text = block.text()
            blk_text_len = len(blk_text)
            blk_no = block.blockNumber()
            line_spaces_lst = self.line_spaces_lst[blk_no]

            if context.cursorPosition >= blpos and context.cursorPosition < blpos + bllen:
                cursor_block = block

            for ii in range(layout.lineCount()):
                line = layout.lineAt(ii)
                if line.textLength() == 0:
                    continue
                num_rspaces, num_lspaces, _, line_pos  = line_spaces_lst[ii]
                char_idx = min(line_pos + num_lspaces, blk_text_len - 1)
                if char_idx < 0:
                    line.draw(painter, QPointF(0, 0))
                    continue

                char = blk_text[char_idx]
                cfmt = self.get_char_fontfmt(blk_no, char_idx)
                fm = cfmt.font_metrics
                selected = False
                if has_selection:
                    sel_start = selection.cursor.selectionStart() - blpos 
                    sel_end = selection.cursor.selectionEnd() - blpos
                    if char_idx < sel_end and char_idx >= sel_start:
                        selected = True
                
                natral_shifted = max(line.naturalTextWidth() - cfmt.br.width(), 0)
                if char in PUNSET_VERNEEDROTATE:
                    char = blk_text[char_idx]
                    line_x, line_y = line.x(), line.y()

                    if char in PUNSET_NONBRACKET:
                        non_bracket_br = cfmt.punc_actual_rect(line, char)
            
                    y_x = line_y - line_x
                    y_p_x = line_y + line_x
                    transform = QTransform(0, 1, 0, -1, 0, 0, y_p_x, y_x, 1)
                    inv_transform = QTransform(0, -1, 0, 1, 0, 0, -y_x, y_p_x, 1)
                    painter.setTransform(transform, True)
                    pun_tbr, pun_br = cfmt.punc_rect(char)
                    hight_comp = pun_tbr.width() - pun_br.width()

                    if char.isalpha():
                        yoff = -cfmt.tbr.top() - fm.ascent() - cfmt.tbr.width()
                        hight_comp = 0
                    elif char in PUNSET_NONBRACKET:
                        yoff =  -non_bracket_br[1] - non_bracket_br[3]
                        if self.punc_align_center:
                            yoff = yoff - cfmt.tbr.width() / 2 + non_bracket_br[3] / 3
                        else:
                            yoff = yoff - cfmt.tbr.width() + non_bracket_br[3]
                    else:
                        yoff = -pun_tbr.top() - fm.ascent() - pun_tbr.height() / 2 - cfmt.br.width() / 2

                    self.line_draw(painter, line, hight_comp,  yoff, selected, selection)
                    painter.setTransform(inv_transform, True)

                elif char in PUNSET_PAUSEORSTOP:
                    pun_tbr, pun_br = cfmt.punc_rect(char)
                    act_rect = cfmt.punc_actual_rect(line, char, cache=True)
                    yoff = -act_rect[1]
                    xoff = -pun_tbr.left()
                    yoff += self.draw_shifted
                    if num_lspaces > 0:
                        if natral_shifted == 0:
                            natral_shifted = num_lspaces * cfmt.space_width
                        else:
                            natral_shifted -= self.draw_shifted
                        xoff -= natral_shifted
                        yoff += natral_shifted
                    if self.punc_align_center or char in {'⁇', '⁉', '⁈', '‼'}:
                        xoff += (cfmt.br.width() - pun_tbr.width()) / 2
                    else:
                        xoff += cfmt.br.width() - pun_tbr.width()
                        xoff -= self.draw_shifted
                    self.line_draw(painter, line, xoff, yoff, selected, selection)

                else:
                    yoff = -cfmt.tbr.top() - fm.ascent() + natral_shifted
                    self.line_draw(painter, line, -natral_shifted, yoff, selected, selection)

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
            if context.cursorPosition < -1:
                cpos = layout.preeditAreaPosition() - (cpos + 2)
            else:
                cpos = context.cursorPosition - blpos

            line = layout.lineForTextPosition(cpos)
            if line.isValid():
                
                pos = line.position()                
                x, y = pos.x(), pos.y()
                if line.textLength() == 0:
                    fm = QFontMetrics(block.charFormat().font())
                else:
                    num_rspaces, num_lspaces, char_yoffset_lst, line_pos = self.line_spaces_lst[blk_no][line.lineNumber()]
                    y = char_yoffset_lst[cpos - line_pos]

                painter.setCompositionMode(QPainter.CompositionMode.RasterOp_NotDestination)
                painter.fillRect(QRectF(x, y, fm.height(), 2), painter.pen().brush())
                if self.has_selection == has_selection:
                    self.update.emit(QRectF(x, y, fm.height(), 2))
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
            rect = blk.layout().boundingRect()
            rect_left = rect.left()
            rect_right = rect.right()
            rect_right, rect_left = self.x_offset_lst[blk_no], self.x_offset_lst[blk_no+1]
            if rect_left <= x and rect_right >= x:
                layout = blk.layout()
                for ii in range(layout.lineCount()):
                    line_top, line_bottom = blk_char_yoffset[ii]
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

    def get_char_fontfmt(self, block_number: int, char_idx: int) -> CharFontFormat:
        frag_idx = self._map_charidx2frag[block_number][char_idx]
        return self.block_charfmt_lst[block_number][frag_idx]

    def layoutBlock(self, block: QTextBlock):
        doc = self.document()
        ls = self.letter_spacing

        block.clearLayout()
        doc_margin = doc.documentMargin()
        line_y_offset = doc_margin
        blk_char_yoffset = []
        blk_line_spaces = []

        block_no = block.blockNumber()
        is_final_block = block == doc.lastBlock()
        blk_text = block.text()
        blk_text_len = len(blk_text)
        if blk_text_len != 0:
            block_width = self.block_ideal_width[block_no]
        else:
            block_width = CharFontFormat(block.charFormat()).tbr.width()

        layout_first_block = block == doc.firstBlock()
        if layout_first_block:
            
            x_offset = self.max_width - doc_margin - block_width
            self.x_offset_lst = [self.max_width - doc_margin]
            self.y_offset_lst = []
        else:
            x_offset = self.x_offset_lst[-1] - block_width * self.line_spacing

        char_idx = 0
        tl = block.layout()
        tl.beginLayout()
        option = doc.defaultTextOption()
        option.setWrapMode(QTextOption.WrapAnywhere)
        tl.setTextOption(option)
        
        while True:
            line = tl.createLine()
            if not line.isValid():
                break

            if self.force_single_char:
                line.setLineWidth(block_width)
                line.setNumColumns(1)
            else:
                line.setLineWidth(block_width)
            
            available_height = self.available_height + doc_margin
            text_len = line.textLength()
            num_rspaces, num_lspaces = 0, 0
            text = blk_text[char_idx: char_idx + text_len].replace('\n', '')
            num_rspaces = text_len - len(text.rstrip())
            num_lspaces = text_len - len(text.lstrip())

            tbr_h = space_w = let_sp_offset = 0
            char_idx += num_lspaces
            if char_idx < blk_text_len:
                cfmt = self.get_char_fontfmt(block_no, char_idx)
                space_w = cfmt.space_width
                let_sp_offset = cfmt.tbr.height() * (ls - 1)
                tbr_h = cfmt.tbr.height() + let_sp_offset
                char = blk_text[char_idx]
                if char in PUNSET_VERNEEDROTATE:
                    tbr, br = cfmt.punc_rect(char)
                    tbr_h = tbr.width()
                    if char.isalpha():
                        cw2 = cfmt.punc_rect(char+char)[1].width()
                        tbr_h = br.width() - (br.width() * 2 - cw2)
                    if char in {'…', '⋯'}:
                        tbr_h = line.naturalTextWidth() - num_lspaces * space_w
                        next_char_idx = char_idx + 1
                        if next_char_idx < blk_text_len and blk_text[next_char_idx] == char:
                            tbr_h -= let_sp_offset
                    tbr_h += let_sp_offset
                elif char in PUNSET_PAUSEORSTOP:
                    if char in {'⁇', '⁉', '⁈', '‼'}:
                        tbr_h = cfmt.punc_actual_rect(line, char, cache=True)[3]
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
            for _ in range(num_lspaces):
                char_yoffset_lst.append(min(available_height - tbr_h, char_yoffset_lst[-1] + space_w))
            blk_line_spaces.append([num_rspaces, num_lspaces, char_yoffset_lst, char_idx - num_lspaces])
            
            char_bottom = char_yoffset_lst[-1] + tbr_h
            if char_bottom - max(let_sp_offset, 0) > available_height:
                # switch to next line
                if char_idx == 0 and layout_first_block:
                    self.min_height = doc_margin + tbr_h
                else:
                    x_offset = x_offset - block_width * self.line_spacing
                line_y_offset = doc_margin
                
                char_yoffset_lst[-1] = line_y_offset
                char_yoffset_lst.append(line_y_offset + tbr_h)
                for _ in range(num_rspaces):
                    char_yoffset_lst.append(min(char_yoffset_lst[-1] + space_w, available_height))
                line_bottom = char_yoffset_lst[-1]
            else:
                char_yoffset_lst.append(char_bottom)
                for _ in range(num_rspaces):
                    char_yoffset_lst.append(min(char_yoffset_lst[-1] + space_w, available_height))
                line_bottom = char_yoffset_lst[-1]

            line.setPosition(QPointF(x_offset, line_y_offset))
            blk_char_yoffset.append([line_y_offset, line_bottom])
            line_y_offset = max(line_bottom, doc_margin)
            char_idx += text_len - num_lspaces
        tl.endLayout()
            
        self.layout_left = x_offset - self.draw_shifted
        self.x_offset_lst.append(x_offset)
        self.y_offset_lst.append(blk_char_yoffset)
        self.line_spaces_lst.append(blk_line_spaces)

    def frameBoundingRect(self, frame: QTextFrame):
        return QRectF(0, 0, max(self.document().pageSize().width(), self.max_width), 2147483647)

    def setLetterSpacing(self, letter_spacing: float):
        if self.letter_spacing != letter_spacing:
            self.letter_spacing = letter_spacing
            self.reLayout()


class HorizontalTextDocumentLayout(SceneTextLayout):

    def __init__(self, doc: QTextDocument):
        super().__init__(doc)
        self.need_ideal_height = True

    def reLayout(self):
        doc = self.document()
        doc_margin = self.document().documentMargin()
        self.y_bottom = 0
        block = doc.firstBlock()
        while block.isValid():
            self.layoutBlock(block)
            block = block.next()
        
        if len(self.y_offset_lst) > 0:
            new_height = self.y_bottom - doc_margin
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
        option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        tl.setTextOption(option)
        font = block.charFormat().font()
        tbr = get_punc_rect('字fg', font.family(), font.pointSizeF(), font.weight(), font.italic())[0]
        fm = QFontMetrics(font)
        doc_margin = self.document().documentMargin()

        idea_height = self.block_ideal_height[block.blockNumber()]
        if block == doc.firstBlock():
            self.x_offset_lst = []
            self.y_offset_lst = []
            y_offset = -tbr.top() - fm.ascent() + doc_margin
        else:
            y_offset = self.y_offset_lst[-1]

        line_idx = 0
        tl.beginLayout()
        while True:
            line = tl.createLine()
            if not line.isValid():
                break
            line.setLeadingIncluded(False)
            line.setLineWidth(self.available_width)
            line.setPosition(QPointF(doc_margin, y_offset))
            self.y_bottom = idea_height + y_offset + line.descent()    #????
            y_offset += idea_height * self.line_spacing
            line_idx += 1
        tl.endLayout()
        self.y_offset_lst.append(y_offset)
        return 1

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
            if context.cursorPosition >= blpos and context.cursorPosition < blpos + bllen:
                cursor_block = block
            layout = block.layout()
            blpos = block.position()
            bllen = block.length()
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
            block = block.next()
        
        if self.foreground_pixmap is not None:
            painter.drawPixmap(0, 0, self.foreground_pixmap)

        if cursor_block is not None:
            block = cursor_block
            blpos = block.position()
            bllen = block.length()
            layout = block.layout()
            if context.cursorPosition < -1:
                cpos = layout.preeditAreaPosition() - (cpos + 2)
            else:
                cpos = context.cursorPosition - blpos
            layout.drawCursor(painter, QPointF(0, 0), cpos, 1)
        painter.restore()