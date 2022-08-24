from qtpy.QtCore import Qt, QRectF, QPointF, QPoint, Signal, QSizeF
from qtpy.QtGui import QTransform, QPalette, QPainter, QTextFrame, QTextBlock, QAbstractTextDocumentLayout, QTextLayout, QFontMetrics, QTextOption, QTextLine, QTextFormat

from typing import List, Set

def print_transform(tr: QTransform):
    print(f'[[{tr.m11(), tr.m12(), tr.m13()}]\n [{tr.m21(), tr.m22(), tr.m23()}]\n [{tr.m31(), tr.m32(), tr.m33()}]]')


PUNSET_HALF = {chr(i) for i in range(0x21, 0x7F)}

# https://www.w3.org/TR/2022/DNOTE-clreq-20220801/#tables_of_chinese_punctuation_marks
# https://www.w3.org/TR/2022/DNOTE-clreq-20220801/#glyphs_sizes_and_positions_in_character_faces_of_punctuation_marks
PUNSET_PAUSEORSTOP = {'。', '．', '，', '、', '：', '；', '！', '‼', '？', '⁇'}     # dont need to rotate, 
PUNSET_BRACKETL = {'「', '『', '“', '‘', '（', '《', '〈', '【', '〖', '〔', '［', '｛'}
PUNSET_BRACKETR = {'」', '』', '”', '’', '）', '》', '〉', '】', '〗', '〕', '］', '｝'}
PUNSET_BRACKET = PUNSET_BRACKETL.union(PUNSET_BRACKETR)

PUNSET_VERNEEDROTATE = {'⸺', '…', '…', '⋯', '⋯', '～', '-', '–', '—', '＿', '﹏', '●', '•'}.union(PUNSET_BRACKET).union(PUNSET_HALF)

class VerticalTextDocumentLayout(QAbstractTextDocumentLayout):
    size_enlarged = Signal()
    def __init__(self, textDocument):
        super().__init__(textDocument)
        self.max_height = 0
        self.max_width = 0
        self.available_width = 0
        self.available_height = 0
        self.x_offset_lst = []
        self.y_offset_lst = []
        self.punt_lst: List[Set] = []
        self.pun_rect_lst: List[List[QRectF]] = []
        self.line_spaces_lst = []
        self.line_spacing = 1.
        self.min_height = 0
        self.layout_left = 0
        self.force_single_char = True
        self.has_selection = False

    def setMaxSize(self, max_width: int, max_height: int):
        self.max_height = max_height
        self.max_width = max_width
        doc_margin = self.document().documentMargin() * 2
        self.available_width = max(max_width -  doc_margin, 0)
        self.available_height = max(max_height - doc_margin, 0)
        self.reLayout()
    
    def setLineSpacing(self, line_spacing: float):
        if self.line_spacing != line_spacing:
            self.line_spacing = line_spacing
            self.reLayout()

    @property
    def align_right(self):
        return False

    def reLayout(self):
        self.min_height = 0
        self.layout_left = 0
        self.punt_lst = []
        self.line_spaces_lst = []
        doc = self.document()
        doc_margin = doc.documentMargin()
        block = doc.firstBlock()
        while block.isValid():
            self.punt_lst.append(set())
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
        if len(context_sel) > 0:
            has_selection = True
            selection = context_sel[0]

        while block.isValid():
            blpos = block.position()
            layout = block.layout()
            bllen = block.length()
            blk_text = block.text()
            if context.cursorPosition >= blpos and context.cursorPosition < blpos + bllen:
                cursor_block = block
            layout = block.layout()
            blpos = block.position()
            bllen = block.length()

            blk_idx = block.blockNumber()
            rotate_pun_set = self.punt_lst[blk_idx]
            line_spaces_lst = self.line_spaces_lst[blk_idx]
            fm = QFontMetrics(block.charFormat().font())
            tbr = fm.tightBoundingRect('字fg')
            width_comp = fm.boundingRect('演').width() - tbr.height()
            for ii in range(layout.lineCount()):
                line = layout.lineAt(ii)
                num_rspaces, num_lspaces, _, line_pos  = line_spaces_lst[ii]
                char_idx = line_pos + num_lspaces
                o = None
                if has_selection:
                    sel_start = selection.cursor.selectionStart() - blpos 
                    sel_end = selection.cursor.selectionEnd() - blpos
                    if char_idx < sel_end and char_idx >= sel_start:
                        o = QTextLayout.FormatRange()
                        o.start = line.textStart()
                        o.length = line.textLength()
                        o.format = selection.format
                        
                if char_idx in rotate_pun_set:
                    line_x, line_y = line.x(), line.y()
                    y_x = line_y - line_x
                    y_p_x = line_y + line_x
                    transform = QTransform(0, 1, 0, -1, 0, 0, y_p_x, y_x, 1)
                    inv_transform = QTransform(0, -1, 0, 1, 0, 0, -y_x, y_p_x, 1)
                    painter.setTransform(transform, True)
                    char = blk_text[char_idx]
                    if not char in PUNSET_BRACKET:
                        line.draw(painter, QPointF(0, -tbr.top() - fm.ascent() - tbr.height() - width_comp), o)
                    else:
                        hight_comp = fm.boundingRectChar(char).width() - fm.boundingRect(char).width() + 2
                        line.draw(painter, QPointF(hight_comp, -tbr.top() - fm.ascent() - tbr.height() - width_comp), o)
                    painter.setTransform(inv_transform, True)
                else:
                    yoff = 0
                    if num_lspaces > 0:
                        yoff = num_lspaces * fm.width(' ')
                        line.draw(painter, QPointF(-yoff, -tbr.top() - fm.ascent() + yoff), o)
                    else:
                        line.draw(painter, QPointF(0, -tbr.top() - fm.ascent()), o)
            block = block.next()
        
        if cursor_block is not None:
            block = cursor_block
            blk_text = block.text()
            blpos = block.position()
            bllen = block.length()
            layout = block.layout()
            if context.cursorPosition < -1:
                cpos = layout.preeditAreaPosition() - (cpos + 2)
            else:
                cpos = context.cursorPosition - blpos

            line = layout.lineForTextPosition(cpos)
            if line.isValid():
                
                pos = line.position()                
                x, y = pos.x(), pos.y()
                len_text = len(blk_text)
                font = block.charFormat().font()
                fm = QFontMetrics(font)
                if len_text > 0:
                    if cpos >= len_text:
                        last_char = blk_text[-1]
                        tbr = fm.tightBoundingRect(last_char)
                        if cpos - 1 in rotate_pun_set:
                            y += tbr.width()
                        else:
                            y += tbr.height()

                num_rspaces, num_lspaces, char_yoffset_lst, line_pos = self.line_spaces_lst[block.blockNumber()][line.lineNumber()]
                if num_rspaces > 0 or num_lspaces > 0:
                    y = char_yoffset_lst[cpos - line_pos]
                
                painter.setCompositionMode(QPainter.RasterOp_NotDestination)
                painter.fillRect(QRectF(x, y, fm.height(), 2), painter.pen().brush())
                if self.has_selection == has_selection:
                    self.update.emit(QRectF(x, y, fm.height(), 2))
                else:
                    self.update.emit(QRectF(0, 0, self.max_width, self.max_height))
            self.has_selection = has_selection  # update this flag when drawing the cursor
        painter.restore()
        
    def documentSize(self) -> QSizeF:
        return QSizeF(self.max_width, self.max_height)

    def hitTest(self, point: QPointF, accuracy: Qt.HitTestAccuracy) -> int:
        blk = self.document().firstBlock()
        x, y = point.x(), point.y()
        off = 0
        while blk.isValid():
            blk_idx = blk.blockNumber()
            blk_char_yoffset = self.y_offset_lst[blk_idx]
            rect = blk.layout().boundingRect()
            rect_left = rect.left()
            rect_right = rect.right()
            rect_right, rect_left = self.x_offset_lst[blk_idx], self.x_offset_lst[blk_idx+1]
            if rect_left <= x and rect_right >= x:
                layout = blk.layout()
                for ii in range(layout.lineCount()):
                    line_top, line_bottom = blk_char_yoffset[ii]
                    line = layout.lineAt(ii)
                    line_xy = line.position()
                    if not line_xy.x() <= x:
                        continue 
                    ntr = line.naturalTextRect()
                    if line_top > y:
                        off = min(off, line.textStart())
                    elif line_bottom < y:
                        off = max(off, line.textStart() + line.textLength())
                    else:
                        num_rspaces, num_lspaces, char_yoffset_lst, line_pos = self.line_spaces_lst[blk_idx][ii]
                        if num_rspaces > 0 or num_lspaces > 0:
                            for ii, (ytop, ybottom) in enumerate(zip(char_yoffset_lst[:-1], char_yoffset_lst[1:])):
                                dis_top, dis_bottom = y - ytop, ybottom - y
                                if dis_top >= 0 and dis_bottom >= 0:
                                    off = ii + line_pos if dis_top < dis_bottom else ii + 1 + line_pos
                                    break
                            break
                        else:
                            if ntr.left() <= x and ntr.right() >= x:
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

    def documentChanged(self, position: int, charsRemoved: int, charsAdded: int) -> None:
        self.reLayout()
        
    def blockBoundingRect(self, block: QTextBlock) -> QRectF:
        if not block.isValid():
            return QRectF()
        br = block.layout().boundingRect()
        rect = QRectF(0, 0, br.width(), br.height())
        return rect

    def layoutBlock(self, block: QTextBlock):
        doc = self.document()
        block.clearLayout()

        line_y_offset = doc.documentMargin()
        blk_char_yoffset = []
        blk_line_spaces = []

        fm = QFontMetrics(block.charFormat().font())
        fm_w = fm.width('字')
        space_w = fm.width(' ')
        line_width = fm_w
        tbr = fm.tightBoundingRect('字fg')
        TBRH = tbr.height()

        layout_first_block = block == doc.firstBlock()
        if layout_first_block:
            x_offset = self.max_width - doc.documentMargin() - fm_w
            self.x_offset_lst = [self.max_width - doc.documentMargin()]
            self.y_offset_lst = []
        else:
            x_offset = self.x_offset_lst[-1] - fm_w*self.line_spacing

        char_idx = 0
        tl = block.layout()
        tl.beginLayout()
        option = doc.defaultTextOption()
        option.setWrapMode(QTextOption.WrapAnywhere)
        # option.setFlags(QTextOption.Flag.IncludeTrailingSpaces)
        tl.setTextOption(option)
        
        blk_text = block.text()
        
        blk_text_len = len(blk_text)
        if blk_text_len == 0:
            TBRH = 0
        while True:
            tbr_h = TBRH

            line = tl.createLine()
            if not line.isValid():
                break

            if self.force_single_char:
                line.setNumColumns(1)
            else:
                line.setLineWidth(fm_w)

            ntw = line.naturalTextWidth()
            text_len = line.textLength()
            num_rspaces, num_lspaces = 0, 0
            char_yoffset_lst = [line_y_offset]
            if text_len > 1:
                text = blk_text[char_idx: char_idx + text_len].replace('\n', '')
                num_rspaces = text_len - len(text.rstrip())
                num_lspaces = text_len - len(text.lstrip())
                for _ in range(num_lspaces):
                    char_yoffset_lst.append(min(self.available_height, char_yoffset_lst[-1] + space_w))
            blk_line_spaces.append([num_rspaces, num_lspaces, char_yoffset_lst, char_idx])
            
            rotated = False
            char_idx += num_lspaces
            if char_idx < blk_text_len:
                char = blk_text[char_idx]
                if char in PUNSET_VERNEEDROTATE:
                    self.punt_lst[-1].add(char_idx)
                    if char in PUNSET_BRACKET:
                        tbr = fm.tightBoundingRect(char)
                    else:
                        tbr = fm.boundingRect(char)
                    tbr_h = tbr.width()
                    rotated = True

            center_align_offset = 0
            if ntw == 0:
                ntw = fm_w
            elif ntw < fm_w / 1.5:
                if not rotated:
                    center_align_offset = fm_w / 4
            
            line_width = max(ntw, fm_w, line_width)
            line.setLineWidth(ntw)
            char_bottom = char_yoffset_lst[-1] + tbr_h
            if char_bottom > self.available_height:
                # switch to next line
                if char_idx == 0 and layout_first_block:
                    self.min_height = line_bottom
                else:
                    x_offset = x_offset - fm_w * self.line_spacing
                line_y_offset = doc.documentMargin()
                line_bottom = line_y_offset + tbr_h
                char_yoffset_lst.append(line_bottom)
                for _ in range(num_rspaces):
                    char_yoffset_lst.append(min(char_yoffset_lst[-1] + space_w, self.available_height))
            else:
                char_yoffset_lst.append(char_bottom)
                for _ in range(num_rspaces):
                    char_yoffset_lst.append(min(char_yoffset_lst[-1] + space_w, self.available_height))
                line_bottom = char_yoffset_lst[-1]

            line.setPosition(QPointF(x_offset+center_align_offset, line_y_offset))
            blk_char_yoffset.append([line_y_offset, line_bottom])
            line_y_offset = line_bottom
            char_idx += text_len - num_lspaces
        tl.endLayout()
            
        self.layout_left = x_offset
        self.x_offset_lst.append(x_offset)
        self.y_offset_lst.append(blk_char_yoffset)
        self.line_spaces_lst.append(blk_line_spaces)
        return 1

    def frameBoundingRect(self, frame: QTextFrame):
        return QRectF(0, 0, max(self.document().pageSize().width(), self.max_width), 2147483647)

    def updateDocumentMargin(self, margin):
        self.max_height = margin *2 + self.available_height
        self.max_width = margin * 2 + self.available_width


class HorizontalTextDocumentLayout(QAbstractTextDocumentLayout):
    size_enlarged = Signal()
    def __init__(self, textDocument):
        super().__init__(textDocument)
        self.max_height = 0
        self.max_width = 0
        self.available_width = 0
        self.available_height = 0
        self.x_offset_lst = []
        self.y_offset_lst = []
        self.line_spacing = 1.

    def setMaxSize(self, max_width: int, max_height: int):
        self.max_height = max_height
        self.max_width = max_width
        doc_margin = self.document().documentMargin() * 2
        self.available_width = max(max_width -  doc_margin, 0)
        self.available_height = max(max_height - doc_margin, 0)
        self.reLayout()

    def setLineSpacing(self, line_spacing: float):
        if self.line_spacing != line_spacing:
            self.line_spacing = line_spacing
            self.reLayout()

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

    def documentChanged(self, position: int, charsRemoved: int, charsAdded: int) -> None:
        self.reLayout()
        
    def blockBoundingRect(self, block: QTextBlock) -> QRectF:
        if not block.isValid():
            return QRectF()
        br = block.layout().boundingRect()
        rect = QRectF(0, 0, br.width(), br.height())
        return rect

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
        fm = QFontMetrics(block.charFormat().font())
        doc_margin = self.document().documentMargin()
        tbr = fm.tightBoundingRect('字fg')
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
            self.y_bottom = tbr.height() + y_offset + line.descent()    #????
            y_offset += tbr.height() * self.line_spacing
            line_idx += 1
        tl.endLayout()
        self.y_offset_lst.append(y_offset)     # vertical text need center alignment ???
        return 1

    def documentSize(self) -> QSizeF:
        return QSizeF(self.max_width, self.max_height)

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

    def updateDocumentMargin(self, margin):
        self.max_height = margin *2 + self.available_height
        self.max_width = margin * 2 + self.available_width