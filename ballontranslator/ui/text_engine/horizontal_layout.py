from typing import List, Optional, Tuple

from qtpy.QtCore import QPointF, QRectF, QSizeF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QPainter,
    QPalette,
    QTextBlock,
    QTextDocument,
    QTextLayout,
    QTextLine,
    QTextOption,
    QTextFormat,
)

from ballontranslator.utils import shared as C
from ballontranslator.utils.fontformat import FontFormat
from .layout import (
    SceneTextLayout,
    _block_cursor_position,
    get_punc_rect,
    paint_context_without_selection_ranges,
    selection_segments_excluding,
)
from .rendering.emphasis import draw_emphasis_marks, emphasis_margins
from .rendering.indexing import _utf16_length, _utf16_slice

class HorizontalTextDocumentLayout(SceneTextLayout):

    def __init__(self, doc: QTextDocument, fontformat: FontFormat):
        super().__init__(doc, fontformat)
        self.need_ideal_height = True
        # QTextLayout hangs overflowing trailing spaces on their preceding
        # line. Keep their document positions, but give only those spaces
        # derived continuation-row geometry for editing and box growth.
        self._space_rows = []
        self._relocated_spaces = []
        self._space_caret_rects = {}
        self._cursor_update_rect = QRectF()

    @staticmethod
    def _cursor_x(line: QTextLine, position: int) -> float:
        value = line.cursorToX(position)
        if isinstance(value, (tuple, list)):
            value = value[0]
        return float(value)

    @classmethod
    def _cursor_span(
        cls,
        line: QTextLine,
        start: int,
        end: int,
    ) -> Tuple[float, float]:
        positions = [
            cls._cursor_x(line, position)
            for position in range(start, end + 1)
        ]
        if not positions:
            return 0.0, 0.0
        return min(positions), max(positions)

    @staticmethod
    def _row_left(
        content_left: float,
        available_width: float,
        row_width: float,
        alignment: Qt.AlignmentFlag,
    ) -> float:
        if alignment & Qt.AlignmentFlag.AlignHCenter:
            return content_left + (available_width - row_width) / 2
        if alignment & Qt.AlignmentFlag.AlignRight:
            return content_left + available_width - row_width
        return content_left

    @staticmethod
    def _line_space_alignment_shift(
        space_width: float,
        alignment: Qt.AlignmentFlag,
        right_to_left: bool,
    ) -> float:
        if alignment & Qt.AlignmentFlag.AlignHCenter:
            return space_width / 2 if right_to_left else -space_width / 2
        if alignment & Qt.AlignmentFlag.AlignRight:
            return 0.0 if right_to_left else -space_width
        return space_width if right_to_left else 0.0

    def _trailing_space_layout(
        self,
        block: QTextBlock,
        line: QTextLine,
    ) -> Tuple[int, List[Tuple[int, float]]]:
        """Return the suffix start and whole spaces Qt cannot fit.

        Qt preserves the positions of hanging spaces but clamps their cursor
        advances to the line edge. We independently fit complete U+0020 cells;
        visible text and every other Unicode separator retain Qt semantics.

        >>> callable(HorizontalTextDocumentLayout._trailing_space_layout)
        True
        """
        text = block.text()
        line_start = line.textStart()
        if line_start + line.textLength() > _utf16_length(text):
            # IME preedit text has layout positions but no document format or
            # stable persistence position until it is committed.
            return line_start + line.textLength(), []
        line_text = _utf16_slice(text, line_start, line.textLength())
        suffix_length = len(line_text) - len(line_text.rstrip(' '))
        if suffix_length == 0 or self.available_width <= 0:
            return line_start + line.textLength(), []

        suffix_start = (
            line_start
            + _utf16_length(line_text[:-suffix_length])
        )
        cursor_left, cursor_right = self._cursor_span(
            line, line_start, suffix_start
        )
        base_advance = cursor_right - cursor_left

        spaces = []
        fitted_width = 0.0
        fit_count = 0
        overflow_started = False
        block_number = block.blockNumber()
        for position in range(suffix_start, suffix_start + suffix_length):
            char_format = self.get_char_fontfmt(block_number, position)
            if char_format is None:
                break
            width = max(
                0.0,
                float(char_format.font_metrics.horizontalAdvance(' ')),
            )
            spaces.append((position, width))
            if (
                not overflow_started
                and base_advance + fitted_width + width
                <= line.width() + 1e-6
            ):
                fitted_width += width
                fit_count += 1
            else:
                overflow_started = True

        if not spaces:
            return suffix_start, []

        # Qt sometimes includes fitting terminal spaces in naturalTextWidth(),
        # but excludes the identical suffix when another word follows. Shift
        # only the width Qt did not already use for paragraph alignment.
        native_space_width = min(
            fitted_width,
            max(0.0, line.naturalTextWidth() - base_advance),
        )
        extra_aligned_width = max(0.0, fitted_width - native_space_width)
        if extra_aligned_width > 1e-6:
            position = line.position()
            position.setX(
                position.x()
                + self._line_space_alignment_shift(
                    extra_aligned_width,
                    block.layout().textOption().alignment(),
                    block.textDirection() == Qt.LayoutDirection.RightToLeft,
                )
            )
            line.setPosition(position)
        return suffix_start + fit_count, spaces[fit_count:]

    def _append_space_rows(
        self,
        block: QTextBlock,
        spaces: List[Tuple[int, float]],
        y_offset: float,
        fallback_height: float,
    ) -> Tuple[
        float,
        Optional[Tuple[QRectF, List[Tuple[int, QRectF]], bool]],
    ]:
        if not spaces:
            return y_offset, None

        rows = []
        row = []
        row_width = 0.0
        for space in spaces:
            width = space[1]
            if row and row_width + width > self.available_width + 1e-6:
                rows.append((row, row_width))
                row = []
                row_width = 0.0
            row.append(space)
            row_width += width
            if width > self.available_width + 1e-6:
                rows.append((row, row_width))
                row = []
                row_width = 0.0
        if row:
            rows.append((row, row_width))

        block_number = block.blockNumber()
        alignment = block.layout().textOption().alignment()
        right_to_left = (
            block.textDirection() == Qt.LayoutDirection.RightToLeft
        )
        content_left = self._effect_padding
        block_rows = self._space_rows[block_number]
        for row_spaces, row_width in rows:
            # The native line height already uses the largest fragment in its
            # complete text range, including the relocated suffix.
            row_height = fallback_height
            row_left = self._row_left(
                content_left,
                self.available_width,
                row_width,
                alignment,
            )
            row_rect = QRectF(
                content_left,
                y_offset,
                self.available_width,
                row_height,
            )
            cells = []
            cursor_x = row_left + row_width if right_to_left else row_left
            self._space_caret_rects[
                block.position() + row_spaces[0][0]
            ] = QRectF(cursor_x - 0.5, y_offset, 1.0, row_height)
            for position, width in row_spaces:
                if right_to_left:
                    cursor_x -= width
                    cell_left = cursor_x
                    caret_x = cursor_x
                else:
                    cell_left = cursor_x
                    cursor_x += width
                    caret_x = cursor_x
                cell_rect = QRectF(
                    cell_left,
                    y_offset,
                    width,
                    row_height,
                )
                cells.append((position, cell_rect))
                self._space_caret_rects[
                    block.position() + position + 1
                ] = QRectF(caret_x - 0.5, y_offset, 1.0, row_height)
            block_rows.append((row_rect, cells, right_to_left))
            self.shrink_width = max(self.shrink_width, row_width)
            self.shrink_height = max(
                self.shrink_height,
                y_offset + row_height - self._effect_padding,
            )
            y_offset += self.calculate_line_spacing(
                row_height, self.line_spacing
            )
        return y_offset, block_rows[-1]

    def _line_can_share_space_row(
        self,
        line: QTextLine,
        row: Tuple[QRectF, List[Tuple[int, QRectF]], bool],
    ) -> bool:
        _row_rect, cells, right_to_left = row
        if not cells:
            return False
        remaining_width = self.available_width - sum(
            cell.width() for _position, cell in cells
        )
        if remaining_width <= 1e-6:
            return False
        line.setLineWidth(remaining_width)
        start = line.textStart()
        end = start + line.textLength()
        cursor_left, cursor_right = self._cursor_span(line, start, end)
        logical_start = self._cursor_x(line, start)
        expected_start = cursor_right if right_to_left else cursor_left
        return (
            line.naturalTextWidth() <= remaining_width + 1e-6
            and cursor_right - cursor_left <= remaining_width + 1e-6
            and abs(logical_start - expected_start) <= 1e-6
        )

    def _merge_line_into_space_row(
        self,
        block: QTextBlock,
        line: QTextLine,
        row: Tuple[QRectF, List[Tuple[int, QRectF]], bool],
        line_content_end: int,
    ) -> None:
        _row_rect, cells, right_to_left = row
        space_width = sum(cell.width() for _position, cell in cells)
        cursor_left, cursor_right = self._cursor_span(
            line, line.textStart(), line_content_end
        )
        line_width = cursor_right - cursor_left
        combined_width = space_width + line_width
        self.shrink_width = max(self.shrink_width, combined_width)
        target_left = self._row_left(
            self._effect_padding,
            self.available_width,
            combined_width,
            block.layout().textOption().alignment(),
        )

        cells_left = min(cell.left() for _position, cell in cells)
        target_cells_left = (
            target_left + line_width if right_to_left else target_left
        )
        cells_shift = target_cells_left - cells_left
        if abs(cells_shift) > 1e-9:
            for position, cell in cells:
                cell.translate(cells_shift, 0.0)
                caret = self._space_caret_rects.get(
                    block.position() + position + 1
                )
                if caret is not None:
                    caret.translate(cells_shift, 0.0)
        _row_rect.setLeft(min(cell.left() for _position, cell in cells))
        _row_rect.setRight(max(cell.right() for _position, cell in cells))

        target_line_start = (
            target_left + line_width
            if right_to_left
            else target_left + space_width
        )
        line_shift = target_line_start - self._cursor_x(
            line, line.textStart()
        )
        if abs(line_shift) > 1e-9:
            position = line.position()
            position.setX(position.x() + line_shift)
            line.setPosition(position)

    def _translate_space_rows_y(self, y_shift: float) -> None:
        if abs(y_shift) <= 1e-9:
            return
        for block_rows in self._space_rows:
            for row_rect, cells, _right_to_left in block_rows:
                row_rect.translate(0.0, y_shift)
                for _position, cell_rect in cells:
                    cell_rect.translate(0.0, y_shift)
        for cursor_rect in self._space_caret_rects.values():
            cursor_rect.translate(0.0, y_shift)

    def source_cursor_rect(
        self, cursor_position: int
    ) -> Optional[QRectF]:
        rect = self._space_caret_rects.get(cursor_position)
        return None if rect is None else QRectF(rect)

    def _paint_space_selection(
        self,
        painter: QPainter,
        block: QTextBlock,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> None:
        rows = self._space_rows[block.blockNumber()]
        if not rows:
            return
        for selection in context.selections:
            if not selection.cursor.hasSelection():
                continue
            selection_start = selection.cursor.selectionStart()
            selection_end = selection.cursor.selectionEnd()
            brush = selection.format.background()
            if brush.style() == Qt.BrushStyle.NoBrush:
                continue
            for _row_rect, cells, _right_to_left in rows:
                for position, cell_rect in cells:
                    absolute = block.position() + position
                    if selection_start <= absolute < selection_end:
                        painter.fillRect(cell_rect, brush)

    def reLayout(self) -> None:
        self._begin_layout_generation()
        doc = self.document()
        doc_margin = self._effect_padding
        self.text_padding = 0
        self.shrink_height = 0
        self.shrink_width = 0
        self._space_rows = []
        self._relocated_spaces = []
        self._space_caret_rects = {}
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
            self._translate_space_rows_y(y_offset)

        self.documentSizeChanged.emit(QSizeF(self.max_width, self.max_height))

    def _space_hit_test(self, point: QPointF) -> Optional[int]:
        for block_number, rows in enumerate(self._space_rows):
            block = self.document().findBlockByNumber(block_number)
            for row_rect, cells, right_to_left in rows:
                if not row_rect.contains(point) or not cells:
                    continue
                for position, cell_rect in cells:
                    if cell_rect.contains(point):
                        after = (
                            point.x() < cell_rect.center().x()
                            if right_to_left
                            else point.x() >= cell_rect.center().x()
                        )
                        return block.position() + position + int(after)
                first_position = cells[0][0]
                last_position = cells[-1][0] + 1
                cells_left = min(cell.left() for _position, cell in cells)
                cells_right = max(cell.right() for _position, cell in cells)
                if right_to_left:
                    return (
                        block.position() + first_position
                        if point.x() > cells_right
                        else block.position() + last_position
                    )
                return (
                    block.position() + first_position
                    if point.x() < cells_left
                    else block.position() + last_position
                )
        return None

    def hitTest(self, point: QPointF, accuracy: Qt.HitTestAccuracy) -> int:
        point = self.map_input_point(point)
        space_hit = self._space_hit_test(point)
        if space_hit is not None:
            return space_hit
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
                        relocated = self._relocated_spaces[
                            blk.blockNumber()
                        ].get(ii)
                        if relocated is not None:
                            off = min(off, relocated[0])
                        break
                    elif ntr.left() > x:
                        off = min(off, line.textStart())
                    else:
                        off = max(off, line.textStart() + line.textLength())
                break
            blk = blk.next()
        return blk.position() + off

    def layoutBlock(self, block: QTextBlock) -> int:
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
        self._space_rows.append([])
        self._relocated_spaces.append({})
        is_last_block = blk_no == self.document().blockCount() - 1
        is_first_block = blk_no == 0
        text_padding = 0
        is_first_line = False
        pending_space_row = None

        while True:
            line = tl.createLine()
            if not line.isValid():
                break
            # line.setLeadingIncluded(False)
            shared_space_row = pending_space_row
            pending_space_row = None
            if (
                shared_space_row is None
                or not self._line_can_share_space_row(
                    line, shared_space_row
                )
            ):
                shared_space_row = None
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
            line_y_offset = (
                shared_space_row[0].top()
                if shared_space_row is not None
                else y_offset
            )
            line_y_offset += over_margin
            line.setPosition(QPointF(doc_margin, line_y_offset + dy))
            relocated_start, relocated_spaces = self._trailing_space_layout(
                block, line
            )
            if shared_space_row is not None:
                self._merge_line_into_space_row(
                    block,
                    line,
                    shared_space_row,
                    relocated_start,
                )
            if relocated_spaces:
                relocated_end = relocated_spaces[-1][0] + 1
                self._relocated_spaces[blk_no][line_idx] = (
                    relocated_start,
                    relocated_end,
                )
            if relocated_spaces:
                cursor_left, cursor_right = self._cursor_span(
                    line, line.textStart(), relocated_start
                )
                tw = cursor_right - cursor_left
            else:
                tw = line.naturalTextWidth()
            shrink_width = max(tw, shrink_width)
            self.shrink_height = max(
                idea_height + line_y_offset - doc_margin + under_margin,
                self.shrink_height,
            )
            line_next_y = line_y_offset + (
                self.calculate_line_spacing(idea_height, self.line_spacing)
                + under_margin
            )
            y_offset = (
                max(y_offset, line_next_y)
                if shared_space_row is not None
                else line_next_y
            )
            y_offset, pending_space_row = self._append_space_rows(
                block,
                relocated_spaces,
                y_offset,
                idea_height,
            )
            if (
                relocated_spaces
                and char_idx + nchar < _utf16_length(block.text())
            ):
                # At a soft-wrap boundary Qt owns the shared cursor position
                # before the following visible glyph. Intermediate spaces and
                # terminal spaces retain layout-owned caret rectangles.
                self._space_caret_rects.pop(
                    block.position() + relocated_spaces[-1][0] + 1,
                    None,
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
            self._paint_space_selection(painter, block, context)
            if render_delegate is None:
                selections = []
                for sel in context.selections:
                    selStart = sel.cursor.selectionStart() - blpos
                    selEnd = sel.cursor.selectionEnd() - blpos
                    if selStart < bllen and selEnd > 0 and selEnd > selStart:
                        for start, end in selection_segments_excluding(
                            max(0, selStart),
                            min(block.length() - 1, selEnd),
                            self._relocated_spaces[
                                block.blockNumber()
                            ].values(),
                        ):
                            o = QTextLayout.FormatRange()
                            o.start = start
                            o.length = end - start
                            o.format = sel.format
                            selections.append(o)
                    elif not sel.cursor.hasSelection() \
                        and sel.format.hasProperty(QTextFormat.FullWidthSelection) \
                        and block.contains(sel.cursor.position()):
                        l = layout.lineForTextPosition(sel.cursor.position() - blpos)
                        for start, end in selection_segments_excluding(
                            l.textStart(),
                            l.textStart() + l.textLength(),
                            self._relocated_spaces[
                                block.blockNumber()
                            ].values(),
                        ):
                            o = QTextLayout.FormatRange()
                            o.start = start
                            o.length = end - start
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
                        painter,
                        block,
                        paint_context_without_selection_ranges(
                            self.document(),
                            block,
                            context,
                            self._relocated_spaces[
                                block.blockNumber()
                            ].values(),
                        ),
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

        if not self.defer_cursor_paint:
            cursor_rect = self.source_cursor_rect(context.cursorPosition)
            if cursor_rect is not None and not cursor_rect.isEmpty():
                painter.setCompositionMode(
                    QPainter.CompositionMode.RasterOp_NotDestination
                )
                painter.fillRect(cursor_rect, painter.pen().brush())
            elif cursor_block is not None:
                block = cursor_block
                layout = block.layout()
                cpos = _block_cursor_position(block, context.cursorPosition)
                if cpos >= 0:
                    layout.drawCursor(painter, QPointF(0, 0), cpos, 1)

            dirty_rect = QRectF()
            current_cursor_rect = (
                QRectF(cursor_rect)
                if cursor_rect is not None else QRectF()
            )
            if current_cursor_rect != self._cursor_update_rect:
                dirty_rect = current_cursor_rect.united(
                    self._cursor_update_rect
                )
            self._cursor_update_rect = current_cursor_rect
            if not dirty_rect.isEmpty():
                if C.USE_PYSIDE6:
                    self.update.emit()
                else:
                    self.update.emit(dirty_rect)
        painter.restore()
