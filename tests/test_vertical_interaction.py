import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QImage,
    QPainter,
    QTextCharFormat,
    QTextCursor,
)
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.textblock import TextBlock


class VerticalInteractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(
        text: str,
        *,
        letter_spacing: float = 1.0,
    ) -> TextBlkItem:
        bounds = [0, 0, 220, 300]
        block = TextBlock(bounds)
        block._bounding_rect = list(bounds)
        block.translation = text
        block.fontformat.vertical = True
        block.fontformat.font_family = 'Noto Sans CJK SC'
        block.fontformat.font_size = 40
        block.fontformat.letter_spacing = letter_spacing
        return TextBlkItem(block, 0)

    @staticmethod
    def _selection_context(
        item: TextBlkItem,
        start: int,
        end: int,
    ) -> QAbstractTextDocumentLayout.PaintContext:
        cursor = QTextCursor(item.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        selection = QAbstractTextDocumentLayout.Selection()
        selection.cursor = cursor
        selection.format = QTextCharFormat()
        selection.format.setBackground(Qt.GlobalColor.red)
        context = QAbstractTextDocumentLayout.PaintContext()
        context.cursorPosition = -1
        context.selections = [selection]
        return context

    def test_joined_punctuation_shares_caret_and_hit_cells(self):
        item = self._make_item('——')
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        cells = item.layout._vertical_line_cells(block, 0)

        self.assertEqual([(cell[0], cell[1]) for cell in cells], [
            (0, 1),
            (1, 2),
        ])
        caret_positions = [
            item.layout.source_cursor_rect(position).top()
            for position in range(3)
        ]
        self.assertEqual(caret_positions, [
            cells[0][2],
            cells[0][3],
            cells[1][3],
        ])

        x = line.x() + 1.0
        for start, end, top, bottom, _is_space in cells:
            height = bottom - top
            self.assertEqual(
                item.layout.hitTest(
                    QPointF(x, top + height / 4),
                    Qt.HitTestAccuracy.FuzzyHit,
                ),
                start,
            )
            self.assertEqual(
                item.layout.hitTest(
                    QPointF(x, bottom - height / 4),
                    Qt.HitTestAccuracy.FuzzyHit,
                ),
                end,
            )

        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        item.setTextCursor(cursor)
        queried = item.inputMethodQuery(
            Qt.InputMethodQuery.ImCursorRectangle
        )
        self.assertEqual(
            QRectF(queried), item.layout.source_cursor_rect(1)
        )

    def test_surrogate_pair_never_exposes_an_internal_hit_cell(self):
        item = self._make_item('😀木')
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        cells = item.layout._vertical_line_cells(block, 0)

        self.assertEqual([(cells[0][0], cells[0][1])], [(0, 2)])
        _start, _end, top, bottom, _is_space = cells[0]
        x = line.x() + 1.0
        self.assertEqual(
            item.layout.hitTest(
                QPointF(x, top + (bottom - top) / 4),
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            0,
        )
        self.assertEqual(
            item.layout.hitTest(
                QPointF(x, bottom - (bottom - top) / 4),
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            2,
        )
        self.assertEqual(
            item.layout.source_cursor_rect(1),
            item.layout.source_cursor_rect(2),
        )

    def test_positive_joined_spacing_remains_a_trailing_run_advance(self):
        normal = self._make_item('——')
        spaced = self._make_item('——', letter_spacing=1.5)
        normal_cells = normal.layout._vertical_line_cells(
            normal.document().firstBlock(), 0
        )
        spaced_cells = spaced.layout._vertical_line_cells(
            spaced.document().firstBlock(), 0
        )

        self.assertEqual(spaced_cells[0][3], normal_cells[0][3])
        self.assertGreater(spaced_cells[-1][3], normal_cells[-1][3])

    def test_selection_uses_vertical_glyph_and_space_cells(self):
        for text, start, end, line_number, selected_cell in (
            ('——', 0, 1, 0, 0),
            ('木   水', 2, 3, 0, 2),
        ):
            with self.subTest(text=text):
                item = self._make_item(text)
                block = item.document().firstBlock()
                line = block.layout().lineAt(line_number)
                cells = item.layout._vertical_line_cells(
                    block, line_number
                )
                context = self._selection_context(item, start, end)
                image = QImage(
                    240,
                    320,
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                try:
                    item.layout.draw(painter, context)
                finally:
                    painter.end()

                width = item.layout._vertical_line_width(
                    block, line_number
                )
                sample_x = round(line.x() + width - 1.0)
                selected = cells[selected_cell]
                selected_y = round((selected[2] + selected[3]) / 2)
                self.assertEqual(
                    image.pixelColor(sample_x, selected_y),
                    Qt.GlobalColor.red,
                )
                unselected = cells[
                    selected_cell + 1
                    if selected_cell + 1 < len(cells)
                    else selected_cell - 1
                ]
                unselected_y = round(
                    (unselected[2] + unselected[3]) / 2
                )
                self.assertNotEqual(
                    image.pixelColor(sample_x, unselected_y),
                    Qt.GlobalColor.red,
                )

    def test_selection_background_overlays_inline_background(self):
        item = self._make_item('木水')
        cursor = QTextCursor(item.document())
        cursor.setPosition(0)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        inline_format = QTextCharFormat()
        inline_format.setBackground(QColor('blue'))
        cursor.mergeCharFormat(inline_format)

        context = self._selection_context(item, 0, 1)
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        cell = item.layout._vertical_line_cells(block, 0)[0]
        width = item.layout._vertical_line_width(block, 0)
        image = QImage(
            240,
            320,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            item.layout.draw(painter, context)
        finally:
            painter.end()

        y = round((cell[2] + cell[3]) / 2)
        for x in (
            round(line.x() + 1.0),
            round(line.x() + width - 1.0),
        ):
            self.assertEqual(
                image.pixelColor(x, y), Qt.GlobalColor.red
            )

    def test_selected_glyph_geometry_is_reused_until_relayout(self):
        item = self._make_item('木火水')
        context = self._selection_context(item, 0, 3)
        image = QImage(
            240,
            320,
            QImage.Format.Format_ARGB32_Premultiplied,
        )

        def draw() -> None:
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            try:
                item.layout.draw(painter, context)
            finally:
                painter.end()

        draw()
        cache_size = len(item.layout._selection_geometry_cache)
        self.assertGreater(cache_size, 0)
        draw()
        self.assertEqual(
            len(item.layout._selection_geometry_cache), cache_size
        )

        item.layout.reLayout()
        self.assertEqual(item.layout._selection_geometry_cache, {})


if __name__ == '__main__':
    unittest.main()
