import os
import unittest
from typing import List


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QEvent, QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QImage,
    QKeyEvent,
    QPainter,
    QTextCharFormat,
    QTextCursor,
)
from qtpy.QtWidgets import QApplication, QGraphicsScene

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
        height: int = 300,
        standard_vertical_roman_alignment: bool = True,
    ) -> TextBlkItem:
        bounds = [0, 0, 220, height]
        block = TextBlock(bounds)
        block._bounding_rect = list(bounds)
        block.translation = text
        block.fontformat.vertical = True
        block.fontformat.font_family = 'Noto Sans CJK SC'
        block.fontformat.font_size = 40
        block.fontformat.letter_spacing = letter_spacing
        block.fontformat.standard_vertical_roman_alignment = (
            standard_vertical_roman_alignment
        )
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
        for roman_alignment in (False, True):
            for text in ('——', '――', '……', '‥‥', '⋯⋯'):
                with self.subTest(
                    roman_alignment=roman_alignment,
                    text=text,
                ):
                    source = f'木{text}水'
                    normal = self._make_item(
                        source,
                        standard_vertical_roman_alignment=roman_alignment,
                    )
                    spaced = self._make_item(
                        source,
                        letter_spacing=1.5,
                        standard_vertical_roman_alignment=roman_alignment,
                    )
                    normal_block = normal.document().firstBlock()
                    spaced_block = spaced.document().firstBlock()
                    normal_line = normal_block.layout().lineForTextPosition(1)
                    spaced_line = spaced_block.layout().lineForTextPosition(1)

                    self.assertEqual(
                        (normal_line.textStart(), normal_line.textLength()),
                        (1, 2),
                    )
                    self.assertEqual(
                        (spaced_line.textStart(), spaced_line.textLength()),
                        (1, 2),
                    )
                    self.assertEqual(
                        spaced_block.layout().lineForTextPosition(
                            3
                        ).textStart(),
                        3,
                    )
                    normal_cells = normal.layout._vertical_line_cells(
                        normal_block, normal_line.lineNumber()
                    )
                    spaced_cells = spaced.layout._vertical_line_cells(
                        spaced_block, spaced_line.lineNumber()
                    )

                    self.assertEqual(len(normal_cells), 2)
                    self.assertEqual(len(spaced_cells), 2)
                    self.assertEqual(
                        spaced_cells[0][3] - spaced_cells[0][2],
                        normal_cells[0][3] - normal_cells[0][2],
                    )
                    self.assertGreater(
                        spaced_cells[-1][3] - spaced_cells[0][2],
                        normal_cells[-1][3] - normal_cells[0][2],
                    )

    def test_joined_punctuation_uses_widest_fragment_column(self):
        for large_position in (0, 1):
            with self.subTest(large_position=large_position):
                item = self._make_item('……', letter_spacing=1.5)
                cursor = QTextCursor(item.document())
                cursor.setPosition(large_position)
                cursor.setPosition(
                    large_position + 1,
                    QTextCursor.MoveMode.KeepAnchor,
                )
                char_format = QTextCharFormat()
                char_format.setFontPointSize(80.0)
                cursor.mergeCharFormat(char_format)
                self.app.processEvents()

                block = item.document().firstBlock()
                line = block.layout().lineAt(0)
                widths = [
                    item.layout.get_char_fontfmt(0, position).tbr.width()
                    for position in range(2)
                ]
                self.assertEqual(
                    (line.textStart(), line.textLength()), (0, 2)
                )
                self.assertAlmostEqual(
                    item.layout._line_record(block, 0)['base_width'],
                    max(widths),
                )

    def test_joined_punctuation_stops_at_tate_chu_yoko_boundary(self):
        item = self._make_item('……', letter_spacing=1.5)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        self.app.processEvents()

        block = item.document().firstBlock()
        layout = block.layout()
        self.assertEqual(
            (layout.lineAt(0).textStart(), layout.lineAt(0).textLength()),
            (0, 1),
        )
        self.assertIsNotNone(item.layout.tate_chu_yoko_cell_rect(block, 1))

    def test_leading_spaces_keep_following_punctuation_joined(self):
        item = self._make_item('  ……水', letter_spacing=1.5)
        block = item.document().firstBlock()
        layout = block.layout()

        self.assertEqual(
            (layout.lineAt(0).textStart(), layout.lineAt(0).textLength()),
            (0, 4),
        )
        self.assertEqual(layout.lineForTextPosition(4).textStart(), 4)

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

    def test_left_right_moves_between_vertical_columns(self):
        probe = self._make_item('A' * 40, height=180)

        def column_positions(item: TextBlkItem) -> List[List[int]]:
            block = item.document().firstBlock()
            layout = block.layout()
            columns = {}
            for line_number in range(layout.lineCount()):
                line = layout.lineAt(line_number)
                columns.setdefault(float(line.x()), []).append(
                    block.position() + line.textStart()
                )
            return [columns[x] for x in sorted(columns, reverse=True)]

        capacity = max(map(len, column_positions(probe)))
        item = self._make_item('A' * (capacity * 3 + 1), height=180)
        columns = column_positions(item)
        self.assertEqual([len(column) for column in columns], [
            capacity,
            capacity,
            capacity,
            1,
        ])

        scene = QGraphicsScene()
        scene.addItem(item)
        item.startEdit()
        original_text = item.toPlainText()
        item.document().clearUndoRedoStacks()
        item.updateUndoSteps()

        def set_position(position: int) -> None:
            cursor = item.textCursor()
            cursor.setPosition(position)
            item.setTextCursor(cursor)

        def press(
            key: Qt.Key,
            modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        ) -> None:
            item.keyPressEvent(QKeyEvent(
                QEvent.Type.KeyPress,
                key,
                modifiers,
            ))

        start = columns[1][1]
        set_position(start)
        press(Qt.Key.Key_Left)
        self.assertEqual(item.textCursor().position(), columns[2][1])
        press(Qt.Key.Key_Right)
        self.assertEqual(item.textCursor().position(), start)

        set_position(start)
        press(Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(
            (item.textCursor().position(), item.textCursor().anchor()),
            (columns[2][1], start),
        )

        # A short column clamps the first move, but the original visual row is
        # retained when moving back into a taller neighboring column.
        sticky_start = columns[2][-1]
        set_position(sticky_start)
        press(Qt.Key.Key_Left)
        self.assertEqual(item.textCursor().position(), len(original_text))
        press(Qt.Key.Key_Right)
        self.assertEqual(item.textCursor().position(), sticky_start)

        self.assertEqual(item.toPlainText(), original_text)
        self.assertEqual(item.document().availableUndoSteps(), 0)


if __name__ == '__main__':
    unittest.main()
