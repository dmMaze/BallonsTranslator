import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QImage,
    QInputMethodEvent,
    QPainter,
    QTextCharFormat,
    QTextCursor,
)
from qtpy.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.textblock import TextBlock


class HorizontalWhitespaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(
        text: str,
        *,
        width: float = 80.0,
        height: float = 120.0,
    ) -> TextBlkItem:
        bounds = [0, 0, width, height]
        block = TextBlock(bounds)
        block._bounding_rect = list(bounds)
        block.translation = text
        block.fontformat.font_family = 'DejaVu Sans'
        block.fontformat.font_size = 24
        block.fontformat.letter_spacing = 1.0
        block.fontformat.line_spacing = 1.0
        return TextBlkItem(block, 0)

    def test_overflowing_spaces_consume_a_row_before_following_text(self):
        spaced = self._make_item('AAAA        B')

        spaced_lines = spaced.document().firstBlock().layout()
        self.assertEqual(spaced_lines.lineCount(), 2)

        moved_space_caret = spaced.layout.source_cursor_rect(6)
        self.assertIsNotNone(moved_space_caret)
        relocated_start = next(iter(
            spaced.layout._relocated_spaces[0].values()
        ))[0]
        first_relocated_caret = spaced.layout.source_cursor_rect(
            relocated_start
        )
        self.assertIsNotNone(first_relocated_caret)
        self.assertEqual(
            first_relocated_caret.top(), moved_space_caret.top()
        )
        following_line = spaced_lines.lineAt(1)
        self.assertEqual(following_line.textStart(), 12)
        self.assertLess(
            abs(following_line.y() - moved_space_caret.top()),
            4.0,
        )
        self.assertGreater(
            following_line.x(),
            spaced.layout.source_cursor_rect(11).center().x(),
        )
        following_rect = following_line.naturalTextRect()
        self.assertEqual(
            spaced.layout.hitTest(
                QPointF(
                    following_rect.right() - 1.0,
                    following_rect.center().y(),
                ),
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            13,
        )
        self.assertEqual(
            spaced.layout.hitTest(
                QPointF(1.0, moved_space_caret.center().y()),
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            5,
        )
        self.assertEqual(
            spaced.layout.hitTest(
                QPointF(
                    moved_space_caret.center().x() - 0.1,
                    moved_space_caret.center().y(),
                ),
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            6,
        )
        # The shared position before B belongs to Qt's following line.
        self.assertIsNone(spaced.layout.source_cursor_rect(12))
        self.assertEqual(spaced.toPlainText(), 'AAAA        B')

    def test_terminal_spaces_create_hittable_continuation_rows(self):
        item = self._make_item(' ' * 20, width=50.0)
        text_layout = item.document().firstBlock().layout()
        self.assertEqual(text_layout.lineCount(), 1)

        second_row_caret = item.layout.source_cursor_rect(13)
        final_caret = item.layout.source_cursor_rect(20)
        self.assertIsNotNone(second_row_caret)
        self.assertIsNotNone(final_caret)
        self.assertGreater(final_caret.top(), second_row_caret.top())
        self.assertEqual(
            item.layout.hitTest(
                QPointF(1.0, final_caret.center().y()),
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            18,
        )

        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(20)
        item.setTextCursor(cursor)
        queried = item.inputMethodQuery(
            Qt.InputMethodQuery.ImCursorRectangle
        )
        self.assertEqual(QRectF(queried), final_caret)

    def test_selection_uses_relocated_space_cells(self):
        item = self._make_item('AAAA        BBB')
        caret = item.layout.source_cursor_rect(6)
        self.assertIsNotNone(caret)

        cursor = QTextCursor(item.document())
        cursor.setPosition(5)
        cursor.setPosition(6, QTextCursor.MoveMode.KeepAnchor)
        selection = QAbstractTextDocumentLayout.Selection()
        selection.cursor = cursor
        selection.format = QTextCharFormat()
        selection.format.setBackground(Qt.GlobalColor.red)
        context = QAbstractTextDocumentLayout.PaintContext()
        context.cursorPosition = -1
        context.selections = [selection]

        image = QImage(
            100,
            140,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            item.layout.draw(painter, context)
        finally:
            painter.end()

        selected_point = QPointF(
            max(1.0, caret.center().x() / 2),
            caret.center().y(),
        ).toPoint()
        self.assertEqual(image.pixelColor(selected_point), Qt.GlobalColor.red)

    def test_paragraph_edits_and_undo_match_fresh_layout(self) -> None:
        item = self._make_item(
            'First line\nSecond line\nThird line', width=480, height=420,
        )

        def check() -> None:
            fresh = self._make_item(item.toPlainText(), width=480, height=420)
            try:
                actual = item.document().firstBlock()
                expected = fresh.document().firstBlock()
                while actual.isValid():
                    a_layout, e_layout = actual.layout(), expected.layout()
                    self.assertEqual(a_layout.lineCount(), e_layout.lineCount())
                    for index in range(a_layout.lineCount()):
                        a_line, e_line = a_layout.lineAt(index), e_layout.lineAt(index)
                        self.assertEqual(a_line.naturalTextRect(), e_line.naturalTextRect())
                        for position in range(actual.length()):
                            self.assertEqual(
                                a_line.cursorToX(position), e_line.cursorToX(position),
                            )
                    actual, expected = actual.next(), expected.next()
                images = []
                for candidate in (item, fresh):
                    image = QImage(500, 450, QImage.Format.Format_ARGB32_Premultiplied)
                    image.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(image)
                    try:
                        candidate.document().drawContents(painter)
                    finally:
                        painter.end()
                    images.append(image)
                self.assertEqual(images[0], images[1])
            finally:
                fresh.deleteLater()

        cursor = QTextCursor(item.document())
        cursor.setPosition(5)
        cursor.insertText('X')
        check()
        cursor.deletePreviousChar()
        check()
        cursor.insertBlock()
        check()
        cursor.deletePreviousChar()
        check()
        item.document().undo()
        check()
        item.document().redo()
        check()

        scene = QGraphicsScene()
        scene.addItem(item)
        view = QGraphicsView(scene)
        self.addCleanup(scene.deleteLater)
        self.addCleanup(view.close)
        view.show()
        item.startEdit()
        view.setFocus()
        item.setFocus()
        self.app.processEvents()
        self.assertTrue(item.hasFocus())
        item.setTextCursor(cursor)
        item.inputMethodEvent(QInputMethodEvent('かな', []))
        self.assertEqual(item.document().firstBlock().layout().preeditAreaText(), 'かな')
        item.inputMethodEvent(QInputMethodEvent())
        check()
        commit = QInputMethodEvent()
        commit.setCommitString('仮名')
        item.inputMethodEvent(commit)
        self.assertIn('仮名', item.toPlainText())
        check()
        item.document().undo()
        check()
        item.document().redo()
        check()


if __name__ == '__main__':
    unittest.main()
