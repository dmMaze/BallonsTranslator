import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QAbstractTextDocumentLayout, QImage, QInputMethodEvent, QPainter, QTextCharFormat, QTextCursor
from qtpy.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from ballontranslator.ui.text_engine.annotations import apply_text_combine_upright
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.glyph import glyph_geometry
from ballontranslator.ui.text_engine.rendering.indexing import _utf16_length
from ballontranslator.utils.textblock import TextBlock


class CombinedWhitespaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _item(
        self, text: str, end: int, standard: bool, start: int = 0, slant: float = 0,
    ) -> TextBlkItem:
        item = TextBlkItem(TextBlock(
            xyxy=[0, 0, 300, 1800], _bounding_rect=[0, 0, 300, 1800],
            translation=text, text_layout_version=1,
            fontformat={'font_family': 'Noto Sans CJK SC',
                        'font_size': 30, 'vertical': True, 'glyph_slant_angle': slant,
                        'standard_vertical_roman_alignment': standard},
        ), 0)
        self.addCleanup(item.deleteLater)
        cursor = QTextCursor(item.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        apply_text_combine_upright(cursor, True)
        item.refreshVerticalLayout()
        return item

    def test_spaces_after_combined_run_have_vertical_cells_and_carets(self) -> None:
        for standard in (True, False):
            for count in (1, 4, 12):
                with self.subTest(standard=standard, count=count):
                    text = '#109' + ' ' * count + 'Title'
                    item = self._item(text, 4, standard)
                    block = item.document().firstBlock()
                    spaces = [c for c in item.layout._vertical_line_cells(block, 0) if c[4]]
                    self.assertEqual(len(spaces), count)
                    combined = item.layout.tate_chu_yoko_cell_rect(block, 0)
                    self.assertTrue(item.layout.is_tate_chu_yoko_line(
                        block, block.layout().lineForTextPosition(3).lineNumber(),
                    ))
                    self.assertAlmostEqual(spaces[0][2], combined.bottom())
                    for start, end, top, bottom, _ in spaces:
                        self.assertGreater(bottom, top)
                        caret = item.layout.source_cursor_rect(start)
                        self.assertAlmostEqual(caret.top(), top)
                        point = QPointF(combined.center().x(), (top + bottom) / 2)
                        self.assertIn(item.layout.hitTest(point, Qt.HitTestAccuracy.FuzzyHit), (start, end))
                    title = item.layout.source_cursor_rect(4 + count)
                    self.assertAlmostEqual(title.top(), spaces[-1][3])
                    self.assertEqual(item.toPlainText(), text)

    def test_authored_internal_space_stays_in_horizontal_run(self) -> None:
        item = self._item('#1 09   Title', 5, True)
        block = item.document().firstBlock()
        spaces = [c for c in item.layout._vertical_line_cells(block, 0) if c[4]]
        self.assertEqual([c[0] for c in spaces], [5, 6, 7])
        cell = item.layout.tate_chu_yoko_cell_rect(block, 0)
        self.assertAlmostEqual(item.layout.source_cursor_rect(2).top(), cell.top())

    def test_trailing_spaces_preserve_combined_ink_and_horizontal_carets(self) -> None:
        prefix = '年😀'
        start = _utf16_length(prefix)
        for text in ('#109', '！', '１２', '1 2', '12 ', '😀', 'ffi', 'אב', 'Aאב'):
            for standard in (True, False):
                with self.subTest(text=text, standard=standard):
                    end = start + _utf16_length(text)
                    reference = self._item(prefix + text + '木', end, standard, start)
                    spaced = self._item(prefix + text + ' ' * 12 + '木', end, standard, start)
                    geometry = []
                    hits = []
                    for item in (reference, spaced):
                        block = item.document().firstBlock()
                        number = block.layout().lineForTextPosition(start).lineNumber()
                        line, offset, transform = item.layout.vertical_line_placement(block, number)
                        cell = item.layout.tate_chu_yoko_cell_rect(block, number)
                        geometry.append((cell, glyph_geometry(
                            line, start, end - start, offset, transform, 0.0,
                        ).bounds))
                        hit = item.layout.hitTest(
                            QPointF(cell.right() - 0.01, cell.center().y()),
                            Qt.HitTestAccuracy.FuzzyHit,
                        )
                        self.assertIn(hit, (start, end))
                        hits.append(hit)
                    self.assertEqual(hits[0], hits[1])
                    self.assertEqual(geometry[0], geometry[1])
                    self.assertEqual(
                        reference.layout.source_cursor_rect(start),
                        spaced.layout.source_cursor_rect(start),
                    )

    def test_trailing_space_formatting_does_not_change_combined_text(self) -> None:
        item = self._item('12    木', 2, False)
        block = item.document().firstBlock()
        before_cell = item.layout.tate_chu_yoko_cell_rect(block, 0)
        before_caret = item.layout.source_cursor_rect(1)
        line, offset, transform = item.layout.vertical_line_placement(block, 0)
        before_ink = glyph_geometry(line, 0, 2, offset, transform, 0.0).bounds
        cursor = QTextCursor(item.document())
        cursor.setPosition(2)
        cursor.setPosition(6, QTextCursor.MoveMode.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setFontPointSize(72)
        cursor.mergeCharFormat(fmt)
        item.refreshVerticalLayout()
        line, offset, transform = item.layout.vertical_line_placement(block, 0)
        self.assertEqual(glyph_geometry(line, 0, 2, offset, transform, 0.0).bounds, before_ink)
        self.assertEqual(item.layout.tate_chu_yoko_cell_rect(block, 0), before_cell)
        self.assertEqual(item.layout.source_cursor_rect(1), before_caret)

    def test_trailing_spaces_paint_selection_in_vertical_cells(self) -> None:
        for slant in (0, 18):
            with self.subTest(slant=slant):
                item = self._item('12   木', 2, False, slant=slant)
                block = item.document().firstBlock()
                selection = QAbstractTextDocumentLayout.Selection()
                selection.cursor = QTextCursor(item.document())
                selection.cursor.setPosition(2)
                selection.cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
                selection.format.setBackground(Qt.GlobalColor.red)
                context = QAbstractTextDocumentLayout.PaintContext()
                context.cursorPosition = -1
                context.selections = [selection]
                image = QImage(300, 300, QImage.Format.Format_ARGB32_Premultiplied)
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                try:
                    item.layout.draw(painter, context)
                finally:
                    painter.end()
                cell = item.layout.tate_chu_yoko_cell_rect(block, 0)
                spaces = [c for c in item.layout._vertical_line_cells(block, 0) if c[4]]
                self.assertEqual(len(spaces), 3)
                for _start, _end, top, bottom, _space in spaces:
                    self.assertEqual(
                        image.pixelColor(round(cell.center().x()), round((top + bottom) / 2)),
                        Qt.GlobalColor.red,
                    )

    def test_preedit_is_not_reclassified_as_trailing_spaces(self) -> None:
        item = self._item('#109    Title', 4, True)
        scene = QGraphicsScene()
        scene.addItem(item)
        view = QGraphicsView(scene)
        try:
            view.show()
            view.setFocus()
            item.startEdit()
            self.app.processEvents()
            cursor = item.textCursor()
            cursor.setPosition(1)
            item.setTextCursor(cursor)
            item.inputMethodEvent(QInputMethodEvent('pin', []))
            block = item.document().firstBlock()
            self.assertEqual(block.layout().preeditAreaText(), 'pin')
            for offset in range(4):
                position = block.layout().preeditAreaPosition() + offset
                number = block.layout().lineForTextPosition(position).lineNumber()
                # Qt 5 and 6 split the native preedit run at different points.
                if not item.layout.is_tate_chu_yoko_line(block, number):
                    continue
                line, _offset, transform = item.layout.vertical_line_placement(block, number)
                native_x = item.layout._line_cursor_x(line, position)
                expected_x = transform.map(QPointF(native_x, line.y())).x()
                # Qt encodes preedit cursor offsets as -(offset + 2).
                caret = item.layout.source_cursor_rect(-(offset + 2))
                self.assertAlmostEqual(caret.center().x(), expected_x)
                self.assertGreater(caret.height(), caret.width())
            self.assertEqual(item.toPlainText(), '#109    Title')
            item.inputMethodEvent(QInputMethodEvent('', []))
            spaces = [c for c in item.layout._vertical_line_cells(block, 0) if c[4]]
            self.assertEqual(len(spaces), 4)
            commit = QInputMethodEvent('', [])
            commit.setCommitString('字')
            item.inputMethodEvent(commit)
            self.assertEqual(item.toPlainText(), '#字109    Title')
            item.undo()
            self.assertEqual(item.toPlainText(), '#109    Title')
        finally:
            item.endEdit(keep_focus=False)
            view.close()
            view.deleteLater()
            scene.removeItem(item)


if __name__ == '__main__':
    unittest.main()
