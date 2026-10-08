import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.annotations import apply_text_combine_upright
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.textblock import TextBlock


class CombinedWhitespaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _item(self, text: str, end: int, standard: bool) -> TextBlkItem:
        item = TextBlkItem(TextBlock(
            xyxy=[0, 0, 300, 1800], _bounding_rect=[0, 0, 300, 1800],
            translation=text, text_layout_version=1,
            fontformat={'font_size': 30, 'vertical': True,
                        'standard_vertical_roman_alignment': standard},
        ), 0)
        self.addCleanup(item.deleteLater)
        cursor = QTextCursor(item.document())
        cursor.setPosition(0)
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


if __name__ == '__main__':
    unittest.main()
