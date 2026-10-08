import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QFontDatabase
from qtpy import API_NAME

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.vertical_layout import _is_hangul
from ballontranslator.ui.text_engine.rendering.glyph import glyph_geometry
from ballontranslator.utils.textblock import TextBlock


class VerticalHangulTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_script_boundary(self) -> None:
        for text in ('아', '아', 'ㅋㅋ', '힣', 'ﾡ'):
            self.assertTrue(_is_hangul(text))
        for text in ('', '。', '「', 'あ', '木', '12'):
            self.assertFalse(_is_hangul(text))

    def test_bearings_baseline_and_overflow_survive_resize(self) -> None:
        database = QFontDatabase() if API_NAME == 'PyQt5' else QFontDatabase
        if 'Noto Serif KR' not in database.families():
            self.skipTest('Noto Serif KR is required for real Hangul outlines')
        item = TextBlkItem(TextBlock(
            xyxy=[0, 0, 80, 600],
            _bounding_rect=[0, 0, 80, 600],
            translation='아야어여오요우유으이',
            fontformat={
                'font_family': 'Noto Serif KR', 'font_size': 30,
                'vertical': True, 'letter_spacing': 1.2,
            },
            text_layout_version=1,
        ), 0)
        for height in (600, 800, 600):
            item.set_size(80, height, set_layout_maxsize=True)
            block = item.document().firstBlock()
            offsets = []
            for index in range(block.layout().lineCount()):
                line, offset, transform = item.layout.vertical_line_placement(
                    block, index
                )
                offsets.append((offset.x(), offset.y()))
                ink = glyph_geometry(
                    line, line.textStart(), line.textLength(),
                    offset, transform, 0.0,
                ).bounds
                self.assertGreaterEqual(ink.top() - line.y(), -1e-5)
                self.assertTrue(
                    item.geometry_controller.source_paint_rect()
                    .adjusted(-1e-5, -1e-5, 1e-5, 1e-5).contains(ink)
                )
            # Equal advance syllables retain the same origin and baseline;
            # their intentionally different ink bearings must remain visible.
            self.assertEqual(len(offsets), 10)
            for x, y in offsets:
                self.assertAlmostEqual(x, offsets[0][0], places=5)
                self.assertAlmostEqual(y, offsets[0][1], places=5)
        item.deleteLater()


if __name__ == '__main__':
    unittest.main()
