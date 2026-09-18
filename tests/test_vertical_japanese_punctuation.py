import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtGui import QFontDatabase, QTextCursor
from qtpy.QtWidgets import QApplication
from qtpy import API_NAME

from ballontranslator.ui.text_engine.annotations import apply_text_combine_upright
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.glyph import glyph_geometry
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock import TextBlock


class JapanesePunctuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _item(self, text: str, font: str, standard: bool = True) -> TextBlkItem:
        database = QFontDatabase() if API_NAME == 'PyQt5' else QFontDatabase
        if font not in database.families():
            self.skipTest(f'{font} is required for the font-specific regression')
        item = TextBlkItem(TextBlock(
            xyxy=[0, 0, 300, 3000], _bounding_rect=[0, 0, 300, 3000],
            translation=text, text_layout_version=1,
            fontformat={
                'font_family': font, 'font_size': 30, 'vertical': True,
                'standard_vertical_roman_alignment': standard,
            },
        ), 0)
        self.addCleanup(item.deleteLater)
        return item

    def test_jlreq_brackets_hyphens_and_leaders_rotate(self) -> None:
        # Literal expectations cover omissions rather than mirroring the sets.
        text = '「」『』（ ）〔〕［］｛｝〈〉《》【】〖〗〘〙〚〛｟｠⦅⦆«»〝〞〟‐–〜～゠―—…‥ー'
        for font in ('MS Mincho', 'Yu Mincho'):
            for standard in (True, False):
                with self.subTest(font=font, standard=standard):
                    item = self._item(text, font, standard)
                    block = item.document().firstBlock()
                    for index, char in enumerate(text):
                        if char.isspace():
                            continue
                        line = block.layout().lineForTextPosition(index)
                        placed, offset, orientation = item.layout.vertical_line_placement(
                            block, line.lineNumber()
                        )
                        self.assertFalse(orientation.isIdentity(), char)
                        ink = glyph_geometry(
                            placed, index, 1, offset, orientation, 0.0
                        ).bounds
                        self.assertFalse(ink.isEmpty(), char)
                        if char in '‐–〜～゠―—…‥ー':
                            self.assertGreater(ink.height(), ink.width(), char)
                        self.assertTrue(
                            item.geometry_controller.source_paint_rect()
                            .adjusted(-0.1, -0.1, 0.1, 0.1).contains(ink), char
                        )
                    self.assertEqual(item.toPlainText(), text)

    def test_compact_brackets_keep_ink_inside_flow_cells(self) -> None:
        text = 'あ（い）〔う〕［え］｛お｝〈か〉《き》【く】「け」『こ』〘さ〙'
        for font in ('MS Mincho', 'Yu Mincho'):
            with self.subTest(font=font), patch.object(
                pcfg, 'compact_vertical_punctuation_spacing', True
            ):
                item = self._item(text, font)
                block = item.document().firstBlock()
                for index, char in enumerate(text):
                    if char not in '（）〔〕［］｛｝〈〉《》【】「」『』〘〙':
                        continue
                    line = block.layout().lineForTextPosition(index)
                    placed, offset, orientation = item.layout.vertical_line_placement(
                        block, line.lineNumber()
                    )
                    ink = glyph_geometry(placed, index, 1, offset, orientation, 0).bounds
                    cells = item.layout._vertical_line_cells(block, line.lineNumber())
                    self.assertGreaterEqual(ink.top(), cells[0][2] - 0.1, char)
                    self.assertLessEqual(ink.bottom(), cells[-1][3] + 0.1, char)
                    hit = item.layout.hitTest(ink.center(), Qt.HitTestAccuracy.FuzzyHit)
                    self.assertIn(hit, (index, index + 1), char)

    def test_colon_policy_and_combined_run_bypass(self) -> None:
        for font in ('MS Mincho', 'Yu Mincho'):
            for standard in (True, False):
                with self.subTest(font=font, standard=standard):
                    item = self._item('：；・！？', font, standard)
                    self.assertEqual(item.layout.needs_vertical_rotation('：'), standard)
                    self.assertEqual(item.layout.needs_vertical_rotation('；'), standard)
                    for char in '・！？':
                        self.assertFalse(item.layout.needs_vertical_rotation(char))
                    cursor = QTextCursor(item.document())
                    cursor.setPosition(0)
                    cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
                    apply_text_combine_upright(cursor, True)
                    item.refreshVerticalLayout()
                    block = item.document().firstBlock()
                    line = block.layout().lineForTextPosition(0)
                    _, _, orientation = item.layout.vertical_line_placement(block, line.lineNumber())
                    # Scaling/translation are allowed, but no sideways rotation.
                    self.assertEqual(orientation.m12(), 0)
                    self.assertEqual(orientation.m21(), 0)
                    self.assertEqual(item.toPlainText(), '：；・！？')


if __name__ == '__main__':
    unittest.main()
