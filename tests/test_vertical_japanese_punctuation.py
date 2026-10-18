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

    def _item(
        self, text: str, font: str, standard: bool = True, font_size: float = 30,
    ) -> TextBlkItem:
        database = QFontDatabase() if API_NAME == 'PyQt5' else QFontDatabase
        if font not in database.families():
            self.skipTest(f'{font} is required for the font-specific regression')
        item = TextBlkItem(TextBlock(
            xyxy=[0, 0, 300, 3000], _bounding_rect=[0, 0, 300, 3000],
            translation=text, text_layout_version=1,
            fontformat={
                'font_family': font, 'font_size': font_size, 'vertical': True,
                'standard_vertical_roman_alignment': standard,
            },
        ), 0)
        self.addCleanup(item.deleteLater)
        return item

    def test_jlreq_brackets_hyphens_and_leaders_rotate(self) -> None:
        # Literal expectations cover omissions rather than mirroring the sets.
        text = '「」『』（ ）〔〕［］｛｝〈〉《》【】〖〗〘〙〚〛｟｠⦅⦆«»〝〞〟‐–〜～゠―—…‥ー'
        for font in ('Noto Serif CJK JP', 'MS Mincho', 'Yu Mincho'):
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
        for font in ('Noto Serif CJK JP', 'MS Mincho', 'Yu Mincho'):
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

    def test_compact_opening_bracket_does_not_overlap_its_neighbors(self) -> None:
        # Malgun Gothic has less leading bearing than the compact-cell trim.
        # The old offset pushed the opening bracket into the preceding 木.
        for standard in (True, False):
            for text in ('木（木', '木  （木'):
                with self.subTest(standard=standard, text=text), patch.object(
                    pcfg, 'compact_vertical_punctuation_spacing', True
                ):
                    item = self._item(text, 'Malgun Gothic', standard, font_size=96)
                    before = (item.toHtml(), item.document().availableUndoSteps())
                    position = text.index('（')
                    for height in (3000, 160, 3000):
                        item.set_size(300, height, set_layout_maxsize=True)
                        block = item.document().firstBlock()
                        geometries = []
                        for index in (0, position, position + 1):
                            line = block.layout().lineForTextPosition(index)
                            placed, offset, orientation = item.layout.vertical_line_placement(
                                block, line.lineNumber()
                            )
                            geometries.append(glyph_geometry(
                                placed, index, 1, offset, orientation, 0.0,
                            ))
                        bracket = geometries[1]
                        line = block.layout().lineForTextPosition(position)
                        cells = item.layout._vertical_line_cells(block, line.lineNumber())
                        cell = next(cell for cell in cells if cell[0] == position)
                        self.assertGreaterEqual(bracket.bounds.top(), cell[2] - 0.1)
                        self.assertLessEqual(bracket.bounds.bottom(), cell[3] + 0.1)
                        for neighbor in (geometries[0], geometries[2]):
                            self.assertFalse(any(
                                ink.intersects(adjacent)
                                for ink in bracket.paths for adjacent in neighbor.paths
                            ))
                        self.assertTrue(
                            item.geometry_controller.source_paint_rect()
                            .adjusted(-0.1, -0.1, 0.1, 0.1).contains(bracket.bounds)
                        )
                        self.assertIn(
                            item.layout.hitTest(bracket.bounds.center(), Qt.HitTestAccuracy.FuzzyHit),
                            (position, position + 1),
                        )
                    self.assertEqual(
                        (item.toHtml(), item.document().availableUndoSteps()), before,
                    )

    def test_colon_policy_and_combined_run_bypass(self) -> None:
        for font in ('Noto Serif CJK JP', 'MS Mincho', 'Yu Mincho'):
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
