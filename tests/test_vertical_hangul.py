import os
import unittest
import unicodedata
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication, QGraphicsScene
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QFontDatabase, QImage, QPainter, QTextCharFormat, QTextCursor
from qtpy import API_NAME

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine import vertical_layout
from ballontranslator.ui.text_engine.annotations import apply_ruby, apply_text_combine_upright
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

    def _item(self, text: str, standard: bool = True) -> TextBlkItem:
        item = TextBlkItem(TextBlock(
            xyxy=[0, 0, 220, 600], _bounding_rect=[0, 0, 220, 600],
            translation=text, text_layout_version=1,
            fontformat={'font_family': 'Noto Serif CJK KR', 'font_size': 30,
                        'vertical': True, 'letter_spacing': 1.0,
                        'standard_vertical_roman_alignment': standard},
        ), 0)
        self.addCleanup(item.deleteLater)
        return item

    @staticmethod
    def _ink(item: TextBlkItem, position: int) -> QRectF:
        block = item.document().firstBlock()
        number = block.layout().lineForTextPosition(position).lineNumber()
        line, offset, transform = item.layout.vertical_line_placement(block, number)
        return glyph_geometry(
            line, line.textStart(), line.textLength(), offset, transform, 0.0,
        ).bounds

    def test_formatted_leading_spaces_do_not_move_hangul_sideways(self) -> None:
        for standard in (True, False):
            with self.subTest(standard=standard):
                reference = self._item('아', standard)
                spaced = self._item('  아', standard)
                cursor = QTextCursor(spaced.document())
                cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
                fmt = QTextCharFormat()
                fmt.setFontPointSize(48)
                cursor.mergeCharFormat(fmt)
                spaced.refreshVerticalLayout()
                self.assertAlmostEqual(
                    self._ink(reference, 0).left(), self._ink(spaced, 2).left(), places=5,
                )

    def test_larger_font_does_not_shift_other_hangul_baselines(self) -> None:
        item = self._item('아야어')
        top = self._ink(item, 0).top()
        cursor = QTextCursor(item.document())
        cursor.setPosition(2)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setFontPointSize(120)
        cursor.mergeCharFormat(fmt)
        item.refreshVerticalLayout()
        self.assertAlmostEqual(self._ink(item, 0).top(), top, places=5)

    def test_decomposed_syllables_have_the_same_placement(self) -> None:
        text = '아야어여'
        items = (self._item(text), self._item(unicodedata.normalize('NFD', text)))
        for position in range(len(text)):
            self.assertEqual(self._ink(items[0], position), self._ink(items[1], position * 2))

    def test_hangul_placement_preserves_mixed_text_flow_and_annotations(self) -> None:
        text = '아A木、야12어여'
        for standard in (True, False):
            items = []
            for enabled in (False, True):
                with patch.object(
                    vertical_layout, '_is_hangul',
                    _is_hangul if enabled else lambda _char: False,
                ):
                    item = self._item(text, standard)
                    cursor = QTextCursor(item.document())
                    cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
                    apply_ruby(cursor, 'group', '가나다')
                    cursor.setPosition(2)
                    cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
                    apply_ruby(cursor, 'group', 'もくてん')
                    cursor.setPosition(5)
                    cursor.setPosition(7, QTextCursor.MoveMode.KeepAnchor)
                    apply_text_combine_upright(cursor, True)
                    item.refreshVerticalLayout()
                    items.append(item)
            before = [(item.toHtml(), item.document().availableUndoSteps()) for item in items]
            for height in (600, 140, 600):
                snapshots = []
                for enabled, item in zip((False, True), items):
                    with patch.object(
                        vertical_layout, '_is_hangul',
                        _is_hangul if enabled else lambda _char: False,
                    ):
                        item.set_size(220, height, set_layout_maxsize=True)
                        item.refresh_font_metrics()
                    block = item.document().firstBlock()
                    lines = block.layout()
                    snapshots.append((
                        [lines.lineAt(n).position() for n in range(lines.lineCount())],
                        [item.layout.source_cursor_rect(p) for p in range(len(text) + 1)],
                        [self._ink(item, p) for p in (1, 2, 3, 5)],
                        [(ruby.cell, ruby.ink_bounds) for ruby in item.layout._vertical_ruby_placements(block)],
                    ))
                    for position in (0, 4, 7, 8):
                        line = lines.lineForTextPosition(position)
                        cells = item.layout._vertical_line_cells(block, line.lineNumber())
                        cell = next(c for c in cells if c[0] == position)
                        point = QPointF(line.x() + 15, (cell[2] + cell[3]) / 2)
                        self.assertIn(item.layout.hitTest(point, Qt.HitTestAccuracy.FuzzyHit), (position, position + 1))
                    self.assertEqual(item.toPlainText(), text)
                self.assertEqual(snapshots[0], snapshots[1])
            self.assertEqual(before, [(item.toHtml(), item.document().availableUndoSteps()) for item in items])

    def test_effect_export_and_geometry_stay_stable_after_relayout(self) -> None:
        for slant in (0, 18):
            with self.subTest(slant=slant):
                block = TextBlock(
                    xyxy=[80, 80, 300, 600], _bounding_rect=[80, 80, 220, 520],
                    translation='아 야어여12木A', text_layout_version=1,
                    fontformat={'font_family': 'Noto Serif CJK KR', 'font_size': 36,
                                'vertical': True, 'letter_spacing': 1.2,
                                'stroke_width': 0.15, 'glyph_slant_angle': slant},
                )
                item = TextBlkItem(block, 0)
                item.setAngle(15)
                scene = QGraphicsScene()
                scene.addItem(item)
                try:
                    def render() -> QImage:
                        image = QImage(500, 750, QImage.Format.Format_ARGB32_Premultiplied)
                        image.fill(Qt.GlobalColor.transparent)
                        painter = QPainter(image)
                        try:
                            scene.render(painter, QRectF(0, 0, 500, 750), QRectF(-50, -50, 500, 750))
                        finally:
                            painter.end()
                        return image

                    preview = render()
                    state = (item.absBoundingRect(qrect=True), list(block._bounding_rect),
                             item.toHtml(), item.document().availableUndoSteps())
                    for _ in range(2):
                        item.refresh_font_metrics()
                        item.repaint_background()
                        self.assertEqual(render(), preview)
                        self.assertEqual(
                            (item.absBoundingRect(qrect=True), list(block._bounding_rect),
                             item.toHtml(), item.document().availableUndoSteps()), state,
                        )
                    item.set_export_effect_render(True)
                    self.assertEqual(render(), preview)
                finally:
                    item.set_export_effect_render(False)
                    scene.clear()

    def test_bearings_baseline_and_overflow_survive_resize(self) -> None:
        database = QFontDatabase() if API_NAME == 'PyQt5' else QFontDatabase
        if 'Noto Serif CJK KR' not in database.families():
            self.skipTest('Noto Serif CJK KR is required for real Hangul outlines')
        item = TextBlkItem(TextBlock(
            xyxy=[0, 0, 80, 600],
            _bounding_rect=[0, 0, 80, 600],
            translation='아야어여오요우유으이',
            fontformat={
                'font_family': 'Noto Serif CJK KR', 'font_size': 30,
                'vertical': True, 'letter_spacing': 1.2,
            },
            text_layout_version=1,
        ), 0)
        self.addCleanup(item.deleteLater)
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


if __name__ == '__main__':
    unittest.main()
