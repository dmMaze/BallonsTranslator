import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, QEvent, QCoreApplication
from qtpy.QtGui import QFont, QFontMetricsF, QRawFont, QTransform
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine import layout
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.glyph import glyph_geometry
from ballontranslator.utils.textblock import TextBlock


class FontMetricsFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        layout.clear_font_metrics_cache()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_supported_reference_keeps_tight_ink(self) -> None:
        font = QFont('DejaVu Sans')
        font.setPointSizeF(30)
        metrics = QFontMetricsF(font)
        for text in ('fg', '木fg'):
            with self.subTest(text=text):
                tight, bounds = layout.get_punc_rect(text, font.family(), 30, font.weight(), False)
                self.assertEqual(tight, metrics.tightBoundingRect(text))
                self.assertEqual(bounds, metrics.boundingRect(text))

    def test_valid_substitute_glyph_metrics_keep_vertical_spacing(self) -> None:
        for family in ('DejaVu Sans', 'Liberation Serif', 'Noto Sans'):
            with self.subTest(family=family):
                font = QFont(family)
                font.setPointSizeF(30)
                metrics = QFontMetricsF(font)
                self.assertFalse(QRawFont.fromFont(font).supportsCharacter(ord('木')))
                expected = metrics.tightBoundingRect('木')
                self.assertFalse(expected.isEmpty())
                actual, _ = layout.get_punc_rect('木', family, 30, font.weight(), False)
                self.assertEqual(actual, expected)
                item = TextBlkItem(TextBlock(
                    xyxy=[0, 0, 200, 500], _bounding_rect=[0, 0, 200, 500],
                    translation='木木', text_layout_version=1,
                    fontformat={'font_family': family, 'font_size': 40,
                                'vertical': True, 'letter_spacing': 1},
                ), 0)
                try:
                    lines = item.document().firstBlock().layout()
                    self.assertAlmostEqual(
                        lines.lineAt(1).y() - lines.lineAt(0).y(), expected.height(),
                    )
                finally:
                    item.deleteLater()

    def test_empty_ink_uses_bounds_then_line_metrics(self) -> None:
        metrics = Mock()
        metrics.tightBoundingRect.return_value = QRectF()
        metrics.ascent.return_value = 24.0
        metrics.descent.return_value = 6.0
        metrics.horizontalAdvance.return_value = 18.0
        raw_font = Mock()
        raw_font.isValid.return_value = True
        raw_font.supportsCharacter.return_value = True
        for bounds, expected in (
            (QRectF(1, -20, 18, 25), QRectF(1, -20, 18, 25)),
            (QRectF(), QRectF(0, -24, 18, 30)),
        ):
            with self.subTest(bounds=bounds), patch.object(
                layout, 'QFontMetricsF', return_value=metrics,
            ), patch.object(layout.QRawFont, 'fromFont', return_value=raw_font):
                layout.clear_font_metrics_cache()
                metrics.boundingRect.return_value = bounds
                tight, _ = layout.get_punc_rect('木fg', 'Sans Serif', 30, QFont.Weight.Normal, False)
                self.assertEqual(tight, expected)

    def test_missing_reference_glyphs_do_not_clip_horizontal_text(self) -> None:
        # An open symbol font exercises valid fallback ink from another face;
        # no proprietary legacy Korean font or synthetic font dependency.
        font = QFont('Noto Sans Symbols2')
        font.setPointSizeF(97.5)
        raw = QRawFont.fromFont(font)
        if not raw.isValid() or 'Symbols' not in raw.familyName():
            self.skipTest('Noto Sans Symbols 2 is required')
        self.assertFalse(any(raw.supportsCharacter(ord(c)) for c in '木fg'))
        self.assertTrue(raw.supportsCharacter(ord('☝')))
        metrics = QFontMetricsF(font)
        tight, bounds = layout.get_punc_rect('木fg', font.family(), 97.5, font.weight(), False)
        self.assertEqual(tight.top(), -metrics.ascent())
        self.assertEqual(tight.height(), metrics.ascent() + metrics.descent())
        self.assertEqual(bounds, metrics.boundingRect('木fg'))
        for alignment in (0, 1, 2):
            with self.subTest(alignment=alignment):
                item = TextBlkItem(TextBlock(
                    xyxy=[0, 0, 500, 800], _bounding_rect=[0, 0, 500, 800],
                    translation='☝\ufe0e\n☝\ufe0e',
                    fontformat={'font_family': font.family(), 'font_size': 130,
                                'alignment': alignment},
                ), 0)
                try:
                    saved = item.toHtml(), item.document().availableUndoSteps()
                    for _ in range(2):
                        item.refresh_font_metrics()
                        block = item.document().firstBlock()
                        while block.isValid():
                            line = block.layout().lineAt(0)
                            ink = glyph_geometry(
                                line, line.textStart(), line.textLength(),
                                QPointF(), QTransform(), 0.0,
                            ).bounds
                            self.assertFalse(ink.isEmpty())
                            self.assertGreaterEqual(ink.top(), 0)
                            block = block.next()
                    self.assertEqual((item.toHtml(), item.document().availableUndoSteps()), saved)
                finally:
                    item.deleteLater()


if __name__ == '__main__':
    unittest.main()
