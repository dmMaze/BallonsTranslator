import os
import unittest
from unittest.mock import patch

import cv2

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QImage, QPainter
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.ui.text_engine.formatting.panel import (
    BoldToolButton,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.fontformat import FontFormat, FontWeight
from ballontranslator.utils.textblock import TextBlock


class SyntheticBoldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _render(
        offset_ratio: float,
        mode: str = 'horizontal',
    ) -> tuple[tuple[int, int, int, int], float]:
        block = TextBlock([0, 0, 180, 100])
        block._bounding_rect = [0, 0, 180, 100]
        block.translation = 'HH'
        block.fontformat.font_size = 48
        block.fontformat.letter_spacing = 1.0
        if mode == 'horizontal':
            block.fontformat.synthetic_bold_offset = [offset_ratio, 0.0]
        elif mode == 'vertical':
            block.fontformat.synthetic_bold_offset = [0.0, offset_ratio]
        else:
            block.fontformat.synthetic_bold_offset = [
                offset_ratio, offset_ratio
            ]
        block.fontformat.synthetic_bold = offset_ratio > 0.0
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)

        image = QImage(
            220,
            140,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        scene.render(
            painter,
            QRectF(0, 0, 220, 140),
            QRectF(-20, -20, 220, 140),
        )
        painter.end()

        points = [
            (x, y)
            for y in range(image.height())
            for x in range(image.width())
            if QColor.fromRgba(image.pixel(x, y)).alpha() > 20
        ]
        bounds = (
            min(x for x, _y in points),
            max(x for x, _y in points),
            min(y for _x, y in points),
            max(y for _x, y in points),
        )
        return bounds, item.padding()

    @staticmethod
    def _render_outline_bounds(
        offsets: tuple[float, float],
    ) -> tuple[int, int, int, int]:
        """Return the red decoration outline around synthetic-bold ink."""
        block = TextBlock([0, 0, 180, 100])
        block._bounding_rect = [0, 0, 180, 100]
        block.translation = 'HH'
        block.fontformat.font_size = 48
        block.fontformat.letter_spacing = 1.0
        block.fontformat.frgb = [255, 255, 255]
        block.fontformat.srgb = [255, 0, 0]
        block.fontformat.stroke_width = 0.08
        block.fontformat.synthetic_bold = any(value > 0.0 for value in offsets)
        block.fontformat.synthetic_bold_offset = list(offsets)
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)

        image = QImage(
            220,
            140,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        scene.render(
            painter,
            QRectF(0, 0, 220, 140),
            QRectF(-20, -20, 220, 140),
        )
        painter.end()

        outline_points = []
        for y in range(image.height()):
            for x in range(image.width()):
                color = QColor.fromRgba(image.pixel(x, y))
                if (
                    color.alpha() > 20
                    and color.red() > 150
                    and color.green() < 100
                    and color.blue() < 100
                ):
                    outline_points.append((x, y))
        return (
            min(x for x, _y in outline_points),
            max(x for x, _y in outline_points),
            min(y for _x, y in outline_points),
            max(y for _x, y in outline_points),
        )

    def test_horizontal_offset_widens_ink_without_increasing_height(self) -> None:
        regular, regular_padding = self._render(0.0)
        emboldened, emboldened_padding = self._render(0.05)

        self.assertLess(emboldened[0], regular[0])
        self.assertGreater(emboldened[1], regular[1])
        self.assertEqual(emboldened[2:], regular[2:])
        self.assertEqual(regular_padding, 0.0)
        self.assertGreater(emboldened_padding, 0.0)

    def test_offset_is_clamped_to_the_supported_range(self) -> None:
        self.assertEqual(
            TextBlock([0, 0, 1, 1]).fontformat.synthetic_bold_offset,
            [0.01, 0.01],
        )
        font_format = FontFormat(synthetic_bold_offset=[9.0, -1.0])
        self.assertEqual(font_format.synthetic_bold_offset, [0.2, 0.0])

    def test_equal_offsets_serialize_as_one_value(self) -> None:
        uniform = FontFormat(
            synthetic_bold=True,
            synthetic_bold_offset=[0.01, 0.01],
        )
        directional = FontFormat(synthetic_bold_offset=[0.02, 0.01])

        self.assertTrue(uniform.to_serializable_dict()['synthetic_bold'])
        self.assertEqual(
            uniform.to_serializable_dict()['synthetic_bold_offset'], 0.01
        )
        self.assertEqual(
            directional.to_serializable_dict()['synthetic_bold_offset'],
            [0.02, 0.01],
        )

    def test_default_offset_is_one_percent_and_disabled(self) -> None:
        font_format = FontFormat()

        self.assertFalse(font_format.synthetic_bold)
        self.assertEqual(font_format.synthetic_bold_offset, [0.01, 0.01])

    def test_uniform_offset_expands_vertical_ink_too(self) -> None:
        regular, _padding = self._render(0.0, 'uniform')
        emboldened, _padding = self._render(0.05, 'uniform')

        regular_height = regular[3] - regular[2]
        emboldened_height = emboldened[3] - emboldened[2]
        self.assertGreater(emboldened_height, regular_height)

    def test_vertical_offset_increases_height_without_widening_ink(self) -> None:
        regular, _padding = self._render(0.0, 'vertical')
        emboldened, _padding = self._render(0.05, 'vertical')

        regular_width = regular[1] - regular[0]
        emboldened_width = emboldened[1] - emboldened[0]
        self.assertEqual(emboldened_width, regular_width)
        self.assertLess(emboldened[2], regular[2])
        self.assertGreater(emboldened[3], regular[3])

    def test_anisotropic_source_captures_the_text_layout_once(self) -> None:
        block = TextBlock([0, 0, 180, 100])
        block._bounding_rect = [0, 0, 180, 100]
        block.translation = 'HH'
        block.fontformat.font_size = 48
        block.fontformat.synthetic_bold = True
        block.fontformat.synthetic_bold_offset = [0.2, 0.0]
        item = TextBlkItem(block, 0)
        renderer = item.effect_renderer
        renderer.release_caches()
        image = QImage(
            220, 140, QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)

        try:
            with patch.object(
                renderer,
                '_paint_live_layout',
                wraps=renderer._paint_live_layout,
            ) as paint_layout, patch.object(
                renderer,
                '_capture_effect_source',
                wraps=renderer._capture_effect_source,
            ) as capture_source:
                self.assertTrue(renderer._draw_cached_synthetic_bold(painter))
                self.assertTrue(renderer._draw_cached_synthetic_bold(painter))
        finally:
            painter.end()

        self.assertEqual(paint_layout.call_count, 1)
        self.assertEqual(capture_source.call_count, 1)

    def test_outline_wraps_the_final_uniformly_emboldened_ink(self) -> None:
        regular_outline = self._render_outline_bounds((0.0, 0.0))
        emboldened_outline = self._render_outline_bounds((0.05, 0.05))

        self.assertLess(emboldened_outline[0], regular_outline[0])
        self.assertGreater(emboldened_outline[1], regular_outline[1])
        self.assertLess(emboldened_outline[2], regular_outline[2])
        self.assertGreater(emboldened_outline[3], regular_outline[3])

    def test_outline_wraps_horizontal_only_emboldened_ink(self) -> None:
        regular_outline = self._render_outline_bounds((0.0, 0.0))
        emboldened_outline = self._render_outline_bounds((0.05, 0.0))

        self.assertLess(emboldened_outline[0], regular_outline[0])
        self.assertGreater(emboldened_outline[1], regular_outline[1])
        # Padding can shift antialias coverage by one raster row even though
        # every anisotropic sample has a zero Y translation.
        self.assertLessEqual(
            abs(emboldened_outline[2] - regular_outline[2]), 1
        )
        self.assertLessEqual(
            abs(emboldened_outline[3] - regular_outline[3]), 1
        )

    def test_emboldened_stroke_uses_one_render_and_one_dilation(self) -> None:
        block = TextBlock([0, 0, 180, 100])
        block._bounding_rect = [0, 0, 180, 100]
        block.translation = 'HH'
        block.fontformat.font_size = 48
        block.fontformat.stroke_width = 0.08
        block.fontformat.synthetic_bold = True
        block.fontformat.synthetic_bold_offset = [0.05, 0.05]
        item = TextBlkItem(block, 0)
        renderer = item.effect_renderer
        rect = renderer.boundingRect()
        canonical = renderer._cached_effect_source(
            rect, 1.0, needs_alpha=True
        )[1]
        renderer.release_caches()

        with patch.object(
            renderer,
            '_paint_stroke_core',
            wraps=renderer._paint_stroke_core,
        ) as paint_stroke_core, patch(
            'ballontranslator.ui.text_engine.effects.renderer.cv2.dilate',
            wraps=cv2.dilate,
        ) as dilate:
            renderer._positioned_stroke_coverage(
                rect,
                1.0,
                renderer._current_stroke(),
                canonical,
            )

        self.assertEqual(paint_stroke_core.call_count, 1)
        self.assertEqual(dilate.call_count, 1)

    def test_xy_offsets_are_edited_independently(self) -> None:
        button = BoldToolButton()
        offsets = []
        button.synthetic_bold_offset_changed.connect(offsets.append)

        button.synthetic_bold_x_box.setValue(5)
        button.synthetic_bold_x_box.param_changed.emit(
            'synthetic_bold_x_offset', 5
        )
        button.synthetic_bold_y_box.setValue(2)
        button.synthetic_bold_y_box.param_changed.emit(
            'synthetic_bold_y_offset', 2
        )

        self.assertEqual(button.synthetic_bold_x_box.value(), 5)
        self.assertEqual(button.synthetic_bold_y_box.value(), 2)
        self.assertEqual(offsets[-1], (0.05, 0.02))

    def test_bold_button_reflects_offsets_without_changing_them(self) -> None:
        button = BoldToolButton()
        offsets = []
        button.synthetic_bold_offset_changed.connect(offsets.append)
        button.set_synthetic_bold(True, (0.01, 0.02))

        self.assertTrue(button.isChecked())
        self.assertEqual(button.synthetic_bold_x_box.value(), 1)
        self.assertEqual(button.synthetic_bold_y_box.value(), 2)
        self.assertEqual(offsets, [])

    def test_bold_button_applies_and_clears_the_current_offsets(self) -> None:
        button = BoldToolButton()
        enabled = []
        button.synthetic_bold_changed.connect(enabled.append)

        button.click()
        button.click()

        self.assertEqual(enabled, [True, False])


if __name__ == '__main__':
    unittest.main()
