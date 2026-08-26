import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image

from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QImage, QPainter, QTextCursor
from qtpy.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.raster import (
    EffectRasterAllocationError,
    EffectRasterPlan,
)
from ballontranslator.utils.fontformat import (
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.rendered_image import RenderedImageLayer
from ballontranslator.utils.text_alpha_mask import (
    AlphaBrushStroke,
    TextAlphaMask,
)
from ballontranslator.utils.text_effects import (
    GlowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
)
from ballontranslator.utils.textblock import TextBlock


class RenderedImageRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _item(
        layer=None,
        *,
        vertical=False,
        stack=TextEffectStack(),
        text='Rendered image source',
    ) -> TextBlkItem:
        block = TextBlock([0, 0, 320, 180], rendered_image=layer)
        block._bounding_rect = [0, 0, 320, 180]
        block.translation = text
        block.vertical = vertical
        block.fontformat.frgb = [230, 20, 30]
        block.fontformat.text_effects = stack
        return TextBlkItem(block, 1)

    @staticmethod
    def _attach(
        item: TextBlkItem, project: ProjImgTrans
    ) -> QGraphicsScene:
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        scene.addItem(item)
        item.effect_renderer.project_assets_changed()
        return scene

    @staticmethod
    def _render(item: TextBlkItem) -> np.ndarray:
        scene = item.scene()
        image = QImage(
            420, 260, QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        scene.render(
            painter,
            QRectF(0, 0, 420, 260),
            QRectF(-30, -30, 420, 260),
        )
        painter.end()
        return pixmap2ndarray(image, keep_alpha=True)

    @staticmethod
    def _import(
        project: ProjImgTrans,
        directory: str,
        name: str,
        pixels: np.ndarray,
    ):
        path = os.path.join(directory, name)
        Image.fromarray(pixels, 'RGBA').save(path)
        return path, project.import_raster_asset(path)

    def test_replace_clears_effect_overflow_and_overlay_is_source_over(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            pixels = np.full((4, 4, 4), (250, 20, 30, 128), np.uint8)
            _source, asset = self._import(
                project, directory, 'half-red.png', pixels
            )
            stack = TextEffectStack(effects=(
                GlowEffect(
                    paint=SolidPaint((0, 255, 0)),
                    size=0.2,
                    spread=0.1,
                ),
                StrokeEffect(
                    width=0.25,
                    paint=SolidPaint((0, 0, 255)),
                ),
            ))
            item = self._item(RenderedImageLayer(asset), stack=stack)
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            bounds = renderer.boundingRect()
            with patch.object(
                renderer,
                '_capture_effect_source',
                wraps=renderer._capture_effect_source,
            ) as capture, patch.object(
                renderer,
                '_generated_effect_pixmap',
                wraps=renderer._generated_effect_pixmap,
            ) as generated:
                replaced = pixmap2ndarray(
                    renderer._render_effect_surface(bounds, 1.0),
                    keep_alpha=True,
                )
            self.assertEqual(capture.call_count, 0)
            self.assertEqual(generated.call_count, 0)
            logical = renderer.logical_unpadded_rect()
            x0 = round(logical.left() - bounds.left())
            y0 = round(logical.top() - bounds.top())
            x1 = round(logical.right() - bounds.left())
            y1 = round(logical.bottom() - bounds.top())
            outside = replaced[..., 3].copy()
            outside[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 0
            self.assertEqual(np.count_nonzero(outside), 0)
            center = replaced[(y0 + y1) // 2, (x0 + x1) // 2]
            self.assertAlmostEqual(int(center[3]), 128, delta=2)
            self.assertGreater(center[0], center[1])

            item.set_rendered_image_layer(
                RenderedImageLayer(asset, mode='overlay')
            )
            overlaid = pixmap2ndarray(
                renderer._render_effect_surface(bounds, 1.0),
                keep_alpha=True,
            )
            self.assertGreater(
                np.count_nonzero(overlaid[..., 3]),
                np.count_nonzero(replaced[..., 3]),
            )
            self.assertGreaterEqual(
                int(overlaid[(y0 + y1) // 2, (x0 + x1) // 2, 3]),
                128,
            )
            scene.removeItem(item)

    def test_transparent_replace_pixels_remain_transparent(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            pixels = np.full((8, 8, 4), (40, 80, 220, 255), np.uint8)
            pixels[:4, :4] = 0
            _source, asset = self._import(
                project, directory, 'alpha.png', pixels
            )
            item = self._item(RenderedImageLayer(asset))
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            bounds = renderer.boundingRect()
            rendered = pixmap2ndarray(
                renderer._render_effect_surface(bounds, 1.0),
                keep_alpha=True,
            )
            logical = renderer.logical_unpadded_rect()
            x = round(logical.left() - bounds.left() + logical.width() * 0.2)
            y = round(logical.top() - bounds.top() + logical.height() * 0.2)
            self.assertEqual(rendered[y, x, 3], 0)
            x = round(logical.left() - bounds.left() + logical.width() * 0.8)
            y = round(logical.top() - bounds.top() + logical.height() * 0.8)
            self.assertGreater(rendered[y, x, 3], 240)
            scene.removeItem(item)

    def test_non_integer_scaling_interpolates_source_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            checker = np.indices((11, 7)).sum(axis=0) % 2
            pixels = np.zeros((11, 7, 4), dtype=np.uint8)
            pixels[checker == 0] = (255, 0, 0, 255)
            pixels[checker == 1] = (0, 0, 255, 255)
            _source, asset = self._import(
                project, directory, 'checker.png', pixels
            )
            item = self._item(RenderedImageLayer(asset))
            scene = self._attach(item, project)
            rendered = pixmap2ndarray(
                item.effect_renderer._render_effect_surface(
                    item.effect_renderer.boundingRect(), 1.0
                ),
                keep_alpha=True,
            )
            visible = rendered[..., 3] == 255
            self.assertTrue(np.any(
                visible
                & (rendered[..., 0] > 20)
                & (rendered[..., 2] > 20)
            ))
            scene.removeItem(item)

    def test_scaling_interpolates_transparent_edges_in_premultiplied_rgba(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            pixels = np.array([[
                [255, 0, 0, 255],
                [0, 0, 255, 0],
            ]], dtype=np.uint8)
            _source, asset = self._import(
                project, directory, 'transparent-edge.png', pixels
            )
            item = self._item(RenderedImageLayer(asset))
            scene = self._attach(item, project)
            rendered = pixmap2ndarray(
                item.effect_renderer._render_effect_surface(
                    item.effect_renderer.boundingRect(), 1.0
                ),
                keep_alpha=True,
            )
            edge = (rendered[..., 3] > 16) & (rendered[..., 3] < 240)
            self.assertGreater(np.count_nonzero(edge), 0)
            self.assertGreater(np.min(rendered[..., 0][edge]), 245)
            self.assertLess(np.max(rendered[..., 2][edge]), 8)
            scene.removeItem(item)

    def test_text_eraser_applies_after_rendered_image(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            pixels = np.full((3, 5, 4), (20, 60, 230, 255), np.uint8)
            _source, asset = self._import(
                project, directory, 'masked.png', pixels
            )
            item = self._item(RenderedImageLayer(asset), text='')
            item.blk.text_alpha_mask = TextAlphaMask(strokes=(
                AlphaBrushStroke('erase', 1000, ((160, 90),)),
            ))
            scene = self._attach(item, project)
            rendered = pixmap2ndarray(
                item.effect_renderer._render_effect_surface(
                    item.effect_renderer.boundingRect(), 1.0
                ),
                keep_alpha=True,
            )
            self.assertEqual(np.count_nonzero(rendered[..., 3]), 0)
            scene.removeItem(item)

    def test_full_tiles_transform_and_asset_change_reuse_source(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            pattern = np.array([
                [[230, 20, 40, 255], [20, 200, 60, 160]],
                [[40, 60, 230, 80], [210, 190, 20, 255]],
            ], dtype=np.uint8)
            _first_path, first = self._import(
                project, directory, 'first.png', pattern
            )
            _second_path, second = self._import(
                project, directory, 'second.png', pattern[::-1].copy()
            )
            item = self._item(
                RenderedImageLayer(first, mode='overlay')
            )
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            bounds = renderer.boundingRect()
            full = renderer._render_effect_surface(bounds, 1.0)
            with patch.object(
                renderer,
                '_capture_effect_source',
                wraps=renderer._capture_effect_source,
            ) as capture:
                item.set_rendered_image_layer(
                    RenderedImageLayer(second, mode='overlay')
                )
                full = renderer._render_effect_surface(bounds, 1.0)
            self.assertEqual(capture.call_count, 0)

            for scale in (1.0, 2.0):
                with self.subTest(scale=scale), patch(
                    'ballontranslator.ui.text_engine.effects.paint.'
                    'premultiply_rgba_in_place',
                    side_effect=AssertionError(
                        'cached project texture was premultiplied per tile'
                    ),
                ):
                    full = renderer._render_effect_surface(bounds, scale)
                    tiled = renderer._new_effect_pixmap(scale, bounds)
                    painter = QPainter(tiled)
                    painter.translate(-bounds.topLeft())
                    renderer.tile_cache.clear()
                    try:
                        renderer._draw_tiled_effects(
                            painter,
                            EffectRasterPlan(
                                'tiles', scale, 0, 0, 96
                            ),
                            bounds,
                        )
                    finally:
                        painter.end()
                    np.testing.assert_array_equal(
                        pixmap2ndarray(full, keep_alpha=True),
                        pixmap2ndarray(tiled, keep_alpha=True),
                    )

            untransformed = self._render(item)
            item.set_text_transform(
                TextTransformStack((SineTextTransform(amplitude_x=0.2),))
            )
            transformed = self._render(item)
            self.assertGreater(np.count_nonzero(transformed[..., 3]), 0)
            self.assertFalse(np.array_equal(untransformed, transformed))
            scene.removeItem(item)

    def test_missing_warm_cache_bypasses_recovers_and_export_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            pixels = np.full((3, 4, 4), (20, 60, 230, 255), np.uint8)
            source_path, asset = self._import(
                project, directory, 'recover.png', pixels
            )
            item = self._item(RenderedImageLayer(asset))
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            textured = self._render(item)
            installed = project.resolve_raster_asset(asset)
            os.unlink(installed)
            renderer.project_assets_changed()
            bypassed = self._render(item)
            visible = bypassed[..., 3] > 160
            self.assertGreater(np.count_nonzero(visible), 0)
            self.assertGreater(
                np.mean(bypassed[..., 0][visible]),
                np.mean(bypassed[..., 2][visible]),
            )
            self.assertFalse(np.array_equal(textured, bypassed))

            renderer.set_export_effect_render(True)
            try:
                with self.assertRaises(EffectRasterAllocationError):
                    renderer._render_effect_surface(
                        renderer.boundingRect(), 1.0
                    )
            finally:
                renderer.set_export_effect_render(False)

            self.assertEqual(project.import_raster_asset(source_path), asset)
            renderer.project_assets_changed()
            recovered = self._render(item)
            np.testing.assert_array_equal(recovered, textured)
            scene.removeItem(item)

    def test_empty_document_modes_render_interactively_and_strictly(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            pixels = np.full((3, 5, 4), (20, 60, 230, 255), np.uint8)
            _source, asset = self._import(
                project, directory, 'empty.png', pixels
            )
            for vertical in (False, True):
                for mode in ('replace', 'overlay'):
                    with self.subTest(vertical=vertical, mode=mode):
                        item = self._item(
                            RenderedImageLayer(asset, mode=mode),
                            vertical=vertical,
                            text='',
                        )
                        scene = self._attach(item, project)
                        renderer = item.effect_renderer
                        renderer.repaint_background()
                        interactive = self._render(item)
                        visible = interactive[..., 3] > 240
                        self.assertGreater(np.count_nonzero(visible), 0)
                        self.assertGreater(
                            np.mean(interactive[..., 2][visible]),
                            np.mean(interactive[..., 0][visible]),
                        )

                        renderer.set_export_effect_render(True)
                        try:
                            strict = pixmap2ndarray(
                                renderer._render_effect_surface(
                                    renderer.boundingRect(), 1.0
                                ),
                                keep_alpha=True,
                            )
                        finally:
                            renderer.set_export_effect_render(False)
                        self.assertGreater(
                            np.count_nonzero(strict[..., 3] > 240), 0
                        )
                        scene.removeItem(item)

    def test_editing_layer_is_still_strictly_selected_for_canvas_export(self):
        for mutation in ('missing', 'digest-mismatch'):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                project = ProjImgTrans()
                project.directory = directory
                project.inpainted_array = np.zeros(
                    (220, 360, 3), dtype=np.uint8
                )
                pixels = np.full(
                    (3, 5, 4), (20, 60, 230, 255), np.uint8
                )
                _source, asset = self._import(
                    project, directory, f'{mutation}.png', pixels
                )
                installed = project.resolve_raster_asset(asset)
                if mutation == 'missing':
                    os.unlink(installed)
                else:
                    with open(installed, 'wb') as corrupted:
                        corrupted.write(b'not the imported raster')

                canvas = Canvas()
                canvas.imgtrans_proj = project
                canvas.baseLayer.setRect(QRectF(0, 0, 360, 220))
                item = self._item(RenderedImageLayer(asset), text='')
                item.setParentItem(canvas.textLayer)
                item.startEdit()
                canvas.editor_index = 1
                canvas.txtblkShapeControl.blk_item = item
                try:
                    self.assertTrue(item.isEditing())
                    self.assertTrue(
                        item.effect_renderer.has_raster_effects()
                    )
                    with self.assertRaises(EffectRasterAllocationError):
                        canvas.render_result_img()
                finally:
                    canvas.deleteLater()
                    self.app.processEvents()

    def test_editing_suppresses_layer_and_restores_for_both_writing_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            pixels = np.full((2, 2, 4), (20, 60, 230, 255), np.uint8)
            _source, asset = self._import(
                project, directory, 'editing.png', pixels
            )
            for vertical in (False, True):
                with self.subTest(vertical=vertical):
                    item = self._item(
                        RenderedImageLayer(asset), vertical=vertical
                    )
                    scene = self._attach(item, project)
                    view = QGraphicsView(scene)
                    view.show()
                    settled = self._render(item)
                    item.startEdit()
                    view.setFocus()
                    item.setFocus()
                    self.app.processEvents()
                    editing_source = self._render(item)
                    source_visible = editing_source[..., 3] > 160
                    self.assertGreater(np.count_nonzero(source_visible), 0)
                    self.assertGreater(
                        np.mean(editing_source[..., 0][source_visible]),
                        np.mean(editing_source[..., 2][source_visible]),
                    )
                    cursor = item.textCursor()
                    cursor.setPosition(0)
                    cursor.setPosition(
                        min(8, len(item.toPlainText())),
                        QTextCursor.MoveMode.KeepAnchor,
                    )
                    item.setTextCursor(cursor)
                    editing = self._render(item)
                    visible = editing[..., 3] > 160
                    self.assertGreater(np.count_nonzero(visible), 0)
                    self.assertFalse(
                        np.array_equal(editing_source, editing)
                    )
                    self.assertFalse(np.array_equal(settled, editing))
                    item.endEdit()
                    restored = self._render(item)
                    np.testing.assert_array_equal(restored, settled)
                    view.close()
                    scene.removeItem(item)


if __name__ == '__main__':
    unittest.main()
