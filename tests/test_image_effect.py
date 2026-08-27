import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image
from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QStyleOptionGraphicsItem,
)

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.raster import (
    EFFECT_RASTER_GUARD,
    EffectRasterAllocationError,
    EffectRasterPlan,
)
from ballontranslator.utils.fontformat import TextTransformStack
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.raster_assets import RasterAssetRef
from ballontranslator.utils.text_effects import (
    FilterEffect,
    GlowEffect,
    ImageEffect,
    StrokeEffect,
    TextEffectStack,
)
from ballontranslator.utils.textblock import TextBlock


class ImageEffectRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _item(
        stack: TextEffectStack,
        text: str = 'Image effect',
        *,
        vertical: bool = False,
    ) -> TextBlkItem:
        block = TextBlock([0, 0, 160, 100])
        block._bounding_rect = [0, 0, 160, 100]
        block.translation = text
        block.vertical = vertical
        block.fontformat.frgb = [230, 20, 30]
        block.fontformat.text_effects = stack
        return TextBlkItem(block, 1)

    @staticmethod
    def _attach(item: TextBlkItem, project: ProjImgTrans) -> QGraphicsScene:
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        scene.addItem(item)
        item.effect_renderer.project_assets_changed()
        return scene

    @staticmethod
    def _asset(
        project: ProjImgTrans,
        directory: str,
        name: str,
        rgba: np.ndarray,
    ):
        path = os.path.join(directory, name)
        Image.fromarray(rgba, 'RGBA').save(path)
        return project.import_raster_asset(path)

    def test_image_placements_use_source_over_and_destination_over(self):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        rect = renderer.logical_unpadded_rect()
        width = max(1, int(np.ceil(rect.width())))
        height = max(1, int(np.ceil(rect.height())))
        blue = np.full((3, 5, 4), (20, 60, 230, 255), dtype=np.uint8)
        asset = RasterAssetRef('assets/' + 'a' * 64 + '.png')

        results = {}
        for mode in ('foreground', 'background'):
            target = QPixmap(width, height)
            target.fill(QColor(230, 20, 30, 255))
            painter = QPainter(target)
            renderer._prepare_effect_surface_painter(painter, 1.0)
            painter.translate(-rect.topLeft())
            renderer._paint_image_effect(
                painter,
                ImageEffect(asset, mode=mode),
                blue,
                QRectF(rect),
                1.0,
            )
            painter.end()
            results[mode] = pixmap2ndarray(target, keep_alpha=True)

        np.testing.assert_array_equal(
            results['foreground'][height // 2, width // 2],
            (20, 60, 230, 255),
        )
        np.testing.assert_array_equal(
            results['background'][height // 2, width // 2],
            (230, 20, 30, 255),
        )

    def test_all_image_prefix_nodes_remain_in_application_order(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            first = self._asset(
                project,
                directory,
                'first.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            second = self._asset(
                project,
                directory,
                'second.png',
                np.full((2, 2, 4), (30, 220, 60, 255), np.uint8),
            )
            # Raw order is topmost-first; application order is reversed.
            stack = TextEffectStack(effects=(
                FilterEffect('builtin:noise'),
                ImageEffect(second, mode='foreground'),
                FilterEffect('builtin:noise'),
                ImageEffect(first, mode='foreground'),
                StrokeEffect(),
            ))
            item = self._item(stack)
            scene = self._attach(item, project)

            nodes = item.effect_renderer._ordered_surface_nodes()

            self.assertEqual(
                tuple(index for index, _effect in nodes),
                (4, 3, 2, 1, 0),
            )
            scene.removeItem(item)

    def test_image_keeps_canonical_and_generated_rasterization(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'foreground.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(
                ImageEffect(asset, mode='foreground'),
                StrokeEffect(width=0.4),
            )))
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            renderer.release_caches()

            with patch.object(
                renderer,
                '_capture_effect_source',
                wraps=renderer._capture_effect_source,
            ) as capture:
                rendered = renderer._render_effect_surface(
                    renderer.boundingRect(), 1.0
                )

            self.assertFalse(rendered.isNull())
            self.assertGreater(capture.call_count, 0)
            scene.removeItem(item)

    def test_image_prefix_cache_tracks_nodes_before_bottom_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'foreground.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            top_filter = FilterEffect('builtin:noise')
            retained_stroke = StrokeEffect(width=0.2)
            image = ImageEffect(asset, mode='foreground')
            lower_filter = FilterEffect('builtin:noise')
            lower_stroke = StrokeEffect(width=0.4)
            stack = TextEffectStack(effects=(
                top_filter,
                retained_stroke,
                image,
                lower_filter,
                lower_stroke,
            ))
            item = self._item(stack)
            scene = self._attach(item, project)
            renderer = item.effect_renderer

            def prefix_key() -> tuple:
                return renderer._effect_cache_key_before_bottom_filter(
                    renderer._effect_cache_input_key(),
                    renderer._ordered_surface_nodes(),
                )[0]

            initial = prefix_key()
            item.set_text_effects(replace(
                stack,
                effects=(
                    top_filter,
                    retained_stroke,
                    image,
                    lower_filter,
                    replace(lower_stroke, width=0.6),
                ),
            ))
            self.assertNotEqual(prefix_key(), initial)

            item.set_text_effects(replace(
                stack,
                effects=(
                    top_filter,
                    replace(retained_stroke, width=0.3),
                    image,
                    lower_filter,
                    lower_stroke,
                ),
            ))
            self.assertEqual(prefix_key(), initial)
            scene.removeItem(item)

    def test_strict_image_resolves_missing_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            foreground = self._asset(
                project,
                directory,
                'foreground.png',
                np.full((2, 2, 4), (30, 220, 60, 255), np.uint8),
            )
            missing = RasterAssetRef(
                'assets/' + 'f' * 64 + '.png', 'missing.png'
            )
            item = self._item(TextEffectStack(effects=(
                ImageEffect(foreground, mode='foreground'),
                ImageEffect(missing, mode='foreground'),
            )))
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            renderer.set_export_effect_render(True)
            try:
                with patch.object(
                    renderer,
                    '_project_raster',
                    wraps=renderer._project_raster,
                ) as project_raster:
                    with self.assertRaises(EffectRasterAllocationError):
                        renderer._render_effect_surface(
                            renderer.boundingRect(), 1.0
                        )
            finally:
                renderer.set_export_effect_render(False)

            self.assertEqual(project_raster.call_count, 1)
            self.assertEqual(project_raster.call_args.args[0], missing)
            scene.removeItem(item)

    def test_strict_canvas_export_keeps_filter_before_image(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            project.inpainted_array = np.zeros((120, 180, 3), np.uint8)
            asset = self._asset(
                project,
                directory,
                'foreground.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(
                ImageEffect(asset, mode='foreground'),
                FilterEffect('builtin:noise'),
            )), text='')
            canvas = Canvas()
            canvas.imgtrans_proj = project
            canvas.baseLayer.setRect(QRectF(0, 0, 180, 120))
            item.setParentItem(canvas.textLayer)
            item.effect_renderer.project_assets_changed()
            try:
                rendered = canvas.render_result_img()
                self.assertFalse(rendered.isNull())
                self.assertIsNone(item.export_effect_error)
            finally:
                canvas.deleteLater()
                self.app.processEvents()

    def test_image_preserves_generated_padding_after_project_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'foreground.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(
                ImageEffect(asset, mode='foreground'),
                GlowEffect(size=2.0, spread=1.0),
                StrokeEffect(width=3.0),
            )))
            plain = self._item(TextEffectStack())
            scene = self._attach(item, project)

            self.assertGreater(item.padding(), 0.0)
            self.assertGreater(
                item.boundingRect().width(), plain.boundingRect().width()
            )
            self.assertTrue(item.effect_renderer._effect_flags()[0])
            self.assertGreater(
                item.effect_renderer._effect_tile_overlap(),
                EFFECT_RASTER_GUARD,
            )
            scene.removeItem(item)

    def test_distorted_glow_uses_canonical_stroke_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'glow-source.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )

            def make_item(width: float):
                item = self._item(TextEffectStack(effects=(
                    GlowEffect(size=0.25, spread=0.1),
                    ImageEffect(asset, mode='foreground'),
                    StrokeEffect(width=width, position='outside'),
                )))
                item.set_text_transform(TextTransformStack(
                    glyph_slant_angle=18.0
                ))
                scene = self._attach(item, project)
                return item, scene

            narrow, narrow_scene = make_item(0.1)
            wide, wide_scene = make_item(1.2)
            self.assertGreater(wide.padding(), narrow.padding())
            self.assertGreater(
                wide.boundingRect().width(), narrow.boundingRect().width()
            )

            renderer = wide.effect_renderer
            pixels = pixmap2ndarray(
                renderer._render_effect_surface(
                    renderer.boundingRect(), 1.0
                ),
                keep_alpha=True,
            )
            border = np.concatenate((
                pixels[0, :, 3], pixels[-1, :, 3],
                pixels[:, 0, 3], pixels[:, -1, 3],
            ))
            self.assertEqual(np.count_nonzero(border), 0)
            narrow_scene.removeItem(narrow)
            wide_scene.removeItem(wide)

    def test_distorted_image_then_expanding_filter_uses_logical_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'blur-source.png',
                np.full((3, 5, 4), (20, 60, 230, 255), np.uint8),
            )
            stack = TextEffectStack(effects=(
                FilterEffect(
                    'builtin:gaussian_blur', params={'radius': 6.0}
                ),
                ImageEffect(asset, mode='foreground'),
            ))
            for vertical in (False, True):
                for text in ('Image effect', ''):
                    with self.subTest(vertical=vertical, empty=not text):
                        item = self._item(
                            stack, text=text, vertical=vertical
                        )
                        item.set_text_transform(TextTransformStack(
                            glyph_slant_angle=18.0
                        ))
                        scene = self._attach(item, project)
                        renderer = item.effect_renderer
                        self.assertTrue(renderer._has_layout_distortion())
                        self.assertGreaterEqual(item.padding(), 6.0)
                        self.assertTrue(renderer.boundingRect().contains(
                            renderer.logical_unpadded_rect().adjusted(
                                -6.0, -6.0, 6.0, 6.0
                            )
                        ))
                        pixels = pixmap2ndarray(
                            renderer._render_effect_surface(
                                renderer.boundingRect(), 1.0
                            ),
                            keep_alpha=True,
                        )
                        logical = renderer.logical_unpadded_rect()
                        bounds = renderer.boundingRect()
                        left = max(0, int(logical.left() - bounds.left()))
                        top = max(0, int(logical.top() - bounds.top()))
                        right = min(
                            pixels.shape[1],
                            int(np.ceil(logical.right() - bounds.left())),
                        )
                        bottom = min(
                            pixels.shape[0],
                            int(np.ceil(logical.bottom() - bounds.top())),
                        )
                        outside = np.ones(pixels.shape[:2], dtype=bool)
                        outside[top:bottom, left:right] = False
                        self.assertGreater(np.count_nonzero(
                            (pixels[:, :, 3] > 0) & outside
                        ), 0)
                        scene.removeItem(item)

    def test_missing_image_does_not_expand_distorted_filter_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            missing = RasterAssetRef(
                'assets/' + 'f' * 64 + '.png', 'missing.png'
            )
            filter_effect = FilterEffect(
                'builtin:gaussian_blur', params={'radius': 6.0}
            )
            for vertical in (False, True):
                for text in ('Image effect', ''):
                    with self.subTest(vertical=vertical, empty=not text):
                        filtered = self._item(
                            TextEffectStack(effects=(filter_effect,)),
                            text=text,
                            vertical=vertical,
                        )
                        missing_image = self._item(
                            TextEffectStack(effects=(
                                filter_effect,
                                ImageEffect(missing, mode='foreground'),
                            )),
                            text=text,
                            vertical=vertical,
                        )
                        transform = TextTransformStack(
                            glyph_slant_angle=18.0
                        )
                        filtered.set_text_transform(transform)
                        missing_image.set_text_transform(transform)
                        filtered_scene = self._attach(filtered, project)
                        missing_scene = self._attach(missing_image, project)
                        self.assertEqual(
                            missing_image.padding(), filtered.padding()
                        )
                        self.assertEqual(
                            missing_image.boundingRect(),
                            filtered.boundingRect(),
                        )
                        if not text:
                            self.assertEqual(missing_image.padding(), 0.0)
                        filtered_scene.removeItem(filtered)
                        missing_scene.removeItem(missing_image)

    def test_sole_missing_image_is_an_interactive_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            missing = RasterAssetRef(
                'assets/' + 'e' * 64 + '.png', 'missing.png'
            )
            item = self._item(TextEffectStack(effects=(
                ImageEffect(missing, mode='foreground'),
            )))
            scene = self._attach(item, project)
            renderer = item.effect_renderer

            self.assertEqual(
                renderer._ordered_surface_nodes(strict_assets=False), ()
            )
            self.assertEqual(renderer._effect_flags(), (False, False))
            self.assertFalse(renderer._renders_completed_foreground())
            self.assertFalse(renderer.requires_no_item_cache())
            item.refresh_cache_policy()
            self.assertEqual(
                item.cacheMode(),
                QGraphicsItem.CacheMode.DeviceCoordinateCache,
            )
            renderer.repaint_background()
            self.assertIsNone(renderer._peek_raster_state())
            scene.removeItem(item)

    def test_raster_memory_failure_stays_inside_callback_and_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            project.inpainted_array = np.zeros((120, 180, 3), np.uint8)
            asset = self._asset(
                project,
                directory,
                'memory.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(ImageEffect(asset),)))
            scene = self._attach(item, project)
            image = QImage(
                220, 160, QImage.Format.Format_ARGB32_Premultiplied
            )
            image.fill(QColor(0, 0, 0, 0))
            with patch.object(
                project, 'load_raster_asset', side_effect=MemoryError('oom')
            ):
                item.effect_renderer.project_assets_changed()
                painter = QPainter(image)
                try:
                    scene.render(painter)
                finally:
                    painter.end()
            scene.removeItem(item)

            canvas = Canvas()
            canvas.imgtrans_proj = project
            canvas.baseLayer.setRect(QRectF(0, 0, 180, 120))
            strict_item = self._item(TextEffectStack(effects=(
                ImageEffect(asset),
            )))
            strict_item.setParentItem(canvas.textLayer)
            strict_item.effect_renderer.project_assets_changed()
            try:
                with patch.object(
                    project,
                    'load_raster_asset',
                    side_effect=MemoryError('strict oom'),
                ):
                    with self.assertRaises(EffectRasterAllocationError):
                        canvas.render_result_img()
            finally:
                canvas.deleteLater()
                self.app.processEvents()

    def test_settled_image_paint_resolves_project_raster_once(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'settled.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(ImageEffect(asset),)))
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            renderer.repaint_background()
            bounds = renderer.boundingRect()
            target = QImage(
                max(1, int(np.ceil(bounds.width()))),
                max(1, int(np.ceil(bounds.height()))),
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            target.fill(QColor(0, 0, 0, 0))
            option = QStyleOptionGraphicsItem()
            option.exposedRect = bounds
            painter = QPainter(target)
            painter.translate(-bounds.topLeft())
            try:
                with patch.object(
                    renderer,
                    '_project_raster',
                    wraps=renderer._project_raster,
                ) as project_raster:
                    renderer.paint_item(
                        painter, option, None, lambda *_args: None
                    )
                self.assertEqual(project_raster.call_count, 1)
            finally:
                painter.end()
                scene.removeItem(item)

    def test_repaint_and_asset_refresh_resolve_project_raster_once(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'repaint-once.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(ImageEffect(asset),)))
            scene = self._attach(item, project)
            renderer = item.effect_renderer

            for action in (
                renderer.repaint_background,
                renderer.project_assets_changed,
            ):
                with self.subTest(action=action.__name__), patch.object(
                    renderer,
                    '_project_raster',
                    wraps=renderer._project_raster,
                ) as project_raster:
                    action()
                self.assertEqual(project_raster.call_count, 1)
            scene.removeItem(item)

    def test_suppressed_repaint_does_not_resolve_project_raster(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'suppressed.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(ImageEffect(asset),)))
            scene = self._attach(item, project)
            renderer = item.effect_renderer

            for state in ('repainting', 'reshaping', 'pre_editing'):
                with self.subTest(state=state):
                    setattr(item, state, True)
                    try:
                        with patch.object(
                            renderer,
                            '_project_raster',
                            wraps=renderer._project_raster,
                        ) as project_raster:
                            renderer.repaint_background()
                        self.assertEqual(project_raster.call_count, 0)
                    finally:
                        setattr(item, state, False)
            scene.removeItem(item)

    def test_image_mode_preview_resolves_project_raster_once(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'preview-once.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(ImageEffect(asset),)))
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            renderer.set_faster_preview(True)
            preview = TextEffectStack(effects=(
                ImageEffect(asset, mode='background'),
            ))

            with patch.object(
                renderer,
                '_project_raster',
                wraps=renderer._project_raster,
            ) as project_raster:
                self.assertTrue(item.set_text_effects(preview, preview=True))
            self.assertEqual(project_raster.call_count, 1)
            scene.removeItem(item)

    def test_strict_full_and_tiled_paint_resolve_project_raster_once(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'strict-once.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(ImageEffect(asset),)))
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            bounds = renderer.boundingRect()
            option = QStyleOptionGraphicsItem()
            option.exposedRect = bounds

            for tiled in (False, True):
                with self.subTest(tiled=tiled):
                    target = QImage(
                        max(1, int(np.ceil(bounds.width()))),
                        max(1, int(np.ceil(bounds.height()))),
                        QImage.Format.Format_ARGB32_Premultiplied,
                    )
                    target.fill(QColor(0, 0, 0, 0))
                    renderer.set_export_effect_render(True)
                    renderer.force_tiles = tiled
                    painter = QPainter(target)
                    painter.translate(-bounds.topLeft())
                    try:
                        with patch.object(
                            renderer,
                            '_project_raster',
                            wraps=renderer._project_raster,
                        ) as project_raster:
                            renderer.paint_item(
                                painter,
                                option,
                                None,
                                lambda *_args: None,
                            )
                        self.assertEqual(project_raster.call_count, 1)
                        self.assertIsNone(renderer.export_error)
                    finally:
                        painter.end()
                        renderer.set_export_effect_render(False)
            scene.removeItem(item)

    def test_tiled_images_resolve_one_shared_asset_once(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'shared.png',
                np.full((3, 5, 4), (20, 60, 230, 180), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(
                ImageEffect(asset, mode='foreground'),
                ImageEffect(asset, mode='background'),
            )))
            scene = self._attach(item, project)
            renderer = item.effect_renderer
            bounds = renderer.boundingRect()
            target = renderer._new_effect_pixmap(1.0, bounds)
            painter = QPainter(target)
            painter.translate(-bounds.topLeft())
            renderer.tile_cache.clear()
            renderer._raster_state().pre_mask_cache.clear()
            renderer._raster_state().pre_filter_cache.clear()
            try:
                with patch.object(
                    renderer,
                    '_project_raster',
                    wraps=renderer._project_raster,
                ) as project_raster:
                    renderer._draw_tiled_effects(
                        painter,
                        EffectRasterPlan('tiles', 1.0, 0, 0, 64),
                        bounds,
                    )
                self.assertEqual(project_raster.call_count, 1)
            finally:
                painter.end()
                scene.removeItem(item)

    def test_image_is_suppressed_only_during_interactive_text_editing(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._asset(
                project,
                directory,
                'editing.png',
                np.full((2, 2, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(ImageEffect(asset),)))
            scene = self._attach(item, project)
            view = QGraphicsView(scene)
            view.show()
            self.assertTrue(item.effect_renderer._ordered_surface_nodes())

            item.startEdit()
            view.setFocus()
            item.setFocus()
            self.app.processEvents()
            self.assertTrue(item.isEditing())
            self.assertFalse(item.effect_renderer._ordered_surface_nodes())
            self.assertTrue(item.effect_renderer.has_raster_effects())

            item.effect_renderer.set_export_effect_render(True)
            try:
                self.assertTrue(item.effect_renderer._ordered_surface_nodes())
            finally:
                item.effect_renderer.set_export_effect_render(False)
            scene.removeItem(item)


if __name__ == '__main__':
    unittest.main()
