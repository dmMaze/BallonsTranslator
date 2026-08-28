import os
import io
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image
from qtpy.QtCore import QCoreApplication, QEvent, QRectF, QTranslator
from qtpy.QtTest import QSignalSpy
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.ui.text_engine.effects.image_generation import (
    ImageGenerationBackend,
    ImageGenerationController,
    ImageGenerationRequest,
    _LIVE_IMAGE_GENERATION_JOBS,
    _encode_generated_png,
    _shutdown_live_image_generation_jobs,
    prepare_image_generation_context,
    validate_logical_crop,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.raster import (
    EffectRasterAllocationError,
)
from ballontranslator.utils.fontformat import (
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.text_effects import (
    HollowEffect,
    ImageGenerationRecipe,
    StrokeEffect,
    TextEffectStack,
)
from ballontranslator.utils.textblock import TextBlock


class _CropItem:
    def __init__(self, rect: QRectF) -> None:
        self.rect = rect

    def absBoundingRect(self, *, qrect: bool = False):
        return QRectF(self.rect) if qrect else self.rect.getRect()


class _BlockingBackend(ImageGenerationBackend):
    def __init__(self, cooperative: bool) -> None:
        self.cooperative = cooperative
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def generate(self, request, stop_event) -> np.ndarray:
        del request
        self.started.set()
        while not self.release.wait(0.005):
            if self.cooperative and stop_event.is_set():
                raise LLMRequestStopped()
        return np.full((2, 3, 4), (20, 80, 220, 170), np.uint8)

    def close(self) -> None:
        self.closed = True


class _ViewBackend(ImageGenerationBackend):
    def __init__(self) -> None:
        self.pixels = np.full((2, 3, 4), (25, 90, 210, 180), np.uint8)

    def generate(self, request, stop_event) -> np.ndarray:
        del request, stop_event
        return self.pixels[:, :, :]

    def close(self) -> None:
        self.pixels.fill(0)
        raise RuntimeError('cleanup failed')


class ImageGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        self.app.processEvents()
        return bool(predicate())

    def test_backend_boundary_import_does_not_register_llm_inpainter(self):
        code = (
            'import sys; '
            'import ballontranslator.ui.text_engine.effects.image_generation; '
            'assert "ballontranslator.modules.inpaint.inpaint_llm" '
            'not in sys.modules'
        )
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_logical_crop_uses_full_fractional_bounds_without_clamping(self):
        image = np.zeros((20, 30, 3), np.uint8)

        crop = validate_logical_crop(
            _CropItem(QRectF(2.25, 3.5, 4.25, 5.1)), image
        )

        self.assertEqual(
            (crop.left, crop.top, crop.right, crop.bottom),
            (2, 3, 7, 9),
        )
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_logical_crop(
                _CropItem(QRectF(-0.01, 0, 4, 4)), image
            )
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_logical_crop(
                _CropItem(QRectF(28, 2, 2.1, 4)), image
            )

    def test_context_source_inpainted_none_and_plain_lettered(self):
        source = np.full((500, 500, 3), (10, 20, 30), np.uint8)
        inpainted = np.full((500, 500, 3), (210, 220, 230), np.uint8)
        project = SimpleNamespace(
            img_array=source,
            inpainted_array=inpainted,
        )
        block = TextBlock([5, 6, 50, 36])
        block._bounding_rect = [5, 6, 50, 36]
        block.translation = 'Lettered'
        block.fontformat.text_effects = TextEffectStack(effects=(
            StrokeEffect(width=0.8),
            HollowEffect(),
        ))
        item = TextBlkItem(block, 1)

        source_crop = prepare_image_generation_context(
            item, project, 'source'
        )
        inpainted_crop = prepare_image_generation_context(
            item, project, 'inpainted'
        )
        lettered_with_effects = prepare_image_generation_context(
            item, project, 'lettered'
        )
        item.blk.fontformat.text_effects = TextEffectStack()
        lettered_plain = prepare_image_generation_context(
            item, project, 'lettered'
        )

        self.assertTrue(np.all(source_crop == (10, 20, 30)))
        self.assertTrue(np.all(inpainted_crop == (210, 220, 230)))
        np.testing.assert_array_equal(lettered_with_effects, lettered_plain)
        self.assertTrue(np.any(lettered_plain != inpainted_crop))
        self.assertFalse(source_crop.flags.writeable)
        self.assertIsNone(prepare_image_generation_context(
            item,
            SimpleNamespace(img_array=None, inpainted_array=None),
            'none',
        ))
        item.deleteLater()

        vertical_block = TextBlock([70, 6, 50, 80])
        vertical_block._bounding_rect = [70, 6, 50, 80]
        vertical_block.translation = '縦書き'
        vertical_block.vertical = True
        vertical_block.fontformat.text_effects = TextEffectStack(effects=(
            StrokeEffect(width=0.8),
            HollowEffect(),
        ))
        vertical_block.fontformat.text_transform = TextTransformStack((
            SineTextTransform(amplitude_x=0.2),
        ))
        vertical_item = TextBlkItem(vertical_block, 2)

        vertical_styled = prepare_image_generation_context(
            vertical_item, project, 'lettered'
        )
        vertical_item.blk.fontformat.text_effects = TextEffectStack()
        vertical_item.set_text_transform(TextTransformStack())
        vertical_plain = prepare_image_generation_context(
            vertical_item, project, 'lettered'
        )

        np.testing.assert_array_equal(vertical_styled, vertical_plain)
        vertical_base = prepare_image_generation_context(
            vertical_item, project, 'inpainted'
        )
        self.assertTrue(np.any(vertical_plain != vertical_base))
        vertical_item.deleteLater()

    def test_plain_lettered_capture_restores_transient_renderer_state(self):
        block = TextBlock([0, 0, 40, 30])
        block._bounding_rect = [0, 0, 40, 30]
        block.translation = 'A'
        item = TextBlkItem(block, 1)
        renderer = item.effect_renderer
        item.layout.deferred_cursor_position = 17
        renderer.capture_plain_logical_rgba(40, 30, 0, 0)
        self.assertEqual(item.layout.deferred_cursor_position, 17)
        delegate = object()
        stroke = StrokeEffect(width=0.5)
        item.layout.render_delegate = delegate
        renderer._render_stroke = stroke
        item.layout.deferred_cursor_position = 23

        with patch.object(
            renderer, '_paint_live_layout', side_effect=RuntimeError('paint')
        ), self.assertRaisesRegex(RuntimeError, 'paint'):
            renderer.capture_plain_logical_rgba(40, 30, 0, 0)

        self.assertIs(item.layout.render_delegate, delegate)
        self.assertIs(renderer._render_stroke, stroke)
        self.assertEqual(item.layout.deferred_cursor_position, 23)
        item.deleteLater()

    def test_validation_messages_use_the_image_generation_context(self):
        class PrefixTranslator(QTranslator):
            def translate(
                self, context, source_text, disambiguation=None, n=-1
            ):
                del disambiguation, n
                if context == 'ImageGeneration':
                    return 'Localized ' + source_text
                return source_text

        translator = PrefixTranslator()
        self.app.installTranslator(translator)
        item = None
        try:
            with self.assertRaises(ValueError) as caught:
                validate_logical_crop(
                    _CropItem(QRectF(0, 0, 2, 2)), None
                )
            self.assertTrue(str(caught.exception).startswith('Localized '))
            block = TextBlock([0, 0, 20, 20])
            block._bounding_rect = [0, 0, 20, 20]
            block.translation = 'A'
            item = TextBlkItem(block, 1)
            project = SimpleNamespace(
                img_array=np.zeros((30, 30, 3), np.uint8),
                inpainted_array=np.zeros((30, 30, 3), np.uint8),
            )
            with patch.object(
                item.effect_renderer,
                'capture_plain_logical_rgba',
                side_effect=EffectRasterAllocationError('allocation'),
            ), self.assertRaises(ValueError) as caught:
                prepare_image_generation_context(item, project, 'lettered')
            self.assertTrue(str(caught.exception).startswith('Localized '))

            with patch.object(
                item.effect_renderer,
                'capture_plain_logical_rgba',
                side_effect=RuntimeError('programming error'),
            ), self.assertRaisesRegex(RuntimeError, 'programming error'):
                prepare_image_generation_context(item, project, 'lettered')
        finally:
            if item is not None:
                item.deleteLater()
            self.app.removeTranslator(translator)

    def test_stop_discards_result_and_interrupts_cooperative_worker(self):
        controller = ImageGenerationController()
        backend = _BlockingBackend(cooperative=True)
        generated = QSignalSpy(controller.generated)
        states = QSignalSpy(controller.state_changed)
        request = ImageGenerationRequest(ImageGenerationRecipe(), None)

        self.assertTrue(controller.start(4, backend, request))
        self.assertTrue(backend.started.wait(1.0))
        self.assertTrue(controller.stop())
        self.assertTrue(self._wait_until(lambda: not controller.active))

        self.assertEqual(len(generated), 0)
        self.assertTrue(backend.closed)
        self.assertEqual(
            [entry[1] for entry in states],
            ['running', 'stopping', 'idle'],
        )
        controller.deleteLater()

    def test_result_is_encoded_before_backend_cleanup_failure(self):
        controller = ImageGenerationController()
        backend = _ViewBackend()
        generated = QSignalSpy(controller.generated)
        failed = QSignalSpy(controller.failed)
        request = ImageGenerationRequest(ImageGenerationRecipe(), None)

        with patch(
            'ballontranslator.ui.text_engine.effects.image_generation.'
            'LOGGER.warning'
        ) as warning:
            self.assertTrue(controller.start(1, backend, request))
            self.assertTrue(self._wait_until(lambda: not controller.active))

        self.assertEqual(len(failed), 0)
        self.assertEqual(len(generated), 1)
        with Image.open(io.BytesIO(bytes(generated[0][1]))) as image:
            self.assertEqual(image.convert('RGBA').getpixel((0, 0)), (
                25, 90, 210, 180
            ))
        warning.assert_called_once()
        controller.deleteLater()

    def test_generated_image_size_is_rejected_before_png_allocation(self):
        image = np.zeros((2, 2, 4), dtype=np.uint8)
        with patch(
            'ballontranslator.utils.raster_assets.'
            'RASTER_ASSET_MAX_PIXELS',
            3,
        ), patch(
            'ballontranslator.ui.text_engine.effects.image_generation.'
            'Image.fromarray',
        ) as from_array, self.assertRaisesRegex(ValueError, 'pixel limit'):
            _encode_generated_png(image)

        from_array.assert_not_called()

    def test_worker_delivery_is_queued_to_the_qt_thread(self):
        controller = ImageGenerationController()
        backend = _BlockingBackend(cooperative=False)
        generated = QSignalSpy(controller.generated)
        request = ImageGenerationRequest(ImageGenerationRecipe(), None)

        self.assertTrue(controller.start(1, backend, request))
        worker = controller._worker
        self.assertTrue(backend.started.wait(1.0))
        backend.release.set()
        worker._thread.join(2.0)

        self.assertTrue(controller.active)
        self.assertEqual(len(generated), 0)
        self.assertTrue(self._wait_until(lambda: not controller.active))
        self.assertEqual(len(generated), 1)
        controller.deleteLater()

    def test_parent_teardown_does_not_destroy_running_http_thread(self):
        parent = QWidget()
        controller = ImageGenerationController(parent)
        backend = _BlockingBackend(cooperative=False)
        request = ImageGenerationRequest(ImageGenerationRecipe(), None)
        self.assertTrue(controller.start(2, backend, request))
        worker = controller._worker
        self.assertIsNotNone(worker)
        self.assertTrue(backend.started.wait(1.0))

        parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.assertTrue(worker._thread.is_alive())
        self.assertIn(worker, _LIVE_IMAGE_GENERATION_JOBS)

        backend.release.set()
        worker._thread.join(2.0)
        self.assertFalse(worker._thread.is_alive())
        self.assertTrue(self._wait_until(
            lambda: worker not in _LIVE_IMAGE_GENERATION_JOBS
        ))
        self.assertTrue(backend.closed)

    def test_app_shutdown_abandons_blocked_post_without_waiting(self):
        controller = ImageGenerationController()
        backend = _BlockingBackend(cooperative=False)
        request = ImageGenerationRequest(ImageGenerationRecipe(), None)
        self.assertTrue(controller.start(3, backend, request))
        worker = controller._worker
        self.assertTrue(backend.started.wait(1.0))

        started = time.monotonic()
        _shutdown_live_image_generation_jobs()

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(worker._thread.is_alive())
        self.assertTrue(worker._thread.daemon)
        self.assertFalse(backend.closed)
        backend.release.set()
        worker._thread.join(2.0)
        self.assertFalse(worker._thread.is_alive())
        self.assertTrue(self._wait_until(lambda: backend.closed))
        self.assertIn(worker, _LIVE_IMAGE_GENERATION_JOBS)
        worker._release_after_finish(worker.token)
        controller.deleteLater()


if __name__ == '__main__':
    unittest.main()
