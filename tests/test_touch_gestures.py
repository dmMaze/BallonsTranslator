import os
import sys
import time
import unittest
from types import SimpleNamespace

# Ensure the local repository is tested, not any pre-installed library
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QPoint, QRectF, QEvent, Qt
from qtpy.QtGui import QPen, QMouseEvent, QTouchEvent, QPointingDevice, QEventPoint, QTabletEvent, QWheelEvent, QNativeGestureEvent
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.canvas import Canvas, CustomGV, detect_pointer_device
from ballontranslator.ui.image_edit import ImageEditMode, StrokeImgItem


def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()


class TouchGesturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas()
        self.canvas.imgtrans_proj = SimpleNamespace(
            img_valid=True,
            inpainted_valid=False,
        )
        self.canvas.baseLayer.setRect(QRectF(0, 0, 1000, 1000))
        self.canvas.setSceneRect(0, 0, 1000, 1000)
        self.canvas.gv.resize(600, 600)
        self.canvas.gv.show()
        _APP.processEvents()

    def tearDown(self) -> None:
        self.canvas.deleteLater()
        _APP.processEvents()

    def test_fit_to_screen_scale_and_fit(self) -> None:
        fit_scale = self.canvas.get_fit_to_screen_scale()
        self.assertGreater(fit_scale, 0.0)
        self.assertLess(fit_scale, 1.0)

        self.canvas.scale_factor = 2.5
        self.canvas.fit_to_screen()
        self.assertAlmostEqual(self.canvas.scale_factor, fit_scale, places=4)

    def test_toggle_zoom_fit_or_step(self) -> None:
        fit_scale = self.canvas.get_fit_to_screen_scale()
        self.canvas.fit_to_screen()
        self.assertAlmostEqual(self.canvas.scale_factor, fit_scale, places=4)

        # First double-tap: zoom in 50%
        self.canvas.toggle_zoom_fit_or_step()
        expected_zoomed = fit_scale * 1.5
        self.assertAlmostEqual(self.canvas.scale_factor, expected_zoomed, places=4)

        # Second double-tap: shrink to fit screen
        self.canvas.toggle_zoom_fit_or_step()
        self.assertAlmostEqual(self.canvas.scale_factor, fit_scale, places=4)

    def test_anchor_zoom_accuracy(self) -> None:
        self.canvas.scale_factor = 1.0
        self.canvas.baseLayer.setScale(1.0)
        self.canvas.setSceneRect(0, 0, 1000, 1000)
        self.canvas.gv.horizontalScrollBar().setValue(200)
        self.canvas.gv.verticalScrollBar().setValue(200)
        _APP.processEvents()

        anchor_vp = QPoint(300, 300)
        scene_pos_before = self.canvas.gv.mapToScene(anchor_vp)

        self.canvas.scaleImage(1.5, anchor_pos=anchor_vp)
        _APP.processEvents()

        scene_pos_after = self.canvas.gv.mapToScene(anchor_vp)
        expected_scene_pos = scene_pos_before * 1.5
        drift = abs(scene_pos_after.x() - expected_scene_pos.x()) + abs(scene_pos_after.y() - expected_scene_pos.y())
        self.assertAlmostEqual(drift, 0.0, places=2)

    def test_cancel_active_gestures_and_strokes(self) -> None:
        self.canvas.addStrokeImageItem(QPointF(10, 10), QPen())
        self.assertIsNotNone(self.canvas.stroke_img_item)

        self.canvas.cancel_active_gestures_and_strokes()
        self.assertIsNone(self.canvas.stroke_img_item)
        self.assertEqual(self.canvas._brush_stroke_button, Qt.MouseButton.NoButton)

    def test_multi_touch_cancels_active_stroke(self) -> None:
        self.canvas.addStrokeImageItem(QPointF(20, 20), QPen())
        self.assertIsNotNone(self.canvas.stroke_img_item)

        dev = QPointingDevice()
        pt1 = QEventPoint(0, QEventPoint.State.Pressed, QPointF(100, 100), QPointF(100, 100))
        pt2 = QEventPoint(1, QEventPoint.State.Pressed, QPointF(200, 200), QPointF(200, 200))
        touch_event = QTouchEvent(QEvent.Type.TouchBegin, dev, Qt.KeyboardModifier.NoModifier, [pt1, pt2])

        handled = self.canvas.gv.viewportEvent(touch_event)
        self.assertTrue(handled)
        self.assertIsNone(self.canvas.stroke_img_item)
        self.assertTrue(self.canvas.gv._touch_active)

    def test_two_finger_pinch_and_pan(self) -> None:
        dev = QPointingDevice()
        pt1 = QEventPoint(0, QEventPoint.State.Pressed, QPointF(200, 200), QPointF(200, 200))
        pt2 = QEventPoint(1, QEventPoint.State.Pressed, QPointF(300, 200), QPointF(300, 200))
        ev_begin = QTouchEvent(QEvent.Type.TouchBegin, dev, Qt.KeyboardModifier.NoModifier, [pt1, pt2])
        self.canvas.gv.viewportEvent(ev_begin)

        initial_scale = self.canvas.scale_factor
        initial_h = self.canvas.gv.horizontalScrollBar().value()

        # Spread fingers apart and translate right (distance increases from 100 to 200, center moves +20px right)
        pt1_u = QEventPoint(0, QEventPoint.State.Updated, QPointF(170, 200), QPointF(170, 200))
        pt2_u = QEventPoint(1, QEventPoint.State.Updated, QPointF(370, 200), QPointF(370, 200))
        ev_update = QTouchEvent(QEvent.Type.TouchUpdate, dev, Qt.KeyboardModifier.NoModifier, [pt1_u, pt2_u])
        handled = self.canvas.gv.viewportEvent(ev_update)

        self.assertTrue(handled)
        self.assertGreater(self.canvas.scale_factor, initial_scale)

    def test_two_finger_tap_undo_signal(self) -> None:
        undo_called = False
        def on_undo():
            nonlocal undo_called
            undo_called = True

        self.canvas.undo_requested.connect(on_undo)

        dev = QPointingDevice()
        pt1 = QEventPoint(0, QEventPoint.State.Pressed, QPointF(200, 200), QPointF(200, 200))
        pt2 = QEventPoint(1, QEventPoint.State.Pressed, QPointF(250, 200), QPointF(250, 200))
        ev_begin = QTouchEvent(QEvent.Type.TouchBegin, dev, Qt.KeyboardModifier.NoModifier, [pt1, pt2])
        self.canvas.gv.viewportEvent(ev_begin)

        pt1_end = QEventPoint(0, QEventPoint.State.Released, QPointF(200, 200), QPointF(200, 200))
        pt2_end = QEventPoint(1, QEventPoint.State.Released, QPointF(250, 200), QPointF(250, 200))
        ev_end = QTouchEvent(QEvent.Type.TouchEnd, dev, Qt.KeyboardModifier.NoModifier, [pt1_end, pt2_end])
        self.canvas.gv.viewportEvent(ev_end)

        self.assertTrue(undo_called)

    def test_three_finger_gestures_disabled_for_windows_compatibility(self) -> None:
        next_page = False
        prev_page = False
        redo_called = False
        def on_next():
            nonlocal next_page
            next_page = True
        def on_prev():
            nonlocal prev_page
            prev_page = True
        def on_redo():
            nonlocal redo_called
            redo_called = True

        self.canvas.next_page_requested.connect(on_next)
        self.canvas.prev_page_requested.connect(on_prev)
        self.canvas.redo_requested.connect(on_redo)

        dev = QPointingDevice()
        # 3 fingers touch, swipe left, and release
        pt1 = QEventPoint(0, QEventPoint.State.Pressed, QPointF(300, 200), QPointF(300, 200))
        pt2 = QEventPoint(1, QEventPoint.State.Pressed, QPointF(350, 200), QPointF(350, 200))
        pt3 = QEventPoint(2, QEventPoint.State.Pressed, QPointF(400, 200), QPointF(400, 200))
        ev_begin = QTouchEvent(QEvent.Type.TouchBegin, dev, Qt.KeyboardModifier.NoModifier, [pt1, pt2, pt3])
        self.canvas.gv.viewportEvent(ev_begin)

        pt1_u = QEventPoint(0, QEventPoint.State.Updated, QPointF(150, 200), QPointF(150, 200))
        pt2_u = QEventPoint(1, QEventPoint.State.Updated, QPointF(200, 200), QPointF(200, 200))
        pt3_u = QEventPoint(2, QEventPoint.State.Updated, QPointF(250, 200), QPointF(250, 200))
        ev_update = QTouchEvent(QEvent.Type.TouchUpdate, dev, Qt.KeyboardModifier.NoModifier, [pt1_u, pt2_u, pt3_u])
        self.canvas.gv.viewportEvent(ev_update)

        pt1_e = QEventPoint(0, QEventPoint.State.Released, QPointF(150, 200), QPointF(150, 200))
        pt2_e = QEventPoint(1, QEventPoint.State.Released, QPointF(200, 200), QPointF(200, 200))
        pt3_e = QEventPoint(2, QEventPoint.State.Released, QPointF(250, 200), QPointF(250, 200))
        ev_end = QTouchEvent(QEvent.Type.TouchEnd, dev, Qt.KeyboardModifier.NoModifier, [pt1_e, pt2_e, pt3_e])
        self.canvas.gv.viewportEvent(ev_end)

        # 3-finger gestures must NOT fire because Windows reserves them
        self.assertFalse(next_page)
        self.assertFalse(prev_page)
        self.assertFalse(redo_called)

    def test_multi_touch_ignores_and_reverts_first_finger_tap(self) -> None:
        from qtpy.QtGui import QUndoCommand
        # Simulate tap 1 pushing a stroke to draw_undo_stack right before multi-touch
        class DummyStrokeCommand(QUndoCommand):
            def __init__(self):
                super().__init__()
                self.undone = False
            def undo(self):
                self.undone = True
            def redo(self):
                pass

        cmd = DummyStrokeCommand()
        self.canvas.draw_undo_stack.push(cmd)
        self.canvas._last_tap_stroke_time = time.time()
        self.canvas._last_tap_was_dragged = False

        dev = QPointingDevice()
        # Finger 1 & 2 land as multi-touch (e.g. pinch or pan)
        pt1 = QEventPoint(0, QEventPoint.State.Pressed, QPointF(100, 100), QPointF(100, 100))
        pt2 = QEventPoint(1, QEventPoint.State.Pressed, QPointF(200, 200), QPointF(200, 200))
        ev_begin = QTouchEvent(QEvent.Type.TouchBegin, dev, Qt.KeyboardModifier.NoModifier, [pt1, pt2])
        self.canvas.gv.viewportEvent(ev_begin)

        # The prior accidental tap 1 stroke must be automatically reverted/undone!
        self.assertTrue(cmd.undone)

    def test_double_click_cancels_pending_inpaint_and_undo_stroke(self) -> None:
        from qtpy.QtGui import QUndoCommand
        class DummyStrokeCommand(QUndoCommand):
            def __init__(self):
                super().__init__()
                self.undone = False
            def undo(self):
                self.undone = True
            def redo(self):
                pass

        cmd = DummyStrokeCommand()
        self.canvas.draw_undo_stack.push(cmd)
        self.canvas._last_tap_stroke_time = time.time()
        self.canvas._last_tap_was_dragged = False

        # Double click on background
        click_pos = QPointF(100, 100)
        dbl_event = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            click_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.gv.mouseDoubleClickEvent(dbl_event)

        # Tap 1's stroke must be undone
        self.assertTrue(cmd.undone)
        # Next release should be suppressed
        self.assertTrue(self.canvas._suppress_next_release)

    def test_double_click_background_toggles_zoom(self) -> None:
        fit_scale = self.canvas.get_fit_to_screen_scale()
        self.canvas.fit_to_screen()
        self.assertAlmostEqual(self.canvas.scale_factor, fit_scale, places=4)

        # Double click on background (100, 100) where no textblock exists
        click_pos = QPointF(100, 100)
        dbl_event = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            click_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.gv.mouseDoubleClickEvent(dbl_event)
        _APP.processEvents()

        expected_scale = fit_scale * 1.5
        self.assertAlmostEqual(self.canvas.scale_factor, expected_scale, places=4)

    def test_multi_touch_never_triggers_inpaint_or_erasing(self) -> None:
        from ballontranslator.ui.image_edit import ImageEditMode
        self.canvas.image_edit_mode = ImageEditMode.InpaintTool

        finish_painting_called = False
        finish_erasing_called = False
        def on_fp(stroke):
            nonlocal finish_painting_called
            finish_painting_called = True
        def on_fe(stroke):
            nonlocal finish_erasing_called
            finish_erasing_called = True

        self.canvas.finish_painting.connect(on_fp)
        self.canvas.finish_erasing.connect(on_fe)

        dev = QPointingDevice()
        # 1. Finger 1 touches down (Qt/Windows may synthesize LeftButton press)
        pt1 = QEventPoint(0, QEventPoint.State.Pressed, QPointF(100, 100), QPointF(100, 100))
        ev_begin = QTouchEvent(QEvent.Type.TouchBegin, dev, Qt.KeyboardModifier.NoModifier, [pt1])
        self.canvas.gv.viewportEvent(ev_begin)

        ev_mp1 = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(100, 100), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.canvas.gv.viewportEvent(ev_mp1)

        # 2. Finger 2 touches down (2 fingers active, Windows often sends RightButton press)
        pt1_u = QEventPoint(0, QEventPoint.State.Updated, QPointF(100, 100), QPointF(100, 100))
        pt2_u = QEventPoint(1, QEventPoint.State.Pressed, QPointF(200, 100), QPointF(200, 100))
        ev_up = QTouchEvent(QEvent.Type.TouchUpdate, dev, Qt.KeyboardModifier.NoModifier, [pt1_u, pt2_u])
        self.canvas.gv.viewportEvent(ev_up)

        ev_rpress = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(150, 100), Qt.MouseButton.RightButton, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
        self.canvas.gv.viewportEvent(ev_rpress)

        # 3. TouchEnd
        pt1_e = QEventPoint(0, QEventPoint.State.Released, QPointF(100, 100), QPointF(100, 100))
        pt2_e = QEventPoint(1, QEventPoint.State.Released, QPointF(200, 100), QPointF(200, 100))
        ev_end = QTouchEvent(QEvent.Type.TouchEnd, dev, Qt.KeyboardModifier.NoModifier, [pt1_e, pt2_e])
        self.canvas.gv.viewportEvent(ev_end)

        # 4. Windows sends synthetic release
        ev_mr = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(150, 100), Qt.MouseButton.RightButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        self.canvas.gv.viewportEvent(ev_mr)

        self.assertFalse(finish_painting_called)
        self.assertFalse(finish_erasing_called)
        self.assertIsNone(self.canvas.stroke_img_item)

    def test_stroke_has_dragged_tracking(self) -> None:
        from ballontranslator.ui.image_edit import StrokeImgItem
        from qtpy.QtCore import QSize
        stroke = StrokeImgItem(QPen(Qt.GlobalColor.black, 4), QPointF(10, 10), QSize(100, 100))
        self.assertFalse(stroke.has_dragged)

        # Tiny jitter (1px) -> still not dragged
        stroke.lineTo(QPointF(11, 10))
        self.assertFalse(stroke.has_dragged)

        # Move 30px -> marked as dragged
        stroke.lineTo(QPointF(40, 10))
        self.assertTrue(stroke.has_dragged)
        stroke.finishPainting()

    def test_drawing_panel_async_inpaint_canceled_by_double_click(self) -> None:
        from ballontranslator.ui.drawingpanel import DrawingPanel
        panel = DrawingPanel(self.canvas)
        self.assertEqual(self.canvas.drawing_panel, panel)

        # Set inpaint tool
        panel.on_use_inpainttool()
        self.assertEqual(panel.currentTool, panel.inpaintTool)

        # Create a stationary tap stroke (has_dragged = False)
        from ballontranslator.ui.image_edit import StrokeImgItem
        from qtpy.QtCore import QSize
        stroke = StrokeImgItem(QPen(Qt.GlobalColor.black, 4), QPointF(50, 50), QSize(100, 100))
        self.canvas.addItem(stroke)
        self.assertFalse(stroke.has_dragged)

        # Tap 1 finishes painting -> should start async debounce timer, NOT run immediately
        panel.on_finish_painting(stroke)
        self.assertTrue(panel._pending_inpaint_timer.isActive())
        self.assertIsNotNone(panel.inpaint_stroke)

        # Tap 2 arrives (Double click) -> should immediately cancel pending inpaint!
        click_pos = QPointF(100, 100)
        dbl_event = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            click_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.gv.mouseDoubleClickEvent(dbl_event)

        self.assertFalse(panel._pending_inpaint_timer.isActive())
        self.assertIsNone(panel.inpaint_stroke)
        panel.deleteLater()

    def test_detect_pointer_device(self) -> None:
        dev = QPointingDevice()
        # 1. Native Gesture (Touchpad)
        g_ev = QNativeGestureEvent(Qt.NativeGestureType.ZoomNativeGesture, dev, QPointF(0, 0), QPointF(0, 0), QPointF(0, 0), 0.1, 0, 0)
        self.assertEqual(detect_pointer_device(g_ev), 'touchpad')

        # 2. Wheel event with pixelDelta (Touchpad)
        w_pad = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(15, -20), QPoint(0, 0), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        self.assertEqual(detect_pointer_device(w_pad), 'touchpad')

        # 3. Wheel event with angleDelta only (Mouse)
        w_mouse = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        self.assertEqual(detect_pointer_device(w_mouse), 'mouse')

        # 4. Touch event (Touchscreen)
        pt1 = QEventPoint(0, QEventPoint.State.Pressed, QPointF(100, 100), QPointF(100, 100))
        t_ev = QTouchEvent(QEvent.Type.TouchBegin, dev, Qt.KeyboardModifier.NoModifier, [pt1])
        self.assertEqual(detect_pointer_device(t_ev), 'touchscreen')

        # 5. Stylus / Tablet event
        tablet_press_type = getattr(QEvent.Type, 'TabletPress', getattr(QEvent, 'TabletPress', None))
        tab_ev = SimpleNamespace(
            type=lambda: tablet_press_type,
            pointingDevice=lambda: dev,
        )
        self.assertEqual(detect_pointer_device(tab_ev), 'stylus')

    def test_tablet_stroke_pressure_and_painting(self) -> None:
        self.canvas.image_edit_mode = ImageEditMode.InpaintTool
        finish_painting_called = False
        def on_fp(stroke):
            nonlocal finish_painting_called
            finish_painting_called = True
            stroke.finishPainting()
            self.canvas.removeItem(stroke)

        self.canvas.finish_painting.connect(on_fp)

        # 1. Tablet press with pressure 0.5
        self.canvas.handle_tablet_press(QPointF(50, 50), is_eraser=False, pressure=0.5)
        self.assertIsNotNone(self.canvas.stroke_img_item)
        stroke = self.canvas.stroke_img_item

        # 2. Tablet move with pressure 0.8
        self.canvas.handle_tablet_move(QPointF(80, 80), pressure=0.8)
        self.assertTrue(stroke.has_dragged)

        # 3. Tablet release
        self.canvas.handle_tablet_release(QPointF(80, 80), is_eraser=False)
        self.assertTrue(finish_painting_called)

    def test_tablet_eraser_behavior(self) -> None:
        self.canvas.image_edit_mode = ImageEditMode.InpaintTool
        finish_erasing_called = False
        def on_fe(stroke):
            nonlocal finish_erasing_called
            finish_erasing_called = True
            stroke.finishPainting()
            self.canvas.removeItem(stroke)

        self.canvas.finish_erasing.connect(on_fe)

        # Tablet press with is_eraser=True (eraser tail of stylus)
        self.canvas.handle_tablet_press(QPointF(50, 50), is_eraser=True, pressure=1.0)
        self.assertTrue(self.canvas._tablet_is_erasing)

        self.canvas.handle_tablet_move(QPointF(70, 70), pressure=1.0)
        self.canvas.handle_tablet_release(QPointF(70, 70), is_eraser=True)

        self.assertTrue(finish_erasing_called)
        self.assertFalse(self.canvas._tablet_is_erasing)

    def test_palm_rejection_during_tablet_use(self) -> None:
        self.canvas.image_edit_mode = ImageEditMode.InpaintTool
        # Tablet is currently active
        self.canvas.gv._tablet_active = True
        self.canvas.gv._last_tablet_time = time.time()

        # Touch event lands from resting palm
        dev = QPointingDevice()
        pt1 = QEventPoint(0, QEventPoint.State.Pressed, QPointF(200, 200), QPointF(200, 200))
        ev_palm = QTouchEvent(QEvent.Type.TouchBegin, dev, Qt.KeyboardModifier.NoModifier, [pt1])
        handled = self.canvas.gv.viewportEvent(ev_palm)

        # Palm touch must be accepted and consumed (preventing inpaint / gestures)
        self.assertTrue(handled)
        self.assertFalse(self.canvas.gv._touch_active)

        self.canvas.gv._tablet_active = False

    def test_touchpad_smooth_pan(self) -> None:
        h_bar = self.canvas.gv.horizontalScrollBar()
        v_bar = self.canvas.gv.verticalScrollBar()
        h_bar.setValue(100)
        v_bar.setValue(100)

        # Emulate touchpad two-finger scroll (pixelDelta is non-null)
        ev_wheel = QWheelEvent(
            QPointF(200, 200),
            QPointF(200, 200),
            QPoint(30, -40),  # pixelDelta
            QPoint(0, 0),     # angleDelta
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.canvas.gv.wheelEvent(ev_wheel)

        self.assertEqual(h_bar.value(), 70)   # 100 - 30
        self.assertEqual(v_bar.value(), 140)  # 100 - (-40)

    def test_touchpad_smooth_zoom(self) -> None:
        initial_scale = self.canvas.scale_factor

        # Emulate touchpad pinch-zoom (pixelDelta with Ctrl modifier)
        ev_zoom = QWheelEvent(
            QPointF(200, 200),
            QPointF(200, 200),
            QPoint(0, 20),   # pixelDelta positive = zoom in
            QPoint(0, 0),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.canvas.gv.wheelEvent(ev_zoom)

        self.assertGreater(self.canvas.scale_factor, initial_scale)


if __name__ == '__main__':
    unittest.main()
