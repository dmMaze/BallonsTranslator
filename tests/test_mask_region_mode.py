import gc
import os
import threading
import time
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch
from typing import Callable

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QKeyEvent, QMouseEvent
from qtpy.QtTest import QTest, QSignalSpy
from qtpy.QtWidgets import QApplication, QDialog, QHBoxLayout, QLineEdit, QMenu, QPlainTextEdit, QTextEdit, QVBoxLayout, QWidget

from ballontranslator.ui import drawingpanel
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import MaskRegionIndex, compute_forward_flood_fill
from ballontranslator.ui.image_edit import ImageEditMode, PenShape
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.mainwindow import PageListView
from ballontranslator.ui.mainwindowbars import TitleBar
from ballontranslator.utils.config import DrawPanelConfig, pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans

LEFT = Qt.MouseButton.LeftButton
RIGHT = Qt.MouseButton.RightButton
CTRL = Qt.KeyboardModifier.ControlModifier


class MaskRegionIndexTests(unittest.TestCase):
    def test_any_single_edge_rejects_entire_addition(self) -> None:
        for edge in ('top', 'bottom', 'left', 'right'):
            with self.subTest(edge=edge):
                mask = np.full((20, 20), 255, np.uint8)
                mask[8:12, 8:12] = 0
                if edge == 'top':
                    mask[:4, 3:6] = 0
                elif edge == 'bottom':
                    mask[-4:, 3:6] = 0
                elif edge == 'left':
                    mask[3:6, :4] = 0
                else:
                    mask[3:6, -4:] = 0
                brush = np.ones_like(mask) * 255
                index = MaskRegionIndex(mask)
                selected = index.touched([0, 0], brush, False)
                overlay, x, y, status = index.preview(selected, False)
                self.assertEqual(status, 'reject')
                self.assertEqual(np.count_nonzero(overlay[..., 3]), 12)
                self.assertTrue(np.any(overlay[..., 3] == 255))
                image = np.zeros((20, 20, 3), np.uint8)
                self.assertIsNone(compute_forward_flood_fill([0, 0, 20, 20], brush, mask, image, index))
                self.assertEqual(mask[9, 9], 0)

    def test_diagonal_regions_remain_separate(self) -> None:
        mask = np.full((6, 6), 255, np.uint8)
        mask[2, 2] = mask[3, 3] = 0
        index = MaskRegionIndex(mask)
        selected = index.touched([2, 2], np.ones((1, 1), np.uint8), False)
        overlay, x, y, status = index.preview(selected, False)
        self.assertEqual((x, y, status), (2, 2, 'add'))
        self.assertEqual(np.count_nonzero(overlay[..., 3]), 1)
        result = compute_forward_flood_fill([2, 2, 3, 3], np.ones((1, 1), np.uint8) * 255,
                                            mask, np.zeros((6, 6, 3), np.uint8), index)
        self.assertIsNotNone(result)
        # The diagonal hole remains unselected by the committed inpaint mask.
        x1, y1, _, _ = result['inpaint_rect']
        self.assertEqual(result['mask'][3-y1, 3-x1], 0)
        self.assertIsNone(index.at(-1, 3))
        self.assertEqual(index.touched([-20, -20], np.ones((2, 2), np.uint8), False), frozenset())


class MaskRegionConfigTests(unittest.TestCase):
    def test_config_defaults_validation_and_legacy_migration(self) -> None:
        self.assertFalse(DrawPanelConfig().mask_region_mode)
        self.assertTrue(DrawPanelConfig(mask_region_mode=True).copy().mask_region_mode)
        for malformed in (1, 'false', None, []):
            with self.subTest(malformed=malformed), self.assertLogs('BallonTranslator', level='WARNING'):
                self.assertFalse(DrawPanelConfig(mask_region_mode=malformed).mask_region_mode)
        with self.assertLogs('BallonTranslator', level='WARNING'):
            migrated = DrawPanelConfig(inpainter_shape=PenShape.FloodFill)
        self.assertEqual(migrated.inpainter_shape, PenShape.Circle)
        self.assertTrue(migrated.mask_region_mode)


class _MaskRegionCanvasFixture:
    """Shared real canvas and panel construction; contains no discoverable tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_config = pcfg.drawpanel
        self.old_opacity = pcfg.mask_transparency
        pcfg.drawpanel = DrawPanelConfig(inpainter_width=6)
        self.canvas = Canvas()
        self.project = ProjImgTrans()
        self.project.img_array = np.full((120, 120, 3), 240, np.uint8)
        self.project.inpainted_array = np.full((120, 120, 3), 180, np.uint8)
        self.project.mask_array = np.full((120, 120), 255, np.uint8)
        self.project.mask_array[30:90, 20:50] = 0
        self.project.mask_array[30:90, 70:100] = 0
        self.canvas.imgtrans_proj = self.project
        self.panel = drawingpanel.DrawingPanel(self.canvas)
        self.panel.module_manager = Mock()
        self.panel.set_config(pcfg.drawpanel)
        self.window = QWidget()
        self.title_bar = TitleBar(self.window)
        self.title_bar.mask_region_toggle_requested.connect(self.panel.toggle_mask_region_mode)
        root_layout = QVBoxLayout(self.window)
        root_layout.addWidget(self.title_bar)
        layout = QHBoxLayout()
        root_layout.addLayout(layout)
        layout.addWidget(self.canvas.gv)
        layout.addWidget(self.panel)
        self.window.resize(900, 600)
        self.window.show()
        self.canvas.updateCanvas()
        self.canvas.setPaintMode(True)
        self.panel.setCurrentToolByName('inpaint')
        self.app.processEvents()
        self.panel.set_mask_region_mode(True)

    def tearDown(self) -> None:
        if self.window is not None:
            self.canvas.clear_states()
            self.assertTrue(self.panel.flood_fill_thread.wait(3000))
            self.app.processEvents()
            self.window.close()
            self.window.deleteLater()
            self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        pcfg.drawpanel = self.old_config
        pcfg.mask_transparency = self.old_opacity

    def pos(self, x: float, y: float) -> QPoint:
        return self.canvas.gv.mapFromScene(self.canvas.baseLayer.mapToScene(QPointF(x, y)))

    def move(self, x: float, y: float, buttons: Qt.MouseButton = Qt.MouseButton.NoButton) -> None:
        viewport = self.canvas.gv.viewport()
        pos = self.pos(x, y)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(pos), QPointF(viewport.mapToGlobal(pos)),
                            Qt.MouseButton.NoButton, buttons, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, event)

    def press(self, x: float, y: float, button: Qt.MouseButton = LEFT,
              modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> None:
        QTest.mousePress(self.canvas.gv.viewport(), button, modifiers, self.pos(x, y))

    def release(self, x: float, y: float, button: Qt.MouseButton = LEFT,
                modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> None:
        QTest.mouseRelease(self.canvas.gv.viewport(), button, modifiers, self.pos(x, y))

    def wait_until(self, predicate: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 3
        while not predicate():
            self.assertLess(time.monotonic(), deadline, 'Region preview did not settle.')
            QTest.qWait(10)

    def overlay(self) -> np.ndarray:
        result = np.zeros((120, 120, 4), np.uint8)
        item = self.panel.region_preview_item
        if item.isVisible():
            crop = pixmap2ndarray(item.pixmap())
            x, y = int(item.pos().x()), int(item.pos().y())
            result[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        return result

    def wait_pixels(self, count: int, status: str) -> None:
        self.wait_until(lambda: np.count_nonzero(self.overlay()[..., 3]) == count
                        and self.panel._region_cursor_status == status)


class MaskRegionShortcutTests(_MaskRegionCanvasFixture, unittest.TestCase):
    def test_shift_toggles_once_and_release_keeps_mode(self) -> None:
        self.panel.set_mask_region_mode(False)
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertTrue(self.canvas.mask_region_mode)
        self.assertEqual(pcfg.mask_transparency, 0.2)
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Shift,
                          Qt.KeyboardModifier.ShiftModifier, '', True)
        self.app.sendEvent(self.canvas.gv, event)
        self.assertTrue(self.canvas.mask_region_mode)
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertTrue(self.canvas.mask_region_mode)
        self.panel.maskTransperancySlider.setValue(80)
        QTest.keyClick(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertFalse(self.canvas.mask_region_mode)
        self.assertEqual(pcfg.mask_transparency, 0.2)
        self.assertFalse(self.panel.inpaintConfigPanel.regionModeCheckBox.isChecked())
        for modifier in (Qt.KeyboardModifier.ControlModifier,
                         Qt.KeyboardModifier.AltModifier,
                         Qt.KeyboardModifier.MetaModifier):
            QTest.keyClick(self.canvas.gv, Qt.Key.Key_Shift, modifier)
            self.assertFalse(pcfg.drawpanel.mask_region_mode)

    def test_shift_toggles_after_brush_controls_retain_focus(self) -> None:
        self.window.activateWindow()
        controls = self.panel.inpaintConfigPanel
        widgets = (
            self.panel.inpaintTool, controls.inpainter_selector.tool_button,
            controls.inpainter_selector.config_button,
            controls.shapeCombobox, controls.thicknessSlider,
            controls.thicknessSpinBox, controls.thicknessSpinBox.lineEdit(),
            controls.regionModeCheckBox, self.panel.maskTransperancySlider,
        )
        requests = QSignalSpy(self.title_bar.mask_region_toggle_requested)
        for widget in widgets:
            with self.subTest(control=type(widget).__name__):
                widget.setFocus()
                self.app.processEvents()
                target = self.app.focusWidget()
                self.assertIsNotNone(target)
                self.assertTrue(target is widget or widget.isAncestorOf(target)
                                or widget is controls.thicknessSpinBox.lineEdit()
                                and target is controls.thicknessSpinBox)
                before = pcfg.drawpanel.mask_region_mode
                count = len(requests)
                QTest.keyClick(target, Qt.Key.Key_Shift)
                self.assertEqual(pcfg.drawpanel.mask_region_mode, not before)
                self.assertEqual(len(requests), count + 1)

    def test_shift_toggles_after_clicking_page_list(self) -> None:
        pages = PageListView()
        pages.addItems(['page 1', 'page 2'])
        self.window.layout().addWidget(pages)
        self.window.activateWindow()
        self.app.processEvents()
        requests = QSignalSpy(self.title_bar.mask_region_toggle_requested)
        QTest.mouseClick(pages.viewport(), LEFT, pos=pages.visualItemRect(pages.item(1)).center())
        self.app.processEvents()
        self.assertIs(self.app.focusWidget(), pages)
        before = pcfg.drawpanel.mask_region_mode
        QTest.keyClick(pages, Qt.Key.Key_Shift)
        self.assertEqual(pcfg.drawpanel.mask_region_mode, not before)
        self.assertEqual(len(requests), 1)
        self.assertEqual(pages.currentRow(), 1)

    def test_window_shift_preserves_text_editing(self) -> None:
        self.window.activateWindow()
        row = self.panel.inpaintConfigPanel.inpainter_selector
        row.prompt_panel.show()
        editors = (row.prompt_edit, QLineEdit(self.window), QTextEdit(self.window), QPlainTextEdit(self.window))
        for editor in editors:
            with self.subTest(editor=type(editor).__name__):
                if editor is not row.prompt_edit:
                    self.window.layout().addWidget(editor)
                editor.setFocus()
                self.app.processEvents()
                self.assertIs(self.app.focusWidget(), editor)
                before = pcfg.drawpanel.mask_region_mode
                QTest.keyClick(editor, Qt.Key.Key_Shift)
                QTest.keyClicks(editor, 'A', Qt.KeyboardModifier.ShiftModifier)
                text = editor.text() if isinstance(editor, QLineEdit) else editor.toPlainText()
                self.assertEqual(text, 'A')
                self.assertEqual(pcfg.drawpanel.mask_region_mode, before)
                editor.hide()

    def test_window_shift_excludes_other_windows_popups_and_modal_dialogs(self) -> None:
        before = pcfg.drawpanel.mask_region_mode
        other = QWidget()
        popup = QMenu(self.window)
        popup.addAction('Action')
        dialog = QDialog(self.window)
        dialog.setModal(True)
        try:
            other.show()
            other.activateWindow()
            self.app.processEvents()
            QTest.keyClick(other, Qt.Key.Key_Shift)
            self.assertEqual(pcfg.drawpanel.mask_region_mode, before)
            other.hide()
            self.window.activateWindow()
            popup.popup(self.window.mapToGlobal(QPoint(10, 10)))
            self.app.processEvents()
            self.assertIs(self.app.activePopupWidget(), popup)
            QTest.keyClick(popup, Qt.Key.Key_Shift)
            self.assertEqual(pcfg.drawpanel.mask_region_mode, before)
            popup.hide()
            dialog.show()
            self.app.processEvents()
            self.assertIs(self.app.activeModalWidget(), dialog)
            QTest.keyClick(dialog, Qt.Key.Key_Shift)
            self.assertEqual(pcfg.drawpanel.mask_region_mode, before)
        finally:
            for widget in (other, popup, dialog):
                widget.close()
                widget.deleteLater()

    def test_window_shift_emits_but_does_not_toggle_outside_inpaint_brushes(self) -> None:
        self.window.activateWindow()
        requests = QSignalSpy(self.title_bar.mask_region_toggle_requested)
        before = pcfg.drawpanel.mask_region_mode
        for tool in ('pen', 'rect', 'hand'):
            self.panel.setCurrentToolByName(tool)
            count = len(requests)
            QTest.keyClick(self.window, Qt.Key.Key_Shift)
            self.assertEqual(len(requests), count + 1)
            self.assertEqual(pcfg.drawpanel.mask_region_mode, before)
            QTest.keyClick(self.panel.inpaintTool, Qt.Key.Key_Shift)
            self.assertEqual(pcfg.drawpanel.mask_region_mode, before)

    def test_window_shift_is_disabled_on_hide_and_restored_on_show(self) -> None:
        self.window.activateWindow()
        requests = QSignalSpy(self.title_bar.mask_region_toggle_requested)
        before = pcfg.drawpanel.mask_region_mode
        self.panel.hide()
        QTest.keyClick(self.window, Qt.Key.Key_Shift)
        self.assertEqual(pcfg.drawpanel.mask_region_mode, before)
        self.assertEqual(len(requests), 1)
        self.panel.show()
        QTest.keyClick(self.window, Qt.Key.Key_Shift)
        self.assertEqual(pcfg.drawpanel.mask_region_mode, not before)
        self.assertEqual(len(requests), 2)

    def test_mode_is_shared_by_both_shapes_and_excludes_other_tools(self) -> None:
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.Rectangle)
        self.assertTrue(self.canvas.mask_region_mode)
        self.assertTrue(self.panel.inpaintConfigPanel.region_mode_row.isVisible())
        self.panel.setCurrentToolByName('pen')
        self.assertFalse(self.canvas.mask_region_mode)
        QTest.keyClick(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertTrue(pcfg.drawpanel.mask_region_mode)
        self.panel.setCurrentToolByName('inpaint')
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.MagicWand)
        self.assertFalse(self.canvas._mask_region_hover_enabled)
        self.assertFalse(self.panel.inpaintConfigPanel.region_mode_row.isVisible())
        QTest.keyClick(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertTrue(pcfg.drawpanel.mask_region_mode)
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.Circle)
        self.assertTrue(self.canvas._mask_region_hover_enabled)

    def test_busy_mode_cannot_toggle_until_inpaint_finishes(self) -> None:
        self.press(35, 60)
        self.release(35, 60)
        self.assertTrue(self.panel.flood_fill_thread.wait(3000))
        self.app.processEvents()
        self.assertEqual(self.canvas.image_edit_mode, ImageEditMode.NONE)
        self.assertFalse(self.panel.inpaintConfigPanel.regionModeCheckBox.isEnabled())
        QTest.keyClick(self.window, Qt.Key.Key_Shift)
        self.assertTrue(pcfg.drawpanel.mask_region_mode)
        self.panel.set_mask_region_mode(False)
        self.assertTrue(pcfg.drawpanel.mask_region_mode)
        self.panel._finish_inpaint()
        self.assertTrue(self.panel.inpaintConfigPanel.regionModeCheckBox.isEnabled())
        QTest.keyClick(self.window, Qt.Key.Key_Shift)
        self.assertFalse(pcfg.drawpanel.mask_region_mode)


class MaskRegionPreviewTests(_MaskRegionCanvasFixture, unittest.TestCase):
    def test_hover_add_clear_and_reject_with_edge_highlighting(self) -> None:
        self.move(35, 60)
        self.wait_pixels(1800, 'add')
        np.testing.assert_allclose(self.overlay()[60, 35], drawingpanel.REGION_ADD_COLOR, atol=1)
        self.move(60, 60)
        self.wait_pixels(10800, 'clear')
        np.testing.assert_allclose(self.overlay()[60, 60], drawingpanel.REGION_CLEAR_COLOR, atol=1)
        self.project.mask_array[:30, 20:50] = 0
        self.canvas.updateLayers()
        self.move(35, 40)
        self.wait_pixels(2700, 'reject')
        overlay = self.overlay()
        np.testing.assert_allclose(overlay[40, 35], drawingpanel.REGION_REJECT_COLOR, atol=1)
        self.assertTrue(np.all(overlay[0, 20:50, 3] == 255))
        self.assertFalse(np.any(overlay[30:90, 70:100, 3]))

    def test_noop_stroke_hides_bucket_until_it_reaches_a_matching_region(self) -> None:
        for shape in (PenShape.Circle, PenShape.Rectangle):
            with self.subTest(shape=shape):
                self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(shape)
                self.panel.inpaintConfigPanel.thicknessSlider.setValue(50)
                self.move(35, 60)
                self.wait_pixels(1800, 'add')
                self.press(60, 60, LEFT)
                self.assertEqual(self.panel._region_cursor_status, 'none')
                bitmap = pixmap2ndarray(self.canvas.gv.viewport().cursor().pixmap())
                cursor = self.canvas.gv.viewport().cursor()
                outside_tip = cursor.hotSpot().x() + 25
                self.assertFalse(bitmap[outside_tip:, outside_tip:, 3].any())
                self.move(35, 60, LEFT)
                self.wait_until(lambda: self.panel._region_cursor_status == 'add')
                bitmap = pixmap2ndarray(self.canvas.gv.viewport().cursor().pixmap())
                self.assertTrue(bitmap[outside_tip:, outside_tip:, 3].any())
                self.canvas.clear_states()
                self.release(35, 60, LEFT)
                self.move(60, 60)
                self.wait_pixels(10800, 'clear')
                self.press(35, 60, RIGHT)
                self.assertEqual(self.panel._region_cursor_status, 'none')
                bitmap = pixmap2ndarray(self.canvas.gv.viewport().cursor().pixmap())
                self.assertFalse(bitmap[outside_tip:, outside_tip:, 3].any())
                self.move(60, 60, RIGHT)
                self.wait_until(lambda: self.panel._region_cursor_status == 'clear')
                self.canvas.clear_states()
                self.release(60, 60, RIGHT)
                self.panel.module_manager.canvas_inpaint.assert_not_called()

    def test_region_icons_stay_beyond_the_tip_and_keep_the_brush_hotspot(self) -> None:
        for shape in (PenShape.Circle, PenShape.Rectangle):
            for zoom in (0.5, 2):
                self.canvas.scaleImage(zoom)
                for status in ('base', 'add', 'clear', 'reject', 'none'):
                    with self.subTest(shape=shape, zoom=zoom, status=status):
                        cursor = self.panel.get_pen_cursor(pen_size=50, shape=shape, region_action=status)
                        pixels = pixmap2ndarray(cursor.pixmap())
                        hotspot = cursor.hotSpot()
                        # Opaque ink belongs to the operation group; the brush
                        # outline is translucent. Keep all ink outside the tip.
                        y, x = np.nonzero(pixels[..., 3] > 200)
                        if status == 'none':
                            self.assertFalse(x.size)
                        else:
                            self.assertTrue(x.size)
                            self.assertTrue(np.all(x > hotspot.x() + 25 * self.canvas.scale_factor))
                            self.assertTrue(np.all(y > hotspot.y() + 25 * self.canvas.scale_factor))
                        self.assertFalse(pixels[hotspot.y(), hotspot.x(), 3])

    def test_live_region_strokes_render_above_preview_without_entering_export(self) -> None:
        for shape in (PenShape.Circle, PenShape.Rectangle):
            for button, x, count, status in ((LEFT, 35, 1800, 'add'), (RIGHT, 60, 10800, 'clear')):
                with self.subTest(shape=shape, button=button):
                    self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(shape)
                    self.move(x, 50)
                    self.wait_pixels(count, status)
                    self.app.processEvents()
                    exported = pixmap2ndarray(self.canvas.render_result_img())
                    point = self.pos(x, 55)
                    before = self.canvas.gv.viewport().grab().toImage().pixelColor(point)
                    self.press(x, 50, button)
                    self.move(x, 70, button)
                    self.wait_pixels(count, status)
                    self.app.processEvents()
                    after = self.canvas.gv.viewport().grab().toImage().pixelColor(point)
                    self.assertNotEqual(before, after)
                    np.testing.assert_array_equal(exported, pixmap2ndarray(self.canvas.render_result_img()))
                    self.assertTrue(self.canvas.stroke_img_item.isVisible())
                    self.canvas.clear_states()
                    self.release(x, 70, button)
                    self.panel.module_manager.canvas_inpaint.assert_not_called()


class MaskRegionStrokeTests(_MaskRegionCanvasFixture, unittest.TestCase):
    def test_left_drag_previews_all_off_regions_for_both_shapes(self) -> None:
        for shape in (PenShape.Circle, PenShape.Rectangle):
            with self.subTest(shape=shape):
                self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(shape)
                self.move(60, 60)
                self.wait_pixels(10800, 'clear')
                self.press(35, 60)
                self.assertEqual(self.panel._region_cursor_status, 'add')
                self.wait_pixels(1800, 'add')
                self.move(85, 60, LEFT)
                self.wait_pixels(3600, 'add')
                self.assertEqual(self.overlay()[60, 60, 3], 0)
                self.canvas.clear_states()
                self.assertFalse(self.panel.region_preview_item.isVisible())
                self.assertIsNone(self.canvas.stroke_img_item)
                self.release(85, 60)

    def test_rejected_drag_hides_all_valid_regions_and_submits_nothing(self) -> None:
        self.project.mask_array[:30, 70:100] = 0
        self.canvas.updateLayers()
        before_mask = self.project.mask_array.copy()
        before_image = self.project.inpainted_array.copy()
        self.press(35, 60)
        self.wait_pixels(1800, 'add')
        self.move(85, 60, LEFT)
        self.wait_pixels(2700, 'reject')
        self.assertEqual(self.overlay()[60, 35, 3], 0)
        self.assertEqual(self.overlay()[0, 85, 3], 255)
        self.release(85, 60)
        self.assertTrue(self.panel.flood_fill_thread.wait(3000))
        self.app.processEvents()
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        np.testing.assert_array_equal(self.project.mask_array, before_mask)
        np.testing.assert_array_equal(self.project.inpainted_array, before_image)

    def test_right_drag_previews_all_on_regions_and_undo_restores_them(self) -> None:
        on_components = 255 - self.project.mask_array
        for shape in (PenShape.Circle, PenShape.Rectangle):
            with self.subTest(shape=shape):
                self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(shape)
                self.project.mask_array[:] = on_components
                self.project.inpainted_array.fill(180)
                self.canvas.updateLayers()
                before = self.project.mask_array.copy()
                self.move(60, 60)
                self.wait_pixels(10800, 'reject')
                self.press(35, 60, RIGHT)
                self.assertEqual(self.panel._region_cursor_status, 'clear')
                self.wait_pixels(1800, 'clear')
                self.move(85, 60, RIGHT)
                self.wait_pixels(3600, 'clear')
                self.assertEqual(self.overlay()[60, 60, 3], 0)
                self.release(85, 60, RIGHT)
                self.assertTrue(self.panel.flood_fill_thread.wait(3000))
                self.app.processEvents()
                self.assertFalse(np.any(self.project.mask_array))
                self.assertTrue(np.all(self.project.inpainted_array[30:90, 20:50] == 240))
                self.canvas.undo()
                np.testing.assert_array_equal(self.project.mask_array, before)
                self.canvas.redo()
                self.assertFalse(np.any(self.project.mask_array))

    def test_addition_completion_preserves_other_holes_and_round_trips_undo(self) -> None:
        before_mask = self.project.mask_array.copy()
        before_image = self.project.inpainted_array.copy()
        self.move(35, 60)
        self.wait_pixels(1800, 'add')
        self.press(35, 60)
        self.release(35, 60)
        self.assertTrue(self.panel.flood_fill_thread.wait(3000))
        self.app.processEvents()
        request = self.panel.module_manager.canvas_inpaint.call_args.args[0]
        image = np.where(request['mask'][..., None] > 0, 80, request['img']).astype(np.uint8)
        self.panel.on_inpaint_finished({
            'inpainted': image, 'mask': request['mask'],
            'inpaint_rect': request['inpaint_rect'],
        })
        expected = before_mask.copy()
        expected[30:90, 20:50] = 255
        np.testing.assert_array_equal(self.project.mask_array, expected)
        self.assertTrue(np.all(self.project.inpainted_array[30:90, 20:50] == 80))
        self.assertTrue(np.all(self.project.inpainted_array[30:90, 70:100] == 180))
        self.assertEqual(self.canvas.draw_undo_stack.count(), 1)
        self.canvas.undo()
        np.testing.assert_array_equal(self.project.mask_array, before_mask)
        np.testing.assert_array_equal(self.project.inpainted_array, before_image)
        self.canvas.redo()
        np.testing.assert_array_equal(self.project.mask_array, expected)

    def test_ctrl_accumulates_components_and_button_direction_never_leaks(self) -> None:
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        self.press(35, 60, LEFT, CTRL)
        self.release(35, 60, LEFT, CTRL)
        self.wait_pixels(1800, 'add')
        self.press(85, 60, LEFT, CTRL)
        self.release(85, 60, LEFT, CTRL)
        self.wait_pixels(3600, 'add')
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
        self.assertTrue(self.panel.flood_fill_thread.wait(3000))
        self.app.processEvents()
        self.panel.module_manager.canvas_inpaint.assert_called_once()
        self.panel._finish_inpaint()
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        self.press(35, 60, LEFT, CTRL)
        self.release(35, 60, LEFT, CTRL)
        self.press(60, 60, RIGHT)
        self.assertEqual(self.panel._region_cursor_status, 'clear')
        self.wait_pixels(10800, 'clear')
        self.canvas.clear_states()

    def test_opposite_release_cannot_finish_region_stroke(self) -> None:
        self.press(35, 60)
        self.press(60, 60, RIGHT)
        self.release(60, 60, RIGHT)
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        self.release(35, 60)
        self.assertTrue(self.panel.flood_fill_thread.wait(3000))
        self.app.processEvents()
        self.panel.module_manager.canvas_inpaint.assert_called_once()

    def test_global_escape_cancels_ctrl_accumulation_before_ctrl_release(self) -> None:
        from ballontranslator.ui.mainwindow import MainWindow
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        self.press(35, 60, LEFT, CTRL)
        self.release(35, 60, LEFT, CTRL)
        self.wait_pixels(1800, 'add')
        MainWindow.shortcutEscape(SimpleNamespace(canvas=self.canvas))
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
        self.app.processEvents()
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        self.assertFalse(self.panel.region_preview_item.isVisible())
        self.assertIsNone(self.panel.inpaint_stroke)


class MaskRegionLifecycleTests(_MaskRegionCanvasFixture, unittest.TestCase):
    def test_preview_export_zoom_cancel_leave_and_page_invalidation(self) -> None:
        before = pixmap2ndarray(self.canvas.render_result_img())
        self.canvas.scaleImage(2)
        self.move(35, 60)
        self.wait_pixels(1800, 'add')
        np.testing.assert_array_equal(before, pixmap2ndarray(self.canvas.render_result_img()))
        self.app.sendEvent(self.canvas.gv.viewport(), QEvent(QEvent.Type.Leave))
        self.assertFalse(self.panel.region_preview_item.isVisible())
        self.move(35, 60)
        self.wait_pixels(1800, 'add')
        self.press(35, 60)
        self.canvas.clear_states()
        self.assertFalse(self.panel.region_preview_item.isVisible())
        self.assertIsNone(self.canvas.stroke_img_item)
        self.release(35, 60)
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        self.project.mask_array.fill(0)
        self.panel.handle_page_changed()
        self.move(35, 60)
        self.wait_pixels(14400, 'reject')
        self.panel.hide()
        self.assertFalse(self.canvas._mask_region_hover_enabled)
        self.assertFalse(self.panel.region_preview_item.isVisible())
        self.assertIsNone(self.panel._region_index)

    def test_stale_pending_source_is_discarded(self) -> None:
        calculate = drawingpanel.MaskRegionIndex
        self.panel.invalidate_mask_regions()
        release = threading.Event()
        started = threading.Event()

        def held_index(mask: np.ndarray) -> MaskRegionIndex:
            started.set()
            release.wait(3)
            return calculate(mask)

        with patch.object(drawingpanel, 'MaskRegionIndex', held_index):
            try:
                self.move(35, 60)
                self.wait_until(started.is_set)
                QTest.keyClick(self.canvas.gv, Qt.Key.Key_Shift)
                self.assertFalse(self.canvas.mask_region_mode)
                self.assertFalse(self.panel.region_preview_item.isVisible())
                self.project.mask_array.fill(0)
                self.canvas.updateLayers()
                QTest.keyClick(self.canvas.gv, Qt.Key.Key_Shift)
            finally:
                release.set()
            self.move(35, 60)
            self.wait_pixels(14400, 'reject')

    def test_cancel_does_not_recreate_preview(self) -> None:
        self.press(35, 60)
        self.wait_pixels(1800, 'add')
        self.canvas.clear_states()
        QTest.qWait(100)
        self.assertFalse(self.panel.region_preview_item.isVisible())
        self.release(35, 60)
        self.panel.module_manager.canvas_inpaint.assert_not_called()

    def test_pending_worker_releases_panel_and_never_applies_after_deletion(self) -> None:
        release = threading.Event()
        started = threading.Event()
        calculate = drawingpanel.MaskRegionIndex

        def held_index(mask: np.ndarray) -> MaskRegionIndex:
            started.set()
            release.wait(3)
            return calculate(mask)

        self.panel.invalidate_mask_regions()
        with patch.object(drawingpanel, 'MaskRegionIndex', held_index):
            try:
                self.move(35, 60)
                self.wait_until(started.is_set)
                future = self.panel._region_index_future
                reference = weakref.ref(self.panel)
                self.window.close()
                self.window.deleteLater()
                self.canvas.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.window = self.panel = self.canvas = None
                gc.collect()
                self.assertIsNone(reference())
            finally:
                release.set()
            # Complete native work and dispatch pending events after deletion.
            future.result(timeout=3)
            self.app.processEvents()
            gc.collect()
            self.assertIsNone(reference())


if __name__ == '__main__':
    unittest.main()
