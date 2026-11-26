import os
import tempfile
import unittest
import weakref
import gc
from copy import deepcopy
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PIL import Image
from qtpy.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QMouseEvent, QPainterPathStroker
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QMenu, QWidget

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.text_engine.bubble_layout import fit_text_in_shape
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager, TextPanel
from ballontranslator.utils import config as C, shared
from ballontranslator.utils.bubble_shape import BubbleShape, bubble_path
from ballontranslator.utils.fontformat import BendTextTransform, FontFormat, TextTransformStack
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


class BubbleEditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Image.new('RGB', (600, 400), 'white').save(os.path.join(self.directory.name, '01.png'))
        self.project = ProjImgTrans(self.directory.name)
        self.project.set_current_img('01.png')
        self.host = QWidget()
        self.canvas = Canvas(self.host)
        self.canvas.editor_index = 1
        self.canvas.imgtrans_proj = self.project
        canvas_patch = patch.object(SW, 'canvas', self.canvas)
        canvas_patch.start()
        self.addCleanup(canvas_patch.stop)
        self.addCleanup(setattr, C, 'active_format', C.active_format)
        with patch.object(shared, 'register_view_widget', create=True):
            self.panel = TextPanel(self.app, self.host)
        self.panel.formatpanel.global_format = FontFormat()
        self.manager = SceneTextManager(self.app, self.host, self.canvas, self.panel, parent=self.host)
        block = TextBlock([100, 80, 400, 280], _bounding_rect=[100, 80, 300, 200],
                          translation='Hello world. This text wraps inside the editable bubble.',
                          fontformat=FontFormat(font_family='DejaVu Sans', font_size=24))
        self.project.pages['01.png'] = [block]
        self.item = self.manager.addTextBlock(block)
        self.item.setSelected(True)
        self.manager._update_selection_panels([self.item])
        layout = QHBoxLayout(self.host)
        layout.addWidget(self.canvas.gv)
        layout.addWidget(self.panel)
        self.host.resize(1100, 700)
        self.host.show()
        self.host.activateWindow()
        self.canvas.gv.setFocus()
        QTest.qWaitForWindowActive(self.host)
        self.app.processEvents()
        self.control = self.canvas.txtblkShapeControl.bubble_control
        self._drag_end = None

    def tearDown(self) -> None:
        self.control.owner.cancelDrag()
        self.release_drag()
        self.canvas.clear_undostack(update_saved_step=True)
        self.manager.clearSceneTextitems()
        self.host.close()
        self.host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def enable(self) -> None:
        self.panel.formatpanel.bubbleShapeChecker.click()
        self.app.processEvents()
        self.assertTrue(self.control.isVisible())

    def drag(self, index: int, part: int, delta: QPointF,
             modifiers=Qt.KeyboardModifier.NoModifier, release=True) -> None:
        point = self.control.points[index * 3 + part]
        self.drag_from(self.canvas.gv.mapFromScene(point.scenePos()), delta, modifiers, release)

    def drag_from(self, start: QPoint, delta: QPointF,
                  modifiers=Qt.KeyboardModifier.NoModifier, release=True) -> None:
        end = start + delta.toPoint()
        viewport = self.canvas.gv.viewport()
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, modifiers, start)
        # Explicit buttons also exercise Qt 5, whose offscreen QTest move drops them.
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(end), QPointF(viewport.mapToGlobal(end)),
                            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, modifiers)
        QApplication.sendEvent(viewport, event)
        self._drag_end = end, modifiers
        self.app.processEvents()
        if release:
            self.release_drag()

    def release_drag(self) -> None:
        if self._drag_end is not None:
            end, modifiers = self._drag_end
            self._drag_end = None
            QTest.mouseRelease(self.canvas.gv.viewport(), Qt.MouseButton.LeftButton, modifiers, end)
            self.app.processEvents()

    def test_toggle_retains_shape_and_restores_controls_undo_and_save(self) -> None:
        self.assertFalse(self.item.blk.bubble_layout)
        self.enable()
        shape = self.item.blk.bubble_shape
        self.assertIsNotNone(self.item.layout.flow_profile)
        self.assertFalse(any(point.isVisible() for point in self.canvas.txtblkShapeControl.ctrlblock_group))
        self.panel.formatpanel.bubbleShapeChecker.click()
        self.assertFalse(self.control.isVisible())
        self.assertIsNone(self.item.layout.flow_profile)
        self.assertEqual(self.item.blk.bubble_shape, shape)
        self.assertTrue(all(point.isVisible() for point in self.canvas.txtblkShapeControl.ctrlblock_group))
        self.canvas.undo_textedit()
        self.assertTrue(self.panel.formatpanel.bubbleShapeChecker.isChecked())
        self.assertTrue(self.control.isVisible())
        self.manager.updateTextBlkList()
        self.project.save()
        loaded = ProjImgTrans(self.directory.name).pages['01.png'][0]
        self.assertTrue(loaded.bubble_layout)
        self.assertEqual(loaded.bubble_shape, shape)
        self.canvas.redo_textedit()
        self.assertFalse(self.panel.formatpanel.bubbleShapeChecker.isChecked())

    def test_anchor_drag_previews_without_mutating_model_then_commits_once(self) -> None:
        self.enable()
        before = deepcopy(self.item.blk.to_dict())
        steps = self.canvas.text_undo_stack.count()
        document_steps = self.item.document().availableUndoSteps()
        text = self.item.toHtml()
        self.drag(0, 0, QPointF(-30, -20), release=False)
        self.assertTrue(self.control.dragging)
        self.assertEqual(self.item.blk.to_dict(), before)
        self.assertNotEqual(self.item.layout.flow_profile.shape, self.item.blk.bubble_shape)
        self.assertEqual(self.canvas.text_undo_stack.count(), steps)
        self.release_drag()
        self.assertEqual(self.canvas.text_undo_stack.count(), steps + 1)
        self.assertEqual(self.item.document().availableUndoSteps(), document_steps)
        self.assertEqual(self.item.toHtml(), text)
        self.assertLess(self.item.absBoundingRect(qrect=True).left(), 100)
        after = self.item.blk.bubble_shape
        self.canvas.undo_textedit()
        self.assertEqual(self.item.blk.bubble_shape, before['bubble_shape'])
        self.assertEqual(self.item.absBoundingRect(qrect=True), QRectF(100, 80, 300, 200))
        self.canvas.redo_textedit()
        self.assertEqual(self.item.blk.bubble_shape, after)

    def test_rotation_grip_rotates_outline_and_text_with_one_undo(self) -> None:
        self.enable()
        view = self.canvas.gv
        owner = self.control.owner
        self.item.setAngle(-91)  # A clockwise quarter-turn crosses the atan2 boundary.
        shape = self.item.blk.bubble_shape
        original_text = self.item.toHtml()
        original_anchor = self.control.points[0].scenePos()
        steps = self.canvas.text_undo_stack.count()
        grip = self.control.rotation_handle
        start = view.mapFromScene(grip.scenePos())
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        self.assertEqual(self.canvas.text_undo_stack.count(), steps)
        center = view.mapFromScene(owner.visualCenterInScene())
        radial = start - center
        end = center + QPoint(-radial.y(), radial.x())
        self.drag_from(start, QPointF(end - start), release=False)
        self.assertAlmostEqual(self.item.rotation(), -1, delta=.6)
        self.assertEqual(self.item.angle, -91)
        self.assertLessEqual((view.mapFromScene(grip.scenePos()) - end).manhattanLength(), 2)
        self.assertNotEqual(original_anchor, self.control.points[0].scenePos())
        geometry_changed = Mock()
        self.item.visual_geometry_changed.connect(geometry_changed)
        self.release_drag()
        geometry_changed.assert_not_called()
        self.assertLessEqual((view.mapFromScene(grip.scenePos()) - end).manhattanLength(), 2)
        angle = self.item.rotation()
        self.assertEqual(self.item.angle, angle)
        self.assertEqual(self.item.blk.bubble_shape, shape)
        self.assertEqual(self.item.toHtml(), original_text)
        self.assertEqual(self.canvas.text_undo_stack.count(), steps + 1)
        self.canvas.undo_textedit()
        self.assertEqual(self.item.angle, -91)
        self.assertEqual(self.control.points[0].scenePos(), original_anchor)
        self.assertEqual(view.mapFromScene(grip.scenePos()), start)
        self.canvas.redo_textedit()
        self.assertEqual(self.item.angle, angle)
        owner.setBlkItem(None)
        owner.setBlkItem(self.item)
        self.assertLessEqual((view.mapFromScene(grip.scenePos()) - end).manhattanLength(), 2)
        self.manager.updateTextBlkList()
        self.project.save()
        loaded = ProjImgTrans(self.directory.name).pages['01.png'][0]
        self.assertEqual((loaded.angle, loaded.bubble_shape), (angle, shape))

    def test_rotation_draft_cancels_on_escape_save_and_target_change(self) -> None:
        self.enable()
        self.host.canvas = self.canvas
        owner = self.control.owner
        for constrained in (True, False):
            if not constrained:
                self.panel.formatpanel.bubbleShapeChecker.click()
            steps = self.canvas.text_undo_stack.count()
            for cancel in (lambda: MainWindow.shortcutEscape(self.host),
                           self.manager.updateTextBlkList,
                           lambda: owner.setBlkItem(None)):
                owner.setBlkItem(self.item)
                self.drag_from(self.rotation_start(), QPointF(45, 0), release=False)
                self.assertNotEqual(self.item.rotation(), self.item.angle)
                cancel()
                self.assertIsNone(self.canvas.mouseGrabberItem())
                self.release_drag()
                self.assertEqual(self.item.rotation(), self.item.angle)
                self.assertEqual(self.canvas.text_undo_stack.count(), steps)

    def rotation_start(self) -> QPoint:
        if self.control.isVisible():
            point = self.control.rotation_handle.scenePos()
        else:
            top = self.control.owner.ctrlblock_group[1]
            point = top.mapToScene(QPointF(0, -12))
        return self.canvas.gv.mapFromScene(point)

    def test_lost_mouse_grab_discards_preview_and_preserves_redo(self) -> None:
        self.enable()
        owner = self.control.owner
        for gesture in ('anchor', 'bubble rotation', 'rectangle rotation'):
            with self.subTest(gesture=gesture):
                if gesture == 'rectangle rotation':
                    self.panel.formatpanel.bubbleShapeChecker.click()
                self.drag_from(self.rotation_start(), QPointF(30, 0))
                self.canvas.undo_textedit()
                stack = self.canvas.text_undo_stack
                index = stack.index()
                shape = self.item.blk.bubble_shape
                if gesture == 'anchor':
                    self.drag(0, 0, QPointF(-15, 8), release=False)
                    self.assertTrue(self.control.dragging)
                else:
                    self.drag_from(self.rotation_start(), QPointF(45, 0), release=False)
                    self.assertTrue(owner.rotating)
                self.item.grabMouse()
                self.assertIs(self.canvas.mouseGrabberItem(), self.item)
                self.item.ungrabMouse()
                self.release_drag()
                self.assertFalse(owner.rotating or self.control.dragging)
                self.assertFalse(owner.angleLabel.isVisible())
                self.assertEqual(self.item.rotation(), self.item.angle)
                if self.item.blk.bubble_layout:
                    self.assertEqual(self.item.layout.flow_profile.shape, shape)
                self.assertEqual(stack.index(), index)
                self.assertTrue(stack.canRedo())

    def test_pending_edits_can_change_selection_or_mouse_capture_during_press(self) -> None:
        self.enable()
        replacement = self.manager.addTextBlock(TextBlock(
            [410, 80, 540, 180], _bounding_rect=[410, 80, 130, 100], translation='Second'))
        owner = self.control.owner
        for target in (None, replacement, self.item):
            for gesture in ('rotation', 'anchor'):
                owner.setBlkItem(self.item)
                settle = Mock(side_effect=(self.item.grabMouse if target is self.item
                                           else lambda: owner.setBlkItem(target)))
                self.canvas.bubble_edit_started.connect(settle)
                try:
                    if gesture == 'rotation':
                        self.drag_from(self.rotation_start(), QPointF(35, 0))
                    else:
                        self.drag(0, 0, QPointF(-15, 8))
                finally:
                    self.canvas.bubble_edit_started.disconnect(settle)
                    if self.canvas.mouseGrabberItem() is self.item:
                        self.item.ungrabMouse()
                settle.assert_called_once()
                self.assertIs(owner.blk_item, target)
                self.assertFalse(owner.rotating or self.control.dragging)
                self.assertEqual((self.item.rotation(), replacement.rotation()), (0, 0))

    def test_rotation_uses_release_position_in_both_modes(self) -> None:
        self.enable()
        view = self.canvas.gv
        for constrained in (True, False):
            if not constrained:
                self.panel.formatpanel.bubbleShapeChecker.click()
            self.item.setAngle(0)
            self.item.setRotation(12.5)  # The Qt pose can precede a project-state sync.
            start = self.rotation_start()
            steps = self.canvas.text_undo_stack.count()
            QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=start + QPoint(50, 0))
            self.assertGreater(self.item.angle, 15)
            self.assertEqual(self.canvas.text_undo_stack.count(), steps + 1)
            self.canvas.undo_textedit()
            self.assertEqual(self.item.angle, 12.5)

    def test_reset_angle_restores_each_block_and_skips_an_empty_change(self) -> None:
        self.enable()
        other = self.manager.addTextBlock(TextBlock(
            [410, 80, 540, 180], _bounding_rect=[410, 80, 130, 100], translation='Second'))
        for item, angle in ((self.item, -30), (other, 45)):
            item.setSelected(True)
            item.setAngle(angle)
        steps = self.canvas.text_undo_stack.count()
        self.manager.onResetAngle()
        self.assertEqual((self.item.angle, other.angle), (0, 0))
        self.manager.onResetAngle()
        self.assertEqual(self.canvas.text_undo_stack.count(), steps + 1)
        self.canvas.undo_textedit()
        self.assertEqual((self.item.angle, other.angle), (-30, 45))
        self.canvas.redo_textedit()
        self.assertEqual((self.item.angle, other.angle), (0, 0))

    def test_right_drag_from_rotation_grip_or_stem_begins_canvas_selection(self) -> None:
        self.enable()
        view = self.canvas.gv
        grip = view.mapFromScene(self.control.rotation_handle.scenePos())
        anchor = (self.control.points[0].scenePos() + self.control.points[3].scenePos()) / 2
        stem = (grip + view.mapFromScene(anchor)) / 2
        for start in (grip, stem):
            QTest.mousePress(view.viewport(), Qt.MouseButton.RightButton, pos=start)
            self.assertIsNotNone(self.canvas.rubber_band_origin)
            with patch.object(QMenu, 'exec', return_value=None):
                QTest.mouseRelease(view.viewport(), Qt.MouseButton.RightButton, pos=start)
            self.assertIsNone(self.canvas.rubber_band_origin)
            self.assertFalse(self.control.dragging)
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=stem)
        self.assertIsNone(self.control.owner.blk_item)

    def test_rotation_grip_keeps_screen_gap_and_avoids_tangent_and_viewport_edge(self) -> None:
        self.enable()
        view = self.canvas.gv
        grip = self.control.rotation_handle
        top = (self.control.points[0].scenePos() + self.control.points[3].scenePos()) / 2
        gap = view.mapFromScene(top) - view.mapFromScene(grip.scenePos())
        self.assertEqual(gap.x(), 0)
        self.assertGreater(gap.y(), 0)
        self.item.setAngle(90)
        right = (self.control.points[0].scenePos() + self.control.points[3].scenePos()) / 2
        self.assertEqual(view.mapFromScene(grip.scenePos()) - view.mapFromScene(right),
                         QPoint(gap.y(), 0))
        self.item.setAngle(0)
        size = grip.deviceTransform(view.viewportTransform()).mapRect(grip.boundingRect()).size()
        shape = self.item.blk.bubble_shape
        nodes = list(shape.nodes)
        nodes[0] = (0, 0, .5, -gap.y() / shape.frame[3], 0, 0)
        self.item.set_bubble_shape(BubbleShape(shape.frame, tuple(nodes)), True, QRectF(*shape.frame))
        self.control.select_anchor(0, False)
        tangent = self.control.points[1]
        tangent_pos = view.mapFromScene(tangent.scenePos())
        self.assertIs(view.itemAt(tangent_pos), tangent)
        self.assertIs(view.itemAt(view.mapFromScene(grip.scenePos())), grip)
        self.item.set_bubble_shape(shape, True, QRectF(*shape.frame))
        self.canvas.scaleImage(1.5)
        self.app.processEvents()
        top = (self.control.points[0].scenePos() + self.control.points[3].scenePos()) / 2
        self.assertEqual(view.mapFromScene(top) - view.mapFromScene(grip.scenePos()), gap)
        self.assertEqual(grip.deviceTransform(view.viewportTransform()).mapRect(grip.boundingRect()).size(), size)
        view.verticalScrollBar().setValue(view.verticalScrollBar().value() + view.mapFromScene(top).y() - 4)
        self.app.processEvents()
        self.assertLess(view.mapFromScene(top).y(), gap.y())
        position = view.mapFromScene(grip.scenePos())
        self.assertGreater(position.y(), view.mapFromScene(top).y())
        self.assertIs(view.itemAt(position), grip)

    def test_zoomed_rotated_edge_keeps_a_reachable_grip(self) -> None:
        self.enable()
        self.canvas.scaleImage(5)
        self.item.setAngle(30)
        view = self.canvas.gv
        edge = self.control.points[9].scenePos() * .375 + self.control.points[0].scenePos() * .625
        pan = view.mapFromScene(edge) - QPoint(63, 117)
        view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + pan.x())
        view.verticalScrollBar().setValue(view.verticalScrollBar().value() + pan.y())
        self.app.processEvents()
        grip = self.control.rotation_handle
        self.assertIs(view.itemAt(view.mapFromScene(grip.scenePos())), grip)

    def test_fallback_stem_points_outside_a_narrow_reversed_outline(self) -> None:
        self.enable()
        rect = BubbleShape.rectangle((100, 80, 10, 1500))
        shape = BubbleShape(rect.frame, tuple(reversed(rect.nodes)))
        self.item.set_bubble_shape(shape, True, QRectF(*shape.frame))
        self.canvas.gv.verticalScrollBar().setValue(330)
        self.app.processEvents()
        links = self.control.links.path()
        self.assertTrue(self.control.rotation_handle.isVisible())
        self.assertFalse(self.control.path().contains(links.pointAtPercent(.25)))

    def test_anchor_drag_preserves_other_scene_points_with_rotation_and_padding(self) -> None:
        self.item.setAngle(23)
        self.enable()
        self.item.setStrokeWidth(.1)
        self.canvas.txtblkShapeControl.requestGeometryRefresh()
        before = [self.control.points[i * 3].scenePos() for i in range(4)]
        self.drag(0, 0, QPointF(-22, -17))
        self.assertFalse(self.control.dragging)
        self.assertNotEqual(self.control.points[0].scenePos(), before[0])
        for index in (1, 2, 3):
            delta = self.control.points[index * 3].scenePos() - before[index]
            self.assertLess(abs(delta.x()) + abs(delta.y()), 1e-5)

    def test_handles_insert_delete_and_shortcuts_preserve_text_block(self) -> None:
        self.enable()
        self.drag(0, 0, QPointF(25, 15), Qt.KeyboardModifier.AltModifier)
        self.assertTrue(self.control.is_smooth(0))
        self.control.select_anchor(0, False)
        anchor = self.control.points[0].scenePos()
        incoming = self.control.points[1].scenePos() - anchor
        self.drag(0, 2, QPointF(12, -9))
        self.assertTrue(self.control.is_smooth(0))
        moved_incoming = self.control.points[1].scenePos() - self.control.points[0].scenePos()
        self.assertAlmostEqual(incoming.x() ** 2 + incoming.y() ** 2,
                               moved_incoming.x() ** 2 + moved_incoming.y() ** 2)
        opposite = self.control.points[1].scenePos()
        self.drag(0, 2, QPointF(10, -12), Qt.KeyboardModifier.AltModifier)
        delta = self.control.points[1].scenePos() - opposite
        self.assertLess(abs(delta.x()) + abs(delta.y()), 1e-5)
        self.assertFalse(self.control.is_smooth(0))
        self.control.convert_selected(False)
        self.assertEqual(self.item.blk.bubble_shape.nodes[0][:2], self.item.blk.bubble_shape.nodes[0][2:4])
        midpoint = self.control.mapToScene(self.control.path().pointAtPercent(.4))
        self.control.insert_anchor(midpoint)
        self.assertEqual(len(self.item.blk.bubble_shape.nodes), 5)
        self.host.canvas = self.canvas
        MainWindow.shortcutDelete(self.host)
        self.assertEqual(len(self.item.blk.bubble_shape.nodes), 4)
        self.assertIn(self.item, self.manager.textblk_item_list)
        self.control.select_anchor(0, False)
        position = self.control.points[0].scenePos()
        QTest.keyClick(self.canvas.gv.viewport(), Qt.Key.Key_Right)
        delta = self.control.points[0].scenePos() - position
        self.assertAlmostEqual(delta.x(), 1.)
        self.assertAlmostEqual(delta.y(), 0.)

    def test_draft_cancels_at_escape_save_selection_and_history_boundaries(self) -> None:
        self.enable()
        for cancel in (lambda: MainWindow.shortcutEscape(self.host),
                       self.panel.formatpanel.resolve_text_transform_edits_for_save,
                       self.manager.updateTextBlkList,
                       self.canvas.clear_states, self.canvas.undo_textedit,
                       lambda: self.canvas.txtblkShapeControl.setBlkItem(None)):
            self.host.canvas = self.canvas
            self.canvas.txtblkShapeControl.setBlkItem(self.item)
            before = self.item.blk.bubble_shape
            self.drag(0, 0, QPointF(-15, 10), release=False)
            self.assertTrue(self.control.dragging)
            cancel()
            self.assertFalse(self.control.dragging)
            self.assertIsNone(self.canvas.mouseGrabberItem())
            self.release_drag()
            self.control.finish_drag()
            if not self.item.blk.bubble_layout:
                self.canvas.redo_textedit()
            self.assertEqual(self.item.blk.bubble_shape, before)

    def test_curve_double_click_and_anchor_context_menu_use_canvas_input(self) -> None:
        self.enable()
        view = self.canvas.gv
        position = view.mapFromScene(self.control.mapToScene(self.control.path().pointAtPercent(.125)))
        QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.app.processEvents()
        self.assertEqual(len(self.item.blk.bubble_shape.nodes), 5)
        generic_menu = Mock()
        self.canvas.context_menu_requested.connect(generic_menu)
        position = view.mapFromScene(self.control.points[0].scenePos())
        with patch.object(QMenu, 'exec_', return_value=None) as menu:
            QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton, pos=position)
        menu.assert_called_once()
        generic_menu.assert_not_called()
        self.assertIsNone(self.canvas.rubber_band_origin)

    def test_only_left_double_click_on_anchor_converts_its_handles(self) -> None:
        self.enable()
        self.drag(0, 0, QPointF(25, 15), Qt.KeyboardModifier.AltModifier)
        shape = self.item.blk.bubble_shape
        view = self.canvas.gv
        for button, part in ((Qt.MouseButton.LeftButton, 1), (Qt.MouseButton.RightButton, 0)):
            position = view.mapFromScene(self.control.points[part].scenePos())
            with patch.object(QMenu, 'exec_', return_value=None):
                QTest.mouseClick(view.viewport(), button, pos=position)
                QTest.mouseDClick(view.viewport(), button, pos=position)
                QTest.mouseRelease(view.viewport(), button, pos=position)
            self.assertEqual(self.item.blk.bubble_shape, shape)
        position = view.mapFromScene(self.control.points[0].scenePos())
        QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertFalse(self.control.is_smooth(0))

    def test_multiple_anchors_and_minimum_node_count(self) -> None:
        self.enable()
        self.control.select_anchor(0, False)
        self.control.select_anchor(1, True)
        before = [self.control.points[i * 3].scenePos() for i in range(4)]
        self.drag(0, 0, QPointF(15, 7))
        for index in (0, 1):
            delta = self.control.points[index * 3].scenePos() - before[index]
            self.assertEqual(delta, QPointF(15, 7))
        for index in (2, 3):
            self.assertEqual(self.control.points[index * 3].scenePos(), before[index])
        self.control.selected_indices = {0}
        self.control.delete_selected()
        self.assertEqual(len(self.item.blk.bubble_shape.nodes), 3)
        self.control.select_anchor(0, False)
        self.control.delete_selected()
        self.assertEqual(len(self.item.blk.bubble_shape.nodes), 3)

    def test_block_shortcuts_cannot_interrupt_an_anchor_drag(self) -> None:
        self.enable()
        self.host.canvas = self.canvas
        self.drag(0, 0, QPointF(-15, 10), release=False)
        preview = self.item.absBoundingRect(qrect=True)
        steps = self.canvas.text_undo_stack.count()
        MainWindow.shortcutDelete(self.host)
        self.assertIn(self.item, self.manager.textblk_item_list)
        QTest.keyClick(self.canvas.gv.viewport(), Qt.Key.Key_Right)
        self.assertTrue(self.canvas.handle_transform_modal_shortcut(
            Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier))
        self.assertTrue(self.control.dragging)
        self.assertEqual(self.item.absBoundingRect(qrect=True), preview)
        self.assertEqual(self.canvas.text_undo_stack.count(), steps)
        self.release_drag()
        self.assertEqual(self.canvas.text_undo_stack.count(), steps + 1)

    def test_context_menu_does_not_act_on_a_replaced_target(self) -> None:
        self.enable()
        replacement = self.manager.addTextBlock(TextBlock(
            [400, 100, 500, 200], translation='Other block',
            bubble_shape=BubbleShape.rectangle((400, 100, 100, 100)), bubble_layout=True))
        for target in (None, replacement):
            self.canvas.txtblkShapeControl.setBlkItem(self.item)
            before = deepcopy(replacement.blk.to_dict())
            steps = self.canvas.text_undo_stack.count()

            def replace_target(menu: QMenu, position) -> object:
                self.canvas.txtblkShapeControl.setBlkItem(target)
                return menu.actions()[0]

            with patch.object(QMenu, 'exec_', replace_target):
                self.control.context_menu(self.host.pos(), self.item.scenePos())
            self.assertEqual(self.canvas.text_undo_stack.count(), steps)
            self.assertEqual(replacement.blk.to_dict(), before)

    def test_shape_and_text_edits_keep_independent_paired_history(self) -> None:
        self.enable()
        original_text = self.item.toPlainText()
        original_shape = self.item.blk.bubble_shape
        editor = self.manager.pairwidget_list[0].e_trans
        editor.setFocus()
        editor.insertPlainText('Added text. ')
        changed_text = editor.toPlainText()
        self.assertEqual(self.item.toPlainText(), changed_text)
        self.canvas.gv.setFocus()
        self.drag(0, 0, QPointF(-20, -15))
        changed_shape = self.item.blk.bubble_shape
        self.assertNotEqual(changed_shape, original_shape)
        self.canvas.undo_textedit()
        self.assertEqual(self.item.blk.bubble_shape, original_shape)
        self.assertEqual(self.item.toPlainText(), changed_text)
        self.canvas.undo_textedit()
        self.assertEqual(self.item.toPlainText(), original_text)
        self.assertEqual(editor.toPlainText(), original_text)
        self.canvas.redo_textedit()
        self.canvas.redo_textedit()
        self.assertEqual(self.item.blk.bubble_shape, changed_shape)
        self.assertEqual(self.item.toPlainText(), changed_text)
        self.assertEqual(editor.toPlainText(), changed_text)

    def test_font_shortcut_during_drag_cancels_draft_before_undo(self) -> None:
        self.enable()
        rect = self.item.absBoundingRect(qrect=True)
        shape = self.item.blk.bubble_shape
        before = self.item.toHtml()
        steps = self.canvas.text_undo_stack.count()
        self.drag(0, 0, QPointF(-20, -15), release=False)
        self.panel.formatpanel.toggle_bold()
        self.assertFalse(self.control.dragging)
        self.release_drag()
        self.assertEqual(self.item.absBoundingRect(qrect=True), rect)
        self.assertEqual(self.item.blk.bubble_shape, shape)
        self.assertNotEqual(self.item.toHtml(), before)
        self.assertEqual(self.canvas.text_undo_stack.count(), steps + 1)
        after = self.item.toHtml()
        self.canvas.undo_textedit()
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.item.absBoundingRect(qrect=True), rect)
        self.canvas.redo_textedit()
        self.assertEqual(self.item.toHtml(), after)
        self.assertEqual(self.item.blk.bubble_shape, shape)

    def test_new_detections_use_bubble_frame_but_saved_frames_take_precedence(self) -> None:
        shape = BubbleShape.rectangle((50, 20, 400, 300))
        for saved, expected in ((None, shape.frame), ([90, 70, 200, 100], (90, 70, 200, 100))):
            block = TextBlock([100, 80, 180, 160], _bounding_rect=saved,
                              bubble_shape=shape, bubble_layout=True, det_model='koharu')
            item = TextBlkItem(block, 1)
            self.addCleanup(item.deleteLater)
            self.assertEqual(item.absBoundingRect(qrect=True).getRect(), expected)
            self.assertEqual(block.xyxy, [100, 80, 180, 160])

    def test_automatic_layout_respects_disabled_constraint_and_manual_vertical_shapes(self) -> None:
        self.enable()
        self.item.setVertical(True)
        with patch.object(C.pcfg, 'let_fntsize_flag', 1):
            self.manager.onAutoLayoutTextblks()
        self.assertIsNotNone(self.item.layout.flow_profile)
        shape = self.item.blk.bubble_shape
        self.panel.formatpanel.bubbleShapeChecker.click()
        self.item.blk.det_model = 'koharu'
        fit_text_in_shape(self.item, auto_size=False, use_bubble_frame=True)
        self.assertFalse(self.item.blk.bubble_layout)
        self.assertEqual(self.item.blk.bubble_shape, shape)
        self.assertIsNone(self.item.layout.flow_profile)

    def test_warped_outline_tracks_constraint_and_export_contains_no_guides(self) -> None:
        self.enable()
        self.item.set_text_transform(TextTransformStack((BendTextTransform(.4),)))
        path = self.item.geometry_controller.bubble_outline_in_item()
        stroker = QPainterPathStroker()
        stroker.setWidth(2.)
        boundary = stroker.createStroke(path)
        source = bubble_path(self.item.blk.bubble_shape.nodes, self.item.logical_unpadded_rect())
        for step in range(40):
            mapped = self.item.geometry_controller.map_source_to_visual(source.pointAtPercent(step / 40))
            self.assertTrue(boundary.contains(mapped))
        self.canvas.updateCanvas()
        self.item.setSelected(True)
        owner = self.control.owner
        owner.setBlkItem(self.item)
        self.control.select_anchor(0, False)
        self.assertTrue(self.control.rotation_handle.isVisible())
        shown = self.canvas.render_result_img()
        owner.hide()  # MainWindow hides the whole frame before saving a result.
        self.assertEqual(shown, self.canvas.render_result_img())
        owner.show()
        self.app.processEvents()
        self.assertTrue(self.control.links.isVisible())
        self.assertTrue(self.control.rotation_handle.isVisible())
        owner.setBlkItem(None)
        self.item.setSelected(False)
        hidden = self.canvas.render_result_img()
        self.assertEqual(shown, hidden)

    def test_scene_clear_cancels_preview_and_releases_target(self) -> None:
        self.enable()
        self.drag(0, 0, QPointF(12, -8), release=False)
        reference = weakref.ref(self.item)
        self.canvas.clear_undostack()
        self.manager.clearSceneTextitems()
        self.item = None
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(self.control.owner.blk_item)
        self.assertFalse(self.control.dragging)
        self.assertIsNone(self.canvas.mouseGrabberItem())

    def test_disabling_vertical_constraint_reflows_like_a_fresh_rectangle(self) -> None:
        self.item.setVertical(True)
        self.enable()
        shape = BubbleShape((100, 80, 300, 200), (
            (1, .5, 1, .224, 1, .776), (.5, 1, .776, 1, .224, 1),
            (0, .5, 0, .776, 0, .224), (.5, 0, .224, 0, .776, 0),
        ))
        self.item.set_bubble_shape(shape, True, QRectF(*shape.frame))
        self.panel.formatpanel.bubbleShapeChecker.click()
        block = TextBlock(**deepcopy(self.item.blk.to_dict()))
        block.rich_text = self.item.toHtml()
        fresh = TextBlkItem(block, 1)
        self.addCleanup(fresh.deleteLater)
        def positions(item):
            return [(line.position(), line.textStart(), line.textLength())
                    for number in range(item.document().blockCount())
                    for layout in [item.document().findBlockByNumber(number).layout()]
                    for index in range(layout.lineCount()) for line in [layout.lineAt(index)]]
        self.assertEqual(positions(self.item), positions(fresh))


if __name__ == '__main__':
    unittest.main()
