import os
import tempfile
import unittest
import weakref
from itertools import product
from types import MethodType
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PIL import Image
from qtpy.QtCore import QCoreApplication, QEvent, QMimeData, QPoint, QPointF, Qt
from qtpy.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage, QTextCursor
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QStackedWidget, QTextEdit, QWidget

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawing_commands import RunBlkTransCommand, StrokeItemUndoCommand
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.mainwindowbars import BottomBar
from ballontranslator.ui.text_engine.editing.manager import (
    SceneTextManager, SceneTextReplacementReason, TextPanel,
)
from ballontranslator.ui.text_engine.editing.widgets import SourceTextEdit
from ballontranslator.ui.text_engine.transforms import grid_control
from ballontranslator.ui.text_engine.transforms.controls import TransformParameterPanel
from ballontranslator.utils import config as C, shared
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


class CanvasSaveStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for name in ('01.png', '02.png'):
            Image.new('RGB', (400, 200), 'white').save(
                os.path.join(self.directory.name, name)
            )
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
        self.manager = SceneTextManager(
            self.app, self.host, self.canvas, self.panel, parent=self.host,
        )
        block = TextBlock(
            [0, 0, 300, 120], _bounding_rect=[0, 0, 300, 120],
            translation='Hello\nworld', text=['source'],
            fontformat=FontFormat(font_family='DejaVu Sans', font_size=24),
        )
        self.project.pages['01.png'] = [block]
        self.item = self.manager.addTextBlock(block)
        self.item.setSelected(True)
        self.manager._update_selection_panels([self.item])
        layout = QHBoxLayout(self.host)
        layout.addWidget(self.canvas.gv)
        layout.addWidget(self.panel)
        self.host.show()
        self.host.activateWindow()
        self.app.processEvents()
        self.host.canvas = self.canvas
        self.host.opening_dir = False
        self.host._llm_context_dirty = False
        # Exercise the actual page-change dirty check; isolate image rendering.
        self.host.saveCurrentPage = Mock(side_effect=self._save_page)
        self._save_page()

    def tearDown(self) -> None:
        self.canvas.clear_undostack(update_saved_step=True)
        self.manager.clearSceneTextitems()
        self.host.close()
        self.host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def _save_page(
        self, update_scene_text: bool = True, save_proj: bool = True, **kwargs,
    ) -> None:
        if update_scene_text:
            self.manager.updateTextBlkList()
        if save_proj:
            self.project.save()
        self.canvas.update_saved_undostep()
        self.canvas.setProjSaveState(False)

    def _revisit_page(self) -> None:
        MainWindow.conditional_save(self.host)
        self.host.saveCurrentPage.assert_called_once()
        self.manager.clearSceneTextitems(SceneTextReplacementReason.PAGE_CHANGE)
        self.project.set_current_img('02.png')
        self.canvas.clear_undostack(update_saved_step=True)
        self.manager.populateSceneTextitems()
        self.manager.clearSceneTextitems(SceneTextReplacementReason.PAGE_CHANGE)
        self.project.set_current_img('01.png')
        self.canvas.clear_undostack(update_saved_step=True)
        self.manager.populateSceneTextitems()
        self.item = self.manager.textblk_item_list[0]

    def _edit(self, source: bool = False) -> SourceTextEdit:
        pair = self.manager.pairwidget_list[0]
        edit = pair.e_source if source else pair.e_trans
        edit.setFocus()
        self.app.processEvents()
        return edit

    def _add_transform(self, kind: str) -> TransformParameterPanel:
        self.panel.formatpanel.text_transform_session.add_transform(kind)
        panel = self.panel.formatpanel.texttransform_panel
        panel.set_expend_area(True, set_config=False)
        self.host.resize(1200, 900)
        self.app.processEvents()
        return panel.transform_panels[0]

    def _drag_canvas(self, start: QPoint, end: QPoint) -> None:
        viewport = self.canvas.gv.viewport()
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(viewport, end)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, end)
        self.app.processEvents()

    def test_transform_enter_keeps_focus_and_both_controls(self) -> None:
        control = self._add_transform('projective').controls['rotation_x']
        initial = self.item.blk.fontformat.text_transform
        control.editor.setFocus()
        control.editor.selectAll()
        QTest.keyClicks(control.editor, '12')
        self.assertTrue(self.canvas.txtblkProjectiveControl.isVisible())
        self.assertEqual(self.item.blk.fontformat.text_transform, initial)
        QTest.keyClick(control.editor, Qt.Key.Key_Return)
        changed = self.item.blk.fontformat.text_transform
        self.assertEqual(changed[0].rotation_x, 12)
        self.assertTrue(control.editor.hasFocus())
        self.assertTrue(self.canvas.txtblkShapeControl.isVisible())
        self.assertTrue(self.canvas.txtblkProjectiveControl.isVisible())
        self.assertEqual(self.panel.formatpanel.text_transform_session.selected_index, 0)
        self.canvas.undo_textedit()
        self.assertEqual(self.item.blk.fontformat.text_transform, initial)
        self.canvas.redo_textedit()
        self.assertEqual(self.item.blk.fontformat.text_transform, changed)
        control.editor.selectAll()
        QTest.keyClicks(control.editor, '15')
        self.assertTrue(self.canvas.txtblkProjectiveControl.isVisible())
        QTest.keyClick(control.editor, Qt.Key.Key_Escape)
        self.assertEqual(self.item.blk.fontformat.text_transform, changed)

    def test_transform_panel_drag_keeps_both_controls(self) -> None:
        control = self._add_transform('grid').controls['horizontal_divisions']
        initial = self.item.blk.fontformat.text_transform
        label = control.label
        QTest.mousePress(label, Qt.MouseButton.LeftButton)
        label.size_ctrl_changed.emit(8)
        self.assertTrue(self.canvas.txtblkGridControl.isVisible())
        self.assertEqual(self.item.blk.fontformat.text_transform, initial)
        QTest.mouseRelease(label, Qt.MouseButton.LeftButton)
        changed = self.item.blk.fontformat.text_transform
        self.assertNotEqual(changed, initial)
        self.assertTrue(self.canvas.txtblkGridControl.isVisible())
        self.assertTrue(self.canvas.txtblkShapeControl.isVisible())
        self.assertEqual(self.panel.formatpanel.text_transform_session.selected_index, 0)
        self.canvas.undo_textedit()
        self.assertEqual(self.item.blk.fontformat.text_transform, initial)
        self.canvas.redo_textedit()
        self.assertEqual(self.item.blk.fontformat.text_transform, changed)

    def test_pending_transform_value_preserves_clicked_canvas_handle(self) -> None:
        session = self.panel.formatpanel.text_transform_session
        cases = (
            ('projective', 'rotation_x', '12'),
            ('projective', 'rotation_x', '0'),
            ('grid', 'horizontal_divisions', '2'),
            ('grid', 'horizontal_divisions', '1'),
        )
        for kind, parameter, value in cases:
            with self.subTest(kind=kind, value=value):
                control = self._add_transform(kind).controls[parameter]
                control.editor.setFocus()
                control.editor.selectAll()
                QTest.keyClicks(control.editor, value)
                self.app.processEvents()
                if kind == 'projective':
                    overlay = self.canvas.txtblkProjectiveControl
                    point = overlay.mapToScene(
                        overlay.rings['z'].path().pointAtPercent(0.0)
                    )
                else:
                    overlay = self.canvas.txtblkGridControl
                    point = overlay.handles[0].scenePos()
                view = self.canvas.gv
                view.centerOn(overlay)
                self.app.processEvents()
                start = view.mapFromScene(point)
                self.assertTrue(view.viewport().rect().contains(start))
                self.assertTrue(control.editor.hasFocus())
                before_position = QPointF(self.item.pos())
                before_angle = self.item.rotation()
                self._drag_canvas(start, start + QPoint(20, 20))
                self.assertEqual(self.item.pos(), before_position)
                self.assertEqual(self.item.rotation(), before_angle)
                transform = self.item.blk.fontformat.text_transform[0]
                self.assertEqual(getattr(transform, parameter), int(value))
                if kind == 'projective':
                    self.assertNotEqual(transform.rotation_z, 0)
                else:
                    self.assertNotEqual(transform.control_points[0], (0, 0))
                self.canvas.undo_textedit()
                transform = self.item.blk.fontformat.text_transform[0]
                self.assertEqual(getattr(transform, parameter), int(value))
                if kind == 'projective':
                    self.assertEqual(transform.rotation_z, 0)
                else:
                    self.assertEqual(transform.control_points[0], (0, 0))
                session.remove_transform(0)

    def test_text_editing_suspends_gizmo_without_deselecting_card(self) -> None:
        session = self.panel.formatpanel.text_transform_session
        for kind in ('projective', 'grid'):
            with self.subTest(kind=kind):
                self._add_transform(kind)
                overlay = self.canvas.active_text_transform_control()
                view = self.canvas.gv
                view.centerOn(overlay)
                self.app.processEvents()
                self.item.startEdit()
                self.app.processEvents()
                self.assertTrue(self.item.isEditing())
                self.assertEqual(session.selected_index, 0)
                self.assertFalse(overlay.isVisible())
                before = self.item.toPlainText()
                view.setFocus()
                QTest.keyClicks(view.viewport(), 'gsr')
                self.assertNotEqual(self.item.toPlainText(), before)
                self.assertFalse(overlay.isVisible())
                self.item.endEdit(keep_focus=False)
                self.assertEqual(session.selected_index, 0)
                self.assertTrue(overlay.isVisible())
                self.assertTrue(self.canvas.txtblkShapeControl.isVisible())
                self.assertTrue(all(handle.isVisible() for handle in
                                    self.canvas.txtblkShapeControl.ctrlblock_group))
                session.remove_transform(0)

    def test_rectangle_gestures_keep_selected_transform(self) -> None:
        session = self.panel.formatpanel.text_transform_session
        for kind in ('projective', 'grid'):
            self._add_transform(kind)
            overlay = self.canvas.active_text_transform_control()
            view = self.canvas.gv
            view.centerOn(overlay)
            self.app.processEvents()
            for gesture in ('resize', 'rotate'):
                with self.subTest(kind=kind, gesture=gesture):
                    shape = self.canvas.txtblkShapeControl
                    handle = shape.ctrlblock_group[1]
                    local = handle.visible_rect.center()
                    if gesture == 'rotate':
                        local.setX(handle.rect().right() - 2)
                    point = handle.mapToScene(local)
                    if kind == 'projective':
                        self.assertTrue(any(ring.contains(ring.mapFromScene(point))
                                            for ring in overlay.rings.values()))
                    start = view.mapFromScene(point)
                    initial = self.item.blk.fontformat.text_transform
                    before = (self.item.rotation(), tuple(self.item.absBoundingRect()))
                    self._drag_canvas(start, start + QPoint(20, -20))
                    after = (self.item.rotation(), tuple(self.item.absBoundingRect()))
                    self.assertNotEqual(after, before)
                    self.assertEqual(after[0] == before[0], gesture == 'resize')
                    self.assertEqual(self.item.blk.fontformat.text_transform, initial)
                    self.assertEqual(session.selected_index, 0)
                    self.assertTrue(overlay.isVisible())
                    self.assertTrue(shape.isVisible())
                    self.canvas.undo_textedit()
                    self.assertEqual((self.item.rotation(),
                                      tuple(self.item.absBoundingRect())), before)
            session.remove_transform(0)

    def test_save_preserves_text_editing_and_selected_eraser(self) -> None:
        self.project.inpainted_array = self.project.img_array.copy()
        self.canvas.updateCanvas()
        self.item.setSelected(True)
        self.manager._update_selection_panels([self.item])
        self.host.st_manager = self.manager
        self.host.imgtrans_proj = self.project
        self.host.rightComicTransStackPanel = Mock()
        self.host.rightComicTransStackPanel.isHidden.return_value = False
        self.host.imsave_thread = Mock()
        session = self.panel.formatpanel.text_transform_session
        eraser = self.panel.formatpanel.texteffect_panel.mask_brush_button
        for kind, erasing in product(('projective', 'grid'), (False, True)):
            with self.subTest(kind=kind, erasing=erasing):
                self._add_transform(kind)
                overlay = self.canvas.active_text_transform_control()
                self.item.startEdit()
                self.assertTrue(self.item.isEditing())
                if erasing:
                    eraser.click()
                    self.assertTrue(self.canvas.alpha_mask_edit_session.active)
                with (
                    patch.object(C.pcfg, 'imgtrans_textblock', False),
                    patch.object(grid_control, 'GridControlPointItem',
                                 wraps=grid_control.GridControlPointItem) as handles,
                ):
                    MainWindow.saveCurrentPage(self.host, restore_interface=True)
                    handles.assert_not_called()
                self.app.processEvents()
                self.assertTrue(self.item.isEditing())
                self.assertEqual(session.selected_index, 0)
                self.assertFalse(overlay.isVisible())
                self.assertEqual(self.canvas.alpha_mask_edit_session.active, erasing)
                self.item.endEdit()
                self.assertEqual(overlay.isVisible(), not erasing)
                self.canvas.alpha_mask_edit_session.deactivate()
                session.remove_transform(0)

    def test_explicit_transform_edits_exit_eraser(self) -> None:
        session = self.panel.formatpanel.text_transform_session
        eraser = self.panel.formatpanel.texteffect_panel.mask_brush_button
        for kind, parameter, value in (
            ('projective', 'rotation_x', '12'),
            ('grid', 'horizontal_divisions', '2'),
        ):
            with self.subTest(kind=kind):
                card = self._add_transform(kind)
                overlay = self.canvas.active_text_transform_control()
                eraser.click()
                self.assertTrue(self.canvas.alpha_mask_edit_session.active)
                card.card_clicked.emit(0)
                self.assertTrue(self.canvas.alpha_mask_edit_session.active)
                card.card_clicked.emit(0)
                self.assertFalse(self.canvas.alpha_mask_edit_session.active)
                self.assertTrue(overlay.isVisible())
                eraser.click()
                control = card.controls[parameter]
                control.editor.setFocus()
                control.editor.selectAll()
                QTest.keyClicks(control.editor, value)
                QTest.keyClick(control.editor, Qt.Key.Key_Return)
                self.assertFalse(self.canvas.alpha_mask_edit_session.active)
                self.assertTrue(overlay.isVisible())
                self.assertEqual(getattr(self.item.blk.fontformat.text_transform[0],
                                         parameter), int(value))
                session.remove_transform(0)

    def test_leaving_text_mode_clears_transform_controls(self) -> None:
        self.project.inpainted_array = self.project.img_array.copy()
        self.canvas.updateCanvas()
        self.host.st_manager = self.manager
        self.host.bottomBar = BottomBar(self.host)
        self.host.rightComicTransStackPanel = QStackedWidget(self.host)
        self.host.rightComicTransStackPanel.addWidget(QWidget())
        self.host.rightComicTransStackPanel.addWidget(self.panel)
        self.host.rightComicTransStackPanel.setCurrentIndex(1)
        self.host.layout().addWidget(self.host.rightComicTransStackPanel)
        self.host.layout().addWidget(self.host.bottomBar)
        self.host.setTextEditMode = MethodType(MainWindow.setTextEditMode, self.host)
        self.host.setTextBlockMode = MethodType(MainWindow.setTextBlockMode, self.host)
        self.host.setPaintMode = MethodType(MainWindow.setPaintMode, self.host)
        self.host.bottomBar.textedit_checkchanged.connect(self.host.setTextEditMode)
        self.host.bottomBar.paintmode_checkchanged.connect(self.host.setPaintMode)
        self.host.rightComicTransStackPanel.currentChanged.connect(
            MethodType(MainWindow.on_transpanel_changed, self.host)
        )
        checker = self.host.bottomBar.texteditChecker
        checker.setChecked(True)
        self.host.resize(1200, 900)
        session = self.panel.formatpanel.text_transform_session
        shape = self.canvas.txtblkShapeControl
        view = self.canvas.gv
        with (
            patch.object(C.pcfg, 'imgtrans_textedit', True),
            patch.object(C.pcfg, 'imgtrans_textblock', False),
            patch.object(C.pcfg, 'imgtrans_paintmode', False),
        ):
            for kind, editing, mode in product(
                ('projective', 'grid'), (False, True), ('off', 'paint'),
            ):
                with self.subTest(kind=kind, editing=editing, mode=mode):
                    self.manager._update_selection_panels([self.item])
                    self._add_transform(kind)
                    handle = shape.ctrlblock_group[1]
                    start = view.mapFromScene(
                        handle.mapToScene(handle.visible_rect.center())
                    )
                    before = tuple(self.item.absBoundingRect())
                    if editing:
                        self.item.startEdit()
                        self.assertTrue(self.item.isEditing())
                    button = self.host.bottomBar.paintChecker if mode == 'paint' else checker
                    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                    self.app.processEvents()
                    self.assertFalse(self.canvas.textLayer.isVisible())
                    self.assertFalse(shape.isVisible())
                    self.assertIsNone(shape.blk_item)
                    self.assertIsNone(self.canvas.active_text_transform_control())
                    self.assertFalse(self.item.isEditing())
                    if mode == 'off':
                        self._drag_canvas(start, start + QPoint(20, -20))
                        self.assertEqual(tuple(self.item.absBoundingRect()), before)
                    QTest.mouseClick(checker, Qt.MouseButton.LeftButton)
                    self.item.setSelected(True)
                    self.manager._update_selection_panels([self.item])
                    session.remove_transform(0)

    def test_transform_hover_and_deselection_restore_rectangle_owner(self) -> None:
        other = self.manager.addTextBlock(TextBlock(
            [320, 0, 400, 100], _bounding_rect=[320, 0, 80, 100],
            translation='Other block', fontformat=FontFormat(font_size=18),
        ))
        session = self.panel.formatpanel.text_transform_session
        shape = self.canvas.txtblkShapeControl
        for kind in ('projective', 'grid'):
            with self.subTest(kind=kind):
                self._add_transform(kind)
                for hovered in (self.item, other, self.item, other):
                    self.manager.onTextBlkItemHoverEnter(hovered.idx)
                    self.assertIs(shape.blk_item, hovered)
                    self.assertTrue(shape.isVisible())
                    self.assertIs(self.canvas.active_transform_control_item(), self.item)
                session.select_transform(-1)
                self.assertIsNone(self.canvas.active_text_transform_control())
                self.assertIs(shape.blk_item, self.item)
                self.assertTrue(shape.isVisible())
                session.remove_transform(0)

    def test_failed_export_does_not_rebuild_suspended_grid(self) -> None:
        self.project.inpainted_array = self.project.img_array.copy()
        self.canvas.updateCanvas()
        self._add_transform('grid')
        self.item.startEdit()
        self.assertTrue(self.item.isEditing())
        with (
            patch.object(self.canvas, 'render', side_effect=RuntimeError('export failed')),
            patch.object(grid_control, 'GridControlPointItem',
                         wraps=grid_control.GridControlPointItem) as handles,
        ):
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                self.canvas.render_result_img()
            handles.assert_not_called()
        self.assertTrue(self.canvas.txtblkShapeControl.isVisible())
        self.assertFalse(self.canvas.txtblkGridControl.isVisible())
        self.assertEqual(self.panel.formatpanel.text_transform_session.selected_index, 0)
        self.item.startEdit()
        self.item.endEdit()
        self.assertTrue(self.canvas.txtblkGridControl.isVisible())

    def test_rectangle_press_batches_multi_selection_refresh(self) -> None:
        items = [self.item]
        for index in range(1, 40):
            items.append(self.manager.addTextBlock(TextBlock(
                [20 * index, 0, 20 * index + 150, 80],
                _bounding_rect=[20 * index, 0, 150, 80],
                translation='Some comic text', fontformat=FontFormat(font_size=18),
            )))
        self.manager.set_blkitems_selection(True, items)
        shape = self.canvas.txtblkShapeControl
        shape.setBlkItem(self.item)
        view = self.canvas.gv
        view.setFocus()
        self.app.processEvents()
        handle = shape.ctrlblock_group[0]
        point = view.mapFromScene(handle.mapToScene(handle.visible_rect.center()))
        with patch.object(self.manager, '_update_selection_panels',
                          wraps=self.manager._update_selection_panels) as refresh:
            QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier, point)
            self.assertEqual(self.canvas.selected_text_items(), [self.item])
            self.assertLessEqual(refresh.call_count, 2)

    def _drop_text(self, edit: SourceTextEdit, position: int, text: str) -> None:
        cursor = QTextCursor(edit.document())
        cursor.setPosition(position)
        point = edit.cursorRect(cursor).center()
        mime = QMimeData()
        mime.setText(text)
        enter = QDragEnterEvent(
            point, Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        self.app.sendEvent(edit.viewport(), enter)
        event = QDropEvent(
            QPointF(point), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        self.app.sendEvent(edit.viewport(), event)
        self.assertTrue(event.isAccepted())

    def test_unfocused_drops_are_saved_and_undoable(self) -> None:
        for source in (False, True):
            with self.subTest(source=source):
                edit = self._edit(source)
                before = edit.toPlainText()
                self.canvas.gv.setFocus()
                self.app.processEvents()
                self.assertFalse(edit.hasFocus())
                self._save_page()
                count = self.canvas.text_undo_stack.count()
                self._drop_text(edit, 0, '\U0001f600new\n')
                after = '\U0001f600new\n' + before
                self.assertEqual(edit.toPlainText(), after)
                self.assertEqual(self.canvas.text_undo_stack.count(), count + 1)
                self.assertTrue(self.canvas.projstate_unsaved)
                if not source:
                    self.assertEqual(self.item.toPlainText(), after)
                self.canvas.undo_textedit()
                self.assertEqual(edit.toPlainText(), before)
                self.assertFalse(self.canvas.projstate_unsaved)
                if not source:
                    self.assertEqual(self.item.toPlainText(), before)
                self.canvas.redo_textedit()
                self._save_page()
                reopened = ProjImgTrans(self.directory.name).pages['01.png'][0]
                saved = reopened.get_text() if source else reopened.translation
                self.assertEqual(saved, after)

    def test_internal_drops_keep_paired_undo_and_redo_synchronized(self) -> None:
        def move_selection(edit: QTextEdit, event: QDropEvent) -> None:
            target = QTextCursor(edit.document())
            target.movePosition(QTextCursor.MoveOperation.End)
            target.beginEditBlock()
            edit.textCursor().removeSelectedText()
            edit.setTextCursor(target)
            edit.insertFromMimeData(event.mimeData())
            target.endEditBlock()
            event.acceptProposedAction()

        edit = self._edit()
        for before, after in (('abcd', 'bcda'), ('aaaa', 'aaaa')):
            with self.subTest(before=before):
                previous = edit.toPlainText()
                edit.selectAll()
                edit.insertPlainText(before)
                cursor = edit.textCursor()
                cursor.setPosition(0)
                cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
                edit.setTextCursor(cursor)
                # Offscreen Qt cannot start native drags; retain its grouped move.
                with patch.object(QTextEdit, 'dropEvent', move_selection):
                    self._drop_text(edit, len(before), before[0])
                for operation, expected in (
                    (self.canvas.undo_textedit, before),
                    (self.canvas.undo_textedit, previous),
                    (self.canvas.redo_textedit, before),
                    (self.canvas.redo_textedit, after),
                ):
                    operation()
                    self.assertEqual(edit.toPlainText(), expected)
                    self.assertEqual(self.item.toPlainText(), expected)

    def test_repeated_text_drop_preserves_inline_formatting(self) -> None:
        edit = self._edit()
        edit.selectAll()
        edit.insertPlainText('aabb')
        self.item.startEdit()
        cursor = self.item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        self.item.setTextCursor(cursor)
        self.item.setFontColor((255, 0, 0), set_selected=True)
        self.item.endEdit()
        edit.setFocus()
        self.app.processEvents()
        self._drop_text(edit, 1, 'a')
        self.assertEqual(self.item.toPlainText(), 'aaabb')
        for position, color in enumerate(('black', 'black', 'red', 'black', 'black')):
            cursor.setPosition(position)
            cursor.setPosition(position + 1, QTextCursor.MoveMode.KeepAnchor)
            self.assertEqual(cursor.charFormat().foreground().color(), QColor(color))

    def test_replacement_color_is_saved_when_undo_branch_is_replaced(self) -> None:
        self.item.setFontColor((255, 0, 0))
        self._save_page()
        self.canvas.undo()
        self.item.setFontColor((0, 0, 255))
        self.assertTrue(self.canvas.text_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)
        self._revisit_page()
        self.assertEqual(self.item.get_fontformat().frgb, [0, 0, 255])
        reopened = ProjImgTrans(self.directory.name)
        self.assertIn('#0000ff', reopened.pages['01.png'][0].rich_text)

    def test_deleted_newline_is_saved_after_undo_and_replacement_edit(self) -> None:
        self.item.setFontColor((255, 0, 0))
        self._save_page()
        self.canvas.undo()
        edit = self._edit()
        cursor = edit.textCursor()
        cursor.setPosition(5)
        edit.setTextCursor(cursor)
        QTest.keyClick(edit, Qt.Key.Key_Delete)
        self.assertEqual(self.item.toPlainText(), 'Helloworld')
        self._revisit_page()
        self.assertEqual(self.item.toPlainText(), 'Helloworld')
        reopened = ProjImgTrans(self.directory.name)
        self.assertEqual(reopened.pages['01.png'][0].translation, 'Helloworld')

    def test_editor_shortcuts_update_save_state_and_redo_to_saved_state(self) -> None:
        self.item.setFontColor((255, 0, 0))
        self._save_page()
        edit = self._edit()
        QTest.keyClick(edit, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertTrue(self.canvas.projstate_unsaved)
        self.assertTrue(self.canvas.text_change_unsaved())
        QTest.keyClick(edit, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        self.assertFalse(self.canvas.projstate_unsaved)
        self.assertFalse(self.canvas.text_change_unsaved())

    def test_merged_typing_after_save_stays_dirty_through_undo_and_redo(self) -> None:
        for source in (False, True):
            with self.subTest(source=source):
                edit = self._edit(source)
                edit.moveCursor(QTextCursor.MoveOperation.End)
                QTest.keyClicks(edit, 'abc')
                self._save_page()
                command_count = self.canvas.text_undo_stack.count()
                QTest.keyClicks(edit, 'd')
                self.assertEqual(self.canvas.text_undo_stack.count(), command_count)
                self.assertTrue(self.canvas.text_change_unsaved())
                self.assertTrue(self.canvas.projstate_unsaved)
                self.canvas.undo_textedit()
                self.assertTrue(self.canvas.projstate_unsaved)
                self.canvas.redo_textedit()
                self.assertTrue(self.canvas.projstate_unsaved)
                self._save_page()
                self.assertFalse(self.canvas.text_change_unsaved())

    def test_merged_typing_can_undo_to_an_earlier_saved_state(self) -> None:
        edit = self._edit()
        edit.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(edit, 'abc')
        self.canvas.undo()
        self.assertEqual(self.item.toPlainText(), 'Hello\nworld')
        self.assertFalse(self.canvas.text_change_unsaved())
        self.assertFalse(self.canvas.projstate_unsaved)

    def test_canvas_typing_after_save_is_saved_on_page_change(self) -> None:
        self.item.startEdit()
        cursor = self.item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.item.setTextCursor(cursor)
        self.canvas.gv.setFocus()
        self.app.processEvents()
        QTest.keyClicks(self.canvas.gv.viewport(), 'abc')
        self._save_page()
        QTest.keyClicks(self.canvas.gv.viewport(), 'd')
        self.assertTrue(self.canvas.text_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)
        self._revisit_page()
        self.assertEqual(self.item.toPlainText(), 'Hello\nworldabcd')

    def test_typing_after_whole_style_reset_is_saved_and_undoable(self) -> None:
        edit = self._edit()
        edit.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(edit, 'abc')
        QTest.keyClick(edit, Qt.Key.Key_Return)
        QTest.keyClicks(edit, 'xyz')
        self.host.setFocus()
        self.app.processEvents()
        self.manager.apply_fontformat(FontFormat(font_size=26))
        self._save_page()
        edit = self._edit()
        edit.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(edit, 'd')
        self.assertTrue(self.canvas.text_change_unsaved())
        self.canvas.undo_textedit()
        self.assertEqual(edit.toPlainText(), 'Hello\nworldabc\nxyz')
        self.assertFalse(self.canvas.text_change_unsaved())
        self.canvas.redo_textedit()
        self._revisit_page()
        self.assertEqual(self.item.toPlainText(), 'Hello\nworldabc\nxyzd')

    def test_canvas_typing_after_style_undo_restores_document_history_baseline(self) -> None:
        self.item.startEdit()
        self.canvas.gv.setFocus()
        self.app.processEvents()
        cursor = self.item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.item.setTextCursor(cursor)
        QTest.keyClicks(self.canvas.gv.viewport(), 'abc')
        self.item.endEdit(keep_focus=False)
        self.host.setFocus()
        self.app.processEvents()
        before_style = self.canvas.text_undo_stack.count()
        self.manager.apply_fontformat(FontFormat(font_size=26))
        self.assertEqual(self.canvas.text_undo_stack.count(), before_style + 1)
        self.item.startEdit()
        self.canvas.gv.setFocus()
        self.app.processEvents()
        self.canvas.undo()
        self.assertEqual(self.canvas.text_undo_stack.count(), before_style + 1)
        self.assertEqual(self.canvas.text_undo_stack.index(), before_style)
        self.canvas.redo()
        self.assertEqual(self.canvas.text_undo_stack.index(), before_style + 1)
        self.canvas.undo()
        self._save_page()
        self.item.startEdit()
        self.canvas.gv.setFocus()
        self.app.processEvents()
        cursor = self.item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.item.setTextCursor(cursor)
        QTest.keyClicks(self.canvas.gv.viewport(), 'd')
        self.assertTrue(self.canvas.text_change_unsaved())
        self._revisit_page()
        self.assertEqual(self.item.toPlainText(), 'Hello\nworldabcd')

    def test_block_inpaint_replay_saves_pixels_and_survives_drawing_history_clear(self) -> None:
        self.project.mask_array = np.zeros((200, 400), np.uint8)
        self.project.inpainted_array = np.full_like(self.project.img_array, 255)
        self.item.blk.region_inpaint_dict = {
            'inpaint_rect': [0, 0, 10, 10],
            'inpainted': np.zeros((10, 10, 3), np.uint8),
            'mask': np.full((10, 10), 255, np.uint8),
        }
        self.canvas.push_text_command(RunBlkTransCommand(
            self.canvas, [self.item], self.manager.pairwidget_list, 3,
        ))
        self._save_page()
        self.canvas.undo()
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertEqual(self.project.inpainted_array[0, 0, 0], 255)
        self.assertEqual(self.project.mask_array[0, 0], 0)
        MainWindow.conditional_save(self.host)
        self.assertFalse(self.host.saveCurrentPage.call_args.kwargs['save_rst_only'])
        self.canvas.clear_draw_stack()
        self.canvas.redo()
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertEqual(self.project.inpainted_array[0, 0, 0], 0)
        self.assertEqual(self.project.mask_array[0, 0], 255)
        command_ref = weakref.ref(self.canvas.text_undo_stack.command(0))
        self.canvas.clear_text_stack()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.assertIsNone(command_ref())

    def test_merged_typing_below_another_canvas_command_invalidates_saved_state(self) -> None:
        edit = self._edit(source=True)
        edit.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(edit, 'abc')
        self._save_page()
        self.item.setFontColor((255, 0, 0))
        QTest.keyClicks(edit, 'd')
        self.canvas.undo()
        self.assertEqual(edit.toPlainText(), 'sourceabcd')
        self.assertTrue(self.canvas.text_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)

    def test_text_deletion_pixel_restore_stays_dirty_outside_draw_history(self) -> None:
        self.project.mask_array = np.full((200, 400), 255, np.uint8)
        self.project.inpainted_array = np.zeros_like(self.project.img_array)
        self.manager.onDeleteBlkItems(1)
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertEqual(self.project.mask_array[10, 10], 0)
        self._save_page()
        self.canvas.undo()
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)
        self.assertEqual(self.project.mask_array[10, 10], 255)
        MainWindow.conditional_save(self.host)
        self.assertFalse(self.host.saveCurrentPage.call_args.kwargs['save_rst_only'])
        self.canvas.redo()
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.assertTrue(self.canvas.projstate_unsaved)

    def test_empty_history_shortcuts_do_not_dirty_saved_state(self) -> None:
        for mode in (0, 1):
            self.canvas.editor_index = mode
            self.canvas.undo()
            self.canvas.redo()
        self.canvas.undo_textedit()
        self.canvas.redo_textedit()
        self.assertFalse(self.canvas.projstate_unsaved)
        self.assertFalse(self.canvas.text_change_unsaved())
        self.assertFalse(self.canvas.draw_change_unsaved())

    def test_drawing_branch_is_dirty_and_history_clear_preserves_unsaved_edits(self) -> None:
        self.canvas.editor_index = 0
        image = QImage(5, 5, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor('red'))
        self.canvas.push_draw_command(StrokeItemUndoCommand(
            self.canvas.drawingLayer, (0, 0, 5, 5), image,
        ))
        self._save_page()
        self.canvas.undo()
        self.canvas.redo()
        self.assertFalse(self.canvas.projstate_unsaved)
        self.canvas.undo()
        self.canvas.push_draw_command(StrokeItemUndoCommand(
            self.canvas.drawingLayer, (10, 10, 5, 5), image,
        ))
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.canvas.clear_draw_stack()
        self.assertTrue(self.canvas.draw_change_unsaved())
        self.canvas.editor_index = 1
        self.item.setFontColor((0, 255, 0))
        self.canvas.clear_text_stack()
        self.assertTrue(self.canvas.text_change_unsaved())
        self.canvas.clear_undostack(update_saved_step=True)
        self.assertFalse(self.canvas.draw_change_unsaved())
        self.assertFalse(self.canvas.text_change_unsaved())


if __name__ == '__main__':
    unittest.main()
