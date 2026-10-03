import os
import tempfile
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PIL import Image
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QColor, QImage, QTextCursor
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawing_commands import RunBlkTransCommand, StrokeItemUndoCommand
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.text_engine.editing.manager import (
    SceneTextManager, SceneTextReplacementReason, TextPanel,
)
from ballontranslator.ui.text_engine.editing.widgets import SourceTextEdit
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
