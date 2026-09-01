import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QWidget

try:
    from qtpy.QtWidgets import QUndoStack
except ImportError:
    from qtpy.QtGui import QUndoStack

from ballontranslator.ui.llm_context_editor import LLMContextEditor
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.bulk_page_summary_editor import (
    BulkPageSummaryDialog,
)
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class LLMContextEditorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _project() -> ProjImgTrans:
        project = ProjImgTrans()
        project.directory = '/unused'
        project.pages = {'001.png': [], '002.png': []}
        project._image_info = {
            '001.png': {'finish_code': 0},
            '002.png': {'finish_code': 0},
        }
        project.current_img = '001.png'
        return project

    def test_editors_write_project_owned_text_and_follow_the_page(self):
        project = self._project()
        panel = LLMContextEditor(QWidget(), project)

        panel.summary_editor.setPlainText('Page one.\nUser note.')
        panel.memory_editor.setPlainText('Shared memory.')

        self.assertEqual(
            project.get_llm_visual_summary('001.png')['text'],
            'Page one.\nUser note.',
        )
        self.assertEqual(
            project.get_llm_compact_memory()['text'],
            'Shared memory.',
        )

        project.set_llm_visual_summary_text('002.png', 'Page two.')
        panel.set_page('002.png')
        self.assertEqual(panel.summary_editor.toPlainText(), 'Page two.')
        self.assertIn('Shared memory.', panel.memory_editor.toPlainText())
        panel.deleteLater()

    def test_empty_project_disables_both_editors(self):
        panel = LLMContextEditor(QWidget(), ProjImgTrans())

        self.assertFalse(panel.summary_editor.isEnabled())
        self.assertFalse(panel.memory_editor.isEnabled())
        panel.deleteLater()

    def test_bulk_summary_commit_is_one_undoable_project_change(self):
        project = self._project()
        project.set_llm_visual_summary(
            '001.png',
            {'version': 1, 'text': 'Old one.', 'marker': 'preserved'},
        )
        untouched = '  Old two.\n\n  \n'
        project.set_llm_visual_summary_text('002.png', untouched)
        project.pages['003.png'] = []
        project._image_info['003.png'] = {'finish_code': 0}
        project.set_llm_visual_summary_text('003.png', 'Old three.')
        stack = QUndoStack()
        push_command = Mock(side_effect=stack.push)
        changed = Mock()
        panel = LLMContextEditor(QWidget(), project)
        panel.set_summary_command_pusher(push_command)
        panel.project_changed.connect(changed)

        panel._open_bulk_summary_editor()
        dialog = panel.findChild(BulkPageSummaryDialog)
        self.assertIsNotNone(dialog)
        self.app.processEvents()
        self.assertTrue(dialog.editor.hasFocus())
        self.assertLess(
            dialog.editor.toPlainText().index('### 001.png'),
            dialog.editor.toPlainText().index('### 002.png'),
        )
        project.set_llm_visual_summary_text(
            '002.png',
            'External edit before commit.',
        )
        dialog.editor.setPlainText(
            '### 001.png\nFirst.\n\n'
            '### unknown.png\nIgnored.\n\n'
            '### 001.PNG\nWrong case.\n\n'
            '### 001.png\nLast wins.\n\n\n'
            '### 002.png   \n\n' + untouched
        )
        panel.close_bulk_summary_editor()
        self.app.processEvents()

        push_command.assert_called_once()
        self.assertEqual(stack.count(), 1)
        self.assertEqual(
            project.get_llm_visual_summary('001.png')['text'],
            'Last wins.',
        )
        self.assertEqual(
            project.get_llm_visual_summary('002.png')['text'],
            untouched,
        )
        self.assertIsNone(project.get_llm_visual_summary('003.png'))
        self.assertEqual(panel.summary_editor.toPlainText(), 'Last wins.')

        stack.undo()
        self.assertEqual(
            project.get_llm_visual_summary('001.png'),
            {'version': 1, 'text': 'Old one.', 'marker': 'preserved'},
        )
        self.assertEqual(
            project.get_llm_visual_summary('002.png')['text'],
            'External edit before commit.',
        )
        self.assertEqual(
            project.get_llm_visual_summary('003.png')['text'],
            'Old three.',
        )
        self.assertEqual(panel.summary_editor.toPlainText(), 'Old one.')

        stack.redo()
        self.assertEqual(
            project.get_llm_visual_summary('001.png')['text'],
            'Last wins.',
        )
        self.assertEqual(
            project.get_llm_visual_summary('002.png')['text'],
            untouched,
        )
        self.assertIsNone(project.get_llm_visual_summary('003.png'))
        changed.assert_not_called()
        panel.deleteLater()

    def test_project_replacement_discards_stale_bulk_text(self):
        project = self._project()
        stack = QUndoStack()
        push_command = Mock(side_effect=stack.push)
        panel = LLMContextEditor(QWidget(), project)
        panel.set_summary_command_pusher(push_command)
        panel._open_bulk_summary_editor()
        dialog = panel.findChild(BulkPageSummaryDialog)
        self.assertIsNotNone(dialog)
        dialog.editor.setPlainText(
            '### 001.png\n\nStale project text.'
        )

        project.pages = {'001.png': []}
        project._image_info = {'001.png': {'finish_code': 0}}
        project.set_llm_visual_summary_text('001.png', 'New project text.')
        panel.set_project(project)
        self.app.processEvents()

        push_command.assert_not_called()
        self.assertEqual(
            project.get_llm_visual_summary('001.png')['text'],
            'New project text.',
        )
        panel.deleteLater()

    def test_bulk_editor_can_reopen_after_deferred_delete(self):
        project = self._project()
        panel = LLMContextEditor(QWidget(), project)
        panel.set_summary_command_pusher(Mock())

        panel._open_bulk_summary_editor()
        first_dialog = panel.findChild(BulkPageSummaryDialog)
        self.assertIsNotNone(first_dialog)
        first_dialog.close()
        QCoreApplication.sendPostedEvents(
            None,
            QEvent.Type.DeferredDelete,
        )
        self.app.processEvents()

        self.assertIsNone(panel.findChild(BulkPageSummaryDialog))
        panel._open_bulk_summary_editor()
        second_dialog = panel.findChild(BulkPageSummaryDialog)
        self.assertIsNotNone(second_dialog)
        self.assertIsNot(second_dialog, first_dialog)
        second_dialog.discard_and_close()
        panel.deleteLater()

    def test_context_dirty_state_survives_canvas_stack_becoming_clean(self):
        canvas = SimpleNamespace(setProjSaveState=Mock())
        title_bar = SimpleNamespace(setTitleContent=Mock())
        owner = SimpleNamespace(
            canvas=canvas,
            titleBar=title_bar,
            _llm_context_dirty=True,
            tr=lambda text: text,
        )

        MainWindow.on_savestate_changed(owner, False)

        canvas.setProjSaveState.assert_called_once_with(True)
        title_bar.setTitleContent.assert_not_called()

    def test_context_only_page_change_saves_without_rendering(self):
        project = SimpleNamespace(save=Mock())
        canvas = SimpleNamespace(
            projstate_unsaved=True,
            text_change_unsaved=Mock(return_value=False),
            draw_change_unsaved=Mock(return_value=False),
            setProjSaveState=Mock(),
        )
        owner = SimpleNamespace(
            canvas=canvas,
            imgtrans_proj=project,
            opening_dir=False,
            _llm_context_dirty=True,
            saveCurrentPage=Mock(),
        )

        MainWindow.conditional_save(owner, keep_exist_as_backup=True)

        project.save.assert_called_once_with(keep_exist_as_backup=True)
        owner.saveCurrentPage.assert_not_called()
        self.assertFalse(owner._llm_context_dirty)
        canvas.setProjSaveState.assert_called_once_with(False)


if __name__ == '__main__':
    unittest.main()
