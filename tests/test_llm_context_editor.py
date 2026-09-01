import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui.llm_context_editor import LLMContextEditor
from ballontranslator.ui.mainwindow import MainWindow
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
