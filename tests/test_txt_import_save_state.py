import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.mainwindow import MainWindow


class TxtImportSaveStateTest(unittest.TestCase):
    """Imported text bypasses the undo stack and must still reach project save."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_apply_txt_import_marks_project_dirty(self):
        canvas = SimpleNamespace(setProjSaveState=Mock(), clear_undostack=Mock())
        manager = SimpleNamespace(updateSceneTextitems=Mock())
        owner = SimpleNamespace(
            canvas=canvas,
            st_manager=manager,
            imgtrans_proj=SimpleNamespace(current_img='001.png'),
            _imported_text_dirty=False,
        )

        MainWindow._apply_txt_import(owner, {'matched_pages': {'001.png'}})

        self.assertTrue(owner._imported_text_dirty)
        canvas.setProjSaveState.assert_called_once_with(True)
        canvas.clear_undostack.assert_called_once_with(update_saved_step=True)
        manager.updateSceneTextitems.assert_called_once_with()

    def test_apply_txt_import_keeps_project_clean_when_nothing_matched(self):
        canvas = SimpleNamespace(setProjSaveState=Mock(), clear_undostack=Mock())
        manager = SimpleNamespace(updateSceneTextitems=Mock())
        owner = SimpleNamespace(
            canvas=canvas,
            st_manager=manager,
            imgtrans_proj=SimpleNamespace(current_img='001.png'),
            _imported_text_dirty=False,
        )

        MainWindow._apply_txt_import(owner, {'matched_pages': set()})

        self.assertFalse(owner._imported_text_dirty)
        canvas.setProjSaveState.assert_not_called()
        canvas.clear_undostack.assert_not_called()
        manager.updateSceneTextitems.assert_not_called()

    def test_import_only_save_persists_project_without_rendering(self):
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
            _llm_context_dirty=False,
            _imported_text_dirty=True,
            saveCurrentPage=Mock(),
        )

        MainWindow.conditional_save(owner, keep_exist_as_backup=True)

        project.save.assert_called_once_with(keep_exist_as_backup=True)
        owner.saveCurrentPage.assert_not_called()
        self.assertFalse(owner._imported_text_dirty)
        canvas.setProjSaveState.assert_called_once_with(False)

    def test_imported_text_survives_canvas_reporting_saved(self):
        canvas = SimpleNamespace(setProjSaveState=Mock())
        title_bar = SimpleNamespace(setTitleContent=Mock())
        owner = SimpleNamespace(
            canvas=canvas,
            titleBar=title_bar,
            _llm_context_dirty=False,
            _imported_text_dirty=True,
            tr=lambda text: text,
        )

        MainWindow.on_savestate_changed(owner, False)

        canvas.setProjSaveState.assert_called_once_with(True)
        title_bar.setTitleContent.assert_not_called()

    def test_imported_text_forces_project_save_alongside_text_edits(self):
        canvas = SimpleNamespace(
            projstate_unsaved=True,
            text_change_unsaved=Mock(return_value=True),
            draw_change_unsaved=Mock(return_value=False),
            setProjSaveState=Mock(),
        )
        owner = SimpleNamespace(
            canvas=canvas,
            imgtrans_proj=SimpleNamespace(save=Mock()),
            opening_dir=False,
            _llm_context_dirty=False,
            _imported_text_dirty=True,
            saveCurrentPage=Mock(),
        )

        MainWindow.conditional_save(owner)

        owner.saveCurrentPage.assert_called_once()
        self.assertTrue(owner.saveCurrentPage.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
