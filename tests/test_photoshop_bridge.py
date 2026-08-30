import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.ps_bridge_dialog import (
    PhotoshopBridgeDialog,
    PhotoshopBridgeUpdateError,
    photoshop_bridge_block_snapshot,
    project_state_matches_disk,
    validate_photoshop_bridge_updates,
)
from ballontranslator.utils.proj_imgtrans import ProjImgTrans, TextBlkEncoder
from ballontranslator.utils.textblock import TextBlock


class PhotoshopBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project = ProjImgTrans()
        self.project.directory = self.temp_dir.name
        self.project.proj_path = os.path.join(self.temp_dir.name, "project.json")
        self.project.current_img = "page.png"
        self.project.pages = {
            "page.png": [
                TextBlock(
                    xyxy=[0, 0, 10, 10],
                    text=["source 1"],
                    translation="one",
                    rich_text="<p>one</p>",
                    _bounding_rect=[0, 0, 10, 10],
                ),
                TextBlock(
                    xyxy=[20, 0, 30, 10],
                    text=["source 2"],
                    translation="",
                    _bounding_rect=[20, 0, 10, 10],
                ),
                TextBlock(
                    xyxy=[40, 0, 50, 10],
                    text=["source 3"],
                    translation="three",
                    _bounding_rect=[40, 0, 10, 10],
                ),
            ]
        }
        self.project.save()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _payload(self, *indices: int) -> dict:
        updates = []
        for index in indices:
            block = self.project.pages["page.png"][index]
            updates.append({
                "block_index": index,
                "translation": f"updated {index + 1}",
                "font_size": block.font_size + index + 1,
                "base": photoshop_bridge_block_snapshot(block),
            })
        return {
            "version": 1,
            "session_id": "test",
            "project_path": self.project.proj_path,
            "page": "page.png",
            "block_count": 3,
            "updates": updates,
        }

    def test_noncontiguous_layer_numbers_update_the_named_blocks(self) -> None:
        payload = self._payload(0, 2)
        fake_window = SimpleNamespace(
            imgtrans_proj=self.project,
            canvas=SimpleNamespace(
                projstate_unsaved=False,
                setProjSaveState=Mock(),
                update_saved_undostep=Mock(),
            ),
            st_manager=SimpleNamespace(updateSceneTextitems=Mock()),
            tr=lambda text: text,
        )

        count = MainWindow.apply_photoshop_bridge_updates(fake_window, payload)

        self.assertEqual(count, 2)
        self.assertEqual(self.project.pages["page.png"][0].translation, "updated 1")
        self.assertEqual(self.project.pages["page.png"][0].rich_text, "")
        self.assertEqual(self.project.pages["page.png"][1].translation, "")
        self.assertEqual(self.project.pages["page.png"][2].translation, "updated 3")
        fake_window.st_manager.updateSceneTextitems.assert_called_once_with()
        self.assertTrue(os.path.isfile(self.project.proj_path + ".backup"))
        with open(
            self.project.proj_path + ".backup", "r", encoding="utf-8"
        ) as backup_file:
            backup = json.load(backup_file)
        self.assertEqual(backup["pages"]["page.png"][0]["translation"], "one")
        with open(self.project.proj_path, "r", encoding="utf-8") as project_file:
            saved = json.load(project_file)
        self.assertEqual(saved["pages"]["page.png"][0]["translation"], "updated 1")

    def test_changed_snapshot_and_duplicate_numbers_are_rejected(self) -> None:
        payload = self._payload(0)
        payload["updates"][0]["base"]["text"] = ["different source"]
        with self.assertRaises(PhotoshopBridgeUpdateError):
            validate_photoshop_bridge_updates(self.project, payload)
        self.assertEqual(self.project.pages["page.png"][0].translation, "one")

        payload = self._payload(0)
        payload["updates"].append(dict(payload["updates"][0]))
        with self.assertRaises(PhotoshopBridgeUpdateError):
            validate_photoshop_bridge_updates(self.project, payload)

    def test_unsaved_app_changes_block_the_update(self) -> None:
        payload = self._payload(0)
        fake_window = SimpleNamespace(
            imgtrans_proj=self.project,
            canvas=SimpleNamespace(projstate_unsaved=True),
            tr=lambda text: text,
        )

        with self.assertRaisesRegex(RuntimeError, "unsaved changes"):
            MainWindow.apply_photoshop_bridge_updates(fake_window, payload)
        self.assertEqual(self.project.pages["page.png"][0].translation, "one")

    def test_disk_change_is_detected_before_the_app_can_overwrite_it(self) -> None:
        self.assertTrue(project_state_matches_disk(self.project))
        self.project.current_img = "another-page.png"
        self.assertTrue(project_state_matches_disk(self.project))
        self.project.current_img = "page.png"
        with open(self.project.proj_path, "r", encoding="utf-8") as project_file:
            disk_data = json.load(project_file)
        disk_data["pages"]["page.png"][0]["translation"] = "external edit"
        with open(self.project.proj_path, "w", encoding="utf-8") as project_file:
            json.dump(disk_data, project_file, cls=TextBlkEncoder)

        self.assertFalse(project_state_matches_disk(self.project))

    def test_bridge_polls_only_for_a_visible_launched_session(self) -> None:
        dialog = PhotoshopBridgeDialog(project=self.project)
        self.assertFalse(dialog._update_timer.isActive())
        dialog._pending_update_path = os.path.join(
            self.temp_dir.name,
            "pending.json",
        )
        dialog.show()
        self.app.processEvents()
        self.assertTrue(dialog._update_timer.isActive())
        dialog.hide()
        self.assertFalse(dialog._update_timer.isActive())
        dialog.deleteLater()

    def test_installer_copies_without_elevation_when_destination_is_writable(
        self,
    ) -> None:
        photoshop_dir = os.path.join(self.temp_dir.name, "Photoshop")
        scripts_dir = os.path.join(photoshop_dir, "Presets", "Scripts")
        os.makedirs(scripts_dir)
        dialog = PhotoshopBridgeDialog(project=self.project)
        with patch(
            "ballontranslator.ui.ps_bridge_dialog.get_photoshop_paths",
            return_value=(photoshop_dir, scripts_dir, None),
        ), patch(
            "ballontranslator.ui.ps_bridge_dialog.shutil.copy2"
        ) as copy, patch.object(
            dialog, "refresh_status"
        ) as refresh, patch.object(
            dialog, "_open_manual_install_folders"
        ) as manual, patch(
            "ballontranslator.ui.ps_bridge_dialog.QMessageBox.information"
        ):
            dialog.on_install_script()

        source_jsx, target_jsx = copy.call_args.args
        self.assertEqual(os.path.basename(source_jsx), "BallonTranslator_PS_Bridge.jsx")
        self.assertEqual(target_jsx, os.path.join(scripts_dir, os.path.basename(source_jsx)))
        refresh.assert_called_once_with()
        manual.assert_not_called()
        dialog.deleteLater()

    def test_installer_falls_back_to_manual_copy_without_elevation(self) -> None:
        photoshop_dir = os.path.join(self.temp_dir.name, "Photoshop")
        scripts_dir = os.path.join(photoshop_dir, "Presets", "Scripts")
        os.makedirs(scripts_dir)
        dialog = PhotoshopBridgeDialog(project=self.project)
        with patch(
            "ballontranslator.ui.ps_bridge_dialog.get_photoshop_paths",
            return_value=(photoshop_dir, scripts_dir, None),
        ), patch(
            "ballontranslator.ui.ps_bridge_dialog.shutil.copy2",
            side_effect=PermissionError("protected"),
        ), patch.object(
            dialog, "refresh_status"
        ) as refresh, patch.object(
            dialog, "_open_manual_install_folders"
        ) as manual, patch(
            "ballontranslator.ui.ps_bridge_dialog.QMessageBox.warning"
        ) as warning:
            dialog.on_install_script()

        manual.assert_called_once()
        source_jsx, destination = manual.call_args.args
        self.assertEqual(destination, scripts_dir)
        refresh.assert_not_called()
        warning.assert_called_once()
        warning_text = warning.call_args.args[2]
        self.assertIn(os.path.normpath(source_jsx), warning_text)
        self.assertIn(os.path.normpath(scripts_dir), warning_text)
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
