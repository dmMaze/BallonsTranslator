import copy
import gc
import json
import os
import unittest
import weakref
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QTextCursor
from qtpy.QtTest import QSignalSpy, QTest
from qtpy.QtWidgets import QApplication, QLayout, QMessageBox, QScrollArea, QVBoxLayout, QWidget

from ballontranslator.modules import codex
from ballontranslator.ui import codex_account
from ballontranslator.ui.codex_settings import CodexSettingsPanel
from ballontranslator.utils.config import ModuleConfig, ProgramConfig, json_dump_program_config, pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, sync_codex_profile


CATALOG = {
    'vision-model': {'modalities': ['text', 'image'], 'efforts': ['low', 'high']},
    'text-model': {'modalities': ['text'], 'efforts': ['none']},
}


class CodexSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.profile = default_codex_profile()
        self.profile.id = 'codex'
        self.profile.model = self.profile.vision_model = 'vision-model'
        self.profile.image_model = 'gpt-image-2'
        self.profile.image_model_options = ['gpt-image-2']
        sync_codex_profile(self.profile, CATALOG)
        self.account = codex.CodexAccount()
        for patcher in (
            patch.object(pcfg.module, 'llm_profiles', [self.profile]),
            patch.object(pcfg.module, 'codex_models', copy.deepcopy(CATALOG)),
            patch.object(codex, 'account', self.account),
            patch.object(codex_account, 'account', self.account),
            patch.object(self.app, '_codex_account_controller', None, create=True),
            patch('ballontranslator.ui.codex_settings.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.controller = codex_account.CodexAccountController.instance()
        self.addCleanup(self.controller.deleteLater)
        warning = patch('ballontranslator.ui.codex_account.QMessageBox.warning')
        self.warning = warning.start()
        self.addCleanup(warning.stop)

    def tearDown(self) -> None:
        if self.controller.worker is not None:
            self.controller.cancel()
            self.assertTrue(self.controller.worker.wait(2000))
            self.app.processEvents()
        self.doCleanups()
        self.drain_deletes()
        super().tearDown()

    def drain_deletes(self) -> None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        gc.collect()

    def make_panel(self) -> CodexSettingsPanel:
        panel = CodexSettingsPanel()
        self.addCleanup(panel.deleteLater)
        return panel

    def test_construction_reads_only_cached_state_and_catalog_capabilities(self) -> None:
        with patch.object(self.account, '_load', side_effect=AssertionError('Unexpected credential IO')), \
                patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected HTTP')):
            panel = self.make_panel()
            self.assertIs(panel.profile, self.profile)
            self.assertEqual(panel.param_widgets['model'].currentText(), 'vision-model')
            self.assertEqual(panel.param_widgets['vision_model'].count(), 1)
            self.assertEqual(panel.param_widgets['image_model'].currentText(), 'gpt-image-2')
            self.assertFalse(panel.account_buttons['login'].isHidden())
            self.assertTrue(panel.account_buttons['logout'].isHidden())
            self.assertTrue(panel.account_status.isHidden())
            self.assertIsNone(self.controller.worker)

    def test_control_edits_persist_on_canonical_profile_and_update_reasoning_choices(self) -> None:
        self.profile.vision_model_options.append('gpt-reasoning')
        panel = self.make_panel()
        panel.param_widgets['model'].setCurrentText('text-model')
        thinking = panel.param_widgets['thinking_level']
        self.assertEqual([thinking.itemText(i) for i in range(thinking.count())], ['Auto', 'Disabled'])
        thinking.setCurrentText('Disabled')
        panel.param_widgets['vision_detail_level'].setCurrentText('high')
        image = panel.param_widgets['image_model']
        self.assertEqual([image.itemText(i) for i in range(image.count())], ['gpt-image-2', 'gpt-reasoning → gpt-image-2'])
        image.setCurrentText('gpt-reasoning → gpt-image-2')
        QTest.keyClick(image.lineEdit(), Qt.Key.Key_Return)
        for key, value in (('prompt', 'Keep translation concise.'), ('vision_prompt', 'Read vertical text.'),
                           ('image_prompt', 'Preserve the line art.')):
            editor = panel.param_widgets[key]
            editor.selectAll()
            editor.insertPlainText(value)
            self.assertEqual(getattr(self.profile, key), value)
        saved = json.loads(json_dump_program_config(ProgramConfig(module=ModuleConfig(
            llm_profiles=[self.profile], codex_models=CATALOG,
        ))))
        restored = next(profile for profile in saved['module']['llm_profiles'] if profile['id'] == 'codex')
        self.assertEqual(restored['model'], 'text-model')
        self.assertEqual(restored['thinking_level'], 'Disabled')
        self.assertEqual(restored['vision_detail_level'], 'high')
        self.assertEqual(restored['vision_prompt'], 'Read vertical text.')
        self.assertEqual(restored['image_model'], 'gpt-reasoning → gpt-image-2')
        self.assertEqual(restored['image_model_options'], ['gpt-image-2'])
        self.assertIs(pcfg.module.llm_profiles[0], self.profile)
        image.setCurrentText('gpt-image-2')
        QTest.keyClick(image.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(self.profile.image_model, 'gpt-image-2')

    def test_saved_model_dropdowns_open_without_signin_or_cached_catalog(self) -> None:
        module = ModuleConfig(llm_profiles=[{
            'id': 'codex', 'backend': 'codex', 'model': 'saved-text',
            'vision_model': 'saved-vision', 'image_model': 'gpt-image-2',
        }], codex_models={})
        with patch.object(pcfg, 'module', module), \
                patch.object(self.account, '_load', side_effect=AssertionError('Unexpected credential IO')), \
                patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected HTTP')):
            panel = self.make_panel()
            panel.show()
            self.app.processEvents()
            self.assertEqual(self.controller.account, '')
            for key, model in (('model', 'saved-text'), ('vision_model', 'saved-vision'),
                               ('image_model', 'gpt-image-2')):
                with self.subTest(key=key):
                    combo = panel.param_widgets[key]
                    self.assertTrue(combo.isEnabled())
                    self.assertEqual(combo.currentText(), model)
                    self.assertTrue(combo.model().item(combo.currentIndex()).isEnabled())
                    QTest.keyClick(combo, Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
                    self.app.processEvents()
                    self.assertTrue(combo.view().isVisible())
                    if key == 'image_model':
                        QTest.keyClick(combo.view(), Qt.Key.Key_Escape)
                    else:
                        self.assertGreater(combo.count(), 1)
                        QTest.keyClick(combo.view(), Qt.Key.Key_Home)
                        QTest.keyClick(combo.view(), Qt.Key.Key_Return)
                        self.assertNotEqual(combo.currentText(), model)
                        self.assertEqual(getattr(panel.profile, key), combo.currentText())
            self.assertTrue(panel.param_widgets['thinking_level'].isEnabled())
            self.assertTrue(panel.param_widgets['vision_detail_level'].isEnabled())

    def test_catalog_refresh_updates_controls_and_preserves_removed_choices(self) -> None:
        self.profile.vision_model_options.append('gpt-reasoning')
        self.profile.image_model = 'gpt-reasoning → gpt-image-2'
        panel = self.make_panel()
        changed = QSignalSpy(panel.profile_ui_updated)
        self.account._loaded = True
        self.account._credentials = {'account_id': 'local-test-account'}
        updated = {'replacement': {'modalities': ['text'], 'efforts': ['medium']}}
        with patch.object(self.account, 'catalog', return_value=updated):
            panel.account_buttons['refresh'].click()
            worker = self.controller.worker
            self.assertIsNotNone(worker)
            self.assertTrue(worker.wait(2000))
            self.app.processEvents()
        self.assertTrue(changed)
        text = panel.param_widgets['model']
        self.assertEqual(text.currentText(), 'vision-model')
        self.assertFalse(text.model().item(text.currentIndex()).isEnabled())
        self.assertTrue(text.model().item(text.findText('replacement')).isEnabled())
        self.assertTrue(panel.param_widgets['vision_model'].isEnabled())
        vision = panel.param_widgets['vision_model']
        self.assertFalse(vision.model().item(vision.currentIndex()).isEnabled())
        self.assertTrue(panel.param_widgets['image_model'].isEnabled())
        self.assertEqual(self.profile.vision_model, 'vision-model')
        image = panel.param_widgets['image_model']
        self.assertEqual(image.currentText(), 'gpt-reasoning → gpt-image-2')
        self.assertFalse(image.model().item(image.currentIndex()).isEnabled())
        image.setCurrentText('gpt-image-2')
        QTest.keyClick(image.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(self.profile.image_model, 'gpt-image-2')
        text.setCurrentText('replacement')
        self.assertEqual(self.profile.model, 'replacement')
        self.assertIn('medium', self.profile.thinking_level_options)

    def test_custom_image_models_commit_once_and_survive_refresh_and_reload(self) -> None:
        panel = self.make_panel()
        panel.show()
        self.app.processEvents()
        combo = panel.param_widgets['image_model']
        changed = QSignalSpy(panel.profile_summary_changed)
        for commit in ('enter', 'focus_loss'):
            with self.subTest(commit=commit):
                combo.setFocus()
                combo.lineEdit().selectAll()
                QTest.keyClicks(combo.lineEdit(), '  custom-image-' + commit + '  ')
                self.assertNotIn('custom-image-' + commit, self.profile.image_model_options)
                if commit == 'enter':
                    QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Return)
                else:
                    panel.param_widgets['model'].setFocus()
                self.app.processEvents()
                self.assertEqual(self.profile.image_model, 'custom-image-' + commit)
                self.assertEqual(combo.currentText(), self.profile.image_model)
        self.assertEqual(len(changed), 2)
        expected = list(self.profile.image_model_options)
        for value in ('   ', '  custom-image-focus_loss  '):
            combo.setFocus()
            combo.lineEdit().selectAll()
            QTest.keyClicks(combo.lineEdit(), value)
            QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Return)
            self.assertEqual(self.profile.image_model_options, expected)
            self.assertEqual(combo.currentText(), 'custom-image-focus_loss')
        self.assertEqual(len(changed), 2)

        sync_codex_profile(self.profile, {'new-text': {'modalities': ['text'], 'efforts': []}})
        self.controller.catalog_changed.emit()
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], expected)
        saved = json.loads(json_dump_program_config(ProgramConfig(module=ModuleConfig(
            llm_profiles=[self.profile], codex_models=CATALOG,
        ))))
        restored = ModuleConfig(**saved['module']).llm_profiles[0]
        self.assertEqual(restored.image_model_options, expected)
        self.assertEqual(restored.image_model, 'custom-image-focus_loss')

        combo.showPopup()
        QTest.keyClick(combo.view(), Qt.Key.Key_Home)
        QTest.keyClick(combo.view(), Qt.Key.Key_Return)
        self.assertEqual(self.profile.image_model, expected[0])

    def test_image_model_buttons_add_delete_and_preserve_an_empty_list(self) -> None:
        panel = self.make_panel()
        panel.show()
        self.app.processEvents()
        combo = panel.param_widgets['image_model']
        original = self.profile.image_model
        panel.add_image_model_btn.click()
        self.assertEqual(combo.currentText(), '')
        self.assertFalse(panel.remove_image_model_btn.isEnabled())
        panel.remove_image_model_btn.click()
        self.assertEqual(self.profile.image_model_options, [original])
        panel.onCatalogChanged()
        self.assertEqual(combo.currentText(), '')
        QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(combo.currentText(), original)
        self.assertTrue(panel.remove_image_model_btn.isEnabled())

        for model in ('image-one', 'image-two', 'image-three', 'image-two'):
            QTest.mouseClick(panel.add_image_model_btn, Qt.MouseButton.LeftButton)
            QTest.keyClicks(combo.lineEdit(), '  ' + model + '  ')
            QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(self.profile.image_model_options, [original, 'image-one', 'image-two', 'image-three'])
        self.assertEqual(self.profile.image_model, 'image-two')
        changed = QSignalSpy(panel.profile_summary_changed)
        for expected in ('image-three', 'image-one', original, ''):
            QTest.mouseClick(panel.remove_image_model_btn, Qt.MouseButton.LeftButton)
            self.assertEqual(self.profile.image_model, expected)
            self.assertEqual(combo.currentText(), expected)
        self.assertEqual(len(changed), 4)
        self.assertFalse(panel.remove_image_model_btn.isEnabled())
        panel.remove_image_model_btn.click()
        self.assertEqual(len(changed), 4)

        sync_codex_profile(self.profile, CATALOG)
        panel.onCatalogChanged()
        self.assertEqual(combo.count(), 0)
        saved = json.loads(json_dump_program_config(ProgramConfig(module=ModuleConfig(
            llm_profiles=[self.profile], codex_models=CATALOG,
        ))))
        restored = ModuleConfig(**saved['module']).llm_profiles[0]
        self.assertEqual(restored.image_model_options, [])
        self.assertEqual(restored.image_model, '')
        panel.add_image_model_btn.click()
        QTest.keyClicks(combo.lineEdit(), 'replacement-image')
        QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(self.profile.image_model_options, ['replacement-image'])
        self.assertEqual(self.profile.image_model, 'replacement-image')

    def test_deleting_a_combination_removes_only_its_image_model(self) -> None:
        self.profile.vision_model_options = ['gpt-one', 'gpt-two', 'other-model']
        self.profile.image_model_options = ['gpt-image-2', 'other-image']
        self.profile.image_model = 'gpt-one → gpt-image-2'
        panel = self.make_panel()
        panel.remove_image_model_btn.click()
        combo = panel.param_widgets['image_model']
        self.assertEqual(self.profile.image_model_options, ['other-image'])
        self.assertEqual(self.profile.image_model, 'other-image')
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ['other-image'])
        self.assertEqual(self.profile.vision_model_options, ['gpt-one', 'gpt-two', 'other-model'])

    def test_unavailable_or_non_gpt_pair_edits_do_not_change_saved_models(self) -> None:
        panel = self.make_panel()
        combo = panel.param_widgets['image_model']
        with patch('ballontranslator.ui.codex_settings.QMessageBox.warning') as warning:
            for value in ('gemini → gpt-image-2', 'gpt-reason → other-image', 'gpt-unlisted → gpt-image-2'):
                combo.setEditText(value)
                self.assertFalse(panel.remove_image_model_btn.isEnabled())
                panel.remove_image_model_btn.click()
                self.assertEqual(self.profile.image_model_options, ['gpt-image-2'])
                QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Return)
                self.assertEqual(self.profile.image_model, 'gpt-image-2')
                self.assertEqual(self.profile.image_model_options, ['gpt-image-2'])
        self.assertEqual(warning.call_count, 3)
        with patch('ballontranslator.ui.codex_settings.QMessageBox.warning'):
            combo.setEditText('gemini → gpt-image-2')
            panel.deleteCurrentImageModel()
            self.assertEqual(self.profile.image_model_options, ['gpt-image-2'])

    def test_external_sync_rebinds_profile_without_losing_prompt_undo_or_emitting_edits(self) -> None:
        panel = self.make_panel()
        editor = panel.param_widgets['prompt']
        editor.moveCursor(QTextCursor.MoveOperation.End)
        editor.insertPlainText(' Added instructions.')
        previous = editor.toPlainText()
        cursor = editor.textCursor().position()
        changed = QSignalSpy(panel.profile_summary_changed)
        panel.syncFromProfile()
        self.assertEqual(editor.textCursor().position(), cursor)
        self.assertTrue(editor.document().isUndoAvailable())
        self.assertFalse(changed)
        replacement = copy.deepcopy(self.profile)
        replacement.model = 'text-model'
        sync_codex_profile(replacement, CATALOG)
        pcfg.module.llm_profiles = [replacement]
        panel.syncFromProfile()
        self.assertIs(panel.profile, replacement)
        self.assertEqual(panel.param_widgets['model'].currentText(), 'text-model')
        self.assertEqual(editor.toPlainText(), previous)
        editor.insertPlainText(' More.')
        self.assertEqual(replacement.prompt, editor.toPlainText())
        self.assertEqual(self.profile.prompt, previous)

    def test_catalog_refresh_preserves_uncommitted_image_model_edit(self) -> None:
        panel = self.make_panel()
        panel.show()
        self.app.processEvents()
        combo = panel.param_widgets['image_model']
        combo.setFocus()
        combo.lineEdit().selectAll()
        QTest.keyClicks(combo.lineEdit(), 'custom-image-draft')
        panel.onCatalogChanged()
        self.assertEqual(combo.currentText(), 'custom-image-draft')
        self.assertEqual(self.profile.image_model, 'gpt-image-2')
        QTest.keyClick(combo.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(self.profile.image_model, 'custom-image-draft')

    def test_failed_catalog_keeps_signout_and_stops_refresh_animation(self) -> None:
        self.controller.account = 'signed-in@example.test'
        panel = self.make_panel()
        with patch.object(self.account, 'catalog', side_effect=RuntimeError('private service details')):
            panel.account_buttons['refresh'].click()
            self.assertFalse(panel.account_buttons['logout'].isEnabled())
            self.assertTrue(panel.refresh_button._busy)
            self.assertTrue(self.controller.worker.wait(2000))
            self.app.processEvents()
        self.assertFalse(panel.account_buttons['logout'].isHidden())
        self.assertTrue(panel.account_buttons['login'].isHidden())
        self.assertTrue(panel.account_buttons['logout'].isEnabled())
        self.assertFalse(panel.refresh_button._busy)
        self.assertTrue(panel.refresh_button.isEnabled())
        self.assertEqual(panel.account_status.text(), 'Connected: signed-in@example.test')
        self.warning.assert_called_once()
        self.assertNotIn('private service details', self.warning.call_args.args[2])
        with patch.object(self.controller, 'start') as start:
            panel.account_buttons['logout'].click()
        start.assert_called_once_with('logout')

    def test_edits_target_replaced_canonical_profile_before_explicit_sync(self) -> None:
        panel = self.make_panel()
        replacement = copy.deepcopy(self.profile)
        pcfg.module.llm_profiles = [replacement]
        panel.param_widgets['model'].setCurrentText('text-model')
        self.assertEqual(replacement.model, 'text-model')
        self.assertEqual(self.profile.model, 'vision-model')
        self.assertIs(panel.profile, replacement)

    def test_account_actions_route_to_controller_and_cancel_remains_available_while_busy(self) -> None:
        panel = self.make_panel()
        with patch.object(self.controller, 'start') as start, \
                patch('ballontranslator.ui.codex_settings.QMessageBox.question') as confirm:
            panel.account_buttons['login'].click()
        start.assert_called_once_with('login')
        confirm.assert_not_called()
        panel.show()
        for action in ('refresh', 'login', 'logout'):
            with self.subTest(action=action):
                worker = codex_account.CodexAccountWorker(action, self.controller)
                self.controller.worker = worker
                self.controller.changed.emit()
                self.assertFalse(panel.account_buttons['login'].isEnabled())
                self.assertFalse(panel.account_buttons['logout'].isEnabled())
                self.assertFalse(panel.refresh_button.isEnabled())
                self.assertEqual(panel.refresh_button._rotation_timer.isActive(), action != 'logout')
                self.assertFalse(panel.account_buttons['cancel'].isHidden())
                panel.account_buttons['cancel'].click()
                self.assertTrue(worker.stop_event.is_set())
                self.assertFalse(panel.account_buttons['cancel'].isEnabled())
                self.controller.worker = None
                worker.deleteLater()
                self.controller.changed.emit()
                self.assertTrue(panel.account_buttons['cancel'].isHidden())
                self.assertTrue(panel.account_buttons['login'].isEnabled())
                self.assertTrue(panel.refresh_button.isEnabled())
                self.assertFalse(panel.refresh_button._rotation_timer.isActive())

    def test_signout_requires_confirmation_and_cancel_preserves_account(self) -> None:
        self.controller.account = 'signed-in@example.test'
        panel = self.make_panel()
        with patch.object(self.controller, 'start') as start, \
                patch('ballontranslator.ui.codex_settings.QMessageBox.question',
                      return_value=QMessageBox.StandardButton.Cancel) as confirm:
            panel.account_buttons['logout'].click()
            confirm.assert_called_once()
            start.assert_not_called()
            self.assertEqual(self.controller.account, 'signed-in@example.test')
            self.assertFalse(panel.account_status.isHidden())
            confirm.return_value = QMessageBox.StandardButton.Yes
            panel.account_buttons['logout'].click()
            start.assert_called_once_with('logout')

    def test_focus_target_and_profile_settings_do_not_depend_on_card_lifecycle(self) -> None:
        panel = self.make_panel()
        panel.show()
        self.app.processEvents()
        panel.focusControl('vision_prompt')
        self.assertTrue(panel.param_widgets['vision_prompt'].hasFocus())
        QTest.keyClicks(panel.param_widgets['vision_prompt'], 'Extra OCR instructions.')
        self.assertIn('Extra OCR instructions.', self.profile.vision_prompt)

    def test_focus_target_scrolls_lower_prompt_into_view(self) -> None:
        scroll = QScrollArea()
        self.addCleanup(scroll.deleteLater)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        panel = CodexSettingsPanel()
        layout.addWidget(panel)
        scroll.setWidget(content)
        scroll.resize(600, 220)
        scroll.show()
        self.app.processEvents()
        scroll.verticalScrollBar().setValue(0)
        panel.focusControl('image_prompt')
        self.app.processEvents()
        self.app.processEvents()
        editor = panel.param_widgets['image_prompt']
        self.assertTrue(editor.hasFocus())
        cursor_rect = editor.cursorRect()
        cursor_rect.moveTopLeft(editor.viewport().mapTo(scroll.viewport(), cursor_rect.topLeft()))
        self.assertTrue(scroll.viewport().rect().contains(cursor_rect))
        self.assertGreater(scroll.verticalScrollBar().value(), 0)

    def test_deleted_panel_and_prompt_editors_release_while_controller_remains_usable(self) -> None:
        panel = CodexSettingsPanel()
        refs = [weakref.ref(panel), *(weakref.ref(panel.param_widgets[key]) for key in (
            'prompt', 'vision_prompt', 'image_prompt',
        ))]
        panel.deleteLater()
        del panel
        self.drain_deletes()
        self.assertTrue(all(ref() is None for ref in refs))
        self.controller.changed.emit()
        self.controller.catalog_changed.emit()
        replacement = self.make_panel()
        self.controller.account = 'replacement@example.test'
        self.controller.changed.emit()
        self.assertEqual(replacement.account_status.text(), 'Connected: replacement@example.test')


if __name__ == '__main__':
    unittest.main()
