import os
import json
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtGui import QContextMenuEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QLineEdit, QMenu, QVBoxLayout, QWidget

from ballontranslator.ui.llm_profile_widgets import LLMProfilesWidget, ProfileCardWidget
from ballontranslator.ui.misc import parse_stylesheet
from ballontranslator.ui.module_tool_button import ModuleSelectionWidget
from ballontranslator.utils import config as config_module, shared
from ballontranslator.utils.config import ModuleConfig, ProgramConfig, pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, LLMProfile, default_profile, profile_by_id, resolve_api_key, profile_to_dict
from ballontranslator.utils.secret_store import SecretStore, is_portable_secret


class LLMProfileModelSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_jev_key_editors_mask_save_to_config_reload_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(shared, 'CONFIG_PATH', os.path.join(directory, 'config.json')), \
                mock.patch.object(pcfg, 'module', ModuleConfig()), mock.patch('requests.post') as post:
            panel = LLMProfilesWidget()
            self.addCleanup(panel.deleteLater)
            widgets = {'typesafe': panel.typesafe_api_key_widget,
                       'openrouter': panel.rows['openrouter'].api_key_widget}
            for provider, widget in widgets.items():
                editor = widget.editor
                self.assertEqual(editor.echoMode(), QLineEdit.EchoMode.Password)
                self.assertEqual(editor.text(), '')
                editor.setText(f'  test-{provider}-key  ' if provider == 'typesafe' else f'test-{provider}-key')
                editor.editingFinished.emit()
                saved = (pcfg.module.ocr_jev_typesafe_api_key if provider == 'typesafe'
                         else profile_by_id(pcfg.module.llm_profiles, 'openrouter').api_key)
                self.assertTrue(is_portable_secret(saved))
                self.assertEqual(SecretStore().resolve(saved).value, f'test-{provider}-key')
            self.assertTrue(config_module.save_config())
            with open(shared.CONFIG_PATH, encoding='utf-8') as stream:
                saved_json = stream.read()
            self.assertNotIn('ocr_jev_openrouter_api_key', saved_json)
            for provider in widgets:
                self.assertNotIn(f'test-{provider}-key', saved_json)
            saved_config = ProgramConfig.load(shared.CONFIG_PATH)
            with mock.patch.object(pcfg, 'module', saved_config.module):
                reloaded = LLMProfilesWidget()
                self.addCleanup(reloaded.deleteLater)
                widgets = {'typesafe': reloaded.typesafe_api_key_widget,
                           'openrouter': reloaded.rows['openrouter'].api_key_widget}
                for provider, widget in widgets.items():
                    self.assertEqual(widget.text(), f'test-{provider}-key')
                    widget.editor.clear()
                    widget.editor.editingFinished.emit()
                self.assertTrue(config_module.save_config())
                cleared = ProgramConfig.load(shared.CONFIG_PATH)
                self.assertEqual(cleared.module.ocr_jev_typesafe_api_key, '')
                self.assertEqual(resolve_api_key(profile_by_id(cleared.module.llm_profiles, 'openrouter')), '')
            post.assert_not_called()

    def test_upgraded_config_exposes_selectable_codex_in_translation_and_ocr_menus(self) -> None:
        cockpit = default_profile('Ollama')
        cockpit.id = 'cockpit'
        cockpit.name = 'Cockpit 本機'
        cockpit.built_in = False
        legacy = default_profile('Codex')
        legacy.id = 'codex'
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'config.json')
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump({'module': {'llm_profiles': [cockpit.to_dict(), legacy.to_dict()],
                           'translator_llm_id': 'cockpit', 'ocr_llm_id': 'cockpit'}}, stream)
            cfg = ProgramConfig.load(path)
        with mock.patch.object(pcfg, 'module', cfg.module):
            for modality, module, selection, model_attr, icon in (
                ('text', 'LLMTranslator', 'translator_llm_id', 'model', 'text.svg'),
                ('vision', 'LLMOCR', 'ocr_llm_id', 'vision_model', 'eye.svg'),
            ):
                with self.subTest(modality=modality):
                    widget = ModuleSelectionWidget(modality, icon, llm_modality=modality)
                    self.addCleanup(widget.deleteLater)
                    widget.selector.addItem(module)
                    widget.menu.rebuildMenu()
                    codex_menu = next((action.menu() for action in widget.menu.actions()
                                       if action.menu() and action.menu().title() == 'Codex App Server'), None)
                    self.assertIsNotNone(codex_menu, f'Codex missing from {modality} menu')
                    selected = mock.Mock()
                    widget.llm_profile_changed.connect(selected)
                    for model in ('gpt-5.5', 'gpt-6-astra', 'gpt-6-sol', 'gpt-6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'):
                        with self.subTest(model=model):
                            selected.reset_mock()
                            model_action = next(action for action in codex_menu.actions()
                                                if action.data() == ('codex-app-server', model_attr, model))
                            model_action.trigger()
                            self.assertEqual(getattr(cfg.module, selection), 'codex-app-server')
                            self.assertEqual(widget.selector.currentText(), module)
                            self.assertEqual(getattr(profile_by_id(cfg.module.llm_profiles, 'codex-app-server'), model_attr), model)
                            selected.assert_called_once_with('codex-app-server')

    def test_backend_switch_exposes_codex_settings_without_rebuilding(self) -> None:
        profile = default_profile('OpenAI')
        profile.model = 'gpt-5.5'
        profile.support_image = False
        profile.thinking_level = 'Disabled'
        profile.thinking_level_options.append('custom-effort')
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)
        backend = card.details.param_widgets['transport']
        executable = card.details.param_widgets['codex_executable']
        save_sessions = card.details.param_widgets['codex_save_sessions']
        thinking = card.details.param_widgets['thinking_level']
        self.assertTrue(executable.isHidden())
        self.assertTrue(save_sessions.isHidden())
        self.assertFalse(save_sessions.isChecked())
        backend.setCurrentText('Codex App Server')
        self.assertEqual(profile.transport, 'Codex App Server')
        self.assertEqual(thinking.currentText(), 'none')
        self.assertEqual([thinking.itemText(i) for i in range(thinking.count())],
                         ['none', 'low', 'medium', 'high', 'xhigh'])
        self.assertFalse(card.details.param_widgets['vision_thinking_level'].isHidden())
        self.assertFalse(executable.isHidden())
        self.assertFalse(save_sessions.isHidden())
        save_sessions.setChecked(True)
        self.assertTrue(profile.codex_save_sessions)
        save_sessions.setChecked(False)
        self.assertFalse(profile.codex_save_sessions)
        self.assertTrue(card.api_summary_widget.isHidden())
        for key in ('require_api_key', 'base_url', 'max_tokens', 'temperature', 'top_p',
                    'frequency_penalty', 'presence_penalty', 'low_vram_mode', 'vision_detail_level'):
            self.assertTrue(card.details.param_widgets[key].isHidden(), key)
        self.assertFalse(card.details.param_widgets['json_schema_response_format'].isHidden())
        card.on_detail_edited('codex_timeout', {'content': '300'})
        card.on_detail_edited('codex_timeout', {'content': '0'})
        self.assertEqual(profile.codex_timeout, 300)
        self.assertEqual(card.details.param_widgets['codex_timeout'].text(), '300')
        backend.setCurrentText('OpenAI-compatible')
        self.assertEqual(profile.transport, 'OpenAI-compatible')
        for option in ('Auto', 'Disabled', 'minimal', 'custom-effort'):
            self.assertGreaterEqual(thinking.findText(option), 0)
        self.assertTrue(executable.isHidden())
        self.assertTrue(save_sessions.isHidden())
        self.assertTrue(card.details.param_widgets['vision_thinking_level'].isHidden())
        self.assertFalse(card.api_summary_widget.isHidden())
        self.assertIs(card.details.param_widgets['codex_executable'], executable)

    def test_codex_text_and_vision_thinking_are_independent(self) -> None:
        profile = default_profile('Codex')
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)
        execution = card.details.param_widgets['codex_execution']
        self.assertEqual(execution.currentText(), 'Python SDK')
        execution.setCurrentText('CLI')
        self.assertEqual(profile.codex_execution, 'CLI')
        execution.setCurrentText('Python SDK')
        self.assertEqual(profile.codex_execution, 'Python SDK')
        thinking = card.details.param_widgets['thinking_level']
        vision_thinking = card.details.param_widgets['vision_thinking_level']
        self.assertEqual(card.model_combo.currentText(), 'gpt-5.6-sol')
        self.assertEqual(card.vision_model_combo.currentText(), 'gpt-5.6-sol')
        self.assertEqual(thinking.currentText(), 'none')
        self.assertEqual(vision_thinking.currentText(), 'none')
        for model in ('gpt-5.6-sol', 'gpt-5.6-terra'):
            with self.subTest(model=model):
                card.model_combo.setCurrentText(model)
                card.vision_model_combo.setCurrentText(model)
                self.assertEqual([thinking.itemText(i) for i in range(thinking.count())],
                                 ['none', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'])
        thinking.setCurrentText('ultra')
        vision_thinking.setCurrentText('max')
        self.assertEqual(profile.thinking_level, 'ultra')
        self.assertEqual(profile.vision_thinking_level, 'max')
        card.vision_model_combo.setCurrentText('gpt-5.6-luna')
        self.assertEqual(profile.thinking_level, 'ultra')
        self.assertEqual(profile.vision_thinking_level, 'max')
        self.assertEqual(vision_thinking.findText('ultra'), -1)
        card.vision_model_combo.setCurrentText('gpt-5.5')
        self.assertEqual(profile.vision_thinking_level, 'none')
        self.assertEqual(profile.thinking_level, 'ultra')
        card.model_combo.setCurrentText('gpt-5.5')
        self.assertEqual(thinking.currentText(), 'none')
        self.assertEqual(thinking.findText('max'), -1)
        card.vision_model_combo.setCurrentText('gpt-6-astra')
        self.assertEqual(vision_thinking.currentText(), 'medium')
        self.assertEqual(vision_thinking.findText('none'), -1)
        self.assertEqual(thinking.currentText(), 'none')
        card.startModelEdit()
        card.model_combo.lineEdit().setText('unknown-model')
        card.finishModelEdit()
        self.assertEqual(thinking.count(), 0)
        self.assertFalse(thinking.isEnabled())
        card.model_combo.setCurrentText('gpt-5.5')
        self.assertEqual(thinking.currentText(), 'none')
        self.assertTrue(thinking.isEnabled())
        card.toggleTextSupport()
        self.assertEqual(vision_thinking.currentText(), 'medium')
        self.assertIs(card.details.param_widgets['thinking_level'], thinking)

    def test_codex_shortcuts_share_thinking_limits_with_profile_editor(self) -> None:
        for modality, module, model_attr, effort_attr in (
            ('text', 'LLMTranslator', 'model', 'thinking_level'),
            ('vision', 'LLMOCR', 'vision_model', 'vision_thinking_level'),
        ):
            profile = default_profile('Codex')
            profile.model = profile.vision_model = 'gpt-6-astra'
            with self.subTest(modality=modality), mock.patch.object(
                pcfg, 'module', ModuleConfig(llm_profiles=[profile]),
            ):
                profile = pcfg.module.llm_profiles[0]
                widget = ModuleSelectionWidget(modality, 'text.svg', llm_modality=modality)
                self.addCleanup(widget.deleteLater)
                widget.selector.addItem(module)
                menu = QMenu(widget)
                widget.menu._buildProfileMenu(menu, profile)
                choices = {action.data()[2]: action for action in menu.actions()
                           if action.data() and action.data()[1] == effort_attr}
                self.assertEqual(set(choices), {'low', 'medium', 'high', 'xhigh', 'max', 'ultra'})
                self.assertFalse(any(action.data() and action.data()[1] == 'vision_detail_level'
                                     for action in menu.actions()))
                choices['ultra'].trigger()
                self.assertEqual(getattr(profile, effort_attr), 'ultra')
                widget.menu.selectLLMProfileSetting('codex-app-server', model_attr, 'gpt-5.6-luna')
                self.assertEqual(getattr(profile, effort_attr), 'none')
                menu.clear()
                widget.menu._buildProfileMenu(menu, profile)
                choices = {action.data()[2]: action for action in menu.actions()
                           if action.data() and action.data()[1] == effort_attr}
                self.assertEqual(set(choices), {'none', 'low', 'medium', 'high', 'xhigh', 'max'})
                self.assertIn('\u2713', choices['none'].text())
                choices['max'].trigger()
                self.assertEqual(getattr(profile, effort_attr), 'max')
                widget.menu.selectLLMProfileSetting('codex-app-server', model_attr, 'gpt-5.5')
                self.assertEqual(getattr(profile, effort_attr), 'none')

    def test_http_ocr_menu_preserves_vision_detail_options(self) -> None:
        profile = default_profile('OpenAI')
        widget = ModuleSelectionWidget('OCR', 'eye.svg', llm_modality='vision')
        self.addCleanup(widget.deleteLater)
        menu = QMenu(widget)
        widget.menu._buildProfileMenu(menu, profile)
        choices = [action.data()[2] for action in menu.actions()
                   if action.data() and action.data()[1] == 'vision_detail_level']
        self.assertEqual(choices, ['None', 'auto', 'low', 'high'])

    def test_model_text_is_selectable_and_vision_add_updates_text_choices(
        self,
    ) -> None:
        profile = LLMProfile(
            id='test',
            name='Test',
            model='text-model',
            model_options=['text-model', 'shared-model'],
            support_vision=True,
            vision_model='vision-model',
            vision_model_options=['vision-model'],
        )
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)

        for combo in (
            card.model_combo,
            card.vision_model_combo,
            card.image_model_combo,
        ):
            self.assertTrue(combo.isEditable())
            self.assertTrue(combo.lineEdit().isReadOnly())
            combo.lineEdit().selectAll()
            self.assertEqual(combo.lineEdit().selectedText(), combo.currentText())

        card.startVisionModelEdit()
        card.vision_model_combo.lineEdit().setText('new-vision-model')
        card.finishVisionModelEdit()

        self.assertEqual(profile.vision_model, 'new-vision-model')
        self.assertEqual(profile.model, 'text-model')
        self.assertIn('new-vision-model', profile.model_options)
        self.assertGreaterEqual(card.model_combo.findText('new-vision-model'), 0)

        card.startVisionModelEdit()
        card.vision_model_combo.lineEdit().setText('shared-model')
        card.finishVisionModelEdit()
        self.assertEqual(profile.model_options.count('shared-model'), 1)

    def image_card(self) -> ProfileCardWidget:
        card = ProfileCardWidget(LLMProfile(
            id='image-test', name='Image test', support_image=True, support_vision=True,
            image_base_url='https://api.example/v1/images/edits',
            image_model='gpt-image-2', image_model_options=['gpt-image-2', 'gpt-image-1'],
            vision_model='gpt-6-sol', vision_model_options=['gpt-6-sol', 'other-vision'],
        ))
        self.addCleanup(card.deleteLater)
        return card

    def test_image_pair_selection_does_not_pollute_saved_options(self) -> None:
        card = self.image_card()
        pair = 'gpt-6-sol → gpt-image-2'
        self.assertGreaterEqual(card.image_model_combo.findText(pair), 0)
        self.assertEqual(card.image_model_combo.findText('other-vision → gpt-image-2'), -1)
        card.image_model_combo.setCurrentText(pair)
        self.assertEqual(card.profile.image_model, pair)
        card.toggleImageSupport()
        card.toggleImageSupport()
        card.syncFromProfile()
        saved = profile_to_dict(card.profile)
        self.assertEqual(saved['image_model'], pair)
        self.assertEqual(saved['image_model_options'], ['gpt-image-2', 'gpt-image-1'])
        self.assertTrue(card.image_model_combo.lineEdit().isReadOnly())

    def test_custom_gateway_base_url_offers_and_saves_image_pairs(self) -> None:
        profile = LLMProfile(id='custom-gateway', name='Custom gateway', support_image=True, support_vision=True)
        profile.image_base_url = 'https://gateway.example/v1'
        profile.vision_model_options = ['gpt-test-reasoning']
        profile.image_model = 'gpt-image-test'
        profile.image_model_options = ['gpt-image-test']
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)
        pair = 'gpt-test-reasoning → gpt-image-test'
        self.assertGreaterEqual(card.image_model_combo.findText(pair), 0)
        card.image_model_combo.setCurrentText(pair)
        saved = profile_to_dict(profile)
        self.assertEqual(saved['image_model'], pair)
        self.assertEqual(saved['image_model_options'], ['gpt-image-test'])
        self.assertEqual(saved['image_base_url'], 'https://gateway.example/v1')

    def test_vision_add_delete_refreshes_pairs_and_preserves_unavailable_selection(self) -> None:
        card = self.image_card()
        image_combo = card.image_model_combo
        card.startVisionModelEdit()
        card.vision_model_combo.lineEdit().setText('gpt-6-luna')
        card.finishVisionModelEdit()
        pair = 'gpt-6-luna → gpt-image-2'
        self.assertGreaterEqual(image_combo.findText(pair), 0)
        self.assertGreaterEqual(image_combo.findText('gpt-6-luna → gpt-image-1'), 0)
        image_combo.setCurrentText(pair)
        card.deleteCurrentVisionModel()
        self.assertNotIn('gpt-6-luna', card.profile.vision_model_options)
        self.assertEqual(image_combo.findText('gpt-6-luna → gpt-image-1'), -1)
        self.assertEqual(image_combo.currentText(), pair)
        self.assertEqual(card.profile.image_model, pair)
        self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
        self.assertIs(card.image_model_combo, image_combo)

    def test_image_pair_delete_removes_underlying_image_and_all_combinations(self) -> None:
        card = self.image_card()
        card.profile.vision_model_options.append('gpt-6-luna')
        card.syncFromProfile()
        card.image_model_combo.setCurrentText('gpt-6-sol → gpt-image-2')
        card.deleteCurrentImageModel()
        choices = [card.image_model_combo.itemText(index) for index in range(card.image_model_combo.count())]
        self.assertEqual(card.profile.image_model_options, ['gpt-image-1'])
        self.assertEqual(card.profile.image_model, 'gpt-image-1')
        self.assertFalse(any('gpt-image-2' in choice for choice in choices))
        self.assertIn('gpt-6-sol → gpt-image-1', choices)
        self.assertIn('gpt-6-luna → gpt-image-1', choices)

    def test_image_add_derives_pairs_and_preserves_draft_during_sync(self) -> None:
        card = self.image_card()
        card.startImageModelEdit()
        card.image_model_combo.lineEdit().setText('gpt-image-3')
        card.profile.vision_model_options.append('gpt-6-luna')
        card.syncFromProfile()
        self.assertEqual(card.image_model_combo.lineEdit().text(), 'gpt-image-3')
        self.assertFalse(card.image_model_combo.lineEdit().isReadOnly())
        self.assertEqual(card.profile.image_model, 'gpt-image-2')
        card.finishImageModelEdit()
        self.assertEqual(card.profile.image_model, 'gpt-image-3')
        self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1', 'gpt-image-3'])
        self.assertGreaterEqual(card.image_model_combo.findText('gpt-6-luna → gpt-image-3'), 0)

    def test_image_add_rejects_pairs_and_restores_previous_value(self) -> None:
        card = self.image_card()
        for value in ('gpt-6-sol → gpt-image-2', 'gpt-6-sol -> gpt-image-3', 'other → gpt-image-2'):
            with self.subTest(value=value), patch('ballontranslator.ui.llm_profile_widgets.QMessageBox.warning') as warning:
                card.startImageModelEdit()
                card.image_model_combo.lineEdit().setText(value)
                self.assertFalse(card.finishImageModelEdit())
                warning.assert_called_once()
                self.assertEqual(card.profile.image_model, 'gpt-image-2')
                self.assertEqual(card.image_model_combo.currentText(), 'gpt-image-2')
                self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
                self.assertTrue(card.image_model_combo.lineEdit().isReadOnly())

    def test_delete_aborts_when_pending_image_add_is_rejected(self) -> None:
        card = self.image_card()
        card.startImageModelEdit()
        card.image_model_combo.lineEdit().setText('other-model → gpt-image-2')
        with patch('ballontranslator.ui.llm_profile_widgets.QMessageBox.warning') as warning:
            card.deleteCurrentImageModel()
        warning.assert_called_once()
        self.assertEqual(card.profile.image_model, 'gpt-image-2')
        self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])

    def test_clicking_delete_during_image_add_cannot_delete_saved_model(self) -> None:
        card = self.image_card()
        card.show()
        card.activateWindow()
        self.app.processEvents()
        card.setActionButtonsVisible(True)
        card.startImageModelEdit()
        card.image_model_combo.lineEdit().setText('other-model -> gpt-image-2')
        self.assertFalse(card.remove_image_model_btn.isEnabled())
        with patch('ballontranslator.ui.llm_profile_widgets.QMessageBox.warning') as warning:
            QTest.mouseClick(card.remove_image_model_btn, Qt.MouseButton.LeftButton)
            self.app.processEvents()
            self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
            QTest.mouseClick(card.model_combo.lineEdit(), Qt.MouseButton.LeftButton)
            self.app.processEvents()
        warning.assert_called_once()
        self.assertEqual(card.image_model_combo.currentText(), 'gpt-image-2')
        self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
        self.assertTrue(card.remove_image_model_btn.isEnabled())
        card.close()

    def test_image_endpoint_edits_refresh_choices_and_native_services_offer_no_pairs(self) -> None:
        card = self.image_card()
        editor = card.details.param_widgets['image_base_url']
        image_combo = card.image_model_combo
        for endpoint in (
            'https://generativelanguage.googleapis.com/v1beta/openai/',
            'https://openrouter.ai/api/v1',
            'https://api.example/v1/images/edits',
            'https://api.example/v1',
        ):
            with self.subTest(endpoint=endpoint):
                editor.setText(endpoint)
                editor.textEdited.emit(endpoint)
                editor.editingFinished.emit()
                self.assertEqual(card.profile.image_base_url, endpoint)
                self.assertEqual(image_combo.findText('gpt-6-sol → gpt-image-2') >= 0,
                                 endpoint.startswith('https://api.example/'))
                self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
                self.assertIs(card.image_model_combo, image_combo)


class LLMProfileTitleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_title_preserves_background_above_and_below_card_border(self) -> None:
        for theme in ('eva-light', 'eva-dark'):
            with self.subTest(theme=theme):
                panel = QWidget()
                self.addCleanup(panel.deleteLater)
                panel.setObjectName('ConfigContentScrollContent')
                panel.setStyleSheet(parse_stylesheet(theme))
                layout = QVBoxLayout(panel)
                card = ProfileCardWidget(LLMProfile(name='Example', title_url='https://example.com'))
                layout.addWidget(card)
                panel.show()
                self.app.processEvents()
                title_rect = card.title_label.geometry().translated(card.pos())
                rendered = panel.grab().toImage()
                # Compare empty title padding with its surroundings, on both
                # sides of the border. Neither background may become a patch.
                title_x = title_rect.right() - 1
                beside_x = title_rect.right() + 5
                above = title_rect.top()
                below = title_rect.bottom()
                self.assertNotEqual(
                    rendered.pixelColor(beside_x, above),
                    rendered.pixelColor(beside_x, below),
                )
                for y in (above, below):
                    self.assertEqual(
                        rendered.pixelColor(title_x, y),
                        rendered.pixelColor(beside_x, y),
                    )
                panel.close()

    def test_link_click_and_single_field_edit_roundtrip(self) -> None:
        profile = LLMProfile(name='Example', title_url='https://example.com')
        url = profile.title_url
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)
        card.show()
        self.app.processEvents()

        with patch('ballontranslator.ui.llm_profile_widgets.QDesktopServices.openUrl') as open_url:
            QTest.mouseClick(card.title_label, Qt.MouseButton.LeftButton)
            self.assertEqual(open_url.call_args[0][0].toString(), url)
            self.assertFalse(card.name_edit.isVisible())

        with patch.object(QMenu, 'exec', lambda menu, *args: menu.actions()[0]):
            position = card.title_label.rect().center()
            self.app.sendEvent(card.title_label, QContextMenuEvent(
                QContextMenuEvent.Reason.Mouse, position,
                card.title_label.mapToGlobal(position),
            ))
        self.assertTrue(card.name_edit.isVisible())
        self.assertEqual(card.name_edit.text(), f'[Example]({url})')
        card.name_edit.setText(f'[A & <B>]({url})')
        QTest.keyClick(card.name_edit, Qt.Key.Key_Return)
        self.assertEqual(profile.name, 'A & <B>')
        self.assertEqual(profile.title_url, url)
        self.assertIn('A &amp; &lt;B&gt;', card.title_label.text())
        card.startNameEdit()
        self.assertEqual(card.name_edit.text(), f'[A & <B>]({url})')
        card.name_edit.setText('Plain <title>')
        QTest.mouseClick(card.model_combo.lineEdit(), Qt.MouseButton.LeftButton)
        self.assertEqual(profile.name, 'Plain <title>')
        self.assertEqual(profile.title_url, '')
        self.assertEqual(card.title_label.textFormat(), Qt.TextFormat.PlainText)
        with patch('ballontranslator.ui.llm_profile_widgets.QDesktopServices.openUrl') as open_url:
            QTest.mouseClick(card.title_label, Qt.MouseButton.LeftButton)
            open_url.assert_not_called()
        QTest.mouseDClick(card.title_label, Qt.MouseButton.LeftButton)
        self.assertTrue(card.name_edit.isVisible())
        card.close()

    def test_malformed_or_non_web_links_remain_plain_text(self) -> None:
        card = ProfileCardWidget(LLMProfile(name='Original'))
        self.addCleanup(card.deleteLater)
        for text in ('[Broken](https://)', '[Local](file:///tmp/file)', '[Broken](https://[bad)', '<b>Plain</b>'):
            with self.subTest(text=text):
                card.startNameEdit()
                card.name_edit.setText(text)
                card.on_name_edit_finished()
                self.assertEqual(card.profile.name, text)
                self.assertEqual(card.profile.title_url, '')
                self.assertEqual(card.title_label.textFormat(), Qt.TextFormat.PlainText)


class APIProfilesPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_repeated_profile_copies_have_unique_ids_and_independent_settings(self) -> None:
        source = default_profile('Gemini')
        source.api_key = 'saved-test-key'
        original = profile_to_dict(source)
        with patch.object(pcfg.module, 'llm_profiles', [source]):
            panel = LLMProfilesWidget()
            self.addCleanup(panel.deleteLater)
            panel.copyProfile(source.id)
            panel.copyProfile(source.id)
            self.app.processEvents()
            first, second = pcfg.module.llm_profiles[1:]
            self.assertEqual(len({source.id, first.id, second.id}), 3)
            for copied in (first, second):
                self.assertIn(copied.id, panel.rows)
                self.assertEqual(profile_to_dict(copied), {
                    **original, 'id': copied.id, 'name': source.name + ' Copy', 'built_in': False,
                })
            first.model_options.append('copy-only-model')
            self.assertNotIn('copy-only-model', source.model_options)
            self.assertNotIn('copy-only-model', second.model_options)
            self.assertEqual(profile_to_dict(source), original)

    def test_codex_is_excluded_from_profile_editing_copy_and_delete(self) -> None:
        codex = default_codex_profile()
        api = default_profile('OpenAI')
        with patch.object(pcfg.module, 'llm_profiles', [codex, api]):
            panel = LLMProfilesWidget()
            self.addCleanup(panel.deleteLater)
            self.assertNotIn(codex.id, panel.rows)
            self.assertIn('base_url', panel.rows[api.id].details.param_widgets)
            panel.rows[api.id].toggleVisionSupport()
            self.assertFalse(api.support_vision)
            previous_clipboard = QApplication.clipboard().text()
            panel.copyProfileAsJson(codex.id)
            self.assertEqual(QApplication.clipboard().text(), previous_clipboard)
            panel.copyProfile(codex.id)
            panel.deleteProfile(codex.id)
            self.assertEqual(pcfg.module.llm_profiles, [codex, api])


if __name__ == '__main__':
    unittest.main()
