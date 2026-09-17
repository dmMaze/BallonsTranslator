import os
import json
import tempfile
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication, QMenu

from ballontranslator.ui.llm_profile_widgets import ProfileCardWidget
from ballontranslator.ui.module_tool_button import ModuleSelectionWidget
from ballontranslator.utils.config import ModuleConfig, ProgramConfig, pcfg
from ballontranslator.utils.llm_profiles import LLMProfile, default_profile


class LLMProfileModelSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_upgraded_config_exposes_selectable_codex_in_translation_and_ocr_menus(self) -> None:
        cockpit = default_profile('Ollama')
        cockpit.id = 'cockpit'
        cockpit.name = 'Cockpit 本機'
        cockpit.built_in = False
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'config.json')
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump({'module': {'llm_profiles': [cockpit.to_dict()],
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
                    widget.rebuildMenu()
                    codex_menu = next((action.menu() for action in widget.menu.actions()
                                       if action.menu() and action.menu().title() == 'Codex'), None)
                    self.assertIsNotNone(codex_menu, f'Codex missing from {modality} menu')
                    selected = mock.Mock()
                    widget.llm_profile_changed.connect(selected)
                    for model in ('gpt-5.5', 'gpt-6-astra', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'):
                        with self.subTest(model=model):
                            selected.reset_mock()
                            model_action = next(action for action in codex_menu.actions()
                                                if action.data() == ('codex', model_attr, model))
                            model_action.trigger()
                            self.assertEqual(getattr(cfg.module, selection), 'codex')
                            self.assertEqual(widget.selector.currentText(), module)
                            self.assertEqual(getattr(cfg.module.llm_profiles[-1], model_attr), model)
                            selected.assert_called_once_with('codex')

    def test_backend_switch_exposes_codex_settings_without_rebuilding(self) -> None:
        profile = default_profile('OpenAI')
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
                widget._buildProfileMenu(menu, profile)
                choices = {action.data()[2]: action for action in menu.actions()
                           if action.data() and action.data()[1] == effort_attr}
                self.assertEqual(set(choices), {'low', 'medium', 'high', 'xhigh', 'max', 'ultra'})
                self.assertFalse(any(action.data() and action.data()[1] == 'vision_detail_level'
                                     for action in menu.actions()))
                choices['ultra'].trigger()
                self.assertEqual(getattr(profile, effort_attr), 'ultra')
                widget.selectLLMProfileSetting('codex', model_attr, 'gpt-5.6-luna')
                self.assertEqual(getattr(profile, effort_attr), 'none')
                menu.clear()
                widget._buildProfileMenu(menu, profile)
                choices = {action.data()[2]: action for action in menu.actions()
                           if action.data() and action.data()[1] == effort_attr}
                self.assertEqual(set(choices), {'none', 'low', 'medium', 'high', 'xhigh', 'max'})
                self.assertIn('\u2713', choices['none'].text())
                choices['max'].trigger()
                self.assertEqual(getattr(profile, effort_attr), 'max')
                widget.selectLLMProfileSetting('codex', model_attr, 'gpt-5.5')
                self.assertEqual(getattr(profile, effort_attr), 'none')

    def test_http_ocr_menu_preserves_vision_detail_options(self) -> None:
        profile = default_profile('OpenAI')
        widget = ModuleSelectionWidget('OCR', 'eye.svg', llm_modality='vision')
        self.addCleanup(widget.deleteLater)
        menu = QMenu(widget)
        widget._buildProfileMenu(menu, profile)
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


if __name__ == '__main__':
    unittest.main()
