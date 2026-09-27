import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication

from ballontranslator.modules import codex
from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.utils.config import ModuleConfig, pcfg
from ballontranslator.utils.llm_profiles import default_profile, profile_by_id


class CodexConfigPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        module = ModuleConfig(
            llm_profiles=[default_profile('OpenAI')],
            codex_models={'example-model': {'modalities': ['text', 'image'], 'efforts': ['high']}},
        )
        patcher = patch.object(pcfg, 'module', module)
        patcher.start()
        self.addCleanup(patcher.stop)
        with patch.object(codex.account, '_load', side_effect=AssertionError('Unexpected credential IO')):
            self.panel = ConfigPanel()
        self.addCleanup(self.panel.deleteLater)

    def tearDown(self) -> None:
        self.doCleanups()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        super().tearDown()

    def test_profile_edit_navigation_opens_codex_and_api_sections(self) -> None:
        self.panel.focusOnLLMProfile('codex', target='vision_model')
        self.app.processEvents()
        self.assertEqual(self.panel.configContent.currentIndex(), self.panel.configContent.section_index['codex'])
        self.assertNotIn('codex', self.panel.llm_profiles_panel.rows)
        self.assertTrue(self.panel.codex_panel.param_widgets['vision_model'].hasFocus())
        self.panel.focusOnLLMProfile('openai', target='api_key', expand_details=False)
        self.app.processEvents()
        self.assertEqual(self.panel.configContent.currentIndex(), self.panel.configContent.section_index['llm_profile'])
        self.assertTrue(self.panel.llm_profiles_panel.rows['openai'].api_key_widget.editor.hasFocus())

    def test_shared_selector_edits_update_canonical_panel(self) -> None:
        profile = profile_by_id(pcfg.module.llm_profiles, 'codex')
        profile.model = 'example-model'
        self.panel.syncLLMProfile(profile.id)
        self.assertEqual(self.panel.codex_panel.param_widgets['model'].currentText(), 'example-model')
        api = profile_by_id(pcfg.module.llm_profiles, 'openai')
        api.model = 'gpt-4o'
        self.panel.syncLLMProfile(api.id)
        self.assertEqual(self.panel.llm_profiles_panel.rows['openai'].model_combo.currentText(), 'gpt-4o')
        self.assertEqual(profile.model, 'example-model')


if __name__ == '__main__':
    unittest.main()
