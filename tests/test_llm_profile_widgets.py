import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtGui import QContextMenuEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QMenu, QVBoxLayout, QWidget

from ballontranslator.ui.llm_profile_widgets import LLMProfilesWidget, ProfileCardWidget
from ballontranslator.ui.misc import parse_stylesheet
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, LLMProfile, default_profile, profile_to_dict


class LLMProfileModelSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

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

    def test_infistar_base_url_offers_and_saves_image_pairs(self) -> None:
        profile = default_profile('Infistar')
        profile.image_base_url = 'https://infistar.cc/v1'
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
        self.assertEqual(saved['image_base_url'], 'https://infistar.cc/v1')

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
