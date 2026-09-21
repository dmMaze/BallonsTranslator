import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtGui import QContextMenuEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QMenu, QVBoxLayout, QWidget

from ballontranslator.ui.llm_profile_widgets import ProfileCardWidget
from ballontranslator.ui.misc import parse_stylesheet
from ballontranslator.utils.llm_profiles import LLMProfile


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


if __name__ == '__main__':
    unittest.main()
