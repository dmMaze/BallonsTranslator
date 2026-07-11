import os
import unittest
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QFrame, QLabel, QPushButton, QTextBrowser

from ballontranslator.ui.update_dialog import (
    RELEASE_WINDOW_WIDTH,
    UpdateReleaseDialog,
    format_release_markdown,
    select_release_note_section,
    simplified_release_date,
    strip_release_images,
)


def get_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class UpdateReleaseDialogTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def test_strips_images_without_removing_release_markdown(self):
        markdown = strip_release_images(
            '<img src="https://example.invalid/demo.gif" />\n\n- Fixed UI\n\n![Demo](demo.png)'
        )

        self.assertEqual(markdown, '- Fixed UI')

    def test_formats_unordered_list_spacing(self):
        self.assertEqual(format_release_markdown('* Fixed UI'), '* \xa0Fixed UI')

    def test_selects_localized_section_and_removes_headings(self):
        markdown = (
            '## Changelog\n'
            '### Fixes\n'
            '- Fixed UI\n'
            '## 更新说明\n'
            '### 修复\n'
            '- 修复界面'
        )

        self.assertEqual(
            select_release_note_section(markdown, 'en_US'),
            '- Fixed UI',
        )
        self.assertEqual(
            select_release_note_section(markdown, 'zh_CN'),
            '- 修复界面',
        )
        self.assertEqual(
            select_release_note_section(markdown, 'zh-CN'),
            '- 修复界面',
        )

    def test_release_section_fallbacks_handle_missing_or_empty_content(self):
        self.assertEqual(
            select_release_note_section('## 更新说明\n- 修复界面', 'en_US'),
            '',
        )
        self.assertEqual(
            select_release_note_section('# Notes\n- Fixed UI', 'en_US'),
            '- Fixed UI',
        )
        self.assertEqual(select_release_note_section('## Changelog', 'en_US'), '')
        self.assertEqual(select_release_note_section('', 'zh_CN'), '')

    def test_localized_sections_are_boundaries_at_different_heading_levels(self):
        markdown = '# Changelog\n- Fixed UI\n### 更新说明\n- 修复界面'

        self.assertEqual(
            select_release_note_section(markdown, 'en_US'),
            '- Fixed UI',
        )
        self.assertEqual(
            select_release_note_section(markdown, 'zh_CN'),
            '- 修复界面',
        )

    def test_preview_dialog_has_only_a_safe_close_action(self):
        release_info = SimpleNamespace(
            tag_name='v1.5.6',
            name='v1.5.6',
            html_url='https://example.invalid/v1.5.6',
            body='',
            published_at='',
        )
        result = SimpleNamespace(
            current_version='1.5.6',
            latest_version='1.5.6',
            release_info=release_info,
        )

        dialog = UpdateReleaseDialog(result, allow_update=False)

        self.assertIsNotNone(dialog.findChild(QPushButton, 'UpdateDialogPrimaryButton'))
        self.assertIsNone(dialog.findChild(QPushButton, 'UpdateDialogCancelButton'))
        dialog.close()

    def test_dialog_uses_linked_version_and_markdown_notes(self):
        release_info = SimpleNamespace(
            tag_name='v1.5.6',
            name='v1.5.6',
            html_url='https://example.invalid/v1.5.6',
            body='## Changes\n\n- Fixed UI',
            published_at='2026-07-03T06:45:13Z',
        )
        result = SimpleNamespace(
            current_version='1.5.5',
            latest_version='1.5.6',
            release_info=release_info,
        )

        dialog = UpdateReleaseDialog(result)
        dialog.show()
        self.app.processEvents()

        self.assertEqual(dialog.width(), RELEASE_WINDOW_WIDTH)
        window_type = getattr(Qt, 'WindowType', Qt)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        self.assertTrue(dialog.windowFlags() & window_type.Popup)
        self.assertTrue(dialog.windowFlags() & window_type.FramelessWindowHint)
        self.assertTrue(dialog.testAttribute(widget_attribute.WA_TranslucentBackground))
        self.assertIsNotNone(dialog.findChild(QFrame, 'UpdateReleaseSurface'))
        self.assertTrue(dialog.findChild(QLabel, 'UpdateReleaseVersion').openExternalLinks())
        self.assertEqual(simplified_release_date(release_info.published_at), '2026-07-03')
        self.assertEqual(dialog.release_notes.verticalScrollBar().maximum(), 0)
        self.assertIsNotNone(dialog.findChild(QTextBrowser, 'UpdateReleaseNotes'))
        restart_notice = dialog.findChild(QLabel, 'UpdateReleaseRestartNotice')
        self.assertIsNotNone(restart_notice)
        self.assertTrue(
            restart_notice.alignment() & Qt.AlignmentFlag.AlignRight
        )
        dialog.close()

if __name__ == '__main__':
    unittest.main()
