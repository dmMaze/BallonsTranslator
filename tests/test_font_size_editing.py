import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QLocale, Qt
from qtpy.QtGui import QTextCursor, QTextCharFormat
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel, FontSizeComboBox
from ballontranslator.ui.text_engine.editing.commands import TextItemEditCommand
from ballontranslator.utils import shared, config as C
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.fontformat import (
    FontFormat, LineSpacingType, TextTransformStack, BendTextTransform,
)


class FontSizeEditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.canvas = Canvas()
        self.canvas.editor_index = 1
        self.canvas_patcher = patch.object(SW, 'canvas', self.canvas)
        self.canvas_patcher.start()
        self.addCleanup(self.canvas_patcher.stop)
        previous = C.active_format
        self.addCleanup(setattr, C, 'active_format', previous)
        with patch.object(shared, 'register_view_widget', lambda *_: None, create=True):
            self.panel = FontFormatPanel(self.app)
        self.panel.global_format = FontFormat()
        block = TextBlock([0, 0, 300, 180])
        block._bounding_rect = [0, 0, 300, 180]
        block.translation = 'One two three four five six seven eight'
        block.fontformat.font_size = 20
        self.item = TextBlkItem(block, 0)
        self.item.setParentItem(self.canvas.textLayer)
        self.item.setSelected(True)
        self.panel.set_textblk_item(self.item)
        self.session = self.panel.font_size_session
        self.box = self.panel.fontsizebox
        self.item.push_undo_stack.connect(self._push_document_edit)
        self.addCleanup(self.canvas.gv.deleteLater)
        self.addCleanup(self.panel.deleteLater)

    def _push_document_edit(self, steps, formatting):
        self.canvas.push_undo_command(TextItemEditCommand(self.item, None, steps, self.panel))

    def test_typing_commits_once_on_enter_and_focus_loss(self):
        self.panel.show()
        edit = self.box.fcombobox
        edit.setFocus()
        self.app.processEvents()
        before = self.item.toHtml()
        edit.lineEdit().selectAll()
        QTest.keyClicks(edit.lineEdit(), '36')
        self.assertEqual(self.item.toHtml(), before)
        QTest.keyClick(edit.lineEdit(), Qt.Key.Key_Return)
        self.assertAlmostEqual(self.item.get_fontformat().font_size, 36)
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.box.drag_label.setFocus()
        self.app.processEvents()
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)

    def test_dropdown_choice_commits_once(self):
        self.panel.show()
        edit = self.box.fcombobox
        edit.setFocus()
        self.app.processEvents()
        edit.showPopup()
        self.app.processEvents()
        index = edit.model().index(edit.findText('36'), 0)
        edit.view().scrollTo(index)
        self.app.processEvents()
        QTest.mouseClick(edit.view().viewport(), Qt.MouseButton.LeftButton,
                         pos=edit.view().visualRect(index).center())
        self.app.processEvents()
        self.assertAlmostEqual(self.item.get_fontformat().font_size, 36)
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)

    def test_typing_previous_size_after_stepper_commits(self):
        self.panel.show()
        self.box.onUpBtnClicked()
        self.app.processEvents()
        edit = self.box.fcombobox
        edit.lineEdit().selectAll()
        QTest.keyClicks(edit.lineEdit(), '20')
        self.box.drag_label.setFocus()
        self.app.processEvents()
        self.assertAlmostEqual(self.item.get_fontformat().font_size, 20)
        self.assertEqual(self.canvas.text_undo_stack.count(), 2)

    def test_whole_item_preview_commit_and_history(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                self.item.setVertical(vertical)
                before = self.item.toHtml()
                rect = self.item.absBoundingRect(qrect=True)
                self.session.begin()
                self.session.preview(100)
                self.assertEqual(self.item.toHtml(), before)
                self.assertIsNotNone(self.item.geometry_controller.preview)
                self.session.commit()
                self.assertIsNone(self.item.geometry_controller.preview)
                self.assertAlmostEqual(self.item.get_fontformat().font_size, 40)
                self.assertAlmostEqual(self.item.absBoundingRect(qrect=True).width(), rect.width() * 2)
                after = self.item.toHtml()
                self.canvas.text_undo_stack.undo()
                self.assertEqual(self.item.toHtml(), before)
                self.assertEqual(self.item.absBoundingRect(qrect=True), rect)
                self.canvas.text_undo_stack.redo()
                self.assertEqual(self.item.toHtml(), after)
                self.canvas.text_undo_stack.undo()

    def test_selected_range_has_no_preview_and_commits_once(self):
        self.item.startEdit()
        cursor = self.item.textCursor()
        cursor.setPosition(3)
        cursor.setPosition(0, QTextCursor.MoveMode.KeepAnchor)
        self.item.setTextCursor(cursor)
        before = self.item.toHtml()
        self.session.begin()
        self.session.preview(100)
        self.assertIsNone(self.item.geometry_controller.preview)
        self.assertEqual(self.item.toHtml(), before)
        self.session.commit()
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.assertEqual(self.item.textCursor().anchor(), 3)
        self.assertEqual(self.item.textCursor().position(), 0)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.toHtml(), before)

    def test_caret_uses_whole_item_preview_and_cancel(self):
        self.item.startEdit()
        cursor = self.item.textCursor()
        cursor.setPosition(2)
        self.item.setTextCursor(cursor)
        before = self.item.toHtml()
        self.session.begin()
        self.session.preview(50)
        self.assertIsNotNone(self.item.geometry_controller.preview)
        self.box.drag_label.cancel_drag()
        self.assertIsNone(self.item.geometry_controller.preview)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.item.textCursor().position(), 2)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_mixed_sizes_keep_ratios_and_previous_typing_history(self):
        self.canvas.gv.show()
        self.canvas.gv.activateWindow()
        self.canvas.gv.setFocus()
        self.item.startEdit()
        self.app.processEvents()
        self.assertTrue(self.item.hasFocus())
        cursor = self.item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText('!')
        cursor.setPosition(0)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setFontPointSize(30)
        cursor.mergeCharFormat(fmt)
        self.item.endEdit()
        self.panel.set_active_format(self.item.get_fontformat(), True)
        before = self.item.toHtml()
        count = self.canvas.text_undo_stack.count()
        self.session.begin()
        self.session.preview(100)
        self.session.commit()
        self.assertEqual(self.canvas.text_undo_stack.count(), count + 1)
        cursor.setPosition(1)
        self.assertEqual(cursor.charFormat().fontPointSize(), 60)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.toHtml(), before)
        self.canvas.text_undo_stack.undo()
        self.canvas.text_undo_stack.undo()
        self.assertFalse(self.item.toPlainText().endswith('!'))

    def test_save_cancels_preview_and_late_release_does_nothing(self):
        before = self.item.toHtml()
        self.session.begin()
        self.session.preview(80)
        self.panel.resolve_text_transform_edits_for_save()
        self.session.commit()
        self.assertIsNone(self.item.geometry_controller.preview)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_multi_item_drag_is_one_command(self):
        block = TextBlock([400, 0, 600, 180])
        block._bounding_rect = [400, 0, 200, 180]
        block.translation = 'Another item'
        block.fontformat.font_size = 30
        other = TextBlkItem(block, 1)
        other.setParentItem(self.canvas.textLayer)
        other.setSelected(True)
        self.panel.set_textblk_item(None, multi_select=True)
        old = [item.toHtml() for item in (self.item, other)]
        self.session.begin()
        self.session.preview(100)
        self.session.commit()
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.assertAlmostEqual(self.item.get_fontformat().font_size, 40)
        self.assertAlmostEqual(other.get_fontformat().font_size, 60)
        self.assertTrue(self.panel.global_mode())
        self.canvas.text_undo_stack.undo()
        self.assertEqual([item.toHtml() for item in (self.item, other)], old)

    def test_distance_spacing_and_existing_transform_survive_resize(self):
        self.item._set_line_spacing_pair(3.0, LineSpacingType.Distance)
        state = TextTransformStack((BendTextTransform(0.3),))
        self.item.set_text_transform(state)
        self.panel.set_active_format(self.item.get_fontformat())
        before = self.item.toHtml()
        self.session.begin()
        self.session.preview(100)
        self.assertEqual(self.item.geometry_controller.canonical(), state)
        self.session.commit()
        self.assertEqual(self.item.geometry_controller.canonical(), state)
        self.assertAlmostEqual(self.item.line_spacing_values()[0], 6)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.toHtml(), before)
        self.assertAlmostEqual(self.item.line_spacing_values()[0], 3)

    def test_click_without_motion_and_escape_leave_no_session(self):
        self.box.drag_label.setFocus()
        self.panel.show()
        self.app.processEvents()
        label = self.box.drag_label
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

        QTest.mouseClick(label, Qt.MouseButton.LeftButton)
        self.assertIsNone(self.session.start_text)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)
        QTest.mousePress(label, Qt.MouseButton.LeftButton)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)
        label.size_ctrl_changed.emit(100)
        self.assertIsNotNone(self.item.geometry_controller.preview)
        QTest.keyClick(label, Qt.Key.Key_Escape)
        self.assertIsNone(self.item.geometry_controller.preview)
        QTest.mouseRelease(label, Qt.MouseButton.LeftButton)
        self.assertIsNone(self.item.geometry_controller.preview)
        self.assertIsNone(self.session.start_text)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_empty_item_resize_undo_restores_insertion_size(self):
        self.item.setPlainText('')
        self.item.startEdit()
        self.panel.set_active_format(self.item.get_fontformat())
        before = self.item.document().defaultFont()
        insertion_size = self.item.textCursor().charFormat().fontPointSize()
        self.session.begin()
        self.session.preview(100)
        self.session.commit()
        after = self.item.document().defaultFont()
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.document().defaultFont(), before)
        self.assertEqual(self.item.textCursor().charFormat().fontPointSize(), insertion_size)
        self.canvas.text_undo_stack.redo()
        self.assertEqual(self.item.document().defaultFont(), after)
        self.assertEqual(self.item.textCursor().charFormat().fontPointSize(), insertion_size * 2)

    def test_resize_uses_live_size_and_refreshes_once(self):
        self.panel.on_param_changed('font_size', 36.04)
        self.panel.set_active_format(self.item.get_fontformat())
        self.assertEqual(self.box.getFontSize(), '36')
        with patch.object(self.session, 'refresh', wraps=self.session.refresh) as refresh:
            self.session.begin()
            self.session.preview(100)
            self.session.commit()
        self.assertEqual(refresh.call_count, 1)
        self.assertAlmostEqual(self.item.fontformat.font_size, 72.08)
        self.assertAlmostEqual(self.item.get_fontformat().font_size, 72.08, places=1)

    def test_blank_paragraph_sizes_follow_resize_and_history(self):
        self.item.setPlainText('One\n\nTwo')
        blank = self.item.document().firstBlock().next()
        before = QTextCursor(blank).charFormat().fontPointSize()
        self.session.begin()
        self.session.preview(100)
        self.session.commit()
        self.assertEqual(QTextCursor(blank).charFormat().fontPointSize(), before * 2)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(QTextCursor(blank).charFormat().fontPointSize(), before)
        self.canvas.text_undo_stack.redo()
        self.assertEqual(QTextCursor(blank).charFormat().fontPointSize(), before * 2)

    def test_size_validator_matches_numeric_parser_in_comma_locale(self):
        previous = QLocale()
        try:
            QLocale.setDefault(QLocale('de_DE'))
            box = FontSizeComboBox(self.panel)
        finally:
            QLocale.setDefault(previous)
        self.addCleanup(box.deleteLater)
        box.setEditText('12,5')
        self.assertFalse(box.lineEdit().hasAcceptableInput())
        box.setEditText('12.5')
        self.assertTrue(box.lineEdit().hasAcceptableInput())
        self.assertEqual(box.value(), 12.5)

    def test_reentering_displayed_size_unifies_selected_mixed_sizes(self):
        self.item.startEdit()
        cursor = self.item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setFontPointSize(30)
        cursor.mergeCharFormat(fmt)
        cursor.setPosition(3)
        cursor.setPosition(0, QTextCursor.MoveMode.KeepAnchor)
        self.item.setTextCursor(cursor)
        self.panel.set_active_format(self.item.get_fontformat())
        self.panel.show()
        edit = self.box.fcombobox
        edit.setFocus()
        self.app.processEvents()
        previous_count = self.canvas.text_undo_stack.count()
        displayed = edit.currentText()
        edit.lineEdit().selectAll()
        QTest.keyClicks(edit.lineEdit(), displayed)
        QTest.keyClick(edit.lineEdit(), Qt.Key.Key_Return)
        self.assertEqual(self.canvas.text_undo_stack.count(), previous_count + 1)
        cursor.setPosition(2)
        self.assertAlmostEqual(cursor.charFormat().fontPointSize(), float(displayed) * 0.75)


if __name__ == '__main__':
    unittest.main()
