import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, QRectF, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager, TextPanel
from ballontranslator.ui.text_engine.formatting import presets
from ballontranslator.utils import config, shared
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


class TextStyleShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.original_styles = list(config.text_styles)
        self.addCleanup(setattr, config, 'active_format', config.active_format)
        self.host = QWidget()
        self.canvas = Canvas(self.host)
        self.canvas.editor_index = 1
        self.canvas.imgtrans_proj = ProjImgTrans()
        canvas_patch = patch.object(SW, 'canvas', self.canvas)
        canvas_patch.start()
        self.addCleanup(canvas_patch.stop)
        with patch.object(shared, 'register_view_widget', create=True):
            self.panel = TextPanel(self.app, self.host)
        self.panel.formatpanel.global_format = FontFormat()
        self.manager = SceneTextManager(
            self.app, self.host, self.canvas, self.panel, parent=self.host,
        )
        self.styles = self.panel.formatpanel.textstyle_panel
        config.text_styles[:] = [
            FontFormat(_style_name=f'Style {index + 1}', font_size=24 + index)
            for index in range(11)
        ]
        self.styles.initStyles(config.text_styles)
        self.items = [
            self.manager.addTextBlock(TextBlock(
                [index * 240, 0, index * 240 + 200, 120],
                _bounding_rect=[index * 240, 0, 200, 120],
                translation='Text',
                fontformat=FontFormat(font_size=16, vertical=bool(index)),
            ))
            for index in range(2)
        ]
        self.canvas.baseLayer.setRect(QRectF(0, 0, 600, 400))
        self.canvas._set_scene_scale(1.0)
        self.canvas.gv.show()
        self.canvas.gv.setFocus()
        self.app.processEvents()
        self.items[0].setSelected(True)
        self.manager._update_selection_panels([self.items[0]])

    def tearDown(self) -> None:
        self.canvas.text_move_session.cancel()
        self.canvas.clear_text_transform_controls()
        self.canvas.text_undo_stack.clear()
        self.manager.clearSceneTextitems()
        for item in self.items:
            item.geometry_controller.release_render_resources()
        self.canvas.gv.close()
        self.canvas.gv.deleteLater()
        self.host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        config.text_styles[:] = self.original_styles

    def test_digits_activate_and_apply_styles_with_undo(self) -> None:
        self.items[1].setSelected(True)
        self.manager._update_selection_panels(self.items)
        for index, digit in enumerate('1234567890'):
            with self.subTest(digit=digit):
                previous = [item.fontformat.font_size for item in self.items]
                QTest.keyClicks(self.canvas.gv.viewport(), digit)
                style = config.text_styles[index]
                self.assertIs(self.styles.active_text_style_label.fontfmt, style)
                self.assertTrue(self.styles.active_text_style_label.active)
                self.assertEqual(self.panel.formatpanel.global_format.font_size, style.font_size)
                self.assertEqual(self.canvas.text_undo_stack.count(), index + 1)
                self.assertEqual(
                    [item.fontformat.font_size for item in self.items],
                    [style.font_size] * 2,
                )
                self.canvas.undo()
                self.assertEqual(
                    [item.fontformat.font_size for item in self.items], previous,
                )
                self.canvas.redo()
                self.assertEqual(
                    [item.fontformat.font_size for item in self.items],
                    [style.font_size] * 2,
                )
        QTest.keyClicks(self.canvas.gv.viewport(), '0')
        self.assertIs(self.styles.active_text_style_label.fontfmt, config.text_styles[9])
        self.assertTrue(self.styles.active_text_style_label.active)

    def test_repeated_style_applies_to_new_selection(self) -> None:
        QTest.keyClicks(self.canvas.gv.viewport(), '1')
        self.assertEqual(self.items[0].fontformat.font_size, 24)
        self.assertEqual(self.items[1].fontformat.font_size, 16)
        self.canvas.clearSelection()
        self.items[1].setSelected(True)
        self.manager._update_selection_panels([self.items[1]])
        QTest.keyClick(self.canvas.gv.viewport(), Qt.Key.Key_1, Qt.KeyboardModifier.KeypadModifier)
        self.assertEqual(self.items[1].fontformat.font_size, 24)
        self.assertTrue(self.styles.active_text_style_label.active)

    def test_shortcuts_follow_reorder_removal_and_reload(self) -> None:
        with patch.object(presets, 'save_text_styles'):
            self.styles._moveStyleLabelToIndex(10, 0)
            QTest.keyClicks(self.canvas.gv.viewport(), '1')
            self.assertEqual(self.items[0].fontformat.font_size, 34)
            self.styles.removeStyleLabel(self.styles.active_text_style_label)
            QTest.keyClicks(self.canvas.gv.viewport(), '1')
            self.assertEqual(self.items[0].fontformat.font_size, 24)
            config.text_styles[:] = [FontFormat(_style_name='Reloaded', font_size=48)]
            self.styles.setStyles(config.text_styles)
            QTest.keyClicks(self.canvas.gv.viewport(), '1')
            active_label = self.styles.active_text_style_label
            self.assertEqual(self.items[0].fontformat.font_size, 48)
            command_count = self.canvas.text_undo_stack.count()
            QTest.keyClicks(self.canvas.gv.viewport(), '234567890')
            self.assertIs(self.styles.active_text_style_label, active_label)
            self.assertEqual(self.canvas.text_undo_stack.count(), command_count)
            self.styles.clearStyles()
            QTest.keyClicks(self.canvas.gv.viewport(), '1234567890')
            self.assertIsNone(self.styles.active_text_style_label)
            self.assertEqual(self.canvas.text_undo_stack.count(), command_count)

    def test_no_selection_only_activates_global_style(self) -> None:
        self.canvas.clearSelection()
        self.manager._update_selection_panels([])
        QTest.keyClicks(self.canvas.gv.viewport(), '0')
        self.assertIs(self.styles.active_text_style_label.fontfmt, config.text_styles[9])
        self.assertEqual(self.panel.formatpanel.global_format.font_size, 33)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)
        self.assertEqual([item.fontformat.font_size for item in self.items], [16, 16])

    def test_typing_digits_does_not_apply_styles(self) -> None:
        item = self.items[0]
        item.startEdit()
        self.app.processEvents()
        QTest.keyClicks(self.canvas.gv.viewport(), '1234567890')
        self.assertIn('1234567890', item.toPlainText())
        self.assertIsNone(self.styles.active_text_style_label)
        self.assertEqual(item.fontformat.font_size, 16)
        item.endEdit()

        edit = self.manager.pairwidget_list[0].e_trans
        QTest.keyClicks(edit, '1234567890')
        self.assertIn('1234567890', edit.toPlainText())
        self.assertIsNone(self.styles.active_text_style_label)

    def test_modified_keys_and_drawing_mode_do_not_apply_styles(self) -> None:
        for modifier in (
            Qt.KeyboardModifier.ControlModifier,
            Qt.KeyboardModifier.AltModifier,
            Qt.KeyboardModifier.ShiftModifier,
            Qt.KeyboardModifier.MetaModifier,
        ):
            QTest.keyClick(self.canvas.gv.viewport(), Qt.Key.Key_1, modifier)
        self.canvas.editor_index = 0
        QTest.keyClicks(self.canvas.gv.viewport(), '1234567890')
        self.assertIsNone(self.styles.active_text_style_label)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)


if __name__ == '__main__':
    unittest.main()
