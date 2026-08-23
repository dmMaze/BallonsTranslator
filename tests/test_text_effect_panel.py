import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication, QWidget

try:
    from qtpy.QtGui import QUndoStack
except ImportError:
    from qtpy.QtWidgets import QUndoStack

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.custom_widget import (
    ColorPickerLabel,
    NestedColorPickerLabel,
)
from ballontranslator.ui.text_engine.formatting.commands import (
    handle_ffmt_change,
)
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.text_effects import (
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
    with_primary_stroke,
)
from ballontranslator.utils.textblock import TextBlock


class _PanelCanvas:
    def __init__(self) -> None:
        self.stack = QUndoStack()
        self.selected = []
        self.gv = QWidget()

    def push_undo_command(self, command) -> None:
        self.stack.push(command)

    def selected_text_items(self):
        return list(self.selected)

    def clear_text_transform_controls(self) -> None:
        pass


class TextEffectPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_canvas = getattr(SW, 'canvas', None)
        self.old_active_format = C.active_format
        self.canvas = _PanelCanvas()
        SW.canvas = self.canvas
        self.registered = []
        self.register_patch = patch.object(
            shared,
            'register_view_widget',
            side_effect=self.registered.append,
            create=True,
        )
        self.register_patch.start()
        self.panel = FontFormatPanel(self.app)
        self.panel.global_format = FontFormat()
        self.panel.set_active_format(self.panel.global_format)

    def tearDown(self) -> None:
        self.panel.deleteLater()
        self.canvas.gv.deleteLater()
        self.register_patch.stop()
        SW.canvas = self.old_canvas
        C.active_format = self.old_active_format
        self.app.processEvents()

    @staticmethod
    def _stack(*strokes, opacity=1.0):
        return TextEffectStack(opacity, tuple(strokes))

    @classmethod
    def _item(cls, stack=None):
        block = TextBlock([0, 0, 320, 180])
        block._bounding_rect = [0, 0, 320, 180]
        block.translation = 'Effect panel'
        if stack is not None:
            block.fontformat.text_effects = stack
        return TextBlkItem(block, 1)

    def test_item_typed_preview_commit_escape_and_one_undo(self):
        before = self._stack(
            StrokeEffect(width=0.12, paint=SolidPaint((10, 20, 30)))
        )
        item = self._item(before)
        self.panel.set_textblk_item(item)
        editor = self.panel.texteffect_panel.overall_opacity_control.editor

        editor.setText('40.0%')
        editor.textEdited.emit('40.0%')
        self.assertEqual(item.blk.fontformat.text_effects, before)
        self.assertAlmostEqual(
            item.effective_text_effects().overall_opacity, 0.4
        )
        self.assertEqual(self.canvas.stack.count(), 0)

        editor.returnPressed.emit()
        self.assertAlmostEqual(
            item.blk.fontformat.text_effects.overall_opacity, 0.4
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        self.canvas.stack.undo()
        self.assertEqual(item.blk.fontformat.text_effects, before)

        editor.setText('25.0%')
        editor.textEdited.emit('25.0%')
        self.assertAlmostEqual(
            item.effective_text_effects().overall_opacity, 0.25
        )
        QApplication.sendEvent(
            editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects(), before)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_stroke_numeric_preview_commit_escape_and_incremental_card(self):
        before = self._stack(StrokeEffect(width=0.12))
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.stroke_cards[0]
        editor = card.width_control.editor

        editor.setText('0.45')
        editor.textEdited.emit('0.45')
        self.assertEqual(item.blk.fontformat.text_effects, before)
        self.assertEqual(item.effective_text_effects()[0].width, 0.45)
        self.assertEqual(self.canvas.stack.count(), 0)

        editor.returnPressed.emit()
        committed = item.blk.fontformat.text_effects
        self.assertEqual(committed[0].width, 0.45)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)

        editor.setText('0.70')
        editor.textEdited.emit('0.70')
        self.assertEqual(item.effective_text_effects()[0].width, 0.7)
        QApplication.sendEvent(
            editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects(), committed)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_global_preview_cancel_and_commit_update_style_only_on_commit(self):
        before = self.panel.global_format.text_effects
        C.active_format = self.panel.global_format
        with patch.object(
            self.panel, 'update_text_style_label'
        ) as update_style:
            self.panel.text_effect_session.preview_value(
                -1, 'overall_opacity', 0.35
            )
            self.assertAlmostEqual(
                self.panel.global_format.text_effects.overall_opacity,
                0.35,
            )
            self.assertEqual(
                C.active_format.text_effects,
                self.panel.global_format.text_effects,
            )
            update_style.assert_not_called()

            self.panel.text_effect_session.cancel_preview()
            self.assertEqual(self.panel.global_format.text_effects, before)
            self.assertEqual(C.active_format.text_effects, before)
            update_style.assert_not_called()

            self.panel.text_effect_session.preview_value(
                -1, 'overall_opacity', 0.6
            )
            self.assertTrue(self.panel.text_effect_session.commit_value(
                -1, 'overall_opacity', 0.6
            ))
            self.assertAlmostEqual(
                self.panel.global_format.text_effects.overall_opacity, 0.6
            )
            self.assertEqual(self.canvas.stack.count(), 0)
            update_style.assert_called_once_with()

        preset = Mock()
        self.panel.textstyle_panel.active_text_style_label = preset
        C.active_format = self.panel.global_format
        self.assertTrue(self.panel.text_effect_session.commit_value(
            -1, 'overall_opacity', 0.7
        ))
        preset.update_style.assert_called_once_with(self.panel.global_format)

    def test_equal_unowned_active_format_is_not_a_global_effect_owner(self):
        before = self.panel.global_format.text_effects
        unowned = self.panel.global_format.deepcopy()
        C.active_format = unowned

        self.panel.text_effect_session.preview_value(
            -1, 'overall_opacity', 0.35
        )
        self.assertEqual(unowned.text_effects, before)
        self.panel.text_effect_session.cancel_preview()
        self.assertEqual(unowned.text_effects, before)

        self.assertTrue(self.panel.text_effect_session.commit_value(
            -1, 'overall_opacity', 0.6
        ))
        self.assertEqual(unowned.text_effects, before)

    def test_structure_operations_each_create_one_canvas_command(self):
        first = StrokeEffect(
            width=0.1, paint=SolidPaint((255, 0, 0))
        )
        second = StrokeEffect(
            width=0.2, paint=SolidPaint((0, 0, 255))
        )
        item = self._item(self._stack(first, second))
        self.panel.set_textblk_item(item)
        effect_panel = self.panel.texteffect_panel

        stroke_action = next(
            action
            for action in effect_panel.add_effect_button.menu().actions()
            if action.data() == 'stroke'
        )
        stroke_action.trigger()
        self.assertEqual(len(item.blk.fontformat.text_effects), 3)
        self.assertEqual(self.canvas.stack.count(), 1)

        effect_panel.stroke_cards[2].delete_button.click()
        self.assertEqual(len(item.blk.fontformat.text_effects), 2)
        self.assertEqual(self.canvas.stack.count(), 2)

        effect_panel.stroke_cards[0].enabled_checkbox.click()
        self.assertFalse(item.blk.fontformat.text_effects[0].enabled)
        self.assertEqual(self.canvas.stack.count(), 3)

        effect_panel.stroke_cards[1].move_up_button.click()
        stack = item.blk.fontformat.text_effects
        self.assertEqual(stack[0].paint.color, (0, 0, 255))
        self.assertEqual(stack[1].paint.color, (255, 0, 0))
        self.assertEqual(self.canvas.stack.count(), 4)
        self.assertEqual(
            [card.index for card in self.panel.texteffect_panel.stroke_cards],
            [0, 1],
        )

    def test_multi_selection_maps_common_structure_and_blocks_mixed_indices(self):
        first = self._item(self._stack(StrokeEffect(width=0.1)))
        second = self._item(self._stack(StrokeEffect(width=0.3)))
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        effect_panel = self.panel.texteffect_panel
        self.assertTrue(effect_panel.mixed_label.isHidden())
        card = effect_panel.stroke_cards[0]
        self.assertEqual(card.width_control.editor.text(), '\N{EM DASH}')
        self.assertTrue(
            self.panel.text_effect_session.commit_value(0, 'width', 0.25)
        )
        self.assertTrue(all(
            item.blk.fontformat.text_effects[0].width == 0.25
            for item in (first, second)
        ))
        self.assertEqual(self.canvas.stack.count(), 1)

        heterogeneous = self._item(self._stack(
            StrokeEffect(width=0.4), StrokeEffect(width=0.5)
        ))
        self.canvas.selected = [first, heterogeneous]
        self.panel.set_textblk_item(None, multi_select=True)
        self.assertFalse(effect_panel.mixed_label.isHidden())
        self.assertFalse(effect_panel.add_effect_button.isEnabled())
        self.assertEqual(effect_panel.stroke_cards, [])
        self.assertFalse(
            self.panel.text_effect_session.commit_value(0, 'width', 0.7)
        )
        self.assertEqual(self.canvas.stack.count(), 1)

        self.assertTrue(self.panel.text_effect_session.commit_value(
            -1, 'overall_opacity', 0.8
        ))
        self.assertTrue(all(
            item.blk.fontformat.text_effects.overall_opacity == 0.8
            for item in (first, heterogeneous)
        ))
        self.assertEqual(self.canvas.stack.count(), 2)

    def test_run_created_stroke_appears_on_selection_refresh(self):
        item = self._item(TextEffectStack())
        self.panel.set_textblk_item(item)
        self.assertEqual(self.panel.texteffect_panel.stroke_cards, [])

        item.blk.fontformat.text_effects = with_primary_stroke(
            item.blk.fontformat.text_effects,
            width=0.32,
            paint=SolidPaint((40, 50, 60)),
        )
        item.fontformat.text_effects = item.blk.fontformat.text_effects
        self.panel.set_textblk_item(item)
        cards = self.panel.texteffect_panel.stroke_cards
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].width_control.editor.text(), '0.32')
        self.assertEqual(cards[0].color_picker.rgb(), (40, 50, 60))

    def test_color_dialog_signal_retains_selected_owner(self):
        item = self._item(self._stack(StrokeEffect(width=0.1)))
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.stroke_cards[0]

        card.color_picker.changingColor.emit()
        self.assertTrue(self.panel.focusOnColorDialog)
        self.panel.set_textblk_item(None)
        self.assertIs(self.panel.textblk_item, item)
        self.assertEqual(self.panel.text_effect_session.items, [item])

        card.color_picker.colorChanged.emit(False)
        self.assertFalse(self.panel.focusOnColorDialog)
        self.assertIs(self.panel.textblk_item, item)
        self.assertEqual(self.panel.text_effect_session.items, [item])

        card.color_picker.changingColor.emit()
        card.color_picker.setPickerColor((70, 80, 90))
        card.color_picker.colorChanged.emit(True)
        self.assertEqual(
            item.blk.fontformat.text_effects[0].paint.color,
            (70, 80, 90),
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)

    def test_page_change_commits_pending_effect_before_owner_merge(self):
        item = self._item(self._stack(StrokeEffect(width=0.1)))
        self.panel.set_textblk_item(item)
        editor = self.panel.texteffect_panel.stroke_cards[0].width_control.editor
        editor.setText('0.55')
        editor.textEdited.emit('0.55')
        self.assertEqual(item.blk.fontformat.text_effects[0].width, 0.1)

        self.panel.resolve_text_transform_edits_for_page_change()

        self.assertEqual(item.blk.fontformat.text_effects[0].width, 0.55)
        self.assertEqual(item.fontformat.text_effects[0].width, 0.55)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIsNone(self.panel.textblk_item)
        self.assertEqual(self.panel.text_effect_session.items, [])

    def test_scene_change_cancels_preview_and_finishes_on_global_effects(self):
        global_effects = self._stack(StrokeEffect(width=0.88))
        self.panel.global_format.text_effects = global_effects
        self.panel.set_active_format(self.panel.global_format)
        item_effects = self._stack(StrokeEffect(width=0.1))
        item = self._item(item_effects)
        self.panel.set_textblk_item(item)
        editor = self.panel.texteffect_panel.stroke_cards[0].width_control.editor
        editor.setText('0.65')
        editor.textEdited.emit('0.65')
        self.assertEqual(item.effective_text_effects()[0].width, 0.65)

        self.panel.cancel_text_transform_edits_for_scene_change()

        self.assertEqual(item.effective_text_effects(), item_effects)
        self.assertIsNone(self.panel.textblk_item)
        self.assertEqual(self.panel.text_effect_session.items, [])
        self.assertEqual(self.panel.text_transform_session.items, [])
        self.assertIsNone(self.panel.text_effect_session.preview_before)
        cards = self.panel.texteffect_panel.stroke_cards
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].width_control.editor.text(), '0.88')
        self.assertEqual(self.canvas.stack.count(), 0)

    def test_panel_config_and_old_controls_are_removed(self):
        effect_view = self.panel.texteffect_panel.view_widget
        self.assertEqual(effect_view.config_name, 'show_text_effect_panel')
        self.assertEqual(effect_view.config_expand_name, 'expand_teffect_panel')
        self.assertIn(effect_view, self.registered)

        old_expand = pcfg.expand_teffect_panel
        try:
            effect_view.set_expend_area(False)
            self.assertFalse(pcfg.expand_teffect_panel)
            effect_view.set_expend_area(True)
            self.assertTrue(pcfg.expand_teffect_panel)
        finally:
            pcfg.expand_teffect_panel = old_expand

        self.assertFalse(hasattr(self.panel, 'strokeWidthBox'))
        self.assertFalse(hasattr(self.panel, 'strokeColorPicker'))
        self.assertFalse(
            hasattr(self.panel.textadvancedfmt_panel, 'opacity_box')
        )
        self.assertIsInstance(self.panel.colorPicker, ColorPickerLabel)
        self.assertNotIsInstance(self.panel.colorPicker, NestedColorPickerLabel)
        self.assertNotIn('opacity', handle_ffmt_change)
        self.assertNotIn('srgb', handle_ffmt_change)
        self.assertNotIn('stroke_width', handle_ffmt_change)
        self.assertFalse(
            self.panel.texteffect_panel.mask_brush_button.isEnabled()
        )
        self.assertFalse(
            self.panel.texteffect_panel.mask_brush_button.isHidden()
        )


if __name__ == '__main__':
    unittest.main()
