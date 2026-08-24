import gc
import os
import unittest
import weakref
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QColor, QKeyEvent
from qtpy.QtWidgets import QApplication, QColorDialog, QDialog, QWidget

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
from ballontranslator.ui.text_engine.formatting.gradient_editor import (
    GradientStopBar,
    LinearGradientEditorDialog,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.text_effects import (
    GradientOverlayEffect,
    GradientStop,
    HollowEffect,
    LinearGradientPaint,
    ShadowEffect,
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

        preset.reset_mock()
        self.assertTrue(self.panel.text_effect_session.add_effect('shadow'))
        self.assertIsInstance(
            self.panel.global_format.text_effects.effects[-1], ShadowEffect
        )
        self.assertEqual(self.canvas.stack.count(), 0)
        preset.update_style.assert_called_once_with(self.panel.global_format)

        preset.reset_mock()
        self.assertTrue(
            self.panel.text_effect_session.add_effect('gradient_overlay')
        )
        self.assertTrue(any(
            isinstance(effect, GradientOverlayEffect)
            for effect in self.panel.global_format.text_effects
        ))
        self.assertEqual(self.canvas.stack.count(), 0)
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

        effect_panel.stroke_cards[0].visibility_button.click()
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

    def test_card_and_menu_icons_expose_visibility_actions(self):
        item = self._item(self._stack(
            StrokeEffect(),
            ShadowEffect(enabled=False),
            HollowEffect(),
            GradientOverlayEffect(),
        ))
        self.panel.set_textblk_item(item)
        effect_panel = self.panel.texteffect_panel
        cards = (
            effect_panel.stroke_cards[0],
            effect_panel.shadow_cards[0],
            effect_panel.hollow_card,
            effect_panel.gradient_overlay_card,
        )
        for card in cards:
            self.assertFalse(card.title_icon_label.pixmap().isNull())
            self.assertFalse(card.visibility_button.icon().isNull())
        self.assertEqual(
            effect_panel.stroke_cards[0].visibility_button.toolTip(),
            'Hide Stroke',
        )
        self.assertEqual(
            effect_panel.shadow_cards[0].visibility_button.toolTip(),
            'Show Shadow',
        )
        self.assertEqual(
            effect_panel.hollow_card.visibility_button.accessibleName(),
            'Hide Hollow',
        )
        self.assertEqual(
            effect_panel.gradient_overlay_card.visibility_button.toolTip(),
            'Hide Gradient Overlay',
        )
        self.assertTrue(all(
            not action.icon().isNull()
            for action in effect_panel.add_effect_actions.values()
        ))

    def test_mixed_eye_click_enables_all_with_one_command(self):
        enabled = self._item(self._stack(StrokeEffect(enabled=True)))
        disabled = self._item(self._stack(StrokeEffect(enabled=False)))
        self.canvas.selected = [enabled, disabled]
        self.panel.set_textblk_item(None, multi_select=True)
        card = self.panel.texteffect_panel.stroke_cards[0]

        mixed_icon_key = card.visibility_button.icon().cacheKey()
        card.visibility_button.set_visibility(False)
        self.assertNotEqual(
            mixed_icon_key, card.visibility_button.icon().cacheKey()
        )
        card.visibility_button.set_visibility(None)
        self.assertEqual(card.visibility_button.toolTip(), 'Show Stroke')
        self.assertEqual(
            card.visibility_button.accessibleName(), 'Show Stroke'
        )
        card.visibility_button.click()

        self.assertTrue(all(
            item.blk.fontformat.text_effects[0].enabled
            for item in (enabled, disabled)
        ))
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_stroke_position_common_mixed_and_one_command_undo(self):
        item = self._item(self._stack(StrokeEffect(position='center')))
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.stroke_cards[0]

        self.assertEqual(card.position_selector.currentData(), 'center')
        card.position_selector.setCurrentIndex(
            card.position_selector.findData('outside')
        )
        self.assertEqual(
            item.blk.fontformat.text_effects[0].position, 'outside'
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        self.canvas.stack.undo()
        self.assertEqual(
            item.blk.fontformat.text_effects[0].position, 'center'
        )
        self.canvas.stack.redo()
        self.assertEqual(
            item.blk.fontformat.text_effects[0].position, 'outside'
        )
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)

        second = self._item(self._stack(StrokeEffect(position='inside')))
        self.canvas.selected = [item, second]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.stroke_cards[0]
        self.assertEqual(mixed_card.position_selector.currentIndex(), -1)
        mixed_card.position_selector.setCurrentIndex(
            mixed_card.position_selector.findData('center')
        )
        self.assertTrue(all(
            target.blk.fontformat.text_effects[0].position == 'center'
            for target in (item, second)
        ))
        self.assertEqual(self.canvas.stack.count(), 2)

    def test_stroke_fill_conversion_and_mixed_selection(self):
        solid = self._item(self._stack(StrokeEffect(
            paint=SolidPaint((12, 34, 56))
        )))
        self.panel.set_textblk_item(solid)
        card = self.panel.texteffect_panel.stroke_cards[0]
        card.fill_type_selector.setCurrentIndex(
            card.fill_type_selector.findData('linear_gradient')
        )
        converted = solid.blk.fontformat.text_effects[0].paint
        self.assertIsInstance(converted, LinearGradientPaint)
        self.assertEqual(converted.stops[0].color, (12, 34, 56))
        self.assertEqual(converted.stops[0].opacity, 1.0)
        self.assertEqual(converted.stops[1].opacity, 0.0)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.canvas.stack.undo()
        self.assertEqual(
            solid.blk.fontformat.text_effects[0].paint,
            SolidPaint((12, 34, 56)),
        )
        self.canvas.stack.redo()
        self.assertEqual(
            solid.blk.fontformat.text_effects[0].paint, converted
        )

        gradient = self._item(self._stack(StrokeEffect(
            paint=LinearGradientPaint()
        )))
        self.canvas.selected = [solid, gradient]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.stroke_cards[0]
        self.assertEqual(
            mixed_card.fill_type_selector.currentData(), 'linear_gradient'
        )
        self.assertFalse(mixed_card.paint_button.isEnabled())
        self.assertEqual(mixed_card.paint_button.text(), 'Mixed')
        self.assertEqual(
            mixed_card.paint_button.toolTip(), 'Mixed Gradient Paint'
        )

        first_solid = self._item(self._stack(StrokeEffect(
            paint=SolidPaint((1, 2, 3))
        )))
        second_solid = self._item(self._stack(StrokeEffect(
            paint=SolidPaint((4, 5, 6))
        )))
        self.canvas.selected = [first_solid, second_solid]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.stroke_cards[0]
        self.assertEqual(
            mixed_card.fill_type_selector.currentData(), 'solid'
        )
        self.assertTrue(mixed_card.paint_button.isEnabled())
        self.assertEqual(mixed_card.paint_button.text(), 'Mixed')
        self.assertEqual(
            mixed_card.paint_button.accessibleName(),
            'Choose Shared Stroke Color',
        )
        with patch.object(
            QColorDialog,
            'getColor',
            return_value=QColor(20, 30, 40),
        ):
            mixed_card.paint_button.click()
        self.assertTrue(all(
            target.blk.fontformat.text_effects[0].paint
            == SolidPaint((20, 30, 40))
            for target in (first_solid, second_solid)
        ))
        self.assertEqual(self.canvas.stack.count(), 2)

        # Heterogeneous paint types choose deterministic defaults for all.
        hetero_solid = self._item(self._stack(StrokeEffect(
            paint=SolidPaint((11, 12, 13))
        )))
        hetero_gradient = self._item(self._stack(StrokeEffect(
            paint=LinearGradientPaint(angle=90.0)
        )))
        self.canvas.selected = [hetero_solid, hetero_gradient]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.stroke_cards[0]
        self.assertEqual(mixed_card.fill_type_selector.currentIndex(), -1)
        self.assertFalse(mixed_card.paint_button.isEnabled())
        self.assertEqual(
            mixed_card.paint_button.accessibleName(), 'Mixed Stroke Paint'
        )
        mixed_card.fill_type_selector.setCurrentIndex(
            mixed_card.fill_type_selector.findData('linear_gradient')
        )
        self.assertTrue(all(
            target.blk.fontformat.text_effects[0].paint
            == LinearGradientPaint()
            for target in (hetero_solid, hetero_gradient)
        ))
        self.assertEqual(self.canvas.stack.count(), 3)

    def test_gradient_editor_stop_operations_and_geometry_preview(self):
        paint = LinearGradientPaint(stops=(
            GradientStop(0.0, (0, 0, 0), 1.0),
            GradientStop(1.0, (200, 100, 0), 0.0),
        ))
        bar = GradientStopBar(paint)
        self.assertEqual(
            bar.toolTip(),
            'Click the strip to add a stop; drag a stop to move it',
        )
        previews = []
        bar.paint_changed.connect(previews.append)
        self.assertTrue(bar.add_stop(0.25))
        self.assertEqual(len(bar.paint.stops), 3)
        self.assertEqual(bar.paint.stops[1].color, (50, 25, 0))
        bar.move_selected(0.4)
        self.assertEqual(bar.paint.stops[1].position, 0.4)
        self.assertTrue(bar.remove_selected())
        self.assertFalse(bar.remove_selected())
        self.assertGreaterEqual(len(previews), 3)

        dialog = LinearGradientEditorDialog(paint)
        dialog_previews = []
        dialog.paint_previewed.connect(dialog_previews.append)
        dialog.stop_color_picker.setPickerColor((9, 8, 7))
        dialog.stop_color_picker.colorChanged.emit(True)
        self.assertEqual(dialog.paint.stops[0].color, (9, 8, 7))
        dialog.angle_editor.setValue(45.0)
        dialog.scale_editor.setValue(150.0)
        dialog.flip_button.click()
        self.assertEqual(dialog.paint.angle, 225.0)
        self.assertEqual(dialog.paint.scale, 1.5)
        self.assertGreaterEqual(len(dialog_previews), 3)

        owned_modal = QWidget(dialog)
        with patch.object(
            QApplication, 'activeModalWidget', return_value=owned_modal
        ):
            self.assertTrue(dialog._preserve_on_outside_click())
        dialog.show()
        QApplication.sendEvent(
            dialog,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        rejected = getattr(getattr(QDialog, 'DialogCode', QDialog), 'Rejected')
        self.assertEqual(dialog.result(), rejected)
        self.assertFalse(dialog.isVisible())
        dialog.deleteLater()
        bar.deleteLater()

    def test_gradient_dialog_preview_cancel_accept_one_undo(self):
        before = self._stack(StrokeEffect(
            paint=LinearGradientPaint()
        ))
        preview = LinearGradientPaint(
            stops=(
                GradientStop(0.0, (255, 0, 0), 1.0),
                GradientStop(1.0, (0, 0, 255), 0.5),
            ),
            angle=30.0,
        )
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.stroke_cards[0]
        rejected = getattr(getattr(QDialog, 'DialogCode', QDialog), 'Rejected')
        accepted = getattr(getattr(QDialog, 'DialogCode', QDialog), 'Accepted')

        def stage_preview(dialog):
            dialog.stop_bar.replace_selected(color=(255, 0, 0))
            dialog.stop_bar.select_stop(1)
            dialog.stop_bar.replace_selected(
                color=(0, 0, 255), opacity=0.5
            )
            dialog.angle_editor.setValue(30.0)
            self.assertEqual(dialog.paint, preview)

        dialog = LinearGradientEditorDialog(before[0].paint)

        def preview_then_reject():
            stage_preview(dialog)
            self.assertEqual(item.blk.fontformat.text_effects, before)
            self.assertEqual(item.effective_text_effects()[0].paint, preview)
            return rejected

        with patch(
            'ballontranslator.ui.text_engine.formatting.effects.'
            'LinearGradientEditorDialog',
            return_value=dialog,
        ), patch.object(dialog, 'exec_', side_effect=preview_then_reject):
            card.paint_button.click()
        self.assertEqual(item.effective_text_effects(), before)
        self.assertEqual(self.canvas.stack.count(), 0)

        dialog = LinearGradientEditorDialog(before[0].paint)

        def preview_then_accept():
            stage_preview(dialog)
            return accepted

        with patch(
            'ballontranslator.ui.text_engine.formatting.effects.'
            'LinearGradientEditorDialog',
            return_value=dialog,
        ), patch.object(dialog, 'exec_', side_effect=preview_then_accept):
            card.paint_button.click()
        self.assertEqual(item.blk.fontformat.text_effects[0].paint, preview)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)
        self.canvas.stack.undo()
        self.assertEqual(item.blk.fontformat.text_effects, before)

    def test_gradient_dialog_releases_filter_and_wrapper_after_card_flow(self):
        item = self._item(self._stack(GradientOverlayEffect()))
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.gradient_overlay_card
        dialog = LinearGradientEditorDialog(
            item.fontformat.text_effects[0].paint
        )
        dialog_ref = weakref.ref(dialog)
        rejected = getattr(getattr(QDialog, 'DialogCode', QDialog), 'Rejected')

        def reject_after_show():
            dialog.show()
            self.app.processEvents()
            self.assertTrue(dialog._outside_click_filter_installed)
            dialog.reject()
            self.assertFalse(dialog._outside_click_filter_installed)
            return rejected

        with patch(
            'ballontranslator.ui.text_engine.formatting.effects.'
            'LinearGradientEditorDialog',
            return_value=dialog,
        ), patch.object(dialog, 'exec_', side_effect=reject_after_show):
            card.paint_button.click()

        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.assertFalse(dialog._outside_click_filter_installed)
        with self.assertRaises(RuntimeError):
            dialog.objectName()
        self.assertIs(dialog_ref(), dialog)
        del dialog
        gc.collect()
        self.assertIsNone(dialog_ref())

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
        self.assertEqual(cards[0].fill_type_selector.currentData(), 'solid')
        self.assertEqual(
            cards[0].paint_button.accessibleName(), 'Choose Stroke Color'
        )
        self.assertEqual(cards[0].paint_button.text(), '')

    def test_color_dialog_signal_retains_selected_owner(self):
        item = self._item(self._stack(StrokeEffect(width=0.1)))
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.stroke_cards[0]

        def choose_color(*_args):
            self.assertTrue(self.panel.focusOnColorDialog)
            self.panel.set_textblk_item(None)
            self.assertIs(self.panel.textblk_item, item)
            self.assertEqual(self.panel.text_effect_session.items, [item])
            return QColor(70, 80, 90)

        with patch.object(QColorDialog, 'getColor', side_effect=choose_color):
            card.paint_button.click()
        self.assertFalse(self.panel.focusOnColorDialog)
        self.assertEqual(
            item.blk.fontformat.text_effects[0].paint.color,
            (70, 80, 90),
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)

    def test_shadow_numeric_preview_commit_escape_keeps_card(self):
        before = self._stack(ShadowEffect(offset=(0.1, 0.2), blur=0.05))
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.shadow_cards[0]
        editor = card.offset_x_control.editor

        editor.setText('0.45')
        editor.textEdited.emit('0.45')
        self.assertEqual(item.blk.fontformat.text_effects, before)
        self.assertEqual(item.effective_text_effects()[0].offset[0], 0.45)
        self.assertEqual(self.canvas.stack.count(), 0)

        editor.returnPressed.emit()
        committed = item.blk.fontformat.text_effects
        self.assertEqual(committed[0].offset, (0.45, 0.2))
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.shadow_cards[0], card)

        blur_editor = card.blur_control.editor
        blur_editor.setText('0.30')
        blur_editor.textEdited.emit('0.30')
        self.assertEqual(item.effective_text_effects()[0].blur, 0.3)
        QApplication.sendEvent(
            blur_editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects(), committed)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_add_shadow_hollow_type_controls_and_uniqueness(self):
        item = self._item(TextEffectStack())
        self.panel.set_textblk_item(item)
        effect_panel = self.panel.texteffect_panel

        effect_panel.add_effect_actions['shadow'].trigger()
        self.assertIsInstance(
            item.blk.fontformat.text_effects[0], ShadowEffect
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        shadow_card = effect_panel.shadow_cards[0]
        shadow_card.type_selector.setCurrentIndex(
            shadow_card.type_selector.findData('long')
        )
        self.assertEqual(
            item.blk.fontformat.text_effects[0].shadow_type, 'long'
        )
        self.assertTrue(shadow_card.blur_control.isHidden())
        self.assertTrue(shadow_card.spread_control.isHidden())
        self.assertEqual(self.canvas.stack.count(), 2)

        effect_panel.add_effect_actions['hollow'].trigger()
        self.assertEqual(self.canvas.stack.count(), 3)
        self.assertIsInstance(
            item.blk.fontformat.text_effects[1], HollowEffect
        )
        self.assertFalse(effect_panel.add_effect_actions['hollow'].isEnabled())
        self.assertFalse(
            self.panel.text_effect_session.add_effect('hollow')
        )
        self.assertEqual(self.canvas.stack.count(), 3)

        effect_panel.hollow_card.visibility_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[1].enabled)
        self.assertEqual(self.canvas.stack.count(), 4)
        effect_panel.hollow_card.delete_button.click()
        self.assertFalse(any(
            isinstance(effect, HollowEffect)
            for effect in item.blk.fontformat.text_effects
        ))
        self.assertEqual(self.canvas.stack.count(), 5)

    def test_gradient_overlay_add_edit_mixed_and_uniqueness(self):
        item = self._item(TextEffectStack())
        self.panel.set_textblk_item(item)
        effect_panel = self.panel.texteffect_panel
        effect_panel.add_effect_actions['gradient_overlay'].trigger()
        overlay = item.blk.fontformat.text_effects[0]
        self.assertIsInstance(overlay, GradientOverlayEffect)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertFalse(
            effect_panel.add_effect_actions['gradient_overlay'].isEnabled()
        )
        self.assertFalse(
            self.panel.text_effect_session.add_effect('gradient_overlay')
        )

        card = effect_panel.gradient_overlay_card
        editor = card.opacity_control.editor
        editor.setText('55.0%')
        editor.textEdited.emit('55.0%')
        self.assertEqual(item.blk.fontformat.text_effects[0], overlay)
        self.assertAlmostEqual(item.effective_text_effects()[0].opacity, 0.55)
        editor.returnPressed.emit()
        self.assertAlmostEqual(
            item.blk.fontformat.text_effects[0].opacity, 0.55
        )
        self.assertEqual(self.canvas.stack.count(), 2)
        self.assertIs(effect_panel.gradient_overlay_card, card)

        card.visibility_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[0].enabled)
        self.assertEqual(self.canvas.stack.count(), 3)
        card.delete_button.click()
        self.assertFalse(any(
            isinstance(effect, GradientOverlayEffect)
            for effect in item.blk.fontformat.text_effects
        ))
        self.assertEqual(self.canvas.stack.count(), 4)
        self.assertTrue(
            effect_panel.add_effect_actions['gradient_overlay'].isEnabled()
        )

        common = self._constant_overlay(angle=0.0)
        different = self._constant_overlay(angle=90.0)
        first = self._item(self._stack(common))
        second = self._item(self._stack(different))
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = effect_panel.gradient_overlay_card
        self.assertEqual(mixed_card.opacity_control.editor.text(), '100.0%')
        self.assertEqual(mixed_card.paint_button.text(), 'Mixed')
        self.assertFalse(mixed_card.paint_button.isEnabled())
        self.assertEqual(
            mixed_card.paint_button.toolTip(), 'Mixed Gradient Paint'
        )

    @staticmethod
    def _constant_overlay(angle: float = 0.0) -> GradientOverlayEffect:
        return GradientOverlayEffect(paint=LinearGradientPaint(
            stops=(
                GradientStop(0.0, (255, 0, 0), 1.0),
                GradientStop(1.0, (0, 0, 255), 1.0),
            ),
            angle=angle,
        ))

    def test_gradient_overlay_dialog_preview_cancel_accept_one_undo(self):
        before = self._stack(self._constant_overlay())
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.gradient_overlay_card
        preview = LinearGradientPaint(
            stops=before[0].paint.stops, angle=60.0
        )
        rejected = getattr(getattr(QDialog, 'DialogCode', QDialog), 'Rejected')
        accepted = getattr(getattr(QDialog, 'DialogCode', QDialog), 'Accepted')

        dialog = LinearGradientEditorDialog(before[0].paint)
        observed = []

        def preview_then_reject():
            dialog.angle_editor.setValue(60.0)
            observed.append((
                item.effective_text_effects()[0].paint,
                item.blk.fontformat.text_effects,
            ))
            return rejected

        with patch(
            'ballontranslator.ui.text_engine.formatting.effects.'
            'LinearGradientEditorDialog',
            return_value=dialog,
        ), patch.object(dialog, 'exec_', side_effect=preview_then_reject):
            card.paint_button.click()
        self.assertEqual(observed, [(preview, before)])
        self.assertEqual(item.effective_text_effects(), before)
        self.assertEqual(self.canvas.stack.count(), 0)

        dialog = LinearGradientEditorDialog(before[0].paint)

        def preview_then_accept():
            dialog.angle_editor.setValue(60.0)
            return accepted

        with patch(
            'ballontranslator.ui.text_engine.formatting.effects.'
            'LinearGradientEditorDialog',
            return_value=dialog,
        ), patch.object(dialog, 'exec_', side_effect=preview_then_accept):
            card.paint_button.click()
        self.assertEqual(item.blk.fontformat.text_effects[0].paint, preview)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.gradient_overlay_card, card)

    def test_shadow_reorder_is_phase_safe_and_mixed_type_does_not_guess(self):
        top = ShadowEffect(color=(255, 0, 0))
        inner = ShadowEffect(shadow_type='inner', color=(0, 255, 0))
        bottom = ShadowEffect(color=(0, 0, 255))
        first = self._item(self._stack(
            top, StrokeEffect(width=0.2), inner, bottom
        ))
        self.panel.set_textblk_item(first)
        cards = self.panel.texteffect_panel.shadow_cards

        self.assertFalse(cards[1].move_up_button.isEnabled())
        cards[2].move_up_button.click()
        effects = first.blk.fontformat.text_effects.effects
        self.assertEqual(effects[0].color, (0, 0, 255))
        self.assertIsInstance(effects[1], StrokeEffect)
        self.assertEqual(effects[2].shadow_type, 'inner')
        self.assertEqual(effects[3].color, (255, 0, 0))
        self.assertEqual(self.canvas.stack.count(), 1)

        second = self._item(self._stack(
            ShadowEffect(shadow_type='inner'),
            StrokeEffect(width=0.3),
            ShadowEffect(shadow_type='inner'),
            ShadowEffect(),
        ))
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.shadow_cards[0]
        self.assertEqual(mixed_card.type_selector.currentIndex(), -1)
        self.assertFalse(mixed_card.move_up_button.isEnabled())
        self.assertFalse(mixed_card.move_down_button.isEnabled())

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
        self.assertFalse(
            hasattr(self.panel.textadvancedfmt_panel, 'shadow_group')
        )
        self.assertIsInstance(self.panel.colorPicker, ColorPickerLabel)
        self.assertNotIsInstance(self.panel.colorPicker, NestedColorPickerLabel)
        self.assertNotIn('opacity', handle_ffmt_change)
        self.assertNotIn('srgb', handle_ffmt_change)
        self.assertNotIn('stroke_width', handle_ffmt_change)
        for name in (
            'shadow_radius', 'shadow_strength',
            'shadow_color', 'shadow_offset',
        ):
            self.assertNotIn(name, handle_ffmt_change)
        self.assertFalse(
            self.panel.texteffect_panel.mask_brush_button.isEnabled()
        )
        self.assertFalse(
            self.panel.texteffect_panel.mask_brush_button.isHidden()
        )


if __name__ == '__main__':
    unittest.main()
