import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFocusEvent, QKeyEvent, QKeySequence, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QColorDialog,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

try:
    from qtpy.QtGui import QUndoStack
except ImportError:
    from qtpy.QtWidgets import QUndoStack

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.custom_widget import (
    ColorPickerLabel,
    NestedColorPickerLabel,
)
from ballontranslator.ui.misc import parse_stylesheet
from ballontranslator.ui.text_engine.formatting.commands import (
    handle_ffmt_change,
)
from ballontranslator.ui.text_engine.formatting.effects import (
    ShadowEffectCard,
    StrokeEffectCard,
    TextEffectPanel,
)
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.formatting.gradient_editor import (
    GradientStopBar,
    InlineLinearGradientEditor,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import (
    FontFormat,
    ProjectiveTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.text_effects import (
    GlowEffect,
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
        self.assertTrue(self.panel.text_effect_session.add_effect('glow'))
        self.assertTrue(any(
            isinstance(effect, GlowEffect)
            for effect in self.panel.global_format.text_effects
        ))
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

    def test_effect_icons_expose_card_visibility_and_hollow_toggle(self):
        item = self._item(self._stack(
            StrokeEffect(),
            ShadowEffect(enabled=False),
            GlowEffect(),
            HollowEffect(),
            GradientOverlayEffect(),
        ))
        self.panel.set_textblk_item(item)
        effect_panel = self.panel.texteffect_panel
        cards = (
            effect_panel.stroke_cards[0],
            effect_panel.shadow_cards[0],
            effect_panel.glow_cards[0],
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
            effect_panel.glow_cards[0].visibility_button.toolTip(),
            'Hide Glow',
        )
        self.assertEqual(
            effect_panel.hollow_toggle_button.accessibleName(),
            'Disable Hollow',
        )
        self.assertTrue(effect_panel.hollow_toggle_button.isChecked())
        self.assertFalse(effect_panel.hollow_toggle_button.icon().isNull())
        self.assertEqual(
            effect_panel.gradient_overlay_card.visibility_button.toolTip(),
            'Hide Gradient',
        )
        self.assertEqual(
            effect_panel.add_effect_actions['gradient_overlay'].text(),
            'Gradient',
        )
        self.assertTrue(all(
            not action.icon().isNull()
            for action in effect_panel.add_effect_actions.values()
        ))
        self.assertNotIn('hollow', effect_panel.add_effect_actions)

    def test_card_action_icons_follow_hover_and_keyboard_focus(self):
        card = StrokeEffectCard(0)
        try:
            card.set_move_enabled(False, True)
            self.assertTrue(card.delete_button.icon().isNull())
            self.assertFalse(card.move_up_button.isEnabled())

            QApplication.sendEvent(card, QEvent(QEvent.Type.Enter))
            self.assertFalse(card.delete_button.icon().isNull())
            self.assertFalse(card.move_up_button.isEnabled())
            QApplication.sendEvent(card, QEvent(QEvent.Type.Leave))
            self.assertTrue(card.delete_button.icon().isNull())

            QApplication.sendEvent(card, QEvent(QEvent.Type.Enter))
            QApplication.sendEvent(
                card.delete_button,
                QFocusEvent(
                    QEvent.Type.FocusIn,
                    Qt.FocusReason.MouseFocusReason,
                ),
            )
            self.assertFalse(card.delete_button.icon().isNull())
            QApplication.sendEvent(card, QEvent(QEvent.Type.Leave))
            self.assertTrue(card.delete_button.icon().isNull())

            QApplication.sendEvent(
                card.delete_button,
                QFocusEvent(
                    QEvent.Type.FocusIn,
                    Qt.FocusReason.TabFocusReason,
                ),
            )
            self.assertFalse(card.delete_button.icon().isNull())
            QApplication.sendEvent(card, QEvent(QEvent.Type.Leave))
            self.assertFalse(card.delete_button.icon().isNull())
            QApplication.sendEvent(
                card.delete_button,
                QFocusEvent(QEvent.Type.FocusOut),
            )
            self.assertTrue(card.delete_button.icon().isNull())
        finally:
            card.deleteLater()

    def test_added_card_scrolls_without_growing_a_constrained_host(self):
        effect_panel = TextEffectPanel(
            'Text Effect', 'test_effect', 'test_effect_expand'
        )
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(effect_panel.view_widget)
        reserved_space = QWidget(host)
        reserved_space.setMinimumHeight(80)
        layout.addWidget(reserved_space, 1)
        host.resize(320, 180)
        host.show()
        self.app.processEvents()
        constrained_height = host.height()

        try:
            effect_panel._set_effect_states([
                self._stack(StrokeEffect(paint=LinearGradientPaint()))
            ])
            self.app.processEvents()

            card = effect_panel.stroke_cards[0]
            self.assertEqual(host.height(), constrained_height)
            self.assertGreater(
                effect_panel.verticalScrollBar().maximum(), 0
            )
            self.assertGreaterEqual(
                card.height(), card.minimumSizeHint().height()
            )
            narrow_card_width = card.width()
            host.resize(480, constrained_height)
            self.app.processEvents()
            self.assertEqual(host.height(), constrained_height)
            self.assertGreater(card.width(), narrow_card_width)
        finally:
            host.deleteLater()
            self.app.processEvents()

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

    def test_hollow_toggle_handles_mixed_presence_with_one_command(self):
        absent = self._item(self._stack(StrokeEffect()))
        present = self._item(self._stack(
            StrokeEffect(), HollowEffect(enabled=True)
        ))
        self.canvas.selected = [absent, present]
        self.panel.set_textblk_item(None, multi_select=True)
        effect_panel = self.panel.texteffect_panel
        toggle = effect_panel.hollow_toggle_button

        self.assertFalse(toggle.isChecked())
        self.assertEqual(
            toggle.accessibleName(),
            'Enable Hollow for All Selected Text',
        )
        self.assertFalse(effect_panel.add_effect_button.isEnabled())
        toggle.click()

        for item in (absent, present):
            hollows = [
                effect
                for effect in item.blk.fontformat.text_effects
                if isinstance(effect, HollowEffect)
            ]
            self.assertEqual(len(hollows), 1)
            self.assertTrue(hollows[0].enabled)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertTrue(toggle.isChecked())
        self.assertEqual(toggle.accessibleName(), 'Disable Hollow')

        self.canvas.stack.undo()
        self.assertFalse(any(
            isinstance(effect, HollowEffect)
            for effect in absent.blk.fontformat.text_effects
        ))
        self.assertEqual(
            toggle.accessibleName(),
            'Enable Hollow for All Selected Text',
        )
        self.canvas.stack.redo()
        self.assertTrue(toggle.isChecked())
        self.assertEqual(toggle.accessibleName(), 'Disable Hollow')

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
        self.assertTrue(card.gradient_editor.isHidden())
        self.assertFalse(card.paint_button.isHidden())
        solid_content_height = (
            self.panel.texteffect_panel.scrollContent.minimumHeight()
        )
        card.fill_type_selector.setCurrentIndex(
            card.fill_type_selector.findData('linear_gradient')
        )
        converted = solid.blk.fontformat.text_effects[0].paint
        self.assertIsInstance(converted, LinearGradientPaint)
        self.assertEqual(converted.stops[0].color, (12, 34, 56))
        self.assertEqual(converted.stops[0].opacity, 1.0)
        self.assertEqual(converted.stops[1].opacity, 0.0)
        self.assertFalse(card.gradient_editor.isHidden())
        self.assertTrue(card.paint_button.isHidden())
        self.assertGreater(
            self.panel.texteffect_panel.scrollContent.minimumHeight(),
            solid_content_height,
        )
        gradient_content_height = (
            self.panel.texteffect_panel.scrollContent.minimumHeight()
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        self.canvas.stack.undo()
        self.assertEqual(
            solid.blk.fontformat.text_effects[0].paint,
            SolidPaint((12, 34, 56)),
        )
        self.assertTrue(card.gradient_editor.isHidden())
        self.assertFalse(card.paint_button.isHidden())
        self.assertLess(
            self.panel.texteffect_panel.scrollContent.minimumHeight(),
            gradient_content_height,
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
        self.assertTrue(mixed_card.paint_button.isHidden())
        self.assertFalse(mixed_card.gradient_editor.isHidden())
        self.assertFalse(mixed_card.gradient_editor.angle_editor.isEnabled())

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

    def test_inline_gradient_stop_actions_numeric_preview_commit_cancel(self):
        paint = LinearGradientPaint(stops=(
            GradientStop(0.0, (0, 0, 0), 1.0),
            GradientStop(1.0, (200, 100, 0), 0.0),
        ))
        editor = InlineLinearGradientEditor(paint)
        bar = editor.stop_bar
        self.assertEqual(
            bar.toolTip(),
            'Click the strip to add a stop; drag a stop to move it',
        )
        previews = []
        commits = []
        cancels = Mock()
        editor.paint_previewed.connect(previews.append)
        editor.paint_commit_requested.connect(commits.append)
        editor.paint_preview_canceled.connect(cancels)

        bar.select_stop(1)
        self.assertEqual(previews, [])
        self.assertEqual(commits, [])
        editor.add_stop_button.click()
        self.assertEqual(len(bar.paint.stops), 3)
        self.assertEqual(len(commits), 1)
        editor.remove_stop_button.click()
        self.assertEqual(len(editor.paint.stops), 2)
        self.assertEqual(len(commits), 2)

        editor.angle_editor.setValue(45.0)
        self.assertEqual(editor.paint.angle, 45.0)
        self.assertEqual(len(commits), 2)
        editor.angle_editor.editingFinished.emit()
        self.assertEqual(len(commits), 3)

        editor.angle_editor.setValue(90.0)
        QApplication.sendEvent(
            editor.angle_editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(editor.paint.angle, 45.0)
        self.assertEqual(cancels.call_count, 1)
        self.assertEqual(len(commits), 3)

        before_color = editor.paint
        editor._on_stop_color_preview(QColor(9, 8, 7))
        self.assertEqual(editor._selected_stop().color, (9, 8, 7))
        editor._on_stop_color_rejected()
        self.assertEqual(editor.paint, before_color)
        self.assertEqual(cancels.call_count, 2)
        editor._on_stop_color_preview(QColor(9, 8, 7))
        editor._on_stop_color_accepted()
        self.assertEqual(editor._selected_stop().color, (9, 8, 7))
        self.assertEqual(len(commits), 4)

        editor.angle_editor.selectAll()
        QTest.keyClicks(editor.angle_editor, '123')
        self.assertEqual(editor.angle_editor.value(), 123.0)
        QTest.keyClick(editor.angle_editor, Qt.Key.Key_Return)
        self.assertEqual(editor.paint.angle, 123.0)
        self.assertEqual(len(commits), 5)
        editor.deleteLater()

    def test_inline_gradient_drag_preview_release_one_undo_and_card_reuse(self):
        before = self._stack(StrokeEffect(
            paint=LinearGradientPaint()
        ))
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.stroke_cards[0]
        editor = card.gradient_editor
        self.assertFalse(editor.isHidden())
        self.assertTrue(card.paint_button.isHidden())
        self.assertEqual(card.findChildren(QColorDialog), [])

        bar = editor.stop_bar
        bar.resize(300, 42)
        bar.show()
        QApplication.sendEvent(bar, QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(7, 31),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ))
        QApplication.sendEvent(bar, QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(90, 31),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ))
        self.assertEqual(item.blk.fontformat.text_effects, before)
        self.assertNotEqual(item.effective_text_effects(), before)
        self.assertEqual(self.canvas.stack.count(), 0)
        QApplication.sendEvent(bar, QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(90, 31),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ))
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)
        committed = item.blk.fontformat.text_effects
        self.assertNotEqual(committed, before)
        self.canvas.stack.undo()
        self.assertEqual(item.blk.fontformat.text_effects, before)

    def test_inline_gradient_escape_owns_active_app_shortcut(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        editor = InlineLinearGradientEditor(LinearGradientPaint())
        layout.addWidget(editor)
        shortcut = QShortcut(QKeySequence('Escape'), host)
        shortcut_hits = Mock()
        shortcut.activated.connect(shortcut_hits)
        host.show()
        self.app.processEvents()

        editor.angle_editor.setFocus()
        editor.angle_editor.setValue(75.0)
        QTest.keyClick(editor.angle_editor, Qt.Key.Key_Escape)
        self.assertEqual(editor.paint.angle, 0.0)
        self.assertEqual(shortcut_hits.call_count, 0)

        bar = editor.stop_bar
        QTest.mousePress(
            bar,
            Qt.MouseButton.LeftButton,
            pos=QPoint(bar.width() // 2, 12),
        )
        self.assertEqual(len(editor.paint.stops), 3)
        QTest.keyClick(bar, Qt.Key.Key_Escape)
        self.assertEqual(len(editor.paint.stops), 2)
        self.assertEqual(shortcut_hits.call_count, 0)
        host.deleteLater()

    def test_inline_gradient_position_bounds_and_mixed_accessibility(self):
        paint = LinearGradientPaint(stops=(
            GradientStop(0.5, (0, 0, 0), 1.0),
            GradientStop(0.5, (255, 255, 255), 1.0),
        ))
        editor = InlineLinearGradientEditor(paint)
        previews = Mock()
        commits = Mock()
        editor.paint_previewed.connect(previews)
        editor.paint_commit_requested.connect(commits)

        editor.stop_position_editor.stepBy(1)
        self.assertEqual(editor.stop_position_editor.value(), 50.0)
        self.assertEqual(editor.paint, paint)
        self.assertEqual(previews.call_count, 0)
        self.assertEqual(commits.call_count, 0)

        editor.set_paint(paint, editable=False)
        self.assertIn('Mixed', editor.stop_bar.accessibleName())
        self.assertEqual(editor.stop_bar.toolTip(), 'Mixed Gradient')
        editor.set_paint(paint, editable=True)
        self.assertEqual(editor.stop_bar.accessibleName(), 'Gradient Stops')
        editor.deleteLater()

    def test_inline_gradient_numeric_suffixes_fit_narrow_card(self):
        old_stylesheet = self.app.styleSheet()
        theme_globals = {
            name: getattr(shared, name)
            for name in (
                'FOREGROUND_FONTCOLOR',
                'SLIDERHANDLE_COLOR',
                'BORDER_COLOR',
                'WIDGET_BACKGROUND_COLOR',
            )
        }
        try:
            for theme in ('eva-light', 'eva-dark'):
                for card_type, effect in (
                    (StrokeEffectCard, StrokeEffect(
                        paint=LinearGradientPaint(angle=35.0, scale=1.25)
                    )),
                    (ShadowEffectCard, ShadowEffect(
                        paint=LinearGradientPaint(angle=35.0, scale=1.25)
                    )),
                ):
                    with self.subTest(theme=theme, card=card_type.__name__):
                        self.app.setStyleSheet(parse_stylesheet(theme))
                        host = QWidget()
                        layout = QVBoxLayout(host)
                        layout.setContentsMargins(11, 11, 11, 11)
                        card = card_type(0)
                        card.set_values([effect])
                        layout.addWidget(card)
                        requested_width = 316
                        host.resize(
                            requested_width, card.sizeHint().height() + 22
                        )
                        host.show()
                        self.app.processEvents()

                        self.assertEqual(host.width(), requested_width)
                        for field in card.gradient_editor._editors():
                            line_edit = field.lineEdit()
                            text_width = (
                                line_edit.fontMetrics().horizontalAdvance(
                                    field.text()
                                )
                            )
                            self.assertGreaterEqual(
                                line_edit.contentsRect().width(),
                                text_width + 4,
                                field.text(),
                            )
                        host.deleteLater()
                        self.app.processEvents()
        finally:
            self.app.setStyleSheet(old_stylesheet)
            for name, value in theme_globals.items():
                setattr(shared, name, value)

    def test_inline_gradient_color_dialog_preview_reject_accept(self):
        before = self._stack(StrokeEffect(paint=LinearGradientPaint()))
        item = self._item(before)
        item.set_text_transform(TextTransformStack((
            ProjectiveTextTransform(),
        )))
        self.canvas.selected = [item]
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.stroke_cards[0]
        editor = card.gradient_editor
        transform_session = self.panel.text_transform_session
        transform_session.preview_parameter_delta(
            0, 'horizontal_scale', 0.25
        )
        self.assertAlmostEqual(
            item._effective_text_transform()[0].horizontal_scale, 1.25
        )
        created_dialogs = []
        real_dialog_class = QColorDialog

        def create_dialog(*args, **kwargs):
            dialog = real_dialog_class(*args, **kwargs)
            created_dialogs.append(dialog)
            return dialog

        def assert_active_preview(color: QColor) -> None:
            dialog = created_dialogs[-1]
            dialog.currentColorChanged.emit(color)
            self.assertTrue(self.panel.focusOnColorDialog)
            self.panel.set_textblk_item(None)
            self.assertEqual(self.canvas.selected, [item])
            self.assertEqual(self.panel.text_effect_session.items, [item])
            self.assertIs(self.panel.textblk_item, item)
            self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)
            self.assertIsNotNone(transform_session.drag_before)
            self.assertAlmostEqual(
                item._effective_text_transform()[0].horizontal_scale, 1.25
            )
            self.assertEqual(item.blk.fontformat.text_effects, before)
            self.assertEqual(
                item.effective_text_effects()[0].paint.stops[0].color,
                (color.red(), color.green(), color.blue()),
            )
            self.assertEqual(self.canvas.stack.count(), 0)

        def preview_then_reject():
            assert_active_preview(QColor(9, 8, 7))
            created_dialogs[-1].reject()
            return 0

        with patch(
            'ballontranslator.ui.text_engine.formatting.gradient_editor.'
            'QColorDialog',
            side_effect=create_dialog,
        ), patch.object(
            real_dialog_class, 'exec_', side_effect=preview_then_reject
        ):
            editor.stop_color_picker.click()

        self.assertFalse(self.panel.focusOnColorDialog)
        self.assertEqual(item.effective_text_effects(), before)
        self.assertEqual(self.canvas.stack.count(), 0)
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)

        def preview_then_accept():
            assert_active_preview(QColor(20, 30, 40))
            created_dialogs[-1].accept()
            return 1

        with patch(
            'ballontranslator.ui.text_engine.formatting.gradient_editor.'
            'QColorDialog',
            side_effect=create_dialog,
        ), patch.object(
            real_dialog_class, 'exec_', side_effect=preview_then_accept
        ):
            editor.stop_color_picker.click()

        self.assertFalse(self.panel.focusOnColorDialog)
        self.assertEqual(
            item.blk.fontformat.text_effects[0].paint.stops[0].color,
            (20, 30, 40),
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)
        self.canvas.stack.undo()
        self.assertEqual(item.blk.fontformat.text_effects, before)
        self.assertIs(self.panel.texteffect_panel.stroke_cards[0], card)
        transform_session.cancel_preview()
        self.assertAlmostEqual(
            item._effective_text_transform()[0].horizontal_scale, 1.0
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

    def test_glow_numeric_preview_commit_escape_keeps_card(self):
        before = self._stack(GlowEffect(size=0.12, spread=0.03))
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.glow_cards[0]
        editor = card.size_control.editor

        editor.setText('0.45')
        editor.textEdited.emit('0.45')
        self.assertEqual(item.blk.fontformat.text_effects, before)
        self.assertEqual(item.effective_text_effects()[0].size, 0.45)
        self.assertEqual(self.canvas.stack.count(), 0)

        editor.returnPressed.emit()
        committed = item.blk.fontformat.text_effects
        self.assertEqual(committed[0].size, 0.45)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.glow_cards[0], card)

        spread_editor = card.spread_control.editor
        spread_editor.setText('0.30')
        spread_editor.textEdited.emit('0.30')
        self.assertEqual(item.effective_text_effects()[0].spread, 0.3)
        QApplication.sendEvent(
            spread_editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects(), committed)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_glow_add_type_reorder_eye_delete_and_mixed_fill(self):
        item = self._item(TextEffectStack())
        self.panel.set_textblk_item(item)
        effect_panel = self.panel.texteffect_panel
        effect_panel.add_effect_actions['glow'].trigger()
        effect_panel.add_effect_actions['glow'].trigger()
        self.assertEqual(len(effect_panel.glow_cards), 2)
        self.assertEqual(self.canvas.stack.count(), 2)
        self.assertTrue(all(
            isinstance(effect, GlowEffect)
            for effect in item.blk.fontformat.text_effects
        ))

        second = effect_panel.glow_cards[1]
        second.type_selector.setCurrentIndex(
            second.type_selector.findData('inner')
        )
        self.assertEqual(
            item.blk.fontformat.text_effects[1].glow_type, 'inner'
        )
        self.assertEqual(second.spread_control.label.text(), 'Choke')
        second.visibility_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[1].enabled)
        second.delete_button.click()
        self.assertEqual(len(item.blk.fontformat.text_effects), 1)
        self.assertEqual(self.canvas.stack.count(), 5)

        interleaved = self._item(self._stack(
            ShadowEffect(paint=SolidPaint((255, 0, 0))),
            GlowEffect(paint=SolidPaint((0, 0, 255))),
            StrokeEffect(),
        ))
        self.panel.set_textblk_item(interleaved)
        self.panel.texteffect_panel.glow_cards[0].move_up_button.click()
        self.assertIsInstance(
            interleaved.blk.fontformat.text_effects[0], GlowEffect
        )
        self.assertIsInstance(
            interleaved.blk.fontformat.text_effects[1], ShadowEffect
        )
        self.assertEqual(self.canvas.stack.count(), 6)

        first_solid = self._item(self._stack(GlowEffect(
            paint=SolidPaint((1, 2, 3))
        )))
        second_solid = self._item(self._stack(GlowEffect(
            paint=SolidPaint((4, 5, 6))
        )))
        self.canvas.selected = [first_solid, second_solid]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.glow_cards[0]
        self.assertEqual(mixed_card.fill_type_selector.currentData(), 'solid')
        self.assertTrue(mixed_card.paint_button.isEnabled())
        self.assertEqual(
            mixed_card.paint_button.accessibleName(),
            'Choose Shared Glow Color',
        )
        with patch.object(
            QColorDialog, 'getColor', return_value=QColor(20, 30, 40)
        ):
            mixed_card.paint_button.click()
        self.assertTrue(all(
            target.blk.fontformat.text_effects[0].paint
            == SolidPaint((20, 30, 40))
            for target in (first_solid, second_solid)
        ))
        self.assertEqual(self.canvas.stack.count(), 7)

        first_gradient = self._item(self._stack(GlowEffect(
            paint=LinearGradientPaint(angle=10.0)
        )))
        second_gradient = self._item(self._stack(GlowEffect(
            paint=LinearGradientPaint(angle=90.0)
        )))
        self.canvas.selected = [first_gradient, second_gradient]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.glow_cards[0]
        self.assertEqual(
            mixed_card.fill_type_selector.currentData(), 'linear_gradient'
        )
        self.assertTrue(mixed_card.paint_button.isHidden())
        self.assertFalse(mixed_card.gradient_editor.isHidden())
        self.assertFalse(mixed_card.gradient_editor.angle_editor.isEnabled())

        outer_type = self._item(self._stack(GlowEffect(glow_type='outer')))
        inner_type = self._item(self._stack(GlowEffect(glow_type='inner')))
        self.canvas.selected = [outer_type, inner_type]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.glow_cards[0]
        self.assertEqual(mixed_card.type_selector.currentIndex(), -1)
        self.assertEqual(
            mixed_card.spread_control.label.text(), 'Spread / Choke'
        )
        mixed_card.type_selector.setCurrentIndex(
            mixed_card.type_selector.findData('outer')
        )
        self.assertTrue(all(
            target.blk.fontformat.text_effects[0].glow_type == 'outer'
            for target in (outer_type, inner_type)
        ))
        self.assertEqual(self.canvas.stack.count(), 8)

        solid_type = self._item(self._stack(GlowEffect(
            paint=SolidPaint((9, 8, 7))
        )))
        gradient_type = self._item(self._stack(GlowEffect(
            paint=LinearGradientPaint(angle=35.0)
        )))
        self.canvas.selected = [solid_type, gradient_type]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.glow_cards[0]
        self.assertEqual(mixed_card.fill_type_selector.currentIndex(), -1)
        self.assertFalse(mixed_card.paint_button.isEnabled())
        mixed_card.fill_type_selector.setCurrentIndex(
            mixed_card.fill_type_selector.findData('linear_gradient')
        )
        self.assertTrue(all(
            target.blk.fontformat.text_effects[0].paint
            == LinearGradientPaint()
            for target in (solid_type, gradient_type)
        ))
        self.assertEqual(self.canvas.stack.count(), 9)

    def test_glow_fill_conversion_and_inline_gradient_one_undo(self):
        before = self._stack(GlowEffect(
            paint=SolidPaint((12, 34, 56))
        ))
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.glow_cards[0]
        self.assertTrue(card.gradient_editor.isHidden())
        self.assertFalse(card.paint_button.isHidden())
        card.fill_type_selector.setCurrentIndex(
            card.fill_type_selector.findData('linear_gradient')
        )
        converted = item.blk.fontformat.text_effects[0].paint
        self.assertIsInstance(converted, LinearGradientPaint)
        self.assertEqual(converted.stops[0].color, (12, 34, 56))
        self.assertEqual(converted.stops[0].opacity, 1.0)
        self.assertEqual(converted.stops[1].opacity, 0.0)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertFalse(card.gradient_editor.isHidden())
        self.assertTrue(card.paint_button.isHidden())

        editor = card.gradient_editor
        self.assertFalse(editor.isHidden())
        self.assertTrue(card.paint_button.isHidden())
        editor.angle_editor.setValue(60.0)
        preview = LinearGradientPaint(stops=converted.stops, angle=60.0)
        self.assertEqual(item.effective_text_effects()[0].paint, preview)
        self.assertEqual(item.blk.fontformat.text_effects[0].paint, converted)
        QApplication.sendEvent(
            editor.angle_editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects()[0].paint, converted)
        self.assertEqual(self.canvas.stack.count(), 1)

        editor.angle_editor.setValue(60.0)
        editor.angle_editor.editingFinished.emit()
        self.assertEqual(item.blk.fontformat.text_effects[0].paint, preview)
        self.assertEqual(self.canvas.stack.count(), 2)
        self.assertIs(self.panel.texteffect_panel.glow_cards[0], card)

    def test_shadow_type_and_hollow_toggle_use_one_command_each(self):
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

        effect_panel.hollow_toggle_button.click()
        self.assertEqual(self.canvas.stack.count(), 3)
        self.assertIsInstance(
            item.blk.fontformat.text_effects[1], HollowEffect
        )
        self.assertTrue(item.blk.fontformat.text_effects[1].enabled)
        self.assertTrue(effect_panel.hollow_toggle_button.isChecked())
        self.assertNotIn('hollow', effect_panel.add_effect_actions)

        effect_panel.hollow_toggle_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[1].enabled)
        self.assertEqual(self.canvas.stack.count(), 4)

        effect_panel.hollow_toggle_button.click()
        hollows = [
            effect
            for effect in item.blk.fontformat.text_effects
            if isinstance(effect, HollowEffect)
        ]
        self.assertEqual(len(hollows), 1)
        self.assertTrue(hollows[0].enabled)
        self.assertEqual(self.canvas.stack.count(), 5)

    def test_shadow_fill_conversion_and_inline_gradient_one_undo(self):
        before = self._stack(ShadowEffect(
            paint=SolidPaint((12, 34, 56))
        ))
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.shadow_cards[0]

        self.assertEqual(card.fill_type_selector.currentData(), 'solid')
        self.assertTrue(card.gradient_editor.isHidden())
        self.assertFalse(card.paint_button.isHidden())
        self.assertEqual(
            card.paint_button.accessibleName(), 'Choose Shadow Color'
        )
        solid_content_height = (
            self.panel.texteffect_panel.scrollContent.minimumHeight()
        )
        card.fill_type_selector.setCurrentIndex(
            card.fill_type_selector.findData('linear_gradient')
        )
        converted = item.blk.fontformat.text_effects[0].paint
        self.assertIsInstance(converted, LinearGradientPaint)
        self.assertEqual(converted.stops[0].color, (12, 34, 56))
        self.assertEqual(converted.stops[0].opacity, 1.0)
        self.assertEqual(converted.stops[1].opacity, 0.0)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertFalse(card.gradient_editor.isHidden())
        self.assertTrue(card.paint_button.isHidden())
        self.assertGreater(
            self.panel.texteffect_panel.scrollContent.minimumHeight(),
            solid_content_height,
        )
        gradient_content_height = (
            self.panel.texteffect_panel.scrollContent.minimumHeight()
        )
        self.canvas.stack.undo()
        self.assertEqual(
            item.blk.fontformat.text_effects[0].paint,
            SolidPaint((12, 34, 56)),
        )
        self.assertTrue(card.gradient_editor.isHidden())
        self.assertFalse(card.paint_button.isHidden())
        self.assertLess(
            self.panel.texteffect_panel.scrollContent.minimumHeight(),
            gradient_content_height,
        )
        self.canvas.stack.redo()
        self.assertEqual(item.blk.fontformat.text_effects[0].paint, converted)

        editor = card.gradient_editor
        editor.angle_editor.setValue(60.0)
        preview = LinearGradientPaint(stops=converted.stops, angle=60.0)
        self.assertEqual(item.effective_text_effects()[0].paint, preview)
        self.assertEqual(item.blk.fontformat.text_effects[0].paint, converted)
        QApplication.sendEvent(
            editor.angle_editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects()[0].paint, converted)
        self.assertEqual(self.canvas.stack.count(), 1)

        editor.angle_editor.setValue(60.0)
        editor.angle_editor.editingFinished.emit()
        self.assertEqual(item.blk.fontformat.text_effects[0].paint, preview)
        self.assertEqual(self.canvas.stack.count(), 2)
        self.assertIs(self.panel.texteffect_panel.shadow_cards[0], card)

        different = self._item(self._stack(ShadowEffect(
            paint=LinearGradientPaint(angle=120.0)
        )))
        self.canvas.selected = [item, different]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.shadow_cards[0]
        self.assertEqual(
            mixed_card.fill_type_selector.currentData(), 'linear_gradient'
        )
        self.assertFalse(mixed_card.gradient_editor.isHidden())
        self.assertFalse(mixed_card.gradient_editor.angle_editor.isEnabled())

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
        self.assertEqual(card.title_label.text(), 'Gradient')
        self.assertIs(effect_panel.gradient_overlay_card, card)

        card.visibility_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[0].enabled)
        self.assertEqual(self.canvas.stack.count(), 2)
        card.delete_button.click()
        self.assertFalse(any(
            isinstance(effect, GradientOverlayEffect)
            for effect in item.blk.fontformat.text_effects
        ))
        self.assertEqual(self.canvas.stack.count(), 3)
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
        self.assertFalse(mixed_card.gradient_editor.isHidden())
        self.assertFalse(mixed_card.gradient_editor.angle_editor.isEnabled())

        self.canvas.selected = [first]
        self.panel.set_textblk_item(None, multi_select=True)
        self.assertIs(effect_panel.gradient_overlay_card, mixed_card)
        self.assertTrue(mixed_card.gradient_editor.angle_editor.isEnabled())

    @staticmethod
    def _constant_overlay(angle: float = 0.0) -> GradientOverlayEffect:
        return GradientOverlayEffect(paint=LinearGradientPaint(
            stops=(
                GradientStop(0.0, (255, 0, 0), 1.0),
                GradientStop(1.0, (0, 0, 255), 1.0),
            ),
            angle=angle,
        ))

    def test_gradient_overlay_inline_preview_cancel_commit_one_undo(self):
        before = self._stack(self._constant_overlay())
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.gradient_overlay_card
        editor = card.gradient_editor
        preview = LinearGradientPaint(
            stops=before[0].paint.stops, angle=60.0
        )
        editor.angle_editor.setValue(60.0)
        self.assertEqual(item.effective_text_effects()[0].paint, preview)
        self.assertEqual(item.blk.fontformat.text_effects, before)
        QApplication.sendEvent(
            editor.angle_editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects(), before)
        self.assertEqual(self.canvas.stack.count(), 0)

        editor.angle_editor.setValue(60.0)
        editor.angle_editor.editingFinished.emit()
        self.assertEqual(item.blk.fontformat.text_effects[0].paint, preview)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.gradient_overlay_card, card)

    def test_gradient_angle_dial_previews_then_commits_once(self):
        before = self._stack(self._constant_overlay())
        item = self._item(before)
        self.panel.set_textblk_item(item)
        gradient = (
            self.panel.texteffect_panel.gradient_overlay_card.gradient_editor
        )
        dial = gradient.angle_dial
        center = QRectF(dial.rect()).center()
        down = center + QPointF(0.0, 8.0)
        left = center - QPointF(8.0, 0.0)

        QApplication.sendEvent(dial, QMouseEvent(
            QEvent.Type.MouseButtonPress,
            down,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ))
        self.assertEqual(item.effective_text_effects()[0].paint.angle, 90.0)
        self.assertEqual(gradient.angle_editor.value(), 90.0)
        self.assertEqual(item.blk.fontformat.text_effects, before)
        self.assertEqual(self.canvas.stack.count(), 0)
        QApplication.sendEvent(dial, QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            down,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ))
        self.assertEqual(item.blk.fontformat.text_effects[0].paint.angle, 90.0)
        self.assertEqual(self.canvas.stack.count(), 1)

        QApplication.sendEvent(dial, QMouseEvent(
            QEvent.Type.MouseButtonPress,
            left,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ))
        self.assertEqual(item.effective_text_effects()[0].paint.angle, 180.0)
        QApplication.sendEvent(dial, QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Escape,
            Qt.KeyboardModifier.NoModifier,
        ))
        self.assertEqual(item.effective_text_effects()[0].paint.angle, 90.0)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_shadow_reorder_is_phase_safe_and_mixed_type_does_not_guess(self):
        top = ShadowEffect(paint=SolidPaint((255, 0, 0)))
        inner = ShadowEffect(
            shadow_type='inner', paint=SolidPaint((0, 255, 0))
        )
        bottom = ShadowEffect(paint=SolidPaint((0, 0, 255)))
        first = self._item(self._stack(
            top, StrokeEffect(width=0.2), inner, bottom
        ))
        self.panel.set_textblk_item(first)
        cards = self.panel.texteffect_panel.shadow_cards

        self.assertFalse(cards[1].move_up_button.isEnabled())
        cards[2].move_up_button.click()
        effects = first.blk.fontformat.text_effects.effects
        self.assertEqual(effects[0].paint, SolidPaint((0, 0, 255)))
        self.assertIsInstance(effects[1], StrokeEffect)
        self.assertEqual(effects[2].shadow_type, 'inner')
        self.assertEqual(effects[3].paint, SolidPaint((255, 0, 0)))
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

    def test_page_change_commits_pending_inline_gradient(self):
        item = self._item(self._stack(self._constant_overlay()))
        self.panel.set_textblk_item(item)
        editor = self.panel.texteffect_panel.gradient_overlay_card.gradient_editor
        editor.angle_editor.setValue(75.0)
        self.assertEqual(item.blk.fontformat.text_effects[0].paint.angle, 0.0)

        self.panel.resolve_text_transform_edits_for_page_change()

        self.assertEqual(item.blk.fontformat.text_effects[0].paint.angle, 75.0)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_history_change_cancels_pending_inline_gradient(self):
        before = self._stack(self._constant_overlay())
        item = self._item(before)
        self.panel.set_textblk_item(item)
        editor = self.panel.texteffect_panel.gradient_overlay_card.gradient_editor
        editor.angle_editor.setValue(75.0)
        self.assertEqual(item.effective_text_effects()[0].paint.angle, 75.0)

        self.panel.resolve_text_transform_edits_for_history_change()

        self.assertEqual(item.effective_text_effects(), before)
        self.assertEqual(editor.paint, before[0].paint)
        self.assertEqual(self.canvas.stack.count(), 0)

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
