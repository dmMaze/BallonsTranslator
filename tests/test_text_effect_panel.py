import os
from dataclasses import replace
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

from qtpy.QtCore import (
    QCoreApplication,
    QEvent,
    QPoint,
    QPointF,
    QRectF,
    QTimer,
    QTranslator,
    Qt,
)
from qtpy.QtGui import QColor, QFocusEvent, QKeyEvent, QKeySequence, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QMessageBox,
    QGraphicsScene,
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
from ballontranslator.ui.text_engine.effects.panel import (
    FilterEffectCard,
    ShadowEffectCard,
    StrokeEffectCard,
    TextFillEffectCard,
    TextEffectPanel,
)
from ballontranslator.ui.text_engine.effects.image_generation import (
    ImageGenerationBackend,
    ImageGenerationRequest,
)
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.formatting.presets import TextStyleLabel
from ballontranslator.ui.text_engine.effects.gradient_editor import (
    GradientStopBar,
    InlineLinearGradientEditor,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.effects.filters import (
    FilterUnavailableError,
    get_filter_registry,
)
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import (
    FontFormat,
    ProjectiveTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.raster_assets import RasterAssetRef
from ballontranslator.utils.text_alpha_mask import TextAlphaMask
from ballontranslator.utils.text_effects import (
    FilterEffect,
    GlowEffect,
    TextFillEffect,
    GradientStop,
    HollowEffect,
    ImageEffect,
    ImageGenerationRecipe,
    LinearGradientPaint,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
    TexturePaint,
    with_primary_stroke,
)
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.llm_profiles import default_profile


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


class _BlockedImageBackend(ImageGenerationBackend):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, request, stop_event) -> np.ndarray:
        del request, stop_event
        self.started.set()
        self.release.wait()
        return np.full((2, 3, 4), 255, np.uint8)


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

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        self.app.processEvents()
        return bool(predicate())

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

    def test_faster_preview_is_opt_in_and_follows_selected_items(self):
        first = self._item(self._stack(StrokeEffect(width=0.12)))
        second = self._item(self._stack(StrokeEffect(width=0.18)))
        effect_panel = self.panel.texteffect_panel

        self.panel.set_textblk_item(first)
        self.assertFalse(effect_panel.faster_preview_toggle.isChecked())
        self.assertFalse(first.effect_renderer.faster_preview)

        effect_panel.faster_preview_toggle.click()
        self.assertTrue(first.effect_renderer.faster_preview)

        self.panel.set_textblk_item(second)
        self.assertTrue(second.effect_renderer.faster_preview)

        effect_panel.faster_preview_toggle.click()
        self.assertFalse(second.effect_renderer.faster_preview)

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
            self.panel.text_effect_session.add_effect('gradient')
        )
        self.assertTrue(any(
            isinstance(effect, TextFillEffect)
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
        self.assertFalse(item.blk.fontformat.text_effects[1].enabled)
        self.assertEqual(self.canvas.stack.count(), 3)

        effect_panel.stroke_cards[1].move_up_button.click()
        stack = item.blk.fontformat.text_effects
        self.assertEqual(stack[0].paint.color, (0, 0, 255))
        self.assertEqual(stack[1].paint.color, (255, 0, 0))
        self.assertEqual(self.canvas.stack.count(), 4)
        self.assertEqual(
            [card.index for card in self.panel.texteffect_panel.stroke_cards],
            [1, 0],
        )

    def test_effect_icons_expose_card_visibility_and_hollow_toggle(self):
        item = self._item(self._stack(
            StrokeEffect(),
            ShadowEffect(enabled=False),
            GlowEffect(),
            HollowEffect(),
            TextFillEffect(),
        ))
        self.panel.set_textblk_item(item)
        effect_panel = self.panel.texteffect_panel
        cards = (
            effect_panel.stroke_cards[0],
            effect_panel.shadow_cards[0],
            effect_panel.glow_cards[0],
            effect_panel.text_fill_cards[0],
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
            effect_panel.text_fill_cards[0].visibility_button.toolTip(),
            'Hide Gradient',
        )
        self.assertEqual(
            [
                effect_panel.add_effect_actions[key].text()
                for key in ('gradient', 'texture')
            ],
            ['Gradient', 'Texture'],
        )
        self.assertNotIn('color', effect_panel.add_effect_actions)
        self.assertNotIn('text_fill', effect_panel.add_effect_actions)
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
        self.assertTrue(effect_panel.add_effect_button.isEnabled())
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
        self.assertEqual(mixed_card.position_selector.currentData(), 'inside')
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
        self.assertTrue(mixed_card.gradient_editor.angle_editor.isEnabled())

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
        self.assertEqual(mixed_card.paint_button.text(), '')
        self.assertEqual(
            mixed_card.paint_button.accessibleName(),
            'Choose Stroke Color',
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

        # The reference card keeps the primary item's exact paint type.
        hetero_solid = self._item(self._stack(StrokeEffect(
            paint=SolidPaint((11, 12, 13))
        )))
        hetero_gradient = self._item(self._stack(StrokeEffect(
            paint=LinearGradientPaint(angle=90.0)
        )))
        self.canvas.selected = [hetero_solid, hetero_gradient]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.stroke_cards[0]
        self.assertEqual(
            mixed_card.fill_type_selector.currentData(), 'linear_gradient'
        )
        self.assertTrue(mixed_card.paint_button.isHidden())
        mixed_card.fill_type_selector.setCurrentIndex(
            mixed_card.fill_type_selector.findData('solid')
        )
        self.assertTrue(all(
            isinstance(
                target.blk.fontformat.text_effects[0].paint, SolidPaint
            )
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

    def test_long_blend_leaf_does_not_grow_card_or_host(self):
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
                with self.subTest(theme=theme):
                    self.app.setStyleSheet(parse_stylesheet(theme))
                    host = QWidget()
                    layout = QVBoxLayout(host)
                    card = StrokeEffectCard(0)
                    card.set_values([StrokeEffect(blend_mode='normal')])
                    layout.addWidget(card)
                    host.resize(
                        host.minimumSizeHint().width(),
                        host.minimumSizeHint().height(),
                    )
                    host.show()
                    self.app.processEvents()
                    before_card_minimum = card.minimumSizeHint().width()
                    before_host_minimum = host.minimumSizeHint().width()
                    before_host_width = host.width()

                    card.set_values([
                        StrokeEffect(blend_mode='linear_dodge')
                    ])
                    layout.activate()
                    self.app.processEvents()

                    self.assertLessEqual(
                        card.minimumSizeHint().width(), before_card_minimum
                    )
                    self.assertLessEqual(
                        host.minimumSizeHint().width(), before_host_minimum
                    )
                    self.assertEqual(host.width(), before_host_width)
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
            'ballontranslator.ui.text_engine.effects.gradient_editor.'
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
            'ballontranslator.ui.text_engine.effects.gradient_editor.'
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

    def test_multi_selection_projects_primary_and_maps_occurrences(self):
        first = self._item(self._stack(StrokeEffect(width=0.1)))
        primary = self._item(self._stack(
            StrokeEffect(width=0.4), StrokeEffect(width=0.5)
        ))
        self.canvas.selected = [first, primary]
        self.panel.set_textblk_item(None, multi_select=True)
        effect_panel = self.panel.texteffect_panel

        self.assertTrue(effect_panel.add_effect_button.isEnabled())
        self.assertEqual(
            [card.width_control.editor.text() for card in effect_panel.stroke_cards],
            ['0.50', '0.40'],
        )
        matched, unmatched = effect_panel.stroke_cards
        self.assertTrue(matched.property('matched'))
        self.assertFalse(unmatched.property('matched'))

        self.assertTrue(
            self.panel.text_effect_session.commit_value(1, 'width', 0.25)
        )
        self.assertEqual(first.blk.fontformat.text_effects[0].width, 0.25)
        self.assertEqual(primary.blk.fontformat.text_effects[1].width, 0.25)
        self.assertEqual(self.canvas.stack.count(), 1)

        self.assertTrue(
            self.panel.text_effect_session.commit_value(0, 'width', 0.7)
        )
        self.assertEqual(first.blk.fontformat.text_effects[0].width, 0.25)
        self.assertEqual(primary.blk.fontformat.text_effects[0].width, 0.7)
        self.assertEqual(self.canvas.stack.count(), 2)

        self.assertTrue(self.panel.text_effect_session.commit_value(
            -1, 'overall_opacity', 0.8
        ))
        self.assertTrue(all(
            item.blk.fontformat.text_effects.overall_opacity == 0.8
            for item in (first, primary)
        ))
        self.assertEqual(self.canvas.stack.count(), 3)

    def test_explicit_primary_overrides_final_item_fallback(self):
        first = self._item(self._stack(StrokeEffect(width=0.45)))
        clicked = self._item(self._stack(StrokeEffect(width=0.15)))
        later = self._item(self._stack(StrokeEffect(width=0.85)))
        self.canvas.selected = [first, clicked, later]

        self.panel.set_textblk_item(None, multi_select=True)
        self.assertEqual(
            self.panel.texteffect_panel.stroke_cards[0]
            .width_control.editor.text(),
            '0.85',
        )
        with (
            patch.object(
                self.panel.texttransform_panel, 'set_transform_items',
                wraps=self.panel.texttransform_panel.set_transform_items,
            ) as set_transform_items,
            patch.object(
                self.panel.texteffect_panel, 'set_effect_items',
                wraps=self.panel.texteffect_panel.set_effect_items,
            ) as set_effect_items,
        ):
            self.panel.set_textblk_item(
                None, multi_select=True, primary_item=clicked
            )

        self.assertEqual(
            self.panel.text_transform_session.items,
            [first, clicked, later],
        )
        self.assertEqual(
            self.panel.text_effect_session.items,
            [clicked, first, later],
        )
        set_transform_items.assert_called_once_with([first, clicked, later])
        set_effect_items.assert_called_once_with([clicked, first, later])
        self.assertIs(self.panel.text_effect_session.items[0], clicked)
        self.assertEqual(
            self.panel.texteffect_panel.stroke_cards[0]
            .width_control.editor.text(),
            '0.15',
        )

    def test_matched_delta_is_relative_and_unmatched_delete_is_primary_only(self):
        other = self._item(self._stack(
            ShadowEffect(), StrokeEffect(width=0.2)
        ))
        primary = self._item(self._stack(
            StrokeEffect(width=0.6), GlowEffect()
        ))
        self.canvas.selected = [other, primary]
        self.panel.set_textblk_item(None, multi_select=True)
        controls = self.panel.texteffect_panel
        stroke_card = controls.stroke_cards[0]
        glow_card = controls.glow_cards[0]
        self.assertTrue(stroke_card.property('matched'))
        self.assertFalse(glow_card.property('matched'))
        self.assertEqual(stroke_card.width_control.editor.text(), '0.60')

        session = self.panel.text_effect_session
        with patch(
            'ballontranslator.ui.text_engine.effects.edit_session.'
            'matched_effect_occurrences'
        ) as recompute_matches:
            session.preview_parameter_delta(0, 'width', 0.1)
            session.preview_parameter_delta(0, 'width', 0.1)
        recompute_matches.assert_not_called()
        self.assertAlmostEqual(
            primary.effective_text_effects()[0].width, 0.7
        )
        self.assertAlmostEqual(other.effective_text_effects()[1].width, 0.3)
        self.assertTrue(session.commit_parameter_delta(0, 'width', 0.1))
        self.assertEqual(self.canvas.stack.count(), 1)

        glow_card.delete_button.click()
        self.assertFalse(any(
            isinstance(effect, GlowEffect)
            for effect in primary.blk.fontformat.text_effects
        ))
        self.assertTrue(any(
            isinstance(effect, ShadowEffect)
            for effect in other.blk.fontformat.text_effects
        ))
        self.assertEqual(self.canvas.stack.count(), 2)

        controls.stroke_cards[0].delete_button.click()
        self.assertTrue(all(
            not any(
                isinstance(effect, StrokeEffect)
                for effect in item.blk.fontformat.text_effects
            )
            for item in (other, primary)
        ))
        self.assertEqual(self.canvas.stack.count(), 3)

    def test_unmatched_reorder_is_primary_only_when_batch_order_is_unaligned(self):
        other = self._item(self._stack(StrokeEffect(width=0.2)))
        primary = self._item(self._stack(
            GlowEffect(size=0.3), StrokeEffect(width=0.4)
        ))
        self.canvas.selected = [other, primary]
        self.panel.set_textblk_item(None, multi_select=True)
        controls = self.panel.texteffect_panel
        glow = controls.glow_cards[0]
        stroke = controls.stroke_cards[0]

        self.assertFalse(glow.property('matched'))
        self.assertTrue(stroke.property('matched'))
        self.assertTrue(glow.move_up_button.isEnabled())
        self.assertFalse(glow.move_down_button.isEnabled())
        self.assertFalse(stroke.move_up_button.isEnabled())
        self.assertFalse(stroke.move_down_button.isEnabled())

        other_before = other.blk.fontformat.text_effects
        glow.move_up_button.click()

        self.assertEqual(other.blk.fontformat.text_effects, other_before)
        self.assertEqual(
            tuple(type(effect) for effect in primary.blk.fontformat.text_effects),
            (StrokeEffect, GlowEffect),
        )
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_heterogeneous_adds_match_and_reorder_requires_aligned_sequences(self):
        first = self._item(self._stack(StrokeEffect(width=0.2)))
        primary = self._item(self._stack(
            GlowEffect(size=0.3), StrokeEffect(width=0.4)
        ))
        self.canvas.selected = [first, primary]
        self.panel.set_textblk_item(None, multi_select=True)
        controls = self.panel.texteffect_panel

        controls.add_effect_actions['glow'].trigger()
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIsInstance(
            first.blk.fontformat.text_effects[0], GlowEffect
        )
        self.assertIsInstance(
            primary.blk.fontformat.text_effects[1], GlowEffect
        )
        added = next(card for card in controls.glow_cards if card.index == 1)
        self.assertTrue(added.property('matched'))
        surplus = next(card for card in controls.glow_cards if card.index == 0)
        self.assertFalse(surplus.property('matched'))
        self.assertTrue(
            self.panel.text_effect_session.commit_value(
                added.index, 'size', 0.55
            )
        )
        self.assertAlmostEqual(first.blk.fontformat.text_effects[0].size, 0.55)
        self.assertAlmostEqual(primary.blk.fontformat.text_effects[1].size, 0.55)

        # The primary's older surplus Glow remains unmatched, so the two
        # movable sequences differ and moving any card would be ambiguous.
        self.assertFalse(added.move_up_button.isEnabled())
        self.assertFalse(added.move_down_button.isEnabled())

        next(
            action for action in controls.filter_add_menu.actions()
            if action.data() == 'builtin:noise'
        ).trigger()
        self.assertEqual(self.canvas.stack.count(), 3)
        self.assertTrue(all(
            isinstance(item.blk.fontformat.text_effects[0], FilterEffect)
            for item in (first, primary)
        ))
        self.assertTrue(controls.filter_cards[-1].property('matched'))
        self.canvas.stack.undo()
        self.assertEqual(controls.filter_cards, [])
        self.canvas.stack.redo()
        filter_card = controls.filter_cards[-1]
        self.assertTrue(filter_card.property('matched'))
        self.assertTrue(
            self.panel.text_effect_session.commit_value(
                filter_card.index, 'enabled', False
            )
        )
        self.assertTrue(all(
            not next(
                effect.enabled
                for effect in item.blk.fontformat.text_effects
                if isinstance(effect, FilterEffect)
            )
            for item in (first, primary)
        ))

        aligned_first = self._item(self._stack(
            StrokeEffect(width=0.1),
            TextFillEffect(paint=LinearGradientPaint()),
            GlowEffect(size=0.2),
        ))
        aligned_primary = self._item(self._stack(
            StrokeEffect(width=0.3),
            GlowEffect(size=0.4),
            TextFillEffect(paint=LinearGradientPaint()),
        ))
        self.canvas.selected = [aligned_first, aligned_primary]
        self.panel.set_textblk_item(None, multi_select=True)
        stroke = controls.stroke_cards[0]
        self.assertTrue(stroke.move_up_button.isEnabled())
        stroke.move_up_button.click()
        self.assertEqual(self.canvas.stack.count(), 5)
        for item in (aligned_first, aligned_primary):
            movable = [
                type(effect)
                for effect in reversed(item.blk.fontformat.text_effects.effects)
                if isinstance(effect, (StrokeEffect, GlowEffect))
            ]
            self.assertEqual(movable, [StrokeEffect, GlowEffect])

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
            item.blk.fontformat.text_effects[0].glow_type, 'inner'
        )
        self.assertEqual(second.spread_control.label.text(), 'Choke')
        second.visibility_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[0].enabled)
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
            interleaved.blk.fontformat.text_effects[1], StrokeEffect
        )
        self.assertIsInstance(
            interleaved.blk.fontformat.text_effects[2], GlowEffect
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
            'Choose Glow Color',
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
        self.assertTrue(mixed_card.gradient_editor.angle_editor.isEnabled())
        self.assertEqual(mixed_card.gradient_editor.angle_editor.value(), 90.0)

        outer_type = self._item(self._stack(GlowEffect(glow_type='outer')))
        inner_type = self._item(self._stack(GlowEffect(glow_type='inner')))
        self.canvas.selected = [outer_type, inner_type]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.glow_cards[0]
        self.assertEqual(mixed_card.type_selector.currentData(), 'inner')
        self.assertEqual(mixed_card.spread_control.label.text(), 'Choke')
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
        self.assertEqual(
            mixed_card.fill_type_selector.currentData(), 'linear_gradient'
        )
        self.assertTrue(mixed_card.paint_button.isHidden())
        mixed_card.fill_type_selector.setCurrentIndex(
            mixed_card.fill_type_selector.findData('solid')
        )
        self.assertTrue(all(
            isinstance(
                target.blk.fontformat.text_effects[0].paint, SolidPaint
            )
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
        self.assertTrue(mixed_card.gradient_editor.angle_editor.isEnabled())
        self.assertEqual(mixed_card.gradient_editor.angle_editor.value(), 120.0)

    def test_foreground_paint_top_level_add_repeat_edit_and_mixed_values(self):
        original = self._constant_text_fill(angle=15.0)
        self.canvas.imgtrans_proj = Mock()
        item = self._item(self._stack(original))
        self.panel.set_textblk_item(item)
        effect_panel = self.panel.texteffect_panel
        effect_panel.add_effect_actions['gradient'].trigger()
        text_fill = item.blk.fontformat.text_effects[0]
        self.assertIsInstance(text_fill, TextFillEffect)
        self.assertIsInstance(text_fill.paint, LinearGradientPaint)
        self.assertEqual(item.blk.fontformat.text_effects[1], original)
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertTrue(all(
            effect_panel.add_effect_actions[key].isEnabled()
            for key in ('gradient', 'texture')
        ))
        self.assertEqual(len(effect_panel.text_fill_cards), 2)
        self.assertEqual(
            [card.index for card in effect_panel.text_fill_cards], [1, 0]
        )
        self.assertEqual(
            [card.title_label.text() for card in effect_panel.text_fill_cards],
            ['Gradient', 'Gradient'],
        )

        card = effect_panel.text_fill_cards[-1]
        self.assertEqual(card.title_label.text(), 'Gradient')

        card.visibility_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[0].enabled)
        self.assertEqual(self.canvas.stack.count(), 2)
        card.delete_button.click()
        self.assertEqual(sum(
            isinstance(effect, TextFillEffect)
            for effect in item.blk.fontformat.text_effects
        ), 1)
        self.assertEqual(self.canvas.stack.count(), 3)
        self.assertTrue(effect_panel.add_effect_actions['gradient'].isEnabled())

        with patch.object(QFileDialog, 'getOpenFileName') as chooser:
            effect_panel.add_effect_actions['texture'].trigger()
        chooser.assert_not_called()
        added_texture = item.blk.fontformat.text_effects[0]
        self.assertEqual(added_texture.paint, TexturePaint())
        self.assertEqual(
            effect_panel.text_fill_cards[-1].title_label.text(), 'Texture'
        )
        self.assertEqual(self.canvas.stack.count(), 4)

        common = self._constant_text_fill(angle=0.0)
        different = self._constant_text_fill(angle=90.0)
        first = self._item(self._stack(common))
        second = self._item(self._stack(different))
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = effect_panel.text_fill_cards[0]
        self.assertFalse(mixed_card.gradient_editor.isHidden())
        self.assertTrue(mixed_card.gradient_editor.angle_editor.isEnabled())
        self.assertEqual(mixed_card.gradient_editor.angle_editor.value(), 90.0)

        self.canvas.selected = [first]
        self.panel.set_textblk_item(None, multi_select=True)
        self.assertIs(effect_panel.text_fill_cards[0], mixed_card)
        self.assertTrue(mixed_card.gradient_editor.angle_editor.isEnabled())

    @staticmethod
    def _constant_text_fill(angle: float = 0.0) -> TextFillEffect:
        return TextFillEffect(paint=LinearGradientPaint(
            stops=(
                GradientStop(0.0, (255, 0, 0), 1.0),
                GradientStop(1.0, (0, 0, 255), 1.0),
            ),
            angle=angle,
        ))

    def test_text_fill_inline_preview_cancel_commit_one_undo(self):
        before = self._stack(self._constant_text_fill())
        item = self._item(before)
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.text_fill_cards[0]
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
        self.assertIs(self.panel.texteffect_panel.text_fill_cards[0], card)

    def test_blend_selectors_display_mixed_and_commit(self):
        expected_modes = (
            'normal',
            'darken',
            'multiply',
            'color_burn',
            'linear_burn',
            'darker_color',
            'lighten',
            'screen',
            'color_dodge',
            'linear_dodge',
            'lighter_color',
        )

        def leaf_actions(selector):
            for root_action in selector.menu().actions():
                submenu = root_action.menu()
                if submenu is None:
                    yield root_action
                else:
                    yield from submenu.actions()

        first = self._item(self._stack(
            StrokeEffect(blend_mode='normal'),
            ShadowEffect(blend_mode='multiply'),
            GlowEffect(blend_mode='linear_dodge'),
            TextFillEffect(blend_mode='darker_color'),
        ))
        self.panel.set_textblk_item(first)
        controls = self.panel.texteffect_panel
        cards = (
            controls.stroke_cards[0],
            controls.shadow_cards[0],
            controls.glow_cards[0],
            controls.text_fill_cards[0],
        )
        self.assertEqual(
            [card.blend_selector.current_mode() for card in cards],
            ['normal', 'multiply', 'linear_dodge', 'darker_color'],
        )
        for card, accessible_name in zip(cards, (
            'Stroke Blend: Normal',
            'Shadow Blend: Multiply',
            'Glow Blend: Linear Dodge (Add)',
            'Gradient Blend: Darker Color',
        )):
            selector = card.blend_selector
            root_actions = selector.menu().actions()
            self.assertEqual(
                [action.text() for action in root_actions],
                ['Normal', 'Darken', 'Lighten'],
            )
            self.assertEqual(root_actions[0].data(), 'normal')
            self.assertIsNone(root_actions[0].menu())
            self.assertEqual(
                [action.data() for action in root_actions[1].menu().actions()],
                list(expected_modes[1:6]),
            )
            self.assertEqual(
                [action.data() for action in root_actions[2].menu().actions()],
                list(expected_modes[6:]),
            )
            actions = list(leaf_actions(selector))
            self.assertEqual(
                [action.data() for action in actions], list(expected_modes)
            )
            self.assertEqual(
                [action.data() for action in actions if action.isChecked()],
                [selector.current_mode()],
            )
            self.assertEqual(selector.accessibleName(), accessible_name)
            self.assertEqual(
                selector.accessibleDescription(), selector.toolTip()
            )

        undo_count = self.canvas.stack.count()
        stroke_actions = list(leaf_actions(cards[0].blend_selector))
        next(
            action for action in stroke_actions
            if action.data() == 'color_burn'
        ).trigger()
        self.assertEqual(
            first.blk.fontformat.text_effects[0].blend_mode, 'color_burn'
        )
        self.assertEqual(self.canvas.stack.count(), undo_count + 1)

        second = self._item(self._stack(
            StrokeEffect(blend_mode='normal'),
            ShadowEffect(blend_mode='darken'),
            GlowEffect(blend_mode='lighten'),
            TextFillEffect(blend_mode='normal'),
        ))
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        cards = (
            controls.stroke_cards[0],
            controls.shadow_cards[0],
            controls.glow_cards[0],
            controls.text_fill_cards[0],
        )
        self.assertEqual(
            [card.blend_selector.current_mode() for card in cards],
            ['normal', 'darken', 'lighten', 'normal'],
        )
        self.assertTrue(all(card.property('matched') for card in cards))
        self.assertTrue(all(
            sum(action.isChecked() for action in leaf_actions(
                card.blend_selector
            )) == 1
            for card in cards
        ))
        undo_count = self.canvas.stack.count()
        fill_card = cards[-1]
        next(
            action for action in leaf_actions(fill_card.blend_selector)
            if action.data() == 'lighter_color'
        ).trigger()
        self.assertTrue(all(
            item.blk.fontformat.text_effects[3].blend_mode == 'lighter_color'
            for item in (first, second)
        ))
        self.assertEqual(self.canvas.stack.count(), undo_count + 1)

    def test_text_fill_opacity_preview_cancel_and_commit(self):
        before = self._stack(TextFillEffect(opacity=0.8))
        item = self._item(before)
        self.panel.set_textblk_item(item)
        editor = (
            self.panel.texteffect_panel.text_fill_cards[0]
            .opacity_control.editor
        )

        editor.setText('35.0%')
        editor.textEdited.emit('35.0%')
        self.assertEqual(item.blk.fontformat.text_effects, before)
        self.assertEqual(item.effective_text_effects()[0].opacity, 0.35)
        QApplication.sendEvent(
            editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects(), before)
        self.assertEqual(self.canvas.stack.count(), 0)

        editor.setText('45.0%')
        editor.textEdited.emit('45.0%')
        editor.returnPressed.emit()
        self.assertEqual(item.blk.fontformat.text_effects[0].opacity, 0.45)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_text_fill_moves_only_with_fills_as_one_multi_item_command(self):
        red = self._constant_text_fill(angle=0.0)
        blue = self._constant_text_fill(angle=30.0)
        green = self._constant_text_fill(angle=60.0)
        yellow = self._constant_text_fill(angle=90.0)
        first_before = self._stack(
            red, StrokeEffect(width=0.2), blue, GlowEffect(size=0.4)
        )
        second_before = self._stack(
            green, StrokeEffect(width=0.5), yellow, GlowEffect(size=0.7)
        )
        first = self._item(first_before)
        second = self._item(second_before)
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        cards = self.panel.texteffect_panel.text_fill_cards
        self.assertEqual([card.index for card in cards], [2, 0])
        top, bottom = cards
        self.assertFalse(top.move_up_button.isEnabled())
        self.assertTrue(top.move_down_button.isEnabled())
        self.assertTrue(bottom.move_up_button.isEnabled())
        self.assertFalse(bottom.move_down_button.isEnabled())
        self.assertTrue(bottom.move_up_button.icon().isNull())
        QApplication.sendEvent(bottom, QEvent(QEvent.Type.Enter))
        self.assertFalse(bottom.move_up_button.icon().isNull())
        QApplication.sendEvent(bottom, QEvent(QEvent.Type.Leave))
        self.assertTrue(bottom.move_up_button.icon().isNull())

        bottom.move_up_button.click()
        self.assertEqual(
            first.blk.fontformat.text_effects.effects,
            (blue, first_before[1], red, first_before[3]),
        )
        self.assertEqual(
            second.blk.fontformat.text_effects.effects,
            (yellow, second_before[1], green, second_before[3]),
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        self.canvas.stack.undo()
        self.assertEqual(first.blk.fontformat.text_effects, first_before)
        self.assertEqual(second.blk.fontformat.text_effects, second_before)

        self.assertFalse(self.panel.text_effect_session.move_effect(2, 1))
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_filter_submenu_repeat_preview_reorder_eye_delete_and_one_undo(self):
        item = self._item(TextEffectStack())
        self.panel.set_textblk_item(item)
        controls = self.panel.texteffect_panel
        actions = {
            action.data(): action for action in controls.filter_add_menu.actions()
        }
        self.assertEqual(
            list(actions),
            [
                'builtin:noise',
                'builtin:grain',
                'builtin:rough_edge',
                'builtin:gaussian_blur',
                'builtin:bloom',
                'builtin:glitch',
            ],
        )
        self.assertTrue(all(not action.icon().isNull() for action in actions.values()))

        actions['builtin:noise'].trigger()
        actions['builtin:grain'].trigger()
        self.assertEqual(len(controls.filter_cards), 2)
        self.assertEqual(self.canvas.stack.count(), 2)
        self.assertEqual(
            [effect.filter_id for effect in item.blk.fontformat.text_effects],
            ['builtin:grain', 'builtin:noise'],
        )
        self.assertEqual(
            [card.filter_id for card in controls.filter_cards],
            ['builtin:noise', 'builtin:grain'],
        )

        noise_card = controls.filter_cards[0]
        amount = noise_card.numeric_controls['amount']
        amount.editor.setText('55.0')
        amount.editor.textEdited.emit('55.0')
        self.assertEqual(
            item.effective_text_effects()[1].params_dict()['amount'], 0.55
        )
        self.assertEqual(
            item.blk.fontformat.text_effects[1].params_dict()['amount'], 0.2
        )
        amount.editor.returnPressed.emit()
        self.assertEqual(self.canvas.stack.count(), 3)
        self.assertIs(controls.filter_cards[0], noise_card)

        mode = noise_card.choice_selectors['mode']
        mode.setCurrentIndex(mode.findData('color'))
        self.assertEqual(
            item.blk.fontformat.text_effects[1].params_dict()['mode'], 'color'
        )
        self.assertEqual(self.canvas.stack.count(), 4)

        controls.filter_cards[0].move_down_button.click()
        self.assertEqual(
            [effect.filter_id for effect in item.blk.fontformat.text_effects],
            ['builtin:noise', 'builtin:grain'],
        )
        self.assertEqual(self.canvas.stack.count(), 5)
        controls.filter_cards[1].visibility_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[0].enabled)
        controls.filter_cards[1].delete_button.click()
        self.assertEqual(len(item.blk.fontformat.text_effects), 1)
        self.assertEqual(self.canvas.stack.count(), 7)
        self.canvas.stack.undo()
        self.assertEqual(len(item.blk.fontformat.text_effects), 2)

    def test_new_filter_cards_and_gaussian_preview_cancel_commit_undo(self):
        before = self._stack(
            FilterEffect('builtin:bloom'),
            FilterEffect('builtin:glitch'),
            FilterEffect('builtin:gaussian_blur'),
        )
        item = self._item(before)
        self.panel.set_textblk_item(item)
        cards = self.panel.texteffect_panel.filter_cards
        self.assertEqual(
            [card.title_label.text() for card in cards],
            ['Gaussian Blur', 'Glitch', 'Bloom'],
        )
        gaussian, glitch, bloom = cards
        self.assertEqual(
            set(bloom.numeric_controls),
            {'threshold', 'radius', 'intensity'},
        )
        self.assertEqual(
            set(glitch.numeric_controls),
            {'shift', 'block_size', 'activity', 'rgb_split', 'seed'},
        )
        self.assertEqual(set(gaussian.numeric_controls), {'radius'})
        radius = gaussian.numeric_controls['radius']

        radius.editor.setText('7.5')
        radius.editor.textEdited.emit('7.5')
        self.assertEqual(
            item.effective_text_effects()[2].params_dict()['radius'], 7.5
        )
        self.assertEqual(item.blk.fontformat.text_effects, before)
        QApplication.sendEvent(
            radius.editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects(), before)
        self.assertEqual(self.canvas.stack.count(), 0)

        radius.editor.setText('8.5')
        radius.editor.textEdited.emit('8.5')
        radius.editor.returnPressed.emit()
        self.assertEqual(
            item.blk.fontformat.text_effects[2].params_dict()['radius'], 8.5
        )
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertIs(self.panel.texteffect_panel.filter_cards[0], gaussian)
        self.canvas.stack.undo()
        self.assertEqual(item.blk.fontformat.text_effects, before)

    def test_filter_structural_mixed_ids_and_missing_card_recovery_controls(self):
        first = self._item(self._stack(FilterEffect('builtin:noise')))
        second = self._item(self._stack(FilterEffect('builtin:grain')))
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        controls = self.panel.texteffect_panel
        self.assertEqual(len(controls.filter_cards), 1)
        self.assertEqual(controls.filter_cards[0].title_label.text(), 'Grain')
        self.assertFalse(controls.filter_cards[0].property('matched'))

        missing = self._item(self._stack(FilterEffect('missing:local')))
        self.panel.set_textblk_item(missing)
        card = controls.filter_cards[0]
        self.assertIsInstance(card, FilterEffectCard)
        self.assertEqual(card.title_label.text(), 'Missing Filter: missing:local')
        self.assertEqual(card.iter_controls(), ())
        self.assertTrue(card.visibility_button.isEnabled())
        self.assertTrue(card.delete_button.isEnabled())
        card.visibility_button.click()
        self.assertFalse(missing.blk.fontformat.text_effects[0].enabled)
        card.delete_button.click()
        self.assertEqual(missing.blk.fontformat.text_effects.effects, ())
        self.canvas.stack.undo()
        self.assertEqual(
            missing.blk.fontformat.text_effects.effects,
            (FilterEffect('missing:local', enabled=False),),
        )

        newer = self._item(self._stack(FilterEffect(
            'builtin:noise', schema_version=9, enabled=False,
            params={'future': 'kept'},
        )))
        self.panel.set_textblk_item(newer)
        card = controls.filter_cards[0]
        self.assertEqual(card.title_label.text(), 'Noise')
        self.assertTrue(card.visibility_button.isEnabled())
        self.assertTrue(card.delete_button.isEnabled())
        self.assertFalse(card.numeric_controls['amount'].isEnabled())
        self.assertIn('schema 9', card.toolTip())

    def test_filter_card_display_uses_static_metadata_without_import(self):
        controls = self.panel.texteffect_panel
        registry = get_filter_registry()
        registry._modules.clear()
        self.panel.set_active_format(FontFormat(text_effects=self._stack(
            FilterEffect('builtin:noise')
        )))

        self.assertEqual(registry._modules, {})
        self.assertEqual(controls.filter_cards[0].title_label.text(), 'Noise')
        self.assertIn('amount', controls.filter_cards[0].numeric_controls)

    def test_filter_runtime_overflow_defaults_during_card_sync(self):
        huge = int('9' * 4001)
        effect = FilterEffect('builtin:noise', params={
            'amount': huge, 'mode': 'monochrome', 'seed': 1,
        })
        item = self._item(self._stack(effect))

        self.panel.set_textblk_item(item)

        card = self.panel.texteffect_panel.filter_cards[0]
        self.assertEqual(card.numeric_controls['amount'].editor.text(), '20.0%')
        self.assertEqual(
            item.blk.fontformat.text_effects[0].params_dict()['amount'], huge
        )

    def test_panel_shared_static_metadata_uses_translation_context(self):
        class PrefixTranslator(QTranslator):
            def translate(
                self, context, source_text, disambiguation=None, n=-1
            ):
                if context == 'TextEffectPanel':
                    return 'Localized ' + source_text
                return source_text

        translator = PrefixTranslator()
        self.app.installTranslator(translator)
        controls = TextEffectPanel(
            'Filters', 'localized_filter_test', 'localized_filter_test_expand'
        )
        custom_card = None
        try:
            actions = {
                action.data(): action.text()
                for action in controls.filter_add_menu.actions()
            }
            self.assertEqual(actions['builtin:noise'], 'Localized Noise')

            controls._set_effect_states((self._stack(
                FilterEffect('builtin:noise')
            ),))
            card = controls.filter_cards[0]
            self.assertEqual(card.title_label.text(), 'Localized Noise')
            self.assertEqual(
                card.numeric_controls['amount'].label.text(),
                'Localized Amount',
            )
            mode = card.choice_selectors['mode']
            self.assertEqual(
                mode.itemText(mode.findData('monochrome')),
                'Localized Monochrome',
            )

            controls._set_effect_states((self._stack(
                StrokeEffect(), ShadowEffect(), GlowEffect(), TextFillEffect()
            ),))
            blend_cards = (
                controls.stroke_cards[0],
                controls.shadow_cards[0],
                controls.glow_cards[0],
                controls.text_fill_cards[0],
            )
            for blend_card in blend_cards:
                selector = blend_card.blend_selector
                root_actions = selector.menu().actions()
                self.assertEqual(
                    [action.text() for action in root_actions],
                    [
                        'Localized Normal',
                        'Localized Darken',
                        'Localized Lighten',
                    ],
                )
                self.assertEqual(
                    [
                        action.text()
                        for action in root_actions[1].menu().actions()
                    ],
                    [
                        'Localized Darken',
                        'Localized Multiply',
                        'Localized Color Burn',
                        'Localized Linear Burn',
                        'Localized Darker Color',
                    ],
                )
                self.assertEqual(
                    [
                        action.text()
                        for action in root_actions[2].menu().actions()
                    ],
                    [
                        'Localized Lighten',
                        'Localized Screen',
                        'Localized Color Dodge',
                        'Localized Linear Dodge (Add)',
                        'Localized Lighter Color',
                    ],
                )
                self.assertEqual(
                    selector.toolTip(),
                    'Localized Blends with earlier output in the text-effect '
                    'stack, not the page image or backdrop.',
                )
                self.assertEqual(selector.text(), 'Localized Normal')
                self.assertTrue(
                    selector.accessibleName().endswith(': Localized Normal')
                )
                selector.set_mode(None)
                self.assertEqual(selector.text(), 'Localized Mixed')
                self.assertTrue(
                    selector.accessibleName().endswith(': Localized Mixed')
                )

            builtin = get_filter_registry().get_spec('builtin:noise')
            custom = replace(
                builtin,
                builtin=False,
                name='Noise',
                params=(replace(
                    builtin.params[0], label='Amount', decimals=6
                ),),
            )
            custom_card = FilterEffectCard(
                0, custom.filter_id, custom, controls
            )
            self.assertEqual(custom_card.title_label.text(), 'Noise')
            self.assertEqual(
                custom_card.numeric_controls['amount'].label.text(), 'Amount'
            )
        finally:
            if custom_card is not None:
                custom_card.deleteLater()
            controls.deleteLater()
            self.app.removeTranslator(translator)
            self.app.processEvents()

    def test_known_filter_lazy_load_failure_disables_only_parameter_editors(self):
        item = self._item(self._stack(FilterEffect('builtin:noise')))
        self.panel.set_textblk_item(item)
        controls = self.panel.texteffect_panel
        card = controls.filter_cards[0]
        registry = Mock()
        registry.get_runtime_failure.return_value = FilterUnavailableError(
            'Noise dependency is unavailable'
        )

        with patch(
            'ballontranslator.ui.text_engine.effects.panel.'
            'get_filter_registry',
            return_value=registry,
        ):
            card.set_values((item.blk.fontformat.text_effects[0],))

        self.assertEqual(card.title_label.text(), 'Noise')
        self.assertTrue(card.visibility_button.isEnabled())
        self.assertTrue(card.delete_button.isEnabled())
        self.assertFalse(card.numeric_controls['amount'].isEnabled())
        self.assertIn('dependency', card.toolTip())

        failing_registry = Mock()
        failing_registry.resolve.side_effect = FilterUnavailableError(
            'Noise dependency is unavailable'
        )
        with patch(
            'ballontranslator.ui.text_engine.effects.edit_session.'
            'get_filter_registry',
            return_value=failing_registry,
        ):
            self.assertFalse(
                self.panel.text_effect_session.commit_value(
                    0, 'param:amount', 0.5
                )
            )
        self.assertEqual(self.canvas.stack.count(), 0)
        self.assertEqual(
            item.blk.fontformat.text_effects[0].params_dict(), {}
        )

    def test_explicit_filter_param_edit_commits_migrated_schema(self):
        effect = FilterEffect(
            'custom:migrate', schema_version=1,
            params={'old_amount': 0.4, 'opaque': 'preserved'},
        )
        runtime = SimpleNamespace(
            spec=SimpleNamespace(schema_version=2),
            params={'amount': 0.4},
        )
        registry = Mock()
        registry.resolve.return_value = runtime

        with patch(
            'ballontranslator.ui.text_engine.effects.edit_session.'
            'get_filter_registry',
            return_value=registry,
        ):
            result = self.panel.text_effect_session._with_value(
                self._stack(effect), 0, 'param:amount', 0.75
            )

        self.assertEqual(result.effects[0].schema_version, 2)
        self.assertEqual(result.effects[0].params_dict(), {'amount': 0.75})
        self.assertEqual(
            effect.params_dict(),
            {'old_amount': 0.4, 'opaque': 'preserved'},
        )

    def test_filter_drag_reads_omitted_param_from_metadata_default(self):
        state = self._stack(FilterEffect('builtin:noise', params={'seed': 4}))

        value = self.panel.text_effect_session._value_at(
            state, 0, 'param:amount'
        )

        self.assertEqual(value, 0.2)
        self.assertEqual(state.effects[0].params_dict(), {'seed': 4})

    def test_filter_card_rebuild_survives_deferred_delete(self):
        item = self._item(self._stack(FilterEffect('builtin:noise')))
        self.panel.set_textblk_item(item)
        controls = self.panel.texteffect_panel
        old_card = controls.filter_cards[0]

        item.blk.fontformat.text_effects = TextEffectStack()
        controls.set_effect_items((item,))
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.assertEqual(controls.filter_cards, [])

        item.blk.fontformat.text_effects = self._stack(
            FilterEffect('builtin:noise')
        )
        controls.set_effect_items((item,))
        new_card = controls.filter_cards[0]
        self.assertIsNot(new_card, old_card)
        new_card.visibility_button.click()
        self.assertFalse(item.blk.fontformat.text_effects[0].enabled)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_filter_and_generated_cards_reorder_as_one_global_sequence(self):
        item = self._item(self._stack(
            StrokeEffect(),
            FilterEffect('builtin:noise'),
            GlowEffect(),
        ))
        self.panel.set_textblk_item(item)
        controls = self.panel.texteffect_panel
        filter_card = controls.filter_cards[0]

        self.assertTrue(filter_card.move_up_button.isEnabled())
        self.assertTrue(filter_card.move_down_button.isEnabled())
        filter_card.move_up_button.click()

        self.assertEqual(
            tuple(type(effect) for effect in item.blk.fontformat.text_effects),
            (StrokeEffect, GlowEffect, FilterEffect),
        )
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_add_generated_ignores_legacy_mid_structural_position(self):
        filter_effect = FilterEffect('builtin:noise')
        inner = ShadowEffect(shadow_type='inner')
        text_fill = TextFillEffect()
        results = []
        for effects in (
            (filter_effect, text_fill, inner),
            (filter_effect, inner, text_fill),
        ):
            item = self._item(self._stack(*effects))
            self.panel.set_textblk_item(item)
            self.assertTrue(
                self.panel.text_effect_session.add_effect('glow')
            )
            results.append(tuple(
                type(effect)
                for effect in item.blk.fontformat.text_effects
                if isinstance(effect, (
                    FilterEffect, StrokeEffect, ShadowEffect, GlowEffect
                ))
            ))

        self.assertEqual(results[0], results[1])
        self.assertEqual(
            results[0],
            (GlowEffect, FilterEffect, ShadowEffect),
        )

    def test_image_add_is_disabled_for_multi_selection_and_remains_primary_only(self):
        asset = RasterAssetRef(
            'assets/' + 'e' * 64 + '.png', 'image.png'
        )
        project = Mock()
        project.import_raster_asset.return_value = asset
        project.resolve_raster_asset.return_value = '/project/assets/image.png'
        project.load_raster_asset.return_value = np.full(
            (2, 2, 4), (20, 60, 230, 255), dtype=np.uint8
        )
        self.canvas.imgtrans_proj = project
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        first = self._item(self._stack(StrokeEffect()))
        second = self._item(self._stack(StrokeEffect()))
        scene.addItem(first)
        scene.addItem(second)
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        controls = self.panel.texteffect_panel
        action = controls.add_effect_actions['image']

        self.assertFalse(action.isEnabled())
        with patch.object(QFileDialog, 'getOpenFileName') as chooser:
            action.trigger()
        chooser.assert_not_called()
        self.assertEqual(self.canvas.stack.count(), 0)
        self.assertFalse(any(
            isinstance(effect, ImageEffect)
            for item in (first, second)
            for effect in item.blk.fontformat.text_effects
        ))

        self.panel.set_textblk_item(second)
        self.assertTrue(action.isEnabled())
        action.trigger()
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertEqual(
            second.blk.fontformat.text_effects.effects,
            (ImageEffect(), StrokeEffect()),
        )
        self.assertEqual(first.blk.fontformat.text_effects.effects, (StrokeEffect(),))
        self.assertEqual(len(controls.image_cards), 1)
        card = controls.image_cards[0]
        self.assertEqual(card.image_field.text(), '')
        self.assertEqual(card.image_button.text(), '')
        self.assertFalse(card.image_button.icon().isNull())
        self.assertIs(card.image_button.parent(), card.image_field)
        self.assertIn('Choose an image', card.image_button.toolTip())
        self.assertIn('Hidden while editing', card.image_button.toolTip())
        placement_hints = [
            card.mode_selector.itemData(
                index, Qt.ItemDataRole.ToolTipRole
            )
            for index in range(card.mode_selector.count())
        ]
        self.assertTrue(all(placement_hints))
        self.assertEqual(
            [
                card.mode_selector.itemText(index)
                for index in range(card.mode_selector.count())
            ],
            ['In Front', 'Behind'],
        )
        self.assertEqual(
            card.mode_selector.toolTip(), placement_hints[0]
        )
        with patch.object(
            QFileDialog,
            'getOpenFileName',
            return_value=('/tmp/image.png', ''),
        ):
            card.image_button.click()
        self.assertEqual(self.canvas.stack.count(), 2)
        self.assertEqual(
            second.blk.fontformat.text_effects.effects[0].asset, asset
        )
        self.assertFalse(any(
            isinstance(effect, ImageEffect)
            for effect in first.blk.fontformat.text_effects
        ))

        card.mode_selector.setCurrentIndex(
            card.mode_selector.findData('background')
        )
        self.assertIn('behind', card.mode_selector.toolTip())
        self.assertEqual(self.canvas.stack.count(), 3)
        card.move_up_button.click()
        self.assertEqual(self.canvas.stack.count(), 4)
        self.assertIsInstance(
            second.blk.fontformat.text_effects.effects[1], ImageEffect
        )
        card = controls.image_cards[0]
        card.delete_button.click()
        self.assertEqual(self.canvas.stack.count(), 5)
        self.assertFalse(any(
            isinstance(effect, ImageEffect)
            for effect in second.blk.fontformat.text_effects.effects
        ))
        self.canvas.stack.undo()
        self.assertEqual(len(controls.image_cards), 1)

        action.trigger()
        self.assertEqual(len(controls.image_cards), 2)

    def test_existing_multi_selection_image_card_is_primary_only(self):
        first = self._item(self._stack(
            ImageEffect(mode='background'), StrokeEffect(), GlowEffect()
        ))
        primary = self._item(self._stack(
            ImageEffect(mode='background'), StrokeEffect()
        ))
        self.canvas.selected = [first, primary]
        self.panel.set_textblk_item(None, multi_select=True)
        card = self.panel.texteffect_panel.image_cards[0]

        self.assertFalse(card.property('matched'))
        self.assertEqual(card.mode_selector.currentData(), 'background')
        card.mode_selector.setCurrentIndex(
            card.mode_selector.findData('foreground')
        )
        self.assertEqual(
            first.blk.fontformat.text_effects[0].mode, 'background'
        )
        self.assertEqual(
            primary.blk.fontformat.text_effects[0].mode, 'foreground'
        )
        self.assertEqual(self.canvas.stack.count(), 1)

        self.assertTrue(card.move_up_button.isEnabled())
        card.move_up_button.click()
        self.assertIsInstance(
            first.blk.fontformat.text_effects[0], ImageEffect
        )
        self.assertIsInstance(
            primary.blk.fontformat.text_effects[1], ImageEffect
        )
        self.assertEqual(self.canvas.stack.count(), 2)

        card = self.panel.texteffect_panel.image_cards[0]
        card.delete_button.click()
        self.assertIsInstance(
            first.blk.fontformat.text_effects[0], ImageEffect
        )
        self.assertFalse(any(
            isinstance(effect, ImageEffect)
            for effect in primary.blk.fontformat.text_effects
        ))
        self.assertEqual(self.canvas.stack.count(), 3)

    def test_image_generation_is_single_item_and_panel_globally_busy(self):
        old_profiles = pcfg.module.llm_profiles
        old_profile_id = pcfg.module.inpaint_llm_id
        profile = default_profile('OpenRouter')
        profile.support_image = True
        profile.image_model = 'image-v2'
        profile.image_model_options = ['image-v2']
        pcfg.module.llm_profiles = [profile]
        pcfg.module.inpaint_llm_id = profile.id
        try:
            project = Mock()
            self.canvas.imgtrans_proj = project
            scene = QGraphicsScene()
            scene.imgtrans_proj = project
            first = self._item(self._stack(ImageEffect(), ImageEffect()))
            second = self._item(self._stack(ImageEffect(), ImageEffect()))
            scene.addItem(first)
            scene.addItem(second)

            self.panel.set_textblk_item(first)
            controls = self.panel.texteffect_panel
            self.assertTrue(controls.image_cards[0].generate_button.isEnabled())
            controls.set_image_generation_state(0, 'running')
            active_card = next(
                card for card in controls.image_cards if card.index == 0
            )
            other_card = next(
                card for card in controls.image_cards if card.index != 0
            )
            self.assertEqual(active_card.generate_button.text(), 'Stop')
            self.assertTrue(active_card.generate_button.isEnabled())
            self.assertFalse(other_card.generate_button.isEnabled())
            self.assertFalse(other_card.model_selector.isEnabled())
            self.assertIn(
                'Another Image generation',
                other_card.generate_button.toolTip(),
            )

            controls.detach_image_generation_card()
            first.blk.fontformat.text_effects = self._stack(
                StrokeEffect(), ImageEffect(), ImageEffect()
            )
            controls.set_effect_items((first,))
            self.assertTrue(all(
                not card.generate_button.isEnabled()
                for card in controls.image_cards
            ))
            self.assertTrue(all(
                card.generate_button.text() == 'Generate'
                for card in controls.image_cards
            ))

            controls.set_image_generation_state(0, 'idle')
            self.canvas.selected = [first, second]
            first.blk.fontformat.text_effects = self._stack(
                ImageEffect(), ImageEffect()
            )
            self.panel.set_textblk_item(None, multi_select=True)
            multi_card = controls.image_cards[0]
            self.assertFalse(multi_card.generate_button.isEnabled())
            self.assertTrue(multi_card.image_button.isEnabled())
            self.assertIn('exactly one', multi_card.generate_button.toolTip())
        finally:
            pcfg.module.llm_profiles = old_profiles
            pcfg.module.inpaint_llm_id = old_profile_id

    def test_image_generation_draft_fields_are_local_to_owned_popup(self):
        old_profiles = pcfg.module.llm_profiles
        old_profile_id = pcfg.module.inpaint_llm_id
        old_inpainter = pcfg.module.inpainter
        first_profile = default_profile('OpenRouter')
        second_profile = default_profile('OpenAI')
        second_profile.image_model = 'image-v2'
        second_profile.image_model_options = ['image-v2']
        pcfg.module.llm_profiles = [first_profile, second_profile]
        pcfg.module.inpaint_llm_id = first_profile.id
        pcfg.module.inpainter = 'lama_large_512px'
        try:
            project = Mock()
            self.canvas.imgtrans_proj = project
            scene = QGraphicsScene()
            scene.imgtrans_proj = project
            item = self._item(self._stack(ImageEffect(
                generation=ImageGenerationRecipe(
                    profile_id=first_profile.id,
                    model=first_profile.image_model_options[0],
                    prompt='Persisted prompt',
                )
            )))
            scene.addItem(item)
            self.panel.set_textblk_item(item)
            card = self.panel.texteffect_panel.image_cards[0]
            selector = card.model_selector

            before = item.blk.fontformat.text_effects
            with patch.object(
                item, 'set_text_effects', wraps=item.set_text_effects
            ) as set_effects, patch.object(
                self.canvas,
                'push_undo_command',
                wraps=self.canvas.push_undo_command,
            ) as push_undo:
                # The custom menu pins the formatting owner for its complete
                # popup transaction, including QMenu's hide notification.
                self.panel.show()

                def clear_selection_while_closing() -> None:
                    self.assertTrue(self.panel.focusOnColorDialog)
                    self.panel.set_textblk_item(None)
                    self.assertIs(self.panel.textblk_item, item)
                    self.assertIs(
                        self.panel.texteffect_panel.image_cards[0], card
                    )

                selector.menu.aboutToHide.connect(
                    clear_selection_while_closing
                )

                def choose_model() -> None:
                    self.assertIs(
                        QApplication.activePopupWidget(), selector.menu
                    )
                    self.assertTrue(self.panel.focusOnColorDialog)
                    self.panel.set_textblk_item(None)
                    self.assertIs(self.panel.textblk_item, item)
                    self.assertIs(
                        self.panel.texteffect_panel.image_cards[0], card
                    )
                    action = next(
                        child
                        for parent in selector.menu.actions()
                        if parent.menu() is not None
                        for child in parent.menu().actions()
                        if child.data()
                        == (second_profile.id, 'image-v2')
                    )
                    action.trigger()
                    selector.menu.close()

                QTimer.singleShot(0, choose_model)
                selector.click()
                self.assertFalse(self.panel.focusOnColorDialog)

                # Native combo popups are top-level windows whose focused
                # list view still belongs to the formatting panel through
                # parentWidget(). Keep the same card if selection clears.
                card.context_selector.showPopup()
                self.app.processEvents()
                self.assertIsNotNone(QApplication.activePopupWidget())
                self.panel.set_textblk_item(None)
                self.assertIs(self.panel.textblk_item, item)
                self.assertIs(
                    self.panel.texteffect_panel.image_cards[0], card
                )
                card.context_selector.hidePopup()

                card.context_selector.setCurrentIndex(
                    card.context_selector.findData('none')
                )
                card.prompt_editor.setPlainText('Keep this draft')
                self.app.processEvents()

                self.assertEqual(set_effects.call_count, 0)
                self.assertEqual(push_undo.call_count, 0)

            self.assertIs(self.panel.texteffect_panel.image_cards[0], card)
            self.assertEqual(selector.profile_id, second_profile.id)
            self.assertEqual(selector.model, 'image-v2')
            self.assertEqual(
                card._generation_draft.model,
                'image-v2',
            )
            self.assertEqual(card.context_selector.currentData(), 'none')
            self.assertEqual(
                card.prompt_editor.toPlainText(), 'Keep this draft'
            )
            self.assertEqual(item.blk.fontformat.text_effects, before)
            self.assertEqual(self.canvas.stack.count(), 0)
            self.assertEqual(pcfg.module.inpaint_llm_id, first_profile.id)
            self.assertEqual(pcfg.module.inpainter, 'lama_large_512px')

            generation_request = Mock()
            card.generate_requested.disconnect()
            card.generate_requested.connect(generation_request)
            card.generate_button.click()
            recipe = generation_request.call_args.args[1]
            self.assertEqual(recipe.profile_id, second_profile.id)
            self.assertEqual(recipe.model, 'image-v2')
            self.assertEqual(recipe.context, 'none')
            self.assertEqual(recipe.prompt, 'Keep this draft')
            self.assertEqual(item.blk.fontformat.text_effects, before)
            self.assertEqual(self.canvas.stack.count(), 0)

            # A real target change projects the persisted recipe normally.
            other = self._item(self._stack(ImageEffect()))
            scene.addItem(other)
            self.panel.set_textblk_item(other)
            self.panel.set_textblk_item(item)
            reset = self.panel.texteffect_panel.image_cards[0]
            self.assertEqual(reset.model_selector.profile_id, first_profile.id)
            self.assertNotEqual(reset.model_selector.model, 'image-v2')
            self.assertEqual(
                reset.prompt_editor.toPlainText(), 'Persisted prompt'
            )
        finally:
            pcfg.module.llm_profiles = old_profiles
            pcfg.module.inpaint_llm_id = old_profile_id
            pcfg.module.inpainter = old_inpainter

    def test_generated_image_recipe_asset_and_history_use_one_command(self):
        old_asset = RasterAssetRef(
            'assets/' + '1' * 64 + '.png', 'old.png'
        )
        new_asset = RasterAssetRef(
            'assets/' + '2' * 64 + '.png', 'generated.png'
        )
        old_recipe = ImageGenerationRecipe(
            profile_id='old-profile',
            model='old-model',
            prompt='Old prompt',
        )
        new_recipe = ImageGenerationRecipe(
            profile_id='new-profile',
            model='new-model',
            context='none',
            prompt='New prompt',
        )
        original = ImageEffect(old_asset, generation=old_recipe)
        project = Mock()
        project.load_identity = object()
        project.current_img = 'page.png'
        project.resolve_raster_asset.return_value = '/project/assets/old.png'
        project.import_raster_asset_bytes.return_value = new_asset
        self.canvas.imgtrans_proj = project
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        item = self._item(self._stack(original))
        scene.addItem(item)
        self.panel.set_textblk_item(item)
        session = self.panel.text_effect_session
        card = self.panel.texteffect_panel.image_cards[0]
        card._generation_draft = new_recipe
        card._generation_draft_dirty = True
        card.model_selector.set_recipe(new_recipe)
        card.context_selector.setCurrentIndex(
            card.context_selector.findData('none')
        )
        card.prompt_editor.setPlainText('New prompt')
        session._pending_image_generation = (
            item,
            0,
            original,
            project,
            project.load_identity,
            project.current_img,
            new_recipe,
        )

        session._finish_image_generation(0, b'generated-png')

        self.assertEqual(self.canvas.stack.count(), 1)
        generated = item.blk.fontformat.text_effects[0]
        self.assertEqual(generated.asset, new_asset)
        self.assertEqual(generated.generation, new_recipe)
        card = self.panel.texteffect_panel.image_cards[0]
        self.assertEqual(card.prompt_editor.toPlainText(), 'New prompt')
        self.assertFalse(card._generation_draft_dirty)
        project.import_raster_asset_bytes.assert_called_once_with(
            b'generated-png', 'generated.png'
        )

        self.canvas.stack.undo()
        card = self.panel.texteffect_panel.image_cards[0]
        self.assertEqual(item.blk.fontformat.text_effects[0], original)
        self.assertEqual(card.prompt_editor.toPlainText(), 'Old prompt')
        self.assertEqual(card.model_selector.model, 'old-model')
        self.canvas.stack.redo()
        card = self.panel.texteffect_panel.image_cards[0]
        self.assertEqual(card.prompt_editor.toPlainText(), 'New prompt')
        self.assertEqual(card.context_selector.currentData(), 'none')

    def test_same_generation_asset_refreshes_project_without_undo(self):
        asset = RasterAssetRef(
            'assets/' + '5' * 64 + '.png', 'generated.png'
        )
        recipe = ImageGenerationRecipe(
            profile_id='artist', model='image-v2', prompt='Texture'
        )
        effect = ImageEffect(asset, generation=recipe)
        project = Mock()
        project.load_identity = object()
        project.current_img = 'page.png'
        project.import_raster_asset_bytes.return_value = asset
        self.canvas.imgtrans_proj = project
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        item = self._item(self._stack(effect))
        scene.addItem(item)
        self.panel.set_textblk_item(item)
        session = self.panel.text_effect_session
        session._pending_image_generation = (
            item,
            0,
            effect,
            project,
            project.load_identity,
            project.current_img,
            recipe,
        )

        with patch.object(
            self.panel.texteffect_panel, 'project_assets_changed'
        ) as refresh_assets:
            session._finish_image_generation(0, b'same-generated-png')

        self.assertEqual(self.canvas.stack.count(), 0)
        self.assertEqual(item.blk.fontformat.text_effects[0], effect)
        project.import_raster_asset_bytes.assert_called_once_with(
            b'same-generated-png', 'generated.png'
        )
        refresh_assets.assert_called_once_with()

    def test_generation_preserves_mode_and_visibility_edits_made_in_flight(self):
        old_asset = RasterAssetRef(
            'assets/' + '3' * 64 + '.png', 'old.png'
        )
        new_asset = RasterAssetRef(
            'assets/' + '4' * 64 + '.png', 'generated.png'
        )
        recipe = ImageGenerationRecipe(
            profile_id='artist', model='image-v2', prompt='Texture'
        )
        original = ImageEffect(old_asset, generation=recipe)
        project = Mock()
        project.load_identity = object()
        project.current_img = 'page.png'
        project.resolve_raster_asset.return_value = '/project/assets/old.png'
        project.import_raster_asset_bytes.return_value = new_asset
        self.canvas.imgtrans_proj = project
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        item = self._item(self._stack(original))
        scene.addItem(item)
        self.panel.set_textblk_item(item)
        session = self.panel.text_effect_session
        session._pending_image_generation = (
            item,
            0,
            original,
            project,
            project.load_identity,
            project.current_img,
            recipe,
        )

        self.assertTrue(session.commit_value(0, 'mode', 'background'))
        self.assertTrue(session.commit_value(0, 'enabled', False))
        session._finish_image_generation(0, b'generated-png')

        generated = item.blk.fontformat.text_effects[0]
        self.assertEqual(generated.asset, new_asset)
        self.assertEqual(generated.mode, 'background')
        self.assertFalse(generated.enabled)
        self.assertEqual(self.canvas.stack.count(), 3)
        self.canvas.stack.undo()
        previous = item.blk.fontformat.text_effects[0]
        self.assertEqual(previous.asset, old_asset)
        self.assertEqual(previous.mode, 'background')
        self.assertFalse(previous.enabled)

    def test_deleted_generation_target_is_stale_and_history_stops_worker(self):
        effect = ImageEffect()
        project = Mock()
        project.load_identity = object()
        project.current_img = 'page.png'
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        item = self._item(self._stack(effect))
        block = item.blk
        original_stack = block.fontformat.text_effects
        scene.addItem(item)
        self.panel.set_textblk_item(item)
        session = self.panel.text_effect_session
        session._pending_image_generation = (
            item,
            0,
            effect,
            project,
            project.load_identity,
            project.current_img,
            ImageGenerationRecipe(),
        )
        with patch.object(session, 'stop_image_generation') as stop:
            session.resolve_for_history_change()
        stop.assert_called_once_with(detach_card=True)
        item.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

        self.assertFalse(session._generation_target_is_current())
        session._finish_image_generation(0, b'stale-png')
        project.import_raster_asset_bytes.assert_not_called()
        self.assertEqual(block.fontformat.text_effects, original_stack)
        self.assertIsNone(session._pending_image_generation)
        session.items = []

    def test_image_generation_context_memory_error_is_reported(self):
        project = Mock()
        project.load_identity = object()
        project.current_img = 'page.png'
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        item = self._item(self._stack(ImageEffect()))
        scene.addItem(item)
        self.panel.set_textblk_item(item)
        session = self.panel.text_effect_session
        recipe = ImageGenerationRecipe(
            profile_id='unused', model='unused', context='source'
        )

        with patch(
            'ballontranslator.ui.text_engine.effects.edit_session.'
            'prepare_image_generation_context',
            side_effect=MemoryError('crop allocation failed'),
        ), patch.object(
            self.panel.texteffect_panel,
            'show_image_generation_context_error',
        ) as show_error:
            self.assertFalse(session.generate_image(0, recipe))

        show_error.assert_called_once_with(0, 'crop allocation failed')
        self.assertIsNone(session._pending_image_generation)

    def test_missing_project_generation_error_uses_translation_context(self):
        class PrefixTranslator(QTranslator):
            def translate(
                self, context, source_text, disambiguation=None, n=-1
            ):
                del disambiguation, n
                if context == 'TextEffectEditSession':
                    return 'Localized ' + source_text
                return source_text

        scene = QGraphicsScene()
        item = self._item(self._stack(ImageEffect()))
        scene.addItem(item)
        self.panel.set_textblk_item(item)
        session = self.panel.text_effect_session
        translator = PrefixTranslator()
        self.app.installTranslator(translator)
        try:
            with patch.object(
                self.panel.texteffect_panel,
                'show_image_generation_context_error',
            ) as show_error:
                self.assertFalse(session.generate_image(
                    0,
                    ImageGenerationRecipe(
                        profile_id='unused', model='unused'
                    ),
                ))
            message = show_error.call_args.args[1]
            self.assertTrue(message.startswith('Localized '))
        finally:
            self.app.removeTranslator(translator)

    def test_image_add_is_disabled_without_project_item_context(self):
        controls = self.panel.texteffect_panel
        self.panel.set_active_format(self.panel.global_format)
        self.assertFalse(controls.add_effect_actions['image'].isEnabled())

        item = self._item(self._stack(StrokeEffect()))
        self.panel.set_textblk_item(item)
        self.assertFalse(controls.add_effect_actions['image'].isEnabled())
        self.assertFalse(self.panel.text_effect_session.add_effect('image'))
        self.assertEqual(self.canvas.stack.count(), 0)

    def test_image_chooser_cancel_and_error_preserve_owner_and_value(self):
        old_asset = RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'old.png'
        )
        project = Mock()
        project.resolve_raster_asset.return_value = '/project/assets/old.png'
        self.canvas.imgtrans_proj = project
        item = self._item(self._stack(ImageEffect(old_asset)))
        self.panel.set_textblk_item(item)
        controls = self.panel.texteffect_panel
        card = controls.image_cards[0]

        with patch.object(
            QFileDialog, 'getOpenFileName', return_value=('', '')
        ) as chooser:
            QTest.mouseClick(
                card.image_field,
                Qt.MouseButton.LeftButton,
                pos=QPoint(4, card.image_field.height() // 2),
            )
        chooser.assert_called_once()
        self.assertEqual(self.canvas.stack.count(), 0)
        self.assertEqual(
            item.blk.fontformat.text_effects[0].asset, old_asset
        )

        project.import_raster_asset.side_effect = ValueError('bad image')

        def choose_while_selection_clears(*_args):
            self.assertTrue(self.panel.focusOnColorDialog)
            self.panel.set_textblk_item(None)
            self.assertIs(self.panel.textblk_item, item)
            return '/tmp/broken.png', 'Images'

        def warning_while_pinned(*_args):
            self.assertTrue(self.panel.focusOnColorDialog)
            self.assertIs(self.panel.textblk_item, item)

        with patch.object(
            QFileDialog,
            'getOpenFileName',
            side_effect=choose_while_selection_clears,
        ), patch.object(
            QMessageBox, 'warning', side_effect=warning_while_pinned
        ):
            card.image_button.click()

        self.assertFalse(self.panel.focusOnColorDialog)
        self.assertEqual(self.canvas.stack.count(), 0)
        self.assertEqual(
            item.blk.fontformat.text_effects[0].asset, old_asset
        )

    def test_image_missing_recovery_and_same_digest_refresh_without_undo(self):
        asset = RasterAssetRef(
            'assets/' + 'b' * 64 + '.png', 'restored.png'
        )
        project = Mock()
        project.resolve_raster_asset.return_value = None
        project.import_raster_asset.return_value = asset
        project.load_raster_asset.return_value = np.full(
            (2, 2, 4), (20, 60, 230, 255), dtype=np.uint8
        )
        self.canvas.imgtrans_proj = project
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        item = self._item(self._stack(ImageEffect(asset)))
        scene.addItem(item)
        self.panel.set_textblk_item(item)
        controls = self.panel.texteffect_panel
        card = controls.image_cards[0]
        self.assertEqual(
            card.image_field.text(), 'Missing: restored.png'
        )
        self.assertEqual(
            card.image_field.accessibleName(), 'Missing: restored.png'
        )

        project.resolve_raster_asset.return_value = (
            '/project/assets/restored.png'
        )
        controls.project_assets_changed()
        self.assertEqual(card.image_field.text(), 'restored.png')

        with patch.object(
            controls,
            'project_assets_changed',
            wraps=controls.project_assets_changed,
        ) as refresh:
            with patch.object(
                QFileDialog,
                'getOpenFileName',
                return_value=('/tmp/restored.png', 'Images'),
            ):
                controls._choose_image_file(card.index)
        self.assertEqual(self.canvas.stack.count(), 0)
        self.assertEqual(refresh.call_count, 1)
        scene.removeItem(item)

    def test_text_fill_texture_import_mapping_preview_cancel_and_commit(self):
        asset = RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'paper.png'
        )
        project = Mock()
        project.import_raster_asset.return_value = asset
        self.canvas.imgtrans_proj = project
        item = self._item(self._stack(TextFillEffect(
            paint=TexturePaint()
        )))
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.text_fill_cards[0]

        self.assertEqual(card.title_label.text(), 'Texture')
        self.assertEqual(
            item.blk.fontformat.text_effects[0].paint, TexturePaint()
        )
        self.assertEqual(card.texture_field.text(), '')
        self.assertEqual(card.texture_button.text(), '')
        self.assertFalse(card.texture_button.icon().isNull())
        self.assertIs(card.texture_button.parent(), card.texture_field)
        self.assertEqual(self.canvas.stack.count(), 0)

        with patch.object(
            QFileDialog, 'getOpenFileName', return_value=('', '')
        ) as chooser:
            QTest.mouseClick(
                card.texture_field,
                Qt.MouseButton.LeftButton,
                pos=QPoint(4, card.texture_field.height() // 2),
            )
        chooser.assert_called_once()

        project.import_raster_asset.side_effect = ValueError('not an image')
        with patch(
            'ballontranslator.ui.text_engine.effects.panel.'
            'QMessageBox.warning'
        ) as warning:
            card.texture_file_requested.emit(card.index, '/tmp/broken.png')
        warning.assert_called_once()
        self.assertEqual(
            item.blk.fontformat.text_effects[0].paint, TexturePaint()
        )
        self.assertEqual(card.texture_field.text(), '')
        self.assertEqual(self.canvas.stack.count(), 0)

        project.import_raster_asset.side_effect = None
        project.import_raster_asset.reset_mock()
        card.texture_file_requested.emit(card.index, '/tmp/paper.png')

        paint = item.blk.fontformat.text_effects[0].paint
        self.assertEqual(paint, TexturePaint(asset))
        project.import_raster_asset.assert_called_once_with('/tmp/paper.png')
        self.assertEqual(self.canvas.stack.count(), 1)
        card.texture_mapping_selector.setCurrentIndex(
            card.texture_mapping_selector.findData('tile')
        )
        self.assertEqual(
            item.blk.fontformat.text_effects[0].paint.mapping, 'tile'
        )
        self.assertEqual(self.canvas.stack.count(), 2)

        editor = card.texture_scale_control.editor
        editor.setText('150.0%')
        editor.textEdited.emit('150.0%')
        self.assertEqual(item.blk.fontformat.text_effects[0].paint.scale, 1.0)
        self.assertEqual(item.effective_text_effects()[0].paint.scale, 1.5)
        QApplication.sendEvent(
            editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(item.effective_text_effects()[0].paint.scale, 1.0)
        editor.setText('150.0%')
        editor.textEdited.emit('150.0%')
        editor.returnPressed.emit()
        self.assertEqual(item.blk.fontformat.text_effects[0].paint.scale, 1.5)
        self.assertEqual(self.canvas.stack.count(), 3)

        project.import_raster_asset.side_effect = ValueError('not an image')
        with patch(
            'ballontranslator.ui.text_engine.effects.panel.'
            'QMessageBox.warning'
        ) as warning:
            card.texture_file_requested.emit(card.index, '/tmp/broken.png')
        warning.assert_called_once()
        self.assertEqual(self.canvas.stack.count(), 3)

    def test_text_fill_texture_errors_route_by_current_card_index(self):
        project = Mock()
        project.import_raster_asset.side_effect = ValueError('bad image')
        self.canvas.imgtrans_proj = project
        item = self._item(self._stack(
            TextFillEffect(paint=TexturePaint()),
            StrokeEffect(),
            TextFillEffect(paint=TexturePaint()),
        ))
        self.panel.set_textblk_item(item)
        controls = self.panel.texteffect_panel
        self.assertEqual(
            [card.index for card in controls.text_fill_cards], [2, 0]
        )

        with patch.object(QMessageBox, 'warning') as warning:
            controls.show_texture_import_error(1, 'stale card')
            warning.assert_not_called()
            for card in controls.text_fill_cards:
                card.texture_file_requested.emit(
                    card.index, '/tmp/broken.png'
                )
        self.assertEqual(warning.call_count, 2)
        self.assertEqual(self.canvas.stack.count(), 0)

    def test_text_fill_texture_is_project_item_only(self):
        global_stack = self._stack(TextFillEffect())
        self.panel.global_format.text_effects = global_stack
        self.panel.set_active_format(self.panel.global_format)
        card = self.panel.texteffect_panel.text_fill_cards[0]
        self.assertEqual(card.title_label.text(), 'Gradient')
        self.assertTrue(
            self.panel.texteffect_panel.add_effect_actions['gradient']
            .isEnabled()
        )
        self.assertFalse(
            self.panel.texteffect_panel.add_effect_actions['texture']
            .isEnabled()
        )

        project = Mock()
        self.canvas.imgtrans_proj = project
        self.assertFalse(
            self.panel.text_effect_session.import_texture(
                card.index, '/tmp/never-imported.png'
            )
        )
        project.import_raster_asset.assert_not_called()
        self.assertEqual(
            self.panel.global_format.text_effects, global_stack
        )

    def test_displaying_global_format_does_not_sanitize_its_model(self):
        asset = RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'paper.png'
        )
        stack = self._stack(
            StrokeEffect(width=0.25),
            TextFillEffect(paint=TexturePaint(asset)),
            GlowEffect(size=0.2),
            FilterEffect('builtin:noise', params={
                'amount': 0.4, 'mode': 'color', 'seed': 3,
            }),
        )
        arbitrary_format = FontFormat(text_effects=stack)

        self.panel.texteffect_panel.set_active_format(arbitrary_format)

        self.assertEqual(arbitrary_format.text_effects, stack)
        self.assertEqual(self.panel.texteffect_panel.text_fill_cards, [])

    def test_active_item_texture_is_omitted_when_updating_portable_preset(self):
        asset = RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'paper.png'
        )
        stack = self._stack(
            StrokeEffect(width=0.25),
            TextFillEffect(paint=TexturePaint(asset)),
            GlowEffect(size=0.2),
            FilterEffect('builtin:noise', params={
                'amount': 0.4, 'mode': 'color', 'seed': 3,
            }),
        )
        item = self._item(stack)
        self.panel.set_textblk_item(item)
        C.active_format = item.get_fontformat()
        preset = TextStyleLabel(fontfmt=FontFormat())
        try:
            with patch(
                'ballontranslator.ui.text_engine.formatting.presets.'
                'save_text_styles'
            ) as save:
                preset.update_style()
            save.assert_called_once_with()
            self.assertEqual(
                preset.fontfmt.text_effects.effects,
                (
                    StrokeEffect(width=0.25),
                    GlowEffect(size=0.2),
                    FilterEffect('builtin:noise', params={
                        'amount': 0.4, 'mode': 'color', 'seed': 3,
                    }),
                ),
            )
            self.assertEqual(item.blk.fontformat.text_effects, stack)
        finally:
            preset.deleteLater()

    def test_text_fill_mixed_texture_fields_and_asset_only_unification(self):
        first_asset = RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'paper.png'
        )
        second_asset = RasterAssetRef(
            'assets/' + 'b' * 64 + '.png', 'cloth.png'
        )
        unified_asset = RasterAssetRef(
            'assets/' + 'c' * 64 + '.png', 'shared.png'
        )
        first = self._item(self._stack(TextFillEffect(
            paint=TexturePaint(first_asset, mapping='tile', scale=1.5)
        )))
        second = self._item(self._stack(TextFillEffect(
            paint=TexturePaint(first_asset, mapping='fit', scale=0.75)
        )))
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        card = self.panel.texteffect_panel.text_fill_cards[0]

        self.assertIn('paper.png', card.texture_field.text())
        self.assertTrue(card.texture_button.isEnabled())
        self.assertEqual(card.texture_mapping_selector.currentData(), 'fit')
        self.assertTrue(card.texture_scale_control.isHidden())
        self.assertTrue(card.property('matched'))

        second_stack = self._stack(TextFillEffect(
            paint=TexturePaint(second_asset, mapping='fit', scale=0.75)
        ))
        second.blk.fontformat.text_effects = second_stack
        second.fontformat.text_effects = second_stack
        self.panel.set_textblk_item(None, multi_select=True)
        card = self.panel.texteffect_panel.text_fill_cards[0]
        self.assertIn('cloth.png', card.texture_field.text())
        self.assertTrue(card.texture_button.isEnabled())

        project = Mock()
        project.import_raster_asset.return_value = unified_asset
        self.canvas.imgtrans_proj = project
        card.texture_file_requested.emit(card.index, '/tmp/shared.png')
        first_paint = first.blk.fontformat.text_effects[0].paint
        second_paint = second.blk.fontformat.text_effects[0].paint
        self.assertEqual(first_paint.asset, unified_asset)
        self.assertEqual(second_paint.asset, unified_asset)
        self.assertEqual((first_paint.mapping, first_paint.scale), ('tile', 1.5))
        self.assertEqual((second_paint.mapping, second_paint.scale), ('fit', 0.75))
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_text_fill_file_dialog_pins_panel_through_cancel_and_error(self):
        item = self._item(self._stack(TextFillEffect(
            paint=TexturePaint()
        )))
        self.panel.set_textblk_item(item)
        card = self.panel.texteffect_panel.text_fill_cards[0]
        transitions = []
        card.color_dialog_active_changed.connect(transitions.append)

        with patch.object(
            QFileDialog, 'getOpenFileName', return_value=('', '')
        ):
            self.assertFalse(card._choose_texture_file())
        self.assertEqual(transitions, [True, False])
        self.assertFalse(self.panel.focusOnColorDialog)

        project = Mock()
        project.import_raster_asset.side_effect = ValueError('bad image')
        self.canvas.imgtrans_proj = project

        def assert_pinned(*_args) -> None:
            self.assertTrue(self.panel.focusOnColorDialog)

        with patch.object(
            QFileDialog,
            'getOpenFileName',
            return_value=('/tmp/broken.png', 'Images'),
        ), patch.object(
            QMessageBox, 'warning', side_effect=assert_pinned
        ):
            self.assertTrue(card._choose_texture_file())
        self.assertEqual(transitions, [True, False, True, False])
        self.assertFalse(self.panel.focusOnColorDialog)
        self.assertEqual(self.canvas.stack.count(), 0)

    def test_foreground_paint_cards_are_fixed_and_texture_scale_is_conditional(self):
        asset = RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'paper.png'
        )
        self.canvas.imgtrans_proj = Mock()
        item = self._item(self._stack(
            TextFillEffect(paint=LinearGradientPaint()),
            TextFillEffect(paint=TexturePaint(asset, mapping='fit')),
        ))
        self.panel.set_textblk_item(item)
        cards = {
            card.title_label.text(): card
            for card in self.panel.texteffect_panel.text_fill_cards
        }
        self.assertEqual(set(cards), {'Gradient', 'Texture'})
        self.assertIsNotNone(cards['Gradient'].gradient_editor)
        self.assertIsNone(cards['Gradient'].texture_field)
        self.assertIsNone(cards['Gradient'].texture_button)
        self.assertIsNotNone(cards['Texture'].texture_field)
        self.assertIsNotNone(cards['Texture'].texture_button)
        self.assertTrue(cards['Texture'].texture_scale_control.isHidden())

        with self.assertRaises(ValueError):
            TextFillEffectCard(3, 'solid')

        cards['Texture'].set_values([TextFillEffect(
            paint=TexturePaint(asset, mapping='tile')
        )])
        self.assertFalse(cards['Texture'].texture_scale_control.isHidden())

    def test_gradient_angle_dial_previews_then_commits_once(self):
        before = self._stack(self._constant_text_fill())
        item = self._item(before)
        self.panel.set_textblk_item(item)
        gradient = (
            self.panel.texteffect_panel.text_fill_cards[0].gradient_editor
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

    def test_effect_reorder_is_global_and_mixed_shadow_type_stays_editable(self):
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

        self.assertTrue(cards[1].move_down_button.isEnabled())
        cards[1].move_down_button.click()
        effects = first.blk.fontformat.text_effects.effects
        self.assertEqual(effects[0].paint, SolidPaint((255, 0, 0)))
        self.assertEqual(effects[1].shadow_type, 'inner')
        self.assertIsInstance(effects[2], StrokeEffect)
        self.assertEqual(effects[3].paint, SolidPaint((0, 0, 255)))
        self.assertEqual(self.canvas.stack.count(), 1)

        second = self._item(self._stack(
            ShadowEffect(shadow_type='inner'),
            ShadowEffect(shadow_type='inner'),
            StrokeEffect(width=0.3),
            ShadowEffect(),
        ))
        self.canvas.selected = [first, second]
        self.panel.set_textblk_item(None, multi_select=True)
        mixed_card = self.panel.texteffect_panel.shadow_cards[2]
        self.assertEqual(mixed_card.type_selector.currentData(), 'inner')
        self.assertTrue(mixed_card.property('matched'))
        self.assertTrue(mixed_card.move_up_button.isEnabled())
        self.assertFalse(mixed_card.move_down_button.isEnabled())

    def test_shutdown_stops_generation_but_ordinary_save_does_not(self):
        session = self.panel.text_effect_session
        with patch.object(session, 'stop_image_generation') as stop:
            self.panel.resolve_text_transform_edits_for_save()
            stop.assert_not_called()

            self.panel.stop_text_effect_generation_for_shutdown()

        stop.assert_called_once_with(detach_card=True)

    def test_whole_format_application_detaches_blocked_generation(self):
        asset = RasterAssetRef(
            'assets/' + '6' * 64 + '.png', 'old.png'
        )
        recipe = ImageGenerationRecipe(
            profile_id='artist', model='image-v2', prompt='Texture'
        )
        effect = ImageEffect(asset, generation=recipe)
        project = Mock()
        project.load_identity = object()
        project.current_img = 'page.png'
        self.canvas.imgtrans_proj = project
        scene = QGraphicsScene()
        scene.imgtrans_proj = project
        item = self._item(self._stack(effect))
        scene.addItem(item)
        self.canvas.selected = [item]
        self.panel.set_textblk_item(item)
        session = self.panel.text_effect_session
        session._pending_image_generation = (
            item,
            0,
            effect,
            project,
            project.load_identity,
            project.current_img,
            recipe,
        )
        backend = _BlockedImageBackend()
        controller = session._image_generation_controller
        self.assertTrue(controller.start(
            0,
            backend,
            ImageGenerationRequest(recipe, None),
        ))
        self.assertTrue(backend.started.wait(1.0))
        editor = Mock()
        manager = SimpleNamespace(
            canvas=self.canvas,
            formatpanel=self.panel,
            pairwidget_list=[None, SimpleNamespace(e_trans=editor)],
        )
        replacement = FontFormat(text_effects=self._stack(
            StrokeEffect(width=0.4),
            ImageEffect(asset, generation=recipe),
        ))
        try:
            SceneTextManager.apply_fontformat(manager, replacement)
            self.assertTrue(controller.active)
            self.assertEqual(
                self.panel.texteffect_panel._image_generation_state,
                'stopping',
            )
            self.assertEqual(
                item.blk.fontformat.text_effects,
                replacement.text_effects,
            )
        finally:
            backend.release.set()
        self.assertTrue(self._wait_until(lambda: not controller.active))
        project.import_raster_asset_bytes.assert_not_called()
        self.assertEqual(self.canvas.stack.count(), 1)
        self.assertEqual(
            item.blk.fontformat.text_effects,
            replacement.text_effects,
        )

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
        item = self._item(self._stack(self._constant_text_fill()))
        self.panel.set_textblk_item(item)
        editor = self.panel.texteffect_panel.text_fill_cards[0].gradient_editor
        editor.angle_editor.setValue(75.0)
        self.assertEqual(item.blk.fontformat.text_effects[0].paint.angle, 0.0)

        self.panel.resolve_text_transform_edits_for_page_change()

        self.assertEqual(item.blk.fontformat.text_effects[0].paint.angle, 75.0)
        self.assertEqual(self.canvas.stack.count(), 1)

    def test_history_change_cancels_pending_inline_gradient(self):
        before = self._stack(self._constant_text_fill())
        item = self._item(before)
        self.panel.set_textblk_item(item)
        editor = self.panel.texteffect_panel.text_fill_cards[0].gradient_editor
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
