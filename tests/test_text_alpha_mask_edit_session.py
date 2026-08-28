import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, QPointF, QRectF, Qt
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsRectItem,
)

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.editing.commands import (
    SetTextAlphaMaskCommand,
)
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.shape_control import (
    CONTROL_ITEM_DATA_KEY,
)
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.fontformat import (
    FontFormat,
    ProjectiveTextTransform,
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.text_alpha_mask import (
    AlphaBrushStroke,
    TextAlphaMask,
)
from ballontranslator.utils.text_effects import (
    ShadowEffect,
    StrokeEffect,
    TextEffectStack,
)
from ballontranslator.utils.textblock import TextBlock


class _MouseEvent:
    def __init__(self, button, position) -> None:
        self._button = button
        self._position = QPointF(position)

    def button(self):
        return self._button

    def scenePos(self):
        return QPointF(self._position)


class TextAlphaMaskEditSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_canvas = getattr(SW, 'canvas', None)
        self.canvas = Canvas()
        SW.canvas = self.canvas
        self.canvas.imgtrans_proj = SimpleNamespace(img_valid=True)
        self.canvas.editor_index = 1
        block = TextBlock([0, 0, 320, 180])
        block._bounding_rect = [0, 0, 320, 180]
        block.translation = 'Alpha mask editing'
        self.item = TextBlkItem(block, 1)
        self.canvas.attach_text_item(self.item)
        self.item.setSelected(True)
        self.session = self.canvas.alpha_mask_edit_session

    def tearDown(self) -> None:
        self.session.deactivate()
        try:
            if self.item.scene() is self.canvas:
                self.canvas.removeItem(self.item)
        except RuntimeError:
            pass
        self.canvas.deleteLater()
        SW.canvas = self.old_canvas
        self.app.processEvents()

    def _scene_point(self, x: float, y: float) -> QPointF:
        origin = self.item.logical_unpadded_rect().topLeft()
        return self.item.geometry_controller.map_source_to_scene(
            QPointF(origin.x() + x, origin.y() + y)
        )

    def _event(self, button, x: float, y: float) -> _MouseEvent:
        return _MouseEvent(button, self._scene_point(x, y))

    def _activate(self) -> None:
        self.assertTrue(self.session.activate(self.item))

    def test_first_activation_inserts_once_and_reentry_is_noop(self):
        self.assertIsNone(self.item.blk.text_alpha_mask)
        self._activate()
        self.assertEqual(self.item.blk.text_alpha_mask, TextAlphaMask())
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self._activate()
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)

        self.canvas.text_undo_stack.undo()
        self.assertIsNone(self.item.blk.text_alpha_mask)
        self.assertFalse(self.session.active)
        self.canvas.text_undo_stack.redo()
        self.assertEqual(self.item.blk.text_alpha_mask, TextAlphaMask())

    def test_live_dot_preview_commit_cancel_and_document_isolation(self):
        self._activate()
        document = self.item.document()
        revision = document.revision()
        html = document.toHtml()
        undo_steps = document.availableUndoSteps()

        press = self._event(Qt.MouseButton.LeftButton, -8.0, 24.0)
        self.assertTrue(self.session.handle_mouse_press(press))
        self.assertEqual(self.item.blk.text_alpha_mask.strokes, ())
        self.assertEqual(
            len(self.item.effective_text_alpha_mask().strokes), 1
        )
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.assertEqual(document.revision(), revision)
        self.assertEqual(document.toHtml(), html)
        self.assertEqual(document.availableUndoSteps(), undo_steps)

        self.assertTrue(self.session.handle_mouse_release(press))
        self.assertEqual(len(self.item.blk.text_alpha_mask.strokes), 1)
        self.assertEqual(self.canvas.text_undo_stack.count(), 2)
        self.assertLess(self.item.blk.text_alpha_mask.strokes[0].points[0][0], 0)
        self.assertEqual(document.revision(), revision)
        self.assertEqual(document.toHtml(), html)
        self.assertEqual(document.availableUndoSteps(), undo_steps)

        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.blk.text_alpha_mask.strokes, ())
        self.canvas.text_undo_stack.redo()
        self.assertEqual(len(self.item.blk.text_alpha_mask.strokes), 1)

        second = self._event(Qt.MouseButton.LeftButton, 40.0, 40.0)
        self.assertTrue(self.session.handle_mouse_press(second))
        count = self.canvas.text_undo_stack.count()
        self.assertTrue(self.session.handle_escape())
        self.assertEqual(self.canvas.text_undo_stack.count(), count)
        self.assertEqual(len(self.item.blk.text_alpha_mask.strokes), 1)
        self.assertFalse(self.item.effect_renderer.has_text_alpha_mask_preview())

    def test_release_promotes_complete_scratch_and_export_uses_canonical(self):
        self._activate()
        event = self._event(Qt.MouseButton.LeftButton, 42.0, 36.0)
        self.assertTrue(self.session.handle_mouse_press(event))
        renderer = self.item.effect_renderer
        scratch = renderer._preview_effect_raster_state
        self.assertIsNotNone(scratch)
        self.assertIsNotNone(scratch.background_pixmap)
        preview_mask = self.item.effective_text_alpha_mask()

        renderer.set_export_effect_render(True)
        try:
            self.assertEqual(
                renderer.effective_text_alpha_mask(), TextAlphaMask()
            )
        finally:
            renderer.set_export_effect_render(False)

        self.assertTrue(self.session.handle_mouse_release(event))
        self.assertEqual(self.item.blk.text_alpha_mask, preview_mask)
        self.assertIs(renderer._effect_raster_state, scratch)
        self.assertIsNone(renderer._preview_effect_raster_state)

    def test_pointer_samples_reuse_stroke_shadow_pre_mask_composite(self):
        self.item.set_text_effects(TextEffectStack(effects=(
            ShadowEffect(blur=0.08, angle=32.0, distance=0.19),
            StrokeEffect(width=0.14),
        )))
        self._activate()
        renderer = self.item.effect_renderer
        with patch.object(
            renderer,
            '_render_pre_mask_effect_surface',
            wraps=renderer._render_pre_mask_effect_surface,
        ) as upstream:
            self.assertTrue(self.session.handle_mouse_press(
                self._event(Qt.MouseButton.LeftButton, 20, 30)
            ))
            self.assertEqual(upstream.call_count, 0)
            for offset in range(1, 11):
                self.assertTrue(self.session.handle_mouse_move(
                    self._event(
                        Qt.MouseButton.NoButton,
                        20 + offset * 5,
                        30 + offset * 3,
                    )
                ))
            self.assertEqual(upstream.call_count, 0)

    def test_commit_rebuilds_when_preview_pixels_are_not_semantically_current(self):
        self._activate()
        event = self._event(Qt.MouseButton.LeftButton, 50.0, 44.0)
        self.assertTrue(self.session.handle_mouse_press(event))
        renderer = self.item.effect_renderer
        scratch = renderer._preview_effect_raster_state
        self.assertIsNotNone(scratch)
        scratch.cache_input_key = ('stale',)

        self.assertTrue(self.session.handle_mouse_release(event))
        self.assertIsNot(renderer._effect_raster_state, scratch)
        self.assertIsNotNone(renderer._effect_raster_state.background_pixmap)

    def test_preview_cancel_restores_canonical_cache_and_rebuilds_after_text_change(self):
        self.item.set_text_alpha_mask(TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 8, ((20, 20),)),
        )))
        renderer = self.item.effect_renderer
        renderer.repaint_background()
        committed = renderer._effect_raster_state
        committed_pixmap = committed.background_pixmap
        preview = TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 16, ((30, 30),)),
        ))
        self.item.set_text_alpha_mask(preview, preview=True)
        self.assertIsNot(renderer._preview_effect_raster_state, committed)
        self.assertTrue(self.item.clear_text_alpha_mask_preview())
        self.assertIs(renderer._effect_raster_state, committed)
        self.assertEqual(
            renderer.background_pixmap.cacheKey(), committed_pixmap.cacheKey()
        )

        self.item.set_text_alpha_mask(preview, preview=True)
        with patch.object(
            renderer,
            '_render_effect_surface',
            wraps=renderer._render_effect_surface,
        ) as render:
            self.item.setPlainText('Changed while mask preview is live')
            calls_before_cancel = render.call_count
            self.assertTrue(self.item.clear_text_alpha_mask_preview())
            self.assertGreater(render.call_count, calls_before_cancel)
        self.assertIsNot(
            renderer.background_pixmap, committed_pixmap
        )

    def test_clear_remove_enabled_and_settings_have_expected_undo_scope(self):
        self.item.set_text_alpha_mask(TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 10, ((10, 10),)),
        )))
        count = self.canvas.text_undo_stack.count()
        self.session.set_mode('restore')
        self.session.set_diameter(31.5)
        self.assertEqual(self.canvas.text_undo_stack.count(), count)

        self.session.set_enabled(False)
        self.assertFalse(self.item.blk.text_alpha_mask.enabled)
        self.assertEqual(self.canvas.text_undo_stack.count(), count + 1)
        self.assertFalse(self.session.active)
        self.canvas.text_undo_stack.undo()
        self.assertTrue(self.item.blk.text_alpha_mask.enabled)

        self.session.clear_mask()
        self.assertEqual(self.item.blk.text_alpha_mask.strokes, ())
        self.assertEqual(self.canvas.text_undo_stack.count(), count + 1)
        self.session.clear_mask()
        self.assertEqual(self.canvas.text_undo_stack.count(), count + 1)

        self.session.remove_mask()
        self.assertIsNone(self.item.blk.text_alpha_mask)
        self.assertEqual(self.canvas.text_undo_stack.count(), count + 2)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.blk.text_alpha_mask, TextAlphaMask())

    def test_rotation_projective_and_nonlinear_mapping_store_local_points(self):
        self._activate()
        transforms = (
            None,
            TextTransformStack((
                ProjectiveTextTransform(
                    horizontal_scale=1.1,
                    vertical_scale=0.9,
                    rotation_z=12.0,
                ),
            )),
            TextTransformStack((SineTextTransform(amplitude_x=0.12),)),
        )
        expected = (-7.0, 28.0)
        for index, transform in enumerate(transforms):
            with self.subTest(index=index):
                self.item.setRotation(19.0 if transform is None else 0.0)
                if transform is not None:
                    self.item.set_text_transform(transform)
                self.item.set_text_alpha_mask(TextAlphaMask())
                event = self._event(
                    Qt.MouseButton.LeftButton, expected[0], expected[1]
                )
                self.assertTrue(self.session.handle_mouse_press(event))
                self.assertTrue(self.session.handle_mouse_release(event))
                point = self.item.blk.text_alpha_mask.strokes[-1].points[0]
                self.assertAlmostEqual(point[0], expected[0], places=4)
                self.assertAlmostEqual(point[1], expected[1], places=4)

        with patch.object(
            self.item.geometry_controller,
            'capture_scene_to_source_mapper',
            return_value=lambda *_args: QPointF(float('nan'), 0),
        ):
            self.assertFalse(self.session.handle_mouse_press(
                self._event(Qt.MouseButton.LeftButton, 10, 10)
            ))
        self.assertFalse(self.item.effect_renderer.has_text_alpha_mask_preview())

    def test_one_stroke_keeps_its_frozen_mapping_across_geometry_change(self):
        self._activate()
        start = self._scene_point(12.0, 18.0)
        end = self._scene_point(72.0, 48.0)
        self.assertTrue(self.session.handle_mouse_press(
            _MouseEvent(Qt.MouseButton.LeftButton, start)
        ))
        self.item.setRotation(37.0)
        self.assertTrue(self.session.handle_mouse_move(
            _MouseEvent(Qt.MouseButton.NoButton, end)
        ))
        self.assertTrue(self.session.handle_mouse_release(
            _MouseEvent(Qt.MouseButton.LeftButton, end)
        ))
        points = self.item.blk.text_alpha_mask.strokes[-1].points
        self.assertAlmostEqual(points[0][0], 12.0, places=4)
        self.assertAlmostEqual(points[0][1], 18.0, places=4)
        self.assertAlmostEqual(points[-1][0], 72.0, places=4)
        self.assertAlmostEqual(points[-1][1], 48.0, places=4)

    def test_right_middle_controls_and_other_text_keep_input_precedence(self):
        self._activate()
        right = self._event(Qt.MouseButton.RightButton, 30, 30)
        middle = self._event(Qt.MouseButton.MiddleButton, 30, 30)
        self.assertFalse(self.session.handle_mouse_press(right))
        self.assertFalse(self.session.handle_mouse_press(middle))
        self.assertTrue(self.session.active)

        point = self._scene_point(30, 30)
        control = QGraphicsRectItem(QRectF(point.x() - 3, point.y() - 3, 6, 6))
        control.setData(CONTROL_ITEM_DATA_KEY, True)
        control.setZValue(20000)
        self.canvas.addItem(control)
        self.assertFalse(self.session.handle_mouse_press(
            _MouseEvent(Qt.MouseButton.LeftButton, point)
        ))
        self.assertTrue(self.session.active)
        self.canvas.removeItem(control)

        other_block = TextBlock([0, 0, 320, 180])
        other_block._bounding_rect = [0, 0, 320, 180]
        other_block.translation = 'Other'
        other = TextBlkItem(other_block, 2)
        self.canvas.attach_text_item(other)
        other.setZValue(self.item.zValue() + 10)
        try:
            self.assertFalse(self.session.handle_mouse_press(
                _MouseEvent(Qt.MouseButton.LeftButton, point)
            ))
            self.assertFalse(self.session.active)
            self.assertEqual(self.item.blk.text_alpha_mask.strokes, ())
        finally:
            self.canvas.removeItem(other)

    def test_real_shape_frame_body_allows_brush_but_handle_keeps_precedence(self):
        self.canvas.txtblkShapeControl.setBlkItem(self.item)
        self.assertTrue(self.canvas.txtblkShapeControl.isVisible())
        self._activate()
        body = self._event(Qt.MouseButton.LeftButton, 160, 90)
        self.assertTrue(self.session.handle_mouse_press(body))
        self.assertTrue(self.session.cancel_active_stroke())

        handle = self.canvas.txtblkShapeControl.ctrlblock_group[0]
        handle_point = handle.mapToScene(handle.visible_rect.center())
        self.assertIs(self.session._top_input_owner(handle_point), handle)
        self.assertFalse(self.session.handle_mouse_press(
            _MouseEvent(Qt.MouseButton.LeftButton, handle_point)
        ))
        self.assertFalse(self.session._drawing)

    def test_selection_remove_hide_and_modal_tool_release_target_overlay(self):
        self._activate()
        self.assertIsNotNone(self.session._cursor_item)
        self.assertTrue(bool(
            self.session._cursor_item.data(CONTROL_ITEM_DATA_KEY)
        ))
        self.item.setSelected(False)
        self.assertFalse(self.session.active)
        self.assertIsNone(self.session._cursor_item)

        self.item.setSelected(True)
        self._activate()
        self.canvas.clearToolStates()
        self.assertFalse(self.session.active)

        self._activate()
        self.canvas.on_hide_canvas()
        self.assertFalse(self.session.active)

        self._activate()
        self.session.resolve_for_page_change()
        self.assertFalse(self.session.active)

        self._activate()
        self.canvas.removeItem(self.item)
        self.assertFalse(self.session.active)
        self.assertIsNone(self.session.target)

    def test_deferred_target_delete_releases_session_and_control(self):
        self._activate()
        cursor = self.session._cursor_item
        self.item.deleteLater()
        QCoreApplication.sendPostedEvents(
            None, QEvent.Type.DeferredDelete
        )
        self.app.processEvents()
        self.assertFalse(self.session.active)
        self.assertIsNone(self.session.target)
        self.assertIsNone(self.session._cursor_item)
        self.assertIsNone(cursor.scene())

    def test_command_factory_and_panel_gating_pinned_card(self):
        self.assertIsNone(SetTextAlphaMaskCommand.create(
            self.item, None, None
        ))
        old_active_format = C.active_format
        registered = []
        with patch.object(
            shared,
            'register_view_widget',
            side_effect=registered.append,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        try:
            panel.global_format = FontFormat()
            panel.set_active_format(panel.global_format)
            controls = panel.texteffect_panel
            self.assertFalse(controls.mask_brush_button.isEnabled())
            self.assertIsNone(controls.alpha_mask_card)

            with patch.object(
                controls,
                '_set_effect_states',
                wraps=controls._set_effect_states,
            ) as effect_refresh:
                panel.set_textblk_item(self.item)
            self.assertEqual(effect_refresh.call_count, 1)
            self.assertTrue(controls.mask_brush_button.isEnabled())
            controls.mask_brush_button.click()
            self.assertTrue(self.session.active)
            self.assertIsNotNone(controls.alpha_mask_card)
            mask_card = controls.alpha_mask_card
            self.assertFalse(mask_card.title_icon_label.pixmap().isNull())
            self.assertEqual(mask_card.title_label.text(), 'Eraser')
            self.assertEqual(
                controls.mask_brush_button.accessibleName(), 'Text Eraser'
            )
            self.assertEqual(
                controls.mask_brush_button.toolTip(), 'Edit Text Eraser'
            )
            self.assertFalse(mask_card.visibility_button.icon().isNull())
            self.assertEqual(
                mask_card.visibility_button.toolTip(), 'Hide'
            )
            self.assertEqual(
                mask_card.visibility_button.accessibleName(),
                'Hide',
            )
            self.assertEqual(
                mask_card.remove_button.toolTip(), 'Delete'
            )
            count = self.canvas.text_undo_stack.count()
            mask_card.visibility_button.click()
            self.assertFalse(self.item.blk.text_alpha_mask.enabled)
            self.assertEqual(self.canvas.text_undo_stack.count(), count + 1)
            self.assertEqual(
                mask_card.visibility_button.toolTip(), 'Show'
            )
            self.canvas.text_undo_stack.undo()
            self.assertTrue(self.item.blk.text_alpha_mask.enabled)

            count = self.canvas.text_undo_stack.count()
            mask_card.mode_selector.setCurrentIndex(1)
            mask_card.diameter_editor.setValue(48.0)
            self.assertEqual(self.canvas.text_undo_stack.count(), count)
            mask_card.size_label.drag_started.emit()
            mask_card.size_label.size_ctrl_changed.emit(4)
            self.assertEqual(self.session.diameter, 52.0)
            mask_card.size_label.drag_canceled.emit()
            self.assertEqual(self.session.diameter, 48.0)
            self.assertEqual(self.canvas.text_undo_stack.count(), count)

            controls.color_dialog_active_changed.emit(True)
            self.assertFalse(self.session.active)
            self.assertTrue(self.session.activate(self.item))
            controls.value_preview_requested.emit(
                -1, 'overall_opacity', 0.75
            )
            self.assertFalse(self.session.active)
            panel.text_effect_session.cancel_preview()

            for signal, argument in (
                (controls.add_filter_requested, 'builtin:noise'),
                (controls.add_effect_requested, 'image'),
            ):
                self.assertTrue(self.session.activate(self.item))
                signal.emit(argument)
                self.assertFalse(self.session.active)

            second_block = TextBlock([0, 0, 100, 80])
            second_block._bounding_rect = [0, 0, 100, 80]
            second = TextBlkItem(second_block, 2)
            self.canvas.attach_text_item(second)
            second.setSelected(True)
            panel.set_textblk_item(multi_select=True)
            self.assertFalse(controls.mask_brush_button.isEnabled())
            self.assertIsNone(controls.alpha_mask_card)
            self.canvas.removeItem(second)
        finally:
            panel.deleteLater()
            C.active_format = old_active_format
            self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
