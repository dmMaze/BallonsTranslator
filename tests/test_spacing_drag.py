import gc
import os
import unittest
import weakref
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, QObject, QRectF, Qt
from qtpy.QtGui import QColor, QImage, QPainter, QTextCursor
from qtpy.QtTest import QSignalSpy, QTest
from qtpy.QtWidgets import QApplication

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.editing.commands import TextItemEditCommand
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.annotations import letter_spacing_value
from ballontranslator.ui.text_engine.rendering.glyph import (
    GLOBAL_GLYPH_GEOMETRY_CACHE, GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE,
)
from ballontranslator.utils import config as C, shared
from ballontranslator.utils.fontformat import (
    BendTextTransform, FontFormat, LineSpacingType, TextTransformStack,
)
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.text_effects import ShadowEffect, StrokeEffect, TextEffectStack


class _DocumentEdits(QObject):
    def __init__(self, canvas: Canvas, panel: FontFormatPanel) -> None:
        super().__init__(canvas)
        self.canvas, self.panel = canvas, panel

    def push(self, steps: int, formatting: bool) -> None:
        self.canvas.push_undo_command(TextItemEditCommand(self.sender(), None, steps, self.panel))


class SpacingDragTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.canvas = Canvas()
        self.canvas.editor_index = 1
        self.canvas_patch = patch.object(SW, 'canvas', self.canvas)
        self.canvas_patch.start()
        self.addCleanup(self.canvas_patch.stop)
        self.addCleanup(setattr, C, 'active_format', C.active_format)
        with patch.object(shared, 'register_view_widget', lambda *_: None, create=True):
            self.panel = FontFormatPanel(self.app)
        self.panel.global_format = FontFormat()
        self.document_edits = _DocumentEdits(self.canvas, self.panel)
        self.item = self.make_item(0)
        self.panel.set_textblk_item(self.item)
        self.addCleanup(self.dispose)

    def dispose(self) -> None:
        if self.panel is not None:
            self.panel.cancel_text_transform_edits_for_scene_change()
            self.panel.deleteLater()
        self.canvas.gv.deleteLater()
        self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def make_item(self, index: int) -> TextBlkItem:
        block = TextBlock([index * 400, 0, index * 400 + 300, 180])
        block._bounding_rect = [index * 400, 0, 300, 180]
        block.translation = 'One two three\nfour five six'
        block.fontformat.font_size = 20
        item = TextBlkItem(block, index)
        item.setParentItem(self.canvas.textLayer)
        item.setSelected(True)
        item.push_undo_stack.connect(self.document_edits.push)
        return item

    def drag(self, name: str, *deltas: int, release: bool = True) -> None:
        label = getattr(self.panel, name + 'SpacingLabel')
        QTest.mousePress(label, Qt.MouseButton.LeftButton)
        for delta in deltas:
            label.size_ctrl_changed.emit(delta)
        if release:
            QTest.mouseRelease(label, Qt.MouseButton.LeftButton)

    def test_preview_preserves_committed_format_and_document_history(self) -> None:
        document = self.item.document()
        before = self.item.toHtml()
        spacing = self.item.fontformat.letter_spacing
        steps = document.availableUndoSteps()
        self.drag('letter', 10, 20, release=False)
        self.assertEqual(document.availableUndoSteps(), steps)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.item.fontformat.letter_spacing, spacing)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_one_drag_restores_defaults_and_text_on_undo(self) -> None:
        for name in ('letter', 'line'):
            with self.subTest(name=name):
                param = name + '_spacing'
                before = self.item.toHtml()
                value = getattr(self.item.fontformat, param)
                self.drag(name, 10, 20)
                self.assertEqual(self.canvas.text_undo_stack.count(), 1)
                self.assertAlmostEqual(getattr(self.item.fontformat, param), value + .3)
                after = self.item.toHtml()
                self.canvas.text_undo_stack.undo()
                self.assertEqual(self.item.toHtml(), before)
                self.assertEqual(getattr(self.item.fontformat, param), value)
                self.canvas.text_undo_stack.redo()
                self.assertEqual(self.item.toHtml(), after)
                self.canvas.text_undo_stack.undo()
                self.canvas.text_undo_stack.clear()

    def test_save_cancels_drag_and_late_release(self) -> None:
        before = self.item.toHtml()
        self.drag('line', 40, release=False)
        self.panel.resolve_text_transform_edits_for_save()
        QTest.mouseRelease(self.panel.lineSpacingLabel, Qt.MouseButton.LeftButton)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_return_to_start_preserves_redo_branch(self) -> None:
        before = self.item.toHtml()
        self.panel.on_param_changed('letter_spacing', 1.8)
        after = self.item.toHtml()
        self.canvas.text_undo_stack.undo()
        self.drag('letter', 10, -10)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.canvas.text_undo_stack.index(), 0)
        self.assertTrue(self.canvas.text_undo_stack.canRedo())
        self.canvas.text_undo_stack.redo()
        self.assertEqual(self.item.toHtml(), after)

    def test_return_to_precise_start_does_not_round_saved_spacing(self) -> None:
        self.panel.on_param_changed('letter_spacing', 1.125)
        before = self.item.toHtml()
        count = self.canvas.text_undo_stack.count()
        self.drag('letter', 10, -10)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.item.fontformat.letter_spacing, 1.125)
        self.assertEqual(self.canvas.text_undo_stack.count(), count)

    def test_multi_item_drag_is_one_command(self) -> None:
        other = self.make_item(1)
        self.panel.set_textblk_item(None, multi_select=True)
        before = [item.toHtml() for item in (self.item, other)]
        self.drag('letter', 10, 20)
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.canvas.text_undo_stack.undo()
        self.assertEqual([item.toHtml() for item in (self.item, other)], before)

    def test_escape_and_hide_restore_preview_and_preserve_redo(self) -> None:
        self.panel.on_param_changed('letter_spacing', 1.8)
        after = self.item.toHtml()
        self.canvas.text_undo_stack.undo()
        before = self.item.toHtml()
        for cancel in ('escape', 'hide'):
            with self.subTest(cancel=cancel):
                self.panel.show()
                self.app.processEvents()
                self.drag('letter', 30, release=False)
                if cancel == 'escape':
                    QTest.keyClick(self.panel.letterSpacingLabel, Qt.Key.Key_Escape)
                else:
                    self.panel.hide()
                QTest.mouseRelease(self.panel.letterSpacingLabel, Qt.MouseButton.LeftButton)
                self.assertEqual(self.item.toHtml(), before)
                self.assertFalse(self.panel.letter_spacing_drag.active)
                self.canvas.text_undo_stack.redo()
                self.assertEqual(self.item.toHtml(), after)
                self.canvas.text_undo_stack.undo()

    def test_page_history_and_target_changes_cancel_late_movement(self) -> None:
        before = self.item.toHtml()
        for resolve in (
            self.panel.resolve_text_transform_edits_for_page_change,
            self.panel.resolve_text_transform_edits_for_history_change,
            self.panel.cancel_text_transform_edits_for_scene_change,
        ):
            self.drag('line', 40, release=False)
            resolve()
            self.panel.lineSpacingLabel.size_ctrl_changed.emit(10)
            QTest.mouseRelease(self.panel.lineSpacingLabel, Qt.MouseButton.LeftButton)
            self.assertEqual(self.item.toHtml(), before)
            self.assertEqual(self.canvas.text_undo_stack.count(), 0)
        self.drag('line', 40, release=False)
        other = self.make_item(1)
        other_before = other.toHtml()
        self.panel.set_textblk_item(other)
        QTest.mouseRelease(self.panel.lineSpacingLabel, Qt.MouseButton.LeftButton)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(other.toHtml(), other_before)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_focus_leaving_drag_cancels_preview_and_late_release(self) -> None:
        self.panel.show()
        self.panel.activateWindow()
        self.app.processEvents()
        before = self.item.toHtml()
        self.drag('line', 30, release=False)
        self.assertTrue(self.panel.lineSpacingLabel.hasFocus())
        self.panel.fontsizebox.fcombobox.setFocus()
        self.app.processEvents()
        self.assertFalse(self.panel.line_spacing_drag.active)
        QTest.mouseRelease(self.panel.lineSpacingLabel, Qt.MouseButton.LeftButton)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)
        self.assertEqual(self.item.opacity(), 1)

    def test_global_defaults_are_only_changed_on_release(self) -> None:
        self.item.setSelected(False)
        self.panel.set_textblk_item(None)
        start = self.panel.global_format.line_spacing
        self.drag('line', 20, release=False)
        self.assertEqual(self.panel.global_format.line_spacing, start)
        self.panel.resolve_text_transform_edits_for_save()
        self.assertEqual(self.panel.global_format.line_spacing, start)
        self.drag('line', 20)
        self.assertAlmostEqual(self.panel.global_format.line_spacing, start + .2)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_selected_text_and_distance_paragraph_keep_their_scope(self) -> None:
        self.item.startEdit()
        cursor = self.item.textCursor()
        cursor.setPosition(5)
        cursor.setPosition(0, QTextCursor.MoveMode.KeepAnchor)
        self.item.setTextCursor(cursor)
        before = self.item.toHtml()
        self.drag('letter', 20, -5, 5, release=False)
        preview = self.panel.letter_spacing_drag.previews[0].toHtml()
        QTest.mouseRelease(self.panel.letterSpacingLabel, Qt.MouseButton.LeftButton)
        self.assertEqual(self.item.toHtml(), preview)
        cursor = self.item.textCursor()
        self.assertEqual((cursor.anchor(), cursor.position()), (5, 0))
        self.assertAlmostEqual(self.item.letter_spacing_value(), 1.35)
        cursor.setPosition(9)
        self.assertAlmostEqual(letter_spacing_value(cursor.charFormat()), 1.15)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.toHtml(), before)

        cursor.setPosition(15)
        self.item.setTextCursor(cursor)
        self.item.setLineSpacingType(LineSpacingType.Distance)
        self.item.setLineSpacing(.6)
        self.panel.set_active_format(self.item.get_fontformat())
        before = self.item.toHtml()
        self.drag('line', 3)
        self.assertEqual(self.item.line_spacing_values(), (.9, LineSpacingType.Distance))
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.toHtml(), before)

    def test_caret_and_empty_item_insertion_spacing_undo(self) -> None:
        for empty in (False, True):
            with self.subTest(empty=empty):
                if empty:
                    self.item.setPlainText('')
                    self.canvas.text_undo_stack.clear()
                self.item.startEdit()
                start = self.item.letter_spacing_value()
                before = self.item.toHtml()
                self.drag('letter', 20)
                self.assertAlmostEqual(self.item.letter_spacing_value(), start + .2)
                if not empty:
                    self.assertEqual(self.item.toHtml(), before)
                self.canvas.text_undo_stack.undo()
                self.assertEqual(self.item.letter_spacing_value(), start)
                self.canvas.text_undo_stack.redo()
                self.assertAlmostEqual(self.item.letter_spacing_value(), start + .2)
                self.canvas.text_undo_stack.undo()

    def test_caret_spacing_history_restores_original_insertion_point(self) -> None:
        self.item.startEdit()
        cursor = self.item.textCursor()
        cursor.setPosition(2)
        self.item.setTextCursor(cursor)
        before = self.item.toHtml()
        start = self.item.letter_spacing_value()
        self.drag('letter', 20)

        cursor.setPosition(5)
        cursor.setPosition(9, QTextCursor.MoveMode.KeepAnchor)
        self.item.setTextCursor(cursor)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.textCursor().position(), 2)
        self.assertFalse(self.item.textCursor().hasSelection())
        self.assertEqual(self.item.letter_spacing_value(), start)
        self.canvas.text_undo_stack.redo()
        self.assertAlmostEqual(self.item.letter_spacing_value(), start + .2)
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)

    def test_caret_spacing_history_survives_deleting_its_surrounding_text(self) -> None:
        self.canvas.gv.show()
        self.canvas.gv.activateWindow()
        self.canvas.gv.setFocus()
        self.item.startEdit()
        self.app.processEvents()
        cursor = self.item.textCursor()
        cursor.setPosition(2)
        self.item.setTextCursor(cursor)
        before = self.item.toHtml()
        start = self.item.letter_spacing_value()
        self.drag('letter', 20)

        cursor = self.item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(9, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        self.item.setTextCursor(cursor)
        self.assertEqual(self.canvas.text_undo_stack.count(), 2)
        self.canvas.text_undo_stack.undo()
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.toHtml(), before)
        self.assertEqual(self.item.textCursor().position(), 2)
        self.assertEqual(self.item.letter_spacing_value(), start)
        self.canvas.text_undo_stack.redo()
        self.assertEqual(self.item.textCursor().position(), 2)
        self.assertAlmostEqual(self.item.letter_spacing_value(), start + .2)
        self.assertEqual(self.canvas.text_undo_stack.count(), 2)

    def render_scene(self) -> QImage:
        image = QImage(600, 600, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor('white'))
        painter = QPainter(image)
        self.canvas.render(painter, QRectF(0, 0, 600, 600), QRectF(-100, -100, 600, 600))
        painter.end()
        return image

    def test_live_render_matches_commit_in_both_writing_modes(self) -> None:
        self.item.setSelected(False)
        for vertical in (False, True):
            self.item.setVertical(vertical)
            for name in ('letter', 'line'):
                with self.subTest(vertical=vertical, name=name):
                    before = self.render_scene()
                    self.drag(name, 20, 10, release=False)
                    preview = self.render_scene()
                    self.assertNotEqual(preview, before)
                    QTest.mouseRelease(getattr(self.panel, name + 'SpacingLabel'), Qt.MouseButton.LeftButton)
                    self.assertEqual(self.render_scene(), preview)
                    self.canvas.text_undo_stack.undo()
                    self.assertEqual(self.render_scene(), before)

    def test_annotations_effects_and_rotation_match_committed_preview(self) -> None:
        self.item.setSelected(False)
        self.item.load_rich_text_html('<p>ABC <ruby>漢字<rt>かんじ</rt></ruby> 🌟</p>')
        self.item.set_text_effects(TextEffectStack(effects=(StrokeEffect(width=.08),)))
        self.item.set_text_transform(TextTransformStack(
            transforms=(BendTextTransform(.2),), glyph_slant_angle=10,
        ))
        self.item.setRotation(20)
        before = self.render_scene()
        self.drag('letter', 10, 20, release=False)
        preview = self.render_scene()
        self.assertNotEqual(preview, before)
        QTest.mouseRelease(self.panel.letterSpacingLabel, Qt.MouseButton.LeftButton)
        self.assertEqual(self.render_scene(), preview)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.render_scene(), before)

    def test_earlier_text_edit_and_clean_marker_survive_drag(self) -> None:
        self.canvas.gv.show()
        self.canvas.gv.activateWindow()
        self.canvas.gv.setFocus()
        self.item.startEdit()
        self.app.processEvents()
        cursor = self.item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText('!')
        self.item.setTextCursor(cursor)
        self.item.endEdit()
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        before = self.item.toHtml()
        self.canvas.text_undo_stack.setClean()
        self.drag('letter', 10, 20, release=False)
        self.assertTrue(self.canvas.text_undo_stack.isClean())
        QTest.mouseRelease(self.panel.letterSpacingLabel, Qt.MouseButton.LeftButton)
        self.assertFalse(self.canvas.text_undo_stack.isClean())
        self.assertEqual(self.canvas.text_undo_stack.count(), 2)
        self.canvas.text_undo_stack.undo()
        self.assertTrue(self.canvas.text_undo_stack.isClean())
        self.assertEqual(self.item.toHtml(), before)
        self.canvas.text_undo_stack.undo()
        self.assertFalse(self.item.toPlainText().endswith('!'))

    def test_preview_items_are_reused_noninteractive_and_released(self) -> None:
        self.drag('letter', 10, release=False)
        session = self.panel.letter_spacing_drag
        references = [weakref.ref(item) for item in session.previews]
        documents = [weakref.ref(item.document()) for item in session.previews]
        self.assertEqual(self.canvas.selected_text_items(), [self.item])
        for delta in (10, 20, -10):
            session.preview(delta)
            self.assertEqual([ref() for ref in references], session.previews)
        session.cancel()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references + documents))
        self.assertEqual(self.item.opacity(), 1)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)

    def test_panel_deletion_restores_source_and_releases_active_preview(self) -> None:
        self.drag('letter', 20, release=False)
        references = [weakref.ref(item) for item in self.panel.letter_spacing_drag.previews]
        references += [weakref.ref(item.document()) for item in self.panel.letter_spacing_drag.previews]
        self.document_edits.panel = None
        self.panel.deleteLater()
        self.panel = None
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertEqual(self.item.opacity(), 1)

    def test_rendered_preview_resources_are_released_at_every_exit(self) -> None:
        other = self.make_item(1)
        other.setRect(QRectF(0, 200, 300, 180))
        for item in (self.item, other):
            item.set_text_effects(TextEffectStack(
                overall_opacity=.6, effects=(StrokeEffect(width=.12), ShadowEffect()),
            ))
            item.set_text_transform(TextTransformStack(
                (BendTextTransform(.2),), glyph_slant_angle=10,
            ))
        self.panel.set_active_format(self.item.get_fontformat())
        caches = (GLOBAL_GLYPH_GEOMETRY_CACHE, GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE)
        for boundary in ('commit', 'cancel', 'save', 'history', 'page', 'scene', 'target', 'delete'):
            with self.subTest(boundary=boundary):
                self.panel.set_textblk_item(None, multi_select=True)
                self.drag('letter', 10, 20, release=False)
                session = self.panel.letter_spacing_drag
                self.render_scene()
                resources = [
                    resource for preview in session.previews for resource in (
                        preview, preview.document(), preview.layout, preview.blk,
                        preview.fontformat, preview.effect_renderer,
                        preview.geometry_controller,
                        preview.geometry_controller.layout_renderer,
                        preview.geometry_controller.surface_renderer,
                        preview.effect_renderer._effect_raster_state,
                    )
                ]
                references = [weakref.ref(resource) for resource in resources]
                destroyed = [
                    QSignalSpy(resource.destroyed) for preview in session.previews
                    for resource in (preview, preview.document(), preview.layout)
                ]
                tokens = {
                    preview.geometry_controller.layout_renderer.geometry_cache.layout_token
                    for preview in session.previews
                }
                self.assertTrue(all(token in caches[0]._namespace_keys for token in tokens))
                if boundary == 'commit':
                    session.commit()
                    self.canvas.text_undo_stack.undo()
                elif boundary == 'cancel':
                    session.cancel()
                elif boundary == 'scene':
                    self.panel.cancel_text_transform_edits_for_scene_change()
                elif boundary == 'target':
                    self.panel.set_textblk_item(other)
                elif boundary == 'delete':
                    self.document_edits.panel = None
                    self.panel.deleteLater()
                    self.panel = None
                else:
                    suffix = boundary if boundary == 'save' else boundary + '_change'
                    getattr(self.panel, 'resolve_text_transform_edits_for_' + suffix)()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.assertTrue(all(len(spy) == 1 for spy in destroyed))
                self.assertEqual(session.previews, [])
                self.assertTrue(all(token not in cache._namespace_keys for cache in caches for token in tokens))
                self.assertEqual(self.item.opacity(), .6)
                self.assertEqual(other.opacity(), .6)
                del resources, destroyed
                gc.collect()
                self.assertTrue(all(ref() is None for ref in references))

    def test_multi_selection_uses_primary_value_and_distance_mode(self) -> None:
        other = self.make_item(1)
        other.setLineSpacingType(LineSpacingType.Distance)
        other.setLineSpacing(.8)
        self.canvas.text_undo_stack.clear()
        self.canvas.set_primary_selected_text_item(other)
        self.panel.set_textblk_item(None, multi_select=True, primary_item=other)
        self.drag('line', 3)
        self.assertEqual(self.item.line_spacing_values(), (1.1, LineSpacingType.Proportional))
        self.assertEqual(other.line_spacing_values(), (1.1, LineSpacingType.Distance))
        self.assertEqual(self.canvas.text_undo_stack.count(), 1)
        self.canvas.text_undo_stack.undo()
        self.assertEqual(self.item.line_spacing_values(), (1.2, LineSpacingType.Proportional))
        self.assertEqual(other.line_spacing_values(), (.8, LineSpacingType.Distance))

    def test_no_motion_and_clamped_drags_do_no_work(self) -> None:
        QTest.mouseClick(self.panel.letterSpacingLabel, Qt.MouseButton.LeftButton)
        self.assertFalse(self.panel.letter_spacing_drag.active)
        self.panel.on_param_changed('letter_spacing', 10)
        count = self.canvas.text_undo_stack.count()
        with patch.object(self.item, 'toHtml', wraps=self.item.toHtml) as serialize:
            self.drag('letter', 1, 10, 100)
            serialize.assert_not_called()
        self.assertEqual(self.canvas.text_undo_stack.count(), count)

    def test_vertical_growth_and_transforms_undo_exactly(self) -> None:
        self.item.setSelected(False)
        self.item.setPlainText('天地玄黄宇宙洪荒')
        self.item.setVertical(True)
        self.item.setRect(QRectF(10, 20, 32, 220))
        self.item.set_text_transform(TextTransformStack((BendTextTransform(.2),)))
        self.panel.set_active_format(self.item.get_fontformat())
        for angle in (0, 25, 90):
            with self.subTest(angle=angle):
                self.item.setRotation(angle)
                before, rect = self.item.toHtml(), self.item.absBoundingRect(qrect=True)
                image = self.render_scene()
                block_rect = list(self.item.blk._bounding_rect)
                self.drag('letter', 70, -30, release=False)
                preview_rect = self.panel.letter_spacing_drag.previews[0].absBoundingRect(qrect=True)
                preview_image = self.render_scene()
                self.assertEqual(self.item.blk._bounding_rect, block_rect)
                QTest.mouseRelease(self.panel.letterSpacingLabel, Qt.MouseButton.LeftButton)
                self.assertEqual(self.item.absBoundingRect(qrect=True), preview_rect)
                self.assertEqual(self.render_scene(), preview_image)
                self.assertEqual(self.canvas.text_undo_stack.count(), 1)
                self.canvas.text_undo_stack.undo()
                self.assertEqual(self.item.toHtml(), before)
                self.assertEqual(self.item.absBoundingRect(qrect=True), rect)
                self.assertEqual(self.render_scene(), image)
                self.canvas.text_undo_stack.clear()


if __name__ == '__main__':
    unittest.main()
