import os
import unittest
from types import SimpleNamespace


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QImage, QInputMethodEvent, QPainter, QTextCursor
from qtpy.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene

try:
    from qtpy.QtGui import QUndoStack
except ImportError:
    from qtpy.QtWidgets import QUndoStack

from ballontranslator.ui.textedit_area import TransPairWidget
from ballontranslator.ui.textedit_commands import (
    MultiPasteCommand,
    ReshapeItemCommand,
    SetTextTransformCommand,
    TextEditCommand,
    propagate_user_edit,
)
from ballontranslator.ui.textitem import TextBlkItem
from ballontranslator.ui.text_effects.glyph import (
    GLOBAL_GLYPH_GEOMETRY_CACHE,
    GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE,
    GlyphGeometry,
    WeightedGlyphGeometryCache,
)
from ballontranslator.ui.text_effects.transform_layout import (
    GlyphSlantLayoutRenderer,
)
from ballontranslator.utils.fontformat import NoTextTransform, SlantTextTransform
from ballontranslator.utils.textblock import TextBlock


TEST_LINES = (
    "Без труда не вытащишь и рыбку из пруда.",
    "冰冻三尺，非一日之寒。",
    "猿も木から落ちる。",
    "Don't judge a book by its cover.",
    "벼는 익을수록 고개를 숙인다.",
    "☀ ☁ ☂ ☃ ★ ☆ ☎ ☯ ♠ ♥ ♦ ♣ ⚠ ⚽ ⚾ ㊗ ㊙ ! @ # $",
)
NEUTRAL = NoTextTransform()
FIRST_TRANSFORM = SlantTextTransform(1.2, 0.9, 12.0, 5.0)
FINAL_TRANSFORMS = (
    SlantTextTransform(0.8, 1.1, -9.0, -4.0),
    SlantTextTransform(1.3, 0.7, 6.0, 8.0),
)


class TextTransformUndoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _make_pair(self, index, text, vertical):
        block = TextBlock([0, 0, 600, 300])
        block._bounding_rect = [0, 0, 600, 300]
        block.vertical = vertical
        block.translation = text
        item = TextBlkItem(block, index)
        pair = TransPairWidget(block, index, False)
        pair.e_trans.setPlainText(item.toPlainText())
        return item, pair

    def _assert_state(self, stack, items, pairs, expected):
        self.assertEqual(stack.index(), expected[0])
        for item, pair, text, transform in zip(
            items, pairs, expected[1], expected[2]
        ):
            self.assertEqual(item.toPlainText(), text)
            self.assertEqual(pair.e_trans.toPlainText(), text)
            self.assertEqual(item.blk.fontformat.text_transform, transform)

    @staticmethod
    def _render_scene(scene):
        image = QImage(
            900,
            600,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        scene.render(
            painter,
            QRectF(0, 0, 900, 600),
            QRectF(-50, -50, 900, 600),
        )
        painter.end()
        return bytes(image.bits().asstring(image.sizeInBytes()))

    @staticmethod
    def _document_steps(items, pairs):
        return tuple(
            (
                item.document().availableUndoSteps(),
                pair.e_trans.document().availableUndoSteps(),
            )
            for item, pair in zip(items, pairs)
        )

    def _push_input_method_commit(
        self,
        stack,
        item,
        pair,
        preedit_text,
        commit_text,
    ):
        edit = pair.e_trans
        pair.show()
        edit.setFocus()
        self.app.processEvents()
        self.assertTrue(edit.hasFocus())
        cursor = edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        edit.setTextCursor(cursor)

        stack_count = stack.count()
        text_before = edit.toPlainText()
        propagated = []
        pushed_steps = []

        def on_propagate(position, added_text, joint_previous):
            propagated.append((position, added_text, joint_previous))
            propagate_user_edit(
                edit,
                item,
                position,
                added_text,
                joint_previous,
            )

        def on_push(num_steps):
            pushed_steps.append(num_steps)
            stack.push(TextEditCommand(edit, num_steps, item))

        edit.propagate_user_edited.connect(on_propagate)
        edit.push_undo_stack.connect(on_push)
        try:
            preedit = QInputMethodEvent(preedit_text, [])
            QApplication.sendEvent(edit, preedit)
            self.assertTrue(edit.pre_editing)
            self.assertEqual(edit.toPlainText(), text_before)
            self.assertEqual(stack.count(), stack_count)

            commit = QInputMethodEvent('', [])
            commit.setCommitString(commit_text)
            QApplication.sendEvent(edit, commit)
            self.app.processEvents()
        finally:
            edit.propagate_user_edited.disconnect(on_propagate)
            edit.push_undo_stack.disconnect(on_push)
            pair.hide()

        self.assertFalse(edit.pre_editing)
        self.assertEqual(propagated, [(len(text_before), commit_text, False)])
        self.assertEqual(pushed_steps, [1])
        self.assertEqual(stack.count(), stack_count + 1)
        self.assertEqual(edit.toPlainText(), text_before + commit_text)
        self.assertEqual(item.toPlainText(), text_before + commit_text)

    def test_mixed_text_transforms_keep_undo_and_pair_widgets_in_sync(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                initial_texts = TEST_LINES[:2]
                first_edit = TEST_LINES[2]
                second_edit = TEST_LINES[3]
                input_method_edit = '入力'
                final_edit = "\n".join(TEST_LINES[4:])
                made = [
                    self._make_pair(index, text, vertical)
                    for index, text in enumerate(initial_texts)
                ]
                items = [entry[0] for entry in made]
                pairs = [entry[1] for entry in made]
                stack = QUndoStack()

                stack.push(
                    MultiPasteCommand(first_edit, [items[0]], [pairs[0].e_trans])
                )
                document_steps = self._document_steps(items, pairs)
                stack.push(
                    SetTextTransformCommand.create(
                        [items[0]], [NEUTRAL], [FIRST_TRANSFORM]
                    )
                )
                self.assertEqual(
                    document_steps,
                    self._document_steps(items, pairs),
                )
                stack.push(
                    MultiPasteCommand(second_edit, [items[1]], [pairs[1].e_trans])
                )
                stack.push(
                    SetTextTransformCommand.create(
                        items,
                        [FIRST_TRANSFORM, NEUTRAL],
                        FINAL_TRANSFORMS,
                    )
                )
                self._push_input_method_commit(
                    stack,
                    items[1],
                    pairs[1],
                    input_method_edit[0],
                    input_method_edit,
                )
                stack.push(
                    MultiPasteCommand(final_edit, [items[0]], [pairs[0].e_trans])
                )

                input_method_text = second_edit + input_method_edit

                states = (
                    (0, initial_texts, (NEUTRAL, NEUTRAL)),
                    (1, (first_edit, initial_texts[1]), (NEUTRAL, NEUTRAL)),
                    (2, (first_edit, initial_texts[1]), (FIRST_TRANSFORM, NEUTRAL)),
                    (3, (first_edit, second_edit), (FIRST_TRANSFORM, NEUTRAL)),
                    (4, (first_edit, second_edit), FINAL_TRANSFORMS),
                    (5, (first_edit, input_method_text), FINAL_TRANSFORMS),
                    (6, (final_edit, input_method_text), FINAL_TRANSFORMS),
                )
                self.assertEqual(stack.count(), len(states) - 1)
                self._assert_state(stack, items, pairs, states[-1])

                for _ in range(3):
                    for expected in reversed(states[:-1]):
                        stack.undo()
                        self._assert_state(stack, items, pairs, expected)

                    for expected in states[1:]:
                        stack.redo()
                        self._assert_state(stack, items, pairs, expected)

                self.assertEqual(stack.count(), len(states) - 1)

    def test_neutral_effect_render_is_stable_after_transform_roundtrip(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                block = TextBlock([0, 0, 600, 300])
                block._bounding_rect = [0, 0, 600, 300]
                block.vertical = vertical
                block.translation = "\n".join(TEST_LINES[:4])
                block.fontformat.stroke_width = 0.08
                block.fontformat.shadow_radius = 0.08
                block.fontformat.shadow_strength = 0.7
                block.fontformat.shadow_offset = [0.08, 0.06]
                block.fontformat.gradient_enabled = True
                block.fontformat.gradient_start_color = [20, 40, 160]
                block.fontformat.gradient_end_color = [220, 80, 40]

                item = TextBlkItem(block, 0)
                scene = QGraphicsScene()
                scene.addItem(item)
                self.app.processEvents()
                neutral_rect = item.sceneBoundingRect()
                neutral_pixels = self._render_scene(scene)

                item.set_text_transform(FIRST_TRANSFORM)
                transformed_pixels = self._render_scene(scene)
                self.assertNotEqual(neutral_pixels, transformed_pixels)

                item.set_text_transform(NEUTRAL)
                self.app.processEvents()
                self.assertEqual(item.sceneBoundingRect(), neutral_rect)
                self.assertEqual(self._render_scene(scene), neutral_pixels)
                scene.removeItem(item)

    def test_global_geometry_cache_isolated_by_layout_and_preview(self):
        cache = WeightedGlyphGeometryCache(max_entries=2, max_weight=10)
        cache.store('first', 1, weight=6)
        cache.store('second', 2, weight=4)
        self.assertEqual(cache.get('first'), 1)
        cache.store('third', 3, weight=4)
        self.assertIsNone(cache.get('second'))
        self.assertEqual(cache.get('first'), 1)
        self.assertEqual(cache.get('third'), 3)

        GLOBAL_GLYPH_GEOMETRY_CACHE.clear()
        GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE.clear()
        self.addCleanup(GLOBAL_GLYPH_GEOMETRY_CACHE.clear)
        self.addCleanup(GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE.clear)
        geometry = GlyphGeometry((), (), QRectF(0, 0, 1, 1))
        first_layout = SimpleNamespace(layout_generation=0)
        second_layout = SimpleNamespace(layout_generation=0)
        first = GlyphSlantLayoutRenderer(first_layout)
        second = GlyphSlantLayoutRenderer(second_layout)
        first.geometry_cache.store('same-local-key', geometry)
        self.assertEqual(
            first.geometry_cache.get('same-local-key'),
            geometry,
        )
        self.assertIsNone(second.geometry_cache.get('same-local-key'))

        self.assertGreater(len(GLOBAL_GLYPH_GEOMETRY_CACHE), 0)
        first_layout.layout_generation += 1
        self.assertIsNone(first.geometry_cache.get('same-local-key'))
        self.assertEqual(len(GLOBAL_GLYPH_GEOMETRY_CACHE), 0)
        global_entries = len(GLOBAL_GLYPH_GEOMETRY_CACHE)
        first.geometry_cache.set_persistent(False)
        first.geometry_cache.store('preview-key', geometry)
        self.assertEqual(first.geometry_cache.get('preview-key'), geometry)
        self.assertEqual(len(GLOBAL_GLYPH_GEOMETRY_CACHE), global_entries)
        self.assertEqual(len(GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE), 1)
        first.geometry_cache.set_persistent(True)
        self.assertIsNone(first.geometry_cache.get('preview-key'))
        self.assertEqual(len(GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE), 0)

        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.set_text_transform(FIRST_TRANSFORM)
        geometry_cache = item.geometry_controller.layout_renderer.geometry_cache
        self.assertTrue(geometry_cache.persistent)
        box_preview = FIRST_TRANSFORM.with_value('horizontal_scale', 1.4)
        item.set_text_transform(box_preview, preview=True)
        self.assertTrue(geometry_cache.persistent)
        item.clear_text_transform_preview()
        glyph_preview = FIRST_TRANSFORM.with_value('glyph_slant_angle', 9.0)
        committed_entries = len(GLOBAL_GLYPH_GEOMETRY_CACHE)
        item.set_text_transform(glyph_preview, preview=True)
        self.assertFalse(geometry_cache.persistent)
        self.assertEqual(
            len(GLOBAL_GLYPH_GEOMETRY_CACHE),
            committed_entries,
        )
        self.assertGreater(len(GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE), 0)
        item.set_text_transform(glyph_preview)
        self.assertTrue(geometry_cache.persistent)
        self.assertEqual(len(GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE), 0)

    def test_effect_padding_is_shrinkable_and_not_document_history(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item, _ = self._make_pair(0, TEST_LINES[0], vertical)
                item.document().clearUndoRedoStacks()

                item.setStrokeWidth(0.2, repaint_background=False)
                wide_stroke_padding = item.padding()
                item.setStrokeWidth(0.05, repaint_background=False)
                self.assertLess(item.padding(), wide_stroke_padding)

                item.setFontSize(40, repaint_background=False)
                large_font_padding = item.padding()
                item.setFontSize(10, repaint_background=False)
                self.assertLess(item.padding(), large_font_padding)

                item.setRelFontSize(2.0, repaint_background=False)
                relative_large_padding = item.padding()
                item.setRelFontSize(0.5, repaint_background=False)
                self.assertLess(item.padding(), relative_large_padding)

                item.setStrokeWidth(0.0, repaint_background=False)
                shadow = item.fontformat.deepcopy()
                shadow.shadow_radius = 0.2
                shadow.shadow_strength = 0.8
                shadow.shadow_offset = [0.0, 0.0]
                item.setShadow(shadow, repaint=False)
                centered_shadow_padding = item.padding()
                self.assertGreater(centered_shadow_padding, 0.0)
                shadow.shadow_offset = [0.8, -0.4]
                item.setShadow(shadow, repaint=False)
                self.assertGreater(item.padding(), centered_shadow_padding)
                shadow.shadow_strength = 0.0
                item.setShadow(shadow, repaint=False)
                self.assertEqual(item.padding(), 0.0)

                item.document().clearUndoRedoStacks()
                logical_rect = item.absBoundingRect(qrect=True)
                item.setPadding(7.0)
                item.setPadding(0.0)
                self.assertEqual(item.document().availableUndoSteps(), 0)
                self.assertEqual(item.document().documentMargin(), 0.0)
                self.assertEqual(item.absBoundingRect(qrect=True), logical_rect)

    def test_none_and_box_only_paths_do_not_create_glyph_renderer(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                GLOBAL_GLYPH_GEOMETRY_CACHE.clear()
                GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE.clear()
                item, _ = self._make_pair(0, TEST_LINES[1], vertical)

                self.assertIsNone(item.geometry_controller.layout_renderer)
                self.assertIsNone(item.layout.render_delegate)
                self.assertIsNone(
                    item.effect_renderer._transformed_effect_state
                )
                self.assertFalse(
                    bool(
                        item.flags()
                        & QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
                    )
                )
                item.layout.reLayout()
                self.assertEqual(len(GLOBAL_GLYPH_GEOMETRY_CACHE), 0)

                box_only = SlantTextTransform(1.2, 0.9, 8.0, 0.0)
                item.set_text_transform(box_only)
                self.assertIsNone(item.layout.render_delegate)
                self.assertTrue(
                    bool(
                        item.flags()
                        & QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
                    )
                )

                item.set_text_transform(FIRST_TRANSFORM)
                self.assertIs(
                    item.layout.render_delegate,
                    item.geometry_controller.layout_renderer,
                )

                item.set_text_transform(NEUTRAL)
                self.assertIsNone(item.geometry_controller.layout_renderer)
                self.assertIsNone(item.layout.render_delegate)
                self.assertFalse(
                    bool(
                        item.flags()
                        & QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
                    )
                )

    def test_resize_undo_stores_alternating_logical_rectangles(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item, _ = self._make_pair(0, TEST_LINES[2], vertical)
                stack = QUndoStack()
                before = item.absBoundingRect(qrect=True)
                after = QRectF(
                    before.x() + 10,
                    before.y() + 15,
                    before.width() - 80,
                    before.height() + 60,
                )
                item.oldRect = QRectF(before)
                item.setRect(after)
                stack.push(ReshapeItemCommand(item))
                stack.push(
                    SetTextTransformCommand.create(
                        [item], [NEUTRAL], [FIRST_TRANSFORM]
                    )
                )

                stack.undo()
                self.assertEqual(item.absBoundingRect(qrect=True), after)
                stack.undo()
                self.assertEqual(item.absBoundingRect(qrect=True), before)
                stack.redo()
                self.assertEqual(item.absBoundingRect(qrect=True), after)
                stack.redo()
                self.assertEqual(item.absBoundingRect(qrect=True), after)


if __name__ == "__main__":
    unittest.main()
