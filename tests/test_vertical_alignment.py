import os
import unittest
from unittest.mock import patch


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.fontformat import TextAlignment
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


class VerticalAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(
        alignment: TextAlignment,
        *,
        bounds=(0, 0, 320, 220),
        text='天地12玄黃宇宙',
        stroke_width=0.0,
        glyph_slant_angle=0.0,
    ) -> TextBlkItem:
        block = TextBlock(
            list(bounds),
            text_layout_version=TEXT_LAYOUT_VERSION,
        )
        block._bounding_rect = list(bounds)
        block.vertical = True
        block.alignment = alignment
        block.translation = text
        block.fontformat.font_size = 32
        block.fontformat.letter_spacing = 1.0
        block.fontformat.stroke_width = stroke_width
        block.fontformat.glyph_slant_angle = glyph_slant_angle
        return TextBlkItem(block, 0)

    @staticmethod
    def _line_x_positions(item: TextBlkItem) -> list[float]:
        positions = []
        block = item.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            positions.extend(
                layout.lineAt(index).x()
                for index in range(layout.lineCount())
            )
            block = block.next()
        return positions

    @staticmethod
    def _sample_hits(item: TextBlkItem) -> list[int]:
        hits = []
        block = item.document().firstBlock()
        while block.isValid():
            block_number = block.blockNumber()
            text_layout = block.layout()
            for line_number in range(text_layout.lineCount()):
                line = text_layout.lineAt(line_number)
                top, bottom = item.layout.y_offset_lst[block_number][
                    line_number
                ]
                point = QPointF(
                    line.x()
                    + max(1.0, line.naturalTextRect().width() / 2),
                    (top + bottom) / 2,
                )
                hits.append(
                    item.layout.hitTest(
                        point,
                        Qt.HitTestAccuracy.FuzzyHit,
                    )
                )
            block = block.next()
        return hits

    @staticmethod
    def _anchor(item: TextBlkItem, alignment: TextAlignment) -> QPointF:
        rect = item.logical_unpadded_rect()
        if alignment == TextAlignment.Left:
            point = rect.topLeft()
        elif alignment == TextAlignment.Center:
            point = QPointF(rect.center().x(), rect.top())
        else:
            point = rect.topRight()
        return item.mapToScene(point)

    def test_unversioned_vertical_blocks_migrate_to_effective_right(self):
        for font_alignment in (
            TextAlignment.Left,
            TextAlignment.Center,
            TextAlignment.Right,
        ):
            with self.subTest(font_alignment=font_alignment):
                block = TextBlock(
                    fontformat={
                        'vertical': True,
                        'alignment': font_alignment,
                    },
                )
                self.assertEqual(block.alignment, TextAlignment.Right)
                self.assertEqual(
                    block.text_layout_version,
                    TEXT_LAYOUT_VERSION,
                )
                self.assertTrue(block.src_is_vertical)

        old_top_level = TextBlock(vertical=True, _alignment=TextAlignment.Left)
        self.assertEqual(old_top_level.alignment, TextAlignment.Right)
        self.assertTrue(old_top_level.src_is_vertical)

    def test_versioned_and_horizontal_blocks_preserve_alignment(self):
        current = TextBlock(
            text_layout_version=TEXT_LAYOUT_VERSION,
            fontformat={
                'vertical': True,
                'alignment': TextAlignment.Center,
            },
        )
        horizontal = TextBlock(
            fontformat={
                'vertical': False,
                'alignment': TextAlignment.Center,
            },
        )
        future = TextBlock(
            text_layout_version=TEXT_LAYOUT_VERSION + 1,
            fontformat={
                'vertical': True,
                'alignment': TextAlignment.Left,
            },
        )
        malformed = TextBlock(
            text_layout_version='invalid',
            fontformat={
                'vertical': True,
                'alignment': TextAlignment.Center,
            },
        )

        self.assertEqual(current.alignment, TextAlignment.Center)
        self.assertEqual(horizontal.alignment, TextAlignment.Center)
        self.assertEqual(future.alignment, TextAlignment.Left)
        self.assertEqual(
            future.text_layout_version,
            TEXT_LAYOUT_VERSION + 1,
        )
        self.assertEqual(malformed.alignment, TextAlignment.Right)
        self.assertEqual(
            malformed.text_layout_version,
            TEXT_LAYOUT_VERSION,
        )
        self.assertIn('text_layout_version', current.to_dict())

    def test_writing_mode_switch_uses_the_saved_item_alignment(self):
        item = self._make_item(TextAlignment.Center)
        item.setVertical(False)
        self.assertEqual(
            item.document().defaultTextOption().alignment(),
            Qt.AlignmentFlag.AlignCenter,
        )
        item.setVertical(True)
        self.assertEqual(item.fontformat.alignment, TextAlignment.Center)
        slack = (
            item.layout.available_width
            - item.layout._column_content_width()
        )
        self.assertLessEqual(
            abs(
                item.layout.layout_left
                - item.layout.effectPadding()
                - slack / 2
            ),
            1 / 64,
        )

    def test_writing_mode_switch_runs_only_the_settled_layout(self):
        item = self._make_item(TextAlignment.Center)
        item.setVertical(False)
        discarded_layout = item.layout

        with patch.object(
            discarded_layout,
            'reLayoutEverything',
            wraps=discarded_layout.reLayoutEverything,
        ) as discarded_relayout:
            item.setVertical(True)

        self.assertEqual(discarded_relayout.call_count, 0)
        self.assertEqual(item.layout.layout_generation, 1)

    def test_writing_mode_switch_detaches_the_old_slant_layout(self):
        item = self._make_item(
            TextAlignment.Center,
            glyph_slant_angle=20.0,
        )
        controller = item.geometry_controller
        old_renderer = controller.layout_renderer
        self.assertIsNotNone(old_renderer)

        observed_renderers = []
        initialize_layout = controller.initialize_layout

        def probe_initialize_layout(*, persistent_cache=True) -> bool:
            # This is the first controller boundary after Qt has adopted the
            # replacement layout. A geometry query here previously followed
            # old_renderer into the deleted VerticalTextDocumentLayout.
            observed_renderers.append(controller.layout_renderer)
            item.boundingRect()
            return initialize_layout(persistent_cache=persistent_cache)

        with patch.object(
            controller,
            'initialize_layout',
            side_effect=probe_initialize_layout,
        ):
            item.setVertical(False)

        self.assertEqual(observed_renderers, [None])
        self.assertIsNotNone(controller.layout_renderer)
        self.assertIsNot(controller.layout_renderer, old_renderer)
        self.assertIs(controller.layout_renderer.layout, item.layout)

        horizontal_renderer = controller.layout_renderer
        item.setVertical(True)
        item.boundingRect()
        self.assertIsNotNone(controller.layout_renderer)
        self.assertIsNot(controller.layout_renderer, horizontal_renderer)
        self.assertIs(controller.layout_renderer.layout, item.layout)

    def test_switching_alignment_translates_layout_and_preserves_hits(self):
        item = self._make_item(
            TextAlignment.Right,
            stroke_width=0.08,
        )
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(2)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.endEdit(keep_focus=False)
        self.app.processEvents()

        layout = item.layout
        slack = layout.available_width - layout._column_content_width()
        self.assertGreater(slack, 0)
        right_x = self._line_x_positions(item)
        right_hits = self._sample_hits(item)
        right_cursor = layout.source_cursor_rect(0)
        logical_rect = item.logical_unpadded_rect()
        model_rect = list(item.blk._bounding_rect)
        item_position = QPointF(item.pos())
        document_revision = item.document().revision()
        right_generation = layout.layout_generation
        old_effect_key = item.effect_renderer.background_pixmap.cacheKey()

        item.setAlignment(TextAlignment.Center)
        center_x = self._line_x_positions(item)
        center_cursor = layout.source_cursor_rect(0)
        self.assertTrue(
            all(
                abs((center - right) + slack / 2) < 1e-6
                for right, center in zip(right_x, center_x)
            )
        )
        self.assertEqual(self._sample_hits(item), right_hits)
        self.assertAlmostEqual(
            center_cursor.x() - right_cursor.x(),
            -slack / 2,
            places=6,
        )
        self.assertEqual(layout.layout_generation, right_generation + 1)
        self.assertNotEqual(
            item.effect_renderer.background_pixmap.cacheKey(),
            old_effect_key,
        )

        item.setAlignment(TextAlignment.Left)
        left_x = self._line_x_positions(item)
        self.assertTrue(
            all(
                abs((left - right) + slack) < 1e-6
                for right, left in zip(right_x, left_x)
            )
        )
        self.assertEqual(self._sample_hits(item), right_hits)
        self.assertEqual(item.logical_unpadded_rect(), logical_rect)
        self.assertEqual(item.blk._bounding_rect, model_rect)
        self.assertEqual(item.pos(), item_position)
        self.assertEqual(item.document().revision(), document_revision)

    def test_alignment_is_item_state_not_a_document_edit(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self._make_item(TextAlignment.Right)
                if not vertical:
                    item.setVertical(False)
                item.startEdit()
                cursor = item.textCursor()
                cursor.setPosition(2)
                cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
                item.setTextCursor(cursor)
                item.document().clearUndoRedoStacks()
                revision = item.document().revision()
                cursor_state = (
                    item.textCursor().position(),
                    item.textCursor().anchor().__pos__(),
                )
                geometry_notifications = []
                item.visual_geometry_changed.connect(
                    lambda: geometry_notifications.append(True)
                )

                item.setAlignment(
                    TextAlignment.Center,
                    restore_cursor=True,
                )

                self.assertEqual(
                    (
                        item.textCursor().position(),
                        item.textCursor().anchor().__pos__(),
                    ),
                    cursor_state,
                )
                self.assertEqual(item.document().availableUndoSteps(), 0)
                self.assertEqual(item.document().revision(), revision)
                self.assertEqual(geometry_notifications, [])

    def test_alignment_updates_slanted_ink_without_eager_repaint(self):
        item = self._make_item(
            TextAlignment.Right,
            glyph_slant_angle=20.0,
        )

        with patch.object(item, 'repaint_background') as repaint:
            item.setAlignment(
                TextAlignment.Left,
                repaint_background=False,
            )

        repaint.assert_not_called()
        self.assertFalse(item._update_effect_padding())

    def test_width_resize_uses_left_center_right_growth_anchors(self):
        expected_line_shift = {
            TextAlignment.Left: 0.0,
            TextAlignment.Center: 40.0,
            TextAlignment.Right: 80.0,
        }
        for alignment, expected_shift in expected_line_shift.items():
            with self.subTest(alignment=alignment):
                item = self._make_item(alignment, text='天地')
                item.setRotation(23)
                before_anchor = self._anchor(item, alignment)
                before_x = self._line_x_positions(item)[0]
                before_generation = item.layout.layout_generation
                rect = item.logical_unpadded_rect()

                item.set_size(
                    rect.width() + 80,
                    rect.height(),
                    set_layout_maxsize=True,
                )

                after_anchor = self._anchor(item, alignment)
                after_x = self._line_x_positions(item)[0]
                self.assertAlmostEqual(
                    after_x - before_x,
                    expected_shift,
                    places=6,
                )
                self.assertAlmostEqual(
                    after_anchor.x(), before_anchor.x(), places=6
                )
                self.assertAlmostEqual(
                    after_anchor.y(), before_anchor.y(), places=6
                )
                expected_generation = before_generation + (
                    alignment != TextAlignment.Left
                )
                self.assertEqual(
                    item.layout.layout_generation,
                    expected_generation,
                )

    def test_fractional_middle_handle_resizes_match_settled_layout(self):
        for alignment in TextAlignment:
            for handle in ('left', 'right'):
                with self.subTest(alignment=alignment, handle=handle):
                    item = self._make_item(
                        alignment,
                        stroke_width=0.08,
                    )
                    initial = item.absBoundingRect(qrect=True)
                    for step in range(1, 201):
                        resized = QRectF(initial)
                        delta = step * 0.37
                        if handle == 'left':
                            resized.setLeft(initial.left() - delta)
                        else:
                            resized.setRight(initial.right() + delta)
                        item.setRect(resized, repaint=False)

                    fast_positions = self._line_x_positions(item)
                    item.layout.reLayout()
                    for fast, settled in zip(
                        fast_positions,
                        self._line_x_positions(item),
                    ):
                        self.assertLessEqual(
                            abs(fast - settled),
                            1 / 64,
                        )

    def test_new_columns_preserve_each_alignment_growth_anchor(self):
        for alignment in TextAlignment:
            with self.subTest(alignment=alignment):
                item = self._make_item(
                    alignment,
                    bounds=(0, 0, 90, 90),
                    text='天',
                )
                item.setRotation(21)
                before_anchor = self._anchor(item, alignment)
                old_width = item.logical_unpadded_rect().width()

                cursor = item.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.insertText(
                    '地玄黃宇宙洪荒日月盈昴辰宿列張'
                )
                item.setTextCursor(cursor)
                self.app.processEvents()

                self.assertGreater(
                    item.logical_unpadded_rect().width(), old_width
                )
                after_anchor = self._anchor(item, alignment)
                self.assertAlmostEqual(
                    after_anchor.x(), before_anchor.x(), places=6
                )
                self.assertAlmostEqual(
                    after_anchor.y(), before_anchor.y(), places=6
                )


if __name__ == '__main__':
    unittest.main()
