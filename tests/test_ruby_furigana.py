import os
import unittest
from unittest.mock import patch


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QEvent, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QInputMethodEvent,
    QKeyEvent,
    QPainter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextLayout,
)
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.ui.text_engine import (
    effect_renderer as effect_rendering,
    horizontal_layout,
)
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.text_engine.rendering import glyph as glyph_rendering
from ballontranslator.ui.text_engine.annotations import (
    RubyValidationError,
    apply_emphasis,
    apply_letter_spacing,
    apply_ruby,
    apply_text_combine_upright,
    create_rich_text_mime,
    insert_rich_text_mime,
    load_rich_text_html,
    prepare_ruby_insertion,
    remove_ruby,
    ruby_containers,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.formatting.advanced import RubyFuriganaGroup
from ballontranslator.ui.text_engine.editing.commands import propagate_user_edit
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.glyph import resolve_paint_spans
from ballontranslator.ui.text_engine.rendering.ruby import (
    RUBY_FONT_SCALE,
    draw_ruby_placement,
)
from ballontranslator.ui.text_engine.rendering.native_document import (
    NATIVE_DOCUMENT_CACHE,
)
from ballontranslator.utils.fontformat import (
    BendTextTransform,
    ProjectiveTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.textblock import TextBlock


def _select(document: QTextDocument, start: int, end: int) -> QTextCursor:
    cursor = QTextCursor(document)
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
    return cursor


class RubyFuriganaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _item(
        *,
        vertical: bool = False,
        text: str = '東京 ABC',
        bounds=(0, 0, 320, 180),
    ) -> TextBlkItem:
        block = TextBlock(list(bounds))
        block._bounding_rect = list(bounds)
        block.vertical = vertical
        block.translation = text
        block.fontformat.font_size = 32
        return TextBlkItem(block, 0)

    def test_group_html_round_trip_keeps_nested_format_entities_and_rp(self):
        html = (
            '<p><ruby style="ruby-position: under; ruby-merge: merge">'
            '<span style="font-weight:700">東&amp;京</span>'
            '<rp>(</rp><rt>とう&amp;きょう</rt><rp>)</rp></ruby></p>'
        )
        document = QTextDocument()
        load_rich_text_html(document, html)

        containers = ruby_containers(document)
        exported = to_rich_text_html(document)
        restored = QTextDocument()
        load_rich_text_html(restored, exported)

        self.assertEqual(document.toPlainText(), '東&京')
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].ruby_type, 'group')
        self.assertEqual(containers[0].position, 'under')
        self.assertEqual(containers[0].units[0].text, 'とう&きょう')
        self.assertIn('<ruby style="ruby-position: under;', exported)
        self.assertIn('<rt>とう&amp;きょう</rt>', exported)
        self.assertNotIn('data-btrans-runtime', exported)
        self.assertTrue(_select(restored, 0, 3).charFormat().font().bold())
        self.assertNotEqual(
            containers[0].container_id,
            ruby_containers(restored)[0].container_id,
        )

    def test_old_html_and_save_load_save_keep_semantic_content(self):
        old = QTextDocument()
        load_rich_text_html(
            old,
            '<p><span style="font-style:italic">No Ruby &amp; safe</span></p>',
        )
        self.assertEqual(old.toPlainText(), 'No Ruby & safe')
        self.assertEqual(ruby_containers(old), ())
        self.assertNotIn('<ruby', to_rich_text_html(old).lower())

        document = QTextDocument()
        document.setPlainText('東京')
        apply_ruby(_select(document, 0, 2), 'mono', 'とう きょう', 'under')
        first_html = to_rich_text_html(document)
        restored = QTextDocument()
        load_rich_text_html(restored, first_html)
        second_html = to_rich_text_html(restored)
        self.assertEqual(first_html, second_html)
        self.assertIn('<span>東</span><rt>とう</rt>', second_html)
        self.assertIn('<span>京</span><rt>きょう</rt>', second_html)

        divisions = QTextDocument()
        load_rich_text_html(
            divisions,
            '<div><ruby>東<rt>とう</rt></ruby></div>'
            '<div><ruby>京<rt>きょう</rt></ruby></div>',
        )
        self.assertEqual(divisions.toPlainText(), '東\n京')
        self.assertEqual(len(ruby_containers(divisions)), 2)

    def test_mono_round_trip_keeps_pair_order_and_distinct_identical_units(self):
        document = QTextDocument()
        document.setPlainText('人人')
        apply_ruby(_select(document, 0, 2), 'mono', 'ひと ひと')

        first = ruby_containers(document)[0]
        exported = to_rich_text_html(document)
        restored = QTextDocument()
        load_rich_text_html(restored, exported)
        second = ruby_containers(restored)[0]

        self.assertEqual([unit.text for unit in first.units], ['ひと', 'ひと'])
        self.assertNotEqual(first.units[0].unit_id, first.units[1].unit_id)
        self.assertIn('ruby-merge: separate', exported)
        self.assertEqual([unit.text for unit in second.units], ['ひと', 'ひと'])
        self.assertNotEqual(second.units[0].unit_id, second.units[1].unit_id)

    def test_malformed_and_nested_ruby_preserve_only_base_text(self):
        for html, expected in (
            ('<p><ruby>漢<rt></rt></ruby>字</p>', '漢字'),
            (
                '<p><ruby>外<ruby>内<rt>ない</rt></ruby><rt>そと</rt></ruby></p>',
                '外内',
            ),
            ('<p><ruby style="ruby-merge: sideways">漢<rt>かん</rt></ruby></p>', '漢'),
            (
                '<p><ruby style="ruby-merge: separate">漢<rt>か ん</rt></ruby></p>',
                '漢',
            ),
            ('<p><ruby>東<br/>京<rt>とうきょう</rt></ruby></p>', '東\n京'),
            ('<p><ruby>漢<rt>か<rt>ん</rt></rt></ruby></p>', '漢'),
            ('<p><ruby><b>東</i><rt>とう</rt></ruby></p>', '東'),
            ('<p><ruby><span>東</rt><rt>とう</rt></ruby></p>', '東'),
            (
                '<p><ruby style="ruby-align: center">'
                '東<rt>とう</rt></ruby></p>',
                '東',
            ),
            (
                '<p><ruby style="ruby-overhang: auto">'
                '東<rt>とう</rt></ruby></p>',
                '東',
            ),
        ):
            with self.subTest(html=html):
                document = QTextDocument()
                load_rich_text_html(document, html)
                self.assertEqual(document.toPlainText(), expected)
                self.assertEqual(ruby_containers(document), ())

        unsupported = QTextDocument()
        load_rich_text_html(
            unsupported,
            '<p><ruby style="ruby-align: center"><b>東</b>'
            '<rp>(</rp><rt>とう</rt><rp>)</rp></ruby></p>',
        )
        self.assertEqual(unsupported.toPlainText(), '東')
        self.assertTrue(_select(unsupported, 0, 1).charFormat().font().bold())

        supported = QTextDocument()
        load_rich_text_html(
            supported,
            '<p><ruby style="ruby-overhang: none">'
            '哈尔滨佛学院<rt>哈佛</rt></ruby></p>',
        )
        self.assertEqual(len(ruby_containers(supported)), 1)
        exported = to_rich_text_html(supported)
        self.assertIn('ruby-align: space-around', exported)
        self.assertIn('ruby-overhang: none', exported)

    def test_custom_clipboard_round_trip_remaps_both_id_levels(self):
        source = QTextDocument()
        source.setPlainText('東京')
        apply_ruby(_select(source, 0, 2), 'mono', 'とう きょう')
        original = ruby_containers(source)[0]
        mime = create_rich_text_mime(_select(source, 0, 2))
        target = QTextDocument()
        cursor = QTextCursor(target)

        self.assertTrue(insert_rich_text_mime(cursor, mime))
        pasted = ruby_containers(target)[0]

        self.assertNotEqual(original.container_id, pasted.container_id)
        self.assertTrue(all(
            left.unit_id != right.unit_id
            for left, right in zip(original.units, pasted.units)
        ))

        first_paste_ids = (
            pasted.container_id,
            tuple(unit.unit_id for unit in pasted.units),
        )
        self.assertTrue(insert_rich_text_mime(cursor, mime))
        adjacent = ruby_containers(target)
        self.assertEqual(len(adjacent), 2)
        self.assertNotEqual(first_paste_ids[0], adjacent[1].container_id)
        self.assertTrue(set(first_paste_ids[1]).isdisjoint(
            unit.unit_id for unit in adjacent[1].units
        ))

    def test_mono_uses_unicode_graphemes_not_python_or_utf16_characters(self):
        document = QTextDocument()
        document.setPlainText('A\u0301👩\u200d👩\u200d👧\u200d👧')
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        apply_ruby(cursor, 'mono', 'えー かぞく')

        container = ruby_containers(document)[0]
        self.assertEqual(len(container.units), 2)
        self.assertGreater(container.units[1].length, 2)
        self.assertEqual([unit.text for unit in container.units], ['えー', 'かぞく'])

        for base in ('A\u0301', '👩\u200d💻'):
            with self.subTest(base=base):
                item = self._item(text=base + 'X')
                end = len(base.encode('utf-16-le')) // 2
                apply_ruby(
                    _select(item.document(), 0, end),
                    'group',
                    'ながいながいながい',
                )
                item.layout.reLayoutEverything()
                block = item.document().firstBlock()
                metric = item.layout._ruby_metrics[0][0]
                line = block.layout().lineForTextPosition(0)
                cell = item.layout._ruby_unit_cell(block, line, metric)
                self.assertAlmostEqual(
                    cell.width(), metric.extent, delta=1.0
                )

    def test_creation_validation_rejects_empty_paragraph_mismatch_and_tate(self):
        document = QTextDocument()
        document.setPlainText('東京\n大阪')
        with self.assertRaises(RubyValidationError):
            apply_ruby(QTextCursor(document), 'group', 'とうきょう')
        with self.assertRaises(RubyValidationError):
            apply_ruby(_select(document, 0, 2), 'group', '')
        with self.assertRaises(RubyValidationError):
            apply_ruby(_select(document, 0, 5), 'group', 'とうきょうおおさか')
        with self.assertRaises(RubyValidationError):
            apply_ruby(_select(document, 0, 2), 'mono', 'とう')

        tate = _select(document, 0, 2)
        apply_text_combine_upright(tate, True)
        with self.assertRaises(RubyValidationError):
            apply_ruby(_select(document, 0, 2), 'group', 'とうきょう')

    def test_partial_overlap_is_rejected_without_changing_existing_ruby(self):
        document = QTextDocument()
        document.setPlainText('東京都')
        apply_ruby(_select(document, 0, 2), 'group', 'とうきょう')
        before = to_rich_text_html(document)

        with self.assertRaises(RubyValidationError):
            apply_ruby(_select(document, 1, 3), 'group', 'きょうと')

        self.assertEqual(to_rich_text_html(document), before)

    def test_caret_update_conversion_remove_undo_and_redo(self):
        document = QTextDocument()
        document.setPlainText('東京')
        apply_ruby(_select(document, 0, 2), 'group', 'とうきょう')
        document.clearUndoRedoStacks()
        caret = QTextCursor(document)
        caret.setPosition(1)

        apply_ruby(caret, 'mono', 'とう きょう', 'under')
        updated = ruby_containers(document)[0]
        self.assertEqual((updated.ruby_type, updated.position), ('mono', 'under'))
        document.undo()
        self.assertEqual(ruby_containers(document)[0].ruby_type, 'group')
        document.redo()
        self.assertEqual(ruby_containers(document)[0].ruby_type, 'mono')
        self.assertTrue(remove_ruby(caret))
        self.assertEqual(ruby_containers(document), ())
        document.undo()
        self.assertEqual(ruby_containers(document)[0].ruby_type, 'mono')

    def test_group_interior_insertion_extends_but_boundaries_do_not(self):
        document = QTextDocument()
        document.setPlainText('東京')
        apply_ruby(_select(document, 0, 2), 'group', 'とうきょう')
        inside = QTextCursor(document)
        inside.setPosition(1)
        prepare_ruby_insertion(inside)
        inside.insertText('X')
        self.assertEqual(ruby_containers(document)[0].length, 3)

        boundary = QTextCursor(document)
        boundary.setPosition(3)
        prepare_ruby_insertion(boundary)
        boundary.insertText('Y')
        self.assertEqual(ruby_containers(document)[0].length, 3)
        self.assertEqual(document.toPlainText(), '東X京Y')

        start = QTextCursor(document)
        start.setPosition(0)
        prepare_ruby_insertion(start)
        start.insertText('Z')
        self.assertEqual(document.toPlainText(), 'Z東X京Y')
        self.assertEqual(ruby_containers(document)[0].start, 1)

    def test_group_deletion_and_creation_undo_redo_remain_native(self):
        document = QTextDocument()
        document.setPlainText('東京')
        apply_ruby(_select(document, 0, 2), 'group', 'とうきょう')
        document.undo()
        self.assertEqual(ruby_containers(document), ())
        document.redo()
        self.assertEqual(len(ruby_containers(document)), 1)

        _select(document, 0, 1).removeSelectedText()
        self.assertEqual(document.toPlainText(), '京')
        self.assertEqual(ruby_containers(document)[0].length, 1)
        _select(document, 0, 1).removeSelectedText()
        self.assertEqual(document.toPlainText(), '')
        self.assertEqual(ruby_containers(document), ())

    def test_side_editor_propagation_obeys_group_and_mono_boundaries(self):
        for ruby_type, reading in (
            ('group', 'とうきょう'),
            ('mono', 'とう きょう'),
        ):
            with self.subTest(ruby_type=ruby_type):
                item = self._item(text='東京')
                apply_ruby(
                    _select(item.document(), 0, 2), ruby_type, reading
                )
                propagate_user_edit(item, 0, 0, 'X')
                self.assertEqual(item.toPlainText(), 'X東京')
                self.assertEqual(ruby_containers(item.document())[0].start, 1)

        mono = self._item(text='東京X')
        apply_ruby(_select(mono.document(), 0, 2), 'mono', 'とう きょう')
        propagate_user_edit(mono, 1, 1, 'A')
        containers = ruby_containers(mono.document())
        self.assertEqual(mono.toPlainText(), '東AX')
        self.assertEqual(len(containers), 1)
        self.assertEqual((containers[0].start, containers[0].units[0].text), (0, 'とう'))

    def test_production_replacement_never_transfers_mono_reading(self):
        item = self._item(text='東京X')
        apply_ruby(_select(item.document(), 0, 2), 'mono', 'とう きょう')
        item.setTextCursor(_select(item.document(), 1, 2))
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_A,
            Qt.KeyboardModifier.NoModifier,
            'A',
        )
        item.keyPressEvent(event)

        containers = ruby_containers(item.document())
        self.assertEqual(item.toPlainText(), '東AX')
        self.assertEqual(len(containers), 1)
        self.assertEqual((containers[0].start, containers[0].units[0].text), (0, 'とう'))

    def test_non_text_keys_use_native_editor(self):
        for key, position, expected in (
            (Qt.Key.Key_Backspace, 1, 'a'),
            (Qt.Key.Key_Delete, 0, 'a'),
            (Qt.Key.Key_Escape, 1, 'aa'),
        ):
            with self.subTest(key=key):
                item = self._item(text='aa')
                scene = QGraphicsScene()
                scene.addItem(item)
                item.startEdit()
                cursor = QTextCursor(item.document())
                cursor.setPosition(position)
                item.setTextCursor(cursor)

                item.keyPressEvent(QKeyEvent(
                    QEvent.Type.KeyPress,
                    key,
                    Qt.KeyboardModifier.NoModifier,
                ))

                self.assertEqual(item.toPlainText(), expected)

    def test_navigation_keys_do_not_edit_or_add_undo_steps(self):
        for key, modifiers, selection, expected in (
            (Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier, (1, 2), (1, 1)),
            (Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier, (1, 2), (2, 2)),
            (Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier, (2, 2), (1, 2)),
        ):
            with self.subTest(key=key, modifiers=modifiers):
                item = self._item(text='abc')
                scene = QGraphicsScene()
                scene.addItem(item)
                item.startEdit()
                item.setTextCursor(_select(item.document(), *selection))
                item.document().clearUndoRedoStacks()
                item.updateUndoSteps()

                item.keyPressEvent(QKeyEvent(
                    QEvent.Type.KeyPress,
                    key,
                    modifiers,
                ))

                cursor = item.textCursor()
                self.assertEqual(item.toPlainText(), 'abc')
                self.assertEqual((cursor.position(), cursor.anchor()), expected)
                self.assertEqual(item.document().availableUndoSteps(), 0)
                self.assertEqual(item.document().pageCount(), 1)

    def test_ime_replacement_range_never_transfers_mono_reading(self):
        item = self._item(text='東京')
        apply_ruby(_select(item.document(), 0, 2), 'mono', 'とう きょう')
        scene = QGraphicsScene()
        scene.addItem(item)
        item.startEdit()
        cursor = QTextCursor(item.document())
        cursor.setPosition(2)
        item.setTextCursor(cursor)
        event = QInputMethodEvent()
        event.setCommitString('A', -1, 1)
        item.inputMethodEvent(event)

        containers = ruby_containers(item.document())
        self.assertEqual(item.toPlainText(), '東A')
        self.assertEqual(len(containers), 1)
        self.assertEqual((containers[0].start, containers[0].units[0].text), (0, 'とう'))

    def test_custom_rich_paste_obeys_group_replacement_and_break_rules(self):
        def custom_mime(text: str):
            source = QTextDocument(text)
            return create_rich_text_mime(
                _select(source, 0, source.characterCount() - 1)
            )

        target = self._item(text='東京')
        apply_ruby(
            _select(target.document(), 0, 2), 'group', 'とうきょう'
        )
        before = ruby_containers(target.document())[0]
        cursor = QTextCursor(target.document())
        cursor.setPosition(1)
        target.setTextCursor(cursor)
        self.assertTrue(target.insert_from_mime_data(custom_mime('X')))
        after = ruby_containers(target.document())
        self.assertEqual(target.toPlainText(), '東X京')
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0].container_id, before.container_id)
        self.assertEqual(after[0].units[0].unit_id, before.units[0].unit_id)
        self.assertEqual(after[0].length, 3)
        self.assertEqual(to_rich_text_html(target.document()).count('<ruby'), 1)

        broken = self._item(text='東京')
        apply_ruby(
            _select(broken.document(), 0, 2), 'group', 'とうきょう'
        )
        cursor = QTextCursor(broken.document())
        cursor.setPosition(1)
        broken.setTextCursor(cursor)
        self.assertTrue(broken.insert_from_mime_data(custom_mime('X\nY')))
        self.assertEqual(broken.toPlainText(), '東X\nY京')
        self.assertEqual(ruby_containers(broken.document()), ())

        mono = self._item(text='東京')
        apply_ruby(_select(mono.document(), 0, 2), 'mono', 'とう きょう')
        mono.setTextCursor(_select(mono.document(), 1, 2))
        self.assertTrue(mono.insert_from_mime_data(custom_mime('A')))
        containers = ruby_containers(mono.document())
        self.assertEqual(mono.toPlainText(), '東A')
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].units[0].text, 'とう')

    def test_break_insertion_removes_group_in_every_plain_text_path(self):
        def grouped_item() -> TextBlkItem:
            item = self._item(text='東京')
            apply_ruby(
                _select(item.document(), 0, 2), 'group', 'とうきょう'
            )
            cursor = QTextCursor(item.document())
            cursor.setPosition(1)
            item.setTextCursor(cursor)
            return item

        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            with self.subTest(key=key):
                key_item = grouped_item()
                key_item.keyPressEvent(QKeyEvent(
                    QEvent.Type.KeyPress,
                    key,
                    Qt.KeyboardModifier.NoModifier,
                    '\n',
                ))
                self.assertEqual(key_item.toPlainText(), '東\n京')
                self.assertEqual(ruby_containers(key_item.document()), ())
                self.assertNotIn(
                    '<ruby', to_rich_text_html(key_item.document())
                )

        paste_item = grouped_item()
        paste_item.insert_plain_text_at_cursor('\u2028')
        self.assertEqual(ruby_containers(paste_item.document()), ())

        ime_item = grouped_item()
        ime_event = QInputMethodEvent()
        ime_event.setCommitString('\n')
        ime_item.inputMethodEvent(ime_event)
        self.assertEqual(ruby_containers(ime_item.document()), ())

        side_item = grouped_item()
        propagate_user_edit(side_item, 1, 0, '\n')
        self.assertEqual(ruby_containers(side_item.document()), ())

    def test_mono_boundary_insertion_and_unit_deletion_are_structural(self):
        document = QTextDocument()
        document.setPlainText('東京')
        apply_ruby(_select(document, 0, 2), 'mono', 'とう きょう')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        prepare_ruby_insertion(cursor)
        cursor.insertText('X')
        self.assertEqual(document.toPlainText(), '東X京')
        self.assertEqual(len(ruby_containers(document)), 2)

        _select(document, 0, 1).removeSelectedText()
        remaining = ruby_containers(document)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].units[0].text, 'きょう')

    def test_ruby_and_tate_reject_overlap_in_both_application_orders(self):
        ruby_first = QTextDocument()
        ruby_first.setPlainText('東京')
        apply_ruby(_select(ruby_first, 0, 2), 'group', 'とうきょう')
        with self.assertRaisesRegex(
            RubyValidationError, 'Tate-chu-yoko cannot overlap Ruby'
        ):
            apply_text_combine_upright(_select(ruby_first, 0, 2), True)
        self.assertEqual(len(ruby_containers(ruby_first)), 1)

        tate_first = QTextDocument()
        tate_first.setPlainText('東京')
        apply_text_combine_upright(_select(tate_first, 0, 2), True)
        with self.assertRaisesRegex(
            RubyValidationError, 'Ruby cannot overlap Tate-chu-yoko'
        ):
            apply_ruby(_select(tate_first, 0, 2), 'group', 'とうきょう')
        self.assertEqual(ruby_containers(tate_first), ())

    def test_ordinary_formatting_preserves_container_and_unit_ids(self):
        document = QTextDocument()
        document.setPlainText('東京')
        apply_ruby(_select(document, 0, 2), 'group', 'とうきょう')
        before = ruby_containers(document)[0]
        modifier = QTextCharFormat()
        modifier.setFontItalic(True)
        _select(document, 1, 2).mergeCharFormat(modifier)
        after = ruby_containers(document)[0]
        self.assertEqual(before.container_id, after.container_id)
        self.assertEqual(before.units[0].unit_id, after.units[0].unit_id)

    def test_horizontal_long_ruby_expands_inline_positions_bounds_and_hit(self):
        item = self._item(text='東京X')
        apply_ruby(_select(item.document(), 0, 2), 'group', 'とてもながいとうきょう')
        item.layout.reLayoutEverything()
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        metric = item.layout._ruby_metrics[0][0]
        cell = item.layout._ruby_unit_cell(block, line, metric)
        x_after = line.cursorToX(2)
        if isinstance(x_after, (tuple, list)):
            x_after = x_after[0]

        self.assertGreater(cell.width(), metric.base_advance)
        self.assertAlmostEqual(cell.width(), metric.extent, delta=1.0)
        self.assertGreater(float(x_after), metric.base_advance)
        self.assertFalse(item.layout.annotation_ink_bounds().isEmpty())
        placement = item.layout._ruby_line_placements(block, line)[0]
        self.assertIn(
            item.layout.hitTest(
                placement.ink_bounds.center(), Qt.HitTestAccuracy.FuzzyHit
            ),
            (0, 1, 2),
        )
        self.assertFalse(item.layout.source_cursor_rect(1).isEmpty())

    def test_horizontal_wrap_keeps_group_and_pairs_indivisible(self):
        group = self._item(text='A東京B', bounds=(0, 0, 65, 220))
        apply_ruby(_select(group.document(), 1, 3), 'group', 'とうきょう')
        group.layout.reLayoutEverything()
        layout = group.document().firstBlock().layout()
        self.assertEqual(
            [(layout.lineAt(index).textStart(), layout.lineAt(index).textLength())
             for index in range(layout.lineCount())],
            [(0, 1), (1, 2), (3, 1)],
        )
        self.assertGreater(layout.lineAt(1).naturalTextWidth(), 65)

        mono = self._item(text='A東京B', bounds=(0, 0, 30, 260))
        apply_ruby(_select(mono.document(), 1, 3), 'mono', 'とう きょう')
        mono.layout.reLayoutEverything()
        layout = mono.document().firstBlock().layout()
        self.assertEqual(
            [(layout.lineAt(index).textStart(), layout.lineAt(index).textLength())
             for index in range(layout.lineCount())],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
        )

    def test_horizontal_wrap_settles_final_ruby_cell_membership(self):
        for width in (120, 150, 180):
            with self.subTest(width=width):
                item = self._item(
                    text='人人人人', bounds=(0, 0, width, 260)
                )
                apply_ruby(
                    _select(item.document(), 0, 4),
                    'mono',
                    'ながい ながい ながい ながい',
                )
                item.layout.reLayoutEverything()
                block = item.document().firstBlock()
                layout = block.layout()
                cell_extent = item.layout._ruby_metrics[0][0].extent
                first_count = max(
                    1, min(4, int((width + 0.02) // cell_extent))
                )
                expected = [(0, first_count)]
                if first_count < 4:
                    expected.append((first_count, 4 - first_count))
                self.assertEqual(
                    [
                        (
                            layout.lineAt(index).textStart(),
                            layout.lineAt(index).textLength(),
                        )
                        for index in range(layout.lineCount())
                    ],
                    expected,
                )
                for line_index in range(layout.lineCount()):
                    line = layout.lineAt(line_index)
                    cells = item.layout._ruby_line_placements(block, line)
                    self.assertLessEqual(
                        sum(cell.cell.width() for cell in cells),
                        width + 0.02,
                    )

    def test_horizontal_wrap_queries_only_grapheme_endpoints(self):
        item = self._item(
            text='😀e\N{COMBINING ACUTE ACCENT}X',
            bounds=(0, 0, 520, 220),
        )
        apply_ruby(
            _select(item.document(), 0, 4),
            'mono',
            'ながい ながい',
        )
        item.layout.reLayoutEverything()
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        with patch.object(
            item.layout,
            '_cursor_x',
            wraps=item.layout._cursor_x,
        ) as cursor_x:
            item.layout._settle_horizontal_ruby_wrap(
                block, line, item.layout._ruby_metrics[0]
            )
        queried = [call.args[-1] for call in cursor_x.call_args_list]
        self.assertEqual(queried, [0, 2, 4, 5])
        self.assertNotIn(1, queried)
        self.assertNotIn(3, queried)
        units = ruby_containers(item.document())[0].units
        layout = block.layout()
        for index in range(layout.lineCount()):
            line = layout.lineAt(index)
            line_end = line.textStart() + line.textLength()
            self.assertFalse(any(
                unit.start < line_end < unit.end for unit in units
            ))

    def test_paint_span_fragment_lookup_is_single_pass_for_many_mono_units(self):
        self.addCleanup(NATIVE_DOCUMENT_CACHE.clear)
        for count in (32, 256):
            with self.subTest(count=count):
                NATIVE_DOCUMENT_CACHE.clear()
                item = self._item(
                    text='A' * count,
                    bounds=(0, 0, count * 160, 180),
                )
                apply_ruby(
                    _select(item.document(), 0, count),
                    'mono',
                    ' '.join('とてもながいとう' for _ in range(count)),
                )
                item.layout.reLayoutEverything()
                block = item.document().firstBlock()
                line = block.layout().lineAt(0)
                with patch.object(
                    item.layout,
                    '_cursor_x',
                    wraps=item.layout._cursor_x,
                ) as cursor_x:
                    item.layout._settle_horizontal_ruby_wrap(
                        block, line, item.layout._ruby_metrics[0]
                    )
                self.assertLessEqual(cursor_x.call_count, count + 1)
                original = type(item.layout).fragment_format_ranges
                iterations = 0
                format_iterations = 0

                class CountingRanges(tuple):
                    def __iter__(self):
                        nonlocal iterations
                        iterations += 1
                        return super().__iter__()

                def counted(layout, block_number, start, end):
                    return CountingRanges(original(
                        layout, block_number, start, end
                    ))

                class CountingFormats(tuple):
                    def __iter__(self):
                        nonlocal format_iterations
                        format_iterations += 1
                        return super().__iter__()

                synthetic_formats = []
                for position in range(count):
                    char_format = QTextCharFormat()
                    char_format.setProperty(0x101010, position)
                    format_range = QTextLayout.FormatRange()
                    format_range.start = position
                    format_range.length = 1
                    format_range.format = char_format
                    synthetic_formats.append(format_range)
                formats = CountingFormats(synthetic_formats)
                self.assertEqual(len(formats), count)

                with patch.object(
                    type(item.layout), 'fragment_format_ranges', counted
                ):
                    spans = resolve_paint_spans(
                        block, line, formats
                    )
                self.assertGreaterEqual(len(spans), count)
                self.assertEqual(iterations, 1)
                self.assertEqual(format_iterations, 1)

                context = QAbstractTextDocumentLayout.PaintContext()
                image = QImage(
                    count * 160,
                    180,
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                image.fill(QColor(0, 0, 0, 0))
                painter = QPainter(image)
                try:
                    with patch(
                        'ballontranslator.ui.text_engine.horizontal_layout.'
                        'draw_slanted_line',
                        wraps=horizontal_layout.draw_slanted_line,
                    ) as paint_line:
                        item.layout.draw(painter, context)
                finally:
                    painter.end()
                self.assertEqual(paint_line.call_count, 1)
                self.assertLessEqual(
                    len(NATIVE_DOCUMENT_CACHE),
                    len(set('とてもながいとう')),
                )

    def test_horizontal_placement_spacing_and_selection_share_ruby_cell(self):
        centers = {}
        for position in ('over', 'under'):
            item = self._item(text='東京X')
            apply_letter_spacing(
                _select(item.document(), 0, 2), 1.4, vertical=False
            )
            apply_ruby(
                _select(item.document(), 0, 2), 'group', 'とうきょう', position
            )
            item.layout.reLayoutEverything()
            block = item.document().firstBlock()
            line = block.layout().lineAt(0)
            placement = item.layout._ruby_line_placements(block, line)[0]
            centers[position] = placement.ink_bounds.center().y()
            self.assertEqual(len(ruby_containers(item.document())), 1)

            context = QAbstractTextDocumentLayout.PaintContext()
            selection = QAbstractTextDocumentLayout.Selection()
            selection.cursor = _select(item.document(), 0, 2)
            selection.format = QTextCharFormat()
            selection.format.setBackground(QColor('#ffee58'))
            context.selections = [selection]
            image = QImage(360, 220, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(QColor(0, 0, 0, 0))
            painter = QPainter(image)
            try:
                item.layout.draw(painter, context)
            finally:
                painter.end()
            self.assertGreater(image.pixelColor(
                round(placement.cell.center().x()),
                round(placement.cell.center().y()),
            ).alpha(), 0)
        self.assertLess(centers['over'], centers['under'])

    def test_vertical_group_and_mono_keep_units_and_annotation_hit_geometry(self):
        for ruby_type, reading, unit_count in (
            ('group', 'とうきょう', 1),
            ('mono', 'とう きょう', 2),
        ):
            with self.subTest(ruby_type=ruby_type):
                item = self._item(vertical=True, text='東京X')
                apply_ruby(
                    _select(item.document(), 0, 2), ruby_type, reading
                )
                item.layout.reLayoutEverything()
                placements = item.layout._vertical_ruby_placements(
                    item.document().firstBlock()
                )
                self.assertEqual(len(placements), unit_count)
                self.assertTrue(all(
                    not placement.ink_bounds.isEmpty()
                    for placement in placements
                ))
                self.assertIn(
                    item.layout.hitTest(
                        placements[0].ink_bounds.center(),
                        Qt.HitTestAccuracy.FuzzyHit,
                    ),
                    (0, 1, 2) if ruby_type == 'group' else (0, 1),
                )

    def test_space_around_expands_latin_and_mixed_but_not_bopomofo(self):
        def group_item(vertical: bool, reading: str) -> TextBlkItem:
            item = self._item(
                vertical=vertical,
                text='是发达省份',
                bounds=(0, 0, 520, 520),
            )
            apply_ruby(
                _select(item.document(), 0, 5), 'group', reading
            )
            item.layout.reLayoutEverything()
            return item

        for vertical in (False, True):
            for reading in ('测试', 'AAA', 'A测试'):
                with self.subTest(vertical=vertical, reading=reading):
                    item = group_item(vertical, reading)
                    block = item.document().firstBlock()
                    metric = item.layout._ruby_metrics[0][0]
                    placement = (
                        item.layout._vertical_ruby_placements(block)[0]
                        if vertical
                        else item.layout._ruby_line_placements(
                            block, block.layout().lineAt(0)
                        )[0]
                    )
                    runs = tuple(reading)
                    self.assertEqual(
                        tuple(
                            run.source.document.toPlainText()
                            for run in placement.paint_runs
                        ),
                        runs,
                    )
                    font_metrics = QFontMetricsF(
                        placement.char_format.font()
                    )
                    advances = [
                        font_metrics.height()
                        if vertical
                        else font_metrics.horizontalAdvance(run)
                        for run in runs
                    ]
                    centers = [
                        geometry.bounds.center().y()
                        if vertical else geometry.bounds.center().x()
                        for geometry in placement.geometries
                    ]
                    if vertical:
                        cross_centers = [
                            geometry.bounds.center().x()
                            for geometry in placement.geometries
                        ]
                        for cross_center in cross_centers[1:]:
                            self.assertAlmostEqual(
                                cross_center, cross_centers[0], delta=0.02
                            )
                    extent = (
                        placement.cell.height()
                        if vertical else placement.cell.width()
                    )
                    expected_gap = (extent - sum(advances)) / len(runs)
                    self.assertGreater(expected_gap, 0.0)
                    for index in range(len(centers) - 1):
                        self.assertAlmostEqual(
                            centers[index + 1] - centers[index]
                            - (advances[index] + advances[index + 1]) / 2,
                            expected_gap,
                            delta=0.6,
                        )
                    expected_center = (
                        placement.cell.center().y()
                        + metric.annotation_center_offset
                        if vertical else placement.cell.center().x()
                    )
                    self.assertAlmostEqual(
                        (
                            centers[0] - advances[0] / 2
                            + centers[-1] + advances[-1] / 2
                        ) / 2,
                        expected_center,
                        delta=0.6,
                    )

            item = group_item(vertical, 'ㄏㄚ')
            block = item.document().firstBlock()
            placement = (
                item.layout._vertical_ruby_placements(block)[0]
                if vertical
                else item.layout._ruby_line_placements(
                    block, block.layout().lineAt(0)
                )[0]
            )
            runs = tuple(
                run.source.document.toPlainText()
                for run in placement.paint_runs
            )
            self.assertEqual(runs, ('ㄏ', 'ㄚ') if vertical else ('ㄏㄚ',))
            centers = [
                geometry.bounds.center().y()
                if vertical else geometry.bounds.center().x()
                for geometry in placement.geometries
            ]
            expected_center = (
                placement.cell.center().y()
                + item.layout._ruby_metrics[0][0].annotation_center_offset
                if vertical else placement.cell.center().x()
            )
            self.assertAlmostEqual(
                (centers[0] + centers[-1]) / 2,
                expected_center,
                delta=0.6,
            )
            if vertical:
                self.assertAlmostEqual(
                    centers[1] - centers[0],
                    QFontMetricsF(placement.char_format.font()).height(),
                    delta=0.6,
                )

    def test_vertical_group_and_mono_use_complete_selected_range(self):
        for ruby_type, reading in (
            ('group', '测试'),
            ('mono', 'A A A A A'),
        ):
            with self.subTest(ruby_type=ruby_type):
                item = self._item(
                    vertical=True,
                    text='是发达省份',
                    bounds=(0, 0, 520, 700),
                )
                item.startEdit()
                item.setTextCursor(_select(item.document(), 0, 5))
                item.setRuby(ruby_type, reading, 'over')
                block = item.document().firstBlock()
                container = ruby_containers(item.document())[0]
                placements = item.layout._vertical_ruby_placements(block)
                self.assertEqual((container.start, container.end), (0, 5))
                self.assertEqual(item.textCursor().selectedText(), '是发达省份')
                final_line = block.layout().lineForTextPosition(4)
                final_cells = item.layout._vertical_line_cells(
                    block, final_line.lineNumber()
                )
                self.assertEqual(final_cells[-1][:2], (4, 5))
                for metric, placement in zip(
                    item.layout._ruby_metrics[0], placements
                ):
                    local_start = metric.unit.start - block.position()
                    local_end = metric.unit.end - block.position()
                    first = block.layout().lineForTextPosition(local_start)
                    last = block.layout().lineForTextPosition(local_end - 1)
                    last_format = item.layout.get_char_fontfmt(
                        block.blockNumber(), local_end - 1
                    )
                    expected_center = (
                        first.y() + last.y() + last_format.tbr.height()
                    ) / 2
                    geometries = placement.geometries
                    self.assertLess(metric.annotation_center_offset, 0.0)
                    self.assertAlmostEqual(
                        (
                            geometries[0].bounds.center().y()
                            + geometries[-1].bounds.center().y()
                        ) / 2,
                        expected_center,
                        delta=0.6,
                    )
                if ruby_type == 'group':
                    self.assertAlmostEqual(
                        placements[0].cell.bottom(),
                        final_cells[-1][3],
                        delta=0.02,
                    )

    def test_vertical_mono_expands_latin_and_mixed_readings_per_pair(self):
        item = self._item(
            vertical=True,
            text='测试',
            bounds=(0, 0, 520, 700),
        )
        selection = _select(item.document(), 0, 2)
        apply_letter_spacing(selection, 4.0, vertical=True)
        apply_ruby(selection, 'mono', 'AAA A测试')
        item.layout.reLayoutEverything()
        block = item.document().firstBlock()
        placements = item.layout._vertical_ruby_placements(block)
        self.assertEqual(len(placements), 2)
        for placement, expected_runs in zip(
            placements, (tuple('AAA'), tuple('A测试'))
        ):
            self.assertEqual(
                tuple(
                    run.source.document.toPlainText()
                    for run in placement.paint_runs
                ),
                expected_runs,
            )
            centers = [
                run.geometry.bounds.center().y()
                for run in placement.paint_runs
            ]
            cross_centers = [
                run.geometry.bounds.center().x()
                for run in placement.paint_runs
            ]
            for cross_center in cross_centers[1:]:
                self.assertAlmostEqual(
                    cross_center, cross_centers[0], delta=0.02
                )
            advances = [
                QFontMetricsF(placement.char_format.font()).height()
            ] * len(expected_runs)
            expected_gap = (
                placement.cell.height() - sum(advances)
            ) / len(expected_runs)
            self.assertGreater(expected_gap, 0.0)
            for first, second in zip(centers, centers[1:]):
                self.assertAlmostEqual(
                    second - first - advances[0],
                    expected_gap,
                    delta=0.6,
                )

    def test_horizontal_latin_base_expands_native_geometry_and_paint(self):
        def kerning_item() -> TextBlkItem:
            item = self._item(text='AVX', bounds=(0, 0, 520, 220))
            char_format = QTextCharFormat()
            font = QFont('DejaVu Sans')
            font.setPointSizeF(24.0)
            char_format.setFont(font)
            _select(item.document(), 0, 3).mergeCharFormat(char_format)
            item.layout.reLayoutEverything()
            return item

        plain = kerning_item()
        plain_line = plain.document().firstBlock().layout().lineAt(0)
        plain_cursors = [
            plain.layout._cursor_x(plain_line, position)
            for position in range(4)
        ]
        metrics = QFontMetricsF(
            _select(plain.document(), 0, 1).charFormat().font()
        )
        self.assertLess(
            plain_cursors[1] - plain_cursors[0],
            metrics.horizontalAdvance('A'),
        )

        item = kerning_item()
        apply_ruby(
            _select(item.document(), 0, 2),
            'group',
            'とてもながいとうきょう',
        )
        item.layout.reLayoutEverything()
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        native_cursors = [
            item.layout._cursor_x(line, position) for position in range(4)
        ]
        metric = item.layout._ruby_metrics[0][0]
        cell = item.layout._ruby_unit_cell(block, line, metric)
        carets = [
            item.layout.source_cursor_rect(position).center().x()
            for position in range(4)
        ]
        self.assertEqual(metric.base_opportunity_ends, (1,))
        self.assertGreater(native_cursors[1], plain_cursors[1])
        self.assertAlmostEqual(
            native_cursors[1] - plain_cursors[1],
            metric.base_gap,
            delta=0.02,
        )
        self.assertAlmostEqual(cell.width(), metric.extent, delta=0.6)
        self.assertAlmostEqual(
            carets[3] - carets[2],
            plain_cursors[3] - plain_cursors[2],
            delta=0.02,
        )
        self.assertGreater(carets[2], native_cursors[2])

        image = QImage(520, 220, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        try:
            with patch.object(
                glyph_rendering,
                'glyph_geometry',
                wraps=glyph_rendering.glyph_geometry,
            ) as geometry:
                item.layout.draw(
                    painter, QAbstractTextDocumentLayout.PaintContext()
                )
        finally:
            painter.end()
        base_spans = {
            (call.args[1], call.args[2]) for call in geometry.call_args_list
        }
        self.assertIn((0, 1), base_spans)
        self.assertIn((1, 1), base_spans)

    def test_long_annotation_distributes_shorter_cjk_and_latin_base(self):
        reading = 'とてもながいとうきょう'
        for vertical in (False, True):
            for base in ('東京', 'AB'):
                with self.subTest(vertical=vertical, base=base):
                    plain = self._item(vertical=vertical, text=base)
                    item = self._item(
                        vertical=vertical,
                        text=base,
                        bounds=(0, 0, 520, 520),
                    )
                    apply_ruby(
                        _select(item.document(), 0, 2), 'group', reading
                    )
                    item.layout.reLayoutEverything()
                    block = item.document().firstBlock()
                    plain_block = plain.document().firstBlock()
                    metric = item.layout._ruby_metrics[0][0]
                    self.assertEqual(metric.base_opportunity_ends, (1,))
                    self.assertGreater(metric.base_gap, 0.0)
                    if vertical:
                        base_delta = (
                            block.layout().lineForTextPosition(1).y()
                            - block.layout().lineForTextPosition(0).y()
                        )
                        plain_delta = (
                            plain_block.layout().lineForTextPosition(1).y()
                            - plain_block.layout().lineForTextPosition(0).y()
                        )
                        cell = item.layout._vertical_ruby_unit_cell(
                            block, metric
                        )
                        cell_extent = cell.height()
                    else:
                        line = block.layout().lineAt(0)
                        plain_line = plain_block.layout().lineAt(0)
                        base_delta = (
                            item.layout._cursor_x(line, 1)
                            - item.layout._cursor_x(line, 0)
                        )
                        plain_delta = (
                            plain.layout._cursor_x(plain_line, 1)
                            - plain.layout._cursor_x(plain_line, 0)
                        )
                        cell = item.layout._ruby_unit_cell(block, line, metric)
                        cell_extent = cell.width()
                    self.assertAlmostEqual(
                        base_delta - plain_delta,
                        metric.base_gap,
                        delta=0.6,
                    )
                    self.assertAlmostEqual(
                        cell_extent, metric.extent, delta=0.6
                    )

    def test_vertical_item_resize_uses_actual_one_sided_annotation_width(self):
        for position in ('over', 'under'):
            with self.subTest(position=position):
                item = self._item(
                    vertical=True,
                    text='哈尔滨佛学院',
                    bounds=(0, 0, 344, 300),
                )
                apply_ruby(
                    _select(item.document(), 0, 6),
                    'group',
                    '哈佛',
                    position,
                )
                apply_emphasis(
                    _select(item.document(), 0, 6),
                    'filled sesame',
                    'over right' if position == 'over' else 'under left',
                )
                item.layout.reLayoutEverything()
                requested = item.absBoundingRect(qrect=True)
                requested.setWidth(1.0)
                item.setRect(requested, padding=False, repaint=False)

                block = item.document().firstBlock()
                metric = item.layout._ruby_metrics[0][0]
                base_cell = item.layout._vertical_ruby_base_cell(
                    block, metric
                )
                placement = item.layout._vertical_ruby_placements(block)[0]
                self.assertAlmostEqual(
                    metric.annotation_cross_extent,
                    max(
                        geometry.bounds.width()
                        for geometry in placement.geometries
                    ),
                    delta=0.02,
                )
                ruby_overflow = (
                    placement.ink_bounds.right() - base_cell.right()
                    if position == 'over'
                    else base_cell.left() - placement.ink_bounds.left()
                )
                record = item.layout._line_record(block, 0)
                occupied = (
                    record['right_margin']
                    if position == 'over' else record['left_margin']
                )
                empty = (
                    record['left_margin']
                    if position == 'over' else record['right_margin']
                )
                self.assertGreaterEqual(occupied, ruby_overflow)
                self.assertEqual(empty, 0.0)
                self.assertAlmostEqual(
                    item.logical_unpadded_rect().width(),
                    record['base_width'] + occupied,
                    delta=0.02,
                )
                self.assertLess(
                    item.logical_unpadded_rect().width(),
                    record['base_width'] + 2 * occupied - 1.0,
                )

                effect = item.fontformat.deepcopy()
                effect.stroke_width = 0.2
                item.set_fontformat(effect)
                self.assertGreater(item.padding(), 0.0)
                self.assertGreaterEqual(
                    item.boundingRect().width(),
                    item.logical_unpadded_rect().width() + 2 * item.padding(),
                )

    def test_vertical_wrap_and_both_placements_use_settled_columns(self):
        group = self._item(
            vertical=True, text='A東京B', bounds=(0, 0, 300, 65)
        )
        apply_ruby(_select(group.document(), 1, 3), 'group', 'とうきょう')
        group.layout.reLayoutEverything()
        layout = group.document().firstBlock().layout()
        self.assertEqual(layout.lineForTextPosition(1).x(), layout.lineForTextPosition(2).x())
        self.assertNotEqual(layout.lineForTextPosition(0).x(), layout.lineForTextPosition(1).x())
        self.assertGreater(group.layout.max_height, 65)

        mono = self._item(
            vertical=True, text='東京', bounds=(0, 0, 240, 30)
        )
        apply_ruby(_select(mono.document(), 0, 2), 'mono', 'とう きょう')
        mono.layout.reLayoutEverything()
        layout = mono.document().firstBlock().layout()
        self.assertNotEqual(layout.lineForTextPosition(0).x(), layout.lineForTextPosition(1).x())

        sides = {}
        for position in ('over', 'under'):
            item = self._item(vertical=True, text='東京X')
            apply_letter_spacing(
                _select(item.document(), 0, 2), 1.3, vertical=True
            )
            apply_ruby(
                _select(item.document(), 0, 2), 'group', 'とうきょう', position
            )
            item.layout.reLayoutEverything()
            placement = item.layout._vertical_ruby_placements(
                item.document().firstBlock()
            )[0]
            metric = item.layout._ruby_metrics[0][0]
            base_cell = item.layout._vertical_ruby_base_cell(
                item.document().firstBlock(), metric
            )
            self.assertAlmostEqual(
                placement.cell.height(),
                max(base_cell.height(), metric.annotation_advance),
                delta=1.1,
            )
            sides[position] = placement.ink_bounds.center().x()
        self.assertGreater(sides['over'], sides['under'])

    def test_ruby_and_emphasis_accumulate_same_side_margin(self):
        plain = self._item(text='東京')
        ruby = self._item(text='東京')
        both = self._item(text='東京')
        for item in (ruby, both):
            apply_ruby(
                _select(item.document(), 0, 2), 'group', 'とうきょう'
            )
            item.layout.reLayoutEverything()
        apply_emphasis(
            _select(both.document(), 0, 2), 'filled dot', 'over right'
        )
        both.layout.reLayoutEverything()

        plain_y = plain.document().firstBlock().layout().lineAt(0).y()
        ruby_y = ruby.document().firstBlock().layout().lineAt(0).y()
        both_y = both.document().firstBlock().layout().lineAt(0).y()
        self.assertGreater(ruby_y, plain_y)
        self.assertGreater(both_y, ruby_y)

    def test_ruby_uses_cached_native_documents_without_mutating_live_document(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self._item(vertical=vertical, text='学院')
                effect = item.fontformat.deepcopy()
                effect.stroke_width = 0.2
                item.set_fontformat(effect)
                cursor = _select(item.document(), 0, 2)
                decorated = QTextCharFormat()
                decorated.setFontUnderline(True)
                decorated.setBackground(QColor('#70ff90'))
                cursor.mergeCharFormat(decorated)
                apply_ruby(cursor, 'group', '佛佛', 'over')
                item.layout.reLayoutEverything()

                live_document = item.document()
                revision = live_document.revision()
                text = live_document.toPlainText()
                html = to_rich_text_html(live_document)
                stroke_context = item.effect_renderer._stroke_paint_context()
                NATIVE_DOCUMENT_CACHE.clear()
                self.addCleanup(NATIVE_DOCUMENT_CACHE.clear)
                block = live_document.firstBlock()
                placements = (
                    item.layout._vertical_ruby_placements(
                        block, stroke_context
                    )
                    if vertical
                    else item.layout._ruby_line_placements(
                        block,
                        block.layout().lineAt(0),
                        stroke_context,
                    )
                )
                self.assertEqual(len(placements), 1)
                placement = placements[0]
                self.assertEqual(len(placement.paint_runs), 2)
                self.assertIs(
                    placement.paint_runs[0].source,
                    placement.paint_runs[1].source,
                )

                image = QImage(
                    800,
                    500,
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                painted_documents = []
                native_draw = QTextDocument.drawContents

                def record_document(document, *args):
                    painted_documents.append(document)
                    return native_draw(document, *args)

                try:
                    with patch.object(
                        QTextDocument,
                        'drawContents',
                        new=record_document,
                    ), patch.object(
                        glyph_rendering,
                        '_draw_dilated_path_stroke',
                    ) as generic_dilation:
                        draw_ruby_placement(painter, placement)
                finally:
                    painter.end()

                source = placement.paint_runs[0].source
                self.assertEqual(
                    painted_documents,
                    [source.document, source.document],
                )
                self.assertIsNot(source.document, live_document)
                self.assertEqual(source.document.toPlainText(), '佛')
                self.assertEqual(source.document.documentMargin(), 0.0)
                self.assertEqual(len(NATIVE_DOCUMENT_CACHE), 1)
                generic_dilation.assert_not_called()
                source_cursor = QTextCursor(source.document)
                source_cursor.select(QTextCursor.SelectionType.Document)
                source_format = source_cursor.charFormat()
                self.assertFalse(source_format.font().underline())
                self.assertEqual(
                    source_format.background().style(),
                    Qt.BrushStyle.NoBrush,
                )
                expected_outline = (
                    stroke_context.selections[0]
                    .format.textOutline().widthF()
                    * RUBY_FONT_SCALE
                )
                self.assertAlmostEqual(
                    source_format.textOutline().widthF(),
                    expected_outline,
                )
                self.assertEqual(live_document.revision(), revision)
                self.assertEqual(live_document.toPlainText(), text)
                self.assertEqual(to_rich_text_html(live_document), html)
                if vertical:
                    with patch.object(
                        glyph_rendering,
                        '_draw_dilated_path_stroke',
                    ) as neutral_generic_dilation:
                        item.repaint_background()
                    neutral_generic_dilation.assert_not_called()

    def test_vertical_glyph_slant_stroke_keeps_ruby_out_of_dilation(self):
        block = TextBlock([0, 0, 220, 260])
        block._bounding_rect = [0, 0, 220, 260]
        block.translation = '学'
        block.vertical = True
        block.fontformat.font_size = 64.0
        block.fontformat.stroke_width = 0.25
        block.fontformat.text_transform = TextTransformStack((), 11.0)
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.addCleanup(scene.removeItem, item)
        apply_ruby(_select(item.document(), 0, 1), 'group', '佛', 'over')
        item.layout.reLayoutEverything()
        self.app.processEvents()

        live_document = item.document()
        document_block = live_document.firstBlock()
        normal = item.layout._vertical_ruby_placements(document_block)[0]
        normal_bounds = QRectF(normal.ink_bounds)
        surface_rect = QRectF(item.boundingRect())
        NATIVE_DOCUMENT_CACHE.clear()
        self.addCleanup(NATIVE_DOCUMENT_CACHE.clear)

        mask_flags = []
        dilation_inputs = []
        native_documents = []
        annotation_calls = []
        draw_mask = item.geometry_controller.draw_layout_selection_mask
        draw_annotations = item.geometry_controller.draw_layout_annotations
        dilate = effect_rendering.cv2.dilate
        native_draw = QTextDocument.drawContents

        def record_mask(painter, context, *, include_annotations=True):
            mask_flags.append(include_annotations)
            return draw_mask(
                painter,
                context,
                include_annotations=include_annotations,
            )

        def record_annotations(painter, context):
            annotation_calls.append(True)
            return draw_annotations(painter, context)

        def record_dilate(source, kernel, *args, **kwargs):
            dilation_inputs.append(source.copy())
            return dilate(source, kernel, *args, **kwargs)

        def record_document(document, *args):
            native_documents.append(document)
            return native_draw(document, *args)

        with patch.object(
            item.geometry_controller,
            'draw_layout_selection_mask',
            new=record_mask,
        ), patch.object(
            item.geometry_controller,
            'draw_layout_annotations',
            new=record_annotations,
        ), patch.object(
            effect_rendering.cv2,
            'dilate',
            new=record_dilate,
        ), patch.object(
            QTextDocument,
            'drawContents',
            new=record_document,
        ), patch.object(
            glyph_rendering,
            '_draw_dilated_path_stroke',
        ) as generic_dilation:
            item.repaint_background()

        def alpha_region(alpha, bounds: QRectF, padding: float = 0.0):
            local = QRectF(bounds).translated(-surface_rect.topLeft())
            local.adjust(-padding, -padding, padding, padding)
            pixels = local.toAlignedRect()
            left = max(0, pixels.left())
            top = max(0, pixels.top())
            right = min(alpha.shape[1], pixels.right() + 1)
            bottom = min(alpha.shape[0], pixels.bottom() + 1)
            return alpha[top:bottom, left:right]

        self.assertTrue(mask_flags)
        self.assertFalse(any(mask_flags))
        self.assertEqual(len(annotation_calls), 1)
        self.assertTrue(dilation_inputs)
        self.assertTrue(all(alpha.any() for alpha in dilation_inputs))
        self.assertTrue(all(
            not alpha_region(alpha, normal_bounds).any()
            for alpha in dilation_inputs
        ))
        generic_dilation.assert_not_called()

        stroke_context = item.effect_renderer._stroke_paint_context()
        outlined = item.layout._vertical_ruby_placements(
            document_block, stroke_context
        )[0]
        source = outlined.paint_runs[0].source
        source_cursor = QTextCursor(source.document)
        source_cursor.select(QTextCursor.SelectionType.Document)
        expected_width = (
            stroke_context.selections[0].format.textOutline().widthF()
            * RUBY_FONT_SCALE
        )
        self.assertAlmostEqual(
            source_cursor.charFormat().textOutline().widthF(),
            expected_width,
        )
        self.assertEqual(
            sum(document is source.document for document in native_documents),
            1,
        )
        final_alpha = pixmap2ndarray(
            item.effect_renderer.background_pixmap,
            keep_alpha=True,
        )[..., 3]
        self.assertTrue(alpha_region(
            final_alpha, outlined.ink_bounds, padding=expected_width
        ).any())
        local_ink = QRectF(outlined.ink_bounds).translated(
            -surface_rect.topLeft()
        )
        self.assertGreater(local_ink.left(), 0.0)
        self.assertGreater(local_ink.top(), 0.0)
        self.assertLess(local_ink.right(), final_alpha.shape[1])
        self.assertLess(local_ink.bottom(), final_alpha.shape[0])

    def test_effects_glyph_slant_mode_switch_and_paint_share_ruby_geometry(self):
        def pixels_in(image, scene_rect, target):
            left = max(0, int((target.left() - scene_rect.left())
                              * image.width() / scene_rect.width()))
            right = min(image.width(), int((target.right() - scene_rect.left())
                                           * image.width() / scene_rect.width()) + 1)
            top = max(0, int((target.top() - scene_rect.top())
                             * image.height() / scene_rect.height()))
            bottom = min(image.height(), int((target.bottom() - scene_rect.top())
                                             * image.height() / scene_rect.height()) + 1)
            return [
                image.pixelColor(x, y)
                for y in range(top, bottom)
                for x in range(left, right)
                if image.pixelColor(x, y).alpha()
            ]

        item = self._item(text='東京X')
        apply_ruby(_select(item.document(), 0, 2), 'group', 'とうきょう')
        item.layout.reLayoutEverything()
        neutral_bounds = item.layout.annotation_ink_bounds()
        transform = TextTransformStack(
            item.fontformat.text_transform.transforms, 18.0
        )
        item.set_text_transform(transform)
        slanted_bounds = item.layout.annotation_ink_bounds()
        self.assertFalse(slanted_bounds.isEmpty())
        self.assertNotEqual(neutral_bounds, slanted_bounds)

        shadow = item.fontformat.deepcopy()
        shadow.stroke_width = 0.2
        shadow.shadow_strength = 1.0
        shadow.shadow_radius = 0.2
        shadow.gradient_start_color = [255, 80, 80]
        shadow.gradient_end_color = [60, 100, 255]
        item.set_fontformat(shadow)
        item.setGradientEnabled(True)
        item.setOpacity(0.65)
        scene = QGraphicsScene()
        scene.addItem(item)
        scene.setSceneRect(item.boundingRect())
        effect_image = QImage(
            420, 240, QImage.Format.Format_ARGB32_Premultiplied
        )
        effect_image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(effect_image)
        try:
            scene.render(painter)
        finally:
            painter.end()
        ruby_bounds = item.layout.annotation_ink_bounds()
        ruby_pixels = pixels_in(effect_image, scene.sceneRect(), ruby_bounds)
        self.assertTrue(ruby_pixels)
        opaque_gradient_pixels = [
            color for color in ruby_pixels
            if color.alpha() > 100 and color.red() > 100 and color.blue() > 40
        ]
        reds = [color.red() for color in opaque_gradient_pixels]
        blues = [color.blue() for color in opaque_gradient_pixels]
        self.assertGreater(max(reds) - min(reds), 10)
        self.assertGreater(max(blues) - min(blues), 10)
        item.setOpacity(1.0)
        opaque_image = QImage(
            420, 240, QImage.Format.Format_ARGB32_Premultiplied
        )
        opaque_image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(opaque_image)
        try:
            scene.render(painter)
        finally:
            painter.end()
        opaque_ruby_pixels = pixels_in(
            opaque_image, scene.sceneRect(), ruby_bounds
        )
        self.assertLess(
            sum(color.alpha() for color in ruby_pixels),
            sum(color.alpha() for color in opaque_ruby_pixels),
        )
        item.setOpacity(0.65)
        effect_ring = pixels_in(
            effect_image, scene.sceneRect(), ruby_bounds.adjusted(-8, -8, 8, 8)
        )
        self.assertGreater(len(effect_ring), len(ruby_pixels))

        item.set_text_transform(TextTransformStack((
            ProjectiveTextTransform(1.08, 0.95, 5.0),
            BendTextTransform(0.25),
        ), 18.0))
        self.assertIsNotNone(item.geometry_controller.visual_mapper)
        self.assertAlmostEqual(item.opacity(), 0.65)

        scene.setSceneRect(item.boundingRect())
        image = QImage(420, 240, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        try:
            scene.render(painter)
        finally:
            painter.end()
        transformed_ruby = item.geometry_controller.visual_mapper.visual_bounds(
            item.layout.annotation_ink_bounds()
        )
        self.assertTrue(pixels_in(image, scene.sceneRect(), transformed_ruby))

        item.setVertical(True)
        self.assertEqual(len(ruby_containers(item.document())), 1)
        self.assertFalse(item.layout.annotation_ink_bounds().isEmpty())
        scene.setSceneRect(item.boundingRect())
        vertical_image = QImage(
            420, 240, QImage.Format.Format_ARGB32_Premultiplied
        )
        vertical_image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(vertical_image)
        try:
            scene.render(painter)
        finally:
            painter.end()
        self.assertTrue(any(
            vertical_image.pixelColor(x, y).alpha() > 0
            for y in range(0, vertical_image.height(), 8)
            for x in range(0, vertical_image.width(), 8)
        ))

    def test_panel_editor_emits_apply_shows_errors_and_exposes_remove(self):
        group = RubyFuriganaGroup()
        edits = []
        removals = []
        group.apply_requested.connect(
            lambda ruby_type, text, position: edits.append(
                (ruby_type, text, position)
            )
        )
        group.remove_requested.connect(lambda: removals.append(True))
        group.set_state(
            'mono', 'とう', 'over',
            editable=False,
        )
        group.apply_button.click()
        self.assertEqual(edits, [('mono', 'とう', 'over')])
        with patch(
            'ballontranslator.ui.text_engine.formatting.advanced.QMessageBox.warning'
        ) as warning:
            group.set_error('Invalid Ruby')
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[2], 'Invalid Ruby')
        group.text_edit.setText('とう きょう')
        group._emit_apply()
        self.assertEqual(
            edits,
            [('mono', 'とう', 'over'), ('mono', 'とう きょう', 'over')],
        )
        group.set_state(
            'group', 'とうきょう', 'under',
            editable=True,
        )
        self.assertTrue(group.remove_button.isEnabled())
        group.remove_button.click()
        self.assertEqual(removals, [True])


if __name__ == '__main__':
    unittest.main()
