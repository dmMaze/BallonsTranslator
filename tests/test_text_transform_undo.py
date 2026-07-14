import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from qtpy.QtWidgets import QApplication, QGraphicsScene, QUndoStack
except ImportError:
    from qtpy.QtGui import QUndoStack
    from qtpy.QtWidgets import QApplication, QGraphicsScene

from qtpy.QtGui import QColor, QFont, QTextCharFormat, QTextCursor, QTextDocument

from ballontranslator.ui.textedit_commands import (
    ApplyFontformatCommand,
    SetTextTransformCommand,
    TextItemEditCommand,
    utf16_code_unit_length,
)
from ballontranslator.ui.textitem import TextBlkItem
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


_APP = QApplication.instance() or QApplication([])


class FakeTextItem:
    def __init__(self, transform):
        self.transform = transform
        self.calls = []
        self.html = '<b>unchanged</b>'
        self.rect = (1, 2, 3, 4)
        self.pos = (5, 6)

    def set_text_transform(
        self,
        horizontal_scale,
        vertical_scale,
        slant_angle,
        *,
        preview=False,
    ):
        self.transform = (horizontal_scale, vertical_scale, slant_angle)
        self.calls.append((self.transform, preview))


class FakeTransEdit:
    def __init__(self):
        self._document = QTextDocument()

    def document(self):
        return self._document

    def undo(self):
        self._document.undo()

    def redo(self):
        self._document.redo()

    def updateUndoSteps(self):
        self.old_undo_steps = self._document.availableUndoSteps()


class TrackingShapeControl:
    def __init__(self, item):
        self.blk_item = item
        self.refresh_count = 0

    def updateBoundingRect(self):
        self.refresh_count += 1


class SetTextTransformCommandTest(unittest.TestCase):
    def test_multi_item_command_is_atomic_and_restores_only_transforms(self):
        items = [
            FakeTextItem((1.0, 1.0, 0.0)),
            FakeTextItem((0.5, 2.0, -10.0)),
        ]
        snapshots = [(item.html, item.rect, item.pos) for item in items]
        refreshes = []
        command = SetTextTransformCommand(
            items,
            [(1.0, 1.0, -0.0), (0.5, 2.0, -10.0)],
            [(1.23456789, 0.01, 90.0), (5.0, 3.0, -50.0)],
            lambda: refreshes.append('refresh'),
        )

        stack = QUndoStack()
        stack.push(command)

        self.assertEqual(stack.count(), 1)
        self.assertEqual(command.childCount(), 0)
        self.assertEqual(items[0].transform, (1.234568, 0.1, 45.0))
        self.assertEqual(items[1].transform, (4.0, 3.0, -45.0))
        self.assertEqual(refreshes, ['refresh'])

        stack.undo()
        self.assertEqual(items[0].transform, (1.0, 1.0, 0.0))
        self.assertEqual(items[1].transform, (0.5, 2.0, -10.0))
        self.assertEqual(refreshes, ['refresh', 'refresh'])

        stack.redo()
        self.assertEqual(items[0].transform, (1.234568, 0.1, 45.0))
        self.assertEqual(items[1].transform, (4.0, 3.0, -45.0))
        self.assertEqual(refreshes, ['refresh', 'refresh', 'refresh'])
        self.assertTrue(all(not preview for item in items for _, preview in item.calls))
        self.assertEqual(
            [(item.html, item.rect, item.pos) for item in items], snapshots
        )

    def test_create_returns_none_when_normalized_values_match(self):
        item = FakeTextItem((4.0, 0.1, -45.0))

        command = SetTextTransformCommand.create(
            [item],
            [(4.0, 0.1, -45.0)],
            [(99.0, 0.0, -99.0)],
        )

        self.assertIsNone(command)
        self.assertEqual(item.calls, [])

    def test_rejects_mismatched_per_item_state(self):
        with self.assertRaisesRegex(ValueError, 'same length'):
            SetTextTransformCommand(
                [FakeTextItem((1.0, 1.0, 0.0))],
                [],
                [(1.0, 1.0, 0.0)],
            )

    def test_undo_then_new_transform_clears_redo_branch(self):
        item = FakeTextItem((1.0, 1.0, 0.0))
        stack = QUndoStack()
        stack.push(
            SetTextTransformCommand(
                [item], [(1.0, 1.0, 0.0)], [(1.5, 1.0, 0.0)]
            )
        )
        stack.undo()
        self.assertTrue(stack.canRedo())

        stack.push(
            SetTextTransformCommand(
                [item], [(1.0, 1.0, 0.0)], [(1.0, 0.75, 5.0)]
            )
        )
        self.assertFalse(stack.canRedo())
        self.assertEqual(stack.count(), 1)
        self.assertEqual(item.transform, (1.0, 0.75, 5.0))

    def test_normalized_noop_never_enters_stack_or_calls_item(self):
        item = FakeTextItem((1.2, 1.0, 0.0))
        stack = QUndoStack()
        command = SetTextTransformCommand.create(
            [item], [(1.2, 1.0, 0.0)], [(1.20000001, 1.0, -0.0)]
        )
        self.assertIsNone(command)
        self.assertEqual(stack.count(), 0)
        self.assertEqual(item.calls, [])


class Utf16CodeUnitLengthTest(unittest.TestCase):
    def test_supplementary_characters_use_two_qt_positions(self):
        self.assertEqual(utf16_code_unit_length('plain'), 5)
        self.assertEqual(utf16_code_unit_length('\U0001f600'), 2)
        self.assertEqual(utf16_code_unit_length('A\U0001f600e\u0301\ufe0f'), 6)


class ApplyFontformatCommandTest(unittest.TestCase):
    def test_prior_text_edit_survives_whole_format_undo_redo_chain(self):
        block = TextBlock(
            xyxy=[0, 0, 160, 70],
            _bounding_rect=[0, 0, 160, 70],
            translation='before',
            fontformat=FontFormat(),
        )
        item = TextBlkItem(block)
        scene = QGraphicsScene()
        scene.addItem(item)
        edit = FakeTransEdit()
        edit.document().setPlainText('before')
        item.document().clearUndoRedoStacks()
        edit.document().clearUndoRedoStacks()
        item.updateUndoSteps()
        edit.updateUndoSteps()

        item_cursor = item.textCursor()
        item_cursor.movePosition(QTextCursor.MoveOperation.End)
        item_cursor.insertText('!')
        edit_cursor = QTextCursor(edit.document())
        edit_cursor.movePosition(QTextCursor.MoveOperation.End)
        edit_cursor.insertText('!')
        text_steps = item.document().availableUndoSteps()
        self.assertGreater(text_steps, 0)

        stack = QUndoStack()
        stack.push(TextItemEditCommand(item, edit, text_steps))
        edited_html = item.toHtml()
        edited_rect = item.absBoundingRect(qrect=True)
        edited_position = item.pos()
        edited_padding = item.padding()
        target = item.fontformat.deepcopy()
        target.vertical = True
        target.bold = True
        target.font_size = 41
        target.stroke_width = 0.14
        target.shadow_radius = 0.4
        target.shadow_strength = 0.8
        target.shadow_offset = [-3.0, 4.0]
        target.gradient_enabled = True
        target.gradient_start_color = [12, 34, 56]
        target.gradient_end_color = [210, 180, 90]
        target.horizontal_scale = 1.4
        target.vertical_scale = 0.7
        target.slant_angle = -12.0
        stack.push(ApplyFontformatCommand([item], [edit], target))

        self.assertEqual(stack.count(), 2)
        self.assertEqual(item.toPlainText(), 'before!')
        self.assertTrue(item.fontformat.vertical)

        stack.undo()
        self.assertEqual(item.toHtml(), edited_html)
        self.assertFalse(item.fontformat.vertical)
        self.assertEqual(item.absBoundingRect(qrect=True), edited_rect)
        self.assertEqual(item.pos(), edited_position)
        self.assertEqual(item.padding(), edited_padding)
        self.assertEqual(item.document().availableUndoSteps(), text_steps)
        stack.undo()
        self.assertEqual(item.toPlainText(), 'before')
        self.assertEqual(edit.document().toPlainText(), 'before')

        stack.redo()
        self.assertEqual(item.toPlainText(), 'before!')
        self.assertEqual(edit.document().toPlainText(), 'before!')
        stack.redo()
        self.assertTrue(item.fontformat.vertical)
        self.assertEqual(item.fontformat.text_transform, (1.4, 0.7, -12.0))

    def test_empty_documents_restore_exact_cursor_and_block_formats(self):
        for formatted in (False, True):
            with self.subTest(formatted=formatted):
                block = TextBlock(
                    xyxy=[20, 30, 140, 90],
                    _bounding_rect=[20, 30, 120, 60],
                    translation='',
                    fontformat=FontFormat(stroke_width=0.02),
                )
                item = TextBlkItem(block)
                scene = QGraphicsScene()
                scene.addItem(item)
                if formatted:
                    cursor = item.textCursor()
                    block_format = QTextCharFormat(cursor.blockCharFormat())
                    block_format.setFontUnderline(True)
                    block_format.setForeground(QColor(91, 42, 17))
                    cursor.setBlockCharFormat(block_format)
                    char_format = QTextCharFormat(cursor.charFormat())
                    char_format.setFontItalic(True)
                    char_format.setFontLetterSpacing(63.0)
                    cursor.setCharFormat(char_format)
                    item.setTextCursor(cursor)
                item.document().clearUndoRedoStacks()

                cursor = item.textCursor()
                before = (
                    item.toHtml(),
                    item.fontformat.deepcopy(),
                    item.absBoundingRect(qrect=True),
                    item.pos(),
                    item.padding(),
                    QFont(item.document().defaultFont()),
                    QTextCharFormat(cursor.charFormat()),
                    QTextCharFormat(cursor.blockCharFormat()),
                    item.document().availableUndoSteps(),
                )
                target = FontFormat(
                    font_family='Arial',
                    font_size=38,
                    bold=True,
                    alignment=2,
                    vertical=True,
                    stroke_width=0.15,
                    shadow_radius=0.3,
                    shadow_strength=0.75,
                    shadow_offset=[-2.0, 3.0],
                    gradient_enabled=True,
                    gradient_start_color=[10, 20, 30],
                    gradient_end_color=[220, 210, 200],
                    horizontal_scale=1.6,
                    vertical_scale=0.65,
                    slant_angle=-13.0,
                )
                stack = QUndoStack()
                stack.push(
                    ApplyFontformatCommand([item], [FakeTransEdit()], target)
                )
                self.assertEqual(stack.count(), 1)
                self.assertEqual(item.toPlainText(), '')
                after = (
                    item.toHtml(),
                    item.fontformat.deepcopy(),
                    item.absBoundingRect(qrect=True),
                    item.pos(),
                    item.padding(),
                    QTextCharFormat(item.textCursor().charFormat()),
                    QTextCharFormat(item.textCursor().blockCharFormat()),
                )

                stack.undo()
                restored_cursor = item.textCursor()
                self.assertEqual(item.toHtml(), before[0])
                self.assertEqual(item.fontformat, before[1])
                self.assertEqual(item.absBoundingRect(qrect=True), before[2])
                self.assertEqual(item.pos(), before[3])
                self.assertEqual(item.padding(), before[4])
                self.assertEqual(item.document().defaultFont(), before[5])
                self.assertEqual(restored_cursor.charFormat(), before[6])
                self.assertEqual(restored_cursor.blockCharFormat(), before[7])
                self.assertEqual(
                    item.document().availableUndoSteps(), before[8]
                )

                stack.redo()
                self.assertEqual(item.toHtml(), after[0])
                self.assertEqual(item.fontformat, after[1])
                self.assertEqual(item.absBoundingRect(qrect=True), after[2])
                self.assertEqual(item.pos(), after[3])
                self.assertEqual(item.padding(), after[4])
                self.assertEqual(item.textCursor().charFormat(), after[5])
                self.assertEqual(item.textCursor().blockCharFormat(), after[6])

    def test_stale_uniform_spacing_is_applied_then_becomes_noop(self):
        block = TextBlock(
            xyxy=[0, 0, 140, 60],
            _bounding_rect=[0, 0, 140, 60],
            translation='stale spacing',
            fontformat=FontFormat(letter_spacing=1.15),
        )
        item = TextBlkItem(block)
        scene = QGraphicsScene()
        scene.addItem(item)
        cursor = QTextCursor(item.document())
        cursor.select(QTextCursor.SelectionType.Document)
        stale_format = QTextCharFormat()
        stale_format.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
        stale_format.setFontLetterSpacing(50.0)
        cursor.mergeCharFormat(stale_format)
        item.document().clearUndoRedoStacks()
        target = item.fontformat.deepcopy()
        stack = QUndoStack()

        stack.push(ApplyFontformatCommand([item], [FakeTransEdit()], target))
        self.assertEqual(stack.count(), 1)
        block_cursor = item.document().firstBlock()
        fragment = block_cursor.begin().fragment()
        self.assertAlmostEqual(
            fragment.charFormat().fontLetterSpacing(), 115.0, places=5
        )

        history_steps = item.document().availableUndoSteps()
        stack.push(ApplyFontformatCommand([item], [FakeTransEdit()], target))
        self.assertEqual(stack.count(), 1)
        self.assertEqual(item.document().availableUndoSteps(), history_steps)

    def test_stale_empty_block_spacing_is_not_misclassified_as_noop(self):
        for text in ('', 'A\n'):
            with self.subTest(text=repr(text)):
                block = TextBlock(
                    xyxy=[0, 0, 140, 60],
                    _bounding_rect=[0, 0, 140, 60],
                    translation=text,
                    fontformat=FontFormat(letter_spacing=1.15),
                )
                item = TextBlkItem(block)
                scene = QGraphicsScene()
                scene.addItem(item)
                target_block = item.document().lastBlock()
                cursor = QTextCursor(target_block)
                stale_format = QTextCharFormat(cursor.charFormat())
                stale_format.setFontLetterSpacingType(
                    QFont.SpacingType.PercentageSpacing
                )
                stale_format.setFontLetterSpacing(50.0)
                cursor.setBlockCharFormat(stale_format)
                cursor.setCharFormat(stale_format)
                item.setTextCursor(cursor)
                item.document().clearUndoRedoStacks()
                target = item.fontformat.deepcopy()
                stack = QUndoStack()

                stack.push(
                    ApplyFontformatCommand([item], [FakeTransEdit()], target)
                )
                self.assertEqual(stack.count(), 1)
                restored = QTextCursor(item.document().lastBlock())
                self.assertAlmostEqual(
                    restored.charFormat().fontLetterSpacing(), 115.0, places=5
                )
                self.assertAlmostEqual(
                    item.document()
                    .lastBlock()
                    .charFormat()
                    .fontLetterSpacing(),
                    115.0,
                    places=5,
                )

                history_steps = item.document().availableUndoSteps()
                stack.push(
                    ApplyFontformatCommand([item], [FakeTransEdit()], target)
                )
                self.assertEqual(stack.count(), 1)
                self.assertEqual(
                    item.document().availableUndoSteps(), history_steps
                )

    def test_identical_gradient_target_is_noop_after_first_apply(self):
        block = TextBlock(
            xyxy=[0, 0, 160, 70],
            _bounding_rect=[0, 0, 160, 70],
            translation='gradient target',
            fontformat=FontFormat(),
        )
        item = TextBlkItem(block)
        scene = QGraphicsScene()
        scene.addItem(item)
        target = item.fontformat.deepcopy()
        target.gradient_enabled = True
        target.gradient_start_color = [3, 40, 90]
        target.gradient_end_color = [240, 170, 20]
        target.gradient_angle = 27.5
        target.gradient_size = 0.8
        stack = QUndoStack()

        stack.push(ApplyFontformatCommand([item], [FakeTransEdit()], target))
        self.assertEqual(stack.count(), 1)
        history_steps = item.document().availableUndoSteps()
        html = item.toHtml()
        stack.push(ApplyFontformatCommand([item], [FakeTransEdit()], target))

        self.assertEqual(stack.count(), 1)
        self.assertEqual(item.document().availableUndoSteps(), history_steps)
        self.assertEqual(item.toHtml(), html)

    def test_multi_item_whole_format_uses_one_stack_entry(self):
        scene = QGraphicsScene()
        items = []
        for index, transform in enumerate(
            ((1.0, 1.0, 0.0), (0.8, 1.3, 11.0))
        ):
            block = TextBlock(
                xyxy=[index * 140, 0, index * 140 + 120, 60],
                _bounding_rect=[index * 140, 0, 120, 60],
                translation=f'item {index}',
                fontformat=FontFormat(
                    horizontal_scale=transform[0],
                    vertical_scale=transform[1],
                    slant_angle=transform[2],
                ),
            )
            item = TextBlkItem(block)
            scene.addItem(item)
            items.append(item)
        before = [
            (
                item.toHtml(),
                item.fontformat.deepcopy(),
                item.absBoundingRect(qrect=True),
                item.pos(),
            )
            for item in items
        ]
        target = FontFormat(
            vertical=True,
            bold=True,
            horizontal_scale=1.6,
            vertical_scale=0.65,
            slant_angle=-9.0,
        )
        stack = QUndoStack()

        stack.push(
            ApplyFontformatCommand(
                items,
                [FakeTransEdit(), FakeTransEdit()],
                target,
            )
        )

        self.assertEqual(stack.count(), 1)
        for item in items:
            self.assertTrue(item.fontformat.vertical)
            self.assertEqual(item.fontformat.text_transform, (1.6, 0.65, -9.0))

        stack.undo()
        for item, state in zip(items, before):
            self.assertEqual(item.toHtml(), state[0])
            self.assertEqual(item.fontformat, state[1])
            self.assertEqual(item.absBoundingRect(qrect=True), state[2])
            self.assertEqual(item.pos(), state[3])

        stack.redo()
        for item in items:
            self.assertTrue(item.fontformat.vertical)
            self.assertEqual(item.fontformat.text_transform, (1.6, 0.65, -9.0))

    def test_uniform_noop_is_removed_from_undo_stack(self):
        block = TextBlock(
            xyxy=[0, 0, 120, 60],
            _bounding_rect=[0, 0, 120, 60],
            translation='plain text',
            fontformat=FontFormat(),
        )
        item = TextBlkItem(block)
        scene = QGraphicsScene()
        scene.addItem(item)
        before_html = item.toHtml()
        before_rect = item.absBoundingRect(qrect=True)
        before_position = item.pos()
        before_undo_steps = item.document().availableUndoSteps()
        shape_control = TrackingShapeControl(item)
        stack = QUndoStack()

        stack.push(
            ApplyFontformatCommand(
                [item],
                [FakeTransEdit()],
                item.fontformat.deepcopy(),
                shape_control,
            )
        )

        self.assertEqual(stack.count(), 0)
        self.assertEqual(shape_control.refresh_count, 0)
        self.assertEqual(item.toHtml(), before_html)
        self.assertEqual(item.absBoundingRect(qrect=True), before_rect)
        self.assertEqual(item.pos(), before_position)
        self.assertEqual(
            item.document().availableUndoSteps(), before_undo_steps
        )

    def test_compound_format_is_one_deterministic_command(self):
        original_format = FontFormat(
            font_family='Arial',
            font_size=24,
            stroke_width=0.03,
            srgb=[12, 34, 56],
            shadow_radius=0.1,
            shadow_strength=0.6,
            shadow_offset=[2.0, -1.0],
            horizontal_scale=1.15,
            vertical_scale=0.85,
            slant_angle=7.0,
        )
        rich_text = (
            '<html><body><p>'
            '<span style="font-size:12pt; font-weight:700; color:#123456;">'
            'Latin</span>'
            '<span style="font-size:19pt; font-style:italic; color:#a02070;">'
            ' \u6f22\U0001f600e\u0301</span>'
            '</p></body></html>'
        )
        block = TextBlock(
            xyxy=[20, 30, 180, 110],
            _bounding_rect=[20, 30, 160, 80],
            translation='Latin \u6f22\U0001f600e\u0301',
            rich_text=rich_text,
            fontformat=original_format,
        )
        item = TextBlkItem(block)
        scene = QGraphicsScene()
        scene.addItem(item)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(8)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        before_html = item.toHtml()
        before_format = item.fontformat.deepcopy()
        before_rect = item.absBoundingRect(qrect=True)
        before_position = item.pos()
        before_transform = item.transform()
        before_cursor = (
            item.textCursor().position(),
            item.textCursor().anchor(),
        )

        target = original_format.deepcopy()
        target.vertical = True
        target.font_size = 42
        target.bold = True
        target.underline = True
        target.alignment = 2
        target.stroke_width = 0.12
        target.shadow_radius = 0.35
        target.shadow_strength = 0.8
        target.shadow_offset = [-4.0, 5.0]
        target.horizontal_scale = 1.8
        target.vertical_scale = 0.55
        target.slant_angle = -18.0

        shape_control = TrackingShapeControl(item)
        command = ApplyFontformatCommand(
            [item], [FakeTransEdit()], target, shape_control
        )
        stack = QUndoStack()
        stack.push(command)
        self.assertEqual(stack.count(), 1)
        self.assertEqual(command.childCount(), 0)
        self.assertEqual(shape_control.refresh_count, 1)
        self.assertTrue(item.fontformat.vertical)
        self.assertEqual(item.fontformat.text_transform, (1.8, 0.55, -18.0))
        self.assertEqual(
            (item.textCursor().position(), item.textCursor().anchor()),
            before_cursor,
        )

        after_html = item.toHtml()
        after_format = item.fontformat.deepcopy()
        after_rect = item.absBoundingRect(qrect=True)
        after_position = item.pos()
        after_transform = item.transform()

        for cycle in range(2):
            with self.subTest(cycle=cycle, operation='undo'):
                stack.undo()
                self.assertEqual(item.toHtml(), before_html)
                self.assertEqual(item.fontformat, before_format)
                self.assertEqual(item.absBoundingRect(qrect=True), before_rect)
                self.assertEqual(item.pos(), before_position)
                self.assertEqual(item.transform(), before_transform)
                self.assertEqual(
                    (item.textCursor().position(), item.textCursor().anchor()),
                    before_cursor,
                )

            with self.subTest(cycle=cycle, operation='redo'):
                stack.redo()
                self.assertEqual(item.toHtml(), after_html)
                self.assertEqual(item.fontformat, after_format)
                self.assertEqual(item.absBoundingRect(qrect=True), after_rect)
                self.assertEqual(item.pos(), after_position)
                self.assertEqual(item.transform(), after_transform)
                self.assertEqual(
                    (item.textCursor().position(), item.textCursor().anchor()),
                    before_cursor,
                )

        self.assertEqual(shape_control.refresh_count, 5)


if __name__ == '__main__':
    unittest.main()
