import copy
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QEvent, QPointF, QRectF, Qt
from qtpy.QtGui import (
    QColor,
    QImage,
    QInputMethodEvent,
    QKeyEvent,
    QPainter,
    QTextCursor,
)
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QStyleOptionGraphicsItem,
)

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
from ballontranslator.ui.text_advanced_format import TextAdvancedFormatPanel
from ballontranslator.ui.texteditshapecontrol import TextBlkShapeControl
from ballontranslator.ui.text_transform_editor import TextTransformEditSession
from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.text_effects.glyph import (
    GLOBAL_GLYPH_GEOMETRY_CACHE,
    GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE,
    GlyphGeometry,
    WeightedGlyphGeometryCache,
)
from ballontranslator.ui.text_effects.transform_layout import (
    GlyphSlantLayoutRenderer,
)
from ballontranslator.ui.text_effects.curvature import CurvatureMapper
from ballontranslator.ui.text_transform import (
    CompositeTextTransformMapper,
    perspective_transform_matrix,
    text_transform_matrix,
)
from ballontranslator.ui.text_transform_variants import (
    compile_text_transform_stack,
)
from ballontranslator.utils.fontformat import (
    CurvatureTextTransform,
    FontFormat,
    PerspectiveTextTransform,
    SlantTextTransform,
    TextTransformStack,
    TextTransformState,
)
from ballontranslator.utils import shared
from ballontranslator.utils.config import ProgramConfig
from ballontranslator.utils.proj_imgtrans import ProjImgTrans, TextBlkEncoder
from ballontranslator.utils.textblock import TextBlock


TEST_LINES = (
    "Без труда не вытащишь и рыбку из пруда.",
    "冰冻三尺，非一日之寒。",
    "猿も木から落ちる。",
    "Don't judge a book by its cover.",
    "벼는 익을수록 고개를 숙인다.",
    "☀ ☁ ☂ ☃ ★ ☆ ☎ ☯ ♠ ♥ ♦ ♣ ⚠ ⚽ ⚾ ㊗ ㊙ ! @ # $",
)
def transform_state(*transforms, glyph_slant_angle=0.0):
    return TextTransformState(
        TextTransformStack(tuple(transforms)),
        glyph_slant_angle,
    )


NEUTRAL = transform_state()
FIRST_TRANSFORM = transform_state(
    SlantTextTransform(1.2, 0.9, 12.0),
    glyph_slant_angle=5.0,
)
FINAL_TRANSFORMS = (
    transform_state(
        SlantTextTransform(0.8, 1.1, -9.0),
        glyph_slant_angle=-4.0,
    ),
    transform_state(
        SlantTextTransform(1.3, 0.7, 6.0),
        glyph_slant_angle=8.0,
    ),
)


class TextTransformTestBase(unittest.TestCase):
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
            self.assertEqual(
                TextTransformState(
                    item.blk.fontformat.text_transform,
                    item.blk.fontformat.glyph_slant_angle,
                ),
                transform,
            )

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


class ExtendedTextTransformModelTest(TextTransformTestBase):
    def test_perspective_and_curvature_payloads_round_trip(self):
        payloads = (
            {
                'transform_type': 'perspective',
                'strength': 0.55,
                'direction': -35.0,
            },
            {
                'transform_type': 'curvature',
                'curvature': -0.75,
            },
        )
        for payload in payloads:
            with self.subTest(transform_type=payload['transform_type']):
                font_format = FontFormat(text_transform=[payload])
                self.assertEqual(
                    font_format.to_serializable_dict()['text_transform'],
                    [payload],
                )

    def test_old_single_transform_payload_is_dropped(self):
        with self.assertLogs('BallonTranslator', level='WARNING') as logs:
            font_format = FontFormat(
                text_transform={
                    'transform_type': 'curvature',
                    'curvature': 0.5,
                },
                font_size=37.0,
            )
        self.assertIn(
            'Ignoring invalid text transform config',
            '\n'.join(logs.output),
        )
        self.assertEqual(font_format.text_transform, TextTransformStack())
        self.assertEqual(font_format.font_size, 37.0)

    def test_program_config_drops_only_an_invalid_text_transform(self):
        payload = {
            'display_lang': 'English',
            'global_fontformat': {
                'font_size': 37.0,
                'opacity': 0.6,
                'text_transform': {
                    'transform_type': 'curvature',
                    'curvature': 0.5,
                },
            },
        }
        with tempfile.NamedTemporaryFile(
            'w+', encoding='utf8'
        ) as config_file:
            json.dump(payload, config_file)
            config_file.flush()
            with self.assertLogs('BallonTranslator', level='WARNING') as logs:
                loaded = ProgramConfig.load(config_file.name)

        self.assertIn(
            'Ignoring invalid text transform config',
            '\n'.join(logs.output),
        )
        self.assertEqual(loaded.display_lang, 'English')
        self.assertEqual(loaded.global_fontformat.font_size, 37.0)
        self.assertEqual(loaded.global_fontformat.opacity, 0.6)
        self.assertEqual(
            loaded.global_fontformat.text_transform,
            TextTransformStack(),
        )

    def test_project_drops_invalid_block_transform_without_rejecting_project(self):
        project_data = {
            'pages': {
                'missing.png': [{
                    'translation': 'preserved',
                    'fontformat': {
                        'font_size': 41.0,
                        'text_transform': {
                            'transform_type': 'slant',
                            'horizontal_scale': 1.0,
                            'vertical_scale': 1.0,
                            'slant_angle': 0.0,
                            'glyph_slant_angle': 0.0,
                        },
                    },
                }],
            },
            'image_info': {},
        }
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            with self.assertLogs('BallonTranslator', level='WARNING') as logs:
                project.load_from_dict(project_data)

        block = project.not_found_pages['missing.png'][0]
        self.assertIn(
            'Ignoring invalid text transform config',
            '\n'.join(logs.output),
        )
        self.assertEqual(block.translation, 'preserved')
        self.assertEqual(block.fontformat.font_size, 41.0)
        self.assertEqual(
            block.fontformat.text_transform,
            TextTransformStack(),
        )

    def test_duplicate_stack_entries_and_glyph_slant_round_trip(self):
        payload = [
            {
                'transform_type': 'slant',
                'horizontal_scale': 1.2,
                'vertical_scale': 1.0,
                'slant_angle': 5.0,
            },
            {
                'transform_type': 'slant',
                'horizontal_scale': 0.8,
                'vertical_scale': 1.1,
                'slant_angle': -3.0,
            },
        ]
        font_format = FontFormat(
            text_transform=payload,
            glyph_slant_angle=7.0,
        )
        serialized = font_format.to_serializable_dict()
        self.assertEqual(serialized['text_transform'], payload)
        self.assertEqual(serialized['glyph_slant_angle'], 7.0)

        block = TextBlock([0, 0, 20, 10])
        block.fontformat = font_format
        restored = TextBlock(
            **json.loads(json.dumps(block, cls=TextBlkEncoder))
        )
        self.assertEqual(
            restored.fontformat.text_transform,
            font_format.text_transform,
        )
        self.assertEqual(restored.fontformat.glyph_slant_angle, 7.0)

    def test_perspective_matrix_is_centered_and_invertible(self):
        rect = QRectF(20, 30, 400, 180)
        for direction in (-180.0, -45.0, 0.0, 90.0, 180.0):
            with self.subTest(direction=direction):
                matrix = perspective_transform_matrix(
                    PerspectiveTextTransform(0.8, direction),
                    rect,
                )
                inverse, invertible = matrix.inverted()
                self.assertTrue(invertible)
                self.assertEqual(matrix.map(rect.center()), rect.center())
                for point in (
                    rect.topLeft(),
                    rect.topRight(),
                    rect.bottomRight(),
                    rect.bottomLeft(),
                ):
                    restored = inverse.map(matrix.map(point))
                    self.assertAlmostEqual(restored.x(), point.x(), places=6)
                    self.assertAlmostEqual(restored.y(), point.y(), places=6)

    def test_curvature_mapper_round_trips_both_writing_modes(self):
        for vertical in (False, True):
            logical = (
                QRectF(10, 20, 160, 420)
                if vertical
                else QRectF(10, 20, 420, 160)
            )
            source = logical.adjusted(-12, -12, 12, 12)
            for curvature in (-1.0, -0.4, 0.0, 0.4, 1.0):
                with self.subTest(
                    vertical=vertical, curvature=curvature
                ):
                    mapper = CurvatureMapper(
                        logical, source, vertical, curvature
                    )
                    for x_ratio, y_ratio in (
                        (0.0, 0.0),
                        (0.2, 0.7),
                        (0.5, 0.5),
                        (0.8, 0.3),
                        (1.0, 1.0),
                    ):
                        point = QPointF(
                            logical.left() + logical.width() * x_ratio,
                            logical.top() + logical.height() * y_ratio,
                        )
                        restored = mapper.inverse_point(
                            mapper.forward_point(point)
                        )
                        self.assertAlmostEqual(
                            restored.x(), point.x(), places=6
                        )
                        self.assertAlmostEqual(
                            restored.y(), point.y(), places=6
                        )


class TextTransformPanelTest(TextTransformTestBase):
    def _make_panel(self):
        previous = getattr(shared, 'register_view_widget', None)
        shared.register_view_widget = lambda *_args: None
        self.addCleanup(
            lambda: (
                delattr(shared, 'register_view_widget')
                if previous is None
                else setattr(shared, 'register_view_widget', previous)
            )
        )
        panel = TextAdvancedFormatPanel(
            'Advanced', 'test_transform', 'test_transform_expand',
            lambda *_args: None,
        )
        self.addCleanup(panel.deleteLater)
        return panel

    def test_add_menu_and_hover_actions_are_generated_from_registry(self):
        panel = self._make_panel()
        self.assertEqual(
            [action.text() for action in panel.add_transform_button.menu().actions()],
            ['Slant', 'Perspective', 'Curvature'],
        )
        added = []
        panel.transform_add_requested.connect(added.append)
        panel.add_transform_button.menu().actions()[1].trigger()
        self.assertEqual(added, ['perspective'])

        panel.set_transform(
            transform_state(SlantTextTransform(), CurvatureTextTransform())
        )
        operation_panel = panel.transform_panels[0]
        self.assertTrue(operation_panel.close_button.isHidden())
        QApplication.sendEvent(operation_panel, QEvent(QEvent.Type.Enter))
        self.assertFalse(operation_panel.close_button.isHidden())

        removed = []
        panel.transform_remove_requested.connect(removed.append)
        operation_panel.close_button.click()
        self.assertEqual(removed, [0])

    def test_multi_selection_only_exposes_matching_stack_indices(self):
        panel = self._make_panel()
        matching = [
            SimpleNamespace(
                blk=SimpleNamespace(
                    fontformat=FontFormat(
                        text_transform=TextTransformStack((
                            SlantTextTransform(1.1, 1.0, 4.0),
                            CurvatureTextTransform(0.4),
                        )),
                    )
                )
            ),
            SimpleNamespace(
                blk=SimpleNamespace(
                    fontformat=FontFormat(
                        text_transform=TextTransformStack((
                            SlantTextTransform(0.9, 1.0, -4.0),
                            CurvatureTextTransform(-0.4),
                        )),
                    )
                )
            ),
        ]
        panel.set_transform_items(matching)
        self.assertEqual(len(panel.transform_panels), 2)
        self.assertTrue(panel.transform_mixed_label.isHidden())
        self.assertFalse(panel.transform_rows.isHidden())
        self.assertEqual(
            panel.transform_panels[0].controls[
                'horizontal_scale'
            ].editor.text(),
            '\N{EM DASH}',
        )

        mixed = [
            matching[0],
            SimpleNamespace(
                blk=SimpleNamespace(
                    fontformat=FontFormat(
                        text_transform=TextTransformStack((
                            CurvatureTextTransform(0.4),
                            SlantTextTransform(),
                        )),
                    )
                )
            ),
        ]
        panel.set_transform_items(mixed)
        self.assertEqual(panel.transform_panels, [])
        self.assertFalse(panel.transform_mixed_label.isHidden())
        self.assertTrue(panel.transform_rows.isHidden())


class TextTransformUndoTest(TextTransformTestBase):
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

    def test_perspective_and_curvature_mix_with_text_undo(self):
        perspective = transform_state(
            PerspectiveTextTransform(0.6, 30.0)
        )
        curvature = transform_state(CurvatureTextTransform(-0.65))
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item, pair = self._make_pair(0, TEST_LINES[0], vertical)
                stack = QUndoStack()
                stack.push(
                    SetTextTransformCommand.create(
                        [item], [NEUTRAL], [perspective]
                    )
                )
                stack.push(
                    MultiPasteCommand(
                        TEST_LINES[1], [item], [pair.e_trans]
                    )
                )
                stack.push(
                    SetTextTransformCommand.create(
                        [item], [perspective], [curvature]
                    )
                )
                expected = (
                    (NEUTRAL, TEST_LINES[0]),
                    (perspective, TEST_LINES[0]),
                    (perspective, TEST_LINES[1]),
                    (curvature, TEST_LINES[1]),
                )
                for _ in range(3):
                    for transform, text in reversed(expected[:-1]):
                        stack.undo()
                        self.assertEqual(
                            TextTransformState(
                                item.blk.fontformat.text_transform,
                                item.blk.fontformat.glyph_slant_angle,
                            ),
                            transform,
                        )
                        self.assertEqual(item.toPlainText(), text)
                        self.assertEqual(pair.e_trans.toPlainText(), text)
                    for transform, text in expected[1:]:
                        stack.redo()
                        self.assertEqual(
                            TextTransformState(
                                item.blk.fontformat.text_transform,
                                item.blk.fontformat.glyph_slant_angle,
                            ),
                            transform,
                        )
                        self.assertEqual(item.toPlainText(), text)
                        self.assertEqual(pair.e_trans.toPlainText(), text)

    def test_stack_structure_edits_are_undoable_for_selected_items(self):
        versions = (
            transform_state(SlantTextTransform(1.15, 0.85, 11.0)),
            transform_state(SlantTextTransform(0.75, 1.25, -7.0)),
        )
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)

        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                items = [
                    self._make_pair(index, TEST_LINES[index], vertical)[0]
                    for index in range(2)
                ]
                for item, transform in zip(items, versions):
                    item.set_text_transform(transform)

                stack = QUndoStack()
                SW.canvas = SimpleNamespace(push_undo_command=stack.push)
                session = object.__new__(TextTransformEditSession)
                session.items = items
                session.controls = SimpleNamespace(
                    set_transform_items=lambda _items: None,
                    finish_pending_transform_edits=lambda: None,
                    cancel_transform_previews=lambda: None,
                )
                session.drag_before = None
                session.drag_key = None

                session.add_transform('curvature')
                self.assertEqual(
                    [
                        tuple(item.blk.fontformat.text_transform)
                        for item in items
                    ],
                    [
                        (
                            versions[0].stack[0],
                            CurvatureTextTransform(),
                        ),
                        (
                            versions[1].stack[0],
                            CurvatureTextTransform(),
                        ),
                    ],
                )
                session.add_transform('slant')
                self.assertEqual(
                    [len(item.blk.fontformat.text_transform) for item in items],
                    [3, 3],
                )
                session.move_transform(2, -1)
                self.assertEqual(
                    [
                        tuple(
                            transform.transform_type
                            for transform in item.blk.fontformat.text_transform
                        )
                        for item in items
                    ],
                    [('slant', 'slant', 'curvature')] * 2,
                )
                session.remove_transform(2)
                self.assertEqual(
                    [len(item.blk.fontformat.text_transform) for item in items],
                    [2, 2],
                )
                stack.undo()
                self.assertEqual(
                    [len(item.blk.fontformat.text_transform) for item in items],
                    [3, 3],
                )
                stack.undo()
                self.assertEqual(
                    [
                        tuple(
                            transform.transform_type
                            for transform in item.blk.fontformat.text_transform
                        )
                        for item in items
                    ],
                    [('slant', 'curvature', 'slant')] * 2,
                )
                stack.undo()
                stack.undo()
                self.assertEqual(
                    [
                        TextTransformState(
                            item.blk.fontformat.text_transform,
                            item.blk.fontformat.glyph_slant_angle,
                        )
                        for item in items
                    ],
                    list(versions),
                )

    def test_mixed_stack_structures_only_allow_append(self):
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        items = [
            self._make_pair(index, TEST_LINES[index], False)[0]
            for index in range(2)
        ]
        initial = (
            transform_state(SlantTextTransform(1.2, 1.0, 5.0)),
            transform_state(CurvatureTextTransform(0.4)),
        )
        for item, state in zip(items, initial):
            item.set_text_transform(state)

        stack = QUndoStack()
        SW.canvas = SimpleNamespace(push_undo_command=stack.push)
        session = object.__new__(TextTransformEditSession)
        session.items = items
        session.controls = SimpleNamespace(
            set_transform_items=lambda _items: None,
            finish_pending_transform_edits=lambda: None,
            cancel_transform_previews=lambda: None,
        )
        session.drag_before = None
        session.drag_key = None

        session.commit_value(0, 'horizontal_scale', 1.5)
        self.assertEqual(stack.count(), 0)
        self.assertEqual(
            tuple(session._state_for_item(item) for item in items),
            initial,
        )

        session.add_transform('perspective')
        self.assertEqual(stack.count(), 1)
        for item in items:
            self.assertEqual(
                item.blk.fontformat.text_transform[-1],
                PerspectiveTextTransform(),
            )
        stack.undo()
        self.assertEqual(
            tuple(session._state_for_item(item) for item in items),
            initial,
        )

    def test_compiler_uses_one_mapping_boundary_for_composed_operations(self):
        logical = QRectF(10, 20, 420, 160)
        source = logical.adjusted(-12, -12, 12, 12)
        first = SlantTextTransform(1.2, 0.9, 8.0)
        second = SlantTextTransform(0.8, 1.1, -4.0)
        matrix_only = TextTransformStack((first, second))
        compiled = compile_text_transform_stack(
            matrix_only, logical, source, False
        )
        self.assertIsNone(compiled.surface_mapper)
        expected = (
            text_transform_matrix(
                first.horizontal_scale,
                first.vertical_scale,
                first.slant_angle,
                logical.center(),
            )
            * text_transform_matrix(
                second.horizontal_scale,
                second.vertical_scale,
                second.slant_angle,
                logical.center(),
            )
        )
        self.assertEqual(compiled.native_matrix, expected)
        reversed_compiled = compile_text_transform_stack(
            TextTransformStack((second, first)),
            logical,
            source,
            False,
        )
        self.assertNotEqual(
            compiled.native_matrix.map(logical.topLeft()),
            reversed_compiled.native_matrix.map(logical.topLeft()),
        )
        neutral_nonlinear = compile_text_transform_stack(
            TextTransformStack((
                CurvatureTextTransform(),
                first,
            )),
            logical,
            source,
            False,
        )
        self.assertIsNone(neutral_nonlinear.surface_mapper)
        self.assertFalse(
            TextTransformStack((CurvatureTextTransform(),)).has_nonlinear
        )

        single_nonlinear = compile_text_transform_stack(
            TextTransformStack((CurvatureTextTransform(0.6),)),
            logical,
            source,
            False,
        ).surface_mapper
        stage = single_nonlinear.stages[0]
        stage_map_rect_path = stage.map_rect_path
        stage_path_calls = 0

        def recording_map_rect_path(rect):
            nonlocal stage_path_calls
            stage_path_calls += 1
            return stage_map_rect_path(rect)

        stage.map_rect_path = recording_map_rect_path
        first_path = single_nonlinear.map_rect_path(source)
        second_path = single_nonlinear.map_rect_path(source)
        self.assertEqual(stage_path_calls, 1)
        self.assertEqual(first_path, second_path)

        nonlinear = TextTransformStack((
            SlantTextTransform(1.1, 0.9, 3.0),
            PerspectiveTextTransform(0.45, -70.0),
            CurvatureTextTransform(0.72),
            SlantTextTransform(0.8, 1.1, -4.0),
            PerspectiveTextTransform(0.2, 25.0),
            CurvatureTextTransform(-0.35),
        ))
        for vertical in (False, True):
            compiled = compile_text_transform_stack(
                nonlinear, logical, source, vertical
            )
            self.assertTrue(compiled.native_matrix.isIdentity())
            self.assertIsInstance(
                compiled.surface_mapper, CompositeTextTransformMapper
            )
            # Adjacent matrix stages on either side of a nonlinear stage fold.
            self.assertEqual(len(compiled.surface_mapper.stages), 4)
            self.assertTrue(nonlinear.has_nonlinear)

            point = QPointF(180, 90)
            restored = compiled.surface_mapper.inverse_point(
                compiled.surface_mapper.forward_point(point)
            )
            self.assertAlmostEqual(restored.x(), point.x(), places=5)
            self.assertAlmostEqual(restored.y(), point.y(), places=5)


class TextTransformRenderingTest(TextTransformTestBase):
    def test_uniform_glyph_paint_batch_matches_per_line_fallback(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                width, height = (
                    (300, 600) if vertical else (600, 300)
                )
                block = TextBlock([0, 0, width, height])
                block._bounding_rect = [0, 0, width, height]
                block.vertical = vertical
                block.translation = (
                    TEST_LINES[0]
                    + "\n\n   \n"
                    + "\n".join(TEST_LINES[1:4])
                )
                block.fontformat.glyph_slant_angle = 20.0
                item = TextBlkItem(block, 0)
                scene = QGraphicsScene()
                scene.addItem(item)
                self.app.processEvents()

                batched = self._render_scene(scene)
                renderer = item.geometry_controller.layout_renderer
                with patch.object(
                    renderer,
                    "draw_uniform_block",
                    return_value=False,
                ):
                    item.update()
                    fallback = self._render_scene(scene)
                self.assertEqual(batched, fallback)

    def test_vertical_width_resize_translates_existing_layout(self):
        block = TextBlock([0, 0, 300, 600])
        block._bounding_rect = [0, 0, 300, 600]
        block.vertical = True
        block.translation = "\n".join(TEST_LINES[:4])
        block.fontformat.glyph_slant_angle = 20.0
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.app.processEvents()
        layout = item.layout

        with patch.object(
            layout, "layoutBlock", wraps=layout.layoutBlock
        ) as layout_block:
            resized = item.absBoundingRect(qrect=True)
            resized.setWidth(resized.width() + 40)
            item.setRect(resized, repaint=False)
            self.assertEqual(layout_block.call_count, 0)
            fast_positions = tuple(
                (
                    block_number,
                    line_number,
                    block_layout.lineAt(line_number).position(),
                )
                for block_number in range(item.document().blockCount())
                for block_layout in (
                    item.document()
                    .findBlockByNumber(block_number)
                    .layout(),
                )
                for line_number in range(block_layout.lineCount())
            )
            fast_pixels = self._render_scene(scene)

            layout.reLayout()
            full_positions = tuple(
                (
                    block_number,
                    line_number,
                    block_layout.lineAt(line_number).position(),
                )
                for block_number in range(item.document().blockCount())
                for block_layout in (
                    item.document()
                    .findBlockByNumber(block_number)
                    .layout(),
                )
                for line_number in range(block_layout.lineCount())
            )
            self.assertEqual(fast_positions, full_positions)
            self.assertEqual(fast_pixels, self._render_scene(scene))

            layout_block.reset_mock()
            resized.setHeight(resized.height() + 40)
            item.setRect(resized, repaint=False)
            self.assertGreater(layout_block.call_count, 0)

    def test_geometry_edits_compile_each_transform_input_once(self):
        state = transform_state(
            SlantTextTransform(1.1, 0.9, 5.0),
            CurvatureTextTransform(0.55),
            PerspectiveTextTransform(0.25, 30.0),
            CurvatureTextTransform(-0.2),
            glyph_slant_angle=22.0,
        )
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                width, height = (
                    (300, 600) if vertical else (600, 300)
                )
                block = TextBlock([0, 0, width, height])
                block._bounding_rect = [0, 0, width, height]
                block.vertical = vertical
                block.translation = "\n".join(TEST_LINES[:4])
                block.fontformat.stroke_width = 0.08
                item = TextBlkItem(block, 0)
                item.set_text_transform(state)
                controller = item.geometry_controller

                target = (
                    'ballontranslator.ui.text_item_geometry.'
                    'compile_text_transform_stack'
                )
                with patch(
                    target, wraps=compile_text_transform_stack
                ) as compile_mock:
                    unchanged = item.absBoundingRect(qrect=True)
                    item.setRect(unchanged, repaint=False)
                    self.app.processEvents()
                    self.assertEqual(compile_mock.call_count, 0)

                    changed = QRectF(unchanged)
                    if vertical:
                        changed.setHeight(changed.height() + 40)
                    else:
                        changed.setWidth(changed.width() + 40)
                    item.setRect(changed, repaint=False)
                    self.app.processEvents()
                    self.assertEqual(compile_mock.call_count, 1)

                    compile_mock.reset_mock()
                    item.startEdit()
                    cursor = item.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.insertText(" reflow")
                    item.setTextCursor(cursor)
                    self.app.processEvents()
                    self.assertLessEqual(compile_mock.call_count, 1)

                    compile_mock.reset_mock()
                    item.setFontSize(38, repaint_background=False)
                    self.app.processEvents()
                    self.assertLessEqual(compile_mock.call_count, 1)
                    self.assertFalse(controller._compile_deferred)
                    compile_mock.reset_mock()
                    self.assertFalse(
                        controller.refresh_compiled_geometry()
                    )
                    self.assertEqual(compile_mock.call_count, 0)
                    item.endEdit()

                    compile_mock.reset_mock()
                    item.setRelFontSize(
                        0.9, repaint_background=False
                    )
                    self.app.processEvents()
                    self.assertLessEqual(compile_mock.call_count, 1)

                    compile_mock.reset_mock()
                    item.setStrokeWidth(
                        0.12, repaint_background=False
                    )
                    self.app.processEvents()
                    self.assertLessEqual(compile_mock.call_count, 1)

                    compile_mock.reset_mock()
                    item.setVertical(not vertical)
                    self.app.processEvents()
                    compile_inputs = {
                        (
                            call.args[1].getRect(),
                            call.args[2].getRect(),
                            call.args[3],
                        )
                        for call in compile_mock.call_args_list
                    }
                    self.assertEqual(
                        len(compile_inputs), compile_mock.call_count
                    )
                    self.assertLessEqual(compile_mock.call_count, 2)

                    compile_mock.reset_mock()
                    mapper = controller.visual_mapper
                    controller.detach_surface_mapper()
                    self.assertTrue(
                        controller.refresh_compiled_geometry()
                    )
                    self.assertEqual(compile_mock.call_count, 0)
                    self.assertIs(controller.visual_mapper, mapper)

    def test_glyph_slant_effects_stay_on_transformed_source_path(self):
        state = transform_state(
            PerspectiveTextTransform(0.3, 25.0),
            CurvatureTextTransform(0.55),
            glyph_slant_angle=20.0,
        )
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                width, height = (
                    (300, 600) if vertical else (600, 300)
                )
                block = TextBlock([0, 0, width, height])
                block._bounding_rect = [0, 0, width, height]
                block.vertical = vertical
                block.translation = "\n".join(TEST_LINES[:3])
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
                view = QGraphicsView(scene)
                view.show()
                self.app.processEvents()
                item.set_text_transform(state)
                controller = item.geometry_controller
                renderer = item.effect_renderer

                def assert_effect_path():
                    self.assertIsNotNone(controller.visual_mapper)
                    self.assertIs(
                        item.layout.render_delegate,
                        controller.layout_renderer,
                    )
                    self.assertFalse(
                        renderer._text_transform_is_neutral()
                    )
                    with (
                        patch.object(
                            renderer,
                            '_paint_cloned_document_stroke',
                        ) as cloned,
                        patch.object(
                            renderer, '_paint_live_layout'
                        ) as live,
                        patch.object(
                            renderer, '_paint_vertical_stroke'
                        ) as vertical_stroke,
                    ):
                        renderer.paint_stroke(None)
                    cloned.assert_not_called()
                    if vertical:
                        vertical_stroke.assert_called_once()
                        live.assert_not_called()
                    else:
                        live.assert_called_once()
                        vertical_stroke.assert_not_called()
                    pixels = self._render_scene(scene)
                    self.assertNotEqual(pixels, bytes(len(pixels)))

                assert_effect_path()
                item.startEdit()
                pair = TransPairWidget(block, 0, False)
                pair.e_trans.setPlainText(item.toPlainText())
                propagated = []
                def record_and_propagate(position, text, joint):
                    propagated.append((position, text))
                    propagate_user_edit(
                        item,
                        pair.e_trans,
                        position,
                        text,
                        joint,
                    )

                item.propagate_user_edited.connect(record_and_propagate)
                cursor = item.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                insert_at = cursor.position()
                cursor.insertText(" edited")
                item.setTextCursor(cursor)
                self.app.processEvents()
                self.assertEqual(len(propagated), 1)
                self.assertIn(
                    propagated[0],
                    (
                        (insert_at, " edited"),
                        (
                            0,
                            item.toPlainText().replace(
                                '\n', '\N{PARAGRAPH SEPARATOR}'
                            ),
                        ),
                    ),
                )
                self.assertEqual(
                    pair.e_trans.toPlainText(), item.toPlainText()
                )
                assert_effect_path()
                item.endEdit()

                rect = item.absBoundingRect(qrect=True)
                rect.setWidth(rect.width() + 40)
                rect.setHeight(rect.height() + 30)
                item.setRect(rect, repaint=False)
                assert_effect_path()

                item.setFontSize(34, repaint_background=False)
                assert_effect_path()
                pair.deleteLater()
                view.close()
                scene.removeItem(item)

    def test_zero_glyph_slant_restores_effects_inside_nonlinear_stack(self):
        stack = TextTransformStack((CurvatureTextTransform(0.55),))
        zero = TextTransformState(stack, 0.0)
        slanted = TextTransformState(stack, 20.0)
        for vertical in (False, True):
            for effect in ("stroke", "shadow"):
                with self.subTest(vertical=vertical, effect=effect):
                    width, height = (
                        (300, 600) if vertical else (600, 300)
                    )
                    block = TextBlock([0, 0, width, height])
                    block._bounding_rect = [0, 0, width, height]
                    block.vertical = vertical
                    block.translation = "\n".join(TEST_LINES[:3])
                    if effect == "stroke":
                        block.fontformat.stroke_width = 0.12
                    else:
                        block.fontformat.shadow_radius = 0.12
                        block.fontformat.shadow_strength = 0.8
                        block.fontformat.shadow_offset = [0.1, 0.1]

                    item = TextBlkItem(block, 0)
                    scene = QGraphicsScene()
                    scene.addItem(item)
                    item.set_text_transform(zero)
                    zero_pixels = self._render_scene(scene)

                    item.set_text_transform(slanted)
                    slanted_pixels = self._render_scene(scene)
                    self.assertNotEqual(slanted_pixels, zero_pixels)

                    renderer = item.effect_renderer
                    with patch.object(
                        renderer,
                        "_repaint_neutral_background",
                        wraps=renderer._repaint_neutral_background,
                    ) as repaint_neutral:
                        item.set_text_transform(zero, preview=True)
                    self.assertEqual(repaint_neutral.call_count, 1)
                    self.assertIsNotNone(
                        renderer.background_pixmap
                    )

                    item.clear_text_transform_preview()
                    self._render_scene(scene)
                    with patch.object(
                        renderer,
                        "_repaint_neutral_background",
                        wraps=renderer._repaint_neutral_background,
                    ) as repaint_neutral:
                        item.set_text_transform(zero)
                    self.assertEqual(repaint_neutral.call_count, 1)
                    self.assertIsNotNone(
                        renderer.background_pixmap
                    )
                    self.assertIsNone(renderer._transformed_effect_state)
                    scene.removeItem(item)

    def test_surface_without_raster_effects_keeps_effect_fast_path(self):
        state = transform_state(
            CurvatureTextTransform(0.55),
            glyph_slant_angle=20.0,
        )
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item, _ = self._make_pair(0, TEST_LINES[1], vertical)
                scene = QGraphicsScene()
                scene.addItem(item)
                item.set_text_transform(state)

                renderer = item.effect_renderer
                self.assertTrue(
                    item.geometry_controller.has_layout_distortion()
                )
                self.assertTrue(
                    item.geometry_controller.uses_surface_warp()
                )
                self.assertTrue(renderer._text_transform_is_neutral())
                self.assertIsNone(renderer._transformed_effect_state)
                pixels = self._render_scene(scene)
                self.assertNotEqual(pixels, bytes(len(pixels)))
                self.assertIsNone(renderer._transformed_effect_state)

                item.set_text_transform(
                    TextTransformState(state.stack, -20.0)
                )
                mirrored_pixels = self._render_scene(scene)
                self.assertNotEqual(mirrored_pixels, pixels)
                self.assertIsNone(renderer._transformed_effect_state)
                scene.removeItem(item)

    def test_reshape_surface_uses_bounded_low_resolution_preview(self):
        class ScaleCapture:
            def __init__(self):
                self.maximum_scale = None

            def release(self):
                pass

            def paint(
                self,
                painter,
                option,
                mapper,
                source_rect,
                cache_key,
                cache_allowed,
                paint_source,
                maximum_scale=None,
            ):
                self.maximum_scale = maximum_scale

        item, _ = self._make_pair(99, TEST_LINES[3], False)
        item.set_text_transform(
            transform_state(
                PerspectiveTextTransform(0.6, 45.0),
                CurvatureTextTransform(0.7),
            )
        )
        capture = ScaleCapture()
        item.geometry_controller.surface_renderer = capture
        image = QImage(
            900,
            600,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(0, 0, 0, 0))
        option = QStyleOptionGraphicsItem()
        option.exposedRect = item.boundingRect()

        item.reshaping = True
        painter = QPainter(image)
        item.geometry_controller.paint_item(
            painter, option, None, lambda *_: None
        )
        painter.end()
        self.assertEqual(capture.maximum_scale, 0.5)

        item.reshaping = False
        painter = QPainter(image)
        item.geometry_controller.paint_item(
            painter, option, None, lambda *_: None
        )
        painter.end()
        self.assertIsNone(capture.maximum_scale)

    def test_surface_composition_renders_through_one_nonlinear_surface(self):
        stack = TextTransformStack((
            SlantTextTransform(1.1, 0.9, 5.0),
            CurvatureTextTransform(0.55),
            PerspectiveTextTransform(0.25, 30.0),
            CurvatureTextTransform(-0.2),
        ))
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item, _ = self._make_pair(99, TEST_LINES[3], vertical)
                scene = QGraphicsScene()
                scene.addItem(item)
                neutral_pixels = self._render_scene(scene)

                item.set_text_transform(TextTransformState(stack))
                controller = item.geometry_controller
                self.assertTrue(item.transform().isIdentity())
                self.assertIsInstance(
                    controller.visual_mapper,
                    CompositeTextTransformMapper,
                )
                renderer = controller.surface_renderer
                self.assertIsNotNone(renderer)
                self.assertNotEqual(self._render_scene(scene), neutral_pixels)
                self.assertIs(controller.surface_renderer, renderer)
                cached_pixmap = renderer.cached_pixmap
                self.assertIsNotNone(cached_pixmap)

                self.assertFalse(controller.refresh_compiled_geometry())
                self.assertIs(renderer.cached_pixmap, cached_pixmap)
                item.set_text_transform(NEUTRAL)
                self.assertIsNone(controller.surface_renderer)
                self.assertEqual(self._render_scene(scene), neutral_pixels)
                controller.release_render_resources()
                scene.removeItem(item)
                self.app.processEvents()

    def test_curvature_selection_requests_a_full_item_update(self):
        class RecordingTextItem(TextBlkItem):
            def __init__(self, *args, **kwargs):
                self.full_update_count = 0
                super().__init__(*args, **kwargs)

            def update(self, *args):
                if not args:
                    self.full_update_count += 1
                return super().update(*args)

        block = TextBlock(
            [40, 40, 540, 220],
            _bounding_rect=[40, 40, 500, 180],
            translation=TEST_LINES[3],
        )
        item = RecordingTextItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)
        item.set_text_transform(
            transform_state(CurvatureTextTransform(0.8))
        )
        item.startEdit()
        item.full_update_count = 0

        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_A,
            Qt.KeyboardModifier.ControlModifier,
        )
        item.keyPressEvent(event)
        self.app.processEvents()

        self.assertTrue(item.textCursor().hasSelection())
        self.assertGreaterEqual(item.full_update_count, 1)
        item.endEdit()

    def test_curvature_defers_and_overlays_cursor_after_surface_warp(self):
        class SourceCapture:
            def __init__(self, cursor_position):
                self.cursor_position = cursor_position
                self.saw_deferred_cursor = False

            def release(self):
                pass

            def paint(
                self,
                painter,
                option,
                mapper,
                source_rect,
                cache_key,
                cache_allowed,
                paint_source,
                maximum_scale=None,
            ):
                def base_paint(*_):
                    self.saw_deferred_cursor = (
                        item.layout.defer_cursor_paint
                    )
                    item.layout.deferred_cursor_position = (
                        self.cursor_position
                    )

                original = item.effect_renderer.paint_item
                item.effect_renderer.paint_item = (
                    lambda source_painter, source_option, source_widget,
                    _base_paint: base_paint()
                )
                try:
                    paint_source(painter, option, None)
                finally:
                    item.effect_renderer.paint_item = original

        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                width, height = (
                    (180, 500) if vertical else (500, 180)
                )
                block = TextBlock(
                    [40, 40, 40 + width, 40 + height],
                    _bounding_rect=[40, 40, width, height],
                    translation=TEST_LINES[3],
                )
                block.vertical = vertical
                item = TextBlkItem(block, 0)
                item.set_text_transform(
                    transform_state(CurvatureTextTransform(0.8))
                )
                item.startEdit()
                cursor = item.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                item.setTextCursor(cursor)
                capture = SourceCapture(cursor.position())
                item.geometry_controller.surface_renderer = capture

                image = QImage(
                    900,
                    700,
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                image.fill(QColor(127, 127, 127))
                before = bytes(
                    image.bits().asstring(image.sizeInBytes())
                )
                painter = QPainter(image)
                painter.translate(100, 100)
                option = QStyleOptionGraphicsItem()
                option.exposedRect = item.boundingRect()
                item.geometry_controller.paint_item(
                    painter, option, None, lambda *_: None
                )
                painter.end()
                after = bytes(image.bits().asstring(image.sizeInBytes()))
                self.assertTrue(capture.saw_deferred_cursor)
                self.assertNotEqual(after, before)
                item.endEdit()

    def test_warped_curvature_surface_maps_layout_hit_tests(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                width, height = (
                    (180, 500) if vertical else (500, 180)
                )
                block = TextBlock(
                    [40, 40, 40 + width, 40 + height],
                    _bounding_rect=[40, 40, width, height],
                    translation=TEST_LINES[3],
                )
                block.vertical = vertical
                item = TextBlkItem(block, 0)
                scene = QGraphicsScene()
                scene.addItem(item)
                neutral_pixels = self._render_scene(scene)
                source_point = item.logical_unpadded_rect().center()
                neutral_hit = item.layout.hitTest(source_point, None)

                item.set_text_transform(
                    transform_state(CurvatureTextTransform(0.7))
                )
                mapper = item.geometry_controller.visual_mapper
                self.assertIsNotNone(mapper)
                self.assertIs(
                    item.layout.input_point_mapper.__self__,
                    item.geometry_controller,
                )
                visual_point = mapper.forward_point(source_point)
                self.assertEqual(
                    item.layout.hitTest(visual_point, None),
                    neutral_hit,
                )
                item.startEdit()
                source_cursor_rect = QGraphicsTextItem.inputMethodQuery(
                    item, Qt.InputMethodQuery.ImCursorRectangle
                )
                self.assertEqual(
                    item.inputMethodQuery(
                        Qt.InputMethodQuery.ImCursorRectangle
                    ),
                    mapper.map_rect_path(
                        QRectF(source_cursor_rect)
                    ).boundingRect(),
                )
                item.endEdit()
                curved_pixels = self._render_scene(scene)
                self.assertNotEqual(curved_pixels, neutral_pixels)
                self.assertIsNotNone(
                    item.geometry_controller.surface_renderer.cached_pixmap
                )
                self.assertTrue(item.contains(visual_point))

                item.set_text_transform(NEUTRAL)
                self.assertIsNone(item.geometry_controller.visual_mapper)
                self.assertIsNone(item.geometry_controller.surface_renderer)
                self.assertIsNone(item.layout.input_point_mapper)
                self.assertEqual(self._render_scene(scene), neutral_pixels)
                item.geometry_controller.release_render_resources()
                scene.removeItem(item)
                self.app.processEvents()

    def test_fresh_items_install_perspective_and_curvature(self):
        transforms = (
            PerspectiveTextTransform(0.6, 25.0),
            CurvatureTextTransform(-0.6),
        )
        for vertical in (False, True):
            for transform in transforms:
                with self.subTest(
                    vertical=vertical,
                    transform=transform.transform_type,
                ):
                    width, height = (
                        (180, 500) if vertical else (500, 180)
                    )
                    block = TextBlock(
                        [20, 30, 20 + width, 30 + height],
                        _bounding_rect=[20, 30, width, height],
                        translation=TEST_LINES[1],
                    )
                    block.vertical = vertical
                    block.fontformat.text_transform = TextTransformStack(
                        (transform,)
                    )
                    item = TextBlkItem(copy.deepcopy(block), 0)
                    if transform.transform_type == 'perspective':
                        self.assertFalse(item.transform().isIdentity())
                        self.assertIsNone(
                            item.geometry_controller.visual_mapper
                        )
                    else:
                        self.assertTrue(item.transform().isIdentity())
                        self.assertIsNotNone(
                            item.geometry_controller.visual_mapper
                        )
                        self.assertIsNotNone(
                            item.layout.input_point_mapper
                        )

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

    def test_persisted_box_transform_is_installed_on_fresh_items(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                block = TextBlock([40, 50, 440, 250])
                block._bounding_rect = [40, 50, 400, 200]
                block.vertical = vertical
                block.angle = 17.0
                block.translation = TEST_LINES[0]
                block.fontformat.text_transform = FIRST_TRANSFORM.stack
                block.fontformat.glyph_slant_angle = (
                    FIRST_TRANSFORM.glyph_slant_angle
                )

                for source in (block, copy.deepcopy(block)):
                    item = TextBlkItem(source, 0)
                    expected = item.geometry_controller.compensated_matrix()
                    self.assertFalse(item.transform().isIdentity())
                    self.assertEqual(item.transform(), expected)

                    before = item.transform()
                    item.setRect(item.absBoundingRect(qrect=True))
                    self.assertEqual(item.transform(), before)

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
        box_preview = transform_state(
            FIRST_TRANSFORM.stack[0].with_value(
                'horizontal_scale', 1.4
            ),
            glyph_slant_angle=FIRST_TRANSFORM.glyph_slant_angle,
        )
        item.set_text_transform(box_preview, preview=True)
        self.assertTrue(geometry_cache.persistent)
        item.clear_text_transform_preview()
        glyph_preview = TextTransformState(FIRST_TRANSFORM.stack, 9.0)
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

                box_only = transform_state(
                    SlantTextTransform(1.2, 0.9, 8.0)
                )
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


class TextTransformGeometryTest(TextTransformTestBase):
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

    def test_transformed_control_hitboxes_stay_outside_text_item(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                scene = QGraphicsScene()
                view = QGraphicsView(scene)
                view.resize(900, 600)
                base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
                scene.addItem(base)
                item, _ = self._make_pair(0, TEST_LINES[0], vertical)
                item.setParentItem(base)
                item.set_logical_position(QPointF(100, 100))
                item.set_text_transform(FIRST_TRANSFORM)
                item.setRotation(23.0)
                control = TextBlkShapeControl(view)
                control.setParentItem(base)
                control.setBlkItem(item)
                view.show()
                self.app.processEvents()
                control.updateBoundingRect()
                self.addCleanup(view.close)

                true_points = [
                    control.handleScenePoint(index) for index in range(8)
                ]
                outward_vectors = control._handle_outward_vectors_device(
                    true_points
                )
                viewport_transform = view.viewportTransform()
                item_transform = item.deviceTransform(viewport_transform)
                item_origin = item_transform.map(QPointF())
                item_axis = item_transform.map(QPointF(1, 0)) - item_origin
                item_angle = math.degrees(
                    math.atan2(item_axis.y(), item_axis.x())
                )
                for index, handle in enumerate(control.ctrlblock_group):
                    with self.subTest(vertical=vertical, handle=index):
                        anchor = viewport_transform.map(true_points[index])
                        hitbox = viewport_transform.map(
                            handle.mapToScene(handle.rect())
                        )
                        visible = viewport_transform.map(
                            handle.mapToScene(handle.visible_rect)
                        )
                        outward = outward_vectors[index]
                        hitbox_projections = [
                            (point.x() - anchor.x()) * outward.x()
                            + (point.y() - anchor.y()) * outward.y()
                            for point in hitbox
                        ]
                        projections = [
                            (point.x() - anchor.x()) * outward.x()
                            + (point.y() - anchor.y()) * outward.y()
                            for point in visible
                        ]
                        self.assertGreaterEqual(
                            min(hitbox_projections), -0.1
                        )
                        self.assertAlmostEqual(
                            min(projections), 0.0, delta=0.1
                        )
                        handle_transform = handle.deviceTransform(
                            viewport_transform
                        )
                        handle_origin = handle_transform.map(QPointF())
                        handle_axis = (
                            handle_transform.map(QPointF(1, 0))
                            - handle_origin
                        )
                        handle_angle = math.degrees(
                            math.atan2(handle_axis.y(), handle_axis.x())
                        )
                        angle_delta = (
                            handle_angle - item_angle + 180
                        ) % 360 - 180
                        self.assertAlmostEqual(
                            angle_delta, 0.0, delta=0.01
                        )


class TextTransformShapeControlTest(TextTransformTestBase):
    def test_curvature_resize_uses_frozen_drag_coordinates(self):
        for vertical in (False, True):
            for curvature in (-0.95, 0.95):
                with self.subTest(
                    vertical=vertical,
                    curvature=curvature,
                ):
                    scene = QGraphicsScene()
                    view = QGraphicsView(scene)
                    base = QGraphicsRectItem(QRectF(0, 0, 900, 700))
                    scene.addItem(base)
                    width, height = (
                        (180, 420) if vertical else (420, 180)
                    )
                    block = TextBlock(
                        [180, 120, 180 + width, 120 + height],
                        _bounding_rect=[180, 120, width, height],
                        translation=TEST_LINES[0],
                    )
                    block.vertical = vertical
                    item = TextBlkItem(block, 0)
                    item.setParentItem(base)
                    item.set_text_transform(
                        transform_state(
                            CurvatureTextTransform(curvature)
                        )
                    )
                    item.setRotation(23.0)
                    control = TextBlkShapeControl(view)
                    control.setParentItem(base)
                    control.setBlkItem(item)
                    view.resize(1000, 800)
                    view.show()
                    self.app.processEvents()
                    self.addCleanup(view.close)

                    handle_index = 5 if vertical else 3
                    opposite_index = (handle_index + 4) % 8
                    mapper = item.geometry_controller.visual_mapper
                    initial_scene_transform = item.sceneTransform()
                    initial_source = QPointF(
                        item.geometry_controller.source_handle_points()[
                            handle_index
                        ]
                    )
                    initial_handle_scene = initial_scene_transform.map(
                        mapper.forward_point(initial_source)
                    )
                    opposite_scene = control.handleScenePoint(
                        opposite_index
                    )
                    initial_size = (
                        item.absBoundingRect(qrect=True).height()
                        if vertical
                        else item.absBoundingRect(qrect=True).width()
                    )

                    control.beginResize(
                        handle_index, initial_handle_scene
                    )
                    sizes = []
                    for extension in (20.0, 40.0, 60.0):
                        target_source = QPointF(initial_source)
                        if vertical:
                            target_source.setY(
                                target_source.y() + extension
                            )
                        else:
                            target_source.setX(
                                target_source.x() + extension
                            )
                        target_scene = initial_scene_transform.map(
                            mapper.forward_point(target_source)
                        )
                        control.resizeFromScene(
                            handle_index, target_scene
                        )
                        rect = item.absBoundingRect(qrect=True)
                        sizes.append(
                            rect.height() if vertical else rect.width()
                        )
                        anchored = control.handleScenePoint(
                            opposite_index
                        )
                        self.assertAlmostEqual(
                            anchored.x(), opposite_scene.x(), places=4
                        )
                        self.assertAlmostEqual(
                            anchored.y(), opposite_scene.y(), places=4
                        )

                    self.assertGreater(sizes[0], initial_size)
                    self.assertGreater(sizes[1], sizes[0])
                    self.assertGreater(sizes[2], sizes[1])
                    control.finishResize()
                    control.setBlkItem(None)
                    item.geometry_controller.release_render_resources()
                    scene.removeItem(base)
                    view.close()
                    self.app.processEvents()

    def test_extended_transform_shape_control_tracks_geometry(self):
        for vertical in (False, True):
            for state in (
                transform_state(
                    PerspectiveTextTransform(0.55, 35.0)
                ),
                transform_state(CurvatureTextTransform(0.65)),
                transform_state(
                    PerspectiveTextTransform(0.55, 35.0),
                    CurvatureTextTransform(0.65),
                ),
            ):
                with self.subTest(
                    vertical=vertical,
                    transforms=tuple(
                        transform.transform_type
                        for transform in state.stack
                    ),
                ):
                    scene = QGraphicsScene()
                    view = QGraphicsView(scene)
                    width, height = (
                        (160, 420) if vertical else (420, 160)
                    )
                    base = QGraphicsRectItem(QRectF(0, 0, 700, 550))
                    scene.addItem(base)
                    block = TextBlock(
                        [100, 70, 100 + width, 70 + height],
                        _bounding_rect=[100, 70, width, height],
                        translation=TEST_LINES[0],
                    )
                    block.vertical = vertical
                    item = TextBlkItem(block, 0)
                    item.setParentItem(base)
                    item.set_text_transform(state)
                    control = TextBlkShapeControl(view)
                    control.setParentItem(base)
                    control.setBlkItem(item)
                    view.resize(800, 650)
                    view.show()
                    self.app.processEvents()
                    self.addCleanup(view.close)

                    for angle in (0.0, 27.0):
                        item.setRotation(angle)
                        control.updateBoundingRect()
                        expected = (
                            item.geometry_controller
                            .visual_handle_points_in_scene()
                        )
                        for index, point in enumerate(expected):
                            actual = control.handleScenePoint(index)
                            self.assertAlmostEqual(
                                actual.x(), point.x(), places=5
                            )
                            self.assertAlmostEqual(
                                actual.y(), point.y(), places=5
                            )
                        self.assertEqual(
                            control.visualPolygonInScene().boundingRect(),
                            item.visual_bounds_in_scene(),
                        )
                    control.setBlkItem(None)
                    item.geometry_controller.release_render_resources()
                    scene.removeItem(base)
                    view.close()
                    self.app.processEvents()

    def test_control_click_preserves_multi_selection(self):
        for transform in (
            NEUTRAL,
            FIRST_TRANSFORM,
            PerspectiveTextTransform(0.5, 25.0),
            CurvatureTextTransform(0.55),
        ):
            state = (
                transform
                if isinstance(transform, TextTransformState)
                else transform_state(transform)
            )
            with self.subTest(
                transform=(
                    state.stack[0].transform_type
                    if state.stack.transforms
                    else 'none'
                )
            ):
                scene = QGraphicsScene()
                view = QGraphicsView(scene)
                view.resize(500, 260)
                base = QGraphicsRectItem(QRectF(0, 0, 420, 180))
                scene.addItem(base)
                items = []
                for index, x in enumerate((20, 180)):
                    block = TextBlock(
                        [x, 30, x + 100, 70],
                        _bounding_rect=[x, 30, 100, 40],
                        translation=TEST_LINES[index],
                    )
                    item = TextBlkItem(block, index)
                    item.setParentItem(base)
                    item.set_text_transform(state)
                    items.append(item)

                control = TextBlkShapeControl(view)
                control.setParentItem(base)
                control.setBlkItem(items[1])
                items[0].setSelected(True)
                move_interactions = []
                items[1].move_interaction_finished.connect(
                    lambda: move_interactions.append(True)
                )
                view.show()
                self.app.processEvents()
                control.updateBoundingRect()
                self.addCleanup(view.close)

                logical = items[1].logical_unpadded_rect()
                interior = items[1].geometry_controller.map_source_to_scene(
                    QPointF(logical.center().x(), logical.top() + 4.0)
                )
                QTest.mouseClick(
                    view.viewport(),
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.ControlModifier,
                    view.mapFromScene(interior),
                )
                self.app.processEvents()
                self.assertTrue(items[0].isSelected())
                self.assertTrue(items[1].isSelected())
                self.assertEqual(move_interactions, [True])

                handle_center = control.handleDisplayScenePoint(1)
                QTest.mouseClick(
                    view.viewport(),
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.ControlModifier,
                    view.mapFromScene(handle_center),
                )
                self.app.processEvents()
                self.assertTrue(items[0].isSelected())
                self.assertFalse(items[1].isSelected())
                control.setBlkItem(None)
                for item in items:
                    item.geometry_controller.release_render_resources()
                scene.removeItem(base)
                view.close()
                self.app.processEvents()

    def test_shape_controls_reset_across_page_item_boundary(self):
        for stale_state in ("editing", "reshape"):
            with self.subTest(stale_state=stale_state):
                scene = QGraphicsScene()
                view = QGraphicsView(scene)
                base = QGraphicsRectItem(QRectF(0, 0, 500, 300))
                scene.addItem(base)
                old_item, _ = self._make_pair(0, TEST_LINES[0], False)
                new_item, _ = self._make_pair(1, TEST_LINES[1], True)
                new_item.set_text_transform(FIRST_TRANSFORM)
                old_item.setParentItem(base)
                new_item.setParentItem(base)
                control = TextBlkShapeControl(view)
                control.setParentItem(base)
                control.setBlkItem(old_item)
                self.addCleanup(view.close)

                if stale_state == "editing":
                    old_item.startEdit()
                    control.startEditing()
                else:
                    control.reshaping = True
                    control.ctrlblock_group[0].drag_mode = (
                        control.ctrlblock_group[0].DRAG_RESHAPE
                    )
                    control.hideControls()

                control.setBlkItem(None)
                control.setBlkItem(new_item)

                self.assertFalse(control.reshaping)
                self.assertTrue(control.isVisible())
                self.assertTrue(
                    all(
                        handle.isVisible()
                        and handle.drag_mode == handle.DRAG_NONE
                        for handle in control.ctrlblock_group
                    )
                )

    def test_shape_control_outline_contrasts_with_background(self):
        view = QGraphicsView()
        control = TextBlkShapeControl(view)
        control.setRect(QRectF(20, 20, 80, 40))
        self.addCleanup(view.close)

        def render(background):
            image = QImage(120, 80, QImage.Format.Format_ARGB32)
            image.fill(background)
            painter = QPainter(image)
            control.paint(
                painter,
                QStyleOptionGraphicsItem(),
                None,
            )
            painter.end()
            return image

        on_black = render(QColor(Qt.GlobalColor.black))
        on_white = render(QColor(Qt.GlobalColor.white))
        contrast_pixels = []
        for y in range(on_black.height()):
            for x in range(on_black.width()):
                black_pixel = on_black.pixelColor(x, y)
                white_pixel = on_white.pixelColor(x, y)
                if (
                    black_pixel.red() > 220
                    and black_pixel.green() > 220
                    and black_pixel.blue() > 220
                    and white_pixel.red() < 35
                    and white_pixel.green() < 35
                    and white_pixel.blue() < 35
                ):
                    contrast_pixels.append((x, y))
        self.assertTrue(contrast_pixels)

    def test_item_owned_selection_guide_can_be_suppressed_for_export(self):
        for transform in (NEUTRAL, FIRST_TRANSFORM):
            with self.subTest(
                transform=(
                    transform.stack[0].transform_type
                    if transform.stack.transforms
                    else 'none'
                )
            ):
                item, _ = self._make_pair(0, TEST_LINES[3], False)
                item.set_logical_position(QPointF(50, 50))
                item.set_text_transform(transform)
                scene = QGraphicsScene()
                scene.addItem(item)
                self.app.processEvents()

                unselected = self._render_scene(scene)
                item.setSelected(True)
                self.app.processEvents()
                selected = self._render_scene(scene)
                self.assertNotEqual(selected, unselected)

                item.under_ctrl = True
                item.update()
                self.app.processEvents()
                self.assertEqual(self._render_scene(scene), selected)

                item.set_ui_guide_suppressed(True)
                self.app.processEvents()
                self.assertEqual(self._render_scene(scene), unselected)


if __name__ == "__main__":
    unittest.main()
