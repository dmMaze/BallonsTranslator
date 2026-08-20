import copy
import json
import math
import os
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QEvent, QPointF, QRectF, Qt
from qtpy.QtGui import (
    QColor,
    QImage,
    QInputMethodEvent,
    QKeyEvent,
    QPainter,
    QTextCharFormat,
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

from ballontranslator.ui.text_engine.editing.widgets import TransPairWidget
from ballontranslator.ui.text_engine.editing.commands import (
    CapitalizeTextItemsCommand,
    MoveBlkItemsCommand,
    MultiPasteCommand,
    ReshapeItemCommand,
    SetTextTransformCommand,
    TextEditCommand,
    propagate_user_edit,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.transforms.panel import TextTransformPanel
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.transforms.controls import (
    CommittedTransformControl,
)
from ballontranslator.ui.text_engine.shape_control import TextBlkShapeControl
from ballontranslator.ui.text_engine.transforms.edit_session import (
    TextTransformEditSession,
)
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.text_engine.rendering.glyph import (
    GLOBAL_GLYPH_GEOMETRY_CACHE,
    GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE,
    GlyphGeometry,
    WeightedGlyphGeometryCache,
)
from ballontranslator.ui.text_engine.rendering.glyph_slant import (
    GlyphSlantLayoutRenderer,
)
from ballontranslator.ui.text_engine.rendering.surface import (
    NonlinearTextSurfaceRenderer,
)
from ballontranslator.ui.text_engine.transforms.bend import BendMapper
from ballontranslator.ui.text_engine.transforms.grid import GridMapper
from ballontranslator.ui.text_engine.transforms.sine import SineMapper
from ballontranslator.ui.text_engine.transforms.grid_control import TextGridTransformControl
from ballontranslator.ui.text_engine.transforms.projective_control import (
    PROJECTIVE_CONTROL_RADIUS,
    TextProjectiveTransformControl,
)
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.text_engine.transforms.modal import ModalPointTransform
from ballontranslator.ui.text_engine.transforms.mapping import (
    CompositeTextTransformMapper,
    projective_transform_matrix,
)
from ballontranslator.ui.text_engine.transforms.registry import (
    compile_text_transform_stack,
)
from ballontranslator.utils.fontformat import (
    BendTextTransform,
    FontFormat,
    GridTextTransform,
    ProjectiveTextTransform,
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils import shared
from ballontranslator.utils import config as C
from ballontranslator.utils.proj_imgtrans import TextBlkEncoder
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
    return TextTransformStack(tuple(transforms), glyph_slant_angle)


NEUTRAL = transform_state()
FIRST_TRANSFORM = transform_state(
    ProjectiveTextTransform(1.2, 0.9, 12.0),
    glyph_slant_angle=5.0,
)
FINAL_TRANSFORMS = (
    transform_state(
        ProjectiveTextTransform(0.8, 1.1, -9.0),
        glyph_slant_angle=-4.0,
    ),
    transform_state(
        ProjectiveTextTransform(1.3, 0.7, 6.0),
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
        pair = TransPairWidget(index, False)
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
                item.blk.fontformat.text_transform,
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

        def on_propagate(position, removed, added_text, joint_previous):
            propagated.append(
                (position, removed, added_text, joint_previous)
            )
            propagate_user_edit(
                item,
                position,
                removed,
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
        self.assertEqual(
            propagated,
            [(len(text_before), 0, commit_text, False)],
        )
        self.assertEqual(pushed_steps, [1])
        self.assertEqual(stack.count(), stack_count + 1)
        self.assertEqual(edit.toPlainText(), text_before + commit_text)
        self.assertEqual(item.toPlainText(), text_before + commit_text)


class ExtendedTextTransformModelTest(TextTransformTestBase):
    def test_projective_and_bend_payloads_round_trip(self):
        payloads = (
            {
                'transform_type': 'projective',
                'horizontal_scale': 1.2,
                'vertical_scale': 0.9,
                'horizontal_slant': 8.0,
                'vertical_slant': -4.0,
                'rotation_x': 25.0,
                'rotation_y': -35.0,
                'rotation_z': 12.0,
                'perspective': 0.55,
            },
            {
                'transform_type': 'bend',
                'bend': -0.75,
            },
        )
        for payload in payloads:
            with self.subTest(transform_type=payload['transform_type']):
                font_format = FontFormat(text_transform=[payload])
                self.assertEqual(
                    font_format.to_serializable_dict()['text_transform'],
                    [payload],
                )

    def test_duplicate_stack_entries_and_glyph_slant_round_trip(self):
        payload = [
            {
                'transform_type': 'projective',
                'horizontal_scale': 1.2,
                'vertical_scale': 1.0,
                'horizontal_slant': 5.0,
                'vertical_slant': 0.0,
                'rotation_x': 0.0,
                'rotation_y': 0.0,
                'rotation_z': 0.0,
                'perspective': 0.0,
            },
            {
                'transform_type': 'projective',
                'horizontal_scale': 0.8,
                'vertical_scale': 1.1,
                'horizontal_slant': -3.0,
                'vertical_slant': 0.0,
                'rotation_x': 0.0,
                'rotation_y': 0.0,
                'rotation_z': 0.0,
                'perspective': 0.0,
            },
        ]
        font_format = FontFormat(
            text_transform=payload,
            glyph_slant_angle=7.0,
        )
        self.assertEqual(font_format.text_transform.glyph_slant_angle, 7.0)
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
        self.assertEqual(
            restored.fontformat.text_transform.glyph_slant_angle,
            7.0,
        )
        self.assertEqual(restored.fontformat.glyph_slant_angle, 7.0)

    def test_projective_matrix_is_centered_and_invertible(self):
        rect = QRectF(20, 30, 400, 180)
        for rotation_x, rotation_y in (
            (-89.0, -45.0),
            (-30.0, 60.0),
            (0.0, 0.0),
            (45.0, 30.0),
            (89.0, 89.0),
        ):
            with self.subTest(rotation_x=rotation_x, rotation_y=rotation_y):
                matrix = projective_transform_matrix(
                    ProjectiveTextTransform(
                        horizontal_scale=1.2,
                        vertical_scale=0.9,
                        horizontal_slant=8.0,
                        vertical_slant=-4.0,
                        rotation_x=rotation_x,
                        rotation_y=rotation_y,
                        rotation_z=20.0,
                        perspective=0.8,
                    ),
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

    def test_projective_parameters_compile_to_one_native_stage_matrix(self):
        rect = QRectF(20, 30, 400, 180)
        transform = ProjectiveTextTransform(
            horizontal_scale=1.2,
            vertical_scale=0.85,
            horizontal_slant=12.0,
            vertical_slant=-7.0,
            rotation_x=25.0,
            rotation_y=-35.0,
            rotation_z=18.0,
            perspective=0.65,
        )

        compiled = compile_text_transform_stack(
            TextTransformStack((transform,)), rect, rect, False
        )

        self.assertIsNone(compiled.surface_mapper)
        self.assertEqual(len(compiled.stages), 1)
        self.assertIsNotNone(compiled.stages[0].mapper)
        self.assertEqual(
            compiled.native_matrix,
            compiled.stages[0].mapper.matrix,
        )
        self.assertEqual(
            compiled.native_matrix,
            projective_transform_matrix(transform, rect),
        )

    def test_live_transform_boundaries_require_typed_stacks(self):
        rect = QRectF(0, 0, 100, 50)
        with self.assertRaisesRegex(TypeError, 'requires TextTransformStack'):
            compile_text_transform_stack(
                [ProjectiveTextTransform()], rect, rect, False
            )

        item, _ = self._make_pair(0, TEST_LINES[0], False)
        with self.assertRaisesRegex(TypeError, 'require TextTransformStack'):
            item.set_text_transform(None)

    def test_persisted_transform_values_are_not_range_validated(self):
        font_format = FontFormat(
            text_transform=[{
                'transform_type': 'projective',
                'horizontal_scale': 5.0,
            }],
            glyph_slant_angle=50.0,
        )
        block = TextBlock([0, 0, 20, 10])
        block.fontformat = font_format
        restored = TextBlock(
            **json.loads(json.dumps(block, cls=TextBlkEncoder))
        )

        self.assertEqual(
            restored.fontformat.text_transform[0].horizontal_scale,
            5.0,
        )
        self.assertEqual(restored.fontformat.glyph_slant_angle, 50.0)

    def test_bend_mapper_round_trips_both_writing_modes(self):
        for vertical in (False, True):
            logical = (
                QRectF(10, 20, 160, 420)
                if vertical
                else QRectF(10, 20, 420, 160)
            )
            source = logical.adjusted(-12, -12, 12, 12)
            for bend in (-1.0, -0.4, 0.0, 0.4, 1.0):
                with self.subTest(
                    vertical=vertical, bend=bend
                ):
                    mapper = BendMapper(
                        logical, source, vertical, bend
                    )
                    source_points = []
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
                        source_points.append(point)
                        restored = mapper.inverse_point(
                            mapper.forward_point(point)
                        )
                        self.assertAlmostEqual(
                            restored.x(), point.x(), places=6
                        )
                        self.assertAlmostEqual(
                            restored.y(), point.y(), places=6
                        )
                    source_x = np.asarray([
                        point.x() for point in source_points
                    ])
                    source_y = np.asarray([
                        point.y() for point in source_points
                    ])
                    visual_x, visual_y = mapper.forward_arrays(
                        source_x, source_y
                    )
                    for index, point in enumerate(source_points):
                        expected = mapper.forward_point(point)
                        self.assertAlmostEqual(
                            visual_x[index], expected.x(), places=6
                        )
                        self.assertAlmostEqual(
                            visual_y[index], expected.y(), places=6
                        )

    def test_sine_payload_neutrality_and_phase_endpoint(self):
        transform = SineTextTransform(
            frequency_x=64,
            frequency_y=3,
            phase_x=1.0,
            phase_y=0.25,
            amplitude_x=1.0,
            amplitude_y=0.4,
        )
        payload = {
            'transform_type': 'sine',
            'frequency_x': 64,
            'frequency_y': 3,
            'phase_x': 1.0,
            'phase_y': 0.25,
            'amplitude_x': 1.0,
            'amplitude_y': 0.4,
        }
        font_format = FontFormat(text_transform=[payload])
        self.assertEqual(font_format.text_transform[0], transform)
        self.assertEqual(
            font_format.to_serializable_dict()['text_transform'], [payload]
        )
        self.assertTrue(SineTextTransform(frequency_x=0).is_neutral())
        self.assertTrue(SineTextTransform(
            amplitude_x=0.0, amplitude_y=0.0
        ).is_neutral())
        self.assertFalse(SineTextTransform().is_neutral())
        self.assertEqual(
            TextTransformStack((SineTextTransform(frequency_x=0.5),))[0]
            .frequency_x,
            0.5,
        )

    def test_sine_mapper_round_trips_ordered_axes_at_extreme_values(self):
        logical = QRectF(10, 20, 420, 160)
        source = logical.adjusted(-12, -12, 12, 12)
        transform = SineTextTransform(
            frequency_x=64,
            frequency_y=64,
            phase_x=1.0,
            phase_y=0.375,
            amplitude_x=1.0,
            amplitude_y=1.0,
        )
        mapper = SineMapper(logical, source, transform)
        source_x = np.asarray([10.0, 61.25, 180.0, 333.75, 430.0])
        source_y = np.asarray([20.0, 44.5, 90.0, 151.5, 180.0])
        visual_x, visual_y = mapper.forward_arrays(source_x, source_y)
        restored_x, restored_y, valid = mapper.inverse_arrays(
            visual_x, visual_y, return_valid=True
        )
        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(restored_x, source_x, atol=1e-9))
        self.assertTrue(np.allclose(restored_y, source_y, atol=1e-9))
        for index in range(len(source_x)):
            point = QPointF(source_x[index], source_y[index])
            mapped = mapper.forward_point(point)
            self.assertAlmostEqual(mapped.x(), visual_x[index], places=6)
            self.assertAlmostEqual(mapped.y(), visual_y[index], places=6)
            restored = mapper.inverse_point(mapped)
            self.assertAlmostEqual(restored.x(), point.x(), places=6)
            self.assertAlmostEqual(restored.y(), point.y(), places=6)
        self.assertGreater(
            mapper.map_rect_path(logical).boundingRect().height(),
            logical.height() * 2.9,
        )

        compiled = compile_text_transform_stack(
            TextTransformStack((transform,)), logical, source, False
        )
        self.assertIsInstance(
            compiled.surface_mapper, CompositeTextTransformMapper
        )
        self.assertTrue(
            compiled.surface_mapper.visual_bounds().contains(source)
        )
        default_bounds = SineMapper(
            logical, source, SineTextTransform()
        ).visual_bounds()
        self.assertEqual(default_bounds.left(), source.left())
        self.assertEqual(default_bounds.right(), source.right())
        self.assertEqual(
            default_bounds.top(), source.top() - logical.height() * 0.1
        )

    def test_grid_payload_divisions_and_interpolation_round_trip(self):
        grid = GridTextTransform(3, 2, 'catmull_rom')
        self.assertEqual(len(grid.control_points), 12)
        self.assertTrue(grid.is_neutral())
        self.assertEqual(
            len(GridTextTransform().control_points),
            4,
        )
        self.assertEqual(len(GridTextTransform(33, 1).control_points), 68)

        points = list(grid.control_points)
        points[5] = (0.7, 0.35)
        grid = grid.with_control_points(points)
        font_format = FontFormat(text_transform=[
            {
                'transform_type': 'grid',
                'horizontal_divisions': grid.horizontal_divisions,
                'vertical_divisions': grid.vertical_divisions,
                'interpolation': grid.interpolation,
                'control_points': grid.control_points,
            }
        ])
        restored = FontFormat(**json.loads(json.dumps(
            font_format.to_serializable_dict()
        )))
        self.assertEqual(restored.text_transform[0], grid)

    def test_grid_bilinear_and_catmull_rom_differ_between_anchors(self):
        logical = QRectF(0, 0, 400, 200)
        source = logical.adjusted(-10, -10, 10, 10)
        base = GridTextTransform(2, 2)
        points = list(base.control_points)
        points[4] = (0.7, 0.3)
        bilinear = GridMapper(
            logical,
            source,
            base.with_control_points(points),
        )
        catmull_rom = GridMapper(
            logical,
            source,
            base.with_control_points(points).with_value(
                'interpolation', 'catmull_rom'
            ),
        )
        anchor = QPointF(200, 100)
        expected_anchor = QPointF(280, 60)
        self.assertEqual(bilinear.forward_point(anchor), expected_anchor)
        self.assertEqual(catmull_rom.forward_point(anchor), expected_anchor)
        between = QPointF(100, 100)
        self.assertNotEqual(
            bilinear.forward_point(between),
            catmull_rom.forward_point(between),
        )
        for mapper in (bilinear, catmull_rom):
            for point in (QPointF(40, 30), QPointF(180, 80), QPointF(350, 170)):
                restored = mapper.inverse_point(mapper.forward_point(point))
                self.assertAlmostEqual(restored.x(), point.x(), places=5)
                self.assertAlmostEqual(restored.y(), point.y(), places=5)

        protruding = list(base.control_points)
        protruding[4] = (1.5, -0.5)
        protruding_mapper = GridMapper(
            logical,
            source,
            base.with_control_points(protruding),
        )
        self.assertTrue(
            protruding_mapper.visual_bounds().contains(QPointF(600, -100))
        )

    def test_grid_inverse_stops_after_convergence(self):
        coordinates = np.meshgrid(
            np.linspace(0.0, 400.0, 24),
            np.linspace(0.0, 200.0, 12),
        )
        for interpolation in ('bilinear', 'catmull_rom'):
            with self.subTest(interpolation=interpolation):
                grid = GridTextTransform(2, 2, interpolation)
                points = list(grid.control_points)
                points[4] = (0.6, 0.4)
                mapper = GridMapper(
                    QRectF(0, 0, 400, 200),
                    QRectF(0, 0, 400, 200),
                    grid.with_control_points(points),
                )
                calls = []
                evaluate = mapper._evaluate

                def counted_evaluate(x, y):
                    calls.append(True)
                    return evaluate(x, y)

                mapper._evaluate = counted_evaluate
                source_x, source_y, valid = mapper.inverse_arrays(
                    *coordinates, return_valid=True
                )
                self.assertEqual(source_x.dtype, np.dtype(np.float32))
                self.assertEqual(source_y.dtype, np.dtype(np.float32))
                self.assertTrue(valid.all())
                self.assertLessEqual(len(calls), mapper.INVERSE_ITERATIONS)
                remapped, _dx, _dy = evaluate(
                    source_x / 400.0, source_y / 200.0
                )
                self.assertTrue(np.allclose(
                    remapped[..., 0] * 400.0,
                    coordinates[0],
                    atol=0.005,
                ))
                self.assertTrue(np.allclose(
                    remapped[..., 1] * 200.0,
                    coordinates[1],
                    atol=0.005,
                ))

    def test_numpy_bilinear_inverse_retries_across_cell_boundaries(self):
        points = (
            (0.1122, 0.2059), (0.7473, 0.0404), (1.0360, 0.2975),
            (-0.2799, 0.4329), (0.3352, 0.2276), (0.7989, 0.5373),
            (-0.2889, 1.2479), (0.3509, 1.2040), (0.6679, 0.9692),
        )
        mapper = GridMapper(
            QRectF(0, 0, 1000, 500),
            QRectF(0, 0, 1000, 500),
            GridTextTransform(2, 2, 'bilinear', points),
        )
        axis = np.linspace(0.25, 0.75, 41, dtype=np.float32)
        source_x, source_y = np.meshgrid(axis * 1000, axis * 500)
        visual_x, visual_y = mapper.forward_arrays(source_x, source_y)
        with patch(
            'ballontranslator.ui.text_engine.transforms.grid.'
            '_compiled_inverse_grid_arrays',
            return_value=None,
        ):
            restored_x, restored_y, valid = mapper.inverse_arrays(
                visual_x, visual_y, return_valid=True
            )

        self.assertTrue(valid.all())
        self.assertLess(
            float(np.max(np.hypot(
                restored_x - source_x,
                restored_y - source_y,
            ))),
            0.02,
        )

    def test_bilinear_grid_outline_keeps_padded_cell_boundary_kinks(self):
        logical = QRectF(0, 0, 100, 100)
        source = logical.adjusted(-10, -20, 30, 20)
        grid = GridTextTransform(2, 1)
        points = list(grid.control_points)
        points[1] = (0.5, -1.0)
        points[4] = (0.5, 1.0)
        mapper = GridMapper(
            logical,
            source,
            grid.with_control_points(points),
        )
        kink = mapper.forward_point(QPointF(50, source.top()))
        self.assertLess(kink.y(), -100.0)
        self.assertTrue(mapper.visual_bounds().contains(kink))

    def test_catmull_rom_bounds_scale_with_deformation_not_box_size(self):
        logical = QRectF(0, 0, 400, 200)
        grid = GridTextTransform(2, 2, 'catmull_rom')
        points = list(grid.control_points)
        points[4] = (0.55, 0.45)
        bounds = GridMapper(
            logical,
            logical,
            grid.with_control_points(points),
        ).visual_bounds()
        self.assertLess(bounds.width(), logical.width() * 1.2)
        self.assertLess(bounds.height(), logical.height() * 1.2)

    def test_grid_compiles_as_one_ordered_composable_surface_mapper(self):
        logical = QRectF(10, 20, 420, 160)
        source = logical.adjusted(-12, -12, 12, 12)
        grid = GridTextTransform(2, 2, 'catmull_rom')
        points = list(grid.control_points)
        points[4] = (0.62, 0.38)
        stack = TextTransformStack((
            ProjectiveTextTransform(1.1, 0.9, 4.0),
            grid.with_control_points(points),
            ProjectiveTextTransform(rotation_y=30.0, perspective=0.25),
        ))
        for vertical in (False, True):
            compiled = compile_text_transform_stack(
                stack, logical, source, vertical
            )
            self.assertTrue(compiled.native_matrix.isIdentity())
            self.assertIsInstance(
                compiled.surface_mapper, CompositeTextTransformMapper
            )
            self.assertEqual(
                tuple(stage.transform.transform_type for stage in compiled.stages),
                ('projective', 'grid', 'projective'),
            )
            point = QPointF(180, 90)
            restored = compiled.surface_mapper.inverse_point(
                compiled.surface_mapper.forward_point(point)
            )
            self.assertAlmostEqual(restored.x(), point.x(), places=5)
            self.assertAlmostEqual(restored.y(), point.y(), places=5)
            source_x = np.asarray([40.0, 180.0, 350.0])
            source_y = np.asarray([30.0, 80.0, 170.0])
            visual_x, visual_y = compiled.surface_mapper.forward_arrays(
                source_x, source_y
            )
            for index in range(len(source_x)):
                expected = compiled.surface_mapper.forward_point(QPointF(
                    source_x[index], source_y[index]
                ))
                self.assertAlmostEqual(
                    visual_x[index], expected.x(), places=6
                )
                self.assertAlmostEqual(
                    visual_y[index], expected.y(), places=6
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
        panel = TextTransformPanel(
            'Text Transform', 'test_transform', 'test_transform_expand',
        )
        self.addCleanup(panel.deleteLater)
        return panel

    @staticmethod
    def _set_stack(panel, state):
        panel._set_transform_states([state])

    def test_panel_normalizes_font_formats_only_at_its_public_boundary(self):
        panel = self._make_panel()
        state = transform_state(GridTextTransform())
        font_format = FontFormat()
        font_format.text_transform = state

        panel.set_active_format(font_format)
        self.assertEqual(panel._transform_panel_types, ('grid',))
        with self.assertRaisesRegex(TypeError, 'requires TextTransformStack'):
            panel._set_transform_states([font_format])

    def test_stack_shape_update_syncs_content_height_once(self):
        panel = self._make_panel()
        with patch.object(
            panel,
            '_sync_content_height',
            wraps=panel._sync_content_height,
        ) as sync_height:
            self._set_stack(panel, transform_state(GridTextTransform()))

        sync_height.assert_called_once_with()

    def test_selected_item_updates_transform_panel_once(self):
        previous_canvas = getattr(SW, 'canvas', None)
        previous_active_format = C.active_format
        canvas = Canvas()
        SW.canvas = canvas
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        self.addCleanup(setattr, C, 'active_format', previous_active_format)
        self.addCleanup(canvas.gv.deleteLater)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        item.set_text_transform(transform_state(GridTextTransform()))

        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            format_panel = FontFormatPanel(self.app)
        format_panel.global_format = FontFormat()
        self.addCleanup(format_panel.deleteLater)

        with patch.object(
            format_panel.texttransform_panel,
            '_set_transform_states',
            wraps=format_panel.texttransform_panel._set_transform_states,
        ) as set_transform_states:
            format_panel.set_textblk_item(item)

        set_transform_states.assert_called_once()

    def test_cursor_letter_spacing_does_not_replace_item_default(self):
        previous_canvas = getattr(SW, 'canvas', None)
        previous_active_format = C.active_format
        canvas = Canvas()
        SW.canvas = canvas
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        self.addCleanup(setattr, C, 'active_format', previous_active_format)
        self.addCleanup(canvas.gv.deleteLater)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLetterSpacing(1.8)

        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            format_panel = FontFormatPanel(self.app)
        format_panel.global_format = FontFormat()
        self.addCleanup(format_panel.deleteLater)

        format_panel.set_textblk_item(item)
        self.assertEqual(format_panel.letterSpacingBox.value(), 1.8)
        format_panel.set_textblk_item()

        self.assertEqual(item.fontformat.letter_spacing, 1.15)

    def test_add_menu_and_hover_actions_are_generated_from_registry(self):
        panel = self._make_panel()
        self.assertEqual(panel.add_transform_button.text(), 'Add')
        self.assertEqual(
            [action.text() for action in panel.add_transform_button.menu().actions()],
            ['Scale / Slant / 3D', 'Bend', 'Sine Wave', 'Grid'],
        )
        self.assertTrue(all(
            not action.icon().isNull()
            for action in panel.add_transform_button.menu().actions()
        ))
        added = []
        panel.transform_add_requested.connect(added.append)
        panel.add_transform_button.menu().actions()[0].trigger()
        self.assertEqual(added, ['projective'])

        self._set_stack(
            panel,
            transform_state(ProjectiveTextTransform(), BendTextTransform())
        )
        operation_panel = panel.transform_panels[0]
        self.assertIsNotNone(operation_panel.title_icon_label.pixmap())
        self.assertFalse(operation_panel.title_icon_label.pixmap().isNull())
        self.assertTrue(operation_panel.close_button.isHidden())
        QApplication.sendEvent(operation_panel, QEvent(QEvent.Type.Enter))
        self.assertFalse(operation_panel.close_button.isHidden())

        removed = []
        panel.transform_remove_requested.connect(removed.append)
        operation_panel.close_button.click()
        self.assertEqual(removed, [0])

    def test_panel_grows_until_its_scrollable_maximum(self):
        panel = self._make_panel()
        initial_height = panel.sizeHint().height()
        self._set_stack(
            panel,
            transform_state(*(
                ProjectiveTextTransform() for _index in range(10)
            )),
        )
        self.assertGreater(panel.sizeHint().height(), initial_height)
        self.assertEqual(panel.sizeHint().height(), panel.MAX_CONTENT_HEIGHT)
        self.assertEqual(panel.maximumHeight(), panel.MAX_CONTENT_HEIGHT)
        panel.setMaximumWidth(300)
        panel.resize(300, panel.MAX_CONTENT_HEIGHT)
        panel.show()
        self.app.processEvents()
        self.assertGreater(panel.verticalScrollBar().maximum(), 0)
        self.app.processEvents()
        self.assertEqual(
            panel.scrollContent.width(), panel.viewport().width()
        )
        self.assertEqual(
            len({operation.width() for operation in panel.transform_panels}),
            1,
        )
        self.assertEqual(
            [operation.y() for operation in panel.transform_panels],
            sorted({operation.y() for operation in panel.transform_panels}),
        )

    def test_transform_cards_select_on_card_and_parameter_interaction(self):
        panel = self._make_panel()
        self._set_stack(
            panel,
            transform_state(
                GridTextTransform(),
                ProjectiveTextTransform(),
            ),
        )
        selected = []
        panel.transform_selected.connect(selected.append)

        QTest.mouseClick(
            panel.transform_panels[1], Qt.MouseButton.LeftButton
        )
        self.assertTrue(panel.transform_panels[1].property('selected'))
        self.assertFalse(panel.transform_panels[0].property('selected'))

        QTest.mouseClick(
            panel.transform_panels[1], Qt.MouseButton.LeftButton
        )
        self.assertFalse(panel.transform_panels[1].property('selected'))

        control = panel.transform_panels[0].controls[
            'horizontal_divisions'
        ]
        control.editor.setText('2')
        control.editor.textEdited.emit('2')
        self.assertTrue(panel.transform_panels[0].property('selected'))
        self.assertEqual(selected, [1, -1, 0])

    def test_parameter_label_drag_keeps_every_transform_selected(self):
        panel = self._make_panel()
        cases = (
            (ProjectiveTextTransform(), 'rotation_x'),
            (BendTextTransform(), 'bend'),
            (SineTextTransform(), 'frequency_x'),
            (GridTextTransform(), 'horizontal_divisions'),
        )
        panel.show()
        self.app.processEvents()
        for transform, parameter in cases:
            with self.subTest(transform=transform.transform_type):
                self._set_stack(panel, transform_state(transform))
                panel.clear_transform_selection(emit=False)
                label = panel.transform_panels[0].controls[parameter].label
                QTest.mousePress(
                    label,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                    label.rect().center(),
                )
                label.size_ctrl_changed.emit(5)
                QTest.mouseRelease(
                    label,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                    label.rect().center(),
                )
                self.assertTrue(
                    panel.transform_panels[0].property('selected')
                )

    def test_sine_controls_use_wave_language_and_percentage_values(self):
        panel = self._make_panel()
        self._set_stack(panel, transform_state(SineTextTransform()))
        controls = panel.transform_panels[0].controls
        self.assertEqual(
            [control.label.text() for control in controls.values()],
            [
                'Segments',
                'Shift',
                'Height',
                'Segments',
                'Shift',
                'Width',
            ],
        )
        self.assertEqual(
            [label.text() for label in panel.transform_panels[0].section_labels],
            ['Left-to-Right Wave', 'Top-to-Bottom Wave'],
        )
        self.assertEqual(controls['frequency_x'].editor.text(), '2')
        self.assertEqual(controls['frequency_x'].drag_step, 0.125)
        self.assertEqual(controls['amplitude_x'].editor.text(), '10.0%')

    def test_controls_canonicalize_values_before_emitting(self):
        floating = CommittedTransformControl(
            'Shift', 'phase_x', 100.0, 0.0, 1.0, '%', 1.0,
        )
        integer = CommittedTransformControl(
            'Segments', 'frequency_x', 1.0, 0.0, 64.0, '', 0.125,
            decimals=0,
        )
        self.addCleanup(floating.deleteLater)
        self.addCleanup(integer.deleteLater)

        self.assertEqual(floating._parse('12.3456789%'), 0.123457)
        self.assertEqual(integer._display_to_canonical(2.5), 2)
        self.assertEqual(integer._parse('2'), 2)
        self.assertIsInstance(integer._parse('0'), int)
        with self.assertRaises(ValueError):
            integer._parse('2.5')
        integer.set_model_value(2)
        integer.editor.setText('2.5')
        integer._on_text_edited()
        self.assertFalse(integer.commit_pending())
        self.assertEqual(integer.editor.text(), '2')

    def test_integer_drag_uses_eight_pixels_per_step_and_stops_at_range_end(self):
        integer = CommittedTransformControl(
            'Segments', 'frequency_x', 1.0, 0.0, 64.0, '', 0.125,
            decimals=0,
        )
        self.addCleanup(integer.deleteLater)
        integer.set_model_value(2)
        integer_previews = []
        integer.preview_requested.connect(
            lambda _name, delta: integer_previews.append(delta)
        )
        integer._start_drag()
        integer._move_drag(7)
        integer._move_drag(1)
        self.assertEqual(integer_previews, [0, 1])
        self.assertTrue(all(
            isinstance(value, int) for value in integer_previews
        ))
        self.assertEqual(integer.editor.text(), '3')
        integer_steps = []
        integer.drag_commit_requested.connect(
            lambda _name, delta: integer_steps.append(delta)
        )
        up_rect, _down_rect = integer.editor._button_rects()
        QTest.mouseClick(
            integer.editor,
            Qt.MouseButton.LeftButton,
            pos=up_rect.center(),
        )
        self.assertEqual(integer_steps, [1])
        integer.set_model_value(64)
        QTest.mouseClick(
            integer.editor,
            Qt.MouseButton.LeftButton,
            pos=up_rect.center(),
        )
        self.assertEqual(integer_steps, [1])
        self.assertIsInstance(integer_steps[0], int)

        bounded = CommittedTransformControl(
            'Shift', 'phase_x', 100.0, 0.0, 1.0, '%', 1.0,
        )
        self.addCleanup(bounded.deleteLater)
        bounded.set_model_value(0.9)
        bounded_previews = []
        bounded_commits = []
        bounded.preview_requested.connect(
            lambda _name, delta: bounded_previews.append(delta)
        )
        bounded.drag_commit_requested.connect(
            lambda _name, delta: bounded_commits.append(delta)
        )
        bounded._start_drag()
        bounded._move_drag(20)
        bounded._move_drag(100)
        self.assertEqual(bounded.editor.text(), '100.0%')
        self.assertAlmostEqual(bounded_previews[-2], 0.1)
        self.assertAlmostEqual(bounded_previews[-1], 0.1)
        bounded._finish_drag()
        self.assertEqual(len(bounded_commits), 1)
        self.assertAlmostEqual(bounded_commits[0], 0.1)

        bounded.set_model_value(None, [0.2, 0.9])
        bounded_previews.clear()
        bounded._start_drag()
        bounded._move_drag(100)
        bounded._move_drag(100)
        self.assertEqual(bounded.editor.text(), 'Δ +10.0%')
        self.assertAlmostEqual(bounded_previews[-1], 0.1)

    def test_grid_division_drag_preserves_integer_through_panel_signal(self):
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        stack = QUndoStack()
        SW.canvas = SimpleNamespace(push_undo_command=stack.push)
        panel = self._make_panel()
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.set_text_transform(transform_state(GridTextTransform()))
        session = TextTransformEditSession(SimpleNamespace(), panel)
        session.replace_targets([item])
        panel.set_transform_items([item])
        previews = []
        panel.transform_preview_requested.connect(
            lambda index, name, value: previews.append((index, name, value))
        )
        control = panel.transform_panels[0].controls[
            'horizontal_divisions'
        ]

        control._start_drag()
        control._move_drag(8)

        self.assertEqual(previews, [(0, 'horizontal_divisions', 1)])
        self.assertIsInstance(previews[0][2], int)
        updated = item._effective_text_transform()[0]
        self.assertEqual(updated.horizontal_divisions, 2)
        self.assertIsInstance(updated.horizontal_divisions, int)
        control._finish_drag()
        self.assertEqual(
            item.blk.fontformat.text_transform[0].horizontal_divisions,
            2,
        )
        self.assertIsInstance(
            item.blk.fontformat.text_transform[0].horizontal_divisions,
            int,
        )
        self.assertEqual(stack.count(), 1)
        stack.undo()
        self.assertEqual(
            item.blk.fontformat.text_transform[0].horizontal_divisions,
            1,
        )

    def test_grid_parameter_selection_binds_controller_and_delete_clears_it(self):
        panel = self._make_panel()
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.set_text_transform(transform_state(
            GridTextTransform()
        ))
        stack = QUndoStack()
        bindings = []
        clears = []
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        SW.canvas = SimpleNamespace(
            push_undo_command=stack.push,
            bind_text_grid_control=lambda item, index, **callbacks:
            bindings.append((item, index, callbacks)),
            clear_text_transform_controls=lambda: clears.append(True),
        )
        session = TextTransformEditSession(SimpleNamespace(), panel)
        session.replace_targets([item])
        panel.set_transform_items([item])

        control = panel.transform_panels[0].controls[
            'horizontal_divisions'
        ]
        control.editor.setText('2')
        control.editor.textEdited.emit('2')
        self.assertEqual(session.selected_index, 0)
        self.assertEqual(bindings[-1][:2], (item, 0))

        panel.transform_panels[0].card_clicked.emit(0)
        self.assertIsNone(session.selected_index)
        self.assertFalse(panel.transform_panels[0].property('selected'))
        self.assertTrue(clears)

    def test_projective_parameter_selection_binds_controller_and_toggle_clears_it(self):
        panel = self._make_panel()
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.set_text_transform(transform_state(ProjectiveTextTransform()))
        stack = QUndoStack()
        bindings = []
        clears = []
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        SW.canvas = SimpleNamespace(
            push_undo_command=stack.push,
            bind_text_projective_control=lambda item, index, **callbacks:
            bindings.append((item, index, callbacks)),
            clear_text_transform_controls=lambda: clears.append(True),
        )
        session = TextTransformEditSession(SimpleNamespace(), panel)
        session.replace_targets([item])
        panel.set_transform_items([item])

        control = panel.transform_panels[0].controls['rotation_x']
        control.editor.setText('12')
        control.editor.textEdited.emit('12')

        self.assertEqual(session.selected_index, 0)
        self.assertEqual(bindings[-1][:2], (item, 0))
        panel.transform_panels[0].card_clicked.emit(0)
        self.assertIsNone(session.selected_index)
        self.assertFalse(panel.transform_panels[0].property('selected'))
        self.assertTrue(clears)

        panel.transform_panels[0].card_clicked.emit(0)
        self.assertEqual(session.selected_index, 0)
        self.assertEqual(bindings[-1][:2], (item, 0))

        session.remove_transform(0)
        self.assertIsNone(session.selected_index)
        self.assertEqual(len(item.blk.fontformat.text_transform), 0)
        self.assertTrue(clears)

    def test_pair_list_selection_binds_projective_and_grid_controllers(self):
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        for transform, control_name in (
            (ProjectiveTextTransform(), 'txtblkProjectiveControl'),
            (GridTextTransform(), 'txtblkGridControl'),
        ):
            with self.subTest(transform=transform.transform_type):
                canvas = Canvas()
                item, pair = self._make_pair(0, TEST_LINES[0], False)
                item.setParentItem(canvas.textLayer)
                state = transform_state(transform)
                item.set_text_transform(state)
                panel = self._make_panel()
                self._set_stack(panel, state)
                SW.canvas = canvas
                session = TextTransformEditSession(SimpleNamespace(), panel)
                panel.transform_panels[0].card_clicked.emit(0)

                def sync_selection():
                    items = canvas.selected_text_items()
                    session.replace_targets(items)
                    panel.set_transform_items(items)

                manager = SimpleNamespace(
                    canvas=canvas,
                    textEditList=SimpleNamespace(
                        checked_list=[SimpleNamespace(idx=0)]
                    ),
                    textblk_item_list=[item],
                    formatpanel=SimpleNamespace(
                        set_textblk_item=lambda *_args, **_kwargs:
                        sync_selection()
                    ),
                )
                manager._update_selection_panels = MethodType(
                    SceneTextManager._update_selection_panels,
                    manager,
                )
                SceneTextManager.on_transwidget_selection_changed(manager)

                control = getattr(canvas, control_name)
                self.assertTrue(item.isSelected())
                self.assertEqual(session.items, [item])
                self.assertIs(control.item, item)
                self.assertTrue(control.isVisible())

                control.clear()
                item.geometry_controller.release_render_resources()
                canvas.removeItem(item)
                pair.deleteLater()
                canvas.gv.close()

    def test_added_transform_is_selected_and_new_grid_binds_controller(self):
        panel = self._make_panel()
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.set_text_transform(transform_state(ProjectiveTextTransform()))
        stack = QUndoStack()
        bindings = []
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        SW.canvas = SimpleNamespace(
            push_undo_command=stack.push,
            bind_text_grid_control=lambda item, index, **_callbacks:
            bindings.append((item, index)),
            clear_text_transform_controls=lambda: None,
        )
        session = TextTransformEditSession(SimpleNamespace(), panel)
        session.replace_targets([item])
        panel.set_transform_items([item])

        session.add_transform('grid')

        self.assertEqual(session.selected_index, 1)
        self.assertTrue(panel.transform_panels[1].property('selected'))
        self.assertFalse(panel.transform_panels[0].property('selected'))
        self.assertEqual(bindings[-1], (item, 1))
        self.assertEqual(stack.count(), 1)

    def test_canvas_scale_shortcut_adds_or_selects_last_projective(self):
        previous_canvas = getattr(SW, 'canvas', None)
        previous_active_format = C.active_format
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        self.addCleanup(setattr, C, 'active_format', previous_active_format)
        C.active_format = None
        cases = (
            (
                transform_state(BendTextTransform()),
                ('bend', 'projective'),
                1,
                1,
            ),
            (
                transform_state(
                    ProjectiveTextTransform(1.1, 1.0),
                    BendTextTransform(),
                    ProjectiveTextTransform(0.9, 1.0),
                ),
                ('projective', 'bend', 'projective'),
                2,
                0,
            ),
        )
        for initial, expected_types, expected_index, undo_count in cases:
            with self.subTest(initial=initial):
                canvas = Canvas()
                canvas.editor_index = 1
                SW.canvas = canvas
                item, pair = self._make_pair(0, TEST_LINES[0], False)
                item.setParentItem(canvas.textLayer)
                item.set_text_transform(initial)
                item.setSelected(True)
                panel = self._make_panel()
                session = TextTransformEditSession(SimpleNamespace(), panel)
                panel.set_transform_items([item])
                manager = type('ManagerStub', (), {})()
                manager.formatpanel = SimpleNamespace(
                    text_transform_session=session
                )

                def update_selection(items):
                    session.replace_targets(items)
                    panel.set_transform_items(items)

                manager._update_selection_panels = update_selection
                manager.on_projective_scale_requested = MethodType(
                    SceneTextManager.on_projective_scale_requested,
                    manager,
                )
                canvas.projective_scale_requested.connect(
                    manager.on_projective_scale_requested
                )

                canvas.gv.resize(800, 500)
                canvas.gv.show()
                canvas.gv.setFocus()
                self.app.processEvents()
                controller = canvas.txtblkProjectiveControl
                controller._cursor_scene_position = lambda: (
                    canvas.gv,
                    controller.scenePos()
                    + QPointF(PROJECTIVE_CONTROL_RADIUS, 0.0),
                )

                QTest.keyClick(canvas.gv.viewport(), Qt.Key.Key_S)
                self.app.processEvents()

                self.assertEqual(
                    tuple(
                        transform.transform_type
                        for transform in item.blk.fontformat.text_transform
                    ),
                    expected_types,
                )
                self.assertEqual(session.selected_index, expected_index)
                self.assertEqual(controller.stack_index, expected_index)
                self.assertEqual(controller._modal_transform.mode, 'scale')
                self.assertEqual(canvas.text_undo_stack.count(), undo_count)

                controller._finish_modal(False)
                controller.clear()
                item.geometry_controller.release_render_resources()
                canvas.removeItem(item)
                pair.deleteLater()
                canvas.gv.close()

    def test_canvas_scale_shortcut_requires_one_nonediting_item(self):
        canvas = Canvas()
        requests = []
        canvas.projective_scale_requested.connect(requests.append)
        first, first_pair = self._make_pair(0, TEST_LINES[0], False)
        second, second_pair = self._make_pair(1, TEST_LINES[1], False)
        first.setParentItem(canvas.textLayer)
        second.setParentItem(canvas.textLayer)

        self.assertFalse(canvas.start_projective_scale())
        canvas.editor_index = 1
        first.setSelected(True)
        first.set_text_transform(transform_state(ProjectiveTextTransform()))
        canvas.bind_text_projective_control(
            first,
            0,
            begin_edit=lambda _index: None,
            preview_transform=lambda _index, _transform: None,
            commit_transform=lambda _index, _transform: None,
            cancel_edit=lambda _index: None,
        )
        first.startEdit()
        self.assertFalse(canvas.handle_transform_modal_shortcut(Qt.Key.Key_S))
        self.assertFalse(canvas.start_projective_scale())
        first.endEdit()
        canvas.clear_text_transform_controls()
        second.setSelected(True)
        self.assertFalse(canvas.start_projective_scale())
        self.assertEqual(requests, [])

        first.geometry_controller.release_render_resources()
        second.geometry_controller.release_render_resources()
        canvas.removeItem(first)
        canvas.removeItem(second)
        first_pair.deleteLater()
        second_pair.deleteLater()
        canvas.gv.close()

    def test_multi_selection_only_exposes_matching_stack_indices(self):
        panel = self._make_panel()
        matching = [
            SimpleNamespace(
                blk=SimpleNamespace(
                    fontformat=FontFormat(
                        text_transform=TextTransformStack((
                            ProjectiveTextTransform(1.1, 1.0, 4.0),
                            BendTextTransform(0.4),
                        )),
                    )
                )
            ),
            SimpleNamespace(
                blk=SimpleNamespace(
                    fontformat=FontFormat(
                        text_transform=TextTransformStack((
                            ProjectiveTextTransform(0.9, 1.0, -4.0),
                            BendTextTransform(-0.4),
                        )),
                    )
                )
            ),
        ]
        panel.set_transform_items(matching)
        self.assertEqual(len(panel.transform_panels), 2)
        self.assertTrue(panel.transform_mixed_label.isHidden())
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
                            BendTextTransform(0.4),
                            ProjectiveTextTransform(),
                        )),
                    )
                )
            ),
        ]
        panel.set_transform_items(mixed)
        self.assertEqual(panel.transform_panels, [])
        self.assertFalse(panel.transform_mixed_label.isHidden())


class TextTransformUndoTest(TextTransformTestBase):
    def test_pair_editor_emits_raw_utf16_replacement_range(self):
        _item, pair = self._make_pair(0, 'aX', False)
        edit = pair.e_trans
        pair.show()
        edit.setFocus()
        self.app.processEvents()
        cursor = edit.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        edit.setTextCursor(cursor)
        propagated = []
        edit.propagate_user_edited.connect(
            lambda *args: propagated.append(args)
        )

        cursor.insertText('\U0001f600')
        self.app.processEvents()
        pair.hide()

        self.assertEqual(edit.toPlainText(), 'a\U0001f600')
        self.assertEqual(propagated, [(1, 1, '\U0001f600', False)])

    def test_pair_editor_emits_empty_ime_replacement_range(self):
        _item, pair = self._make_pair(0, 'aX', False)
        edit = pair.e_trans
        pair.show()
        edit.setFocus()
        self.app.processEvents()
        cursor = edit.textCursor()
        cursor.setPosition(1)
        edit.setTextCursor(cursor)
        propagated = []
        edit.propagate_user_edited.connect(
            lambda *args: propagated.append(args)
        )

        event = QInputMethodEvent('', [])
        event.setCommitString('', 0, 1)
        QApplication.sendEvent(edit, event)
        self.app.processEvents()
        pair.hide()

        self.assertEqual(edit.toPlainText(), 'a')
        self.assertEqual(propagated, [(1, 1, '', False)])

    def test_cancelled_preedit_does_not_capture_next_key_edit(self):
        _item, pair = self._make_pair(0, 'aX', False)
        edit = pair.e_trans
        pair.show()
        edit.setFocus()
        self.app.processEvents()
        cursor = edit.textCursor()
        cursor.setPosition(1)
        edit.setTextCursor(cursor)
        propagated = []
        edit.propagate_user_edited.connect(
            lambda *args: propagated.append(args)
        )

        QApplication.sendEvent(edit, QInputMethodEvent('z', []))
        QApplication.sendEvent(edit, QInputMethodEvent('', []))
        cursor = edit.textCursor()
        cursor.insertText('Z')
        self.app.processEvents()
        pair.hide()

        self.assertEqual(edit.toPlainText(), 'aZX')
        self.assertEqual(propagated, [(1, 0, 'Z', False)])

    def test_capitalize_selected_items_is_one_synced_undo_command(self):
        original = 'hELLO WORLD. next ONE!'
        capitalized = 'Hello world. Next one!'
        item, pair = self._make_pair(0, original, False)
        second_original = '😀 aNOTHER item. sECOND sentence!'
        second_capitalized = '😀 Another item. Second sentence!'
        second_item, second_pair = self._make_pair(
            1,
            second_original,
            True,
        )

        colors = (QColor(210, 20, 30), QColor(20, 80, 210))
        for position, color in zip((1, 2), colors):
            cursor = QTextCursor(item.document())
            cursor.setPosition(position)
            cursor.setPosition(
                position + 1,
                QTextCursor.MoveMode.KeepAnchor,
            )
            char_format = QTextCharFormat()
            char_format.setForeground(color)
            char_format.setFontItalic(True)
            cursor.mergeCharFormat(char_format)

        stack = QUndoStack()
        canvas = SimpleNamespace(
            textEditMode=lambda: True,
            selected_text_items=lambda: [item, second_item],
            push_undo_command=stack.push,
        )
        manager = SimpleNamespace(
            canvas=canvas,
            pairwidget_list=[pair, second_pair],
        )
        unexpected_history = []
        item.push_undo_stack.connect(
            lambda *_args: unexpected_history.append('item')
        )
        pair.e_trans.push_undo_stack.connect(
            lambda *_args: unexpected_history.append('pair')
        )
        pair.show()
        pair.e_trans.setFocus()
        self.app.processEvents()

        SceneTextManager.capitalize_selected_textitems(manager)

        self.assertEqual(stack.count(), 1)
        self.assertEqual(stack.index(), 1)
        self.assertEqual(item.toPlainText(), capitalized)
        self.assertEqual(pair.e_trans.toPlainText(), capitalized)
        self.assertEqual(second_item.toPlainText(), second_capitalized)
        self.assertEqual(second_pair.e_trans.toPlainText(), second_capitalized)
        self.assertEqual(unexpected_history, [])
        for position, color in zip((1, 2), colors):
            cursor = QTextCursor(item.document())
            cursor.setPosition(position)
            cursor.setPosition(
                position + 1,
                QTextCursor.MoveMode.KeepAnchor,
            )
            self.assertEqual(cursor.charFormat().foreground().color(), color)
            self.assertTrue(cursor.charFormat().fontItalic())

        SceneTextManager.capitalize_selected_textitems(manager)
        self.assertEqual(stack.count(), 1)

        stack.undo()
        self.assertEqual(item.toPlainText(), original)
        self.assertEqual(pair.e_trans.toPlainText(), original)
        self.assertEqual(second_item.toPlainText(), second_original)
        self.assertEqual(second_pair.e_trans.toPlainText(), second_original)
        stack.redo()
        self.assertEqual(item.toPlainText(), capitalized)
        self.assertEqual(pair.e_trans.toPlainText(), capitalized)
        self.assertEqual(second_item.toPlainText(), second_capitalized)
        self.assertEqual(second_pair.e_trans.toPlainText(), second_capitalized)
        pair.hide()

    def test_capitalize_command_rejects_unsynchronized_pair(self):
        item, pair = self._make_pair(0, 'hELLO', False)
        pair.e_trans.setPlainText('different')

        self.assertIsNone(CapitalizeTextItemsCommand.create(
            [item],
            [pair.e_trans],
        ))

    def test_parameter_preview_repaints_only_changed_items(self):
        states = (
            transform_state(ProjectiveTextTransform(horizontal_scale=1.0)),
            transform_state(ProjectiveTextTransform(horizontal_scale=1.2)),
        )
        preview_changes = (True, False)
        cancel_changes = (False, True)
        updates = []

        def make_item(index, state):
            return SimpleNamespace(
                blk=SimpleNamespace(
                    fontformat=FontFormat(text_transform=state)
                ),
                _effective_text_transform=lambda: state,
                set_text_transform=(
                    lambda _state, *, preview=False:
                    preview_changes[index]
                ),
                clear_text_transform_preview=(
                    lambda: cancel_changes[index]
                ),
                update=lambda: updates.append(index),
            )

        session = object.__new__(TextTransformEditSession)
        session.items = [
            make_item(index, state)
            for index, state in enumerate(states)
        ]
        session.drag_before = None
        session.drag_key = None

        session.preview_parameter_delta(0, 'horizontal_scale', 0.1)
        self.assertEqual(updates, [0])

        updates.clear()
        session.preview_parameter_delta(0, 'vertical_scale', 0.1)
        self.assertCountEqual(updates, [0, 1])
        self.assertEqual(len(updates), 2)

        updates.clear()
        session.cancel_preview()
        self.assertEqual(updates, [1])

    def test_command_transitions_schedule_repaint_without_session_repaint(self):
        previous_canvas = getattr(SW, 'canvas', None)
        previous_active_format = C.active_format
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        self.addCleanup(setattr, C, 'active_format', previous_active_format)
        C.active_format = None

        transitions = (
            (
                'matrix',
                transform_state(ProjectiveTextTransform(horizontal_scale=1.2)),
            ),
            ('surface', transform_state(BendTextTransform(0.4))),
            ('glyph-layout', transform_state(glyph_slant_angle=10.0)),
        )
        for render_path, after in transitions:
            with self.subTest(render_path=render_path):
                scene = QGraphicsScene()
                item, _ = self._make_pair(0, TEST_LINES[0], False)
                scene.addItem(item)
                repaints = []
                scene.changed.connect(repaints.append)
                self.app.processEvents()
                repaints.clear()

                refreshed_states = []
                session = object.__new__(TextTransformEditSession)
                session.items = [item]
                session.controls = SimpleNamespace(
                    set_transform_items=lambda _items: refreshed_states.append(
                        item.blk.fontformat.text_transform
                    )
                )
                session.selected_index = None
                SW.canvas = SimpleNamespace(
                    clear_text_transform_controls=lambda: None,
                )
                stack = QUndoStack()
                command = SetTextTransformCommand.create(
                    [item], [NEUTRAL], [after], session._sync_transform_ui
                )

                stack.push(command)
                self.app.processEvents()
                self.assertTrue(repaints)
                self.assertEqual(refreshed_states[-1], after)

                repaints.clear()
                stack.undo()
                self.app.processEvents()
                self.assertTrue(repaints)
                self.assertEqual(refreshed_states[-1], NEUTRAL)

                repaints.clear()
                stack.redo()
                self.app.processEvents()
                self.assertTrue(repaints)
                self.assertEqual(refreshed_states[-1], after)

                stack.undo()
                item.set_text_transform(after, preview=True)
                self.app.processEvents()
                repaints.clear()
                stack.push(SetTextTransformCommand(
                    [item], [NEUTRAL], [after], session._sync_transform_ui
                ))
                self.app.processEvents()
                self.assertTrue(
                    repaints,
                    f'{render_path} preview commit did not schedule repaint',
                )
                self.assertIsNone(item.geometry_controller.preview)
                self.assertEqual(refreshed_states[-1], after)

                item.geometry_controller.release_render_resources()
                scene.removeItem(item)

    def test_grid_modal_preview_switch_and_cancel_do_not_create_undo(self):
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        grid = GridTextTransform(2, 2, 'catmull_rom')
        item.set_text_transform(transform_state(grid))
        stack = QUndoStack()
        SW.canvas = SimpleNamespace(push_undo_command=stack.push)
        session = object.__new__(TextTransformEditSession)
        session.host = SimpleNamespace()
        session.items = [item]
        session.controls = SimpleNamespace(
            set_transform_items=lambda _items: None,
            finish_pending_transform_edits=lambda: None,
            cancel_transform_previews=lambda: None,
        )
        session.drag_before = None
        session.drag_key = None
        session.grid_before = None
        session.grid_index = None
        session.selected_index = 0
        controller = TextGridTransformControl()
        controller.setParentItem(base)
        controller.bind(
            item,
            0,
            begin_edit=session.begin_grid_edit,
            preview_points=session.preview_grid_points,
            commit_points=session.commit_grid_points,
            cancel_edit=session.cancel_grid_edit,
        )
        controller._set_selected_indices({0, 1})
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)
        start = controller.handles[0].scenePos()

        self.assertTrue(controller._start_modal('translate', start))
        with patch(
            'ballontranslator.ui.text_engine.geometry.'
            'compile_text_transform_stack',
            wraps=compile_text_transform_stack,
        ) as compile_mock:
            self.assertTrue(controller._update_modal(
                start + QPointF(30, 10)
            ))
            self.app.processEvents()
            self.assertLessEqual(compile_mock.call_count, 1)
        self.assertEqual(stack.count(), 0)
        self.assertTrue(controller._switch_modal(
            'scale', start + QPointF(30, 10)
        ))
        self.assertEqual(stack.count(), 0)
        self.assertEqual(
            item._effective_text_transform()[0], grid
        )
        self.assertTrue(controller._finish_modal(False))
        self.assertEqual(stack.count(), 0)
        self.assertEqual(item._effective_text_transform()[0], grid)

        self.assertTrue(controller._start_modal('translate', start))
        self.assertTrue(controller._update_modal(start + QPointF(20, 15)))
        self.assertEqual(stack.count(), 0)
        self.assertTrue(controller._finish_modal(True))
        self.assertEqual(stack.count(), 1)
        self.assertNotEqual(item.blk.fontformat.text_transform[0], grid)
        stack.undo()
        self.assertEqual(item.blk.fontformat.text_transform[0], grid)
        controller.clear()
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_grid_handle_preview_commits_one_undoable_state(self):
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        grid = GridTextTransform(2, 2, 'catmull_rom')
        points = list(grid.control_points)
        points[4] = (0.56, 0.44)
        grid = grid.with_control_points(points)
        item.set_text_transform(transform_state(grid))
        stack = QUndoStack()
        SW.canvas = SimpleNamespace(push_undo_command=stack.push)
        session = object.__new__(TextTransformEditSession)
        session.host = SimpleNamespace()
        session.items = [item]
        session.controls = SimpleNamespace(
            set_transform_items=lambda _items: None,
            finish_pending_transform_edits=lambda: None,
            cancel_transform_previews=lambda: None,
        )
        session.drag_before = None
        session.drag_key = None
        session.grid_before = None
        session.grid_index = None
        session.selected_index = 0

        points = list(grid.control_points)
        points[4] = (0.62, 0.38)
        session.begin_grid_edit(0)
        session.preview_grid_points(0, points)
        self.assertEqual(
            item._effective_text_transform()[0].control_points[4],
            (0.62, 0.38),
        )
        self.assertEqual(
            item.blk.fontformat.text_transform[0], grid
        )
        session.commit_grid_points(0, points)
        self.assertEqual(stack.count(), 1)
        self.assertEqual(
            item.blk.fontformat.text_transform[0].control_points[4],
            (0.62, 0.38),
        )
        stack.undo()
        self.assertEqual(item.blk.fontformat.text_transform[0], grid)

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

    def test_projective_bend_and_sine_mix_with_text_undo(self):
        projective = transform_state(
            ProjectiveTextTransform(rotation_y=30.0, perspective=0.6)
        )
        bend = transform_state(BendTextTransform(-0.65))
        sine = transform_state(SineTextTransform())
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item, pair = self._make_pair(0, TEST_LINES[0], vertical)
                stack = QUndoStack()
                stack.push(
                    SetTextTransformCommand.create(
                        [item], [NEUTRAL], [projective]
                    )
                )
                stack.push(
                    MultiPasteCommand(
                        TEST_LINES[1], [item], [pair.e_trans]
                    )
                )
                stack.push(
                    SetTextTransformCommand.create(
                        [item], [projective], [bend]
                    )
                )
                stack.push(
                    SetTextTransformCommand.create(
                        [item], [bend], [sine]
                    )
                )
                expected = (
                    (NEUTRAL, TEST_LINES[0]),
                    (projective, TEST_LINES[0]),
                    (projective, TEST_LINES[1]),
                    (bend, TEST_LINES[1]),
                    (sine, TEST_LINES[1]),
                )
                for _ in range(3):
                    for transform, text in reversed(expected[:-1]):
                        stack.undo()
                        self.assertEqual(
                            item.blk.fontformat.text_transform,
                            transform,
                        )
                        self.assertEqual(item.toPlainText(), text)
                        self.assertEqual(pair.e_trans.toPlainText(), text)
                    for transform, text in expected[1:]:
                        stack.redo()
                        self.assertEqual(
                            item.blk.fontformat.text_transform,
                            transform,
                        )
                        self.assertEqual(item.toPlainText(), text)
                        self.assertEqual(pair.e_trans.toPlainText(), text)

    def test_stack_structure_edits_are_undoable_for_selected_items(self):
        versions = (
            transform_state(ProjectiveTextTransform(1.15, 0.85, 11.0)),
            transform_state(ProjectiveTextTransform(0.75, 1.25, -7.0)),
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
                    select_transform=lambda _index, emit=False: None,
                    clear_transform_selection=lambda emit=False: None,
                )
                session.drag_before = None
                session.drag_key = None

                session.add_transform('bend')
                self.assertEqual(
                    [
                        tuple(item.blk.fontformat.text_transform)
                        for item in items
                    ],
                    [
                        (
                            versions[0][0],
                            BendTextTransform(),
                        ),
                        (
                            versions[1][0],
                            BendTextTransform(),
                        ),
                    ],
                )
                session.add_transform('projective')
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
                    [('projective', 'projective', 'bend')] * 2,
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
                    [('projective', 'bend', 'projective')] * 2,
                )
                stack.undo()
                stack.undo()
                self.assertEqual(
                    [item.blk.fontformat.text_transform for item in items],
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
            transform_state(ProjectiveTextTransform(1.2, 1.0, 5.0)),
            transform_state(BendTextTransform(0.4)),
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
            select_transform=lambda _index, emit=False: None,
            clear_transform_selection=lambda emit=False: None,
        )
        session.drag_before = None
        session.drag_key = None

        session.commit_value(0, 'horizontal_scale', 1.5)
        self.assertEqual(stack.count(), 0)
        self.assertEqual(
            tuple(session._state_for_item(item) for item in items),
            initial,
        )

        session.add_transform('projective')
        self.assertEqual(stack.count(), 1)
        for item in items:
            self.assertEqual(
                item.blk.fontformat.text_transform[-1],
                ProjectiveTextTransform(),
            )
        stack.undo()
        self.assertEqual(
            tuple(session._state_for_item(item) for item in items),
            initial,
        )

    def test_compiler_uses_one_mapping_boundary_for_composed_operations(self):
        logical = QRectF(10, 20, 420, 160)
        source = logical.adjusted(-12, -12, 12, 12)
        first = ProjectiveTextTransform(1.2, 0.9, 8.0)
        second = ProjectiveTextTransform(0.8, 1.1, -4.0)
        matrix_only = TextTransformStack((first, second))
        compiled = compile_text_transform_stack(
            matrix_only, logical, source, False
        )
        self.assertIsNone(compiled.surface_mapper)
        expected = (
            projective_transform_matrix(
                first, compiled.stages[0].context.source_bounds
            )
            * projective_transform_matrix(
                second, compiled.stages[1].context.source_bounds
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
                BendTextTransform(),
                first,
            )),
            logical,
            source,
            False,
        )
        self.assertIsNone(neutral_nonlinear.surface_mapper)
        self.assertFalse(
            TextTransformStack((BendTextTransform(),)).has_nonlinear
        )

        single_nonlinear = compile_text_transform_stack(
            TextTransformStack((BendTextTransform(0.6),)),
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
            ProjectiveTextTransform(1.1, 0.9, 3.0),
            ProjectiveTextTransform(rotation_y=-70.0, perspective=0.45),
            BendTextTransform(0.72),
            ProjectiveTextTransform(0.8, 1.1, -4.0),
            ProjectiveTextTransform(rotation_y=25.0, perspective=0.2),
            BendTextTransform(-0.35),
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
    def test_item_exposes_the_export_effect_error_value(self):
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        failure = RuntimeError('incomplete transformed export')

        item.effect_renderer.export_error = failure

        self.assertIs(item.export_effect_error, failure)

    def test_nonlinear_surface_sampling_tracks_render_quality(self):
        class IdentityMapper:
            geometry_key = ('identity',)

            @staticmethod
            def visual_bounds(rect):
                return QRectF(rect)

            @staticmethod
            def inverse_arrays(x, y, *, return_valid=False):
                if return_valid:
                    return x, y, np.ones_like(x, dtype=bool)
                return x, y

        renderer = NonlinearTextSurfaceRenderer()
        source_rect = QRectF(0, 0, 40, 20)
        option = QStyleOptionGraphicsItem()
        image = QImage(
            100,
            60,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(Qt.GlobalColor.transparent)

        def paint_source(painter, _option, _widget):
            painter.fillRect(source_rect, QColor(255, 255, 255))

        def sampled_warp(*, high_quality, maximum_scale=None):
            painter = QPainter(image)
            try:
                painter.scale(1.5, 1.5)
                with patch.object(
                    renderer, '_warp', wraps=renderer._warp
                ) as warp:
                    renderer.paint(
                        painter,
                        option,
                        IdentityMapper(),
                        source_rect,
                        ('content', high_quality),
                        False,
                        paint_source,
                        maximum_scale=maximum_scale,
                        high_quality=high_quality,
                    )
                return warp.call_args.args[-2:]
            finally:
                painter.end()

        render_scale, interpolation = sampled_warp(high_quality=True)
        self.assertEqual(render_scale, 2.0)
        self.assertEqual(interpolation, cv2.INTER_CUBIC)

        render_scale, interpolation = sampled_warp(
            high_quality=False, maximum_scale=0.5
        )
        self.assertEqual(render_scale, 0.5)
        self.assertEqual(interpolation, cv2.INTER_LINEAR)

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
            ProjectiveTextTransform(1.1, 0.9, 5.0),
            BendTextTransform(0.55),
            ProjectiveTextTransform(rotation_y=30.0, perspective=0.25),
            BendTextTransform(-0.2),
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
                    'ballontranslator.ui.text_engine.geometry.'
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
            ProjectiveTextTransform(rotation_y=25.0, perspective=0.3),
            BendTextTransform(0.55),
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
                pair = TransPairWidget(0, False)
                pair.e_trans.setPlainText(item.toPlainText())
                propagated = []
                def record_and_propagate(position, removed, text, joint):
                    propagated.append((position, text))
                    propagate_user_edit(
                        pair.e_trans,
                        position,
                        removed,
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

    def test_effects_and_glyph_slant_bypass_outer_device_cache(self):
        for slant in (0.0, 20.0):
            with self.subTest(slant=slant):
                block = TextBlock([0, 0, 150, 500])
                block._bounding_rect = [0, 0, 150, 500]
                block.vertical = True
                block.translation = '天是否！！！'
                block.fontformat.font_size = 52
                block.fontformat.stroke_width = 0.18
                block.fontformat.text_transform = TextTransformStack(
                    (), slant
                )
                item = TextBlkItem(block, 0)
                scene = QGraphicsScene()
                scene.addItem(item)

                self.assertEqual(
                    item.cacheMode(), QGraphicsItem.CacheMode.NoCache
                )
                source = scene.itemsBoundingRect()
                image = QImage(
                    max(1, round(source.width() * 4)),
                    max(1, round(source.height() * 4)),
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                try:
                    scene.render(painter, QRectF(image.rect()), source)
                finally:
                    painter.end()

                renderer = item.effect_renderer
                self.assertEqual(renderer.background_pixmap_scale, 4.0)
                self.assertEqual(
                    renderer.background_pixmap.devicePixelRatioF(), 4.0
                )
                scene.removeItem(item)

        item, pair = self._make_pair(0, TEST_LINES[0], False)
        self.assertEqual(
            item.cacheMode(),
            QGraphicsItem.CacheMode.DeviceCoordinateCache,
        )
        item.set_text_transform(TextTransformStack((), 20.0))
        self.assertEqual(
            item.cacheMode(), QGraphicsItem.CacheMode.NoCache
        )
        item.set_text_transform(TextTransformStack())
        self.assertEqual(
            item.cacheMode(),
            QGraphicsItem.CacheMode.DeviceCoordinateCache,
        )
        pair.deleteLater()

    def test_zero_glyph_slant_restores_effects_inside_nonlinear_stack(self):
        stack = TextTransformStack((BendTextTransform(0.55),))
        zero = TextTransformStack(stack.transforms, 0.0)
        slanted = TextTransformStack(stack.transforms, 20.0)
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
                    item.set_text_transform(zero, preview=True)
                    self._render_scene(scene)
                    self.assertIsNotNone(
                        renderer.background_pixmap
                    )

                    item.clear_text_transform_preview()
                    self._render_scene(scene)
                    item.set_text_transform(zero)
                    self._render_scene(scene)
                    self.assertIsNotNone(
                        renderer.background_pixmap
                    )
                    self.assertIsNotNone(renderer._effect_raster_state)
                    scene.removeItem(item)

    def test_surface_without_raster_effects_keeps_effect_fast_path(self):
        state = transform_state(
            BendTextTransform(0.55),
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
                self.assertIsNone(renderer._effect_raster_state)
                pixels = self._render_scene(scene)
                self.assertNotEqual(pixels, bytes(len(pixels)))
                self.assertIsNone(renderer._effect_raster_state)

                item.set_text_transform(
                    TextTransformStack(state.transforms, -20.0)
                )
                mirrored_pixels = self._render_scene(scene)
                self.assertNotEqual(mirrored_pixels, pixels)
                self.assertIsNone(renderer._effect_raster_state)
                scene.removeItem(item)

    def test_interactive_surface_uses_bounded_low_resolution_preview(self):
        class ScaleCapture:
            def __init__(self):
                self.maximum_scale = None
                self.high_quality = None

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
                high_quality=True,
            ):
                self.maximum_scale = maximum_scale
                self.high_quality = high_quality

        item, _ = self._make_pair(99, TEST_LINES[3], False)
        item.set_text_transform(
            transform_state(
                ProjectiveTextTransform(rotation_y=45.0, perspective=0.6),
                BendTextTransform(0.7),
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
        self.assertFalse(capture.high_quality)

        item.reshaping = False
        painter = QPainter(image)
        item.geometry_controller.paint_item(
            painter, option, None, lambda *_: None
        )
        painter.end()
        self.assertIsNone(capture.maximum_scale)
        self.assertTrue(capture.high_quality)

        item.set_text_transform(
            transform_state(
                ProjectiveTextTransform(rotation_y=45.0, perspective=0.6),
                BendTextTransform(0.6),
            ),
            preview=True,
        )
        painter = QPainter(image)
        item.geometry_controller.paint_item(
            painter, option, None, lambda *_: None
        )
        painter.end()
        self.assertEqual(capture.maximum_scale, 0.5)
        self.assertFalse(capture.high_quality)

    def test_surface_composition_renders_through_one_nonlinear_surface(self):
        stack = TextTransformStack((
            ProjectiveTextTransform(1.1, 0.9, 5.0),
            SineTextTransform(
                frequency_x=3,
                frequency_y=2,
                phase_x=0.25,
                phase_y=1.0,
            ),
            BendTextTransform(0.55),
            ProjectiveTextTransform(rotation_y=30.0, perspective=0.25),
            BendTextTransform(-0.2),
        ))
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item, _ = self._make_pair(99, TEST_LINES[3], vertical)
                scene = QGraphicsScene()
                scene.addItem(item)
                neutral_pixels = self._render_scene(scene)

                item.set_text_transform(stack)
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

    def test_bend_selection_requests_a_full_item_update(self):
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
            transform_state(BendTextTransform(0.8))
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

    def test_grid_editing_cache_tracks_selection_and_ime(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                width, height = (
                    (120, 240) if vertical else (240, 120)
                )
                block = TextBlock(
                    [0, 0, width, height],
                    _bounding_rect=[0, 0, width, height],
                    translation=TEST_LINES[0],
                )
                block.vertical = vertical
                item = TextBlkItem(block, 0)
                scene = QGraphicsScene()
                scene.addItem(item)
                grid = GridTextTransform(2, 2, 'catmull_rom')
                points = list(grid.control_points)
                points[4] = (0.56, 0.44)
                item.set_text_transform(transform_state(
                    grid.with_control_points(points)
                ))
                self._render_scene(scene)
                renderer = item.geometry_controller.surface_renderer
                self.assertIsNotNone(renderer.cached_pixmap)
                self.assertIsNotNone(renderer.cached_remap)
                mapper = item.geometry_controller.visual_mapper

                with patch.object(
                    renderer, '_warp', wraps=renderer._warp
                ) as warp, patch.object(
                    mapper, 'inverse_arrays', wraps=mapper.inverse_arrays
                ) as inverse:
                    item.startEdit()
                    self._render_scene(scene)
                    self.assertEqual(warp.call_count, 0)

                    cursor = item.textCursor()
                    cursor.setPosition(1)
                    item.setTextCursor(cursor)
                    self._render_scene(scene)
                    self.assertEqual(warp.call_count, 0)

                    cursor.setPosition(0)
                    cursor.setPosition(
                        3, QTextCursor.MoveMode.KeepAnchor
                    )
                    item.setTextCursor(cursor)
                    self._render_scene(scene)
                    self._render_scene(scene)
                    self.assertEqual(warp.call_count, 1)

                    cursor.setPosition(
                        4, QTextCursor.MoveMode.KeepAnchor
                    )
                    item.setTextCursor(cursor)
                    self._render_scene(scene)
                    self.assertEqual(warp.call_count, 2)

                    cursor.clearSelection()
                    item.setTextCursor(cursor)
                    self._render_scene(scene)
                    self._render_scene(scene)
                    self.assertEqual(warp.call_count, 3)

                    cursor.insertText('X')
                    item.setTextCursor(cursor)
                    self._render_scene(scene)
                    self._render_scene(scene)
                    self.assertEqual(warp.call_count, 4)

                    item.inputMethodEvent(QInputMethodEvent('かな', []))
                    self.assertTrue(item.pre_editing)
                    self.assertIsNone(renderer.cached_pixmap)
                    self.assertIsNotNone(renderer.cached_remap)
                    self._render_scene(scene)
                    self._render_scene(scene)
                    self.assertEqual(warp.call_count, 5)
                    self.assertEqual(inverse.call_count, 0)

                    item.inputMethodEvent(QInputMethodEvent('', []))
                    self.assertFalse(item.pre_editing)

                    changed = list(points)
                    changed[4] = (0.58, 0.42)
                    item.set_text_transform(transform_state(
                        grid.with_control_points(changed)
                    ))
                    self.assertIsNone(renderer.cached_remap)
                    changed_mapper = item.geometry_controller.visual_mapper
                    with patch.object(
                        changed_mapper,
                        'inverse_arrays',
                        wraps=changed_mapper.inverse_arrays,
                    ) as changed_inverse:
                        self._render_scene(scene)
                    self.assertGreater(changed_inverse.call_count, 0)
                    self.assertEqual(warp.call_count, 6)

                item.endEdit()
                item.geometry_controller.release_render_resources()
                scene.removeItem(item)

    def test_bend_defers_and_overlays_cursor_after_surface_warp(self):
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
                high_quality=True,
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
                    transform_state(BendTextTransform(0.8))
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

    def test_nonlinear_surface_uses_each_layouts_cursor_orientation(self):
        nonlinear_transforms = (
            BendTextTransform(0.7),
            SineTextTransform(),
            GridTextTransform().with_control_points((
                (0.0, 0.0),
                (1.05, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
            )),
        )
        for vertical in (False, True):
            for transform in nonlinear_transforms:
                with self.subTest(
                    vertical=vertical,
                    transform=transform.transform_type,
                ):
                    self._assert_nonlinear_cursor_orientation(
                        vertical, transform
                    )

    def _assert_nonlinear_cursor_orientation(self, vertical, transform):
        item, _ = self._make_pair(0, TEST_LINES[0], vertical)
        item.set_text_transform(transform_state(transform))
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(2)
        item.setTextCursor(cursor)
        item.layout.deferred_cursor_position = cursor.position()
        mapper = item.geometry_controller.visual_mapper
        mapped_rects = []
        original_map_rect_path = mapper.map_rect_path

        def capture_rect(rect):
            mapped_rects.append(QRectF(rect))
            return original_map_rect_path(rect)

        image = QImage(
            900, 600, QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QColor(127, 127, 127))
        painter = QPainter(image)
        with patch.object(mapper, 'map_rect_path', capture_rect):
            item.geometry_controller._paint_surface_cursor(
                painter, mapper, export_render=False
            )
        painter.end()

        self.assertEqual(len(mapped_rects), 1)
        if vertical:
            self.assertGreater(
                mapped_rects[0].width(), mapped_rects[0].height()
            )
        else:
            self.assertGreater(
                mapped_rects[0].height(), mapped_rects[0].width()
            )
        item.endEdit()
        item.geometry_controller.release_render_resources()

    def test_cached_surface_probes_native_cursor_visibility(self):
        class CacheHit:
            def release(self):
                pass

            def paint(self, *_args, **_kwargs):
                return True

        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item, _ = self._make_pair(0, TEST_LINES[0], vertical)
                item.set_text_transform(transform_state(
                    BendTextTransform(0.7)
                ))
                item.startEdit()
                cursor = item.textCursor()
                cursor.setPosition(2)
                item.setTextCursor(cursor)
                item.geometry_controller.surface_renderer = CacheHit()
                probes = []

                def base_paint(painter, _option, _widget):
                    probes.append((
                        item.layout.defer_cursor_paint,
                        painter.opacity(),
                    ))
                    item.layout.deferred_cursor_position = cursor.position()

                image = QImage(
                    900, 600, QImage.Format.Format_ARGB32_Premultiplied
                )
                image.fill(QColor(0, 0, 0, 0))
                painter = QPainter(image)
                option = QStyleOptionGraphicsItem()
                option.exposedRect = item.boundingRect()
                item.geometry_controller.paint_item(
                    painter, option, None, base_paint
                )
                painter.end()

                self.assertEqual(probes, [(True, 0.0)])
                self.assertEqual(
                    item.geometry_controller._surface_cursor_position,
                    cursor.position(),
                )
                item.endEdit()
                item.geometry_controller.release_render_resources()

    def test_warped_bend_surface_maps_layout_hit_tests(self):
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
                    transform_state(BendTextTransform(0.7))
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
                source_cursor_rect = item.layout.source_cursor_rect(
                    item.textCursor().position()
                )
                if source_cursor_rect is None:
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

    def test_fresh_items_install_projective_and_bend(self):
        transforms = (
            ProjectiveTextTransform(rotation_y=25.0, perspective=0.6),
            BendTextTransform(-0.6),
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
                    if transform.transform_type == 'projective':
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
                restored_pixels = self._render_scene(scene)
                # Qt's process-global glyph raster cache can change a few
                # antialiasing levels after the slanted render. Geometry stays
                # exact above; keep this check strict enough to catch a shifted,
                # clipped, or otherwise visibly changed neutral effect.
                delta = np.abs(
                    np.frombuffer(restored_pixels, dtype=np.uint8).astype(
                        np.int16
                    )
                    - np.frombuffer(neutral_pixels, dtype=np.uint8).astype(
                        np.int16
                    )
                )
                changed_pixels = np.count_nonzero(
                    np.any(delta.reshape(-1, 4), axis=1)
                )
                self.assertLessEqual(int(delta.max()), 24)
                self.assertLessEqual(
                    changed_pixels,
                    (900 * 600) // 50,
                )
                scene.removeItem(item)

    def test_persisted_projective_transform_is_installed_on_fresh_items(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                block = TextBlock([40, 50, 440, 250])
                block._bounding_rect = [40, 50, 400, 200]
                block.vertical = vertical
                block.angle = 17.0
                block.translation = TEST_LINES[0]
                block.fontformat.text_transform = FIRST_TRANSFORM

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
        projective_preview = transform_state(
            FIRST_TRANSFORM[0].with_value(
                'horizontal_scale', 1.4
            ),
            glyph_slant_angle=FIRST_TRANSFORM.glyph_slant_angle,
        )
        item.set_text_transform(projective_preview, preview=True)
        self.assertTrue(geometry_cache.persistent)
        item.clear_text_transform_preview()
        glyph_preview = TextTransformStack(FIRST_TRANSFORM.transforms, 9.0)
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

    def test_none_and_projective_only_paths_do_not_create_glyph_renderer(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                GLOBAL_GLYPH_GEOMETRY_CACHE.clear()
                GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE.clear()
                item, _ = self._make_pair(0, TEST_LINES[1], vertical)

                self.assertIsNone(item.geometry_controller.layout_renderer)
                self.assertIsNone(item.layout.render_delegate)
                self.assertIsNone(
                    item.effect_renderer._effect_raster_state
                )
                self.assertFalse(
                    bool(
                        item.flags()
                        & QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
                    )
                )
                item.layout.reLayout()
                self.assertEqual(len(GLOBAL_GLYPH_GEOMETRY_CACHE), 0)

                projective_only = transform_state(
                    ProjectiveTextTransform(1.2, 0.9, 8.0)
                )
                item.set_text_transform(projective_only)
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
    def test_canvas_coalesces_view_geometry_refreshes(self):
        canvas = Canvas()
        with (
            patch.object(
                canvas.txtblkShapeControl, 'requestGeometryRefresh'
            ) as shape_refresh,
            patch.object(
                canvas.txtblkGridControl, 'requestGeometryRefresh'
            ) as grid_refresh,
            patch.object(
                canvas.txtblkProjectiveControl, 'requestGeometryRefresh'
            ) as projective_refresh,
        ):
            canvas.hscroll_bar.valueChanged.emit(1)
            canvas.vscroll_bar.valueChanged.emit(1)
            canvas.gv.view_resized.emit()
            self.app.processEvents()

            shape_refresh.assert_called_once_with()
            grid_refresh.assert_called_once_with()
            projective_refresh.assert_called_once_with()

            shape_refresh.reset_mock()
            grid_refresh.reset_mock()
            projective_refresh.reset_mock()
            canvas.refresh_text_shape_control()
            canvas.gv.deleteLater()
            QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()
            shape_refresh.assert_not_called()
            grid_refresh.assert_not_called()
            projective_refresh.assert_not_called()

    def test_shape_control_refreshes_once_per_settled_geometry_change(self):
        canvas = Canvas()
        canvas.imgtrans_proj = SimpleNamespace(img_valid=True)
        canvas.baseLayer.setRect(0, 0, 1200, 800)
        canvas.setSceneRect(canvas.baseLayer.boundingRect())
        canvas.gv.resize(600, 400)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        control = canvas.txtblkShapeControl
        control.setBlkItem(item)
        canvas.gv.show()
        self.app.processEvents()
        self.addCleanup(canvas.gv.close)

        with patch.object(
            control,
            'updateBoundingRect',
            wraps=control.updateBoundingRect,
        ) as refresh_geometry:
            canvas.scaleImage(1.1)
            self.app.processEvents()
            refresh_geometry.assert_called_once_with()

            refresh_geometry.reset_mock()
            center = control.visualCenterInScene()
            control.rotateFromScene(center + QPointF(100, 100), 0.0)
            refresh_geometry.assert_called_once_with()

            refresh_geometry.reset_mock()
            handle_index = 4
            control.beginResize(handle_index)
            control.resizeFromScene(
                handle_index,
                control.handleScenePoint(handle_index) + QPointF(20, 20),
            )
            refresh_geometry.assert_called_once_with()
            control.finishResize()

    def test_transform_preview_skips_hidden_shape_and_commit_rebind(self):
        canvas = Canvas()
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        before_grid = GridTextTransform(2, 2)
        before = transform_state(before_grid)
        after = transform_state(
            before_grid.with_value('horizontal_divisions', 3)
        )
        item.set_text_transform(before)
        canvas.txtblkShapeControl.setBlkItem(item)
        callbacks = {
            'begin_edit': lambda _index: None,
            'preview_points': lambda _index, _points: None,
            'commit_points': lambda _index, _points: None,
            'cancel_edit': lambda _index: None,
        }
        canvas.bind_text_grid_control(item, 0, **callbacks)
        self.assertFalse(canvas.txtblkShapeControl.isVisible())
        notifications = []
        item.visual_geometry_changed.connect(
            lambda: notifications.append(True)
        )

        with (
            patch.object(
                canvas.txtblkShapeControl,
                'updateBoundingRect',
                wraps=canvas.txtblkShapeControl.updateBoundingRect,
            ) as hidden_shape_refresh,
            patch.object(
                item.geometry_controller,
                'grid_control_geometry',
                wraps=item.geometry_controller.grid_control_geometry,
            ) as grid_refresh,
        ):
            item.set_text_transform(after, preview=True)
            hidden_shape_refresh.assert_not_called()
            grid_refresh.assert_called_once_with(0)

            hidden_shape_refresh.reset_mock()
            grid_refresh.reset_mock()
            notifications.clear()
            self.assertTrue(item.set_text_transform(after, preview=False))
            canvas.bind_text_grid_control(item, 0, **callbacks)
            hidden_shape_refresh.assert_not_called()
            grid_refresh.assert_not_called()
            self.assertEqual(notifications, [])

    def test_projective_control_same_target_rebind_is_idempotent(self):
        canvas = Canvas()
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        item.set_text_transform(transform_state(ProjectiveTextTransform()))
        callbacks = {
            'begin_edit': lambda _index: None,
            'preview_transform': lambda _index, _transform: None,
            'commit_transform': lambda _index, _transform: None,
            'cancel_edit': lambda _index: None,
        }
        canvas.bind_text_projective_control(item, 0, **callbacks)

        with patch.object(
            canvas.txtblkProjectiveControl,
            'requestGeometryRefresh',
        ) as refresh_geometry:
            canvas.bind_text_projective_control(item, 0, **callbacks)

        refresh_geometry.assert_not_called()

    def test_content_padding_change_refreshes_gradient_once(self):
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.fontformat.gradient_enabled = True
        renderer = item.effect_renderer
        with (
            patch.object(
                renderer,
                '_effect_padding',
                return_value=renderer.padding() + 10.0,
            ),
            patch.object(renderer, '_refresh_gradient_geometry') as refresh,
        ):
            item.on_content_changed()

        refresh.assert_called_once_with()

    def test_settled_format_geometry_notifies_visual_controllers(self):
        for transform in (
            ProjectiveTextTransform(1.2, 0.9, 8.0),
            ProjectiveTextTransform(rotation_y=25.0, perspective=0.5),
            BendTextTransform(0.55),
            GridTextTransform().with_control_points((
                (0.0, 0.0),
                (1.1, 0.05),
                (-0.05, 1.0),
                (1.0, 1.0),
            )),
        ):
            for vertical in (False, True):
                with self.subTest(
                    transform=transform.transform_type,
                    vertical=vertical,
                ):
                    item, _ = self._make_pair(0, TEST_LINES[0], vertical)
                    item.set_text_transform(transform_state(transform))
                    controller = item.geometry_controller
                    notifications = []
                    item.visual_geometry_changed.connect(
                        lambda: notifications.append(True)
                    )

                    item.is_formatting = True
                    changed = item.absBoundingRect(qrect=True)
                    changed.setWidth(changed.width() + 24.0)
                    item.setRect(changed, repaint=False)
                    self.assertTrue(controller._compile_deferred)
                    notifications.clear()
                    self.assertTrue(controller.flush_deferred_compilation())
                    item.is_formatting = False
                    self.assertEqual(notifications, [True])

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
                item._old_rect = QRectF(before)
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

    def test_manual_move_and_reshape_sync_xyxy_without_replacing_lines(self):
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.blk.lines = [
            [[0, 0], [600, 0], [600, 150], [0, 150]],
            [[0, 150], [600, 150], [600, 300], [0, 300]],
        ]
        original_lines = copy.deepcopy(item.blk.lines)
        stack = QUndoStack()

        before_move = item.logical_position()
        item.setPos(item.pos() + QPointF(20, 30))
        after_move = item.logical_position()
        stack.push(MoveBlkItemsCommand(
            [item],
            before_positions=[before_move],
            after_positions=[after_move],
        ))
        self.assertEqual(item.blk.xyxy, [20, 30, 620, 330])
        self.assertEqual(item.blk.lines, original_lines)
        stack.undo()
        self.assertEqual(item.blk.xyxy, [0, 0, 600, 300])
        stack.redo()
        self.assertEqual(item.blk.xyxy, [20, 30, 620, 330])

        before_reshape = item.absBoundingRect(qrect=True)
        after_reshape = QRectF(25, 35, 500, 200)
        item._old_rect = QRectF(before_reshape)
        item.setRect(after_reshape)
        stack.push(ReshapeItemCommand(item))
        self.assertEqual(item.blk.xyxy, [25, 35, 525, 235])
        self.assertEqual(item.blk.lines, original_lines)
        stack.undo()
        self.assertEqual(item.blk.xyxy, [20, 30, 620, 330])
        stack.redo()
        self.assertEqual(item.blk.xyxy, [25, 35, 525, 235])

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
    def test_modal_point_transform_matches_blender_style_relative_math(self):
        tool = ModalPointTransform()
        points = (QPointF(0, 0), QPointF(2, 0))
        self.assertTrue(tool.begin(tool.TRANSLATE, points, QPointF(10, 10)))
        moved = tool.update(QPointF(13, 14))
        self.assertEqual(moved, (QPointF(3, 4), QPointF(5, 4)))

        reset = tool.constrain('x', QPointF(13, 14))
        self.assertEqual(reset, points)
        constrained = tool.update(QPointF(18, 30))
        self.assertEqual(constrained, (QPointF(5, 0), QPointF(7, 0)))

        reset = tool.switch_mode(tool.ROTATE, QPointF(2, 0))
        self.assertEqual(reset, points)
        rotated = tool.update(QPointF(1, 1))
        self.assertAlmostEqual(rotated[0].x(), 1.0, places=6)
        self.assertAlmostEqual(rotated[0].y(), -1.0, places=6)
        self.assertAlmostEqual(rotated[1].x(), 1.0, places=6)
        self.assertAlmostEqual(rotated[1].y(), 1.0, places=6)

        reset = tool.switch_mode(tool.SCALE, QPointF(2, 0))
        self.assertEqual(reset, points)
        scaled = tool.update(QPointF(3, 0))
        self.assertEqual(scaled, (QPointF(-1, 0), QPointF(3, 0)))
        self.assertEqual(tool.cancel(), points)

        boundary = ModalPointTransform()
        start_angle = math.radians(179.0)
        end_angle = math.radians(-179.0)
        start = QPointF(math.cos(start_angle), math.sin(start_angle))
        end = QPointF(math.cos(end_angle), math.sin(end_angle))
        self.assertTrue(boundary.begin(boundary.ROTATE, (QPointF(),), start))
        boundary.update(end)
        self.assertAlmostEqual(boundary.rotation_delta(), 2.0, places=6)

    def test_projective_controller_modal_rotate_scale_and_axis_constraints(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        initial = ProjectiveTextTransform()
        item.set_text_transform(transform_state(initial))
        controller = TextProjectiveTransformControl()
        controller.setParentItem(base)
        begun = []
        previews = []
        committed = []
        canceled = []
        controller.bind(
            item,
            0,
            begin_edit=begun.append,
            preview_transform=lambda index, transform:
            previews.append((index, transform)),
            commit_transform=lambda index, transform:
            committed.append((index, transform)),
            cancel_edit=canceled.append,
        )
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)
        center = controller.scenePos()
        start = center + QPointF(PROJECTIVE_CONTROL_RADIUS, 0.0)
        controller._cursor_scene_position = lambda: (view, start)

        def press(key):
            return controller.handle_shortcut(key)

        self.assertTrue(press(Qt.Key.Key_R))
        self.assertEqual(controller._modal_transform.axis, 'z')
        self.assertTrue(controller._update_modal(
            center + QPointF(0.0, PROJECTIVE_CONTROL_RADIUS)
        ))
        self.assertAlmostEqual(previews[-1][1].rotation_z, 90.0)
        with patch.object(
            controller._modal_transform,
            'rotation_delta',
            return_value=500.0,
        ):
            self.assertTrue(controller._preview_modal())
        self.assertEqual(previews[-1][1].rotation_z, 180.0)

        self.assertTrue(press(Qt.Key.Key_X))
        self.assertEqual(previews[-1][1], initial)
        self.assertTrue(controller._update_modal(
            center - QPointF(0.0, PROJECTIVE_CONTROL_RADIUS)
        ))
        self.assertNotEqual(previews[-1][1].rotation_x, 0.0)
        with patch.object(
            controller._modal_transform,
            'rotation_delta',
            return_value=500.0,
        ):
            self.assertTrue(controller._preview_modal())
        self.assertEqual(previews[-1][1].rotation_x, 89.0)

        self.assertTrue(press(Qt.Key.Key_S))
        self.assertEqual(previews[-1][1], initial)
        with patch.object(
            controller._modal_transform,
            'scale_factor',
            return_value=10.0,
        ):
            self.assertTrue(controller._preview_modal())
        self.assertEqual(previews[-1][1].horizontal_scale, 4.0)
        self.assertEqual(previews[-1][1].vertical_scale, 4.0)
        with patch.object(
            controller._modal_transform,
            'scale_factor',
            return_value=0.0,
        ):
            self.assertTrue(controller._preview_modal())
        self.assertEqual(previews[-1][1].horizontal_scale, 0.1)
        self.assertEqual(previews[-1][1].vertical_scale, 0.1)
        scale_start = QPointF(controller._modal_transform.start_mouse)
        vector = scale_start - center
        self.assertTrue(controller._update_modal(center + vector * 1.5))
        self.assertAlmostEqual(previews[-1][1].horizontal_scale, 1.5)
        self.assertAlmostEqual(previews[-1][1].vertical_scale, 1.5)

        self.assertTrue(press(Qt.Key.Key_Y))
        self.assertEqual(previews[-1][1], initial)
        scale_start = QPointF(controller._modal_transform.start_mouse)
        vector = scale_start - center
        self.assertTrue(controller._update_modal(center + vector * 1.25))
        self.assertEqual(previews[-1][1].horizontal_scale, 1.0)
        self.assertAlmostEqual(previews[-1][1].vertical_scale, 1.25)
        self.assertTrue(controller._finish_modal(True))

        self.assertEqual(begun, [0])
        self.assertEqual(len(committed), 1)
        self.assertEqual(canceled, [])
        controller.clear()
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_projective_controller_keeps_a_fixed_device_size(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        item.set_text_transform(transform_state(ProjectiveTextTransform()))
        controller = TextProjectiveTransformControl()
        controller.setParentItem(base)
        controller.bind(
            item,
            0,
            begin_edit=lambda _index: None,
            preview_transform=lambda _index, _transform: None,
            commit_transform=lambda _index, _transform: None,
            cancel_edit=lambda _index: None,
        )
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)

        before = controller.deviceTransform(
            view.viewportTransform()
        ).mapRect(controller.boundingRect())
        view.scale(2.0, 2.0)
        controller.requestGeometryRefresh()
        after = controller.deviceTransform(
            view.viewportTransform()
        ).mapRect(controller.boundingRect())

        self.assertAlmostEqual(before.width(), after.width(), places=5)
        self.assertAlmostEqual(before.height(), after.height(), places=5)
        controller.clear()
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_projective_controller_keeps_its_pivot_during_preview(self):
        scene = QGraphicsScene()
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        initial = ProjectiveTextTransform(
            rotation_y=35.0,
            perspective=0.7,
        )
        item.set_text_transform(transform_state(initial))
        controller = TextProjectiveTransformControl()
        controller.setParentItem(base)
        controller.bind(
            item,
            0,
            begin_edit=lambda _index: None,
            preview_transform=lambda _index, _transform: None,
            commit_transform=lambda _index, _transform: None,
            cancel_edit=lambda _index: None,
        )
        before = QPointF(controller.scenePos())

        item.set_text_transform(
            transform_state(initial.with_value('rotation_z', 55.0)),
            preview=True,
        )

        self.assertAlmostEqual(controller.scenePos().x(), before.x())
        self.assertAlmostEqual(controller.scenePos().y(), before.y())
        stage = item.geometry_controller.compiled.stages[0]
        expected = item.mapToScene(stage.context.source_bounds.center())
        self.assertEqual(controller.scenePos(), expected)
        controller.clear()
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_projective_controller_commit_creates_one_undo_command(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        initial = ProjectiveTextTransform()
        item.set_text_transform(transform_state(initial))
        controller = TextProjectiveTransformControl()
        controller.setParentItem(base)
        previous_register = getattr(shared, 'register_view_widget', None)
        shared.register_view_widget = lambda *_args: None
        self.addCleanup(
            lambda: (
                delattr(shared, 'register_view_widget')
                if previous_register is None
                else setattr(
                    shared, 'register_view_widget', previous_register
                )
            )
        )
        panel = TextTransformPanel(
            'Text Transform', 'test_transform', 'test_transform_expand'
        )
        self.addCleanup(panel.deleteLater)
        stack = QUndoStack()
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        SW.canvas = SimpleNamespace(
            push_undo_command=stack.push,
            bind_text_projective_control=lambda bound_item, index, **callbacks:
            controller.bind(bound_item, index, **callbacks),
            clear_text_transform_controls=controller.clear,
        )
        session = TextTransformEditSession(SimpleNamespace(), panel)
        session.replace_targets([item])
        panel.set_transform_items([item])
        panel.transform_panels[0].card_clicked.emit(0)
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)
        center = controller.scenePos()
        start = center + QPointF(PROJECTIVE_CONTROL_RADIUS, 0.0)
        controller._cursor_scene_position = lambda: (view, start)

        self.assertTrue(controller.handle_shortcut(Qt.Key.Key_R))
        self.assertTrue(controller._update_modal(
            center + QPointF(0.0, PROJECTIVE_CONTROL_RADIUS)
        ))
        self.assertTrue(controller._finish_modal(True))

        self.assertEqual(stack.count(), 1)
        changed = item.blk.fontformat.text_transform[0]
        self.assertAlmostEqual(changed.rotation_z, 90.0)
        stack.undo()
        self.assertEqual(item.blk.fontformat.text_transform[0], initial)
        controller.clear()
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_shape_handle_bounds_cover_its_shifted_manual_paint(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        base = QGraphicsRectItem(QRectF(0, 0, 400, 300))
        scene.addItem(base)
        control = TextBlkShapeControl(view)
        control.setParentItem(base)
        handle = control.ctrlblock_group[0]
        handle.setDeviceAngle(0.0)
        handle.setOutwardDeviceVector(QPointF(0.92, 0.39), True)

        guard = handle.pen_width / 2.0 + 1.0
        painted = handle.visible_rect.adjusted(
            -guard, -guard, guard, guard
        )
        self.assertTrue(handle.boundingRect().contains(painted))
        # Expanding the paint bounds must not expand the interaction hitbox.
        self.assertFalse(handle.shape().contains(painted.topLeft()))

        scene.removeItem(base)

    def test_active_transform_hover_suppresses_shape_only_for_bound_item(self):
        first = SimpleNamespace(hasFocus=lambda: False)
        second = SimpleNamespace(hasFocus=lambda: False)
        calls = []
        shape = SimpleNamespace(blk_item=first)

        def bind_shape(item):
            calls.append(('bind', item))
            shape.blk_item = item

        shape.setBlkItem = bind_shape
        shape.hide = lambda: calls.append(('hide', shape.blk_item))
        manager = SimpleNamespace(
            is_editting=lambda: False,
            textblk_item_list=[first, second],
            canvas=SimpleNamespace(
                active_transform_control_item=lambda: first,
            ),
            txtblkShapeControl=shape,
        )

        SceneTextManager.onTextBlkItemHoverEnter(manager, 0)
        SceneTextManager.onTextBlkItemHoverEnter(manager, 1)
        SceneTextManager.onTextBlkItemHoverEnter(manager, 0)

        self.assertEqual(
            calls,
            [
                ('hide', first),
                ('bind', second),
                ('bind', first),
                ('hide', first),
            ],
        )

    def test_grid_binding_restores_selected_shape_owner_when_cleared(self):
        selected = SimpleNamespace(isSelected=lambda: True)
        hovered = SimpleNamespace(isSelected=lambda: False)
        calls = []
        shape = SimpleNamespace(blk_item=hovered)

        def bind_shape(item):
            calls.append(('shape', item))
            shape.blk_item = item

        shape.setBlkItem = bind_shape
        shape.hide = lambda: calls.append(('hide', shape.blk_item))
        shape.show = lambda: calls.append(('show', shape.blk_item))
        shape.requestGeometryRefresh = lambda: calls.append(('refresh', None))
        grid = SimpleNamespace(item=None)

        def bind_grid(item, index, **_callbacks):
            calls.append(('grid', item, index))
            grid.item = item

        grid.bind = bind_grid
        grid.clear = lambda: setattr(grid, 'item', None)
        projective = SimpleNamespace(item=None, clear=lambda: None)
        canvas = SimpleNamespace(
            txtblkShapeControl=shape,
            txtblkGridControl=grid,
            txtblkProjectiveControl=projective,
            _rubber_band_target=None,
            selected_text_items=lambda: [selected],
        )
        canvas._restore_shape_after_transform_control = (
            lambda had_binding:
            Canvas._restore_shape_after_transform_control(canvas, had_binding)
        )

        Canvas.bind_text_grid_control(canvas, selected, 2)
        shape.blk_item = hovered
        Canvas.clear_text_transform_controls(canvas)

        self.assertEqual(
            calls,
            [
                ('shape', selected),
                ('grid', selected, 2),
                ('hide', selected),
                ('shape', selected),
            ],
        )

    def test_grid_overlay_tracks_composed_stack_in_both_writing_modes(self):
        grid = GridTextTransform(2, 2, 'catmull_rom')
        points = list(grid.control_points)
        points[4] = (0.62, 0.38)
        stack = transform_state(
            ProjectiveTextTransform(1.08, 0.95, 4.0),
            grid.with_control_points(points),
            ProjectiveTextTransform(rotation_y=25.0, perspective=0.2),
        )
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                scene = QGraphicsScene()
                base = QGraphicsRectItem(QRectF(0, 0, 900, 700))
                scene.addItem(base)
                item, _ = self._make_pair(0, TEST_LINES[0], vertical)
                item.setParentItem(base)
                item.setRotation(12.0)
                item.set_text_transform(stack)
                controller = TextGridTransformControl()
                controller.setParentItem(base)
                controller.bind(
                    item,
                    1,
                    begin_edit=lambda _index: None,
                    preview_points=lambda _index, _points: None,
                    commit_points=lambda _index, _points: None,
                    cancel_edit=lambda _index: None,
                )

                compiled = item.geometry_controller.compiled
                grid_mapper = compiled.stages[1].mapper
                prefix_mapper = compiled.stages[0].mapper
                for handle, stage_point in zip(
                    controller.handles,
                    grid_mapper.control_source_points(),
                ):
                    source = prefix_mapper.inverse_point(
                        stage_point, extrapolate=True
                    )
                    expected = item.mapToScene(
                        compiled.surface_mapper.forward_point(source)
                    )
                    actual = handle.scenePos()
                    self.assertAlmostEqual(
                        actual.x(), expected.x(), places=5
                    )
                    self.assertAlmostEqual(
                        actual.y(), expected.y(), places=5
                    )

                image = QImage(
                    900, 700, QImage.Format.Format_ARGB32_Premultiplied
                )
                image.fill(QColor(0, 0, 0, 0))
                painter = QPainter(image)
                scene.render(painter)
                painter.end()
                self.assertIsNotNone(
                    controller._overlay_renderer.cached_pixmap
                )
                destination = controller._overlay_mapper.visual_bounds(
                    controller._overlay_source_rect
                )
                item_to_control, valid = item.itemTransform(controller)
                self.assertTrue(valid)
                self.assertTrue(
                    controller.boundingRect().contains(
                        item_to_control.mapRect(destination)
                    )
                )
                self.assertGreaterEqual(
                    controller._overlay_renderer.cached_pixmap.width(),
                    math.ceil(destination.width()),
                )
                self.assertGreaterEqual(
                    controller._overlay_renderer.cached_pixmap.height(),
                    math.ceil(destination.height()),
                )
                mapper = controller._overlay_mapper
                cached_pixmap_key = (
                    controller._overlay_renderer.cached_pixmap.cacheKey()
                )
                cached_remap = controller._overlay_renderer.cached_remap
                item.setPos(item.pos() + QPointF(25.0, 15.0))
                item.moving.emit(item)
                self.assertIs(controller._overlay_mapper, mapper)

                image.fill(QColor(0, 0, 0, 0))
                painter = QPainter(image)
                scene.render(painter)
                painter.end()
                self.assertEqual(
                    controller._overlay_renderer.cached_pixmap.cacheKey(),
                    cached_pixmap_key,
                )
                self.assertIs(
                    controller._overlay_renderer.cached_remap,
                    cached_remap,
                )
                controller.clear()
                item.geometry_controller.release_render_resources()
                scene.removeItem(base)

    def test_grid_controller_double_click_enters_edit_and_deselects(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.resize(800, 500)
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        item.set_text_transform(transform_state(
            GridTextTransform()
        ))
        controller = TextGridTransformControl()
        controller.setParentItem(base)
        previous_register = getattr(shared, 'register_view_widget', None)
        shared.register_view_widget = lambda *_args: None
        self.addCleanup(
            lambda: (
                delattr(shared, 'register_view_widget')
                if previous_register is None
                else setattr(
                    shared, 'register_view_widget', previous_register
                )
            )
        )
        panel = TextTransformPanel(
            'Text Transform', 'test_transform', 'test_transform_expand'
        )
        self.addCleanup(panel.deleteLater)
        stack = QUndoStack()
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        SW.canvas = SimpleNamespace(
            push_undo_command=stack.push,
            bind_text_grid_control=lambda bound_item, index, **callbacks:
            controller.bind(bound_item, index, **callbacks),
            clear_text_transform_controls=controller.clear,
        )
        session = TextTransformEditSession(SimpleNamespace(), panel)
        session.replace_targets([item])
        panel.set_transform_items([item])
        panel.transform_panels[0].card_clicked.emit(0)
        item.begin_edit.connect(
            lambda _index: session.select_transform(-1)
        )
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)

        scene_pos = item.geometry_controller.map_source_to_scene(
            item.boundingRect().center()
        )
        QTest.mouseDClick(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            view.mapFromScene(scene_pos),
        )
        self.app.processEvents()

        self.assertTrue(item.isEditing())
        self.assertIsNone(session.selected_index)
        self.assertFalse(panel.transform_panels[0].property('selected'))
        self.assertIsNone(controller.item)
        self.assertFalse(controller.isVisible())

        item.endEdit()
        self.app.processEvents()
        self.assertFalse(controller.isVisible())
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_projective_ring_double_click_enters_edit_and_deselects(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.resize(800, 500)
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        item.set_text_transform(transform_state(ProjectiveTextTransform()))
        controller = TextProjectiveTransformControl()
        controller.setParentItem(base)
        previous_register = getattr(shared, 'register_view_widget', None)
        shared.register_view_widget = lambda *_args: None
        self.addCleanup(
            lambda: (
                delattr(shared, 'register_view_widget')
                if previous_register is None
                else setattr(
                    shared, 'register_view_widget', previous_register
                )
            )
        )
        panel = TextTransformPanel(
            'Text Transform', 'test_transform', 'test_transform_expand'
        )
        self.addCleanup(panel.deleteLater)
        previous_canvas = getattr(SW, 'canvas', None)
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        SW.canvas = SimpleNamespace(
            push_undo_command=QUndoStack().push,
            bind_text_projective_control=
            lambda bound_item, index, **callbacks:
            controller.bind(bound_item, index, **callbacks),
            clear_text_transform_controls=controller.clear,
        )
        session = TextTransformEditSession(SimpleNamespace(), panel)
        session.replace_targets([item])
        panel.set_transform_items([item])
        panel.transform_panels[0].card_clicked.emit(0)
        item.begin_edit.connect(
            lambda _index: session.select_transform(-1)
        )
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)

        ring_point = controller.rings['z'].path().pointAtPercent(0.0)
        scene_pos = controller.mapToScene(ring_point)
        QTest.mouseDClick(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            view.mapFromScene(scene_pos),
        )
        self.app.processEvents()

        self.assertTrue(item.isEditing())
        self.assertIsNone(session.selected_index)
        self.assertFalse(panel.transform_panels[0].property('selected'))
        self.assertIsNone(controller.item)
        self.assertFalse(controller.isVisible())

        item.endEdit()
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_grid_controller_group_drag_moves_selected_circle_handles(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        grid = GridTextTransform(2, 2, 'catmull_rom')
        points = list(grid.control_points)
        points[4] = (0.56, 0.44)
        grid = grid.with_control_points(points)
        item.set_text_transform(transform_state(grid))
        controller = TextGridTransformControl()
        controller.setParentItem(base)
        begun = []
        previews = []
        committed = []
        canceled = []
        controller.bind(
            item,
            0,
            begin_edit=begun.append,
            preview_points=lambda index, points: previews.append(
                (index, points)
            ),
            commit_points=lambda index, points: committed.append(
                (index, points)
            ),
            cancel_edit=canceled.append,
        )
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)

        self.assertEqual(len(controller.handles), 9)
        self.assertFalse(controller.path().isEmpty())
        self.assertIsNotNone(controller._overlay_mapper)
        grid_mapper = controller._overlay_mapper.stages[0]
        original_forward_point = grid_mapper.forward_point
        original_inverse_point = grid_mapper.inverse_point
        grid_mapper.forward_point = lambda _point: self.fail(
            'controller handles must use batched forward mapping'
        )
        try:
            controller.requestGeometryRefresh()
        finally:
            grid_mapper.forward_point = original_forward_point
        grid_mapper.inverse_point = lambda *_args, **_kwargs: self.fail(
            'dragging a Grid handle must not invert that same Grid stage'
        )

        try:
            image = QImage(800, 500, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(QColor(0, 0, 0, 0))
            painter = QPainter(image)
            scene.render(painter)
            painter.end()
            self.assertIsNotNone(controller._overlay_renderer.cached_pixmap)
            controller._set_selected_indices({0})
            start = controller.handles[1].scenePos()
            self.assertTrue(controller.begin_handle_drag(
                1, start, Qt.KeyboardModifier.ControlModifier
            ))
            self.assertEqual(controller.selected_indices, {0, 1})
            self.assertTrue(controller.move_handle_drag(
                start + QPointF(20.0, 10.0)
            ))
            self.assertTrue(controller.finish_handle_drag())
        finally:
            grid_mapper.inverse_point = original_inverse_point

        self.assertEqual(begun, [0])
        self.assertEqual(canceled, [])
        self.assertTrue(previews)
        self.assertEqual(len(committed), 1)
        moved = committed[0][1]
        delta0 = (
            moved[0][0] - grid.control_points[0][0],
            moved[0][1] - grid.control_points[0][1],
        )
        delta1 = (
            moved[1][0] - grid.control_points[1][0],
            moved[1][1] - grid.control_points[1][1],
        )
        self.assertAlmostEqual(delta0[0], delta1[0], places=6)
        self.assertAlmostEqual(delta0[1], delta1[1], places=6)
        self.assertEqual(moved[2], grid.control_points[2])
        controller.clear()
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_grid_modal_shortcuts_reset_modes_and_commit_once(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        base = QGraphicsRectItem(QRectF(0, 0, 800, 500))
        scene.addItem(base)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(base)
        grid = GridTextTransform(2, 2, 'catmull_rom')
        item.set_text_transform(transform_state(grid))
        controller = TextGridTransformControl()
        controller.setParentItem(base)
        begun = []
        previews = []
        committed = []
        canceled = []
        controller.bind(
            item,
            0,
            begin_edit=begun.append,
            preview_points=lambda index, points: previews.append(
                (index, points)
            ),
            commit_points=lambda index, points: committed.append(
                (index, points)
            ),
            cancel_edit=canceled.append,
        )
        controller._set_selected_indices({0, 1})
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)
        start = controller.handles[0].scenePos()
        controller._cursor_scene_position = lambda: (view, start)

        def press(key):
            return controller.handle_shortcut(key)

        self.assertTrue(press(Qt.Key.Key_G))
        self.assertTrue(controller._update_modal(start + QPointF(25, 10)))
        self.assertEqual(begun, [0])
        self.assertTrue(previews)
        self.assertEqual(committed, [])
        moved = previews[-1][1]
        self.assertNotEqual(moved, grid.control_points)

        self.assertTrue(press(Qt.Key.Key_X))
        self.assertEqual(previews[-1][1], grid.control_points)
        self.assertTrue(controller.modal_indicator.isVisible())
        self.assertTrue(controller._update_modal(start + QPointF(40, 30)))
        constrained = previews[-1][1]
        self.assertAlmostEqual(constrained[0][1], grid.control_points[0][1])

        self.assertTrue(press(Qt.Key.Key_S))
        self.assertEqual(previews[-1][1], grid.control_points)
        self.assertEqual(controller._modal_transform.mode, 'scale')
        self.assertTrue(controller._update_modal(start + QPointF(70, 30)))
        self.assertTrue(press(Qt.Key.Key_R))
        self.assertEqual(previews[-1][1], grid.control_points)
        self.assertEqual(controller._modal_transform.mode, 'rotate')
        self.assertTrue(controller._update_modal(start + QPointF(50, 60)))
        self.assertTrue(controller._finish_modal(False))
        self.assertEqual(canceled, [0])
        self.assertEqual(committed, [])

        self.assertTrue(press(Qt.Key.Key_G))
        self.assertTrue(controller._update_modal(start + QPointF(20, 15)))
        self.assertTrue(controller._finish_modal(True))
        self.assertEqual(begun, [0, 0])
        self.assertEqual(len(committed), 1)
        self.assertEqual(canceled, [0])
        controller.clear()
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_canvas_routes_grid_modal_key_and_unheld_mouse_following(self):
        canvas = Canvas()
        canvas.gv.resize(800, 500)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        item.set_text_transform(transform_state(
            GridTextTransform(2, 2, 'bilinear')
        ))
        previews = []
        committed = []
        canceled = []
        canvas.bind_text_grid_control(
            item,
            0,
            begin_edit=lambda _index: None,
            preview_points=lambda index, points: previews.append(
                (index, points)
            ),
            commit_points=lambda index, points: committed.append(
                (index, points)
            ),
            cancel_edit=canceled.append,
        )
        controller = canvas.txtblkGridControl
        controller._set_selected_indices({0, 1})
        canvas.gv.show()
        canvas.gv.setFocus()
        self.app.processEvents()
        self.addCleanup(canvas.gv.close)
        start = controller.handles[0].scenePos()
        controller._cursor_scene_position = lambda: (canvas.gv, start)

        QTest.keyClick(canvas.gv.viewport(), Qt.Key.Key_G)
        self.app.processEvents()
        self.assertTrue(controller._modal_transform.active)
        target = start + QPointF(35, 20)
        viewport_target = canvas.gv.mapFromScene(target)
        QTest.mouseMove(canvas.gv.viewport(), viewport_target)
        self.app.processEvents()
        self.assertTrue(previews)
        self.assertEqual(committed, [])

        QTest.mouseClick(
            canvas.gv.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            viewport_target,
        )
        self.app.processEvents()
        self.assertFalse(controller._modal_transform.active)
        self.assertEqual(committed, [])
        self.assertEqual(canceled, [0])

        QTest.keyClick(canvas.gv.viewport(), Qt.Key.Key_G)
        self.app.processEvents()
        self.assertTrue(controller._modal_transform.active)
        viewport_target = canvas.gv.mapFromScene(target + QPointF(20, 10))
        QTest.mouseMove(canvas.gv.viewport(), viewport_target)
        self.app.processEvents()
        QTest.mouseClick(
            canvas.gv.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            viewport_target,
        )
        self.app.processEvents()
        self.assertFalse(controller._modal_transform.active)
        self.assertEqual(len(committed), 1)
        self.assertEqual(canceled, [0])
        controller.clear()
        item.geometry_controller.release_render_resources()

    def test_draw_tool_shortcut_routes_rotate_to_selected_grid_handles(self):
        canvas = Canvas()
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        item.set_text_transform(transform_state(
            GridTextTransform(2, 2, 'bilinear')
        ))
        canvas.bind_text_grid_control(
            item,
            0,
            begin_edit=lambda _index: None,
            preview_points=lambda _index, _points: None,
            commit_points=lambda _index, _points: None,
            cancel_edit=lambda _index: None,
        )
        controller = canvas.txtblkGridControl
        controller._set_selected_indices({0, 1})
        start = controller.handles[0].scenePos()
        controller._cursor_scene_position = lambda: (canvas.gv, start)
        drawing_tools = []
        drawing_panel = SimpleNamespace(
            canvas=canvas,
            isVisible=lambda: True,
            setCurrentToolByName=drawing_tools.append,
        )

        DrawingPanel.shortcutSetCurrentToolByName(
            drawing_panel, 'rect', Qt.Key.Key_R
        )

        self.assertEqual(controller._modal_transform.mode, 'rotate')
        self.assertEqual(drawing_tools, [])
        controller._finish_modal(False)
        controller.clear()
        item.geometry_controller.release_render_resources()

    def test_draw_tool_shortcut_routes_to_selected_projective_stage(self):
        canvas = Canvas()
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        item.set_text_transform(transform_state(ProjectiveTextTransform()))
        canvas.bind_text_projective_control(
            item,
            0,
            begin_edit=lambda _index: None,
            preview_transform=lambda _index, _transform: None,
            commit_transform=lambda _index, _transform: None,
            cancel_edit=lambda _index: None,
        )
        controller = canvas.txtblkProjectiveControl
        start = controller.scenePos() + QPointF(PROJECTIVE_CONTROL_RADIUS, 0.0)
        controller._cursor_scene_position = lambda: (canvas.gv, start)
        drawing_tools = []
        drawing_panel = SimpleNamespace(
            canvas=canvas,
            isVisible=lambda: True,
            setCurrentToolByName=drawing_tools.append,
        )

        DrawingPanel.shortcutSetCurrentToolByName(
            drawing_panel, 'rect', Qt.Key.Key_S
        )

        self.assertEqual(controller._modal_transform.mode, 'scale')
        self.assertEqual(drawing_tools, [])
        controller._finish_modal(False)
        controller.clear()
        item.geometry_controller.release_render_resources()

    def test_grid_rubber_selection_can_start_on_canvas_background(self):
        canvas = Canvas()
        context_requests = []
        canvas.context_menu_requested.connect(
            lambda *args: context_requests.append(args)
        )
        canvas.gv.resize(800, 500)
        item, _ = self._make_pair(0, TEST_LINES[0], False)
        item.setParentItem(canvas.textLayer)
        item.set_text_transform(transform_state(
            GridTextTransform(2, 2, 'bilinear')
        ))
        canvas.bind_text_grid_control(
            item,
            0,
            begin_edit=lambda _index: None,
            preview_points=lambda _index, _points: None,
            commit_points=lambda _index, _points: None,
            cancel_edit=lambda _index: None,
        )
        controller = canvas.txtblkGridControl
        controller._set_selected_indices({0})
        canvas.gv.show()
        self.app.processEvents()
        self.addCleanup(canvas.gv.close)
        bounds = controller.mapRectToScene(controller.path().boundingRect())
        start = bounds.bottomRight() + QPointF(30.0, 30.0)
        end = bounds.topLeft() - QPointF(10.0, 10.0)
        canvas.setSceneRect(QRectF(start, end).normalized().adjusted(
            -50.0, -50.0, 50.0, 50.0
        ))
        canvas.gv.fitInView(
            canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self.app.processEvents()
        self.assertTrue(canvas._is_grid_rubber_origin(start))

        start_view = canvas.gv.mapFromScene(start)
        end_view = canvas.gv.mapFromScene(end)
        QTest.mousePress(
            canvas.gv.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            start_view,
        )
        self.app.processEvents()
        self.assertTrue(canvas.rubber_band.isVisible())
        QTest.mouseMove(canvas.gv.viewport(), end_view)
        QTest.mouseRelease(
            canvas.gv.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            end_view,
        )
        self.app.processEvents()

        self.assertEqual(
            controller.selected_indices,
            set(range(len(controller.handles))),
        )
        self.assertFalse(canvas.rubber_band.isVisible())
        self.assertEqual(context_requests, [])

        controller._set_selected_indices(set())
        inside = (
            controller.handles[0].scenePos()
            + controller.handles[4].scenePos()
        ) / 2.0
        inside_end = controller.handles[0].scenePos() - QPointF(5.0, 5.0)
        self.assertTrue(canvas._is_grid_rubber_origin(inside))
        QTest.mousePress(
            canvas.gv.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            canvas.gv.mapFromScene(inside),
        )
        QTest.mouseMove(
            canvas.gv.viewport(), canvas.gv.mapFromScene(inside_end)
        )
        QTest.mouseRelease(
            canvas.gv.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            canvas.gv.mapFromScene(inside_end),
        )
        self.app.processEvents()
        self.assertEqual(controller.selected_indices, {0})
        self.assertEqual(context_requests, [])
        controller.clear()
        item.geometry_controller.release_render_resources()
        canvas.removeItem(item)

    def test_canvas_rubber_band_still_selects_scene_items(self):
        canvas = Canvas()
        canvas.imgtrans_proj = SimpleNamespace(img_valid=True)
        canvas.gv.resize(500, 350)
        selectable = QGraphicsRectItem(QRectF(100, 100, 60, 40))
        selectable.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
            True,
        )
        selectable.setParentItem(canvas.baseLayer)
        canvas.setSceneRect(0, 0, 300, 240)
        canvas.gv.fitInView(
            canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        canvas.gv.show()
        self.app.processEvents()
        self.addCleanup(canvas.gv.close)
        start = QPointF(80, 80)
        end = QPointF(180, 160)

        self.assertTrue(canvas._begin_rubber_band(
            start,
            Qt.KeyboardModifier.NoModifier,
            Qt.MouseButton.RightButton,
            target='scene',
            on_update=canvas._select_scene_items_in_rect,
        ))
        self.assertTrue(canvas._update_rubber_band(end))

        self.assertTrue(canvas.rubber_band.isVisible())
        self.assertTrue(selectable.isSelected())
        self.assertTrue(canvas._finish_rubber_band(
            end, Qt.MouseButton.RightButton
        ))
        self.assertFalse(canvas.rubber_band.isVisible())

    def test_bend_resize_uses_frozen_drag_coordinates(self):
        for vertical in (False, True):
            for bend in (-0.95, 0.95):
                with self.subTest(
                    vertical=vertical,
                    bend=bend,
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
                            BendTextTransform(bend)
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

    def test_grid_resize_continues_from_the_previous_drag_sample(self):
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        base = QGraphicsRectItem(QRectF(0, 0, 900, 700))
        scene.addItem(base)
        block = TextBlock(
            [180, 120, 600, 300],
            _bounding_rect=[180, 120, 420, 180],
            translation=TEST_LINES[0],
        )
        item = TextBlkItem(block, 0)
        item.setParentItem(base)
        grid = GridTextTransform(2, 2, 'bilinear')
        points = list(grid.control_points)
        points[4] = (0.35, 0.65)
        item.set_text_transform(transform_state(
            grid.with_control_points(points)
        ))
        control = TextBlkShapeControl(view)
        control.setParentItem(base)
        control.setBlkItem(item)
        view.show()
        self.app.processEvents()
        self.addCleanup(view.close)

        handle_index = 4
        initial_transform = item.sceneTransform()
        mapper = item.geometry_controller.visual_mapper
        initial_source = QPointF(
            item.geometry_controller.source_handle_points()[handle_index]
        )
        initial_scene = initial_transform.map(
            mapper.forward_point(initial_source)
        )
        control.beginResize(handle_index, initial_scene)
        frozen_mapper = control._resize_scene_to_source
        calls = []

        def tracked_mapper(scene_point, previous_source):
            result = frozen_mapper(scene_point, previous_source)
            calls.append((QPointF(previous_source), QPointF(result)))
            return result

        control._resize_scene_to_source = tracked_mapper
        for extension in (15.0, 30.0, 45.0):
            target = initial_source + QPointF(extension, extension)
            control.resizeFromScene(
                handle_index,
                initial_transform.map(mapper.forward_point(target)),
            )

        self.assertEqual(len(calls), 3)
        for previous, prior_result in zip(
            (call[0] for call in calls[1:]),
            (call[1] for call in calls[:-1]),
        ):
            self.assertAlmostEqual(previous.x(), prior_result.x(), places=6)
            self.assertAlmostEqual(previous.y(), prior_result.y(), places=6)
        control.finishResize()
        control.setBlkItem(None)
        item.geometry_controller.release_render_resources()
        scene.removeItem(base)

    def test_extended_transform_shape_control_tracks_geometry(self):
        for vertical in (False, True):
            for state in (
                FIRST_TRANSFORM,
                transform_state(
                    ProjectiveTextTransform(rotation_y=35.0, perspective=0.55)
                ),
                transform_state(BendTextTransform(0.65)),
                transform_state(SineTextTransform()),
                transform_state(
                    ProjectiveTextTransform(rotation_y=35.0, perspective=0.55),
                    BendTextTransform(0.65),
                ),
                transform_state(
                    GridTextTransform().with_control_points((
                        (0.0, 0.0),
                        (1.08, 0.04),
                        (-0.04, 1.0),
                        (1.0, 1.0),
                    ))
                ),
            ):
                with self.subTest(
                    vertical=vertical,
                    transforms=tuple(
                        transform.transform_type
                        for transform in state
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
                            control.mapToScene(control.shape()).boundingRect(),
                            item.visual_bounds_in_scene(),
                        )

                    item.setFontSize(38, repaint_background=False)
                    self.app.processEvents()
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
                        control.mapToScene(control.shape()).boundingRect(),
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
            ProjectiveTextTransform(rotation_y=25.0, perspective=0.5),
            BendTextTransform(0.55),
        ):
            state = (
                transform
                if isinstance(transform, TextTransformStack)
                else transform_state(transform)
            )
            with self.subTest(
                transform=(
                    state[0].transform_type
                    if state.transforms
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

                handle_center = control.ctrlblock_group[1].scenePos()
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

    def test_item_owned_selection_guide_can_be_suppressed_for_export(self):
        for transform in (NEUTRAL, FIRST_TRANSFORM):
            with self.subTest(
                transform=(
                    transform[0].transform_type
                    if transform.transforms
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
