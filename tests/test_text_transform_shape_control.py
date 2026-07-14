import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from ballontranslator.ui.texteditshapecontrol import TextBlkShapeControl
from ballontranslator.ui.textedit_commands import ReshapeItemCommand
from ballontranslator.ui.textitem import TextBlkItem
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


_APP = QApplication.instance() or QApplication([])


class NoPixmapTextItem(TextBlkItem):
    def toPixmap(self):
        raise AssertionError('rotation preview must use the live item')


def make_control(item_class=TextBlkItem):
    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    base_layer = QGraphicsRectItem()
    scene.addItem(base_layer)
    base_layer.setPos(17, -3)
    base_layer.setScale(1.4)
    text_layer = QGraphicsRectItem(base_layer)
    text_layer.setPos(5, 8)

    fontformat = FontFormat(
        horizontal_scale=1.5,
        vertical_scale=0.7,
        slant_angle=15.0,
    )
    block = TextBlock(
        xyxy=[10, 20, 110, 70],
        _bounding_rect=[10, 20, 100, 50],
        translation='x',
        angle=23,
        fontformat=fontformat,
    )
    item = item_class(block)
    item.setParentItem(text_layer)
    control = TextBlkShapeControl(view)
    control.setParentItem(base_layer)
    control.setBlkItem(item)
    return scene, view, item, block, control


class TextTransformShapeControlTests(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected, places=7):
        self.assertAlmostEqual(actual.x(), expected.x(), places=places)
        self.assertAlmostEqual(actual.y(), expected.y(), places=places)

    def assertPolygonAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_point, expected_point in zip(actual, expected):
            self.assertPointAlmostEqual(actual_point, expected_point)

    @staticmethod
    def expectedHandlePoints(polygon):
        corners = [QPointF(point) for point in polygon]
        edges = [
            (corners[index] + corners[(index + 1) % 4]) / 2
            for index in range(4)
        ]
        return [
            point
            for index in range(4)
            for point in (corners[index], edges[index])
        ]

    def test_exact_polygon_and_eight_device_stable_handles(self):
        scene, view, item, block, control = make_control()
        expected_polygon = item.visual_polygon_in_scene()

        self.assertPolygonAlmostEqual(
            control.visualPolygonInScene(), expected_polygon
        )
        expected_handles = self.expectedHandlePoints(expected_polygon)
        for index, (handle, expected) in enumerate(
            zip(control.ctrlblock_group, expected_handles)
        ):
            with self.subTest(handle=index):
                self.assertPointAlmostEqual(handle.scenePos(), expected)
                self.assertTrue(
                    handle.flags()
                    & QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
                )

        original_rects = [QRectF(handle.rect()) for handle in control.ctrlblock_group]
        for zoom in (0.25, 4.0):
            view.resetTransform()
            view.scale(zoom, zoom)
            control.updateScale(zoom)
            for handle, original_rect in zip(
                control.ctrlblock_group, original_rects
            ):
                self.assertEqual(handle.rect(), original_rect)
                device = handle.deviceTransform(view.viewportTransform())
                self.assertAlmostEqual(device.m11(), 1.0)
                self.assertAlmostEqual(device.m12(), 0.0)
                self.assertAlmostEqual(device.m21(), 0.0)
                self.assertAlmostEqual(device.m22(), 1.0)

    def test_corner_resize_keeps_opposite_scene_anchor_and_document(self):
        scene, view, item, block, control = make_control()
        old_abs = item.absBoundingRect(qrect=True)
        old_html = item.document().toHtml()
        old_revision = item.document().revision()
        opposite = QPointF(control.handleScenePoint(0))
        reshaped = []
        item.reshaped.connect(lambda reshaped_item: reshaped.append(reshaped_item))

        control.beginResize(4)
        item.startReshape()
        local = item.logical_unpadded_rect()
        target = item.mapToScene(
            QPointF(local.right() + 24, local.bottom() + 17)
        )
        control.resizeFromScene(4, target)

        new_abs = item.absBoundingRect(qrect=True)
        self.assertAlmostEqual(new_abs.width(), old_abs.width() + 24)
        self.assertAlmostEqual(new_abs.height(), old_abs.height() + 17)
        self.assertPointAlmostEqual(control.handleScenePoint(0), opposite)
        self.assertPointAlmostEqual(
            item.transformOriginPoint(), item.logical_unpadded_rect().center()
        )
        self.assertPolygonAlmostEqual(
            control.visualPolygonInScene(), item.visual_polygon_in_scene()
        )
        self.assertEqual(item.document().toHtml(), old_html)
        self.assertEqual(item.document().revision(), old_revision)

        item.endReshape()
        self.assertEqual(reshaped, [item])
        self.assertEqual(item.oldRect, old_abs)

    def test_edge_resize_keeps_opposite_midpoint_fixed(self):
        scene, view, item, block, control = make_control()
        old_abs = item.absBoundingRect(qrect=True)
        opposite = QPointF(control.handleScenePoint(7))

        control.beginResize(3)
        local = item.logical_unpadded_rect()
        target = item.mapToScene(
            QPointF(local.right() + 13, local.center().y() + 100)
        )
        control.resizeFromScene(3, target)

        new_abs = item.absBoundingRect(qrect=True)
        self.assertAlmostEqual(new_abs.width(), old_abs.width() + 13)
        self.assertAlmostEqual(new_abs.height(), old_abs.height())
        self.assertPointAlmostEqual(control.handleScenePoint(7), opposite)

    def test_reshape_command_refreshes_exact_polygon_on_undo_and_redo(self):
        scene, view, item, block, control = make_control()
        old_rect = item.absBoundingRect(qrect=True)
        item.startReshape()
        control.beginResize(4)
        logical = item.logical_unpadded_rect()
        control.resizeFromScene(
            4,
            item.mapToScene(
                QPointF(logical.right() + 19, logical.bottom() + 11)
            ),
        )
        new_rect = item.absBoundingRect(qrect=True)
        command = ReshapeItemCommand(item, control)

        command.redo()
        command.undo()
        self.assertEqual(item.absBoundingRect(qrect=True), old_rect)
        self.assertPolygonAlmostEqual(
            control.visualPolygonInScene(), item.visual_polygon_in_scene()
        )

        command.redo()
        self.assertEqual(item.absBoundingRect(qrect=True), new_rect)
        self.assertPolygonAlmostEqual(
            control.visualPolygonInScene(), item.visual_polygon_in_scene()
        )

    def test_live_rotation_preview_restores_canonical_angle(self):
        scene, view, item, block, control = make_control(NoPixmapTextItem)
        old_html = item.document().toHtml()
        old_revision = item.document().revision()
        original_angle = item.rotation()
        center = control.visualCenterInScene()

        rotate_start, captured_angle = control.beginRotation(
            center + QPointF(100, 0)
        )
        preview_angle = control.rotateFromScene(
            center + QPointF(0, 100), rotate_start
        )

        self.assertAlmostEqual(preview_angle, original_angle + 90)
        self.assertAlmostEqual(item.rotation(), preview_angle)
        self.assertEqual(block.angle, original_angle)
        self.assertPolygonAlmostEqual(
            control.visualPolygonInScene(), item.visual_polygon_in_scene()
        )

        committed_angle = control.finishRotationPreview(captured_angle)
        self.assertAlmostEqual(committed_angle, preview_angle)
        self.assertAlmostEqual(item.rotation(), original_angle)
        self.assertEqual(block.angle, original_angle)
        self.assertEqual(item.document().toHtml(), old_html)
        self.assertEqual(item.document().revision(), old_revision)

    def test_legacy_move_and_rotation_refresh_calls_rebuild_polygon(self):
        scene, view, item, block, control = make_control()

        item.setPos(item.pos() + QPointF(11, -7))
        control.setPos(item.pos())
        self.assertPolygonAlmostEqual(
            control.visualPolygonInScene(), item.visual_polygon_in_scene()
        )

        old_reported_angle = control.rotation()
        item.setAngle(37)
        self.assertEqual(control.rotation(), old_reported_angle)
        control.setRotation(37)
        self.assertEqual(control.rotation(), 37)
        self.assertPolygonAlmostEqual(
            control.visualPolygonInScene(), item.visual_polygon_in_scene()
        )


if __name__ == '__main__':
    unittest.main()
