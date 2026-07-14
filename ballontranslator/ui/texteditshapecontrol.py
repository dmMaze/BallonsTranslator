import math

from qtpy.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QLabel,
    QStyleOptionGraphicsItem,
    QWidget,
)
from qtpy.QtCore import QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF, QTransform

from .cursor import resizeCursorList, rotateCursorList
from .textitem import TextBlkItem


CBEDGE_WIDTH = 30
VISUALIZE_HITBOX = False


class ControlBlockItem(QGraphicsRectItem):
    DRAG_NONE = 0
    DRAG_RESHAPE = 1
    DRAG_ROTATE = 2
    CURSOR_IDX = -1

    def __init__(self, parent, idx: int):
        super().__init__(parent)
        self.idx = idx
        self.ctrl = parent
        self.edge_width = 0
        self.drag_mode = self.DRAG_NONE
        self.rotate_start = 0.0
        self.rotate_original = 0.0
        self.setAcceptHoverEvents(True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        self.updateEdgeWidth(CBEDGE_WIDTH)

    def updateEdgeWidth(self, edge_width: float):
        self.edge_width = edge_width
        visible_len = edge_width / 2
        self.pen_width = edge_width / CBEDGE_WIDTH * 2
        self.visible_rect = QRectF(
            -visible_len / 2,
            -visible_len / 2,
            visible_len,
            visible_len,
        )
        self.setRect(-edge_width / 2, -edge_width / 2, edge_width, edge_width)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget,
    ) -> None:
        painter.setPen(
            QPen(
                QColor(75, 75, 75),
                self.pen_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.SquareCap,
            )
        )
        painter.fillRect(self.visible_rect, QColor(200, 200, 200, 125))
        painter.drawRect(self.visible_rect)
        if VISUALIZE_HITBOX:
            painter.setPen(
                QPen(
                    QColor(75, 125, 0),
                    self.pen_width,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.SquareCap,
                )
            )
            painter.drawRect(self.boundingRect())

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        return super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        angle_idx = self.get_angle_idx(self.ctrl.handleSceneAngle(self.idx))
        if self.visible_rect.contains(event.pos()):
            self.setCursor(resizeCursorList[angle_idx % 4])
        else:
            self.setCursor(rotateCursorList[angle_idx])
        self.CURSOR_IDX = angle_idx
        return super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if self.drag_mode == self.DRAG_NONE:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        return super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.ctrl.ctrlblockPressed()
        if event.button() == Qt.MouseButton.LeftButton and self.ctrl.blk_item is not None:
            blk_item = self.ctrl.blk_item
            blk_item.setSelected(True)
            if self.visible_rect.contains(event.pos()):
                self.ctrl.reshaping = True
                self.drag_mode = self.DRAG_RESHAPE
                self.ctrl.beginResize(self.idx)
                blk_item.startReshape()
            else:
                self.drag_mode = self.DRAG_ROTATE
                self.rotate_start, self.rotate_original = self.ctrl.beginRotation(
                    event.scenePos()
                )
                self.updateAngleLabelPos()
        event.accept()

    def updateAngleLabelPos(self):
        angle_label = self.ctrl.angleLabel
        gv = angle_label.parent()
        pos = gv.mapFromScene(self.ctrl.handleScenePoint(self.idx))
        x = max(min(pos.x(), gv.width() - angle_label.width()), 0)
        y = max(min(pos.y(), gv.height() - angle_label.height()), 0)
        angle_label.move(QPoint(x, y))
        angle = self.ctrl.blk_item.rotation() if self.ctrl.blk_item is not None else 0
        angle_label.setText("{:.1f}\N{DEGREE SIGN}".format(angle))
        if not angle_label.isVisible():
            angle_label.setVisible(True)
            angle_label.raise_()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        blk_item = self.ctrl.blk_item
        if blk_item is None:
            return
        if self.drag_mode == self.DRAG_RESHAPE:
            self.ctrl.resizeFromScene(self.idx, event.scenePos())
        elif self.drag_mode == self.DRAG_ROTATE:
            self.ctrl.rotateFromScene(event.scenePos(), self.rotate_start)
            angle_idx = self.get_angle_idx(self.ctrl.handleSceneAngle(self.idx))
            if self.CURSOR_IDX != angle_idx:
                self.setCursor(rotateCursorList[angle_idx])
                self.CURSOR_IDX = angle_idx
            self.updateAngleLabelPos()
        event.accept()

    @staticmethod
    def get_angle_idx(angle) -> int:
        return int((angle + 22.5) % 360 / 45)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.ctrl.blk_item is not None:
            self.ctrl.reshaping = False
            if self.drag_mode == self.DRAG_RESHAPE:
                self.ctrl.blk_item.endReshape()
            elif self.drag_mode == self.DRAG_ROTATE:
                preview_angle = self.ctrl.finishRotationPreview(
                    self.rotate_original
                )
                self.ctrl.blk_item.rotated.emit(preview_angle)
            self.drag_mode = self.DRAG_NONE
            self.ctrl.angleLabel.setVisible(False)
            self.ctrl.blk_item.update()
            self.ctrl.updateBoundingRect()
        return super().mouseReleaseEvent(event)


class TextBlkShapeControl(QGraphicsRectItem):
    blk_item: TextBlkItem = None
    ctrl_block: ControlBlockItem = None
    reshaping: bool = False

    def __init__(self, parent) -> None:
        super().__init__()
        self.gv = parent
        self._visual_polygon = QPolygonF()
        self._reported_angle = 0.0
        self._updating_bounds = False
        self._resize_old_logical_rect = None
        self._resize_opposite_scene = None
        self._resize_opposite_idx = None
        self.ctrlblock_group = [ControlBlockItem(self, idx) for idx in range(8)]

        pen = QPen(QColor(69, 71, 87), 2, Qt.PenStyle.SolidLine)
        pen.setDashPattern([7, 14])
        self.setPen(pen)
        self.setVisible(False)

        self.angleLabel = QLabel(parent)
        self.angleLabel.setText("{:.1f}\N{DEGREE SIGN}".format(0.0))
        self.angleLabel.setObjectName("angleLabel")
        self.angleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angleLabel.setHidden(True)

        self.current_scale = 1.0
        self.need_rescale = False
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    @staticmethod
    def _handle_points(polygon: QPolygonF):
        corners = [QPointF(point) for point in polygon]
        if len(corners) != 4:
            return []
        edges = [
            (corners[index] + corners[(index + 1) % 4]) / 2
            for index in range(4)
        ]
        return [
            point
            for index in range(4)
            for point in (corners[index], edges[index])
        ]

    @classmethod
    def _item_handle_points_in_scene(cls, item: TextBlkItem):
        return cls._handle_points(item.visual_polygon_in_scene())

    def setBlkItem(self, blk_item: TextBlkItem):
        if self.blk_item == blk_item and self.isVisible():
            return
        if self.blk_item is not None:
            self.blk_item.under_ctrl = False
            if self.blk_item.isEditing():
                self.blk_item.endEdit()
            self.blk_item.update()

        self.blk_item = blk_item
        if blk_item is None:
            self._visual_polygon = QPolygonF()
            self._reported_angle = 0.0
            self.hide()
            return
        blk_item.under_ctrl = True
        blk_item.update()
        self.updateBoundingRect()
        self.show()

    def updateBoundingRect(self):
        if self.blk_item is None:
            return

        scene_polygon = self.blk_item.visual_polygon_in_scene()
        parent = self.parentItem()
        if parent is None:
            parent_polygon = QPolygonF([QPointF(point) for point in scene_polygon])
        else:
            parent_polygon = QPolygonF(
                [parent.mapFromScene(point) for point in scene_polygon]
            )
        bounds = parent_polygon.boundingRect()
        origin = bounds.topLeft()
        local_polygon = QPolygonF(
            [point - origin for point in parent_polygon]
        )

        self._updating_bounds = True
        try:
            self.prepareGeometryChange()
            self._visual_polygon = local_polygon
            self._reported_angle = self.blk_item.rotation()
            super().setTransform(QTransform(), False)
            super().setRotation(0.0)
            super().setPos(origin)
            super().setRect(local_polygon.boundingRect())
            self.updateControlBlocks()
            self.update()
        finally:
            self._updating_bounds = False

    def visualPolygonInScene(self) -> QPolygonF:
        return QPolygonF([self.mapToScene(point) for point in self._visual_polygon])

    def shape(self) -> QPainterPath:
        if len(self._visual_polygon) != 4:
            return super().shape()
        path = QPainterPath()
        path.addPolygon(self._visual_polygon)
        path.closeSubpath()
        return path

    def setRect(self, *args):
        if self.blk_item is not None and not self._updating_bounds:
            self.updateBoundingRect()
            return
        super().setRect(*args)
        self._visual_polygon = QPolygonF()
        self.updateControlBlocks()

    def setPos(self, *args):
        if self.blk_item is not None and not self._updating_bounds:
            self.updateBoundingRect()
            return
        return super().setPos(*args)

    def rotation(self) -> float:
        if self.blk_item is not None:
            return self._reported_angle
        return super().rotation()

    def setRotation(self, angle: float) -> None:
        if self.blk_item is not None and not self._updating_bounds:
            self._reported_angle = angle
            self.updateBoundingRect()
            return
        super().setRotation(angle)

    def updateControlBlocks(self):
        if len(self._visual_polygon) == 4:
            points = self._handle_points(self._visual_polygon)
        else:
            rect = self.rect()
            polygon = QPolygonF(
                [
                    rect.topLeft(),
                    rect.topRight(),
                    rect.bottomRight(),
                    rect.bottomLeft(),
                ]
            )
            points = self._handle_points(polygon)
        for ctrlblock, point in zip(self.ctrlblock_group, points):
            ctrlblock.setPos(point)

    def setAngle(self, angle: float) -> None:
        self.setRotation(angle)

    def visualCenterInScene(self) -> QPointF:
        polygon = self.blk_item.visual_polygon_in_scene()
        if len(polygon) != 4:
            return self.sceneBoundingRect().center()
        return sum((QPointF(point) for point in polygon), QPointF()) / 4

    def handleScenePoint(self, idx: int) -> QPointF:
        if self.blk_item is not None:
            points = self._item_handle_points_in_scene(self.blk_item)
            if len(points) == 8:
                return points[idx]
        return self.ctrlblock_group[idx].scenePos()

    def handleSceneAngle(self, idx: int) -> float:
        vector = self.handleScenePoint(idx) - self.visualCenterInScene()
        return math.degrees(math.atan2(vector.y(), vector.x()))

    def beginResize(self, idx: int):
        self._resize_old_logical_rect = QRectF(
            self.blk_item.logical_unpadded_rect()
        )
        self._resize_opposite_idx = (idx + 4) % 8
        self._resize_opposite_scene = QPointF(
            self._item_handle_points_in_scene(self.blk_item)[
                self._resize_opposite_idx
            ]
        )

    def resizeFromScene(self, idx: int, scene_pos: QPointF):
        if self.blk_item is None or self._resize_opposite_scene is None:
            return

        item = self.blk_item
        current_local = item.logical_unpadded_rect()
        mouse_local = item.mapFromScene(scene_pos)
        left = current_local.left()
        right = current_local.right()
        top = current_local.top()
        bottom = current_local.bottom()
        minimum = 1.0

        if idx in (0, 6, 7):
            left = min(mouse_local.x(), right - minimum)
        elif idx in (2, 3, 4):
            right = max(mouse_local.x(), left + minimum)
        if idx in (0, 1, 2):
            top = min(mouse_local.y(), bottom - minimum)
        elif idx in (4, 5, 6):
            bottom = max(mouse_local.y(), top + minimum)

        new_local = QRectF(QPointF(left, top), QPointF(right, bottom))
        current_abs = item.absBoundingRect(qrect=True)
        new_abs = QRectF(
            current_abs.x() + new_local.x() - current_local.x(),
            current_abs.y() + new_local.y() - current_local.y(),
            new_local.width(),
            new_local.height(),
        )
        item.setRect(new_abs)

        moved_anchor = self._item_handle_points_in_scene(item)[
            self._resize_opposite_idx
        ]
        parent = item.parentItem()
        if parent is None:
            parent_delta = self._resize_opposite_scene - moved_anchor
        else:
            parent_delta = (
                parent.mapFromScene(self._resize_opposite_scene)
                - parent.mapFromScene(moved_anchor)
            )
        item.setPos(item.pos() + parent_delta)
        item.blk._bounding_rect = item.absBoundingRect()
        self.updateBoundingRect()

    def beginRotation(self, scene_pos: QPointF):
        rotate_vec = scene_pos - self.visualCenterInScene()
        pointer_angle = math.degrees(math.atan2(rotate_vec.y(), rotate_vec.x()))
        original_angle = self.blk_item.rotation()
        return original_angle - pointer_angle, original_angle

    def rotateFromScene(self, scene_pos: QPointF, rotate_start: float) -> float:
        rotate_vec = scene_pos - self.visualCenterInScene()
        pointer_angle = math.degrees(math.atan2(rotate_vec.y(), rotate_vec.x()))
        preview_angle = pointer_angle + rotate_start
        self.blk_item.setRotation(preview_angle)
        self.updateBoundingRect()
        return preview_angle

    def finishRotationPreview(self, original_angle: float) -> float:
        preview_angle = self.blk_item.rotation()
        # Keep the model angle as the command's sole owner during preview.
        self.blk_item.setRotation(original_angle)
        self.updateBoundingRect()
        return preview_angle

    def ctrlblockPressed(self):
        self.scene().clearSelection()
        if self.blk_item is not None:
            self.blk_item.endEdit()

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget=...,
    ) -> None:
        if len(self._visual_polygon) != 4:
            painter.setCompositionMode(QPainter.CompositionMode.RasterOp_NotDestination)
            return super().paint(painter, option, widget)
        painter.setCompositionMode(QPainter.CompositionMode.RasterOp_NotDestination)
        painter.setPen(self.pen())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(self._visual_polygon)

    def hideControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.hide()

    def showControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.show()

    def updateScale(self, scale: float):
        if not self.isVisible():
            if scale != self.current_scale:
                self.need_rescale = True
                self.current_scale = scale
            return

        self.current_scale = scale
        pen = self.pen()
        pen.setWidthF(2 / max(abs(scale), 1e-9))
        self.setPen(pen)
        # The handles ignore scene/view transforms, so their device size is stable.
        for ctrl in self.ctrlblock_group:
            ctrl.updateEdgeWidth(CBEDGE_WIDTH)

    def show(self) -> None:
        super().show()
        if self.need_rescale:
            self.updateScale(self.current_scale)
            self.need_rescale = False
        self.setZValue(1)

    def startEditing(self):
        self.setCursor(Qt.CursorShape.IBeamCursor)
        for ctrlb in self.ctrlblock_group:
            ctrlb.hide()

    def endEditing(self):
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        if self.isVisible():
            for ctrlb in self.ctrlblock_group:
                ctrlb.show()
