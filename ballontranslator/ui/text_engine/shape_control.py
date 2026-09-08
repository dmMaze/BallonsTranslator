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
from qtpy.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)

from ..cursor import (
    resizeCursorList,
    resize_handle_scene_angle,
    rotateCursorList,
    scene_angle_to_cursor_index,
)
from .item import TextBlkItem
from .transforms.mapping import rect_polygon


CBEDGE_WIDTH = 30
VISUALIZE_HITBOX = False
PROXY_HANDLE_VIEWPORT_INSET = 12.0
CONTROL_DEVICE_GUARD = 2.0
CONTROL_ITEM_DATA_KEY = 0x1238


def device_pixels_to_local(item: QGraphicsItem, pixels: float) -> float:
    """Return a conservative item-local radius for a device-pixel radius.

    >>> device_pixels_to_local(QGraphicsRectItem(), 2.0)
    2.0
    """
    radii = [float(pixels)]
    scene = item.scene()
    if scene is None:
        return radii[0]
    for view in scene.views():
        inverse, invertible = item.deviceTransform(
            view.viewportTransform()
        ).inverted()
        if not invertible:
            continue
        origin = inverse.map(QPointF())
        for x, y in (
            (pixels, 0.0),
            (0.0, pixels),
            (pixels, pixels),
            (pixels, -pixels),
        ):
            delta = inverse.map(QPointF(x, y)) - origin
            radii.append(max(abs(delta.x()), abs(delta.y())))
    return max(radii)


class ControlBlockItem(QGraphicsRectItem):
    """Fixed-device-size resize/rotation handle for the shape overlay.

    >>> ControlBlockItem.DRAG_NONE
    0
    """

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
        self._outward_device = QPointF()
        self._attached_to_item = False
        self._device_angle = 0.0
        self.setAcceptHoverEvents(True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        # QGraphicsRectItem otherwise expands shape() for its default pen,
        # making a boundary-attached exterior hitbox overlap the text item.
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setData(CONTROL_ITEM_DATA_KEY, True)
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.updateEdgeWidth(CBEDGE_WIDTH)

    def updateEdgeWidth(self, edge_width: float):
        self.edge_width = edge_width
        self.pen_width = edge_width / CBEDGE_WIDTH * 2
        self.setRect(-edge_width / 2, -edge_width / 2, edge_width, edge_width)
        self._updateVisibleRect()

    def setOutwardDeviceVector(
        self, outward: QPointF, attached_to_item: bool
    ) -> None:
        self._outward_device = QPointF(outward)
        self._attached_to_item = bool(attached_to_item)
        self._updateVisibleRect()

    def setDeviceAngle(self, angle: float) -> None:
        self._device_angle = float(angle)
        self.setRotation(self._device_angle)
        self._updateVisibleRect()

    def _outwardInLocal(self, outward: QPointF) -> QPointF:
        radians = math.radians(self._device_angle)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        return QPointF(
            cosine * outward.x() + sine * outward.y(),
            -sine * outward.x() + cosine * outward.y(),
        )

    def supportRadius(self, outward: QPointF) -> float:
        local = self._outwardInLocal(outward)
        return self.edge_width / 2 * (
            abs(local.x()) + abs(local.y())
        )

    def _updateVisibleRect(self) -> None:
        visible_len = self.edge_width / 2
        center = QPointF()
        if self._attached_to_item:
            # The hitbox center is shifted by its support radius. Shift the
            # smaller painted block back by the radius difference so its inner
            # edge/corner touches the item's border, matching the old control.
            outward_local = self._outwardInLocal(self._outward_device)
            support = abs(outward_local.x()) + abs(outward_local.y())
            center = outward_local * (
                -(self.edge_width - visible_len) / 2 * support
            )
        visible_rect = QRectF(
            center.x() - visible_len / 2,
            center.y() - visible_len / 2,
            visible_len,
            visible_len,
        )
        if getattr(self, 'visible_rect', None) == visible_rect:
            return
        # The painted block is shifted inside its larger interaction hitbox.
        # Notify Qt when that manual paint extent changes so old pixels are
        # included in the viewport's dirty region during movement.
        self.prepareGeometryChange()
        self.visible_rect = visible_rect
        self.update()

    def boundingRect(self) -> QRectF:
        bounds = QRectF(super().boundingRect())
        visible_rect = getattr(self, 'visible_rect', QRectF())
        if visible_rect.isNull():
            return bounds
        # paint() supplies its own pen because the item's real pen is NoPen.
        guard = getattr(self, 'pen_width', 0.0) / 2.0 + 1.0
        return bounds.united(
            visible_rect.adjusted(-guard, -guard, guard, guard)
        )

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
        if self.ctrl.blk_item is None:
            return super().hoverMoveEvent(event)
        if self.visible_rect.contains(event.pos()):
            angle_idx = scene_angle_to_cursor_index(
                self.ctrl.resizeHandleSceneAngle(self.idx)
            )
            self.setCursor(resizeCursorList[angle_idx % 4])
        else:
            angle_idx = scene_angle_to_cursor_index(
                self.ctrl.handleSceneAngle(self.idx)
            )
            self.setCursor(rotateCursorList[angle_idx])
        self.CURSOR_IDX = angle_idx
        return super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if self.drag_mode == self.DRAG_NONE:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        return super().hoverLeaveEvent(event)

    def resetInteraction(self) -> None:
        scene = self.scene()
        if scene is not None and scene.mouseGrabberItem() is self:
            self.ungrabMouse()
        self.drag_mode = self.DRAG_NONE
        self.rotate_start = 0.0
        self.rotate_original = 0.0
        self.CURSOR_IDX = -1
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.ctrl.blk_item is not None:
            blk_item = self.ctrl.blk_item
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Control-click is selection input even when the fixed-size
                # handle happens to be the topmost scene item.
                blk_item.setSelected(not blk_item.isSelected())
                event.accept()
                return
            self.ctrl.ctrlblockPressed()
            blk_item.setSelected(True)
            if self.visible_rect.contains(event.pos()):
                self.ctrl.reshaping = True
                self.drag_mode = self.DRAG_RESHAPE
                self.ctrl.beginResize(self.idx, event.scenePos())
                blk_item.startReshape()
            else:
                self.drag_mode = self.DRAG_ROTATE
                self.rotate_start, self.rotate_original = self.ctrl.beginRotation(
                    event.scenePos(), self.idx
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
            self.ctrl.rotateFromScene(event.scenePos(), self.rotate_start, self.idx)
            angle_idx = scene_angle_to_cursor_index(
                self.ctrl.handleSceneAngle(self.idx)
            )
            if self.CURSOR_IDX != angle_idx:
                self.setCursor(rotateCursorList[angle_idx])
                self.CURSOR_IDX = angle_idx
            self.updateAngleLabelPos()
        event.accept()

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
            self.ctrl.finishResize()
            self.ctrl.finishProxyDrag()
        return super().mouseReleaseEvent(event)


class TextBlkShapeControl(QGraphicsRectItem):
    """Render and manipulate the active text item's visual geometry.

    >>> vector = TextBlkShapeControl._normalized_device_vector(3.0, 4.0)
    >>> (round(vector.x(), 1), round(vector.y(), 1))
    (0.6, 0.8)
    """

    blk_item: TextBlkItem = None
    ctrl_block: ControlBlockItem = None
    reshaping: bool = False

    def __init__(self, parent) -> None:
        super().__init__()
        self.gv = parent
        self._visual_path = QPainterPath()
        self._outline_bounds = QRectF()
        self._true_handle_scene_points = []
        self._display_handle_scene_points = []
        self._reported_angle = 0.0
        self._updating_bounds = False
        self._resize_opposite_scene = None
        self._resize_opposite_idx = None
        self._resize_initial_local = None
        self._resize_initial_abs = None
        self._resize_initial_source_handle = None
        self._resize_previous_source = None
        self._resize_scene_to_source = None
        self._proxy_drag_idx = None
        self._proxy_pointer_device_start = None
        self._proxy_actual_scene_start = None
        self.ctrlblock_group = [ControlBlockItem(self, idx) for idx in range(8)]

        pen = QPen(QColor(69, 71, 87), 2, Qt.PenStyle.SolidLine)
        pen.setDashPattern([7, 14])
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setData(CONTROL_ITEM_DATA_KEY, True)
        self.setVisible(False)
        self.setZValue(1)

        self.angleLabel = QLabel(parent)
        self.angleLabel.setText("{:.1f}\N{DEGREE SIGN}".format(0.0))
        self.angleLabel.setObjectName("angleLabel")
        self.angleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angleLabel.setHidden(True)

        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def requestGeometryRefresh(self) -> None:
        if not self.isVisible():
            return
        if self.blk_item is not None:
            self.updateBoundingRect()
        else:
            self.refreshDeviceGeometry()
            if self.isVisible():
                self.updateControlBlocks()

    def boundingRect(self) -> QRectF:
        outline_bounds = getattr(self, '_outline_bounds', QRectF())
        if not outline_bounds.isNull():
            return QRectF(outline_bounds)
        return super().boundingRect()

    def refreshDeviceGeometry(self) -> bool:
        geometry_bounds = (
            self._visual_path.boundingRect()
            if not self._visual_path.isEmpty()
            else super().rect()
        )
        if geometry_bounds.isNull():
            return False
        guard = device_pixels_to_local(
            self, self.pen().widthF() / 2.0 + CONTROL_DEVICE_GUARD
        )
        bounds = geometry_bounds.adjusted(
            -guard, -guard, guard, guard
        )
        if bounds == self._outline_bounds:
            return False
        self.prepareGeometryChange()
        self._outline_bounds = bounds
        self.update()
        return True

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
        return item.geometry_controller.visual_handle_points_in_scene()

    def setBlkItem(self, blk_item: TextBlkItem):
        if (
            blk_item is not None
            and self.blk_item == blk_item
            and self.isVisible()
        ):
            return
        if self.blk_item is not None:
            try:
                self.blk_item.visual_geometry_changed.disconnect(
                    self._on_item_geometry_changed
                )
                self.blk_item.moving.disconnect(
                    self._on_item_geometry_changed
                )
            except (RuntimeError, TypeError):
                pass
            self.blk_item.under_ctrl = False
            if self.blk_item.isEditing():
                self.blk_item.endEdit()
            self.blk_item.update()

        self.blk_item = blk_item
        if blk_item is None:
            self.resetInteraction()
            self._visual_path = QPainterPath()
            self._outline_bounds = QRectF()
            self._true_handle_scene_points = []
            self._display_handle_scene_points = []
            self._reported_angle = 0.0
            self.hide()
            self.requestGeometryRefresh()
            return
        self.resetInteraction()
        blk_item.under_ctrl = True
        blk_item.visual_geometry_changed.connect(
            self._on_item_geometry_changed
        )
        blk_item.moving.connect(self._on_item_geometry_changed)
        blk_item.update()
        self.show()
        self.requestGeometryRefresh()

    def _on_item_geometry_changed(self, *_args: object) -> None:
        self.requestGeometryRefresh()

    def updateBoundingRect(self):
        if self.blk_item is None:
            return

        scene_path = self.blk_item.geometry_controller.visual_outline_in_scene()
        parent = self.parentItem()
        if parent is None:
            parent_path = QPainterPath(scene_path)
        else:
            inverse, invertible = parent.sceneTransform().inverted()
            parent_path = (
                inverse.map(scene_path) if invertible else QPainterPath()
            )
        bounds = parent_path.boundingRect()
        origin = bounds.topLeft()
        local_path = QPainterPath(parent_path)
        local_path.translate(-origin)
        local_bounds = local_path.boundingRect()
        guard = device_pixels_to_local(
            self, self.pen().widthF() / 2.0 + CONTROL_DEVICE_GUARD
        )
        outline_bounds = local_bounds.adjusted(-guard, -guard, guard, guard)
        if (
            self._visual_path == local_path
            and self.pos() == origin
            and self.rect() == local_bounds
            and self._outline_bounds == outline_bounds
            and self._reported_angle == self.blk_item.rotation()
        ):
            self.updateControlBlocks()
            return False

        self._updating_bounds = True
        try:
            self.prepareGeometryChange()
            self._visual_path = local_path
            self._outline_bounds = outline_bounds
            self._reported_angle = self.blk_item.rotation()
            super().setTransform(QTransform(), False)
            super().setRotation(0.0)
            super().setPos(origin)
            super().setRect(local_bounds)
            self.updateControlBlocks()
            self.update()
        finally:
            self._updating_bounds = False
        return True

    def shape(self) -> QPainterPath:
        if self._visual_path.isEmpty():
            return super().shape()
        return QPainterPath(self._visual_path)

    def setRect(self, *args):
        if self.blk_item is not None and not self._updating_bounds:
            self.updateBoundingRect()
            return
        super().setRect(*args)
        self._visual_path = QPainterPath()
        rect = super().rect()
        guard = device_pixels_to_local(
            self, self.pen().widthF() / 2.0 + CONTROL_DEVICE_GUARD
        )
        self.prepareGeometryChange()
        self._outline_bounds = rect.adjusted(-guard, -guard, guard, guard)
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
        if self.blk_item is not None:
            true_scene_points = self._item_handle_points_in_scene(self.blk_item)
        else:
            polygon = rect_polygon(self.rect())
            true_scene_points = [
                self.mapToScene(point) for point in self._handle_points(polygon)
            ]
        self._true_handle_scene_points = [
            QPointF(point) for point in true_scene_points
        ]
        local_frames = (
            self.blk_item is not None
            and self.blk_item.geometry_controller
            .compiled.needs_local_handle_frames
        )
        if local_frames:
            outward_vectors, device_angles = self._handle_frames_device(
                self._true_handle_scene_points,
                self.blk_item.geometry_controller
                .visual_handle_tangents_in_scene(),
            )
        else:
            outward_vectors = self._handle_outward_vectors_device(
                self._true_handle_scene_points
            )
            device_angles = [
                self._item_device_angle()
                for _ in self._true_handle_scene_points
            ]
        for ctrlblock, device_angle in zip(
            self.ctrlblock_group, device_angles
        ):
            ctrlblock.setDeviceAngle(device_angle)
        handle_placements = [
            self._outward_handle_scene_point(
                point,
                outward,
                ctrlblock.supportRadius(outward),
            )
            for point, outward, ctrlblock in zip(
                self._true_handle_scene_points,
                outward_vectors,
                self.ctrlblock_group,
            )
        ]
        self._display_handle_scene_points = [
            point for point, _attached in handle_placements
        ]
        for ctrlblock, point in zip(
            self.ctrlblock_group, self._display_handle_scene_points
        ):
            ctrlblock.setPos(self.mapFromScene(point))
        for ctrlblock, outward, (_point, attached) in zip(
            self.ctrlblock_group,
            outward_vectors,
            handle_placements,
        ):
            ctrlblock.setOutwardDeviceVector(outward, attached)

    @staticmethod
    def _normalized_device_vector(x: float, y: float) -> QPointF:
        length = math.hypot(x, y)
        if length == 0.0:
            return QPointF()
        return QPointF(x / length, y / length)

    def _handle_outward_vectors_device(self, scene_points):
        """Return supporting outward normals for corner/edge handles."""
        if len(scene_points) != 8:
            return [QPointF() for _ in scene_points]
        transform = self.gv.viewportTransform()
        corners = [transform.map(scene_points[index]) for index in range(0, 8, 2)]
        signed_area = sum(
            point.x() * corners[(index + 1) % 4].y()
            - point.y() * corners[(index + 1) % 4].x()
            for index, point in enumerate(corners)
        )
        orientation = 1.0 if signed_area >= 0.0 else -1.0
        edge_normals = []
        for index, point in enumerate(corners):
            edge = corners[(index + 1) % 4] - point
            edge_normals.append(
                self._normalized_device_vector(
                    orientation * edge.y(),
                    -orientation * edge.x(),
                )
            )

        center = sum(corners, QPointF()) / 4
        vectors = []
        for index in range(4):
            corner_normal = (
                edge_normals[(index - 1) % 4] + edge_normals[index]
            )
            corner_normal = self._normalized_device_vector(
                corner_normal.x(), corner_normal.y()
            )
            if corner_normal.isNull():
                radial = corners[index] - center
                corner_normal = self._normalized_device_vector(
                    radial.x(), radial.y()
                )
            vectors.extend((corner_normal, edge_normals[index]))
        return vectors

    def _handle_frames_device(self, scene_points, scene_tangents=None):
        """Return local outward normals and tangent angles for eight handles."""
        if len(scene_points) != 8:
            return (
                [QPointF() for _ in scene_points],
                [self._item_device_angle() for _ in scene_points],
            )
        transform = self.gv.viewportTransform()
        points = [transform.map(point) for point in scene_points]
        if scene_tangents is None or len(scene_tangents) != len(points):
            device_tangents = None
        else:
            device_origin = transform.map(QPointF())
            device_tangents = [
                transform.map(QPointF(tangent)) - device_origin
                for tangent in scene_tangents
            ]
        signed_area = sum(
            point.x() * points[(index + 1) % len(points)].y()
            - point.y() * points[(index + 1) % len(points)].x()
            for index, point in enumerate(points)
        )
        orientation = 1.0 if signed_area >= 0.0 else -1.0
        outward_vectors = []
        angles = []
        for index in range(len(points)):
            boundary_tangent = (
                points[(index + 1) % 8] - points[(index - 1) % 8]
            )
            boundary_tangent = self._normalized_device_vector(
                boundary_tangent.x(), boundary_tangent.y()
            )
            outward = QPointF(
                orientation * boundary_tangent.y(),
                -orientation * boundary_tangent.x(),
            )
            outward_vectors.append(outward)
            tangent = (
                boundary_tangent
                if device_tangents is None
                else device_tangents[index]
            )
            angles.append(math.degrees(math.atan2(tangent.y(), tangent.x())))
        return outward_vectors, angles

    def _item_device_angle(self) -> float:
        if self.blk_item is None:
            return 0.0
        transform = self.blk_item.deviceTransform(
            self.gv.viewportTransform()
        )
        origin = transform.map(QPointF())
        local_x = transform.map(QPointF(1.0, 0.0)) - origin
        if local_x.isNull():
            return self.blk_item.rotation()
        return math.degrees(math.atan2(local_x.y(), local_x.x()))

    def _outward_handle_scene_point(
        self,
        scene_point: QPointF,
        outward: QPointF,
        support_radius: float,
    ):
        """Place a device-sized hitbox outside and against its support line."""
        view = self.gv
        viewport = view.viewport()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return QPointF(scene_point), False
        transform = view.viewportTransform()
        inverse, invertible = transform.inverted()
        if not invertible:
            return QPointF(scene_point), False
        device = transform.map(scene_point)
        display = device + outward * support_radius
        attached = QRectF(viewport.rect()).contains(device)
        if not attached:
            inset = PROXY_HANDLE_VIEWPORT_INSET + support_radius
            left = inset
            top = inset
            right = max(left, viewport.width() - inset)
            bottom = max(top, viewport.height() - inset)
            display = QPointF(
                min(max(display.x(), left), right),
                min(max(display.y(), top), bottom),
            )
        return inverse.map(display), attached

    def setAngle(self, angle: float) -> None:
        self.setRotation(angle)

    def visualCenterInScene(self) -> QPointF:
        return self.blk_item.geometry_controller.visual_rotation_center_in_scene()

    def handleScenePoint(self, idx: int) -> QPointF:
        if len(self._true_handle_scene_points) == 8:
            return QPointF(self._true_handle_scene_points[idx])
        if self.blk_item is not None:
            points = self._item_handle_points_in_scene(self.blk_item)
            if len(points) == 8:
                return QPointF(points[idx])
        return self.ctrlblock_group[idx].scenePos()

    def handleSceneAngle(self, idx: int) -> float:
        vector = self.handleScenePoint(idx) - self.visualCenterInScene()
        return math.degrees(math.atan2(vector.y(), vector.x()))

    def resizeHandleSceneAngle(self, idx: int) -> float:
        if (
            not self.blk_item.geometry_controller
            .compiled.needs_local_handle_frames
        ):
            points = self._item_handle_points_in_scene(self.blk_item)
            if len(points) == 8:
                return resize_handle_scene_angle(
                    points[2] - points[0], idx
                )
        points = self._item_handle_points_in_scene(self.blk_item)
        if len(points) != 8:
            return self.rotation() + 45.0 * idx - 135.0
        tangent = points[(idx + 1) % 8] - points[(idx - 1) % 8]
        signed_area = sum(
            point.x() * points[(index + 1) % len(points)].y()
            - point.y() * points[(index + 1) % len(points)].x()
            for index, point in enumerate(points)
        )
        orientation = 1.0 if signed_area >= 0.0 else -1.0
        outward = QPointF(
            orientation * tangent.y(),
            -orientation * tangent.x(),
        )
        return math.degrees(math.atan2(outward.y(), outward.x()))

    def _beginProxyDrag(self, idx: int, pointer_scene: QPointF):
        self._proxy_drag_idx = idx
        transform = self.gv.viewportTransform()
        self._proxy_pointer_device_start = transform.map(pointer_scene)
        self._proxy_actual_scene_start = QPointF(self.handleScenePoint(idx))

    def _proxySceneTarget(self, idx: int, pointer_scene: QPointF) -> QPointF:
        if (
            self._proxy_drag_idx != idx
            or self._proxy_pointer_device_start is None
            or self._proxy_actual_scene_start is None
        ):
            return QPointF(pointer_scene)
        inverse, invertible = self.gv.viewportTransform().inverted()
        if not invertible:
            return QPointF(pointer_scene)
        current_device = self.gv.viewportTransform().map(pointer_scene)
        scene_start = inverse.map(self._proxy_pointer_device_start)
        scene_current = inverse.map(current_device)
        return self._proxy_actual_scene_start + scene_current - scene_start

    def finishProxyDrag(self):
        self._proxy_drag_idx = None
        self._proxy_pointer_device_start = None
        self._proxy_actual_scene_start = None

    def resetInteraction(self) -> None:
        """Clear transient state before the control crosses an item/page boundary."""
        self.reshaping = False
        self.finishResize()
        self.finishProxyDrag()
        self.angleLabel.hide()
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        for ctrlblock in self.ctrlblock_group:
            ctrlblock.resetInteraction()
            ctrlblock.show()

    def beginResize(self, idx: int, pointer_scene: QPointF = None):
        if pointer_scene is not None:
            self._beginProxyDrag(idx, pointer_scene)
        item = self.blk_item
        self._resize_opposite_idx = (idx + 4) % 8
        self._resize_opposite_scene = QPointF(
            self._item_handle_points_in_scene(item)[
                self._resize_opposite_idx
            ]
        )
        self._resize_initial_local = QRectF(item.logical_unpadded_rect())
        self._resize_initial_abs = QRectF(item.absBoundingRect(qrect=True))
        self._resize_initial_source_handle = QPointF(
            item.geometry_controller.source_handle_points()[idx]
        )
        self._resize_previous_source = QPointF(
            self._resize_initial_source_handle
        )
        self._resize_scene_to_source = (
            item.geometry_controller.capture_scene_to_source_mapper()
        )

    def finishResize(self) -> None:
        self._resize_opposite_scene = None
        self._resize_opposite_idx = None
        self._resize_initial_local = None
        self._resize_initial_abs = None
        self._resize_initial_source_handle = None
        self._resize_previous_source = None
        self._resize_scene_to_source = None

    def resizeFromScene(self, idx: int, scene_pos: QPointF):
        if (
            self.blk_item is None
            or self._resize_opposite_scene is None
            or self._resize_initial_local is None
            or self._resize_initial_abs is None
            or self._resize_initial_source_handle is None
            or self._resize_previous_source is None
            or self._resize_scene_to_source is None
        ):
            return

        item = self.blk_item
        scene_pos = self._proxySceneTarget(idx, scene_pos)
        initial_local = self._resize_initial_local
        mouse_local = self._resize_scene_to_source(
            scene_pos,
            self._resize_previous_source,
        )
        self._resize_previous_source = QPointF(mouse_local)
        left = initial_local.left()
        right = initial_local.right()
        top = initial_local.top()
        bottom = initial_local.bottom()
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
        initial_abs = self._resize_initial_abs
        new_abs = QRectF(
            initial_abs.x() + new_local.x() - initial_local.x(),
            initial_abs.y() + new_local.y() - initial_local.y(),
            new_local.width(),
            new_local.height(),
        )
        # Publish only the settled geometry after restoring the visual anchor.
        item.setRect(new_abs, notify=False)

        # Resizing changes nonlinear geometry, so restore the untouched visual
        # handle to the scene position captured at drag start.
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
        item.visual_geometry_changed.emit()

    def beginRotation(self, scene_pos: QPointF, idx: int = None):
        if idx is not None:
            self._beginProxyDrag(idx, scene_pos)
            scene_pos = self._proxySceneTarget(idx, scene_pos)
        rotate_vec = scene_pos - self.visualCenterInScene()
        pointer_angle = math.degrees(math.atan2(rotate_vec.y(), rotate_vec.x()))
        original_angle = self.blk_item.rotation()
        return original_angle - pointer_angle, original_angle

    def rotateFromScene(
        self, scene_pos: QPointF, rotate_start: float, idx: int = None
    ) -> float:
        if idx is not None:
            scene_pos = self._proxySceneTarget(idx, scene_pos)
        rotate_vec = scene_pos - self.visualCenterInScene()
        pointer_angle = math.degrees(math.atan2(rotate_vec.y(), rotate_vec.x()))
        preview_angle = pointer_angle + rotate_start
        self.blk_item.setRotation(preview_angle)
        return preview_angle

    def finishRotationPreview(self, original_angle: float) -> float:
        preview_angle = self.blk_item.rotation()
        # Keep the model angle as the command's sole owner during preview.
        self.blk_item.setRotation(original_angle)
        self.finishProxyDrag()
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
        painter.save()
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        path = self._visual_path
        painter.setPen(self.pen())
        if path.isEmpty():
            painter.drawRect(self.rect())
        else:
            painter.drawPath(path)
        painter.restore()

    def hideControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.hide()
        self.requestGeometryRefresh()

    def showControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.show()
        self.requestGeometryRefresh()

    def startEditing(self):
        self.setCursor(Qt.CursorShape.IBeamCursor)
        for ctrlb in self.ctrlblock_group:
            ctrlb.hide()
        self.requestGeometryRefresh()

    def endEditing(self):
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        if self.isVisible():
            for ctrlb in self.ctrlblock_group:
                ctrlb.show()
        self.requestGeometryRefresh()
