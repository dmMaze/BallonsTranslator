import math
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Optional, Set

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
    QRegion,
    QTransform,
)

from .cursor import (
    resizeCursorList,
    resize_handle_scene_angle,
    rotateCursorList,
    scene_angle_to_cursor_index,
)
from .textitem import TEXTRECT_SELECTED_COLOR, TEXTRECT_SHOW_COLOR, TextBlkItem


CBEDGE_WIDTH = 30
VISUALIZE_HITBOX = False
UI_OVERLAY_ITEM_DATA_KEY = 0x1238
OVERLAY_DEVICE_GUARD = 2.0
PROXY_HANDLE_VIEWPORT_INSET = 12.0


def _device_pixels_to_local(item: QGraphicsItem, pixels: float) -> float:
    """Return a conservative item-local radius for a device-pixel radius.

    >>> _device_pixels_to_local(QGraphicsRectItem(), 2.0)
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


def _is_effectively_visible(item: QGraphicsItem) -> bool:
    current = item
    while current is not None:
        if not current.isVisible():
            return False
        current = current.parentItem()
    return True


class TextGuideOverlayItem(QGraphicsItem):
    """Paint one reusable, input-transparent text guide in scene overlay state."""

    def __init__(self, parent: QGraphicsItem):
        super().__init__(parent)
        self._polygon = QPolygonF()
        self._selected = False
        self._bounds = QRectF()
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setData(UI_OVERLAY_ITEM_DATA_KEY, True)
        self.setZValue(10.0)
        self.hide()

    @property
    def polygon(self) -> QPolygonF:
        return QPolygonF(self._polygon)

    def setGuide(self, polygon: QPolygonF, selected: bool) -> bool:
        polygon = QPolygonF(polygon)
        pen_width = 3.5 if selected else 3.0
        guard = _device_pixels_to_local(
            self, pen_width / 2.0 + OVERLAY_DEVICE_GUARD
        )
        bounds = polygon.boundingRect().adjusted(-guard, -guard, guard, guard)
        if (
            self._polygon == polygon
            and self._selected == selected
            and self._bounds == bounds
        ):
            return False
        self.prepareGeometryChange()
        self._polygon = polygon
        self._selected = selected
        self._bounds = bounds
        self.update()
        return True

    def boundingRect(self) -> QRectF:
        return QRectF(self._bounds)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        if not self._polygon.isEmpty():
            path.addPolygon(self._polygon)
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._polygon.isEmpty():
            return
        painter.save()
        try:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            pen = QPen(
                TEXTRECT_SELECTED_COLOR if self._selected else TEXTRECT_SHOW_COLOR,
                3.5 if self._selected else 3.0,
                Qt.PenStyle.DashLine if self._selected else Qt.PenStyle.SolidLine,
            )
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawPolygon(self._polygon)
        finally:
            painter.restore()


class OverlayFootprintInvalidator:
    """Invalidate old and new device-space overlay footprints for every view."""

    def __init__(
        self,
        scene,
        overlay_items: Callable[[], Iterable[QGraphicsItem]],
    ) -> None:
        self.scene = scene
        self._overlay_items = overlay_items
        self._regions: Dict[object, QRegion] = {}

    @staticmethod
    def _item_device_region(item: QGraphicsItem, view) -> QRegion:
        if not _is_effectively_visible(item):
            return QRegion()
        transform = item.deviceTransform(view.viewportTransform())
        rect = transform.mapRect(item.boundingRect()).toAlignedRect()
        rect.adjust(
            -int(OVERLAY_DEVICE_GUARD),
            -int(OVERLAY_DEVICE_GUARD),
            int(OVERLAY_DEVICE_GUARD),
            int(OVERLAY_DEVICE_GUARD),
        )
        return QRegion(rect).intersected(QRegion(view.viewport().rect()))

    def _region_for_view(self, view) -> QRegion:
        region = QRegion()
        for item in self._overlay_items():
            if item.scene() is self.scene:
                region |= self._item_device_region(item, view)
        return region

    def capture_current_regions(self, refresh_existing: bool = True) -> None:
        """Capture pre-mutation footprints using each view's current transform."""
        for view in tuple(self.scene.views()):
            if refresh_existing or view not in self._regions:
                self._regions[view] = self._region_for_view(view)

    def sync(self, update_geometry: Callable[[], None]) -> None:
        views = tuple(self.scene.views())
        old_regions = {
            view: QRegion(self._regions.get(view, self._region_for_view(view)))
            for view in views
        }
        update_geometry()
        new_regions = {view: self._region_for_view(view) for view in views}
        self._regions = new_regions
        for view in views:
            dirty = old_regions[view] | new_regions[view]
            if not dirty.isEmpty():
                view.viewport().update(dirty)


class TextOverlayManager:
    """Own reusable guide overlays and their device-space invalidation."""

    def __init__(self, scene, parent: QGraphicsItem, shape_control) -> None:
        self.scene = scene
        self.parent = parent
        self.shape_control = shape_control
        self._items: Set[TextBlkItem] = set()
        self._guides: Dict[TextBlkItem, TextGuideOverlayItem] = {}
        self._pool = []
        self._textblock_mode = False
        self._syncing = False
        self._sync_pending = False
        self._batch_depth = 0
        self._batch_pending = False
        self.invalidator = OverlayFootprintInvalidator(
            scene, self._visible_overlay_items
        )

    @contextmanager
    def batch_update(self):
        """Coalesce a group of overlay mutations into one old/new sync."""
        if self._batch_depth == 0:
            self.invalidator.capture_current_regions(refresh_existing=True)
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._batch_pending:
                self._batch_pending = False
                self.sync_overlays()

    def _visible_overlay_items(self):
        for overlay in self._guides.values():
            if overlay.isVisible():
                yield overlay
        control = self.shape_control
        if control.isVisible():
            yield control
            for handle in control.ctrlblock_group:
                if handle.isVisible():
                    yield handle

    def _acquire_overlay(self, item: TextBlkItem) -> TextGuideOverlayItem:
        overlay = self._guides.get(item)
        if overlay is not None:
            return overlay
        if self._pool:
            overlay = self._pool.pop()
        else:
            overlay = TextGuideOverlayItem(self.parent)
        self._guides[item] = overlay
        return overlay

    def _release_overlay(self, item: TextBlkItem) -> None:
        overlay = self._guides.pop(item, None)
        if overlay is None:
            return
        overlay.hide()
        overlay.setGuide(QPolygonF(), False)
        self._pool.append(overlay)

    def register_item(self, item: TextBlkItem) -> None:
        self._items.add(item)
        self.sync_overlays()

    def unregister_item(self, item: TextBlkItem) -> None:
        if item not in self._items and item not in self._guides:
            return
        self.invalidator.capture_current_regions(
            refresh_existing=self._batch_depth == 0
        )
        self._items.discard(item)
        self._release_overlay(item)
        self.sync_overlays()

    def clear(self) -> None:
        self.invalidator.capture_current_regions(
            refresh_existing=self._batch_depth == 0
        )
        self._items.clear()
        for item in tuple(self._guides):
            self._release_overlay(item)
        self.sync_overlays()

    def set_textblock_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._textblock_mode == enabled:
            self.sync_overlays()
            return
        self._textblock_mode = enabled
        self.sync_overlays()

    def overlay_for_item(self, item: TextBlkItem):
        return self._guides.get(item)

    def _item_parent_polygon(self, item: TextBlkItem) -> QPolygonF:
        return QPolygonF(
            [self.parent.mapFromScene(point) for point in item.visual_polygon_in_scene()]
        )

    def _update_geometry(self) -> None:
        control = self.shape_control
        if control.blk_item is not None and control.blk_item.scene() is self.scene:
            control.updateBoundingRect()
        control.refreshDeviceGeometry()
        if control.isVisible():
            control.updateControlBlocks()
        active_item = control.blk_item if control.isVisible() else None

        for item in tuple(self._items):
            if item.scene() is not self.scene or not _is_effectively_visible(item):
                overlay = self._guides.get(item)
                if overlay is not None:
                    overlay.hide()
                continue
            selected = item.isSelected()
            should_show = item is not active_item and (
                selected or self._textblock_mode
            )
            if not should_show:
                overlay = self._guides.get(item)
                if overlay is not None:
                    overlay.hide()
                continue
            overlay = self._acquire_overlay(item)
            overlay.setGuide(self._item_parent_polygon(item), selected)
            overlay.show()

    def sync_overlays(self, *_args, **_kwargs) -> None:
        if self._batch_depth:
            self._batch_pending = True
            return
        if self._syncing:
            self._sync_pending = True
            return
        self._syncing = True
        try:
            while True:
                self._sync_pending = False
                self.invalidator.sync(self._update_geometry)
                if not self._sync_pending:
                    break
        finally:
            self._syncing = False


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
        self.setData(UI_OVERLAY_ITEM_DATA_KEY, True)
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
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

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.ctrl.ctrlblockPressed()
        if event.button() == Qt.MouseButton.LeftButton and self.ctrl.blk_item is not None:
            blk_item = self.ctrl.blk_item
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
            self.ctrl.finishProxyDrag()
        return super().mouseReleaseEvent(event)


class TextBlkShapeControl(QGraphicsRectItem):
    blk_item: TextBlkItem = None
    ctrl_block: ControlBlockItem = None
    reshaping: bool = False

    def __init__(self, parent) -> None:
        super().__init__()
        self.gv = parent
        self._visual_polygon = QPolygonF()
        self._outline_bounds = QRectF()
        self._true_handle_scene_points = []
        self._display_handle_scene_points = []
        self._reported_angle = 0.0
        self._updating_bounds = False
        self._resize_old_logical_rect = None
        self._resize_opposite_scene = None
        self._resize_opposite_idx = None
        self._proxy_drag_idx = None
        self._proxy_pointer_device_start = None
        self._proxy_actual_scene_start = None
        self.overlay_sync_callback: Optional[Callable[[], None]] = None
        self.ctrlblock_group = [ControlBlockItem(self, idx) for idx in range(8)]

        pen = QPen(TEXTRECT_SELECTED_COLOR, 3.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setData(UI_OVERLAY_ITEM_DATA_KEY, True)
        self.setVisible(False)

        self.angleLabel = QLabel(parent)
        self.angleLabel.setText("{:.1f}\N{DEGREE SIGN}".format(0.0))
        self.angleLabel.setObjectName("angleLabel")
        self.angleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angleLabel.setHidden(True)

        self.current_scale = 1.0
        self.need_rescale = False
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def requestOverlaySync(self):
        if self.overlay_sync_callback is not None:
            self.overlay_sync_callback()
        elif self.blk_item is not None:
            # Standalone/tests may construct the control without a manager.
            # Production installs the callback and keeps invalidation ownership
            # in TextOverlayManager.
            self.updateBoundingRect()

    def boundingRect(self) -> QRectF:
        outline_bounds = getattr(self, '_outline_bounds', QRectF())
        if not outline_bounds.isNull():
            return QRectF(outline_bounds)
        return super().boundingRect()

    def refreshDeviceGeometry(self) -> bool:
        geometry_bounds = (
            self._visual_polygon.boundingRect()
            if not self._visual_polygon.isEmpty()
            else super().rect()
        )
        if geometry_bounds.isNull():
            return False
        guard = _device_pixels_to_local(
            self, self.pen().widthF() / 2.0 + OVERLAY_DEVICE_GUARD
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
            self._outline_bounds = QRectF()
            self._true_handle_scene_points = []
            self._display_handle_scene_points = []
            self._reported_angle = 0.0
            self.hide()
            self.requestOverlaySync()
            return
        blk_item.under_ctrl = True
        blk_item.update()
        self.show()
        self.requestOverlaySync()

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
        local_bounds = local_polygon.boundingRect()
        guard = _device_pixels_to_local(
            self, self.pen().widthF() / 2.0 + OVERLAY_DEVICE_GUARD
        )
        outline_bounds = local_bounds.adjusted(-guard, -guard, guard, guard)
        if (
            self._visual_polygon == local_polygon
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
            self._visual_polygon = local_polygon
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
        rect = super().rect()
        guard = _device_pixels_to_local(
            self, self.pen().widthF() / 2.0 + OVERLAY_DEVICE_GUARD
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
        if len(self._visual_polygon) == 4:
            true_scene_points = self._item_handle_points_in_scene(self.blk_item)
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
            true_scene_points = [
                self.mapToScene(point) for point in self._handle_points(polygon)
            ]
        self._true_handle_scene_points = [
            QPointF(point) for point in true_scene_points
        ]
        self._display_handle_scene_points = [
            self._clamped_handle_scene_point(point)
            for point in self._true_handle_scene_points
        ]
        for ctrlblock, point in zip(
            self.ctrlblock_group, self._display_handle_scene_points
        ):
            ctrlblock.setPos(self.mapFromScene(point))

    def _clamped_handle_scene_point(self, scene_point: QPointF) -> QPointF:
        view = self.gv
        viewport = view.viewport()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return QPointF(scene_point)
        transform = view.viewportTransform()
        inverse, invertible = transform.inverted()
        if not invertible:
            return QPointF(scene_point)
        device = transform.map(scene_point)
        if QRectF(viewport.rect()).contains(device):
            return QPointF(scene_point)
        inset = PROXY_HANDLE_VIEWPORT_INSET
        left = inset
        top = inset
        right = max(left, viewport.width() - inset)
        bottom = max(top, viewport.height() - inset)
        clamped = QPointF(
            min(max(device.x(), left), right),
            min(max(device.y(), top), bottom),
        )
        return inverse.map(clamped)

    def setAngle(self, angle: float) -> None:
        self.setRotation(angle)

    def visualCenterInScene(self) -> QPointF:
        polygon = self.blk_item.visual_polygon_in_scene()
        if len(polygon) != 4:
            return self.sceneBoundingRect().center()
        return sum((QPointF(point) for point in polygon), QPointF()) / 4

    def handleScenePoint(self, idx: int) -> QPointF:
        if len(self._true_handle_scene_points) == 8:
            return QPointF(self._true_handle_scene_points[idx])
        if self.blk_item is not None:
            points = self._item_handle_points_in_scene(self.blk_item)
            if len(points) == 8:
                return QPointF(points[idx])
        return self.ctrlblock_group[idx].scenePos()

    def handleDisplayScenePoint(self, idx: int) -> QPointF:
        if len(self._display_handle_scene_points) == 8:
            return QPointF(self._display_handle_scene_points[idx])
        return self.ctrlblock_group[idx].scenePos()

    def handleSceneAngle(self, idx: int) -> float:
        vector = self.handleScenePoint(idx) - self.visualCenterInScene()
        return math.degrees(math.atan2(vector.y(), vector.x()))

    def resizeHandleSceneAngle(self, idx: int) -> float:
        polygon = self.blk_item.visual_polygon_in_scene()
        if len(polygon) == 4:
            return resize_handle_scene_angle(polygon[1] - polygon[0], idx)
        return self.rotation() + 45.0 * idx - 135.0

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

    def beginResize(self, idx: int, pointer_scene: QPointF = None):
        if pointer_scene is not None:
            self._beginProxyDrag(idx, pointer_scene)
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
        scene_pos = self._proxySceneTarget(idx, scene_pos)
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
        self.requestOverlaySync()

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
        self.requestOverlaySync()
        return preview_angle

    def finishRotationPreview(self, original_angle: float) -> float:
        preview_angle = self.blk_item.rotation()
        # Keep the model angle as the command's sole owner during preview.
        self.blk_item.setRotation(original_angle)
        self.finishProxyDrag()
        if self.overlay_sync_callback is None:
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
        painter.save()
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceOver
        )
        painter.setPen(self.pen())
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        if len(self._visual_polygon) != 4:
            painter.drawRect(self.rect())
        else:
            painter.drawPolygon(self._visual_polygon)
        painter.restore()

    def hideControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.hide()
        self.requestOverlaySync()

    def showControls(self):
        for ctrl in self.ctrlblock_group:
            ctrl.show()
        self.requestOverlaySync()

    def updateScale(self, scale: float):
        if not self.isVisible():
            if scale != self.current_scale:
                self.need_rescale = True
                self.current_scale = scale
            return

        self.current_scale = scale
        # The handles ignore scene/view transforms, so their device size is stable.
        for ctrl in self.ctrlblock_group:
            ctrl.updateEdgeWidth(CBEDGE_WIDTH)
        self.requestOverlaySync()

    def show(self) -> None:
        super().show()
        if self.need_rescale:
            self.need_rescale = False
        self.setZValue(11)

    def startEditing(self):
        self.setCursor(Qt.CursorShape.IBeamCursor)
        for ctrlb in self.ctrlblock_group:
            ctrlb.hide()
        self.requestOverlaySync()

    def endEditing(self):
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        if self.isVisible():
            for ctrlb in self.ctrlblock_group:
                ctrlb.show()
        self.requestOverlaySync()
