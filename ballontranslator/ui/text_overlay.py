"""Reusable text guides and viewport-aware overlay invalidation."""

from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Set

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QBrush, QPainter, QPainterPath, QPen, QPolygonF, QRegion
from qtpy.QtWidgets import QGraphicsItem, QGraphicsRectItem


UI_OVERLAY_ITEM_DATA_KEY = 0x1238
OVERLAY_DEVICE_GUARD = 2.0


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


class TextGuideOverlayItem(QGraphicsItem):
    """Paint one reusable, input-transparent text guide."""

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
        guard = device_pixels_to_local(
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
        from .textitem import TEXTRECT_SELECTED_COLOR, TEXTRECT_SHOW_COLOR

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
        if not item.isVisible():
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
        for view in tuple(self.scene.views()):
            if refresh_existing or view not in self._regions:
                self._regions[view] = self._region_for_view(view)

    def sync(self, update_geometry: Callable[[], None]) -> None:
        views = tuple(self.scene.views())
        old_regions = {}
        for view in views:
            old_region = self._regions.get(view)
            if old_region is None:
                old_region = self._region_for_view(view)
            old_regions[view] = QRegion(old_region)
        update_geometry()
        new_regions = {view: self._region_for_view(view) for view in views}
        self._regions = new_regions
        for view in views:
            dirty = old_regions[view] | new_regions[view]
            if not dirty.isEmpty():
                view.viewport().update(dirty)


class TextOverlayManager:
    """Own reusable guide overlays and targeted device invalidation."""

    def __init__(self, scene, parent: QGraphicsItem, shape_control) -> None:
        self.scene = scene
        self.parent = parent
        self.shape_control = shape_control
        self._items: Set[object] = set()
        self._guides: Dict[object, TextGuideOverlayItem] = {}
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

    def _acquire_overlay(self, item) -> TextGuideOverlayItem:
        overlay = self._guides.get(item)
        if overlay is not None:
            return overlay
        overlay = self._pool.pop() if self._pool else TextGuideOverlayItem(self.parent)
        self._guides[item] = overlay
        return overlay

    def _release_overlay(self, item) -> None:
        overlay = self._guides.pop(item, None)
        if overlay is None:
            return
        overlay.hide()
        overlay.setGuide(QPolygonF(), False)
        self._pool.append(overlay)

    def register_item(self, item) -> None:
        self._items.add(item)
        self.sync_overlays()

    def unregister_item(self, item) -> None:
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
            return
        self._textblock_mode = enabled
        self.sync_overlays()

    def overlay_for_item(self, item):
        return self._guides.get(item)

    def _item_parent_polygon(self, item) -> QPolygonF:
        return QPolygonF(
            [self.parent.mapFromScene(point) for point in item.visual_polygon_in_scene()]
        )

    def _update_viewport_geometry(self) -> None:
        """Refresh only viewport-dependent proxy handles during scrolling."""
        control = self.shape_control
        control.refreshDeviceGeometry()
        if control.isVisible():
            control.updateControlBlocks()

    def _update_geometry(self) -> None:
        control = self.shape_control
        if control.blk_item is not None and control.blk_item.scene() is self.scene:
            control.updateBoundingRect()
        else:
            self._update_viewport_geometry()
        active_item = control.blk_item if control.isVisible() else None

        for item in tuple(self._items):
            self._update_item_geometry(item, active_item)

    def _update_item_geometry(self, item, active_item) -> None:
        if item.scene() is not self.scene or not item.isVisible():
            overlay = self._guides.get(item)
            if overlay is not None:
                overlay.hide()
            return
        selected = item.isSelected()
        should_show = item is not active_item and (
            selected or self._textblock_mode
        )
        if not should_show:
            overlay = self._guides.get(item)
            if overlay is not None:
                overlay.hide()
            return
        overlay = self._acquire_overlay(item)
        overlay.setGuide(self._item_parent_polygon(item), selected)
        overlay.show()

    def _sync(self, update_geometry: Callable[[], None]) -> None:
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
                self.invalidator.sync(update_geometry)
                if not self._sync_pending:
                    break
        finally:
            self._syncing = False

    def sync_viewport(self, *_args, **_kwargs) -> None:
        self._sync(self._update_viewport_geometry)

    def sync_items(self, items, *, update_control=True) -> None:
        """Refresh only changed items during interactive geometry updates."""
        items = tuple(dict.fromkeys(items))

        def update_geometry():
            control = self.shape_control
            if update_control:
                if (
                    control.blk_item is not None
                    and control.blk_item.scene() is self.scene
                ):
                    control.updateBoundingRect()
                else:
                    self._update_viewport_geometry()
            active_item = control.blk_item if control.isVisible() else None
            for item in items:
                if item in self._items:
                    self._update_item_geometry(item, active_item)

        self._sync(update_geometry)

    def sync_overlays(self, *_args, **_kwargs) -> None:
        self._sync(self._update_geometry)
