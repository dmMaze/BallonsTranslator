"""Scene controller for one selected free-form Grid text transform."""

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from qtpy.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
)

from ballontranslator.utils.fontformat import GridTextTransform

from ..rendering.surface import NonlinearTextSurfaceRenderer
from ..rendering.raster import RASTER_BOUNDARY_FAILURES
from ...cursor import rotateCursorList
from .modal import ModalPointTransform
from ..shape_control import CONTROL_ITEM_DATA_KEY


GRID_HANDLE_RADIUS = 5.0
GRID_LINE_WIDTH = 1.25


class GridControlPointItem(QGraphicsEllipseItem):
    """One circular, selectable Grid handle."""

    def __init__(self, controller, index):
        super().__init__(
            -GRID_HANDLE_RADIUS,
            -GRID_HANDLE_RADIUS,
            GRID_HANDLE_RADIUS * 2.0,
            GRID_HANDLE_RADIUS * 2.0,
            controller,
        )
        self.controller = controller
        self.index = int(index)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setData(CONTROL_ITEM_DATA_KEY, True)
        pen = QPen(QColor(30, 147, 229), 1.5)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(2.0)
        self._selected = None
        self.set_selected(False)

    def set_selected(self, selected):
        selected = bool(selected)
        if self._selected == selected:
            return
        self._selected = selected
        self.setBrush(
            QBrush(
                QColor(30, 147, 229)
                if selected
                else QColor(255, 255, 255)
            )
        )
        self.update()

    def mousePressEvent(self, event):
        if self.controller.begin_handle_drag(
            self.index, event.scenePos(), event.modifiers()
        ):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.controller.move_handle_drag(event.scenePos()):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.controller.finish_handle_drag():
            event.accept()
            return
        super().mouseReleaseEvent(event)


class TextGridTransformControl(QGraphicsPathItem):
    """Edit one selected Grid stage for exactly one text item.

    >>> TextGridTransformControl.__name__
    'TextGridTransformControl'
    """

    def __init__(self):
        super().__init__()
        pen = QPen(QColor(30, 147, 229, 190), GRID_LINE_WIDTH)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setData(CONTROL_ITEM_DATA_KEY, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        self.setZValue(20.0)

        self.modal_indicator = QGraphicsPathItem(self)
        indicator_pen = QPen(QColor(30, 147, 229), 1.25)
        indicator_pen.setCosmetic(True)
        indicator_pen.setStyle(Qt.PenStyle.DashLine)
        self.modal_indicator.setPen(indicator_pen)
        self.modal_indicator.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.modal_indicator.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.modal_indicator.setData(CONTROL_ITEM_DATA_KEY, True)
        self.modal_indicator.setZValue(4.0)
        self.modal_indicator.hide()

        self.modal_origin = QGraphicsEllipseItem(
            -2.5, -2.5, 5.0, 5.0, self
        )
        self.modal_origin.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        self.modal_origin.setPen(QPen(Qt.PenStyle.NoPen))
        self.modal_origin.setBrush(QBrush(QColor(230, 135, 35)))
        self.modal_origin.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.modal_origin.setData(CONTROL_ITEM_DATA_KEY, True)
        self.modal_origin.setZValue(5.0)
        self.modal_origin.hide()

        self.item = None
        self.stack_index = -1
        self.handles = []
        self.selected_indices = set()
        self._begin_edit = None
        self._preview_points = None
        self._commit_points = None
        self._cancel_edit = None
        self._drag_mapping = None
        self._drag_start_grid = None
        self._drag_previous_grid = None
        self._drag_initial_points = None
        self._drag_latest_points = None
        self._overlay_mapper = None
        self._overlay_source_rect = QRectF()
        self._overlay_transform = None
        self._overlay_renderer = NonlinearTextSurfaceRenderer()
        self._modal_transform = ModalPointTransform()
        self._modal_indices = ()
        self._modal_initial_points = None
        self._modal_latest_points = None
        self._modal_scene_to_grid = None
        self._modal_grid_rect = QRectF()
        self._modal_view = None
        self._modal_restore_cursor = None
        self._modal_finish_button = None
        self.setVisible(False)

    def bind(
        self,
        item,
        stack_index,
        *,
        begin_edit,
        preview_points,
        commit_points,
        cancel_edit,
    ):
        binding_changed = self.item is not item or self.stack_index != stack_index
        refresh_geometry = binding_changed or not self.isVisible()
        if binding_changed and self._modal_transform.active:
            self._finish_modal(False)
        if self.item is not item:
            self.clear()
            self.item = item
            item.visual_geometry_changed.connect(self.requestGeometryRefresh)
            item.moving.connect(self.requestGeometryRefresh)
        elif binding_changed:
            self.selected_indices.clear()
            self._clear_drag()
        self.stack_index = int(stack_index)
        self._begin_edit = begin_edit
        self._preview_points = preview_points
        self._commit_points = commit_points
        self._cancel_edit = cancel_edit
        self.show()
        if refresh_geometry:
            self.requestGeometryRefresh()

    def clear(self):
        if self._modal_transform.active:
            self._finish_modal(False)
        else:
            self._release_modal_mouse()
            self._clear_modal_visuals()
        if self.item is not None:
            try:
                self.item.visual_geometry_changed.disconnect(
                    self.requestGeometryRefresh
                )
                self.item.moving.disconnect(self.requestGeometryRefresh)
            except (RuntimeError, TypeError):
                pass
        self.item = None
        self.stack_index = -1
        self.selected_indices.clear()
        self._clear_drag()
        self._overlay_mapper = None
        self._overlay_source_rect = QRectF()
        self._overlay_transform = None
        self._overlay_renderer.release()
        self.setPath(QPainterPath())
        for handle in self.handles:
            handle.setParentItem(None)
            if handle.scene() is not None:
                handle.scene().removeItem(handle)
        self.handles = []
        self.hide()

    def _grid_transform(self):
        if self.item is None:
            return None
        stack = self.item._effective_text_transform()
        if self.stack_index < 0 or self.stack_index >= len(stack):
            return None
        transform = stack[self.stack_index]
        return transform if isinstance(transform, GridTextTransform) else None

    def _ensure_handle_count(self, count):
        while len(self.handles) < count:
            self.handles.append(
                GridControlPointItem(self, len(self.handles))
            )
        while len(self.handles) > count:
            handle = self.handles.pop()
            handle.setParentItem(None)
            if handle.scene() is not None:
                handle.scene().removeItem(handle)
        self.selected_indices.intersection_update(range(count))

    def requestGeometryRefresh(self, *_args):
        if self.item is None:
            return
        geometry = self.item.geometry_controller.grid_control_geometry(
            self.stack_index
        )
        if geometry is None:
            self.clear()
            return
        visual_points, mapper, source_rect, transform = geometry
        previous_mapper = self._overlay_mapper
        if (
            previous_mapper is mapper
            or (
                previous_mapper is not None
                and previous_mapper.geometry_key == mapper.geometry_key
            )
        ):
            # Movement changes item-to-control placement, not Grid geometry.
            # Keep the mapper and its already-computed visual bounds warm.
            mapper = previous_mapper
        self._overlay_mapper = mapper
        self._overlay_source_rect = source_rect
        self._overlay_transform = transform
        item_to_control, valid = self.item.itemTransform(self)
        if not valid:
            return
        local_points = item_to_control.map(visual_points)
        self._ensure_handle_count(len(local_points))
        for index, (handle, local_point) in enumerate(
            zip(self.handles, local_points)
        ):
            handle.setPos(local_point)
            handle.set_selected(index in self.selected_indices)

        paint_bounds = item_to_control.mapRect(
            mapper.visual_bounds(source_rect)
        )
        if local_points:
            handle_bounds = QPolygonF(local_points).boundingRect().adjusted(
                -GRID_HANDLE_RADIUS,
                -GRID_HANDLE_RADIUS,
                GRID_HANDLE_RADIUS,
                GRID_HANDLE_RADIUS,
            )
            paint_bounds = paint_bounds.united(handle_bounds)
        path = QPainterPath()
        if not paint_bounds.isEmpty():
            # The Catmull-Rom-warped guide may extend beyond every handle. The
            # item's bounds must cover all pixels it paints so moving it also
            # invalidates the guide's previous scene position.
            path.addRect(paint_bounds)
        self.setPath(path)
        self.update()

    def _paint_grid_source(self, painter, _option, _widget):
        transform = self._overlay_transform
        rect = self._overlay_source_rect
        if transform is None or rect.isEmpty():
            return
        path = QPainterPath()
        for row in range(transform.vertical_divisions + 1):
            y = rect.top() + (
                rect.height() * row / transform.vertical_divisions
            )
            path.moveTo(rect.left(), y)
            path.lineTo(rect.right(), y)
        for column in range(transform.horizontal_divisions + 1):
            x = rect.left() + (
                rect.width() * column / transform.horizontal_divisions
            )
            path.moveTo(x, rect.top())
            path.lineTo(x, rect.bottom())
        pen = QPen(QColor(30, 147, 229, 190), GRID_LINE_WIDTH)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def paint(self, painter: QPainter, option, widget=None):
        if self.item is None or self._overlay_mapper is None:
            return
        item_to_control, valid = self.item.itemTransform(self)
        if not valid:
            return
        painter.save()
        try:
            painter.setTransform(item_to_control, True)
            # Warp one guide surface; mapping dense guide segments as separate
            # scene paths makes Grid interaction stall at high subdivision.
            self._overlay_renderer.paint(
                painter,
                option,
                self._overlay_mapper,
                self._overlay_source_rect,
                (
                    'grid-controller',
                    self.stack_index,
                    self._overlay_mapper.geometry_key,
                ),
                cache_allowed=True,
                paint_source=self._paint_grid_source,
                high_quality=False,
            )
        except RASTER_BOUNDARY_FAILURES:
            pass
        finally:
            painter.restore()

    def shape(self):
        path = QPainterPath(super().shape())
        bounds = self.path().boundingRect()
        if not bounds.isEmpty():
            path.addRect(bounds)
        return path

    def _set_selected_indices(self, indices):
        self.selected_indices = {
            int(index)
            for index in indices
            if 0 <= int(index) < len(self.handles)
        }
        for index, handle in enumerate(self.handles):
            handle.set_selected(index in self.selected_indices)
        if self.selected_indices:
            self.setFocus()

    def _cursor_scene_position(self):
        scene = self.scene()
        if scene is None:
            return None, None
        global_pos = QCursor.pos()
        views = scene.views()
        for view in views:
            viewport_pos = view.viewport().mapFromGlobal(global_pos)
            if view.viewport().rect().contains(viewport_pos):
                return view, view.mapToScene(viewport_pos)
        if not views:
            return None, None
        view = views[0]
        return view, view.mapToScene(view.viewport().mapFromGlobal(global_pos))

    def _set_modal_cursor(self) -> None:
        view = self._modal_view
        if view is None or not self._modal_transform.active:
            return
        viewport = view.viewport()
        if self._modal_restore_cursor is None:
            self._modal_restore_cursor = QCursor(viewport.cursor())
        tool = self._modal_transform
        if tool.mode == tool.ROTATE:
            cursor = rotateCursorList[0]
        elif tool.mode == tool.SCALE:
            cursor = Qt.CursorShape.SizeFDiagCursor
        elif tool.axis == 'x':
            cursor = Qt.CursorShape.SizeHorCursor
        elif tool.axis == 'y':
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.SizeAllCursor
        viewport.setCursor(cursor)

    def _update_modal_indicator(self) -> None:
        tool = self._modal_transform
        if not tool.active:
            return
        path = QPainterPath()
        pen = QPen(self.modal_indicator.pen())
        if tool.mode == tool.TRANSLATE and tool.axis is not None:
            view = self._modal_view
            if view is not None:
                visible = view.mapToScene(view.viewport().rect()).boundingRect()
            else:
                visible = self.scene().sceneRect()
            if tool.axis == 'x':
                start = QPointF(visible.left(), tool.origin.y())
                end = QPointF(visible.right(), tool.origin.y())
                pen.setColor(QColor(215, 70, 70))
            else:
                start = QPointF(tool.origin.x(), visible.top())
                end = QPointF(tool.origin.x(), visible.bottom())
                pen.setColor(QColor(70, 175, 90))
            path.moveTo(self.mapFromScene(start))
            path.lineTo(self.mapFromScene(end))
        elif tool.mode in (tool.ROTATE, tool.SCALE):
            pen.setColor(
                QColor(230, 135, 35)
                if tool.mode == tool.ROTATE
                else QColor(30, 147, 229)
            )
            path.moveTo(self.mapFromScene(tool.origin))
            path.lineTo(self.mapFromScene(tool.current_mouse))
        self.modal_indicator.setPen(pen)
        self.modal_indicator.setPath(path)
        self.modal_indicator.setVisible(not path.isEmpty())
        show_origin = (
            tool.mode in (tool.ROTATE, tool.SCALE)
            or tool.axis is not None
        )
        self.modal_origin.setPos(self.mapFromScene(tool.origin))
        self.modal_origin.setVisible(show_origin)
        self._set_modal_cursor()

    def _clear_modal_visuals(self) -> None:
        self.modal_indicator.setPath(QPainterPath())
        self.modal_indicator.hide()
        self.modal_origin.hide()
        if self._modal_view is not None and self._modal_restore_cursor is not None:
            self._modal_view.viewport().setCursor(self._modal_restore_cursor)
        self._modal_view = None
        self._modal_restore_cursor = None

    def _release_modal_mouse(self) -> None:
        scene = self.scene()
        if scene is not None and scene.mouseGrabberItem() is self:
            self.ungrabMouse()
        self._modal_finish_button = None

    def _reset_modal_preview(self) -> None:
        if self._modal_initial_points is None:
            return
        self._modal_latest_points = self._modal_initial_points
        self._preview_points(self.stack_index, self._modal_initial_points)

    def _preview_modal_scene_points(self, scene_points) -> bool:
        if (
            self._modal_scene_to_grid is None
            or self._modal_initial_points is None
            or self._modal_grid_rect.isEmpty()
        ):
            return False
        output_x, output_y, valid = self._modal_scene_to_grid(scene_points)
        rect = self._modal_grid_rect
        points = list(self._modal_initial_points)
        for offset, index in enumerate(self._modal_indices):
            if not bool(valid[offset]):
                continue
            points[index] = (
                (float(output_x[offset]) - rect.left()) / rect.width(),
                (float(output_y[offset]) - rect.top()) / rect.height(),
            )
        self._modal_latest_points = tuple(points)
        self._preview_points(self.stack_index, self._modal_latest_points)
        return True

    def _start_modal(self, mode: str, scene_pos: QPointF) -> bool:
        if (
            self.item is None
            or not self.isVisible()
            or not self.selected_indices
            or self._drag_mapping is not None
        ):
            return False
        if self._modal_transform.active:
            return self._switch_modal(mode, scene_pos)
        transform = self._grid_transform()
        mapping = (
            self.item.geometry_controller
            .capture_scene_to_grid_output_array_mapper(self.stack_index)
        )
        if transform is None or mapping is None:
            return False
        indices = tuple(sorted(self.selected_indices))
        scene_points = tuple(self.handles[index].scenePos() for index in indices)
        if not self._modal_transform.begin(mode, scene_points, scene_pos):
            return False
        self._modal_indices = indices
        self._modal_initial_points = tuple(transform.control_points)
        self._modal_latest_points = self._modal_initial_points
        self._modal_scene_to_grid = mapping
        self._modal_grid_rect = QRectF(self._overlay_source_rect)
        self._begin_edit(self.stack_index)
        self.grabMouse()
        self._update_modal_indicator()
        return True

    def _switch_modal(self, mode: str, scene_pos: QPointF) -> bool:
        if not self._modal_transform.active:
            return False
        previous_mode = self._modal_transform.mode
        self._modal_transform.switch_mode(mode, scene_pos)
        if previous_mode != mode:
            self._reset_modal_preview()
        self._update_modal_indicator()
        return True

    def _constrain_modal(self, axis: str, scene_pos: QPointF) -> bool:
        if self._modal_transform.constrain(axis, scene_pos) is None:
            return False
        self._reset_modal_preview()
        self._update_modal_indicator()
        return True

    def _update_modal(self, scene_pos: QPointF) -> bool:
        if not self._modal_transform.active:
            return False
        scene_points = self._modal_transform.update(scene_pos)
        changed = self._preview_modal_scene_points(scene_points)
        self._update_modal_indicator()
        return changed

    def _finish_modal(self, commit: bool, *, release_mouse: bool = True) -> bool:
        if not self._modal_transform.active:
            return False
        initial = self._modal_initial_points
        latest = self._modal_latest_points
        if commit:
            self._modal_transform.finish()
        else:
            self._modal_transform.cancel()
        self._modal_indices = ()
        self._modal_initial_points = None
        self._modal_latest_points = None
        self._modal_scene_to_grid = None
        self._modal_grid_rect = QRectF()
        self._clear_modal_visuals()
        if release_mouse:
            self._release_modal_mouse()
        if commit and latest is not None and latest != initial:
            self._commit_points(self.stack_index, latest)
        else:
            self._cancel_edit(self.stack_index)
        return True

    def handle_shortcut(
        self,
        key,
        modifiers=Qt.KeyboardModifier.NoModifier,
    ) -> bool:
        """Handle a Grid modal shortcut from either key or app actions."""
        if (
            self.item is None
            or not self.isVisible()
            or not self.selected_indices
            or modifiers != Qt.KeyboardModifier.NoModifier
        ):
            return False
        if key == Qt.Key.Key_Escape and self._modal_transform.active:
            return self._finish_modal(False)
        modes = {
            Qt.Key.Key_G: ModalPointTransform.TRANSLATE,
            Qt.Key.Key_R: ModalPointTransform.ROTATE,
            Qt.Key.Key_S: ModalPointTransform.SCALE,
        }
        mode = modes.get(key)
        if mode is not None:
            if self._modal_transform.active:
                scene_pos = QPointF(self._modal_transform.current_mouse)
            else:
                self._modal_view, scene_pos = self._cursor_scene_position()
            return scene_pos is not None and self._start_modal(mode, scene_pos)
        if self._modal_transform.active and key in (Qt.Key.Key_X, Qt.Key.Key_Y):
            if self._modal_transform.mode == ModalPointTransform.ROTATE:
                return False
            axis = 'x' if key == Qt.Key.Key_X else 'y'
            return self._constrain_modal(
                axis, self._modal_transform.current_mouse
            )
        return False

    def handle_modal_mouse_press(self, event) -> bool:
        if not self._modal_transform.active or event.button() not in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            return False
        self._modal_finish_button = event.button()
        self._finish_modal(
            event.button() == Qt.MouseButton.LeftButton,
            release_mouse=False,
        )
        event.accept()
        return True

    def handle_modal_mouse_move(self, event) -> bool:
        if self._modal_finish_button is not None:
            event.accept()
            return True
        if not self._modal_transform.active:
            return False
        self._update_modal(event.scenePos())
        event.accept()
        return True

    def handle_modal_mouse_release(self, event) -> bool:
        if self._modal_finish_button is None:
            return False
        if event.button() == self._modal_finish_button:
            self._release_modal_mouse()
        event.accept()
        return True

    def begin_handle_drag(self, index, scene_pos, modifiers):
        transform = self._grid_transform()
        if transform is None:
            return False
        toggle = bool(
            modifiers
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.ShiftModifier
            )
        )
        if toggle:
            selected = set(self.selected_indices)
            if index in selected:
                selected.remove(index)
                self._set_selected_indices(selected)
                return True
            selected.add(index)
            self._set_selected_indices(selected)
        elif index not in self.selected_indices:
            self._set_selected_indices({index})
        if not self.selected_indices:
            return True

        captured = (
            self.item.geometry_controller
            .capture_scene_to_grid_output_mapper(self.stack_index)
        )
        if captured is None:
            return False
        scene_to_grid, normalize_delta = captured
        self._drag_mapping = (scene_to_grid, normalize_delta)
        self._drag_start_grid = scene_to_grid(scene_pos)
        self._drag_previous_grid = QPointF(self._drag_start_grid)
        self._drag_initial_points = tuple(transform.control_points)
        self._drag_latest_points = self._drag_initial_points
        self._begin_edit(self.stack_index)
        return True

    def move_handle_drag(self, scene_pos):
        if self._drag_mapping is None:
            return False
        scene_to_grid, normalize_delta = self._drag_mapping
        current = scene_to_grid(scene_pos, self._drag_previous_grid)
        self._drag_previous_grid = QPointF(current)
        delta = normalize_delta(current - self._drag_start_grid)
        points = list(self._drag_initial_points)
        for index in self.selected_indices:
            x, y = points[index]
            points[index] = (x + delta.x(), y + delta.y())
        self._drag_latest_points = tuple(points)
        self._preview_points(self.stack_index, self._drag_latest_points)
        return True

    def finish_handle_drag(self):
        if self._drag_mapping is None:
            return False
        points = self._drag_latest_points
        unchanged = points == self._drag_initial_points
        self._clear_drag()
        if unchanged:
            self._cancel_edit(self.stack_index)
        else:
            self._commit_points(self.stack_index, points)
        return True

    def _clear_drag(self):
        self._drag_mapping = None
        self._drag_start_grid = None
        self._drag_previous_grid = None
        self._drag_initial_points = None
        self._drag_latest_points = None

    def select_handles_in_scene_rect(self, scene_rect, modifiers) -> bool:
        if self.item is None or not self.isVisible():
            return False
        enclosed = {
            index
            for index, handle in enumerate(self.handles)
            if scene_rect.contains(handle.scenePos())
        }
        toggle = bool(
            modifiers
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.ShiftModifier
            )
        )
        self._set_selected_indices(
            self.selected_indices.symmetric_difference(enclosed)
            if toggle
            else enclosed
        )
        return True

    def mousePressEvent(self, event):
        if self.handle_modal_mouse_press(event):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.handle_modal_mouse_move(event):
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self.item is None
        ):
            return super().mouseDoubleClickEvent(event)
        item = self.item
        item.startEdit(pos=item.mapFromScene(event.scenePos()))
        item.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.handle_modal_mouse_release(event):
            return
        super().mouseReleaseEvent(event)
