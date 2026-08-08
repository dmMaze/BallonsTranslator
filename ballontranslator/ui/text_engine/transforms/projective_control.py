"""Fixed-size 3D rotation control for one selected Projective stage."""

from dataclasses import replace
import math

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QTransform,
)
from qtpy.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem

from ballontranslator.utils.fontformat import (
    ProjectiveTextTransform,
    TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MAX,
    TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MIN,
    TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MAX,
    TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MIN,
    TEXT_TRANSFORM_PRECISION,
    TEXT_TRANSFORM_SCALE_MAX,
    TEXT_TRANSFORM_SCALE_MIN,
)

from ...cursor import rotateCursorList
from .modal import ModalPointTransform
from ..shape_control import CONTROL_ITEM_DATA_KEY


PROJECTIVE_CONTROL_RADIUS = 68.0
PROJECTIVE_CONTROL_PEN_WIDTH = 4.0
PROJECTIVE_CONTROL_ACTIVE_PEN_WIDTH = 6.0
PROJECTIVE_CONTROL_HIT_WIDTH = 14.0
PROJECTIVE_AXIS_COLORS = {
    'x': QColor(220, 70, 75),
    'y': QColor(90, 190, 70),
    'z': QColor(65, 125, 235),
}


def _canonical_ui_value(value: float) -> float:
    value = round(value, TEXT_TRANSFORM_PRECISION)
    return 0.0 if value == 0.0 else value


def _display_rotation(point):
    """Apply the controller-only small X/Y tilt to one 3D point."""
    x, y, z = point
    angle_x = math.radians(-12.0)
    angle_y = math.radians(12.0)
    cosine_x, sine_x = math.cos(angle_x), math.sin(angle_x)
    cosine_y, sine_y = math.cos(angle_y), math.sin(angle_y)
    y, z = cosine_x * y - sine_x * z, sine_x * y + cosine_x * z
    x, z = cosine_y * x + sine_y * z, -sine_y * x + cosine_y * z
    return x, y, z


def _axis_ring_path(axis: str) -> QPainterPath:
    bases = {
        'x': ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        'y': ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        'z': ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    }
    first, second = (_display_rotation(point) for point in bases[axis])
    circle = QPainterPath()
    circle.addEllipse(QRectF(-1.0, -1.0, 2.0, 2.0))
    radius = PROJECTIVE_CONTROL_RADIUS
    return QTransform(
        radius * first[0], radius * first[1],
        radius * second[0], radius * second[1],
        0.0, 0.0,
    ).map(circle)


class ProjectiveRotationRing(QGraphicsPathItem):
    """One selectable axis ring on the projective controller."""

    def __init__(self, controller, axis: str):
        super().__init__(_axis_ring_path(axis), controller)
        self.controller = controller
        self.axis = axis
        self.hovered = False
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setData(CONTROL_ITEM_DATA_KEY, True)
        self.setCursor(rotateCursorList[0])
        self.setZValue(1.0)

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(PROJECTIVE_CONTROL_HIT_WIDTH)
        return stroker.createStroke(self.path())

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.controller._sync_ring_styles()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.controller._sync_ring_styles()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if self.controller.begin_axis_rotation(self.axis, event.scenePos()):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.controller.handle_modal_mouse_move(event):
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.controller.enter_text_edit(event):
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        if self.controller.handle_modal_mouse_release(event):
            return
        super().mouseReleaseEvent(event)


class TextProjectiveTransformControl(QGraphicsPathItem):
    """Edit one selected Projective stage for exactly one text item.

    The display tilt and device-pixel size belong only to this overlay; the
    selected transform still compiles to one item-local ``QTransform``.

    >>> TextProjectiveTransformControl.__name__
    'TextProjectiveTransformControl'
    """

    def __init__(self):
        super().__init__()
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setData(CONTROL_ITEM_DATA_KEY, True)
        self.setZValue(21.0)

        self.rings = {
            axis: ProjectiveRotationRing(self, axis)
            for axis in ('x', 'y', 'z')
        }
        outline = QPainterPath()
        for ring in self.rings.values():
            outline.addPath(ring.path())
        self.setPath(outline)

        self.modal_indicator = QGraphicsPathItem(self)
        indicator_pen = QPen(QColor(245, 245, 245), 1.25)
        indicator_pen.setCosmetic(True)
        indicator_pen.setStyle(Qt.PenStyle.DashLine)
        self.modal_indicator.setPen(indicator_pen)
        self.modal_indicator.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.modal_indicator.setData(CONTROL_ITEM_DATA_KEY, True)
        self.modal_indicator.setZValue(5.0)
        self.modal_indicator.hide()

        self.modal_origin = QGraphicsEllipseItem(-2.5, -2.5, 5.0, 5.0, self)
        self.modal_origin.setPen(QPen(Qt.PenStyle.NoPen))
        self.modal_origin.setBrush(QBrush(QColor(230, 135, 35)))
        self.modal_origin.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.modal_origin.setData(CONTROL_ITEM_DATA_KEY, True)
        self.modal_origin.setZValue(6.0)

        self.item = None
        self.stack_index = -1
        self._begin_edit = None
        self._preview_transform = None
        self._commit_transform = None
        self._cancel_edit = None
        self._modal_transform = ModalPointTransform()
        self._modal_initial_transform = None
        self._modal_latest_transform = None
        self._modal_view = None
        self._modal_restore_cursor = None
        self._modal_finish_button = None
        self._direct_rotation = False
        self._sync_ring_styles()
        self.hide()

    def bind(
        self,
        item,
        stack_index,
        *,
        begin_edit,
        preview_transform,
        commit_transform,
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
        self.stack_index = int(stack_index)
        self._begin_edit = begin_edit
        self._preview_transform = preview_transform
        self._commit_transform = commit_transform
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
        self._begin_edit = None
        self._preview_transform = None
        self._commit_transform = None
        self._cancel_edit = None
        self.hide()

    def _projective_transform(self):
        if self.item is None:
            return None
        stack = self.item._effective_text_transform()
        if self.stack_index < 0 or self.stack_index >= len(stack):
            return None
        transform = stack[self.stack_index]
        return (
            transform
            if isinstance(transform, ProjectiveTextTransform)
            else None
        )

    def requestGeometryRefresh(self, *_args):
        if self.item is None:
            return
        center = (
            self.item.geometry_controller
            .projective_control_center_in_scene(self.stack_index)
        )
        if center is None:
            self.clear()
            return
        parent = self.parentItem()
        self.setPos(parent.mapFromScene(center) if parent is not None else center)

    def _sync_ring_styles(self):
        active_axis = (
            self._modal_transform.axis
            if self._modal_transform.active
            and self._modal_transform.mode == ModalPointTransform.ROTATE
            else None
        )
        for axis, ring in self.rings.items():
            color = QColor(PROJECTIVE_AXIS_COLORS[axis])
            if active_axis is not None and axis != active_axis:
                color.setAlpha(70)
            else:
                color.setAlpha(255 if active_axis == axis or ring.hovered else 210)
            emphasized = active_axis == axis or ring.hovered
            pen = QPen(
                color,
                PROJECTIVE_CONTROL_ACTIVE_PEN_WIDTH
                if emphasized
                else PROJECTIVE_CONTROL_PEN_WIDTH,
            )
            pen.setCosmetic(True)
            ring.setPen(pen)
            ring.setZValue(2.0 if emphasized else 1.0)

    def _cursor_scene_position(self):
        scene = self.scene()
        if scene is None:
            return None, None
        global_pos = QCursor.pos()
        for view in scene.views():
            viewport_pos = view.viewport().mapFromGlobal(global_pos)
            if view.viewport().rect().contains(viewport_pos):
                return view, view.mapToScene(viewport_pos)
        views = scene.views()
        if not views:
            return None, None
        view = views[0]
        return view, view.mapToScene(view.viewport().mapFromGlobal(global_pos))

    def _set_modal_cursor(self):
        if self._modal_view is None or not self._modal_transform.active:
            return
        viewport = self._modal_view.viewport()
        if self._modal_restore_cursor is None:
            self._modal_restore_cursor = QCursor(viewport.cursor())
        tool = self._modal_transform
        if tool.mode == ModalPointTransform.ROTATE:
            cursor = rotateCursorList[0]
        elif tool.axis == 'x':
            cursor = Qt.CursorShape.SizeHorCursor
        elif tool.axis == 'y':
            cursor = Qt.CursorShape.SizeVerCursor
        else:
            cursor = Qt.CursorShape.SizeFDiagCursor
        viewport.setCursor(cursor)

    def _update_modal_indicator(self):
        tool = self._modal_transform
        if not tool.active:
            return
        path = QPainterPath()
        pen = QPen(self.modal_indicator.pen())
        if tool.mode == ModalPointTransform.SCALE and tool.axis is not None:
            if tool.axis == 'x':
                path.moveTo(-10000.0, 0.0)
                path.lineTo(10000.0, 0.0)
            else:
                path.moveTo(0.0, -10000.0)
                path.lineTo(0.0, 10000.0)
            pen.setColor(PROJECTIVE_AXIS_COLORS[tool.axis])
        else:
            path.moveTo(QPointF())
            path.lineTo(self.mapFromScene(tool.current_mouse))
            pen.setColor(QColor(245, 245, 245))
        self.modal_indicator.setPen(pen)
        self.modal_indicator.setPath(path)
        self.modal_indicator.show()
        self._set_modal_cursor()
        self._sync_ring_styles()

    def _clear_modal_visuals(self):
        self.modal_indicator.setPath(QPainterPath())
        self.modal_indicator.hide()
        if self._modal_view is not None and self._modal_restore_cursor is not None:
            self._modal_view.viewport().setCursor(self._modal_restore_cursor)
        self._modal_view = None
        self._modal_restore_cursor = None
        self._direct_rotation = False
        self._sync_ring_styles()

    def _release_modal_mouse(self):
        scene = self.scene()
        if scene is not None and scene.mouseGrabberItem() is self:
            self.ungrabMouse()
        self._modal_finish_button = None

    def _reset_modal_preview(self):
        if self._modal_initial_transform is None:
            return
        self._modal_latest_transform = self._modal_initial_transform
        self._preview_transform(
            self.stack_index, self._modal_initial_transform
        )

    def _preview_modal(self):
        initial = self._modal_initial_transform
        if initial is None:
            return False
        tool = self._modal_transform
        if tool.mode == tool.ROTATE:
            axis = tool.axis or 'z'
            minimum, maximum = (
                (
                    TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MIN,
                    TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MAX,
                )
                if axis == 'z'
                else (
                    TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MIN,
                    TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MAX,
                )
            )
            value = getattr(initial, f'rotation_{axis}')
            value = min(
                max(value + tool.rotation_delta(), minimum), maximum
            )
            transform = initial.with_value(
                f'rotation_{axis}',
                _canonical_ui_value(value),
            )
        else:
            factor = tool.scale_factor()
            updates = {}
            if tool.axis != 'y':
                updates['horizontal_scale'] = _canonical_ui_value(
                    min(
                        max(
                            initial.horizontal_scale * factor,
                            TEXT_TRANSFORM_SCALE_MIN,
                        ),
                        TEXT_TRANSFORM_SCALE_MAX,
                    )
                )
            if tool.axis != 'x':
                updates['vertical_scale'] = _canonical_ui_value(
                    min(
                        max(
                            initial.vertical_scale * factor,
                            TEXT_TRANSFORM_SCALE_MIN,
                        ),
                        TEXT_TRANSFORM_SCALE_MAX,
                    )
                )
            transform = replace(initial, **updates)
        self._modal_latest_transform = transform
        self._preview_transform(self.stack_index, transform)
        return True

    def _start_modal(
        self,
        mode: str,
        scene_pos: QPointF,
        *,
        axis: str = None,
        direct: bool = False,
    ) -> bool:
        if self.item is None or not self.isVisible():
            return False
        if self._modal_transform.active:
            return self._switch_modal(mode, scene_pos)
        transform = self._projective_transform()
        if transform is None:
            return False
        origin = self.scenePos()
        if not self._modal_transform.begin(mode, (origin,), scene_pos):
            return False
        if axis is not None:
            self._modal_transform.constrain(axis, scene_pos)
        self._modal_initial_transform = transform
        self._modal_latest_transform = transform
        self._direct_rotation = direct
        self._begin_edit(self.stack_index)
        if not direct:
            self.grabMouse()
        self._update_modal_indicator()
        return True

    def _switch_modal(self, mode: str, scene_pos: QPointF) -> bool:
        if not self._modal_transform.active:
            return False
        previous = self._modal_transform.mode
        self._modal_transform.switch_mode(mode, scene_pos)
        if mode == ModalPointTransform.ROTATE:
            self._modal_transform.constrain('z', scene_pos)
        if previous != mode:
            self._reset_modal_preview()
        self._update_modal_indicator()
        return True

    def _constrain_modal(self, axis: str) -> bool:
        tool = self._modal_transform
        if tool.constrain(axis, tool.current_mouse) is None:
            return False
        self._reset_modal_preview()
        self._update_modal_indicator()
        return True

    def _update_modal(self, scene_pos: QPointF) -> bool:
        if not self._modal_transform.active:
            return False
        self._modal_transform.update(scene_pos)
        changed = self._preview_modal()
        self._update_modal_indicator()
        return changed

    def _finish_modal(self, commit: bool, *, release_mouse: bool = True) -> bool:
        if not self._modal_transform.active:
            return False
        initial = self._modal_initial_transform
        latest = self._modal_latest_transform
        if commit:
            self._modal_transform.finish()
        else:
            self._modal_transform.cancel()
        self._modal_initial_transform = None
        self._modal_latest_transform = None
        self._clear_modal_visuals()
        if release_mouse:
            self._release_modal_mouse()
        if commit and latest is not None and latest != initial:
            self._commit_transform(self.stack_index, latest)
        else:
            self._cancel_edit(self.stack_index)
        return True

    def begin_axis_rotation(self, axis: str, scene_pos: QPointF) -> bool:
        self._modal_view = self.scene().views()[0] if self.scene().views() else None
        return self._start_modal(
            ModalPointTransform.ROTATE,
            scene_pos,
            axis=axis,
            direct=True,
        )

    def enter_text_edit(self, event) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or self.item is None:
            return False
        item = self.item
        item.startEdit(pos=item.mapFromScene(event.scenePos()))
        item.update()
        event.accept()
        return True

    def handle_shortcut(
        self,
        key,
        modifiers=Qt.KeyboardModifier.NoModifier,
    ) -> bool:
        if (
            self.item is None
            or not self.isVisible()
            or modifiers != Qt.KeyboardModifier.NoModifier
        ):
            return False
        if key == Qt.Key.Key_Escape and self._modal_transform.active:
            return self._finish_modal(False)
        modes = {
            Qt.Key.Key_R: ModalPointTransform.ROTATE,
            Qt.Key.Key_S: ModalPointTransform.SCALE,
        }
        mode = modes.get(key)
        if mode is not None:
            if self._modal_transform.active:
                scene_pos = QPointF(self._modal_transform.current_mouse)
            else:
                self._modal_view, scene_pos = self._cursor_scene_position()
            if scene_pos is None:
                return False
            started = self._start_modal(mode, scene_pos)
            if (
                started
                and mode == ModalPointTransform.ROTATE
                and self._modal_transform.axis is None
            ):
                self._modal_transform.constrain('z', scene_pos)
                self._update_modal_indicator()
            return started
        if self._modal_transform.active and key in (
            Qt.Key.Key_X,
            Qt.Key.Key_Y,
            Qt.Key.Key_Z,
        ):
            axis = {
                Qt.Key.Key_X: 'x',
                Qt.Key.Key_Y: 'y',
                Qt.Key.Key_Z: 'z',
            }[key]
            if (
                self._modal_transform.mode == ModalPointTransform.SCALE
                and axis == 'z'
            ):
                return False
            return self._constrain_modal(axis)
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
        if self._direct_rotation and self._modal_transform.active:
            self._update_modal(event.scenePos())
            self._finish_modal(event.button() == Qt.MouseButton.LeftButton)
            event.accept()
            return True
        if self._modal_finish_button is None:
            return False
        if event.button() == self._modal_finish_button:
            self._release_modal_mouse()
        event.accept()
        return True

    def mouseMoveEvent(self, event):
        if self.handle_modal_mouse_move(event):
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.handle_modal_mouse_release(event):
            return
        super().mouseReleaseEvent(event)
