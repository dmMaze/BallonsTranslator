"""Canvas-owned input and undo session for one TextBlock alpha mask."""

from dataclasses import replace
import math
from typing import Optional, TYPE_CHECKING

from qtpy.QtCore import QObject, QPointF, Qt, Signal
from qtpy.QtGui import QColor, QPainterPath, QPen
from qtpy.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
)

from ballontranslator.utils.text_alpha_mask import (
    ALPHA_BRUSH_MODES,
    AlphaBrushStroke,
    TextAlphaMask,
    simplify_alpha_brush_points,
)
from ..editing.commands import SetTextAlphaMaskCommand
from ..item import TextBlkItem
from ..shape_control import CONTROL_ITEM_DATA_KEY

if TYPE_CHECKING:
    from .panel import TextEffectPanel
    from ...canvas import Canvas


class TextAlphaMaskEditSession(QObject):
    """Own one transient brush stroke and its selected canvas target.

    >>> TextAlphaMaskEditSession.DEFAULT_DIAMETER
    24.0
    """

    state_changed = Signal()
    DEFAULT_DIAMETER = 24.0

    def __init__(self, canvas: "Canvas") -> None:
        super().__init__(canvas)
        self.canvas = canvas
        self.controls = None
        self.target = None
        self.mode = 'erase'
        self.diameter = self.DEFAULT_DIAMETER
        self._cursor_item = None
        self._drawing = False
        self._stroke_mapper = None
        self._stroke_origin = None
        self._stroke_previous_source = None
        self._stroke_points = []
        self._stroke_before = None
        self._stroke_mode = self.mode
        self._stroke_diameter = self.diameter

    def bind_controls(self, controls: "TextEffectPanel") -> None:
        """Bind the one production panel without adding a second owner."""
        if self.controls is controls:
            return
        if self.controls is not None:
            raise RuntimeError('alpha mask controls are already bound')
        self.controls = controls
        controls.set_alpha_mask_session(self)
        controls.mask_edit_requested.connect(self.set_editing)
        controls.mask_enabled_requested.connect(self.set_enabled)
        controls.mask_mode_changed.connect(self.set_mode)
        controls.mask_diameter_changed.connect(self.set_diameter)
        controls.mask_clear_requested.connect(self.clear_mask)
        controls.mask_remove_requested.connect(self.remove_mask)
        controls.value_commit_requested.connect(self._on_effect_edit_started)
        controls.value_preview_requested.connect(self._on_effect_edit_started)
        controls.parameter_preview_requested.connect(
            self._on_effect_edit_started
        )
        controls.parameter_commit_requested.connect(
            self._on_effect_edit_started
        )
        controls.add_effect_requested.connect(self._on_effect_edit_started)
        controls.add_filter_requested.connect(self._on_effect_edit_started)
        controls.remove_effect_requested.connect(self._on_effect_edit_started)
        controls.move_effect_requested.connect(self._on_effect_edit_started)
        controls.rendered_image_enabled_requested.connect(
            self._on_effect_edit_started
        )
        controls.rendered_image_mode_requested.connect(
            self._on_effect_edit_started
        )
        controls.rendered_image_remove_requested.connect(
            self._on_effect_edit_started
        )
        controls.color_dialog_active_changed.connect(
            self._on_effect_color_dialog_active_changed
        )
        self.state_changed.connect(controls.refresh_alpha_mask_state)
        controls.destroyed.connect(self._on_controls_destroyed)

    def _on_controls_destroyed(self, *_args: object) -> None:
        self.controls = None

    def _on_effect_edit_started(self, *_args: object) -> None:
        self.deactivate()

    def _on_effect_color_dialog_active_changed(self, active: bool) -> None:
        if active:
            self.deactivate()

    @property
    def active(self) -> bool:
        return self.target is not None

    def can_activate(self, item: Optional[TextBlkItem]) -> bool:
        """Return whether one attached current-page item owns canvas input."""
        project = self.canvas.imgtrans_proj
        try:
            return bool(
                item is not None
                and self.canvas.textEditMode()
                and project is not None
                and project.img_valid
                and item.scene() is self.canvas
                and item.parentItem() is self.canvas.textLayer
                and self.canvas.selected_text_items() == [item]
                and not self.canvas.path_reorder_active
            )
        except RuntimeError:
            return False

    def _eligible_selected_item(self) -> Optional[TextBlkItem]:
        items = self.canvas.selected_text_items()
        if len(items) != 1 or not self.can_activate(items[0]):
            return None
        return items[0]

    def set_editing(self, enabled: bool) -> None:
        if enabled:
            self.activate(self._eligible_selected_item())
        else:
            self.deactivate()

    def activate(self, item: Optional[TextBlkItem]) -> bool:
        if item is self.target and self.can_activate(item):
            self._ensure_cursor_item()
            self.state_changed.emit()
            return True
        if not self.can_activate(item):
            self.deactivate()
            return False

        self.deactivate()
        assert item is not None
        self.canvas.cancel_path_reorder()
        self.canvas.clear_text_transform_controls()
        if self.controls is not None:
            self.controls.finish_pending_effect_edits()
            self.controls.cancel_effect_previews()
        self.target = item
        try:
            item.destroyed.connect(self._on_target_destroyed)
        except (AttributeError, RuntimeError):
            self.target = None
            self.state_changed.emit()
            return False

        mask = item.blk.text_alpha_mask
        if mask is None:
            self._push_change(item, TextAlphaMask())
        elif not mask.enabled:
            self._push_change(item, replace(mask, enabled=True))
        self._ensure_cursor_item()
        self.canvas.gv.setFocus()
        self.state_changed.emit()
        return True

    def deactivate(self) -> None:
        had_state = self.target is not None or self._drawing
        self.cancel_active_stroke()
        target = self.target
        self.target = None
        if target is not None:
            try:
                target.destroyed.disconnect(self._on_target_destroyed)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._remove_cursor_item()
        if had_state:
            self.state_changed.emit()

    def _on_target_destroyed(self, *_args: object) -> None:
        self.target = None
        self._drawing = False
        self._clear_stroke_state()
        self._remove_cursor_item()
        self.state_changed.emit()

    def _ensure_cursor_item(self) -> None:
        if self._cursor_item is not None:
            return
        cursor_item = QGraphicsPathItem()
        cursor_item.setData(CONTROL_ITEM_DATA_KEY, True)
        cursor_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        pen = QPen(QColor(30, 147, 229, 220), 1.5)
        pen.setCosmetic(True)
        cursor_item.setPen(pen)
        cursor_item.setZValue(10000.0)
        cursor_item.hide()
        self.canvas.addItem(cursor_item)
        self._cursor_item = cursor_item

    def _remove_cursor_item(self) -> None:
        cursor_item = self._cursor_item
        self._cursor_item = None
        try:
            if cursor_item is not None and cursor_item.scene() is self.canvas:
                QGraphicsScene.removeItem(self.canvas, cursor_item)
        except RuntimeError:
            pass

    @staticmethod
    def _is_finite_point(point: QPointF) -> bool:
        return math.isfinite(point.x()) and math.isfinite(point.y())

    def _map_stroke_point(self, scene_point: QPointF) -> Optional[QPointF]:
        mapper = self._stroke_mapper
        origin = self._stroke_origin
        if mapper is None or origin is None:
            return None
        try:
            source = mapper(scene_point, self._stroke_previous_source)
        except (ArithmeticError, RuntimeError, ValueError):
            return None
        if source is None or not self._is_finite_point(source):
            return None
        self._stroke_previous_source = QPointF(source)
        return QPointF(source.x() - origin.x(), source.y() - origin.y())

    def _update_cursor(self, scene_point: QPointF) -> None:
        target = self.target
        cursor_item = self._cursor_item
        if target is None or cursor_item is None:
            return
        mapper = target.geometry_controller.capture_scene_to_source_mapper()
        if mapper is None:
            cursor_item.hide()
            return
        try:
            source = mapper(scene_point)
        except (ArithmeticError, RuntimeError, ValueError):
            cursor_item.hide()
            return
        if source is None or not self._is_finite_point(source):
            cursor_item.hide()
            return
        path = QPainterPath()
        radius = self.diameter / 2.0
        for index in range(33):
            angle = 2.0 * math.pi * index / 32.0
            source_point = QPointF(
                source.x() + radius * math.cos(angle),
                source.y() + radius * math.sin(angle),
            )
            try:
                scene_outline = target.geometry_controller.map_source_to_scene(
                    source_point
                )
            except (ArithmeticError, RuntimeError, ValueError):
                cursor_item.hide()
                return
            if not self._is_finite_point(scene_outline):
                cursor_item.hide()
                return
            if index == 0:
                path.moveTo(scene_outline)
            else:
                path.lineTo(scene_outline)
        cursor_item.setPath(path)
        cursor_item.show()

    def _top_input_owner(
        self, scene_point: QPointF
    ) -> Optional[QGraphicsItem]:
        shape_frame = self.canvas.txtblkShapeControl
        for item in self.canvas.items(scene_point):
            if item is self._cursor_item:
                continue
            # The ordinary selection frame spans the whole text body. Its
            # child handle items still carry control precedence.
            if item is shape_frame:
                continue
            if bool(item.data(CONTROL_ITEM_DATA_KEY)):
                return item
            if isinstance(item, TextBlkItem):
                return item
        return None

    def handle_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        if not self.active or event.button() != Qt.MouseButton.LeftButton:
            return False
        owner = self._top_input_owner(event.scenePos())
        if owner is not None and owner is not self.target:
            if isinstance(owner, TextBlkItem):
                self.deactivate()
            return False
        if owner is not None and bool(owner.data(CONTROL_ITEM_DATA_KEY)):
            return False

        target = self.target
        assert target is not None
        mapper = target.geometry_controller.capture_scene_to_source_mapper()
        if mapper is None:
            return False
        mask = target.blk.text_alpha_mask
        if mask is None or not mask.enabled:
            self.deactivate()
            return False
        self._stroke_mapper = mapper
        self._stroke_origin = QPointF(target.logical_unpadded_rect().topLeft())
        self._stroke_previous_source = None
        self._stroke_points = []
        self._stroke_before = mask
        self._stroke_mode = self.mode
        self._stroke_diameter = self.diameter
        point = self._map_stroke_point(event.scenePos())
        if point is None:
            self._clear_stroke_state()
            return False
        self._drawing = True
        self._stroke_points.append((point.x(), point.y()))
        self._preview_active_stroke()
        self._update_cursor(event.scenePos())
        return True

    def handle_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        if not self.active:
            return False
        self._update_cursor(event.scenePos())
        if not self._drawing:
            return False
        point = self._map_stroke_point(event.scenePos())
        if point is None:
            return True
        value = (point.x(), point.y())
        if value != self._stroke_points[-1]:
            self._stroke_points.append(value)
            self._preview_active_stroke()
        return True

    def handle_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        if not self._drawing or event.button() != Qt.MouseButton.LeftButton:
            return False
        point = self._map_stroke_point(event.scenePos())
        if point is not None:
            value = (point.x(), point.y())
            if value != self._stroke_points[-1]:
                self._stroke_points.append(value)
        target = self.target
        before = self._stroke_before
        assert before is not None
        points = simplify_alpha_brush_points(self._stroke_points)
        stroke = AlphaBrushStroke(
            self._stroke_mode, self._stroke_diameter, points
        )
        after = replace(before, strokes=before.strokes + (stroke,))
        if target is not None:
            target.set_text_alpha_mask(after, preview=True)
        self._drawing = False
        self._clear_stroke_state()
        if target is not None:
            self._push_change(target, after)
        return True

    def _preview_active_stroke(self) -> None:
        target = self.target
        before = self._stroke_before
        if target is None or before is None or not self._stroke_points:
            return
        stroke = AlphaBrushStroke(
            self._stroke_mode,
            self._stroke_diameter,
            tuple(self._stroke_points),
        )
        target.set_text_alpha_mask(
            replace(before, strokes=before.strokes + (stroke,)),
            preview=True,
        )

    def cancel_active_stroke(self) -> bool:
        if not self._drawing:
            return False
        target = self.target
        self._drawing = False
        self._clear_stroke_state()
        if target is not None:
            try:
                target.clear_text_alpha_mask_preview()
            except RuntimeError:
                pass
        return True

    def _clear_stroke_state(self) -> None:
        self._stroke_mapper = None
        self._stroke_origin = None
        self._stroke_previous_source = None
        self._stroke_points = []
        self._stroke_before = None

    def _push_change(
        self,
        item: TextBlkItem,
        after: Optional[TextAlphaMask],
    ) -> bool:
        command = SetTextAlphaMaskCommand.create(
            item,
            item.blk.text_alpha_mask,
            after,
            self._on_command_applied,
        )
        if command is None:
            return False
        self.canvas.push_undo_command(command)
        return True

    def _on_command_applied(self) -> None:
        target = self.target
        if target is not None and (
            target.scene() is not self.canvas
            or target.blk.text_alpha_mask is None
            or not target.blk.text_alpha_mask.enabled
        ):
            self.deactivate()
            return
        self.state_changed.emit()

    def set_mode(self, mode: str) -> None:
        if mode not in ALPHA_BRUSH_MODES:
            raise ValueError('alpha brush mode must be erase or restore')
        if self.mode == mode:
            return
        self.mode = mode
        self.state_changed.emit()

    def set_diameter(self, diameter: float) -> None:
        value = float(diameter)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError('alpha brush diameter must be finite and positive')
        if self.diameter == value:
            return
        self.diameter = value
        self.state_changed.emit()

    def set_enabled(self, enabled: bool) -> None:
        item = self._eligible_selected_item()
        if item is None or item.blk.text_alpha_mask is None:
            return
        self.cancel_active_stroke()
        mask = item.blk.text_alpha_mask
        self._push_change(item, replace(mask, enabled=bool(enabled)))

    def clear_mask(self) -> None:
        item = self._eligible_selected_item()
        if item is None or item.blk.text_alpha_mask is None:
            return
        self.cancel_active_stroke()
        mask = item.blk.text_alpha_mask
        self._push_change(item, replace(mask, strokes=()))

    def remove_mask(self) -> None:
        item = self._eligible_selected_item()
        if item is None or item.blk.text_alpha_mask is None:
            return
        self.deactivate()
        self._push_change(item, None)

    def handle_escape(self) -> bool:
        if not self.active:
            return False
        self.deactivate()
        return True

    def handle_selection_changed(self) -> None:
        if self.active and self._eligible_selected_item() is not self.target:
            self.deactivate()

    def resolve_for_save(self) -> None:
        self.cancel_active_stroke()

    def resolve_for_history_change(self) -> None:
        self.cancel_active_stroke()

    def resolve_for_page_change(self) -> None:
        self.deactivate()

    def cancel_for_scene_change(self) -> None:
        self.deactivate()
