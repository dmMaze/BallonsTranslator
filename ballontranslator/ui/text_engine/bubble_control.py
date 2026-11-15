"""Direct editing of a text block's closed cubic wrapping constraint."""

import math
from typing import Optional, TYPE_CHECKING

from qtpy.QtCore import QCoreApplication, QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QBrush, QColor, QPainter, QPainterPath, QPainterPathStroker, QPen
from qtpy.QtWidgets import (
    QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem, QGraphicsSceneMouseEvent,
    QMenu, QStyleOptionGraphicsItem, QWidget,
)

from ballontranslator.utils.bubble_shape import BubbleShape
from .bubble_layout import bubble_path
from .editing.commands import BubbleShapeState, SetBubbleShapeCommand
from .shape_control import CONTROL_ITEM_DATA_KEY, device_pixels_to_local

if TYPE_CHECKING:
    from .item import TextBlkItem
    from .shape_control import TextBlkShapeControl


class BubbleControlPoint(QGraphicsRectItem):
    """A fixed-size anchor or tangent handle; the controller owns its gesture."""

    def __init__(self, controller: 'BubbleShapeControl', index: int, part: int) -> None:
        radius = 4.5 if part == 0 else 3.5
        super().__init__(-radius, -radius, radius * 2, radius * 2, controller)
        self.controller, self.index, self.part = controller, index, part
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setData(CONTROL_ITEM_DATA_KEY, True)
        self.setZValue(3 if part == 0 else 2)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: Optional[QWidget] = None) -> None:
        color = QColor(30, 147, 229)
        painter.setPen(QPen(color, 1.25))
        painter.setBrush(color if self.part == 0 and self.index in self.controller.selected_indices
                         else QColor('white'))
        if self.part == 0:
            painter.drawRect(self.rect())
        else:
            painter.drawEllipse(self.rect())

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.controller.select_anchor(self.index, False)
        else:
            self.controller.begin_drag(self.index, self.part, event.scenePos(), event.modifiers())
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.controller.move_drag(event.scenePos(), event.modifiers())
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.controller.finish_drag()
        elif event.button() == Qt.MouseButton.RightButton:
            self.controller.context_menu(event.screenPos(), event.scenePos())
        event.accept()

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.part == 0 and event.button() == Qt.MouseButton.LeftButton:
            self.controller.cancel_drag()
            self.controller.select_anchor(self.index, False)
            self.controller.convert_selected(not self.controller.is_smooth(self.index))
        event.accept()


class BubbleShapeControl(QGraphicsPathItem):
    """Preview derived geometry and commit one shape/frame command per gesture.

    >>> BubbleShapeControl.__name__
    'BubbleShapeControl'
    """

    def __init__(self, owner: 'TextBlkShapeControl') -> None:
        super().__init__(owner)
        self.owner = owner
        self.points: list[BubbleControlPoint] = []
        self.selected_indices: set[int] = set()
        self._bound_item: Optional[TextBlkItem] = None
        self._before: Optional[BubbleShapeState] = None
        self._after: Optional[BubbleShapeState] = None
        self._source_rect = QRectF()
        self._page_rect = QRectF()
        self._nodes: list[list[float]] = []
        self._scene_to_source = None
        self._previous_source = None
        self._drag_start = QPointF()
        self._drag_index = self._drag_part = 0
        self._pull_handles = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setData(CONTROL_ITEM_DATA_KEY, True)
        self.setZValue(5)
        pen = QPen(QColor(30, 147, 229), 1.5)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.links = QGraphicsPathItem(self)
        link_pen = QPen(QColor(30, 147, 229, 160), 1)
        link_pen.setCosmetic(True)
        self.links.setPen(link_pen)
        self.links.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.links.setData(CONTROL_ITEM_DATA_KEY, True)
        self.setToolTip(QCoreApplication.translate('BubbleShapeControl',
            'Drag anchors or handles. Shift: constrain / select more anchors.\n'
            'Alt-drag an anchor: pull handles. Alt-drag a handle: move it independently.\n'
            'Double-click the curve to add an anchor; double-click an anchor to switch smooth/corner.\n'
            'Delete: remove selected anchors. Escape: cancel the current drag.'))
        self.hide()

    @property
    def dragging(self) -> bool:
        return self._before is not None

    def bind(self, item: Optional['TextBlkItem']) -> None:
        if item is self._bound_item:
            return
        self.cancel_drag()
        self._bound_item = item
        self.selected_indices.clear()
        self.hide()

    def refresh(self) -> None:
        item = self._bound_item
        if item is None or item.layout.flow_profile is None:
            self.hide()
            return
        shape = item.layout.flow_profile.shape
        self.setPath(self.owner.shape())
        count = len(shape.nodes) * 3
        while len(self.points) > count:
            point = self.points.pop()
            point.setParentItem(None)
            if point.scene() is not None:
                point.scene().removeItem(point)
        while len(self.points) < count:
            number = len(self.points)
            self.points.append(BubbleControlPoint(self, number // 3, number % 3))
        self.selected_indices.intersection_update(range(len(shape.nodes)))
        frame = item.logical_unpadded_rect()
        links = QPainterPath()
        for index, node in enumerate(shape.nodes):
            positions = []
            for part in range(3):
                source = QPointF(frame.left() + node[part * 2] * frame.width(),
                                 frame.top() + node[part * 2 + 1] * frame.height())
                position = self.mapFromScene(item.geometry_controller.map_source_to_scene(source))
                point = self.points[index * 3 + part]
                point.setPos(position)
                visible = part == 0 or (index in self.selected_indices and node[part * 2:part * 2 + 2] != node[:2])
                point.setVisible(visible)
                point.update()
                positions.append(position)
                if part and visible:
                    links.moveTo(positions[0])
                    links.lineTo(position)
        self.links.setPath(links)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(device_pixels_to_local(self, 10))
        return stroker.createStroke(self.path())

    def select_anchor(self, index: int, extend: bool) -> None:
        if extend:
            self.selected_indices.symmetric_difference_update({index})
        elif index not in self.selected_indices:
            self.selected_indices = {index}
        self.setFocus()
        self.refresh()

    def is_smooth(self, index: int) -> bool:
        node = self._bound_item.layout.flow_profile.shape.nodes[index]
        a = QPointF(node[2] - node[0], node[3] - node[1])
        b = QPointF(node[4] - node[0], node[5] - node[1])
        scale = math.hypot(a.x(), a.y()) * math.hypot(b.x(), b.y())
        return (scale > 1e-10 and a.x() * b.x() + a.y() * b.y() < 0
                and abs(a.x() * b.y() - a.y() * b.x()) < scale * 1e-5)

    def _capture(self) -> None:
        self.scene().bubble_edit_started.emit()
        item = self._bound_item
        self._source_rect = item.logical_unpadded_rect()
        self._page_rect = item.absBoundingRect(qrect=True)
        self._before = (item.blk.bubble_shape, item.blk.bubble_layout, self._page_rect.getRect())
        self._after = self._before
        rect = self._source_rect
        self._nodes = [[rect.left() + value * rect.width() if offset % 2 == 0
                        else rect.top() + value * rect.height()
                        for offset, value in enumerate(node)] for node in self._before[0].nodes]

    def begin_drag(self, index: int, part: int, position: QPointF,
                   modifiers: Qt.KeyboardModifier) -> None:
        self.cancel_drag()
        item = self._bound_item
        item.endEdit(keep_focus=False)
        item.setSelected(True)
        self.select_anchor(index, bool(part == 0 and modifiers & Qt.KeyboardModifier.ShiftModifier))
        if part == 0 and index not in self.selected_indices:
            return
        self._capture()
        self._scene_to_source = item.geometry_controller.capture_scene_to_source_mapper()
        if self._scene_to_source is None:
            self._clear_drag()
            return
        self._drag_start = self._scene_to_source(position)
        self._previous_source = self._drag_start
        self._drag_index, self._drag_part = index, part
        self._pull_handles = part == 0 and bool(modifiers & Qt.KeyboardModifier.AltModifier)
        self._smooth = self.is_smooth(index)

    def _state_from_nodes(self, nodes: list[list[float]]) -> Optional[BubbleShapeState]:
        bounds = bubble_path(nodes).boundingRect()
        if min(bounds.width(), bounds.height()) < 1:
            return None
        delta = bounds.center() - self._source_rect.center()
        angle = math.radians(self._bound_item.rotation())
        delta = QPointF(math.cos(angle) * delta.x() - math.sin(angle) * delta.y(),
                        math.sin(angle) * delta.x() + math.cos(angle) * delta.y())
        center = self._page_rect.center() + delta
        rect = QRectF(center.x() - bounds.width() / 2, center.y() - bounds.height() / 2,
                      bounds.width(), bounds.height())
        normalized = tuple(tuple((value - bounds.left()) / bounds.width() if offset % 2 == 0
                                 else (value - bounds.top()) / bounds.height()
                                 for offset, value in enumerate(node)) for node in nodes)
        try:
            shape = BubbleShape(rect.getRect(), normalized)
        except ValueError:
            return None
        return shape, True, rect.getRect()

    def move_drag(self, position: QPointF, modifiers: Qt.KeyboardModifier) -> None:
        if not self.dragging:
            return
        source = self._scene_to_source(position, self._previous_source)
        self._previous_source = source
        delta = source - self._drag_start
        nodes = [list(node) for node in self._nodes]
        part = 2 if self._pull_handles else self._drag_part
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if part == 0:
            if shift:
                delta = QPointF(delta.x(), 0) if abs(delta.x()) >= abs(delta.y()) else QPointF(0, delta.y())
            for index in self.selected_indices:
                for offset in (0, 2, 4):
                    nodes[index][offset] += delta.x()
                    nodes[index][offset + 1] += delta.y()
        else:
            node = nodes[self._drag_index]
            anchor = QPointF(*node[:2])
            old = anchor if self._pull_handles else QPointF(*node[part * 2:part * 2 + 2])
            vector = old + delta - anchor
            length = math.hypot(vector.x(), vector.y())
            if shift:
                angle = round(math.atan2(vector.y(), vector.x()) / (math.pi / 4)) * (math.pi / 4)
                vector = QPointF(math.cos(angle) * length, math.sin(angle) * length)
            node[part * 2:part * 2 + 2] = (anchor.x() + vector.x(), anchor.y() + vector.y())
            if self._pull_handles or self._smooth and not modifiers & Qt.KeyboardModifier.AltModifier:
                opposite = 6 - part * 2
                other_length = length if self._pull_handles else math.hypot(
                    node[opposite] - anchor.x(), node[opposite + 1] - anchor.y())
                ratio = other_length / length if length > 1e-9 else 0
                node[opposite:opposite + 2] = (anchor.x() - vector.x() * ratio,
                                              anchor.y() - vector.y() * ratio)
        state = self._before if nodes == self._nodes else self._state_from_nodes(nodes)
        if state is not None and state != self._after:
            self._after = state
            shape, enabled, rect = state
            self._bound_item.set_bubble_shape(shape, enabled, QRectF(*rect), preview=True)

    def _clear_drag(self) -> None:
        self._before = self._after = None
        self._nodes = []
        self._scene_to_source = self._previous_source = None

    def finish_drag(self) -> None:
        if not self.dragging:
            return
        before, after = self._before, self._after
        self._clear_drag()
        if before != after:
            self.scene().push_text_command(SetBubbleShapeCommand([self._bound_item], [before], [after]))

    def cancel_drag(self) -> None:
        if not self.dragging:
            return
        shape, enabled, rect = self._before
        self._clear_drag()
        grabber = self.scene().mouseGrabberItem() if self.scene() is not None else None
        if grabber is self or grabber in self.points:
            grabber.ungrabMouse()
        self._bound_item.set_bubble_shape(shape, enabled, QRectF(*rect), preview=True)

    def _commit_nodes(self, nodes: list[list[float]]) -> None:
        state = self._state_from_nodes(nodes)
        if state is not None:
            self._after = state
        self.finish_drag()
        self.refresh()

    def delete_selected(self) -> None:
        self.cancel_drag()
        self._capture()
        nodes = [node for index, node in enumerate(self._nodes) if index not in self.selected_indices]
        if len(nodes) < 3 or len(nodes) == len(self._nodes):
            self._clear_drag()
            return
        self.selected_indices.clear()
        self._commit_nodes(nodes)

    def convert_selected(self, smooth: bool) -> None:
        self.cancel_drag()
        self._capture()
        nodes = [list(node) for node in self._nodes]
        for index in self.selected_indices:
            anchor = QPointF(*nodes[index][:2])
            if not smooth:
                nodes[index][2:] = nodes[index][:2] * 2
                continue
            previous = QPointF(*nodes[index - 1][:2])
            following = QPointF(*nodes[(index + 1) % len(nodes)][:2])
            tangent = following - previous
            length = math.hypot(tangent.x(), tangent.y())
            if length <= 1e-9:
                continue
            for part, neighbor, sign in ((1, previous, -1), (2, following, 1)):
                handle_length = math.hypot(nodes[index][part * 2] - anchor.x(),
                                           nodes[index][part * 2 + 1] - anchor.y())
                if handle_length < 1e-6:
                    handle_length = math.hypot(neighbor.x() - anchor.x(), neighbor.y() - anchor.y()) / 3
                handle = anchor + tangent * (sign * handle_length / length)
                nodes[index][part * 2:part * 2 + 2] = (handle.x(), handle.y())
        if nodes == self._nodes:
            self._clear_drag()
        else:
            self._commit_nodes(nodes)

    def insert_anchor(self, scene_position: QPointF) -> None:
        item = self._bound_item
        shape = item.blk.bubble_shape
        if len(shape.nodes) >= 64:
            return
        mapping = item.geometry_controller.capture_scene_to_source_mapper()
        if mapping is None:
            return
        point = mapping(scene_position)
        self.cancel_drag()
        self._capture()
        best = (math.inf, 0, .5)
        # A bounded search is only needed on insertion; de Casteljau preserves the curve.
        for index, node in enumerate(self._nodes):
            following = self._nodes[(index + 1) % len(self._nodes)]
            a, b, c, d = (QPointF(*values) for values in (node[:2], node[4:], following[2:4], following[:2]))
            def distance(t: float) -> float:
                q = a * (1 - t) ** 3 + b * (3 * t * (1 - t) ** 2) + c * (3 * t * t * (1 - t)) + d * t ** 3
                return (q.x() - point.x()) ** 2 + (q.y() - point.y()) ** 2
            sample = min(range(33), key=lambda n: distance(n / 32))
            low, high = max(0., (sample - 1) / 32), min(1., (sample + 1) / 32)
            for _ in range(16):
                first, second = (2 * low + high) / 3, (low + 2 * high) / 3
                if distance(first) < distance(second):
                    high = second
                else:
                    low = first
            t = (low + high) / 2
            best = min(best, (distance(t), index, t))
        _, index, t = best
        self._after = (shape.split_segment(index, t), True, self._page_rect.getRect())
        self.selected_indices = {index + 1}
        self.finish_drag()

    def handle_shortcut(self, key: int, modifiers: Qt.KeyboardModifier) -> bool:
        if not self.isVisible() or self._bound_item is None or self._bound_item.isEditing():
            return False
        if not self.dragging and not self.hasFocus():
            return False
        if key == Qt.Key.Key_Escape:
            if self.dragging:
                self.cancel_drag()
            else:
                self.selected_indices.clear()
                self.clearFocus()
                self.refresh()
            return True
        if self.dragging:
            # Keep block deletion, movement and tool shortcuts out of this gesture.
            return True
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            return True
        if key == Qt.Key.Key_A and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.selected_indices = set(range(len(self._bound_item.blk.bubble_shape.nodes)))
            self.refresh()
            return True
        directions = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
                      Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        if key in directions and self.selected_indices:
            self._capture()
            dx, dy = directions[key]
            step = 10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
            nodes = [list(node) for node in self._nodes]
            for index in self.selected_indices:
                for offset in (0, 2, 4):
                    nodes[index][offset] += dx * step
                    nodes[index][offset + 1] += dy * step
            self._commit_nodes(nodes)
            return True
        return False

    def context_menu(self, screen_position: QPoint, scene_position: QPointF) -> None:
        self.cancel_drag()
        self.setFocus()
        item = self._bound_item
        menu = QMenu(self.owner.gv)
        try:
            add = menu.addAction(QCoreApplication.translate('BubbleShapeControl', 'Add anchor'))
            delete = menu.addAction(QCoreApplication.translate('BubbleShapeControl', 'Delete selected anchors'))
            smooth = menu.addAction(QCoreApplication.translate('BubbleShapeControl', 'Make smooth'))
            corner = menu.addAction(QCoreApplication.translate('BubbleShapeControl', 'Make corner'))
            size = len(item.blk.bubble_shape.nodes)
            add.setEnabled(size < 64)
            delete.setEnabled(bool(self.selected_indices) and size - len(self.selected_indices) >= 3)
            smooth.setEnabled(bool(self.selected_indices))
            corner.setEnabled(bool(self.selected_indices))
            chosen = menu.exec_(screen_position)
            if self._bound_item is not item or not self.isVisible():
                return
            if chosen is add:
                self.insert_anchor(scene_position)
            elif chosen is delete:
                self.delete_selected()
            elif chosen is smooth or chosen is corner:
                self.convert_selected(chosen is smooth)
        finally:
            menu.deleteLater()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_indices.clear()
            self.refresh()
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.context_menu(event.screenPos(), event.scenePos())
        event.accept()

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.insert_anchor(event.scenePos())
        event.accept()
