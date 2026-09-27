"""Canvas-owned modal translation of selected text items."""

from typing import List, Optional, Sequence, TYPE_CHECKING

from qtpy.QtCore import QObject, QPointF, Qt
from qtpy.QtGui import QColor, QCursor, QPainterPath, QPen, QTransform
from qtpy.QtWidgets import QGraphicsPathItem, QGraphicsSceneMouseEvent

from .commands import MoveBlkItemsCommand
from ..item import TextBlkItem
from ..shape_control import CONTROL_ITEM_DATA_KEY
from ..transforms.modal import ModalPointTransform

if TYPE_CHECKING:
    from ...canvas import Canvas


class TextItemMoveSession(QObject):
    """Preview item positions and publish one ordinary movement command.

    >>> TextItemMoveSession.__name__
    'TextItemMoveSession'
    """

    def __init__(self, canvas: "Canvas") -> None:
        super().__init__(canvas)
        self.canvas = canvas
        self.tool = ModalPointTransform()
        self.items: List[TextBlkItem] = []
        self._before: List[QPointF] = []
        self._scene_to_parent: List[QTransform] = []
        self._guide: Optional[QGraphicsPathItem] = None
        self._finish_button: Optional[Qt.MouseButton] = None
        self._restore_cursor: Optional[QCursor] = None

    @property
    def active(self) -> bool:
        return self.tool.active

    def start(self) -> bool:
        canvas = self.canvas
        items = canvas.selected_text_items()
        if (
            self.active
            or not canvas.textEditMode()
            or canvas.editing_textblkitem is not None
            or canvas.creating_textblock
            or canvas.path_reorder_active
            or canvas.mouseGrabberItem() is not None
            or not items
            or any(item.isEditing() for item in items)
        ):
            return False
        canvas.alpha_mask_edit_session.deactivate()
        self.items = items
        self._before = [item.logical_position() for item in items]
        parent_to_scene = [
            item.parentItem().sceneTransform()
            if item.parentItem() is not None else QTransform()
            for item in items
        ]
        # Movement is in scene axes; logical positions belong to each parent,
        # independently of paint padding, item rotation, and the effect stack.
        self._scene_to_parent = [matrix.inverted()[0] for matrix in parent_to_scene]
        points = [
            matrix.map(position)
            for matrix, position in zip(parent_to_scene, self._before)
        ]
        view = canvas.gv
        mouse = view.mapToScene(view.viewport().mapFromGlobal(QCursor.pos()))
        self.tool.begin(ModalPointTransform.TRANSLATE, points, mouse)
        self._restore_cursor = QCursor(view.viewport().cursor())
        self._guide = QGraphicsPathItem()
        pen = QPen(QColor(30, 147, 229), 1.25, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._guide.setPen(pen)
        self._guide.setData(CONTROL_ITEM_DATA_KEY, True)
        self._guide.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._guide.setZValue(200)
        canvas.addItem(self._guide)
        self._guide.grabMouse()
        self._update_indicator()
        return True

    def _preview(self, points: Sequence[QPointF]) -> None:
        for item, matrix, point in zip(self.items, self._scene_to_parent, points):
            # Match ordinary mouse dragging: keep the model untouched until
            # MoveBlkItemsCommand commits, including source geometry on cancel.
            item.setPos(item.pos() + matrix.map(point) - item.logical_position())
            item.moving.emit(item)

    def _update_indicator(self) -> None:
        path = QPainterPath()
        cursor = Qt.CursorShape.SizeAllCursor
        axis = self.tool.axis
        if axis is not None:
            view = self.canvas.gv
            visible = view.mapToScene(view.viewport().rect()).boundingRect()
            origin = self.tool.origin
            if axis == 'x':
                path.moveTo(visible.left(), origin.y())
                path.lineTo(visible.right(), origin.y())
                cursor = Qt.CursorShape.SizeHorCursor
            else:
                path.moveTo(origin.x(), visible.top())
                path.lineTo(origin.x(), visible.bottom())
                cursor = Qt.CursorShape.SizeVerCursor
        self._guide.setPath(path)
        self._guide.setCursor(cursor)
        self.canvas.gv.viewport().setCursor(cursor)

    def _release_mouse(self) -> None:
        if self._guide is not None:
            if self.canvas.mouseGrabberItem() is self._guide:
                self._guide.ungrabMouse()
            self.canvas.removeItem(self._guide)
            self._guide = None
        if self._restore_cursor is not None:
            self.canvas.gv.viewport().setCursor(self._restore_cursor)
            self._restore_cursor = None
        self._finish_button = None

    def _finish(self, commit: bool, *, release_mouse: bool = True) -> None:
        items, before = self.items, self._before
        after = [item.logical_position() for item in items]
        if commit:
            self.tool.finish()
        else:
            self._preview(self.tool.cancel())
        self.items = []
        self._before = []
        self._scene_to_parent = []
        self._guide.setPath(QPainterPath())
        if release_mouse:
            self._release_mouse()
        if commit:
            if before != after:
                self.canvas.push_text_command(MoveBlkItemsCommand(items, before, after))

    def cancel(self) -> None:
        if self.active:
            self._finish(False)
        else:
            self._release_mouse()

    def handle_shortcut(
        self, key: int, modifiers: Qt.KeyboardModifier
    ) -> bool:
        if not self.active:
            return False
        if modifiers != Qt.KeyboardModifier.NoModifier:
            return True
        if key == Qt.Key.Key_Escape:
            self.cancel()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish(True)
        elif key in (Qt.Key.Key_X, Qt.Key.Key_Y):
            axis = 'x' if key == Qt.Key.Key_X else 'y'
            self._preview(self.tool.constrain(axis, self.tool.current_mouse))
            self._update_indicator()
        elif key in (Qt.Key.Key_R, Qt.Key.Key_S):
            self.cancel()
            return False
        return True

    def handle_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        if self._finish_button is not None:
            return True
        if not self.active:
            return False
        self._preview(self.tool.update(event.scenePos()))
        self._update_indicator()
        return True

    def handle_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        if not self.active:
            return False
        button = event.button()
        if button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            # Keep the grab through release so confirmation cannot start a
            # selection drag, and cancellation cannot open the context menu.
            self._finish(button == Qt.MouseButton.LeftButton, release_mouse=False)
            self._finish_button = button
        return True

    def handle_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        if self._finish_button is not None:
            if event.button() == self._finish_button:
                self._release_mouse()
            return True
        return self.active
