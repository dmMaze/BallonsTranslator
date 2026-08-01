"""Reusable Blender-style modal transforms for selected 2D points."""

import math

from qtpy.QtCore import QPointF


class ModalPointTransform:
    """Transform a frozen point selection relative to mouse movement.

    The caller owns event routing, preview/commit transactions, coordinate
    conversion, cursors, and indicators. This class only owns modal math.

    >>> tool = ModalPointTransform()
    >>> tool.begin(tool.TRANSLATE, (QPointF(2, 3),), QPointF(10, 10))
    True
    >>> point = tool.update(QPointF(14, 12))[0]
    >>> (point.x(), point.y())
    (6.0, 5.0)
    """

    TRANSLATE = 'translate'
    ROTATE = 'rotate'
    SCALE = 'scale'
    MODES = frozenset((TRANSLATE, ROTATE, SCALE))

    def __init__(self) -> None:
        self.mode = None
        self.axis = None
        self.initial_points = ()
        self.result_points = ()
        self.origin = QPointF()
        self.start_mouse = QPointF()
        self.current_mouse = QPointF()

    @property
    def active(self) -> bool:
        return self.mode is not None

    def begin(self, mode: str, points, mouse: QPointF) -> bool:
        if mode not in self.MODES or self.active:
            return False
        points = tuple(QPointF(point) for point in points)
        if not points:
            return False
        self.initial_points = points
        self.result_points = tuple(QPointF(point) for point in points)
        self.origin = sum(points, QPointF()) / len(points)
        self.mode = mode
        self.axis = None
        self.start_mouse = QPointF(mouse)
        self.current_mouse = QPointF(mouse)
        return True

    def switch_mode(self, mode: str, mouse: QPointF):
        if not self.active or mode not in self.MODES:
            return None
        if mode == self.mode:
            return tuple(QPointF(point) for point in self.result_points)
        self.mode = mode
        self.axis = None
        self.start_mouse = QPointF(mouse)
        self.current_mouse = QPointF(mouse)
        self.result_points = tuple(
            QPointF(point) for point in self.initial_points
        )
        return tuple(QPointF(point) for point in self.result_points)

    def constrain(self, axis: str, mouse: QPointF):
        if not self.active or self.mode != self.TRANSLATE:
            return None
        if axis not in ('x', 'y'):
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis
        self.start_mouse = QPointF(mouse)
        self.current_mouse = QPointF(mouse)
        self.result_points = tuple(
            QPointF(point) for point in self.initial_points
        )
        return tuple(QPointF(point) for point in self.result_points)

    def update(self, mouse: QPointF):
        if not self.active:
            return ()
        self.current_mouse = QPointF(mouse)
        if self.mode == self.TRANSLATE:
            delta = self.current_mouse - self.start_mouse
            if self.axis == 'x':
                delta.setY(0.0)
            elif self.axis == 'y':
                delta.setX(0.0)
            result = tuple(point + delta for point in self.initial_points)
        elif self.mode == self.ROTATE:
            start = self.start_mouse - self.origin
            current = self.current_mouse - self.origin
            if start.isNull() or current.isNull():
                cosine, sine = 1.0, 0.0
            else:
                start_angle = math.atan2(start.y(), start.x())
                current_angle = math.atan2(current.y(), current.x())
                angle = current_angle - start_angle
                cosine, sine = math.cos(angle), math.sin(angle)
            result = tuple(
                QPointF(
                    self.origin.x()
                    + cosine * (point.x() - self.origin.x())
                    - sine * (point.y() - self.origin.y()),
                    self.origin.y()
                    + sine * (point.x() - self.origin.x())
                    + cosine * (point.y() - self.origin.y()),
                )
                for point in self.initial_points
            )
        else:
            start_distance = math.hypot(
                self.start_mouse.x() - self.origin.x(),
                self.start_mouse.y() - self.origin.y(),
            )
            current_distance = math.hypot(
                self.current_mouse.x() - self.origin.x(),
                self.current_mouse.y() - self.origin.y(),
            )
            factor = (
                current_distance / start_distance
                if start_distance > 1e-9
                else 1.0
            )
            result = tuple(
                self.origin + (point - self.origin) * factor
                for point in self.initial_points
            )
        self.result_points = tuple(QPointF(point) for point in result)
        return tuple(QPointF(point) for point in self.result_points)

    def finish(self):
        if not self.active:
            return ()
        result = tuple(QPointF(point) for point in self.result_points)
        self._reset()
        return result

    def cancel(self):
        if not self.active:
            return ()
        result = tuple(QPointF(point) for point in self.initial_points)
        self._reset()
        return result

    def _reset(self) -> None:
        self.mode = None
        self.axis = None
        self.initial_points = ()
        self.result_points = ()
        self.origin = QPointF()
        self.start_mouse = QPointF()
        self.current_mouse = QPointF()
