"""Reusable Blender-style modal transforms for selected 2D points."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

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
        self._reset()

    @property
    def active(self) -> bool:
        return self.mode is not None

    def begin(
        self, mode: str, points: Sequence[QPointF], mouse: QPointF
    ) -> bool:
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
        self._rotation_degrees = 0.0
        return True

    def switch_mode(
        self, mode: str, mouse: QPointF
    ) -> Optional[Tuple[QPointF, ...]]:
        if not self.active or mode not in self.MODES:
            return None
        if mode == self.mode:
            return tuple(QPointF(point) for point in self.result_points)
        self.mode = mode
        self.axis = None
        self.start_mouse = QPointF(mouse)
        self.current_mouse = QPointF(mouse)
        self._rotation_degrees = 0.0
        # A mode switch restarts from begin()'s points; the owning control
        # resets its model preview too, so modal previews never compound.
        self.result_points = tuple(
            QPointF(point) for point in self.initial_points
        )
        return tuple(QPointF(point) for point in self.result_points)

    def constrain(
        self, axis: str, mouse: QPointF
    ) -> Optional[Tuple[QPointF, ...]]:
        if not self.active:
            return None
        valid_axes = (
            ('x', 'y', 'z')
            if self.mode == self.ROTATE
            else ('x', 'y')
        )
        if axis not in valid_axes:
            raise ValueError(
                f"axis must be one of {', '.join(valid_axes)}"
            )
        self.axis = axis
        self.start_mouse = QPointF(mouse)
        self.current_mouse = QPointF(mouse)
        self._rotation_degrees = 0.0
        # Changing the axis restarts the same operation-start transaction.
        self.result_points = tuple(
            QPointF(point) for point in self.initial_points
        )
        return tuple(QPointF(point) for point in self.result_points)

    def update(self, mouse: QPointF) -> Tuple[QPointF, ...]:
        if not self.active:
            return ()
        if self.mode == self.ROTATE:
            previous = self.current_mouse - self.origin
            current = QPointF(mouse) - self.origin
            if not previous.isNull() and not current.isNull():
                delta = (
                    math.atan2(current.y(), current.x())
                    - math.atan2(previous.y(), previous.x())
                )
                delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
                self._rotation_degrees += math.degrees(delta)
        self.current_mouse = QPointF(mouse)
        if self.mode == self.TRANSLATE:
            delta = self.current_mouse - self.start_mouse
            if self.axis == 'x':
                delta.setY(0.0)
            elif self.axis == 'y':
                delta.setX(0.0)
            result = tuple(point + delta for point in self.initial_points)
        elif self.mode == self.ROTATE:
            angle = math.radians(self.rotation_delta())
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
            factor = self.scale_factor()
            result = tuple(
                QPointF(
                    self.origin.x()
                    + (point.x() - self.origin.x())
                    * (factor if self.axis != 'y' else 1.0),
                    self.origin.y()
                    + (point.y() - self.origin.y())
                    * (factor if self.axis != 'x' else 1.0),
                )
                for point in self.initial_points
            )
        self.result_points = tuple(QPointF(point) for point in result)
        return tuple(QPointF(point) for point in self.result_points)

    def rotation_delta(self) -> float:
        """Return the current clockwise screen-space angle in degrees.

        >>> tool = ModalPointTransform()
        >>> tool.begin(tool.ROTATE, (QPointF(),), QPointF(1, 0))
        True
        >>> tool.update(QPointF(0, 1))
        (PyQt6.QtCore.QPointF(),)
        >>> round(tool.rotation_delta())
        90
        """
        return self._rotation_degrees

    def scale_factor(self) -> float:
        """Return the current radial scale relative to modal start."""
        start_distance = math.hypot(
            self.start_mouse.x() - self.origin.x(),
            self.start_mouse.y() - self.origin.y(),
        )
        current_distance = math.hypot(
            self.current_mouse.x() - self.origin.x(),
            self.current_mouse.y() - self.origin.y(),
        )
        return (
            current_distance / start_distance
            if start_distance > 1e-9
            else 1.0
        )

    def finish(self) -> Tuple[QPointF, ...]:
        if not self.active:
            return ()
        result = tuple(QPointF(point) for point in self.result_points)
        self._reset()
        return result

    def cancel(self) -> Tuple[QPointF, ...]:
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
        self._rotation_degrees = 0.0
