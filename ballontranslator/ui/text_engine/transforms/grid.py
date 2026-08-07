"""Invertible free-form grid mapping for transformed text surfaces."""

import math
import threading

import numpy as np
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPainterPath

from ballontranslator.utils.fontformat import GridTextTransform
from ballontranslator.utils.logger import logger as LOGGER


_numba_inverse_grid_arrays = None
_numba_backend_checked = False


def start_grid_numba_warmup() -> threading.Thread:
    """Load or compile Grid kernels outside the Qt event thread."""
    def warmup() -> None:
        try:
            from .grid_numba import warm_grid_numba_cache
            warm_grid_numba_cache()
            LOGGER.info('Grid transform acceleration is ready.')
        except Exception as error:
            LOGGER.warning(
                f'Grid transform acceleration is unavailable: {error}'
            )

    thread = threading.Thread(
        target=warmup,
        name='GridNumbaWarmup',
        daemon=True,
    )
    thread.start()
    return thread


def _compiled_inverse_grid_arrays(*args, **kwargs):
    """Use the warmed backend without making Grid depend on startup timing."""
    global _numba_backend_checked, _numba_inverse_grid_arrays
    if not _numba_backend_checked:
        _numba_backend_checked = True
        try:
            from .grid_numba import inverse_grid_arrays
        except Exception:
            inverse_grid_arrays = None
        _numba_inverse_grid_arrays = inverse_grid_arrays
    if _numba_inverse_grid_arrays is None:
        return None
    return _numba_inverse_grid_arrays(*args, **kwargs)


class GridMapper:
    """Map a logical rectangle through a bilinear or Catmull-Rom grid.

    Control points are normalized against ``logical_rect`` so formatting and
    layout changes resize the deformation with the text instead of detaching
    it from the new geometry.

    >>> mapper = GridMapper(
    ...     QRectF(0, 0, 100, 50), QRectF(0, 0, 100, 50),
    ...     GridTextTransform(),
    ... )
    >>> point = mapper.forward_point(QPointF(25, 20))
    >>> (point.x(), point.y())
    (25.0, 20.0)
    """

    INVERSE_ITERATIONS = 12
    INVERSE_TOLERANCE = 1e-5
    INVERSE_CONVERGENCE_TOLERANCE = 1e-9
    # Float32 raster maps cannot reliably reach the scalar solver's cutoff.
    INVERSE_ARRAY_CONVERGENCE_TOLERANCE = 1e-6
    OUTLINE_SAMPLES_PER_CELL = 8
    CATMULL_ROM_RANGE_OVERSHOOT = 0.28125

    def __init__(
        self,
        logical_rect: QRectF,
        source_rect: QRectF,
        transform: GridTextTransform,
    ) -> None:
        self.logical_rect = QRectF(logical_rect)
        self.source_rect = QRectF(source_rect)
        self.transform = transform
        if self.logical_rect.width() <= 0.0 or self.logical_rect.height() <= 0.0:
            raise ValueError('grid rectangle must have positive dimensions')
        self.horizontal = self.transform.horizontal_divisions
        self.vertical = self.transform.vertical_divisions
        self.points = np.asarray(
            self.transform.control_points, dtype=np.float64
        ).reshape(self.vertical + 1, self.horizontal + 1, 2)
        # Keep exact interaction geometry while making dense raster work float32.
        self._array_points = self.points.astype(np.float32)
        self._catmull_rom_points = (
            self._pad_catmull_rom_points(self.points)
            if self.transform.interpolation == 'catmull_rom'
            else None
        )
        self._array_catmull_rom_points = (
            self._pad_catmull_rom_points(self._array_points)
            if self.transform.interpolation == 'catmull_rom'
            else None
        )

    @staticmethod
    def _pad_catmull_rom_points(points):
        rows, columns, _ = points.shape
        padded = np.empty(
            (rows + 2, columns + 2, 2), dtype=points.dtype
        )
        padded[1:-1, 1:-1] = points
        padded[1:-1, 0] = 2.0 * points[:, 0] - points[:, 1]
        padded[1:-1, -1] = 2.0 * points[:, -1] - points[:, -2]
        padded[0] = 2.0 * padded[1] - padded[2]
        padded[-1] = 2.0 * padded[-2] - padded[-3]
        return padded

    @property
    def geometry_key(self):
        rect = self.logical_rect
        source = self.source_rect
        return (
            type(self),
            self.transform,
            rect.x(), rect.y(), rect.width(), rect.height(),
            source.x(), source.y(), source.width(), source.height(),
        )

    def _source_to_normalized(self, x, y):
        return (
            (x - self.logical_rect.left()) / self.logical_rect.width(),
            (y - self.logical_rect.top()) / self.logical_rect.height(),
        )

    def _normalized_to_source(self, x, y):
        return (
            self.logical_rect.left() + x * self.logical_rect.width(),
            self.logical_rect.top() + y * self.logical_rect.height(),
        )

    @staticmethod
    def _catmull_rom_polynomial(
        p0, p1, p2, p3, value, *, derivative=False
    ):
        """Evaluate one uniform Catmull-Rom segment and optional derivative.

        >>> points = tuple(np.array([x, 0.0]) for x in range(4))
        >>> value, slope = GridMapper._catmull_rom_polynomial(
        ...     *points, np.asarray(0.5), derivative=True
        ... )
        >>> (value.tolist(), slope.tolist())
        ([1.5, 0.0], [1.0, 0.0])
        """
        value = np.asarray(value, dtype=p0.dtype)[..., None]
        linear = p2 - p0
        quadratic = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
        cubic = -p0 + 3.0 * p1 - 3.0 * p2 + p3
        output = 0.5 * (
            2.0 * p1
            + value * (linear + value * (quadratic + value * cubic))
        )
        if not derivative:
            return output
        slope = 0.5 * (
            linear
            + value * (2.0 * quadratic + value * 3.0 * cubic)
        )
        return output, slope

    def _cell_coordinates(self, x, y):
        scaled_x = x * self.horizontal
        scaled_y = y * self.vertical
        column = np.clip(
            np.floor(scaled_x).astype(np.int64),
            0,
            self.horizontal - 1,
        )
        row = np.clip(
            np.floor(scaled_y).astype(np.int64),
            0,
            self.vertical - 1,
        )
        return column, row, scaled_x - column, scaled_y - row

    def _evaluate_bilinear(self, x, y):
        column, row, local_x, local_y = self._cell_coordinates(x, y)
        points = self._array_points if x.dtype == np.float32 else self.points
        p00 = points[row, column]
        p10 = points[row, column + 1]
        p01 = points[row + 1, column]
        p11 = points[row + 1, column + 1]
        wx = local_x[..., None]
        wy = local_y[..., None]
        linear_x = p10 - p00
        linear_y = p01 - p00
        cross = p11 - p10 - p01 + p00
        output = p00 + linear_x * wx + linear_y * wy + cross * wx * wy
        derivative_x = (linear_x + cross * wy) * self.horizontal
        derivative_y = (linear_y + cross * wx) * self.vertical
        return output, derivative_x, derivative_y

    def _evaluate_catmull_rom(self, x, y):
        column, row, local_x, local_y = self._cell_coordinates(x, y)
        catmull_rom_points = (
            self._array_catmull_rom_points
            if x.dtype == np.float32
            else self._catmull_rom_points
        )
        rows = []
        row_derivatives = []
        for y_index in range(4):
            points = tuple(
                catmull_rom_points[
                    row + y_index, column + x_index
                ]
                for x_index in range(4)
            )
            row_output, row_derivative = self._catmull_rom_polynomial(
                *points, local_x, derivative=True
            )
            rows.append(row_output)
            row_derivatives.append(row_derivative)
        output, derivative_y = self._catmull_rom_polynomial(
            *rows, local_y, derivative=True
        )
        derivative_x = self._catmull_rom_polynomial(
            *row_derivatives, local_y
        )
        derivative_x *= self.horizontal
        derivative_y *= self.vertical
        return output, derivative_x, derivative_y

    def _evaluate(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        dtype = (
            np.float32
            if x.dtype == np.float32 and y.dtype == np.float32
            else np.float64
        )
        x = x.astype(dtype, copy=False)
        y = y.astype(dtype, copy=False)
        if self.transform.interpolation == 'catmull_rom':
            return self._evaluate_catmull_rom(x, y)
        return self._evaluate_bilinear(x, y)

    def forward_point(self, source: QPointF) -> QPointF:
        normalized_x, normalized_y = self._source_to_normalized(
            source.x(), source.y()
        )
        output, _dx, _dy = self._evaluate(normalized_x, normalized_y)
        visual_x, visual_y = self._normalized_to_source(
            float(output[0]), float(output[1])
        )
        return QPointF(visual_x, visual_y)

    def forward_arrays(self, source_x, source_y):
        normalized_x, normalized_y = self._source_to_normalized(
            source_x, source_y
        )
        output, _dx, _dy = self._evaluate(normalized_x, normalized_y)
        return self._normalized_to_source(
            output[..., 0], output[..., 1]
        )

    def _inverse_normalized(
        self,
        visual_x,
        visual_y,
        initial_x,
        initial_y,
        clamp_bounds=None,
    ):
        dense = np.ndim(initial_x) > 0 or np.ndim(initial_y) > 0
        dtype = np.float32 if dense else np.float64
        visual_x = np.asarray(visual_x, dtype=dtype)
        visual_y = np.asarray(visual_y, dtype=dtype)
        source_x = np.asarray(initial_x, dtype=dtype).copy()
        source_y = np.asarray(initial_y, dtype=dtype).copy()
        if clamp_bounds is not None:
            source_left, source_top, source_right, source_bottom = clamp_bounds
            np.clip(source_x, source_left, source_right, out=source_x)
            np.clip(source_y, source_top, source_bottom, out=source_y)
        finite = np.isfinite(source_x) & np.isfinite(source_y)
        tolerance_squared = self.INVERSE_TOLERANCE ** 2
        convergence_tolerance = (
            self.INVERSE_ARRAY_CONVERGENCE_TOLERANCE
            if dense
            else self.INVERSE_CONVERGENCE_TOLERANCE
        )
        convergence_squared = convergence_tolerance ** 2
        residual_squared = None
        for _iteration in range(self.INVERSE_ITERATIONS):
            output, derivative_x, derivative_y = self._evaluate(
                source_x, source_y
            )
            residual_x = output[..., 0] - visual_x
            residual_y = output[..., 1] - visual_y
            residual_squared = (
                residual_x * residual_x + residual_y * residual_y
            )
            determinant = (
                derivative_x[..., 0] * derivative_y[..., 1]
                - derivative_x[..., 1] * derivative_y[..., 0]
            )
            stable = np.isfinite(determinant) & (np.abs(determinant) > 1e-10)
            finite &= stable
            if np.all(
                (~finite) | (residual_squared <= convergence_squared)
            ):
                break
            safe = np.where(stable, determinant, 1.0)
            delta_x = (
                residual_x * derivative_y[..., 1]
                - residual_y * derivative_y[..., 0]
            ) / safe
            delta_y = (
                derivative_x[..., 0] * residual_y
                - derivative_x[..., 1] * residual_x
            ) / safe
            if dense:
                np.clip(delta_x, -0.5, 0.5, out=delta_x)
                np.clip(delta_y, -0.5, 0.5, out=delta_y)
            else:
                delta_x = np.clip(delta_x, -0.5, 0.5)
                delta_y = np.clip(delta_y, -0.5, 0.5)
            source_x -= delta_x
            source_y -= delta_y
            if clamp_bounds is not None:
                np.clip(source_x, source_left, source_right, out=source_x)
                np.clip(source_y, source_top, source_bottom, out=source_y)
        else:
            output, _dx, _dy = self._evaluate(source_x, source_y)
            residual_x = output[..., 0] - visual_x
            residual_y = output[..., 1] - visual_y
            residual_squared = (
                residual_x * residual_x + residual_y * residual_y
            )
        finite &= (
            np.isfinite(residual_squared)
            & (residual_squared <= tolerance_squared)
        )
        return source_x, source_y, finite

    def _retry_bilinear_inverse(
        self, visual_x, visual_y, source_x, source_y, valid
    ):
        """Retry failed dense points inside their neighboring bilinear cells.

        >>> GridMapper.INVERSE_TOLERANCE > 0.0
        True
        """
        failed = np.flatnonzero(~valid.ravel())
        if failed.size == 0:
            return source_x, source_y, valid
        flat_visual_x = np.asarray(visual_x).ravel()
        flat_visual_y = np.asarray(visual_y).ravel()
        flat_source_x = source_x.ravel()
        flat_source_y = source_y.ravel()
        flat_valid = valid.ravel()
        base_columns = np.clip(
            np.floor(flat_source_x[failed] * self.horizontal).astype(np.int64),
            0,
            self.horizontal - 1,
        )
        base_rows = np.clip(
            np.floor(flat_source_y[failed] * self.vertical).astype(np.int64),
            0,
            self.vertical - 1,
        )
        resolved = np.zeros(failed.shape, dtype=bool)
        points = self._array_points
        guard = np.float32(1e-5)
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                pending = ~resolved
                if not np.any(pending):
                    return source_x, source_y, valid
                columns = np.clip(
                    base_columns + column_offset,
                    0,
                    self.horizontal - 1,
                )
                rows = np.clip(
                    base_rows + row_offset,
                    0,
                    self.vertical - 1,
                )
                cell_points = np.stack((
                    points[rows, columns],
                    points[rows, columns + 1],
                    points[rows + 1, columns],
                    points[rows + 1, columns + 1],
                ), axis=1)
                targets_x = flat_visual_x[failed]
                targets_y = flat_visual_y[failed]
                candidates = pending & (
                    (targets_x >= cell_points[:, :, 0].min(axis=1) - guard)
                    & (targets_x <= cell_points[:, :, 0].max(axis=1) + guard)
                    & (targets_y >= cell_points[:, :, 1].min(axis=1) - guard)
                    & (targets_y <= cell_points[:, :, 1].max(axis=1) + guard)
                )
                if not np.any(candidates):
                    continue
                candidate_indices = np.flatnonzero(candidates)
                candidate_columns = columns[candidates]
                candidate_rows = rows[candidates]
                p00 = cell_points[candidates, 0]
                linear_x = cell_points[candidates, 1] - p00
                linear_y = cell_points[candidates, 2] - p00
                offset = np.stack((
                    targets_x[candidates] - p00[:, 0],
                    targets_y[candidates] - p00[:, 1],
                ), axis=1)
                determinant = (
                    linear_x[:, 0] * linear_y[:, 1]
                    - linear_x[:, 1] * linear_y[:, 0]
                )
                stable = np.abs(determinant) > 1e-10
                safe = np.where(stable, determinant, 1.0)
                local_x = np.where(
                    stable,
                    (offset[:, 0] * linear_y[:, 1]
                     - offset[:, 1] * linear_y[:, 0]) / safe,
                    0.5,
                )
                local_y = np.where(
                    stable,
                    (linear_x[:, 0] * offset[:, 1]
                     - linear_x[:, 1] * offset[:, 0]) / safe,
                    0.5,
                )
                left = candidate_columns / self.horizontal
                top = candidate_rows / self.vertical
                right = (candidate_columns + 1) / self.horizontal
                bottom = (candidate_rows + 1) / self.vertical
                initial_x = left + np.clip(local_x, 0.0, 1.0) / self.horizontal
                initial_y = top + np.clip(local_y, 0.0, 1.0) / self.vertical
                restored_x, restored_y, cell_valid = self._inverse_normalized(
                    targets_x[candidates],
                    targets_y[candidates],
                    initial_x,
                    initial_y,
                    (left, top, right, bottom),
                )
                accepted = candidate_indices[cell_valid]
                if accepted.size == 0:
                    continue
                output_indices = failed[accepted]
                flat_source_x[output_indices] = restored_x[cell_valid]
                flat_source_y[output_indices] = restored_y[cell_valid]
                flat_valid[output_indices] = True
                resolved[accepted] = True
        return source_x, source_y, valid

    def inverse_point(
        self,
        visual: QPointF,
        previous_source: QPointF = None,
        *,
        extrapolate: bool = False,
    ) -> QPointF:
        visual_x, visual_y = self._source_to_normalized(
            visual.x(), visual.y()
        )
        if previous_source is None:
            initial_x, initial_y = visual_x, visual_y
        else:
            initial_x, initial_y = self._source_to_normalized(
                previous_source.x(), previous_source.y()
            )
        source_x, source_y, valid = self._inverse_normalized(
            visual_x,
            visual_y,
            initial_x,
            initial_y,
            None if extrapolate else self._normalized_source_bounds(),
        )
        if not bool(valid) and previous_source is not None:
            return QPointF(previous_source)
        x, y = self._normalized_to_source(float(source_x), float(source_y))
        return QPointF(x, y)

    def inverse_arrays(self, visual_x, visual_y, *, return_valid=False):
        normalized_x, normalized_y = self._source_to_normalized(
            visual_x, visual_y
        )
        points = (
            self._array_catmull_rom_points
            if self.transform.interpolation == 'catmull_rom'
            else self._array_points
        )
        source_bounds = self._normalized_source_bounds()
        compiled = _compiled_inverse_grid_arrays(
            points,
            self.horizontal,
            self.vertical,
            normalized_x,
            normalized_y,
            catmull_rom=self.transform.interpolation == 'catmull_rom',
            source_bounds=source_bounds,
        )
        if compiled is None:
            source_x, source_y, valid = self._inverse_normalized(
                normalized_x,
                normalized_y,
                normalized_x,
                normalized_y,
                None,
            )
            source_left, source_top, source_right, source_bottom = source_bounds
            source_epsilon = 4.0 * self.INVERSE_ARRAY_CONVERGENCE_TOLERANCE
            valid &= (
                (source_x >= source_left - source_epsilon)
                & (source_x <= source_right + source_epsilon)
                & (source_y >= source_top - source_epsilon)
                & (source_y <= source_bottom + source_epsilon)
            )
            if self.transform.interpolation == 'bilinear' and not np.all(valid):
                source_x, source_y, valid = self._retry_bilinear_inverse(
                    normalized_x,
                    normalized_y,
                    source_x,
                    source_y,
                    valid,
                )
        else:
            source_x, source_y, valid = compiled
        source_left, source_top, source_right, source_bottom = source_bounds
        source_epsilon = 4.0 * self.INVERSE_ARRAY_CONVERGENCE_TOLERANCE
        valid &= (
            (source_x >= source_left - source_epsilon)
            & (source_x <= source_right + source_epsilon)
            & (source_y >= source_top - source_epsilon)
            & (source_y <= source_bottom + source_epsilon)
        )
        x, y = self._normalized_to_source(source_x, source_y)
        if return_valid:
            return x, y, valid
        return x, y

    def _normalized_source_bounds(self):
        source_left, source_top = self._source_to_normalized(
            self.source_rect.left(), self.source_rect.top()
        )
        source_right, source_bottom = self._source_to_normalized(
            self.source_rect.right(), self.source_rect.bottom()
        )
        return source_left, source_top, source_right, source_bottom

    def control_source_points(self):
        return tuple(
            QPointF(*self._normalized_to_source(
                column / self.horizontal,
                row / self.vertical,
            ))
            for row in range(self.vertical + 1)
            for column in range(self.horizontal + 1)
        )

    def normalized_output_delta(self, delta: QPointF) -> QPointF:
        return QPointF(
            delta.x() / self.logical_rect.width(),
            delta.y() / self.logical_rect.height(),
        )

    def map_rect_path(self, rect: QRectF) -> QPainterPath:
        rect = QRectF(rect).normalized()
        if self.transform.interpolation == 'bilinear':
            x_coordinates = [rect.left()]
            x_coordinates.extend(
                self.logical_rect.left()
                + self.logical_rect.width() * column / self.horizontal
                for column in range(1, self.horizontal)
                if rect.left()
                < self.logical_rect.left()
                + self.logical_rect.width() * column / self.horizontal
                < rect.right()
            )
            x_coordinates.append(rect.right())
            y_coordinates = [rect.top()]
            y_coordinates.extend(
                self.logical_rect.top()
                + self.logical_rect.height() * row / self.vertical
                for row in range(1, self.vertical)
                if rect.top()
                < self.logical_rect.top()
                + self.logical_rect.height() * row / self.vertical
                < rect.bottom()
            )
            y_coordinates.append(rect.bottom())
            source_points = (
                [QPointF(x, rect.top()) for x in x_coordinates[:-1]]
                + [QPointF(rect.right(), y) for y in y_coordinates[:-1]]
                + [
                    QPointF(x, rect.bottom())
                    for x in reversed(x_coordinates[1:])
                ]
                + [
                    QPointF(rect.left(), y)
                    for y in reversed(y_coordinates[1:])
                ]
            )
        else:
            segments_x = max(
                1, self.horizontal * self.OUTLINE_SAMPLES_PER_CELL
            )
            segments_y = max(
                1, self.vertical * self.OUTLINE_SAMPLES_PER_CELL
            )
            source_points = []
            for index in range(segments_x):
                source_points.append(QPointF(
                    rect.left() + rect.width() * index / segments_x,
                    rect.top(),
                ))
            for index in range(segments_y):
                source_points.append(QPointF(
                    rect.right(),
                    rect.top() + rect.height() * index / segments_y,
                ))
            for index in range(segments_x):
                source_points.append(QPointF(
                    rect.right() - rect.width() * index / segments_x,
                    rect.bottom(),
                ))
            for index in range(segments_y):
                source_points.append(QPointF(
                    rect.left(),
                    rect.bottom() - rect.height() * index / segments_y,
                ))
        coordinates = np.asarray(
            [(point.x(), point.y()) for point in source_points],
            dtype=np.float64,
        )
        mapped_x, mapped_y = self.forward_arrays(
            coordinates[:, 0], coordinates[:, 1]
        )
        points = tuple(
            QPointF(float(x), float(y))
            for x, y in zip(mapped_x, mapped_y)
        )
        path = QPainterPath()
        if points:
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        return path

    def visual_bounds(self, source_rect: QRectF = None) -> QRectF:
        bounds = self.map_rect_path(
            self.source_rect if source_rect is None else source_rect
        ).boundingRect()
        minimum = self.points.min(axis=(0, 1))
        maximum = self.points.max(axis=(0, 1))
        if self.transform.interpolation == 'catmull_rom':
            # Catmull-Rom reproduces the regular identity lattice exactly, so
            # Only handle displacement can contribute Catmull-Rom overshoot.
            displacement = self.points.copy()
            displacement[:, :, 0] -= np.linspace(
                0.0, 1.0, self.horizontal + 1
            )
            displacement[:, :, 1] -= np.linspace(
                0.0, 1.0, self.vertical + 1
            )[:, None]
            minimum = displacement.min(axis=(0, 1))
            maximum = displacement.max(axis=(0, 1))
            expansion = (
                maximum - minimum
            ) * self.CATMULL_ROM_RANGE_OVERSHOOT
            minimum -= expansion
            maximum += expansion
            maximum += (1.0, 1.0)
        left, top = self._normalized_to_source(*minimum)
        right, bottom = self._normalized_to_source(*maximum)
        return bounds.united(
            QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        )

    def local_tangent(self, source: QPointF) -> QPointF:
        step = max(
            self.logical_rect.width(), self.logical_rect.height()
        ) * 1e-5
        start = self.forward_point(source)
        end = self.forward_point(QPointF(source.x() + step, source.y()))
        tangent = end - start
        length = math.hypot(tangent.x(), tangent.y())
        return tangent / length if length else QPointF(1.0, 0.0)
