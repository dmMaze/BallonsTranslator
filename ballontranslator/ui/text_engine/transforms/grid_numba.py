"""Cached parallel inverse kernels for Grid text transforms."""

import math
import os
import os.path as osp
import threading

from ballontranslator.utils import shared


NUMBA_CACHE_DIR = osp.join(shared.cache_dir, 'numba')
# Configure the documented cache boundary before Numba reads its environment.
os.environ['NUMBA_CACHE_DIR'] = NUMBA_CACHE_DIR

import numpy as np
from numba import njit, prange


_warmup_lock = threading.Lock()
_warmup_complete = False


@njit(inline='always')
def _catmull_rom(p0, p1, p2, p3, value):
    linear = p2 - p0
    quadratic = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
    cubic = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    output = 0.5 * (
        2.0 * p1
        + value * (linear + value * (quadratic + value * cubic))
    )
    slope = 0.5 * (
        linear + value * (2.0 * quadratic + value * 3.0 * cubic)
    )
    return output, slope


@njit(inline='always')
def _evaluate_bilinear(points, horizontal, vertical, x, y):
    scaled_x = x * horizontal
    scaled_y = y * vertical
    column = min(max(int(math.floor(scaled_x)), 0), horizontal - 1)
    row = min(max(int(math.floor(scaled_y)), 0), vertical - 1)
    local_x = scaled_x - column
    local_y = scaled_y - row
    p00x = points[row, column, 0]
    p00y = points[row, column, 1]
    linear_xx = points[row, column + 1, 0] - p00x
    linear_xy = points[row, column + 1, 1] - p00y
    linear_yx = points[row + 1, column, 0] - p00x
    linear_yy = points[row + 1, column, 1] - p00y
    cross_x = (
        points[row + 1, column + 1, 0]
        - points[row, column + 1, 0]
        - points[row + 1, column, 0]
        + p00x
    )
    cross_y = (
        points[row + 1, column + 1, 1]
        - points[row, column + 1, 1]
        - points[row + 1, column, 1]
        + p00y
    )
    output_x = (
        p00x + linear_xx * local_x + linear_yx * local_y
        + cross_x * local_x * local_y
    )
    output_y = (
        p00y + linear_xy * local_x + linear_yy * local_y
        + cross_y * local_x * local_y
    )
    derivative_xx = (linear_xx + cross_x * local_y) * horizontal
    derivative_xy = (linear_xy + cross_y * local_y) * horizontal
    derivative_yx = (linear_yx + cross_x * local_x) * vertical
    derivative_yy = (linear_yy + cross_y * local_x) * vertical
    return (
        output_x, output_y,
        derivative_xx, derivative_xy,
        derivative_yx, derivative_yy,
    )


@njit(inline='always')
def _evaluate_catmull_rom(points, horizontal, vertical, x, y):
    scaled_x = x * horizontal
    scaled_y = y * vertical
    column = min(max(int(math.floor(scaled_x)), 0), horizontal - 1)
    row = min(max(int(math.floor(scaled_y)), 0), vertical - 1)
    local_x = scaled_x - column
    local_y = scaled_y - row

    row_x0 = row_x1 = row_x2 = row_x3 = 0.0
    row_y0 = row_y1 = row_y2 = row_y3 = 0.0
    derivative_x0 = derivative_x1 = derivative_x2 = derivative_x3 = 0.0
    derivative_y0 = derivative_y1 = derivative_y2 = derivative_y3 = 0.0
    for offset_y in range(4):
        index_y = row + offset_y
        value_x, slope_x = _catmull_rom(
            points[index_y, column, 0],
            points[index_y, column + 1, 0],
            points[index_y, column + 2, 0],
            points[index_y, column + 3, 0],
            local_x,
        )
        value_y, slope_y = _catmull_rom(
            points[index_y, column, 1],
            points[index_y, column + 1, 1],
            points[index_y, column + 2, 1],
            points[index_y, column + 3, 1],
            local_x,
        )
        if offset_y == 0:
            row_x0, row_y0 = value_x, value_y
            derivative_x0, derivative_y0 = slope_x, slope_y
        elif offset_y == 1:
            row_x1, row_y1 = value_x, value_y
            derivative_x1, derivative_y1 = slope_x, slope_y
        elif offset_y == 2:
            row_x2, row_y2 = value_x, value_y
            derivative_x2, derivative_y2 = slope_x, slope_y
        else:
            row_x3, row_y3 = value_x, value_y
            derivative_x3, derivative_y3 = slope_x, slope_y

    output_x, derivative_yx = _catmull_rom(
        row_x0, row_x1, row_x2, row_x3, local_y
    )
    output_y, derivative_yy = _catmull_rom(
        row_y0, row_y1, row_y2, row_y3, local_y
    )
    derivative_xx, _unused = _catmull_rom(
        derivative_x0, derivative_x1,
        derivative_x2, derivative_x3,
        local_y,
    )
    derivative_xy, _unused = _catmull_rom(
        derivative_y0, derivative_y1,
        derivative_y2, derivative_y3,
        local_y,
    )
    return (
        output_x, output_y,
        derivative_xx * horizontal, derivative_xy * horizontal,
        derivative_yx * vertical, derivative_yy * vertical,
    )


@njit(inline='always')
def _inverse_point(
    points,
    horizontal,
    vertical,
    visual_x,
    visual_y,
    catmull_rom,
):
    source_x = visual_x
    source_y = visual_y
    finite = math.isfinite(source_x) and math.isfinite(source_y)
    residual_squared = math.inf
    converged = False
    for _iteration in range(12):
        values = (
            _evaluate_catmull_rom(
                points, horizontal, vertical, source_x, source_y
            )
            if catmull_rom
            else _evaluate_bilinear(
                points, horizontal, vertical, source_x, source_y
            )
        )
        residual_x = values[0] - visual_x
        residual_y = values[1] - visual_y
        residual_squared = residual_x * residual_x + residual_y * residual_y
        determinant = values[2] * values[5] - values[3] * values[4]
        stable = math.isfinite(determinant) and abs(determinant) > 1e-10
        finite = finite and stable
        if not finite:
            break
        if residual_squared <= 1e-12:
            converged = True
            break
        delta_x = (
            residual_x * values[5] - residual_y * values[4]
        ) / determinant
        delta_y = (
            values[2] * residual_y - values[3] * residual_x
        ) / determinant
        source_x -= min(max(delta_x, -0.5), 0.5)
        source_y -= min(max(delta_y, -0.5), 0.5)

    if finite and not converged:
        values = (
            _evaluate_catmull_rom(
                points, horizontal, vertical, source_x, source_y
            )
            if catmull_rom
            else _evaluate_bilinear(
                points, horizontal, vertical, source_x, source_y
            )
        )
        residual_x = values[0] - visual_x
        residual_y = values[1] - visual_y
        residual_squared = residual_x * residual_x + residual_y * residual_y
    valid = (
        finite
        and math.isfinite(residual_squared)
        and residual_squared <= 1e-10
    )
    return source_x, source_y, valid


@njit(inline='always')
def _inverse_bilinear_cell(
    points,
    horizontal,
    vertical,
    visual_x,
    visual_y,
    column,
    row,
):
    p00x = points[row, column, 0]
    p00y = points[row, column, 1]
    p10x = points[row, column + 1, 0]
    p10y = points[row, column + 1, 1]
    p01x = points[row + 1, column, 0]
    p01y = points[row + 1, column, 1]
    p11x = points[row + 1, column + 1, 0]
    p11y = points[row + 1, column + 1, 1]
    guard = 1e-5
    if (
        visual_x < min(p00x, p10x, p01x, p11x) - guard
        or visual_x > max(p00x, p10x, p01x, p11x) + guard
        or visual_y < min(p00y, p10y, p01y, p11y) - guard
        or visual_y > max(p00y, p10y, p01y, p11y) + guard
    ):
        return 0.0, 0.0, False

    linear_xx = p10x - p00x
    linear_xy = p10y - p00y
    linear_yx = p01x - p00x
    linear_yy = p01y - p00y
    cross_x = p11x - p10x - p01x + p00x
    cross_y = p11y - p10y - p01y + p00y
    affine_determinant = linear_xx * linear_yy - linear_xy * linear_yx
    if abs(affine_determinant) > 1e-10:
        offset_x = visual_x - p00x
        offset_y = visual_y - p00y
        local_x = (
            offset_x * linear_yy - offset_y * linear_yx
        ) / affine_determinant
        local_y = (
            linear_xx * offset_y - linear_xy * offset_x
        ) / affine_determinant
        local_x = min(max(local_x, 0.0), 1.0)
        local_y = min(max(local_y, 0.0), 1.0)
    else:
        local_x = local_y = 0.5

    best_x = local_x
    best_y = local_y
    best_residual = math.inf
    for _iteration in range(12):
        output_x = (
            p00x + linear_xx * local_x + linear_yx * local_y
            + cross_x * local_x * local_y
        )
        output_y = (
            p00y + linear_xy * local_x + linear_yy * local_y
            + cross_y * local_x * local_y
        )
        residual_x = output_x - visual_x
        residual_y = output_y - visual_y
        residual = residual_x * residual_x + residual_y * residual_y
        if residual > best_residual:
            local_x = (local_x + best_x) * 0.5
            local_y = (local_y + best_y) * 0.5
            continue
        best_x = local_x
        best_y = local_y
        best_residual = residual
        if residual <= 1e-12:
            break
        derivative_xx = linear_xx + cross_x * local_y
        derivative_xy = linear_xy + cross_y * local_y
        derivative_yx = linear_yx + cross_x * local_x
        derivative_yy = linear_yy + cross_y * local_x
        determinant = (
            derivative_xx * derivative_yy - derivative_xy * derivative_yx
        )
        if not math.isfinite(determinant) or abs(determinant) <= 1e-10:
            break
        delta_x = (
            residual_x * derivative_yy - residual_y * derivative_yx
        ) / determinant
        delta_y = (
            derivative_xx * residual_y - derivative_xy * residual_x
        ) / determinant
        local_x = min(max(local_x - delta_x, 0.0), 1.0)
        local_y = min(max(local_y - delta_y, 0.0), 1.0)

    valid = (
        math.isfinite(best_residual)
        and best_residual <= 1e-10
        and -guard <= best_x <= 1.0 + guard
        and -guard <= best_y <= 1.0 + guard
    )
    return (
        (column + best_x) / horizontal,
        (row + best_y) / vertical,
        valid,
    )


@njit(inline='always')
def _retry_nearby_bilinear_cells(
    points,
    horizontal,
    vertical,
    visual_x,
    visual_y,
    source_x,
    source_y,
):
    start_column = min(
        max(int(math.floor(source_x * horizontal)), 0), horizontal - 1
    )
    start_row = min(
        max(int(math.floor(source_y * vertical)), 0), vertical - 1
    )
    for row in range(max(0, start_row - 1), min(vertical, start_row + 2)):
        for column in range(
            max(0, start_column - 1), min(horizontal, start_column + 2)
        ):
            result = _inverse_bilinear_cell(
                points,
                horizontal,
                vertical,
                visual_x,
                visual_y,
                column,
                row,
            )
            if result[2]:
                return result
    return source_x, source_y, False


@njit(cache=True, parallel=True)
def _inverse_bilinear(
    points,
    horizontal,
    vertical,
    visual_x,
    visual_y,
    source_left,
    source_top,
    source_right,
    source_bottom,
):
    source_x = np.empty_like(visual_x)
    source_y = np.empty_like(visual_y)
    valid = np.empty(visual_x.shape, dtype=np.bool_)
    flat_x = visual_x.ravel()
    flat_y = visual_y.ravel()
    flat_source_x = source_x.ravel()
    flat_source_y = source_y.ravel()
    flat_valid = valid.ravel()
    for index in prange(flat_x.size):
        result = _inverse_point(
            points, horizontal, vertical,
            flat_x[index], flat_y[index], False,
        )
        source_epsilon = 4e-6
        in_source = (
            source_left - source_epsilon <= result[0]
            <= source_right + source_epsilon
            and source_top - source_epsilon <= result[1]
            <= source_bottom + source_epsilon
        )
        flat_source_x[index] = result[0]
        flat_source_y[index] = result[1]
        flat_valid[index] = result[2] and in_source
    return source_x, source_y, valid


@njit(cache=True, parallel=True)
def _retry_invalid_bilinear(
    points,
    horizontal,
    vertical,
    visual_x,
    visual_y,
    source_x,
    source_y,
    valid,
):
    flat_x = visual_x.ravel()
    flat_y = visual_y.ravel()
    flat_source_x = source_x.ravel()
    flat_source_y = source_y.ravel()
    flat_valid = valid.ravel()
    for index in prange(flat_x.size):
        if flat_valid[index]:
            continue
        result = _retry_nearby_bilinear_cells(
            points,
            horizontal,
            vertical,
            flat_x[index],
            flat_y[index],
            flat_source_x[index],
            flat_source_y[index],
        )
        if result[2]:
            flat_source_x[index] = result[0]
            flat_source_y[index] = result[1]
            flat_valid[index] = True


@njit(cache=True, parallel=True)
def _inverse_catmull_rom(
    points,
    horizontal,
    vertical,
    visual_x,
    visual_y,
    source_left,
    source_top,
    source_right,
    source_bottom,
):
    source_x = np.empty_like(visual_x)
    source_y = np.empty_like(visual_y)
    valid = np.empty(visual_x.shape, dtype=np.bool_)
    flat_x = visual_x.ravel()
    flat_y = visual_y.ravel()
    flat_source_x = source_x.ravel()
    flat_source_y = source_y.ravel()
    flat_valid = valid.ravel()
    for index in prange(flat_x.size):
        result = _inverse_point(
            points, horizontal, vertical,
            flat_x[index], flat_y[index], True,
        )
        flat_source_x[index] = result[0]
        flat_source_y[index] = result[1]
        flat_valid[index] = result[2]
    return source_x, source_y, valid


def warm_grid_numba_cache() -> None:
    """Load or compile raster and bilinear-retry signatures once per process.

    >>> NUMBA_CACHE_DIR.endswith(osp.join('.btrans_cache', 'numba'))
    True
    """
    global _warmup_complete
    with _warmup_lock:
        if _warmup_complete:
            return
        bilinear_points = np.asarray(
            (((0.0, 0.0), (1.0, 0.0)),
             ((0.0, 1.0), (1.0, 1.0))),
            dtype=np.float32,
        )
        axis = np.arange(-1.0, 3.0, dtype=np.float32)
        catmull_x, catmull_y = np.meshgrid(axis, axis)
        catmull_rom_points = np.stack((catmull_x, catmull_y), axis=-1)
        coordinate = np.full((1, 1), 0.5, dtype=np.float32)
        bilinear = _inverse_bilinear(
            bilinear_points, 1, 1, coordinate, coordinate,
            0.0, 0.0, 1.0, 1.0,
        )
        catmull_rom = _inverse_catmull_rom(
            catmull_rom_points, 1, 1, coordinate, coordinate,
            0.0, 0.0, 1.0, 1.0,
        )
        retry_source_x = np.zeros_like(coordinate)
        retry_source_y = np.zeros_like(coordinate)
        retry_valid = np.zeros_like(coordinate, dtype=np.bool_)
        _retry_invalid_bilinear(
            bilinear_points,
            1,
            1,
            coordinate,
            coordinate,
            retry_source_x,
            retry_source_y,
            retry_valid,
        )
        if (
            not bool(bilinear[2][0, 0])
            or not bool(catmull_rom[2][0, 0])
            or not bool(retry_valid[0, 0])
        ):
            raise RuntimeError('Numba Grid kernel warm-up failed validation')
        _warmup_complete = True


def inverse_grid_arrays(
    points,
    horizontal,
    vertical,
    visual_x,
    visual_y,
    *,
    catmull_rom,
    source_bounds=(0.0, 0.0, 1.0, 1.0),
):
    """Return a compiled inverse result, or ``None`` before warm-up.

    >>> inverse_grid_arrays(
    ...     np.zeros((2, 2, 2), dtype=np.float32), 1, 1,
    ...     np.zeros((1, 1), dtype=np.float32),
    ...     np.zeros((1, 1), dtype=np.float32), catmull_rom=False,
    ... ) is None or _warmup_complete
    True
    """
    if not _warmup_complete:
        return None
    visual_x = np.asarray(visual_x, dtype=np.float32)
    visual_y = np.asarray(visual_y, dtype=np.float32)
    kernel = _inverse_catmull_rom if catmull_rom else _inverse_bilinear
    result = kernel(
        points,
        horizontal,
        vertical,
        visual_x,
        visual_y,
        *source_bounds,
    )
    if not catmull_rom and not np.all(result[2]):
        # Keep the common kernel unchanged; only seam failures pay for a
        # cell-constrained retry.
        _retry_invalid_bilinear(
            points,
            horizontal,
            vertical,
            visual_x,
            visual_y,
            *result,
        )
    return result
