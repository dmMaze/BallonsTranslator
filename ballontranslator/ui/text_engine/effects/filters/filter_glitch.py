"""Deterministic horizontal block displacement and RGB separation."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ballontranslator.ui.text_engine.effects.filters._procedural import (
    coordinate_noise,
)
from ballontranslator.ui.text_engine.effects.filters.registry import FilterContext
from ballontranslator.utils.text_effects import FilterScalar


FILTER_META = {
    'filter_id': 'builtin:glitch',
    'name': 'Glitch',
    'schema_version': 1,
    'order': 60,
    'expands_alpha': True,
    'params': (
        {
            'key': 'shift', 'label': 'Shift', 'kind': 'float',
            'default': 6.0, 'minimum': 0.0, 'maximum': 64.0,
            'step': 0.1, 'decimals': 1, 'suffix': ' px',
        },
        {
            'key': 'block_size', 'label': 'Block Size', 'kind': 'float',
            'default': 8.0, 'minimum': 1.0, 'maximum': 64.0,
            'step': 0.1, 'decimals': 1, 'suffix': ' px',
        },
        {
            'key': 'activity', 'label': 'Activity', 'kind': 'float',
            'default': 0.25, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
        {
            'key': 'rgb_split', 'label': 'RGB Split', 'kind': 'float',
            'default': 2.0, 'minimum': 0.0, 'maximum': 32.0,
            'step': 0.1, 'decimals': 1, 'suffix': ' px',
        },
        {
            'key': 'seed', 'label': 'Seed', 'kind': 'int',
            'default': 0, 'minimum': 0, 'maximum': 2147483647,
            'step': 1,
        },
    ),
}
_BAND_ROWS = 256


def tile_halo(params: Mapping[str, FilterScalar], render_scale: float) -> int:
    if float(params['activity']) <= 0.0:
        return 0
    return (
        int(math.ceil(float(params['shift']) * render_scale))
        + int(math.ceil(float(params['rgb_split']) * render_scale))
    )


def _row_shifts(
    height: int,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> tuple[np.ndarray, np.ndarray]:
    block_size = max(
        1, int(math.ceil(float(params['block_size']) * context.render_scale))
    )
    absolute_y = context.origin_y + np.arange(height, dtype=np.int64)
    block_ids = np.floor_divide(absolute_y, block_size)
    first_block = int(block_ids[0])
    block_count = int(block_ids[-1] - first_block + 1)
    seed = int(params['seed'])
    active_noise = coordinate_noise(
        block_count, 1, 0, first_block, seed, channel=20
    )[:, 0]
    shift_noise = coordinate_noise(
        block_count, 1, 0, first_block, seed, channel=21
    )[:, 0]
    index = (block_ids - first_block).astype(np.intp)
    active = (
        (active_noise[index] + np.float32(1.0)) * np.float32(0.5)
        < np.float32(float(params['activity']))
    )
    shift_radius = int(math.ceil(
        float(params['shift']) * context.render_scale
    ))
    shifts = np.rint(shift_noise[index] * shift_radius).astype(np.int32)
    shifts[~active] = 0
    return shifts, active


def _sample_rows(
    source: np.ndarray,
    top: int,
    bottom: int,
    offsets: np.ndarray,
) -> np.ndarray:
    width = source.shape[1]
    columns = np.arange(width, dtype=np.int32)[np.newaxis, :]
    indices = columns + offsets[:, np.newaxis]
    valid = (indices >= 0) & (indices < width)
    clipped = np.clip(indices, 0, max(0, width - 1))
    rows = np.arange(top, bottom, dtype=np.intp)[:, np.newaxis]
    sampled = source[rows, clipped]
    sampled[~valid] = 0
    return sampled


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    """Apply seeded row-block displacement and split-channel sampling.

    >>> pixel = np.zeros((1, 1, 4), dtype=np.uint8)
    >>> params = {'activity': 0.0, 'shift': 6.0, 'rgb_split': 2.0}
    >>> apply(pixel, params, FilterContext(1.0, 0, 0)) is pixel
    True
    """
    activity = float(params['activity'])
    if activity <= 0.0 or rgba.size == 0:
        return rgba
    shift = int(math.ceil(float(params['shift']) * context.render_scale))
    split = int(math.ceil(
        float(params['rgb_split']) * context.render_scale
    ))
    if shift == 0 and split == 0:
        return rgba
    height, width = rgba.shape[:2]
    row_shifts, active = _row_shifts(height, params, context)
    active_split = np.where(active, split, 0).astype(np.int32)
    output = np.zeros(rgba.shape, dtype=np.uint8)

    for top in range(0, height, _BAND_ROWS):
        bottom = min(height, top + _BAND_ROWS)
        center_offsets = row_shifts[top:bottom]
        split_offsets = active_split[top:bottom]
        center = _sample_rows(
            rgba, top, bottom, center_offsets
        )
        red = _sample_rows(
            rgba, top, bottom, center_offsets + split_offsets
        )
        blue = _sample_rows(
            rgba, top, bottom, center_offsets - split_offsets
        )
        alpha = np.maximum.reduce((
            center[:, :, 3], red[:, :, 3], blue[:, :, 3]
        ))
        target = output[top:bottom]
        target[:, :, 3] = alpha
        nonzero = alpha > 0
        denominator = alpha.astype(np.uint32)
        for channel, sample in ((0, red), (1, center), (2, blue)):
            numerator = (
                sample[:, :, channel].astype(np.uint32)
                * sample[:, :, 3].astype(np.uint32)
            )
            values = np.zeros(alpha.shape, dtype=np.uint32)
            values[nonzero] = (
                numerator[nonzero] + denominator[nonzero] // 2
            ) // denominator[nonzero]
            target[:, :, channel] = values.astype(np.uint8)
    return output
