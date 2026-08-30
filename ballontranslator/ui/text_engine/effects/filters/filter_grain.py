"""Soft coordinate-stable pigment and alpha grain."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ballontranslator.ui.text_engine.effects.filters._procedural import (
    blurred_coordinate_noise,
)
from ballontranslator.ui.text_engine.effects.filters.registry import FilterContext
from ballontranslator.utils.text_effects import FilterScalar


FILTER_META = {
    'filter_id': 'builtin:grain',
    'name': 'Grain',
    'schema_version': 1,
    'order': 20,
    'params': (
        {
            'key': 'amount', 'label': 'Amount', 'kind': 'float',
            'default': 0.25, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
        {
            'key': 'size', 'label': 'Size', 'kind': 'float',
            'default': 2.0, 'minimum': 0.25, 'maximum': 8.0,
            'step': 0.1, 'decimals': 1,
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
    return int(math.ceil(float(params['size']) * render_scale * 2.0))


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    amount = float(params['amount'])
    if amount <= 0.0 or rgba.size == 0:
        return rgba
    height, width = rgba.shape[:2]
    radius = tile_halo(params, context.render_scale)
    seed = int(params['seed'])
    for top in range(0, height, _BAND_ROWS):
        rows = min(_BAND_ROWS, height - top)
        grain = blurred_coordinate_noise(
            rows, width, context.origin_x, context.origin_y + top,
            seed, radius,
        )
        target = rgba[top:top + rows]
        target[:, :, :3] = np.clip(
            target[:, :, :3].astype(np.float32)
            + grain[:, :, np.newaxis] * np.float32(amount * 72.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        alpha_factor = np.clip(
            1.0 + grain * np.float32(amount * 0.35), 0.0, 1.0
        )
        target[:, :, 3] = np.rint(
            target[:, :, 3].astype(np.float32) * alpha_factor
        ).astype(np.uint8)
    return rgba
