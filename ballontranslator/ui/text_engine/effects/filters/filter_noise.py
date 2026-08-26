"""Coordinate-deterministic additive pigment noise."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ballontranslator.ui.text_engine.effects.filters._procedural import (
    coordinate_noise,
)
from ballontranslator.ui.text_engine.effects.filters.registry import FilterContext
from ballontranslator.utils.text_effects import FilterScalar


FILTER_META = {
    'filter_id': 'builtin:noise',
    'name': 'Noise',
    'schema_version': 1,
    'order': 10,
    'params': (
        {
            'key': 'amount', 'label': 'Amount', 'kind': 'float',
            'default': 0.2, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
        {
            'key': 'mode', 'label': 'Color', 'kind': 'choice',
            'default': 'monochrome',
            'choices': (('Monochrome', 'monochrome'), ('Color', 'color')),
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
    return 0


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    amount = float(params['amount'])
    if amount <= 0.0 or rgba.size == 0:
        return rgba
    height, width = rgba.shape[:2]
    seed = int(params['seed'])
    color = params['mode'] == 'color'
    strength = np.float32(amount * 64.0)
    for top in range(0, height, _BAND_ROWS):
        rows = min(_BAND_ROWS, height - top)
        if color:
            for channel in range(3):
                noise = coordinate_noise(
                    rows, width, context.origin_x, context.origin_y + top,
                    seed, channel,
                )
                rgba[top:top + rows, :, channel] = np.clip(
                    rgba[top:top + rows, :, channel].astype(np.float32)
                    + noise * strength,
                    0.0,
                    255.0,
                ).astype(np.uint8)
        else:
            noise = coordinate_noise(
                rows, width, context.origin_x, context.origin_y + top, seed
            )
            rgba[top:top + rows, :, :3] = np.clip(
                rgba[top:top + rows, :, :3].astype(np.float32)
                + noise[:, :, np.newaxis] * strength,
                0.0,
                255.0,
            ).astype(np.uint8)
    return rgba
