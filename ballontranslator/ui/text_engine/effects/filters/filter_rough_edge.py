"""Deterministically grow a noisy, locally colored text silhouette."""

from __future__ import annotations

import math
from typing import Mapping

import cv2
import numpy as np

from ballontranslator.ui.text_engine.effects.filters._procedural import (
    blurred_coordinate_noise,
    coordinate_noise,
)
from ballontranslator.ui.text_engine.effects.filters.registry import FilterContext
from ballontranslator.utils.text_effects import FilterScalar


FILTER_META = {
    'filter_id': 'builtin:rough_edge',
    'name': 'Rough Edge',
    'schema_version': 1,
    'order': 30,
    'expands_alpha': True,
    'params': (
        {
            'key': 'amount', 'label': 'Amount', 'kind': 'float',
            'default': 0.35, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
        {
            'key': 'size', 'label': 'Size', 'kind': 'float',
            'default': 2.0, 'minimum': 0.25, 'maximum': 8.0,
            'step': 0.1, 'decimals': 1,
        },
        {
            'key': 'hardness', 'label': 'Hardness', 'kind': 'float',
            'default': 0.6, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
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
    return max(1, int(math.ceil(float(params['size']) * render_scale)))


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    """Grow the original silhouette through a noisy displaced threshold.

    >>> pixels = np.zeros((2, 3, 4), dtype=np.uint8)
    >>> apply(pixels, {'amount': 0.0}, FilterContext(1.0, 0, 0)) is pixels
    True
    """
    amount = float(params['amount'])
    if amount <= 0.0 or rgba.size == 0:
        return rgba
    radius = tile_halo(params, context.render_scale)
    alpha = rgba[:, :, 3].copy()
    kernel_size = radius * 2 + 1
    hardness = float(params['hardness'])
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    seed = int(params['seed'])
    threshold = np.float32(0.5 - hardness * 0.2)
    slope = np.float32(10.0 + hardness * 40.0)
    height, width = alpha.shape
    for top in range(0, height, _BAND_ROWS):
        bottom = min(height, top + _BAND_ROWS)
        source_top = max(0, top - radius)
        source_bottom = min(height, bottom + radius)
        dilated_source = cv2.dilate(
            alpha[source_top:source_bottom],
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        offset = top - source_top
        dilated = dilated_source[offset:offset + bottom - top]
        source_alpha = alpha[top:bottom]
        active_zone = dilated > 12
        coarse_noise = blurred_coordinate_noise(
            bottom - top,
            width,
            context.origin_x,
            context.origin_y + top,
            seed,
            radius,
            channel=1,
        )
        fine_noise = coordinate_noise(
            bottom - top,
            width,
            context.origin_x,
            context.origin_y + top,
            seed,
        )
        # Match the original effect's coarse-plus-fine displaced threshold.
        # The scale restores approximately normal-noise variance while keeping
        # the randomness coordinate-stable across full and tiled rendering.
        noise = (
            coarse_noise * np.float32(1.4)
            + fine_noise * np.float32(0.6)
        )
        displaced = (
            source_alpha.astype(np.float32) / np.float32(255.0)
            + noise * np.float32(amount * 0.6)
        )
        new_alpha = np.zeros_like(displaced, dtype=np.float32)
        new_alpha[active_zone] = np.float32(1.0) / (
            np.float32(1.0)
            + np.exp(-slope * (displaced[active_zone] - threshold))
        )

        target = rgba[top:bottom]
        transparent_growth = (
            (source_alpha == 0)
            & (new_alpha >= np.float32(1.0 / 255.0))
        )
        if np.any(transparent_growth):
            # Extend the nearest visible source color instead of inventing a
            # black fringe or sampling one tile-local "dominant" color.
            source_region = alpha[source_top:source_bottom] > 12
            if np.any(source_region):
                distance_input = (~source_region).astype(np.uint8)
                _, labels = cv2.distanceTransformWithLabels(
                    distance_input,
                    cv2.DIST_L2,
                    5,
                    labelType=cv2.DIST_LABEL_PIXEL,
                )
                source_labels = labels[source_region]
                color_lookup = np.zeros(
                    (int(source_labels.max()) + 1, 3), dtype=np.uint8
                )
                color_lookup[source_labels] = rgba[
                    source_top:source_bottom, :, :3
                ][source_region]
                target_labels = labels[
                    offset:offset + bottom - top
                ]
                target[transparent_growth, :3] = color_lookup[
                    target_labels[transparent_growth]
                ]
        target[:, :, 3] = np.clip(
            new_alpha * np.float32(255.0), 0.0, 255.0
        ).astype(np.uint8)
    return rgba
