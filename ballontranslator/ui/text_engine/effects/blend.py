"""Exact alpha-correct pixel blending for modes not provided by QPainter."""

import numpy as np


CUSTOM_BLEND_MODES = frozenset({
    'linear_burn',
    'darker_color',
    'linear_dodge',
    'lighter_color',
})
_BLEND_CHUNK_PIXELS = 262_144


def composite_custom_blend_rgba(
    destination: np.ndarray,
    source: np.ndarray,
    blend_mode: str,
) -> np.ndarray:
    """Source-over one straight-RGBA8 layer with exact integer rounding.

    Row chunks bound working memory without changing the result, which keeps
    full and tiled rendering on the same pixel contract.

    >>> dst = np.array([[[100, 150, 200, 255]]], dtype=np.uint8)
    >>> src = np.array([[[200, 50, 100, 255]]], dtype=np.uint8)
    >>> composite_custom_blend_rgba(dst, src, 'linear_burn')[0, 0].tolist()
    [45, 0, 45, 255]
    """
    if blend_mode not in CUSTOM_BLEND_MODES:
        raise ValueError('unsupported custom text-effect blend mode')
    if (
        not isinstance(destination, np.ndarray)
        or not isinstance(source, np.ndarray)
        or destination.dtype != np.uint8
        or source.dtype != np.uint8
    ):
        raise TypeError('blend layers must be RGBA8 arrays')
    if (
        destination.shape != source.shape
        or destination.ndim != 3
        or destination.shape[2] != 4
    ):
        raise ValueError('blend layers must have matching RGBA shapes')

    if destination.size == 0:
        return np.ascontiguousarray(destination)
    output = np.empty(destination.shape, dtype=np.uint8)
    rows_per_chunk = max(
        1, _BLEND_CHUNK_PIXELS // max(1, destination.shape[1])
    )

    for start in range(0, destination.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, destination.shape[0])
        backdrop_pixels = destination[start:stop].reshape(-1, 4)
        source_pixels = source[start:stop].reshape(-1, 4)
        backdrop_rgb = backdrop_pixels[:, :3].astype(np.uint32)
        source_rgb = source_pixels[:, :3].astype(np.uint32)
        backdrop_alpha = backdrop_pixels[:, 3:4].astype(np.uint32)
        source_alpha = source_pixels[:, 3:4].astype(np.uint32)

        if blend_mode == 'linear_burn':
            channel_sum = backdrop_rgb + source_rgb
            blended_rgb = np.where(
                channel_sum > 255, channel_sum - 255, 0
            ).astype(np.uint32)
        elif blend_mode == 'linear_dodge':
            blended_rgb = np.minimum(
                backdrop_rgb + source_rgb, 255
            ).astype(np.uint32)
        else:
            # Compare encoded channel totals exactly; equal totals retain the
            # destination color in both whole-color modes.
            source_total = np.sum(
                source_pixels[:, :3], axis=1, dtype=np.uint16
            )
            backdrop_total = np.sum(
                backdrop_pixels[:, :3], axis=1, dtype=np.uint16
            )
            source_wins = (
                source_total < backdrop_total
            )
            if blend_mode == 'lighter_color':
                source_wins = source_total > backdrop_total
            blended_rgb = np.where(
                source_wins[:, np.newaxis], source_rgb, backdrop_rgb
            ).astype(np.uint32)

        inverse_source_alpha = 255 - source_alpha
        inverse_backdrop_alpha = 255 - backdrop_alpha
        output_alpha = (
            source_alpha * 255
            + backdrop_alpha * inverse_source_alpha
        )
        # Blend functions affect only the overlap. Source-over still owns
        # partial coverage and transparent-edge alpha.
        numerator = (
            source_alpha * inverse_backdrop_alpha * source_rgb
            + source_alpha * backdrop_alpha * blended_rgb
            + inverse_source_alpha * backdrop_alpha * backdrop_rgb
        )
        output_rgb = np.zeros_like(numerator, dtype=np.uint32)
        # floor(N / A + 0.5), including exact half values.
        np.floor_divide(
            numerator * 2 + output_alpha,
            output_alpha * 2,
            out=output_rgb,
            where=output_alpha > 0,
        )

        result = np.empty(source_pixels.shape, dtype=np.uint8)
        result[:, :3] = output_rgb.astype(np.uint8)
        result[:, 3:4] = ((output_alpha + 127) // 255).astype(np.uint8)
        output[start:stop] = result.reshape(
            stop - start, destination.shape[1], 4
        )

    return np.ascontiguousarray(output)
