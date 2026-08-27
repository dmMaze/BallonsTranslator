"""Immutable TextBlock-owned rendered-image layer values."""

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Optional, Union

from .logger import logger as LOGGER
from .raster_assets import RasterAssetRef, coerce_raster_asset_ref


RENDERED_IMAGE_LAYER_VERSION = 1


@dataclass(frozen=True)
class RenderedImageLayer:
    """One full-RGBA project image composited over isolated text output.

    >>> ref = RasterAssetRef('assets/' + 'a' * 64 + '.png')
    >>> RenderedImageLayer(ref).mode
    'replace'
    >>> RenderedImageLayer().asset is None
    True
    """

    asset: Optional[RasterAssetRef] = None
    version: int = RENDERED_IMAGE_LAYER_VERSION
    enabled: bool = True
    mode: str = 'replace'

    def __post_init__(self) -> None:
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or self.version != RENDERED_IMAGE_LAYER_VERSION
        ):
            raise ValueError('unsupported rendered image layer version')
        if not isinstance(self.enabled, bool):
            raise TypeError('rendered image enabled must be a bool')
        if self.asset is not None and not isinstance(
            self.asset, RasterAssetRef
        ):
            raise TypeError(
                'rendered image asset must be RasterAssetRef or None'
            )
        if self.mode not in {'replace', 'overlay'}:
            raise ValueError('rendered image mode must be replace or overlay')
        object.__setattr__(self, 'version', int(self.version))

    def to_serializable_dict(self) -> dict:
        return {
            'version': self.version,
            'enabled': self.enabled,
            'asset': (
                None
                if self.asset is None
                else self.asset.to_serializable_dict()
            ),
            'mode': self.mode,
        }


def load_rendered_image_layer(
    payload: Optional[Union[RenderedImageLayer, Mapping[str, object]]],
) -> Optional[RenderedImageLayer]:
    """Load one optional layer without rejecting its surrounding TextBlock.

    >>> load_rendered_image_layer(None) is None
    True
    >>> load_rendered_image_layer({'version': 99}) is None
    True
    """
    if payload is None or isinstance(payload, RenderedImageLayer):
        return payload
    if not isinstance(payload, Mapping):
        LOGGER.warning('Ignoring malformed Image layer %r.', payload)
        return None
    unknown = set(payload) - {'version', 'enabled', 'asset', 'mode'}
    if unknown:
        LOGGER.warning(
            'Ignoring unknown Image layer fields: %s.',
            sorted(unknown),
        )
    try:
        raw_asset = payload.get('asset')
        asset = (
            None
            if raw_asset is None
            else coerce_raster_asset_ref(raw_asset)
        )
        return RenderedImageLayer(
            version=payload.get('version'),
            enabled=payload.get('enabled', True),
            asset=asset,
            mode=payload.get('mode', 'replace'),
        )
    except (TypeError, ValueError) as error:
        LOGGER.warning('Ignoring malformed Image layer: %s.', error)
        return None
