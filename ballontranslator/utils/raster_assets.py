"""Generic immutable references to project-managed raster assets."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import PurePosixPath
import re
from typing import Mapping, Union


_RASTER_ASSET_PATH_PATTERN = re.compile(
    r'^assets/(?P<digest>[0-9a-f]{64})\.[a-z0-9]+$'
)

RASTER_ASSET_MAX_PIXELS = 64 * 1024 * 1024
RASTER_ASSET_MAX_DECODED_BYTES = RASTER_ASSET_MAX_PIXELS * 4


def validate_raster_dimensions(width: int, height: int) -> int:
    """Validate shared decoded-raster bounds and return the pixel count.

    >>> validate_raster_dimensions(3, 2)
    6
    """
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
    ):
        raise TypeError('raster dimensions must be integers')
    pixels = int(width) * int(height)
    if pixels <= 0 or pixels > RASTER_ASSET_MAX_PIXELS:
        raise ValueError('raster asset exceeds the pixel limit')
    if pixels * 4 > RASTER_ASSET_MAX_DECODED_BYTES:
        raise ValueError('raster asset exceeds the decoded-byte limit')
    return pixels


@dataclass(frozen=True)
class RasterAssetRef:
    """Reference one content-addressed raster inside a project.

    The value has no text-effect semantics so drawing brushes can reuse it.

    >>> RasterAssetRef('assets/' + 'a' * 64 + '.png').digest == 'a' * 64
    True
    """

    path: str
    display_name: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError('raster asset path must be a string')
        path = PurePosixPath(self.path)
        if (
            '\\' in self.path
            or path.is_absolute()
            or not _RASTER_ASSET_PATH_PATTERN.fullmatch(self.path)
        ):
            raise ValueError(
                'raster asset path must be a content-addressed file in assets'
            )
        if not isinstance(self.display_name, str):
            raise TypeError('raster asset display name must be a string')
        display_name = self.display_name.replace('\\', '/')
        if self.display_name and PurePosixPath(display_name).name != display_name:
            raise ValueError('raster asset display name must not contain a path')

    @property
    def digest(self) -> str:
        """Return the SHA-256 hex digest encoded by the validated path."""
        match = _RASTER_ASSET_PATH_PATTERN.fullmatch(self.path)
        assert match is not None
        return match.group('digest')

    def to_serializable_dict(self) -> dict:
        return {
            'path': self.path,
            'display_name': self.display_name,
        }


def coerce_raster_asset_ref(
    value: Union[RasterAssetRef, Mapping[str, object]],
) -> RasterAssetRef:
    """Return a validated reference from its strict persisted payload.

    >>> coerce_raster_asset_ref(
    ...     {'path': 'assets/' + 'b' * 64 + '.png'}
    ... ).digest == 'b' * 64
    True
    """
    if isinstance(value, RasterAssetRef):
        return value
    if not isinstance(value, Mapping):
        raise ValueError('raster asset must be a value or typed payload')
    unexpected = set(value) - {'path', 'display_name'}
    if unexpected:
        raise ValueError(
            f'unsupported raster asset fields: {sorted(unexpected)}'
        )
    return RasterAssetRef(
        path=value.get('path'),
        display_name=value.get('display_name', ''),
    )
