"""Generic immutable references to project-managed raster assets."""

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Mapping, Union


_RASTER_ASSET_PATH_PATTERN = re.compile(
    r'^assets/(?P<digest>[0-9a-f]{64})\.[a-z0-9]+$'
)


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
