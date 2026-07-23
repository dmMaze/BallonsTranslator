"""Project persistence boundary for text transforms.

The runtime model is polymorphic, but schema v2 deliberately remains the
existing flat slant quartet. Old configs use the same flat representation and
therefore require no migration or rewrite.
"""

import math
from typing import Tuple

from .fontformat import (
    TEXT_TRANSFORM_BOX_SLANT_MAX,
    TEXT_TRANSFORM_BOX_SLANT_MIN,
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    TEXT_TRANSFORM_SCALE_MAX,
    TEXT_TRANSFORM_SCALE_MIN,
    SlantTextTransform,
    TextTransform,
    normalize_text_transform,
)


TEXT_TRANSFORM_SCHEMA_VERSION = 2
_MISSING = object()
_CANONICAL_TRANSFORM_FIELDS = SlantTextTransform.component_fields
_TRANSFORM_BOUNDS = {
    'horizontal_scale': (TEXT_TRANSFORM_SCALE_MIN, TEXT_TRANSFORM_SCALE_MAX),
    'vertical_scale': (TEXT_TRANSFORM_SCALE_MIN, TEXT_TRANSFORM_SCALE_MAX),
    'slant_angle': (TEXT_TRANSFORM_BOX_SLANT_MIN, TEXT_TRANSFORM_BOX_SLANT_MAX),
    'glyph_slant_angle': (
        TEXT_TRANSFORM_GLYPH_SLANT_MIN,
        TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    ),
}
_NONCANONICAL_TRANSFORM_FIELDS = (
    *_CANONICAL_TRANSFORM_FIELDS,
    'italic_angle',
    'rich_text_transform_version',
)
_INTERMEDIATE_STRETCH_MARKER = 'ballontranslator-logical-stretch-v1:'


class TextTransformPayloadError(ValueError):
    """Base class for project text-transform payload failures."""


class UnsupportedTextTransformVersionError(TextTransformPayloadError):
    pass


class InvalidTextTransformPayloadError(TextTransformPayloadError):
    pass


def _payload_version(value, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTextTransformPayloadError(
            f"{location} must be an integer schema version"
        )
    value = float(value)
    if not math.isfinite(value) or not value.is_integer() or value < 0:
        raise InvalidTextTransformPayloadError(
            f"{location} must be an integer schema version"
        )
    return int(value)


def _payload_number(value, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTextTransformPayloadError(
            f"{location} must be a finite number"
        )
    value = float(value)
    if not math.isfinite(value):
        raise InvalidTextTransformPayloadError(
            f"{location} must be a finite number"
        )
    return value


def _canonical_v2_block_transform(
    block: dict,
    location: str,
) -> Tuple[TextTransform, dict]:
    if 'fontformat' not in block or not isinstance(block['fontformat'], dict):
        raise InvalidTextTransformPayloadError(
            f"{location}.fontformat must be an object in schema v2"
        )
    fontformat = block['fontformat']

    for field_name in _NONCANONICAL_TRANSFORM_FIELDS:
        if field_name in block:
            raise InvalidTextTransformPayloadError(
                f"{location}.{field_name} is not canonical in schema v2"
            )
    for field_name in ('italic_angle', 'rich_text_transform_version'):
        if field_name in fontformat:
            raise InvalidTextTransformPayloadError(
                f"{location}.fontformat.{field_name} is not canonical in schema v2"
            )
    rich_text = block.get('rich_text')
    if isinstance(rich_text, str) and _INTERMEDIATE_STRETCH_MARKER in rich_text:
        raise InvalidTextTransformPayloadError(
            f"{location}.rich_text contains unsupported intermediate transform metadata"
        )

    raw = []
    for field_name in _CANONICAL_TRANSFORM_FIELDS:
        if field_name not in fontformat:
            raise InvalidTextTransformPayloadError(
                f"{location}.fontformat.{field_name} is required in schema v2"
            )
        value = _payload_number(
            fontformat[field_name], f"{location}.fontformat.{field_name}"
        )
        minimum, maximum = _TRANSFORM_BOUNDS[field_name]
        if value < minimum or value > maximum:
            raise InvalidTextTransformPayloadError(
                f"{location}.fontformat.{field_name} is outside "
                f"the canonical range [{minimum}, {maximum}]"
            )
        raw.append(value)

    return normalize_text_transform(*raw), fontformat


def _official_legacy_fontformat(block: dict, location: str) -> dict:
    """Validate an upstream block before adding a neutral quartet."""
    fontformat = block.get('fontformat', {})
    if fontformat is None:
        fontformat = {}
    if not isinstance(fontformat, dict):
        raise InvalidTextTransformPayloadError(
            f"{location}.fontformat must be an object"
        )

    for field_name in _NONCANONICAL_TRANSFORM_FIELDS:
        if field_name in block or field_name in fontformat:
            raise InvalidTextTransformPayloadError(
                f"{location} contains unsupported intermediate transform field "
                f"{field_name}"
            )
    rich_text = block.get('rich_text')
    if isinstance(rich_text, str) and _INTERMEDIATE_STRETCH_MARKER in rich_text:
        raise InvalidTextTransformPayloadError(
            f"{location}.rich_text contains unsupported intermediate transform metadata"
        )
    return fontformat


def migrate_text_transform_payload(proj_dict: dict) -> dict:
    """Return canonical schema-v2 dictionaries without mutating the input.

    Only dictionaries that receive canonical fields are copied. Potentially
    large text, geometry, and mask values remain shared until ``TextBlock``
    construction consumes them, avoiding a redundant whole-project deepcopy.

    >>> migrate_text_transform_payload({'pages': {}})['text_transform_schema_version']
    2
    """
    if not isinstance(proj_dict, dict):
        raise InvalidTextTransformPayloadError("project payload must be an object")

    version_value = proj_dict.get('text_transform_schema_version', _MISSING)
    if version_value is _MISSING:
        root_version = None
    else:
        root_version = _payload_version(
            version_value,
            'text_transform_schema_version',
        )
        if root_version != TEXT_TRANSFORM_SCHEMA_VERSION:
            raise UnsupportedTextTransformVersionError(
                f"unsupported text transform schema version {root_version}"
            )

    for field_name in _NONCANONICAL_TRANSFORM_FIELDS:
        if field_name in proj_dict:
            raise InvalidTextTransformPayloadError(
                f"project root field {field_name} is not canonical"
            )

    pages = proj_dict.get('pages')
    if not isinstance(pages, dict):
        raise InvalidTextTransformPayloadError("pages must be an object")

    migrated_pages = {}
    for page_name, blocks in pages.items():
        if not isinstance(blocks, list):
            raise InvalidTextTransformPayloadError(f"pages.{page_name} must be a list")
        migrated_blocks = []
        for index, block in enumerate(blocks):
            location = f"pages.{page_name}[{index}]"
            if not isinstance(block, dict):
                raise InvalidTextTransformPayloadError(f"{location} must be an object")
            if root_version is None:
                fontformat = _official_legacy_fontformat(block, location)
                transform = SlantTextTransform()
            else:
                transform, fontformat = _canonical_v2_block_transform(block, location)
            canonical_fontformat = dict(fontformat)
            canonical_fontformat.update(transform.flat_dict())
            canonical_block = dict(block)
            canonical_block['fontformat'] = canonical_fontformat
            migrated_blocks.append(canonical_block)
        migrated_pages[page_name] = migrated_blocks

    migrated = dict(proj_dict)
    migrated['pages'] = migrated_pages
    migrated['text_transform_schema_version'] = TEXT_TRANSFORM_SCHEMA_VERSION
    return migrated
