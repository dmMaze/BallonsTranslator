"""UI/runtime registration for supported text-transform variants."""

from dataclasses import dataclass
from typing import Callable, Tuple

from qtpy.QtCore import QCoreApplication

from ballontranslator.utils.fontformat import (
    TEXT_TRANSFORM_BOX_SLANT_MAX,
    TEXT_TRANSFORM_BOX_SLANT_MIN,
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    TEXT_TRANSFORM_SCALE_MAX,
    TEXT_TRANSFORM_SCALE_MIN,
    TEXT_TRANSFORM_TYPES,
    TextTransform,
)

from .text_effects.transform_layout import GlyphSlantLayoutRenderer
from .text_transform import (
    NoTextTransformStrategy,
    SlantTextTransformStrategy,
    TextTransformStrategy,
)


@dataclass(frozen=True)
class TransformControlSpec:
    """Describe one scalar control shown for a transform variant."""

    name: str
    attribute_name: str
    label: Callable[[], str]
    factor: float
    minimum: float
    maximum: float
    suffix: str


@dataclass(frozen=True)
class TextTransformVariantSpec:
    """Bind one persisted type to its UI and rendering strategy.

    >>> TEXT_TRANSFORM_VARIANTS[0].transform_type
    'none'
    """

    transform_type: str
    label: Callable[[], str]
    strategy: TextTransformStrategy
    controls: Tuple[TransformControlSpec, ...] = ()

    @property
    def layout_renderer_factory(self):
        return getattr(self.strategy, 'layout_renderer_factory', None)


SLANT_CONTROLS = (
    TransformControlSpec(
        'horizontal_scale_control',
        'horizontal_scale',
        lambda: QCoreApplication.translate(
            'TextAdvancedFormatPanel', 'Horizontal Scale'
        ),
        100.0,
        TEXT_TRANSFORM_SCALE_MIN,
        TEXT_TRANSFORM_SCALE_MAX,
        '%',
    ),
    TransformControlSpec(
        'vertical_scale_control',
        'vertical_scale',
        lambda: QCoreApplication.translate(
            'TextAdvancedFormatPanel', 'Vertical Scale'
        ),
        100.0,
        TEXT_TRANSFORM_SCALE_MIN,
        TEXT_TRANSFORM_SCALE_MAX,
        '%',
    ),
    TransformControlSpec(
        'slant_angle_control',
        'slant_angle',
        lambda: QCoreApplication.translate(
            'TextAdvancedFormatPanel', 'Box Slant'
        ),
        1.0,
        TEXT_TRANSFORM_BOX_SLANT_MIN,
        TEXT_TRANSFORM_BOX_SLANT_MAX,
        '\N{DEGREE SIGN}',
    ),
    TransformControlSpec(
        'glyph_slant_angle_control',
        'glyph_slant_angle',
        lambda: QCoreApplication.translate(
            'TextAdvancedFormatPanel', 'Glyph Slant'
        ),
        1.0,
        TEXT_TRANSFORM_GLYPH_SLANT_MIN,
        TEXT_TRANSFORM_GLYPH_SLANT_MAX,
        '\N{DEGREE SIGN}',
    ),
)


TEXT_TRANSFORM_VARIANTS = (
    TextTransformVariantSpec(
        'none',
        lambda: QCoreApplication.translate('TextAdvancedFormatPanel', 'None'),
        NoTextTransformStrategy(),
    ),
    TextTransformVariantSpec(
        'slant',
        lambda: QCoreApplication.translate('TextAdvancedFormatPanel', 'Slant'),
        SlantTextTransformStrategy(GlyphSlantLayoutRenderer),
        SLANT_CONTROLS,
    ),
)
TEXT_TRANSFORM_VARIANTS_BY_TYPE = {
    variant.transform_type: variant for variant in TEXT_TRANSFORM_VARIANTS
}
if set(TEXT_TRANSFORM_VARIANTS_BY_TYPE) != set(TEXT_TRANSFORM_TYPES):
    raise RuntimeError(
        'persisted and UI/runtime text-transform variants must be registered together'
    )


def text_transform_variant(transform_type: str) -> TextTransformVariantSpec:
    try:
        return TEXT_TRANSFORM_VARIANTS_BY_TYPE[transform_type]
    except KeyError as error:
        raise ValueError(
            f'unsupported live text transform type {transform_type}'
        ) from error


def text_transform_strategy(transform: TextTransform) -> TextTransformStrategy:
    return text_transform_variant(transform.transform_type).strategy
