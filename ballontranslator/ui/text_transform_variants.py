"""UI/runtime registration for supported text-transform variants."""

from dataclasses import dataclass
import math
from typing import Callable, Tuple

from qtpy.QtCore import QCoreApplication, QRectF
from qtpy.QtGui import QTransform

from ballontranslator.utils.fontformat import (
    TEXT_TRANSFORM_CURVATURE_MAX,
    TEXT_TRANSFORM_CURVATURE_MIN,
    TEXT_TRANSFORM_BOX_SLANT_MAX,
    TEXT_TRANSFORM_BOX_SLANT_MIN,
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    TEXT_TRANSFORM_GRID_DIVISION_MAX,
    TEXT_TRANSFORM_GRID_DIVISION_MIN,
    TEXT_TRANSFORM_PERSPECTIVE_DIRECTION_MAX,
    TEXT_TRANSFORM_PERSPECTIVE_DIRECTION_MIN,
    TEXT_TRANSFORM_PERSPECTIVE_STRENGTH_MAX,
    TEXT_TRANSFORM_PERSPECTIVE_STRENGTH_MIN,
    TEXT_TRANSFORM_SCALE_MAX,
    TEXT_TRANSFORM_SCALE_MIN,
    TEXT_TRANSFORM_TYPES,
    TextTransformStack,
    coerce_text_transform_stack,
)

from .text_transform import (
    CompiledTextTransform,
    CompositeTextTransformMapper,
    CompiledTransformStage,
    MatrixTransformMapper,
    TransformStageContext,
    curvature_transform_stage,
    grid_transform_stage,
    perspective_transform_stage,
    rect_polygon,
    slant_transform_stage,
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
    decimals: int = 1
    choices: tuple = ()


@dataclass(frozen=True)
class TextTransformVariantSpec:
    """Bind one persisted type to controls and one geometry-stage factory.

    >>> TEXT_TRANSFORM_VARIANTS[0].transform_type
    'slant'
    """

    transform_type: str
    label: Callable[[], str]
    stage_factory: Callable
    controls: Tuple[TransformControlSpec, ...] = ()


SLANT_CONTROLS = (
    TransformControlSpec(
        'horizontal_scale_control',
        'horizontal_scale',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Horizontal Scale'
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
            'TextTransformPanel', 'Vertical Scale'
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
            'TextTransformPanel', 'Box Slant'
        ),
        1.0,
        TEXT_TRANSFORM_BOX_SLANT_MIN,
        TEXT_TRANSFORM_BOX_SLANT_MAX,
        '\N{DEGREE SIGN}',
    ),
)

GLYPH_SLANT_CONTROL = TransformControlSpec(
    'glyph_slant_angle_control',
    'glyph_slant_angle',
    lambda: QCoreApplication.translate(
        'TextTransformPanel', 'Glyph Slant'
    ),
    1.0,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    '\N{DEGREE SIGN}',
)

PERSPECTIVE_CONTROLS = (
    TransformControlSpec(
        'perspective_strength_control',
        'strength',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Strength'
        ),
        100.0,
        TEXT_TRANSFORM_PERSPECTIVE_STRENGTH_MIN,
        TEXT_TRANSFORM_PERSPECTIVE_STRENGTH_MAX,
        '%',
    ),
    TransformControlSpec(
        'perspective_direction_control',
        'direction',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Direction'
        ),
        1.0,
        TEXT_TRANSFORM_PERSPECTIVE_DIRECTION_MIN,
        TEXT_TRANSFORM_PERSPECTIVE_DIRECTION_MAX,
        '\N{DEGREE SIGN}',
    ),
)

CURVATURE_CONTROLS = (
    TransformControlSpec(
        'curvature_control',
        'curvature',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Curvature'
        ),
        100.0,
        TEXT_TRANSFORM_CURVATURE_MIN,
        TEXT_TRANSFORM_CURVATURE_MAX,
        '%',
    ),
)

GRID_CONTROLS = (
    TransformControlSpec(
        'grid_horizontal_divisions_control',
        'horizontal_divisions',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Horizontal Divisions'
        ),
        1.0,
        TEXT_TRANSFORM_GRID_DIVISION_MIN,
        TEXT_TRANSFORM_GRID_DIVISION_MAX,
        '',
        decimals=0,
    ),
    TransformControlSpec(
        'grid_vertical_divisions_control',
        'vertical_divisions',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Vertical Divisions'
        ),
        1.0,
        TEXT_TRANSFORM_GRID_DIVISION_MIN,
        TEXT_TRANSFORM_GRID_DIVISION_MAX,
        '',
        decimals=0,
    ),
    TransformControlSpec(
        'grid_sampling_control',
        'sampling',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Sampling'
        ),
        1.0,
        0.0,
        0.0,
        '',
        choices=(
            (
                'bilinear',
                lambda: QCoreApplication.translate(
                    'TextTransformPanel', 'Straight'
                ),
            ),
            (
                'catmull_rom',
                lambda: QCoreApplication.translate(
                    'TextTransformPanel', 'Smooth'
                ),
            ),
        ),
    ),
)


TEXT_TRANSFORM_VARIANTS = (
    TextTransformVariantSpec(
        'slant',
        lambda: QCoreApplication.translate('TextTransformPanel', 'Slant'),
        slant_transform_stage,
        SLANT_CONTROLS,
    ),
    TextTransformVariantSpec(
        'perspective',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Perspective'
        ),
        perspective_transform_stage,
        PERSPECTIVE_CONTROLS,
    ),
    TextTransformVariantSpec(
        'curvature',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Curvature'
        ),
        curvature_transform_stage,
        CURVATURE_CONTROLS,
    ),
    TextTransformVariantSpec(
        'grid',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Grid'
        ),
        grid_transform_stage,
        GRID_CONTROLS,
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


def _validate_matrix_stage(matrix: QTransform, source_bounds: QRectF) -> None:
    coefficients = (
        matrix.m11(), matrix.m12(), matrix.m13(),
        matrix.m21(), matrix.m22(), matrix.m23(),
        matrix.m31(), matrix.m32(), matrix.m33(),
    )
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError('transform matrix must be finite and invertible')
    _, invertible = matrix.inverted()
    if not invertible:
        raise ValueError('transform matrix must be finite and invertible')

    corners = (
        source_bounds.topLeft(),
        source_bounds.topRight(),
        source_bounds.bottomRight(),
        source_bounds.bottomLeft(),
    )
    denominators = [
        matrix.m13() * point.x()
        + matrix.m23() * point.y()
        + matrix.m33()
        for point in corners
    ]
    maximum = max(abs(value) for value in denominators)
    if (
        maximum == 0.0
        or min(abs(value) for value in denominators) <= maximum * 1e-9
        or min(denominators) < 0.0 < max(denominators)
    ):
        raise ValueError('projective transform crosses its source horizon')
    for point in corners:
        mapped = matrix.map(point)
        if not math.isfinite(mapped.x()) or not math.isfinite(mapped.y()):
            raise ValueError('transform matrix must map to finite coordinates')


_NONLINEAR_MAPPER_METHODS = (
    'forward_point',
    'forward_arrays',
    'inverse_point',
    'inverse_arrays',
    'visual_bounds',
    'geometry_key',
)


def compile_text_transform_stack(
    stack,
    logical_rect: QRectF,
    source_rect: QRectF,
    vertical: bool,
) -> CompiledTextTransform:
    """Compile registered operations to one native matrix or surface mapper.

    >>> compiled = compile_text_transform_stack(
    ...     TextTransformStack(), QRectF(0, 0, 10, 5),
    ...     QRectF(0, 0, 10, 5), False,
    ... )
    >>> compiled.is_identity
    True
    """
    stack = coerce_text_transform_stack(stack)
    original_logical = QRectF(logical_rect)
    original_source = QRectF(source_rect)
    logical_bounds = QRectF(original_logical)
    source_bounds = QRectF(original_source)
    runtime_stages = []
    stage_records = []
    combined_matrix = QTransform()
    has_nonlinear = False

    for transform_index, transform in enumerate(stack):
        context = TransformStageContext(
            QRectF(logical_bounds),
            QRectF(source_bounds),
            bool(vertical),
        )
        if transform.is_neutral():
            stage_records.append(
                CompiledTransformStage(
                    transform_index, transform, context
                )
            )
            continue
        stage = text_transform_variant(
            transform.transform_type
        ).stage_factory(transform, context)
        if transform.is_nonlinear:
            if isinstance(stage, QTransform) or not all(
                hasattr(stage, name) for name in _NONLINEAR_MAPPER_METHODS
            ):
                raise TypeError(
                    f'{transform.transform_type} must build a nonlinear mapper'
                )
            mapper_stage = stage
            has_nonlinear = True
        else:
            if not isinstance(stage, QTransform):
                raise TypeError(
                    f'{transform.transform_type} must build a QTransform'
                )
            _validate_matrix_stage(stage, source_bounds)
            mapper_stage = MatrixTransformMapper(stage)
            if not has_nonlinear:
                combined_matrix = combined_matrix * stage

        stage_records.append(
            CompiledTransformStage(
                transform_index,
                transform,
                context,
                mapper_stage,
            )
        )

        # Folding as stages are added keeps deep matrix runs cheap while
        # preserving one inverse mapper on the nonlinear rendering path.
        if (
            not transform.is_nonlinear
            and runtime_stages
            and isinstance(runtime_stages[-1], MatrixTransformMapper)
        ):
            runtime_stages[-1] = MatrixTransformMapper(
                runtime_stages[-1].matrix * mapper_stage.matrix
            )
        else:
            runtime_stages.append(mapper_stage)

        if transform_index == len(stack) - 1:
            # Bounds are stage input, so the final output outline is computed
            # lazily by the item and renderer instead of twice per mouse move.
            continue

        if has_nonlinear:
            partial = CompositeTextTransformMapper(
                runtime_stages,
                original_logical,
                original_source,
                vertical,
            )
            logical_bounds = partial.visual_bounds(original_logical)
            source_bounds = partial.visual_bounds(original_source)
        else:
            # Projective matrices map each edge to another straight edge while
            # its homogeneous denominator stays away from the horizon.
            logical_bounds = combined_matrix.map(
                rect_polygon(original_logical)
            ).boundingRect()
            source_bounds = combined_matrix.map(
                rect_polygon(original_source)
            ).boundingRect()
        if (
            logical_bounds.isEmpty()
            or source_bounds.isEmpty()
            or not all(
                math.isfinite(value)
                for value in (
                    logical_bounds.left(),
                    logical_bounds.top(),
                    logical_bounds.right(),
                    logical_bounds.bottom(),
                    source_bounds.left(),
                    source_bounds.top(),
                    source_bounds.right(),
                    source_bounds.bottom(),
                )
            )
        ):
            raise ValueError('transform stage produced invalid bounds')

    if not runtime_stages:
        return CompiledTextTransform(
            stack,
            QTransform(),
            stages=tuple(stage_records),
        )

    if has_nonlinear:
        return CompiledTextTransform(
            stack,
            QTransform(),
            CompositeTextTransformMapper(
                runtime_stages,
                original_logical,
                original_source,
                vertical,
            ),
            tuple(stage_records),
        )

    _validate_matrix_stage(combined_matrix, original_source)
    return CompiledTextTransform(
        stack,
        combined_matrix,
        stages=tuple(stage_records),
    )
