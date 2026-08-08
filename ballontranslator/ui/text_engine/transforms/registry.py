"""UI/runtime registration for supported text-transform variants."""

from dataclasses import dataclass
import math
from typing import Callable, Tuple

from qtpy.QtCore import QCoreApplication, QRectF
from qtpy.QtGui import QTransform

from ballontranslator.utils.fontformat import (
    TEXT_TRANSFORM_BEND_MAX,
    TEXT_TRANSFORM_BEND_MIN,
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    TEXT_TRANSFORM_GRID_DIVISION_MAX,
    TEXT_TRANSFORM_GRID_DIVISION_MIN,
    TEXT_TRANSFORM_PROJECTIVE_PERSPECTIVE_MAX,
    TEXT_TRANSFORM_PROJECTIVE_PERSPECTIVE_MIN,
    TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MAX,
    TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MIN,
    TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MAX,
    TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MIN,
    TEXT_TRANSFORM_PROJECTIVE_SLANT_MAX,
    TEXT_TRANSFORM_PROJECTIVE_SLANT_MIN,
    TEXT_TRANSFORM_SCALE_MAX,
    TEXT_TRANSFORM_SCALE_MIN,
    TEXT_TRANSFORM_SINE_AMPLITUDE_MAX,
    TEXT_TRANSFORM_SINE_AMPLITUDE_MIN,
    TEXT_TRANSFORM_SINE_FREQUENCY_MAX,
    TEXT_TRANSFORM_SINE_FREQUENCY_MIN,
    TEXT_TRANSFORM_SINE_PHASE_MAX,
    TEXT_TRANSFORM_SINE_PHASE_MIN,
    TEXT_TRANSFORM_TYPES,
    TextTransform,
    TextTransformStack,
)

from .mapping import (
    CompiledTextTransform,
    CompositeTextTransformMapper,
    CompiledTransformStage,
    MatrixTransformMapper,
    TransformStageContext,
    bend_transform_stage,
    grid_transform_stage,
    projective_transform_stage,
    rect_polygon,
    sine_transform_stage,
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
    shortcut: Callable[[], str] = None
    section: Callable[[], str] = None
    section_columns: int = 2


@dataclass(frozen=True)
class TextTransformVariantSpec:
    """Bind one persisted type to controls and one geometry-stage factory.

    >>> TEXT_TRANSFORM_VARIANTS[0].transform_type
    'projective'
    """

    transform_type: str
    label: Callable[[], str]
    icon_name: str
    stage_factory: Callable
    controls: Tuple[TransformControlSpec, ...] = ()


PROJECTIVE_CONTROLS = (
    TransformControlSpec(
        'horizontal_scale_control',
        'horizontal_scale',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Horizontal'
        ),
        100.0,
        TEXT_TRANSFORM_SCALE_MIN,
        TEXT_TRANSFORM_SCALE_MAX,
        '%',
        shortcut=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Shortcut: S → X'
        ),
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Scale'
        ),
    ),
    TransformControlSpec(
        'vertical_scale_control',
        'vertical_scale',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Vertical'
        ),
        100.0,
        TEXT_TRANSFORM_SCALE_MIN,
        TEXT_TRANSFORM_SCALE_MAX,
        '%',
        shortcut=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Shortcut: S → Y'
        ),
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Scale'
        ),
    ),
    TransformControlSpec(
        'horizontal_slant_control',
        'horizontal_slant',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Horizontal'
        ),
        1.0,
        TEXT_TRANSFORM_PROJECTIVE_SLANT_MIN,
        TEXT_TRANSFORM_PROJECTIVE_SLANT_MAX,
        '\N{DEGREE SIGN}',
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Slant'
        ),
    ),
    TransformControlSpec(
        'vertical_slant_control',
        'vertical_slant',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Vertical'
        ),
        1.0,
        TEXT_TRANSFORM_PROJECTIVE_SLANT_MIN,
        TEXT_TRANSFORM_PROJECTIVE_SLANT_MAX,
        '\N{DEGREE SIGN}',
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Slant'
        ),
    ),
    TransformControlSpec(
        'rotation_x_control',
        'rotation_x',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'X'
        ),
        1.0,
        TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MIN,
        TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MAX,
        '\N{DEGREE SIGN}',
        shortcut=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Shortcut: R → X'
        ),
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Rotation'
        ),
        section_columns=3,
    ),
    TransformControlSpec(
        'rotation_y_control',
        'rotation_y',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Y'
        ),
        1.0,
        TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MIN,
        TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MAX,
        '\N{DEGREE SIGN}',
        shortcut=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Shortcut: R → Y'
        ),
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Rotation'
        ),
        section_columns=3,
    ),
    TransformControlSpec(
        'rotation_z_control',
        'rotation_z',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Z'
        ),
        1.0,
        TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MIN,
        TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MAX,
        '\N{DEGREE SIGN}',
        shortcut=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Shortcut: R (or R → Z)'
        ),
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Rotation'
        ),
        section_columns=3,
    ),
    TransformControlSpec(
        'perspective_control',
        'perspective',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Perspective'
        ),
        100.0,
        TEXT_TRANSFORM_PROJECTIVE_PERSPECTIVE_MIN,
        TEXT_TRANSFORM_PROJECTIVE_PERSPECTIVE_MAX,
        '%',
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

BEND_CONTROLS = (
    TransformControlSpec(
        'bend_control',
        'bend',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Amount'
        ),
        100.0,
        TEXT_TRANSFORM_BEND_MIN,
        TEXT_TRANSFORM_BEND_MAX,
        '%',
    ),
)

SINE_CONTROLS = (
    TransformControlSpec(
        'sine_frequency_x_control',
        'frequency_x',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Segments'
        ),
        1.0,
        TEXT_TRANSFORM_SINE_FREQUENCY_MIN,
        TEXT_TRANSFORM_SINE_FREQUENCY_MAX,
        '',
        decimals=0,
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Left-to-Right Wave'
        ),
    ),
    TransformControlSpec(
        'sine_phase_x_control',
        'phase_x',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Shift'
        ),
        100.0,
        TEXT_TRANSFORM_SINE_PHASE_MIN,
        TEXT_TRANSFORM_SINE_PHASE_MAX,
        '%',
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Left-to-Right Wave'
        ),
    ),
    TransformControlSpec(
        'sine_amplitude_x_control',
        'amplitude_x',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Height'
        ),
        100.0,
        TEXT_TRANSFORM_SINE_AMPLITUDE_MIN,
        TEXT_TRANSFORM_SINE_AMPLITUDE_MAX,
        '%',
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Left-to-Right Wave'
        ),
    ),
    TransformControlSpec(
        'sine_frequency_y_control',
        'frequency_y',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Segments'
        ),
        1.0,
        TEXT_TRANSFORM_SINE_FREQUENCY_MIN,
        TEXT_TRANSFORM_SINE_FREQUENCY_MAX,
        '',
        decimals=0,
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Top-to-Bottom Wave'
        ),
    ),
    TransformControlSpec(
        'sine_phase_y_control',
        'phase_y',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Shift'
        ),
        100.0,
        TEXT_TRANSFORM_SINE_PHASE_MIN,
        TEXT_TRANSFORM_SINE_PHASE_MAX,
        '%',
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Top-to-Bottom Wave'
        ),
    ),
    TransformControlSpec(
        'sine_amplitude_y_control',
        'amplitude_y',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Width'
        ),
        100.0,
        TEXT_TRANSFORM_SINE_AMPLITUDE_MIN,
        TEXT_TRANSFORM_SINE_AMPLITUDE_MAX,
        '%',
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Top-to-Bottom Wave'
        ),
    ),
)

GRID_CONTROLS = (
    TransformControlSpec(
        'grid_horizontal_divisions_control',
        'horizontal_divisions',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Horizontal'
        ),
        1.0,
        TEXT_TRANSFORM_GRID_DIVISION_MIN,
        TEXT_TRANSFORM_GRID_DIVISION_MAX,
        '',
        decimals=0,
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Divisions'
        ),
    ),
    TransformControlSpec(
        'grid_vertical_divisions_control',
        'vertical_divisions',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Vertical'
        ),
        1.0,
        TEXT_TRANSFORM_GRID_DIVISION_MIN,
        TEXT_TRANSFORM_GRID_DIVISION_MAX,
        '',
        decimals=0,
        section=lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Divisions'
        ),
    ),
    TransformControlSpec(
        'grid_interpolation_control',
        'interpolation',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Interpolation'
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
        'projective',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Scale / Slant / 3D'
        ),
        'text_transform_projective.svg',
        projective_transform_stage,
        PROJECTIVE_CONTROLS,
    ),
    TextTransformVariantSpec(
        'bend',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Bend'
        ),
        'text_transform_bend.svg',
        bend_transform_stage,
        BEND_CONTROLS,
    ),
    TextTransformVariantSpec(
        'sine',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Sine Wave'
        ),
        'text_transform_sine.svg',
        sine_transform_stage,
        SINE_CONTROLS,
    ),
    TextTransformVariantSpec(
        'grid',
        lambda: QCoreApplication.translate(
            'TextTransformPanel', 'Grid'
        ),
        'text_transform_grid.svg',
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
    stack: TextTransformStack,
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
    if not isinstance(stack, TextTransformStack):
        raise TypeError('text transform compiler requires TextTransformStack')
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
