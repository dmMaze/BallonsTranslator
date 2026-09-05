"""Bound generated effect reach without changing the saved stack."""

from dataclasses import replace
import math

from ballontranslator.utils.text_effects import (
    GlowEffect, ShadowEffect, StrokeEffect, TextEffectStack,
)


def limit_effect_radii(
    stack: TextEffectStack, font_size: float, max_reach: float
) -> TextEffectStack:
    """Saturate generated radii inside a shared logical halo budget.

    Stroke keeps its source width; Shadow and Glow use the remaining reach.
    Preserve distance, then blur, then spread so increasing spread stops at
    the boundary without pulling an existing shadow back toward the text.

    >>> stack = TextEffectStack(effects=(StrokeEffect(width=10.0),))
    >>> limit_effect_radii(stack, 200.0, 100.0).effects[0].width
    1.0
    """
    if font_size <= 0.0:
        return stack
    budget = max(0.0, max_reach) / font_size
    effects = list(stack.effects)
    stroke_reach = 0.0
    for index, effect in enumerate(effects):
        if isinstance(effect, StrokeEffect) and not effect.is_neutral():
            width = min(effect.width, budget * 2.0)
            if width != effect.width:
                effects[index] = replace(effect, width=width)
            stroke_reach = max(stroke_reach, width / 2.0)
    for index, effect in enumerate(effects):
        if effect.is_neutral():
            continue
        if isinstance(effect, ShadowEffect):
            remaining = max(
                0.0, budget - (0.0 if effect.shadow_type == 'inner' else stroke_reach)
            )
            angle = math.radians(effect.angle)
            projection = max(abs(math.cos(angle)), abs(math.sin(angle)))
            distance = min(effect.distance, remaining / projection)
            remaining = max(0.0, remaining - distance * projection)
            parameters = {'distance': distance}
            if effect.shadow_type != 'long':
                blur = min(effect.blur, remaining)
                parameters.update(blur=blur, spread=min(effect.spread, remaining - blur))
            if any(getattr(effect, name) != value for name, value in parameters.items()):
                effects[index] = replace(effect, **parameters)
        elif isinstance(effect, GlowEffect):
            remaining = max(
                0.0, budget - (0.0 if effect.glow_type == 'inner' else stroke_reach)
            )
            size = min(effect.size, remaining)
            spread = min(effect.spread, remaining - size)
            if size != effect.size or spread != effect.spread:
                effects[index] = replace(effect, size=size, spread=spread)
    return stack if tuple(effects) == stack.effects else replace(stack, effects=tuple(effects))
