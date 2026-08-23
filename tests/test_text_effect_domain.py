from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from ballontranslator.utils.text_effects import (
    HollowEffect,
    SHADOW_BLUR_LIMIT,
    SHADOW_OFFSET_LIMIT,
    SHADOW_SPREAD_LIMIT,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
    coerce_text_effect_stack,
    effect_phase,
    ensure_primary_stroke,
    primary_stroke,
    with_non_stroke_effects,
    with_primary_stroke,
)


class TextEffectDomainTest(unittest.TestCase):
    def test_values_are_deeply_immutable(self):
        paint = SolidPaint([12, 34, 56])
        stroke = StrokeEffect(width=0.2, paint=paint)
        stack = TextEffectStack(0.75, [stroke])

        self.assertEqual(paint.color, (12, 34, 56))
        self.assertEqual(stack.effects, (stroke,))
        with self.assertRaises(FrozenInstanceError):
            stroke.width = 0.4
        with self.assertRaises(FrozenInstanceError):
            stack.effects = ()

        shadow = ShadowEffect(color=[4, 5, 6], offset=[0.2, -0.3])
        self.assertEqual(shadow.color, (4, 5, 6))
        self.assertEqual(shadow.offset, (0.2, -0.3))
        with self.assertRaises(FrozenInstanceError):
            shadow.blur = 0.2

    def test_ensure_inserts_default_primary_stroke(self):
        original = TextEffectStack(overall_opacity=0.4)

        result = ensure_primary_stroke(original)

        self.assertIsNone(primary_stroke(original))
        self.assertEqual(result.overall_opacity, 0.4)
        self.assertEqual(result.effects, (StrokeEffect(),))
        self.assertEqual(result.effects[0].position, 'center')

    def test_stroke_position_is_strict_and_keeps_positional_paint_compatibility(self):
        paint = SolidPaint((1, 2, 3))
        positional = StrokeEffect(True, 1.0, 'normal', 0.2, paint)

        self.assertIs(positional.paint, paint)
        self.assertEqual(positional.position, 'center')
        for position in ('inside', 'center', 'outside'):
            self.assertEqual(StrokeEffect(position=position).position, position)
        with self.assertRaises(ValueError):
            StrokeEffect(position='future')

    def test_ensure_and_equal_update_are_no_ops_for_existing_stroke(self):
        stack = TextEffectStack(effects=(StrokeEffect(width=0.3),))

        self.assertIs(ensure_primary_stroke(stack), stack)
        self.assertIs(with_primary_stroke(stack, width=0.3), stack)

    def test_update_preserves_unspecified_values_and_effect_order(self):
        first = StrokeEffect(
            enabled=False,
            opacity=0.4,
            width=0.2,
            paint=SolidPaint((10, 20, 30)),
            position='outside',
        )
        second = StrokeEffect(
            width=0.8,
            paint=SolidPaint((200, 210, 220)),
        )
        stack = TextEffectStack(0.6, (first, second))

        result = with_primary_stroke(stack, width=0.5)

        self.assertEqual(result.overall_opacity, 0.6)
        self.assertEqual(result.effects[0], StrokeEffect(
            enabled=False,
            opacity=0.4,
            width=0.5,
            paint=first.paint,
            position='outside',
        ))
        self.assertIs(result.effects[1], second)
        self.assertEqual(stack.effects, (first, second))

    def test_missing_stroke_is_inserted_then_updated(self):
        result = with_primary_stroke(
            TextEffectStack(),
            width=0.0,
            paint=SolidPaint((90, 80, 70)),
        )

        self.assertEqual(result.effects, (StrokeEffect(
            width=0.0,
            paint=SolidPaint((90, 80, 70)),
        ),))

    def test_non_stroke_override_preserves_all_stroke_cards(self):
        strokes = (
            StrokeEffect(width=0.2),
            StrokeEffect(width=0.7, paint=SolidPaint((4, 5, 6))),
        )
        target = TextEffectStack(0.4, strokes)
        source = TextEffectStack(0.8, (StrokeEffect(width=0.9),))

        result = with_non_stroke_effects(target, source)

        self.assertEqual(result.overall_opacity, 0.8)
        self.assertEqual(result.effects, strokes)
        self.assertIs(with_non_stroke_effects(result, source), result)

    def test_non_stroke_override_copies_typed_values_by_phase_boundary(self):
        first = StrokeEffect(width=0.2)
        second = StrokeEffect(width=0.7)
        drop = ShadowEffect(shadow_type='drop', offset=(0.2, 0.3))
        hollow = HollowEffect()
        inner = ShadowEffect(shadow_type='inner', blur=0.2)
        source = TextEffectStack(0.8, (drop, hollow, inner))
        target = TextEffectStack(0.4, (first, second))

        result = with_non_stroke_effects(target, source)

        self.assertEqual(
            result.effects, (drop, first, second, hollow, inner)
        )
        self.assertEqual(
            [effect for effect in result if not isinstance(effect, StrokeEffect)],
            [drop, hollow, inner],
        )

    def test_neutral_state_tracks_opacity_and_active_strokes(self):
        self.assertTrue(TextEffectStack().is_neutral())
        self.assertFalse(TextEffectStack(overall_opacity=0.5).is_neutral())
        self.assertTrue(TextEffectStack(
            effects=(StrokeEffect(enabled=False),)
        ).is_neutral())
        self.assertTrue(TextEffectStack(
            effects=(StrokeEffect(width=0.0),)
        ).is_neutral())
        self.assertTrue(TextEffectStack(
            effects=(StrokeEffect(opacity=0.0),)
        ).is_neutral())

        active = TextEffectStack(effects=(StrokeEffect(),))
        self.assertTrue(active.has_active_effects)
        self.assertFalse(active.is_neutral())
        self.assertTrue(TextEffectStack(
            effects=(ShadowEffect(enabled=False), HollowEffect(enabled=False))
        ).is_neutral())
        self.assertFalse(TextEffectStack(
            effects=(ShadowEffect(shadow_type='long'),)
        ).is_neutral())
        self.assertFalse(TextEffectStack(
            effects=(HollowEffect(),)
        ).is_neutral())

    def test_live_values_are_strictly_validated(self):
        invalid_constructors = (
            lambda: SolidPaint((0, 1, 256)),
            lambda: SolidPaint((0.0, 1, 2)),
            lambda: StrokeEffect(enabled=1),
            lambda: StrokeEffect(opacity=-0.1),
            lambda: StrokeEffect(opacity=float('nan')),
            lambda: StrokeEffect(blend_mode='multiply'),
            lambda: StrokeEffect(width=float('inf')),
            lambda: StrokeEffect(paint={'paint_type': 'solid'}),
            lambda: ShadowEffect(enabled=1),
            lambda: ShadowEffect(opacity=1.1),
            lambda: ShadowEffect(blend_mode='multiply'),
            lambda: ShadowEffect(shadow_type='outer'),
            lambda: ShadowEffect(color=(0, 1, 256)),
            lambda: ShadowEffect(offset=(0, float('inf'))),
            lambda: ShadowEffect(blur=-0.1),
            lambda: ShadowEffect(spread=-0.1),
            lambda: HollowEffect(enabled=1),
            lambda: TextEffectStack(overall_opacity=1.1),
            lambda: TextEffectStack(effects=({'effect_type': 'stroke'},)),
            lambda: TextEffectStack(effects=(HollowEffect(), HollowEffect())),
        )

        for constructor in invalid_constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()

        with self.assertRaisesRegex(ValueError, 'at least 0.0'):
            StrokeEffect(width=-0.1)

        stack = TextEffectStack()
        with self.assertRaises(ValueError):
            with_primary_stroke(stack, width=-0.1)
        with self.assertRaises(TypeError):
            with_primary_stroke(stack, paint={'paint_type': 'solid'})
        with self.assertRaises(TypeError):
            with_primary_stroke(stack, future_parameter=True)

    def test_payload_loading_drops_only_malformed_entries(self):
        payload = {
            'overall_opacity': 0.65,
            'future_stack_field': True,
            'effects': [
                {
                    'effect_type': 'stroke',
                    'width': 0.2,
                    'paint': {
                        'paint_type': 'solid',
                        'color': [1, 2, 3],
                    },
                },
                {'effect_type': 'glow', 'blur': 0.5},
                {'effect_type': 'stroke', 'width': -1},
                {'effect_type': 'stroke', 'position': 'diagonal'},
                {'effect_type': 'stroke', 'future_field': 1},
                {
                    'effect_type': 'stroke',
                    'paint': {'paint_type': 'gradient'},
                },
                {
                    'effect_type': 'stroke',
                    'width': 0.4,
                    'position': 'outside',
                    'paint': {
                        'paint_type': 'solid',
                        'color': [4, 5, 6],
                    },
                },
            ],
        }

        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            stack = coerce_text_effect_stack(payload)

        self.assertEqual(stack.overall_opacity, 0.65)
        self.assertEqual(
            stack.effects,
            (
                StrokeEffect(width=0.2, paint=SolidPaint((1, 2, 3))),
                StrokeEffect(
                    width=0.4,
                    paint=SolidPaint((4, 5, 6)),
                    position='outside',
                ),
            ),
        )
        self.assertEqual(warning.call_count, 6)

    def test_payload_keeps_mixed_order_and_isolates_duplicate_hollow(self):
        payload = {'effects': [
            {
                'effect_type': 'shadow',
                'shadow_type': 'drop',
                'offset': [0.2, -0.1],
                'blur': 0.3,
            },
            {'effect_type': 'stroke', 'width': 0.2},
            {'effect_type': 'hollow', 'enabled': True},
            {'effect_type': 'hollow', 'enabled': False},
            {
                'effect_type': 'shadow',
                'shadow_type': 'inner',
                'spread': 0.1,
            },
        ]}

        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            stack = coerce_text_effect_stack(payload)

        self.assertEqual(
            tuple(effect.effect_type for effect in stack.effects),
            ('shadow', 'stroke', 'hollow', 'shadow'),
        )
        self.assertEqual(stack.effects[0].shadow_type, 'drop')
        self.assertEqual(stack.effects[-1].shadow_type, 'inner')
        warning.assert_called_once()

    def test_effect_phase_is_fixed_by_typed_value(self):
        self.assertEqual(effect_phase(ShadowEffect()), 'exterior')
        self.assertEqual(
            effect_phase(ShadowEffect(shadow_type='long')), 'exterior'
        )
        self.assertEqual(
            effect_phase(ShadowEffect(shadow_type='inner')), 'interior'
        )
        self.assertEqual(effect_phase(StrokeEffect()), 'stroke')
        self.assertEqual(effect_phase(HollowEffect()), 'foreground')

    def test_shadow_geometry_limits_match_live_and_passive_boundaries(self):
        boundary = ShadowEffect(
            offset=(-SHADOW_OFFSET_LIMIT, SHADOW_OFFSET_LIMIT),
            blur=SHADOW_BLUR_LIMIT,
            spread=SHADOW_SPREAD_LIMIT,
        )
        self.assertEqual(boundary.offset, (-10.0, 10.0))
        with self.assertRaises(ValueError):
            ShadowEffect(offset=(SHADOW_OFFSET_LIMIT + 0.01, 0.0))
        with self.assertRaises(ValueError):
            ShadowEffect(blur=SHADOW_BLUR_LIMIT + 0.01)
        with self.assertRaises(ValueError):
            ShadowEffect(spread=SHADOW_SPREAD_LIMIT + 0.01)

        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            stack = coerce_text_effect_stack({'effects': [
                {
                    'effect_type': 'shadow',
                    'offset': [SHADOW_OFFSET_LIMIT + 0.01, 0],
                },
                {'effect_type': 'stroke', 'width': 0.2},
            ]})
        self.assertEqual(stack.effects, (StrokeEffect(width=0.2),))
        warning.assert_called_once()

    def test_invalid_stack_fields_fall_back_independently(self):
        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            stack = coerce_text_effect_stack({
                'overall_opacity': 'opaque',
                'effects': [{'effect_type': 'stroke', 'width': 0.2}],
            })

        self.assertEqual(
            stack,
            TextEffectStack(effects=(StrokeEffect(width=0.2),)),
        )
        self.assertEqual(warning.call_count, 1)

    def test_serialization_uses_stable_semantic_string_fields(self):
        stack = TextEffectStack(0.8, (StrokeEffect(
            enabled=False,
            opacity=0.5,
            width=0.25,
            paint=SolidPaint((7, 8, 9)),
            position='inside',
        ),))

        payload = {
            'overall_opacity': 0.8,
            'effects': [{
                'effect_type': 'stroke',
                'enabled': False,
                'opacity': 0.5,
                'blend_mode': 'normal',
                'width': 0.25,
                'position': 'inside',
                'paint': {
                    'paint_type': 'solid',
                    'color': [7, 8, 9],
                },
            }],
        }
        self.assertEqual(stack.to_serializable_dict(), payload)
        self.assertEqual(coerce_text_effect_stack(payload), stack)
        self.assertIs(coerce_text_effect_stack(stack), stack)

    def test_shadow_and_hollow_serialization_is_stable(self):
        stack = TextEffectStack(effects=(
            ShadowEffect(
                enabled=False,
                opacity=0.4,
                shadow_type='long',
                color=(9, 8, 7),
                offset=(-0.2, 0.4),
                blur=0.3,
                spread=0.1,
            ),
            HollowEffect(),
        ))

        payload = stack.to_serializable_dict()

        self.assertEqual(payload['effects'][0], {
            'effect_type': 'shadow',
            'enabled': False,
            'opacity': 0.4,
            'blend_mode': 'normal',
            'shadow_type': 'long',
            'color': [9, 8, 7],
            'offset': [-0.2, 0.4],
            'blur': 0.3,
            'spread': 0.1,
        })
        self.assertEqual(payload['effects'][1], {
            'effect_type': 'hollow',
            'enabled': True,
        })
        self.assertEqual(coerce_text_effect_stack(payload), stack)


if __name__ == '__main__':
    unittest.main()
