from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from ballontranslator.utils.text_effects import (
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
    coerce_text_effect_stack,
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

    def test_ensure_inserts_default_primary_stroke(self):
        original = TextEffectStack(overall_opacity=0.4)

        result = ensure_primary_stroke(original)

        self.assertIsNone(primary_stroke(original))
        self.assertEqual(result.overall_opacity, 0.4)
        self.assertEqual(result.effects, (StrokeEffect(),))

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
            lambda: TextEffectStack(overall_opacity=1.1),
            lambda: TextEffectStack(effects=({'effect_type': 'stroke'},)),
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
                {'effect_type': 'shadow', 'blur': 0.5},
                {'effect_type': 'stroke', 'width': -1},
                {'effect_type': 'stroke', 'future_field': 1},
                {
                    'effect_type': 'stroke',
                    'paint': {'paint_type': 'gradient'},
                },
                {
                    'effect_type': 'stroke',
                    'width': 0.4,
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
                StrokeEffect(width=0.4, paint=SolidPaint((4, 5, 6))),
            ),
        )
        self.assertEqual(warning.call_count, 5)

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
        ),))

        payload = {
            'overall_opacity': 0.8,
            'effects': [{
                'effect_type': 'stroke',
                'enabled': False,
                'opacity': 0.5,
                'blend_mode': 'normal',
                'width': 0.25,
                'paint': {
                    'paint_type': 'solid',
                    'color': [7, 8, 9],
                },
            }],
        }
        self.assertEqual(stack.to_serializable_dict(), payload)
        self.assertEqual(coerce_text_effect_stack(payload), stack)
        self.assertIs(coerce_text_effect_stack(stack), stack)


if __name__ == '__main__':
    unittest.main()
