from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from ballontranslator.utils.raster_assets import RasterAssetRef
from ballontranslator.utils.text_effects import (
    FilterEffect,
    GlowEffect,
    TextFillEffect,
    GradientStop,
    HollowEffect,
    ImageEffect,
    ImageGenerationRecipe,
    LinearGradientPaint,
    SHADOW_BLUR_LIMIT,
    SHADOW_DISTANCE_LIMIT,
    SHADOW_SPREAD_LIMIT,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TEXT_EFFECT_BLEND_MODES,
    TextEffectStack,
    TexturePaint,
    coerce_text_effect_stack,
    effect_phase,
    effect_structure_key,
    effect_paint_fallback_color,
    ensure_primary_stroke,
    primary_stroke,
    with_non_stroke_effects,
    with_primary_stroke,
)


class TextEffectDomainTest(unittest.TestCase):
    def test_image_effect_is_repeatable_hashable_strict_and_neutral_empty(self):
        asset = RasterAssetRef('assets/' + 'a' * 64 + '.png')
        effect = ImageEffect(asset, mode='background')
        stack = coerce_text_effect_stack({'effects': (
            effect.to_serializable_dict(),
            ImageEffect().to_serializable_dict(),
            {'effect_type': 'image', 'mode': 'invalid'},
        )})

        self.assertEqual(
            stack.effects, (effect, ImageEffect(), ImageEffect())
        )
        self.assertEqual(effect_phase(effect), 'image')
        self.assertEqual(hash(effect), hash(ImageEffect(asset, mode='background')))
        self.assertTrue(ImageEffect().is_neutral())
        self.assertEqual(ImageEffect().mode, 'foreground')
        with self.assertRaises(ValueError):
            ImageEffect(asset, mode='replace')
        with self.assertRaises(ValueError):
            ImageEffect(asset, mode='overlay')

    def test_image_generation_recipe_round_trip_and_passive_recovery(self):
        asset = RasterAssetRef(
            'assets/' + '9' * 64 + '.png', 'generated.png'
        )
        recipe = ImageGenerationRecipe(
            backend='future-local',
            profile_id='artist',
            model='diffusion-v2',
            context='lettered',
            prompt='Paint lettering texture',
        )
        effect = ImageEffect(asset, mode='foreground', generation=recipe)

        self.assertEqual(
            coerce_text_effect_stack({
                'effects': [effect.to_serializable_dict()]
            }).effects,
            (effect,),
        )

        malformed = effect.to_serializable_dict()
        malformed.update({
            'asset': {'path': '../outside.png'},
            'enabled': 'yes',
            'mode': 'future-mode',
            'future': 'ignored',
            'generation': {
                'backend': 'removed-backend',
                'profile_id': 5,
                'model': 'kept-model',
                'context': 'future-context',
                'prompt': ['bad'],
                'future': True,
            },
        })
        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            loaded = coerce_text_effect_stack({'effects': [malformed]})[0]

        self.assertEqual(loaded.asset, None)
        self.assertTrue(loaded.enabled)
        self.assertEqual(loaded.mode, 'foreground')
        self.assertEqual(loaded.generation, ImageGenerationRecipe(
            backend='removed-backend',
            model='kept-model',
        ))
        self.assertGreaterEqual(warning.call_count, 6)

    def test_filter_effect_is_repeatable_hashable_and_structurally_typed(self):
        noise = FilterEffect(
            'builtin:noise', params={'seed': 7, 'amount': 0.25}
        )
        newer = FilterEffect(
            'builtin:noise', schema_version=2, params=noise.params
        )
        stack = TextEffectStack(effects=(noise, noise, newer))

        self.assertEqual(hash(noise), hash(FilterEffect(
            'builtin:noise', params=(('amount', 0.25), ('seed', 7))
        )))
        self.assertEqual(noise.params, (('amount', 0.25), ('seed', 7)))
        self.assertEqual(effect_phase(noise), 'filter')
        self.assertEqual(
            effect_structure_key(noise), ('filter', 'builtin:noise', 1)
        )
        self.assertNotEqual(
            effect_structure_key(noise), effect_structure_key(newer)
        )
        self.assertEqual(len(stack.effects), 3)

    def test_filter_effect_passive_load_isolates_generic_fields_and_params(self):
        stack = coerce_text_effect_stack({'effects': (
            {
                'effect_type': 'filter',
                'filter_id': 'future:kept',
                'schema_version': 9,
                'enabled': False,
                'params': {'future': 'opaque', 'bad': [1, 2]},
            },
            {
                'effect_type': 'filter',
                'filter_id': 'future:defaults',
                'schema_version': 0,
                'enabled': 'bad',
                'params': 'bad',
            },
            StrokeEffect(width=0.3).to_serializable_dict(),
        )})

        self.assertEqual(stack.effects[0], FilterEffect(
            'future:kept', 9, False, {'future': 'opaque'}
        ))
        self.assertEqual(stack.effects[1], FilterEffect('future:defaults'))
        self.assertIsInstance(stack.effects[2], StrokeEffect)

    def test_linear_gradient_values_are_immutable_and_strict(self):
        hard_transition = LinearGradientPaint(
            stops=(
                GradientStop(0.0, (1, 2, 3), 0.0),
                GradientStop(0.5, (4, 5, 6), 0.5),
                GradientStop(0.5, (7, 8, 9), 1.0),
                GradientStop(1.0, (10, 11, 12), 1.0),
            ),
            angle=-90,
            scale=4.0,
        )

        self.assertEqual(hard_transition.angle, 270.0)
        self.assertEqual(hard_transition.scale, 4.0)
        self.assertEqual(len(hard_transition.stops), 4)
        with self.assertRaises(FrozenInstanceError):
            hard_transition.angle = 0.0

        invalid_values = (
            lambda: GradientStop(-0.001),
            lambda: GradientStop(1.001),
            lambda: GradientStop(float('nan')),
            lambda: GradientStop(color=(0, 1, 256)),
            lambda: GradientStop(opacity=-0.001),
            lambda: GradientStop(opacity=1.001),
            lambda: LinearGradientPaint(stops=(GradientStop(),)),
            lambda: LinearGradientPaint(
                stops=tuple(GradientStop(index / 32) for index in range(33))
            ),
            lambda: LinearGradientPaint(stops=(
                GradientStop(0.8), GradientStop(0.2)
            )),
            lambda: LinearGradientPaint(stops=(
                GradientStop(), {'position': 1.0}
            )),
            lambda: LinearGradientPaint(angle=float('inf')),
            lambda: LinearGradientPaint(scale=0.099),
            lambda: LinearGradientPaint(scale=4.001),
        )
        for constructor in invalid_values:
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()

        self.assertEqual(LinearGradientPaint(angle=360).angle, 0.0)
        self.assertEqual(LinearGradientPaint(scale=0.1).scale, 0.1)
        self.assertIsInstance(LinearGradientPaint(
            stops=[GradientStop(), GradientStop(1.0)]
        ).stops, tuple)
        self.assertEqual(len(LinearGradientPaint(stops=tuple(
            GradientStop(index / 31) for index in range(32)
        )).stops), 32)

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

        shadow = ShadowEffect(
            paint=SolidPaint([4, 5, 6]), angle=315.0, distance=0.3
        )
        self.assertEqual(shadow.paint, SolidPaint((4, 5, 6)))
        self.assertEqual(shadow.angle, 315.0)
        self.assertEqual(shadow.distance, 0.3)
        with self.assertRaises(FrozenInstanceError):
            shadow.blur = 0.2

    def test_ensure_inserts_default_primary_stroke(self):
        original = TextEffectStack(overall_opacity=0.4)

        result = ensure_primary_stroke(original)

        self.assertIsNone(primary_stroke(original))
        self.assertEqual(result.overall_opacity, 0.4)
        self.assertEqual(result.effects, (StrokeEffect(),))
        self.assertEqual(result.effects[0].position, 'outside')

    def test_stroke_position_defaults_outside_and_keeps_positional_paint(self):
        paint = SolidPaint((1, 2, 3))
        positional = StrokeEffect(True, 1.0, 'normal', 0.2, paint)

        self.assertIs(positional.paint, paint)
        self.assertEqual(positional.position, 'outside')
        for position in ('inside', 'center', 'outside'):
            self.assertEqual(StrokeEffect(position=position).position, position)
        with self.assertRaises(ValueError):
            StrokeEffect(position='future')

    def test_typed_stroke_without_position_keeps_center_compatibility(self):
        stack = coerce_text_effect_stack({
            'effects': [{'effect_type': 'stroke', 'width': 0.2}],
        })

        self.assertEqual(stack.effects[0].position, 'center')

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

    def test_run_inserted_stroke_is_applied_last(self):
        filter_effect = FilterEffect('builtin:noise')
        text_fill = TextFillEffect()

        result = ensure_primary_stroke(TextEffectStack(effects=(
            filter_effect, text_fill
        )))

        self.assertIsInstance(result.effects[0], StrokeEffect)
        self.assertIs(result.effects[1], filter_effect)
        self.assertIs(result.effects[2], text_fill)

    def test_run_stroke_ignores_legacy_mid_structural_position(self):
        filter_effect = FilterEffect('builtin:noise')
        inner = ShadowEffect(shadow_type='inner')
        text_fill = TextFillEffect()
        legacy = ensure_primary_stroke(TextEffectStack(effects=(
            filter_effect, text_fill, inner
        )))
        normalized = ensure_primary_stroke(TextEffectStack(effects=(
            filter_effect, inner, text_fill
        )))

        movable_types = (
            FilterEffect, StrokeEffect, ShadowEffect, GlowEffect
        )
        self.assertEqual(
            tuple(
                type(effect) for effect in legacy.effects
                if isinstance(effect, movable_types)
            ),
            tuple(
                type(effect) for effect in normalized.effects
                if isinstance(effect, movable_types)
            ),
        )
        self.assertEqual(
            tuple(
                type(effect) for effect in legacy.effects
                if isinstance(effect, movable_types)
            ),
            (StrokeEffect, FilterEffect, ShadowEffect),
        )

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

    def test_non_stroke_override_ignores_mid_structural_position(self):
        first = StrokeEffect(width=0.2)
        second = StrokeEffect(width=0.7)
        drop = ShadowEffect(
            shadow_type='drop', angle=45.0, distance=0.3
        )
        glow = GlowEffect(size=0.2)
        hollow = HollowEffect()
        inner = ShadowEffect(shadow_type='inner', blur=0.2)
        text_fill = TextFillEffect()
        source = TextEffectStack(
            0.8, (drop, glow, hollow, text_fill, inner)
        )
        target = TextEffectStack(0.4, (first, second))

        result = with_non_stroke_effects(target, source)

        self.assertEqual(
            result.effects,
            (drop, glow, hollow, text_fill, inner, first, second),
        )
        self.assertEqual(
            [effect for effect in result if not isinstance(effect, StrokeEffect)],
            [drop, glow, hollow, text_fill, inner],
        )

        normalized = with_non_stroke_effects(
            target,
            TextEffectStack(
                0.8, (drop, glow, inner, hollow, text_fill)
            ),
        )
        movable_types = (
            FilterEffect, StrokeEffect, ShadowEffect, GlowEffect
        )
        self.assertEqual(
            tuple(
                type(effect) for effect in result.effects
                if isinstance(effect, movable_types)
            ),
            tuple(
                type(effect) for effect in normalized.effects
                if isinstance(effect, movable_types)
            ),
        )

    def test_text_fill_is_repeatable_strict_and_replacement_active(self):
        transparent = LinearGradientPaint(stops=(
            GradientStop(0.0, (255, 0, 0), 0.0),
            GradientStop(1.0, (0, 0, 255), 0.0),
        ))
        self.assertTrue(TextFillEffect(enabled=False).is_neutral())
        self.assertFalse(TextFillEffect(opacity=0.0).is_neutral())
        self.assertFalse(
            TextFillEffect(paint=transparent).is_neutral()
        )
        self.assertEqual(effect_phase(TextFillEffect()), 'foreground')
        self.assertEqual(
            effect_structure_key(TextFillEffect()),
            ('text_fill', 'linear_gradient'),
        )
        self.assertEqual(
            effect_structure_key(TextFillEffect(
                paint=TexturePaint()
            )),
            ('text_fill', 'texture'),
        )
        fills = (TextFillEffect(), TextFillEffect(opacity=0.5))
        self.assertEqual(TextEffectStack(effects=fills).effects, fills)
        for constructor in (
            lambda: TextFillEffect(enabled=1),
            lambda: TextFillEffect(blend_mode='overlay'),
            lambda: TextFillEffect(opacity=-0.01),
            lambda: TextFillEffect(opacity=1.01),
            lambda: TextFillEffect(paint=SolidPaint()),
            lambda: TextFillEffect(paint=object()),
        ):
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()

    def test_text_fill_payload_round_trip_and_isolation(self):
        text_fill = TextFillEffect(
            paint=LinearGradientPaint(
                stops=(
                    GradientStop(0.0, (1, 2, 3), 0.25),
                    GradientStop(1.0, (4, 5, 6), 0.75),
                ),
                angle=45.0,
                scale=1.5,
            ),
        )
        payload = TextEffectStack(effects=(
            StrokeEffect(width=0.2), text_fill,
        )).to_serializable_dict()
        self.assertEqual(payload['effects'][1], {
            'effect_type': 'text_fill',
            'enabled': True,
            'opacity': 1.0,
            'blend_mode': 'normal',
            'paint': text_fill.paint.to_serializable_dict(),
        })
        self.assertEqual(
            coerce_text_effect_stack(payload).effects,
            (StrokeEffect(width=0.2), text_fill),
        )
        for legacy_type in ('gradient', 'gradient_overlay'):
            legacy_payload = dict(payload['effects'][1])
            legacy_payload.update({
                'effect_type': legacy_type,
                'opacity': 0.4,
            })
            self.assertEqual(
                coerce_text_effect_stack({
                    'effects': [legacy_payload]
                }).effects,
                (text_fill,),
            )

        old_single_payload = dict(payload['effects'][1])
        old_single_payload.pop('opacity')
        self.assertEqual(
            coerce_text_effect_stack({
                'effects': [old_single_payload]
            }).effects,
            (text_fill,),
        )

        malformed = {'effects': [
            payload['effects'][1],
            {'effect_type': 'text_fill', 'opacity': -0.1},
            {'effect_type': 'text_fill'},
            {'effect_type': 'stroke', 'width': 0.3},
            {
                'effect_type': 'text_fill',
                'paint': {'paint_type': 'solid', 'color': [1, 2, 3]},
            },
        ]}
        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            loaded = coerce_text_effect_stack(malformed)
        self.assertEqual(loaded.effects, (
            text_fill,
            StrokeEffect(width=0.3, position='center'),
        ))
        self.assertEqual(warning.call_count, 3)

    def test_blend_modes_are_strict_live_and_recover_on_passive_load(self):
        self.assertEqual(TEXT_EFFECT_BLEND_MODES, (
            'normal',
            'darken', 'multiply', 'color_burn', 'linear_burn',
            'darker_color',
            'lighten', 'screen', 'color_dodge', 'linear_dodge',
            'lighter_color',
        ))
        constructors = (
            StrokeEffect,
            ShadowEffect,
            GlowEffect,
            TextFillEffect,
        )
        for constructor in constructors:
            for blend_mode in TEXT_EFFECT_BLEND_MODES:
                with self.subTest(
                    effect=constructor.__name__, blend_mode=blend_mode
                ):
                    self.assertEqual(
                        constructor(blend_mode=blend_mode).blend_mode,
                        blend_mode,
                    )
                    effect = constructor(blend_mode=blend_mode)
                    self.assertEqual(
                        coerce_text_effect_stack({
                            'effects': [effect.to_serializable_dict()]
                        }).effects,
                        (effect,),
                    )
            with self.subTest(effect=constructor.__name__, invalid='strict'):
                with self.assertRaises(ValueError):
                    constructor(blend_mode='overlay')

        payloads = []
        for constructor in constructors:
            payload = constructor().to_serializable_dict()
            payload['blend_mode'] = 'future-mode'
            payloads.append(payload)
        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            loaded = coerce_text_effect_stack({'effects': payloads})

        self.assertEqual(
            tuple(type(effect) for effect in loaded.effects), constructors
        )
        self.assertTrue(all(
            effect.blend_mode == 'normal' for effect in loaded.effects
        ))
        self.assertEqual(warning.call_count, len(constructors))

    def test_texture_paint_round_trip_and_asset_reference_validation(self):
        asset = RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'paper.png'
        )
        texture = TexturePaint(asset, mapping='tile', scale=1.5)
        fill = TextFillEffect(paint=texture)

        loaded = coerce_text_effect_stack(
            TextEffectStack(effects=(fill,)).to_serializable_dict()
        )

        self.assertEqual(loaded.effects, (fill,))
        self.assertEqual(texture.to_serializable_dict()['asset'], {
            'path': asset.path,
            'display_name': 'paper.png',
        })
        empty = TextFillEffect(paint=TexturePaint())
        self.assertTrue(empty.is_neutral())
        empty_loaded = coerce_text_effect_stack(
            TextEffectStack(effects=(empty,)).to_serializable_dict()
        )
        self.assertEqual(empty_loaded.effects, (empty,))
        self.assertIsNone(empty.paint.to_serializable_dict()['asset'])
        for effect_type in (StrokeEffect, ShadowEffect, GlowEffect):
            with self.subTest(effect_type=effect_type.__name__):
                with self.assertRaises(TypeError):
                    effect_type(paint=texture)
        for constructor in (
            lambda: RasterAssetRef('../paper.png'),
            lambda: RasterAssetRef('/tmp/paper.png'),
            lambda: RasterAssetRef(
                'assets/' + 'a' * 64 + '.png', '../paper.png'
            ),
            lambda: TexturePaint(asset, mapping='future'),
            lambda: TexturePaint(asset, scale=0.09),
            lambda: TexturePaint('paper.png'),
        ):
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()

    def test_glow_is_strict_repeatable_neutral_and_serializable(self):
        transparent = LinearGradientPaint(stops=(
            GradientStop(0.0, (255, 0, 0), 0.0),
            GradientStop(1.0, (0, 0, 255), 0.0),
        ))
        outer = GlowEffect(
            opacity=0.6,
            paint=LinearGradientPaint(angle=45.0, scale=1.5),
            size=SHADOW_BLUR_LIMIT,
            spread=SHADOW_SPREAD_LIMIT,
        )
        inner = GlowEffect(
            glow_type='inner', paint=SolidPaint((4, 5, 6)), size=0.3
        )
        stack = TextEffectStack(effects=(outer, inner))

        self.assertEqual(effect_phase(outer), 'exterior')
        self.assertEqual(effect_phase(inner), 'interior')
        self.assertEqual(
            coerce_text_effect_stack(stack.to_serializable_dict()), stack
        )
        self.assertTrue(GlowEffect(enabled=False).is_neutral())
        self.assertTrue(GlowEffect(opacity=0.0).is_neutral())
        self.assertTrue(GlowEffect(size=0.0, spread=0.0).is_neutral())
        self.assertTrue(GlowEffect(paint=transparent).is_neutral())
        self.assertFalse(GlowEffect(size=0.0, spread=0.1).is_neutral())

        invalid_values = (
            lambda: GlowEffect(enabled=1),
            lambda: GlowEffect(opacity=1.01),
            lambda: GlowEffect(blend_mode='overlay'),
            lambda: GlowEffect(glow_type='future'),
            lambda: GlowEffect(paint=object()),
            lambda: GlowEffect(size=-0.01),
            lambda: GlowEffect(size=SHADOW_BLUR_LIMIT + 0.01),
            lambda: GlowEffect(spread=-0.01),
            lambda: GlowEffect(spread=SHADOW_SPREAD_LIMIT + 0.01),
        )
        for constructor in invalid_values:
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()

    def test_bad_glow_payload_drops_only_that_entry(self):
        valid = GlowEffect(glow_type='inner', size=0.4, spread=0.2)
        payload = TextEffectStack(effects=(
            StrokeEffect(width=0.2), valid,
        )).to_serializable_dict()
        payload['effects'].insert(1, {
            'effect_type': 'glow',
            'size': SHADOW_BLUR_LIMIT + 0.01,
        })
        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            loaded = coerce_text_effect_stack(payload)

        self.assertEqual(loaded.effects, (StrokeEffect(width=0.2), valid))
        warning.assert_called_once()

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
        transparent_gradient = LinearGradientPaint(stops=(
            GradientStop(0.0, (255, 0, 0), 0.0),
            GradientStop(1.0, (0, 0, 255), 0.0),
        ))
        self.assertTrue(StrokeEffect(paint=transparent_gradient).is_neutral())
        self.assertTrue(ShadowEffect(paint=transparent_gradient).is_neutral())

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
            lambda: StrokeEffect(blend_mode='overlay'),
            lambda: StrokeEffect(width=float('inf')),
            lambda: StrokeEffect(paint={'paint_type': 'solid'}),
            lambda: ShadowEffect(enabled=1),
            lambda: ShadowEffect(opacity=1.1),
            lambda: ShadowEffect(blend_mode='overlay'),
            lambda: ShadowEffect(shadow_type='outer'),
            lambda: ShadowEffect(paint={'paint_type': 'solid'}),
            lambda: ShadowEffect(angle=float('inf')),
            lambda: ShadowEffect(distance=-0.1),
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
                    'paint': {'paint_type': 'linear_gradient', 'stops': [
                        {'position': 0.0, 'color': [0, 0, 0], 'opacity': 1.0},
                    ]},
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
                StrokeEffect(
                    width=0.2,
                    paint=SolidPaint((1, 2, 3)),
                    position='center',
                ),
                StrokeEffect(
                    width=0.4,
                    paint=SolidPaint((4, 5, 6)),
                    position='outside',
                ),
            ),
        )
        self.assertEqual(warning.call_count, 6)

    def test_gradient_payload_round_trip_and_bad_entry_isolation(self):
        paint = LinearGradientPaint(
            stops=(
                GradientStop(0.0, (1, 2, 3), 0.25),
                GradientStop(0.4, (4, 5, 6), 0.5),
                GradientStop(1.0, (7, 8, 9), 1.0),
            ),
            angle=405,
            scale=1.75,
        )
        gradient = StrokeEffect(width=0.3, paint=paint)
        payload = TextEffectStack(effects=(gradient,)).to_serializable_dict()

        self.assertEqual(payload['effects'][0]['paint'], {
            'paint_type': 'linear_gradient',
            'stops': [
                {'position': 0.0, 'color': [1, 2, 3], 'opacity': 0.25},
                {'position': 0.4, 'color': [4, 5, 6], 'opacity': 0.5},
                {'position': 1.0, 'color': [7, 8, 9], 'opacity': 1.0},
            ],
            'angle': 45.0,
            'scale': 1.75,
        })
        self.assertEqual(coerce_text_effect_stack(payload)[0], gradient)

        malformed = {'effects': [
            payload['effects'][0],
            {
                'effect_type': 'stroke',
                'paint': {
                    'paint_type': 'linear_gradient',
                    'stops': [
                        {'position': 0.8, 'color': [0, 0, 0], 'opacity': 1.0},
                        {'position': 0.2, 'color': [255, 255, 255], 'opacity': 1.0},
                    ],
                },
            },
            {
                'effect_type': 'stroke',
                'paint': {'paint_type': 'solid', 'color': [9, 8, 7]},
            },
        ]}
        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            loaded = coerce_text_effect_stack(malformed)
        self.assertEqual(loaded.effects, (
            gradient,
            StrokeEffect(
                paint=SolidPaint((9, 8, 7)),
                position='center',
            ),
        ))
        warning.assert_called_once()
        self.assertEqual(effect_paint_fallback_color(paint), (1, 2, 3))
        self.assertEqual(
            effect_paint_fallback_color(SolidPaint((8, 7, 6))),
            (8, 7, 6),
        )

    def test_payload_keeps_mixed_order_and_isolates_duplicate_hollow(self):
        payload = {'effects': [
            {
                'effect_type': 'shadow',
                'shadow_type': 'drop',
                'angle': 330.0,
                'distance': 0.2,
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
            angle=-90.0,
            distance=SHADOW_DISTANCE_LIMIT,
            blur=SHADOW_BLUR_LIMIT,
            spread=SHADOW_SPREAD_LIMIT,
        )
        self.assertEqual(boundary.angle, 270.0)
        self.assertEqual(boundary.distance, 10.0)
        with self.assertRaises(ValueError):
            ShadowEffect(distance=SHADOW_DISTANCE_LIMIT + 0.01)
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
                    'offset': [0.2, 0.1],
                },
                {
                    'effect_type': 'shadow',
                    'distance': SHADOW_DISTANCE_LIMIT + 0.01,
                },
                {'effect_type': 'stroke', 'width': 0.2},
            ]})
        self.assertEqual(
            stack.effects,
            (StrokeEffect(width=0.2, position='center'),),
        )
        self.assertEqual(warning.call_count, 2)

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
            TextEffectStack(effects=(StrokeEffect(
                width=0.2,
                position='center',
            ),)),
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
        paint = LinearGradientPaint(stops=(
            GradientStop(0.0, (9, 8, 7), 0.25),
            GradientStop(1.0, (1, 2, 3), 1.0),
        ), angle=35.0, scale=1.4)
        stack = TextEffectStack(effects=(
            ShadowEffect(
                enabled=False,
                opacity=0.4,
                shadow_type='long',
                paint=paint,
                angle=120.0,
                distance=0.4,
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
            'paint': paint.to_serializable_dict(),
            'angle': 120.0,
            'distance': 0.4,
            'blur': 0.3,
            'spread': 0.1,
        })
        self.assertEqual(payload['effects'][1], {
            'effect_type': 'hollow',
            'enabled': True,
        })
        self.assertEqual(coerce_text_effect_stack(payload), stack)

        legacy = {'effects': [{
            'effect_type': 'shadow',
            'shadow_type': 'inner',
            'color': [9, 8, 7],
        }]}
        self.assertEqual(
            coerce_text_effect_stack(legacy).effects,
            (ShadowEffect(
                shadow_type='inner', paint=SolidPaint((9, 8, 7))
            ),),
        )
        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            malformed = coerce_text_effect_stack({'effects': [{
                'effect_type': 'shadow', 'color': None,
            }]})
        self.assertEqual(malformed.effects, ())
        warning.assert_called_once()


if __name__ == '__main__':
    unittest.main()
