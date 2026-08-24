import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ballontranslator.utils import shared
from ballontranslator.utils.config import (
    ProgramConfig,
    json_dump_program_config,
    load_textstyle_from,
    pcfg,
    text_styles,
)
from ballontranslator.utils.fontformat import (
    FontFormat,
    normalize_fontformat_effect_payload,
)
from ballontranslator.utils.io_utils import json_dump_nested_obj
from ballontranslator.utils.proj_imgtrans import ProjImgTrans, TextBlkEncoder
from ballontranslator.utils.text_effects import (
    GlowEffect,
    GradientOverlayEffect,
    GradientStop,
    HollowEffect,
    LinearGradientPaint,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
    primary_stroke,
)
from ballontranslator.utils.textblock import (
    TextBlock,
    normalize_textblock_effect_payload,
)


class TextEffectPersistenceTest(unittest.TestCase):
    def test_passive_normalization_does_not_share_bridge_lists(self):
        first_payload, _ = normalize_fontformat_effect_payload({})
        second_payload, _ = normalize_fontformat_effect_payload({})
        first = FontFormat(**first_payload)
        second = FontFormat(**second_payload)

        first.shadow_color[0] = 255
        first.shadow_offset[0] = 2.0

        self.assertEqual(second.shadow_color, [0, 0, 0])
        self.assertEqual(second.shadow_offset, [0.0, 0.0])

        first_saved = first.to_serializable_dict()
        second_saved = second.to_serializable_dict()
        first_saved['shadow_color'][0] = 128
        self.assertEqual(second_saved['shadow_color'], [0, 0, 0])
        self.assertEqual(first.shadow_color, [255, 0, 0])

    def test_nested_legacy_migration_and_new_field_precedence(self):
        legacy = FontFormat(
            opacity=0.7,
            stroke_width=0.2,
            srgb=[1, 2, 3],
        )
        stroke = primary_stroke(legacy.text_effects)
        self.assertEqual(legacy.text_effects.overall_opacity, 0.7)
        self.assertEqual(stroke.width, 0.2)
        self.assertEqual(stroke.paint, SolidPaint((1, 2, 3)))
        self.assertEqual(stroke.position, 'outside')

        explicit_empty = FontFormat(
            text_effects={'overall_opacity': 0.9, 'effects': []},
            opacity=0.1,
            stroke_width=0.8,
            srgb=[9, 8, 7],
        )
        self.assertEqual(
            explicit_empty.text_effects,
            TextEffectStack(overall_opacity=0.9),
        )
        self.assertEqual(explicit_empty.stroke_width, 0.0)

        overlay = GradientOverlayEffect(opacity=0.7)
        authoritative_overlay = FontFormat(
            text_effects=TextEffectStack(effects=(overlay,)),
            gradient_enabled=True,
            gradient_start_color=[255, 0, 0],
            gradient_end_color=[0, 0, 255],
        )
        self.assertEqual(
            authoritative_overlay.text_effects.effects, (overlay,)
        )

        with patch(
            'ballontranslator.utils.text_effects.LOGGER.warning'
        ) as warning:
            malformed = FontFormat(
                text_effects=['malformed'],
                opacity=0.2,
                stroke_width=0.6,
            )
        self.assertEqual(malformed.text_effects, TextEffectStack())
        warning.assert_called_once()

    def test_legacy_views_and_merge_use_only_the_canonical_stack(self):
        fontformat = FontFormat()
        fontformat.srgb = [10, 20, 30]
        fontformat.stroke_width = 0.25
        fontformat.opacity = 0.6

        self.assertNotIn('srgb', vars(fontformat))
        self.assertNotIn('stroke_width', vars(fontformat))
        self.assertNotIn('opacity', vars(fontformat))
        self.assertEqual(fontformat.srgb, [10, 20, 30])
        self.assertEqual(fontformat.stroke_width, 0.25)
        self.assertEqual(fontformat.opacity, 0.6)
        self.assertTrue({
            'srgb', 'stroke_width', 'opacity', 'text_effects'
        }.issubset(FontFormat.params()))

        target = FontFormat(text_effects=TextEffectStack(
            0.8,
            (
                StrokeEffect(width=0.4, paint=SolidPaint((4, 5, 6))),
                StrokeEffect(width=0.7, paint=SolidPaint((7, 8, 9))),
            ),
        ))
        changed = fontformat.merge(target, compare=True)
        self.assertTrue({
            'text_effects', 'srgb', 'stroke_width', 'opacity'
        }.issubset(changed))
        self.assertEqual(fontformat.text_effects, target.text_effects)

    def test_explicit_write_dual_writes_only_compatible_primary_stroke(self):
        fontformat = FontFormat(text_effects=TextEffectStack(
            0.75,
            (
                StrokeEffect(width=0.3, paint=SolidPaint((3, 4, 5))),
                StrokeEffect(width=0.8, paint=SolidPaint((8, 9, 10))),
            ),
        ))

        serialized = fontformat.to_serializable_dict()

        self.assertEqual(
            serialized['text_effects'],
            fontformat.text_effects.to_serializable_dict(),
        )
        self.assertEqual(serialized['opacity'], 0.75)
        self.assertEqual(serialized['stroke_width'], 0.3)
        self.assertEqual(serialized['srgb'], [3, 4, 5])
        self.assertEqual(serialized['shadow_radius'], 0.0)
        self.assertEqual(serialized['shadow_strength'], 1.0)
        self.assertEqual(serialized['shadow_color'], [0, 0, 0])
        self.assertEqual(serialized['shadow_offset'], [0.0, 0.0])
        self.assertFalse(serialized['gradient_enabled'])

        disabled_primary = FontFormat(text_effects=TextEffectStack(
            effects=(
                StrokeEffect(enabled=False, width=0.4),
                StrokeEffect(width=0.8, paint=SolidPaint((8, 9, 10))),
            )
        )).to_serializable_dict()
        self.assertEqual(disabled_primary['stroke_width'], 0.0)
        self.assertEqual(disabled_primary['srgb'], [0, 0, 0])

        gradient = LinearGradientPaint(stops=(
            GradientStop(0.0, (12, 34, 56), 0.4),
            GradientStop(1.0, (210, 220, 230), 1.0),
        ))
        gradient_write = FontFormat(text_effects=TextEffectStack(effects=(
            StrokeEffect(width=0.3, paint=gradient),
        ))).to_serializable_dict()
        self.assertEqual(gradient_write['srgb'], [12, 34, 56])
        self.assertEqual(
            gradient_write['text_effects']['effects'][0]['paint'][
                'paint_type'
            ],
            'linear_gradient',
        )
        transparent_write = FontFormat(text_effects=TextEffectStack(effects=(
            StrokeEffect(width=0.3, paint=LinearGradientPaint(stops=(
                GradientStop(0.0, (12, 34, 56), 0.0),
                GradientStop(1.0, (210, 220, 230), 0.0),
            ))),
        ))).to_serializable_dict()
        self.assertEqual(transparent_write['stroke_width'], 0.0)
        self.assertEqual(transparent_write['srgb'], [0, 0, 0])

    def test_project_flat_migration_orders_fields_and_aggregates_notices(self):
        legacy = {
            'opacity': 0.6,
            'default_stroke_width': 0.3,
            'bg_colors': [4, 5, 6],
            'shadow_radius': 0.2,
        }
        authoritative = {
            'text_effects': {'overall_opacity': 0.9, 'effects': []},
            'opacity': 0.1,
            'default_stroke_width': 0.8,
            'bg_colors': [9, 8, 7],
            'gradient_enabled': True,
        }
        project = ProjImgTrans()
        project.directory = '/tmp'

        with patch(
            'ballontranslator.utils.proj_imgtrans.find_all_imgs',
            return_value=[],
        ), patch(
            'ballontranslator.utils.fontformat.LOGGER.warning'
        ) as warning:
            project.load_from_dict({
                'pages': {'missing.png': [legacy, legacy, authoritative]},
                'image_info': {},
            })

        blocks = project.not_found_pages['missing.png']
        self.assertEqual(blocks[0].opacity, 0.6)
        self.assertEqual(blocks[0].stroke_width, 0.3)
        self.assertEqual(blocks[0].bg_colors, [4, 5, 6])
        self.assertEqual(
            primary_stroke(blocks[0].fontformat.text_effects).position,
            'outside',
        )
        self.assertEqual(
            blocks[1].fontformat.text_effects,
            blocks[0].fontformat.text_effects,
        )
        self.assertEqual(blocks[2].fontformat.text_effects, TextEffectStack(0.9))
        self.assertEqual(warning.call_count, 2)
        self.assertIn('Shadow', warning.call_args_list[0].args[0])
        self.assertIn('Gradient', warning.call_args_list[1].args[0])

    def test_style_and_global_config_loaders_migrate_once_per_owner(self):
        old_styles = list(text_styles)
        old_style_path = pcfg.text_styles_path
        overlay_stack = TextEffectStack(effects=(
            GradientOverlayEffect(
                opacity=0.65,
                paint=LinearGradientPaint(angle=35.0, scale=1.2),
            ),
        ))
        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, 'styles.json')
            config_path = os.path.join(directory, 'config.json')
            with open(style_path, 'w', encoding='utf8') as handle:
                json.dump([
                    {
                        '_style_name': 'legacy',
                        'opacity': 0.6,
                        'stroke_width': 0.2,
                        'srgb': [1, 2, 3],
                        'shadow_radius': 0.2,
                    },
                    {'gradient_enabled': True},
                    {
                        '_style_name': 'typed overlay',
                        'text_effects': overlay_stack.to_serializable_dict(),
                    },
                ], handle)
            with open(config_path, 'w', encoding='utf8') as handle:
                json.dump({
                    'global_fontformat': {
                        'opacity': 0.8,
                        'stroke_width': 0.4,
                        'srgb': [4, 5, 6],
                        'shadow_radius': 0.3,
                        'gradient_enabled': True,
                    }
                }, handle)

            try:
                with patch(
                    'ballontranslator.utils.fontformat.LOGGER.warning'
                ) as style_warning:
                    load_textstyle_from(style_path)
                self.assertEqual(style_warning.call_count, 2)
                self.assertEqual(len(text_styles), 3)
                self.assertEqual(text_styles[0].opacity, 0.6)
                self.assertEqual(text_styles[0].stroke_width, 0.2)
                self.assertEqual(text_styles[2].text_effects, overlay_stack)

                with patch(
                    'ballontranslator.utils.fontformat.LOGGER.warning'
                ) as config_warning:
                    config = ProgramConfig.load(config_path)
                self.assertEqual(config_warning.call_count, 2)
                self.assertEqual(config.global_fontformat.opacity, 0.8)
                self.assertEqual(config.global_fontformat.stroke_width, 0.4)

                saved_styles = json.loads(json_dump_nested_obj(text_styles))
                saved_config = json.loads(json_dump_program_config(config))
                self.assertIn('text_effects', saved_styles[0])
                self.assertIn(
                    'text_effects', saved_config['global_fontformat']
                )
                self.assertEqual(
                    saved_styles[2]['text_effects'],
                    overlay_stack.to_serializable_dict(),
                )

                config.global_fontformat.text_effects = overlay_stack
                saved_config = json.loads(json_dump_program_config(config))
                with open(config_path, 'w', encoding='utf8') as handle:
                    json.dump(saved_config, handle)
                reloaded = ProgramConfig.load(config_path)
                self.assertEqual(
                    reloaded.global_fontformat.text_effects, overlay_stack
                )
            finally:
                text_styles[:] = old_styles
                pcfg.text_styles_path = old_style_path

    def test_project_round_trip_preserves_full_stack_in_headless_mode(self):
        stack = TextEffectStack(
            0.65,
            (
                ShadowEffect(
                    shadow_type='drop',
                    color=(10, 20, 30),
                    offset=(0.3, -0.2),
                    blur=0.1,
                    spread=0.05,
                ),
                GlowEffect(
                    paint=LinearGradientPaint(angle=15.0),
                    size=0.16,
                    spread=0.04,
                ),
                StrokeEffect(
                    width=0.2,
                    paint=LinearGradientPaint(stops=(
                        GradientStop(0.0, (1, 2, 3), 0.25),
                        GradientStop(1.0, (4, 5, 6), 1.0),
                    ), angle=32, scale=1.4),
                    position='outside',
                ),
                StrokeEffect(
                    width=0.7,
                    paint=SolidPaint((7, 8, 9)),
                    position='inside',
                ),
                HollowEffect(enabled=False),
                GradientOverlayEffect(
                    opacity=0.8,
                    paint=LinearGradientPaint(stops=(
                        GradientStop(0.0, (90, 80, 70), 0.2),
                        GradientStop(1.0, (10, 20, 30), 1.0),
                    ), angle=75.0, scale=0.8),
                ),
                GlowEffect(
                    glow_type='inner',
                    paint=SolidPaint((240, 230, 120)),
                    size=0.12,
                    spread=0.03,
                ),
                ShadowEffect(shadow_type='inner', opacity=0.6),
            ),
        )
        block = TextBlock()
        block.fontformat.text_effects = stack

        with patch.object(shared, 'HEADLESS', True):
            payload = json.loads(json.dumps(block, cls=TextBlkEncoder))
            normalized, _ = normalize_textblock_effect_payload(payload)
            restored = TextBlock(**normalized)

        self.assertEqual(restored.fontformat.text_effects, stack)

    def test_typed_shadow_and_hollow_round_trip_in_preset_and_global_config(self):
        old_styles = list(text_styles)
        old_style_path = pcfg.text_styles_path
        stack = TextEffectStack(0.7, (
            ShadowEffect(shadow_type='long', offset=(0.4, 0.2)),
            GlowEffect(size=0.3, spread=0.1),
            StrokeEffect(paint=LinearGradientPaint()),
            HollowEffect(),
        ))
        payload = FontFormat(text_effects=stack).to_serializable_dict()
        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, 'styles.json')
            config_path = os.path.join(directory, 'config.json')
            with open(style_path, 'w', encoding='utf8') as handle:
                json.dump([dict(payload, _style_name='typed')], handle)
            with open(config_path, 'w', encoding='utf8') as handle:
                json.dump({'global_fontformat': payload}, handle)
            try:
                load_textstyle_from(style_path)
                config = ProgramConfig.load(config_path)
                self.assertEqual(text_styles[0].text_effects, stack)
                self.assertEqual(config.global_fontformat.text_effects, stack)
            finally:
                text_styles[:] = old_styles
                pcfg.text_styles_path = old_style_path

    def test_ocr_stroke_detection_updates_only_primary_width(self):
        second = StrokeEffect(width=0.8, paint=SolidPaint((8, 9, 10)))
        gradient = LinearGradientPaint(stops=(
            GradientStop(0.0, (250, 250, 250), 1.0),
            GradientStop(1.0, (10, 20, 30), 0.0),
        ))
        block = TextBlock()
        block.fontformat.text_effects = TextEffectStack(effects=(
            StrokeEffect(
                width=0.0, paint=gradient, position='outside'
            ),
            second,
        ))
        block.fontformat.frgb = [0, 0, 0]

        block.recalulate_stroke_width(stroke_width=0.35)

        self.assertEqual(
            primary_stroke(block.fontformat.text_effects).width,
            0.35,
        )
        self.assertIs(primary_stroke(block.fontformat.text_effects).paint, gradient)
        self.assertEqual(
            primary_stroke(block.fontformat.text_effects).position,
            'outside',
        )
        self.assertIs(block.fontformat.text_effects.effects[1], second)

        block.fontformat.srgb = [7, 6, 5]
        self.assertEqual(
            primary_stroke(block.fontformat.text_effects).paint,
            SolidPaint((7, 6, 5)),
        )

    def test_ocr_color_total_rounds_once_without_persisted_scratch(self):
        block = TextBlock(
            lines=[
                [[0, line], [10, line], [10, line + 1], [0, line + 1]]
                for line in range(10)
            ],
            bg_colors=np.zeros(3, dtype=np.float32),
        )

        samples = [np.array([1, 2, 3]) for _ in range(10)]
        color_total = sum(samples, np.zeros(3, dtype=np.float32))
        block.update_font_colors(color_total, color_total)
        stroke = primary_stroke(block.fontformat.text_effects)
        payload = json.loads(json.dumps(block, cls=TextBlkEncoder))

        self.assertEqual(stroke.paint, SolidPaint((1, 2, 3)))
        self.assertFalse(any('accumul' in key for key in vars(block)))
        self.assertNotIn('accumul', json.dumps(payload))


if __name__ == '__main__':
    unittest.main()
