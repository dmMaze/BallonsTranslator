import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

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
from ballontranslator.utils.raster_assets import (
    RasterAssetRef,
    coerce_raster_asset_ref,
)
from ballontranslator.utils.rendered_image import RenderedImageLayer
from ballontranslator.utils.rgba import premultiply_rgba_in_place
from ballontranslator.utils.text_effects import (
    FilterEffect,
    GlowEffect,
    TextFillEffect,
    GradientStop,
    HollowEffect,
    LinearGradientPaint,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
    TexturePaint,
    primary_stroke,
)
from ballontranslator.utils.textblock import (
    TextBlock,
    normalize_textblock_effect_payload,
)


class TextEffectPersistenceTest(unittest.TestCase):
    def test_rendered_image_layer_round_trip_and_permissive_failure(self):
        asset = RasterAssetRef(
            'assets/' + 'c' * 64 + '.png', 'rendered.png'
        )
        layer = RenderedImageLayer(asset, enabled=False, mode='overlay')
        block = TextBlock(rendered_image=layer)
        block.translation = 'preserved'

        payload = json.loads(json.dumps(block, cls=TextBlkEncoder))
        restored = TextBlock(**payload)
        self.assertEqual(restored.rendered_image, layer)

        payload['rendered_image']['version'] = 99
        malformed = TextBlock(**payload)
        self.assertIsNone(malformed.rendered_image)
        self.assertEqual(malformed.translation, 'preserved')
        self.assertIsNone(TextBlock().rendered_image)

    def test_raster_asset_payload_coercion_is_shared_and_strict(self):
        payload = {'path': 'assets/' + 'd' * 64 + '.webp'}
        self.assertEqual(coerce_raster_asset_ref(payload).digest, 'd' * 64)
        with self.assertRaises(ValueError):
            coerce_raster_asset_ref({**payload, 'future': True})

    def test_raster_import_hashes_and_decodes_one_stable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'changing.png')
            red = np.full((2, 3, 4), (230, 20, 30, 255), dtype=np.uint8)
            blue = np.full((2, 3, 4), (20, 30, 230, 255), dtype=np.uint8)
            Image.fromarray(red).save(source_path)
            with open(source_path, 'rb') as source:
                original_bytes = source.read()
            real_image_open = Image.open

            def mutate_source_after_snapshot(path, *args, **kwargs):
                if os.path.basename(path).startswith('.import-'):
                    Image.fromarray(blue).save(source_path)
                return real_image_open(path, *args, **kwargs)

            project = ProjImgTrans()
            project.directory = directory
            with patch(
                'ballontranslator.utils.proj_imgtrans.Image.open',
                side_effect=mutate_source_after_snapshot,
            ):
                asset = project.import_raster_asset(source_path)

            self.assertEqual(
                asset.digest, hashlib.sha256(original_bytes).hexdigest()
            )
            with open(project.resolve_raster_asset(asset), 'rb') as installed:
                self.assertEqual(installed.read(), original_bytes)
            np.testing.assert_array_equal(
                project.load_raster_asset(asset), red
            )

    def test_raster_import_rejects_resource_bombs_and_wide_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            source_path = os.path.join(directory, 'texture.png')
            Image.fromarray(np.zeros((3, 3, 4), dtype=np.uint8)).save(
                source_path
            )
            with patch(
                'ballontranslator.utils.proj_imgtrans.'
                'RASTER_ASSET_MAX_SOURCE_BYTES',
                1,
            ):
                with self.assertRaises(ValueError):
                    project.import_raster_asset(source_path)
            with patch(
                'ballontranslator.utils.proj_imgtrans.Image.open',
                side_effect=Image.DecompressionBombError('too large'),
            ):
                with self.assertRaises(ValueError):
                    project.import_raster_asset(source_path)

            wide_path = os.path.join(directory, 'wide.png')
            Image.fromarray(
                np.array([[0, 65535]], dtype=np.uint16)
            ).save(wide_path)
            with self.assertRaises(ValueError):
                project.import_raster_asset(wide_path)

    def test_raster_decode_accepts_full_page_texture_dimensions(self):
        with tempfile.NamedTemporaryFile() as source:
            height, width = 7016, 4960
            rgba = np.broadcast_to(
                np.zeros((1, 1, 4), dtype=np.uint8),
                (height, width, 4),
            )
            image = MagicMock()
            image.__enter__.return_value = image
            image.format = 'PNG'
            image.size = (width, height)
            image.mode = 'RGB'

            with patch(
                'ballontranslator.utils.proj_imgtrans.Image.open',
                return_value=image,
            ), patch(
                'ballontranslator.utils.proj_imgtrans.np.array',
                return_value=rgba,
            ), patch(
                'ballontranslator.utils.proj_imgtrans.np.ascontiguousarray',
                return_value=rgba,
            ):
                extension, decoded = ProjImgTrans._decode_raster_asset_snapshot(
                    source.name
                )

            self.assertEqual(extension, '.png')
            self.assertEqual(decoded.shape, (height, width, 4))

    def test_raster_decode_cache_is_byte_bounded(self):
        project = ProjImgTrans()
        first = RasterAssetRef('assets/' + 'a' * 64 + '.png')
        second = RasterAssetRef('assets/' + 'b' * 64 + '.png')
        pixels = np.zeros((2, 2, 4), dtype=np.uint8)

        with patch(
            'ballontranslator.utils.proj_imgtrans.'
            'RASTER_ASSET_DECODE_CACHE_MAX_BYTES',
            pixels.nbytes,
        ):
            project._cache_raster_asset(first, pixels, (1, 1, 1, 1))
            project._cache_raster_asset(second, pixels.copy(), (2, 2, 2, 2))

        self.assertEqual(tuple(project._raster_asset_cache), (second.path,))

    def test_changed_raster_bytes_are_reverified_before_interactive_decode(self):
        for replacement in ('invalid-bytes', 'valid-wrong-digest'):
            with (
                self.subTest(replacement=replacement),
                tempfile.TemporaryDirectory() as directory,
            ):
                source_path = os.path.join(directory, 'source.png')
                red = np.full(
                    (2, 3, 4), (230, 20, 30, 255), dtype=np.uint8
                )
                Image.fromarray(red, 'RGBA').save(source_path)
                project = ProjImgTrans()
                project.directory = directory
                asset = project.import_raster_asset(source_path)
                np.testing.assert_array_equal(
                    project.load_raster_asset(asset), red
                )
                installed = project.resolve_raster_asset(asset)
                if replacement == 'invalid-bytes':
                    with open(installed, 'wb') as handle:
                        handle.write(b'not an image')
                else:
                    project._raster_asset_cache.clear()
                    blue = np.full(
                        (2, 3, 4), (20, 30, 230, 255), dtype=np.uint8
                    )
                    Image.fromarray(blue, 'RGBA').save(installed)

                with patch(
                    'ballontranslator.utils.proj_imgtrans.LOGGER.warning'
                ) as warning:
                    self.assertIsNone(project.load_raster_asset(asset))
                warning.assert_called_once()
                with self.assertRaises(OSError):
                    project.load_raster_asset(asset, strict=True)

    def test_non_strict_raster_stat_race_bypasses(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'source.png')
            Image.fromarray(
                np.zeros((2, 3, 4), dtype=np.uint8), 'RGBA'
            ).save(source_path)
            project = ProjImgTrans()
            project.directory = directory
            asset = project.import_raster_asset(source_path)

            with patch.object(
                project,
                '_raster_asset_signature',
                side_effect=FileNotFoundError('changed during resolve'),
            ), patch(
                'ballontranslator.utils.proj_imgtrans.LOGGER.warning'
            ) as warning:
                self.assertIsNone(project.load_raster_asset(asset))
            warning.assert_called_once()

    def test_strict_raster_load_rejects_replacement_after_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'source.png')
            replacement_path = os.path.join(directory, 'replacement.png')
            red = np.full((2, 3, 4), (230, 20, 30, 255), dtype=np.uint8)
            blue = np.full((2, 3, 4), (20, 30, 230, 255), dtype=np.uint8)
            Image.fromarray(red, 'RGBA').save(source_path)
            Image.fromarray(blue, 'RGBA').save(replacement_path)
            project = ProjImgTrans()
            project.directory = directory
            asset = project.import_raster_asset(source_path)
            installed = project.resolve_raster_asset(asset)
            original_hash = project._hash_raster_asset_file

            def replace_after_hash(path):
                digest = original_hash(path)
                os.replace(replacement_path, path)
                return digest

            with patch.object(
                project,
                '_hash_raster_asset_file',
                side_effect=replace_after_hash,
            ) as raster_hash:
                with self.assertRaisesRegex(OSError, 'changed while'):
                    project.load_raster_asset(asset, strict=True)
            raster_hash.assert_called_once_with(installed)
            self.assertNotIn(asset.path, project._raster_asset_cache)

    def test_project_cache_reuses_and_invalidates_premultiplied_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'source.png')
            straight = np.full(
                (2, 3, 4), (200, 100, 50, 128), dtype=np.uint8
            )
            Image.fromarray(straight, 'RGBA').save(source_path)
            project = ProjImgTrans()
            project.directory = directory
            asset = project.import_raster_asset(source_path)

            with patch(
                'ballontranslator.utils.proj_imgtrans.'
                'premultiply_rgba_in_place',
                wraps=premultiply_rgba_in_place,
            ) as premultiply, patch.object(
                project,
                '_hash_raster_asset_file',
                side_effect=AssertionError('warm interactive hit hashed'),
            ):
                first = project.load_raster_asset(asset, premultiplied=True)
                second = project.load_raster_asset(asset, premultiplied=True)
                self.assertIs(first, second)
                self.assertFalse(first.flags.writeable)
                self.assertEqual(first[0, 0].tolist(), [100, 50, 25, 128])
                np.testing.assert_array_equal(
                    project.load_raster_asset(asset), straight
                )
                self.assertEqual(premultiply.call_count, 1)

            with patch(
                'ballontranslator.utils.proj_imgtrans.'
                'premultiply_rgba_in_place',
                wraps=premultiply_rgba_in_place,
            ) as premultiply:
                self.assertEqual(project.import_raster_asset(source_path), asset)
                project.load_raster_asset(asset, premultiplied=True)
                self.assertEqual(premultiply.call_count, 1)

    def test_project_imports_and_resolves_content_addressed_raster_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'paper texture.png')
            Image.fromarray(np.array([
                [[255, 0, 0, 255], [0, 255, 0, 128]],
            ], dtype=np.uint8), 'RGBA').save(source_path)
            project = ProjImgTrans()
            project.directory = directory

            first = project.import_raster_asset(source_path)
            second = project.import_raster_asset(source_path)
            resolved = project.resolve_raster_asset(first)

            self.assertEqual(first, second)
            self.assertIsInstance(first, RasterAssetRef)
            self.assertEqual(first.display_name, 'paper texture.png')
            self.assertTrue(first.path.startswith('assets/'))
            self.assertTrue(os.path.isfile(resolved))
            self.assertEqual(
                TextFillEffect(paint=TexturePaint(first)).to_serializable_dict()[
                    'paint'
                ]['asset']['path'],
                first.path,
            )

            with open(resolved, 'wb') as installed:
                installed.write(b'corrupt')
            with self.assertRaises(OSError):
                project.import_raster_asset(source_path)

            os.unlink(resolved)
            with patch(
                'ballontranslator.utils.proj_imgtrans.LOGGER.warning'
            ) as warning:
                self.assertIsNone(project.resolve_raster_asset(first))
            warning.assert_called_once()
            with self.assertRaises(FileNotFoundError):
                project.resolve_raster_asset(first, strict=True)

            jxl_path = os.path.join(directory, 'paper texture.jxl')
            Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8)).save(jxl_path)
            self.assertTrue(
                project.import_raster_asset(jxl_path).path.endswith('.jxl')
            )

    def test_project_raster_assets_cannot_escape_through_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory, \
                tempfile.TemporaryDirectory() as outside:
            source_path = os.path.join(directory, 'texture.png')
            Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8)).save(
                source_path
            )
            try:
                os.symlink(outside, os.path.join(directory, 'assets'))
            except OSError as error:
                self.skipTest(f'symlink creation is unavailable: {error}')
            project = ProjImgTrans()
            project.directory = directory

            with self.assertRaises(OSError):
                project.import_raster_asset(source_path)
            asset = RasterAssetRef('assets/' + 'a' * 64 + '.png')
            self.assertIsNone(project.resolve_raster_asset(asset))
            with self.assertRaises(FileNotFoundError):
                project.resolve_raster_asset(asset, strict=True)

    def test_application_styles_discard_only_project_texture_fills(self):
        asset = RasterAssetRef('assets/' + 'a' * 64 + '.png', 'paper.png')
        stack = TextEffectStack(effects=(
            StrokeEffect(width=0.25),
            TextFillEffect(paint=TexturePaint(asset)),
            GlowEffect(size=0.2),
        ))
        payload = FontFormat(text_effects=stack).to_serializable_dict()
        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, 'styles.json')
            config_path = os.path.join(directory, 'config.json')
            with open(style_path, 'w', encoding='utf8') as handle:
                json.dump([payload], handle)
            with open(config_path, 'w', encoding='utf8') as handle:
                json.dump({'global_fontformat': payload}, handle)
            old_styles = list(text_styles)
            old_style_path = pcfg.text_styles_path
            try:
                load_textstyle_from(style_path)
                config = ProgramConfig.load(config_path)
                expected = (StrokeEffect(width=0.25), GlowEffect(size=0.2))
                self.assertEqual(text_styles[0].text_effects.effects, expected)
                self.assertEqual(
                    config.global_fontformat.text_effects.effects, expected
                )
            finally:
                text_styles[:] = old_styles
                pcfg.text_styles_path = old_style_path

    def test_project_payload_passively_preserves_texture_fill_reference(self):
        asset = RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'missing-paper.png'
        )
        stack = TextEffectStack(effects=(
            StrokeEffect(width=0.25),
            TextFillEffect(
                paint=TexturePaint(asset, mapping='tile', scale=1.5)
            ),
        ))
        block = TextBlock()
        block.fontformat.text_effects = stack

        payload = json.loads(json.dumps(block, cls=TextBlkEncoder))
        normalized, _notices = normalize_textblock_effect_payload(payload)
        restored = TextBlock(**normalized)

        self.assertEqual(restored.fontformat.text_effects, stack)

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

        text_fill = TextFillEffect()
        authoritative_text_fill = FontFormat(
            text_effects=TextEffectStack(effects=(text_fill,)),
            gradient_enabled=True,
            gradient_start_color=[255, 0, 0],
            gradient_end_color=[0, 0, 255],
        )
        self.assertEqual(
            authoritative_text_fill.text_effects.effects, (text_fill,)
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
        text_fill_stack = TextEffectStack(effects=(
            TextFillEffect(
                paint=LinearGradientPaint(angle=35.0, scale=1.2),
            ),
        ))
        legacy_gradient_payload = text_fill_stack.to_serializable_dict()
        legacy_gradient_payload['effects'][0][
            'effect_type'
        ] = 'gradient_overlay'
        legacy_gradient_payload['effects'][0]['opacity'] = 0.65
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
                        '_style_name': 'typed text_fill',
                        'text_effects': legacy_gradient_payload,
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
                self.assertEqual(text_styles[2].text_effects, text_fill_stack)

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
                    text_fill_stack.to_serializable_dict(),
                )

                config.global_fontformat.text_effects = text_fill_stack
                saved_config = json.loads(json_dump_program_config(config))
                with open(config_path, 'w', encoding='utf8') as handle:
                    json.dump(saved_config, handle)
                reloaded = ProgramConfig.load(config_path)
                self.assertEqual(
                    reloaded.global_fontformat.text_effects, text_fill_stack
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
                    paint=LinearGradientPaint(stops=(
                        GradientStop(0.0, (10, 20, 30), 0.4),
                        GradientStop(1.0, (60, 50, 40), 1.0),
                    ), angle=25.0),
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
                TextFillEffect(
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
            ShadowEffect(
                shadow_type='long',
                offset=(0.4, 0.2),
                paint=LinearGradientPaint(angle=55.0),
            ),
            GlowEffect(size=0.3, spread=0.1),
            StrokeEffect(paint=LinearGradientPaint()),
            HollowEffect(),
            FilterEffect(
                'future:grain',
                schema_version=8,
                params={'future': 'preserved', 'amount': 0.4},
            ),
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
