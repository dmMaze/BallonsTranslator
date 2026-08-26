from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ballontranslator.ui.text_engine.effects.filters import (
    FilterContext,
    FilterMetadataError,
    FilterRegistry,
    FilterUnavailableError,
)
from ballontranslator.utils.text_effects import FilterEffect


def _plugin_source(
    filter_id='custom:demo',
    *,
    body='def apply(rgba, params, context):\n    return rgba\n\n'
    'def tile_halo(params, render_scale):\n    return 0\n',
):
    return f'''FILTER_META = {{
    "filter_id": {filter_id!r},
    "name": "Demo",
    "schema_version": 1,
    "params": ({{
        "key": "amount", "label": "Amount", "kind": "float",
        "default": 0.5, "minimum": 0.0, "maximum": 1.0,
    }},),
}}

{body}'''


class TextFilterRegistryTest(unittest.TestCase):
    def _registry(self, builtin, custom):
        return FilterRegistry(Path(builtin), Path(custom))

    def test_discovery_executes_nothing_and_imports_only_active_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            marker = root / 'executed'
            source = _plugin_source(body=(
                f'open({str(marker)!r}, "w").write("yes")\n\n'
                'def apply(rgba, params, context):\n    return rgba\n\n'
                'def tile_halo(params, render_scale):\n    return 0\n'
            ))
            (custom / 'filter_demo.py').write_text(source, encoding='utf-8')
            registry = self._registry(builtin, custom)

            self.assertEqual([spec.filter_id for spec in registry.specs], [
                'custom:demo'
            ])
            self.assertFalse(marker.exists())
            runtime = registry.resolve(FilterEffect('custom:demo'))
            self.assertTrue(marker.exists())
            image = np.zeros((2, 3, 4), dtype=np.uint8)
            self.assertIs(
                runtime.apply(
                    image, runtime.params, FilterContext(1.0, 0, 0)
                ),
                image,
            )

    def test_builtin_precedence_and_custom_malformed_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            (builtin / 'filter_demo.py').write_text(
                _plugin_source('builtin:demo'), encoding='utf-8'
            )
            (custom / 'filter_demo.py').write_text(
                _plugin_source('builtin:demo'), encoding='utf-8'
            )
            (custom / 'filter_broken.py').write_text(
                'FILTER_META = object()\n', encoding='utf-8'
            )
            registry = self._registry(builtin, custom)

            with patch(
                'ballontranslator.ui.text_engine.effects.filters.registry.'
                'LOGGER.warning'
            ) as warning:
                specs = registry.specs
            self.assertEqual(len(specs), 1)
            self.assertTrue(specs[0].builtin)
            self.assertGreaterEqual(warning.call_count, 2)

    def test_builtin_metadata_error_is_loud(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            (builtin / 'filter_bad.py').write_text(
                'FILTER_META = object()\n', encoding='utf-8'
            )
            with self.assertRaises(FilterMetadataError):
                self._registry(builtin, custom).specs

    def test_numeric_metadata_failures_are_isolated_and_bounded(self):
        huge_bound = '9' * 4001
        replacements = {
            'huge_bound': ('"maximum": 1.0', f'"maximum": {huge_bound}'),
            'inf_factor': (
                '"maximum": 1.0,',
                '"maximum": 1.0, "display_factor": 1e999,',
            ),
            'huge_decimals': (
                '"maximum": 1.0,',
                '"maximum": 1.0, "decimals": 999999999999,',
            ),
            'bool_min': ('"minimum": 0.0', '"minimum": True'),
            'string_step': (
                '"maximum": 1.0,',
                '"maximum": 1.0, "step": "fast",',
            ),
            'reversed': (
                '"minimum": 0.0, "maximum": 1.0',
                '"minimum": 2.0, "maximum": 1.0',
            ),
            'bad_suffix': (
                '"maximum": 1.0,',
                '"maximum": 1.0, "suffix": 3,',
            ),
            'bad_expands_alpha': (
                '"schema_version": 1,',
                '"schema_version": 1, "expands_alpha": "yes",',
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            (custom / 'filter_valid.py').write_text(
                _plugin_source('custom:valid'), encoding='utf-8'
            )
            for name, (old, new) in replacements.items():
                source = _plugin_source(f'custom:{name}').replace(old, new)
                (custom / f'filter_{name}.py').write_text(
                    source, encoding='utf-8'
                )

            with patch(
                'ballontranslator.ui.text_engine.effects.filters.registry.'
                'LOGGER.warning'
            ) as warning:
                specs = self._registry(builtin, custom).specs
            self.assertEqual(
                [spec.filter_id for spec in specs], ['custom:valid']
            )
            self.assertEqual(warning.call_count, len(replacements))

            (builtin / 'filter_bad.py').write_text(
                _plugin_source('builtin:bad').replace(
                    '"maximum": 1.0', f'"maximum": {huge_bound}'
                ),
                encoding='utf-8',
            )
            with self.assertRaises(FilterMetadataError):
                self._registry(builtin, custom).specs

    def test_custom_symlink_and_path_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            outside = root / 'filter_link.py'
            outside.write_text(_plugin_source('custom:link'), encoding='utf-8')
            (custom / 'filter_link.py').symlink_to(outside)
            (custom / 'filter_wrong.py').write_text(
                _plugin_source('custom:different'), encoding='utf-8'
            )

            with patch(
                'ballontranslator.ui.text_engine.effects.filters.registry.'
                'LOGGER.warning'
            ) as warning:
                self.assertEqual(self._registry(builtin, custom).specs, ())
            self.assertEqual(warning.call_count, 2)

    def test_lazy_runtime_failures_are_isolated_and_cached(self):
        cases = {
            'dependency': 'import definitely_missing_filter_dependency\n',
            'runtime_meta': 'FILTER_META["schema_version"] = 2\n',
        }
        for name, prefix in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                builtin = root / 'builtin'
                custom = root / 'custom'
                builtin.mkdir()
                custom.mkdir()
                path = custom / 'filter_demo.py'
                path.write_text(
                    _plugin_source(body=prefix + '\n' + (
                        'def apply(rgba, params, context):\n    return rgba\n\n'
                        'def tile_halo(params, render_scale):\n    return 0\n'
                    )),
                    encoding='utf-8',
                )
                registry = self._registry(builtin, custom)
                self.assertEqual(len(registry.specs), 1)
                with self.assertRaises(FilterUnavailableError):
                    registry.resolve(FilterEffect('custom:demo'))
                self.assertIn('custom:demo', registry._failures)
                self.assertNotIn(
                    '_ballontranslator_custom_text_filter_filter_demo',
                    sys.modules,
                )

    def test_restart_snapshot_rejects_changes_and_ignores_additions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            path = custom / 'filter_demo.py'
            path.write_text(_plugin_source(), encoding='utf-8')
            registry = self._registry(builtin, custom)
            self.assertEqual(len(registry.specs), 1)

            (custom / 'filter_later.py').write_text(
                _plugin_source('custom:later'), encoding='utf-8'
            )
            path.write_text(_plugin_source() + '\n# changed\n', encoding='utf-8')
            self.assertIsNone(registry.get_spec('custom:later'))
            with self.assertRaisesRegex(FilterUnavailableError, 'restart'):
                registry.resolve(FilterEffect('custom:demo'))

    def test_scan_to_import_symlink_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            path = custom / 'filter_demo.py'
            path.write_text(_plugin_source(), encoding='utf-8')
            replacement = root / 'filter_demo.py'
            replacement.write_text(_plugin_source(), encoding='utf-8')
            registry = self._registry(builtin, custom)
            spec = registry.get_spec('custom:demo')
            self.assertIs(registry._spec_by_id['custom:demo'], spec)

            path.unlink()
            path.symlink_to(replacement)
            with self.assertRaisesRegex(FilterUnavailableError, 'source path'):
                registry.resolve(FilterEffect('custom:demo'))

    def test_active_params_drop_unknowns_and_isolate_invalid_known_values(self):
        registry = FilterRegistry(custom_dir=Path('/path/that/does/not/exist'))
        spec = registry.get_spec('builtin:noise')
        original = FilterEffect('builtin:noise', params={
            'amount': 99.0,
            'mode': 'future',
            'seed': -3,
            'opaque_future': 'preserved',
        })

        active = spec.normalize_params(original.params_dict())

        self.assertEqual(active, spec.default_params())
        self.assertNotIn('opaque_future', active)
        self.assertIn(('opaque_future', 'preserved'), original.params)
        with patch(
            'ballontranslator.ui.text_engine.effects.filters.registry.'
            'LOGGER.warning'
        ) as warning:
            registry.resolve(original)
            registry.resolve(original)
        self.assertEqual(warning.call_count, 3)

    def test_runtime_overflow_defaults_without_mutating_opaque_params(self):
        registry = FilterRegistry(custom_dir=Path('/missing/custom/filters'))
        huge = int('9' * 4001)
        effect = FilterEffect('builtin:noise', params={
            'amount': huge, 'mode': 'monochrome', 'seed': 1,
        })

        with patch(
            'ballontranslator.ui.text_engine.effects.filters.registry.'
            'LOGGER.warning'
        ) as warning:
            first = registry.resolve(effect)
            second = registry.resolve(effect)

        self.assertEqual(first.params['amount'], 0.2)
        self.assertEqual(second.params['amount'], 0.2)
        self.assertEqual(effect.params_dict()['amount'], huge)
        self.assertEqual(warning.call_count, 1)

    def test_invalid_param_warnings_are_message_stable_and_capped(self):
        registry = FilterRegistry(custom_dir=Path('/missing/custom/filters'))
        with patch(
            'ballontranslator.ui.text_engine.effects.filters.registry.'
            'LOGGER.warning'
        ) as warning:
            registry.resolve(FilterEffect('builtin:noise', params={
                'amount': True, 'mode': 'monochrome', 'seed': 1,
            }))
            registry.resolve(FilterEffect('builtin:noise', params={
                'amount': 99.0, 'mode': 'monochrome', 'seed': 1,
            }))
        self.assertEqual(warning.call_count, 1)

        capped = FilterRegistry(custom_dir=Path('/missing/custom/filters'))
        capped._param_warnings.update(
            (f'custom:{index}', 1, 'amount') for index in range(64)
        )
        with patch(
            'ballontranslator.ui.text_engine.effects.filters.registry.'
            'LOGGER.warning'
        ) as warning:
            capped.resolve(FilterEffect('builtin:noise', params={
                'amount': 99.0, 'mode': 'monochrome', 'seed': 1,
            }))
        self.assertEqual(warning.call_count, 0)
        self.assertEqual(len(capped._param_warnings), 64)

    def test_older_active_schema_migrates_only_during_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            body = (
                'def migrate_params(from_version, params):\n'
                '    params["amount"] = params.get("old_amount", 0.5)\n'
                '    return params\n\n'
                'def apply(rgba, params, context):\n    return rgba\n\n'
                'def tile_halo(params, render_scale):\n    return 0\n'
            )
            source = _plugin_source(body=body).replace(
                '"schema_version": 1', '"schema_version": 2'
            )
            (custom / 'filter_demo.py').write_text(source, encoding='utf-8')
            registry = self._registry(builtin, custom)
            effect = FilterEffect(
                'custom:demo', schema_version=1,
                params={'old_amount': 0.7, 'opaque': 'kept'},
            )

            self.assertEqual(registry._modules, {})
            runtime = registry.resolve(effect)
            self.assertEqual(runtime.params, {'amount': 0.7})
            self.assertIn('custom:demo', registry._modules)
            self.assertEqual(effect.params_dict()['opaque'], 'kept')

    def test_newer_schema_is_rejected_before_lazy_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builtin = root / 'builtin'
            custom = root / 'custom'
            builtin.mkdir()
            custom.mkdir()
            marker = root / 'imported'
            (custom / 'filter_demo.py').write_text(
                _plugin_source(body=(
                    f'open({str(marker)!r}, "w").write("yes")\n\n'
                    'def apply(rgba, params, context):\n    return rgba\n\n'
                    'def tile_halo(params, render_scale):\n    return 0\n'
                )),
                encoding='utf-8',
            )
            registry = self._registry(builtin, custom)

            with self.assertRaisesRegex(FilterUnavailableError, 'incompatible'):
                registry.resolve(FilterEffect('custom:demo', schema_version=2))
            self.assertFalse(marker.exists())
            self.assertEqual(registry._modules, {})

    def test_builtin_algorithms_are_deterministic_and_obey_alpha_contract(self):
        registry = FilterRegistry(custom_dir=Path('/path/that/does/not/exist'))
        image = np.zeros((12, 15, 4), dtype=np.uint8)
        image[2:10, 3:12] = (100, 140, 180, 255)
        context = FilterContext(1.25, -7, 11)
        for spec in registry.specs:
            with self.subTest(filter_id=spec.filter_id):
                runtime = registry.resolve(FilterEffect(
                    spec.filter_id, params=spec.default_params()
                ))
                first = runtime.apply(image.copy(), runtime.params, context)
                second = runtime.apply(image.copy(), runtime.params, context)
                np.testing.assert_array_equal(first, second)
                if not spec.expands_alpha:
                    self.assertTrue(np.all(
                        first[:, :, 3][image[:, :, 3] == 0] == 0
                    ))

    def test_expanding_filter_metadata_and_defaults(self):
        registry = FilterRegistry(custom_dir=Path('/missing/custom/filters'))
        expected = {
            'builtin:gaussian_blur': (
                'Gaussian Blur', 40, {'radius': 2.0}, 3,
            ),
            'builtin:bloom': (
                'Bloom', 50,
                {'threshold': 0.6, 'radius': 6.0, 'intensity': 0.8},
                8,
            ),
            'builtin:glitch': (
                'Glitch', 60,
                {
                    'shift': 6.0, 'block_size': 8.0, 'activity': 0.25,
                    'rgb_split': 2.0, 'seed': 0,
                },
                11,
            ),
        }

        for filter_id, (name, order, defaults, halo) in expected.items():
            with self.subTest(filter_id=filter_id):
                spec = registry.get_spec(filter_id)
                self.assertIsNotNone(spec)
                self.assertEqual(spec.name, name)
                self.assertEqual(spec.order, order)
                self.assertTrue(spec.expands_alpha)
                self.assertEqual(spec.default_params(), defaults)
                runtime = registry.resolve(FilterEffect(filter_id))
                self.assertEqual(runtime.tile_halo(runtime.params, 1.25), halo)

    def test_expanding_filter_neutral_values_are_exact_noops(self):
        registry = FilterRegistry(custom_dir=Path('/missing/custom/filters'))
        image = np.arange(9 * 11 * 4, dtype=np.uint8).reshape(9, 11, 4)
        cases = (
            ('builtin:gaussian_blur', {'radius': 0.0}),
            ('builtin:bloom', {
                'threshold': 0.0, 'radius': 5.0, 'intensity': 0.0,
            }),
            ('builtin:glitch', {
                'shift': 8.0, 'block_size': 3.0, 'activity': 0.0,
                'rgb_split': 4.0, 'seed': 7,
            }),
            ('builtin:glitch', {
                'shift': 0.0, 'block_size': 3.0, 'activity': 1.0,
                'rgb_split': 0.0, 'seed': 7,
            }),
        )

        for filter_id, params in cases:
            with self.subTest(filter_id=filter_id):
                runtime = registry.resolve(FilterEffect(
                    filter_id, params=params
                ))
                result = runtime.apply(
                    image, runtime.params, FilterContext(2.0, -13, -17)
                )
                self.assertIs(result, image)
                np.testing.assert_array_equal(result, image)
                self.assertEqual(runtime.tile_halo(runtime.params, 2.0), 0)

    def test_gaussian_blur_expands_only_within_its_declared_halo(self):
        registry = FilterRegistry(custom_dir=Path('/missing/custom/filters'))
        image = np.zeros((31, 35, 4), dtype=np.uint8)
        image[15, 17] = (80, 150, 230, 255)
        runtime = registry.resolve(FilterEffect(
            'builtin:gaussian_blur', params={'radius': 2.0}
        ))

        result = runtime.apply(
            image, runtime.params, FilterContext(1.0, 0, 0)
        )

        expanded = (image[:, :, 3] == 0) & (result[:, :, 3] > 0)
        self.assertGreater(np.count_nonzero(expanded), 0)
        halo = runtime.tile_halo(runtime.params, 1.0)
        allowed = cv2.dilate(
            (image[:, :, 3] > 0).astype(np.uint8),
            np.ones((halo * 2 + 1, halo * 2 + 1), dtype=np.uint8),
        )
        self.assertFalse(np.any(expanded & (allowed == 0)))
        self.assertFalse(np.any(
            result[:, :, :3][result[:, :, 3] == 0] != 0
        ))

        tiny = np.zeros((9, 9, 4), dtype=np.uint8)
        tiny[4, 4] = (255, 100, 50, 1)
        quantized = runtime.apply(
            tiny, runtime.params, FilterContext(1.0, 0, 0)
        )
        self.assertEqual(np.count_nonzero(quantized[:, :, 3]), 0)
        self.assertEqual(np.count_nonzero(quantized[:, :, :3]), 0)

    def test_bloom_bright_pass_and_threshold_one(self):
        registry = FilterRegistry(custom_dir=Path('/missing/custom/filters'))

        def bloom(pixel, threshold, radius):
            image = np.zeros((31, 31, 4), dtype=np.uint8)
            image[15, 15] = pixel
            runtime = registry.resolve(FilterEffect(
                'builtin:bloom', params={
                    'threshold': threshold, 'radius': radius,
                    'intensity': 1.0,
                },
            ))
            return image, runtime.apply(
                image, runtime.params, FilterContext(1.0, 0, 0)
            )

        dark, dark_result = bloom((100, 100, 100, 255), 0.6, 0.0)
        np.testing.assert_array_equal(dark_result, dark)
        bright, bright_result = bloom((255, 240, 220, 128), 0.6, 0.0)
        self.assertGreater(
            int(bright_result[15, 15, 3]), int(bright[15, 15, 3])
        )

        white, white_result = bloom((255, 255, 255, 255), 1.0, 2.0)
        self.assertGreater(
            np.count_nonzero(white_result[:, :, 3]),
            np.count_nonzero(white[:, :, 3]),
        )
        subwhite, subwhite_result = bloom((254, 254, 254, 255), 1.0, 2.0)
        np.testing.assert_array_equal(subwhite_result, subwhite)
        self.assertFalse(np.any(
            white_result[:, :, :3][white_result[:, :, 3] == 0] != 0
        ))

    def test_glitch_is_seeded_crop_stable_and_can_split_visible_alpha(self):
        registry = FilterRegistry(custom_dir=Path('/missing/custom/filters'))
        random = np.random.default_rng(123)
        image = random.integers(0, 256, (64, 96, 4), dtype=np.uint8)
        image[:, :, 3] = random.integers(32, 256, image.shape[:2], dtype=np.uint8)
        params = {
            'shift': 5.0, 'block_size': 4.0, 'activity': 1.0,
            'rgb_split': 3.0, 'seed': 19,
        }
        runtime = registry.resolve(FilterEffect('builtin:glitch', params=params))
        context = FilterContext(1.0, -41, -27)
        full = runtime.apply(image, runtime.params, context)
        repeated = runtime.apply(image, runtime.params, context)
        np.testing.assert_array_equal(full, repeated)

        changed_seed = registry.resolve(FilterEffect(
            'builtin:glitch', params={**params, 'seed': 20}
        ))
        self.assertFalse(np.array_equal(
            full, changed_seed.apply(image, changed_seed.params, context)
        ))

        halo = runtime.tile_halo(runtime.params, context.render_scale)
        x, y, width, height = 24, 20, 42, 25
        left, top = x - halo, y - halo
        right, bottom = x + width + halo, y + height + halo
        tile = image[top:bottom, left:right]
        tile_result = runtime.apply(
            tile,
            runtime.params,
            FilterContext(
                1.0, context.origin_x + left, context.origin_y + top
            ),
        )
        np.testing.assert_array_equal(
            full[y:y + height, x:x + width],
            tile_result[halo:halo + height, halo:halo + width],
        )

        blocked_source = np.zeros((8, 41, 4), dtype=np.uint8)
        blocked_source[:, 20] = (255, 255, 255, 255)
        blocked_runtime = registry.resolve(FilterEffect(
            'builtin:glitch', params={
                'shift': 10.0, 'block_size': 4.0, 'activity': 1.0,
                'rgb_split': 0.0, 'seed': 19,
            },
        ))
        blocked = blocked_runtime.apply(
            blocked_source,
            blocked_runtime.params,
            FilterContext(1.0, 0, -4),
        )
        for row in range(1, 4):
            np.testing.assert_array_equal(blocked[0], blocked[row])
        for row in range(5, 8):
            np.testing.assert_array_equal(blocked[4], blocked[row])
        self.assertFalse(np.array_equal(blocked[0], blocked[4]))

        split_source = np.zeros((11, 25, 4), dtype=np.uint8)
        split_source[:, 12] = (255, 255, 255, 255)
        split_runtime = registry.resolve(FilterEffect(
            'builtin:glitch', params={
                'shift': 0.0, 'block_size': 1.0, 'activity': 1.0,
                'rgb_split': 3.0, 'seed': 1,
            },
        ))
        split = split_runtime.apply(
            split_source, split_runtime.params, FilterContext(1.0, 0, -5)
        )
        self.assertGreater(
            np.count_nonzero(split[:, :, 3]),
            np.count_nonzero(split_source[:, :, 3]),
        )
        self.assertFalse(np.any(split[:, :, :3][split[:, :, 3] == 0] != 0))

    def test_rough_edge_grows_only_its_declared_halo(self):
        registry = FilterRegistry(custom_dir=Path('/missing/custom/filters'))
        image = np.zeros((30, 34, 4), dtype=np.uint8)
        image[8:22, 10:24] = (40, 90, 170, 255)
        runtime = registry.resolve(FilterEffect(
            'builtin:rough_edge', params={
                'amount': 1.0, 'size': 2.0,
                'hardness': 0.8, 'seed': 17,
            },
        ))

        result = runtime.apply(
            image.copy(), runtime.params, FilterContext(1.0, -4, 9)
        )

        expanded = (image[:, :, 3] == 0) & (result[:, :, 3] > 0)
        self.assertGreater(np.count_nonzero(expanded), 0)
        self.assertGreater(int(result[:, :, 3][expanded].max()), 200)
        halo = int(runtime.tile_halo(runtime.params, 1.0))
        allowed = cv2.dilate(
            (image[:, :, 3] > 0).astype(np.uint8),
            np.ones((halo * 2 + 1, halo * 2 + 1), dtype=np.uint8),
        )
        self.assertFalse(np.any(expanded & (allowed == 0)))


if __name__ == '__main__':
    unittest.main()
