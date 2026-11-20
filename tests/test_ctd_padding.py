"""Opt-in CTD padding tests against checkout sources; no model inference or IO."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ballontranslator.modules.base import merge_config_module_params
from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.modules.lazy_registry import _scan_file, validate_lazy_module_specs
from ballontranslator.modules.textdetector import detector_ctd as ctd_module
from ballontranslator.modules.textdetector.ctd_padding import _intersections
from ballontranslator.modules.textdetector.detector_ctd import ComicTextDetector
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.config import ProgramConfig
from ballontranslator.utils.imgproc_utils import rotate_polygons, xywh2xyxypoly

CTD_SOURCE = Path(ctd_module.__file__).resolve()


class CTDPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector_class = ComicTextDetector
        cls.default_params = deepcopy(ComicTextDetector.params)

    def setUp(self):
        self.detector_class.params = deepcopy(self.default_params)
        self.detector = self.detector_class()
        self.detector.updateParam('Detect box padding (px)', 4)
        self.image = np.zeros((80, 100, 3), dtype=np.uint8)
        self.mask = np.arange(8000, dtype=np.uint8).reshape(80, 100)

    def tearDown(self):
        self.detector_class.params = deepcopy(self.default_params)

    def test_sources_are_from_this_checkout(self):
        expected = Path(__file__).resolve().parents[1] / 'ballontranslator/modules/textdetector/detector_ctd.py'
        self.assertEqual(CTD_SOURCE, expected)

    def test_default_zero_preserves_legacy_geometry_mask_and_font_processing(self):
        self.assertEqual(self.default_params['Detect box padding (px)'], 0)
        self.detector.updateParam('Detect box padding (px)', 0)
        blocks = [self.block(_bounding_rect=[20, 20, 40, 30]), self.block(angle=3)]
        original = [(deepcopy(block.xyxy), deepcopy(block.lines), deepcopy(block._bounding_rect)) for block in blocks]
        self.detector.model = lambda image: (None, self.mask, blocks)
        self.detector.set_param_value('font size multiplier', 1.5)
        self.detector.set_param_value('font size max', 16)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
        expected_mask = cv2.dilate(self.mask, element)
        with patch('ballontranslator.modules.textdetector.detector_ctd.pad_ctd_boxes') as pad:
            mask, returned = self.detector.detect(self.image)
        pad.assert_not_called()
        self.assertIs(returned, blocks)
        self.assertEqual([(block.xyxy, block.lines, block._bounding_rect) for block in blocks], original)
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertTrue(all(block.font_size == 16 and block._detected_font_size == 16 for block in blocks))
        self.assertTrue(all(block.det_model == 'ctd' for block in blocks))

    def block(self, box=(20, 20, 60, 50), **kwargs):
        return TextBlock(
            xyxy=list(box), lines=[[[20, 20], [60, 20], [60, 50], [20, 50]]],
            _detected_font_size=12, **kwargs,
        )

    def detect_blocks(self, blocks):
        # Run real inherited _detect (font/mask processing), stub only inference.
        self.detector.model = lambda image: (None, self.mask, blocks)
        self.detector.set_param_value('mask dilate size', 0)
        return self.detector.detect(self.image)

    def test_enabled_preserves_mask_lines_font_and_source_min_rect(self):
        block = self.block(_bounding_rect=[1, 2, 3, 4])
        lines, source_rect = deepcopy(block.lines), block.min_rect().copy()
        mask_bytes = self.mask.tobytes()
        mask, blocks = self.detect_blocks([block])
        self.assertIs(mask, self.mask)
        self.assertEqual(mask.tobytes(), mask_bytes)
        self.assertIs(blocks[0], block)
        self.assertEqual(block.xyxy, [16, 16, 64, 54])
        self.assertEqual(block.bounding_rect(), [16, 16, 48, 38])
        self.assertEqual(block.to_dict()['_bounding_rect'], [16, 16, 48, 38])
        self.assertEqual(block.to_dict()['xyxy'], [16, 16, 64, 54])
        self.assertEqual(block.lines, lines)
        np.testing.assert_array_equal(block.min_rect(), source_rect)
        self.assertEqual(block.font_size, 12)
        self.assertEqual(block._detected_font_size, 12)
        self.assertEqual(block.det_model, 'ctd')

    def test_line_ocr_crop_matches_ctd_for_zero_and_default_padding(self):
        self.image = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
        upstream = self.block(det_model='ctd')
        expected = upstream.get_transformed_region(self.image, 0, 24)
        for padding in (0, 4):
            with self.subTest(padding=padding):
                self.detector.updateParam('Detect box padding (px)', padding)
                block = self.block()
                self.detect_blocks([block])
                np.testing.assert_array_equal(block.get_transformed_region(self.image, 0, 24), expected)

    def test_mask_and_font_match_upstream_processing(self):
        self.detector.model = lambda image: (None, self.mask, [self.block()])
        self.detector.set_param_value('font size multiplier', 1.5)
        self.detector.set_param_value('font size max', 16)
        expected_mask, expected_blocks = ComicTextDetector._detect(self.detector, self.image, None)
        mask, blocks = self.detector.detect(self.image)
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertEqual(blocks[0].font_size, expected_blocks[0].font_size)
        self.assertEqual(blocks[0]._detected_font_size, expected_blocks[0]._detected_font_size)

    def test_custom_and_clipping_in_original_pixels(self):
        self.detector.updateParam('Detect box padding (px)', '8')
        block = self.block((2, 3, 99, 79))
        self.detect_blocks([block])
        self.assertEqual(block.xyxy, [0, 0, 100, 80])
        self.assertEqual(block.bounding_rect(), [0, 0, 100, 80])

    def test_zero_and_empty(self):
        self.detector.updateParam('Detect box padding (px)', 0)
        block = self.block(_bounding_rect=[20, 20, 40, 30])
        original = block.to_dict(deep_copy=True)
        self.detect_blocks([block])
        self.assertEqual(block.xyxy, original['xyxy'])
        self.assertEqual(block._bounding_rect, original['_bounding_rect'])
        mask, blocks = self.detect_blocks([])
        self.assertIs(mask, self.mask)
        self.assertEqual(blocks, [])

    def test_vertical_and_invalid_boxes_are_unchanged(self):
        blocks = [self.block(src_is_vertical=True), self.block(angle=3, src_is_vertical=True),
                  self.block((60, 20, 20, 50)), self.block((200, 200, 210, 210))]
        originals = [(deepcopy(b.xyxy), deepcopy(b._bounding_rect)) for b in blocks]
        self.detect_blocks(blocks)
        self.assertEqual([(b.xyxy, b._bounding_rect) for b in blocks], originals)

    def angled_block(self, angle, center=(100, 100), half_size=(30, 20)):
        center, half = np.array(center, dtype=float), np.array(half_size, dtype=float)
        rect = [*(center-half), *(2*half)]
        lines = rotate_polygons(center, xywh2xyxypoly(np.array([rect])),
                                -angle, to_int=False).reshape(-1, 4, 2).tolist()
        # A symmetric integer AABB matches CTD's source rotation center.
        radius = np.ceil(np.abs(np.array(lines).reshape(-1, 2)-center).max(axis=0))
        lower = np.floor(center-radius).astype(int)
        upper = np.rint(2*center).astype(int)-lower
        return TextBlock(xyxy=[*lower, *upper],
                         lines=lines, angle=angle, _bounding_rect=rect, _detected_font_size=12)

    def assert_rotated_coverage(self, block, original_center):
        rect = block.bounding_rect()
        polygon = rotate_polygons([rect[0]+rect[2]/2, rect[1]+rect[3]/2],
                                  xywh2xyxypoly(np.array([rect])),
                                  -block.angle, to_int=False).reshape(-1, 2)
        self.assertTrue((polygon.min(axis=0) >= np.array(block.xyxy[:2])).all())
        self.assertTrue((polygon.max(axis=0) <= np.array(block.xyxy[2:])).all())
        np.testing.assert_array_equal(block.center(), original_center)
        self.assertGreaterEqual(block.xyxy[0], 0)
        self.assertGreaterEqual(block.xyxy[1], 0)
        self.assertLessEqual(block.xyxy[2], self.image.shape[1])
        self.assertLessEqual(block.xyxy[3], self.image.shape[0])

    def test_rotated_local_padding_and_unchanged_line_ocr(self):
        self.image = np.arange(200*200*3, dtype=np.uint8).reshape(200, 200, 3)
        self.mask = np.zeros((200, 200), dtype=np.uint8)
        self.detector.updateParam('Detect box padding (px)', 16)
        for angle in (1, -1, 3, -3, 10, -10, -0.0790864, -2.7882571, 0.1693675):
            with self.subTest(angle=angle):
                block = self.angled_block(angle)
                original_center = block.center().copy()
                lines, source_rect = deepcopy(block.lines), block.min_rect().copy()
                block.det_model = 'ctd'
                expected_crop = block.get_transformed_region(self.image, 0, 24)
                original_local = np.array(block.bounding_rect())
                mask, _ = self.detect_blocks([block])
                self.assertIs(mask, self.mask)
                self.assertEqual(block.angle, angle)
                self.assertEqual(block.lines, lines)
                self.assertEqual(block.font_size, 12)
                self.assertEqual(block._detected_font_size, 12)
                self.assertEqual(block.det_model, 'ctd')
                self.assertGreaterEqual(block.bounding_rect()[2], original_local[2]+32)
                self.assertGreaterEqual(block.bounding_rect()[3], original_local[3]+32)
                self.assert_rotated_coverage(block, original_center)
                np.testing.assert_array_equal(block.min_rect(), source_rect)
                np.testing.assert_array_equal(block.get_transformed_region(self.image, 0, 24), expected_crop)
                self.assertEqual(block.to_dict()['_bounding_rect'], block.bounding_rect())

    def test_rotated_zero_keeps_original_geometry(self):
        self.detector.updateParam('Detect box padding (px)', 0)
        block = self.angled_block(3, center=(50, 40), half_size=(20, 10))
        original = block.to_dict(deep_copy=True)
        self.detect_blocks([block])
        self.assertEqual(block.xyxy, original['xyxy'])
        self.assertEqual(block._bounding_rect, original['_bounding_rect'])
        self.assertEqual(block.angle, original['angle'])
        self.assertEqual(block.lines, original['lines'])

    def test_rotated_half_pixel_center_and_native_rounding_offset(self):
        self.image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.mask = np.zeros((200, 200), dtype=np.uint8)
        block = self.angled_block(-2.7882571, center=(100.5, 100.5))
        # Saved CTD native rect can differ from xyxy's center by one pixel.
        block._bounding_rect[0] -= 1
        source_rect = block.min_rect().copy()
        center = block.center().copy()
        block.det_model = 'ctd'
        source_crop = block.get_transformed_region(self.image, 0, 24)
        self.detect_blocks([block])
        self.assert_rotated_coverage(block, center)
        np.testing.assert_array_equal(block.min_rect(), source_rect)
        np.testing.assert_array_equal(block.get_transformed_region(self.image, 0, 24), source_crop)

    def test_rotated_boundary_reduces_padding_without_moving_lettering(self):
        self.detector.updateParam('Detect box padding (px)', 16)
        for angle in (10, -10, 0.1693675):
            with self.subTest(angle=angle):
                block = self.angled_block(angle, center=(30, 40), half_size=(20, 15))
                original = np.array(block.bounding_rect())
                original_center = block.center().copy()
                source_rect = block.min_rect().copy()
                with patch.object(self.detector.logger, 'warning') as warning:
                    self.detect_blocks([block])
                warning.assert_called_once()
                self.assertIn('reduced rotated box padding', warning.call_args.args[0])
                self.assertGreater(block.bounding_rect()[2], original[2])
                self.assertLess(block.bounding_rect()[2], original[2]+32)
                self.assert_rotated_coverage(block, original_center)
                np.testing.assert_array_equal(block.min_rect(), source_rect)

    def test_rotated_out_of_page_source_is_retained_with_warning(self):
        block = self.angled_block(10, center=(1, 40), half_size=(20, 15))
        original = block.to_dict(deep_copy=True)
        with patch.object(self.detector.logger, 'warning') as warning:
            self.detect_blocks([block])
        warning.assert_called_once()
        self.assertIn('no room for padding', warning.call_args.args[0])
        self.assertEqual(block.xyxy, original['xyxy'])
        self.assertEqual(block.bounding_rect(), original['_bounding_rect'])

    def prepare_neighbor_page(self):
        self.image = np.arange(200*200*3, dtype=np.uint8).reshape(200, 200, 3)
        self.mask = np.zeros((200, 200), dtype=np.uint8)
        self.detector.updateParam('Detect box padding (px)', 16)

    def neighbor_block(self, box, **kwargs):
        x1, y1, x2, y2 = box
        return TextBlock(xyxy=list(box), lines=[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
                         _detected_font_size=12, **kwargs)

    def test_adjacent_original_neighbors_limit_padding_order_independently(self):
        self.prepare_neighbor_page()
        original_boxes = [[40, 40, 80, 60], [90, 40, 130, 60]]
        outcomes = []
        for boxes in (original_boxes, list(reversed(original_boxes))):
            blocks = [self.neighbor_block(box) for box in boxes]
            original_lines = [deepcopy(block.lines) for block in blocks]
            with patch.object(self.detector.logger, 'warning') as warnings:
                mask, _ = self.detect_blocks(blocks)
            self.assertIs(mask, self.mask)
            self.assertEqual(warnings.call_count, 2)
            outcomes.append({tuple(box): block.xyxy for box, block in zip(boxes, blocks)})
            for index, (block, lines) in enumerate(zip(blocks, original_lines)):
                self.assertEqual(block.lines, lines)
                self.assertEqual(block.font_size, 12)
                self.assertEqual(block._detected_font_size, 12)
                self.assertEqual(block.det_model, 'ctd')
                self.assertFalse(_intersections(np.array(block.xyxy),
                                np.array([boxes[1-index]])).any())
                self.assertIn('xyxy=', warnings.call_args_list[index].args[0])
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][(40, 40, 80, 60)], [30, 30, 90, 70])
        self.assertEqual(outcomes[0][(90, 40, 130, 60)], [80, 30, 140, 70])

    def test_touching_originals_skip_positive_padding(self):
        self.prepare_neighbor_page()
        boxes = [[40, 40, 80, 60], [80, 40, 130, 60]]
        blocks = [self.neighbor_block(box) for box in boxes]
        with patch.object(self.detector.logger, 'warning') as warnings:
            self.detect_blocks(blocks)
        self.assertEqual([block.xyxy for block in blocks], boxes)
        self.assertEqual(warnings.call_count, 2)
        self.assertTrue(all('no room for padding' in call.args[0] for call in warnings.call_args_list))

    def test_diagonal_originals_guard_aabb_but_allow_empty_margin_overlap(self):
        self.prepare_neighbor_page()
        boxes = [[40, 40, 80, 60], [90, 70, 130, 90]]
        blocks = [self.neighbor_block(box) for box in boxes]
        self.detect_blocks(blocks)
        self.assertEqual(blocks[0].xyxy, [30, 30, 90, 70])
        self.assertEqual(blocks[1].xyxy, [80, 60, 140, 100])
        self.assertTrue(_intersections(np.array(blocks[0].xyxy),
                        np.array([blocks[1].xyxy])).any())
        for index, block in enumerate(blocks):
            self.assertFalse(_intersections(np.array(block.xyxy),
                             np.array([boxes[1-index]])).any())

    def test_rotated_crop_aabb_guard_includes_vertical_obstacle(self):
        self.prepare_neighbor_page()
        block = self.angled_block(45, center=(100, 100), half_size=(30, 10))
        obstacle = self.neighbor_block([145, 55, 155, 65], src_is_vertical=True)
        original_center = block.center().copy()
        original_source_rect = block.min_rect().copy()
        block.det_model = 'ctd'
        original_crop = block.get_transformed_region(self.image, 0, 24)
        with patch.object(self.detector.logger, 'warning') as warnings:
            self.detect_blocks([block, obstacle])
        self.assertLess(block.bounding_rect()[2], 60+32)
        self.assertFalse(_intersections(np.array(block.xyxy),
                         np.array([obstacle.xyxy])).any())
        self.assertEqual(obstacle.xyxy, [145, 55, 155, 65])
        self.assertEqual(warnings.call_count, 2)
        self.assertIn('original neighbor', warnings.call_args_list[0].args[0])
        self.assertIn('orientation manually', warnings.call_args_list[1].args[0])
        self.assert_rotated_coverage(block, original_center)
        np.testing.assert_array_equal(block.min_rect(), original_source_rect)
        np.testing.assert_array_equal(block.get_transformed_region(self.image, 0, 24), original_crop)

    def test_preexisting_overlap_and_zero_preserve_originals(self):
        self.prepare_neighbor_page()
        boxes = [[40, 40, 80, 60], [70, 40, 110, 60]]
        for padding in (16, 0):
            blocks = [self.neighbor_block(box) for box in boxes]
            self.detector.updateParam('Detect box padding (px)', padding)
            with patch.object(self.detector.logger, 'warning') as warnings:
                self.detect_blocks(blocks)
            self.assertEqual([block.xyxy for block in blocks], boxes)
            self.assertEqual([block._bounding_rect for block in blocks], [None, None])
            if padding:
                self.assertEqual(warnings.call_count, 2)
                self.assertTrue(all('pre-existing overlap' in call.args[0] for call in warnings.call_args_list))
            else:
                warnings.assert_not_called()

    def test_partial_out_of_page_original_is_not_shrunk(self):
        self.prepare_neighbor_page()
        block = self.neighbor_block([-2, 40, 80, 60])
        with patch.object(self.detector.logger, 'warning') as warnings:
            self.detect_blocks([block])
        self.assertEqual(block.xyxy, [-2, 40, 80, 60])
        warnings.assert_called_once()
        self.assertIn('outside page', warnings.call_args.args[0])

    def test_full_integer_range_and_malformed_params(self):
        for padding in range(65):
            with self.subTest(padding=padding):
                self.detector.updateParam('Detect box padding (px)', padding)
                block = self.block()
                self.detect_blocks([block])
                self.assertEqual(block.xyxy, [max(0, 20-padding), max(0, 20-padding),
                                              min(100, 60+padding), min(80, 50+padding)])
        for value in [-1, 65, 1000000, 4.5, 4.0, True, None, [], {}, '4.5', '-1', 'NaN']:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'whole number from 0 to 64'):
                    self.detector.updateParam('Detect box padding (px)', value)
                self.detector.params['Detect box padding (px)']['value'] = value
                self.detector.model = None
                with patch.object(ctd_module, 'load_ctd_model') as load:
                    with self.assertRaisesRegex(ModuleRunError, 'whole number from 0 to 64'):
                        self.detector.detect(self.image)
                    load.assert_not_called()

    def test_existing_model_lifecycle_loads_once(self):
        self.assertIsNone(self.detector.model)
        self.assertEqual(self.detector._load_model_keys, {'model'})
        model = lambda image: (None, self.mask, [])
        with patch.object(ctd_module, 'load_ctd_model', return_value=model) as loading:
            self.detector.detect(self.image)
            self.detector.detect(self.image)
        self.assertEqual(loading.call_count, 1)

    def test_malformed_live_padding_fails_before_loaded_model_inference(self):
        self.detector.params['Detect box padding (px)']['value'] = '4.5'
        with patch.object(self.detector, 'model') as model:
            with self.assertRaisesRegex(ModuleRunError, 'whole number from 0 to 64'):
                self.detector.detect(self.image)
            model.assert_not_called()

    def test_real_lazy_metadata_is_opt_in_without_model_loading(self):
        with patch('importlib.import_module', side_effect=AssertionError('scan must not import')), \
             patch.object(ctd_module, 'load_ctd_model', side_effect=AssertionError('scan must not load')):
            spec = _scan_file(str(CTD_SOURCE), 'textdetector')[0]
        self.assertEqual(spec.key, 'ctd')
        self.assertEqual(spec.import_path, 'ballontranslator.modules.textdetector.detector_ctd')
        self.assertIsNone(spec.resolved_class)
        self.assertEqual(validate_lazy_module_specs([spec]), [])
        self.assertEqual(spec.params['Detect box padding (px)'], 0)
        self.assertEqual(spec.download_file_list, ComicTextDetector.download_file_list)
        self.assertEqual(len(spec.download_file_list), 2)
        self.assertEqual(spec.dependencies, ComicTextDetector.dependencies)

    def test_normal_config_json_roundtrip(self):
        spec = _scan_file(str(CTD_SOURCE), 'textdetector')[0]
        config = {'unrelated': {'existing': 'unchanged'}}
        merge_config_module_params(config, ['ctd'], lambda key: spec)
        config['ctd']['Detect box padding (px)'] = {'value': 7}
        restored = json.loads(json.dumps(config))
        merge_config_module_params(restored, ['ctd'], lambda key: spec)
        self.assertEqual(restored['unrelated'], {'existing': 'unchanged'})
        detector = self.detector_class(**restored['ctd'])
        self.assertEqual(detector.get_param_value('Detect box padding (px)'), 7)
        self.assertIsNone(detector.model)

    def test_old_ctd_config_gets_zero_without_changing_existing_settings(self):
        spec = _scan_file(str(CTD_SOURCE), 'textdetector')[0]
        config = {'ctd': {'detect_size': {'value': 1024}, 'mask dilate size': {'value': 1}}}
        merge_config_module_params(config, ['ctd'], lambda key: spec)
        detector = self.detector_class(**config['ctd'])
        self.assertEqual(detector.get_param_value('Detect box padding (px)'), 0)
        self.assertEqual(detector.get_param_value('detect_size'), 1024)
        self.assertEqual(detector.get_param_value('mask dilate size'), 1)
        self.assertIsNone(detector.model)

    def test_invalid_saved_padding_recovers_through_real_config_load(self):
        key = 'Detect box padding (px)'
        for value in (65, -1, 'malformed', True, 4.5, 4.0, None, [], {}, '4.5'):
            for structured in (False, True):
                with self.subTest(value=value, structured=structured):
                    saved_value = {'value': value, 'display_name': 'Saved padding'} if structured else value
                    payload = {
                        'recent_proj_list': ['synthetic-project'],
                        'module': {
                            'textdetector': 'ctd', 'enable_ocr': False,
                            'textdetector_params': {
                                'ctd': {key: saved_value, 'detect_size': {'value': 1024}, 'mask dilate size': 1},
                                'other_detector': {'keep': 9},
                            },
                        },
                    }
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'config.json'
                        original_json = json.dumps(payload)
                        path.write_text(original_json, encoding='utf-8')
                        with self.assertLogs('BallonTranslator', level='WARNING') as messages:
                            loaded = ProgramConfig.load(str(path))
                        self.assertEqual(path.read_text(encoding='utf-8'), original_json)
                    self.assertTrue(any('Discard invalid saved CTD' in message for message in messages.output))
                    params = loaded.module.textdetector_params['ctd']
                    recovered = params[key]
                    self.assertEqual(recovered.get('value') if isinstance(recovered, dict) else recovered, 0)
                    if structured:
                        self.assertEqual(params[key]['display_name'], 'Saved padding')
                    self.assertEqual(params['detect_size'], {'value': 1024})
                    self.assertEqual(params['mask dilate size'], 1)
                    self.assertEqual(loaded.module.textdetector_params['other_detector'], {'keep': 9})
                    self.assertEqual(loaded.module.textdetector, 'ctd')
                    self.assertFalse(loaded.module.enable_ocr)
                    self.assertEqual(loaded.recent_proj_list, ['synthetic-project'])
                    spec = _scan_file(str(CTD_SOURCE), 'textdetector')[0]
                    merge_config_module_params(loaded.module.textdetector_params, ['ctd'], lambda name: spec)
                    detector = self.detector_class(**loaded.module.textdetector_params['ctd'])
                    self.assertEqual(detector.get_param_value(key), 0)

    def test_valid_and_missing_saved_padding_survive_real_config_load(self):
        key = 'Detect box padding (px)'
        for saved in ({}, {key: 16}, {key: {'value': 16}}, {key: '4'}, {key: {'value': 64}}):
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps({'module': {'textdetector_params': {'ctd': saved}}}), encoding='utf-8')
                with patch('ballontranslator.utils.config.LOGGER.warning') as warning:
                    loaded = ProgramConfig.load(str(path))
                warning.assert_not_called()
                self.assertEqual(loaded.module.textdetector_params['ctd'], saved)
                spec = _scan_file(str(CTD_SOURCE), 'textdetector')[0]
                merge_config_module_params(loaded.module.textdetector_params, ['ctd'], lambda name: spec)
                detector = self.detector_class(**loaded.module.textdetector_params['ctd'])
                expected = saved.get(key, 0)
                expected = expected['value'] if isinstance(expected, dict) else expected
                self.assertEqual(int(detector.get_param_value(key)), int(expected))


if __name__ == '__main__':
    unittest.main()
