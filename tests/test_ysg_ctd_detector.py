import doctest
import os
import subprocess
import sys
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ballontranslator.modules import TEXTDETECTORS
from ballontranslator.modules.base import merge_config_module_params
from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.utils.io_utils import json_dump_nested_obj
from ballontranslator.utils.textblock import TextBlock


def setUpModule():
    # Resolving a detector replaces its lazy spec. Keep startup tests isolated.
    global registry_patch
    registry_patch = patch.dict(TEXTDETECTORS.module_dict, {
        key: deepcopy(TEXTDETECTORS.get_spec(key)) for key in ('ysgyolo', 'ysg_ctd')
    })
    registry_patch.start()
    global detector_ysg_ctd, YSGYoloDetector, YSGYoloCTDDetector, _combine_refined_mask
    from ballontranslator.modules.textdetector import detector_ysg_ctd
    from ballontranslator.modules.textdetector.detector_ysg import YSGYoloDetector
    from ballontranslator.modules.textdetector.detector_ysg_ctd import (
        YSGYoloCTDDetector, _combine_refined_mask,
    )


def tearDownModule():
    registry_patch.stop()


def _block(points):
    block = TextBlock(lines=[points])
    block.adjust_bbox()
    return block


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self

    def numpy(self):
        return self.values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return _Tensor(self.values[index])

    def __int__(self):
        return int(self.values)


def _prediction(boxes=(), polygons=(), names=None):
    return SimpleNamespace(
        names={0: 'balloon'} if names is None else names,
        boxes=SimpleNamespace(cls=_Tensor(np.zeros(len(boxes))), xyxy=_Tensor(boxes)),
        obb=SimpleNamespace(cls=_Tensor(np.zeros(len(polygons))), xyxyxyxy=_Tensor(polygons)),
    )


class MaskCombinationTests(unittest.TestCase):
    def test_refinement_clips_sfx_and_falls_back_per_line(self):
        first = _block([[5, 5], [20, 5], [20, 20], [5, 20]])
        missed = _block([[25, 5], [35, 5], [35, 20], [25, 20]])
        first.lines.extend(missed.lines)
        fine = np.zeros((40, 40), np.uint8)
        fine[10:12, 10:12] = 255
        fine[30:35, 5:10] = 255
        result = _combine_refined_mask(fine, [first], [TextBlock(xyxy=[5, 5, 20, 20])])
        expected = np.zeros_like(fine)
        expected[10:12, 10:12] = 255
        expected[5:21, 25:36] = 255
        np.testing.assert_array_equal(result, expected)

    def test_empty_ctd_refinement_keeps_polygon_even_with_matching_block(self):
        points = [[-5, 5], [10, -5], [25, 5], [10, 20]]
        fine = np.zeros((16, 16), np.uint8)
        result = _combine_refined_mask(fine, [_block(points)], [TextBlock(xyxy=[0, 0, 16, 16])])
        expected = np.zeros_like(fine)
        cv2.fillPoly(expected, [np.array(points, np.int32)], 255)
        np.testing.assert_array_equal(result, expected)

    def test_neighbor_pixels_cannot_suppress_rotated_line_fallback(self):
        diamond = _block([[2, 10], [10, 2], [18, 10], [10, 18]])
        neighbor = _block([[2, 2], [5, 2], [5, 5], [2, 5]])
        fine = np.zeros((24, 24), np.uint8)
        fine[3, 3] = 255
        ctd = [TextBlock(xyxy=[2, 2, 18, 18])]
        result = _combine_refined_mask(fine, [diamond, neighbor], ctd)
        self.assertEqual(result[10, 10], 255)
        self.assertEqual(result[3, 3], 255)
        self.assertEqual(result[2, 2], 0)
        np.testing.assert_array_equal(result, _combine_refined_mask(fine, [neighbor, diamond], ctd))

    def test_fallback_pixels_do_not_count_as_ctd_refinement_for_later_lines(self):
        first = _block([[1, 1], [10, 1], [10, 10], [1, 10]])
        second = _block([[8, 1], [18, 1], [18, 10], [8, 10]])
        fine = np.zeros((20, 20), np.uint8)
        result = _combine_refined_mask(fine, [first, second], [TextBlock(xyxy=[0, 0, 20, 20])])
        self.assertEqual(result[5, 15], 255)
        np.testing.assert_array_equal(result, _combine_refined_mask(fine, [second, first], [TextBlock(xyxy=[0, 0, 20, 20])]))

    def test_outside_image_and_empty_blocks(self):
        fine = np.full((10, 10), 255, np.uint8)
        outside = _block([[-10, -10], [-5, -10], [-5, -5], [-10, -5]])
        self.assertFalse(np.any(_combine_refined_mask(fine, [outside], [])))
        self.assertFalse(np.any(_combine_refined_mask(fine, [], [])))


class HybridDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = TEXTDETECTORS.resolve_module('ysg_ctd')(**{
            'merge text lines': False, 'mask dilate size': 0, 'device': 'cpu',
        })
        self.image = np.zeros((60, 60, 3), np.uint8)
        self.prediction = _prediction(boxes=[[10, 10, 40, 30]])
        self.detector.model = Mock()
        self.detector.model.predict.return_value = [self.prediction]
        self.fine = np.zeros((60, 60), np.uint8)
        self.fine[20:23, 20:23] = 255
        self.fine[45:55, 45:55] = 255
        self.detector.ctd_model = Mock(return_value=(None, self.fine, [TextBlock(xyxy=[10, 10, 40, 30])]))
        self.addCleanup(self.detector.unload_model)

    def test_detection_keeps_yolo_blocks_and_parameters(self):
        self.detector.updateParam('font size multiplier', 1.5)
        self.detector.updateParam('confidence threshold', 0.4)
        _, original = YSGYoloDetector._detect(self.detector, self.image)
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual(len(blocks), len(original))
        self.assertEqual(blocks[0].lines, original[0].lines)
        self.assertEqual(blocks[0].font_size, original[0].font_size)
        self.assertEqual(blocks[0].det_model, 'ysg_ctd')
        self.assertEqual(self.detector.model.predict.call_args.kwargs['conf'], 0.4)
        expected = np.zeros_like(mask)
        expected[20:23, 20:23] = 255
        np.testing.assert_array_equal(mask, expected)

    def test_merged_lines_and_oriented_boxes_keep_yolo_geometry(self):
        self.detector.updateParam('merge text lines', True)
        self.detector.model.predict.return_value = [_prediction(
            polygons=[[[10, 20], [25, 10], [40, 20], [25, 30]]],
        )]
        _, expected = YSGYoloDetector._detect(self.detector, self.image)
        mask, blocks = self.detector.detect(self.image)
        np.testing.assert_array_equal(blocks[0].lines, expected[0].lines)
        np.testing.assert_array_equal(blocks[0].xyxy, expected[0].xyxy)
        self.assertEqual(mask[20, 20], 255)
        self.assertEqual(mask[10, 10], 0)

    def test_complete_blocks_and_coarse_masks_match_ysg_across_options(self) -> None:
        for oriented in (False, True):
            for merge in (False, True):
                for dilation in (0, 2):
                    with self.subTest(oriented=oriented, merge=merge, dilation=dilation):
                        boxes = [[10, 10, 20, 30], [25, 10, 35, 30], [40, 10, 50, 30]]
                        polygons = [cv2.boxPoints(((15 + 15 * i, 20), (10, 20), 15)) for i in range(3)]
                        prediction = _prediction(
                            boxes=() if oriented else boxes,
                            polygons=polygons if oriented else (),
                            names={0: 'balloon', 1: 'other'},
                        )
                        detections = prediction.obb if oriented else prediction.boxes
                        detections.cls = _Tensor([0, 0, 1])
                        self.detector.model.predict.return_value = [prediction]
                        self.detector.updateParam('merge text lines', merge)
                        self.detector.updateParam('mask dilate size', dilation)
                        self.detector.updateParam('font size multiplier', 1.3)
                        self.detector.updateParam('font size min', 12)
                        self.detector.updateParam('font size max', 28)
                        expected_mask, expected_blocks = YSGYoloDetector._detect(self.detector, self.image)
                        for block in expected_blocks:
                            block.det_model = 'ysg_ctd'
                        self.detector.ctd_model.return_value = (None, np.zeros_like(self.fine), [])
                        mask, blocks = self.detector.detect(self.image)
                        np.testing.assert_array_equal(mask, expected_mask)
                        self.assertEqual(
                            json_dump_nested_obj([block.to_dict() for block in blocks], sort_keys=True),
                            json_dump_nested_obj([block.to_dict() for block in expected_blocks], sort_keys=True),
                        )

    def test_dilation_applies_once_to_final_mask(self):
        self.detector.updateParam('mask dilate size', 2)
        with patch.object(cv2, 'dilate', wraps=cv2.dilate) as dilate:
            mask, _ = self.detector.detect(self.image)
        self.assertEqual(dilate.call_count, 1)
        self.detector.model.predict.assert_called_once()
        self.detector.ctd_model.assert_called_once()
        expected = np.zeros_like(mask)
        expected[20:23, 20:23] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (2, 2))
        np.testing.assert_array_equal(mask, cv2.dilate(expected, kernel))

    def test_box_outputs_are_transferred_once_per_tensor(self):
        prediction = _prediction(
            boxes=[[1, 1, 8, 8], [10, 10, 30, 30], [35, 35, 45, 45]],
            polygons=[[[10, 20], [25, 10], [40, 20], [25, 30]]],
        )
        self.detector.model.predict.return_value = [prediction]
        self.detector.detect(self.image)
        for tensor in (prediction.boxes.cls, prediction.boxes.xyxy,
                       prediction.obb.cls, prediction.obb.xyxyxyxy):
            self.assertEqual(tensor.cpu_calls, 1)

    def test_disabled_labels_skip_both_inference_calls(self):
        self.detector.updateParam('label', {name: False for name in self.detector.get_param_value('label')})
        mask, blocks = self.detector.detect(self.image)
        self.assertFalse(np.any(mask))
        self.assertEqual(blocks, [])
        self.detector.model.predict.assert_not_called()
        self.detector.ctd_model.assert_not_called()

    def test_ctd_failure_uses_yolo_mask_but_yolo_failure_propagates(self):
        self.detector.updateParam('mask dilate size', 2)
        expected, _ = YSGYoloDetector._detect(self.detector, self.image)
        self.detector.ctd_model.side_effect = RuntimeError('CTD failed')
        with patch.object(self.detector.logger, 'warning') as warning, \
                patch.object(cv2, 'dilate', wraps=cv2.dilate) as dilate:
            mask, blocks = self.detector.detect(self.image)
        warning.assert_called_once()
        self.assertEqual(dilate.call_count, 1)
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(len(blocks), 1)
        self.detector.model.predict.side_effect = RuntimeError('YSG failed')
        with self.assertRaises(ModuleRunError):
            self.detector.detect(self.image)

    def test_empty_predictions_or_unselected_model_labels_skip_ctd(self):
        for prediction in (_prediction(), _prediction(names={0: 'other'})):
            with self.subTest(names=prediction.names):
                self.detector.model.predict.return_value = [prediction]
                mask, blocks = self.detector.detect(self.image)
                self.assertEqual(mask.shape, (60, 60))
                self.assertEqual(mask.dtype, np.uint8)
                self.assertFalse(np.any(mask))
                self.assertEqual(blocks, [])
        self.detector.ctd_model.assert_not_called()

    def test_rgba_input_is_converted_by_shared_detect_entrypoint(self):
        self.detector.detect(np.zeros((60, 60, 4), np.uint8))
        self.assertEqual(self.detector.ctd_model.call_args.args[0].shape, (60, 60, 3))

    def test_settings_do_not_mutate_original_or_other_instances(self):
        original = deepcopy(YSGYoloDetector.params)
        self.detector.updateParam('confidence threshold', 0.8)
        with patch.object(detector_ysg_ctd, 'find_model_paths') as find_paths:
            another = YSGYoloCTDDetector()
        find_paths.assert_not_called()
        self.addCleanup(another.unload_model)
        self.assertEqual(another.get_param_value('confidence threshold'), 0.3)
        self.assertEqual(YSGYoloDetector.params, original)

    def test_only_changed_backend_is_invalidated_without_loading(self):
        ysg, ctd = self.detector.model, self.detector.ctd_model
        with patch.object(self.detector, 'load_model') as load:
            self.detector.updateParam('ctd detect size', '1280')
            self.assertIs(self.detector.ctd_model, ctd)
            self.detector.updateParam('model path', 'data/models/custom.pt')
            self.assertIsNone(self.detector.model)
            self.assertIs(self.detector.ctd_model, ctd)
            self.detector.model = ysg
            self.detector.updateParam('ctd det rearrange max batches', 8)
            self.assertIs(self.detector.ctd_model, ctd)
            self.assertEqual(ctd.det_rearrange_max_batches, 8)
            self.detector.updateParam('ctd detect size', 1024)
            self.assertIs(self.detector.ctd_model, ctd)
            self.assertEqual(ctd.detect_size, 1024)
            self.assertIs(self.detector.model, ysg)
            self.detector.updateParam('device', 'cuda')
            self.assertIsNone(self.detector.model)
            self.assertIsNone(self.detector.ctd_model)
            load.assert_not_called()

    def test_unload_releases_both_backends(self):
        self.assertTrue(self.detector.all_model_loaded())
        self.assertTrue(self.detector.unload_model())
        self.assertFalse(self.detector.all_model_loaded())
        self.assertIsNone(self.detector.model)
        self.assertIsNone(self.detector.ctd_model)

    def test_ctd_load_uses_device_options_and_preserves_existing_instance(self):
        # Keep tests runnable without either inference dependency installed.
        fake_ctd = Mock()
        modules = {
            'ballontranslator.modules.textdetector.ctd': SimpleNamespace(CTDModel=fake_ctd),
            'ballontranslator.modules.textdetector.detector_ctd': SimpleNamespace(
                CTD_ONNX_PATH='ctd.onnx', CTD_TORCH_PATH='ctd.pt',
            ),
            'ultralytics': SimpleNamespace(YOLO=Mock()),
        }
        with patch.dict(sys.modules, modules), \
                patch.object(detector_ysg_ctd.osp, 'exists', return_value=True):
            for device, path in (('cpu', 'ctd.onnx'), ('cuda', 'ctd.pt')):
                self.detector.updateParam('device', device)
                self.detector.ctd_model = None
                self.detector.load_model()
                fake_ctd.assert_called_with(path, device=device, detect_size=1280, det_rearrange_max_batches=4)
                fake_ctd.reset_mock()
                self.detector.load_model()
                fake_ctd.assert_not_called()

    def test_missing_checkpoint_prefers_primary_without_rescanning_models(self) -> None:
        self.detector.updateParam('model path', 'data/models/missing.pt')
        constructor = Mock()
        with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=constructor)}), \
                patch.object(self.detector, 'flush') as flush, \
                patch.object(detector_ysg_ctd.osp, 'exists', side_effect=lambda path: path != 'data/models/missing.pt'), \
                patch.object(self.detector.logger, 'warning'):
            self.detector.load_model()
            self.detector.load_model()
        constructor.assert_called_once_with('data/models/ysgyolo_yolo26_2.0.pt')
        flush.assert_not_called()

    def test_missing_all_checkpoints_fails_before_backend_import_or_download(self) -> None:
        self.detector.updateParam('model path', 'data/models/missing.pt')
        constructor = Mock()
        with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=constructor)}), \
                patch.object(self.detector, 'flush', return_value=[]), \
                patch.object(detector_ysg_ctd.osp, 'exists', return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, 'Choose an existing model path'):
                self.detector.load_model()
        constructor.assert_not_called()

    def test_yolo_reload_keeps_ctd_and_ctd_tuning_keeps_both_models(self) -> None:
        ysg, ctd = self.detector.model, self.detector.ctd_model
        constructor = Mock()
        with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=constructor)}), \
                patch.object(detector_ysg_ctd.osp, 'exists', return_value=True):
            self.detector.updateParam('ctd detect size', 1024)
            self.detector.updateParam('ctd det rearrange max batches', 8)
            self.detector.load_model()
            self.assertIs(self.detector.model, ysg)
            self.assertIs(self.detector.ctd_model, ctd)
            constructor.assert_not_called()
            self.detector.updateParam('model path', 'data/models/ysgyolo_yolo26OBB_2.0.pt')
            self.detector.load_model()
            self.detector.load_model()
        constructor.assert_called_once_with('data/models/ysgyolo_yolo26OBB_2.0.pt')
        self.assertIs(self.detector.ctd_model, ctd)


class HybridMetadataTests(unittest.TestCase):
    def test_lazy_discovery_does_not_import_either_backend(self):
        result = subprocess.run(
            [sys.executable, '-c', '''
import sys
from ballontranslator.modules import TEXTDETECTORS
from ballontranslator.modules.lazy_registry import validate_lazy_module_specs
spec = TEXTDETECTORS.get_spec('ysg_ctd')
assert spec is not None
assert not validate_lazy_module_specs([spec])
assert len(spec.download_file_list) == 4
assert spec.params['model path']['value'] == 'data/models/ysgyolo_yolo26_2.0.pt'
assert set(spec.dependencies) == {'torch', 'ultralytics>=8.4.14', 'torchvision', 'einops'}
assert not any(name in sys.modules for name in ('torch', 'ultralytics',
    'ballontranslator.modules.textdetector.detector_ysg',
    'ballontranslator.modules.textdetector.detector_ysg_ctd',
    'ballontranslator.modules.textdetector.ctd'))
'''], capture_output=True, text=True, env=os.environ.copy(),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_older_config_gets_independent_defaults_and_saved_values_survive(self):
        saved = {'ysgyolo': {'confidence threshold': 0.2}}
        merge_config_module_params(saved, ['ysgyolo', 'ysg_ctd'], TEXTDETECTORS.get)
        self.assertEqual(saved['ysg_ctd']['confidence threshold']['value'], 0.3)
        saved['ysg_ctd']['confidence threshold']['value'] = 0.7
        merge_config_module_params(saved, ['ysgyolo', 'ysg_ctd'], TEXTDETECTORS.get)
        self.assertEqual(saved['ysg_ctd']['confidence threshold']['value'], 0.7)
        self.assertEqual(saved['ysgyolo']['confidence threshold']['value'], 0.2)

    def test_doctests(self):
        self.assertEqual(doctest.testmod(detector_ysg_ctd).failed, 0)


if __name__ == '__main__':
    unittest.main()
