import doctest
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from qtpy.QtCore import QRectF
from qtpy.QtGui import QTransform

from ballontranslator.modules import TEXTDETECTORS
from ballontranslator.modules.base import merge_config_module_params
from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.modules.textdetector import detector_koharu
from ballontranslator.modules.textdetector.detector_koharu import KoharuDetector
from ballontranslator.utils.io_utils import json_dump_nested_obj
from ballontranslator.utils.bubble_shape import _text_mask_path, bubble_path
from ballontranslator.utils.textblock import TextBlock


def prediction() -> dict:
    logits = torch.full((1, 300, 5), -20.0)
    # Include a fifth background slot and a below-threshold text candidate.
    for index, (label, score) in enumerate([(0, .8), (1, .9), (2, .99), (3, .99), (4, .99), (0, .2)]):
        logits[0, index, label] = torch.logit(torch.tensor(score))
    boxes = torch.tensor([[[.25, .5, .2, .5], [.75, .5, .2, .5]]])
    boxes = torch.cat([boxes, torch.full((1, 298, 4), .5)], dim=1)
    masks = torch.full((1, 300, 8, 8), -2.0)
    masks[0, 0, 2:6, 1:3] = 2
    masks[0, 1, 2:6, 5:7] = 2
    masks[0, 2:6] = 2
    return {'pred_logits': logits, 'pred_boxes': boxes, 'pred_masks': masks}


def text_strips(vertical: bool = False, angle: float = 0) -> np.ndarray:
    """Three short lines whose overall bounds suggest the opposite writing mode."""
    y, x = np.mgrid[:160, :160].astype(np.float64) + 0.5 - 80
    radians = np.deg2rad(angle)
    inline = x * np.cos(radians) + y * np.sin(radians)
    block = -x * np.sin(radians) + y * np.cos(radians)
    if vertical:
        inline, block = block, inline
    return (np.abs(inline) < 30) & (np.min(np.abs(block[..., None] - [-40, 0, 40]), axis=-1) < 2)


def expected_cleanup(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Reference disk morphology without the production distance transforms."""
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = (x*x + y*y <= radius*radius).astype(np.uint8)
    return cv2.morphologyEx(cv2.dilate(mask, disk), cv2.MORPH_CLOSE, disk)


class KoharuTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log = patch.object(KoharuDetector.logger, 'info')
        self.log.start()
        self.addCleanup(self.log.stop)
        self.detector = KoharuDetector(device='cpu', **{'detect size': 768})
        self.output = prediction()
        self.detector.model = Mock(return_value=self.output)
        self.addCleanup(self.detector.unload_model)
        self.img = np.zeros((40, 80, 3), np.uint8)
        self.img[..., 0] = 255

    def expected_mask(self, index: int) -> np.ndarray:
        instance = self.output['pred_masks'][:, index:index+1]
        return (F.interpolate(instance, (40, 80), mode='bilinear', align_corners=False)[0, 0].numpy() > 0).astype(np.uint8) * 255

    def test_rgb_preprocessing_and_filtered_segmentation(self) -> None:
        mask, blocks = self.detector.detect(self.img)
        pixels = self.detector.model.call_args.args[0]
        self.assertEqual(tuple(pixels.shape), (1, 3, 768, 768))
        torch.testing.assert_close(pixels[0, :, 20, 20], torch.tensor([(1-.485)/.229, -.456/.224, -.406/.225]))
        self.assertFalse(pixels.requires_grad)
        self.assertEqual({b.label for b in blocks}, {'text', 'onomatopoeia'})
        self.assertEqual(len(blocks), 2)
        np.testing.assert_array_equal(mask, expected_cleanup(self.expected_mask(0) | self.expected_mask(1)))
        self.assertTrue(all(b.det_model == 'koharu' and b.font_size > 0 for b in blocks))
        for block in blocks:
            restored = TextBlock(**json.loads(json_dump_nested_obj(block.to_dict())))
            self.assertEqual(restored.label, block.label)
            self.assertEqual(restored.lines, block.lines)
            self.assertEqual(restored.xyxy, block.xyxy)

    def test_bubble_association_chooses_smallest_containing_mask(self) -> None:
        self.detector.num_select = 4
        masks = np.zeros((4, 80, 120), np.uint8)
        masks[0, 30:45, 20:35] = 1
        masks[1, 60:68, 100:110] = 1
        masks[2, 10:60, 10:55] = 1
        masks[3, 5:75, 5:90] = 1
        logits = torch.full((1, 4, 5), -20.)
        logits[0, [0, 1, 2, 3], [0, 0, 2, 2]] = 20
        self.detector.model.return_value = {
            'pred_logits': logits,
            'pred_boxes': torch.tensor([[[.23, .47, .13, .19], [.875, .8, .09, .1],
                                          [.27, .44, .38, .63], [.4, .5, .72, .88]]]),
            'pred_masks': torch.from_numpy(np.where(masks[None], 2., -2.)).float(),
        }
        mask, blocks = self.detector.detect(np.zeros((80, 120, 3), np.uint8))
        np.testing.assert_array_equal(mask, expected_cleanup(np.maximum(masks[0], masks[1]) * 255))
        bubble_block = next(b for b in blocks if b.xyxy[0] < 50)
        self.assertIsNotNone(bubble_block.bubble_shape)
        self.assertTrue(bubble_block.bubble_layout)
        self.assertLess(bubble_block.bubble_shape.frame[2], 55)
        self.assertIsNone(next(b for b in blocks if b.xyxy[0] > 50).bubble_shape)
        self.assertFalse(next(b for b in blocks if b.xyxy[0] > 50).bubble_layout)
        self.assertEqual(bubble_block._detected_font_size, -1)

    def test_joined_bubble_allocates_separate_flow_shapes(self) -> None:
        self.detector.num_select = 3
        masks = np.zeros((3, 100, 180), np.uint8)
        masks[0, 40:60, 35:55] = 1
        masks[1, 40:60, 125:145] = 1
        cv2.ellipse(masks[2], (45, 50), (40, 45), 0, 0, 360, 1, -1)
        cv2.ellipse(masks[2], (135, 50), (40, 45), 0, 0, 360, 1, -1)
        masks[2, 45:55, 70:110] = 1
        logits = torch.full((1, 3, 5), -20.)
        logits[0, [0, 1, 2], [0, 0, 2]] = 20
        self.detector.model.return_value = {
            'pred_logits': logits,
            'pred_boxes': torch.tensor([[[.25, .5, .12, .2], [.75, .5, .12, .2], [.5, .5, .98, .96]]]),
            'pred_masks': torch.from_numpy(np.where(masks[None], 2., -2.)).float(),
        }
        mask, blocks = self.detector.detect(np.zeros((100, 180, 3), np.uint8))
        self.assertEqual(len(blocks), 2)
        frames = sorted(block.bubble_shape.frame for block in blocks)
        self.assertTrue(all(block.bubble_layout for block in blocks))
        self.assertLessEqual(frames[0][0] + frames[0][2], frames[1][0] + 1)
        np.testing.assert_array_equal(mask, expected_cleanup(np.maximum(masks[0], masks[1]) * 255))

    def test_disconnected_bubble_uses_the_component_containing_its_text(self) -> None:
        self.detector.num_select = 2
        masks = np.zeros((2, 100, 200), np.uint8)
        masks[0, 20:40, 15:30] = 1
        masks[1, 10:60, 5:45] = 1
        masks[1, 5:95, 70:190] = 1
        logits = torch.full((1, 2, 5), -20.)
        logits[0, 0, 0] = logits[0, 1, 2] = 20
        self.detector.model.return_value = {
            'pred_logits': logits,
            'pred_boxes': torch.tensor([[[.1125, .3, .075, .2], [.5, .5, .98, .98]]]),
            'pred_masks': torch.from_numpy(np.where(masks[None], 2., -2.)).float(),
        }
        _, blocks = self.detector.detect(np.zeros((100, 200, 3), np.uint8))
        self.assertEqual(len(blocks), 1)
        x, y, width, height = blocks[0].bubble_shape.frame
        self.assertLess(x, 20)
        self.assertLess(x + width, 50)
        self.assertLess(y, 25)
        self.assertGreater(y + height, 40)

    def test_single_and_shared_bubbles_cover_only_their_box_clipped_text(self) -> None:
        for shared in (False, True):
            with self.subTest(shared=shared):
                text_boxes = ([[.275, .5, .35, .9], [.725, .5, .35, .9]] if shared
                              else [[.5, .5, .55, .92]])
                count = len(text_boxes)
                self.detector.num_select = count + 1
                masks = np.zeros((count + 1, 120, 200), np.uint8)
                if shared:
                    masks[0, 40:80, 45:65] = masks[1, 40:80, 135:155] = 1
                    masks[0, 8, 55] = 1
                else:
                    masks[0, 35:85, 75:125] = 1
                    masks[0, 8, 100] = 1
                expected_text = masks[:count].copy()
                masks[:count, :3] = masks[:count, -3:] = 1
                masks[:count, :, :5] = masks[:count, :, -5:] = 1
                cv2.ellipse(masks[-1], (100, 60), (90, 45), 0, 0, 360, 1, -1)
                logits = torch.full((1, count + 1, 5), -20.)
                logits[0, :count, 0] = logits[0, count, 2] = 20
                self.detector.model.return_value = {
                    'pred_logits': logits,
                    'pred_boxes': torch.tensor([text_boxes + [[.5, .5, .9, .75]]]),
                    'pred_masks': torch.from_numpy(np.where(masks[None], 2., -2.)).float(),
                }
                mask, blocks = self.detector.detect(np.zeros((120, 200, 3), np.uint8))
                np.testing.assert_array_equal(mask, expected_cleanup(masks[:count].max(axis=0) * 255))
                self.assertEqual(len(blocks), count)
                for block in blocks:
                    self.assertTrue(block.bubble_layout)
                    shape = block.bubble_shape
                    path = bubble_path(shape.nodes, QRectF(*shape.frame))
                    center = QRectF(*shape.frame).center()
                    transform = QTransform().translate(center.x(), center.y()).rotate(block.angle)
                    transform.translate(-center.x(), -center.y())
                    index = int(shared and block.xyxy[0] > 100)
                    required = _text_mask_path(expected_text[index]).translated(.5, .5)
                    page_path = transform.map(path)
                    self.assertTrue(required.subtracted(page_path).isEmpty())
                    x1, y1, x2, y2 = block.xyxy
                    self.assertFalse(page_path.contains(QRectF(x1, y1, x2 - x1, y2 - y1)))
                    if shared:
                        other = _text_mask_path(expected_text[1 - index]).translated(.5, .5)
                        self.assertFalse(page_path.intersects(other))

    def test_box_clipping_uses_native_coordinates_before_bubble_association(self) -> None:
        # The 8x8 masks project onto an 80x40 page, with fractional native x bounds.
        self.output['pred_masks'][0, 2] = -2
        self.output['pred_masks'][0, 2, 1:7, :4] = 2
        _, baseline = self.detector.detect(self.img)
        original = next(b for b in baseline if b.label == 'text').bubble_shape
        self.assertIsNotNone(original)
        self.output['pred_masks'][0, 0] = 2
        saved_masks = self.output['pred_masks'].clone()
        mask, blocks = self.detector.detect(self.img)
        block = next(b for b in blocks if b.label == 'text')
        self.assertEqual(block.bubble_shape, original)
        self.assertTrue(block.bubble_layout)
        self.assertTrue(np.all(mask == 255))
        torch.testing.assert_close(self.output['pred_masks'], saved_masks)
        # An instance with foreground only outside its box has no layout constraint.
        self.output['pred_masks'][0, 0] = -2
        self.output['pred_masks'][0, 0, 0] = 2
        _, blocks = self.detector.detect(self.img)
        block = next(b for b in blocks if b.label == 'text')
        self.assertIsNone(block.bubble_shape)
        self.assertFalse(block.bubble_layout)

    def test_labels_filter_both_blocks_and_masks(self) -> None:
        for index, label in enumerate(('text', 'onomatopoeia')):
            self.detector.updateParam('label', {name: name == label for name in ('text', 'onomatopoeia')})
            mask, blocks = self.detector.detect(self.img)
            self.assertEqual([b.label for b in blocks], [label])
            np.testing.assert_array_equal(mask, expected_cleanup(self.expected_mask(index)))
        self.detector.updateParam('label', {'text': False, 'onomatopoeia': False})
        self.detector.model.reset_mock()
        mask, blocks = self.detector.detect(self.img)
        self.assertEqual(blocks, [])
        self.assertFalse(mask.any())
        self.detector.model.assert_not_called()

    def test_thresholds_empty_predictions_and_topk_background(self) -> None:
        self.detector.updateParam('text threshold', 1)
        mask, blocks = self.detector.detect(self.img)
        self.assertEqual([b.label for b in blocks], ['onomatopoeia'])
        np.testing.assert_array_equal(mask, expected_cleanup(self.expected_mask(1)))
        self.detector.updateParam('onomatopoeia threshold', 1)
        mask, blocks = self.detector.detect(self.img)
        self.assertEqual(blocks, [])
        self.assertEqual(mask.shape, self.img.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertFalse(mask.any())
        # Ranking happens across all five logits, before label filtering.
        self.detector.num_select = 1
        self.detector.updateParam('text threshold', 0)
        mask, blocks = self.detector.detect(self.img)
        self.assertEqual(blocks, [])

    def test_font_multiplier_applies_once_without_changing_masks(self) -> None:
        original_mask, original = self.detector.detect(self.img)
        self.detector.updateParam('font size multiplier', 1.5)
        mask, blocks = self.detector.detect(self.img)
        np.testing.assert_array_equal(mask, original_mask)
        for before, after in zip(original, blocks):
            self.assertEqual(after.font_size, before.font_size * 1.5)
            self.assertEqual(after._detected_font_size, -1)

    def test_cleanup_matches_disk_morphology_at_edges_and_overlaps(self) -> None:
        for shape, radius in [((40, 80), 1), ((100, 256), 2), ((100, 768), 5),
                              ((100, 1024), 6), ((200, 2048), 12)]:
            with self.subTest(shape=shape):
                raw = np.zeros(shape, np.uint8)
                raw[:12, :12] = 255
                raw[-12:, -12:] = 255
                raw[20:40, 30:50] = 255
                raw[25:35, 35:45] = 0
                raw[20:40, 52:70] = 255
                instance = torch.from_numpy(np.where(raw > 0, 2., -2.).astype(np.float32))
                self.output['pred_masks'] = instance[None, None].expand(1, 300, *shape)
                self.detector.updateParam('label', {'text': True, 'onomatopoeia': False})
                mask, blocks = self.detector.detect(np.zeros((*shape, 3), np.uint8))
                np.testing.assert_array_equal(mask, expected_cleanup(raw, radius))
                self.assertEqual([b.label for b in blocks], ['text'])
                self.assertEqual(mask.dtype, np.uint8)
                self.assertEqual(mask[0, 0], 255)
                self.assertEqual(mask[-1, -1], 255)

    def test_cleanup_preserves_empty_instance_masks(self) -> None:
        self.output['pred_masks'].fill_(-2.)
        mask, blocks = self.detector.detect(self.img)
        self.assertEqual(len(blocks), 2)
        self.assertFalse(mask.any())
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, self.img.shape[:2])

    def test_boxes_clip_to_page_and_degenerate_boxes_are_ignored(self) -> None:
        self.output['pred_boxes'][0, 0] = torch.tensor([0, 0, .5, .5])
        self.output['pred_boxes'][0, 1] = torch.tensor([2, 2, .1, .1])
        mask, blocks = self.detector.detect(self.img)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].xyxy, [0, 0, 20, 10])
        np.testing.assert_array_equal(mask, expected_cleanup(self.expected_mask(0)))

    def test_mask_lines_determine_direction_rotation_and_ocr_geometry(self) -> None:
        self.detector.num_select = 1
        for vertical, angle in ((False, 0), (True, 0), (False, -22),
                                (True, -8.5), (False, 12.5), (True, 12.5), (False, 2)):
            with self.subTest(vertical=vertical, angle=angle):
                foreground = text_strips(vertical, angle)
                box_size = [.75, .5] if vertical else [.5, .75]
                self.detector.model.return_value = {
                    'pred_logits': torch.tensor([[[20., -20., -20., -20., -20.]]]),
                    'pred_boxes': torch.tensor([[[.5, .5, *box_size]]]),
                    'pred_masks': torch.from_numpy(np.where(foreground, 2., -2.)[None, None]).float(),
                }
                image = np.repeat(np.where(foreground, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
                mask, (block,) = self.detector.detect(image)
                np.testing.assert_array_equal(mask, expected_cleanup(foreground.astype(np.uint8) * 255))
                self.assertEqual(block.src_is_vertical, vertical)
                self.assertEqual(block.vertical, vertical)
                expected_angle = 0 if abs(angle) < 3 else angle
                self.assertAlmostEqual(block.angle, expected_angle, delta=.75)
                line = block.lines_array()[0]
                edge = line[1] - line[0]
                self.assertAlmostEqual(np.rad2deg(np.arctan2(edge[1], edge[0])), block.angle, delta=1)
                crop = block.get_transformed_region(image, 0, 48)
                self.assertEqual(crop.shape[0], 48)
                self.assertGreater(crop.shape[1], 0)
                self.assertTrue(np.any(crop < 128))
                restored = TextBlock(**json.loads(json_dump_nested_obj(block.to_dict())))
                self.assertEqual((restored.angle, restored.src_is_vertical, restored.vertical),
                                 (block.angle, vertical, vertical))

    def test_each_instance_uses_its_own_mask_before_cleanup(self) -> None:
        self.detector.num_select = 2
        self.detector.model.return_value = {
            'pred_logits': torch.tensor([[[20., -20., -20., -20., -20.],
                                          [-20., 20., -20., -20., -20.]]]),
            'pred_boxes': torch.tensor([[[.5, .5, .9, .9], [.5, .5, .9, .9]]]),
            'pred_masks': torch.from_numpy(np.where(np.stack([text_strips(), text_strips(True)]),
                                                    2., -2.)[None]).float(),
        }
        image = np.full((160, 160, 3), 255, np.uint8)
        _, blocks = self.detector.detect(image)
        self.assertEqual({block.label: block.src_is_vertical for block in blocks},
                         {'text': False, 'onomatopoeia': True})

    def test_ambiguous_or_empty_masks_fall_back_to_box_shape(self) -> None:
        mask = np.zeros((80, 80), bool)
        for populated in (False, True):
            if populated:
                mask[30:50, 30:50] = True
            for box, vertical in (([10, 0, 50, 80], True), ([0, 10, 80, 50], False),
                                  ([0, 0, 80, 80], False)):
                with self.subTest(populated=populated, box=box):
                    self.assertEqual(detector_koharu._mask_angle(mask, np.array(box)), (0., vertical))
        # A nearly square mask uses the box inside the 2% tie margin, but not beyond it.
        for width, vertical in ((49, False), (48, True)):
            mask = np.zeros((160, 160), bool)
            mask[30:80, 30:30 + width] = True
            self.assertEqual(detector_koharu._mask_angle(mask, np.array([10, 20, 150, 100])),
                             (0., vertical))

    def test_direction_samples_only_the_box_margin_and_clips_page_edges(self) -> None:
        foreground = np.zeros((160, 160), bool)
        foreground[1:4, :35] = foreground[25:28, :35] = foreground[49:52, :35] = True
        box = np.array([-1.25, -1.5, 36.25, 54.5])
        expected = detector_koharu._mask_angle(foreground, box)
        self.assertEqual(expected, (0., False))
        foreground[:, 45:150] = True
        self.assertEqual(detector_koharu._mask_angle(foreground, box), expected)

    def test_rgba_and_inference_failures_use_shared_entrypoint(self) -> None:
        mask, blocks = self.detector.detect(np.dstack((self.img, np.full((40, 80), 255, np.uint8))))
        self.assertEqual(len(blocks), 2)
        self.detector.model.side_effect = RuntimeError('inference failed')
        with self.assertRaisesRegex(ModuleRunError, 'inference failed'):
            self.detector.detect(self.img)

    def test_invalid_edits_keep_previous_value_and_loaded_model(self) -> None:
        model = self.detector.model
        for key, value in [('text threshold', float('nan')), ('text threshold', 1.1),
                           ('onomatopoeia threshold', -.1), ('detect size', 1024),
                           ('font size multiplier', 0), ('device', 'bad'),
                           ('label', {'text': 'false'}), ('model path', None)]:
            with self.subTest(key=key, value=value):
                previous = self.detector.get_param_value(key)
                with patch.object(self.detector.logger, 'warning') as warning:
                    self.detector.updateParam(key, value)
                warning.assert_called_once()
                self.assertEqual(self.detector.get_param_value(key), previous)
                self.assertIs(self.detector.model, model)

    def test_saved_settings_discard_only_invalid_portions(self) -> None:
        with patch.object(KoharuDetector.logger, 'warning') as warning:
            detector = KoharuDetector(**{
                'text threshold': {'value': .6}, 'onomatopoeia threshold': float('inf'),
                'mask dilate size': 7, 'removed setting': 12,
                'mask postprocessing': {'value': 'ctd'},
                'label': {'value': {'text': False, 'onomatopoeia': 'false', 'bubble': True}},
            })
        self.assertTrue(warning.called)
        self.assertEqual(detector.get_param_value('text threshold'), .6)
        self.assertEqual(detector.get_param_value('onomatopoeia threshold'), .3)
        self.assertNotIn('mask dilate size', detector.params)
        self.assertNotIn('mask postprocessing', detector.params)
        self.assertEqual(detector.get_param_value('label'), {'text': False, 'onomatopoeia': True})
        self.assertFalse(detector.all_model_loaded())
        self.assertEqual(KoharuDetector().get_param_value('text threshold'), .25)

    def test_reloads_are_lazy_and_only_for_model_or_device_changes(self) -> None:
        model = self.detector.model
        with patch.object(self.detector, 'load_model') as load:
            self.detector.updateParam('device', 'cpu')
            self.detector.updateParam('detect size', 1152)
            self.detector.updateParam('text threshold', '.4')
            self.assertIs(self.detector.model, model)
            self.detector.updateParam('device', 'mps')
            self.assertIsNone(self.detector.model)
            self.detector.model = model
            self.detector.updateParam('model path', 'other/model.safetensors')
            self.assertIsNone(self.detector.model)
            load.assert_not_called()

    def test_loading_is_local_strict_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.safetensors'
            path.touch()
            config_path = path.with_name('config.json')
            config = {'architecture': 'RFDETRSeg2XLarge', 'resolution': 1920, 'num_select': 160,
                      'classes': {'0': 'text', '1': 'onomatopoeia', '2': 'bubble', '3': 'panel'}}
            config_path.write_text(json.dumps(config), encoding='utf-8')
            model = Mock()
            model.to.return_value = model
            model.eval.return_value = model
            factory = Mock(return_value=SimpleNamespace(model=SimpleNamespace(model=model)))
            weights = Mock(return_value={})
            self.detector.updateParam('model path', str(path))
            with patch.dict(sys.modules, {'rfdetr': SimpleNamespace(RFDETRSeg2XLarge=factory),
                                          'safetensors.torch': SimpleNamespace(load_file=weights)}):
                self.detector.load_model()
                self.detector.load_model()
                factory.assert_called_once_with(pretrain_weights=None, resolution=1920,
                    positional_encoding_size=160, num_classes=4, num_select=160, device='cpu')
                weights.assert_called_once_with(str(path))
                model.load_state_dict.assert_called_once_with({}, strict=True)
                self.assertTrue(self.detector.unload_model())
                model.load_state_dict.side_effect = RuntimeError('incompatible weights')
                with self.assertRaisesRegex(RuntimeError, 'incompatible weights'):
                    self.detector.load_model()
                self.assertFalse(self.detector.all_model_loaded())
                config['resolution'] = 1000
                config_path.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'multiple of 24'):
                    self.detector.load_model()

    def test_missing_files_fail_with_setup_message(self) -> None:
        self.detector.updateParam('model path', '/missing/koharu/model.safetensors')
        with self.assertRaisesRegex(FileNotFoundError, 'config.json together'):
            self.detector.load_model()


class KoharuMetadataTests(unittest.TestCase):
    def test_metadata_is_lazy(self) -> None:
        result = subprocess.run([sys.executable, '-c', '''
import sys
from pathlib import Path
from ballontranslator.modules import TEXTDETECTORS
from ballontranslator.modules.lazy_registry import validate_lazy_module_specs
spec = TEXTDETECTORS.get_spec('koharu')
assert spec is not None and not validate_lazy_module_specs([spec])
assert spec.params['label']['value'] == {'text': True, 'onomatopoeia': True}
assert spec.params['detect size']['value'] == 1152
assert 'mask postprocessing' not in spec.params
assert 'mask dilate size' not in spec.params
assert 'transformers>=5.1,<6' in spec.dependencies
download, = spec.download_file_list
assert download['files'] == ['model.safetensors', 'config.json']
assert (Path(download['save_dir']) / download['files'][0]).as_posix() == spec.params['model path']['value']
assert len(download['sha256_pre_calculated']) == 2
assert all(len(digest) == 64 for digest in download['sha256_pre_calculated'])
assert not any(name in sys.modules for name in ('torch', 'transformers', 'rfdetr',
    'ballontranslator.modules.textdetector.detector_koharu'))
'''], capture_output=True, text=True, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_older_config_discards_removed_settings_without_changing_ctd(self) -> None:
        saved = {'ctd': {'mask dilate size': 4}, 'koharu': {
            'mask postprocessing': {'value': 'ctd'}, 'mask dilate size': {'value': 7},
            'text threshold': {'value': .7},
        }}
        with patch('ballontranslator.modules.base.LOGGER.warning') as warning:
            merge_config_module_params(saved, ['ctd', 'koharu'], TEXTDETECTORS.get)
        warning.assert_any_call('Found invalid koharu config: mask postprocessing')
        warning.assert_any_call('Found invalid koharu config: mask dilate size')
        self.assertEqual(saved['ctd']['mask dilate size'], 4)
        self.assertEqual(saved['koharu']['text threshold']['value'], .7)
        self.assertEqual(saved['koharu']['detect size']['value'], 1152)
        self.assertNotIn('mask postprocessing', saved['koharu'])
        self.assertNotIn('mask dilate size', saved['koharu'])

    def test_doctest(self) -> None:
        self.assertEqual(doctest.testmod(detector_koharu).failed, 0)


if __name__ == '__main__':
    unittest.main()
