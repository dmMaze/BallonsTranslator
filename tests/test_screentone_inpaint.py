import copy
import unittest
from typing import Tuple
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ballontranslator.modules.inpaint.inpaint_default import LamaInpainterMPE, LamaLarge, torch
from ballontranslator.modules.inpaint.screentone import restore_screentone
from ballontranslator.modules.lazy_registry import _scan_file, validate_lazy_module_specs


def _screen(
    diagonal: bool = False, gradient: bool = False, offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices((128, 128))
    if diagonal:
        dots = ((x + y + offset) % 8 < 2) & ((y - x + offset) % 8 < 2)
        mean = 200 - 170 / 16
    else:
        dots = ((x + offset) % 6 < 2) & ((y + offset) % 6 < 2)
        mean = 200 - 170 / 9
    light = 0.55 + 0.4*x/128 if gradient else np.ones_like(x)
    source = np.repeat(np.rint(np.where(dots, 30, 200) * light).astype(np.uint8)[..., None], 3, axis=2)
    base = np.repeat(np.rint(mean * light).astype(np.uint8)[..., None], 3, axis=2)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:90, 45:80] = 255
    damaged = source.copy()
    damaged[mask > 0] = 255
    return source, damaged, mask, base


class ScreentoneTest(unittest.TestCase):
    def test_period_and_phase_are_restored_without_losing_shading(self) -> None:
        for diagonal in (False, True):
            for gradient in (False, True):
                for offset in (0, 3):
                    with self.subTest(diagonal=diagonal, gradient=gradient, offset=offset):
                        source, damaged, mask, base = _screen(diagonal, gradient, offset)
                        restored = restore_screentone(damaged, mask, base)
                        error = np.abs(restored.astype(float) - source)[mask > 0].mean()
                        self.assertLess(error, 5)
                        np.testing.assert_array_equal(restored[mask == 0], base[mask == 0])

    def test_dense_glyph_masks_receive_texture_not_just_fallback(self) -> None:
        y, x = np.indices((619, 174))
        gray = 72 + 45*np.cos(np.pi*(x+y)/2) + 38*np.sin(np.pi*(x-y)/2)
        source = np.repeat(np.clip(np.rint(gray), 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for left in (8, 62, 116):
            for row, char in enumerate('TEXTMASKDATA'):
                cv2.putText(mask, char, (left, 43 + row*51),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 7, cv2.LINE_AA)
        mask = cv2.dilate((mask > 0).astype(np.uint8)*255, np.ones((3, 3), np.uint8))
        damaged = source.copy()
        damaged[mask > 0] = 255
        base = np.full_like(source, 72)
        restored = restore_screentone(damaged, mask, base)
        before = np.abs(base.astype(float) - source)[mask > 0].mean()
        after = np.abs(restored.astype(float) - source)[mask > 0].mean()
        self.assertLess(after, before * 0.3)
        np.testing.assert_array_equal(restored[mask == 0], base[mask == 0])

    def test_masked_content_is_never_a_donor_and_inputs_are_unchanged(self) -> None:
        _, damaged, mask, base = _screen()
        original_image, original_mask, original_base = damaged.copy(), mask.copy(), base.copy()
        first = restore_screentone(damaged, mask, base)
        np.testing.assert_array_equal(damaged, original_image)
        np.testing.assert_array_equal(mask, original_mask)
        np.testing.assert_array_equal(base, original_base)
        damaged[mask > 0] = (0, 150, 50)
        np.testing.assert_array_equal(first, restore_screentone(damaged, mask, base))

    def test_reconstructed_black_linework_is_not_replaced_by_dots(self) -> None:
        _, damaged, mask, base = _screen(gradient=True)
        base[60:64, :] = 0
        restored = restore_screentone(damaged, mask, base)
        np.testing.assert_array_equal(restored[60:64], base[60:64])

    def test_generated_brightness_blobs_do_not_modulate_the_restored_screen(self) -> None:
        source, damaged, mask, base = _screen(gradient=True)
        y, x = np.indices(mask.shape)
        # A smooth model artifact must not change the source tone's contrast.
        blob = 18*np.exp(-((x-63)**2+(y-63)**2)/220)
        base = np.clip(base.astype(float)+blob[..., None], 0, 255).astype(np.uint8)
        restored = restore_screentone(damaged, mask, base)
        error = np.abs(restored.astype(float)-source).mean(axis=2)
        boundary = (mask > 0) & (cv2.erode(mask, np.ones((3, 3), np.uint8)) == 0)
        self.assertLess(error[mask > 0].mean(), 5)
        self.assertLess(error[boundary].mean(), 5)
        np.testing.assert_array_equal(restored[mask == 0], base[mask == 0])

    def test_nonperiodic_and_insufficient_evidence_keep_normal_result(self) -> None:
        source, _, mask, base = _screen()
        rng = np.random.default_rng(3)
        noise = np.repeat(rng.integers(0, 256, mask.shape, dtype=np.uint8)[..., None], 3, axis=2)
        for image in (np.full(noise.shape, 127, np.uint8), noise):
            self.assertIs(restore_screentone(image, mask, base), base)
        for fill in (0, 255):
            self.assertIs(restore_screentone(source, np.full(mask.shape, fill, np.uint8), base), base)
        mask[:] = 255
        mask[:2] = 0
        self.assertIs(restore_screentone(source, mask, base), base)

    def test_color_screens_are_not_treated_as_monochrome(self) -> None:
        _, damaged, mask, base = _screen()
        damaged[:, :, 0] = 0
        self.assertIs(restore_screentone(damaged, mask, base), base)

    def test_solid_region_keeps_the_normal_result(self) -> None:
        source, _, mask, base = _screen()
        source[28:105, 30:95] = 127
        source[mask > 0] = 255
        base[:] = 127
        restored = restore_screentone(source, mask, base)
        np.testing.assert_array_equal(restored[52:78, 54:71], base[52:78, 54:71])

    def test_single_line_is_not_a_repeating_screen(self) -> None:
        _, _, mask, base = _screen()
        source = np.full((128, 128, 3), 192, np.uint8)
        cv2.line(source, (0, 30), (127, 80), (0, 0, 0), 3)
        source[mask > 0] = 255
        self.assertIs(restore_screentone(source, mask, base), base)

    def test_invalid_inputs_are_rejected(self) -> None:
        _, damaged, mask, base = _screen()
        for image, invalid_mask, result in ((damaged.astype(float), mask, base),
                                           (damaged, mask[:-1], base),
                                           (damaged, mask.astype(float), base),
                                           (damaged, mask, base[:-1]),
                                           (damaged, mask, base.astype(float))):
            with self.assertRaises(ValueError):
                restore_screentone(image, invalid_mask, result)

    def test_option_metadata_is_lazy_and_disabled_by_default(self) -> None:
        specs = _scan_file('ballontranslator/modules/inpaint/inpaint_default.py', 'inpainter')
        for spec in specs:
            if spec.key in ('lama_mpe', 'lama_large_512px'):
                option = spec.params['preserve screentones']
                self.assertEqual(option['type'], 'checkbox')
                self.assertIs(option['value'], False)
                self.assertEqual(validate_lazy_module_specs([spec]), [])


@unittest.skipIf(torch is None, 'torch is not installed')
class ScreentoneIntegrationTest(unittest.TestCase):
    def test_one_normal_inference_is_followed_by_masked_texture_restoration(self) -> None:
        for cls in (LamaInpainterMPE, LamaLarge):
            with self.subTest(model=cls.__name__), patch.object(cls, 'params', copy.deepcopy(cls.params)):
                inpainter = cls(**copy.deepcopy(cls.params))
                inpainter.device, inpainter.precision = 'cpu', 'fp32'
                source, damaged, mask, base = _screen()
                model = Mock()
                model.load_masked_position_encoding.return_value = (
                    np.zeros(mask.shape, np.int32), np.zeros(mask.shape, np.int32),
                    np.zeros(mask.shape + (4,), np.int32),
                )
                model.return_value = torch.from_numpy(base).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                inpainter.model = model
                results, inputs = [], []
                for enabled in (False, True):
                    inpainter.set_param_value('preserve screentones', enabled)
                    results.append(inpainter._inpaint(damaged, mask))
                    inputs.append(model.call_args.args[0].clone())
                self.assertEqual(model.call_count, 2)
                self.assertTrue(torch.equal(inputs[0], inputs[1]))
                np.testing.assert_array_equal(results[1][mask == 0], damaged[mask == 0])
                before = np.abs(results[0].astype(float) - source)[mask > 0].mean()
                after = np.abs(results[1].astype(float) - source)[mask > 0].mean()
                self.assertLess(after, before * 0.3)

    def test_disabled_or_unrecognized_pattern_keeps_normal_path(self) -> None:
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(LamaLarge, 'params', copy.deepcopy(LamaLarge.params)):
                inpainter = LamaLarge(**copy.deepcopy(LamaLarge.params))
                inpainter.device, inpainter.precision = 'cpu', 'fp32'
                inpainter.set_param_value('preserve screentones', enabled)
                image = np.full((64, 64, 3), 127, np.uint8)
                mask = np.zeros((64, 64), np.uint8)
                mask[20:40, 20:40] = 255
                model = Mock()
                model.load_masked_position_encoding.return_value = (
                    np.zeros(mask.shape, np.int32), np.zeros(mask.shape, np.int32),
                    np.zeros(mask.shape + (4,), np.int32),
                )
                model.return_value = torch.full((1, 3, 64, 64), 0.5)
                inpainter.model = model
                with patch('ballontranslator.modules.inpaint.screentone.restore_screentone', wraps=restore_screentone) as restore:
                    result = inpainter._inpaint(image, mask)
                self.assertEqual(restore.call_count, int(enabled))
                np.testing.assert_array_equal(result[mask > 0], 128)
                np.testing.assert_array_equal(result[mask == 0], image[mask == 0])


if __name__ == '__main__':
    unittest.main()
