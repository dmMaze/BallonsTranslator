import copy
import unittest
from typing import Tuple, Type
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ballontranslator.modules.inpaint.inpaint_default import (
    LamaInpainterMPE,
    LamaLarge,
    torch,
)
from ballontranslator.utils.imgproc_utils import resize_keepasp


def _position_encoding(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    positions = np.zeros((height, width), dtype=np.int32)
    directions = np.zeros((height, width, 4), dtype=np.int32)
    return positions, positions, directions


def _pattern(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width))
    dots = ((x % 5 < 2) & (y % 5 < 2)) * 255
    return np.stack((dots, x % 256, y % 256), axis=2).astype(np.uint8)


@unittest.skipIf(torch is None, 'torch is not installed')
class LamaPaddingTest(unittest.TestCase):
    def _inpainter(
        self, cls: Type[LamaInpainterMPE], limit: int = 1024,
    ) -> LamaInpainterMPE:
        with patch.object(cls, 'params', copy.deepcopy(cls.params)):
            inpainter = cls(**copy.deepcopy(cls.params))
        inpainter.device = 'cpu'
        inpainter.precision = 'fp32'
        inpainter.inpaint_size = limit
        inpainter.model = Mock()
        inpainter.model.load_masked_position_encoding.side_effect = (
            _position_encoding
        )
        return inpainter

    def test_stride_padding_preserves_image_pixels_and_mask_coordinates(self) -> None:
        for cls in (LamaInpainterMPE, LamaLarge):
            for height, width in ((786, 172), (172, 786), (65, 65), (1, 7)):
                with self.subTest(model=cls.__name__, shape=(height, width)):
                    inpainter = self._inpainter(cls)
                    image = _pattern(height, width)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[:, width // 2] = 255
                    mask[-1, -1] = 255
                    original_image, original_mask = image.copy(), mask.copy()

                    pixels, masked, _, _, _, _, bottom, right = (
                        inpainter.inpaint_preprocess(image, mask)
                    )

                    side = (max(height, width) + 63) // 64 * 64
                    self.assertEqual(tuple(pixels.shape), (1, 3, side, side))
                    self.assertEqual((side - bottom, side - right), (height, width))
                    actual = pixels[0, :, :height, :width].permute(1, 2, 0).numpy()
                    expected = image.astype(np.float32) / 255.0
                    expected[mask > 0] = 0
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(
                        masked[0, 0, :height, :width].numpy(), mask / 255.0
                    )
                    np.testing.assert_array_equal(image, original_image)
                    np.testing.assert_array_equal(mask, original_mask)

    def test_aligned_input_keeps_existing_reflection_padding(self) -> None:
        for cls in (LamaInpainterMPE, LamaLarge):
            with self.subTest(model=cls.__name__):
                inpainter = self._inpainter(cls)
                image = _pattern(64, 128)
                mask = np.zeros((64, 128), dtype=np.uint8)
                pixels, _, _, _, _, _, bottom, right = (
                    inpainter.inpaint_preprocess(image, mask)
                )
                expected = cv2.copyMakeBorder(
                    image, 0, 64, 0, 0, cv2.BORDER_REFLECT
                ).astype(np.float32) / 255.0
                np.testing.assert_array_equal(
                    pixels[0].permute(1, 2, 0).numpy(), expected
                )
                self.assertEqual((bottom, right), (64, 0))

    def test_size_limit_still_downscales_before_padding(self) -> None:
        for cls in (LamaInpainterMPE, LamaLarge):
            for height, width in ((157, 91), (91, 157)):
                with self.subTest(model=cls.__name__, shape=(height, width)):
                    inpainter = self._inpainter(cls, limit=128)
                    image = _pattern(height, width)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[5:height // 2, 7:width // 2] = 255
                    expected_image = resize_keepasp(image, 128, stride=None)
                    expected_mask = resize_keepasp(mask, 128, stride=None) >= 128
                    scaled_height, scaled_width = expected_image.shape[:2]

                    pixels, masked, _, _, _, _, bottom, right = (
                        inpainter.inpaint_preprocess(image, mask)
                    )

                    self.assertEqual((128 - bottom, 128 - right),
                                     (scaled_height, scaled_width))
                    expected = expected_image.astype(np.float32) / 255.0
                    expected[expected_mask] = 0
                    np.testing.assert_array_equal(
                        pixels[0, :, :scaled_height, :scaled_width]
                        .permute(1, 2, 0).numpy(), expected
                    )
                    np.testing.assert_array_equal(
                        masked[0, 0, :scaled_height, :scaled_width].numpy(),
                        expected_mask,
                    )

    def test_output_removes_padding_before_mask_compositing(self) -> None:
        for cls in (LamaInpainterMPE, LamaLarge):
            for height, width in ((79, 65), (65, 79), (128, 128)):
                with self.subTest(model=cls.__name__, shape=(height, width)):
                    inpainter = self._inpainter(cls)
                    image = np.full((height, width, 3), 127, dtype=np.uint8)
                    generated = _pattern(height, width)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[1:-1, 1:-1] = 255
                    side = (max(height, width) + 63) // 64 * 64
                    padded = cv2.copyMakeBorder(
                        generated, 0, side - height, 0, side - width,
                        cv2.BORDER_REFLECT,
                    )
                    inpainter.model.return_value = torch.from_numpy(
                        padded
                    ).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                    result = inpainter._inpaint(image, mask)

                    self.assertEqual(result.shape, image.shape)
                    self.assertEqual(result.dtype, np.uint8)
                    np.testing.assert_array_equal(result[mask > 0], generated[mask > 0])
                    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])


if __name__ == '__main__':
    unittest.main()
