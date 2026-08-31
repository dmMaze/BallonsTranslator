import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ballontranslator.utils.io_utils import imread


class ImageIOTest(unittest.TestCase):
    def test_imread_normalizes_16_bit_grayscale_png_to_uint8_rgb(self):
        source = np.array(
            [[0, 256], [32768, 65535]],
            dtype=np.uint16,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'gray16.png'
            Image.fromarray(source).save(image_path)

            loaded = imread(str(image_path))

        self.assertEqual(loaded.dtype, np.uint8)
        self.assertEqual(loaded.shape, (2, 2, 3))
        np.testing.assert_array_equal(
            loaded[:, :, 0],
            np.array([[0, 1], [128, 255]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(loaded[:, :, 0], loaded[:, :, 1])
        np.testing.assert_array_equal(loaded[:, :, 1], loaded[:, :, 2])


if __name__ == '__main__':
    unittest.main()
