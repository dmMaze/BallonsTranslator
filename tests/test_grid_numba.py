import os
import os.path as osp
import unittest

import numpy as np
from qtpy.QtCore import QRectF

from ballontranslator.ui.text_engine.transforms.grid import GridMapper
from ballontranslator.ui.text_engine.transforms.grid_numba import (
    NUMBA_CACHE_DIR,
    inverse_grid_arrays,
    warm_grid_numba_cache,
)
from ballontranslator.utils import shared
from ballontranslator.utils.fontformat import GridTextTransform


class GridNumbaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warm_grid_numba_cache()

    def test_cache_uses_persistent_app_directory(self):
        expected = osp.join(shared.cache_dir, 'numba')
        self.assertEqual(NUMBA_CACHE_DIR, expected)
        self.assertEqual(os.environ['NUMBA_CACHE_DIR'], expected)

    def test_warmed_kernel_preserves_padded_source_pixels(self):
        logical = QRectF(0, 0, 100, 50)
        source = logical.adjusted(-10, -5, 10, 5)
        visual_x = np.asarray([[-5.0, 50.0, 105.0]], dtype=np.float32)
        visual_y = np.asarray([[-2.0, 25.0, 52.0]], dtype=np.float32)
        for interpolation in ('bilinear', 'catmull_rom'):
            with self.subTest(interpolation=interpolation):
                mapper = GridMapper(
                    logical,
                    source,
                    GridTextTransform(1, 1, interpolation),
                )
                restored_x, restored_y, valid = mapper.inverse_arrays(
                    visual_x, visual_y, return_valid=True
                )
                self.assertTrue(valid.all())
                self.assertTrue(np.allclose(restored_x, visual_x))
                self.assertTrue(np.allclose(restored_y, visual_y))

    def test_warmup_makes_both_signatures_available(self):
        coordinates = np.full((1, 1), 0.5, dtype=np.float32)
        linear_points = np.asarray(
            (((0.0, 0.0), (1.0, 0.0)),
             ((0.0, 1.0), (1.0, 1.0))),
            dtype=np.float32,
        )
        self.assertIsNotNone(inverse_grid_arrays(
            linear_points,
            1,
            1,
            coordinates,
            coordinates,
            catmull_rom=False,
        ))

    def test_bilinear_cell_boundaries_do_not_leave_inverse_tears(self):
        points = (
            (0.1122, 0.2059), (0.7473, 0.0404), (1.0360, 0.2975),
            (-0.2799, 0.4329), (0.3352, 0.2276), (0.7989, 0.5373),
            (-0.2889, 1.2479), (0.3509, 1.2040), (0.6679, 0.9692),
        )
        mapper = GridMapper(
            QRectF(0, 0, 1000, 500),
            QRectF(0, 0, 1000, 500),
            GridTextTransform(2, 2, 'bilinear', points),
        )
        axis = np.linspace(0.25, 0.75, 81, dtype=np.float32)
        source_x, source_y = np.meshgrid(axis * 1000, axis * 500)
        visual_x, visual_y = mapper.forward_arrays(source_x, source_y)
        restored_x, restored_y, valid = mapper.inverse_arrays(
            visual_x, visual_y, return_valid=True
        )

        self.assertTrue(valid.all())
        self.assertLess(
            float(np.max(np.hypot(
                restored_x - source_x,
                restored_y - source_y,
            ))),
            0.02,
        )


if __name__ == '__main__':
    unittest.main()
