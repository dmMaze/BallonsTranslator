"""The auto font size is capped against the page median."""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.pipeline_formatting import (
    cap_font_size_outliers,
)
from ballontranslator.utils.textblock import TextBlock


def blocks_with(sizes) -> list:
    blocks = []
    for size in sizes:
        block = TextBlock(translation='x')
        block.fontformat.font_size = float(size)
        blocks.append(block)
    return blocks


class FontSizeCapTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_outlier_is_pulled_down_to_the_limit(self) -> None:
        blocks = blocks_with((50.0, 55.0, 60.0, 400.0))
        self.assertEqual(cap_font_size_outliers(blocks, 3.0), 1)
        self.assertAlmostEqual(blocks[-1].fontformat.font_size, 57.5 * 3.0)

    def test_normal_sizes_are_untouched(self) -> None:
        blocks = blocks_with((50.0, 55.0, 60.0))
        self.assertEqual(cap_font_size_outliers(blocks, 3.0), 0)
        self.assertEqual(
            [block.fontformat.font_size for block in blocks],
            [50.0, 55.0, 60.0],
        )

    def test_zero_ratio_disables_the_cap(self) -> None:
        blocks = blocks_with((50.0, 55.0, 400.0))
        self.assertEqual(cap_font_size_outliers(blocks, 0), 0)
        self.assertAlmostEqual(blocks[-1].fontformat.font_size, 400.0)

    def test_two_blocks_are_not_enough_for_a_median(self) -> None:
        blocks = blocks_with((50.0, 400.0))
        self.assertEqual(cap_font_size_outliers(blocks, 3.0), 0)


if __name__ == '__main__':
    unittest.main()
