import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from ballontranslator.ui.text_engine.editing.drop_sync import (
    changed_range,
    utf16_length,
)


class ChangedRangeTest(unittest.TestCase):
    def test_insertion(self) -> None:
        self.assertEqual(changed_range('abcd', 'abXYcd'), (2, 0, 2))

    def test_deletion(self) -> None:
        self.assertEqual(changed_range('abcd', 'ad'), (1, 2, 0))

    def test_replacement(self) -> None:
        self.assertEqual(changed_range('one two', 'one six'), (4, 3, 3))

    def test_move_inside_the_same_text_is_one_range(self) -> None:
        # Dragging "ab" to the end removes it at the start and inserts it
        # at the end; one range spanning both describes the change.
        self.assertEqual(changed_range('abcd', 'cdab'), (0, 4, 4))

    def test_unchanged(self) -> None:
        self.assertEqual(changed_range('same', 'same'), (4, 0, 0))

    def test_positions_count_utf16_units(self) -> None:
        # An emoji is one Python character but two Qt positions.
        self.assertEqual(utf16_length('\U0001f600a'), 3)
        self.assertEqual(
            changed_range('\U0001f600ab', '\U0001f600aXb'), (3, 0, 1)
        )


if __name__ == '__main__':
    unittest.main()
