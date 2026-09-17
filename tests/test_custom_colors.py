import json
import os
import tempfile
import unittest
from unittest.mock import patch

from qtpy.QtGui import QColor

from ballontranslator.ui.mainwindow import (
    _current_custom_colors,
    _restore_custom_colors,
)
from ballontranslator.utils.config import ProgramConfig


class CustomColorPersistenceTests(unittest.TestCase):
    def test_restore_sets_each_saved_color(self) -> None:
        with (
            patch(
                'ballontranslator.ui.mainwindow.QColorDialog.customCount',
                return_value=2,
            ),
            patch(
                'ballontranslator.ui.mainwindow.QColorDialog.setCustomColor'
            ) as set_custom_color,
        ):
            _restore_custom_colors(['#123456', '#abcdef', '#ffffff'])

        self.assertEqual(set_custom_color.call_count, 2)
        self.assertEqual(set_custom_color.call_args_list[0].args[0], 0)
        self.assertEqual(set_custom_color.call_args_list[0].args[1].name(), '#123456')
        self.assertEqual(set_custom_color.call_args_list[1].args[0], 1)
        self.assertEqual(set_custom_color.call_args_list[1].args[1].name(), '#abcdef')

    def test_current_colors_are_serializable_hex_strings(self) -> None:
        with (
            patch(
                'ballontranslator.ui.mainwindow.QColorDialog.customCount',
                return_value=2,
            ),
            patch(
                'ballontranslator.ui.mainwindow.QColorDialog.customColor',
                side_effect=[QColor('#123456'), QColor('#abcdef')],
            ),
        ):
            colors = _current_custom_colors()

        self.assertEqual(colors, ['#123456', '#abcdef'])

    def test_config_load_discards_only_invalid_custom_colors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, 'config.json')
            with open(config_path, 'w', encoding='utf8') as config_file:
                json.dump(
                    {
                        'custom_colors': [
                            '#123456',
                            'invalid',
                            42,
                            '#ABCDEF',
                        ]
                    },
                    config_file,
                )

            config = ProgramConfig.load(config_path)

        self.assertEqual(config.custom_colors, ['#123456', '#ABCDEF'])


if __name__ == '__main__':
    unittest.main()
