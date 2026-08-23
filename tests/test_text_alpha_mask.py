import copy
from dataclasses import FrozenInstanceError
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.proj_imgtrans import ProjImgTrans, TextBlkEncoder
from ballontranslator.utils.text_alpha_mask import (
    AlphaBrushStroke,
    TextAlphaMask,
    load_text_alpha_mask,
    simplify_alpha_brush_points,
)
from ballontranslator.utils.textblock import TextBlock


class TextAlphaMaskDomainTest(unittest.TestCase):
    def test_deterministic_simplification_preserves_visible_turns_and_endpoints(self):
        points = ((0, 0), (1, 0.01), (2, 0), (2, 2), (3, 2))
        simplified = simplify_alpha_brush_points(points)
        self.assertEqual(simplified[0], (0.0, 0.0))
        self.assertEqual(simplified[-1], (3.0, 2.0))
        self.assertIn((2.0, 0.0), simplified)
        self.assertEqual(simplified, simplify_alpha_brush_points(points))

    def test_live_values_are_typed_immutable_and_strict(self):
        stroke = AlphaBrushStroke('erase', 12, ((-3, 4), (20, 30)))
        mask = TextAlphaMask(strokes=(stroke,))

        self.assertEqual(stroke.points, ((-3.0, 4.0), (20.0, 30.0)))
        self.assertFalse(mask.is_neutral())
        self.assertRaises(FrozenInstanceError, setattr, stroke, 'diameter', 4)
        self.assertRaises(FrozenInstanceError, setattr, mask, 'enabled', False)
        for args in (
            ('paint', 3, ((0, 0),)),
            ('erase', 0, ((0, 0),)),
            ('erase', float('inf'), ((0, 0),)),
            ('restore', 3, ()),
            ('restore', 3, ((0, float('nan')),)),
        ):
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    AlphaBrushStroke(*args)
        with self.assertRaises(ValueError):
            TextAlphaMask(version=2)
        with self.assertRaises(ValueError):
            TextAlphaMask(version=1.0)
        with self.assertRaises(TypeError):
            TextAlphaMask(enabled=1)
        with self.assertRaises(TypeError):
            TextAlphaMask(strokes=({},))

        self.assertTrue(TextAlphaMask().is_neutral())
        self.assertTrue(TextAlphaMask(enabled=False, strokes=(stroke,)).is_neutral())

    def test_permissive_loader_isolates_bad_strokes_and_points(self):
        payload = {
            'version': 1,
            'enabled': True,
            'removed_field': 'ignored',
            'strokes': [
                {
                    'mode': 'erase',
                    'diameter': 4,
                    'points': [[1, 2], ['bad'], [3, 4]],
                    'old_field': 1,
                },
                {'mode': 'erase', 'diameter': -1, 'points': [[5, 6]]},
                {'mode': 'restore', 'diameter': 2, 'points': [[7, 8]]},
                'broken',
            ],
        }
        with patch(
            'ballontranslator.utils.text_alpha_mask.LOGGER.warning'
        ) as warning:
            loaded = load_text_alpha_mask(payload)

        self.assertEqual(len(loaded.strokes), 2)
        self.assertEqual(loaded.strokes[0].points, ((1.0, 2.0), (3.0, 4.0)))
        self.assertEqual(loaded.strokes[1].mode, 'restore')
        self.assertGreaterEqual(warning.call_count, 4)

        invalid_payloads = (
            [],
            {'version': 99, 'enabled': True, 'strokes': []},
            {'version': 1.0, 'enabled': True, 'strokes': []},
        )
        with patch('ballontranslator.utils.text_alpha_mask.LOGGER.warning'):
            for invalid in invalid_payloads:
                with self.subTest(invalid=invalid):
                    self.assertIsNone(load_text_alpha_mask(invalid))
        with patch('ballontranslator.utils.text_alpha_mask.LOGGER.warning'):
            recovered = load_text_alpha_mask({
                'version': 1,
                'enabled': 'yes',
                'strokes': {},
            })
        self.assertEqual(recovered, TextAlphaMask())

    def test_stable_json_project_round_trip_and_old_project_default(self):
        mask = TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 6.5, ((-2, 1), (3, 4))),
            AlphaBrushStroke('restore', 2, ((0, 0),)),
        ))
        block = TextBlock(text_alpha_mask=mask)
        serialized = json.dumps(block, cls=TextBlkEncoder)
        payload = json.loads(serialized)

        self.assertEqual(payload['text_alpha_mask'], {
            'version': 1,
            'enabled': True,
            'strokes': [
                {
                    'mode': 'erase',
                    'diameter': 6.5,
                    'points': [[-2.0, 1.0], [3.0, 4.0]],
                },
                {
                    'mode': 'restore',
                    'diameter': 2.0,
                    'points': [[0.0, 0.0]],
                },
            ],
        })
        self.assertEqual(TextBlock(**payload).text_alpha_mask, mask)
        self.assertNotIn('_mask_generation', serialized)
        self.assertNotIn('render_scale', serialized)
        self.assertNotIn('cache_key', serialized)
        payload.pop('text_alpha_mask')
        self.assertIsNone(TextBlock(**payload).text_alpha_mask)

        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            valid_record = json.loads(json.dumps(block, cls=TextBlkEncoder))
            invalid_record = dict(valid_record, translation='still loads')
            invalid_record['text_alpha_mask'] = {
                'version': 999,
                'enabled': True,
                'strokes': [],
            }
            with patch(
                'ballontranslator.utils.text_alpha_mask.LOGGER.warning'
            ):
                project.load_from_dict({
                    'pages': {
                        'missing.png': [valid_record, invalid_record]
                    },
                    'image_info': {},
                })
            self.assertEqual(
                project.not_found_pages['missing.png'][0].text_alpha_mask,
                mask,
            )
            self.assertEqual(len(project.not_found_pages['missing.png']), 2)
            self.assertEqual(
                project.not_found_pages['missing.png'][1].translation,
                'still loads',
            )
            self.assertIsNone(
                project.not_found_pages['missing.png'][1].text_alpha_mask
            )

    def test_copy_duplicate_retains_mask_but_styles_do_not_own_it(self):
        mask = TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 5, ((1, 2),)),
        ))
        block = TextBlock(text_alpha_mask=mask)

        self.assertIs(copy.copy(block).text_alpha_mask, mask)
        self.assertEqual(copy.deepcopy(block).text_alpha_mask, mask)
        style = block.fontformat.deepcopy()
        self.assertIsInstance(style, FontFormat)
        self.assertFalse(hasattr(style, 'text_alpha_mask'))
        self.assertNotIn(
            'text_alpha_mask', json.loads(json.dumps(
                style.to_serializable_dict()
            )),
        )

    def test_textblock_load_discards_only_invalid_optional_mask(self):
        with patch('ballontranslator.utils.text_alpha_mask.LOGGER.warning'):
            block = TextBlock(
                translation='survives',
                text_alpha_mask={'version': 999, 'strokes': []},
            )
        self.assertEqual(block.translation, 'survives')
        self.assertIsNone(block.text_alpha_mask)


if __name__ == '__main__':
    unittest.main()
