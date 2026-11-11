import unittest

from ballontranslator.utils.multi_extract import (
    box_iou,
    conflict_count,
    conflict_payload,
    group_regions,
    has_conflict,
    iter_conflicts,
    merge_passes,
    readings,
    resolve,
)
from ballontranslator.utils.textblock import (
    TextBlock,
    normalize_textblock_effect_payload,
)


def block(x, y, w, h, text, line=None):
    blk = TextBlock(_bounding_rect=[x, y, w, h])
    blk.text = [text] if line is None else list(line)
    return blk


class BoxIoUTest(unittest.TestCase):

    def test_identical_boxes(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_touching_boxes_do_not_overlap(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [10, 0, 10, 10]), 0.0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 10, 10]), 1 / 3)

    def test_contained_box(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [2, 2, 4, 4]), 16 / 100)


class GroupRegionsTest(unittest.TestCase):

    def test_same_region_in_every_pass_forms_one_group(self):
        passes = [[block(0, 0, 10, 10, 'あ')], [block(0, 0, 10, 10, 'あ')], [block(0, 0, 10, 10, 'あ')]]

        self.assertEqual(group_regions(passes), [[(0, 0), (1, 0), (2, 0)]])

    def test_disjoint_regions_stay_separate(self):
        passes = [[block(0, 0, 10, 10, 'あ'), block(80, 0, 10, 10, 'い')], [block(0, 0, 10, 10, 'あ')]]

        self.assertEqual(len(group_regions(passes)), 2)

    def test_threshold_decides_a_weak_overlap(self):
        passes = [[block(0, 0, 10, 10, 'あ')], [block(5, 0, 10, 10, 'あ')]]

        self.assertEqual(len(group_regions(passes, iou_threshold=0.3)), 1)
        self.assertEqual(len(group_regions(passes, iou_threshold=0.5)), 2)

    def test_a_block_matches_at_most_one_block_per_pass(self):
        # The second, weaker pair must not chain through the shared middle pass.
        passes = [
            [block(0, 0, 10, 10, 'あ')],
            [block(0, 0, 12, 10, 'あ'), block(5, 0, 10, 10, 'い')],
        ]

        groups = group_regions(passes)

        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(group) for group in groups), [1, 2])


class MergePassesTest(unittest.TestCase):

    def test_matching_passes_produce_no_conflict(self):
        passes = [[block(0, 0, 10, 10, 'あ')], [block(0, 0, 10, 10, 'あ')]]

        result = merge_passes(passes)

        self.assertEqual([blk.get_text() for blk in result.blocks], ['あ'])
        self.assertEqual(result.text_conflicts, [])
        self.assertEqual(result.weak_regions, [])
        self.assertEqual(result.conflicts[0].support, (0, 1))

    def test_region_only_one_pass_saw_is_kept_and_flagged(self):
        passes = [[block(10, 10, 10, 10, 'あ')], []]

        result = merge_passes(passes)

        self.assertEqual([blk.get_text() for blk in result.blocks], ['あ'])
        self.assertEqual(len(result.weak_regions), 1)
        self.assertTrue(result.weak_regions[0].single_pass)

    def test_disagreeing_text_keeps_reference_and_reports_candidates(self):
        passes = [
            [block(0, 0, 10, 10, 'そういえば')],
            [block(0, 0, 10, 10, 'そういえは')],
            [block(0, 0, 10, 10, 'そういえば')],
        ]

        result = merge_passes(passes)
        conflict = result.conflicts[0]

        self.assertEqual(result.blocks[0].get_text(), 'そういえば')
        self.assertEqual(conflict.chosen_pass, 0)
        self.assertEqual(conflict.texts, ['そういえば', 'そういえは'])
        self.assertEqual([index for index, _ in conflict.candidates], [1, 2])
        self.assertEqual(len(result.text_conflicts), 1)
        self.assertEqual(result.weak_regions, [])

    def test_reference_pass_selects_geometry_and_text(self):
        passes = [
            [block(0, 0, 10, 10, 'あ')],
            [block(0, 0, 11, 11, 'い')],
        ]

        result = merge_passes(passes, reference_pass=1)

        self.assertEqual(result.blocks[0].bounding_rect(), [0, 0, 11, 11])
        self.assertEqual(result.blocks[0].get_text(), 'い')
        self.assertEqual(result.conflicts[0].chosen_pass, 1)
        self.assertEqual(result.conflicts[0].texts, ['い', 'あ'])

    def test_empty_reference_read_falls_back_to_another_pass(self):
        passes = [[block(0, 0, 10, 10, '')], [block(0, 0, 10, 10, 'あ')]]

        result = merge_passes(passes)

        self.assertEqual(result.blocks[0].get_text(), 'あ')
        self.assertEqual(result.text_conflicts, [])
        self.assertEqual(result.conflicts[0].chosen_pass, 0)

    def test_merged_blocks_do_not_share_state_with_the_passes(self):
        passes = [[block(0, 0, 10, 10, 'あ')], [block(0, 0, 10, 10, 'あ')]]

        result = merge_passes(passes)
        result.blocks[0].text = ['changed']

        self.assertEqual(passes[0][0].get_text(), 'あ')

    def test_empty_passes_merge_to_an_empty_page(self):
        self.assertEqual(merge_passes([[], []]).blocks, [])
        self.assertEqual(merge_passes([]).conflicts, [])

    def test_merged_blocks_come_back_in_reading_order(self):
        passes = [[block(0, 0, 10, 10, 'left'), block(200, 0, 10, 10, 'right')]]

        texts = [blk.get_text() for blk in merge_passes(passes).blocks]

        self.assertEqual(sorted(texts), ['left', 'right'])
        self.assertEqual(len(texts), 2)


class ConflictPayloadTest(unittest.TestCase):

    def _payload(self):
        passes = [
            [block(0, 0, 10, 10, 'そういえば')],
            [block(0, 0, 10, 10, 'そういえは')],
            [block(0, 0, 10, 10, 'そういえば')],
        ]
        return conflict_payload(merge_passes(passes).conflicts[0])

    def test_payload_lists_chosen_pass_support_and_candidates(self):
        payload = self._payload()

        self.assertEqual(payload['chosen'], 0)
        self.assertEqual(payload['support'], [0, 1, 2])
        self.assertEqual(
            payload['candidates'],
            [
                {'pass': 1, 'text': ['そういえは']},
                {'pass': 2, 'text': ['そういえば']},
            ],
        )

    def test_payload_survives_a_project_round_trip(self):
        block_with_conflict = block(0, 0, 10, 10, 'そういえば')
        block_with_conflict.multi_extract = self._payload()

        record = normalize_textblock_effect_payload(
            block_with_conflict.to_dict(deep_copy=True)
        )[0]
        reloaded = TextBlock(**record)

        self.assertEqual(reloaded.multi_extract, block_with_conflict.multi_extract)

    def test_blocks_without_a_payload_stay_empty_on_reload(self):
        record = normalize_textblock_effect_payload(
            block(0, 0, 10, 10, 'あ').to_dict(deep_copy=True)
        )[0]

        self.assertEqual(TextBlock(**record).multi_extract, {})


class ResolveTest(unittest.TestCase):

    def _block(self):
        blk = block(0, 0, 10, 10, 'そういえば')
        blk.multi_extract = {
            'chosen': 0,
            'support': [0, 1],
            'candidates': [{'pass': 1, 'text': ['そういえは']}],
        }
        return blk

    def test_readings_list_the_chosen_pass_first(self):
        self.assertEqual(
            readings(self._block()),
            [(0, ['そういえば']), (1, ['そういえは'])],
        )

    def test_choosing_another_pass_replaces_the_source_text(self):
        blk = self._block()

        old_text, old_payload = resolve(blk, 1)

        self.assertEqual(old_text, ['そういえば'])
        self.assertEqual(old_payload['chosen'], 0)
        self.assertEqual(blk.get_text(), 'そういえは')
        self.assertFalse(has_conflict(blk))

    def test_choosing_the_current_pass_only_marks_it_reviewed(self):
        blk = self._block()

        resolve(blk, 0)

        self.assertEqual(blk.get_text(), 'そういえば')
        self.assertFalse(has_conflict(blk))

    def test_unknown_pass_keeps_the_text_but_clears_the_conflict(self):
        blk = self._block()

        resolve(blk, 9)

        self.assertEqual(blk.get_text(), 'そういえば')
        self.assertFalse(has_conflict(blk))

    def test_string_candidates_are_normalised_to_lines(self):
        blk = block(0, 0, 10, 10, 'あ')
        blk.multi_extract = {
            'chosen': 0,
            'candidates': [{'pass': 1, 'text': 'い'}],
        }

        resolve(blk, 1)

        self.assertEqual(blk.text, ['い'])

    def test_resolving_does_not_mutate_the_shared_payload(self):
        blk = self._block()

        _, old_payload = resolve(blk, 1)
        old_payload['candidates'].append({'pass': 5, 'text': ['x']})

        self.assertEqual(len(blk.multi_extract), 0)


class ProjectConflictScanTest(unittest.TestCase):

    def test_counts_and_lists_only_unreviewed_blocks(self):
        pending = block(0, 0, 10, 10, 'あ')
        pending.multi_extract = {'chosen': 0, 'candidates': []}
        pages = {'a.jpg': [block(0, 0, 10, 10, 'あ'), pending], 'b.jpg': []}

        self.assertEqual(conflict_count(pages), 1)
        self.assertEqual(iter_conflicts(pages), [('a.jpg', pending)])


if __name__ == '__main__':
    unittest.main()