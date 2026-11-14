import itertools
import unittest

import torch
from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor

from ballontranslator.modules.ocr.ocr_manga import VectorizedNoRepeatNGramLogitsProcessor


class VectorizedNoRepeatNGramTest(unittest.TestCase):
    """The vectorized processor must return exactly what transformers' own processor returns."""

    def assert_same(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, ngram_size: int) -> None:
        expected = NoRepeatNGramLogitsProcessor(ngram_size)(input_ids, scores.clone())
        actual = VectorizedNoRepeatNGramLogitsProcessor(ngram_size)(input_ids, scores.clone())
        self.assertTrue(torch.equal(actual, expected), f'ngram_size={ngram_size}, shape={tuple(input_ids.shape)}')

    def test_all_short_sequences(self) -> None:
        # Every sequence over a 3-token vocabulary up to length 7, as one batch per length.
        generator = torch.Generator().manual_seed(0)
        for ngram_size in range(1, 5):
            for length in range(1, 8):
                input_ids = torch.tensor(list(itertools.product(range(3), repeat=length)))
                scores = torch.randn(input_ids.shape[0], 5, generator=generator)  # vocab wider than the ids used
                self.assert_same(input_ids, scores, ngram_size)

    def test_manga_ocr_sized_inputs(self) -> None:
        # manga-ocr's vocabulary and no_repeat_ngram_size=3, beam-search-sized batches, repetitive ids.
        generator = torch.Generator().manual_seed(1)
        for _ in range(200):
            batch, length = int(torch.randint(1, 33, (1,))), int(torch.randint(1, 80, (1,)))
            alphabet = int(torch.randint(2, 40, (1,)))
            input_ids = torch.randint(0, alphabet, (batch, length), generator=generator)
            scores = torch.randn(batch, 6144, generator=generator)
            self.assert_same(input_ids, scores, 3)


if __name__ == '__main__':
    unittest.main()
