import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from safetensors.torch import save_file
import torch
from transformers import PaddleOCRVLConfig, PaddleOCRVLForConditionalGeneration

from ballontranslator.modules.ocr import ocr_paddleVL_manga


class PaddleOCRTransformersTests(unittest.TestCase):
    def test_legacy_checkpoint_preserves_weights_and_position_embeddings(self) -> None:
        # The downloaded checkpoint has flat text config and independent output weights.
        config = {
            'model_type': 'paddleocr_vl', 'tie_word_embeddings': False,
            'vocab_size': 16, 'hidden_size': 32, 'intermediate_size': 64,
            'num_hidden_layers': 1, 'num_attention_heads': 4,
            'num_key_value_heads': 2, 'head_dim': 8,
            'image_token_id': 13, 'vision_start_token_id': 12, 'vision_end_token_id': 14,
            'rope_scaling': {'rope_type': 'default', 'mrope_section': [1, 1, 2]},
            'vision_config': {
                'model_type': 'paddleocr_vl', 'hidden_size': 32,
                'intermediate_size': 64, 'num_hidden_layers': 1,
                'num_attention_heads': 4, 'image_size': 8, 'patch_size': 2,
            },
        }
        native_config = PaddleOCRVLConfig(**copy.deepcopy(config))
        native_config.tie_word_embeddings = False
        original = PaddleOCRVLForConditionalGeneration(native_config)
        embeddings = original.model.visual.vision_model.embeddings
        with torch.no_grad():
            original.get_input_embeddings().weight.fill_(1)
            original.get_output_embeddings().weight.fill_(2)
            embeddings.patch_embedding.weight.zero_()
            embeddings.patch_embedding.bias.zero_()
            positions = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32).square() / 512
            embeddings.position_embedding.weight.copy_(positions)
        legacy_weights = {
            key.replace('model.visual.', 'visual.')
               .replace('model.projector.', 'mlp_AR.')
               .replace('model.language_model.', 'model.'): tensor
            for key, tensor in original.state_dict().items()
        }
        processor = Mock(return_value={
            'input_ids': torch.tensor([[1, 12, 13, 13, 13, 13, 14, 2]]),
            'attention_mask': torch.ones(1, 8, dtype=torch.long),
            'pixel_values': torch.zeros(16, 3, 2, 2),
            'image_grid_thw': torch.tensor([[1, 4, 4]]),
        })
        processor.tokenizer.eos_token_id = 2
        processor.batch_decode.return_value = ['recognized']
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'config.json').write_text(json.dumps(config), encoding='utf-8')
            save_file(legacy_weights, str(Path(tmp, 'model.safetensors')))
            with patch.object(ocr_paddleVL_manga, 'MODEL_PATH', tmp), \
                    patch.object(ocr_paddleVL_manga.AutoProcessor, 'from_pretrained', return_value=processor):
                ocr = ocr_paddleVL_manga.PaddleOCRVLManga(device='cpu', max_new_tokens=2)
                self.addCleanup(ocr.unload_model)
                ocr.load_model()

        self.assertFalse(ocr.model.training)
        self.assertEqual(ocr.model.dtype, torch.float32)
        self.assertIsNot(ocr.model.get_input_embeddings().weight, ocr.model.get_output_embeddings().weight)
        loaded_weights = ocr.model.state_dict()
        for key, expected in original.state_dict().items():
            torch.testing.assert_close(loaded_weights[key], expected, rtol=0, atol=0)
        # A non-square image grid must retain the legacy bilinear interpolation.
        actual = ocr.model.model.visual.vision_model.embeddings(
            torch.zeros(1, 6, 3, 2, 2), torch.tensor([[1, 2, 3]]),
        )
        expected = torch.nn.functional.interpolate(
            positions.T.reshape(1, 32, 4, 4), size=(2, 3), mode='bilinear', align_corners=False,
        ).flatten(2).squeeze(0).T
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)

        captured_positions = []

        def capture_positions(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
            captured_positions.append(kwargs['position_ids'].clone())

        hook = ocr.model.model.language_model.register_forward_pre_hook(capture_positions, with_kwargs=True)
        self.addCleanup(hook.remove)
        self.assertEqual(ocr.ocr_img(np.zeros((8, 8, 3), dtype=np.uint8)), ['recognized'])
        # Four image tokens occupy a 2x2 spatial grid, not four successive text positions.
        expected_positions = torch.tensor([
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            [[0, 1, 2, 2, 2, 2, 4, 5]],
            [[0, 1, 2, 2, 3, 3, 4, 5]],
            [[0, 1, 2, 3, 2, 3, 4, 5]],
        ])
        torch.testing.assert_close(captured_positions[0], expected_positions)
        torch.testing.assert_close(captured_positions[1], torch.tensor([[[8]], [[6]], [[6]], [[6]]]))


if __name__ == '__main__':
    unittest.main()
