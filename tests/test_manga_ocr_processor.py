import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from ballontranslator.modules.ocr import ocr_manga


class MangaOCRProcessorTests(unittest.TestCase):
    def test_legacy_checkpoint_preprocessing_and_recognition(self) -> None:
        # Existing MangaOCR downloads describe the old ViTFeatureExtractor.
        config = {'feature_extractor_type': 'ViTFeatureExtractor', 'do_resize': True,
                  'size': 224, 'resample': 2, 'do_normalize': True,
                  'image_mean': [.5, .5, .5], 'image_std': [.5, .5, .5]}
        image = np.random.default_rng(42).integers(0, 256, (75, 39, 3), dtype=np.uint8)
        model = Mock(device='cpu')
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        tokenizer = Mock()
        tokenizer.decode.return_value = 'こ ん に ち は'
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'preprocessor_config.json').write_text(json.dumps(config), encoding='utf-8')
            with patch.object(ocr_manga.AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
                    patch.object(ocr_manga.VisionEncoderDecoderModel, 'from_pretrained', return_value=model):
                ocr = ocr_manga.MangaOcr(tmp)
                self.assertEqual(ocr(image), 'こんにちは')
        # Match the legacy PIL path; recent ViTImageProcessor defaults to torchvision.
        resized = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR))
        normalized = (resized.astype(np.float64) / 255).astype(np.float32)
        expected = torch.from_numpy(((normalized - .5) / .5).transpose(2, 0, 1)[None])
        torch.testing.assert_close(model.generate.call_args.args[0], expected, rtol=0, atol=0)
        tokenizer.decode.assert_called_once()
        model.to.assert_called_once_with('cpu')


if __name__ == '__main__':
    unittest.main()
