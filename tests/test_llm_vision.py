import base64
import unittest
from unittest import mock

import cv2
import numpy as np

from ballontranslator.modules.llm_vision import encode_chat_image


class LLMVisionEncodingTest(unittest.TestCase):
    def test_rgb_and_rgba_are_reordered_without_mutating_source(self):
        cases = (
            (np.array([[[10, 20, 30]]], dtype=np.uint8), [30, 20, 10]),
            (np.array([[[10, 20, 30, 40]]], dtype=np.uint8), [30, 20, 10, 40]),
        )
        for image, expected in cases:
            with self.subTest(channels=image.shape[-1]):
                original = image.copy()
                encoded = np.frombuffer(b'jpeg', dtype=np.uint8)
                with mock.patch(
                    'ballontranslator.modules.llm_vision.cv2.imencode',
                    return_value=(True, encoded),
                ) as imencode:
                    encode_chat_image(image)

                self.assertTrue(np.array_equal(image, original))
                self.assertEqual(
                    imencode.call_args.args[1][0, 0].tolist(),
                    expected,
                )

    def test_encoding_parameters_and_image_part(self):
        image = np.zeros((2, 2), dtype=np.uint8)
        encoded_bytes = b'jpeg bytes'
        encoded = np.frombuffer(encoded_bytes, dtype=np.uint8)

        with mock.patch(
            'ballontranslator.modules.llm_vision.cv2.imencode',
            return_value=(True, encoded),
        ) as imencode:
            default = encode_chat_image(image)
            self.assertEqual(imencode.call_args.args, ('.jpg', image))

            explicit = encode_chat_image(
                image,
                detail='HIGH',
                jpeg_quality=85,
            )
            self.assertEqual(
                imencode.call_args.args,
                (
                    '.jpg',
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                ),
            )
        self.assertEqual(
            explicit.data_url,
            'data:image/jpeg;base64,'
            + base64.b64encode(encoded_bytes).decode('ascii'),
        )
        self.assertEqual(
            explicit.image_part()['image_url']['detail'],
            'HIGH',
        )
        self.assertNotIn('detail', default.image_part()['image_url'])

    def test_encoding_failure_uses_the_callers_message(self):
        with mock.patch(
            'ballontranslator.modules.llm_vision.cv2.imencode',
            return_value=(False, None),
        ):
            with self.assertRaisesRegex(RuntimeError, '^OCR failed$'):
                encode_chat_image(
                    np.zeros((2, 2), dtype=np.uint8),
                    failure_message='OCR failed',
                )


if __name__ == '__main__':
    unittest.main()
