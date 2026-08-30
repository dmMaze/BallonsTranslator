"""Shared OpenAI-compatible chat image encoding."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class EncodedChatImage:
    """One immutable JPEG image part for a chat request.

    >>> EncodedChatImage('data:image/jpeg;base64,AA==', 'auto').detail
    'auto'
    """

    data_url: str
    detail: str

    def image_part(self) -> Dict[str, Any]:
        image_url = {'url': self.data_url}
        if self.detail.lower() != 'none':
            image_url['detail'] = self.detail
        return {'type': 'image_url', 'image_url': image_url}


def encode_chat_image(
    image: np.ndarray,
    *,
    detail: str = 'None',
    jpeg_quality: Optional[int] = None,
    failure_message: str = 'Failed to encode image.',
) -> EncodedChatImage:
    """Encode one RGB/RGBA project image for an OpenAI image content part.

    Grayscale arrays pass through unchanged. Omitting ``jpeg_quality`` also
    preserves OpenCV's default JPEG parameters, as used by LLM OCR.
    """
    encoded_image = image
    # Project images use Pillow's channel order; OpenCV encoders do not.
    if image.ndim == 3 and image.shape[-1] == 3:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[-1] == 4:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    if jpeg_quality is None:
        success, buffer = cv2.imencode('.jpg', encoded_image)
    else:
        success, buffer = cv2.imencode(
            '.jpg',
            encoded_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
    if not success:
        raise RuntimeError(failure_message)

    return EncodedChatImage(
        data_url=(
            'data:image/jpeg;base64,'
            + base64.b64encode(buffer.tobytes()).decode('ascii')
        ),
        detail=str(detail or 'None'),
    )
