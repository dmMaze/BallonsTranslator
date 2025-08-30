from typing import Union, Tuple, Callable

import cv2
import numpy as np
from qtpy.QtGui import QColor, QPixmap, QImage

from .misc import pixmap2ndarray, ndarray2pixmap


def apply_shadow_effect(img: Union[QPixmap, QImage, np.ndarray], color: QColor, strength=1.0, radius=21) -> Tuple[
    QPixmap, np.ndarray, np.ndarray]:
    if isinstance(color, QColor):
        color = [color.red(), color.green(), color.blue()]

    if not isinstance(img, np.ndarray):
        img = pixmap2ndarray(img, keep_alpha=True)

    mask = img[..., -1].copy()
    ksize = radius * 2 + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize / 6)
    if strength != 1:
        mask = np.clip(mask.astype(np.float32) * strength, 0, 255).astype(np.uint8)
    bg_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    bg_img[..., :3] = np.array(color, np.uint8)
    bg_img[..., 3] = mask

    result = ndarray2pixmap(bg_img)
    return result, img
