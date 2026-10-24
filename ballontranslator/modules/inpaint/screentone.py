"""Restore native-resolution periodic texture over a normal LaMa result."""

from typing import Optional, Tuple

import cv2
import numpy as np

from ballontranslator.utils.logger import logger as LOGGER


def _find_lattice(
    gray: np.ndarray, known: np.ndarray,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Find two independent integer translations supported by known pixels.

    >>> _find_lattice(np.zeros((32, 32), np.float32),
    ...               np.ones((32, 32), bool)) is None
    True
    """
    weights = known.astype(np.float32)
    low = cv2.GaussianBlur(gray * weights, (0, 0), 2)
    low /= np.maximum(cv2.GaussianBlur(weights, (0, 0), 2), 1e-4)
    signal = (gray - low) * weights
    if known.sum() < 512 or np.std(signal[known]) < 5:
        return None

    # Zero padding and pairwise energies avoid wraparound and masked-text peaks.
    shape = tuple(cv2.getOptimalDFTSize(2 * size) for size in gray.shape)
    spectrum = np.fft.rfft2(signal, s=shape)
    valid_spectrum = np.fft.rfft2(weights, s=shape)
    numerator = np.fft.irfft2(spectrum * spectrum.conj(), s=shape)
    energy = np.fft.irfft2(
        np.fft.rfft2(signal**2, s=shape) * valid_spectrum.conj(), s=shape
    )
    overlap = np.fft.irfft2(valid_spectrum * valid_spectrum.conj(), s=shape)
    radius = min(24, min(gray.shape) // 3)
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    denominator = np.sqrt(np.maximum(energy[y, x] * energy[-y, -x], 0))
    score = numerator[y, x] / np.maximum(denominator, 1e-5)
    score[overlap[y, x] < max(128, known.sum() * 0.2)] = 0
    peaks = score >= cv2.dilate(score, np.ones((3, 3), np.uint8)) - 1e-8
    peaks &= (score > 0.85) & (x*x + y*y >= 4)
    peaks &= (y > 0) | ((y == 0) & (x > 0))
    candidates = [
        (float(score[i, j]), int(x[i, j]), int(y[i, j]))
        for i, j in zip(*np.where(peaks))
    ]
    if not candidates:
        return None
    best = max(candidate[0] for candidate in candidates)
    candidates = sorted(
        (candidate for candidate in candidates if candidate[0] >= best - 0.04),
        key=lambda candidate: candidate[1]**2 + candidate[2]**2,
    )
    _, ax, ay = candidates[0]
    for _, bx, by in candidates[1:]:
        if 4 <= abs(ax * by - ay * bx) <= 144:
            return (ax, ay), (bx, by)
    return None


def restore_screentone(
    image: np.ndarray, mask: np.ndarray, result: np.ndarray,
) -> np.ndarray:
    """Match nearby screen pixels and shading while keeping strong LaMa edges.

    Only masked, locally supported pixels are changed. Pattern detection and
    donors use the original unmasked image; unsupported regions retain the
    normal inference result. Nothing is mutated and no extra inference runs.

    >>> image = np.full((32, 32, 3), 127, np.uint8)
    >>> restore_screentone(image, np.zeros((32, 32), np.uint8), image) is image
    True
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Screentone restoration requires a uint8 RGB image.')
    if mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
        raise ValueError('Screentone restoration requires a matching uint8 mask.')
    if result.dtype != np.uint8 or result.shape != image.shape:
        raise ValueError('Screentone restoration requires a matching uint8 RGB result.')
    texture = _estimate_texture(image, mask)
    if texture is None:
        return result
    detail, background, confidence, period = texture

    # Use the normal result to protect reconstructed structural edges.
    sigma = max(1.0, period / 3)
    smooth = cv2.GaussianBlur(result.astype(np.float32), (0, 0), sigma)
    shade = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(shade, cv2.CV_32F, 1, 0, ksize=3) / 8
    grad_y = cv2.Sobel(shade, cv2.CV_32F, 0, 1, ksize=3) / 8
    radius = int(np.ceil(sigma))
    edges = cv2.dilate(cv2.magnitude(grad_x, grad_y),
                       np.ones((2*radius + 1, 2*radius + 1), np.uint8))
    confidence *= 1 - np.clip((edges - 6) / 10, 0, 1)
    # Match both brightness and contrast to the same source donors. Scaling the
    # dots by model brightness would retain glyph-shaped low-frequency blobs.
    restored = smooth + (background - shade + detail)[..., None]
    blended = result * (1 - confidence[..., None]) + restored * confidence[..., None]
    output = result.copy()
    selected = (mask > 127) & (confidence > 0)
    output[selected] = np.clip(np.rint(blended[selected]), 0, 255).astype(np.uint8)
    LOGGER.debug('Screentone restoration: %.1f%% of masked pixels have strong support.',
                 100 * float(np.mean(confidence[mask > 127] > 0.5)))
    return output


def _estimate_texture(
    image: np.ndarray, mask: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Interpolate local same-phase donors, excluding text and masked pixels.

    >>> image = np.full((32, 32, 3), 127, np.uint8)
    >>> _estimate_texture(image, np.full((32, 32), 255, np.uint8)) is None
    True
    """
    height, width = mask.shape
    missing = mask > 127
    if min(height, width) < 16 or not missing.any() or missing.all():
        return None
    # This first version handles monochrome screens, not colored print screens.
    if np.mean(np.ptp(image[~missing].astype(np.int16), axis=1)) > 8:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    known = cv2.erode((~missing).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if known.sum() < 512:
        return None

    # Bound FFT work without resizing fine dots. Try the best intact crop first.
    sample_h, sample_w = min(height, 192), min(width, 192)
    samples = []
    tops = sorted(set(range(0, height - sample_h + 1, sample_h)) | {height - sample_h})
    lefts = sorted(set(range(0, width - sample_w + 1, sample_w)) | {width - sample_w})
    for top in tops:
        for left in lefts:
            region = np.s_[top:top + sample_h, left:left + sample_w]
            count = int(known[region].sum())
            samples.append((count, top, left))
    lattice = None
    for count, top, left in sorted(samples, reverse=True)[:8]:
        if count < 512:
            continue
        region = np.s_[top:top + sample_h, left:left + sample_w]
        lattice = _find_lattice(gray[region], known[region])
        if lattice is not None:
            break
    if lattice is None:
        return None

    (ax, ay), (bx, by) = lattice
    determinant = abs(ax * by - ay * bx)
    y, x = np.indices((height, width), dtype=np.int32)
    phases = ((by*x - bx*y) % determinant) * determinant
    phases += (-ay*x + ax*y) % determinant
    keys, counts = np.unique(phases[known], return_counts=True)
    if len(keys) != determinant or counts.min() < 4:
        return None
    period = max(np.hypot(ax, ay), np.hypot(bx, by))
    sigma = max(8.0, period * 3)
    predicted = np.zeros_like(gray)
    background = np.zeros_like(gray)
    support = np.full_like(gray, np.inf)
    # Estimate each phase from its own valid neighbors, instead of interpolating
    # a binary reliability flag through glyph-shaped holes. Keep memory O(HW).
    for key in keys:
        phase = phases == key
        weights = (known & phase).astype(np.float32)
        mass = cv2.GaussianBlur(weights, (0, 0), sigma)
        estimate = cv2.GaussianBlur(gray * weights, (0, 0), sigma) / np.maximum(mass, 1e-8)
        background += estimate / len(keys)
        np.copyto(predicted, estimate, where=phase)
        support = np.minimum(support, mass * len(keys))

    # Prediction residual checks phase agreement, not only whether a lattice
    # exists elsewhere in the crop. Distant/absent donors contribute no repair.
    weights = known.astype(np.float32)
    noise = cv2.GaussianBlur((gray - predicted)**2 * weights, (0, 0), sigma)
    energy = cv2.GaussianBlur((gray - background)**2 * weights, (0, 0), sigma)
    confidence = np.clip((1 - noise / np.maximum(energy, 1e-5) - 0.75) / 0.2, 0, 1)
    confidence *= np.clip((support - 0.01) / 0.04, 0, 1)
    strength = cv2.GaussianBlur((predicted - background)**2 * weights, (0, 0), sigma)
    strength /= np.maximum(cv2.GaussianBlur(weights, (0, 0), sigma), 1e-8)
    confidence[strength < 25] = 0
    if not np.any(confidence[missing] > 0):
        return None
    return predicted - background, background, confidence, float(period)
