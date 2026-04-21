"""
Image-quality metrics for phosphene percept comparison.

SSIM
    Structural similarity (skimage).  Measures luminance, contrast, and
    structural correlation.  Range [−1, 1]; 1 = identical.

LPIPS
    Learned Perceptual Image Patch Similarity (lpips package, optional).
    Lower = more perceptually similar.  Range [0, ~1+].
    Falls back to None if lpips is not installed.

Notes on interpretation
-----------------------
Phosphene images are sparse, binary-ish grayscale; natural images are dense
RGB.  Direct pixel-level similarity numbers are therefore NOT meaningful as
absolute quality scores between a phosphene and its source image — they are
meaningful as *relative* comparisons between two phosphene encodings of the
same input.

When computing metrics against a reconstructed image (decoder output), set
``mode="reconstruction"``; when comparing two phosphenes set
``mode="phosphene_vs_phosphene"``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# SSIM from scikit-image
try:
    from skimage.metrics import structural_similarity as _ssim_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    _ssim_fn = None  # type: ignore

# LPIPS (optional heavy dependency)
try:
    import lpips as _lpips_lib
    import torch as _torch
    LPIPS_AVAILABLE = True
    _lpips_fn = _lpips_lib.LPIPS(net="vgg", verbose=False)
    _lpips_fn.eval()
except Exception:
    LPIPS_AVAILABLE = False
    _lpips_fn = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVAL_SIZE: Tuple[int, int] = (256, 256)  # canonical comparison resolution


def _to_gray_float(img: np.ndarray, size: Tuple[int, int] = _EVAL_SIZE) -> np.ndarray:
    """Resize to *size* and return float32 [0,1] single-channel."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    img = cv2.resize(img.astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _to_rgb_tensor(img: np.ndarray, size: Tuple[int, int] = _EVAL_SIZE):
    """Convert any image to a normalised [−1,1] RGB torch tensor (B=1,C,H,W)."""
    import torch
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    if img.max() > 1.5:
        img = img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    # Normalise to [−1, 1]
    img = img * 2.0 - 1.0
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ssim(
    a: np.ndarray,
    b: np.ndarray,
    *,
    size: Tuple[int, int] = _EVAL_SIZE,
) -> float:
    """Compute SSIM between two images (any format, any size).

    Both images are resized to *size*, converted to float32 [0,1] grayscale,
    then SSIM is computed with a 7-pixel Gaussian window.

    Returns NaN if skimage is unavailable.
    """
    if not SKIMAGE_AVAILABLE or _ssim_fn is None:
        return float("nan")
    fa = _to_gray_float(a, size)
    fb = _to_gray_float(b, size)
    score = _ssim_fn(fa, fb, data_range=1.0)
    return float(score)


def compute_lpips(
    a: np.ndarray,
    b: np.ndarray,
    *,
    size: Tuple[int, int] = _EVAL_SIZE,
) -> Optional[float]:
    """Compute LPIPS between two images.

    Returns None if lpips / torch is not installed, or on error.
    Lower = more perceptually similar.
    """
    if not LPIPS_AVAILABLE or _lpips_fn is None:
        return None
    try:
        ta = _to_rgb_tensor(a, size)
        tb = _to_rgb_tensor(b, size)
        with _torch.no_grad():
            score = _lpips_fn(ta, tb)
        return float(score.item())
    except Exception:
        return None
