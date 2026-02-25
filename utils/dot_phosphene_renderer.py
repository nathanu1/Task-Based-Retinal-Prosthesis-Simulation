"""
Simple sparse dot phosphene renderer.

Goal:
- Render a stimulation grid (e.g. 60x60 or 6x10) as *dots* (Gaussian blobs) on black,
  without cortical magnification or axon streaks.
- Designed for "dot-tracing" subject outlines + a few interior/foreground dots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import cv2


BlendMode = Literal["sum", "max"]


@dataclass
class DotRenderParams:
    sigma_px: float = 1.6
    intensity_gamma: float = 0.85
    output_gamma: float = 0.85
    blend: BlendMode = "sum"
    jitter_px: float = 0.0


def render_dots_from_grid(
    stim_grid: np.ndarray,
    *,
    output_size: Tuple[int, int],
    params: Optional[DotRenderParams] = None,
) -> np.ndarray:
    """
    Args:
        stim_grid: (H,W) in [0,1]
        output_size: (out_h, out_w)
    Returns:
        uint8 grayscale (out_h,out_w)
    """
    if params is None:
        params = DotRenderParams()
    stim = np.asarray(stim_grid, dtype=np.float32)
    if stim.ndim != 2:
        raise ValueError("stim_grid must be 2D")
    stim = np.clip(stim, 0.0, 1.0)
    sh, sw = stim.shape
    out_h, out_w = output_size

    # Pick active electrodes
    ys, xs = np.nonzero(stim > 1e-6)
    if ys.size == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    inten = stim[ys, xs].astype(np.float32)
    inten = np.power(np.clip(inten, 0.0, 1.0), float(params.intensity_gamma)).astype(np.float32)

    sigma = float(max(0.5, params.sigma_px))
    k = int(np.ceil(4.0 * sigma)) * 2 + 1
    k = max(3, k | 1)
    c = k // 2
    ax = (np.arange(k, dtype=np.float32) - c).astype(np.float32)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    ker = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma + 1e-8)).astype(np.float32)
    ker = ker / (float(np.max(ker)) + 1e-8)

    canvas = np.zeros((out_h, out_w), dtype=np.float32)
    rng = np.random.default_rng(0)
    for y0, x0, a in zip(ys.tolist(), xs.tolist(), inten.tolist()):
        cy = int((y0 + 0.5) / max(1, sh) * out_h)
        cx = int((x0 + 0.5) / max(1, sw) * out_w)
        if params.jitter_px and params.jitter_px > 1e-6:
            cy = int(cy + rng.normal(0.0, float(params.jitter_px)))
            cx = int(cx + rng.normal(0.0, float(params.jitter_px)))
        top = cy - c
        left = cx - c
        y1 = max(0, top)
        x1 = max(0, left)
        y2 = min(out_h, top + k)
        x2 = min(out_w, left + k)
        ky1 = y1 - top
        kx1 = x1 - left
        ky2 = ky1 + (y2 - y1)
        kx2 = kx1 + (x2 - x1)
        if y2 <= y1 or x2 <= x1:
            continue
        patch = ker[ky1:ky2, kx1:kx2] * float(a)
        if params.blend == "max":
            canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], patch)
        else:
            canvas[y1:y2, x1:x2] = np.clip(canvas[y1:y2, x1:x2] + patch, 0.0, 1.0)

    # Tiny blur to reduce pixelation at small sigmas
    if sigma < 1.2:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=0.35, sigmaY=0.35)

    # Normalize and output gamma
    mn, mx = float(canvas.min()), float(canvas.max())
    if mx - mn > 1e-8:
        canvas = (canvas - mn) / (mx - mn)
    canvas = np.power(np.clip(canvas, 0.0, 1.0), float(params.output_gamma)).astype(np.float32)
    return (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)

