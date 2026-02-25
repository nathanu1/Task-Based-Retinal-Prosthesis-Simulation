"""Edge-priority phosphene renderer: bright dots on black from edges (and optional saliency)."""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Tuple

DEFAULT_GRID = 60
DEFAULT_OUTPUT_SIZE = (480, 640)
DEFAULT_BLOB_SIGMA = 2.5
DEFAULT_EDGE_LOW, DEFAULT_EDGE_HIGH = 50, 150


def image_to_edge_stim(image: np.ndarray, grid_size: int = DEFAULT_GRID,
    low: int = DEFAULT_EDGE_LOW, high: int = DEFAULT_EDGE_HIGH,
    saliency: Optional[np.ndarray] = None, edge_weight: float = 0.7, saliency_weight: float = 0.3,
    multi_scale: bool = True) -> np.ndarray:
    """Build stimulation map from edges (priority) and optional saliency."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = np.asarray(gray, dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if multi_scale:
        e1 = cv2.Canny(gray, 30, 80)
        e2 = cv2.Canny(gray, low, high)
        e3 = cv2.Canny(gray, 100, 200)
        edges = np.maximum(np.maximum(e1, e2), e3)
    else:
        edges = cv2.Canny(gray, low, high)
    edge_grid = cv2.resize(edges.astype(np.float32) / 255.0, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    stim = edge_weight * edge_grid
    if saliency is not None and saliency.size > 0:
        sal = np.asarray(saliency, dtype=np.float32).squeeze()
        if sal.max() > sal.min():
            sal = (sal - sal.min()) / (sal.max() - sal.min())
        sal_grid = cv2.resize(sal, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
        stim = stim + saliency_weight * sal_grid
    return np.clip(stim.astype(np.float64), 0.0, 1.0)


def render_stim_as_phosphenes(stim_grid: np.ndarray, output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    blob_sigma: float = DEFAULT_BLOB_SIGMA, black_background: bool = True, contrast_stretch: bool = True) -> np.ndarray:
    """Render stimulation grid as bright Gaussian blobs on black."""
    stim_h, stim_w = stim_grid.shape
    out_h, out_w = output_size
    stim = np.clip(np.asarray(stim_grid, dtype=np.float64), 0, 1)
    canvas = np.zeros((out_h, out_w), dtype=np.float64) if black_background else np.full((out_h, out_w), 0.05)
    k = int(blob_sigma * 4) | 1
    center = k // 2
    y, x = np.ogrid[:k, :k]
    gauss = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * blob_sigma ** 2))
    gauss = gauss / gauss.max()
    for i in range(stim_h):
        for j in range(stim_w):
            if stim[i, j] < 0.02:
                continue
            y_pos = int((i + 0.5) / stim_h * out_h)
            x_pos = int((j + 0.5) / stim_w * out_w)
            scaled = gauss * stim[i, j]
            top, left = y_pos - center, x_pos - center
            y1, y2 = max(0, top), min(out_h, top + k)
            x1, x2 = max(0, left), min(out_w, left + k)
            py1, py2 = y1 - top, y2 - top
            px1, px2 = x1 - left, x2 - left
            if py2 > py1 and px2 > px1:
                canvas[y1:y2, x1:x2] = np.minimum(1.0, canvas[y1:y2, x1:x2] + scaled[py1:py2, px1:px2])
    if contrast_stretch:
        cmin, cmax = canvas.min(), canvas.max()
        if cmax - cmin > 1e-6:
            canvas = (canvas - cmin) / (cmax - cmin)
        elif cmax > 0:
            canvas = canvas / cmax
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


def render_edge_priority_phosphene(image: np.ndarray, grid_size: int = DEFAULT_GRID,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE, saliency: Optional[np.ndarray] = None,
    edge_low: int = DEFAULT_EDGE_LOW, edge_high: int = DEFAULT_EDGE_HIGH, blob_sigma: float = DEFAULT_BLOB_SIGMA) -> np.ndarray:
    """One-shot: image → edges (+ saliency) → stim map → phosphene image."""
    stim = image_to_edge_stim(image, grid_size=grid_size, low=edge_low, high=edge_high,
        saliency=saliency, edge_weight=0.75, saliency_weight=0.25)
    return render_stim_as_phosphenes(stim, output_size=output_size, blob_sigma=blob_sigma,
        black_background=True, contrast_stretch=True)
