"""
AI-inference fusion: combines YOLO, segmentation, saliency, edges, motion
to produce the best phosphene representation from any input image.

Works like Dynaphos conceptually: takes inference from all models and fuses
to maximize information preservation for prosthetic vision.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Dict, List, Any, Tuple


def _normalize(m: np.ndarray) -> np.ndarray:
    mn, mx = m.min(), m.max()
    if mx - mn > 1e-8:
        return ((m - mn) / (mx - mn)).astype(np.float32)
    return np.clip(m, 0, 1).astype(np.float32)


def fuse_ai_attention(
    image: np.ndarray,
    saliency: Optional[np.ndarray] = None,
    segmentation: Optional[np.ndarray] = None,
    edges: Optional[np.ndarray] = None,
    motion_magnitude: Optional[np.ndarray] = None,
    yolo_heatmap: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Fuse all model outputs into a unified attention map [0, 1].

    High values = salient regions to preserve in phosphene.
    Uses saliency as primary driver; boosts with segmentation foreground,
    edges for contours, motion for dynamic regions, YOLO for detected objects.
    """
    h, w = image.shape[:2]
    if weights is None:
        weights = {"saliency": 2.0, "segmentation": 1.2, "edges": 1.0, "motion": 0.8, "yolo": 1.5}

    fused = np.zeros((h, w), dtype=np.float32)

    if saliency is not None and saliency.size > 0:
        sal = _normalize(cv2.resize(saliency.astype(np.float32).squeeze(), (w, h)))
        fused += weights["saliency"] * sal

    if segmentation is not None:
        seg = _normalize(cv2.resize(segmentation.astype(np.float32), (w, h)))
        fused += weights["segmentation"] * seg

    if edges is not None:
        ed = _normalize(cv2.resize(edges.astype(np.float32), (w, h)))
        fused += weights["edges"] * ed

    if motion_magnitude is not None:
        mot = _normalize(cv2.resize(motion_magnitude.astype(np.float32), (w, h)))
        fused += weights["motion"] * mot

    if yolo_heatmap is not None:
        yh = _normalize(cv2.resize(yolo_heatmap.astype(np.float32), (w, h)))
        fused += weights["yolo"] * yh

    fused = _normalize(fused)
    fused = cv2.GaussianBlur(fused, (7, 7), 1.5)
    fused = _normalize(fused)
    return np.clip(fused, 0, 1).astype(np.float32)


def yolo_detections_to_heatmap(
    detections: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    sigma: float = 40.0,
) -> np.ndarray:
    """Convert YOLO bounding boxes to a spatial heatmap (objects = high)."""
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for d in detections:
        x1, y1, x2, y2 = [int(round(x)) for x in d["bbox"][:4]]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        conf = d.get("conf", 1.0)
        y_coords = np.arange(h, dtype=np.float32)
        x_coords = np.arange(w, dtype=np.float32)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)) * conf
        heatmap = heatmap + gauss
    return _normalize(heatmap)


def attention_to_heatmap_colored(attention: np.ndarray) -> np.ndarray:
    """
    Convert attention map to heatmap visualization: blue=low, green=mid, yellow=red=high.
    Matches reference: warm colors = high attention, cool = low.
    """
    att = _normalize(attention)
    att_uint8 = (att * 255).astype(np.uint8)
    colored = cv2.applyColorMap(att_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def attention_to_phosphene(
    attention: np.ndarray,
    output_size: Tuple[int, int],
    grid_size: int = 60,
    blob_sigma: float = 2.5,
    threshold: float = 0.15,
) -> np.ndarray:
    """
    Render attention map as sparse white blobs on black (phosphene).
    High-attention regions become bright Gaussian blobs.
    """
    h, w = attention.shape[:2]
    stim_grid = cv2.resize(attention, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    stim_grid = np.clip(stim_grid, 0, 1).astype(np.float64)
    stim_grid[stim_grid < threshold] = 0

    out_h, out_w = output_size
    canvas = np.zeros((out_h, out_w), dtype=np.float64)
    k = int(blob_sigma * 4) | 1
    center = k // 2
    y, x = np.ogrid[:k, :k]
    gauss = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * blob_sigma ** 2))
    gauss = gauss / gauss.max()

    for i in range(grid_size):
        for j in range(grid_size):
            if stim_grid[i, j] < 0.02:
                continue
            y_pos = int((i + 0.5) / grid_size * out_h)
            x_pos = int((j + 0.5) / grid_size * out_w)
            scaled = gauss * stim_grid[i, j]
            top, left = y_pos - center, x_pos - center
            y1, y2 = max(0, top), min(out_h, top + k)
            x1, x2 = max(0, left), min(out_w, left + k)
            py1, py2 = y1 - top, y2 - top
            px1, px2 = x1 - left, x2 - left
            if py2 > py1 and px2 > px1:
                canvas[y1:y2, x1:x2] = np.minimum(1.0, canvas[y1:y2, x1:x2] + scaled[py1:py2, px1:px2])

    cmin, cmax = canvas.min(), canvas.max()
    if cmax - cmin > 1e-6:
        canvas = (canvas - cmin) / (cmax - cmin)
    elif cmax > 0:
        canvas = canvas / cmax
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
