"""Monocular depth estimation (optional).

Uses MiDaS via torch.hub when available. This is used to boost near-field
regions for safety-oriented phosphene allocation.

Falls back gracefully if model weights cannot be loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn > 1e-8:
        return (x - mn) / (mx - mn)
    return np.clip(x, 0, 1)


@dataclass
class DepthResult:
    depth: np.ndarray  # higher = farther (MiDaS convention varies)
    inv_depth: np.ndarray  # higher = nearer


class DepthEstimator:
    """MiDaS depth estimator via torch.hub (small model by default)."""

    def __init__(self, model_type: str = "MiDaS_small"):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for depth estimation")
        self.model_type = model_type
        self.model = None
        self.transform = None

    def _ensure_loaded(self):
        if self.model is not None and self.transform is not None:
            return
        # Requires internet on first run to download weights
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)  # type: ignore
        self.model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")  # type: ignore
        self.transform = midas_transforms.small_transform if "small" in self.model_type.lower() else midas_transforms.default_transform
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def predict(self, bgr: np.ndarray, output_size: Optional[Tuple[int, int]] = None) -> DepthResult:
        self._ensure_loaded()
        assert self.model is not None and self.transform is not None

        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(img_rgb)
        if torch.cuda.is_available():
            inp = inp.cuda()
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = pred.detach().cpu().numpy().astype(np.float32)

        # MiDaS depth is relative; use inverse-depth as "near" prior
        depth_n = _normalize(depth)
        inv = 1.0 - depth_n

        if output_size is not None:
            h, w = output_size
            depth_n = cv2.resize(depth_n, (w, h), interpolation=cv2.INTER_CUBIC)
            inv = cv2.resize(inv, (w, h), interpolation=cv2.INTER_CUBIC)

        return DepthResult(depth=depth_n, inv_depth=_normalize(inv))

