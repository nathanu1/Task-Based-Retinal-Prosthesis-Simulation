"""Bandwidth allocation for phosphene stimulation."""

import numpy as np
import cv2
from typing import Optional, Literal
from ..task_policy.schemas import TaskParams


class BandwidthAllocator:
    """Allocates limited phosphene bandwidth."""

    def __init__(self, strategy: str = "foveated", max_active: int = 200):
        self.strategy = strategy
        self.max_active = max_active

    def allocate(self, fused_map: np.ndarray, task_params: TaskParams) -> np.ndarray:
        """Allocate stimulation from fused map; output same shape for device grid."""
        h, w = fused_map.shape
        target_h, target_w = 60, 60
        resized = cv2.resize(fused_map, (target_w, target_h))
        flat = resized.flatten()
        k = min(self.max_active, flat.size)
        thresh = np.partition(flat, -k)[-k] if k > 0 else flat.max()
        mask = resized >= max(thresh, 1e-6)
        out = resized * mask.astype(np.float32)
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        return out
