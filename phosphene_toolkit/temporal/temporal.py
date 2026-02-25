"""Temporal stabilization for phosphene stimulation."""

import numpy as np
from typing import Optional


class TemporalStabilizer:
    """Stabilizes stimulation over time."""

    def __init__(self, smoothing: float = 0.7):
        self.smoothing = smoothing
        self.prev_map: Optional[np.ndarray] = None

    def stabilize(
        self,
        fused_map: np.ndarray,
        motion_magnitude: Optional[np.ndarray] = None,
        frame_time: Optional[float] = None,
    ) -> np.ndarray:
        """Smooth map with previous frame."""
        if self.prev_map is None:
            self.prev_map = fused_map.copy()
            return fused_map
        alpha = 1.0 - self.smoothing
        out = alpha * fused_map + (1 - alpha) * self.prev_map
        out = np.clip(out, 0, 1).astype(np.float32)
        self.prev_map = out.copy()
        return out

    def reset(self):
        self.prev_map = None
