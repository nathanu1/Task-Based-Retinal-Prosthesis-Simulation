"""Task-conditional fusion of perception maps with spatial weighting."""

import numpy as np
import cv2
from typing import Dict, List, Optional
from ..task_policy.schemas import TaskParams


def _normalize_map(m: np.ndarray) -> np.ndarray:
    mn, mx = m.min(), m.max()
    if mx - mn > 1e-8:
        return ((m - mn) / (mx - mn)).astype(np.float32)
    return np.clip(m, 0, 1).astype(np.float32)


class TaskConditionalFusion:
    """Fuses segmentation, saliency, motion with task weights and spatial gaussian."""

    def fuse_maps(
        self,
        segmentation: np.ndarray,
        saliency: np.ndarray,
        motion_magnitude: np.ndarray,
        task_params: TaskParams,
        class_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fuse maps weighted by task params with optional foveal emphasis."""
        h, w = segmentation.shape[:2]
        seg_n = _normalize_map(segmentation)
        sal_n = _normalize_map(saliency)
        mot_n = _normalize_map(motion_magnitude)
        w_seg = 1.0
        w_sal = 1.5
        w_mot = task_params.motion_weight
        fused = w_seg * seg_n + w_sal * sal_n + w_mot * mot_n
        fused = _normalize_map(fused)
        fused = cv2.GaussianBlur(fused, (5, 5), 1)
        fused = _normalize_map(fused)
        return np.clip(fused, 0, 1).astype(np.float32)
