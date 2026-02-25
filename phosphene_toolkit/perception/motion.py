"""Farneback optical flow motion detection with flow visualization."""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple


class FarnebackMotionDetector:
    """Farneback optical flow motion detector with HSV flow visualization."""

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15):
        self.prev_gray: Optional[np.ndarray] = None
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize

    def detect_motion(self, frame: np.ndarray) -> Dict:
        """Detect motion; returns magnitude, direction, and flow_vis (HSV color-coded)."""
        h, w = frame.shape[:2]
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        magnitude = np.zeros((h, w), dtype=np.float32)
        direction = np.zeros((h, w), dtype=np.float32)
        flow_vis = np.zeros((h, w, 3), dtype=np.uint8)
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                self.pyr_scale, self.levels, self.winsize, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mag_norm = np.clip(mag * 5, 0, 1)
            magnitude = mag_norm.astype(np.float32)
            direction = np.arctan2(flow[..., 1], flow[..., 0])
            flow_vis = _flow_to_hsv(flow, mag)
        self.prev_gray = gray.copy()
        return {"magnitude": magnitude, "direction": direction, "flow_vis": flow_vis}

    def reset(self):
        self.prev_gray = None


def compute_motion_between_frames(prev_frame: np.ndarray, curr_frame: np.ndarray) -> Dict:
    """
    Compute optical flow between two consecutive frames (for video tracking).
    Returns magnitude, direction, flow_vis.
    """
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    h, w = curr_gray.shape[:2]
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mag_norm = np.clip(mag * 5, 0, 1).astype(np.float32)
    direction = np.arctan2(flow[..., 1], flow[..., 0])
    flow_vis = _flow_to_hsv(flow, mag)
    return {"magnitude": mag_norm, "direction": direction, "flow_vis": flow_vis, "flow": flow}


def _flow_to_hsv(flow: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Convert optical flow to HSV color visualization."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    mag_norm = np.clip(mag * 3, 0, 1)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
