"""Improved edge detection: multi-scale Canny, Sobel, Scharr, Laplacian."""

import numpy as np
import cv2
from typing import Literal, Optional, Tuple

EdgeMethod = Literal["canny", "canny_multi", "sobel", "scharr", "laplacian", "structured"]


def compute_edges(
    image: np.ndarray,
    method: EdgeMethod = "canny_multi",
    low: int = 50,
    high: int = 150,
    blur_ksize: int = 3,
) -> np.ndarray:
    """
    Compute edge map [0, 1] using chosen method.

    - canny: Single-scale Canny
    - canny_multi: Multi-scale Canny (combines low/medium/high thresholds)
    - sobel: Sobel magnitude
    - scharr: Scharr magnitude (finer gradients)
    - laplacian: Laplacian of Gaussian
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = np.asarray(gray, dtype=np.uint8)
    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if method == "canny":
        edges = cv2.Canny(gray, low, high)
        return (edges.astype(np.float32) / 255.0)

    if method == "canny_multi":
        e1 = cv2.Canny(gray, 30, 80)
        e2 = cv2.Canny(gray, low, high)
        e3 = cv2.Canny(gray, 100, 200)
        combined = np.maximum(np.maximum(e1.astype(np.float32), e2.astype(np.float32)), e3.astype(np.float32))
        return np.clip(combined / 255.0, 0, 1).astype(np.float32)

    if method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        return mag.astype(np.float32)

    if method == "scharr":
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        return mag.astype(np.float32)

    if method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        lap = np.abs(lap)
        lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
        return lap.astype(np.float32)

    # fallback
    edges = cv2.Canny(gray, low, high)
    return (edges.astype(np.float32) / 255.0)
