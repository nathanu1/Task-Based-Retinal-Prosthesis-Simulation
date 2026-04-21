"""
Phosphene spatial coverage metrics.

Global coverage
    Fraction of the percept image that is "lit" (above a brightness threshold).
    A percept with many isolated dots has lower coverage than one with
    uniformly spread activations.

Radial / eccentricity-binned coverage
    The image is divided into N concentric annular bins from the image centre
    (eccentricity 0 = centre, 1 = corner).  Coverage is reported per bin,
    capturing whether phosphenes are concentrated centrally or spread
    peripherally.

    This is relevant because prosthetic vision with a limited electrode count
    benefits from placing phosphenes where the user's visual attention is;
    central coverage matters more for reading, peripheral for navigation.

Phosphene coverage heatmap
    Smoothed binary activation mask resized to a standard resolution for
    visual inspection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


_BRIGHTNESS_THR = 15          # uint8 pixel value considered "lit"
_RADIAL_BINS = 8               # number of eccentricity rings
_COVERAGE_EVAL_SIZE = (256, 256)


def _to_gray_u8(img: np.ndarray, size: Tuple[int, int] = _COVERAGE_EVAL_SIZE) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    return cv2.resize(img.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def radial_coverage_bins(
    phosphene_img: np.ndarray,
    *,
    n_bins: int = _RADIAL_BINS,
    threshold: int = _BRIGHTNESS_THR,
    size: Tuple[int, int] = _COVERAGE_EVAL_SIZE,
) -> Dict[str, object]:
    """Compute per-eccentricity-ring coverage fractions.

    Parameters
    ----------
    phosphene_img : array
        Uint8 grayscale phosphene image (H×W or H×W×C).
    n_bins : int
        Number of concentric rings.
    threshold : int
        Pixel brightness (0–255) above which a pixel counts as lit.
    size : tuple
        Evaluation resolution (H, W).

    Returns
    -------
    dict with:
        bin_centers : list[float]  – normalised eccentricity [0, 1]
        bin_coverage : list[float] – fraction of lit pixels per ring
        bin_pixel_count : list[int]
        mean_eccentricity : float  – coverage-weighted mean eccentricity
    """
    img = _to_gray_u8(phosphene_img, size)
    h, w = img.shape
    lit = (img > threshold).astype(np.float32)

    # Build eccentricity map: 0 at centre, 1 at farthest corner
    cy, cx = h / 2.0, w / 2.0
    ys = np.arange(h, dtype=np.float32) - cy
    xs = np.arange(w, dtype=np.float32) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    # Normalise by the max possible distance (corner)
    max_dist = float(np.sqrt(cy**2 + cx**2)) + 1e-8
    ecc = np.sqrt(yy**2 + xx**2) / max_dist
    ecc = np.clip(ecc, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = [float(0.5 * (edges[i] + edges[i + 1])) for i in range(n_bins)]
    bin_coverage: List[float] = []
    bin_pixel_count: List[int] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (ecc >= lo) & (ecc < hi)
        total = int(mask.sum())
        active = int((lit[mask]).sum())
        bin_pixel_count.append(total)
        bin_coverage.append(float(active / max(total, 1)))

    # Coverage-weighted mean eccentricity
    cov_arr = np.array(bin_coverage, dtype=np.float64)
    cen_arr = np.array(bin_centers, dtype=np.float64)
    mean_ecc = float(np.dot(cov_arr, cen_arr) / (cov_arr.sum() + 1e-8))

    return {
        "bin_centers": bin_centers,
        "bin_coverage": bin_coverage,
        "bin_pixel_count": bin_pixel_count,
        "mean_eccentricity": mean_ecc,
    }


def compute_coverage_metrics(
    phosphene_img: np.ndarray,
    *,
    threshold: int = _BRIGHTNESS_THR,
    n_radial_bins: int = _RADIAL_BINS,
    size: Tuple[int, int] = _COVERAGE_EVAL_SIZE,
) -> Dict[str, object]:
    """Compute all coverage metrics for one phosphene image.

    Returns
    -------
    dict with:
        global_coverage : float   – fraction of lit pixels (global)
        lit_pixels : int
        total_pixels : int
        radial : dict             – output of radial_coverage_bins()
        coverage_heatmap : np.ndarray – smoothed float32 H×W heatmap [0,1]
    """
    img = _to_gray_u8(phosphene_img, size)
    lit_mask = (img > threshold).astype(np.float32)
    lit_px = int(lit_mask.sum())
    total_px = int(lit_mask.size)

    radial = radial_coverage_bins(
        img,
        n_bins=n_radial_bins,
        threshold=threshold,
        size=size,
    )

    # Smoothed coverage heatmap
    heatmap = cv2.GaussianBlur(lit_mask, (0, 0), 5.0)
    hm_max = heatmap.max()
    if hm_max > 1e-8:
        heatmap = heatmap / hm_max

    return {
        "global_coverage": float(lit_px / max(total_px, 1)),
        "lit_pixels": lit_px,
        "total_pixels": total_px,
        "radial": radial,
        "coverage_heatmap": heatmap.astype(np.float32),
    }
