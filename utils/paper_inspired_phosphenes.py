"""
Paper-inspired cortical phosphene renderer (eLife 2024, e85812).

Goal: mimic the *appearance* and key modeling ingredients reported in:
van der Grinten et al. (2024) "Towards biologically plausible phosphene simulation..."
https://doi.org/10.7554/eLife.85812

We implement a pragmatic approximation of the paper's components:
- Cortical magnification factor M(r) (Eq. 4): M = k(b-a)/((r+a)(r+b))
- Current-dependent spread D and phosphene size P = D/M (Eq. 5-6) in relative units
- Each phosphene rendered as a Gaussian blob where ~2 std dev covers the phosphene size
- Electrode-wise thresholds (fixed per run) to sparsify output

This module takes an "attention" / "stimulation" map in [0,1] and renders a
paper-like phosphene percept as uint8 on black background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2


def _normalize(m: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(m)), float(np.max(m))
    if mx - mn > 1e-8:
        return ((m - mn) / (mx - mn)).astype(np.float32)
    return np.clip(m, 0, 1).astype(np.float32)


def cortical_magnification(
    r_deg: np.ndarray,
    *,
    a: float = 0.75,
    b: float = 120.0,
    k: float = 17.3,
) -> np.ndarray:
    """Eq. 4 in the paper (mm/deg), used here as a relative magnification factor."""
    r = np.asarray(r_deg, dtype=np.float32)
    return (k * (b - a)) / ((r + a) * (r + b) + 1e-8)


def _fixed_threshold_grid(shape: Tuple[int, int], mean: float, std: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    thr = rng.normal(loc=mean, scale=std, size=shape).astype(np.float32)
    return np.clip(thr, 0.0, 1.0)


@dataclass(frozen=True)
class PaperInspiredParams:
    # Visual field size (deg). Used to convert deg -> pixels.
    fov_deg: float = 16.0  # half-width/half-height in degrees

    # "Current" mapping (uA-like, relative; we only need a stable scale)
    i_max: float = 128.0
    i0: float = 23.9  # rheobase-like offset (paper cites ~23.9 uA)

    # Pulse parameters (drive duty-cycle-like scaling; optional)
    pulse_width_us: float = 170.0
    freq_hz: float = 300.0

    # Threshold grid (sparseness control)
    thr_mean: float = 0.20
    thr_std: float = 0.07
    thr_seed: int = 0
    # Increase thresholds outside "priority" regions to suppress background texture
    # Keep this moderate; allocation should mainly be controlled upstream (budgeting).
    thr_outside_boost: float = 0.10

    # Size model constants (chosen to match figure-like sizes)
    # P_deg ≈ (size_base + size_gain * I_uA) / M(r)
    size_base: float = 0.18
    size_gain: float = 0.0018

    # Render clamps
    # Keep phosphenes crisp (your reference has small bright dots)
    sigma_px_min: float = 0.7
    sigma_px_max: float = 3.0

    # Intensity shaping
    intensity_gamma: float = 0.65
    sigmoid_scale: float = 10.0  # higher = sharper thresholding

    # Post-processing (reduce blur + boost contrast)
    final_blur_sigma: float = 0.15
    output_gamma: float = 0.75


def render_paper_inspired_phosphenes(
    attention_map: np.ndarray,
    *,
    output_size: Tuple[int, int],
    grid_size: int = 60,
    params: Optional[PaperInspiredParams] = None,
    priority_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render a paper-inspired phosphene percept from an attention/stimulation map in [0,1].

    Returns:
        uint8 grayscale image (H, W), black background with bright phosphene blobs.
    """
    if params is None:
        params = PaperInspiredParams()

    att = _normalize(np.asarray(attention_map, dtype=np.float32))
    stim = cv2.resize(att, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    stim = np.clip(stim, 0, 1).astype(np.float32)

    # Map to "current" and apply a simple duty-cycle scaling (paper's Eq. 7 style)
    i_uA = stim * params.i_max
    duty = (params.pulse_width_us * 1e-6) * params.freq_hz  # Pw * f
    i_eff = np.maximum(0.0, (i_uA - params.i0)) * float(duty)
    i_eff_n = _normalize(i_eff)

    # Fixed per-electrode thresholds (paper samples per electrode; we keep it stable)
    thr = _fixed_threshold_grid((grid_size, grid_size), params.thr_mean, params.thr_std, seed=params.thr_seed)
    if priority_mask is not None and priority_mask.size > 0:
        pm = _normalize(np.asarray(priority_mask, dtype=np.float32))
        pm_g = cv2.resize(pm, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
        pm_g = np.clip(pm_g, 0, 1).astype(np.float32)
        # Outside high-priority regions, require higher effective current to activate
        thr = np.clip(thr + (1.0 - pm_g) * float(params.thr_outside_boost), 0.0, 1.0)
    active = (i_eff_n > thr).astype(np.float32)

    # Smooth thresholding to produce graded intensity while preserving sparsity
    x = (i_eff_n - thr) * params.sigmoid_scale
    intensity = (1.0 / (1.0 + np.exp(-x))) * active
    intensity = np.power(np.clip(intensity, 0, 1), params.intensity_gamma).astype(np.float32)

    out_h, out_w = output_size
    canvas = np.zeros((out_h, out_w), dtype=np.float32)

    # Electrode positions in visual field coordinates (deg)
    ys = np.linspace(-params.fov_deg, params.fov_deg, grid_size, dtype=np.float32)
    xs = np.linspace(-params.fov_deg, params.fov_deg, grid_size, dtype=np.float32)
    yy_deg, xx_deg = np.meshgrid(ys, xs, indexing="ij")
    r_deg = np.sqrt(xx_deg * xx_deg + yy_deg * yy_deg)
    m = cortical_magnification(r_deg)

    # Paper-inspired size model (relative): P_deg = (size_base + size_gain * I_uA) / M(r)
    p_deg = (params.size_base + params.size_gain * i_uA) / (m + 1e-8)
    sigma_deg = 0.5 * p_deg  # 2σ ≈ size

    # deg -> pixels conversion (assume symmetric FOV)
    px_per_deg_x = out_w / (2.0 * params.fov_deg)
    px_per_deg_y = out_h / (2.0 * params.fov_deg)
    sigma_px = sigma_deg * (0.5 * (px_per_deg_x + px_per_deg_y))
    sigma_px = np.clip(sigma_px, params.sigma_px_min, params.sigma_px_max).astype(np.float32)

    # Render each active electrode as a Gaussian blob with its sigma
    for i in range(grid_size):
        for j in range(grid_size):
            inten = float(intensity[i, j])
            if inten < 0.01:
                continue

            s = float(sigma_px[i, j])
            k = int(np.ceil(s * 4)) * 2 + 1
            if k < 3:
                k = 3
            cy = int((i + 0.5) / grid_size * out_h)
            cx = int((j + 0.5) / grid_size * out_w)

            # Create kernel
            ax = np.arange(k, dtype=np.float32) - (k // 2)
            yy, xx = np.meshgrid(ax, ax, indexing="ij")
            ker = np.exp(-(xx * xx + yy * yy) / (2.0 * s * s + 1e-8))
            ker = (ker / (np.max(ker) + 1e-8)) * inten

            top = cy - k // 2
            left = cx - k // 2
            y1 = max(0, top)
            x1 = max(0, left)
            y2 = min(out_h, top + k)
            x2 = min(out_w, left + k)
            ky1 = y1 - top
            kx1 = x1 - left
            ky2 = ky1 + (y2 - y1)
            kx2 = kx1 + (x2 - x1)
            if y2 > y1 and x2 > x1:
                canvas[y1:y2, x1:x2] = np.clip(canvas[y1:y2, x1:x2] + ker[ky1:ky2, kx1:kx2], 0, 1)

    # Minimal blur for clean dots (avoid smeary phosphene map)
    if params.final_blur_sigma and params.final_blur_sigma > 1e-6:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=float(params.final_blur_sigma), sigmaY=float(params.final_blur_sigma))

    # Contrast boost
    canvas = _normalize(canvas)
    canvas = np.power(np.clip(canvas, 0, 1), float(params.output_gamma)).astype(np.float32)
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)

