"""
Adaptive encoding policy utilities.

This module is intentionally model-agnostic: given (candidate -> stim grid),
it provides:
- candidate definitions
- foreground/boundary proxy masks on the electrode grid
- a foreground-fidelity proxy score
- hysteresis selection for video
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass(frozen=True)
class EncodingCandidate:
    name: str
    outline_mode: str  # "seg_boundary" | "pixel_edges"
    edge_method: str  # "canny_multi" | "scharr" | "sobel" | "laplacian" | "canny"
    contrast_mode: str  # "none" | "clahe" | "gamma" | "retinex"
    gamma: float = 1.0
    use_bilateral: bool = True
    fill_weight: float = 0.18
    bg_suppress: float = 0.75


@dataclass(frozen=True)
class CandidateScore:
    score: float
    fg_energy: float
    boundary_energy: float
    leak_energy: float
    active_dots: int
    dot_count_error: float
    flicker: float


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.min(x)) if x.size else 0.0
    mx = float(np.max(x)) if x.size else 1.0
    if mx - mn < 1e-8:
        return np.clip(x, 0.0, 1.0)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)


def resize_to_grid(m: np.ndarray, grid_n: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError("resize_to_grid expects 2D array")
    if cv2 is None:
        # Nearest-neighbor fallback without OpenCV
        h, w = m.shape
        ys = (np.linspace(0, h - 1, grid_n)).astype(np.int32)
        xs = (np.linspace(0, w - 1, grid_n)).astype(np.int32)
        return m[ys][:, xs].astype(np.float32)
    return cv2.resize(m, (grid_n, grid_n), interpolation=cv2.INTER_AREA).astype(np.float32)


def grid_foreground_mask(
    *,
    gate: Optional[np.ndarray],
    object_mask_u8: Optional[np.ndarray],
    grid_n: int,
    gate_thresh: float = 0.25,
) -> np.ndarray:
    """
    Build a foreground proxy on the electrode grid using (object mask OR gate).
    Returns float32 mask in [0,1].
    """
    fg = np.zeros((grid_n, grid_n), dtype=np.float32)
    if object_mask_u8 is not None and object_mask_u8.size > 0:
        obj = (np.asarray(object_mask_u8, dtype=np.uint8) > 0).astype(np.float32)
        fg = np.maximum(fg, (resize_to_grid(obj, grid_n) > 0.10).astype(np.float32))
    if gate is not None and gate.size > 0:
        g = _normalize01(np.asarray(gate, dtype=np.float32))
        fg = np.maximum(fg, (resize_to_grid(g, grid_n) > float(gate_thresh)).astype(np.float32))
    return np.clip(fg, 0.0, 1.0).astype(np.float32)


def grid_boundary_mask(*, object_mask_u8: Optional[np.ndarray], grid_n: int) -> np.ndarray:
    """
    Boundary proxy from the object segmentation mask.
    Returns float32 mask in [0,1] on the electrode grid.
    """
    if object_mask_u8 is None or object_mask_u8.size == 0:
        return np.zeros((grid_n, grid_n), dtype=np.float32)
    m = (np.asarray(object_mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2 is None:
        # crude boundary: difference of dilated/eroded via convolution-free approx
        mm = (m > 0).astype(np.float32)
        gy, gx = np.gradient(mm)
        b = (np.abs(gx) + np.abs(gy)) > 0
        return (resize_to_grid(b.astype(np.float32), grid_n) > 0.05).astype(np.float32)
    ker = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, ker)
    b = (grad > 0).astype(np.float32)
    return (resize_to_grid(b, grid_n) > 0.05).astype(np.float32)


def score_stim_grid(
    *,
    stim_grid: np.ndarray,
    fg_mask_g: np.ndarray,
    boundary_mask_g: Optional[np.ndarray] = None,
    target_active_dots: Optional[int] = None,
    prev_stim_grid: Optional[np.ndarray] = None,
    stim_thr: float = 1e-6,
    weights: Optional[Dict[str, float]] = None,
) -> CandidateScore:
    """
    Foreground-fidelity proxy score for a (grid_n x grid_n) stimulation grid.

    Interpretable terms:
    - fg_energy: fraction of stimulation energy on foreground
    - boundary_energy: fraction of energy on boundary proxy (if provided)
    - leak_energy: fraction of energy off foreground
    - dot_count_error: relative error vs target active electrodes
    - flicker: mean abs change of active mask vs prev (video stability)
    """
    if weights is None:
        weights = {
            "fg": 1.35,
            "boundary": 0.70,
            "leak": 0.85,
            "dot_err": 0.15,
            "flicker": 0.30,
        }

    stim = np.clip(np.asarray(stim_grid, dtype=np.float32), 0.0, 1.0)
    fg = (np.asarray(fg_mask_g, dtype=np.float32) > 0.5).astype(np.float32)
    bnd = None
    if boundary_mask_g is not None and np.asarray(boundary_mask_g).size > 0:
        bnd = (np.asarray(boundary_mask_g, dtype=np.float32) > 0.5).astype(np.float32)

    total_e = float(np.sum(stim) + 1e-8)
    fg_e = float(np.sum(stim * fg))
    leak_e = float(np.sum(stim * (1.0 - fg)))
    bnd_e = float(np.sum(stim * bnd)) if bnd is not None else 0.0

    fg_energy = _safe_div(fg_e, total_e)
    leak_energy = _safe_div(leak_e, total_e)
    boundary_energy = _safe_div(bnd_e, total_e) if bnd is not None else 0.0

    active_dots = int(np.sum(stim > float(stim_thr)))
    dot_count_error = 0.0
    if target_active_dots is not None and int(target_active_dots) > 0:
        dot_count_error = abs(active_dots - int(target_active_dots)) / float(target_active_dots)

    flicker = 0.0
    if prev_stim_grid is not None and np.asarray(prev_stim_grid).shape == stim.shape:
        prev = np.asarray(prev_stim_grid, dtype=np.float32)
        flicker = float(np.mean(np.abs((stim > stim_thr).astype(np.float32) - (prev > stim_thr).astype(np.float32))))

    score = (
        float(weights["fg"]) * fg_energy
        + float(weights["boundary"]) * boundary_energy
        - float(weights["leak"]) * leak_energy
        - float(weights["dot_err"]) * dot_count_error
        - float(weights["flicker"]) * flicker
    )

    return CandidateScore(
        score=float(score),
        fg_energy=float(fg_energy),
        boundary_energy=float(boundary_energy),
        leak_energy=float(leak_energy),
        active_dots=int(active_dots),
        dot_count_error=float(dot_count_error),
        flicker=float(flicker),
    )


def default_candidates(*, has_object_boundary: bool) -> List[EncodingCandidate]:
    """
    Small candidate set: fast enough to evaluate per frame.
    """
    c: List[EncodingCandidate] = []

    if has_object_boundary:
        c.append(
            EncodingCandidate(
                name="boundary_first_fill_low",
                outline_mode="seg_boundary",
                edge_method="canny_multi",
                contrast_mode="none",
                fill_weight=0.28,
                bg_suppress=0.72,
                use_bilateral=False,
            )
        )
        c.append(
            EncodingCandidate(
                name="boundary_first_fill_high",
                outline_mode="seg_boundary",
                edge_method="canny_multi",
                contrast_mode="none",
                fill_weight=0.40,
                bg_suppress=0.78,
                use_bilateral=False,
            )
        )

    # Pixel-edge fallbacks with contrast help
    c.extend(
        [
            EncodingCandidate(
                name="edges_clahe_canny_multi",
                outline_mode="pixel_edges",
                edge_method="canny_multi",
                contrast_mode="clahe",
                fill_weight=0.24,
                bg_suppress=0.68,
                use_bilateral=True,
            ),
            EncodingCandidate(
                name="edges_gamma_scharr",
                outline_mode="pixel_edges",
                edge_method="scharr",
                contrast_mode="gamma",
                gamma=0.75,
                fill_weight=0.28,
                bg_suppress=0.72,
                use_bilateral=True,
            ),
            EncodingCandidate(
                name="edges_retinex_canny_multi",
                outline_mode="pixel_edges",
                edge_method="canny_multi",
                contrast_mode="retinex",
                fill_weight=0.26,
                bg_suppress=0.72,
                use_bilateral=True,
            ),
            EncodingCandidate(
                name="edges_texture_suppression",
                outline_mode="pixel_edges",
                edge_method="canny_multi",
                contrast_mode="clahe",
                fill_weight=0.16,
                bg_suppress=0.84,
                use_bilateral=True,
            ),
            EncodingCandidate(
                name="edges_detail_balanced",
                outline_mode="pixel_edges",
                edge_method="scharr",
                contrast_mode="clahe",
                fill_weight=0.32,
                bg_suppress=0.60,
                use_bilateral=True,
            ),
        ]
    )
    return c


def select_with_hysteresis(
    *,
    scored: List[Tuple[EncodingCandidate, CandidateScore]],
    prev_choice_name: Optional[str],
    prev_score: Optional[float],
    stable_frames: int,
    min_hold_frames: int = 3,
    margin: float = 0.03,
) -> Tuple[EncodingCandidate, CandidateScore, bool]:
    """
    Select best candidate with lightweight hysteresis.

    Returns (candidate, score, switched?).
    """
    if not scored:
        raise ValueError("select_with_hysteresis: empty scored list")

    scored_sorted = sorted(scored, key=lambda t: float(t[1].score), reverse=True)
    best_c, best_s = scored_sorted[0]

    if prev_choice_name is None or prev_score is None:
        return best_c, best_s, True

    if stable_frames < int(min_hold_frames):
        # hold unless the improvement is significant
        if float(best_s.score) > float(prev_score) + float(margin):
            return best_c, best_s, (best_c.name != prev_choice_name)
        # keep previous if it exists in list, else take best
        for c, s in scored:
            if c.name == prev_choice_name:
                return c, s, False
        return best_c, best_s, True

    # after hold, switch if better by margin
    if float(best_s.score) > float(prev_score) + float(margin):
        return best_c, best_s, (best_c.name != prev_choice_name)

    # otherwise keep previous if possible
    for c, s in scored:
        if c.name == prev_choice_name:
            return c, s, False
    return best_c, best_s, True


def clahe_u8(gray_u8: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    if cv2 is None:
        return np.asarray(gray_u8, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    return clahe.apply(np.asarray(gray_u8, dtype=np.uint8))


def gamma_u8(gray_u8: np.ndarray, gamma: float) -> np.ndarray:
    g = float(max(1e-3, gamma))
    x = np.asarray(gray_u8, dtype=np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), g)
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)


def retinex_ssr_u8(gray_u8: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """
    Single-scale retinex (SSR) approximation:
      R = log(I) - log(GaussianBlur(I))
    normalized to uint8.
    """
    if cv2 is None:
        return np.asarray(gray_u8, dtype=np.uint8)
    I = np.asarray(gray_u8, dtype=np.float32) / 255.0
    I = np.clip(I, 1e-6, 1.0)
    blur = cv2.GaussianBlur(I, (0, 0), float(sigma))
    blur = np.clip(blur, 1e-6, 1.0)
    R = np.log(I) - np.log(blur)
    R = _normalize01(R)
    return (R * 255.0).astype(np.uint8)

