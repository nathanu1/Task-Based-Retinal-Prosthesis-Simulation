"""
Dots Baseline backend — the original adaptive 60×60 stimulation grid +
Gaussian dot renderer.  This is the full existing pipeline extracted from
app.py with no behavioural changes.

DotsParams
    All sidebar parameters that control the dots pipeline.

DotsBackend
    Implements PhospheneBackend.  Accepts a PerceptionContext (containing
    pre-computed saliency, segmentation, edges, YOLO, gate, near-field) and
    runs the adaptive candidate selection → stimulation grid → dot render.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BackendResult, PerceptionContext, PhospheneBackend, TASK_PRESETS
from utils.adaptive_encoding import (
    EncodingCandidate,
    CandidateScore,
    default_candidates,
    grid_foreground_mask,
    grid_boundary_mask,
    score_stim_grid,
    select_with_hysteresis,
    clahe_u8,
    gamma_u8,
    retinex_ssr_u8,
)
from utils.ai_inference_fusion import fuse_ai_attention
from utils.dot_phosphene_renderer import DotRenderParams, render_dots_from_grid


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class DotsParams:
    """All tunable parameters for the Dots Baseline backend."""

    # Dot budget
    grid_n: int = 60
    dot_density: float = 0.12

    # Budget allocation weights
    subject_weight: float = 0.50
    near_weight: float = 0.22
    fill_weight: float = 0.18
    bg_suppress: float = 0.75

    # Outline source
    outline_source: str = "Segmentation boundary (preferred)"

    # Mask cleanup
    mask_kernel: int = 7
    target_class: str = "auto"
    interior_frac: float = 0.20
    fg_frac: float = 0.10
    min_sep: int = 1

    # Adaptive encoding controls
    use_adaptive: bool = True
    adaptive_hold: int = 3
    adaptive_margin: float = 0.03
    adaptive_gate_thresh: float = 0.18

    # Dot renderer
    dot_sigma: float = 1.6
    dot_blend: str = "sum"
    dot_jitter: float = 0.0

    # Edge prefilter
    use_bilateral: bool = True

    # Temporal smoothing (0 = off)
    temporal_smooth: float = 0.55

    @classmethod
    def from_task_preset(cls, task_mode: str) -> "DotsParams":
        """Return a DotsParams pre-populated with task-preset defaults."""
        p = cls()
        if task_mode in TASK_PRESETS:
            tp = TASK_PRESETS[task_mode]
            p.near_weight = tp.near_weight
            p.subject_weight = tp.subject_weight
            p.fill_weight = tp.fill_weight
            p.bg_suppress = tp.bg_suppress
        return p


# ---------------------------------------------------------------------------
# Private helpers (pixel-level)
# ---------------------------------------------------------------------------

def _robust_normalize(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi - lo < 1e-8:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _binary_fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    flood = m.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(flood) & cv2.bitwise_not(m)
    return ((m | holes) > 0).astype(np.uint8) * 255


def _keep_largest_component(mask_u8: np.ndarray, *, roi: Optional[tuple] = None) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(m.shape[1], int(x2)), min(m.shape[0], int(y2))
        roi_m = np.zeros_like(m)
        roi_m[y1:y2, x1:x2] = m[y1:y2, x1:x2]
        m = roi_m
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m > 0).astype(np.uint8) * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(1 + np.argmax(areas))
    return (labels == k).astype(np.uint8) * 255


def _mask_boundary(mask_u8: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    ker = np.ones((k, k), np.uint8)
    grad = cv2.morphologyEx((mask_u8 > 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, ker)
    b = (grad > 0).astype(np.float32)
    b_u8 = (b * 255).astype(np.uint8)
    b_u8 = cv2.erode(b_u8, np.ones((3, 3), np.uint8), iterations=1)
    return b_u8.astype(np.float32) / 255.0


def _contour_points_to_grid_mask(contour: np.ndarray, hw: tuple, grid_n: int, k: int) -> np.ndarray:
    h, w = hw
    pts = contour.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 4 or k <= 0:
        return np.zeros((grid_n, grid_n), dtype=np.uint8)
    d = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)], axis=0)
    total = float(s[-1] + 1e-8)
    targets = (np.linspace(0.0, total, num=k, endpoint=False) + total / max(k, 1) * 0.5).astype(np.float32)
    idx = np.clip(np.searchsorted(s, targets, side="left"), 0, pts.shape[0] - 1)
    sp = pts[idx]
    gx = np.clip((sp[:, 0] / max(1, w - 1)) * grid_n, 0, grid_n - 1).astype(np.int32)
    gy = np.clip((sp[:, 1] / max(1, h - 1)) * grid_n, 0, grid_n - 1).astype(np.int32)
    m = np.zeros((grid_n, grid_n), dtype=np.uint8)
    m[gy, gx] = 1
    return m


def _topk_minsep_mask(score: np.ndarray, k: int, *, min_sep: int = 0, forbid: Optional[np.ndarray] = None) -> np.ndarray:
    s = np.asarray(score, dtype=np.float32)
    h, w = s.shape
    if k <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    blocked = np.zeros((h, w), dtype=np.uint8)
    if forbid is not None:
        blocked = np.maximum(blocked, (np.asarray(forbid) > 0).astype(np.uint8))
    flat = s.reshape(-1)
    M = int(min(flat.size, max(k * 25, k + 10)))
    if M >= flat.size:
        cand = np.argsort(flat)[::-1]
    else:
        idx = np.argpartition(flat, -M)[-M:]
        cand = idx[np.argsort(flat[idx])[::-1]]
    out = np.zeros((h, w), dtype=np.uint8)
    r = int(max(0, min_sep))
    sel = 0
    for ii in cand.tolist():
        if sel >= k:
            break
        if flat[ii] <= 1e-8:
            break
        y = int(ii // w)
        x = int(ii - y * w)
        if blocked[y, x] != 0:
            continue
        out[y, x] = 1
        sel += 1
        if r > 0:
            y1, y2 = max(0, y - r), min(h, y + r + 1)
            x1, x2 = max(0, x - r), min(w, x + r + 1)
            blocked[y1:y2, x1:x2] = 1
    return out


def _topk_mask(score: np.ndarray, k: int, forbid: np.ndarray) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(score, dtype=np.uint8)
    s = score.copy()
    s[np.asarray(forbid, dtype=bool)] = -1.0
    flat = s.reshape(-1)
    if k >= flat.size:
        return (flat >= 0).astype(np.uint8).reshape(score.shape)
    idx = np.argpartition(flat, -k)[-k:]
    m = np.zeros_like(flat, dtype=np.uint8)
    m[idx] = 1
    return m.reshape(score.shape)


def _mask_bbox(mask_u8: np.ndarray) -> Optional[tuple]:
    ys, xs = np.where(np.asarray(mask_u8, dtype=np.uint8) > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0)


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / (union + 1e-8))


def _candidate_gray(gray_u8: np.ndarray, cand: EncodingCandidate, *, bilateral_d: int = 7) -> np.ndarray:
    g = np.asarray(gray_u8, dtype=np.uint8)
    mode = str(cand.contrast_mode)
    if mode == "clahe":
        g = clahe_u8(g, clip_limit=2.0, tile_grid=(8, 8))
    elif mode == "gamma":
        g = gamma_u8(g, cand.gamma)
    elif mode == "retinex":
        g = retinex_ssr_u8(g, sigma=30.0)
    if bool(cand.use_bilateral):
        try:
            g = cv2.bilateralFilter(g, d=int(bilateral_d), sigmaColor=55, sigmaSpace=7)
        except Exception:
            pass
    return np.asarray(g, dtype=np.uint8)


def _luminance_from_gray(gray_u8: np.ndarray) -> np.ndarray:
    lum = np.asarray(gray_u8, dtype=np.float32) / 255.0
    lum = cv2.GaussianBlur(lum, (0, 0), 1.0)
    lum = _robust_normalize(lum, 2.0, 98.0)
    lp = cv2.GaussianBlur(lum, (0, 0), 3.0)
    detail = np.abs(lum - lp).astype(np.float32)
    detail = _robust_normalize(detail, 2.0, 99.0)
    return _robust_normalize(0.65 * lum + 0.35 * detail, 1.0, 99.0).astype(np.float32)


def _compute_edges_local(img_bgr: np.ndarray, method: str = "canny_multi", low: int = 50, high: int = 150) -> np.ndarray:
    try:
        from phosphene_toolkit.perception.edges import compute_edges as _ce
        return _ce(img_bgr, method=method, low=low, high=high)
    except Exception:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return (cv2.Canny(gray, low, high) / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Core candidate builder
# ---------------------------------------------------------------------------

def _build_for_candidate(
    cand: Optional[EncodingCandidate],
    *,
    ctx: PerceptionContext,
    p: DotsParams,
    fusion_weights: Dict[str, float],
    object_mask_u8: Optional[np.ndarray],
    gate: np.ndarray,
    near: np.ndarray,
    motion_magnitude: Optional[np.ndarray],
    luminance: np.ndarray,
    gray_c: np.ndarray,
) -> Dict[str, Any]:
    """Build stimulation grid for one candidate configuration.

    ``cand=None`` means manual (sidebar settings) rather than adaptive.
    Returns a dict with: edges_map, motion_edges, attention_map, priority_obj,
    edge_g, lum_g, obj_g, near_g, stim_g, budget, edge_density, use_boundary.
    """
    grid_n = p.grid_n

    # Determine outline mode and parameters per candidate or manual settings
    if cand is None:
        outline_mode = "seg_boundary" if p.outline_source.startswith("Segmentation") else "pixel_edges"
        edge_method = "canny_multi"
        cand_fill = float(p.fill_weight)
        cand_bg = float(p.bg_suppress)
        gray_base = gray_c.astype(np.uint8)
        lum_map = luminance
        cand_use_bilateral = bool(p.use_bilateral)
    else:
        outline_mode = str(cand.outline_mode)
        edge_method = str(cand.edge_method)
        cand_fill = float(cand.fill_weight)
        cand_bg = float(cand.bg_suppress)
        gray_base = _candidate_gray(ctx.gray.astype(np.uint8), cand)
        lum_map = _luminance_from_gray(gray_base)
        cand_use_bilateral = bool(cand.use_bilateral)

    has_obj = object_mask_u8 is not None and int(np.sum(np.asarray(object_mask_u8) > 0)) > 50
    use_boundary = (outline_mode == "seg_boundary") and has_obj

    # --- Outline / edge map ---
    if use_boundary:
        edges_map_local = _mask_boundary(object_mask_u8, ksize=3)
        edges_map_local = np.clip(
            edges_map_local * (object_mask_u8.astype(np.float32) / 255.0), 0.0, 1.0
        ).astype(np.float32)
        if float(np.mean(edges_map_local > 0.05)) < 0.0015:
            use_boundary = False

    if not use_boundary:
        gray_for_edges = gray_base.astype(np.uint8)
        if cand is None and cand_use_bilateral:
            try:
                gray_for_edges = cv2.bilateralFilter(gray_for_edges, d=7, sigmaColor=55, sigmaSpace=7)
            except Exception:
                pass
        edges_map_local = _compute_edges_local(ctx.image, method=edge_method)
        edges_u8 = (np.clip(edges_map_local, 0, 1) * 255).astype(np.uint8)
        edges_u8 = cv2.morphologyEx(edges_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        edges_map_local = edges_u8.astype(np.float32) / 255.0
        pad = max(2, int(0.02 * min(edges_map_local.shape[0], edges_map_local.shape[1])))
        edges_map_local[:pad, :] = 0.0
        edges_map_local[-pad:, :] = 0.0
        edges_map_local[:, :pad] = 0.0
        edges_map_local[:, -pad:] = 0.0
        p_exp = 1.0 + 3.0 * float(cand_bg)
        gated = (0.02 + 0.98 * np.power(np.clip(gate, 0.0, 1.0), p_exp)).astype(np.float32)
        edges_map_local = np.clip(edges_map_local * gated, 0.0, 1.0).astype(np.float32)
        edges_map_local[gate < (0.10 + 0.20 * float(cand_bg))] *= (1.0 - 0.85 * float(cand_bg))

    # --- Attention map ---
    attention_map_local = fuse_ai_attention(
        ctx.image,
        saliency=ctx.saliency_map,
        segmentation=ctx.segmentation_fg if ctx.segmentation_fg is not None else ctx.segmentation_map,
        edges=edges_map_local,
        motion_magnitude=motion_magnitude,
        yolo_heatmap=ctx.yolo_heatmap,
        weights=fusion_weights,
    )

    # --- Motion edges ---
    if motion_magnitude is not None and motion_magnitude.size > 0:
        mot = np.clip(motion_magnitude.astype(np.float32), 0, 1)
        mot = (mot - mot.min()) / (mot.max() - mot.min() + 1e-8)
    else:
        mot = np.zeros_like(edges_map_local, dtype=np.float32)
    motion_edges_local = np.clip(edges_map_local * (1.0 + 2.0 * mot), 0, 1).astype(np.float32)
    motion_edges_local = cv2.GaussianBlur(motion_edges_local, (0, 0), 0.6)
    motion_edges_local = (motion_edges_local - motion_edges_local.min()) / (motion_edges_local.max() - motion_edges_local.min() + 1e-8)

    # --- Priority map ---
    priority_obj_local = attention_map_local.copy()
    if ctx.yolo_heatmap is not None and ctx.yolo_heatmap.size > 0:
        yh = np.clip(ctx.yolo_heatmap.astype(np.float32), 0, 1)
        yh = (yh - yh.min()) / (yh.max() - yh.min() + 1e-8)
        priority_obj_local = np.maximum(priority_obj_local, yh)
    priority_obj_local = (priority_obj_local - priority_obj_local.min()) / (priority_obj_local.max() - priority_obj_local.min() + 1e-8)
    priority_obj_local = cv2.GaussianBlur(priority_obj_local, (0, 0), 1.6)
    priority_obj_local = (priority_obj_local - priority_obj_local.min()) / (priority_obj_local.max() - priority_obj_local.min() + 1e-8)

    # --- Resize to electrode grid ---
    edge_g = cv2.resize(motion_edges_local, (grid_n, grid_n), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum_g = cv2.resize(lum_map, (grid_n, grid_n), interpolation=cv2.INTER_AREA).astype(np.float32)
    obj_g = cv2.resize(priority_obj_local, (grid_n, grid_n), interpolation=cv2.INTER_AREA).astype(np.float32)
    near_g = cv2.resize(near, (grid_n, grid_n), interpolation=cv2.INTER_AREA).astype(np.float32)
    edge_g = _robust_normalize(edge_g, 1.0, 99.0)
    lum_g = _robust_normalize(lum_g, 1.0, 99.0)
    near_g = _robust_normalize(near_g, 1.0, 99.0)

    lum_fg = np.clip(lum_g * (0.10 + 0.90 * obj_g), 0.0, 1.0)
    score_g = np.clip(
        (1.0 - float(cand_fill)) * edge_g + float(cand_fill) * lum_fg, 0.0, 1.0
    ).astype(np.float32)
    score_g = _robust_normalize(score_g, 1.0, 99.0)

    # --- Adaptive budget ---
    edge_density = float(np.mean(edge_g > 0.25))
    base_budget = int(grid_n * grid_n * float(p.dot_density))
    if cand is None:
        budget = int(np.clip(
            base_budget * (0.9 / (edge_density + 0.15)),
            grid_n * grid_n * 0.05, grid_n * grid_n * 0.16,
        ))
    else:
        budget = int(np.clip(
            base_budget * (1.05 / (edge_density + 0.35)),
            grid_n * grid_n * 0.07, grid_n * grid_n * 0.20,
        ))

    obj_budget = int(budget * float(p.subject_weight))
    near_budget = int(budget * float(p.near_weight))
    global_budget = max(0, budget - obj_budget - near_budget)

    used = np.zeros((grid_n, grid_n), dtype=np.uint8)
    if use_boundary and object_mask_u8 is not None:
        k_in = int(float(p.interior_frac) * budget)
        k_fg = int(float(p.fg_frac) * budget)
        k_outline = max(0, budget - k_in - k_fg)

        contours, _ = cv2.findContours(
            (object_mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours and k_outline > 0:
            contour = max(contours, key=lambda c: float(cv2.contourArea(c)))
            used = _contour_points_to_grid_mask(contour, (ctx.image.shape[0], ctx.image.shape[1]), grid_n, k_outline)
        elif k_outline > 0:
            used = _topk_minsep_mask(score_g, k_outline, min_sep=int(p.min_sep))

        mask_g = cv2.resize(
            (object_mask_u8.astype(np.float32) / 255.0), (grid_n, grid_n), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        mask_g = np.clip(mask_g, 0.0, 1.0)
        mask_bin = (mask_g > 0.25).astype(np.uint8) * 255

        if k_in > 0:
            inner = cv2.erode(mask_bin, np.ones((3, 3), np.uint8), iterations=1)
            inner_f = inner.astype(np.float32) / 255.0
            score_in = _robust_normalize(0.60 * obj_g + 0.40 * lum_g, 1.0, 99.0) * inner_f
            forbid = (used > 0).astype(np.uint8)
            used = np.maximum(used, _topk_minsep_mask(score_in, k_in, min_sep=int(p.min_sep), forbid=forbid))

        if k_fg > 0:
            outside = (mask_g < 0.15).astype(np.float32)
            score_fg = _robust_normalize(near_g, 1.0, 99.0) * outside
            forbid = (used > 0).astype(np.uint8)
            used = np.maximum(used, _topk_minsep_mask(score_fg, k_fg, min_sep=max(0, int(p.min_sep) - 1), forbid=forbid))
    else:
        m_obj = _topk_mask(score_g * (0.25 + 0.75 * obj_g), obj_budget, used)
        used = np.maximum(used, m_obj)
        m_near = _topk_mask(score_g * (0.30 + 0.70 * near_g), near_budget, used)
        used = np.maximum(used, m_near)
        m_global = _topk_mask(score_g, global_budget, used)
        used = np.maximum(used, m_global)

    score_combined = score_g * (0.75 * obj_g + 0.15 * near_g + 0.10)
    score_combined = np.clip(_robust_normalize(score_combined, 1.0, 99.0), 0.0, 1.0)
    stim_g = np.clip(score_combined * used.astype(np.float32), 0.0, 1.0).astype(np.float32)

    return {
        "edges_map": edges_map_local,
        "motion_edges": motion_edges_local,
        "attention_map": attention_map_local,
        "priority_obj": priority_obj_local,
        "edge_g": edge_g,
        "lum_g": lum_g,
        "obj_g": obj_g,
        "near_g": near_g,
        "stim_g": stim_g,
        "budget": int(budget),
        "edge_density": float(edge_density),
        "use_boundary": bool(use_boundary),
    }


# ---------------------------------------------------------------------------
# DotsBackend
# ---------------------------------------------------------------------------

class DotsBackend(PhospheneBackend):
    """Dots Baseline: adaptive 60×60 electrode grid + Gaussian dot renderer.

    This is the original pipeline from app.py, unchanged in behaviour,
    wrapped in the common PhospheneBackend interface.
    """

    name = "Dots Baseline"
    available = True
    unavailable_reason = ""

    def run(
        self,
        context: PerceptionContext,
        params: Dict[str, Any],
        prev_stim_grid: Optional[np.ndarray] = None,
    ) -> BackendResult:
        t_total = time.perf_counter()
        timings: Dict[str, float] = {}

        # Unpack parameters
        p = DotsParams(
            grid_n=int(params.get("grid_n", 60)),
            dot_density=float(params.get("dot_density", 0.12)),
            subject_weight=float(params.get("subject_weight", 0.50)),
            near_weight=float(params.get("near_weight", 0.22)),
            fill_weight=float(params.get("fill_weight", 0.18)),
            bg_suppress=float(params.get("bg_suppress", 0.75)),
            outline_source=str(params.get("outline_source", "Segmentation boundary (preferred)")),
            mask_kernel=int(params.get("mask_kernel", 7)),
            target_class=str(params.get("target_class", "auto")),
            interior_frac=float(params.get("interior_frac", 0.20)),
            fg_frac=float(params.get("fg_frac", 0.10)),
            min_sep=int(params.get("min_sep", 1)),
            use_adaptive=bool(params.get("use_adaptive", True)),
            adaptive_hold=int(params.get("adaptive_hold", 3)),
            adaptive_margin=float(params.get("adaptive_margin", 0.03)),
            adaptive_gate_thresh=float(params.get("adaptive_gate_thresh", 0.18)),
            dot_sigma=float(params.get("dot_sigma", 1.6)),
            dot_blend=str(params.get("dot_blend", "sum")),
            dot_jitter=float(params.get("dot_jitter", 0.0)),
            use_bilateral=bool(params.get("use_bilateral", True)),
            temporal_smooth=float(params.get("temporal_smooth", 0.55)),
        )

        # Task-preset fusion weights (may be overridden by caller)
        task_mode = context.task_mode
        if task_mode in TASK_PRESETS:
            fusion_weights = dict(TASK_PRESETS[task_mode].fusion_weights)
        else:
            fusion_weights = {"saliency": 2.4, "segmentation": 2.4, "edges": 0.25, "motion": 1.0, "yolo": 3.2}

        # Also allow per-call weight override
        if "fusion_weights" in params:
            fusion_weights = dict(params["fusion_weights"])

        # gray_clahe is the CLAHE-enhanced uint8 gray used in candidate building
        luminance = context.luminance
        gray_c = context.gray_clahe

        # --- Adaptive or manual selection ---
        t0 = time.perf_counter()
        has_obj = context.object_mask_u8 is not None and int(np.sum(np.asarray(context.object_mask_u8) > 0)) > 50

        adaptive_state = params.get("_adaptive_state", {})
        prev_choice = adaptive_state.get("choice_name")
        prev_score = adaptive_state.get("choice_score")
        stable_frames = int(adaptive_state.get("stable_frames", 0))

        chosen_res: Optional[Dict[str, Any]] = None
        adaptive_debug: Optional[Dict[str, Any]] = None

        if p.use_adaptive:
            candidates = default_candidates(has_object_boundary=bool(has_obj))
            fg_mask_g = grid_foreground_mask(
                gate=context.gate,
                object_mask_u8=context.object_mask_u8,
                grid_n=p.grid_n,
                gate_thresh=float(p.adaptive_gate_thresh),
            )
            boundary_mask_g = grid_boundary_mask(object_mask_u8=context.object_mask_u8, grid_n=p.grid_n) if has_obj else None

            cache: Dict[str, Dict[str, Any]] = {}
            scored: List[Tuple[EncodingCandidate, CandidateScore]] = []
            for cand in candidates:
                res = _build_for_candidate(
                    cand,
                    ctx=context,
                    p=p,
                    fusion_weights=fusion_weights,
                    object_mask_u8=context.object_mask_u8,
                    gate=context.gate,
                    near=context.near,
                    motion_magnitude=context.motion_magnitude,
                    luminance=luminance,
                    gray_c=gray_c,
                )
                cache[cand.name] = res
                cs = score_stim_grid(
                    stim_grid=res["stim_g"],
                    fg_mask_g=fg_mask_g,
                    boundary_mask_g=boundary_mask_g,
                    target_active_dots=int(res["budget"]),
                    prev_stim_grid=prev_stim_grid,
                )
                scored.append((cand, cs))

            chosen_c, chosen_s, switched = select_with_hysteresis(
                scored=scored,
                prev_choice_name=prev_choice,
                prev_score=prev_score,
                stable_frames=stable_frames,
                min_hold_frames=int(p.adaptive_hold),
                margin=float(p.adaptive_margin),
            )
            if switched:
                stable_frames = 0
            stable_frames += 1
            chosen_res = cache.get(chosen_c.name) or _build_for_candidate(
                chosen_c,
                ctx=context,
                p=p,
                fusion_weights=fusion_weights,
                object_mask_u8=context.object_mask_u8,
                gate=context.gate,
                near=context.near,
                motion_magnitude=context.motion_magnitude,
                luminance=luminance,
                gray_c=gray_c,
            )
            adaptive_debug = {
                "chosen": chosen_c.name,
                "score": float(chosen_s.score),
                "fg_energy": float(chosen_s.fg_energy),
                "boundary_energy": float(chosen_s.boundary_energy),
                "leak": float(chosen_s.leak_energy),
                "flicker": float(chosen_s.flicker),
                "stable_frames": stable_frames,
                "new_state": {
                    "choice_name": chosen_c.name,
                    "choice_score": float(chosen_s.score),
                    "stable_frames": stable_frames,
                },
                "candidates": [
                    {"name": c.name, "score": round(float(s.score), 4), "dots": int(s.active_dots)}
                    for c, s in sorted(scored, key=lambda t: float(t[1].score), reverse=True)
                ],
            }
        else:
            chosen_res = _build_for_candidate(
                None,
                ctx=context,
                p=p,
                fusion_weights=fusion_weights,
                object_mask_u8=context.object_mask_u8,
                gate=context.gate,
                near=context.near,
                motion_magnitude=context.motion_magnitude,
                luminance=luminance,
                gray_c=gray_c,
            )

        timings["encoding_ms"] = (time.perf_counter() - t0) * 1000.0

        stim_g: np.ndarray = chosen_res["stim_g"]

        # --- Temporal smoothing ---
        if prev_stim_grid is not None and float(p.temporal_smooth) > 1e-6:
            if np.asarray(prev_stim_grid).shape == stim_g.shape:
                a = float(np.clip(p.temporal_smooth, 0.0, 0.95))
                stim_g = (a * np.asarray(prev_stim_grid, dtype=np.float32) + (1.0 - a) * stim_g).astype(np.float32)

        # --- Render dots ---
        t0 = time.perf_counter()
        percept = render_dots_from_grid(
            stim_g,
            output_size=(context.image.shape[0], context.image.shape[1]),
            params=DotRenderParams(
                sigma_px=float(p.dot_sigma),
                blend=str(p.dot_blend),
                jitter_px=float(p.dot_jitter),
            ),
        )
        timings["render_ms"] = (time.perf_counter() - t0) * 1000.0
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0

        intermediate_maps = {
            "edges_map": chosen_res["edges_map"],
            "motion_edges": chosen_res["motion_edges"],
            "attention_map": chosen_res["attention_map"],
            "priority_obj": chosen_res["priority_obj"],
            "edge_g": chosen_res["edge_g"],
            "lum_g": chosen_res["lum_g"],
            "obj_g": chosen_res["obj_g"],
            "near_g": chosen_res["near_g"],
            "gate": context.gate,
            "near": context.near,
        }
        if context.saliency_map is not None:
            intermediate_maps["saliency"] = context.saliency_map
        if context.segmentation_map is not None:
            intermediate_maps["segmentation"] = context.segmentation_map
        if context.segmentation_fg is not None:
            intermediate_maps["segmentation_fg"] = context.segmentation_fg

        metadata: Dict[str, Any] = {
            "budget": int(chosen_res["budget"]),
            "edge_density": float(chosen_res["edge_density"]),
            "use_boundary": bool(chosen_res["use_boundary"]),
            "grid_n": p.grid_n,
            "dot_density": p.dot_density,
        }
        if adaptive_debug is not None:
            metadata["adaptive"] = adaptive_debug

        return BackendResult(
            backend_name=self.name,
            input_image=context.image,
            task_mode=context.task_mode,
            stimulation_grid=stim_g,
            phosphene_image=percept,
            intermediate_maps=intermediate_maps,
            timing_info=timings,
            metadata=metadata,
        )
