"""
Task-Based Retinal Prosthesis Simulation — Benchmarking Workbench
=================================================================
A modular Streamlit app for comparing phosphene encoder backends,
evaluating perceptual quality metrics, and exploring intermediate
feature maps for neuroprosthetic rendering research.

Architecture overview
---------------------
PerceptionContext (encoders/base.py)
    Pre-computed once per frame: segmentation, saliency, YOLO, edges,
    motion, fused foreground gate, near-field prior.

PhospheneBackend (encoders/base.py)
    Common interface; backends in encoders/:
      • DotsBackend       — adaptive 60×60 grid + Gaussian dot renderer
      • ToolkitBackend    — PhospheneEncoderTool.process_frame()
      • P2PRetinalBackend — pulse2percept AxonMap/Scoreboard (Argus II)
      • DynaphosBackend   — DynaphosModel + Orion cortical implant

BackendResult (encoders/base.py)
    Standard output: stimulation_grid, phosphene_image, intermediate_maps,
    timing_info, metadata.

Metrics (metrics/)
    SSIM, LPIPS, electrode efficiency proxy, radial coverage.

Tabs
    Demo | Compare | Evaluation | Diagnostics | Batch

Extending
---------
Add a new backend by subclassing PhospheneBackend and registering it in
encoders/registry.py.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Encoders + registry
# ---------------------------------------------------------------------------
from encoders.base import TASK_PRESETS, BackendResult, PerceptionContext
from encoders.registry import (
    BACKEND_NAMES,
    available_backend_names,
    get_backend,
    list_backends,
    set_toolkit_encoder,
)
from encoders.dots_backend import DotsParams

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from metrics import (
    LPIPS_AVAILABLE,
    compute_coverage_metrics,
    compute_electrode_metrics,
    compute_lpips,
    compute_ssim,
    radial_coverage_bins,
)

# ---------------------------------------------------------------------------
# Existing utils (unchanged)
# ---------------------------------------------------------------------------
from utils.adaptive_encoding import (
    CandidateScore,
    EncodingCandidate,
    clahe_u8,
    default_candidates,
    gamma_u8,
    grid_boundary_mask,
    grid_foreground_mask,
    retinex_ssr_u8,
    score_stim_grid,
    select_with_hysteresis,
)
from utils.ai_inference_fusion import (
    attention_to_heatmap_colored,
    fuse_ai_attention,
    yolo_detections_to_heatmap,
)
from utils.dot_phosphene_renderer import DotRenderParams, render_dots_from_grid

st.set_page_config(
    page_title="Retinal Prosthesis Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Cached model loaders
# ============================================================

@st.cache_resource
def load_policy():
    try:
        from phosphene_toolkit.task_policy import RuleBasedTaskPolicy
        return RuleBasedTaskPolicy()
    except Exception:
        return None


@st.cache_resource
def load_saliency():
    try:
        from phosphene_toolkit.perception.saliency import SpectralResidualSaliency
        return SpectralResidualSaliency()
    except Exception:
        return None


@st.cache_resource
def load_encoder():
    """Load PhospheneEncoderTool (used for segmentation + toolkit backend)."""
    try:
        from phosphene_toolkit import PhospheneEncoderTool
        config = {
            "device": {
                "grid_size": [60, 60], "amplitude_levels": 256,
                "max_amplitude_per_electrode": 1.0, "global_power_cap": 100.0,
                "spatial_spread_sigma": 1.2, "temporal_freq_hz": 20.0,
                "duty_cycle": 0.1, "dropout_rate": 0.0,
            },
            "observer": {
                "phosphene_size_mean": 2.0, "phosphene_size_std": 0.5,
                "elongation_factor": 1.5, "spatial_jitter_std": 0.3,
                "brightness_gamma": 0.8, "adaptation_rate": 0.1, "noise_level": 0.05,
            },
            "perception": {
                "segmentation_model": "deeplabv3_resnet50",
                "input_size": [480, 640], "fast_mode": False,
            },
            "fusion": {"allocation_strategy": "foveated", "max_active_phosphenes": 200},
        }
        enc = PhospheneEncoderTool(config=config)
        set_toolkit_encoder(enc)   # make available to ToolkitBackend
        return enc
    except Exception as e:
        st.warning(f"PhospheneEncoderTool unavailable: {e}")
        return None


@st.cache_resource
def load_depth():
    try:
        from phosphene_toolkit.perception.depth import DepthEstimator
        return DepthEstimator(model_type="MiDaS_small")
    except Exception:
        return None


# ============================================================
# Helpers
# ============================================================

def _robust_normalize(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.percentile(x, p_lo)), float(np.percentile(x, p_hi))
    if hi - lo < 1e-8:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _binary_fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood) & cv2.bitwise_not(m)
    return ((m | holes) > 0).astype(np.uint8) * 255


def _keep_largest_component(mask_u8: np.ndarray, *, roi=None) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(m.shape[1], int(x2)), min(m.shape[0], int(y2))
        tmp = np.zeros_like(m)
        tmp[y1:y2, x1:x2] = m[y1:y2, x1:x2]
        m = tmp
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m > 0).astype(np.uint8) * 255
    k = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == k).astype(np.uint8) * 255


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / (union + 1e-8))


def _mask_bbox(mask_u8: np.ndarray) -> Optional[tuple]:
    ys, xs = np.where(np.asarray(mask_u8) > 0)
    if ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0)


def run_yolo_labeler(img, conf=0.25):
    try:
        from utils.detection import run_yolo_detection
        return run_yolo_detection(img, conf_threshold=conf)
    except Exception:
        return img.copy(), []


def compute_edges(img, method="canny_multi", low=50, high=150):
    try:
        from phosphene_toolkit.perception.edges import compute_edges as _ce
        return _ce(img, method=method, low=low, high=high)
    except Exception:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (cv2.Canny(gray, low, high) / 255.0).astype(np.float32)


# ============================================================
# Perception context builder
# ============================================================

def build_perception_context(
    img: np.ndarray,
    *,
    enc,
    sal,
    depth_est,
    is_video: bool,
    frames: List[np.ndarray],
    frame_idx: int,
    task_mode: str,
    mask_kernel: int,
    target_class: str,
) -> PerceptionContext:
    """Compute all shared perception outputs for *img*."""

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Luminance with CLAHE + local contrast
    try:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe_obj.apply(gray)
    except Exception:
        gray_clahe = gray
    lum = _robust_normalize(gray_clahe.astype(np.float32) / 255.0, 2.0, 98.0)
    lum_blur = cv2.GaussianBlur(lum, (0, 0), 3.0)
    detail = _robust_normalize(np.abs(lum - lum_blur), 2.0, 99.0)
    luminance = _robust_normalize(0.65 * lum + 0.35 * detail, 1.0, 99.0)

    # Saliency
    saliency_map = None
    if sal:
        try:
            saliency_map = sal.compute_saliency(img, method="combined")
        except Exception:
            pass

    # Segmentation via toolkit encoder
    segmentation_map: Optional[np.ndarray] = None
    segmentation_fg: Optional[np.ndarray] = None
    pred_full: Optional[np.ndarray] = None
    if enc is not None and hasattr(enc, "segmentation"):
        try:
            seg_result = enc.segmentation.segment(img)
            segmentation_map = seg_result.get("segmentation")
            pm = seg_result.get("pred_mask")
            if pm is not None:
                pred_full = cv2.resize(pm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                segmentation_fg = cv2.GaussianBlur(
                    (pred_full > 0).astype(np.float32), (0, 0), 1.2
                )
        except Exception:
            pass

    # YOLO
    annotated_yolo, detections = run_yolo_labeler(img, conf=0.25)
    yolo_heatmap: Optional[np.ndarray] = None
    primary_det: Optional[Dict] = None
    yolo_roi: Optional[tuple] = None
    if detections:
        det_sorted = sorted(detections, key=lambda d: float(d.get("conf", 0)), reverse=True)[:3]
        yolo_heatmap = yolo_detections_to_heatmap(det_sorted, img.shape[:2], sigma=60.0)
        primary_det = det_sorted[0]

    # Object mask
    object_mask_u8: Optional[np.ndarray] = None
    if pred_full is not None and pred_full.size > 0:
        class_id = None
        if target_class != "foreground":
            try:
                from phosphene_toolkit.perception.segmentation import COCO_LABELS
                n2i = {n: i for i, n in enumerate(COCO_LABELS)}
                if target_class == "auto" and primary_det is not None:
                    class_id = n2i.get(str(primary_det.get("class", "")).strip().lower())
                elif target_class in n2i:
                    class_id = n2i.get(target_class)
            except Exception:
                pass
        if class_id is not None:
            object_mask_u8 = ((pred_full == int(class_id)).astype(np.uint8) * 255)
            if int(np.sum(object_mask_u8 > 0)) < 100:
                object_mask_u8 = ((pred_full > 0).astype(np.uint8) * 255)
        else:
            object_mask_u8 = ((pred_full > 0).astype(np.uint8) * 255)

        # Restrict to YOLO ROI
        roi = None
        if primary_det is not None:
            x1, y1, x2, y2 = [float(v) for v in primary_det.get("bbox", (0, 0, w, h))[:4]]
            mx, my = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            roi = (x1 - mx, y1 - my, x2 + mx, y2 + my)
            yolo_roi = roi
            rx1, ry1 = max(0, int(roi[0])), max(0, int(roi[1]))
            rx2, ry2 = min(w, int(roi[2])), min(h, int(roi[3]))
            tmp = np.zeros_like(object_mask_u8)
            tmp[ry1:ry2, rx1:rx2] = object_mask_u8[ry1:ry2, rx1:rx2]
            object_mask_u8 = tmp

        k = max(3, int(mask_kernel) | 1)
        ker = np.ones((k, k), np.uint8)
        object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_CLOSE, ker, iterations=1)
        object_mask_u8 = _binary_fill_holes(object_mask_u8)
        object_mask_u8 = cv2.morphologyEx(
            object_mask_u8, cv2.MORPH_OPEN,
            np.ones((max(3, k // 2) | 1, max(3, k // 2) | 1), np.uint8), iterations=1
        )
        object_mask_u8 = _keep_largest_component(object_mask_u8, roi=yolo_roi)

        # Drift guard
        if yolo_roi is not None and int(np.sum(object_mask_u8 > 0)) > 50:
            mb = _mask_bbox(object_mask_u8)
            if mb is not None and _bbox_iou(mb, yolo_roi) < 0.08:
                rx1, ry1 = max(0, int(yolo_roi[0])), max(0, int(yolo_roi[1]))
                rx2, ry2 = min(w, int(yolo_roi[2])), min(h, int(yolo_roi[3]))
                tmp = np.zeros_like(object_mask_u8)
                tmp[ry1:ry2, rx1:rx2] = 255
                object_mask_u8 = tmp

    # Fallback mask from YOLO bbox + gate
    gate = np.zeros((h, w), dtype=np.float32)
    if segmentation_fg is not None:
        gate = np.maximum(gate, np.clip(segmentation_fg, 0, 1))
    if yolo_heatmap is not None:
        gate = np.maximum(gate, _robust_normalize(yolo_heatmap, 1.0, 99.0))
    if saliency_map is not None:
        sm = cv2.resize(saliency_map.astype(np.float32).squeeze(), (w, h))
        gate = np.maximum(gate, _robust_normalize(sm, 2.0, 98.0))
    gate = _robust_normalize(cv2.GaussianBlur(np.clip(gate, 0, 1), (0, 0), 2.0), 1.0, 99.0)

    if object_mask_u8 is None and primary_det is not None:
        try:
            x1, y1, x2, y2 = [float(v) for v in primary_det.get("bbox", (0, 0, w, h))[:4]]
            mx, my = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
            bx1, by1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            bx2, by2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
            bb = np.zeros((h, w), dtype=np.uint8)
            bb[by1:by2, bx1:bx2] = 255
            gm = (gate > 0.12).astype(np.uint8) * 255
            object_mask_u8 = cv2.morphologyEx(
                cv2.bitwise_and(bb, gm), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
            )
            object_mask_u8 = _binary_fill_holes(object_mask_u8)
            object_mask_u8 = _keep_largest_component(object_mask_u8, roi=(bx1, by1, bx2, by2))
            yolo_roi = (bx1, by1, bx2, by2)
        except Exception:
            pass

    # Edges
    edges_map = compute_edges(img)

    # Motion
    motion_magnitude: Optional[np.ndarray] = None
    motion_result: Optional[Dict] = None
    if is_video and len(frames) > 1 and frame_idx > 0:
        try:
            from phosphene_toolkit.perception.motion import compute_motion_between_frames
            motion_result = compute_motion_between_frames(frames[frame_idx - 1], frames[frame_idx])
            motion_magnitude = motion_result.get("magnitude")
        except Exception:
            pass

    # Near-field prior (bottom region + optional depth)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    near = np.power(np.clip(yy, 0, 1), 2.2)
    if depth_est is not None:
        try:
            depth_res = depth_est.predict(img, output_size=(h, w))
            near = np.maximum(near, depth_res.inv_depth.astype(np.float32))
        except Exception:
            pass
    near = cv2.GaussianBlur(near, (0, 0), 3.0)
    near = (near - near.min()) / (near.max() - near.min() + 1e-8)

    return PerceptionContext(
        image=img,
        gray=gray,
        gray_clahe=gray_clahe,
        luminance=luminance,
        saliency_map=saliency_map,
        segmentation_map=segmentation_map,
        segmentation_fg=segmentation_fg,
        pred_full=pred_full,
        object_mask_u8=object_mask_u8,
        detections=detections,
        annotated_yolo=annotated_yolo,
        yolo_heatmap=yolo_heatmap,
        primary_det=primary_det,
        yolo_roi=yolo_roi,
        edges_map=edges_map,
        motion_magnitude=motion_magnitude,
        motion_result=motion_result,
        gate=gate,
        near=near,
        task_mode=task_mode,
        task_preset=TASK_PRESETS.get(task_mode),
    )


# ============================================================
# Sidebar
# ============================================================

def render_sidebar() -> Dict[str, Any]:
    """Render the sidebar and return all UI parameter values."""
    with st.sidebar:
        st.header("Workbench Settings")

        # ---- Backend selection ----
        st.subheader("Encoder Backends")
        avail_names = available_backend_names()
        all_backends_info = list_backends()

        # Show unavailability warnings
        unavailable = [n for n in BACKEND_NAMES if not all_backends_info[n][1]]
        if unavailable:
            with st.expander("Unavailable backends", expanded=False):
                for name in unavailable:
                    reason = all_backends_info[name][2]
                    st.caption(f"**{name}**: {reason}")

        primary_backend = st.selectbox(
            "Primary backend",
            avail_names,
            index=0,
            help="Backend used in the Demo and Evaluation tabs.",
        )

        enable_comparison = st.checkbox("Enable A/B comparison", value=False)
        secondary_backend = primary_backend
        if enable_comparison:
            secondary_opts = [n for n in avail_names if n != primary_backend] or avail_names
            secondary_backend = st.selectbox(
                "Secondary backend (B)",
                secondary_opts,
                index=0,
            )

        # ---- Task mode ----
        st.subheader("Task Mode")
        task_mode = st.selectbox(
            "Task preset",
            list(TASK_PRESETS.keys()),
            index=0,
            help="Adjusts cue fusion weights and electrode budget allocation.",
        )
        if task_mode in TASK_PRESETS:
            st.caption(TASK_PRESETS[task_mode].description)

        # ---- Calibration (Dots baseline) ----
        dots_params: Dict[str, Any] = {}
        if "Dots Baseline" in (primary_backend, secondary_backend):
            with st.expander("Dots: Calibration", expanded=False):
                preset = TASK_PRESETS.get(task_mode)
                dots_params["dot_density"] = st.slider(
                    "Dot density", 0.05, 0.22, 0.12, 0.01,
                    help="Overall % of electrodes activated (approx).",
                )
                dots_params["subject_weight"] = st.slider(
                    "Subject priority", 0.30, 0.70,
                    float(preset.subject_weight) if preset else 0.50, 0.05,
                )
                dots_params["near_weight"] = st.slider(
                    "Near-field priority", 0.10, 0.55,
                    float(preset.near_weight) if preset else 0.22, 0.05,
                )
                dots_params["fill_weight"] = st.slider(
                    "Interior fill (luminance)", 0.0, 0.45,
                    float(preset.fill_weight) if preset else 0.18, 0.01,
                )
                dots_params["bg_suppress"] = st.slider(
                    "Background suppression", 0.0, 1.0,
                    float(preset.bg_suppress) if preset else 0.75, 0.05,
                )
                dots_params["temporal_smooth"] = st.slider(
                    "Temporal smoothing (video)", 0.0, 0.90, 0.55, 0.05,
                )
                dots_params["mask_kernel"] = st.slider("Mask cleanup kernel", 3, 11, 7, 2)
                dots_params["target_class"] = st.selectbox(
                    "Segmentation target",
                    ["auto", "dog", "person", "cat", "foreground"],
                    index=0,
                )
                dots_params["interior_frac"] = st.slider("Interior dot fraction", 0.0, 0.60, 0.20, 0.05)
                dots_params["fg_frac"] = st.slider("Foreground dot fraction", 0.0, 0.40, 0.10, 0.05)
                dots_params["min_sep"] = st.slider("Min dot spacing (grid)", 0, 4, 1, 1)

            with st.expander("Dots: Adaptive encoding", expanded=False):
                dots_params["use_adaptive"] = True
                dots_params["adaptive_hold"] = st.slider("Hold frames (video)", 0, 10, 3, 1)
                dots_params["adaptive_margin"] = st.slider("Switch margin", 0.0, 0.15, 0.03, 0.01)
                dots_params["adaptive_gate_thresh"] = st.slider("Foreground gate threshold", 0.05, 0.60, 0.18, 0.05)

            with st.expander("Dots: Renderer", expanded=False):
                dots_params["dot_sigma"] = st.slider("Dot σ (px)", 0.8, 3.5, 1.6, 0.1)
                dots_params["dot_blend"] = st.selectbox("Dot blend", ["sum", "max"], index=0)
                dots_params["dot_jitter"] = st.slider("Dot jitter (px)", 0.0, 1.5, 0.0, 0.1)

            with st.expander("Dots: Edge prefilter", expanded=False):
                dots_params["use_bilateral"] = st.checkbox("Bilateral filter before edges", value=True)
                dots_params["outline_source"] = st.selectbox(
                    "Outline source",
                    ["Segmentation boundary (preferred)", "Pixel edges (fallback)"],
                    index=0,
                )

        # ---- pulse2percept model selector ----
        p2p_params: Dict[str, Any] = {}
        if "p2p Retinal (Argus II)" in (primary_backend, secondary_backend):
            with st.expander("p2p Retinal options", expanded=False):
                p2p_params["p2p_model"] = st.selectbox(
                    "Perceptual model", ["axon_map", "scoreboard"], index=0,
                    help="AxonMap: biologically accurate axon streaks. Scoreboard: simpler blobs.",
                )

        # ---- Dynaphos options ----
        dynaphos_params: Dict[str, Any] = {}
        if "Dynaphos Cortical (p2p)" in (primary_backend, secondary_backend):
            with st.expander("Dynaphos options", expanded=False):
                from encoders.dynaphos_backend import PREPROCESS_MODES
                dynaphos_params["dynaphos_preprocess"] = st.selectbox(
                    "Preprocessing", PREPROCESS_MODES, index=0,
                    help="How to convert the image to a cortical stimulation pattern.",
                )

        # ---- Perception ----
        with st.expander("Perception / diagnostics", expanded=False):
            mask_kernel = dots_params.get("mask_kernel", 7)
            target_class = dots_params.get("target_class", "auto")
            show_candidate_table = st.checkbox("Show adaptive candidates table", value=False)

    return {
        "primary_backend": primary_backend,
        "secondary_backend": secondary_backend,
        "enable_comparison": enable_comparison,
        "task_mode": task_mode,
        "dots_params": dots_params,
        "p2p_params": p2p_params,
        "dynaphos_params": dynaphos_params,
        "mask_kernel": int(dots_params.get("mask_kernel", 7)),
        "target_class": str(dots_params.get("target_class", "auto")),
        "show_candidate_table": show_candidate_table,
    }


# ============================================================
# Build backend params dict
# ============================================================

def _make_backend_params(backend_name: str, ui: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"task_mode": ui["task_mode"]}
    if backend_name == "Dots Baseline":
        params.update(ui["dots_params"])
    elif backend_name == "p2p Retinal (Argus II)":
        params.update(ui["p2p_params"])
    elif backend_name == "Dynaphos Cortical (p2p)":
        params.update(ui["dynaphos_params"])
    return params


# ============================================================
# Session-state helpers for temporal smoothing
# ============================================================

def _get_prev_stim(backend_name: str) -> Optional[np.ndarray]:
    key = f"_prev_stim_{backend_name}"
    return st.session_state.get(key)


def _set_prev_stim(backend_name: str, stim: Optional[np.ndarray]) -> None:
    key = f"_prev_stim_{backend_name}"
    if stim is not None:
        st.session_state[key] = stim.copy()
    else:
        st.session_state.pop(key, None)


def _get_adaptive_state(backend_name: str) -> Dict[str, Any]:
    return st.session_state.get(f"_adaptive_{backend_name}", {})


def _set_adaptive_state(backend_name: str, state: Dict[str, Any]) -> None:
    st.session_state[f"_adaptive_{backend_name}"] = state


def _reset_temporal_state(upload_id: str) -> None:
    if st.session_state.get("_upload_id") != upload_id:
        st.session_state["_upload_id"] = upload_id
        for key in list(st.session_state.keys()):
            if key.startswith("_prev_stim") or key.startswith("_adaptive"):
                del st.session_state[key]


# ============================================================
# Run one backend
# ============================================================

def run_backend(
    backend_name: str,
    context: PerceptionContext,
    ui: Dict[str, Any],
) -> BackendResult:
    backend = get_backend(backend_name)
    params = _make_backend_params(backend_name, ui)
    params["_adaptive_state"] = _get_adaptive_state(backend_name)
    prev_stim = _get_prev_stim(backend_name)

    result = backend.run(context, params, prev_stim_grid=prev_stim)

    # Persist state
    if result.stimulation_grid is not None:
        _set_prev_stim(backend_name, result.stimulation_grid)
    meta = result.metadata
    if "adaptive" in meta and "new_state" in meta["adaptive"]:
        _set_adaptive_state(backend_name, meta["adaptive"]["new_state"])

    return result


# ============================================================
# Tab renderers
# ============================================================

def render_demo_tab(ctx: PerceptionContext, result: BackendResult, ui: Dict[str, Any]) -> None:
    """Demo tab: shows the attention/gate map and stimulation grid as introspection."""
    st.caption("The top panel above shows Input | Motion/Edges | Phosphene in real time.")
    st.divider()

    # Stimulation grid heatmap (electrode-level view)
    col_stim, col_gate = st.columns(2)
    with col_stim:
        st.subheader("Stimulation grid")
        if result.stimulation_grid is not None:
            sg = np.clip(result.stimulation_grid, 0, 1)
            sg_vis = cv2.applyColorMap((sg * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            st.image(cv2.cvtColor(sg_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Grid: {sg.shape[0]}×{sg.shape[1]}")
        else:
            st.info("No stimulation grid for this backend.")

    with col_gate:
        st.subheader("Foreground gate")
        gate_vis = cv2.applyColorMap((np.clip(ctx.gate, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        st.image(cv2.cvtColor(gate_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("High values = foreground / salient region")

    # Adaptive candidates table
    if ui.get("show_candidate_table") and "adaptive" in result.metadata:
        cands = result.metadata["adaptive"].get("candidates", [])
        if cands:
            st.caption(f"Adaptive choice: `{result.metadata['adaptive']['chosen']}` "
                       f"(stable: {result.metadata['adaptive']['stable_frames']} frames)")
            st.dataframe(cands, use_container_width=True, height=200)

    # Sidebar frame metrics
    with st.sidebar:
        with st.expander("Frame metrics", expanded=False):
            if result.stimulation_grid is not None:
                em = compute_electrode_metrics(result.stimulation_grid)
                st.write({
                    "active_electrodes": int(em["active_count"]),
                    "active_%": round(100.0 * em["active_ratio"], 1),
                    "efficiency_proxy": round(em["efficiency_proxy"], 3),
                    "flicker": round(em["flicker"], 3),
                })
            for k, v in result.timing_info.items():
                if "ms" in k:
                    st.caption(f"{k}: {v:.1f} ms")


def render_compare_tab(
    ctx: PerceptionContext,
    result_a: BackendResult,
    result_b: BackendResult,
    ui: Dict[str, Any],
) -> None:
    img_rgb = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)

    st.subheader("A/B Comparison")
    c_img, c_a, c_b = st.columns(3)
    with c_img:
        st.caption("Input")
        st.image(img_rgb, channels="RGB", use_container_width=True)
    with c_a:
        st.caption(f"A: {result_a.backend_name}")
        if result_a.error:
            st.error(result_a.error)
        else:
            st.image(result_a.phosphene_image, channels="GRAY", use_container_width=True)
    with c_b:
        st.caption(f"B: {result_b.backend_name}")
        if result_b.error:
            st.error(result_b.error)
        else:
            st.image(result_b.phosphene_image, channels="GRAY", use_container_width=True)

    # Metric table
    st.divider()
    st.subheader("Metrics")
    rows = []
    for label, res in [("A", result_a), ("B", result_b)]:
        row: Dict[str, Any] = {"Backend": f"{label}: {res.backend_name}"}
        if res.stimulation_grid is not None:
            em = compute_electrode_metrics(res.stimulation_grid)
            row["Active %"] = round(100.0 * em["active_ratio"], 1)
            row["Efficiency"] = round(em["efficiency_proxy"], 3)
            row["Flicker"] = round(em["flicker"], 3)
        if not res.error:
            cm = compute_coverage_metrics(res.phosphene_image)
            row["Coverage %"] = round(100.0 * cm["global_coverage"], 1)
        row["Latency (ms)"] = round(res.timing_info.get("total_ms", 0), 1)
        rows.append(row)

    if result_a.stimulation_grid is not None and result_b.stimulation_grid is not None:
        if result_a.phosphene_image.shape == result_b.phosphene_image.shape:
            ssim_val = compute_ssim(result_a.phosphene_image, result_b.phosphene_image)
            st.caption(f"SSIM (A vs B): **{ssim_val:.4f}** (1 = identical)")
            if LPIPS_AVAILABLE:
                lpips_val = compute_lpips(result_a.phosphene_image, result_b.phosphene_image)
                if lpips_val is not None:
                    st.caption(f"LPIPS (A vs B): **{lpips_val:.4f}** (0 = identical)")

    st.dataframe(rows, use_container_width=True)


def render_evaluation_tab(result: BackendResult, ctx: PerceptionContext) -> None:
    st.subheader("Evaluation Metrics")

    if result.error:
        st.error(f"Backend failed: {result.error}")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Electrode metrics**")
        if result.stimulation_grid is not None:
            em = compute_electrode_metrics(result.stimulation_grid)
            st.metric("Active electrodes", int(em["active_count"]))
            st.metric("Active %", f"{100.0*em['active_ratio']:.1f}%")
            st.metric("Efficiency proxy", f"{em['efficiency_proxy']:.3f}")
            st.metric("Redundancy", f"{em['redundancy']:.3f}")
            st.metric("Flicker", f"{em['flicker']:.3f}")
            with st.expander("What is efficiency proxy?", expanded=False):
                st.markdown(
                    "**Formula**: `efficiency = (1 − active_ratio) × (1 − redundancy)`\n\n"
                    "- `active_ratio` = fraction of electrodes that are active.\n"
                    "- `redundancy` = mean fraction of 8-connected neighbours that are also active "
                    "(penalises clumped electrodes).\n"
                    "- High efficiency: sparse, well-separated activations."
                )
        else:
            st.info("No stimulation grid available for this backend.")

    with c2:
        st.markdown("**Phosphene coverage**")
        cm = compute_coverage_metrics(result.phosphene_image)
        st.metric("Global coverage", f"{100.0*cm['global_coverage']:.1f}%")
        st.metric("Lit pixels", cm["lit_pixels"])
        radial = cm["radial"]
        st.metric("Coverage-weighted eccentricity",
                  f"{radial['mean_eccentricity']:.3f} (0=centre, 1=corner)")

        # Radial bar chart
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 2.5))
            xs = radial["bin_centers"]
            ys = [v * 100 for v in radial["bin_coverage"]]
            ax.bar(xs, ys, width=0.09, color="steelblue", alpha=0.8)
            ax.set_xlabel("Eccentricity (0=centre)")
            ax.set_ylabel("Coverage %")
            ax.set_title("Radial coverage")
            ax.set_ylim(0, 100)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception:
            st.dataframe({"eccentricity": radial["bin_centers"], "coverage": radial["bin_coverage"]})

    st.divider()

    # Image quality metrics vs input
    st.subheader("Image-quality vs. input (phosphene ↔ input)")
    st.caption(
        "These numbers compare the phosphene percept against the original image. "
        "They reflect structural similarity, not perceptual quality — use them for "
        "relative comparison between backends, not absolute quality assessment."
    )
    ssim_val = compute_ssim(result.phosphene_image, ctx.image)
    c3, c4 = st.columns(2)
    with c3:
        st.metric("SSIM (percept vs. input)", f"{ssim_val:.4f}", help="1 = identical structure")
    with c4:
        if LPIPS_AVAILABLE:
            lpips_val = compute_lpips(result.phosphene_image, ctx.image)
            if lpips_val is not None:
                st.metric("LPIPS (percept vs. input)", f"{lpips_val:.4f}", help="0 = identical percept")
            else:
                st.info("LPIPS computation failed.")
        else:
            st.info("Install `lpips` for LPIPS metrics: `pip install lpips`")

    st.divider()

    # Timing
    st.subheader("Latency breakdown")
    timing_data = {k: round(v, 2) for k, v in result.timing_info.items()}
    st.json(timing_data)

    # Coverage heatmap
    st.subheader("Coverage heatmap")
    hm = (cm["coverage_heatmap"] * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_HOT)
    st.image(cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    st.caption("Bright = denser phosphene coverage")


def render_diagnostics_tab(ctx: PerceptionContext, result: BackendResult) -> None:
    """Show all intermediate introspection maps."""
    st.subheader("Diagnostics / Introspection Maps")

    named_maps = {
        "Saliency": ctx.saliency_map,
        "Segmentation FG": ctx.segmentation_fg,
        "Foreground Gate": ctx.gate,
        "Near-field": ctx.near,
        "Edges (raw)": ctx.edges_map,
    }
    # Add backend intermediates
    for k in ("motion_edges", "attention_map", "priority_obj", "edge_g", "stim_g"):
        v = result.intermediate_maps.get(k)
        if v is not None:
            named_maps[k.replace("_", " ").title()] = v

    # Depth
    if "near" in result.intermediate_maps:
        named_maps["Near-field (refined)"] = result.intermediate_maps["near"]

    available_maps = {k: v for k, v in named_maps.items() if v is not None and np.asarray(v).size > 0}
    if not available_maps:
        st.info("No intermediate maps available.")
        return

    cols_per_row = 3
    keys = list(available_maps.keys())
    for row_start in range(0, len(keys), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, key in enumerate(keys[row_start:row_start + cols_per_row]):
            arr = available_maps[key]
            arr_f = np.asarray(arr, dtype=np.float32)
            if arr_f.ndim == 3:
                arr_f = arr_f.mean(axis=2)
            arr_f = np.clip(arr_f, 0, 1)
            if arr_f.max() <= 1.0:
                arr_f = arr_f * 255.0
            arr_u8 = arr_f.astype(np.uint8)
            colored = cv2.applyColorMap(cv2.resize(arr_u8, (256, 256)), cv2.COLORMAP_INFERNO)
            with cols[i]:
                st.caption(key)
                st.image(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    # YOLO tab
    st.divider()
    st.subheader("YOLO Detections")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, key="diag_yolo_conf")
    annotated, detections = run_yolo_labeler(ctx.image, conf=conf_thresh)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    if detections:
        for d in detections:
            st.caption(f"**{d['class']}** ({d['conf']:.2f}) bbox {[round(v, 1) for v in d['bbox']]}")


def render_batch_tab(ui: Dict[str, Any], enc, sal, depth_est) -> None:
    """Upload multiple images, run selected backend, export metrics as CSV/JSON."""
    st.subheader("Batch Evaluation")
    st.caption("Upload multiple images; selected backend is run on each and metrics are aggregated.")

    uploaded_batch = st.file_uploader(
        "Upload images (multi-select)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if not uploaded_batch:
        st.info("Upload images above to run batch evaluation.")
        return

    backend_name = ui["primary_backend"]
    if st.button(f"Run batch ({len(uploaded_batch)} images) with **{backend_name}**"):
        results_rows = []
        progress = st.progress(0)
        for i, file in enumerate(uploaded_batch):
            arr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            ctx = build_perception_context(
                img, enc=enc, sal=sal, depth_est=depth_est,
                is_video=False, frames=[img], frame_idx=0,
                task_mode=ui["task_mode"],
                mask_kernel=ui["mask_kernel"],
                target_class=ui["target_class"],
            )
            backend = get_backend(backend_name)
            params = _make_backend_params(backend_name, ui)
            t0 = time.perf_counter()
            res = backend.run(ctx, params)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            row: Dict[str, Any] = {"filename": file.name, "backend": backend_name, "task": ui["task_mode"]}
            if res.stimulation_grid is not None:
                em = compute_electrode_metrics(res.stimulation_grid)
                row.update({k: round(v, 4) for k, v in em.items() if isinstance(v, float)})
            if not res.error:
                cm = compute_coverage_metrics(res.phosphene_image)
                row["global_coverage"] = round(cm["global_coverage"], 4)
                row["mean_eccentricity"] = round(cm["radial"]["mean_eccentricity"], 4)
                ssim_val = compute_ssim(res.phosphene_image, img)
                row["ssim_vs_input"] = round(ssim_val, 4)
            row["latency_ms"] = round(latency_ms, 2)
            row["error"] = res.error or ""
            results_rows.append(row)
            progress.progress((i + 1) / len(uploaded_batch))

        if results_rows:
            st.success(f"Completed {len(results_rows)} images.")
            st.dataframe(results_rows, use_container_width=True)

            # Export CSV
            import csv
            csv_buf = io.StringIO()
            if results_rows:
                writer = csv.DictWriter(csv_buf, fieldnames=list(results_rows[0].keys()))
                writer.writeheader()
                writer.writerows(results_rows)
            st.download_button("Download CSV", csv_buf.getvalue(), file_name="batch_results.csv")

            # Export JSON
            json_str = json.dumps(results_rows, indent=2)
            st.download_button("Download JSON", json_str, file_name="batch_results.json")


# ============================================================
# Main
# ============================================================

def main() -> None:
    st.title("Retinal Prosthesis Simulation Workbench")
    st.caption("Multi-backend encoder comparison, quantitative evaluation, and introspection")

    ui = render_sidebar()

    # Pre-load models
    enc = load_encoder()
    sal = load_saliency()
    depth_est = load_depth()

    # File uploader
    uploaded = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "gif"],
    )
    if uploaded is None:
        st.info("Upload an image or video to get started.")
        return

    # Reset temporal state on new file
    upload_id = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
    _reset_temporal_state(upload_id)

    # Load media
    is_video = uploaded.type.startswith("video") or uploaded.name.lower().endswith((".mp4", ".avi", ".mov", ".gif"))
    frames: List[np.ndarray] = []
    frame_idx = 0
    img: Optional[np.ndarray] = None

    if is_video:
        bytes_data = uploaded.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(bytes_data)
            path = f.name
        try:
            cap = cv2.VideoCapture(path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
        if not frames:
            st.warning("No frames found in video.")
            return
        frame_idx = st.slider("Frame", 0, len(frames) - 1, 0)
        img = frames[frame_idx]
    else:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not decode image/video.")
        return

    # ── Show input immediately (before any slow processing) ─────────────────
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    top_col_in, top_col_mid, top_col_out = st.columns(3)
    with top_col_in:
        st.subheader("Input")
        st.image(img_rgb, use_container_width=True)
    # mid and out columns are filled in after processing (see placeholders below)
    mid_placeholder = top_col_mid.empty()
    out_placeholder = top_col_out.empty()

    # ── Build perception context (shared across all backends) ────────────────
    with st.spinner("Running perception pipeline…"):
        ctx = build_perception_context(
            img,
            enc=enc,
            sal=sal,
            depth_est=depth_est,
            is_video=is_video,
            frames=frames,
            frame_idx=frame_idx,
            task_mode=ui["task_mode"],
            mask_kernel=ui["mask_kernel"],
            target_class=ui["target_class"],
        )

    # ── Run primary backend ──────────────────────────────────────────────────
    with st.spinner(f"Running {ui['primary_backend']}…"):
        result_a = run_backend(ui["primary_backend"], ctx, ui)

    # ── Fill in the top-row placeholders ────────────────────────────────────
    with mid_placeholder.container():
        st.subheader("Motion / Edges")
        motion_edges = result_a.intermediate_maps.get("motion_edges")
        if motion_edges is None:
            motion_edges = result_a.intermediate_maps.get("edges_map")
        if motion_edges is not None:
            st.image((np.clip(motion_edges, 0, 1) * 255).astype(np.uint8), use_container_width=True)
        else:
            st.image(np.zeros_like(img[:, :, 0]), use_container_width=True)

    with out_placeholder.container():
        st.subheader(f"Phosphene — {result_a.backend_name}")
        if result_a.error:
            st.error(result_a.error)
        else:
            st.image(result_a.phosphene_image, use_container_width=True)
        st.caption(f"Task: {result_a.task_mode}")

    # ── Run secondary backend if comparison enabled ──────────────────────────
    result_b: Optional[BackendResult] = None
    if ui["enable_comparison"] and ui["secondary_backend"] != ui["primary_backend"]:
        with st.spinner(f"Running {ui['secondary_backend']}…"):
            result_b = run_backend(ui["secondary_backend"], ctx, ui)
    elif ui["enable_comparison"]:
        result_b = result_a

    # ── Tabs (analysis views) ────────────────────────────────────────────────
    tab_names = ["Demo", "Compare", "Evaluation", "Diagnostics", "Batch"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_demo_tab(ctx, result_a, ui)

    with tabs[1]:
        if not ui["enable_comparison"]:
            st.info("Enable 'A/B comparison' in the sidebar to use this tab.")
        elif result_b is not None:
            render_compare_tab(ctx, result_a, result_b, ui)

    with tabs[2]:
        render_evaluation_tab(result_a, ctx)

    with tabs[3]:
        render_diagnostics_tab(ctx, result_a)

    with tabs[4]:
        render_batch_tab(ui, enc, sal, depth_est)


if __name__ == "__main__":
    main()
