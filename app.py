"""
Task-Based Retinal Prosthesis Simulation — Benchmarking Workbench
=================================================================
A modular Streamlit app for comparing phosphene encoder backends,
evaluating perceptual quality, and inspecting intermediate
feature maps for neuroprosthetic rendering research.

UI architecture (Pass 1 — structure & hierarchy)
------------------------------------------------
The interface is organised around a single principle: the *result* is
the product, everything else is supporting context.

1. **Sidebar** holds only navigation-level controls:
   - Image input (the user's source material)
   - Configuration (task mode, backend, compare toggle)
   - Advanced settings (a single collapsed expander; only the
     parameter groups for the *currently selected* backend appear
     inside it)

2. **Main area**
   - Landing state when no image is loaded: a single centred prompt.
   - Result row when an image is loaded: a stable 3-column layout
     (Input | Edges/Motion | Phosphene) in single mode, or
     (Input | Backend A | Backend B) in compare mode. Headline
     metrics appear directly below each phosphene output.
   - Tabs (Stimulation | Compare | Evaluate | Diagnostics | Batch)
     hold deeper analysis. They are visually secondary to the result
     row above.

Module architecture (unchanged)
-------------------------------
PerceptionContext  — pre-computed perception (encoders/base.py)
PhospheneBackend   — common backend interface (encoders/)
BackendResult      — standard backend output
metrics/           — SSIM, LPIPS, electrode efficiency, coverage
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

from encoders.base import TASK_PRESETS, BackendResult, PerceptionContext
from encoders.dots_backend import DotsParams
from encoders.registry import (
    BACKEND_NAMES,
    available_backend_names,
    get_backend,
    list_backends,
    set_toolkit_encoder,
)
from metrics import (
    LPIPS_AVAILABLE,
    compute_coverage_metrics,
    compute_electrode_metrics,
    compute_lpips,
    compute_ssim,
)
from utils.adaptive_encoding import (
    clahe_u8,
    gamma_u8,
    retinex_ssr_u8,
)
from utils.ai_inference_fusion import (
    fuse_ai_attention,
    yolo_detections_to_heatmap,
)


st.set_page_config(
    page_title="Retinal Prosthesis Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Pass 2 — visual theme
# ----------------------------------------------------------------
# Linear-inspired polish: a single restrained accent, a tighter type
# scale, subtle 1px borders instead of heavy shadows, calm tab rhythm,
# and a tonally muted metric block. The intent is to recede so the
# image/result panels carry the visual weight.
#
# All visual settings live here (and in .streamlit/config.toml). The
# rest of the app is structural.
# ============================================================

_THEME_CSS = """
<style>
:root {
  --rpw-bg: #FBFBFD;
  --rpw-surface: #FFFFFF;
  --rpw-surface-2: #F3F4F8;
  --rpw-border: rgba(14, 17, 22, 0.08);
  --rpw-border-strong: rgba(14, 17, 22, 0.14);
  --rpw-text: #0E1116;
  --rpw-text-muted: #5B6270;
  --rpw-text-subtle: #8B92A1;
  --rpw-accent: #5B6CFF;
  --rpw-accent-soft: rgba(91, 108, 255, 0.10);
  --rpw-radius: 10px;
  --rpw-radius-sm: 6px;
}

/* ---------- Type scale ----------------------------------------- */
html, body, [class*="css"] {
  font-feature-settings: "ss01", "cv11";
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.block-container {
  padding-top: 2.2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1400px;
}
h1, h2, h3, h4 {
  letter-spacing: -0.012em;
  font-weight: 600;
}
h1 { font-size: 1.55rem !important; line-height: 1.2; margin-bottom: 0.15rem !important; }
h2 { font-size: 1.10rem !important; line-height: 1.3; }
h3 { font-size: 0.95rem !important; line-height: 1.3; }
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--rpw-text-muted) !important;
  font-size: 0.82rem !important;
}

/* Tighten the title block at the top of the app */
[data-testid="stAppViewContainer"] > .main h1 + p,
[data-testid="stAppViewContainer"] > .main h1 + div {
  color: var(--rpw-text-muted);
  font-size: 0.86rem;
  margin-top: 0.05rem;
}

/* ---------- Sidebar -------------------------------------------- */
[data-testid="stSidebar"] {
  background: var(--rpw-surface-2);
  border-right: 1px solid var(--rpw-border);
}
[data-testid="stSidebar"] .block-container {
  padding-top: 1.5rem !important;
}
[data-testid="stSidebar"] h3 {
  font-size: 0.72rem !important;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--rpw-text-subtle) !important;
  margin: 1.4rem 0 0.5rem 0 !important;
}
[data-testid="stSidebar"] h3:first-of-type {
  margin-top: 0.2rem !important;
}

/* ---------- Inputs --------------------------------------------- */
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
  border-radius: var(--rpw-radius-sm) !important;
  border-color: var(--rpw-border) !important;
}
.stSlider [data-baseweb="slider"] > div > div {
  background: var(--rpw-accent) !important;
}
.stCheckbox > label {
  font-size: 0.86rem !important;
  color: var(--rpw-text) !important;
}

/* File uploader: calmer, less candy */
[data-testid="stFileUploader"] section {
  border: 1px dashed var(--rpw-border-strong) !important;
  background: var(--rpw-surface) !important;
  border-radius: var(--rpw-radius) !important;
  padding: 0.85rem 1rem !important;
}
[data-testid="stFileUploader"] section button {
  background: var(--rpw-surface) !important;
  border: 1px solid var(--rpw-border-strong) !important;
  color: var(--rpw-text) !important;
  font-weight: 500 !important;
}

/* ---------- Buttons -------------------------------------------- */
.stButton > button, .stDownloadButton > button {
  border-radius: var(--rpw-radius-sm) !important;
  border: 1px solid var(--rpw-border-strong) !important;
  background: var(--rpw-surface) !important;
  color: var(--rpw-text) !important;
  font-weight: 500 !important;
  padding: 0.42rem 0.95rem !important;
  transition: background 120ms ease, border-color 120ms ease, transform 60ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--rpw-accent-soft) !important;
  border-color: var(--rpw-accent) !important;
  color: var(--rpw-accent) !important;
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: translateY(1px);
}
.stButton > button:focus, .stDownloadButton > button:focus {
  box-shadow: 0 0 0 3px var(--rpw-accent-soft) !important;
  outline: none !important;
}

/* ---------- Containers (cards) --------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--rpw-radius) !important;
  border-color: var(--rpw-border) !important;
  background: var(--rpw-surface);
  box-shadow: 0 1px 0 rgba(14, 17, 22, 0.02);
}

/* Image rounding inside cards */
[data-testid="stImage"] img {
  border-radius: var(--rpw-radius-sm);
}

/* ---------- Metrics -------------------------------------------- */
[data-testid="stMetric"] {
  background: transparent;
  padding: 0.15rem 0;
}
[data-testid="stMetricLabel"] {
  color: var(--rpw-text-subtle) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
[data-testid="stMetricValue"] {
  font-size: 1.15rem !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em;
  color: var(--rpw-text) !important;
}
[data-testid="stMetricDelta"] {
  font-size: 0.75rem !important;
}

/* ---------- Tabs ----------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.25rem;
  border-bottom: 1px solid var(--rpw-border);
  margin-bottom: 1.1rem;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  color: var(--rpw-text-muted) !important;
  font-weight: 500 !important;
  font-size: 0.86rem !important;
  padding: 0.55rem 0.85rem !important;
  border-radius: var(--rpw-radius-sm) var(--rpw-radius-sm) 0 0 !important;
  transition: color 120ms ease, background 120ms ease;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--rpw-text) !important;
  background: var(--rpw-surface-2) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--rpw-accent) !important;
  background: transparent !important;
  box-shadow: inset 0 -2px 0 var(--rpw-accent);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ---------- Dividers ------------------------------------------- */
hr, [data-testid="stDivider"] {
  border-color: var(--rpw-border) !important;
  background: var(--rpw-border) !important;
  margin: 1.25rem 0 !important;
}

/* ---------- Expanders ------------------------------------------ */
.streamlit-expanderHeader, [data-testid="stExpander"] details > summary {
  font-weight: 500 !important;
  font-size: 0.86rem !important;
  color: var(--rpw-text) !important;
}
[data-testid="stExpander"] {
  border: 1px solid var(--rpw-border) !important;
  border-radius: var(--rpw-radius-sm) !important;
  background: var(--rpw-surface);
}

/* ---------- Alerts (calmer) ------------------------------------ */
[data-testid="stAlert"] {
  border-radius: var(--rpw-radius-sm) !important;
  border: 1px solid var(--rpw-border) !important;
  font-size: 0.86rem !important;
}

/* ---------- Dataframe ------------------------------------------ */
[data-testid="stDataFrame"] {
  border-radius: var(--rpw-radius-sm);
  border: 1px solid var(--rpw-border);
  overflow: hidden;
}

/* ---------- Reduced motion preference -------------------------- */
@media (prefers-reduced-motion: reduce) {
  * {
    transition: none !important;
    animation: none !important;
  }
}
</style>
"""


def _inject_theme() -> None:
    """Inject the Pass 2 theme. Idempotent — safe across reruns."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


# ============================================================
# Cached model loaders (heavy; loaded once per session)
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
    """Load PhospheneEncoderTool (used for segmentation + the Toolkit backend)."""
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
        set_toolkit_encoder(enc)
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
# Generic helpers
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


def _card():
    """Return a bordered Streamlit container, falling back gracefully.

    `st.container(border=True)` was added in Streamlit 1.29. Older
    versions silently degrade to a plain container so the layout still
    works; the CSS theme already adds restraint to non-bordered blocks.
    """
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def _heatmap_rgb(arr: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO,
                 size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert a 2D array in [0,1] to a colourised RGB image for display."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = np.clip(a, 0, 1)
    if a.max() <= 1.0:
        a = a * 255.0
    a_u8 = a.astype(np.uint8)
    if size is not None:
        a_u8 = cv2.resize(a_u8, size, interpolation=cv2.INTER_LINEAR)
    bgr = cv2.applyColorMap(a_u8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# Perception context builder (shared across all backends)
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
    """Compute all shared perception outputs for *img* once per frame."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe_obj.apply(gray)
    except Exception:
        gray_clahe = gray
    lum = _robust_normalize(gray_clahe.astype(np.float32) / 255.0, 2.0, 98.0)
    lum_blur = cv2.GaussianBlur(lum, (0, 0), 3.0)
    detail = _robust_normalize(np.abs(lum - lum_blur), 2.0, 99.0)
    luminance = _robust_normalize(0.65 * lum + 0.35 * detail, 1.0, 99.0)

    saliency_map = None
    if sal:
        try:
            saliency_map = sal.compute_saliency(img, method="combined")
        except Exception:
            pass

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

    annotated_yolo, detections = run_yolo_labeler(img, conf=0.25)
    yolo_heatmap: Optional[np.ndarray] = None
    primary_det: Optional[Dict] = None
    yolo_roi: Optional[tuple] = None
    if detections:
        det_sorted = sorted(detections, key=lambda d: float(d.get("conf", 0)), reverse=True)[:3]
        yolo_heatmap = yolo_detections_to_heatmap(det_sorted, img.shape[:2], sigma=60.0)
        primary_det = det_sorted[0]

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

        if primary_det is not None:
            x1, y1, x2, y2 = [float(v) for v in primary_det.get("bbox", (0, 0, w, h))[:4]]
            mx, my = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            yolo_roi = (x1 - mx, y1 - my, x2 + mx, y2 + my)
            rx1, ry1 = max(0, int(yolo_roi[0])), max(0, int(yolo_roi[1]))
            rx2, ry2 = min(w, int(yolo_roi[2])), min(h, int(yolo_roi[3]))
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

        if yolo_roi is not None and int(np.sum(object_mask_u8 > 0)) > 50:
            mb = _mask_bbox(object_mask_u8)
            if mb is not None and _bbox_iou(mb, yolo_roi) < 0.08:
                rx1, ry1 = max(0, int(yolo_roi[0])), max(0, int(yolo_roi[1]))
                rx2, ry2 = min(w, int(yolo_roi[2])), min(h, int(yolo_roi[3]))
                tmp = np.zeros_like(object_mask_u8)
                tmp[ry1:ry2, rx1:rx2] = 255
                object_mask_u8 = tmp

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

    edges_map = compute_edges(img)

    motion_magnitude: Optional[np.ndarray] = None
    motion_result: Optional[Dict] = None
    if is_video and len(frames) > 1 and frame_idx > 0:
        try:
            from phosphene_toolkit.perception.motion import compute_motion_between_frames
            motion_result = compute_motion_between_frames(frames[frame_idx - 1], frames[frame_idx])
            motion_magnitude = motion_result.get("magnitude")
        except Exception:
            pass

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
# Backend params + execution
# ============================================================

def _make_backend_params(backend_name: str, ui: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"task_mode": ui["task_mode"]}
    if backend_name == "Dots Baseline":
        params.update(ui["dots_params"])
    elif backend_name == "p2p Retinal (Argus II)":
        params.update(ui["p2p_params"])
    elif backend_name == "Dynaphos Cortical (p2p)":
        params.update(ui["dynaphos_params"])
    elif backend_name == "Learned Encoder (E2E)":
        params.update(ui["learned_params"])
    return params


def _get_prev_stim(backend_name: str) -> Optional[np.ndarray]:
    return st.session_state.get(f"_prev_stim_{backend_name}")


def _set_prev_stim(backend_name: str, stim: Optional[np.ndarray]) -> None:
    if stim is not None:
        st.session_state[f"_prev_stim_{backend_name}"] = stim.copy()
    else:
        st.session_state.pop(f"_prev_stim_{backend_name}", None)


def _get_adaptive_state(backend_name: str) -> Dict[str, Any]:
    return st.session_state.get(f"_adaptive_{backend_name}", {})


def _set_adaptive_state(backend_name: str, state: Dict[str, Any]) -> None:
    st.session_state[f"_adaptive_{backend_name}"] = state


def _reset_temporal_state(upload_id: str) -> None:
    """Clear per-backend temporal smoothing / adaptive state on new media."""
    if st.session_state.get("_upload_id") != upload_id:
        st.session_state["_upload_id"] = upload_id
        for key in list(st.session_state.keys()):
            if key.startswith("_prev_stim") or key.startswith("_adaptive"):
                del st.session_state[key]


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
    if result.stimulation_grid is not None:
        _set_prev_stim(backend_name, result.stimulation_grid)
    meta = result.metadata
    if "adaptive" in meta and "new_state" in meta["adaptive"]:
        _set_adaptive_state(backend_name, meta["adaptive"]["new_state"])
    return result


# ============================================================
# SIDEBAR
# ----------------------------------------------------------------
# Two visible sections + one collapsed Advanced expander.
# Backend-specific controls are only shown for the selected backend.
# ============================================================

def _format_backend_label(name: str, info: Dict[str, Tuple[Any, bool, str]]) -> str:
    """Append `(unavailable)` suffix to disabled backends in the selectbox."""
    avail = info.get(name, (None, False, ""))[1]
    return name if avail else f"{name}  (unavailable)"


def _render_advanced_for_backend(
    backend_name: str,
    task_mode: str,
    dots_params: Dict[str, Any],
    p2p_params: Dict[str, Any],
    dynaphos_params: Dict[str, Any],
    learned_params: Dict[str, Any],
) -> None:
    """Render only the parameter group(s) relevant to *backend_name*."""
    preset = TASK_PRESETS.get(task_mode)

    if backend_name == "Dots Baseline":
        st.markdown("**Dots — calibration**")
        dots_params["dot_density"] = st.slider("Dot density", 0.05, 0.22, 0.12, 0.01)
        dots_params["subject_weight"] = st.slider(
            "Subject priority", 0.30, 0.70,
            float(preset.subject_weight) if preset else 0.50, 0.05,
        )
        dots_params["near_weight"] = st.slider(
            "Near-field priority", 0.10, 0.55,
            float(preset.near_weight) if preset else 0.22, 0.05,
        )
        dots_params["fill_weight"] = st.slider(
            "Interior fill", 0.0, 0.45,
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
            "Segmentation target", ["auto", "dog", "person", "cat", "foreground"], index=0,
        )
        dots_params["interior_frac"] = st.slider("Interior dot fraction", 0.0, 0.60, 0.20, 0.05)
        dots_params["fg_frac"] = st.slider("Foreground dot fraction", 0.0, 0.40, 0.10, 0.05)
        dots_params["min_sep"] = st.slider("Min dot spacing", 0, 4, 1, 1)

        st.markdown("**Dots — adaptive encoding**")
        dots_params["use_adaptive"] = True
        dots_params["adaptive_hold"] = st.slider("Hold frames (video)", 0, 10, 3, 1)
        dots_params["adaptive_margin"] = st.slider("Switch margin", 0.0, 0.15, 0.03, 0.01)
        dots_params["adaptive_gate_thresh"] = st.slider(
            "Foreground gate threshold", 0.05, 0.60, 0.18, 0.05,
        )

        st.markdown("**Dots — renderer**")
        dots_params["dot_sigma"] = st.slider("Dot σ (px)", 0.8, 3.5, 1.6, 0.1)
        dots_params["dot_blend"] = st.selectbox("Dot blend", ["sum", "max"], index=0)
        dots_params["dot_jitter"] = st.slider("Dot jitter (px)", 0.0, 1.5, 0.0, 0.1)

        st.markdown("**Dots — edge prefilter**")
        dots_params["use_bilateral"] = st.checkbox("Bilateral filter before edges", value=True)
        dots_params["outline_source"] = st.selectbox(
            "Outline source",
            ["Segmentation boundary (preferred)", "Pixel edges (fallback)"],
            index=0,
        )

    elif backend_name == "p2p Retinal (Argus II)":
        st.markdown("**p2p Retinal — model**")
        p2p_params["p2p_model"] = st.selectbox(
            "Perceptual model", ["axon_map", "scoreboard"], index=0,
            help="AxonMap: biologically accurate axon streaks. Scoreboard: simpler blobs.",
        )

    elif backend_name == "Dynaphos Cortical (p2p)":
        st.markdown("**Dynaphos — preprocessing**")
        from encoders.dynaphos_backend import PREPROCESS_MODES
        dynaphos_params["dynaphos_preprocess"] = st.selectbox(
            "Preprocessing", PREPROCESS_MODES, index=0,
            help="How to convert the image to a cortical stimulation pattern.",
        )

    elif backend_name == "Learned Encoder (E2E)":
        st.markdown("**Learned encoder — checkpoint**")
        from encoders.learned_backend import available_checkpoints
        ckpts = available_checkpoints()
        if not ckpts:
            st.caption("No trained checkpoint found under `runs/paper_eval/`.")
        else:
            labels = [label for label, _ in ckpts]
            learned_params["learned_ckpt"] = st.selectbox(
                "Trained model", labels, index=0,
                help="End-to-end trained encoder. MoE picks an expert per image (adaptive candidate).",
            )
            learned_params["learned_renderer"] = st.selectbox(
                "Renderer",
                ["differentiable", "argus_ii"],
                index=0,
                format_func=lambda r: {
                    "differentiable": "Differentiable simulator (training)",
                    "argus_ii": "pulse2percept Argus II",
                }.get(r, r),
                help="Render the learned stimulation grid with the training "
                     "simulator, or feed it into the Argus II retinal model.",
            )
            if learned_params.get("learned_renderer") == "argus_ii":
                learned_params["p2p_model"] = st.selectbox(
                    "Argus II model", ["axon_map", "scoreboard"], index=0,
                    key="learned_p2p_model",
                )

    else:
        st.caption("This backend has no advanced settings.")


def render_sidebar() -> Dict[str, Any]:
    """Render the sidebar.

    Returns a dict of UI state used by main() to drive processing.
    The sidebar has three visible sections:

      1. Image input — file uploader (and frame slider if video)
      2. Configuration — task mode, backend, optional comparison
      3. Advanced settings — single collapsed expander; only the
         parameter groups for the active backend(s) are shown inside.
    """
    backends_info = list_backends()

    with st.sidebar:
        st.markdown("### Image input")
        uploaded = st.file_uploader(
            "Upload image or video",
            type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "gif"],
            label_visibility="collapsed",
            key="sidebar_uploader",
        )

        st.markdown("### Configuration")
        task_mode = st.selectbox(
            "Task mode",
            list(TASK_PRESETS.keys()),
            index=0,
            help="Adjusts cue fusion weights and electrode budget allocation.",
        )

        # Backend selectbox: include unavailable entries with a clear suffix
        backend_options = list(BACKEND_NAMES)
        primary_backend = st.selectbox(
            "Backend",
            backend_options,
            index=0,
            format_func=lambda n: _format_backend_label(n, backends_info),
        )

        # Inline disabled-state warning if user picks an unavailable backend
        primary_avail, primary_reason = backends_info[primary_backend][1], backends_info[primary_backend][2]
        if not primary_avail:
            st.warning(f"This backend is unavailable. {primary_reason}")

        enable_comparison = st.checkbox("Compare with another backend", value=False)
        secondary_backend = primary_backend
        if enable_comparison:
            secondary_opts = [n for n in backend_options if n != primary_backend] or backend_options
            secondary_backend = st.selectbox(
                "Backend B",
                secondary_opts,
                index=0,
                format_func=lambda n: _format_backend_label(n, backends_info),
            )
            sec_avail, sec_reason = backends_info[secondary_backend][1], backends_info[secondary_backend][2]
            if not sec_avail:
                st.warning(f"Backend B is unavailable. {sec_reason}")

        # ---- Advanced (collapsed by default) ----
        dots_params: Dict[str, Any] = {}
        p2p_params: Dict[str, Any] = {}
        dynaphos_params: Dict[str, Any] = {}
        learned_params: Dict[str, Any] = {}
        show_diagnostics = False
        show_candidate_table = False

        with st.expander("Advanced settings", expanded=False):
            # Render parameters for the active backend(s) only.
            # Avoid duplicating Dots block when both A and B are Dots.
            shown_backends: List[str] = []
            for name in (primary_backend, secondary_backend):
                if name in shown_backends:
                    continue
                shown_backends.append(name)
                if len(shown_backends) > 1:
                    st.divider()
                st.caption(name)
                _render_advanced_for_backend(
                    name, task_mode, dots_params, p2p_params, dynaphos_params,
                    learned_params,
                )

            st.divider()
            st.caption("Display")
            show_diagnostics = st.checkbox(
                "Show diagnostic maps", value=False,
                help="Reveal segmentation, saliency, depth, attention, and intermediate maps in the Diagnostics tab.",
            )
            show_candidate_table = st.checkbox(
                "Show adaptive candidates table", value=False,
                help="In the Stimulation tab, list all evaluated encoding candidates and their scores.",
            )

    return {
        "uploaded": uploaded,
        "task_mode": task_mode,
        "primary_backend": primary_backend,
        "secondary_backend": secondary_backend,
        "enable_comparison": enable_comparison,
        "primary_available": primary_avail,
        "secondary_available": backends_info[secondary_backend][1],
        "dots_params": dots_params,
        "p2p_params": p2p_params,
        "dynaphos_params": dynaphos_params,
        "learned_params": learned_params,
        "mask_kernel": int(dots_params.get("mask_kernel", 7)),
        "target_class": str(dots_params.get("target_class", "auto")),
        "show_candidate_table": show_candidate_table,
        "show_diagnostics": show_diagnostics,
    }


# ============================================================
# RESULT ROW (always visible above the tabs)
# ============================================================

def _summary_metrics(result: BackendResult) -> Tuple[str, str, str]:
    """Return (active electrodes, coverage %, latency ms) as display strings.

    Returns "—" for any value that cannot be computed.
    """
    active = "—"
    if result.stimulation_grid is not None:
        em = compute_electrode_metrics(result.stimulation_grid)
        active = f"{int(em['active_count'])}"

    coverage = "—"
    if not result.error and result.phosphene_image is not None:
        try:
            cm = compute_coverage_metrics(result.phosphene_image)
            coverage = f"{100.0 * cm['global_coverage']:.1f}%"
        except Exception:
            pass

    latency_ms = result.timing_info.get("total_ms")
    latency = f"{latency_ms:.0f} ms" if latency_ms is not None else "—"

    return active, coverage, latency


def _panel_eyebrow(label: str) -> None:
    """A small uppercase eyebrow label for cards (Linear-style)."""
    st.markdown(
        f'<div style="font-size:0.70rem;font-weight:600;letter-spacing:0.07em;'
        f'text-transform:uppercase;color:var(--rpw-text-subtle);'
        f'margin:0 0 0.45rem 0;">{label}</div>',
        unsafe_allow_html=True,
    )


def _render_phosphene_panel(
    result: BackendResult,
    eyebrow: str,
    title: str,
    *,
    show_metrics: bool = True,
) -> None:
    """Render a phosphene result inside a bordered card."""
    with _card():
        _panel_eyebrow(eyebrow)
        st.markdown(
            f'<div style="font-weight:600;font-size:0.95rem;color:var(--rpw-text);'
            f'margin:0 0 0.5rem 0;">{title}</div>',
            unsafe_allow_html=True,
        )
        if result.error:
            st.error(result.error)
        else:
            st.image(result.phosphene_image, use_container_width=True)
        if show_metrics:
            active, coverage, latency = _summary_metrics(result)
            m1, m2, m3 = st.columns(3)
            m1.metric("Active", active)
            m2.metric("Coverage", coverage)
            m3.metric("Latency", latency)


def render_result_row(
    ctx: PerceptionContext,
    result_a: BackendResult,
    result_b: Optional[BackendResult],
    enable_comparison: bool,
) -> None:
    """The primary visual surface of the app.

    Each column is a bordered card. The eyebrow label gives the user a
    consistent way to scan the row ("INPUT / EDGES / PHOSPHENE" or
    "INPUT / BACKEND A / BACKEND B"); the bold title underneath names
    the specific content. This separation of *role* from *identity*
    makes A/B comparisons easier to scan.
    """
    img_rgb = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB)

    if not enable_comparison or result_b is None:
        col_in, col_mid, col_out = st.columns(3)
        with col_in:
            with _card():
                _panel_eyebrow("Input")
                st.image(img_rgb, use_container_width=True)
                st.caption(f"Task · {ctx.task_mode}")
        with col_mid:
            with _card():
                _panel_eyebrow("Edges / Motion")
                edges_view = result_a.intermediate_maps.get("motion_edges")
                if edges_view is None:
                    edges_view = result_a.intermediate_maps.get("edges_map")
                if edges_view is None:
                    edges_view = ctx.edges_map
                if edges_view is not None:
                    st.image(
                        (np.clip(edges_view, 0, 1) * 255).astype(np.uint8),
                        use_container_width=True,
                    )
                else:
                    st.image(np.zeros_like(ctx.gray), use_container_width=True)
                st.caption("Foreground-gated edge map")
        with col_out:
            _render_phosphene_panel(
                result_a,
                eyebrow="Phosphene",
                title=result_a.backend_name,
            )
        return

    col_in, col_a, col_b = st.columns(3)
    with col_in:
        with _card():
            _panel_eyebrow("Input")
            st.image(img_rgb, use_container_width=True)
            st.caption(f"Task · {ctx.task_mode}")
    with col_a:
        _render_phosphene_panel(result_a, eyebrow="Backend A", title=result_a.backend_name)
    with col_b:
        _render_phosphene_panel(result_b, eyebrow="Backend B", title=result_b.backend_name)


# ============================================================
# TABS
# ============================================================

def render_stimulation_tab(
    ctx: PerceptionContext,
    result: BackendResult,
    show_candidate_table: bool,
) -> None:
    """Show the underlying stimulation grid and the foreground gate.

    These are *secondary* views — the primary phosphene output already
    appears in the result row above. The Stimulation tab exists so the
    user can inspect what the encoder is doing under the hood.
    """
    col_stim, col_gate = st.columns(2)
    with col_stim:
        with _card():
            _panel_eyebrow("Stimulation grid")
            if result.stimulation_grid is not None:
                sg = np.clip(result.stimulation_grid, 0, 1)
                st.image(_heatmap_rgb(sg, size=(384, 384)), use_container_width=True)
                st.caption(f"{sg.shape[0]} × {sg.shape[1]} electrodes")
            else:
                st.info("This backend does not expose an intermediate stimulation grid.")

    with col_gate:
        with _card():
            _panel_eyebrow("Foreground gate")
            st.image(_heatmap_rgb(ctx.gate, size=(384, 384)), use_container_width=True)
            st.caption("High values = foreground / salient region")

    if show_candidate_table and "adaptive" in result.metadata:
        cands = result.metadata["adaptive"].get("candidates", [])
        if cands:
            st.divider()
            st.caption(
                f"Adaptive choice · `{result.metadata['adaptive']['chosen']}`  "
                f"(stable for {result.metadata['adaptive']['stable_frames']} frame(s))"
            )
            st.dataframe(cands, use_container_width=True, height=220)


def render_compare_tab(
    ctx: PerceptionContext,
    result_a: Optional[BackendResult],
    result_b: Optional[BackendResult],
    enable_comparison: bool,
) -> None:
    """Side-by-side stimulation grids and a compact metric comparison."""
    if not enable_comparison or result_b is None:
        st.caption(
            "Enable *Compare with another backend* in the sidebar to populate this tab."
        )
        return

    # Stimulation grids side by side (the result row already shows percepts)
    col_a, col_b = st.columns(2)
    for col, label, res in [(col_a, "A", result_a), (col_b, "B", result_b)]:
        with col:
            with _card():
                _panel_eyebrow(f"Backend {label} · stimulation")
                st.markdown(
                    f'<div style="font-weight:600;font-size:0.92rem;color:var(--rpw-text);'
                    f'margin:0 0 0.5rem 0;">{res.backend_name}</div>',
                    unsafe_allow_html=True,
                )
                if res.stimulation_grid is not None:
                    st.image(
                        _heatmap_rgb(np.clip(res.stimulation_grid, 0, 1), size=(384, 384)),
                        use_container_width=True,
                    )
                else:
                    st.info("No stimulation grid for this backend.")

    st.divider()
    st.markdown("**Metric comparison**")

    rows: List[Dict[str, Any]] = []
    for label, res in [("A", result_a), ("B", result_b)]:
        row: Dict[str, Any] = {"": f"{label} · {res.backend_name}"}
        if res.stimulation_grid is not None:
            em = compute_electrode_metrics(res.stimulation_grid)
            row["Active"] = int(em["active_count"])
            row["Active %"] = round(100.0 * em["active_ratio"], 1)
            row["Efficiency"] = round(em["efficiency_proxy"], 3)
            row["Flicker"] = round(em["flicker"], 3)
        if not res.error:
            cm = compute_coverage_metrics(res.phosphene_image)
            row["Coverage %"] = round(100.0 * cm["global_coverage"], 1)
        row["Latency (ms)"] = round(res.timing_info.get("total_ms", 0.0), 1)
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # A vs B image-level similarity (only meaningful when shapes match)
    if (
        result_a.phosphene_image is not None
        and result_b.phosphene_image is not None
        and result_a.phosphene_image.shape == result_b.phosphene_image.shape
        and not result_a.error
        and not result_b.error
    ):
        ssim_val = compute_ssim(result_a.phosphene_image, result_b.phosphene_image)
        cols = st.columns(2)
        cols[0].metric("SSIM (A vs B)", f"{ssim_val:.3f}", help="1 = identical")
        if LPIPS_AVAILABLE:
            lpips_val = compute_lpips(result_a.phosphene_image, result_b.phosphene_image)
            if lpips_val is not None:
                cols[1].metric("LPIPS (A vs B)", f"{lpips_val:.3f}", help="0 = identical")


def render_evaluate_tab(result: BackendResult, ctx: PerceptionContext) -> None:
    """Per-image quantitative evaluation."""
    if result.error:
        st.error(f"Backend failed: {result.error}")
        return

    col_e, col_c = st.columns(2)

    with col_e:
        with _card():
            _panel_eyebrow("Electrode metrics")
            if result.stimulation_grid is not None:
                em = compute_electrode_metrics(result.stimulation_grid)
                mc1, mc2 = st.columns(2)
                mc1.metric("Active electrodes", int(em["active_count"]))
                mc2.metric("Active %", f"{100.0 * em['active_ratio']:.1f}%")
                mc3, mc4 = st.columns(2)
                mc3.metric("Efficiency proxy", f"{em['efficiency_proxy']:.3f}")
                mc4.metric("Redundancy", f"{em['redundancy']:.3f}")
                st.metric("Flicker", f"{em['flicker']:.3f}")
                with st.expander("How is the efficiency proxy defined?"):
                    st.markdown(
                        "`efficiency = (1 − active_ratio) × (1 − redundancy)`\n\n"
                        "* `active_ratio`: fraction of electrodes that are active.\n"
                        "* `redundancy`: mean fraction of 8-connected neighbours of "
                        "each active electrode that are also active.\n\n"
                        "High efficiency means sparse, well-separated activations."
                    )
            else:
                st.info("This backend does not expose a stimulation grid.")

    with col_c:
        with _card():
            _panel_eyebrow("Phosphene coverage")
            cm = compute_coverage_metrics(result.phosphene_image)
            radial = cm["radial"]
            cc1, cc2 = st.columns(2)
            cc1.metric("Global coverage", f"{100.0 * cm['global_coverage']:.1f}%")
            cc2.metric("Mean eccentricity", f"{radial['mean_eccentricity']:.3f}")

            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 2.4))
                xs = radial["bin_centers"]
                ys = [v * 100 for v in radial["bin_coverage"]]
                ax.bar(xs, ys, width=0.09, color="#5B6CFF", alpha=0.9)
                ax.set_xlabel("Eccentricity", fontsize=8, color="#5B6270")
                ax.set_ylabel("Coverage %", fontsize=8, color="#5B6270")
                ax.set_ylim(0, max(100.0, max(ys) * 1.1 if ys else 100.0))
                ax.tick_params(left=False, bottom=False, labelsize=7, colors="#8B92A1")
                ax.grid(axis="y", color="#0E111614", linewidth=0.6)
                ax.set_axisbelow(True)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception:
                st.dataframe(
                    {"eccentricity": radial["bin_centers"], "coverage": radial["bin_coverage"]},
                    hide_index=True,
                )

    st.divider()

    # Reference-only quality scores
    with _card():
        _panel_eyebrow("Image similarity vs. input")
        st.caption(
            "Reference only. Phosphene percepts are sparse and binary-like, so absolute "
            "scores against a natural image are not perceptually meaningful — use them "
            "to compare backends, not to judge absolute quality."
        )
        sim_cols = st.columns(2)
        ssim_val = compute_ssim(result.phosphene_image, ctx.image)
        sim_cols[0].metric("SSIM", f"{ssim_val:.3f}", help="1 = identical structure")
        if LPIPS_AVAILABLE:
            lpips_val = compute_lpips(result.phosphene_image, ctx.image)
            sim_cols[1].metric(
                "LPIPS",
                f"{lpips_val:.3f}" if lpips_val is not None else "—",
                help="0 = identical perception",
            )
        else:
            sim_cols[1].metric("LPIPS", "—", help="pip install lpips to enable")

    bottom_l, bottom_r = st.columns(2)
    with bottom_l:
        with _card():
            _panel_eyebrow("Latency breakdown")
            timing_data = {k: round(v, 2) for k, v in result.timing_info.items()}
            st.json(timing_data, expanded=False)
    with bottom_r:
        with _card():
            _panel_eyebrow("Coverage heatmap")
            hm = (cm["coverage_heatmap"] * 255).astype(np.uint8)
            st.image(_heatmap_rgb(hm / 255.0, colormap=cv2.COLORMAP_HOT), use_container_width=True)


def render_diagnostics_tab(
    ctx: PerceptionContext,
    result: BackendResult,
    show_diagnostics: bool,
) -> None:
    """Lower-priority introspection maps. Hidden by default behind a toggle."""
    if not show_diagnostics:
        st.caption(
            "Diagnostic maps are hidden by default. "
            "Enable *Show diagnostic maps* under Advanced settings to display them here."
        )
        return

    named_maps: Dict[str, Optional[np.ndarray]] = {
        "Saliency": ctx.saliency_map,
        "Segmentation FG": ctx.segmentation_fg,
        "Foreground gate": ctx.gate,
        "Near-field / depth": ctx.near,
        "Edges (raw)": ctx.edges_map,
    }
    for k in ("motion_edges", "attention_map", "priority_obj", "edge_g", "stim_g"):
        v = result.intermediate_maps.get(k)
        if v is not None:
            named_maps[k.replace("_", " ").title()] = v

    available_maps = {
        k: v for k, v in named_maps.items()
        if v is not None and np.asarray(v).size > 0
    }

    if not available_maps:
        st.info("No intermediate maps were produced for this image.")
    else:
        with _card():
            _panel_eyebrow("Intermediate maps")
            cols_per_row = 3
            keys = list(available_maps.keys())
            for row_start in range(0, len(keys), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for i, key in enumerate(keys[row_start:row_start + cols_per_row]):
                    with row_cols[i]:
                        st.markdown(
                            f'<div style="font-size:0.78rem;font-weight:500;'
                            f'color:var(--rpw-text-muted);margin-bottom:0.25rem;">{key}</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            _heatmap_rgb(available_maps[key], size=(256, 256)),
                            use_container_width=True,
                        )

    with _card():
        _panel_eyebrow("YOLO detections")
        conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, key="diag_yolo_conf")
        annotated, detections = run_yolo_labeler(ctx.image, conf=conf_thresh)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        if detections:
            for d in detections:
                st.caption(f"**{d['class']}** ({d['conf']:.2f})  ·  bbox {[round(v, 1) for v in d['bbox']]}")


def render_batch_tab(ui: Dict[str, Any], enc, sal, depth_est) -> None:
    """Run the active backend over a batch of images and export results."""
    backend_name = ui["primary_backend"]

    uploaded_batch = st.file_uploader(
        "Upload images (multi-select)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if not uploaded_batch:
        st.caption("Upload one or more images above to enable batch evaluation.")
        return

    if not ui["primary_available"]:
        st.warning("The selected backend is unavailable; batch run disabled.")
        return

    if not st.button(f"Run batch · {len(uploaded_batch)} images · {backend_name}"):
        return

    results_rows: List[Dict[str, Any]] = []
    progress = st.progress(0.0)

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
            row["ssim_vs_input"] = round(compute_ssim(res.phosphene_image, img), 4)
        row["latency_ms"] = round(latency_ms, 2)
        row["error"] = res.error or ""
        results_rows.append(row)
        progress.progress((i + 1) / len(uploaded_batch))

    if not results_rows:
        st.warning("No images could be decoded.")
        return

    st.success(f"Completed {len(results_rows)} images.")
    st.dataframe(results_rows, use_container_width=True, hide_index=True)

    import csv
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=list(results_rows[0].keys()))
    writer.writeheader()
    writer.writerows(results_rows)
    dcol1, dcol2 = st.columns(2)
    dcol1.download_button("Download CSV", csv_buf.getvalue(), file_name="batch_results.csv")
    dcol2.download_button("Download JSON", json.dumps(results_rows, indent=2), file_name="batch_results.json")


# ============================================================
# Media decoding (image or video)
# ============================================================

def _decode_uploaded_media(
    uploaded,
) -> Tuple[Optional[np.ndarray], List[np.ndarray], int, bool]:
    """Decode an uploaded image or video file.

    Returns (img, frames, frame_idx, is_video).  img is None on failure.
    """
    is_video = uploaded.type.startswith("video") or uploaded.name.lower().endswith(
        (".mp4", ".avi", ".mov", ".gif")
    )
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
            return None, [], 0, True
        frame_idx = st.slider("Frame", 0, len(frames) - 1, 0, key="frame_slider")
        img = frames[frame_idx]
    else:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return img, frames, frame_idx, is_video


# ============================================================
# Landing state (when no image is loaded)
# ============================================================

def render_landing() -> None:
    """Empty state shown when no image has been uploaded yet."""
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("&nbsp;")
        with _card():
            _panel_eyebrow("Get started")
            st.markdown(
                '<div style="font-weight:600;font-size:1.05rem;color:var(--rpw-text);'
                'margin:0 0 0.55rem 0;">Upload an image or short video</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "Use the **Image input** section in the sidebar to load a file. "
                "Then choose a **Task mode** and a **Backend** — the phosphene percept "
                "and key metrics will appear right here."
            )
            st.caption(
                "Tip · enable *Compare with another backend* in the sidebar to view A / B results side by side."
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    _inject_theme()

    st.title("Retinal Prosthesis Workbench")
    st.caption("Multi-backend phosphene encoder comparison and evaluation")

    ui = render_sidebar()

    # Heavy models are cached across sessions and reruns.
    enc = load_encoder()
    sal = load_saliency()
    depth_est = load_depth()

    uploaded = ui["uploaded"]
    if uploaded is None:
        render_landing()
        return

    # Reset per-backend temporal state when the user uploads a new file.
    upload_id = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
    _reset_temporal_state(upload_id)

    img, frames, frame_idx, is_video = _decode_uploaded_media(uploaded)
    if img is None:
        if is_video:
            st.warning("No frames found in this video.")
        else:
            st.error("Could not decode the uploaded file.")
        return

    # If the chosen backend is unavailable we still surface the message but
    # do not attempt to run it; the result panel will show the inline error.
    if not ui["primary_available"]:
        st.warning(
            f"Backend '{ui['primary_backend']}' is unavailable — choose a different backend in the sidebar."
        )

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

    with st.spinner(f"Running {ui['primary_backend']}…"):
        result_a = run_backend(ui["primary_backend"], ctx, ui)

    result_b: Optional[BackendResult] = None
    if ui["enable_comparison"]:
        if ui["secondary_backend"] != ui["primary_backend"]:
            with st.spinner(f"Running {ui['secondary_backend']}…"):
                result_b = run_backend(ui["secondary_backend"], ctx, ui)
        else:
            result_b = result_a

    # ----- Result row (primary surface) -----
    render_result_row(ctx, result_a, result_b, ui["enable_comparison"])

    st.divider()

    # ----- Tabs (secondary, for deeper inspection) -----
    tabs = st.tabs(["Stimulation", "Compare", "Evaluate", "Diagnostics", "Batch"])

    with tabs[0]:
        render_stimulation_tab(ctx, result_a, ui["show_candidate_table"])

    with tabs[1]:
        render_compare_tab(ctx, result_a, result_b, ui["enable_comparison"])

    with tabs[2]:
        render_evaluate_tab(result_a, ctx)

    with tabs[3]:
        render_diagnostics_tab(ctx, result_a, ui["show_diagnostics"])

    with tabs[4]:
        render_batch_tab(ui, enc, sal, depth_est)


if __name__ == "__main__":
    main()
