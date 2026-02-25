"""
Task-Based Retinal Prosthesis Simulation — Streamlit app.

Translates images and video into phosphene representations using AI inference
from YOLO, DeepLab V3, saliency, edge detection, and motion. Fuses all model
outputs to produce the best phosphene representation for any input.
"""

import streamlit as st
import numpy as np
import cv2
import time
from typing import Optional, List, Dict, Any, Tuple

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

st.set_page_config(page_title="Retinal Prosthesis Simulator", layout="wide")

def _robust_normalize(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi - lo < 1e-8:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _binary_fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask (0/255) using flood fill."""
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    flood = m.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(flood) & cv2.bitwise_not(m)
    filled = m | holes
    return (filled > 0).astype(np.uint8) * 255


def _keep_largest_component(mask_u8: np.ndarray, *, roi: Optional[tuple] = None) -> np.ndarray:
    """Keep the largest connected component (optionally within ROI). mask_u8 is 0/255."""
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
    out = (labels == k).astype(np.uint8) * 255
    return out


def _mask_boundary(mask_u8: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Return a ~1-2px boundary map (float32 in [0,1]) from a 0/255 mask."""
    k = max(3, int(ksize) | 1)
    ker = np.ones((k, k), np.uint8)
    grad = cv2.morphologyEx((mask_u8 > 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, ker)
    b = (grad > 0).astype(np.float32)
    # thin a bit: keep only local maxima by eroding once
    b_u8 = (b * 255).astype(np.uint8)
    b_u8 = cv2.erode(b_u8, np.ones((3, 3), np.uint8), iterations=1)
    return (b_u8.astype(np.float32) / 255.0)


def _contour_points_to_grid_mask(contour: np.ndarray, hw: tuple, grid_n: int, k: int) -> np.ndarray:
    """
    Sample exactly k points along contour arc-length and place them on a grid_n x grid_n mask.
    Returns uint8 mask in {0,1}.
    """
    h, w = hw
    pts = contour.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 4 or k <= 0:
        return np.zeros((grid_n, grid_n), dtype=np.uint8)
    # cumulative arc length
    d = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)], axis=0)
    total = float(s[-1] + 1e-8)
    # uniform samples along arc length
    targets = (np.linspace(0.0, total, num=k, endpoint=False) + total / max(k, 1) * 0.5).astype(np.float32)
    idx = np.searchsorted(s, targets, side="left")
    idx = np.clip(idx, 0, pts.shape[0] - 1)
    sp = pts[idx]
    # map to grid coordinates
    gx = np.clip((sp[:, 0] / max(1, w - 1)) * grid_n, 0, grid_n - 1).astype(np.int32)
    gy = np.clip((sp[:, 1] / max(1, h - 1)) * grid_n, 0, grid_n - 1).astype(np.int32)
    m = np.zeros((grid_n, grid_n), dtype=np.uint8)
    m[gy, gx] = 1
    return m


def _topk_minsep_mask(score: np.ndarray, k: int, *, min_sep: int = 0, forbid: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Greedy top-k selection with optional minimum spacing (in grid cells).
    Returns uint8 mask in {0,1}.
    """
    s = np.asarray(score, dtype=np.float32)
    h, w = s.shape
    if k <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    blocked = np.zeros((h, w), dtype=np.uint8)
    if forbid is not None:
        blocked = np.maximum(blocked, (np.asarray(forbid) > 0).astype(np.uint8))

    flat = s.reshape(-1)
    # Candidate pool: avoid sorting everything
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
            y1 = max(0, y - r)
            y2 = min(h, y + r + 1)
            x1 = max(0, x - r)
            x2 = min(w, x + r + 1)
            blocked[y1:y2, x1:x2] = 1
    return out


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
    try:
        from phosphene_toolkit import PhospheneEncoderTool
        config = {
            "device": {"grid_size": [60, 60], "amplitude_levels": 256, "max_amplitude_per_electrode": 1.0,
                       "global_power_cap": 100.0, "spatial_spread_sigma": 1.2, "temporal_freq_hz": 20.0,
                       "duty_cycle": 0.1, "dropout_rate": 0.0},
            "observer": {"phosphene_size_mean": 2.0, "phosphene_size_std": 0.5, "elongation_factor": 1.5,
                         "spatial_jitter_std": 0.3, "brightness_gamma": 0.8, "adaptation_rate": 0.1, "noise_level": 0.05},
            "perception": {"segmentation_model": "deeplabv3_resnet50", "input_size": [480, 640], "fast_mode": False},
            "fusion": {"allocation_strategy": "foveated", "max_active_phosphenes": 200}
        }
        return PhospheneEncoderTool(config=config)
    except Exception as e:
        st.warning(f"Pipeline encoder not available: {e}")
        return None


@st.cache_resource
def load_depth():
    """Optional depth model (MiDaS)."""
    try:
        from phosphene_toolkit.perception.depth import DepthEstimator
        return DepthEstimator(model_type="MiDaS_small")
    except Exception:
        return None


def run_yolo_labeler(img: np.ndarray, conf: float = 0.25) -> tuple:
    from utils.detection import run_yolo_detection
    return run_yolo_detection(img, conf_threshold=conf)


def compute_edges(img: np.ndarray, *, method: str = "canny_multi", low: int = 50, high: int = 150) -> np.ndarray:
    try:
        from phosphene_toolkit.perception.edges import compute_edges as _compute_edges
        # method is validated inside phosphene_toolkit; keep a safe fallback below
        return _compute_edges(img, method=method, low=int(low), high=int(high))
    except Exception:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (cv2.Canny(gray, int(low), int(high)) / 255.0).astype(np.float32)


def _candidate_gray(gray_u8: np.ndarray, cand: EncodingCandidate, *, bilateral_d: int = 7) -> np.ndarray:
    g = np.asarray(gray_u8, dtype=np.uint8)
    mode = str(cand.contrast_mode)
    if mode == "clahe":
        g = clahe_u8(g, clip_limit=2.0, tile_grid=(8, 8))
    elif mode == "gamma":
        g = gamma_u8(g, cand.gamma)
    elif mode == "retinex":
        g = retinex_ssr_u8(g, sigma=30.0)
    # else: "none"
    if bool(cand.use_bilateral):
        try:
            g = cv2.bilateralFilter(g, d=int(bilateral_d), sigmaColor=55, sigmaSpace=7)
        except Exception:
            pass
    return np.asarray(g, dtype=np.uint8)


def _luminance_from_gray(gray_u8: np.ndarray) -> np.ndarray:
    lum = (np.asarray(gray_u8, dtype=np.float32) / 255.0)
    lum = cv2.GaussianBlur(lum, (0, 0), 1.0)
    lum = _robust_normalize(lum, 2.0, 98.0)
    lp = cv2.GaussianBlur(lum, (0, 0), 3.0)
    detail = np.abs(lum - lp).astype(np.float32)
    detail = _robust_normalize(detail, 2.0, 99.0)
    lum = _robust_normalize(0.65 * lum + 0.35 * detail, 1.0, 99.0)
    return lum.astype(np.float32)


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / (union + 1e-8))


def _mask_bbox(mask_u8: np.ndarray) -> Optional[tuple]:
    ys, xs = np.where(np.asarray(mask_u8, dtype=np.uint8) > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0)


def main():
    st.title("Task-Based Retinal Prosthesis Simulation")
    st.caption("Adaptive Visual Inference Fusion for Neuroprosthetic Rendering")

    with st.sidebar:
        st.header("Settings")
        renderer_mode = st.selectbox("Phosphene Encoder", ["Dots (simple)"], index=0)
        encoding_policy = "Auto (adaptive)"
        st.caption("Encoding policy: Auto (adaptive)")
        outline_source = st.selectbox(
            "Outline source",
            ["Segmentation boundary (preferred)", "Pixel edges (fallback)"],
            index=0,
            help="For clean, closed silhouettes, derive the outline from the segmentation mask boundary instead of Canny edges.",
        )
        task_policy = load_policy()
        preset = "navigation"
        if task_policy:
            presets = task_policy.get_available_presets()
            preset = st.selectbox("Task preset", presets, index=0)
        with st.expander("Processing"):
            st.caption("YOLO labeler, Segmentation, Saliency & Attention, Edge Detection, Fusion & Motion in tabs.")

        # Minimal calibration controls (kept clean in one expander)
        with st.expander("Calibration", expanded=False):
            dot_density = st.slider("Dot density", 0.05, 0.22, 0.12, 0.01, help="Overall % of electrodes activated (approx).")
            subject_weight = st.slider("Subject priority", 0.30, 0.70, 0.50, 0.05, help="Budget share for main subject edges.")
            near_weight = st.slider("Near-field priority", 0.10, 0.55, 0.22, 0.05, help="Budget share for near-field safety edges.")
            fill_weight = st.slider(
                "Interior fill (luminance)",
                0.0,
                0.45,
                0.18,
                0.01,
                help="Mix some luminance/contrast into stimulation so the percept doesn't collapse to only bright edges.",
            )
            bg_suppress = st.slider(
                "Background suppression",
                0.0,
                1.0,
                0.75,
                0.05,
                help="How strongly to suppress edges outside segmented/YOLO/saliency foreground (higher = less grass texture).",
            )
            temporal_smooth = st.slider(
                "Temporal smoothing (video)",
                0.0,
                0.90,
                0.55,
                0.05,
                help="Exponential smoothing on the stimulation grid for video: higher = more stable but more lag.",
            )
            mask_kernel = st.slider(
                "Mask cleanup kernel",
                3,
                11,
                7,
                2,
                help="Morphological cleanup size for segmentation masks (close/fill/open).",
            )
            target_class = st.selectbox(
                "Segmentation target",
                ["auto", "dog", "person", "foreground"],
                index=0,
                help="Which class to outline when using segmentation boundary. Auto uses top YOLO class if supported.",
            )
            interior_frac = st.slider(
                "Interior dot fraction",
                0.0,
                0.60,
                0.20,
                0.05,
                help="Fraction of the dot budget used inside the subject mask (in addition to the outline).",
            )
            fg_frac = st.slider(
                "Foreground dot fraction",
                0.0,
                0.40,
                0.10,
                0.05,
                help="Fraction of the dot budget used for foreground/near-field dots (useful for navigation).",
            )
            min_sep = st.slider(
                "Min dot spacing (grid)",
                0,
                4,
                1,
                1,
                help="Minimum spacing between selected dots on the 60×60 grid (reduces clumps).",
            )

        adaptive_hold = 3
        adaptive_margin = 0.03
        adaptive_gate_thresh = 0.18
        show_candidate_table = False
        if encoding_policy.startswith("Auto"):
            with st.expander("Adaptive encoding", expanded=False):
                adaptive_hold = st.slider("Hold frames (video)", 0, 10, 3, 1, help="Minimum frames to keep the current encoding before switching (unless clearly better).")
                adaptive_margin = st.slider("Switch margin", 0.0, 0.15, 0.03, 0.01, help="Required score improvement before switching encodings.")
                adaptive_gate_thresh = st.slider("Foreground gate threshold", 0.05, 0.60, 0.18, 0.05, help="Threshold on the fused foreground gate used for scoring (higher = stricter foreground).")
                show_candidate_table = st.checkbox("Show candidate comparison", value=False)

        dot_sigma = 1.6
        dot_blend = "sum"
        dot_jitter = 0.0
        with st.expander("Dot renderer", expanded=False):
            dot_sigma = st.slider("Dot σ (px)", 0.8, 3.5, 1.6, 0.1)
            dot_blend = st.selectbox("Dot blend", ["sum", "max"], index=0)
            dot_jitter = st.slider("Dot jitter (px)", 0.0, 1.5, 0.0, 0.1)
        with st.expander("Edge prefilter", expanded=False):
            use_bilateral = st.checkbox(
                "Bilateral filter before edges",
                value=True,
                help="Reduces high-frequency textures (e.g., grass) before edge extraction.",
            )

    uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "gif"])
    if uploaded is None:
        st.info("Upload an image or video to start.")
        return

    # Reset temporal/adaptive state on new input to avoid drift from prior media.
    upload_id = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
    if st.session_state.get("_current_upload_id") != upload_id:
        st.session_state["_current_upload_id"] = upload_id
        st.session_state.pop("_stim_g_smooth", None)
        st.session_state.pop("_prev_stim_grid", None)
        st.session_state.pop("_adaptive_choice_name", None)
        st.session_state.pop("_adaptive_choice_score", None)
        st.session_state.pop("_adaptive_stable_frames", None)

    is_video = uploaded.type.startswith("video") or uploaded.name.lower().endswith((".mp4", ".avi", ".mov", ".gif"))
    frames: List[np.ndarray] = []
    if is_video:
        bytes_data = uploaded.read()
        import tempfile
        import os
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
            st.warning("No frames in video.")
            return
        frame_idx = st.slider("Frame", 0, len(frames) - 1, 0)
        img = frames[frame_idx]
    else:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not load image.")
        return

    # Match reference layout: Input | Motion/Edges | Phosphene
    col_img, col_mid, col_out = st.columns(3)
    with col_img:
        st.subheader("Input")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    # Motion: for video, use prev/curr frames; for image, zeros
    motion_magnitude = None
    motion_result = None
    if is_video and len(frames) > 1 and frame_idx > 0:
        from phosphene_toolkit.perception.motion import compute_motion_between_frames
        motion_result = compute_motion_between_frames(frames[frame_idx - 1], frames[frame_idx])
        motion_magnitude = motion_result.get("magnitude")

    # Run all models (phosphene stim is edge-driven, but we build a priority mask to suppress background texture)
    saliency_map = None
    sal = load_saliency()
    if sal:
        saliency_map = sal.compute_saliency(img, method="combined")

    # Add a low-frequency luminance/contrast channel (helps differentiate images beyond just edges)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_c = clahe.apply(gray)
    except Exception:
        gray_c = gray
    luminance = (gray_c.astype(np.float32) / 255.0)
    luminance = cv2.GaussianBlur(luminance, (0, 0), 1.0)
    luminance = _robust_normalize(luminance, 2.0, 98.0)
    # Add local contrast (helps when raw luminance is flat or lighting dominates)
    lp = cv2.GaussianBlur(luminance, (0, 0), 3.0)
    detail = np.abs(luminance - lp).astype(np.float32)
    detail = _robust_normalize(detail, 2.0, 99.0)
    luminance = _robust_normalize(0.65 * luminance + 0.35 * detail, 1.0, 99.0)

    enc = load_encoder()
    segmentation_map = None
    segmentation_fg = None
    pred_full = None
    if enc and hasattr(enc, "segmentation"):
        try:
            seg_result = enc.segmentation.segment(img)
            segmentation_map = seg_result.get("segmentation")
            pred_mask = seg_result.get("pred_mask")
            if pred_mask is not None:
                # Foreground = any non-background class
                pred_full = cv2.resize(pred_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                segmentation_fg = (pred_full > 0).astype(np.float32)
                segmentation_fg = cv2.GaussianBlur(segmentation_fg, (0, 0), 1.2)
        except Exception:
            pass

    annotated, detections = run_yolo_labeler(img, conf=0.25)
    yolo_heatmap = None
    if detections:
        from utils.ai_inference_fusion import yolo_detections_to_heatmap
        # Prefer the most confident detections (usually the main subject)
        det_sorted = sorted(detections, key=lambda d: float(d.get("conf", 0.0)), reverse=True)[:3]
        yolo_heatmap = yolo_detections_to_heatmap(det_sorted, img.shape[:2], sigma=60.0)
        primary_det = det_sorted[0] if len(det_sorted) > 0 else None
    else:
        primary_det = None

    from utils.ai_inference_fusion import (
        fuse_ai_attention,
        attention_to_heatmap_colored,
    )
    from utils.dot_phosphene_renderer import DotRenderParams, render_dots_from_grid

    # Build an object mask for boundary outline (class-specific when possible)
    object_mask_u8 = None
    yolo_roi = None
    if pred_full is not None and pred_full.size > 0:
        # Try to pick a class id based on YOLO or user selection
        class_id = None
        if target_class != "foreground":
            try:
                from phosphene_toolkit.perception.segmentation import COCO_LABELS
                name_to_id = {n: i for i, n in enumerate(COCO_LABELS)}
                if target_class == "auto" and primary_det is not None:
                    cname = str(primary_det.get("class", "")).strip().lower()
                    class_id = name_to_id.get(cname)
                elif target_class in name_to_id:
                    class_id = name_to_id.get(target_class)
            except Exception:
                class_id = None
        if class_id is not None:
            object_mask_u8 = ((pred_full == int(class_id)).astype(np.uint8) * 255)
            # If class-specific mask is tiny/empty, fall back to foreground mask.
            if int(np.sum(object_mask_u8 > 0)) < 100:
                object_mask_u8 = ((pred_full > 0).astype(np.uint8) * 255)
        else:
            object_mask_u8 = ((pred_full > 0).astype(np.uint8) * 255)

        # If YOLO exists, restrict to its ROI (plus margin) to avoid background leaks
        roi = None
        if primary_det is not None:
            x1, y1, x2, y2 = [float(v) for v in primary_det.get("bbox", (0, 0, img.shape[1], img.shape[0]))[:4]]
            mx = 0.08 * (x2 - x1)
            my = 0.08 * (y2 - y1)
            roi = (x1 - mx, y1 - my, x2 + mx, y2 + my)
            yolo_roi = roi
        if roi is not None:
            x1, y1, x2, y2 = roi
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
            tmp = np.zeros_like(object_mask_u8)
            tmp[y1:y2, x1:x2] = object_mask_u8[y1:y2, x1:x2]
            object_mask_u8 = tmp

        # Clean mask: close -> fill holes -> open -> keep largest component
        k = max(3, int(mask_kernel) | 1)
        ker = np.ones((k, k), np.uint8)
        object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_CLOSE, ker, iterations=1)
        object_mask_u8 = _binary_fill_holes(object_mask_u8)
        object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_OPEN, np.ones((max(3, k // 2) | 1, max(3, k // 2) | 1), np.uint8), iterations=1)
        object_mask_u8 = _keep_largest_component(object_mask_u8, roi=roi)
        # Sanity: if segmentation mask drifts far from YOLO ROI, prefer ROI-based mask.
        if yolo_roi is not None and int(np.sum(object_mask_u8 > 0)) > 50:
            mb = _mask_bbox(object_mask_u8)
            if mb is not None and _bbox_iou(mb, yolo_roi) < 0.08:
                x1, y1, x2, y2 = yolo_roi
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
                tmp = np.zeros_like(object_mask_u8)
                tmp[y1:y2, x1:x2] = 255
                object_mask_u8 = tmp

    # Semantic gating BEFORE edges: foreground = segmentation OR yolo OR saliency.
    gate = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    if segmentation_fg is not None and segmentation_fg.size > 0:
        gate = np.maximum(gate, np.clip(segmentation_fg.astype(np.float32), 0.0, 1.0))
    if yolo_heatmap is not None and yolo_heatmap.size > 0:
        yh = np.clip(yolo_heatmap.astype(np.float32), 0.0, 1.0)
        yh = _robust_normalize(yh, 1.0, 99.0)
        gate = np.maximum(gate, yh)
    if saliency_map is not None and saliency_map.size > 0:
        sm = np.asarray(saliency_map, dtype=np.float32)
        sm = cv2.resize(sm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        sm = np.clip(_robust_normalize(sm, 2.0, 98.0), 0.0, 1.0)
        gate = np.maximum(gate, sm)
    gate = cv2.GaussianBlur(np.clip(gate, 0.0, 1.0), (0, 0), 2.0)
    gate = np.clip(_robust_normalize(gate, 1.0, 99.0), 0.0, 1.0)

    # Fallback object mask: if segmentation-specific mask is missing but YOLO has a primary box,
    # derive a coarse object region from bbox ∩ gate to preserve thin limbs/appendages.
    if (object_mask_u8 is None or int(np.sum(object_mask_u8 > 0)) < 50) and primary_det is not None:
        try:
            x1, y1, x2, y2 = [float(v) for v in primary_det.get("bbox", (0, 0, img.shape[1], img.shape[0]))[:4]]
            mx = 0.10 * (x2 - x1)
            my = 0.10 * (y2 - y1)
            x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            x2, y2 = min(img.shape[1], int(x2 + mx)), min(img.shape[0], int(y2 + my))
            bb = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            bb[y1:y2, x1:x2] = 255
            gm = (gate > 0.12).astype(np.uint8) * 255
            object_mask_u8 = cv2.bitwise_and(bb, gm)
            object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            object_mask_u8 = _binary_fill_holes(object_mask_u8)
            object_mask_u8 = _keep_largest_component(object_mask_u8, roi=(x1, y1, x2, y2))
            yolo_roi = (x1, y1, x2, y2)
        except Exception:
            pass

    # Near-field (safety) prior: bottom region + (optional) inverse depth.
    yy = np.linspace(0.0, 1.0, img.shape[0], dtype=np.float32)[:, None]
    near = np.repeat(yy, img.shape[1], axis=1)
    near = np.power(np.clip(near, 0, 1), 2.2)  # concentrate on bottom region
    depth_est = load_depth()
    if depth_est is not None:
        try:
            depth_res = depth_est.predict(img, output_size=(img.shape[0], img.shape[1]))
            near = np.maximum(near, depth_res.inv_depth.astype(np.float32))
        except Exception:
            pass
    near = cv2.GaussianBlur(near, (0, 0), 3.0)
    near = (near - near.min()) / (near.max() - near.min() + 1e-8)

    def _topk_mask(score: np.ndarray, k: int, forbid: np.ndarray) -> np.ndarray:
        if k <= 0:
            return np.zeros_like(score, dtype=np.uint8)
        s = score.copy()
        s[np.asarray(forbid, dtype=bool)] = -1.0
        flat = s.reshape(-1)
        if k >= flat.size:
            m = (flat >= 0).astype(np.uint8).reshape(score.shape)
            return m
        idx = np.argpartition(flat, -k)[-k:]
        m = np.zeros_like(flat, dtype=np.uint8)
        m[idx] = 1
        return m.reshape(score.shape)

    def _build_for_candidate(cand: Optional[EncodingCandidate]) -> Dict[str, Any]:
        # Candidate = None means "manual pipeline" using sidebar settings.
        if cand is None:
            outline_mode = "seg_boundary" if outline_source.startswith("Segmentation") else "pixel_edges"
            edge_method = "canny_multi"
            cand_fill = float(fill_weight)
            cand_bg = float(bg_suppress)
            gray_base = gray_c.astype(np.uint8)
            lum_map = luminance
            cand_use_bilateral = bool(use_bilateral)
        else:
            outline_mode = str(cand.outline_mode)
            edge_method = str(cand.edge_method)
            cand_fill = float(cand.fill_weight)
            cand_bg = float(cand.bg_suppress)
            gray_base = _candidate_gray(gray.astype(np.uint8), cand)
            lum_map = _luminance_from_gray(gray_base)
            cand_use_bilateral = bool(cand.use_bilateral)

        has_obj = object_mask_u8 is not None and int(np.sum(np.asarray(object_mask_u8) > 0)) > 50
        use_boundary = (outline_mode == "seg_boundary") and has_obj

        # Outline map: either from segmentation boundary (preferred) or from pixel edges (fallback)
        if use_boundary:
            edges_map_local = _mask_boundary(object_mask_u8, ksize=3)
            edges_map_local = np.clip(
                edges_map_local * (object_mask_u8.astype(np.float32) / 255.0), 0.0, 1.0
            ).astype(np.float32)
            # Boundary can be too sparse/fragmented on some masks; auto-fallback to pixel edges.
            if float(np.mean(edges_map_local > 0.05)) < 0.0015:
                use_boundary = False
        else:
            edges_map_local = None

        if not use_boundary:
            gray_for_edges = gray_base.astype(np.uint8)
            if cand is None and cand_use_bilateral:
                try:
                    gray_for_edges = cv2.bilateralFilter(gray_for_edges, d=7, sigmaColor=55, sigmaSpace=7)
                except Exception:
                    pass
            edges_map_local = compute_edges(gray_for_edges, method=edge_method)
            edges_u8 = (np.clip(edges_map_local, 0, 1) * 255).astype(np.uint8)
            edges_u8 = cv2.morphologyEx(edges_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            edges_map_local = (edges_u8.astype(np.float32) / 255.0)
            pad = max(2, int(0.02 * min(edges_map_local.shape[0], edges_map_local.shape[1])))
            edges_map_local[:pad, :] = 0.0
            edges_map_local[-pad:, :] = 0.0
            edges_map_local[:, :pad] = 0.0
            edges_map_local[:, -pad:] = 0.0
            p = 1.0 + 3.0 * float(cand_bg)
            gated = (0.02 + 0.98 * np.power(np.clip(gate, 0.0, 1.0), p)).astype(np.float32)
            edges_map_local = np.clip(edges_map_local * gated, 0.0, 1.0).astype(np.float32)
            edges_map_local[gate < (0.10 + 0.20 * float(cand_bg))] *= (1.0 - 0.85 * float(cand_bg))

        # Build a semantic-ish attention map: downweight raw edges so texture (grass) doesn't dominate
        attention_map_local = fuse_ai_attention(
            img,
            saliency=saliency_map,
            segmentation=segmentation_fg if segmentation_fg is not None else segmentation_map,
            edges=edges_map_local,
            motion_magnitude=motion_magnitude,
            yolo_heatmap=yolo_heatmap,
            weights={"saliency": 2.4, "segmentation": 2.4, "edges": 0.25, "motion": 1.0, "yolo": 3.2},
        )

        # Motion-tracked edges (like your example)
        if motion_magnitude is not None and motion_magnitude.size > 0:
            mot = np.clip(motion_magnitude.astype(np.float32), 0, 1)
            mot = (mot - mot.min()) / (mot.max() - mot.min() + 1e-8)
        else:
            mot = np.zeros_like(edges_map_local, dtype=np.float32)

        motion_edges_local = np.clip(edges_map_local * (1.0 + 2.0 * mot), 0, 1).astype(np.float32)
        motion_edges_local = cv2.GaussianBlur(motion_edges_local, (0, 0), 0.6)
        motion_edges_local = (motion_edges_local - motion_edges_local.min()) / (motion_edges_local.max() - motion_edges_local.min() + 1e-8)

        # Priority mask: combine AI attention + (optional) YOLO heatmap (subject), but ALWAYS preserve some near-field foreground.
        priority_obj_local = attention_map_local.copy()
        if yolo_heatmap is not None and yolo_heatmap.size > 0:
            yh = np.clip(yolo_heatmap.astype(np.float32), 0, 1)
            yh = (yh - yh.min()) / (yh.max() - yh.min() + 1e-8)
            priority_obj_local = np.maximum(priority_obj_local, yh)
        priority_obj_local = (priority_obj_local - priority_obj_local.min()) / (priority_obj_local.max() - priority_obj_local.min() + 1e-8)
        priority_obj_local = cv2.GaussianBlur(priority_obj_local, (0, 0), 1.6)
        priority_obj_local = (priority_obj_local - priority_obj_local.min()) / (priority_obj_local.max() - priority_obj_local.min() + 1e-8)

        # Adaptive budgeting on the electrode grid: allocate dots to (subject edges) + (near-field edges) + (global edges).
        grid_n_local = 60
        edge_g_local = cv2.resize(motion_edges_local, (grid_n_local, grid_n_local), interpolation=cv2.INTER_AREA).astype(np.float32)
        lum_g_local = cv2.resize(lum_map, (grid_n_local, grid_n_local), interpolation=cv2.INTER_AREA).astype(np.float32)
        obj_g_local = cv2.resize(priority_obj_local, (grid_n_local, grid_n_local), interpolation=cv2.INTER_AREA).astype(np.float32)
        near_g_local = cv2.resize(near, (grid_n_local, grid_n_local), interpolation=cv2.INTER_AREA).astype(np.float32)
        edge_g_local = _robust_normalize(edge_g_local, 1.0, 99.0)
        lum_g_local = _robust_normalize(lum_g_local, 1.0, 99.0)
        near_g_local = _robust_normalize(near_g_local, 1.0, 99.0)

        lum_fg = np.clip(lum_g_local * (0.10 + 0.90 * obj_g_local), 0.0, 1.0)
        score_g_local = np.clip(
            (1.0 - float(cand_fill)) * edge_g_local + float(cand_fill) * lum_fg, 0.0, 1.0
        ).astype(np.float32)
        score_g_local = _robust_normalize(score_g_local, 1.0, 99.0)

        # Determine dot budget
        edge_density = float(np.mean(edge_g_local > 0.25))
        base_budget = int(grid_n_local * grid_n_local * float(dot_density))  # user-calibrated
        if cand is None:
            budget_local = int(np.clip(base_budget * (0.9 / (edge_density + 0.15)), grid_n_local * grid_n_local * 0.05, grid_n_local * grid_n_local * 0.16))
        else:
            # Adaptive candidates get a mild density-dependent budget boost to preserve thin structures.
            budget_local = int(
                np.clip(
                    base_budget * (1.05 / (edge_density + 0.35)),
                    grid_n_local * grid_n_local * 0.07,
                    grid_n_local * grid_n_local * 0.20,
                )
            )

        obj_budget = int(budget_local * float(subject_weight))
        near_budget = int(budget_local * float(near_weight))
        global_budget = max(0, budget_local - obj_budget - near_budget)

        used = np.zeros((grid_n_local, grid_n_local), dtype=np.uint8)
        if use_boundary:
            k_in = int(float(interior_frac) * budget_local)
            k_fg = int(float(fg_frac) * budget_local)
            k_outline = max(0, budget_local - k_in - k_fg)

            contours, _ = cv2.findContours((object_mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours and k_outline > 0:
                contour = max(contours, key=lambda c: float(cv2.contourArea(c)))
                used = _contour_points_to_grid_mask(contour, (img.shape[0], img.shape[1]), grid_n_local, k_outline)
            elif k_outline > 0:
                used = _topk_minsep_mask(score_g_local, k_outline, min_sep=int(min_sep))

            mask_g = cv2.resize((object_mask_u8.astype(np.float32) / 255.0), (grid_n_local, grid_n_local), interpolation=cv2.INTER_AREA).astype(np.float32)
            mask_g = np.clip(mask_g, 0.0, 1.0)
            mask_bin = (mask_g > 0.25).astype(np.uint8) * 255

            if k_in > 0:
                inner = cv2.erode(mask_bin, np.ones((3, 3), np.uint8), iterations=1)
                inner_f = (inner.astype(np.float32) / 255.0)
                score_in = _robust_normalize(0.60 * obj_g_local + 0.40 * lum_g_local, 1.0, 99.0) * inner_f
                forbid = (used > 0).astype(np.uint8)
                used_in = _topk_minsep_mask(score_in, k_in, min_sep=int(min_sep), forbid=forbid)
                used = np.maximum(used, used_in)

            if k_fg > 0:
                outside = (mask_g < 0.15).astype(np.float32)
                score_fg = _robust_normalize(near_g_local, 1.0, 99.0) * outside
                forbid = (used > 0).astype(np.uint8)
                used_fg = _topk_minsep_mask(score_fg, k_fg, min_sep=max(0, int(min_sep) - 1), forbid=forbid)
                used = np.maximum(used, used_fg)
        else:
            m_obj = _topk_mask(score_g_local * (0.25 + 0.75 * obj_g_local), obj_budget, used)
            used = np.maximum(used, m_obj)
            m_near = _topk_mask(score_g_local * (0.30 + 0.70 * near_g_local), near_budget, used)
            used = np.maximum(used, m_near)
            m_global = _topk_mask(score_g_local, global_budget, used)
            used = np.maximum(used, m_global)

        score_combined = score_g_local * (0.75 * obj_g_local + 0.15 * near_g_local + 0.10)
        score_combined = np.clip(_robust_normalize(score_combined, 1.0, 99.0), 0.0, 1.0)
        stim_g_local = np.clip(score_combined * used.astype(np.float32), 0.0, 1.0).astype(np.float32)

        return {
            "edges_map": edges_map_local,
            "motion_edges": motion_edges_local,
            "attention_map": attention_map_local,
            "priority_obj": priority_obj_local,
            "edge_g": edge_g_local,
            "lum_g": lum_g_local,
            "obj_g": obj_g_local,
            "near_g": near_g_local,
            "stim_g": stim_g_local,
            "budget": int(budget_local),
            "edge_density": float(edge_density),
            "use_boundary": bool(use_boundary),
        }

    adaptive_debug: Optional[Dict[str, Any]] = None
    if encoding_policy.startswith("Auto"):
        has_obj = object_mask_u8 is not None and int(np.sum(np.asarray(object_mask_u8) > 0)) > 50
        candidates = default_candidates(has_object_boundary=bool(has_obj))

        fg_mask_g = grid_foreground_mask(
            gate=gate, object_mask_u8=object_mask_u8, grid_n=60, gate_thresh=float(adaptive_gate_thresh)
        )
        boundary_mask_g = grid_boundary_mask(object_mask_u8=object_mask_u8, grid_n=60) if has_obj else None

        prev_for_flicker = st.session_state.get("_prev_stim_grid")
        scored: List[Tuple[EncodingCandidate, CandidateScore]] = []
        cache: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            res = _build_for_candidate(cand)
            cache[cand.name] = res
            cs = score_stim_grid(
                stim_grid=res["stim_g"],
                fg_mask_g=fg_mask_g,
                boundary_mask_g=boundary_mask_g,
                target_active_dots=int(res["budget"]),
                prev_stim_grid=prev_for_flicker,
            )
            scored.append((cand, cs))

        prev_choice = st.session_state.get("_adaptive_choice_name")
        prev_score = st.session_state.get("_adaptive_choice_score")
        stable_frames = int(st.session_state.get("_adaptive_stable_frames", 0))
        chosen_c, chosen_s, switched = select_with_hysteresis(
            scored=scored,
            prev_choice_name=str(prev_choice) if prev_choice is not None else None,
            prev_score=float(prev_score) if prev_score is not None else None,
            stable_frames=stable_frames,
            min_hold_frames=int(adaptive_hold),
            margin=float(adaptive_margin),
        )
        if switched:
            stable_frames = 0
        stable_frames += 1
        st.session_state["_adaptive_choice_name"] = chosen_c.name
        st.session_state["_adaptive_choice_score"] = float(chosen_s.score)
        st.session_state["_adaptive_stable_frames"] = int(stable_frames)

        chosen_res = cache.get(chosen_c.name) or _build_for_candidate(chosen_c)
        edges_map = chosen_res["edges_map"]
        motion_edges = chosen_res["motion_edges"]
        attention_map = chosen_res["attention_map"]
        priority_obj = chosen_res["priority_obj"]
        edge_g = chosen_res["edge_g"]
        obj_g = chosen_res["obj_g"]
        near_g = chosen_res["near_g"]
        stim_g = chosen_res["stim_g"]

        adaptive_debug = {"chosen": chosen_c.name, "score": chosen_s, "stable_frames": stable_frames}

        if show_candidate_table:
            rows = []
            for c, s in sorted(scored, key=lambda t: float(t[1].score), reverse=True):
                rows.append(
                    {
                        "candidate": c.name,
                        "score": round(float(s.score), 4),
                        "fg_energy": round(float(s.fg_energy), 3),
                        "bnd_energy": round(float(s.boundary_energy), 3),
                        "leak": round(float(s.leak_energy), 3),
                        "dots": int(s.active_dots),
                        "flicker": round(float(s.flicker), 3),
                    }
                )
            with st.sidebar:
                st.caption("Adaptive candidates (higher score = better)")
                st.dataframe(rows, use_container_width=True, height=230)
    else:
        # Kept only as defensive fallback; UI is adaptive-only.
        manual_res = _build_for_candidate(None)
        edges_map = manual_res["edges_map"]
        motion_edges = manual_res["motion_edges"]
        attention_map = manual_res["attention_map"]
        priority_obj = manual_res["priority_obj"]
        edge_g = manual_res["edge_g"]
        obj_g = manual_res["obj_g"]
        near_g = manual_res["near_g"]
        stim_g = manual_res["stim_g"]

    with col_mid:
        st.subheader("Motion / Edges")
        st.image((motion_edges * 255).astype(np.uint8), use_container_width=True)

    if adaptive_debug is not None:
        with st.sidebar:
            cs: CandidateScore = adaptive_debug["score"]
            st.caption(f"Adaptive choice: `{adaptive_debug['chosen']}` (stable: {adaptive_debug['stable_frames']} frames)")
            st.write(
                {
                    "score": round(float(cs.score), 4),
                    "fg_energy": round(float(cs.fg_energy), 3),
                    "boundary_energy": round(float(cs.boundary_energy), 3),
                    "leak": round(float(cs.leak_energy), 3),
                    "flicker": round(float(cs.flicker), 3),
                    "dots": int(cs.active_dots),
                }
            )

    grid_n = 60
    # (edge_g, obj_g, near_g, stim_g) are produced above (manual or adaptive)

    # Temporal stabilization (video): smooth stimulation to reduce flicker and keep object stable.
    if is_video and float(temporal_smooth) > 1e-6:
        prev_s = st.session_state.get("_stim_g_smooth")
        if prev_s is None or np.asarray(prev_s).shape != stim_g.shape:
            prev_s = stim_g.copy()
        a = float(np.clip(temporal_smooth, 0.0, 0.95))
        stim_g = (a * np.asarray(prev_s, dtype=np.float32) + (1.0 - a) * stim_g).astype(np.float32)
        st.session_state["_stim_g_smooth"] = stim_g.copy()
    stim = cv2.resize(stim_g, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    stim = np.clip(stim, 0.0, 1.0)

    # Priority mask for renderer thresholding: blend object + near-field (so grass/foreground isn't eliminated)
    priority = np.clip(0.65 * priority_obj + 0.35 * near, 0, 1).astype(np.float32)
    priority = (priority - priority.min()) / (priority.max() - priority.min() + 1e-8)

    # Dots-only encoder: render directly from sparse stimulation grid.
    percept = render_dots_from_grid(
        stim_g,
        output_size=(img.shape[0], img.shape[1]),
        params=DotRenderParams(sigma_px=float(dot_sigma), blend=str(dot_blend), jitter_px=float(dot_jitter)),
    )

    # Minimal metrics (clean + interpretable)
    from utils.processing_metrics import compute_metrics
    prev_stim = st.session_state.get("_prev_stim_grid")
    metrics = compute_metrics(
        edge_grid=edge_g,
        stim_grid=stim_g,
        priority_obj_grid=obj_g,
        near_grid=near_g,
        prev_stim_grid=prev_stim,
    )
    st.session_state["_prev_stim_grid"] = stim_g.copy()
    with st.sidebar:
        with st.expander("Metrics", expanded=False):
            st.write(
                {
                    "active_dots": int(metrics["active_dots"]),
                    "active_pct": round(metrics["active_pct"], 1),
                    "edge_density": round(metrics["edge_density"], 3),
                    "flicker": round(metrics["flicker"], 3),
                }
            )

    with col_out:
        st.subheader("Phosphene")
        st.image(percept, channels="GRAY", use_container_width=True)
        st.caption("Phosphene Representation")

    tabs = st.tabs(["YOLO Labeler", "Segmentation", "Saliency & Attention", "Edge Detection", "Fusion & Motion"])
    with tabs[0]:
        st.caption("YOLO object detection with bounding boxes and class labels")
        conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, key="yolo_conf")
        annotated, detections = run_yolo_labeler(img, conf=conf_thresh)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if detections:
            st.write("**Detected objects:**")
            for d in detections:
                st.write(f"- **{d['class']}** ({d['conf']:.2f}) at bbox {[round(x,1) for x in d['bbox']]}")

    with tabs[1]:
        st.caption("DeepLab V3 segmentation (colored overlay)")
        enc = load_encoder()
        if enc and hasattr(enc, "segmentation"):
            seg = enc.segmentation.segment(img)
            overlay = seg.get("colored_overlay")
            sm = seg.get("segmentation", np.zeros_like(img[:, :, 0], dtype=np.float32))
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            with col_b:
                sm_vis = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
                st.image((sm_vis * 255).astype(np.uint8), use_container_width=True)
        else:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    with tabs[2]:
        st.caption("Saliency & attention heatmap (warm = high, cool = low)")
        saliency_heatmap = attention_to_heatmap_colored(attention_map)
        st.image(saliency_heatmap, channels="RGB", use_container_width=True)

    with tabs[3]:
        st.caption("Edge detection")
        st.image((edges_map * 255).astype(np.uint8), use_container_width=True)

    with tabs[4]:
        st.caption("Fusion & motion (optical flow tracking)")
        if motion_result is not None:
            mag = motion_result.get("magnitude", np.zeros_like(img[:, :, 0], dtype=np.float32))
            flow_vis = motion_result.get("flow_vis")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Motion magnitude** (brighter = more motion)")
                mag_vis = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
                st.image((mag_vis * 255).astype(np.uint8), use_container_width=True)
            with col_b:
                st.write("**Optical flow** (color = direction, brightness = magnitude)")
                if flow_vis is not None and flow_vis.size > 0:
                    st.image(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                else:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        else:
            if is_video and frame_idx == 0:
                st.info("Select frame 1 or later to see motion (optical flow between consecutive frames).")
            else:
                st.info("Upload a video and select frame ≥ 1 to see motion tracking.")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)


if __name__ == "__main__":
    main()
