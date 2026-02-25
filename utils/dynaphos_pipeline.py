"""
Dynaphos image/video → phosphene pipeline using pulse2percept.

Translates images and videos into phosphene representations using DynaphosModel
(eLife 2024) and Orion cortical implant. Uses CorticalPhospheneSimulator internally
with edge/grayscale preprocessing for stimulus generation.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Tuple, Generator, Dict, Any, List
from PIL import Image

try:
    from .cortical_phosphene import CorticalPhospheneSimulator, DYNAPHOS_AVAILABLE
except ImportError:
    try:
        from cortical_phosphene import CorticalPhospheneSimulator, DYNAPHOS_AVAILABLE
    except ImportError:
        CorticalPhospheneSimulator = None  # type: ignore
        DYNAPHOS_AVAILABLE = False


def _clahe_u8(gray_u8: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    return clahe.apply(np.asarray(gray_u8, dtype=np.uint8))


def _gamma_u8(gray_u8: np.ndarray, gamma: float) -> np.ndarray:
    g = float(max(1e-3, gamma))
    x = np.asarray(gray_u8, dtype=np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), g)
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)


def _retinex_ssr_u8(gray_u8: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    I = np.asarray(gray_u8, dtype=np.float32) / 255.0
    I = np.clip(I, 1e-6, 1.0)
    blur = cv2.GaussianBlur(I, (0, 0), float(sigma))
    blur = np.clip(blur, 1e-6, 1.0)
    R = np.log(I) - np.log(blur)
    mn, mx = float(np.min(R)), float(np.max(R))
    if mx - mn > 1e-8:
        R = (R - mn) / (mx - mn)
    return (np.clip(R, 0.0, 1.0) * 255.0).astype(np.uint8)


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn > 1e-8:
        return np.clip((x - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _score_stim_proxy(
    stim_g: np.ndarray,
    *,
    fg_mask_g: Optional[np.ndarray] = None,
    boundary_mask_g: Optional[np.ndarray] = None,
    target_active: Optional[int] = None,
    prev_stim_g: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    stim = np.clip(np.asarray(stim_g, dtype=np.float32), 0.0, 1.0)
    total_e = float(np.sum(stim) + 1e-8)
    active = int(np.sum(stim > 1e-6))

    fg_energy = 0.0
    leak_energy = 0.0
    if fg_mask_g is not None and np.asarray(fg_mask_g).size > 0:
        fg = (np.asarray(fg_mask_g, dtype=np.float32) > 0.5).astype(np.float32)
        fg_energy = float(np.sum(stim * fg)) / total_e
        leak_energy = float(np.sum(stim * (1.0 - fg))) / total_e

    boundary_energy = 0.0
    if boundary_mask_g is not None and np.asarray(boundary_mask_g).size > 0:
        b = (np.asarray(boundary_mask_g, dtype=np.float32) > 0.5).astype(np.float32)
        boundary_energy = float(np.sum(stim * b)) / total_e

    dot_err = 0.0
    if target_active is not None and int(target_active) > 0:
        dot_err = abs(active - int(target_active)) / float(target_active)

    flicker = 0.0
    if prev_stim_g is not None and np.asarray(prev_stim_g).shape == stim.shape:
        prev = np.asarray(prev_stim_g, dtype=np.float32)
        flicker = float(np.mean(np.abs((stim > 1e-6).astype(np.float32) - (prev > 1e-6).astype(np.float32))))

    score = 1.30 * fg_energy + 0.55 * boundary_energy - 1.20 * leak_energy - 0.25 * dot_err - 0.45 * flicker
    return {
        "score": float(score),
        "fg_energy": float(fg_energy),
        "boundary_energy": float(boundary_energy),
        "leak": float(leak_energy),
        "dots": float(active),
        "dot_err": float(dot_err),
        "flicker": float(flicker),
    }


def _stim_from_mode(
    gray_u8: np.ndarray,
    *,
    mode: str,
    grid_size: Tuple[int, int],
    edge_method: str = "canny",
    low: int = 50,
    high: int = 150,
) -> np.ndarray:
    h, w = grid_size
    if mode == "grayscale":
        return cv2.resize(gray_u8.astype(np.float32) / 255.0, (w, h)).astype(np.float64)
    if mode == "invert":
        return (1.0 - cv2.resize(gray_u8.astype(np.float32) / 255.0, (w, h))).astype(np.float64)

    # edges
    g = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    if edge_method == "canny_multi":
        e1 = cv2.Canny(g, 30, 80)
        e2 = cv2.Canny(g, int(low), int(high))
        e3 = cv2.Canny(g, 100, 200)
        edges = np.maximum(np.maximum(e1, e2), e3)
    else:
        edges = cv2.Canny(g, int(low), int(high))
    return cv2.resize(edges.astype(np.float32) / 255.0, (w, h)).astype(np.float64)


def _image_to_stim_grid(
    image: np.ndarray,
    preprocess: str = "edges",
    grid_size: Tuple[int, int] = (60, 60),
    *,
    fg_mask_g: Optional[np.ndarray] = None,
    boundary_mask_g: Optional[np.ndarray] = None,
    prev_stim_g: Optional[np.ndarray] = None,
    target_active: Optional[int] = None,
    return_debug: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """Convert image to stimulation grid [0, 1]."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray_u8 = np.asarray(gray, dtype=np.uint8)

    if preprocess != "auto":
        if preprocess == "edges":
            stim = _stim_from_mode(gray_u8, mode="edges", grid_size=grid_size, edge_method="canny", low=50, high=150)
        elif preprocess == "invert":
            stim = _stim_from_mode(gray_u8, mode="invert", grid_size=grid_size)
        else:
            stim = _stim_from_mode(gray_u8, mode="grayscale", grid_size=grid_size)
        stim = np.clip(stim, 0.0, 1.0).astype(np.float64)
        if return_debug:
            return stim, {"chosen": preprocess}
        return stim

    # Auto mode: evaluate a small set of candidates and pick best by proxy score.
    candidates: List[Dict[str, Any]] = [
        {"name": "edges", "prep": "none", "mode": "edges", "edge_method": "canny_multi"},
        {"name": "edges_clahe", "prep": "clahe", "mode": "edges", "edge_method": "canny_multi"},
        {"name": "edges_gamma075", "prep": "gamma", "gamma": 0.75, "mode": "edges", "edge_method": "canny_multi"},
        {"name": "edges_retinex", "prep": "retinex", "mode": "edges", "edge_method": "canny_multi"},
        {"name": "grayscale", "prep": "clahe", "mode": "grayscale"},
        {"name": "invert", "prep": "clahe", "mode": "invert"},
    ]

    best = None
    best_debug: Dict[str, Any] = {}
    all_debug: List[Dict[str, Any]] = []
    for cand in candidates:
        g = gray_u8
        if cand.get("prep") == "clahe":
            g = _clahe_u8(g)
        elif cand.get("prep") == "gamma":
            g = _gamma_u8(g, float(cand.get("gamma", 1.0)))
        elif cand.get("prep") == "retinex":
            g = _retinex_ssr_u8(g, sigma=30.0)

        stim = _stim_from_mode(
            g,
            mode=str(cand["mode"]),
            grid_size=grid_size,
            edge_method=str(cand.get("edge_method", "canny")),
            low=50,
            high=150,
        )
        stim = np.clip(stim, 0.0, 1.0).astype(np.float64)
        dbg = _score_stim_proxy(
            stim,
            fg_mask_g=fg_mask_g,
            boundary_mask_g=boundary_mask_g,
            target_active=target_active,
            prev_stim_g=prev_stim_g,
        )
        dbg["candidate"] = str(cand["name"])
        all_debug.append(dbg)

        if best is None or float(dbg["score"]) > float(best_debug["score"]):
            best = stim
            best_debug = dbg

    if best is None:
        best = _stim_from_mode(gray_u8, mode="edges", grid_size=grid_size, edge_method="canny", low=50, high=150)
        best_debug = {"candidate": "edges_fallback", "score": 0.0}

    best = np.clip(best, 0.0, 1.0).astype(np.float64)
    if return_debug:
        return best, {"chosen": best_debug.get("candidate"), "score": best_debug, "all": all_debug}
    return best


class DynaphosPipeline:
    """
    End-to-end image/video → phosphene pipeline using DynaphosModel.

    Translates visual input into phosphene representations as perceived
    by cortical visual prosthesis (Orion) users.
    """

    def __init__(self, preprocess: str = "edges"):
        """
        Args:
            preprocess: "grayscale" | "edges" | "invert" | "auto" - image preprocessing
        """
        if not DYNAPHOS_AVAILABLE:
            raise ImportError("Dynaphos pipeline requires pulse2percept>=0.9")
        self.simulator = CorticalPhospheneSimulator()
        self.preprocess = preprocess
        self._prev_stim_g: Optional[np.ndarray] = None

    def image_to_phosphene(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int] = (480, 640),
        *,
        fg_mask_g: Optional[np.ndarray] = None,
        boundary_mask_g: Optional[np.ndarray] = None,
        target_active: Optional[int] = None,
        return_debug: bool = False,
    ) -> np.ndarray:
        """Convert single image to phosphene representation."""
        res = _image_to_stim_grid(
            image,
            self.preprocess,
            fg_mask_g=fg_mask_g,
            boundary_mask_g=boundary_mask_g,
            prev_stim_g=self._prev_stim_g,
            target_active=target_active,
            return_debug=return_debug,
        )
        if isinstance(res, tuple):
            stim_grid, dbg = res
        else:
            stim_grid, dbg = res, None
        self._prev_stim_g = np.asarray(stim_grid, dtype=np.float32)
        ph = self.simulator.simulate_from_grid(stim_grid, output_size=output_size, as_uint8=True)
        if return_debug and dbg is not None:
            # attach debug as attribute for downstream inspection
            setattr(self, "last_debug", dbg)
        return ph

    def video_to_phosphenes(
        self,
        video_path: str,
        output_size: Tuple[int, int] = (480, 640),
        max_frames: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Convert video to phosphene frame sequence."""
        cap = cv2.VideoCapture(video_path)
        count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and count >= max_frames:
                    break
                yield self.image_to_phosphene(frame, output_size=output_size)
                count += 1
        finally:
            cap.release()


def image_to_dynaphos_phosphene(
    image: np.ndarray,
    output_size: Tuple[int, int] = (480, 640),
    preprocess: str = "edges",
    *,
    fg_mask_g: Optional[np.ndarray] = None,
    boundary_mask_g: Optional[np.ndarray] = None,
    target_active: Optional[int] = None,
    return_debug: bool = False,
) -> Optional[np.ndarray]:
    """Convenience: convert image to Dynaphos phosphene. Returns None if pulse2percept not available."""
    if not DYNAPHOS_AVAILABLE:
        return None
    pipeline = DynaphosPipeline(preprocess=preprocess)
    return pipeline.image_to_phosphene(
        image,
        output_size=output_size,
        fg_mask_g=fg_mask_g,
        boundary_mask_g=boundary_mask_g,
        target_active=target_active,
        return_debug=return_debug,
    )
