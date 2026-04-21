"""
pulse2percept Retinal backend — simulates an Argus II retinal implant using
pulse2percept's AxonMapModel (default) or ScoreboardModel.

The stimulation grid from the DotsBackend (60×60) is downsampled to the
Argus II layout (6×10 electrodes), converted to micro-amp currents, and
passed through pulse2percept's perceptual model, which produces elongated
phosphene streaks along retinal axon fibres.

Falls back gracefully if pulse2percept is not installed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .base import BackendResult, PerceptionContext, PhospheneBackend

try:
    from utils.phosphene import (
        PhospheneSimulator,
        image_to_edge_stim_map,
        PULSE2PERCEPT_AVAILABLE,
        ARGUS_II_ROWS,
        ARGUS_II_COLS,
        MAX_CURRENT_MICROAMPS,
    )
except ImportError:
    PULSE2PERCEPT_AVAILABLE = False
    PhospheneSimulator = None  # type: ignore
    image_to_edge_stim_map = None  # type: ignore
    ARGUS_II_ROWS, ARGUS_II_COLS = 6, 10
    MAX_CURRENT_MICROAMPS = 50.0


def _stim_grid_to_argus(stim_grid: np.ndarray, implant: Any) -> Dict[str, float]:
    """Downsample any 2-D float stim_grid → Argus II 6×10 current dict."""
    from PIL import Image as PILImage
    s = np.clip(np.asarray(stim_grid, dtype=np.float32), 0, 1)
    img = PILImage.fromarray((s * 255).astype(np.uint8))
    resized = img.resize((ARGUS_II_COLS, ARGUS_II_ROWS), PILImage.BILINEAR)
    flat = np.array(resized, dtype=np.float32).flatten() / 255.0
    flat = np.clip(flat, 0, 1) * float(MAX_CURRENT_MICROAMPS)
    electrode_names = list(implant.electrodes.keys())
    n = min(len(electrode_names), len(flat))
    return {name: float(flat[i]) if i < n else 0.0 for i, name in enumerate(electrode_names)}


class P2PRetinalBackend(PhospheneBackend):
    """pulse2percept Argus II retinal implant simulation.

    Uses AxonMapModel (biologically plausible axon-streak phosphenes) or
    ScoreboardModel (simpler blob phosphenes without axon distortion).

    The 60×60 stimulation grid from the perception pipeline is mapped to the
    6×10 Argus II electrode array and then passed through the chosen model.
    The output is resized to match the input image dimensions.

    model_key: "axon_map" | "scoreboard"
    """

    name = "p2p Retinal (Argus II)"
    available = PULSE2PERCEPT_AVAILABLE
    unavailable_reason = "" if PULSE2PERCEPT_AVAILABLE else "pulse2percept not installed (pip install pulse2percept)"

    # Cached simulators keyed by model name — set externally or lazily built
    _simulators: Dict[str, Any] = {}

    @classmethod
    def get_simulator(cls, model_key: str = "axon_map") -> Optional[Any]:
        """Return or lazily create a PhospheneSimulator for the given model."""
        if not PULSE2PERCEPT_AVAILABLE or PhospheneSimulator is None:
            return None
        if model_key not in cls._simulators:
            try:
                sim = PhospheneSimulator(perceptual_model=model_key)
                cls._simulators[model_key] = sim
            except Exception:
                return None
        return cls._simulators.get(model_key)

    def run(
        self,
        context: PerceptionContext,
        params: Dict[str, Any],
        prev_stim_grid: Optional[np.ndarray] = None,
    ) -> BackendResult:
        if not self.available:
            return self._black_percept(context, self.unavailable_reason)

        model_key = str(params.get("p2p_model", "axon_map"))
        t_total = time.perf_counter()

        sim = self.get_simulator(model_key)
        if sim is None:
            return self._black_percept(context, f"Could not load pulse2percept simulator ({model_key}).")

        # Use the stimulation grid from params if provided (from DotsBackend run),
        # otherwise derive a simple edge-based stim map from the raw image.
        stim_grid = params.get("stim_grid_override")
        if stim_grid is None:
            # Build a basic 6×10 stim map from image edges
            gray = context.gray
            if image_to_edge_stim_map is not None:
                stim_grid = image_to_edge_stim_map(gray, grid_size=32)
            else:
                stim_grid = cv2.resize(gray.astype(np.float32) / 255.0, (10, 6))
        stim_grid = np.asarray(stim_grid, dtype=np.float32)

        h, w = context.image.shape[:2]
        output_size = (h, w)

        t0 = time.perf_counter()
        try:
            percept_data = sim.simulate_from_grid(stim_grid, output_size=output_size, as_uint8=True)
        except Exception as exc:
            return self._black_percept(context, f"pulse2percept simulation failed: {exc}")
        sim_ms = (time.perf_counter() - t0) * 1000.0

        # Normalise and convert to uint8
        if percept_data.dtype != np.uint8:
            p = np.asarray(percept_data, dtype=np.float64)
            p_min, p_max = p.min(), p.max()
            if p_max - p_min > 1e-8:
                p = (p - p_min) / (p_max - p_min)
            percept_data = (np.clip(p, 0, 1) * 255).astype(np.uint8)
        if percept_data.ndim == 3:
            percept_data = percept_data[:, :, 0]

        if percept_data.shape[:2] != (h, w):
            percept_data = cv2.resize(percept_data, (w, h), interpolation=cv2.INTER_LINEAR)

        total_ms = (time.perf_counter() - t_total) * 1000.0
        timings = {"simulation_ms": sim_ms, "total_ms": total_ms}

        # Expose a normalised stim grid for metric computation
        stim_out = np.clip(stim_grid.astype(np.float32), 0, 1)
        if stim_out.max() > 1.0:
            stim_out = stim_out / (stim_out.max() + 1e-8)

        intermediate_maps: Dict[str, np.ndarray] = {}
        if context.saliency_map is not None:
            intermediate_maps["saliency"] = context.saliency_map
        if context.edges_map is not None:
            intermediate_maps["edges_map"] = context.edges_map

        metadata = {
            "p2p_model": model_key,
            "argus_grid": f"{ARGUS_II_ROWS}×{ARGUS_II_COLS}",
        }

        return BackendResult(
            backend_name=self.name,
            input_image=context.image,
            task_mode=context.task_mode,
            stimulation_grid=stim_out,
            phosphene_image=percept_data,
            intermediate_maps=intermediate_maps,
            timing_info=timings,
            metadata=metadata,
        )
