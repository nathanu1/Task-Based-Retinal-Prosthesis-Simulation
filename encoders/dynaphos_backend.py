"""
Dynaphos Cortical backend — uses DynaphosModel (eLife 2024) + Orion cortical
implant via pulse2percept to produce cortical phosphene percepts.

The DynaphosPipeline from utils/dynaphos_pipeline.py handles image → stim
grid → cortical phosphene.  This backend wraps that pipeline in the common
PhospheneBackend interface.

Falls back gracefully if pulse2percept >= 0.9 is not installed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .base import BackendResult, PerceptionContext, PhospheneBackend

try:
    from utils.dynaphos_pipeline import DynaphosPipeline, DYNAPHOS_AVAILABLE
except ImportError:
    DYNAPHOS_AVAILABLE = False
    DynaphosPipeline = None  # type: ignore

# Preprocessing modes exposed to the user
PREPROCESS_MODES = ["edges", "clahe", "gamma", "retinex", "grayscale", "invert"]


class DynaphosBackend(PhospheneBackend):
    """Cortical Dynaphos phosphene simulation (DynaphosModel + Orion implant).

    Parameters accepted via ``params``
    -----------------------------------
    preprocess : str
        Image-to-stimulus preprocessing mode passed to DynaphosPipeline.
        One of: "edges" | "clahe" | "gamma" | "retinex" | "grayscale" | "invert".
    """

    name = "Dynaphos Cortical (p2p)"
    available = DYNAPHOS_AVAILABLE
    unavailable_reason = "" if DYNAPHOS_AVAILABLE else "pulse2percept>=0.9 not installed (pip install pulse2percept)"

    # Lazily created per preprocessing mode
    _pipelines: Dict[str, Any] = {}

    @classmethod
    def get_pipeline(cls, preprocess: str = "edges") -> Optional[Any]:
        if not DYNAPHOS_AVAILABLE or DynaphosPipeline is None:
            return None
        if preprocess not in cls._pipelines:
            try:
                cls._pipelines[preprocess] = DynaphosPipeline(preprocess=preprocess)
            except Exception:
                return None
        return cls._pipelines.get(preprocess)

    def run(
        self,
        context: PerceptionContext,
        params: Dict[str, Any],
        prev_stim_grid: Optional[np.ndarray] = None,
    ) -> BackendResult:
        if not self.available:
            return self._black_percept(context, self.unavailable_reason)

        preprocess = str(params.get("dynaphos_preprocess", "edges"))
        t_total = time.perf_counter()

        pipeline = self.get_pipeline(preprocess)
        if pipeline is None:
            return self._black_percept(context, "Could not build DynaphosPipeline.")

        h, w = context.image.shape[:2]
        output_size = (h, w)

        # Pass foreground/boundary hints derived from perception context
        fg_mask_g = None
        boundary_mask_g = None
        if context.object_mask_u8 is not None:
            from utils.adaptive_encoding import grid_foreground_mask, grid_boundary_mask
            fg_mask_g = grid_foreground_mask(
                gate=context.gate,
                object_mask_u8=context.object_mask_u8,
                grid_n=60,
            )
            boundary_mask_g = grid_boundary_mask(object_mask_u8=context.object_mask_u8, grid_n=60)

        t0 = time.perf_counter()
        try:
            percept_arr = pipeline.image_to_phosphene(
                context.image,
                output_size=output_size,
                fg_mask_g=fg_mask_g,
                boundary_mask_g=boundary_mask_g,
                return_debug=True,
            )
            # Debug info is attached as pipeline.last_debug by the pipeline implementation
            debug_info = getattr(pipeline, "last_debug", {}) or {}
        except Exception as exc:
            return self._black_percept(context, f"DynaphosPipeline failed: {exc}")

        sim_ms = (time.perf_counter() - t0) * 1000.0

        if percept_arr is None or not isinstance(percept_arr, np.ndarray):
            return self._black_percept(context, "DynaphosPipeline returned None.")

        # Normalise → uint8
        if percept_arr.dtype != np.uint8:
            p = np.asarray(percept_arr, dtype=np.float64)
            if p.max() - p.min() > 1e-8:
                p = (p - p.min()) / (p.max() - p.min())
            percept_arr = (np.clip(p, 0, 1) * 255).astype(np.uint8)
        if percept_arr.ndim == 3:
            percept_arr = percept_arr[:, :, 0]

        if percept_arr.shape[:2] != (h, w):
            percept_arr = cv2.resize(percept_arr, (w, h), interpolation=cv2.INTER_LINEAR)

        total_ms = (time.perf_counter() - t_total) * 1000.0

        stim_g = debug_info.get("stim_g") if isinstance(debug_info, dict) else None
        if stim_g is not None:
            stim_g = np.clip(np.asarray(stim_g, dtype=np.float32), 0, 1)

        intermediate_maps: Dict[str, np.ndarray] = {}
        if context.saliency_map is not None:
            intermediate_maps["saliency"] = context.saliency_map
        if context.edges_map is not None:
            intermediate_maps["edges_map"] = context.edges_map
        if stim_g is not None:
            intermediate_maps["stim_g"] = stim_g

        return BackendResult(
            backend_name=self.name,
            input_image=context.image,
            task_mode=context.task_mode,
            stimulation_grid=stim_g,
            phosphene_image=percept_arr,
            intermediate_maps=intermediate_maps,
            timing_info={"simulation_ms": sim_ms, "total_ms": total_ms},
            metadata={"preprocess": preprocess, "dynaphos_debug": {k: v for k, v in debug_info.items() if not isinstance(v, np.ndarray)}},
        )
