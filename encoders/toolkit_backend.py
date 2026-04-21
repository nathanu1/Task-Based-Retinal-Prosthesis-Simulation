"""
Toolkit Full Pipeline backend — uses PhospheneEncoderTool.process_frame()
to produce a biologically-styled percept with task-conditioned fusion,
bandwidth allocation, and perceptual observer model.

Falls back gracefully if phosphene_toolkit is unavailable or process_frame
raises an error.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .base import BackendResult, PerceptionContext, PhospheneBackend

try:
    from phosphene_toolkit import PhospheneEncoderTool as _PET
    _TOOLKIT_AVAILABLE = True
except ImportError:
    _PET = None  # type: ignore
    _TOOLKIT_AVAILABLE = False

# Map app task modes to task strings the toolkit understands
_TASK_MAP = {
    "General Scene": "general",
    "Navigation": "navigation: avoid obstacles",
    "Object Emphasis": "identify and locate object",
}


class ToolkitBackend(PhospheneBackend):
    """Full PhospheneEncoderTool pipeline backend.

    Uses ``process_frame(frame, task, return_intermediates=True)`` which runs:
      perception → fusion → temporal stabilisation → bandwidth allocation →
      device constraints → observer perceptual model.

    The toolkit encoder is cached at startup to avoid repeated model loading.
    """

    name = "Toolkit Pipeline"
    available = _TOOLKIT_AVAILABLE
    unavailable_reason = "" if _TOOLKIT_AVAILABLE else "phosphene_toolkit not importable"

    # Shared encoder instance (set by the Streamlit @st.cache_resource loader)
    _encoder_instance: Optional[Any] = None

    @classmethod
    def set_encoder(cls, enc: Any) -> None:
        cls._encoder_instance = enc

    def run(
        self,
        context: PerceptionContext,
        params: Dict[str, Any],
        prev_stim_grid: Optional[np.ndarray] = None,
    ) -> BackendResult:
        if not self.available or _PET is None:
            return self._black_percept(context, self.unavailable_reason)

        enc = self._encoder_instance
        if enc is None:
            return self._black_percept(context, "Toolkit encoder not loaded — call ToolkitBackend.set_encoder() first.")

        task_mode = context.task_mode
        task_str = _TASK_MAP.get(task_mode, "general")

        t0 = time.perf_counter()
        try:
            result = enc.process_frame(
                context.image,
                task=task_str,
                return_intermediates=True,
            )
        except Exception as exc:
            return self._black_percept(context, f"process_frame failed: {exc}")

        total_ms = (time.perf_counter() - t0) * 1000.0

        # process_frame returns {'stimulation_plan', 'percept', 'timings', ...}
        percept_arr = result.get("percept")
        if percept_arr is None:
            return self._black_percept(context, "Toolkit process_frame returned no percept.")

        # Ensure uint8 grayscale
        if percept_arr.dtype != np.uint8:
            p = np.asarray(percept_arr, dtype=np.float32)
            p_min, p_max = p.min(), p.max()
            if p_max - p_min > 1e-8:
                p = (p - p_min) / (p_max - p_min)
            percept_arr = (np.clip(p, 0, 1) * 255).astype(np.uint8)
        if percept_arr.ndim == 3 and percept_arr.shape[2] == 1:
            percept_arr = percept_arr[:, :, 0]

        # Resize to match input image
        h, w = context.image.shape[:2]
        if percept_arr.shape[:2] != (h, w):
            import cv2
            percept_arr = cv2.resize(percept_arr, (w, h), interpolation=cv2.INTER_LINEAR)

        stim_plan = result.get("stimulation_plan")
        if stim_plan is not None:
            stim_plan = np.asarray(stim_plan, dtype=np.float32)
            s_min, s_max = stim_plan.min(), stim_plan.max()
            if s_max - s_min > 1e-8:
                stim_plan = (stim_plan - s_min) / (s_max - s_min)

        timings = dict(result.get("timings", {}))
        timings["total_ms"] = total_ms

        intermediates = result.get("intermediates", {})
        intermediate_maps: Dict[str, np.ndarray] = {}
        for k, v in intermediates.items():
            if isinstance(v, np.ndarray):
                intermediate_maps[k] = v

        # Always expose shared perception maps
        if context.saliency_map is not None:
            intermediate_maps["saliency"] = context.saliency_map
        if context.segmentation_fg is not None:
            intermediate_maps["segmentation_fg"] = context.segmentation_fg
        if context.edges_map is not None:
            intermediate_maps["edges_map"] = context.edges_map

        metadata = {
            "task_string": task_str,
            "active_phosphenes": result.get("percept_info", {}).get("active_phosphenes", 0),
            "device_diagnostics": result.get("device_diagnostics", {}),
        }

        return BackendResult(
            backend_name=self.name,
            input_image=context.image,
            task_mode=context.task_mode,
            stimulation_grid=stim_plan,
            phosphene_image=percept_arr,
            intermediate_maps=intermediate_maps,
            timing_info=timings,
            metadata=metadata,
        )
