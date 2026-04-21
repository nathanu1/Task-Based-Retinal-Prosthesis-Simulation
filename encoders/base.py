"""
Base classes and shared data structures for all phosphene encoder backends.

PerceptionContext
    Pre-computed perception results (segmentation, saliency, edges, motion,
    near-field, YOLO) shared across backends for a single frame.

BackendResult
    Standardised output from any backend: stimulation_grid, phosphene_image,
    intermediate maps, timing, and optional error.

PhospheneBackend
    Abstract base class every backend must implement.

TaskPreset
    Named weight config for multi-cue fusion (General Scene / Navigation /
    Object Emphasis).  Backends receive the active preset name via ``params``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Task presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskPreset:
    name: str
    description: str
    fusion_weights: Dict[str, float]
    near_weight: float
    subject_weight: float
    fill_weight: float
    bg_suppress: float


TASK_PRESETS: Dict[str, TaskPreset] = {
    "General Scene": TaskPreset(
        name="General Scene",
        description="Balanced across all visual cues. Good default for natural images.",
        fusion_weights={"saliency": 2.4, "segmentation": 2.4, "edges": 0.25, "motion": 1.0, "yolo": 3.2},
        near_weight=0.22,
        subject_weight=0.50,
        fill_weight=0.18,
        bg_suppress=0.75,
    ),
    "Navigation": TaskPreset(
        name="Navigation",
        description="Emphasises near-field obstacles and motion for safe movement.",
        fusion_weights={"saliency": 1.0, "segmentation": 1.5, "edges": 0.5, "motion": 2.0, "yolo": 2.0},
        near_weight=0.42,
        subject_weight=0.35,
        fill_weight=0.10,
        bg_suppress=0.60,
    ),
    "Object Emphasis": TaskPreset(
        name="Object Emphasis",
        description="Maximises foreground object boundaries, suppresses background clutter.",
        fusion_weights={"saliency": 2.0, "segmentation": 3.5, "edges": 0.15, "motion": 0.5, "yolo": 4.0},
        near_weight=0.15,
        subject_weight=0.65,
        fill_weight=0.25,
        bg_suppress=0.85,
    ),
}


# ---------------------------------------------------------------------------
# Perception context
# ---------------------------------------------------------------------------

@dataclass
class PerceptionContext:
    """Pre-computed perception results shared across all backends for one frame.

    All spatial arrays are in the same resolution as ``image`` unless
    otherwise noted (e.g. segmentation downsampled versions).
    """

    # Raw frame
    image: np.ndarray                           # H×W×3 BGR uint8
    gray: np.ndarray                            # H×W uint8 (raw)
    gray_clahe: np.ndarray                      # H×W uint8 CLAHE-enhanced (basis for luminance)
    luminance: np.ndarray                       # H×W float32 [0,1] CLAHE+detail-enhanced

    # AI perception outputs
    saliency_map: Optional[np.ndarray]          # H×W float32 [0,1]
    segmentation_map: Optional[np.ndarray]      # H×W float32 class scores
    segmentation_fg: Optional[np.ndarray]       # H×W float32 foreground probability
    pred_full: Optional[np.ndarray]             # H×W uint8 class indices
    object_mask_u8: Optional[np.ndarray]        # H×W uint8 {0,255} cleaned object mask

    # YOLO
    detections: List[Dict[str, Any]]            # list of {class, conf, bbox}
    annotated_yolo: Optional[np.ndarray]        # H×W×3 BGR with boxes drawn
    yolo_heatmap: Optional[np.ndarray]          # H×W float32 [0,1]
    primary_det: Optional[Dict[str, Any]]
    yolo_roi: Optional[Tuple[float, float, float, float]]

    # Edges + motion
    edges_map: np.ndarray                       # H×W float32 [0,1]
    motion_magnitude: Optional[np.ndarray]      # H×W float32
    motion_result: Optional[Dict[str, Any]]

    # Fused priority maps
    gate: np.ndarray                            # H×W float32 [0,1] foreground gate
    near: np.ndarray                            # H×W float32 [0,1] near-field weight

    # Task info
    task_mode: str = "General Scene"
    task_preset: Optional[TaskPreset] = None


# ---------------------------------------------------------------------------
# Backend result
# ---------------------------------------------------------------------------

@dataclass
class BackendResult:
    """Standardised output from any phosphene encoder backend.

    Fields
    ------
    backend_name
        Human-readable name of the backend that produced this result.
    input_image
        Original BGR frame.
    task_mode
        Active task preset name.
    stimulation_grid
        2-D float32 array in [0,1] representing electrode activations.
        Shape is backend-dependent (e.g. 60×60 for Dots, 6×10 for ArgusII).
        May be None if the backend does not expose an intermediate grid.
    phosphene_image
        Rendered percept as uint8 grayscale H×W or H×W×1.
    intermediate_maps
        Dict of named intermediate arrays (edges, saliency, attention, …)
        used by the Diagnostics tab.  Values are float32 or uint8 arrays.
    timing_info
        Dict of stage name → milliseconds.
    metadata
        Backend-specific key/value pairs (params used, model names, …).
    error
        Non-None means the backend failed; contains the error message.
        ``phosphene_image`` will be a black placeholder in that case.
    """

    backend_name: str
    input_image: np.ndarray
    task_mode: str
    stimulation_grid: Optional[np.ndarray]
    phosphene_image: np.ndarray                 # uint8 grayscale
    intermediate_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    timing_info: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class PhospheneBackend(ABC):
    """Abstract base class for all phosphene encoder backends.

    Subclasses must set class-level attributes ``name``, ``available``, and
    ``unavailable_reason``, then implement ``run()``.
    """

    name: str = "Unnamed"
    available: bool = False
    unavailable_reason: str = ""

    @abstractmethod
    def run(
        self,
        context: PerceptionContext,
        params: Dict[str, Any],
        prev_stim_grid: Optional[np.ndarray] = None,
    ) -> BackendResult:
        """Run the backend on *context* and return a BackendResult.

        Parameters
        ----------
        context
            Pre-computed perception results.
        params
            Backend-specific hyper-parameters (from sidebar sliders/selects).
        prev_stim_grid
            Previous frame's stimulation grid for temporal smoothing (video).
        """
        ...

    def _black_percept(self, context: PerceptionContext, error: str) -> BackendResult:
        """Helper: return a placeholder result when the backend fails."""
        h, w = context.image.shape[:2]
        return BackendResult(
            backend_name=self.name,
            input_image=context.image,
            task_mode=context.task_mode,
            stimulation_grid=None,
            phosphene_image=np.zeros((h, w), dtype=np.uint8),
            error=error,
        )
