"""
Phosphene encoder backends for the retinal prosthesis benchmarking workbench.

Architecture
------------
Each backend implements PhospheneBackend and returns a BackendResult.
Backends share a PerceptionContext computed once per frame (segmentation,
saliency, YOLO, edges, motion, gate, near-field).

Extending with new backends
---------------------------
1. Subclass PhospheneBackend in a new file under encoders/
2. Set ``name``, ``available``, and ``unavailable_reason``
3. Implement ``run(context, params) -> BackendResult``
4. Register the class in registry.py
"""

from .base import BackendResult, PerceptionContext, PhospheneBackend, TaskPreset, TASK_PRESETS

__all__ = [
    "BackendResult",
    "PerceptionContext",
    "PhospheneBackend",
    "TaskPreset",
    "TASK_PRESETS",
]
