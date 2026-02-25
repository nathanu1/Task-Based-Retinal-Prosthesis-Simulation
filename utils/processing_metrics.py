"""Minimal, interpretable processing metrics for phosphene pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np


def _safe_mean(x: np.ndarray) -> float:
    if x is None or x.size == 0:  # type: ignore
        return 0.0
    return float(np.mean(x))


def compute_metrics(
    *,
    edge_grid: np.ndarray,
    stim_grid: np.ndarray,
    priority_obj_grid: Optional[np.ndarray] = None,
    near_grid: Optional[np.ndarray] = None,
    prev_stim_grid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return a compact set of metrics to track quality + stability."""
    edge_grid = np.asarray(edge_grid, dtype=np.float32)
    stim_grid = np.asarray(stim_grid, dtype=np.float32)

    active = float(np.sum(stim_grid > 1e-6))
    total = float(stim_grid.size)
    edge_density = float(np.mean(edge_grid > 0.25))

    obj_cov = _safe_mean(priority_obj_grid) if priority_obj_grid is not None else 0.0
    near_cov = _safe_mean(near_grid) if near_grid is not None else 0.0

    flicker = 0.0
    if prev_stim_grid is not None and prev_stim_grid.size == stim_grid.size:
        prev = np.asarray(prev_stim_grid, dtype=np.float32).reshape(stim_grid.shape)
        flicker = float(np.mean(np.abs((stim_grid > 1e-6).astype(np.float32) - (prev > 1e-6).astype(np.float32))))

    return {
        "active_dots": active,
        "active_pct": 100.0 * active / max(total, 1.0),
        "edge_density": edge_density,
        "obj_priority_mean": obj_cov,
        "near_priority_mean": near_cov,
        "flicker": flicker,
    }

