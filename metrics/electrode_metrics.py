"""
Electrode-level metrics for a phosphene stimulation grid.

Active electrode count
    Number of grid cells with amplitude > threshold.

Active electrode ratio
    active_count / total_electrodes.

Electrode efficiency proxy
    Measures how well the active electrodes are "used" — combining two factors:

    1. **Sparsity bonus**: reward low activation ratios (less is more).
       sparsity = 1 − active_ratio

    2. **Local redundancy penalty**: penalise clusters of adjacent active
       electrodes that overlap in perceptual space without adding information.
       redundancy = mean fraction of 8-neighbours that are also active,
       averaged over all active cells.

    Formula:
        efficiency = sparsity × (1 − redundancy)

    Range: [0, 1].  High efficiency = sparse, well-separated active
    electrodes.  Low efficiency = dense or clumped activations.

Flicker
    Mean absolute change in the active mask vs. the previous frame's grid.
    Range [0, 1].  0 = identical; 1 = completely different.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_ACTIVE_THR = 1e-6


def _local_redundancy(stim_bin: np.ndarray) -> float:
    """Mean fraction of 8-connected neighbours that are also active."""
    if stim_bin.sum() == 0:
        return 0.0
    h, w = stim_bin.shape
    s = stim_bin.astype(np.float32)
    # Shift in 8 directions and sum
    neighbours = np.zeros_like(s)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            rolled = np.roll(np.roll(s, dy, axis=0), dx, axis=1)
            # Zero out wrapped edges
            if dy < 0:
                rolled[h + dy:, :] = 0
            elif dy > 0:
                rolled[:dy, :] = 0
            if dx < 0:
                rolled[:, w + dx:] = 0
            elif dx > 0:
                rolled[:, :dx] = 0
            neighbours += rolled
    # Max possible neighbours is 8; compute fraction per active cell
    frac = neighbours[s > 0] / 8.0
    return float(frac.mean())


def compute_electrode_metrics(
    stim_grid: np.ndarray,
    *,
    prev_stim_grid: Optional[np.ndarray] = None,
    active_thr: float = _ACTIVE_THR,
) -> Dict[str, float]:
    """Compute electrode-level metrics for a 2-D stimulation grid.

    Parameters
    ----------
    stim_grid : np.ndarray
        2-D float array in [0, 1] (any shape).
    prev_stim_grid : optional
        Previous frame grid for flicker computation.
    active_thr : float
        Amplitude threshold above which an electrode is considered active.

    Returns
    -------
    dict with keys:
        active_count, total_electrodes, active_ratio,
        efficiency_proxy, sparsity, redundancy, flicker, mean_amplitude
    """
    stim = np.clip(np.asarray(stim_grid, dtype=np.float32), 0, 1)
    total = int(stim.size)
    stim_bin = (stim > float(active_thr)).astype(np.float32)
    active = int(stim_bin.sum())
    active_ratio = float(active / max(total, 1))

    sparsity = 1.0 - active_ratio
    redundancy = _local_redundancy(stim_bin)
    # Efficiency: high when activations are sparse and non-clustered
    efficiency = float(sparsity * (1.0 - redundancy))

    flicker = 0.0
    if prev_stim_grid is not None and np.asarray(prev_stim_grid).shape == stim.shape:
        prev_bin = (np.asarray(prev_stim_grid, dtype=np.float32) > float(active_thr)).astype(np.float32)
        flicker = float(np.mean(np.abs(stim_bin - prev_bin)))

    return {
        "active_count": float(active),
        "total_electrodes": float(total),
        "active_ratio": float(active_ratio),
        "efficiency_proxy": float(efficiency),
        "sparsity": float(sparsity),
        "redundancy": float(redundancy),
        "flicker": float(flicker),
        "mean_amplitude": float(stim[stim_bin > 0].mean()) if active > 0 else 0.0,
    }
