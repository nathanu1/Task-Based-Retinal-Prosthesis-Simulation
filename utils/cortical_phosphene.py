"""Cortical (V1) phosphene simulation using pulse2percept DynaphosModel."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

DYNAPHOS_AVAILABLE = False
DynaphosModel = None  # type: ignore
Orion = None  # type: ignore
BiphasicPulseTrain = None  # type: ignore

try:
    from pulse2percept.models.cortex import DynaphosModel as _DynaphosModel
    from pulse2percept.implants.cortex import Orion as _Orion
    DynaphosModel = _DynaphosModel
    Orion = _Orion
    try:
        from pulse2percept.stimuli import BiphasicPulseTrain as _BPT
        BiphasicPulseTrain = _BPT
    except ImportError:
        pass
    DYNAPHOS_AVAILABLE = True
except ImportError:
    pass

if not DYNAPHOS_AVAILABLE:
    print("Warning: Cortical (dynaphos) simulator not available. Install pulse2percept>=0.9")

CORTICAL_GRID_ROWS, CORTICAL_GRID_COLS = 6, 10
MAX_AMPLITUDE_UA = 128.0


def _stim_grid_to_cortical_currents(stim_grid: np.ndarray, electrode_names: list) -> dict:
    from PIL import Image
    stim = np.asarray(stim_grid, dtype=np.float64)
    if stim.ndim != 2:
        raise ValueError("stim_grid must be 2D (H, W)")
    img = Image.fromarray((np.clip(stim, 0, 1) * 255).astype(np.uint8))
    resized = img.resize((CORTICAL_GRID_COLS, CORTICAL_GRID_ROWS), Image.BILINEAR)
    grid_flat = np.array(resized, dtype=np.float64).flatten() / 255.0
    grid_flat = np.clip(grid_flat, 0.0, 1.0) * MAX_AMPLITUDE_UA
    n = min(len(electrode_names), len(grid_flat))
    return {name: float(grid_flat[i]) if i < n else 0.0 for i, name in enumerate(electrode_names)}


class CorticalPhospheneSimulator:
    """Cortical (V1) phosphene simulator using DynaphosModel (eLife 2024)."""

    def __init__(self):
        if not DYNAPHOS_AVAILABLE or DynaphosModel is None or Orion is None:
            raise ImportError("Cortical simulator requires pulse2percept>=0.9")
        self.implant = Orion()
        self.model = DynaphosModel()
        self.model.build()

    def simulate_from_grid(self, stim_grid: np.ndarray, output_size=(256, 256), as_uint8=True) -> np.ndarray:
        electrode_names = [e.name for e in self.implant.electrodes]
        amp_dict = _stim_grid_to_cortical_currents(stim_grid, electrode_names)
        stim_dict = amp_dict
        if BiphasicPulseTrain is not None:
            try:
                stim_dict = {name: BiphasicPulseTrain(amp, pulse_dur=0.17, interphase_dur=0.17, freq=300)
                            for name, amp in amp_dict.items()}
            except Exception:
                pass
        try:
            percept = self.model.predict_percept(self.implant, stim_dict)
        except Exception:
            from PIL import Image
            out = (np.clip(stim_grid, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(out).resize((output_size[1] if output_size else 256, output_size[0] if output_size else 256), Image.BILINEAR)
            return np.array(img)
        out = np.asarray(percept.data, dtype=np.float64)
        pmin, pmax = out.min(), out.max()
        out = (out - pmin) / (pmax - pmin + 1e-8) if pmax - pmin > 1e-8 else np.clip(out, 0, 1)
        if output_size:
            from PIL import Image
            h, w = output_size
            img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
            img = img.resize((w, h), Image.BILINEAR)
            out = np.array(img, dtype=np.float64) / 255.0
        return (np.clip(out, 0, 1) * 255).astype(np.uint8) if as_uint8 else out
