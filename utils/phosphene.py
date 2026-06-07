"""Phosphene simulation using pulse2percept."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from PIL import Image

try:
    from pulse2percept.models import AxonMapModel
    from pulse2percept.implants import ArgusII
    PULSE2PERCEPT_AVAILABLE = True
    try:
        from pulse2percept.models import ScoreboardModel
        SCOREBOARD_AVAILABLE = True
    except ImportError:
        ScoreboardModel = None  # type: ignore
        SCOREBOARD_AVAILABLE = False
except ImportError:
    PULSE2PERCEPT_AVAILABLE = False
    AxonMapModel = None  # type: ignore
    ScoreboardModel = None  # type: ignore
    SCOREBOARD_AVAILABLE = False
    print("Warning: pulse2percept not available. Install with: pip install pulse2percept")

ARGUS_II_ROWS, ARGUS_II_COLS = 6, 10
MAX_CURRENT_MICROAMPS = 50.0
DEFAULT_GRID_SIZE = 32


def image_to_edge_stim_map(
    image: Union[np.ndarray, "Image.Image"],
    grid_size: int = DEFAULT_GRID_SIZE,
    low: int = 50,
    high: int = 150,
) -> np.ndarray:
    """Build stimulation map from edge detection."""
    try:
        from PIL import Image as PILImage
        import cv2
    except ImportError:
        raise ImportError("PIL and opencv-python required for edge stim map")
    if hasattr(image, "convert"):
        image = np.array(image.convert("L"))
    elif image.ndim == 3:
        image = np.array(PILImage.fromarray(image.astype(np.uint8)).convert("L"))
    image = np.asarray(image, dtype=np.uint8)
    edges = cv2.Canny(image, low, high)
    edges_f = edges.astype(np.float32) / 255.0
    out = cv2.resize(edges_f, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    return np.clip(out.astype(np.float64), 0.0, 1.0)


def _stim_grid_to_electrode_currents(stim_grid: np.ndarray, implant: object) -> dict:
    """Map 2D stim grid to Argus II 6×10 electrode currents."""
    stim = np.asarray(stim_grid, dtype=np.float64)
    if stim.ndim != 2:
        raise ValueError("stim_grid must be 2D (H, W)")
    from PIL import Image
    img = Image.fromarray((np.clip(stim, 0, 1) * 255).astype(np.uint8))
    resized = img.resize((ARGUS_II_COLS, ARGUS_II_ROWS), Image.BILINEAR)
    grid_6x10 = np.array(resized, dtype=np.float64) / 255.0
    currents_flat = np.clip(grid_6x10.flatten(), 0.0, 1.0) * MAX_CURRENT_MICROAMPS
    electrode_names = list(implant.electrodes.keys())
    n = min(len(electrode_names), len(currents_flat))
    return {name: float(currents_flat[i]) if i < n else 0.0 for i, name in enumerate(electrode_names)}


def generate_phosphene_image(stim_20x20: np.ndarray, implant=None, model=None, save_path=None) -> np.ndarray:
    """Generate phosphene image from stimulation map."""
    if not PULSE2PERCEPT_AVAILABLE:
        raise ImportError("pulse2percept is required. Install with: pip install pulse2percept")
    if implant is None:
        implant = ArgusII()
    if model is None:
        model = AxonMapModel()
        model.build()
    stim_dict = _stim_grid_to_electrode_currents(stim_20x20, implant)
    # pulse2percept's predict_percept reads the stimulus from implant.stim;
    # its second positional arg is t_percept, NOT the stimulus.
    implant.stim = stim_dict
    percept = model.predict_percept(implant)
    percept_data = percept.data
    if save_path:
        from PIL import Image
        pn = (percept_data - percept_data.min()) / (percept_data.max() - percept_data.min() + 1e-8)
        Image.fromarray((pn * 255).astype(np.uint8)).save(save_path)
    return percept_data


class PhospheneSimulator:
    """Wrapper for pulse2percept with AxonMapModel or ScoreboardModel."""

    def __init__(self, perceptual_model: str = "axon_map"):
        if not PULSE2PERCEPT_AVAILABLE:
            raise ImportError("pulse2percept is required. Install with: pip install pulse2percept")
        self.implant = ArgusII()
        perceptual_model = (perceptual_model or "axon_map").strip().lower()
        if perceptual_model == "scoreboard" and SCOREBOARD_AVAILABLE and ScoreboardModel is not None:
            self.model = ScoreboardModel(xrange=(-20, 20), yrange=(-20, 20), xystep=0.25)
        else:
            self.model = AxonMapModel()
        self.model.build()

    def simulate(self, stim_20x20: np.ndarray, save_path=None) -> np.ndarray:
        return generate_phosphene_image(stim_20x20, self.implant, self.model, save_path)

    def simulate_from_grid(self, stim_grid: np.ndarray, output_size=(256, 256), as_uint8=True) -> np.ndarray:
        if not PULSE2PERCEPT_AVAILABLE:
            raise ImportError("pulse2percept is required")
        stim_dict = _stim_grid_to_electrode_currents(stim_grid, self.implant)
        # predict_percept reads the stimulus from implant.stim; the second
        # positional argument is t_percept, not the stimulus dict.
        self.implant.stim = stim_dict
        percept = self.model.predict_percept(self.implant)
        if percept is None or not hasattr(percept, 'data'):
            h, w = (output_size if output_size else (256, 256))
            blank = np.zeros((h, w), dtype=np.float64)
            return (blank * 255).astype(np.uint8) if as_uint8 else blank
        out = np.asarray(percept.data, dtype=np.float64)
        # pulse2percept returns (Y, X, T); collapse the trailing time axis so
        # downstream PIL/cv2 ops receive a plain 2-D image.
        if out.ndim == 3:
            out = out[..., 0]
        pmin, pmax = out.min(), out.max()
        out = (out - pmin) / (pmax - pmin + 1e-8) if pmax - pmin > 1e-8 else np.clip(out, 0, 1)
        if output_size:
            from PIL import Image
            h, w = output_size
            img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
            img = img.resize((w, h), Image.BILINEAR)
            out = np.array(img, dtype=np.float64) / 255.0
        return (np.clip(out, 0, 1) * 255).astype(np.uint8) if as_uint8 else out

    def simulate_edges_from_image(self, image, grid_size=32, output_size=(256, 256), as_uint8=True, canny_low=50, canny_high=150):
        stim_grid = image_to_edge_stim_map(image, grid_size=grid_size, low=canny_low, high=canny_high)
        return self.simulate_from_grid(stim_grid, output_size=output_size, as_uint8=as_uint8)
