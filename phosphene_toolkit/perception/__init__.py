from .segmentation import DeepLabV3Segmentation
from .saliency import SpectralResidualSaliency
from .motion import FarnebackMotionDetector, compute_motion_between_frames
from .edges import compute_edges
try:
    from .depth import DepthEstimator, DepthResult
except Exception:  # optional dependency/weights
    DepthEstimator = None  # type: ignore
    DepthResult = None  # type: ignore

__all__ = [
    'DeepLabV3Segmentation',
    'SpectralResidualSaliency',
    'FarnebackMotionDetector',
    'compute_motion_between_frames',
    'compute_edges',
    'DepthEstimator',
    'DepthResult',
]
