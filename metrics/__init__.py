"""
Phosphene benchmarking metrics.

Modules
-------
image_metrics
    SSIM and LPIPS between phosphene percepts (or percept vs. input).
electrode_metrics
    Active electrode count, ratio, efficiency proxy, flicker.
coverage_metrics
    Global and radial/eccentricity-binned phosphene coverage.
"""

from .image_metrics import compute_ssim, compute_lpips, LPIPS_AVAILABLE
from .electrode_metrics import compute_electrode_metrics
from .coverage_metrics import compute_coverage_metrics, radial_coverage_bins

__all__ = [
    "compute_ssim",
    "compute_lpips",
    "LPIPS_AVAILABLE",
    "compute_electrode_metrics",
    "compute_coverage_metrics",
    "radial_coverage_bins",
]
