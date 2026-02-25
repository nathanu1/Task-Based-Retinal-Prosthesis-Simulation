"""Differentiable phosphene simulator (PyTorch).

This is a pragmatic, real-time differentiable renderer inspired by the dynaphos/pulse2percept
style papers: electrode stimulation grid -> perceptual phosphene image.

Key design goals:
- fully differentiable (no Python loops over electrodes)
- stable gradients for end-to-end training
- produces dot-like phosphenes on black background

We approximate phosphenes as a Gaussian blur of an electrode grid, plus a mild nonlinearity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel2d(sigma: float, kernel_size: int) -> torch.Tensor:
    """Create a (1,1,K,K) Gaussian kernel."""
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma + 1e-8))
    ker = ker / (ker.sum() + 1e-8)
    return ker.view(1, 1, kernel_size, kernel_size)


@dataclass(frozen=True)
class SimParams:
    output_hw: Tuple[int, int] = (128, 128)
    blur_sigma: float = 1.5
    blur_kernel: int = 9
    brightness_gamma: float = 0.8
    pow_eps: float = 1e-4  # prevents infinite gradients at 0 when gamma<1
    noise_std: float = 0.0
    contrast_eps: float = 1e-6


class DifferentiablePhospheneSimulator(nn.Module):
    """Differentiable phosphene renderer: stim grid -> percept image in [0,1]."""

    def __init__(self, params: Optional[SimParams] = None):
        super().__init__()
        self.params = params or SimParams()

        k = int(self.params.blur_kernel)
        if k % 2 == 0:
            k += 1
        ker = gaussian_kernel2d(float(self.params.blur_sigma), k)
        self.register_buffer("_gauss", ker)

    def forward(self, stim_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stim_grid: (B, 1, Hg, Wg) in [0,1]
        Returns:
            percept: (B, 1, H, W) in [0,1]
        """
        if stim_grid.ndim != 4:
            raise ValueError("stim_grid must be (B,1,Hg,Wg)")
        stim = stim_grid.clamp(0, 1)

        H, W = self.params.output_hw
        stim_up = F.interpolate(stim, size=(H, W), mode="bilinear", align_corners=False)

        pad = self._gauss.shape[-1] // 2
        blurred = F.conv2d(stim_up, self._gauss, padding=pad)

        # Nonlinearity (brightness)
        base = blurred.clamp(0, 1)
        g = float(self.params.brightness_gamma)
        eps = float(max(0.0, self.params.pow_eps))
        if eps > 0.0:
            # Keep percept(0)=0 while making gradients finite near 0.
            percept = torch.pow(base + eps, g) - (eps ** g)
        else:
            percept = torch.pow(base, g)
        percept = percept.clamp(0, 1)

        if self.training and self.params.noise_std and self.params.noise_std > 0:
            percept = (percept + torch.randn_like(percept) * float(self.params.noise_std)).clamp(0, 1)

        # Contrast stretch per batch item (safe)
        b = percept.shape[0]
        p_flat = percept.view(b, -1)
        pmin = p_flat.min(dim=1).values.view(b, 1, 1, 1)
        pmax = p_flat.max(dim=1).values.view(b, 1, 1, 1)
        percept = (percept - pmin) / (pmax - pmin + float(self.params.contrast_eps))
        return percept.clamp(0, 1)

