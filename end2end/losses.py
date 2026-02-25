"""Loss functions for end-to-end prosthetic vision training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Sobel edge magnitude. x: (B,1,H,W) -> (B,1,H,W) in [0,~]."""
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("x must be (B,1,H,W)")
    device = x.device
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return mag


def normalize_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each sample to [0,1]."""
    b = x.shape[0]
    flat = x.view(b, -1)
    mn = flat.min(dim=1).values.view(b, 1, 1, 1)
    mx = flat.max(dim=1).values.view(b, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute mean weighted MSE."""
    return torch.mean(weight * (pred - target) ** 2)

