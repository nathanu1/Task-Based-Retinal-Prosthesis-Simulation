"""Utilities for end-to-end training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def save_grid_triplet(
    out_path: str | Path,
    inputs: torch.Tensor,
    phos: torch.Tensor,
    recons: torch.Tensor,
    max_rows: int = 4,
):
    """Save a grid: rows are sequences; columns are frames. Stacks input/phos/recon vertically."""
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # inputs/phos/recons: (B,T,1,H,W) or (T,1,H,W) for B=1
    if inputs.ndim == 4:
        inputs = inputs.unsqueeze(0)
        phos = phos.unsqueeze(0)
        recons = recons.unsqueeze(0)

    B, T, _, H, W = inputs.shape
    B = min(B, max_rows)

    def to_u8(x: torch.Tensor) -> np.ndarray:
        x = x.detach().cpu().clamp(0, 1).numpy()
        return (x * 255).astype(np.uint8)

    inp_u = to_u8(inputs[:B])
    pho_u = to_u8(phos[:B])
    rec_u = to_u8(recons[:B])

    # build canvas: height = B * (3*H), width = T*W
    canvas = np.zeros((B * 3 * H, T * W), dtype=np.uint8)
    for b in range(B):
        for t in range(T):
            x0 = t * W
            y0 = b * 3 * H
            canvas[y0 : y0 + H, x0 : x0 + W] = inp_u[b, t, 0]
            canvas[y0 + H : y0 + 2 * H, x0 : x0 + W] = pho_u[b, t, 0]
            canvas[y0 + 2 * H : y0 + 3 * H, x0 : x0 + W] = rec_u[b, t, 0]

    Image.fromarray(canvas).save(out_path.as_posix())

