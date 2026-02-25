"""MovingMNIST-style video generator (no external dataset required)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torchvision.datasets import MNIST
    from torchvision import transforms
    TORCHVISION_OK = True
except Exception:
    TORCHVISION_OK = False
    MNIST = None  # type: ignore
    transforms = None  # type: ignore


@dataclass(frozen=True)
class MovingMNISTParams:
    seq_len: int = 5
    canvas_hw: Tuple[int, int] = (128, 128)
    digit_hw: Tuple[int, int] = (28, 28)
    max_speed: int = 4
    digits_per_seq: int = 2
    # Robustness augmentations (simulate lighting + clutter)
    aug_prob: float = 0.85
    aug_brightness: float = 0.25
    aug_contrast: float = 0.35
    aug_gamma: Tuple[float, float] = (0.7, 1.6)
    aug_noise_std: float = 0.05
    aug_blur_prob: float = 0.25
    aug_shadow_prob: float = 0.25
    aug_invert_prob: float = 0.05
    aug_bg_texture_prob: float = 0.55
    aug_bg_strength: float = 0.35


class MovingMNIST(Dataset):
    """Generates sequences of moving MNIST digits on black background."""

    def __init__(self, root: str = "./data", train: bool = True, params: MovingMNISTParams | None = None, length: int = 20000):
        if not TORCHVISION_OK:
            raise ImportError("torchvision is required for MovingMNIST")
        self.params = params or MovingMNISTParams()
        self.length = int(length)
        self.mnist = MNIST(root=root, train=train, download=True, transform=transforms.ToTensor())

        self.H, self.W = self.params.canvas_hw
        self.dH, self.dW = self.params.digit_hw

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.params
        seq = torch.zeros((p.seq_len, 1, self.H, self.W), dtype=torch.float32)

        # sample digits
        digits = []
        for _ in range(p.digits_per_seq):
            ridx = np.random.randint(0, len(self.mnist))
            img, _ = self.mnist[ridx]
            digits.append(img)  # (1,28,28)

        # init positions and velocities
        pos = []
        vel = []
        for _ in range(p.digits_per_seq):
            y = np.random.randint(0, self.H - self.dH)
            x = np.random.randint(0, self.W - self.dW)
            vy = np.random.randint(-p.max_speed, p.max_speed + 1)
            vx = np.random.randint(-p.max_speed, p.max_speed + 1)
            if vy == 0 and vx == 0:
                vy = 1
            pos.append([y, x])
            vel.append([vy, vx])

        for t in range(p.seq_len):
            frame = torch.zeros((1, self.H, self.W), dtype=torch.float32)
            for di, dimg in enumerate(digits):
                y, x = pos[di]
                frame[:, y : y + self.dH, x : x + self.dW] = torch.maximum(
                    frame[:, y : y + self.dH, x : x + self.dW], dimg
                )
                # update position with bounce
                y2 = y + vel[di][0]
                x2 = x + vel[di][1]
                if y2 < 0 or y2 > self.H - self.dH:
                    vel[di][0] *= -1
                    y2 = y + vel[di][0]
                if x2 < 0 or x2 > self.W - self.dW:
                    vel[di][1] *= -1
                    x2 = x + vel[di][1]
                pos[di] = [y2, x2]
            seq[t] = frame

        if float(p.aug_prob) > 1e-6:
            seq = self._augment_sequence(seq)
        return seq  # (T,1,H,W)

    def _augment_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Apply simple photometric + background clutter augmentations.
        Designed to stress encoders that rely on high contrast.
        """
        p = self.params
        if np.random.rand() > float(p.aug_prob):
            return seq

        x = seq.clone()

        # Optional textured background (per sequence, consistent across frames)
        if np.random.rand() < float(p.aug_bg_texture_prob):
            bg = self._random_background((self.H, self.W))
            bg = bg * float(p.aug_bg_strength)
            x = torch.clamp(x + bg[None, None, :, :], 0.0, 1.0)

        # Brightness/contrast jitter (per sequence)
        b = (np.random.rand() * 2 - 1) * float(p.aug_brightness)
        c = 1.0 + (np.random.rand() * 2 - 1) * float(p.aug_contrast)
        x = torch.clamp((x - 0.5) * c + 0.5 + b, 0.0, 1.0)

        # Gamma jitter (per sequence)
        g0, g1 = float(p.aug_gamma[0]), float(p.aug_gamma[1])
        gamma = np.random.uniform(min(g0, g1), max(g0, g1))
        x = torch.clamp(torch.pow(torch.clamp(x, 0.0, 1.0), float(gamma)), 0.0, 1.0)

        # Synthetic shadow (per sequence, consistent across frames)
        if np.random.rand() < float(p.aug_shadow_prob):
            shadow = self._random_shadow((self.H, self.W))
            x = torch.clamp(x * shadow[None, None, :, :], 0.0, 1.0)

        # Optional inversion (rare)
        if np.random.rand() < float(p.aug_invert_prob):
            x = 1.0 - x

        # Additive Gaussian noise
        if float(p.aug_noise_std) > 1e-6:
            noise = torch.randn_like(x) * float(p.aug_noise_std)
            x = torch.clamp(x + noise, 0.0, 1.0)

        # Blur (per sequence, consistent across frames)
        if np.random.rand() < float(p.aug_blur_prob):
            x = self._gaussian_blur_torch(x, sigma=float(np.random.uniform(0.6, 1.4)))

        return x

    def _random_background(self, hw: Tuple[int, int]) -> torch.Tensor:
        H, W = int(hw[0]), int(hw[1])
        # Mixture: smooth gradient + band-limited noise
        yy = torch.linspace(0.0, 1.0, H).view(H, 1).repeat(1, W)
        xx = torch.linspace(0.0, 1.0, W).view(1, W).repeat(H, 1)
        a = float(np.random.uniform(-0.6, 0.6))
        b = float(np.random.uniform(-0.6, 0.6))
        grad = torch.clamp(0.5 + a * (xx - 0.5) + b * (yy - 0.5), 0.0, 1.0)
        noise = torch.rand((H, W)) * 2.0 - 1.0
        # cheap smoothing using avg pooling twice
        noise = noise.view(1, 1, H, W)
        noise = torch.nn.functional.avg_pool2d(noise, kernel_size=9, stride=1, padding=4)
        noise = torch.nn.functional.avg_pool2d(noise, kernel_size=9, stride=1, padding=4)
        noise = noise.view(H, W)
        noise = torch.clamp((noise - noise.min()) / (noise.max() - noise.min() + 1e-8), 0.0, 1.0)
        bg = 0.55 * grad + 0.45 * noise
        return torch.clamp(bg, 0.0, 1.0)

    def _random_shadow(self, hw: Tuple[int, int]) -> torch.Tensor:
        H, W = int(hw[0]), int(hw[1])
        yy = torch.linspace(0.0, 1.0, H).view(H, 1).repeat(1, W)
        xx = torch.linspace(0.0, 1.0, W).view(1, W).repeat(H, 1)
        # random line half-plane shadow with soft edge
        theta = float(np.random.uniform(0.0, np.pi))
        nx = float(np.cos(theta))
        ny = float(np.sin(theta))
        cx = float(np.random.uniform(0.2, 0.8))
        cy = float(np.random.uniform(0.2, 0.8))
        d = (xx - cx) * nx + (yy - cy) * ny
        width = float(np.random.uniform(0.05, 0.15))
        soft = torch.sigmoid(d / max(1e-3, width))
        dark = float(np.random.uniform(0.35, 0.75))
        shadow = (1.0 - soft) * dark + soft * 1.0
        return torch.clamp(shadow, 0.2, 1.0)

    def _gaussian_blur_torch(self, x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        # x: (T,1,H,W)
        if sigma <= 1e-6:
            return x
        k = int(np.ceil(float(sigma) * 4)) * 2 + 1
        k = max(3, min(k, 31))
        ax = torch.arange(k, dtype=torch.float32) - (k // 2)
        ker = torch.exp(-(ax * ax) / (2.0 * float(sigma) * float(sigma) + 1e-8))
        ker = ker / (ker.sum() + 1e-8)
        ker2d = (ker[:, None] * ker[None, :]).view(1, 1, k, k)
        ker2d = ker2d.to(x.device, dtype=x.dtype)
        return torch.nn.functional.conv2d(x, ker2d, padding=k // 2)

