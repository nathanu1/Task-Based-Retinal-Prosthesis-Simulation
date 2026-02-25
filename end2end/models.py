"""Encoder/Decoder models for end-to-end prosthetic vision training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def straight_through_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Hard top-k mask in forward, soft gradients in backward (STE)."""
    if k <= 0:
        return torch.zeros_like(scores)
    b = scores.shape[0]
    flat = scores.view(b, -1)
    k = min(k, flat.shape[1])
    _, idx = torch.topk(flat, k=k, dim=1)
    hard = torch.zeros_like(flat)
    hard.scatter_(1, idx, 1.0)
    hard = hard.view_as(scores)
    # STE: pretend hard == soft in backward
    return hard + (scores - scores.detach())


def gumbel_softmax_st(
    logits: torch.Tensor,
    *,
    tau: float = 1.0,
    hard: bool = True,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gumbel-softmax with straight-through hard selection.

    Returns:
      y: same shape as logits, one-hot in forward if hard=True
      probs: softmax(logits) (no gumbel), useful for logging/regularization
    """
    probs = torch.softmax(logits, dim=dim)
    if not logits.requires_grad and not logits.is_floating_point():
        return probs, probs
    if hard:
        y = F.gumbel_softmax(logits, tau=float(tau), hard=True, dim=dim)
    else:
        y = F.gumbel_softmax(logits, tau=float(tau), hard=False, dim=dim)
    return y, probs


@dataclass(frozen=True)
class EncParams:
    stim_hw: Tuple[int, int] = (60, 60)
    topk_frac: float = 0.12  # fraction of electrodes active
    temperature: float = 1.0


class EncoderCNN(nn.Module):
    """Lightweight encoder producing a stimulation grid (B,1,Hg,Wg) in [0,1]."""

    def __init__(self, in_ch: int = 1, params: Optional[EncParams] = None):
        super().__init__()
        self.params = params or EncParams()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,1,H,W) normalized [0,1]
        Returns:
            stim: (B,1,Hg,Wg) in [0,1]
        """
        f = self.backbone(x)
        s = self.head(f)
        s = F.interpolate(s, size=self.params.stim_hw, mode="bilinear", align_corners=False)
        # soft scores
        s = torch.sigmoid(s / float(self.params.temperature))

        # enforce sparsity with STE top-k
        k = int(self.params.topk_frac * self.params.stim_hw[0] * self.params.stim_hw[1])
        mask = straight_through_topk(s, k=k)
        stim = (s * mask).clamp(0, 1)
        return stim


class EncoderBackbone(nn.Module):
    """Encoder with a lightweight pretrained backbone (MobileNetV3 / ResNet18).

    For object recognition, pretrained features help generalize beyond MNIST.
    Input is grayscale; we repeat channels to 3 for pretrained backbones.
    """

    def __init__(self, backbone: str = "mobilenet_v3_small", params: Optional[EncParams] = None, freeze_backbone: bool = True):
        super().__init__()
        self.params = params or EncParams()
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone

        try:
            import torchvision.models as M
            from torchvision.models import (
                mobilenet_v3_small,
                resnet18,
            )
        except Exception as e:
            raise ImportError("torchvision is required for pretrained backbones") from e

        if backbone == "mobilenet_v3_small":
            # Newer torchvision uses weights enums; fall back if unavailable
            try:
                from torchvision.models import MobileNet_V3_Small_Weights
                weights = MobileNet_V3_Small_Weights.DEFAULT
                net = mobilenet_v3_small(weights=weights)
            except Exception:
                net = mobilenet_v3_small(pretrained=True)  # type: ignore
            self.feat = net.features
            feat_ch = 576
        elif backbone == "resnet18":
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT
                net = resnet18(weights=weights)
            except Exception:
                net = resnet18(pretrained=True)  # type: ignore
            # take conv1..layer4
            self.feat = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4
            )
            feat_ch = 512
        else:
            raise ValueError("backbone must be one of: mobilenet_v3_small, resnet18")

        if freeze_backbone:
            for p in self.feat.parameters():
                p.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Conv2d(feat_ch, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W) -> repeat to 3ch
        if x.shape[1] == 1:
            x3 = x.repeat(1, 3, 1, 1)
        else:
            x3 = x
        f = self.feat(x3)
        s = self.head(f)
        s = F.interpolate(s, size=self.params.stim_hw, mode="bilinear", align_corners=False)
        s = torch.sigmoid(s / float(self.params.temperature))
        k = int(self.params.topk_frac * self.params.stim_hw[0] * self.params.stim_hw[1])
        mask = straight_through_topk(s, k=k)
        return (s * mask).clamp(0, 1)


class DecoderCNN(nn.Module):
    """Decoder reconstructing the input image from the percept (B,1,H,W)."""

    def __init__(self, out_hw: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.out_hw = out_hw
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, percept: torch.Tensor) -> torch.Tensor:
        y = self.net(percept)
        y = F.interpolate(y, size=self.out_hw, mode="bilinear", align_corners=False)
        return y.clamp(0, 1)


class GatingCNN(nn.Module):
    """Small gating network producing expert logits from input."""

    def __init__(self, in_ch: int, n_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(64, int(n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)
        # global average pool
        g = f.mean(dim=(-2, -1))
        return self.head(g)


class EncoderMoE(nn.Module):
    """
    Mixture-of-Encoders: selects one expert encoder per sample using a gating network.

    Experts are any of:
      - "cnn"
      - torchvision backbones supported by EncoderBackbone ("mobilenet_v3_small", "resnet18")
    """

    def __init__(
        self,
        *,
        expert_names: List[str],
        in_ch: int = 1,
        params: Optional[EncParams] = None,
        freeze_backbone: bool = True,
        gate_tau: float = 1.0,
        adaptive_bias_lr: float = 0.10,
        usage_ema_decay: float = 0.99,
        bias_clip: float = 0.50,
    ):
        super().__init__()
        if not expert_names:
            raise ValueError("expert_names must be non-empty")
        self.params = params or EncParams()
        self.expert_names = list(expert_names)
        self.gate_tau = float(gate_tau)
        self.adaptive_bias_lr = float(adaptive_bias_lr)
        self.usage_ema_decay = float(usage_ema_decay)
        self.bias_clip = float(bias_clip)

        experts: List[nn.Module] = []
        for name in self.expert_names:
            if name == "cnn":
                experts.append(EncoderCNN(in_ch=in_ch, params=self.params))
            else:
                experts.append(EncoderBackbone(backbone=name, params=self.params, freeze_backbone=freeze_backbone))
        self.experts = nn.ModuleList(experts)
        self.gate = GatingCNN(in_ch=in_ch, n_experts=len(self.expert_names))

        # Adaptive expert bias (non-gradient load balancing, inspired by utilization EMA)
        self.expert_bias = nn.Parameter(torch.zeros(len(self.expert_names), dtype=torch.float32), requires_grad=False)
        self.register_buffer("usage_ema", torch.full((len(self.expert_names),), 1.0 / len(self.expert_names), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
          stim: (B,1,Hg,Wg)
          info: dict with gate logits/probs/selection
        """
        logits = self.gate(x)  # (B,E)
        if self.expert_bias.numel() == logits.shape[-1]:
            logits = logits + self.expert_bias.view(1, -1)
        # Safety: avoid gumbel edge-cases for E==1 (can produce NaNs in some torch builds)
        if logits.shape[-1] == 1:
            probs = torch.ones_like(logits)
            y = torch.ones_like(logits)
        else:
            # Clamp logits to a sane range to prevent overflow in softmax under extreme gradients
            logits = logits.clamp(-20.0, 20.0)
            tau = float(max(0.05, self.gate_tau))
            if self.training:
                y, probs = gumbel_softmax_st(logits, tau=tau, hard=True, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                # deterministic expert choice in eval
                idx = torch.argmax(probs, dim=-1)
                y = F.one_hot(idx, num_classes=probs.shape[-1]).to(dtype=probs.dtype)

        # Update utilization EMA and adaptive bias (no auxiliary loss)
        if self.training and logits.shape[-1] > 1 and self.adaptive_bias_lr > 1e-8:
            with torch.no_grad():
                # y is one-hot in training (hard routing). Fall back to probs if not.
                usage = y if y.dtype.is_floating_point else y.to(dtype=torch.float32)
                if usage.ndim == 2:
                    usage = usage.mean(dim=0)  # (E,)
                else:
                    usage = probs.mean(dim=0)

                decay = float(min(max(self.usage_ema_decay, 0.0), 0.9999))
                self.usage_ema.mul_(decay).add_(usage.to(self.usage_ema.dtype) * (1.0 - decay))

                target = 1.0 / float(self.usage_ema.numel())
                bias_update = float(self.adaptive_bias_lr) * (target - self.usage_ema)
                self.expert_bias.add_(bias_update.to(self.expert_bias.dtype))
                self.expert_bias.clamp_(-abs(float(self.bias_clip)), abs(float(self.bias_clip)))

        # expert outputs
        stim_list = [e(x) for e in self.experts]  # each (B,1,Hg,Wg)
        stim_stack = torch.stack(stim_list, dim=1)  # (B,E,1,Hg,Wg)
        w = y.view(y.shape[0], y.shape[1], 1, 1, 1)
        stim = (stim_stack * w).sum(dim=1).clamp(0, 1)
        info: Dict[str, Any] = {
            "gate_logits": logits,
            "gate_probs": probs,
            "gate_hard": y,
            "usage_ema": self.usage_ema.detach().clone(),
            "expert_bias": self.expert_bias.detach().clone(),
        }
        return stim, info

