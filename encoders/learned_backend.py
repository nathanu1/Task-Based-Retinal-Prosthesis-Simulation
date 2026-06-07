"""
Learned Encoder backend — drives phosphene encoding with the end-to-end
*trained* encoder (``end2end/``) instead of a hand-tuned, rule-based pipeline.

The encoder was trained jointly with a differentiable phosphene simulator and a
reconstruction decoder (de Ruyter van Steveninck-style end-to-end optimisation),
so it produces an *adaptive*, data-driven stimulation grid: the network decides
which electrodes to activate to best preserve object structure under a strict
sparsity (top-k) budget.

Two encoder variants are supported, auto-detected from the checkpoint:

* **CNN**            — a single learned :class:`EncoderCNN`.
* **MoE (gated)**    — a :class:`EncoderMoE` mixture-of-encoders whose gating
  network picks one expert per image. The chosen expert is reported as the
  "adaptive encoding candidate" for that frame.

The learned 60×60 stimulation grid can be rendered two ways:

* ``differentiable`` — the same differentiable simulator used during training
  (fast, consistent with how the encoder was optimised).
* ``argus_ii``       — fed as a stim-grid override into the pulse2percept
  Argus II model, i.e. the learned encoding *replaces* the edge-based stim map
  the p2p backend used by default.

Falls back gracefully (black percept + message) when torch or the checkpoints
are unavailable.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BackendResult, PerceptionContext, PhospheneBackend

try:
    import torch

    from end2end.models import (
        EncoderBackbone,
        EncoderCNN,
        EncoderMoE,
        EncParams,
    )
    from end2end.simulator import DifferentiablePhospheneSimulator, SimParams

    _TORCH_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - import guard
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
    _IMPORT_ERROR = str(_exc)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
# Human-readable label -> path, ordered best-first. Only existing files are
# offered to the UI.
_CANDIDATE_CKPTS: List[Tuple[str, str]] = [
    ("CNN (5-epoch)", "runs/paper_eval/e2e_cnn_full/ckpt_epoch5.pt"),
    ("CNN (1-epoch)", "runs/paper_eval/e2e_cnn/ckpt_epoch1.pt"),
    ("MoE (gated)", "runs/paper_eval/e2e_moe/ckpt_epoch1.pt"),
]


def available_checkpoints() -> List[Tuple[str, str]]:
    """Return [(label, path)] for checkpoints that exist on disk."""
    return [(label, path) for label, path in _CANDIDATE_CKPTS if os.path.isfile(path)]


def _infer_expert_names(enc_sd: Dict[str, Any]) -> List[str]:
    """Infer MoE expert names from an encoder state-dict.

    ``cnn`` experts expose a ``backbone.*`` submodule; pretrained-backbone
    experts expose ``feat.*`` and a ``head.0.weight`` whose input channel count
    distinguishes mobilenet_v3_small (576) from resnet18 (512).
    """
    idxs = sorted({int(m.group(1)) for k in enc_sd if (m := re.match(r"experts\.(\d+)\.", k))})
    names: List[str] = []
    for i in idxs:
        sub = [k for k in enc_sd if k.startswith(f"experts.{i}.")]
        if any(".backbone." in k for k in sub):
            names.append("cnn")
            continue
        head_w = enc_sd.get(f"experts.{i}.head.0.weight")
        feat_ch = int(head_w.shape[1]) if head_w is not None else 576
        names.append("resnet18" if feat_ch == 512 else "mobilenet_v3_small")
    return names or ["cnn"]


class _LoadedModel:
    """Bundle of a loaded encoder + differentiable simulator + metadata."""

    def __init__(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu")
        enc_sd: Dict[str, Any] = ck["encoder"]
        ep = ck.get("enc_params", {}) or {}
        sp = ck.get("sim_params", {}) or {}

        self.stim_hw: Tuple[int, int] = tuple(ep.get("stim_hw", (60, 60)))  # type: ignore[assignment]
        self.topk_frac = float(ep.get("topk_frac", 0.0556))
        temperature = float(ep.get("temperature", 1.0))
        self.input_hw: Tuple[int, int] = tuple(sp.get("output_hw", (128, 128)))  # type: ignore[assignment]

        enc_params = EncParams(
            stim_hw=self.stim_hw, topk_frac=self.topk_frac, temperature=temperature
        )

        self.is_moe = any(k.startswith("experts.") for k in enc_sd)
        self.expert_names: List[str] = []
        if self.is_moe:
            self.expert_names = _infer_expert_names(enc_sd)
            enc = EncoderMoE(
                expert_names=self.expert_names,
                in_ch=1,
                params=enc_params,
                freeze_backbone=True,
            )
        elif any(k.startswith("feat.") for k in enc_sd):
            head_w = enc_sd.get("head.0.weight")
            feat_ch = int(head_w.shape[1]) if head_w is not None else 576
            backbone = "resnet18" if feat_ch == 512 else "mobilenet_v3_small"
            self.expert_names = [backbone]
            enc = EncoderBackbone(backbone=backbone, params=enc_params, freeze_backbone=True)
        else:
            self.expert_names = ["cnn"]
            enc = EncoderCNN(in_ch=1, params=enc_params)

        enc.load_state_dict(enc_sd)
        enc.eval()
        self.encoder = enc

        self.simulator = DifferentiablePhospheneSimulator(
            params=SimParams(
                output_hw=self.input_hw,
                blur_sigma=float(sp.get("blur_sigma", 1.5)),
                blur_kernel=int(sp.get("blur_kernel", 9)),
                brightness_gamma=float(sp.get("brightness_gamma", 0.8)),
                pow_eps=float(sp.get("pow_eps", 1e-4)),
                noise_std=float(sp.get("noise_std", 0.0)),
                contrast_eps=float(sp.get("contrast_eps", 1e-6)),
            )
        ).eval()
        self.epoch = int(ck.get("epoch", 0))


class LearnedEncoderBackend(PhospheneBackend):
    """End-to-end *trained* encoder backend.

    Produces a learned, adaptive stimulation grid from the input image and
    renders it either with the differentiable training simulator or with the
    pulse2percept Argus II model.
    """

    name = "Learned Encoder (E2E)"
    available = _TORCH_AVAILABLE and bool(available_checkpoints())
    if not _TORCH_AVAILABLE:
        unavailable_reason = f"PyTorch / end2end unavailable ({_IMPORT_ERROR})"
    elif not available_checkpoints():
        unavailable_reason = "No trained checkpoint found under runs/paper_eval/"
    else:
        unavailable_reason = ""

    # Cache loaded models by checkpoint path (heavy; load once).
    _models: Dict[str, _LoadedModel] = {}

    @classmethod
    def get_model(cls, ckpt_path: str) -> Optional[_LoadedModel]:
        if not _TORCH_AVAILABLE:
            return None
        if ckpt_path not in cls._models:
            cls._models[ckpt_path] = _LoadedModel(ckpt_path)
        return cls._models.get(ckpt_path)

    def _preprocess(self, context: PerceptionContext, input_hw: Tuple[int, int]) -> "torch.Tensor":
        """Grayscale luminance → (1,1,H,W) float tensor in [0,1]."""
        gray = context.gray_clahe if context.gray_clahe is not None else context.gray
        g = np.asarray(gray, dtype=np.float32) / 255.0
        g = cv2.resize(g, (input_hw[1], input_hw[0]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(g)[None, None].float()

    def run(
        self,
        context: PerceptionContext,
        params: Dict[str, Any],
        prev_stim_grid: Optional[np.ndarray] = None,
    ) -> BackendResult:
        if not self.available:
            return self._black_percept(context, self.unavailable_reason)

        ckpts = available_checkpoints()
        label_to_path = {label: path for label, path in ckpts}
        ckpt_label = str(params.get("learned_ckpt", ckpts[0][0]))
        ckpt_path = label_to_path.get(ckpt_label, ckpts[0][1])
        renderer = str(params.get("learned_renderer", "differentiable"))

        t_total = time.perf_counter()

        try:
            model = self.get_model(ckpt_path)
        except Exception as exc:
            return self._black_percept(context, f"Failed to load checkpoint: {exc}")
        if model is None:
            return self._black_percept(context, "Learned encoder unavailable.")

        h, w = context.image.shape[:2]

        # ----- Encode (adaptive stimulation grid) -----
        t0 = time.perf_counter()
        chosen_expert: Optional[str] = None
        gate_probs: Optional[List[float]] = None
        try:
            with torch.no_grad():
                x = self._preprocess(context, model.input_hw)
                if model.is_moe:
                    stim_t, info = model.encoder(x)
                    probs = info.get("gate_probs")
                    if probs is not None:
                        gate_probs = [float(v) for v in probs.squeeze(0).tolist()]
                        chosen_expert = model.expert_names[int(np.argmax(gate_probs))]
                else:
                    stim_t = model.encoder(x)
                    chosen_expert = model.expert_names[0]
        except Exception as exc:
            return self._black_percept(context, f"Learned encoder forward failed: {exc}")
        encode_ms = (time.perf_counter() - t0) * 1000.0

        stim_grid = stim_t.squeeze().cpu().numpy().astype(np.float32)
        stim_grid = np.clip(stim_grid, 0.0, 1.0)

        # ----- Render percept -----
        t1 = time.perf_counter()
        render_used = renderer
        percept_u8: Optional[np.ndarray] = None
        render_error: Optional[str] = None

        if renderer == "argus_ii":
            percept_u8, render_error = self._render_argus(stim_grid, context, params)
            if percept_u8 is None:
                # Fall back to the differentiable simulator rather than failing.
                render_used = "differentiable"

        if percept_u8 is None:
            try:
                with torch.no_grad():
                    perc = model.simulator(stim_t)
                p = perc.squeeze().cpu().numpy().astype(np.float32)
                percept_u8 = (np.clip(p, 0.0, 1.0) * 255.0).astype(np.uint8)
            except Exception as exc:
                return self._black_percept(context, f"Differentiable render failed: {exc}")

        if percept_u8.ndim == 3:
            percept_u8 = percept_u8[:, :, 0]
        if percept_u8.shape[:2] != (h, w):
            percept_u8 = cv2.resize(percept_u8, (w, h), interpolation=cv2.INTER_LINEAR)

        total_ms = (time.perf_counter() - t_total) * 1000.0
        timings = {
            "encode_ms": encode_ms,
            "render_ms": (time.perf_counter() - t1) * 1000.0,
            "total_ms": total_ms,
        }

        intermediate_maps: Dict[str, np.ndarray] = {"learned_stim_grid": stim_grid}
        if context.edges_map is not None:
            intermediate_maps["edges_map"] = context.edges_map
        if context.saliency_map is not None:
            intermediate_maps["saliency"] = context.saliency_map

        active = int(np.sum(stim_grid > 1e-6))
        metadata: Dict[str, Any] = {
            "checkpoint": ckpt_label,
            "variant": "MoE" if model.is_moe else "CNN",
            "renderer": render_used,
            "stim_grid": f"{stim_grid.shape[0]}×{stim_grid.shape[1]}",
            "topk_frac": round(model.topk_frac, 4),
            "active_electrodes": active,
            "trained_epochs": model.epoch,
            "adaptive_candidate": chosen_expert or "—",
        }
        if model.is_moe and gate_probs is not None:
            metadata["expert_names"] = model.expert_names
            metadata["gate_probs"] = [round(p, 3) for p in gate_probs]
            # Surface the per-image adaptive choice as a candidate table.
            metadata["adaptive"] = {
                "chosen": chosen_expert or "—",
                "stable_frames": 0,
                "candidates": [
                    {"candidate": n, "gate_prob": round(p, 3), "chosen": n == chosen_expert}
                    for n, p in zip(model.expert_names, gate_probs)
                ],
            }
        if render_error:
            metadata["render_note"] = render_error

        return BackendResult(
            backend_name=self.name,
            input_image=context.image,
            task_mode=context.task_mode,
            stimulation_grid=stim_grid,
            phosphene_image=percept_u8,
            intermediate_maps=intermediate_maps,
            timing_info=timings,
            metadata=metadata,
        )

    def _render_argus(
        self,
        stim_grid: np.ndarray,
        context: PerceptionContext,
        params: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Render the learned stim grid through pulse2percept Argus II.

        Returns (percept_uint8 | None, note). On any failure returns
        (None, reason) so the caller can fall back to the differentiable sim.
        """
        try:
            from .p2p_retinal_backend import P2PRetinalBackend
        except Exception as exc:
            return None, f"p2p backend import failed ({exc})"

        model_key = str(params.get("p2p_model", "axon_map"))
        sim = P2PRetinalBackend.get_simulator(model_key)
        if sim is None:
            return None, "pulse2percept Argus II model unavailable"

        h, w = context.image.shape[:2]
        try:
            percept = sim.simulate_from_grid(stim_grid, output_size=(h, w), as_uint8=True)
        except Exception as exc:
            return None, f"Argus II render failed ({exc})"
        return np.asarray(percept, dtype=np.uint8), None
