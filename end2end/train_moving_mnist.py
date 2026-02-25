"""Train end-to-end encoder->phosphene simulator->decoder on MovingMNIST.

This follows the schematic described in de Ruyter van Steveninck et al. (2022a) and
the eLife dynaphos-style pipeline: encoder outputs stimulation, simulator produces phosphenes,
decoder reconstructs input. Optimize with backprop through the simulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MovingMNIST, MovingMNISTParams
from .models import EncoderCNN, EncoderBackbone, DecoderCNN, EncParams, EncoderMoE
from .simulator import DifferentiablePhospheneSimulator, SimParams
from .utils import save_grid_triplet
from .losses import sobel_magnitude, normalize_per_sample, weighted_mse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="runs/moving_mnist")
    p.add_argument("--topk-frac", type=float, default=0.12)
    p.add_argument("--sim-hw", type=int, default=128)
    p.add_argument("--lambda-sparsity", type=float, default=0.5)
    p.add_argument("--lambda-boundary", type=float, default=1.2, help="Edge/boundary supervision on recon (object-centric).")
    p.add_argument("--lambda-phos-boundary", type=float, default=0.6, help="Regularize phosphene percept towards object boundaries.")
    p.add_argument("--lambda-temporal", type=float, default=0.25, help="Penalize flicker between consecutive stim frames.")
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["cnn", "mobilenet_v3_small", "resnet18"])
    p.add_argument("--finetune-backbone", action="store_true", help="Allow backbone finetuning (slower, better).")
    p.add_argument("--moe", action="store_true", help="Use a mixture-of-encoders with a gating network.")
    p.add_argument("--moe-experts", type=str, default="cnn,mobilenet_v3_small", help="Comma-separated expert names when --moe is set.")
    p.add_argument("--moe-tau", type=float, default=1.0, help="Gumbel-softmax temperature for gating (lower -> harder selection).")
    p.add_argument("--lambda-gate-entropy", type=float, default=0.02, help="Entropy penalty to encourage confident expert selection.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    ds = MovingMNIST(
        root=str(out_dir / "data"),
        train=True,
        params=MovingMNISTParams(seq_len=args.seq_len, digits_per_seq=args.digits, canvas_hw=(args.sim_hw, args.sim_hw)),
        length=20000,
    )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)

    enc_params = EncParams(stim_hw=(60, 60), topk_frac=args.topk_frac)
    if bool(args.moe):
        expert_names = [s.strip() for s in str(args.moe_experts).split(",") if s.strip()]
        enc = EncoderMoE(
            expert_names=expert_names,
            in_ch=1,
            params=enc_params,
            freeze_backbone=not args.finetune_backbone,
            gate_tau=float(args.moe_tau),
        ).to(device)
    else:
        if args.backbone == "cnn":
            enc = EncoderCNN(in_ch=1, params=enc_params).to(device)
        else:
            enc = EncoderBackbone(backbone=args.backbone, params=enc_params, freeze_backbone=not args.finetune_backbone).to(device)
    sim = DifferentiablePhospheneSimulator(params=SimParams(output_hw=(args.sim_hw, args.sim_hw))).to(device)
    dec = DecoderCNN(out_hw=(args.sim_hw, args.sim_hw)).to(device)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)

    # target sparsity: average stimulation energy, encourages consistent dot density
    target_mean = float(args.topk_frac) * 0.35

    step = 0
    for epoch in range(1, args.epochs + 1):
        enc.train()
        dec.train()
        sim.train()

        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}")
        for seq in pbar:
            # seq: (B,T,1,H,W)
            seq = seq.to(device)
            B, T, C, H, W = seq.shape

            # flatten time for batched processing
            x = seq.view(B * T, C, H, W)
            gate_info = None
            if bool(args.moe):
                stim, gate_info = enc(x)  # type: ignore[misc]
            else:
                stim = enc(x)  # (B*T,1,60,60)
            phos = sim(stim)  # (B*T,1,H,W)
            recon = dec(phos)  # (B*T,1,H,W)

            # Object-centric boundary targets from input (MovingMNIST digits = objects)
            # Boundary target = Sobel magnitude of input; weight loss more on object strokes.
            with torch.no_grad():
                obj_mask = (x > 0.10).float()
                edge_t = normalize_per_sample(sobel_magnitude(x))
                # emphasize edges near objects
                weight = 0.20 + 0.80 * obj_mask

            edge_p = normalize_per_sample(sobel_magnitude(recon))

            # losses
            recon_loss = F.mse_loss(recon, x)
            boundary_loss = weighted_mse(edge_p, edge_t, weight)
            phos_boundary_loss = F.mse_loss(phos, edge_t)

            sparsity_loss = (stim.mean() - target_mean).abs()
            # temporal loss (flicker) on stimulation grid across frames
            stim_seq = stim.view(B, T, 1, stim.shape[-2], stim.shape[-1])
            flicker = torch.mean(torch.abs(stim_seq[:, 1:] - stim_seq[:, :-1]))

            gate_entropy = torch.tensor(0.0, device=device)
            if gate_info is not None and "gate_probs" in gate_info:
                probs = gate_info["gate_probs"].clamp(1e-8, 1.0)
                ent = -(probs * torch.log(probs)).sum(dim=-1)  # (B*T,)
                gate_entropy = ent.mean()

            loss = (
                recon_loss
                + float(args.lambda_boundary) * boundary_loss
                + float(args.lambda_phos_boundary) * phos_boundary_loss
                + float(args.lambda_temporal) * flicker
                + float(args.lambda_sparsity) * sparsity_loss
                + float(args.lambda_gate_entropy) * gate_entropy
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            pbar.set_postfix({
                "loss": float(loss.detach().cpu()),
                "recon": float(recon_loss.detach().cpu()),
                "bnd": float(boundary_loss.detach().cpu()),
                "flick": float(flicker.detach().cpu()),
                "sparse": float(sparsity_loss.detach().cpu()),
                "gateH": float(gate_entropy.detach().cpu()) if gate_info is not None else 0.0,
            })

            # sample viz
            if step % 400 == 0:
                enc.eval()
                dec.eval()
                sim.eval()
                with torch.no_grad():
                    seq_v = seq[:4]  # (b,T,1,H,W)
                    xv = seq_v.view(-1, 1, H, W)
                    if bool(args.moe):
                        stimv, _ = enc(xv)  # type: ignore[misc]
                    else:
                        stimv = enc(xv)
                    phov = sim(stimv)
                    recv = dec(phov)
                    save_grid_triplet(out_dir / f"epoch{epoch}_step{step}.png",
                                      inputs=seq_v,
                                      phos=phov.view(seq_v.shape),
                                      recons=recv.view(seq_v.shape))
                enc.train()
                dec.train()
                sim.train()

        # save checkpoint
        ckpt = {
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "sim_params": sim.params.__dict__,
            "enc_params": enc.params.__dict__,
            "epoch": epoch,
        }
        torch.save(ckpt, out_dir / f"ckpt_epoch{epoch}.pt")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

