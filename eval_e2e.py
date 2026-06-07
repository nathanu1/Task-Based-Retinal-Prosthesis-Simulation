import argparse, json, warnings, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
from end2end.data import MovingMNIST, MovingMNISTParams
from end2end.models import EncoderCNN, EncoderBackbone, EncoderMoE, DecoderCNN, EncParams
from end2end.simulator import DifferentiablePhospheneSimulator, SimParams
from end2end.losses import sobel_magnitude, normalize_per_sample, weighted_mse

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--sim-hw", type=int, default=128)
ap.add_argument("--topk-frac", type=float, default=0.0556)
ap.add_argument("--moe", action="store_true")
ap.add_argument("--moe-experts", default="cnn,mobilenet_v3_small")
ap.add_argument("--backbone", default="cnn")
ap.add_argument("--device", default="cuda")
ap.add_argument("--length", type=int, default=960)
a = ap.parse_args()

dev = torch.device(a.device)
ck = torch.load(a.ckpt, map_location=dev)
ep = EncParams(stim_hw=(60, 60), topk_frac=a.topk_frac)
if a.moe:
    experts = [s.strip() for s in a.moe_experts.split(",") if s.strip()]
    enc = EncoderMoE(expert_names=experts, in_ch=1, params=ep, freeze_backbone=True)
elif a.backbone == "cnn":
    enc = EncoderCNN(in_ch=1, params=ep)
else:
    enc = EncoderBackbone(backbone=a.backbone, params=ep, freeze_backbone=True)
enc.load_state_dict(ck["encoder"]); enc.to(dev).eval()
dec = DecoderCNN(out_hw=(a.sim_hw, a.sim_hw)); dec.load_state_dict(ck["decoder"]); dec.to(dev).eval()
sim = DifferentiablePhospheneSimulator(params=SimParams(output_hw=(a.sim_hw, a.sim_hw))).to(dev).eval()

ds = MovingMNIST(root="runs/eval_data", train=False,
                 params=MovingMNISTParams(seq_len=5, digits_per_seq=2, canvas_hw=(a.sim_hw, a.sim_hw)),
                 length=a.length)
dl = DataLoader(ds, batch_size=32, shuffle=False)
tm = a.topk_frac * 0.35
agg = {k: [] for k in ["loss","recon","bnd","phos","flick","sparse","active_frac"]}
experts_n = len(a.moe_experts.split(",")) if a.moe else 0
usage = np.zeros(max(experts_n, 1))
with torch.no_grad():
    for seq in dl:
        seq = seq.to(dev); B,T,C,H,W = seq.shape; x = seq.view(B*T,C,H,W)
        if a.moe:
            stim, info = enc(x); sel = info["gate_hard"].argmax(-1).cpu().numpy()
            for s in sel: usage[s] += 1
        else:
            stim = enc(x)
        phos = sim(stim); recon = dec(phos)
        obj = (x>0.10).float(); edge_t = normalize_per_sample(sobel_magnitude(x)); wgt = 0.20+0.80*obj
        edge_p = normalize_per_sample(sobel_magnitude(recon))
        rl=F.mse_loss(recon,x); bl=weighted_mse(edge_p,edge_t,wgt); pl=F.mse_loss(phos,edge_t)
        sl=(stim.mean()-tm).abs()
        ss=stim.view(B,T,1,stim.shape[-2],stim.shape[-1]); fl=torch.mean(torch.abs(ss[:,1:]-ss[:,:-1]))
        loss=rl+1.2*bl+0.6*pl+0.25*fl+0.5*sl
        for k,v in zip(["loss","recon","bnd","phos","flick","sparse","active_frac"],
                       [loss,rl,bl,pl,fl,sl,(stim>1e-6).float().mean()]):
            agg[k].append(float(v))
print("HELD-OUT RESULTS:")
for k in agg: print(f"  {k:12s}: {np.mean(agg[k]):.4f} +/- {np.std(agg[k]):.4f}")
print(f"  active electrodes ~= {np.mean(agg['active_frac'])*3600:.0f} / 3600")
if a.moe:
    usage = usage/usage.sum()
    print("  expert routing:", {a.moe_experts.split(',')[i].strip(): f"{usage[i]*100:.1f}%" for i in range(experts_n)})
