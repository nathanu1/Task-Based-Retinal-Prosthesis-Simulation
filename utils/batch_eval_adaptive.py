"""
Batch evaluation harness for adaptive encoding.

Usage:
  python -m utils.batch_eval_adaptive --in path/to/images --out runs/adaptive_eval

Saves:
  - per-image composite grid (candidates + chosen)
  - per-image JSON metrics
  - summary CSV
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

from utils.adaptive_encoding import (
    EncodingCandidate,
    CandidateScore,
    default_candidates,
    grid_foreground_mask,
    grid_boundary_mask,
    score_stim_grid,
)


def _robust_normalize(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi - lo < 1e-8:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _mask_boundary(mask_u8: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    ker = np.ones((k, k), np.uint8)
    grad = cv2.morphologyEx((mask_u8 > 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, ker)
    b = (grad > 0).astype(np.float32)
    return b


def _simple_gate(gray_u8: np.ndarray) -> np.ndarray:
    """
    Lightweight foreground proxy when you don't want to run segmentation/YOLO:
    - local contrast + edges, normalized
    """
    g = np.asarray(gray_u8, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(g)
    edges = cv2.Canny(cv2.GaussianBlur(g2, (5, 5), 0), 50, 150).astype(np.float32) / 255.0
    lum = (g2.astype(np.float32) / 255.0)
    lp = cv2.GaussianBlur(lum, (0, 0), 3.0)
    detail = np.abs(lum - lp).astype(np.float32)
    gate = _robust_normalize(0.65 * edges + 0.35 * detail, 1.0, 99.0)
    gate = cv2.GaussianBlur(gate, (0, 0), 1.2)
    return _robust_normalize(gate, 1.0, 99.0)


def _candidate_stim(
    gray_u8: np.ndarray,
    *,
    cand: EncodingCandidate,
    grid_n: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a candidate stimulation grid + an edges visualization map (H,W float32 in [0,1]).
    This is a simplified version of what `app.py` does, meant for offline comparison.
    """
    from utils.adaptive_encoding import clahe_u8, gamma_u8, retinex_ssr_u8

    g = np.asarray(gray_u8, dtype=np.uint8)
    if cand.contrast_mode == "clahe":
        g = clahe_u8(g)
    elif cand.contrast_mode == "gamma":
        g = gamma_u8(g, cand.gamma)
    elif cand.contrast_mode == "retinex":
        g = retinex_ssr_u8(g)

    if cand.use_bilateral:
        try:
            g = cv2.bilateralFilter(g, d=7, sigmaColor=55, sigmaSpace=7)
        except Exception:
            pass

    # Edges
    if cand.edge_method == "scharr":
        gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx * gx + gy * gy)
        edges = _robust_normalize(mag, 1.0, 99.0)
    else:
        edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150).astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (0, 0), 0.6)
        edges = _robust_normalize(edges, 1.0, 99.0)

    # Luminance for fill
    lum = (g.astype(np.float32) / 255.0)
    lum = cv2.GaussianBlur(lum, (0, 0), 1.0)
    lum = _robust_normalize(lum, 2.0, 98.0)
    lp = cv2.GaussianBlur(lum, (0, 0), 3.0)
    detail = np.abs(lum - lp).astype(np.float32)
    detail = _robust_normalize(detail, 2.0, 99.0)
    lum = _robust_normalize(0.65 * lum + 0.35 * detail, 1.0, 99.0)

    # Grid score -> sparse top-k
    edge_g = cv2.resize(edges, (grid_n, grid_n), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum_g = cv2.resize(lum, (grid_n, grid_n), interpolation=cv2.INTER_AREA).astype(np.float32)
    edge_g = _robust_normalize(edge_g, 1.0, 99.0)
    lum_g = _robust_normalize(lum_g, 1.0, 99.0)
    score_g = np.clip((1.0 - float(cand.fill_weight)) * edge_g + float(cand.fill_weight) * lum_g, 0.0, 1.0)
    score_g = _robust_normalize(score_g, 1.0, 99.0)

    # fixed budget for comparison
    budget = int(np.clip(grid_n * grid_n * 0.12, grid_n * grid_n * 0.05, grid_n * grid_n * 0.16))
    flat = score_g.reshape(-1)
    idx = np.argpartition(flat, -budget)[-budget:]
    used = np.zeros_like(flat, dtype=np.float32)
    used[idx] = 1.0
    used = used.reshape(score_g.shape)
    stim_g = np.clip(score_g * used, 0.0, 1.0).astype(np.float32)
    return stim_g, edges


def _render_dots(stim_g: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    from utils.dot_phosphene_renderer import DotRenderParams, render_dots_from_grid

    return render_dots_from_grid(
        np.asarray(stim_g, dtype=np.float32),
        output_size=out_hw,
        params=DotRenderParams(sigma_px=1.6, blend="sum", jitter_px=0.0),
    )


def _tile(images: List[np.ndarray], cols: int) -> np.ndarray:
    rows = int(np.ceil(len(images) / max(1, cols)))
    h = max(int(im.shape[0]) for im in images)
    w = max(int(im.shape[1]) for im in images)
    out = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        hh, ww = im.shape[:2]
        out[r * h : r * h + hh, c * w : c * w + ww] = im
    return out


def _list_images(in_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Input directory of images")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory for eval artifacts")
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image").mkdir(parents=True, exist_ok=True)

    paths = _list_images(in_dir)
    if not paths:
        raise SystemExit(f"No images found in {in_dir}")

    summary_rows: List[Dict[str, Any]] = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gate = _simple_gate(gray)
        fg_mask_g = grid_foreground_mask(gate=gate, object_mask_u8=None, grid_n=int(args.grid), gate_thresh=0.25)
        boundary_mask_g = None

        candidates = default_candidates(has_object_boundary=False)
        scored: List[Tuple[EncodingCandidate, CandidateScore]] = []
        stim_cache: Dict[str, np.ndarray] = {}
        img_cache: Dict[str, np.ndarray] = {}
        for cand in candidates:
            stim_g, _ = _candidate_stim(gray, cand=cand, grid_n=int(args.grid))
            stim_cache[cand.name] = stim_g
            cs = score_stim_grid(stim_grid=stim_g, fg_mask_g=fg_mask_g, boundary_mask_g=boundary_mask_g, target_active_dots=int(args.grid * args.grid * 0.12))
            scored.append((cand, cs))

        scored_sorted = sorted(scored, key=lambda t: float(t[1].score), reverse=True)
        chosen_c, chosen_s = scored_sorted[0]

        # Render composites
        tiles: List[np.ndarray] = []
        for cand, cs in scored_sorted[:6]:
            ph = _render_dots(stim_cache[cand.name], out_hw=(h, w))
            label = f"{cand.name} s={cs.score:.3f}"
            ph2 = cv2.putText(ph.copy(), label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 200, 2, cv2.LINE_AA)
            tiles.append(ph2)

        chosen = _render_dots(stim_cache[chosen_c.name], out_hw=(h, w))
        chosen = cv2.putText(chosen.copy(), f"CHOSEN: {chosen_c.name}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        tiles.append(chosen)

        grid_img = _tile(tiles, cols=int(args.cols))
        rel = p.relative_to(in_dir)
        out_base = (out_dir / "per_image" / rel).with_suffix("")
        out_base.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_base) + "_grid.png", grid_img)

        record = {
            "path": str(rel).replace("\\\\", "/"),
            "chosen": chosen_c.name,
            "chosen_score": float(chosen_s.score),
            "chosen_fg_energy": float(chosen_s.fg_energy),
            "chosen_leak": float(chosen_s.leak_energy),
        }
        with open(str(out_base) + "_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": record,
                    "candidates": [
                        {
                            "name": c.name,
                            "score": float(s.score),
                            "fg_energy": float(s.fg_energy),
                            "boundary_energy": float(s.boundary_energy),
                            "leak": float(s.leak_energy),
                            "dots": int(s.active_dots),
                        }
                        for c, s in scored_sorted
                    ],
                },
                f,
                indent=2,
            )
        summary_rows.append(record)

    # Write summary CSV
    import csv

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wcsv.writeheader()
        for r in summary_rows:
            wcsv.writerow(r)

    print(f"Done. Wrote {len(summary_rows)} results to {out_dir}")


if __name__ == "__main__":
    main()

