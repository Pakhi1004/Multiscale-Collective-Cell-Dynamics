"""Evaluate a checkpoint: per-window R^2 and per-run R^2 (predictions averaged
over all windows of a run - Ka is constant within a run).

  python src/evaluate.py runs/cvt_Ka_100ep [--split val] [--stride 8]
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from dataset import SeqDataset, CROP, unscale_alpha
from model import CvT
from train import predict, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="val")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    args = json.load(open(os.path.join(a.run_dir, "args.json")))
    ck = torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    m = CvT(ck["n_out"], img=CROP).to(dev); m.load_state_dict(ck["model"])
    ds = SeqDataset(a.split, ck["target"], stride=a.stride, skip=args.get("skip", 1))
    P, Y, K = predict(m, DataLoader(ds, 64, num_workers=2), dev)
    runs = K.unique()
    Pr = torch.stack([P[K == k].mean(0) for k in runs]); Yr = torch.stack([Y[K == k][0] for k in runs])
    names = ["Ka", "alpha"][: ck["n_out"]]
    print(f"{a.run_dir}  split={a.split}  epoch={ck['epoch']}  windows={len(P)} runs={len(runs)}")
    for i, n in enumerate(names):
        print(f"  {n:5s} per-window R2 {r2(P[:, i:i+1], Y[:, i:i+1])[0]:.4f}   per-run R2 {r2(Pr[:, i:i+1], Yr[:, i:i+1])[0]:.4f}")
    for v in sorted(Yr[:, 0].unique().tolist()):
        sel = Y[:, 0] == v
        print(f"  true Ka {v:.2f}: pred {P[sel,0].mean():.3f} ± {P[sel,0].std():.3f} (window sd)")


if __name__ == "__main__":
    main()
