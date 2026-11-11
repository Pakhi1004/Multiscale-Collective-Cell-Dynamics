"""Step 4b: train the CvT on simulated sequences.

  python src/train.py --target Ka                 # 1-parameter model
  python src/train.py --target Ka_alpha           # 2-parameter model
  python src/train.py --target Ka --epochs 2 --samples 512 --tag smoke   # quick test

Paper settings: Adam lr 1e-4, ExponentialLR gamma 0.99, batch 32, 100 epochs, MSE.
Each epoch draws --samples random augmented training windows (fresh augmentation
every epoch); validation = held-out seed 4, no augmentation. Best checkpoint by
validation R^2 (mean over outputs) -> runs/<tag>/best.pt ; log -> runs/<tag>/log.csv
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
sys.path.insert(0, os.path.dirname(__file__))
from dataset import SeqDataset, Augment, ROOT, CROP
from model import CvT, n_params


def r2(pred, true):
    ss_res = ((pred - true) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).tolist()


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, Y, K = [], [], []
    for x, y, k in loader:
        P.append(model(x.to(dev)).cpu()); Y.append(y); K.append(k)
    return torch.cat(P), torch.cat(Y), torch.cat(K)


def make_aug(kind):
    if kind == "none": return None
    if kind == "paper": return Augment(contrast=(0.75, 1.3), gamma=(0.8, 1.25), brightness=0.12)
    return Augment()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Ka", choices=["Ka", "Ka_alpha"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples", type=int, default=6000, help="training windows per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--val_stride", type=int, default=24)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip", type=int, default=1, help="frame gap inside a window")
    ap.add_argument("--aug", default="full", choices=["full", "paper", "none"],
                    help="full = shift/flip/rot/crop + small brightness; paper = + strong contrast/gamma")
    args = ap.parse_args()
    tag = args.tag or f"cvt_{args.target}"
    out = os.path.join(ROOT, "runs", tag); os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)

    torch.manual_seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    tr = SeqDataset("train", args.target, stride=args.stride, augment=make_aug(args.aug), seed=args.seed, skip=args.skip)
    va = SeqDataset("val", args.target, stride=args.val_stride, skip=args.skip)
    n_out = 1 if args.target == "Ka" else 2
    model = CvT(n_out=n_out, img=CROP).to(dev)
    print(f"device={dev} params={n_params(model):,} train_windows={len(tr)} val_windows={len(va)}")

    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(tr, args.batch, sampler=RandomSampler(tr, num_samples=args.samples, generator=g),
                    num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    vl = DataLoader(va, 64, num_workers=args.workers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, args.gamma)
    loss_fn = nn.MSELoss()

    best, logf = -1e9, open(os.path.join(out, "log.csv"), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["epoch", "train_mse", "val_mse"] + [f"val_r2_{i}" for i in range(n_out)] + ["lr", "sec"])
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, _ in tl:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        P, Y, _ = predict(model, vl, dev)
        vmse, vr2 = float(((P - Y) ** 2).mean()), r2(P, Y)
        score = float(np.mean(vr2))
        log.writerow([ep, tot / n, vmse] + vr2 + [sched.get_last_lr()[0], round(time.time() - t0, 1)]); logf.flush()
        flag = ""
        if score > best:
            best = score; flag = " *"
            torch.save({"model": model.state_dict(), "n_out": n_out, "target": args.target,
                        "epoch": ep, "val_r2": vr2}, os.path.join(out, "best.pt"))
        print(f"ep {ep:3d} train {tot/n:.5f} val {vmse:.5f} R2 {[round(v,4) for v in vr2]} "
              f"{time.time()-t0:.0f}s{flag}", flush=True)
    torch.save({"model": model.state_dict(), "n_out": n_out, "target": args.target, "epoch": args.epochs},
               os.path.join(out, "last.pt"))
    print("best val R2", best, "->", out)


if __name__ == "__main__":
    main()
