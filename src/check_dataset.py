"""Step 3c: sanity checks - split summary, leakage, batch shape, figures."""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from dataset import SeqDataset, Augment, summary, runs_for, unscale_alpha, ROOT

FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

def main():
    summary()

    # 1. no run shared between splits
    names = {sp: {r["run"] for r in runs_for(sp)} for sp in ["train", "val", "test", "extrap", "stability"]}
    sps = list(names)
    for a in range(len(sps)):
        for b in range(a + 1, len(sps)):
            assert not names[sps[a]] & names[sps[b]], (sps[a], sps[b])
    print("leakage check: no run appears in two splits  OK")

    # 2. batch shape through a DataLoader with workers
    ds = SeqDataset("train", target="Ka_alpha", augment=Augment())
    x, y, _ = next(iter(DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)))
    print("batch x", tuple(x.shape), x.dtype, f"[{x.min():.2f},{x.max():.2f}]", " y", tuple(y.shape))
    assert x.shape[1:] == (8, 1, 128, 128)

    # 3. does raw grey spread separate Ka? (why augmentation matters)
    raw = SeqDataset("train", stride=200)
    st = {}
    for i in range(len(raw)):
        xx, yy, _ = raw[i]
        st.setdefault(round(float(yy[0]), 2), []).append(float(xx[xx > 0.05].std()))
    print("cell-grey std by Ka (raw):", {k: round(np.mean(v), 4) for k, v in sorted(st.items())})

    # 4. figure: Ka contrast, one window raw vs augmented
    eval_ds = SeqDataset("train", target="Ka_alpha", stride=500)
    pick = {}
    for i, (k, s) in enumerate(eval_ds.windows):
        r = eval_ds.runs[k]
        if r["alpha"] == 80 and s == 0 and r["seed"] == 0: pick[r["Ka"]] = i
    fig, ax = plt.subplots(1, len(pick), figsize=(3 * len(pick), 3.2))
    for a, (ka, i) in zip(ax, sorted(pick.items())):
        a.imshow(eval_ds[i][0][0, 0], cmap="gray", vmin=0, vmax=1); a.set_title(f"Ka={ka}, α=80"); a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "step3_Ka_comparison.png"), dpi=110); plt.close()

    i0 = pick[min(pick)]
    xr, y0, _ = eval_ds[i0]
    fig, ax = plt.subplots(4, 8, figsize=(16, 8.4))
    for c in range(8):
        ax[0, c].imshow(xr[c, 0], cmap="gray", vmin=0, vmax=1); ax[0, c].set_title(f"frame {c}")
    for row, ep in zip(range(1, 4), range(3)):
        eval_ds.augment, eval_ds.epoch, eval_ds.fixed_aug = Augment(), ep, True
        xa = eval_ds[i0][0]
        for c in range(8): ax[row, c].imshow(xa[c, 0], cmap="gray", vmin=0, vmax=1)
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(f"row 1: raw window   rows 2-4: augmented   label Ka={y0[0]:.2f}, α={unscale_alpha(y0[1]):.0f}")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "step3_augmentation.png"), dpi=90); plt.close()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
