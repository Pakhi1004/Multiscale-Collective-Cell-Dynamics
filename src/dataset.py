"""Step 3b: windowed dataset of rendered monolayer sequences.

Sample = 8 consecutive frames (~5 min apart, ~35 min span) from one run,
tensor (8, 1, 96, 96) float in [0,1]; label = (Ka,) or (Ka, alpha_scaled).

Splits (by run/seed, never by window, so overlapping windows cannot leak):
  train : sim_data/train, seeds 0-3
  val   : sim_data/train, seed 4          (model selection / early stopping)
  test  : sim_data/test     (off-grid Ka, alpha)   - final evaluation only
  extrap: sim_data/extrap   (Ka 0.05, 0.60)        - final evaluation only
  stability: 238 frames -> 231 windows at stride 1 (paper Fig. 2F)
"""
import csv, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "data", "frames")

SEQ_LEN = 8
VAL_SEEDS = {4}
ALPHA_LO, ALPHA_HI = 60.0, 100.0


def scale_alpha(a):   return (a - ALPHA_LO) / (ALPHA_HI - ALPHA_LO)
def unscale_alpha(s): return s * (ALPHA_HI - ALPHA_LO) + ALPHA_LO


def load_index():
    with open(os.path.join(FRAMES, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["Ka"], r["alpha"] = float(r["Ka"]), float(r["alpha"])
        r["seed"], r["T"] = int(r["seed"]), int(r["T"])
    return rows


def runs_for(split, rows=None):
    rows = rows if rows is not None else load_index()
    if split == "train":
        return [r for r in rows if r["split"] == "train" and r["seed"] not in VAL_SEEDS]
    if split == "val":
        return [r for r in rows if r["split"] == "train" and r["seed"] in VAL_SEEDS]
    return [r for r in rows if r["split"] == split]


CROP = 96   # every image the network sees is a 96x96 patch at native pixel scale


def center_crop(x, c=CROP):
    H, W = x.shape[-2:]
    i, j = (H - c) // 2, (W - c) // 2
    return x[..., i:i + c, j:j + c]


class Augment:
    """Same random transform for all 8 frames of a window (training only).

    - random periodic shift (images are periodic, so this is free translation)
    - flips / 90-degree rotations (the physics has no preferred direction)
    - random CROP x CROP patch, NO resize -> train and eval images have the
      same pixel scale and sharpness (resizing only in training caused a
      train/val shift: val R2 -1.98 vs 0.65 with matched processing)
    - small brightness shift only. Contrast/gamma are off by default: a
      0.75-1.3 contrast factor is larger than the whole Ka effect on area
      spread (std 0.060 at Ka=0.1 -> 0.040 at Ka=0.5) and erased the signal.
    """
    def __init__(self, crop=CROP, brightness=0.05, contrast=(1.0, 1.0), gamma=(1.0, 1.0)):
        self.crop, self.brightness, self.contrast, self.gamma = crop, brightness, contrast, gamma

    def __call__(self, x, g):                     # x: (T,1,H,W) float [0,1]
        T, _, H, W = x.shape
        x = torch.roll(x, (int(g.integers(H)), int(g.integers(W))), (2, 3))
        if g.random() < 0.5: x = x.flip(3)
        x = torch.rot90(x, int(g.integers(4)), (2, 3))
        c = self.crop
        i, j = int(g.integers(H - c + 1)), int(g.integers(W - c + 1))
        x = x[:, :, i:i + c, j:j + c]
        if self.gamma != (1.0, 1.0):
            x = x.clamp(0, 1) ** g.uniform(*self.gamma)
        m = x.mean()
        x = (x - m) * g.uniform(*self.contrast) + m + g.uniform(-self.brightness, self.brightness)
        return x.clamp(0, 1)


class SeqDataset(Dataset):
    def __init__(self, split, target="Ka", stride=4, augment=None, seed=0,
                 max_windows_per_run=None, skip=1):
        assert target in ("Ka", "Ka_alpha")
        self.split, self.target, self.augment, self.seed = split, target, augment, seed
        self.skip = skip                           # frame gap inside a window (1 = consecutive)
        self.runs = runs_for(split)
        if not self.runs:
            raise ValueError(f"no runs for split {split!r}")
        self.windows = []                          # (run_idx, start)
        for k, r in enumerate(self.runs):
            starts = list(range(0, r["T"] - (SEQ_LEN - 1) * skip, stride))
            if max_windows_per_run:
                starts = starts[:max_windows_per_run]
            self.windows += [(k, s) for s in starts]
        self._mm = {}
        self.epoch = 0
        self.fixed_aug = False

    def set_epoch(self, e): self.epoch = e      # new augmentations every epoch

    def _frames(self, k):
        if k not in self._mm:                    # lazily memory-map per worker
            r = self.runs[k]
            self._mm[k] = np.load(os.path.join(FRAMES, r["split"], r["run"] + ".npy"), mmap_mode="r")
        return self._mm[k]

    def label(self, r):
        if self.target == "Ka":
            return torch.tensor([r["Ka"]], dtype=torch.float32)
        return torch.tensor([r["Ka"], scale_alpha(r["alpha"])], dtype=torch.float32)

    def __len__(self): return len(self.windows)

    def __getitem__(self, i):
        k, s = self.windows[i]
        x = torch.from_numpy(np.array(self._frames(k)[s:s + (SEQ_LEN - 1) * self.skip + 1:self.skip], dtype=np.float32) / 255.0)
        x = x.unsqueeze(1)                       # (8,1,128,128)
        if self.augment is None:
            return center_crop(x), self.label(self.runs[k]), k   # (8,1,96,96)
        else:
            # fixed_aug=True -> reproducible (figures); else fresh draw from the
            # worker's torch RNG, which differs per worker and per epoch
            sd = (self.seed, self.epoch, i) if self.fixed_aug else int(torch.randint(2**62, (1,)))
            g = np.random.default_rng(sd)
            x = self.augment(x, g)
        return x, self.label(self.runs[k]), k


def summary():
    for sp in ["train", "val", "test", "extrap", "stability"]:
        stride = 1 if sp == "stability" else 4
        ds = SeqDataset(sp, stride=stride)
        ka = sorted({r["Ka"] for r in ds.runs}); al = sorted({r["alpha"] for r in ds.runs})
        print(f"{sp:9s} runs={len(ds.runs):3d} windows={len(ds):6d}  Ka={ka}  alpha={al}")


if __name__ == "__main__":
    summary()
