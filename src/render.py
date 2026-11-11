"""Step 3a: render vertex-model NPZ runs into uint8 image stacks.

Each run -> data/frames/<split>/<run>.npy  with shape (T, 128, 128), uint8.
Plus data/frames/index.csv listing split, run, Ka, alpha, seed, T.

Rendering = periodic Voronoi on cell centres (nearest-centre labelling on a
pixel grid with periodic KD-tree), cells filled by area with ONE fixed mapping
for every run (never per-image normalisation), black 1-px edges at 256 px,
then 2x2 average-pooled to 128 px.
"""
import argparse, csv, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim_data")
OUT = os.path.join(ROOT, "data", "frames")
SPLITS = ["train", "test", "extrap", "stability"]

HI = 256            # internal render size
LO = 128            # saved size
A_MIN, A_MAX = 0.6, 1.4      # fixed area range for ALL runs
G_SMALL, G_LARGE = 255, 20   # lighter = smaller (paper Fig. 1C); edges = 0


def area_to_grey(a):
    t = np.clip((a - A_MIN) / (A_MAX - A_MIN), 0.0, 1.0)
    return G_SMALL + t * (G_LARGE - G_SMALL)


def render_frame(pts, areas, L, grid):
    tree = cKDTree(np.mod(pts, L), boxsize=L)
    _, lab = tree.query(grid)
    lab = lab.reshape(HI, HI)
    img = area_to_grey(areas)[lab]
    edge = (lab != np.roll(lab, -1, 0)) | (lab != np.roll(lab, -1, 1))
    img[edge] = 0.0
    img = img.reshape(LO, 2, LO, 2).mean((1, 3))
    return np.round(img).astype(np.uint8)


def render_run(job):
    split, fname = job
    out = os.path.join(OUT, split, fname.replace(".npz", ".npy"))
    d = np.load(os.path.join(SIM, split, fname))
    L = float(d["L"])
    c = (np.arange(HI) + 0.5) * L / HI
    gx, gy = np.meshgrid(c, c)           # row = y, col = x
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    meta = dict(split=split, run=fname[:-4], Ka=float(d["Ka"]),
                alpha=float(d["alpha"]), seed=int(d["seed"]),
                T=int(d["areas"].shape[0]))
    if os.path.exists(out):
        return meta
    pts, areas = d["pts"], d["areas"]
    stack = np.empty((pts.shape[0], LO, LO), np.uint8)
    for t in range(pts.shape[0]):
        stack[t] = render_frame(pts[t], areas[t], L, grid)
    np.save(out + ".tmp.npy", stack)
    os.replace(out + ".tmp.npy", out)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="debug: runs per split")
    args = ap.parse_args()
    jobs = []
    for s in SPLITS:
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
        files = sorted(f for f in os.listdir(os.path.join(SIM, s)) if f.endswith(".npz"))
        if args.limit:
            files = files[: args.limit]
        jobs += [(s, f) for f in files]
    t0, rows = time.time(), []
    with Pool(args.workers) as p:
        for i, m in enumerate(p.imap_unordered(render_run, jobs), 1):
            rows.append(m)
            print(f"[{i}/{len(jobs)}] {m['split']}/{m['run']}  {time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["run"]))
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("done", len(rows), "runs")


if __name__ == "__main__":
    sys.exit(main())
