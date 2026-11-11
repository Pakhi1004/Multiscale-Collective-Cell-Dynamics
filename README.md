# Multiscale Collective Cell Dynamics — physics-guided ML

B.Tech project reproducing the physics-guided part of *Machine learning multiscale collective cell dynamics: From single-cell characterization to multicellular monolayer modeling* (Xu et al., PNAS 2026).

A convolutional transformer (CvT) is trained on vertex-model simulations, whose parameters are known exactly, to infer the area stiffness **Ka** (and later the mechanochemical coupling **α**) from short image sequences of a cell monolayer.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | *(vertex-model simulator, run separately)* | ERK-feedback vertex model, N=300 cells, 1500 frames (~5 min apart) per run → `sim_data/<split>/*.npz` |
| 2 | `src/render.py` | Periodic Voronoi of cell centres, filled greyscale-by-area (one fixed range), 128×128 → `data/frames/<split>/*.npy` |
| 3 | `src/dataset.py`, `src/check_dataset.py` | 8-frame windows, split by seed, augmentation (shift/flip/rotate, 96×96 crop, small brightness change) |
| 4 | `src/model.py`, `src/train.py` | CvT: per-frame CNN → time embedding → 3 attention layers (2/4/6 heads) → Ka |
| 5 | `src/evaluate.py` | Per-window and per-run (averaged) R² |

## Data splits

| Split | Ka | α | Runs |
|---|---|---|---|
| train | 0.1–0.5 | 60–100 | seeds 0–3 (100 runs) |
| val | 0.1–0.5 | 60–100 | seed 4 (25 runs) |
| test | 0.15–0.45 | 65–95 | 32 runs (values between the training grid) |
| extrap | 0.05, 0.60 | 80 | 4 runs |
| stability | 0.3 | 80 | 1 run, 231 windows |

The data (~5 GB) is not stored in this repo. Put the simulator output in `sim_data/`, then run `python src/render.py` to build `data/`.

## Usage

```bash
python src/render.py                       # NPZ -> image stacks
python src/check_dataset.py                # split, leakage and batch checks + figures
python src/train.py --target Ka --epochs 100 --skip 4 --lr 3e-4 --tag cvt_Ka_skip4_lr3e-4_100ep
python src/evaluate.py runs/cvt_Ka_skip4_lr3e-4_100ep
```

## Results so far (Ka-only model, validation seed)

| Model | Per-window R² | Per-run R² |
|---|---|---|
| Consecutive frames, lr 1e-4, 100 epochs | 0.837 | 0.933 |
| **Every 4th frame, lr 3e-4, 100 epochs** | **0.892** | **0.957** |

Ka values 0.1–0.4 are recovered within 0.01 on average. Ka = 0.5 is under-predicted (0.45), because area spread barely changes between Ka 0.4 and 0.5 in these simulations.

Changes from the paper's recipe: no strong contrast/gamma augmentation (in this data it erases the Ka signal), crop without resizing (keeps train and eval images at the same pixel scale), and a window spacing of 4 frames (8 frames cover ~2.5 h).
