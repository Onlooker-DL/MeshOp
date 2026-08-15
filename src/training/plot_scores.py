#!/usr/bin/env python3
"""Operator-side comparison plot: label (target) vs prediction.

Reads predictions.mat (pred_score / target_score_test on the 64x128 polar
grid) and draws, for each requested test sample, a 1x3 figure on the disk:
    [ Label (target score) ] [ Predicted score ] [ |Pred - Label| ]

Usage:
    python experiments/cd_disk/operator/plot_scores.py \
        --predictions result/operators/cd_disk/fno/cd_b3000_mse/predictions.mat \
        [--samples 1 4 69] [--out figures/cd_disk/operator]

Requires numpy, scipy, h5py, matplotlib (all in requirements.txt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_predictions(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        pred_data = np.asarray(f["pred_score"][:])
        target_data = np.asarray(f["target_score_test"][:])
        r = np.asarray(f["query_r"][:]).ravel().astype(np.float32)
        th = np.asarray(f["query_theta"][:]).ravel().astype(np.float32)
        nr, nth = int(r.size), int(th.size)
        if pred_data.ndim == 3 and pred_data.shape[0] == nr and pred_data.shape[1] == nth:
            pred = pred_data.astype(np.float32)          # already (nr,nth,N)
            target = target_data.astype(np.float32)
        else:
            pred = np.transpose(pred_data, (2, 1, 0)).astype(np.float32)  # (N,nth,nr) -> (nr,nth,N)
            target = np.transpose(target_data, (2, 1, 0)).astype(np.float32)
        src = np.asarray(f["source_attempt_id_test"][:]).ravel()
        max_score = float(np.asarray(f["max_score"][:]).ravel()[0])
    return pred, target, r, th, src, max_score


def polar_to_cart(r, th, field, n=260):
    """Interpolate an (nr, nth) polar field onto an n x n Cartesian disk."""
    from scipy.interpolate import griddata

    RR, TT = np.meshgrid(r, th, indexing="ij")
    xp = (RR * np.cos(TT)).ravel()
    yp = (RR * np.sin(TT)).ravel()
    vals = np.asarray(field, dtype=np.float64).ravel()
    g = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(g, g)
    inside = (X**2 + Y**2) <= 1.0
    F = np.full(X.shape, np.nan)
    if np.any(inside):
        F[inside] = griddata(
            (xp, yp), vals, (X[inside], Y[inside]), method="linear"
        )
        # nearest fallback for any remaining holes (e.g. near the boundary)
        still_nan = np.isnan(F) & inside
        if np.any(still_nan):
            F[still_nan] = griddata(
                (xp, yp), vals, (X[still_nan], Y[still_nan]), method="nearest"
            )
    return X, Y, F


def plot_sample(pred, target, r, th, src, max_score, k, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tgt = target[:, :, k]
    pre = np.clip(pred[:, :, k], 0.0, max_score)
    X, Y, Ft = polar_to_cart(r, th, tgt)
    _, _, Fp = polar_to_cart(r, th, pre)
    _, _, Fd = polar_to_cart(r, th, np.abs(pre - tgt))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    im = axes[0].imshow(
        Ft, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis",
        vmin=0.0, vmax=max_score,
    )
    axes[0].set_title("Label (target score)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].imshow(
        Fp, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis",
        vmin=0.0, vmax=max_score,
    )
    axes[1].set_title("Predicted score")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    im = axes[2].imshow(
        Fd, origin="lower", extent=(-1, 1, -1, 1), cmap="hot",
    )
    axes[2].set_title("|Pred - Label|")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle(f"test {k+1} (source {int(src[k])})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"[plot_scores] saved {out_path.with_suffix('.pdf')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Operator label-vs-prediction plot.")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--samples", type=int, nargs="*",
                    help="1-based sample indices (default: first 3 test samples)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pred, target, r, th, src, max_score = load_predictions(args.predictions)
    N = pred.shape[2]

    if args.samples:
        samples = [k - 1 for k in args.samples if 1 <= k <= N]
    else:
        samples = list(range(max(N - 100, 0), max(N - 100, 0) + min(3, N)))

    if not samples:
        raise SystemExit("no valid samples to plot")

    if args.out is None:
        # Same convention as the Burgers operator pipeline:
        # figures/operators/<problem>/<model>/<experiment>/
        exp_dir = args.predictions.parent            # .../operators/<problem>/<model>/<experiment>
        model_dir = exp_dir.parent
        problem = model_dir.parent.name
        model_name = model_dir.name
        experiment = exp_dir.name
        out_dir = ROOT / "figures" / "operators" / problem / model_name / experiment
    else:
        out_dir = args.out
        model_name = args.predictions.parent.parent.name
        experiment = args.predictions.parent.name

    for k in samples:
        out_path = out_dir / (
            f"{model_name}_{experiment}_test{k+1:03d}_scores.pdf"
        )
        plot_sample(pred, target, r, th, src, max_score, k, out_path)


if __name__ == "__main__":
    main()