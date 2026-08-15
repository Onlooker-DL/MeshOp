#!/usr/bin/env python3
"""Compute full score metrics from a saved predictions.mat (no re-training).

Reads pred_score / target_score_test and writes (merges) final_metrics.json
with the same metric definitions as the Burgers training script:
mae, rmse, rounded_accuracy, within_0p5, within_1, under/over rates,
ceil_level_mae, and the final train/test MSE losses.

Usage:
    python experiments/cd_disk/operator/compute_metrics_from_predictions.py \
        --predictions result/operators/cd_disk/fno/cd_b3000_mse/predictions.mat \
        --train-samples 3000 --test-samples 100 \
        --threshold 0.5 --max-score 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHOD_TEXT = {
    "fno": "FNO2d continuous refinement-score regression (CD disk)",
    "cno": "CNO refinement-score regression (CD disk)",
    "deeponet": "DeepONet refinement-score regression (CD disk)",
    "pod_deeponet": "POD-DeepONet refinement-score regression (CD disk)",
}


def load_predictions(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        pred_data = np.asarray(f["pred_score"][:])
        target_data = np.asarray(f["target_score_test"][:])
        r = np.asarray(f["query_r"][:]).ravel()
        th = np.asarray(f["query_theta"][:]).ravel()
        nr, nth = int(r.size), int(th.size)
        if pred_data.ndim == 3 and pred_data.shape[0] == nr and pred_data.shape[1] == nth:
            pred = pred_data.astype(np.float32)          # already (nr,nth,N)
            target = target_data.astype(np.float32)
        else:
            pred = np.transpose(pred_data, (2, 1, 0)).astype(np.float32)  # (N,nth,nr) -> (nr,nth,N)
            target = np.transpose(target_data, (2, 1, 0)).astype(np.float32)
        max_score = float(np.asarray(f["max_score"][:]).ravel()[0])
    return pred, target, max_score


def generation(s: np.ndarray, max_score: float, threshold: float) -> np.ndarray:
    s = np.clip(s, 0.0, max_score)
    return np.clip(np.floor(s + 1.0 - threshold), 0, int(max_score))


def score_metrics(pred: np.ndarray, target: np.ndarray,
                  max_score: float, threshold: float) -> dict:
    p = np.clip(pred, 0.0, max_score)
    t = target
    d = p - t
    gp = generation(p, max_score, threshold)
    gt = generation(t, max_score, threshold)
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "rounded_accuracy": float(np.mean(gp == gt)),
        "within_0p5": float(np.mean(np.abs(d) <= 0.5)),
        "within_1": float(np.mean(np.abs(d) <= 1.0)),
        "under_refinement_rate": float(np.mean(gp < gt)),
        "over_refinement_rate": float(np.mean(gp > gt)),
        "ceil_level_mae": float(np.mean(np.abs(gp - gt))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute metrics from predictions.mat")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--train-samples", type=int, required=True)
    ap.add_argument("--test-samples", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-score", type=int, default=12)
    args = ap.parse_args()

    pred, target, max_score = load_predictions(args.predictions)
    N = pred.shape[2]
    # New test-only exports contain exactly the held-out test set, so there
    # are no training samples in predictions.mat.  Legacy full exports
    # (train+test) are still supported.
    if args.train_samples + args.test_samples <= N:
        has_train = True
        train_idx = np.arange(args.train_samples)
        test_idx = np.arange(N - args.test_samples, N)
    else:
        has_train = False
        train_idx = np.arange(0)
        test_idx = np.arange(N)

    if has_train:
        train_loss = float(np.mean((pred[..., train_idx] - target[..., train_idx]) ** 2))
    else:
        train_loss = None
    test_loss = float(np.mean((pred[..., test_idx] - target[..., test_idx]) ** 2))

    # E_score: mean over test samples of ||pred - target||_2 / ||target||_2
    p_test = np.clip(pred[..., test_idx], 0.0, max_score).reshape(len(test_idx), -1)
    t_test = target[..., test_idx].reshape(len(test_idx), -1)
    denom = np.maximum(np.linalg.norm(t_test, axis=1), 1e-12)
    rel_per_sample = np.linalg.norm(p_test - t_test, axis=1) / denom
    mean_relative_score_error = float(np.mean(rel_per_sample))
    if has_train:
        train_metrics = score_metrics(pred[..., train_idx], target[..., train_idx],
                                      max_score, args.threshold)
    else:
        train_metrics = None
    test_metrics = score_metrics(pred[..., test_idx], target[..., test_idx],
                                 max_score, args.threshold)

    out_dir = args.predictions.parent
    metrics_file = out_dir / "final_metrics.json"
    merged = {}
    if metrics_file.exists():
        merged = json.loads(metrics_file.read_text(encoding="utf-8"))
    problem = str(merged.get("problem", "cd_disk"))
    name = str(merged.get("model", "fno"))
    merged.update({
        "code_version": f"{name.upper()}_CD_DISK_V1",
        "method": METHOD_TEXT.get(name, f"{name} refinement-score regression (CD disk)"),
        "problem": problem,
        "model": name,
        "training_samples": args.train_samples,
        "test_samples": args.test_samples,
        "final_training_loss": train_loss,
        "final_training_metrics": train_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "mean_relative_score_error": mean_relative_score_error,
        "relative_score_error_per_sample": [float(x) for x in rel_per_sample],
        "metrics_source": "predictions.mat (recomputed, no re-training)",
    })
    # Keep timing/storage untouched if already present; mark unavailable otherwise.
    if "timing" not in merged:
        merged["timing"] = {
            "available": False,
            "note": "requires model weights; measure on the next training run",
        }
    if "parameter_storage_mib" not in merged:
        merged["parameter_storage_mib"] = None

    metrics_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged, indent=2))
    print(f"wrote {metrics_file}")


if __name__ == "__main__":
    main()