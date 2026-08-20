#!/usr/bin/env python3
"""Compare the source AFEM solution_query field with the spectral reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    flat_difference = difference.reshape(difference.shape[0], -1)
    flat_target = target.astype(np.float64).reshape(target.shape[0], -1)
    relative = np.linalg.norm(flat_difference, axis=1) / np.maximum(
        np.linalg.norm(flat_target, axis=1), 1.0e-12
    )
    return {
        "mse": float(np.mean(np.square(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "mean_relative_l2": float(np.mean(relative)),
        "median_relative_l2": float(np.median(relative)),
        "maximum_relative_l2": float(np.max(relative)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--spectral",
        type=Path,
        default=Path("data/burgers/burgers_spectral_3100.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/burgers/afem_baseline/final_metrics.json"),
    )
    args = parser.parse_args()

    with h5py.File(args.spectral, "r") as handle:
        spectral = np.asarray(handle["solution"], dtype=np.float32)
        x = np.asarray(handle["query_x"], dtype=np.float32).reshape(-1)
        t = np.asarray(handle["query_t"], dtype=np.float32).reshape(-1)
        source_ids = np.asarray(
            handle["source_attempt_id"], dtype=np.int64
        ).reshape(-1)
    if spectral.shape != (3100, x.size, t.size):
        raise ValueError(f"Unexpected spectral shape {spectral.shape}.")

    with h5py.File(args.source, "r") as handle:
        if "solution_query" not in handle:
            raise KeyError("The source MAT file has no solution_query field.")
        raw = np.asarray(handle["solution_query"], dtype=np.float32)
        if raw.shape[0] < 3100 or raw.shape[1:] != (t.size, x.size):
            raise ValueError(
                "Expected MATLAB v7.3 solution_query layout [sample,t,x], "
                f"got {raw.shape}."
            )
        afem = np.transpose(raw[:3100], (0, 2, 1))

    train_slice = slice(0, 3000)
    test_slice = slice(3000, 3100)
    result = {
        "method": "AFEM final finite iterate sampled on the 101x101 query grid",
        "reference": "512-point Fourier ETDRK4 spectral solution",
        "training_range_matlab": [1, 3000],
        "test_range_matlab": [3001, 3100],
        "source_attempt_id_test_min": int(source_ids[test_slice].min()),
        "source_attempt_id_test_max": int(source_ids[test_slice].max()),
        "train_metrics": metrics(afem[train_slice], spectral[train_slice]),
        "test_metrics": metrics(afem[test_slice], spectral[test_slice]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
