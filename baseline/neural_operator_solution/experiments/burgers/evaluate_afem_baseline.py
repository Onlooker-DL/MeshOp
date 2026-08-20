#!/usr/bin/env python3
"""Compare the source AFEM solution_query field with the spectral reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.physical_l2 import (  # noqa: E402
    burgers_quadrature_weights,
    solution_metrics,
)


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
        if "source_dataset_index" in handle:
            source_indices = np.asarray(
                handle["source_dataset_index"], dtype=np.int64
            ).reshape(-1)
        else:
            source_indices = np.arange(1, spectral.shape[0] + 1)
    if spectral.shape != (3100, x.size, t.size):
        raise ValueError(f"Unexpected spectral shape {spectral.shape}.")
    if source_indices.shape != (3100,):
        raise ValueError(
            f"Unexpected source index shape {source_indices.shape}."
        )
    if np.any(source_indices < 1) or np.unique(source_indices).size != 3100:
        raise ValueError("Source dataset indices must be unique and positive.")

    with h5py.File(args.source, "r") as handle:
        if "solution_query" not in handle:
            raise KeyError("The source MAT file has no solution_query field.")
        raw = np.asarray(handle["solution_query"], dtype=np.float32)
        if raw.shape[0] < int(source_indices.max()) or raw.shape[1:] != (
            t.size,
            x.size,
        ):
            raise ValueError(
                "Expected MATLAB v7.3 solution_query layout [sample,t,x], "
                f"got {raw.shape}."
            )
        afem = np.transpose(raw[source_indices - 1], (0, 2, 1))

    train_slice = slice(0, 3000)
    test_slice = slice(3000, 3100)
    train_source_indices = source_indices[train_slice]
    test_source_indices = source_indices[test_slice]
    quadrature_weights = burgers_quadrature_weights(x, t)
    result = {
        "method": "AFEM final finite iterate sampled on the 101x101 query grid",
        "reference": "512-point Fourier ETDRK4 spectral solution",
        "training_range_matlab": [
            int(train_source_indices.min()),
            int(train_source_indices.max()),
        ],
        "test_range_matlab": [
            int(test_source_indices.min()),
            int(test_source_indices.max()),
        ],
        "source_attempt_id_test_min": int(source_ids[test_slice].min()),
        "source_attempt_id_test_max": int(source_ids[test_slice].max()),
        "relative_l2_evaluation": {
            "definition": "mean of per-sample physical-domain relative L2 errors",
            "quadrature": "periodic trapezoidal x tensor nonperiodic trapezoidal t",
            "legacy_metric": "*_metrics.mean_relative_discrete_l2",
        },
        "train_metrics": solution_metrics(
            afem[train_slice], spectral[train_slice], quadrature_weights
        ),
        "test_metrics": solution_metrics(
            afem[test_slice], spectral[test_slice], quadrature_weights
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
