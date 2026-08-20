#!/usr/bin/env python3
"""Recompute physical relative L2 metrics from an existing predictions.h5."""

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
    cd_disk_quadrature_weights,
    reaction_diffusion_quadrature_weights,
    solution_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved Direct predictions with physical L2 quadrature."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_and_evaluate(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if "prediction" not in handle or "target" not in handle:
            raise KeyError("predictions.h5 must contain prediction and target datasets.")
        prediction = np.asarray(handle["prediction"], dtype=np.float32)
        target = np.asarray(handle["target"], dtype=np.float32)
        keys = set(handle.keys())
        if {"query_x", "query_t"}.issubset(keys):
            problem = "burgers"
            weights = burgers_quadrature_weights(
                np.asarray(handle["query_x"]), np.asarray(handle["query_t"])
            )
            quadrature = "periodic trapezoidal x tensor nonperiodic trapezoidal t"
        elif {"r", "theta"}.issubset(keys):
            problem = "cd_disk"
            weights = cd_disk_quadrature_weights(
                np.asarray(handle["r"]), np.asarray(handle["theta"])
            )
            quadrature = "polar control-volume r dr tensor periodic trapezoidal dtheta"
        elif {"x", "y", "z"}.issubset(keys):
            problem = "reaction_diffusion"
            weights = reaction_diffusion_quadrature_weights(
                np.asarray(handle["x"]),
                np.asarray(handle["y"]),
                np.asarray(handle["z"]),
            )
            quadrature = "tensor-product trapezoidal dx dy dz on the stored nonuniform z grid"
        else:
            raise KeyError(
                "Could not infer the problem from coordinate datasets in predictions.h5."
            )
        source_key = next(
            (name for name in ("source_dataset_index", "source_sample_id", "source_attempt_id") if name in handle),
            None,
        )
        source_ids = (
            np.asarray(handle[source_key], dtype=np.int64).reshape(-1)
            if source_key is not None
            else None
        )

    result: dict[str, object] = {
        "problem": problem,
        "prediction_file": str(path.resolve()),
        "samples": int(prediction.shape[0]),
        "test_metrics": solution_metrics(prediction, target, weights),
        "relative_l2_evaluation": {
            "definition": "mean of per-sample physical-domain relative L2 errors",
            "quadrature": quadrature,
            "legacy_metric": "test_metrics.mean_relative_discrete_l2",
        },
    }
    if source_ids is not None:
        result["source_id_dataset"] = source_key
        result["source_id_min"] = int(source_ids.min())
        result["source_id_max"] = int(source_ids.max())
    return result


def main() -> None:
    args = parse_args()
    prediction_path = args.predictions.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    result = load_and_evaluate(prediction_path)
    output = (
        args.output.resolve()
        if args.output is not None
        else prediction_path.parent / "physical_l2_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

