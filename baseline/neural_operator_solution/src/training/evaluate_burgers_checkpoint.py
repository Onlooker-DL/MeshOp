#!/usr/bin/env python3
"""Evaluate a trained Burgers direct-solution checkpoint on source 5001:5100."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.physical_l2 import (
    burgers_quadrature_weights,
    per_sample_relative_l2,
    solution_metrics,
)
from src.models import DeepONet, PODCoefficientNet
from src.training.train_burgers import (
    GridDataset,
    ScalarNormalizer,
    build_grid_model,
    load_data,
    make_loader,
    normalized_coordinates,
    parameter_counts,
    seed_everything,
)


TRAIN_ROWS = np.arange(0, 3000, dtype=np.int64)
TEST_ROWS = np.arange(3000, 3100, dtype=np.int64)
EXPECTED_SOURCE_INDICES = np.concatenate(
    (
        np.arange(1, 3001, dtype=np.int64),
        np.arange(5001, 5101, dtype=np.int64),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an existing Burgers final_model.pt and evaluate only the "
            "100 source samples 5001:5100. No optimizer step is performed."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pod-basis", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.timing_repeats < 1 or args.warmup_repeats < 0:
        raise ValueError("Timing repeats must be positive and warmup nonnegative.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative.")
    return args


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=device)
    if not isinstance(loaded, dict) or "model_state_dict" not in loaded:
        raise ValueError(f"Invalid final checkpoint: {path}")
    return loaded


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def validate_source_mapping(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "source_dataset_index" not in handle:
            raise KeyError(
                "The HDF5 file has no source_dataset_index. Run the new "
                "generator or replacement script before checkpoint evaluation."
            )
        indices = np.asarray(
            handle["source_dataset_index"], dtype=np.int64
        ).reshape(-1)
    if not np.array_equal(indices, EXPECTED_SOURCE_INDICES):
        raise ValueError(
            "Unexpected source mapping. Expected HDF5 rows 1:3000 -> source "
            "1:3000 and rows 3001:3100 -> source 5001:5100."
        )
    return indices


def checkpoint_normalizer(
    checkpoint: dict[str, Any], name: str
) -> ScalarNormalizer:
    try:
        values = checkpoint["normalization"][name]
        return ScalarNormalizer(float(values["mean"]), float(values["std"]))
    except (KeyError, TypeError, ValueError) as error:
        raise KeyError(
            f"Checkpoint is missing the saved {name} normalization."
        ) from error


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_prediction(
    function: Callable[[], np.ndarray],
    device: torch.device,
    warmup_repeats: int,
    timing_repeats: int,
) -> tuple[np.ndarray, list[float]]:
    for _ in range(warmup_repeats):
        function()
    times: list[float] = []
    prediction: np.ndarray | None = None
    for _ in range(timing_repeats):
        synchronize(device)
        start = time.perf_counter()
        prediction = function()
        synchronize(device)
        times.append(time.perf_counter() - start)
    assert prediction is not None
    return prediction, times


def grid_predictor(
    name: str,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    data: Any,
    device: torch.device,
    workers: int,
) -> tuple[torch.nn.Module, Callable[[], np.ndarray]]:
    u0_norm = checkpoint_normalizer(checkpoint, "initial_condition")
    forcing_norm = checkpoint_normalizer(checkpoint, "forcing")
    initial_sensors = data.initial_condition_sensors[TEST_ROWS]
    forcing_sensors = data.forcing_sensors[TEST_ROWS]
    grid_initial = np.stack(
        [np.interp(data.query_x, data.sensor_x, row) for row in initial_sensors]
    ).astype(np.float32)
    grid_forcing = np.stack(
        [np.interp(data.query_x, data.sensor_x, row) for row in forcing_sensors]
    ).astype(np.float32)
    initial = u0_norm.encode(grid_initial)
    forcing = forcing_norm.encode(grid_forcing)
    xn, tn, _ = normalized_coordinates(data.query_x, data.query_t)
    grid_xn = xn
    if name == "fno":
        grid_xn = (
            2.0 * (data.query_x - float(data.query_x.min())) / 2.0 - 1.0
        ).astype(np.float32)
    x_channel = np.broadcast_to(
        grid_xn[:, None], (grid_xn.size, tn.size)
    ).astype(np.float32)
    t_channel = np.broadcast_to(
        tn[None, :], (grid_xn.size, tn.size)
    ).astype(np.float32)
    target = data.solution[TEST_ROWS]
    local_rows = np.arange(TEST_ROWS.size, dtype=np.int64)
    dataset = GridDataset(
        initial, forcing, target, local_rows, x_channel, t_channel
    )
    batch_size = int(
        config["training"].get("inference_sample_batch_size", 8)
    )
    loader = make_loader(
        dataset,
        batch_size,
        False,
        workers,
        int(config.get("seed", 42)) + 2,
        device,
    )
    model_config = dict(
        checkpoint.get(
            "model_config",
            {k: v for k, v in config["model"].items() if k != "name"},
        )
    )
    model = build_grid_model(name, model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    def predict() -> np.ndarray:
        blocks: list[np.ndarray] = []
        with torch.inference_mode():
            for inputs, _ in loader:
                output = model(inputs.to(device, non_blocking=True))
                blocks.append(output[:, 0].float().cpu().numpy())
        return np.concatenate(blocks, axis=0)

    return model, predict


def normalized_test_branch(
    checkpoint: dict[str, Any], data: Any
) -> np.ndarray:
    if "branch_mean" not in checkpoint or "branch_std" not in checkpoint:
        raise KeyError("Checkpoint is missing branch_mean or branch_std.")
    branch_raw = np.concatenate(
        (
            data.initial_condition_sensors[TEST_ROWS],
            data.forcing_sensors[TEST_ROWS],
        ),
        axis=1,
    ).astype(np.float32)
    mean = as_numpy(checkpoint["branch_mean"]).reshape(1, -1)
    std = as_numpy(checkpoint["branch_std"]).reshape(1, -1)
    if mean.shape[1] != branch_raw.shape[1] or std.shape != mean.shape:
        raise ValueError("Checkpoint branch normalization has the wrong shape.")
    return np.asarray((branch_raw - mean) / std, dtype=np.float32)


def deeponet_predictor(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    data: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, Callable[[], np.ndarray]]:
    branch = normalized_test_branch(checkpoint, data)
    _, _, coordinates = normalized_coordinates(data.query_x, data.query_t)
    coords = torch.from_numpy(coordinates).to(device)
    model_config = dict(
        checkpoint.get(
            "model_config",
            {k: v for k, v in config["model"].items() if k != "name"},
        )
    )
    model = DeepONet(
        branch_dim=branch.shape[1], coordinate_dim=2, **model_config
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    sample_batch = int(
        config["training"].get("inference_sample_batch_size", 8)
    )
    query_batch = int(config["training"].get("query_batch_size", 4096))

    def predict() -> np.ndarray:
        output = np.empty((branch.shape[0], coordinates.shape[0]), np.float32)
        with torch.inference_mode():
            for sample_start in range(0, branch.shape[0], sample_batch):
                sample_end = min(sample_start + sample_batch, branch.shape[0])
                branch_tensor = torch.from_numpy(
                    branch[sample_start:sample_end]
                ).to(device)
                for query_start in range(0, coordinates.shape[0], query_batch):
                    query_end = min(query_start + query_batch, coordinates.shape[0])
                    pred = model(branch_tensor, coords[query_start:query_end])
                    output[
                        sample_start:sample_end, query_start:query_end
                    ] = pred.float().cpu().numpy()
        return output.reshape(
            branch.shape[0], data.query_x.size, data.query_t.size
        )

    return model, predict


def pod_predictor(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    data: Any,
    basis_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Callable[[], np.ndarray]]:
    branch = normalized_test_branch(checkpoint, data)
    if not basis_path.is_file():
        raise FileNotFoundError(f"POD basis not found: {basis_path}")
    with np.load(basis_path) as saved:
        field_mean = np.asarray(saved["field_mean"], dtype=np.float32)
        basis = np.asarray(saved["basis"], dtype=np.float32)
    model_config = dict(
        checkpoint.get(
            "model_config",
            {k: v for k, v in config["model"].items() if k != "name"},
        )
    )
    model = PODCoefficientNet(
        branch_dim=branch.shape[1], rank=basis.shape[0], **model_config
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    def predict() -> np.ndarray:
        with torch.inference_mode():
            coefficients = model(
                torch.from_numpy(branch).to(device)
            ).float().cpu().numpy()
        fields = coefficients @ basis + field_mean[None, :]
        return fields.reshape(
            branch.shape[0], data.query_x.size, data.query_t.size
        )

    return model, predict


def per_sample_metrics(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> list[dict[str, float]]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    physical_relative, discrete_relative = per_sample_relative_l2(
        prediction, target, weights
    )
    rows: list[dict[str, float]] = []
    for local_id in range(prediction.shape[0]):
        diff = difference[local_id]
        truth = target[local_id].astype(np.float64)
        rows.append(
            {
                "test_id": local_id + 1,
                "source_dataset_index": local_id + 5001,
                "mse": float(np.mean(np.square(diff))),
                "rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "mae": float(np.mean(np.abs(diff))),
                "relative_l2": float(physical_relative[local_id]),
                "relative_discrete_l2": float(discrete_relative[local_id]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    name = str(config["model"]["name"]).lower()
    if name not in {"fno", "cno", "deeponet", "pod_deeponet"}:
        raise ValueError(f"Unsupported Burgers model: {name}")

    seed_everything(
        int(config.get("seed", 42)), bool(config.get("deterministic", False))
    )
    device = torch.device(args.device or config["training"].get("device", "cuda:0"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.set_device(device)

    data_path = project_path(args.data or Path(config["data"]["path"]))
    source_indices = validate_source_mapping(data_path)
    data = load_data(data_path)
    experiment = str(config["experiment"])
    trained_dir = (
        PROJECT_ROOT / "results" / "burgers" / name / experiment
    ).resolve()
    checkpoint_path = project_path(args.checkpoint) if args.checkpoint else (
        trained_dir / "final_model.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = safe_torch_load(checkpoint_path, device)
    checkpoint_name = str(checkpoint.get("model_name", name)).lower()
    if checkpoint_name != name:
        raise ValueError(
            f"Config model is {name}, but checkpoint model is {checkpoint_name}."
        )

    if name in {"fno", "cno"}:
        model, predictor = grid_predictor(
            name, checkpoint, config, data, device, args.num_workers
        )
    elif name == "deeponet":
        model, predictor = deeponet_predictor(
            checkpoint, config, data, device
        )
    else:
        basis_path = project_path(args.pod_basis) if args.pod_basis else (
            trained_dir / "pod_basis.npz"
        )
        model, predictor = pod_predictor(
            checkpoint, config, data, basis_path, device
        )

    prediction, inference_times = timed_prediction(
        predictor,
        device,
        args.warmup_repeats,
        args.timing_repeats,
    )
    target = data.solution[TEST_ROWS]
    if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
        raise ValueError(
            f"Invalid prediction shape/values: {prediction.shape}, expected {target.shape}."
        )
    quadrature_weights = burgers_quadrature_weights(data.query_x, data.query_t)
    metrics = solution_metrics(prediction, target, quadrature_weights)
    rows = per_sample_metrics(prediction, target, quadrature_weights)
    for row, source_id in zip(rows, data.source_ids[TEST_ROWS], strict=True):
        row["source_attempt_id"] = int(source_id)

    output_dir = project_path(args.output_dir) if args.output_dir else (
        trained_dir / "evaluation_test5001_5100"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_time = float(np.mean(inference_times))
    report = {
        "problem": "burgers_direct_solution",
        "evaluation_only": True,
        "model": name,
        "experiment": experiment,
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "training_samples": 3000,
        "test_samples": 100,
        "train_source_range_matlab": [1, 3000],
        "test_source_range_matlab": [5001, 5100],
        "source_dataset_indices_verified": bool(
            np.array_equal(source_indices, EXPECTED_SOURCE_INDICES)
        ),
        "parameter_count": parameter_counts(model),
        "warmup_repeats": args.warmup_repeats,
        "timing_repeats": args.timing_repeats,
        "inference_times_sec": inference_times,
        "mean_test_inference_time_sec": mean_time,
        "mean_inference_time_sec_per_sample": mean_time / 100.0,
        "inference_throughput_samples_per_sec": 100.0 / mean_time,
        "test_metrics": metrics,
        "relative_l2_evaluation": {
            "definition": "mean of per-sample physical-domain relative L2 errors",
            "quadrature": "periodic trapezoidal x tensor nonperiodic trapezoidal t",
            "legacy_metric": "test_metrics.mean_relative_discrete_l2",
        },
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "per_sample_test_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with h5py.File(output_dir / "predictions.h5", "w") as handle:
        handle.create_dataset("prediction", data=prediction, compression="lzf")
        handle.create_dataset("target", data=target, compression="lzf")
        handle.create_dataset("query_x", data=data.query_x)
        handle.create_dataset("query_t", data=data.query_t)
        handle.create_dataset(
            "source_dataset_index", data=source_indices[TEST_ROWS]
        )
        handle.create_dataset(
            "source_attempt_id", data=data.source_ids[TEST_ROWS]
        )
        handle.attrs["mean_test_inference_time_sec"] = mean_time
        handle.attrs["relative_l2_definition"] = (
            "mean of per-sample physical-domain quadrature relative L2 errors"
        )
        for key, value in metrics.items():
            handle.attrs[key] = value
    print(json.dumps(report, indent=2), flush=True)
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

