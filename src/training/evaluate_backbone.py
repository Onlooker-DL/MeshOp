#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate an already-trained DeepONet, POD-DeepONet, Transolver, or CNO
checkpoint using the same score metrics reported by the Burgers FNO.

This script does not train or modify model parameters.

Expected experiment directory
-----------------------------
<out_dir>/
    resolved_config.json
    final_model.pt
    predictions.mat                 # normally already present
    pod_basis.npz                   # POD-DeepONet only
    final_metrics.json              # old report; backed up once

Example
-------
python src/training/evaluate_backbone.py \
    --out-dir result/operators/burgers/deeponet/b3000_mse \
    --device cuda:0

Outputs
-------
<out_dir>/final_metrics.json
<out_dir>/per_sample_test_metrics.csv
<out_dir>/predictions.mat

The old final_metrics.json is preserved as
final_metrics_before_unified_evaluation.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from src.data import OperatorData, load_problem_data
from src.inference.export_scores import export_predictions
from src.models import CNO, DeepONet, PODCoefficientNet, Transolver
from src.training.train_backbone import (
    _make_cno_grid_data,
    _predict_cno,
)


CODE_VERSION = "GENERIC_BACKBONE_UNIFIED_EVALUATION_V1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_torch_load(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except Exception:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_config(
    out_dir: Path,
    config_path: Path | None,
) -> dict[str, Any]:
    path = (
        config_path
        if config_path is not None
        else out_dir / "resolved_config.json"
    )
    config = _load_json(path)

    for key in (
        "problem",
        "model",
        "data",
        "training",
        "score",
    ):
        if key not in config:
            raise KeyError(
                f"Configuration is missing '{key}': {path}"
            )

    return config


def _expected_split(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    ntrain = int(data_cfg["train_samples"])
    ntest = int(data_cfg["test_samples"])
    mode = str(
        data_cfg.get("split_mode", "random_exact")
    )

    if mode == "prefix_train_tail_test":
        total = int(data_cfg["total_samples"])
        if total < ntrain + ntest:
            raise ValueError(
                "total_samples is smaller than "
                "train_samples + test_samples."
            )
        train_indices = np.arange(
            ntrain,
            dtype=np.int64,
        )
        test_indices = np.arange(
            total - ntest,
            total,
            dtype=np.int64,
        )
        return train_indices, test_indices

    if mode == "random_exact":
        rng = np.random.default_rng(
            int(config.get("seed", 900))
        )
        permutation = rng.permutation(
            ntrain + ntest
        ).astype(np.int64)
        return (
            permutation[:ntrain],
            permutation[ntrain:],
        )

    raise ValueError(
        f"Unsupported split_mode: {mode}"
    )


def _load_split(
    out_dir: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prefer the indices stored in predictions.mat because these are the
    exact indices used by the completed experiment.
    """
    prediction_path = out_dir / "predictions.mat"

    if prediction_path.exists():
        loaded = sio.loadmat(
            prediction_path,
            variable_names=[
                "train_indices",
                "test_indices",
            ],
        )
        if (
            "train_indices" in loaded
            and "test_indices" in loaded
        ):
            train_indices = (
                np.asarray(
                    loaded["train_indices"],
                    dtype=np.int64,
                ).reshape(-1)
                - 1
            )
            test_indices = (
                np.asarray(
                    loaded["test_indices"],
                    dtype=np.int64,
                ).reshape(-1)
                - 1
            )
        else:
            train_indices, test_indices = (
                _expected_split(config)
            )
    else:
        train_indices, test_indices = (
            _expected_split(config)
        )

    expected_train = int(
        config["data"]["train_samples"]
    )
    expected_test = int(
        config["data"]["test_samples"]
    )

    if train_indices.size != expected_train:
        raise ValueError(
            f"Loaded {train_indices.size} training indices, "
            f"expected {expected_train}."
        )
    if test_indices.size != expected_test:
        raise ValueError(
            f"Loaded {test_indices.size} test indices, "
            f"expected {expected_test}."
        )
    if np.intersect1d(
        train_indices,
        test_indices,
    ).size:
        raise ValueError(
            "Training and test indices overlap."
        )
    if (
        train_indices.size == 0
        or test_indices.size == 0
        or min(
            int(train_indices.min()),
            int(test_indices.min()),
        )
        < 0
    ):
        raise ValueError(
            "Split contains an invalid negative index."
        )

    return train_indices, test_indices


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _generation(
    score: np.ndarray,
    maximum: int,
    threshold: float,
) -> np.ndarray:
    """
    Match the FNO/export rule exactly:

        floor(clip(score) + 1 - threshold + 1e-6)

    For threshold=0.5 this is standard rounding.
    """
    clipped = np.clip(
        np.asarray(score),
        0.0,
        float(maximum),
    )
    return np.clip(
        np.floor(
            clipped
            + 1.0
            - float(threshold)
            + 1.0e-6
        ),
        0,
        maximum,
    ).astype(np.int16)


@dataclass
class MetricAccumulator:
    n_samples: int

    def __post_init__(self) -> None:
        self.count = 0
        self.raw_squared_sum = 0.0
        self.absolute_sum = 0.0
        self.squared_sum = 0.0
        self.rounded_correct = 0
        self.within_half = 0
        self.within_one = 0
        self.under_count = 0
        self.over_count = 0
        self.generation_absolute_sum = 0.0

        self.sample_count = np.zeros(
            self.n_samples,
            dtype=np.int64,
        )
        self.sample_absolute_sum = np.zeros(
            self.n_samples,
            dtype=np.float64,
        )
        self.sample_squared_sum = np.zeros(
            self.n_samples,
            dtype=np.float64,
        )
        self.sample_target_squared_sum = np.zeros(
            self.n_samples,
            dtype=np.float64,
        )
        self.sample_under_count = np.zeros(
            self.n_samples,
            dtype=np.int64,
        )
        self.sample_over_count = np.zeros(
            self.n_samples,
            dtype=np.int64,
        )
        self.sample_within_one_count = np.zeros(
            self.n_samples,
            dtype=np.int64,
        )

    def update(
        self,
        raw_prediction: np.ndarray,
        target: np.ndarray,
        local_sample_start: int,
        maximum_generation: int,
        generation_threshold: float,
    ) -> None:
        raw64 = np.asarray(
            raw_prediction,
            dtype=np.float64,
        )
        target64 = np.asarray(
            target,
            dtype=np.float64,
        )

        if raw64.shape != target64.shape:
            raise ValueError(
                "Prediction/target shape mismatch: "
                f"{raw64.shape} versus {target64.shape}."
            )

        clipped = np.clip(
            raw64,
            0.0,
            float(maximum_generation),
        )
        raw_difference = raw64 - target64
        difference = clipped - target64
        absolute_difference = np.abs(difference)

        prediction_generation = _generation(
            clipped,
            maximum_generation,
            generation_threshold,
        )
        target_generation = _generation(
            target64,
            maximum_generation,
            generation_threshold,
        )
        generation_difference = (
            prediction_generation
            - target_generation
        )

        number = int(difference.size)
        self.count += number
        self.raw_squared_sum += float(
            np.sum(raw_difference**2)
        )
        self.absolute_sum += float(
            np.sum(absolute_difference)
        )
        self.squared_sum += float(
            np.sum(difference**2)
        )
        self.rounded_correct += int(
            np.sum(generation_difference == 0)
        )
        self.within_half += int(
            np.sum(absolute_difference <= 0.5)
        )
        self.within_one += int(
            np.sum(absolute_difference <= 1.0)
        )
        self.under_count += int(
            np.sum(generation_difference < 0)
        )
        self.over_count += int(
            np.sum(generation_difference > 0)
        )
        self.generation_absolute_sum += float(
            np.sum(
                np.abs(generation_difference)
            )
        )

        batch_size = int(difference.shape[0])
        flattened_difference = difference.reshape(
            batch_size,
            -1,
        )
        flattened_target = target64.reshape(
            batch_size,
            -1,
        )
        flattened_generation_difference = (
            generation_difference.reshape(
                batch_size,
                -1,
            )
        )
        flattened_absolute_difference = (
            absolute_difference.reshape(
                batch_size,
                -1,
            )
        )

        destination = slice(
            local_sample_start,
            local_sample_start + batch_size,
        )

        points_in_chunk = int(
            flattened_difference.shape[1]
        )
        self.sample_count[destination] += (
            points_in_chunk
        )
        self.sample_absolute_sum[destination] += (
            np.sum(
                flattened_absolute_difference,
                axis=1,
            )
        )
        self.sample_squared_sum[destination] += (
            np.sum(
                flattened_difference**2,
                axis=1,
            )
        )
        self.sample_target_squared_sum[
            destination
        ] += np.sum(
            flattened_target**2,
            axis=1,
        )
        self.sample_under_count[destination] += (
            np.sum(
                flattened_generation_difference < 0,
                axis=1,
            )
        )
        self.sample_over_count[destination] += (
            np.sum(
                flattened_generation_difference > 0,
                axis=1,
            )
        )
        self.sample_within_one_count[
            destination
        ] += np.sum(
            flattened_absolute_difference <= 1.0,
            axis=1,
        )

    def finalize(
        self,
    ) -> tuple[
        float,
        dict[str, float],
        dict[str, np.ndarray],
    ]:
        denominator = max(self.count, 1)

        loss = (
            self.raw_squared_sum / denominator
        )

        metrics = {
            "mae": (
                self.absolute_sum / denominator
            ),
            "rmse": math.sqrt(
                self.squared_sum / denominator
            ),
            "rounded_accuracy": (
                self.rounded_correct / denominator
            ),
            "within_0p5": (
                self.within_half / denominator
            ),
            "within_1": (
                self.within_one / denominator
            ),
            "under_refinement_rate": (
                self.under_count / denominator
            ),
            "over_refinement_rate": (
                self.over_count / denominator
            ),
            "ceil_level_mae": (
                self.generation_absolute_sum
                / denominator
            ),
        }

        sample_denominator = np.maximum(
            self.sample_count,
            1,
        ).astype(np.float64)

        per_sample = {
            "mae": (
                self.sample_absolute_sum
                / sample_denominator
            ),
            "rmse": np.sqrt(
                self.sample_squared_sum
                / sample_denominator
            ),
            "relative_l2": np.sqrt(
                self.sample_squared_sum
            )
            / np.maximum(
                np.sqrt(
                    self.sample_target_squared_sum
                ),
                1.0e-12,
            ),
            "under_rate": (
                self.sample_under_count
                / sample_denominator
            ),
            "over_rate": (
                self.sample_over_count
                / sample_denominator
            ),
            "within_one": (
                self.sample_within_one_count
                / sample_denominator
            ),
        }

        return (
            float(loss),
            {
                key: float(value)
                for key, value in metrics.items()
            },
            per_sample,
        )


def _build_model(
    name: str,
    checkpoint: dict[str, Any],
    data: OperatorData,
    out_dir: Path,
    device: torch.device,
) -> tuple[
    nn.Module,
    dict[str, np.ndarray] | None,
]:
    model_cfg = dict(
        checkpoint["model_config"]
    )

    if name == "deeponet":
        branch_mean = np.asarray(
            checkpoint["branch_mean"],
            dtype=np.float32,
        ).reshape(-1)

        model = DeepONet(
            branch_dim=int(branch_mean.size),
            coordinate_dim=int(
                data.coordinates.shape[1]
            ),
            latent_dim=int(
                model_cfg.get("latent_dim", 256)
            ),
            hidden_dim=int(
                model_cfg.get("hidden_dim", 256)
            ),
            branch_depth=int(
                model_cfg.get("branch_depth", 4)
            ),
            trunk_depth=int(
                model_cfg.get("trunk_depth", 4)
            ),
            activation=str(
                model_cfg.get("activation", "gelu")
            ),
            dropout=float(
                model_cfg.get("dropout", 0.0)
            ),
        ).to(device)

        auxiliary = None

    elif name == "transolver":
        branch_mean = np.asarray(
            checkpoint["branch_mean"],
            dtype=np.float32,
        ).reshape(-1)

        model = Transolver(
            branch_dim=int(branch_mean.size),
            coordinate_dim=int(
                data.coordinates.shape[1]
            ),
            width=int(
                model_cfg.get("width", 128)
            ),
            layers=int(
                model_cfg.get("layers", 4)
            ),
            heads=int(
                model_cfg.get("heads", 8)
            ),
            slices=int(
                model_cfg.get("slices", 32)
            ),
            dropout=float(
                model_cfg.get("dropout", 0.0)
            ),
        ).to(device)

        auxiliary = None

    elif name == "cno":
        model = CNO(
            dim=int(model_cfg.get("dim", 2)),
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            width=int(model_cfg.get("width", 16)),
            levels=int(model_cfg.get("levels", 3)),
            n_res_bottleneck=int(
                model_cfg.get("n_res_bottleneck", 4)
            ),
            n_res_intermediate=int(
                model_cfg.get("n_res_intermediate", 2)
            ),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            activation=str(
                model_cfg.get("activation", "gelu")
            ),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)

        auxiliary = None

    elif name == "pod_deeponet":
        pod_path = out_dir / "pod_basis.npz"
        if not pod_path.exists():
            raise FileNotFoundError(
                f"POD basis not found: {pod_path}"
            )

        pod = np.load(pod_path)
        basis = np.ascontiguousarray(
            pod["basis"],
            dtype=np.float32,
        )
        mean = np.ascontiguousarray(
            pod["mean"],
            dtype=np.float32,
        ).reshape(-1)

        branch_mean = np.asarray(
            checkpoint["branch_mean"],
            dtype=np.float32,
        ).reshape(-1)

        rank = int(
            checkpoint.get(
                "pod_rank",
                basis.shape[0],
            )
        )
        if basis.shape[0] != rank:
            raise ValueError(
                "POD rank in checkpoint and basis file "
                "do not match."
            )

        model = PODCoefficientNet(
            branch_dim=int(branch_mean.size),
            rank=rank,
            hidden_dim=int(
                model_cfg.get("hidden_dim", 256)
            ),
            depth=int(
                model_cfg.get("depth", 4)
            ),
            activation=str(
                model_cfg.get("activation", "gelu")
            ),
            dropout=float(
                model_cfg.get("dropout", 0.0)
            ),
        ).to(device)

        auxiliary = {
            "basis": basis,
            "mean": mean,
        }

    else:
        raise ValueError(
            f"Unsupported model: {name}"
        )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint.get("model_state_dict"),
    )
    if state_dict is None:
        raise KeyError(
            "Checkpoint contains neither state_dict "
            "nor model_state_dict."
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model, auxiliary


def _normalization(
    checkpoint: dict[str, Any],
    data: OperatorData,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    torch.Tensor,
]:
    checkpoint_name = str(
        checkpoint.get("model_name", "")
    )
    if checkpoint_name == "cno":
        pair_mean = np.asarray(
            checkpoint["branch_mean"],
            dtype=np.float32,
        ).reshape(-1)
        pair_std = np.asarray(
            checkpoint["branch_std"],
            dtype=np.float32,
        ).reshape(-1)
        if pair_mean.size < 1 or pair_std.size < 1:
            raise ValueError(
                "CNO checkpoint must store normalization scalars."
            )
        return pair_mean, pair_std, None

    branch_mean = np.asarray(
        checkpoint["branch_mean"],
        dtype=np.float32,
    ).reshape(-1)
    branch_std = np.asarray(
        checkpoint["branch_std"],
        dtype=np.float32,
    ).reshape(-1)
    branch_std = np.maximum(
        branch_std,
        1.0e-6,
    )

    if (
        branch_mean.size
        != data.branch.shape[1]
        or branch_std.size
        != data.branch.shape[1]
    ):
        raise ValueError(
            "Checkpoint branch normalization does not "
            "match the current data."
        )

    if (
        "coordinate_min" in checkpoint
        and "coordinate_max" in checkpoint
    ):
        coordinate_min = np.asarray(
            checkpoint["coordinate_min"],
            dtype=np.float32,
        ).reshape(-1)
        coordinate_max = np.asarray(
            checkpoint["coordinate_max"],
            dtype=np.float32,
        ).reshape(-1)
    else:
        coordinate_min = np.asarray(
            data.coordinates.min(axis=0),
            dtype=np.float32,
        )
        coordinate_max = np.asarray(
            data.coordinates.max(axis=0),
            dtype=np.float32,
        )

    coordinate_scale = np.maximum(
        coordinate_max - coordinate_min,
        1.0e-8,
    )
    normalized_coordinates = (
        2.0
        * (
            np.asarray(
                data.coordinates,
                dtype=np.float32,
            )
            - coordinate_min
        )
        / coordinate_scale
        - 1.0
    )

    coordinate_tensor = torch.from_numpy(
        np.ascontiguousarray(
            normalized_coordinates,
            dtype=np.float32,
        )
    ).to(device)

    return (
        branch_mean,
        branch_std,
        coordinate_tensor,
    )


@torch.inference_mode()
def _evaluate_point_model(
    model: nn.Module,
    data: OperatorData,
    indices: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    coordinates: torch.Tensor,
    device: torch.device,
    sample_batch_size: int,
    query_batch_size: int,
    maximum_generation: int,
    generation_threshold: float,
    collect_predictions: bool,
) -> tuple[
    float,
    dict[str, float],
    dict[str, np.ndarray],
    np.ndarray | None,
]:
    number_of_samples = int(indices.size)
    number_of_points = int(
        np.prod(data.grid_shape)
    )

    prediction_flat = (
        np.empty(
            (
                number_of_samples,
                number_of_points,
            ),
            dtype=np.float32,
        )
        if collect_predictions
        else None
    )

    accumulator = MetricAccumulator(
        number_of_samples
    )

    for local_start in range(
        0,
        number_of_samples,
        sample_batch_size,
    ):
        local_stop = min(
            local_start + sample_batch_size,
            number_of_samples,
        )
        sample_ids = indices[
            local_start:local_stop
        ]

        normalized_branch = (
            np.asarray(
                data.branch[sample_ids],
                dtype=np.float32,
            )
            - branch_mean
        ) / branch_std

        branch_tensor = torch.from_numpy(
            np.ascontiguousarray(
                normalized_branch,
                dtype=np.float32,
            )
        ).to(device)

        target_batch = np.asarray(
            data.target[sample_ids],
            dtype=np.float32,
        ).reshape(
            sample_ids.size,
            number_of_points,
        )

        for query_start in range(
            0,
            number_of_points,
            query_batch_size,
        ):
            query_stop = min(
                query_start + query_batch_size,
                number_of_points,
            )

            raw = model(
                branch_tensor,
                coordinates[
                    query_start:query_stop
                ],
            ).float().cpu().numpy()

            accumulator.update(
                raw_prediction=raw,
                target=target_batch[
                    :,
                    query_start:query_stop,
                ],
                local_sample_start=local_start,
                maximum_generation=(
                    maximum_generation
                ),
                generation_threshold=(
                    generation_threshold
                ),
            )

            if prediction_flat is not None:
                prediction_flat[
                    local_start:local_stop,
                    query_start:query_stop,
                ] = np.clip(
                    raw,
                    0.0,
                    float(maximum_generation),
                )

    (
        loss,
        metrics,
        per_sample,
    ) = accumulator.finalize()

    prediction = (
        prediction_flat.reshape(
            (
                number_of_samples,
                *data.grid_shape,
            )
        )
        if prediction_flat is not None
        else None
    )

    return (
        loss,
        metrics,
        per_sample,
        prediction,
    )


@torch.inference_mode()
def _evaluate_pod_model(
    model: nn.Module,
    auxiliary: dict[str, np.ndarray],
    data: OperatorData,
    indices: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    device: torch.device,
    sample_batch_size: int,
    query_batch_size: int,
    maximum_generation: int,
    generation_threshold: float,
    collect_predictions: bool,
) -> tuple[
    float,
    dict[str, float],
    dict[str, np.ndarray],
    np.ndarray | None,
]:
    basis = auxiliary["basis"]
    pod_mean = auxiliary["mean"]

    number_of_samples = int(indices.size)
    number_of_points = int(
        np.prod(data.grid_shape)
    )

    if basis.shape[1] != number_of_points:
        raise ValueError(
            "POD basis point dimension does not match "
            "the current target grid."
        )
    if pod_mean.size != number_of_points:
        raise ValueError(
            "POD mean point dimension does not match "
            "the current target grid."
        )

    prediction_flat = (
        np.empty(
            (
                number_of_samples,
                number_of_points,
            ),
            dtype=np.float32,
        )
        if collect_predictions
        else None
    )

    accumulator = MetricAccumulator(
        number_of_samples
    )

    for local_start in range(
        0,
        number_of_samples,
        sample_batch_size,
    ):
        local_stop = min(
            local_start + sample_batch_size,
            number_of_samples,
        )
        sample_ids = indices[
            local_start:local_stop
        ]

        normalized_branch = (
            np.asarray(
                data.branch[sample_ids],
                dtype=np.float32,
            )
            - branch_mean
        ) / branch_std

        branch_tensor = torch.from_numpy(
            np.ascontiguousarray(
                normalized_branch,
                dtype=np.float32,
            )
        ).to(device)

        coefficient_prediction = model(
            branch_tensor
        ).float().cpu().numpy()

        target_batch = np.asarray(
            data.target[sample_ids],
            dtype=np.float32,
        ).reshape(
            sample_ids.size,
            number_of_points,
        )

        for query_start in range(
            0,
            number_of_points,
            query_batch_size,
        ):
            query_stop = min(
                query_start + query_batch_size,
                number_of_points,
            )

            raw = (
                coefficient_prediction
                @ basis[
                    :,
                    query_start:query_stop,
                ]
                + pod_mean[
                    query_start:query_stop
                ]
            )

            accumulator.update(
                raw_prediction=raw,
                target=target_batch[
                    :,
                    query_start:query_stop,
                ],
                local_sample_start=local_start,
                maximum_generation=(
                    maximum_generation
                ),
                generation_threshold=(
                    generation_threshold
                ),
            )

            if prediction_flat is not None:
                prediction_flat[
                    local_start:local_stop,
                    query_start:query_stop,
                ] = np.clip(
                    raw,
                    0.0,
                    float(maximum_generation),
                )

    (
        loss,
        metrics,
        per_sample,
    ) = accumulator.finalize()

    prediction = (
        prediction_flat.reshape(
            (
                number_of_samples,
                *data.grid_shape,
            )
        )
        if prediction_flat is not None
        else None
    )

    return (
        loss,
        metrics,
        per_sample,
        prediction,
    )


@torch.inference_mode()
def _cno_normalization_from_pair(
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    problem: str,
) -> dict[str, float]:
    """Convert the normalization scalars saved by the CNO trainer."""
    mean = np.asarray(branch_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(branch_std, dtype=np.float32).reshape(-1)
    if problem == "reaction_diffusion":
        if mean.size < 1 or std.size < 1:
            raise ValueError(
                "CNO checkpoint must store a boundary normalization scalar."
            )
        return {
            "boundary_mean": float(mean[0]),
            "boundary_std": float(std[0]),
        }
    if mean.size < 2 or std.size < 2:
        raise ValueError(
            "CNO checkpoint must store (u, forcing) "
            "normalization pairs."
        )
    return {
        "u_mean": float(mean[0]),
        "u_std": float(std[0]),
        "forcing_mean": float(mean[1]),
        "forcing_std": float(std[1]),
    }


@torch.inference_mode()
def _evaluate_cno_model(
    model: nn.Module,
    data: OperatorData,
    indices: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    coordinates: torch.Tensor | None,
    device: torch.device,
    sample_batch_size: int,
    query_batch_size: int,
    maximum_generation: int,
    generation_threshold: float,
    collect_predictions: bool,
) -> tuple[
    float,
    dict[str, float],
    dict[str, np.ndarray],
    np.ndarray | None,
]:
    """Evaluate a CNO on full score grids (targets are grid fields)."""
    grid, _ = _make_cno_grid_data(data, indices)
    normalized = _cno_normalization_from_pair(
        branch_mean,
        branch_std,
        data.problem,
    )

    number_of_samples = int(indices.size)
    number_of_points = int(np.prod(data.grid_shape))

    prediction_flat = (
        np.empty(
            (number_of_samples, number_of_points),
            dtype=np.float32,
        )
        if collect_predictions
        else None
    )
    accumulator = MetricAccumulator(number_of_samples)

    target_full = np.asarray(
        data.target[indices],
        dtype=np.float32,
    ).reshape(number_of_samples, number_of_points)

    for local_start in range(
        0,
        number_of_samples,
        sample_batch_size,
    ):
        local_stop = min(
            local_start + sample_batch_size,
            number_of_samples,
        )
        sample_ids = indices[local_start:local_stop]
        raw, _ = _predict_cno(
            model,
            grid,
            normalized,
            sample_ids,
            device,
            sample_batch_size,
        )
        raw_flat = np.asarray(
            raw,
            dtype=np.float32,
        ).reshape(sample_ids.size, -1)

        accumulator.update(
            raw_prediction=raw_flat,
            target=target_full[local_start:local_stop],
            local_sample_start=local_start,
            maximum_generation=maximum_generation,
            generation_threshold=generation_threshold,
        )

        if prediction_flat is not None:
            prediction_flat[local_start:local_stop] = np.clip(
                raw_flat,
                0.0,
                float(maximum_generation),
            )

    (
        loss,
        metrics,
        per_sample,
    ) = accumulator.finalize()

    prediction = (
        prediction_flat.reshape(
            (number_of_samples,) + data.grid_shape
        )
        if prediction_flat is not None
        else None
    )

    return (
        loss,
        metrics,
        per_sample,
        prediction,
    )


@torch.inference_mode()
def _time_cno_model(
    model: nn.Module,
    data: OperatorData,
    indices: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    coordinates: torch.Tensor | None,
    device: torch.device,
    sample_batch_size: int,
    query_batch_size: int,
    repeats: int,
) -> dict[str, float]:
    grid, _ = _make_cno_grid_data(data, indices)
    normalized = _cno_normalization_from_pair(
        branch_mean,
        branch_std,
        data.problem,
    )

    def one_pass() -> None:
        _predict_cno(
            model,
            grid,
            normalized,
            indices,
            device,
            sample_batch_size,
        )

    one_pass()
    _synchronize(device)

    elapsed = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        one_pass()
        _synchronize(device)
        elapsed.append(time.perf_counter() - start)

    total = float(np.median(elapsed))
    samples = int(indices.size)

    return {
        "total_test_inference_time_sec": total,
        "mean_inference_time_sec_per_sample": (
            total / max(samples, 1)
        ),
        "throughput_samples_per_sec": (
            samples / max(total, 1.0e-12)
        ),
        "timing_repeats": int(repeats),
        "timed_samples": samples,
    }


def _time_point_model(
    model: nn.Module,
    data: OperatorData,
    indices: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    coordinates: torch.Tensor,
    device: torch.device,
    sample_batch_size: int,
    query_batch_size: int,
    repeats: int,
) -> dict[str, float]:
    number_of_points = int(
        np.prod(data.grid_shape)
    )

    def one_pass() -> None:
        for local_start in range(
            0,
            indices.size,
            sample_batch_size,
        ):
            sample_ids = indices[
                local_start:
                local_start + sample_batch_size
            ]
            normalized_branch = (
                np.asarray(
                    data.branch[sample_ids],
                    dtype=np.float32,
                )
                - branch_mean
            ) / branch_std
            branch_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    normalized_branch,
                    dtype=np.float32,
                )
            ).to(device)

            for query_start in range(
                0,
                number_of_points,
                query_batch_size,
            ):
                output = model(
                    branch_tensor,
                    coordinates[
                        query_start:
                        query_start + query_batch_size
                    ],
                )
                _ = output.float().cpu().numpy()

    # Warm-up.
    one_pass()
    _synchronize(device)

    elapsed = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        one_pass()
        _synchronize(device)
        elapsed.append(
            time.perf_counter() - start
        )

    total = float(np.median(elapsed))
    samples = int(indices.size)

    return {
        "total_test_inference_time_sec": total,
        "mean_inference_time_sec_per_sample": (
            total / max(samples, 1)
        ),
        "throughput_samples_per_sec": (
            samples / max(total, 1.0e-12)
        ),
        "timing_repeats": int(repeats),
        "timed_samples": samples,
    }


@torch.inference_mode()
def _time_pod_model(
    model: nn.Module,
    auxiliary: dict[str, np.ndarray],
    data: OperatorData,
    indices: np.ndarray,
    branch_mean: np.ndarray,
    branch_std: np.ndarray,
    device: torch.device,
    sample_batch_size: int,
    query_batch_size: int,
    repeats: int,
) -> dict[str, float]:
    basis = auxiliary["basis"]
    pod_mean = auxiliary["mean"]
    number_of_points = int(
        np.prod(data.grid_shape)
    )

    def one_pass() -> None:
        for local_start in range(
            0,
            indices.size,
            sample_batch_size,
        ):
            sample_ids = indices[
                local_start:
                local_start + sample_batch_size
            ]
            normalized_branch = (
                np.asarray(
                    data.branch[sample_ids],
                    dtype=np.float32,
                )
                - branch_mean
            ) / branch_std
            branch_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    normalized_branch,
                    dtype=np.float32,
                )
            ).to(device)

            coefficients = model(
                branch_tensor
            ).float().cpu().numpy()

            for query_start in range(
                0,
                number_of_points,
                query_batch_size,
            ):
                query_stop = min(
                    query_start + query_batch_size,
                    number_of_points,
                )
                output = (
                    coefficients
                    @ basis[
                        :,
                        query_start:query_stop,
                    ]
                    + pod_mean[
                        query_start:query_stop
                    ]
                )
                _ = np.asarray(output)

    # Warm-up.
    one_pass()
    _synchronize(device)

    elapsed = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        one_pass()
        _synchronize(device)
        elapsed.append(
            time.perf_counter() - start
        )

    total = float(np.median(elapsed))
    samples = int(indices.size)

    return {
        "total_test_inference_time_sec": total,
        "mean_inference_time_sec_per_sample": (
            total / max(samples, 1)
        ),
        "throughput_samples_per_sec": (
            samples / max(total, 1.0e-12)
        ),
        "timing_repeats": int(repeats),
        "timed_samples": samples,
    }


def _save_per_sample_csv(
    path: Path,
    test_indices: np.ndarray,
    source_attempt_id: np.ndarray,
    per_sample: dict[str, np.ndarray],
) -> tuple[int, int]:
    best = int(
        np.argmin(per_sample["mae"])
    )
    worst = int(
        np.argmax(per_sample["mae"])
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_order",
                "dataset_index_zero_based",
                "dataset_index_matlab_one_based",
                "source_attempt_id",
                "mae",
                "rmse",
                "relative_l2",
                "under_refinement_rate",
                "over_refinement_rate",
                "within_one_rate",
            ]
        )

        for local_index in range(
            test_indices.size
        ):
            dataset_index = int(
                test_indices[local_index]
            )
            writer.writerow(
                [
                    local_index + 1,
                    dataset_index,
                    dataset_index + 1,
                    int(
                        source_attempt_id[
                            dataset_index
                        ]
                    ),
                    float(
                        per_sample["mae"][
                            local_index
                        ]
                    ),
                    float(
                        per_sample["rmse"][
                            local_index
                        ]
                    ),
                    float(
                        per_sample["relative_l2"][
                            local_index
                        ]
                    ),
                    float(
                        per_sample["under_rate"][
                            local_index
                        ]
                    ),
                    float(
                        per_sample["over_rate"][
                            local_index
                        ]
                    ),
                    float(
                        per_sample["within_one"][
                            local_index
                        ]
                    ),
                ]
            )

    return best, worst


def _training_time_from_old_report(
    out_dir: Path,
) -> float:
    old_report = out_dir / "final_metrics.json"
    if not old_report.exists():
        return 0.0

    try:
        report = _load_json(old_report)
    except Exception:
        return 0.0

    return float(
        report.get(
            "training_time_sec",
            0.0,
        )
    )


def _backup_old_report(out_dir: Path) -> None:
    source = out_dir / "final_metrics.json"
    backup = (
        out_dir
        / "final_metrics_before_unified_evaluation.json"
    )

    if source.exists() and not backup.exists():
        shutil.copy2(source, backup)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an already-trained generic backbone "
            "using the FNO score-metric schema."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=(
            "Experiment result directory containing "
            "final_model.pt and resolved_config.json."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional configuration JSON. By default, "
            "<out-dir>/resolved_config.json is used."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Evaluation device. Defaults to training.device "
            "from resolved_config.json."
        ),
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--skip-train-metrics",
        action="store_true",
        help=(
            "Skip full training-set evaluation. The default "
            "computes it to match the FNO report."
        ),
    )

    args = parser.parse_args()

    if (
        args.sample_batch_size is not None
        and args.sample_batch_size < 1
    ):
        raise ValueError(
            "--sample-batch-size must be positive."
        )
    if (
        args.query_batch_size is not None
        and args.query_batch_size < 1
    ):
        raise ValueError(
            "--query-batch-size must be positive."
        )
    if (
        args.timing_repeats is not None
        and args.timing_repeats < 1
    ):
        raise ValueError(
            "--timing-repeats must be positive."
        )

    return args


def main() -> None:
    wall_start = time.perf_counter()
    args = parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    config = _load_config(
        out_dir,
        args.config,
    )

    name = str(config["model"]["name"])
    if name not in {
        "deeponet",
        "pod_deeponet",
        "transolver",
        "cno",
    }:
        raise ValueError(
            "This evaluator supports only deeponet, "
            "pod_deeponet, transolver, and cno."
        )

    configured_device = str(
        config["training"].get(
            "device",
            "cuda:0",
        )
    )
    device = torch.device(
        args.device
        if args.device is not None
        else configured_device
    )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )
        torch.cuda.set_device(device)

    data_path = (
        PROJECT_ROOT
        / str(config["data"]["path"])
    )
    data = load_problem_data(
        str(config["problem"]),
        data_path,
    )

    train_indices, test_indices = _load_split(
        out_dir,
        config,
    )

    if max(
        int(train_indices.max()),
        int(test_indices.max()),
    ) >= data.branch.shape[0]:
        raise ValueError(
            "Split index exceeds the loaded data set."
        )

    checkpoint_path = (
        out_dir / "final_model.pt"
    )
    checkpoint = _safe_torch_load(
        checkpoint_path,
        device,
    )

    checkpoint_name = str(
        checkpoint.get(
            "model_name",
            name,
        )
    )
    if checkpoint_name != name:
        raise ValueError(
            f"Checkpoint model is '{checkpoint_name}', "
            f"but configuration model is '{name}'."
        )

    model, auxiliary = _build_model(
        name,
        checkpoint,
        data,
        out_dir,
        device,
    )

    (
        branch_mean,
        branch_std,
        coordinates,
    ) = _normalization(
        checkpoint,
        data,
        device,
    )

    train_cfg = config["training"]
    score_cfg = config["score"]

    sample_batch_size = (
        int(args.sample_batch_size)
        if args.sample_batch_size is not None
        else int(
            train_cfg.get(
                "inference_sample_batch_size",
                2,
            )
        )
    )
    query_batch_size = (
        int(args.query_batch_size)
        if args.query_batch_size is not None
        else int(
            train_cfg.get(
                "query_batch_size",
                8192,
            )
        )
    )
    timing_repeats = (
        int(args.timing_repeats)
        if args.timing_repeats is not None
        else int(
            train_cfg.get(
                "timing_repeats",
                3,
            )
        )
    )

    maximum_generation = int(
        score_cfg["maximum"]
    )
    generation_threshold = float(
        score_cfg["generation_threshold"]
    )

    print("=" * 78)
    print("Unified evaluation of trained neural-operator backbone")
    print("=" * 78)
    print(f"problem              : {config['problem']}")
    print(f"model                : {name}")
    print(f"checkpoint           : {checkpoint_path}")
    print(f"data                 : {data_path}")
    print(f"device               : {device}")
    print(f"train samples        : {train_indices.size}")
    print(f"test samples         : {test_indices.size}")
    print(f"sample batch size    : {sample_batch_size}")
    print(f"query batch size     : {query_batch_size}")
    print(f"timing repeats       : {timing_repeats}")
    print(
        f"generation rule      : threshold="
        f"{generation_threshold}, max={maximum_generation}"
    )
    print("=" * 78)

    if name == "pod_deeponet":
        evaluator = _evaluate_pod_model
    elif name == "cno":
        evaluator = _evaluate_cno_model
    else:
        evaluator = _evaluate_point_model

    if args.skip_train_metrics:
        train_loss = math.nan
        train_metrics = {
            "mae": math.nan,
            "rmse": math.nan,
            "rounded_accuracy": math.nan,
            "within_0p5": math.nan,
            "within_1": math.nan,
            "under_refinement_rate": math.nan,
            "over_refinement_rate": math.nan,
            "ceil_level_mae": math.nan,
        }
    else:
        print("[evaluation] full training set...")
        if name == "pod_deeponet":
            (
                train_loss,
                train_metrics,
                _,
                _,
            ) = evaluator(
                model=model,
                auxiliary=auxiliary,
                data=data,
                indices=train_indices,
                branch_mean=branch_mean,
                branch_std=branch_std,
                device=device,
                sample_batch_size=(
                    sample_batch_size
                ),
                query_batch_size=(
                    query_batch_size
                ),
                maximum_generation=(
                    maximum_generation
                ),
                generation_threshold=(
                    generation_threshold
                ),
                collect_predictions=False,
            )
        else:
            (
                train_loss,
                train_metrics,
                _,
                _,
            ) = evaluator(
                model=model,
                data=data,
                indices=train_indices,
                branch_mean=branch_mean,
                branch_std=branch_std,
                coordinates=coordinates,
                device=device,
                sample_batch_size=(
                    sample_batch_size
                ),
                query_batch_size=(
                    query_batch_size
                ),
                maximum_generation=(
                    maximum_generation
                ),
                generation_threshold=(
                    generation_threshold
                ),
                collect_predictions=False,
            )

    print("[evaluation] held-out test set...")
    if name == "pod_deeponet":
        (
            test_loss,
            test_metrics,
            test_per_sample,
            test_prediction,
        ) = evaluator(
            model=model,
            auxiliary=auxiliary,
            data=data,
            indices=test_indices,
            branch_mean=branch_mean,
            branch_std=branch_std,
            device=device,
            sample_batch_size=(
                sample_batch_size
            ),
            query_batch_size=(
                query_batch_size
            ),
            maximum_generation=(
                maximum_generation
            ),
            generation_threshold=(
                generation_threshold
            ),
            collect_predictions=True,
        )
    else:
        (
            test_loss,
            test_metrics,
            test_per_sample,
            test_prediction,
        ) = evaluator(
            model=model,
            data=data,
            indices=test_indices,
            branch_mean=branch_mean,
            branch_std=branch_std,
            coordinates=coordinates,
            device=device,
            sample_batch_size=(
                sample_batch_size
            ),
            query_batch_size=(
                query_batch_size
            ),
            maximum_generation=(
                maximum_generation
            ),
            generation_threshold=(
                generation_threshold
            ),
            collect_predictions=True,
        )

    if test_prediction is None:
        raise RuntimeError(
            "Test predictions were not collected."
        )

    print("[timing] repeated test inference...")
    if name == "pod_deeponet":
        timing = _time_pod_model(
            model=model,
            auxiliary=auxiliary,
            data=data,
            indices=test_indices,
            branch_mean=branch_mean,
            branch_std=branch_std,
            device=device,
            sample_batch_size=(
                sample_batch_size
            ),
            query_batch_size=(
                query_batch_size
            ),
            repeats=timing_repeats,
        )
    elif name == "cno":
        timing = _time_cno_model(
            model=model,
            data=data,
            indices=test_indices,
            branch_mean=branch_mean,
            branch_std=branch_std,
            coordinates=coordinates,
            device=device,
            sample_batch_size=(
                sample_batch_size
            ),
            query_batch_size=(
                query_batch_size
            ),
            repeats=timing_repeats,
        )
    else:
        timing = _time_point_model(
            model=model,
            data=data,
            indices=test_indices,
            branch_mean=branch_mean,
            branch_std=branch_std,
            coordinates=coordinates,
            device=device,
            sample_batch_size=(
                sample_batch_size
            ),
            query_batch_size=(
                query_batch_size
            ),
            repeats=timing_repeats,
        )

    best_local, worst_local = (
        _save_per_sample_csv(
            out_dir
            / "per_sample_test_metrics.csv",
            test_indices,
            np.asarray(
                data.source_attempt_id,
                dtype=np.int64,
            ),
            test_per_sample,
        )
    )

    training_time = (
        _training_time_from_old_report(
            out_dir
        )
    )

    report: dict[str, Any] = {
        "code_version": CODE_VERSION,
        "problem": str(config["problem"]),
        "model": name,
        "method": (
            f"{name} continuous refinement-score regression"
        ),
        "loss": "mse",
        "split": {
            "training_samples": int(
                train_indices.size
            ),
            "held_out_test_samples": int(
                test_indices.size
            ),
        },
        "checkpoint_selection": (
            "fixed final epoch"
        ),
        "training_time_sec": float(
            training_time
        ),
        "final_training_set_loss": float(
            train_loss
        ),
        "final_training_set_metrics": {
            key: float(value)
            for key, value
            in train_metrics.items()
        },
        "test_loss": float(test_loss),
        "test_metrics": {
            key: float(value)
            for key, value
            in test_metrics.items()
        },
        "generation_rule": {
            "threshold": float(
                generation_threshold
            ),
            "minimum_generation": 0,
            "maximum_generation": int(
                maximum_generation
            ),
            "used_in_training_loss": False,
        },
        "per_sample_relative_l2": {
            "mean": float(
                np.mean(
                    test_per_sample[
                        "relative_l2"
                    ]
                )
            ),
            "median": float(
                np.median(
                    test_per_sample[
                        "relative_l2"
                    ]
                )
            ),
            "worst": float(
                np.max(
                    test_per_sample[
                        "relative_l2"
                    ]
                )
            ),
        },
        "timing": timing,
        "trainable_parameters": int(
            _count_parameters(model)
        ),
        "device": str(device),
        "precision": "float32",
        "best_mae_sample_matlab": int(
            test_indices[best_local]
        )
        + 1,
        "worst_mae_sample_matlab": int(
            test_indices[worst_local]
        )
        + 1,
        "best_mae_source_attempt_id": int(
            data.source_attempt_id[
                test_indices[best_local]
            ]
        ),
        "worst_mae_source_attempt_id": int(
            data.source_attempt_id[
                test_indices[worst_local]
            ]
        ),
    }

    if name == "pod_deeponet":
        report["pod_rank"] = int(
            auxiliary["basis"].shape[0]
        )
        pod_file = np.load(
            out_dir / "pod_basis.npz"
        )
        if "captured_energy" in pod_file:
            report[
                "pod_captured_energy"
            ] = float(
                np.asarray(
                    pod_file[
                        "captured_energy"
                    ]
                ).reshape(-1)[0]
            )

    _backup_old_report(out_dir)

    with (
        out_dir / "final_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    extras: dict[str, Any] = {
        "operator_name": np.array(
            [name],
            dtype=object,
        ),
        "test_mae": np.array(
            [[test_metrics["mae"]]],
            dtype=np.float64,
        ),
        "test_rmse": np.array(
            [[test_metrics["rmse"]]],
            dtype=np.float64,
        ),
        "test_rounded_accuracy": np.array(
            [[
                test_metrics[
                    "rounded_accuracy"
                ]
            ]],
            dtype=np.float64,
        ),
        "test_within_half_rate": np.array(
            [[test_metrics["within_0p5"]]],
            dtype=np.float64,
        ),
        "test_within_one_rate": np.array(
            [[test_metrics["within_1"]]],
            dtype=np.float64,
        ),
        "test_under_refinement_rate": np.array(
            [[
                test_metrics[
                    "under_refinement_rate"
                ]
            ]],
            dtype=np.float64,
        ),
        "test_over_refinement_rate": np.array(
            [[
                test_metrics[
                    "over_refinement_rate"
                ]
            ]],
            dtype=np.float64,
        ),
        "test_ceil_level_mae": np.array(
            [[
                test_metrics[
                    "ceil_level_mae"
                ]
            ]],
            dtype=np.float64,
        ),
        "training_time_sec": np.array(
            [[training_time]],
            dtype=np.float64,
        ),
        "mean_inference_time_sec_per_sample": np.array(
            [[
                timing[
                    "mean_inference_time_sec_per_sample"
                ]
            ]],
            dtype=np.float64,
        ),
        "per_sample_mae": np.asarray(
            test_per_sample["mae"],
            dtype=np.float64,
        ).reshape(-1, 1),
        "per_sample_rmse": np.asarray(
            test_per_sample["rmse"],
            dtype=np.float64,
        ).reshape(-1, 1),
        "per_sample_relative_l2": np.asarray(
            test_per_sample["relative_l2"],
            dtype=np.float64,
        ).reshape(-1, 1),
        "per_sample_under_rate": np.asarray(
            test_per_sample["under_rate"],
            dtype=np.float64,
        ).reshape(-1, 1),
        "per_sample_over_rate": np.asarray(
            test_per_sample["over_rate"],
            dtype=np.float64,
        ).reshape(-1, 1),
        "per_sample_within_one_rate": np.asarray(
            test_per_sample["within_one"],
            dtype=np.float64,
        ).reshape(-1, 1),
    }

    temporary_prediction_path = (
        out_dir / "predictions_unified_tmp.mat"
    )

    export_predictions(
        temporary_prediction_path,
        data,
        test_prediction,
        test_indices,
        train_indices,
        maximum_generation,
        generation_threshold,
        float(
            timing[
                "total_test_inference_time_sec"
            ]
        ),
        extras,
    )

    temporary_prediction_path.replace(
        out_dir / "predictions.mat"
    )

    total_wall_time = (
        time.perf_counter() - wall_start
    )

    print("\n" + "=" * 78)
    print("Unified evaluation finished")
    print("=" * 78)
    if not args.skip_train_metrics:
        print(
            f"final training-set MAE   : "
            f"{train_metrics['mae']:.6e}"
        )
        print(
            f"final training-set RMSE  : "
            f"{train_metrics['rmse']:.6e}"
        )
    print(
        f"test loss                : "
        f"{test_loss:.6e}"
    )
    print(
        f"test MAE                 : "
        f"{test_metrics['mae']:.6e}"
    )
    print(
        f"test RMSE                : "
        f"{test_metrics['rmse']:.6e}"
    )
    print(
        f"generation accuracy      : "
        f"{100.0 * test_metrics['rounded_accuracy']:.4f}%"
    )
    print(
        f"within 0.5               : "
        f"{100.0 * test_metrics['within_0p5']:.4f}%"
    )
    print(
        f"within one score level   : "
        f"{100.0 * test_metrics['within_1']:.4f}%"
    )
    print(
        f"under-refinement rate    : "
        f"{100.0 * test_metrics['under_refinement_rate']:.4f}%"
    )
    print(
        f"over-refinement rate     : "
        f"{100.0 * test_metrics['over_refinement_rate']:.4f}%"
    )
    print(
        f"generation MAE           : "
        f"{test_metrics['ceil_level_mae']:.6e}"
    )
    print(
        f"mean per-sample rel L2   : "
        f"{np.mean(test_per_sample['relative_l2']):.6e}"
    )
    print(
        f"inference time/sample    : "
        f"{timing['mean_inference_time_sec_per_sample']:.6e} s"
    )
    print(
        f"evaluation wall time     : "
        f"{total_wall_time:.3f} s"
    )
    print(
        f"metrics JSON             : "
        f"{out_dir / 'final_metrics.json'}"
    )
    print(
        f"prediction MAT           : "
        f"{out_dir / 'predictions.mat'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
