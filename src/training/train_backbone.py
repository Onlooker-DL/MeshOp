from __future__ import annotations

import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import OperatorData, load_problem_data
from src.inference.export_scores import export_predictions
from src.models import (
    CNO,
    DeepONet,
    MultiBranchDeepONet,
    PODCoefficientNet,
    Transolver,
)


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _load_split(
    path: Path,
    ntrain: int,
    ntest: int,
) -> tuple[np.ndarray, np.ndarray]:
    split = np.load(path)
    train = np.asarray(split["train_indices"], dtype=np.int64).reshape(-1)
    test = np.asarray(split["test_indices"], dtype=np.int64).reshape(-1)

    if train.size != ntrain or test.size != ntest:
        raise ValueError(
            "Shared split sizes do not match the YAML configuration."
        )
    if np.intersect1d(train, test).size:
        raise ValueError("Shared train/test split overlaps.")
    if (
        train.size == 0
        or test.size == 0
        or min(int(train.min()), int(test.min())) < 0
    ):
        raise ValueError(
            "Shared split contains an invalid negative index."
        )

    return train, test


def _coordinate_normalization(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    scale = np.maximum(high - low, 1.0e-8)
    normalized = 2.0 * (coordinates - low) / scale - 1.0

    return (
        normalized.astype(np.float32),
        low.astype(np.float32),
        high.astype(np.float32),
    )


def _make_point_model(
    name: str,
    branch_dim: int,
    coordinate_dim: int,
    cfg: dict[str, Any],
) -> nn.Module:
    if name == "deeponet":
        return DeepONet(
            branch_dim=branch_dim,
            coordinate_dim=coordinate_dim,
            latent_dim=int(cfg.get("latent_dim", 256)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            branch_depth=int(cfg.get("branch_depth", 4)),
            trunk_depth=int(cfg.get("trunk_depth", 4)),
            activation=str(cfg.get("activation", "gelu")),
            dropout=float(cfg.get("dropout", 0.0)),
        )

    if name == "deeponet_multi":
        return MultiBranchDeepONet(
            branch_dim=branch_dim,
            coordinate_dim=coordinate_dim,
            latent_dim=int(cfg.get("latent_dim", 64)),
            hidden_dim=int(cfg.get("hidden_dim", 28)),
            branch_depth=int(cfg.get("branch_depth", 4)),
            trunk_depth=int(cfg.get("trunk_depth", 4)),
            branch_split=int(cfg.get("branch_split", 0)),
            activation=str(cfg.get("activation", "gelu")),
            dropout=float(cfg.get("dropout", 0.0)),
        )

    if name == "transolver":
        return Transolver(
            branch_dim=branch_dim,
            coordinate_dim=coordinate_dim,
            width=int(cfg.get("width", 128)),
            layers=int(cfg.get("layers", 4)),
            heads=int(cfg.get("heads", 8)),
            slices=int(cfg.get("slices", 32)),
            dropout=float(cfg.get("dropout", 0.0)),
        )

    raise ValueError(name)


def _target_slice(
    target: np.ndarray,
    sample_ids: np.ndarray,
    query_ids: np.ndarray,
) -> np.ndarray:
    batch = np.asarray(
        target[sample_ids],
        dtype=np.float32,
    ).reshape(sample_ids.size, -1)

    return np.ascontiguousarray(
        batch[:, query_ids],
        dtype=np.float32,
    )


def _predict_point_model(
    model: nn.Module,
    branch: np.ndarray,
    coordinates: torch.Tensor,
    device: torch.device,
    sample_batch_size: int,
    query_batch_size: int,
) -> tuple[np.ndarray, float]:
    model.eval()

    output = np.empty(
        (branch.shape[0], coordinates.shape[0]),
        dtype=np.float32,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, branch.shape[0], sample_batch_size):
            b = torch.from_numpy(
                branch[i : i + sample_batch_size]
            ).to(device)

            chunks = []
            for q in range(0, coordinates.shape[0], query_batch_size):
                chunks.append(
                    model(
                        b,
                        coordinates[q : q + query_batch_size],
                    ).cpu()
                )

            output[i : i + b.shape[0]] = torch.cat(
                chunks,
                dim=1,
            ).numpy()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return output, time.perf_counter() - start


def _make_cno_grid_data(
    data: OperatorData,
    train_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build Burgers query-grid fields and their training-set normalization.

    Returns a grid dict (u0_query, forcing_query, x_channel, t_channel) and a
    normalization dict (u_mean, u_std, forcing_mean, forcing_std). The dense
    sensor-grid fields stored in the metadata are interpolated onto the common
    query grid so the CNO input and target share the same resolution, exactly
    as the FNO path does.
    """
    if data.problem == "reaction_diffusion":
        metadata = data.metadata
        boundary = np.asarray(
            metadata["boundary"],
            dtype=np.float32,
        )
        query_x = np.asarray(
            metadata["query_x"], dtype=np.float64
        ).reshape(-1)
        query_y = np.asarray(
            metadata["query_y"], dtype=np.float64
        ).reshape(-1)
        query_z = np.asarray(
            metadata["query_z"], dtype=np.float64
        ).reshape(-1)

        train_b = boundary[train_indices]
        normalized = {
            "boundary_mean": float(
                np.mean(train_b, dtype=np.float64)
            ),
            "boundary_std": float(
                max(
                    np.std(train_b, dtype=np.float64),
                    1.0e-8,
                )
            ),
        }

        def _norm(axis_values: np.ndarray) -> np.ndarray:
            span = float(axis_values.max() - axis_values.min())
            return (
                2.0
                * (axis_values - float(axis_values.min()))
                / max(span, 1.0e-12)
                - 1.0
            )

        xx, yy, zz = np.meshgrid(
            _norm(query_x),
            _norm(query_y),
            _norm(query_z),
            indexing="ij",
        )
        grid = {
            "boundary": boundary,
            "x_channel": xx.astype(np.float32),
            "y_channel": yy.astype(np.float32),
            "z_channel": zz.astype(np.float32),
        }
        return grid, normalized

    metadata = data.metadata
    query_x = np.asarray(metadata["query_x"], dtype=np.float64).reshape(-1)
    query_t = np.asarray(metadata["query_t"], dtype=np.float64).reshape(-1)
    x_input = np.asarray(metadata["x_input"], dtype=np.float64).reshape(-1)
    u0 = np.asarray(metadata["U0"], dtype=np.float32)
    forcing = np.asarray(metadata["F"], dtype=np.float32)

    if u0.ndim != 2 or u0.shape[1] != x_input.size:
        raise ValueError("metadata U0 must be (samples, nx_input).")
    if forcing.shape != u0.shape:
        raise ValueError("metadata F shape does not match U0.")

    u0_query = np.stack(
        [np.interp(query_x, x_input, row) for row in u0],
        axis=0,
    ).astype(np.float32)
    forcing_query = np.stack(
        [np.interp(query_x, x_input, row) for row in forcing],
        axis=0,
    ).astype(np.float32)

    train_u = u0_query[train_indices]
    train_f = forcing_query[train_indices]
    normalized = {
        "u_mean": float(np.mean(train_u, dtype=np.float64)),
        "u_std": float(
            max(np.std(train_u, dtype=np.float64), 1.0e-8)
        ),
        "forcing_mean": float(
            np.mean(train_f, dtype=np.float64)
        ),
        "forcing_std": float(
            max(np.std(train_f, dtype=np.float64), 1.0e-8)
        ),
    }

    x_period = float(query_x.max() - query_x.min())
    t_range = float(query_t.max() - query_t.min())
    x_norm = (
        2.0 * (query_x - float(query_x.min()))
        / max(x_period, 1.0e-12)
        - 1.0
    )
    t_norm = (
        2.0 * (query_t - float(query_t.min()))
        / max(t_range, 1.0e-12)
        - 1.0
    )
    xx, tt = np.meshgrid(x_norm, t_norm, indexing="ij")

    grid = {
        "u0_query": u0_query,
        "forcing_query": forcing_query,
        "x_channel": xx.astype(np.float32),
        "t_channel": tt.astype(np.float32),
    }
    return grid, normalized


def _cno_input_tensor(
    grid: dict[str, Any],
    sample_ids: np.ndarray,
    normalized: dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    """Stack a batch of 4-channel (u0, forcing, x, t) grid fields."""
    if "boundary" in grid:
        # Reaction-diffusion: boundary field broadcast along z + x/y/z coords.
        b = (
            grid["boundary"][sample_ids]
            - normalized["boundary_mean"]
        ) / normalized["boundary_std"]
        nz = grid["z_channel"].shape[2]
        b_field = np.repeat(b[:, :, :, None], nz, axis=3)
        x_channel = np.broadcast_to(
            grid["x_channel"][None, ...],
            (sample_ids.size,) + grid["x_channel"].shape,
        )
        y_channel = np.broadcast_to(
            grid["y_channel"][None, ...],
            (sample_ids.size,) + grid["y_channel"].shape,
        )
        z_channel = np.broadcast_to(
            grid["z_channel"][None, ...],
            (sample_ids.size,) + grid["z_channel"].shape,
        )
        stack = np.stack(
            [b_field, x_channel, y_channel, z_channel],
            axis=1,
        )
        return torch.from_numpy(
            np.ascontiguousarray(stack, dtype=np.float32)
        ).to(device)

    u = (
        grid["u0_query"][sample_ids] - normalized["u_mean"]
    ) / normalized["u_std"]
    f = (
        grid["forcing_query"][sample_ids]
        - normalized["forcing_mean"]
    ) / normalized["forcing_std"]
    nt = grid["t_channel"].shape[1]
    u_field = np.repeat(u[:, :, None], nt, axis=2)
    f_field = np.repeat(f[:, :, None], nt, axis=2)
    x_channel = np.broadcast_to(
        grid["x_channel"][None, ...],
        (sample_ids.size,) + grid["x_channel"].shape,
    )
    t_channel = np.broadcast_to(
        grid["t_channel"][None, ...],
        (sample_ids.size,) + grid["t_channel"].shape,
    )
    stack = np.stack(
        [u_field, f_field, x_channel, t_channel],
        axis=1,
    )
    return torch.from_numpy(
        np.ascontiguousarray(stack, dtype=np.float32)
    ).to(device)


def _predict_cno(
    model: nn.Module,
    grid: dict[str, Any],
    normalized: dict[str, float],
    indices: np.ndarray,
    device: torch.device,
    sample_batch_size: int,
) -> tuple[np.ndarray, float]:
    """Predict full score grids for the given sample indices."""
    model.eval()
    output = np.empty(
        (indices.size,) + grid["x_channel"].shape,
        dtype=np.float32,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, indices.size, sample_batch_size):
            sample_ids = indices[i : i + sample_batch_size]
            inputs = _cno_input_tensor(
                grid, sample_ids, normalized, device
            )
            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    prediction = model(inputs).float()
            else:
                prediction = model(inputs).float()
            output[i : i + sample_ids.size] = (
                prediction.squeeze(1).cpu().numpy()
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - start


def _compute_score_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    generation_threshold: float,
    minimum_generation: int,
    maximum_generation: int,
) -> tuple[
    dict[str, float],
    dict[str, float],
    np.ndarray,
]:
    """
    Compute continuous-score and integer-generation metrics.

    Both prediction and target must be on the original, unnormalized
    refinement-score scale.
    """
    prediction64 = np.asarray(
        prediction,
        dtype=np.float64,
    )
    target64 = np.asarray(
        target,
        dtype=np.float64,
    )

    if prediction64.shape != target64.shape:
        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{prediction64.shape} versus {target64.shape}."
        )

    difference = prediction64 - target64
    absolute_difference = np.abs(difference)

    prediction_generation = np.floor(
        prediction64 + generation_threshold
    )
    target_generation = np.floor(
        target64 + generation_threshold
    )

    prediction_generation = np.clip(
        prediction_generation,
        minimum_generation,
        maximum_generation,
    ).astype(np.int64)

    target_generation = np.clip(
        target_generation,
        minimum_generation,
        maximum_generation,
    ).astype(np.int64)

    generation_difference = (
        prediction_generation - target_generation
    )

    mse = float(np.mean(difference**2))
    mae = float(np.mean(absolute_difference))
    rmse = float(np.sqrt(mse))

    test_metrics = {
        "mae": mae,
        "rmse": rmse,
        "rounded_accuracy": float(
            np.mean(generation_difference == 0)
        ),
        "within_0p5": float(
            np.mean(absolute_difference <= 0.5)
        ),
        "within_1": float(
            np.mean(absolute_difference <= 1.0)
        ),
        "under_refinement_rate": float(
            np.mean(generation_difference < 0)
        ),
        "over_refinement_rate": float(
            np.mean(generation_difference > 0)
        ),
        # Keep the same field name used by the Burgers FNO output.
        "ceil_level_mae": float(
            np.mean(np.abs(generation_difference))
        ),
    }

    per_sample_difference = difference.reshape(
        difference.shape[0],
        -1,
    )
    per_sample_target = target64.reshape(
        target64.shape[0],
        -1,
    )

    per_sample_relative_l2 = (
        np.linalg.norm(per_sample_difference, axis=1)
        / np.maximum(
            np.linalg.norm(per_sample_target, axis=1),
            1.0e-12,
        )
    )

    relative_l2_metrics = {
        "mean": float(np.mean(per_sample_relative_l2)),
        "median": float(np.median(per_sample_relative_l2)),
        "worst": float(np.max(per_sample_relative_l2)),
    }

    return (
        test_metrics,
        relative_l2_metrics,
        per_sample_relative_l2,
    )


def _train_point_model(
    name: str,
    data: OperatorData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: dict[str, Any],
    out_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_cfg = config["model"]
    train_cfg = config["training"]

    branch_train = np.asarray(
        data.branch[train_indices],
        dtype=np.float32,
    )
    branch_mean = branch_train.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    branch_std = branch_train.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    branch_std = np.maximum(branch_std, 1.0e-6)

    normalized_branch = (
        data.branch.astype(np.float32) - branch_mean
    ) / branch_std

    (
        normalized_coordinates,
        coordinate_min,
        coordinate_max,
    ) = _coordinate_normalization(data.coordinates)

    coordinates = torch.from_numpy(
        normalized_coordinates
    ).to(device)

    model = _make_point_model(
        name,
        normalized_branch.shape[1],
        normalized_coordinates.shape[1],
        model_cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(
            train_cfg.get("weight_decay", 1.0e-6)
        ),
    )

    epochs = int(train_cfg.get("epochs", 500))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg.get("eta_min", 1.0e-5)),
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices)),
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    query_points = min(
        int(train_cfg.get("train_query_points", 4096)),
        data.coordinates.shape[0],
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(config.get("seed", 900)) + 31
    )

    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for (sample_ids_tensor,) in loader:
            sample_ids = sample_ids_tensor.numpy().astype(
                np.int64
            )

            qids = torch.randperm(
                data.coordinates.shape[0],
                generator=generator,
            )[:query_points].numpy()

            query_index_tensor = torch.from_numpy(
                qids
            ).to(device)

            branch_tensor = torch.from_numpy(
                normalized_branch[sample_ids]
            ).to(device)

            target_tensor = torch.from_numpy(
                _target_slice(
                    data.target,
                    sample_ids,
                    qids,
                )
            ).to(device)

            prediction = model(
                branch_tensor,
                coordinates.index_select(
                    0,
                    query_index_tensor,
                ),
            )

            loss = torch.mean(
                (prediction - target_tensor) ** 2
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

            total_loss += (
                float(loss.detach()) * sample_ids.size
            )
            total_items += sample_ids.size

        scheduler.step()

        epoch_loss = total_loss / max(
            total_items,
            1,
        )

        history.append(
            {
                "epoch": epoch,
                "training_mse": epoch_loss,
            }
        )

        if (
            epoch == 1
            or epoch
            % int(train_cfg.get("log_every", 10))
            == 0
            or epoch == epochs
        ):
            print(
                f"[{name}] epoch {epoch:4d}/{epochs}: "
                f"MSE={epoch_loss:.6e}"
            )

    torch.save(
        {
            "model_name": name,
            "model_config": model_cfg,
            "state_dict": model.state_dict(),
            "branch_mean": branch_mean,
            "branch_std": branch_std,
            "coordinate_min": coordinate_min,
            "coordinate_max": coordinate_max,
            "epoch": epochs,
        },
        out_dir / "final_model.pt",
    )

    test_branch = np.ascontiguousarray(
        normalized_branch[test_indices]
    )

    prediction_flat, inference_time = (
        _predict_point_model(
            model,
            test_branch,
            coordinates,
            device,
            int(
                train_cfg.get(
                    "inference_sample_batch_size",
                    2,
                )
            ),
            int(
                train_cfg.get(
                    "query_batch_size",
                    8192,
                )
            ),
        )
    )

    prediction = prediction_flat.reshape(
        (test_indices.size,) + data.grid_shape
    )

    return prediction, {
        "history": history,
        "inference_time_sec": inference_time,
        "trainable_parameters": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
        "normalization": {
            "branch_mean_shape": list(
                branch_mean.shape
            ),
            "coordinate_min": coordinate_min.tolist(),
            "coordinate_max": coordinate_max.tolist(),
        },
    }


def _train_cno(
    name: str,
    data: OperatorData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: dict[str, Any],
    out_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train the convolutional neural operator on full score grids."""
    if data.problem not in {"burgers", "reaction_diffusion"}:
        raise ValueError(
            f"The CNO grid-input path is not implemented for {data.problem}."
        )

    model_cfg = config["model"]
    train_cfg = config["training"]
    grid, normalized = _make_cno_grid_data(data, train_indices)
    dim = 3 if data.problem == "reaction_diffusion" else 2

    model = CNO(
        dim=dim,
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
        activation=str(model_cfg.get("activation", "gelu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"[cno] trainable parameters = {trainable_parameters}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-6)),
    )

    epochs = int(train_cfg.get("epochs", 500))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg.get("eta_min", 1.0e-5)),
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices)),
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    use_amp = bool(train_cfg.get("bf16_amp", False)) and device.type == "cuda"
    target_full = np.asarray(data.target, dtype=np.float32)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for (sample_ids_tensor,) in loader:
            sample_ids = sample_ids_tensor.numpy().astype(
                np.int64
            )
            inputs = _cno_input_tensor(
                grid, sample_ids, normalized, device
            )
            target = torch.from_numpy(
                np.ascontiguousarray(
                    target_full[sample_ids],
                    dtype=np.float32,
                )
            ).unsqueeze(1).to(device)

            if use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    prediction = model(inputs)
                    loss = torch.mean((prediction - target) ** 2)
            else:
                prediction = model(inputs)
                loss = torch.mean((prediction - target) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
            optimizer.step()

            total_loss += (
                float(loss.detach()) * sample_ids.size
            )
            total_items += sample_ids.size

        scheduler.step()

        epoch_loss = total_loss / max(total_items, 1)
        history.append(
            {
                "epoch": epoch,
                "training_mse": epoch_loss,
            }
        )

        if (
            epoch == 1
            or epoch % int(train_cfg.get("log_every", 10)) == 0
            or epoch == epochs
        ):
            print(
                f"[{name}] epoch {epoch:4d}/{epochs}: "
                f"MSE={epoch_loss:.6e}"
            )

    if data.problem == "reaction_diffusion":
        checkpoint_mean = np.asarray(
            [normalized["boundary_mean"]],
            dtype=np.float32,
        )
        checkpoint_std = np.asarray(
            [normalized["boundary_std"]],
            dtype=np.float32,
        )
    else:
        checkpoint_mean = np.asarray(
            [
                normalized["u_mean"],
                normalized["forcing_mean"],
            ],
            dtype=np.float32,
        )
        checkpoint_std = np.asarray(
            [
                normalized["u_std"],
                normalized["forcing_std"],
            ],
            dtype=np.float32,
        )

    torch.save(
        {
            "model_name": name,
            "model_config": model_cfg,
            "state_dict": model.state_dict(),
            "branch_mean": checkpoint_mean,
            "branch_std": checkpoint_std,
            "epoch": epochs,
        },
        out_dir / "final_model.pt",
    )

    prediction, inference_time = _predict_cno(
        model,
        grid,
        normalized,
        test_indices,
        device,
        int(
            train_cfg.get(
                "inference_sample_batch_size",
                2,
            )
        ),
    )

    return prediction, {
        "history": history,
        "inference_time_sec": inference_time,
        "trainable_parameters": trainable_parameters,
        "normalization": (
            {
                "boundary_mean": normalized["boundary_mean"],
                "boundary_std": normalized["boundary_std"],
            }
            if data.problem == "reaction_diffusion"
            else {
                "u_mean": normalized["u_mean"],
                "u_std": normalized["u_std"],
                "forcing_mean": normalized["forcing_mean"],
                "forcing_std": normalized["forcing_std"],
            }
        ),
    }


def _randomized_pod(
    target: np.ndarray,
    train_indices: np.ndarray,
    energy: float,
    maximum_rank: int,
    oversampling: int,
    seed: int,
    sample_chunk: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Training-only randomized POD without materializing a centered
    3-D tensor.
    """
    n = train_indices.size
    points = int(np.prod(target.shape[1:]))

    maximum_rank = min(maximum_rank, n - 1)
    sketch_rank = min(
        maximum_rank + oversampling,
        n - 1,
    )

    mean = np.zeros(points, dtype=np.float64)

    for start in range(0, n, sample_chunk):
        ids = train_indices[start : start + sample_chunk]
        mean += np.asarray(
            target[ids],
            dtype=np.float32,
        ).reshape(ids.size, points).sum(
            axis=0,
            dtype=np.float64,
        )

    mean = (mean / n).astype(np.float32)

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(
        (points, sketch_rank),
        dtype=np.float32,
    )

    sketch = np.empty(
        (n, sketch_rank),
        dtype=np.float32,
    )

    total_energy = 0.0

    for start in range(0, n, sample_chunk):
        ids = train_indices[start : start + sample_chunk]
        centered = np.asarray(
            target[ids],
            dtype=np.float32,
        ).reshape(ids.size, points) - mean

        sketch[start : start + ids.size] = (
            centered @ omega
        )

        total_energy += float(
            np.sum(centered.astype(np.float64) ** 2)
        )

    del omega

    q, _ = np.linalg.qr(
        sketch,
        mode="reduced",
    )

    compressed = np.zeros(
        (sketch_rank, points),
        dtype=np.float32,
    )

    for start in range(0, n, sample_chunk):
        ids = train_indices[start : start + sample_chunk]
        centered = np.asarray(
            target[ids],
            dtype=np.float32,
        ).reshape(ids.size, points) - mean

        compressed += (
            q[start : start + ids.size].T
            @ centered
        )

    _, singular_values, basis = np.linalg.svd(
        compressed,
        full_matrices=False,
    )

    captured = (
        np.cumsum(
            singular_values.astype(np.float64) ** 2
        )
        / max(total_energy, 1.0e-30)
    )

    eligible = np.flatnonzero(captured >= energy)

    rank = (
        int(eligible[0] + 1)
        if eligible.size
        else maximum_rank
    )
    rank = min(rank, maximum_rank)

    captured_energy = float(captured[rank - 1])

    if captured_energy < energy:
        warnings.warn(
            f"maximum_rank={maximum_rank} captures "
            f"{captured_energy:.6f}, below the requested POD "
            f"energy {energy:.6f}; raise maximum_rank.",
            RuntimeWarning,
        )

    basis = np.ascontiguousarray(
        basis[:rank],
        dtype=np.float32,
    )

    coefficients = np.empty(
        (n, rank),
        dtype=np.float32,
    )

    for start in range(0, n, sample_chunk):
        ids = train_indices[start : start + sample_chunk]
        centered = np.asarray(
            target[ids],
            dtype=np.float32,
        ).reshape(ids.size, points) - mean

        coefficients[start : start + ids.size] = (
            centered @ basis.T
        )

    return (
        mean,
        basis,
        coefficients,
        singular_values[:rank],
        captured_energy,
    )


def _train_pod(
    data: OperatorData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: dict[str, Any],
    out_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_cfg = config["model"]
    train_cfg = config["training"]
    pod_cfg = config.get("pod", {})

    (
        mean,
        basis,
        coefficients,
        singular_values,
        captured_energy,
    ) = _randomized_pod(
        data.target,
        train_indices,
        float(pod_cfg.get("energy", 0.99)),
        int(pod_cfg.get("maximum_rank", 128)),
        int(pod_cfg.get("oversampling", 16)),
        int(config.get("seed", 900)) + 101,
        int(pod_cfg.get("sample_chunk", 8)),
    )

    np.savez_compressed(
        out_dir / "pod_basis.npz",
        mean=mean,
        basis=basis,
        singular_values=singular_values,
        captured_energy=np.array([captured_energy]),
        train_indices=train_indices,
    )

    branch_train = np.asarray(
        data.branch[train_indices],
        dtype=np.float32,
    )

    branch_mean = branch_train.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    branch_std = np.maximum(
        branch_train.std(
            axis=0,
            dtype=np.float64,
        ).astype(np.float32),
        1.0e-6,
    )

    normalized_train = (
        branch_train - branch_mean
    ) / branch_std

    model = PODCoefficientNet(
        branch_dim=normalized_train.shape[1],
        rank=basis.shape[0],
        hidden_dim=int(
            model_cfg.get("hidden_dim", 256)
        ),
        depth=int(model_cfg.get("depth", 4)),
        activation=str(
            model_cfg.get("activation", "gelu")
        ),
        dropout=float(
            model_cfg.get("dropout", 0.0)
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(
            train_cfg.get("weight_decay", 1.0e-6)
        ),
    )

    epochs = int(train_cfg.get("epochs", 500))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(
            train_cfg.get("eta_min", 1.0e-5)
        ),
    )

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_train),
            torch.from_numpy(coefficients),
        ),
        batch_size=int(
            train_cfg.get("batch_size", 16)
        ),
        shuffle=True,
        num_workers=int(
            train_cfg.get("num_workers", 0)
        ),
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0

        for branch_batch, coefficient_batch in loader:
            branch_batch = branch_batch.to(device)
            coefficient_batch = (
                coefficient_batch.to(device)
            )

            prediction = model(branch_batch)
            loss = torch.mean(
                (prediction - coefficient_batch) ** 2
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    train_cfg.get(
                        "grad_clip",
                        1.0,
                    )
                ),
            )

            optimizer.step()

            total += (
                float(loss.detach())
                * branch_batch.shape[0]
            )
            count += branch_batch.shape[0]

        scheduler.step()

        epoch_loss = total / max(count, 1)

        history.append(
            {
                "epoch": epoch,
                "training_coefficient_mse": (
                    epoch_loss
                ),
            }
        )

        if (
            epoch == 1
            or epoch
            % int(train_cfg.get("log_every", 10))
            == 0
            or epoch == epochs
        ):
            print(
                "[pod_deeponet] "
                f"epoch {epoch:4d}/{epochs}: "
                f"coefficient MSE={epoch_loss:.6e}"
            )

    torch.save(
        {
            "model_name": "pod_deeponet",
            "model_config": model_cfg,
            "state_dict": model.state_dict(),
            "branch_mean": branch_mean,
            "branch_std": branch_std,
            "pod_rank": basis.shape[0],
            "epoch": epochs,
        },
        out_dir / "final_model.pt",
    )

    test_branch = (
        np.asarray(
            data.branch[test_indices],
            dtype=np.float32,
        )
        - branch_mean
    ) / branch_std

    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    with torch.no_grad():
        coefficient_prediction = model(
            torch.from_numpy(test_branch).to(device)
        ).cpu().numpy()

    prediction_flat = (
        coefficient_prediction @ basis + mean
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    inference_time = time.perf_counter() - start

    prediction = prediction_flat.reshape(
        (test_indices.size,) + data.grid_shape
    )

    return prediction, {
        "history": history,
        "inference_time_sec": inference_time,
        "trainable_parameters": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
        "pod_rank": int(basis.shape[0]),
        "pod_captured_energy": captured_energy,
    }


def train(
    config: dict[str, Any],
    root: Path,
    out_dir: Path,
    split_file: Path,
) -> Path:
    name = str(config["model"]["name"])

    if name not in {
        "deeponet",
        "deeponet_multi",
        "pod_deeponet",
        "transolver",
        "cno",
    }:
        raise ValueError(
            f"Generic trainer cannot train {name}."
        )

    seed = int(config.get("seed", 900))
    deterministic = bool(
        config.get("deterministic", False)
    )

    _seed_everything(seed, deterministic)

    device = torch.device(
        str(
            config["training"].get(
                "device",
                "cuda:0",
            )
        )
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    data_path = root / str(config["data"]["path"])
    data = load_problem_data(
        str(config["problem"]),
        data_path,
    )

    configured_total = config["data"].get(
        "total_samples"
    )

    if (
        configured_total is not None
        and int(configured_total)
        != data.branch.shape[0]
    ):
        raise ValueError(
            f"Configured data.total_samples="
            f"{configured_total}, but {data_path} "
            f"contains {data.branch.shape[0]} samples."
        )

    train_indices, test_indices = _load_split(
        split_file,
        int(config["data"]["train_samples"]),
        int(config["data"]["test_samples"]),
    )

    if (
        max(
            int(train_indices.max()),
            int(test_indices.max()),
        )
        >= data.branch.shape[0]
    ):
        raise ValueError(
            "Split index exceeds the loaded data set."
        )

    start = time.perf_counter()

    if name == "pod_deeponet":
        predictions, details = _train_pod(
            data,
            train_indices,
            test_indices,
            config,
            out_dir,
            device,
        )
    elif name == "cno":
        predictions, details = _train_cno(
            name,
            data,
            train_indices,
            test_indices,
            config,
            out_dir,
            device,
        )
    else:
        predictions, details = _train_point_model(
            name,
            data,
            train_indices,
            test_indices,
            config,
            out_dir,
            device,
        )

    training_time = time.perf_counter() - start

    target = np.asarray(
        data.target[test_indices],
        dtype=np.float32,
    )

    minimum_generation = int(
        config["score"].get(
            "minimum",
            0,
        )
    )
    maximum_generation = int(
        config["score"]["maximum"]
    )
    generation_threshold = float(
        config["score"]["generation_threshold"]
    )

    clipped = np.clip(
        predictions,
        float(minimum_generation),
        float(maximum_generation),
    ).astype(np.float32, copy=False)

    (
        test_metrics,
        relative_l2_metrics,
        per_sample_relative_l2,
    ) = _compute_score_metrics(
        prediction=clipped,
        target=target,
        generation_threshold=generation_threshold,
        minimum_generation=minimum_generation,
        maximum_generation=maximum_generation,
    )

    inference_time = float(
        details["inference_time_sec"]
    )
    mean_inference_time = (
        inference_time / test_indices.size
    )
    throughput = (
        test_indices.size / inference_time
        if inference_time > 0.0
        else math.inf
    )

    metrics = {
        "problem": config["problem"],
        "model": name,
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
        "test_mse": float(
            test_metrics["rmse"] ** 2
        ),
        "test_metrics": test_metrics,
        "generation_rule": {
            "threshold": generation_threshold,
            "minimum_generation": (
                minimum_generation
            ),
            "maximum_generation": (
                maximum_generation
            ),
            "used_in_training_loss": False,
        },
        "per_sample_relative_l2": (
            relative_l2_metrics
        ),
        # Preserve the previous top-level fields for compatibility.
        "mean_per_sample_relative_l2": float(
            relative_l2_metrics["mean"]
        ),
        "median_per_sample_relative_l2": float(
            relative_l2_metrics["median"]
        ),
        "timing": {
            # Preserve the previous field.
            "test_inference_time_sec": (
                inference_time
            ),
            # Add the FNO-style field.
            "total_test_inference_time_sec": (
                inference_time
            ),
            "mean_inference_time_sec_per_sample": (
                mean_inference_time
            ),
            "throughput_samples_per_sec": (
                throughput
            ),
            "timed_samples": int(
                test_indices.size
            ),
        },
        "trainable_parameters": int(
            details["trainable_parameters"]
        ),
        "device": str(device),
        "precision": "float32",
    }

    if "pod_rank" in details:
        metrics["pod_rank"] = int(
            details["pod_rank"]
        )
        metrics["pod_captured_energy"] = float(
            details["pod_captured_energy"]
        )

    with (
        out_dir / "final_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    with (
        out_dir / "history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            details["history"],
            handle,
            indent=2,
        )

    prediction_path = out_dir / "predictions.mat"

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
        "rounded_accuracy": np.array(
            [[test_metrics["rounded_accuracy"]]],
            dtype=np.float64,
        ),
        "within_0p5": np.array(
            [[test_metrics["within_0p5"]]],
            dtype=np.float64,
        ),
        "within_1": np.array(
            [[test_metrics["within_1"]]],
            dtype=np.float64,
        ),
        "under_refinement_rate": np.array(
            [[
                test_metrics[
                    "under_refinement_rate"
                ]
            ]],
            dtype=np.float64,
        ),
        "over_refinement_rate": np.array(
            [[
                test_metrics[
                    "over_refinement_rate"
                ]
            ]],
            dtype=np.float64,
        ),
        "ceil_level_mae": np.array(
            [[test_metrics["ceil_level_mae"]]],
            dtype=np.float64,
        ),
        "mean_per_sample_relative_l2": np.array(
            [[relative_l2_metrics["mean"]]],
            dtype=np.float64,
        ),
        "median_per_sample_relative_l2": np.array(
            [[relative_l2_metrics["median"]]],
            dtype=np.float64,
        ),
        "worst_per_sample_relative_l2": np.array(
            [[relative_l2_metrics["worst"]]],
            dtype=np.float64,
        ),
        "per_sample_relative_l2": np.asarray(
            per_sample_relative_l2,
            dtype=np.float64,
        ).reshape(-1, 1),
    }

    export_predictions(
        prediction_path,
        data,
        clipped,
        test_indices,
        train_indices,
        maximum_generation,
        generation_threshold,
        inference_time,
        extras,
    )

    return prediction_path
