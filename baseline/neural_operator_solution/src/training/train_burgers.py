#!/usr/bin/env python3
"""Train a neural operator to predict the Burgers solution u(x,t)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import CNO, DeepONet, FNO2dSolutionOperator, PODCoefficientNet


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


@dataclass(frozen=True)
class DataBundle:
    initial_condition: np.ndarray
    forcing: np.ndarray
    sensor_x: np.ndarray
    initial_condition_sensors: np.ndarray
    forcing_sensors: np.ndarray
    solution: np.ndarray
    query_x: np.ndarray
    query_t: np.ndarray
    source_ids: np.ndarray


@dataclass(frozen=True)
class ScalarNormalizer:
    mean: float
    std: float

    def encode(self, value: np.ndarray) -> np.ndarray:
        return np.asarray((value - self.mean) / self.std, dtype=np.float32)

    def decode(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value * self.std + self.mean, dtype=np.float32)

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


def load_data(path: Path) -> DataBundle:
    with h5py.File(path, "r") as handle:
        required = {
            "initial_condition",
            "forcing",
            "sensor_x",
            "initial_condition_sensors",
            "forcing_sensors",
            "solution",
            "query_x",
            "query_t",
            "source_attempt_id",
        }
        missing = required.difference(handle.keys())
        if missing:
            raise KeyError(f"Missing spectral-data fields: {sorted(missing)}")
        completed_attr = np.asarray(
            handle.attrs.get("completed_samples", 0)
        ).reshape(-1)
        completed = int(completed_attr[0]) if completed_attr.size else 0
        if completed != 3100:
            raise RuntimeError(
                f"Expected 3100 completed spectral samples, found {completed}."
            )
        initial = np.asarray(handle["initial_condition"], dtype=np.float32)
        forcing = np.asarray(handle["forcing"], dtype=np.float32)
        sensor_x = np.asarray(handle["sensor_x"], dtype=np.float32).reshape(-1)
        initial_sensors = np.asarray(
            handle["initial_condition_sensors"], dtype=np.float32
        )
        forcing_sensors = np.asarray(handle["forcing_sensors"], dtype=np.float32)
        solution = np.asarray(handle["solution"], dtype=np.float32)
        query_x = np.asarray(handle["query_x"], dtype=np.float32).reshape(-1)
        query_t = np.asarray(handle["query_t"], dtype=np.float32).reshape(-1)
        source_ids = np.asarray(handle["source_attempt_id"], dtype=np.int64).reshape(-1)

    expected = (3100, query_x.size, query_t.size)
    if initial.shape != expected[:2] or forcing.shape != expected[:2]:
        raise ValueError("Initial-condition or forcing shape is inconsistent.")
    if sensor_x.size != 256:
        raise ValueError(f"Expected 256 branch sensors, found {sensor_x.size}.")
    if initial_sensors.shape != (3100, 256) or forcing_sensors.shape != (3100, 256):
        raise ValueError("The 256-point branch-sensor fields are inconsistent.")
    if solution.shape != expected:
        raise ValueError(f"Solution shape {solution.shape}, expected {expected}.")
    if source_ids.size != 3100:
        raise ValueError("source_attempt_id must contain 3100 entries.")
    for name, value in (
        ("initial_condition", initial),
        ("forcing", forcing),
        ("initial_condition_sensors", initial_sensors),
        ("forcing_sensors", forcing_sensors),
        ("solution", solution),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf.")
    ic_error = float(np.max(np.abs(solution[:, :, 0] - initial)))
    if ic_error > 2.0e-5:
        raise ValueError(f"Solution/initial-condition mismatch: {ic_error:.3e}.")
    return DataBundle(
        initial,
        forcing,
        sensor_x,
        initial_sensors,
        forcing_sensors,
        solution,
        query_x,
        query_t,
        source_ids,
    )


def scalar_normalizer(value: np.ndarray) -> ScalarNormalizer:
    mean = float(np.mean(value, dtype=np.float64))
    std = max(float(np.std(value, dtype=np.float64)), 1.0e-8)
    return ScalarNormalizer(mean, std)


def normalized_coordinates(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xn = 2.0 * (x - x.min()) / max(float(x.max() - x.min()), 1.0e-12) - 1.0
    tn = 2.0 * (t - t.min()) / max(float(t.max() - t.min()), 1.0e-12) - 1.0
    xx, tt = np.meshgrid(xn, tn, indexing="ij")
    coordinates = np.stack((xx.reshape(-1), tt.reshape(-1)), axis=1)
    return xn.astype(np.float32), tn.astype(np.float32), coordinates.astype(np.float32)


class GridDataset(Dataset):
    def __init__(
        self,
        initial: np.ndarray,
        forcing: np.ndarray,
        target: np.ndarray,
        indices: np.ndarray,
        x_channel: np.ndarray,
        t_channel: np.ndarray,
    ) -> None:
        self.initial = initial
        self.forcing = forcing
        self.target = target
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_channel = torch.from_numpy(x_channel)
        self.t_channel = torch.from_numpy(t_channel)

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        nx, nt = self.target.shape[1:]
        u0 = np.broadcast_to(self.initial[index, :, None], (nx, nt))
        forcing = np.broadcast_to(self.forcing[index, :, None], (nx, nt))
        inputs = np.stack((u0, forcing, self.x_channel.numpy(), self.t_channel.numpy()))
        return (
            torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)),
            torch.from_numpy(self.target[index][None, ...]),
        )


class BranchDataset(Dataset):
    def __init__(self, branch: np.ndarray, target: np.ndarray, indices: np.ndarray) -> None:
        self.branch = branch
        self.target = target.reshape(target.shape[0], -1)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return torch.from_numpy(self.branch[index]), torch.from_numpy(self.target[index])


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator if shuffle else None,
    )


def parameter_counts(model: nn.Module) -> dict[str, int]:
    tensor_elements = 0
    real_parameters = 0
    storage_bytes = 0
    complex_elements = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        tensor_elements += count
        multiplier = 2 if parameter.is_complex() else 1
        real_parameters += multiplier * count
        complex_elements += count if parameter.is_complex() else 0
        storage_bytes += count * parameter.element_size()
    return {
        "trainable_tensor_elements": int(tensor_elements),
        "complex_trainable_elements": int(complex_elements),
        "real_trainable_parameters": int(real_parameters),
        "parameter_storage_bytes": int(storage_bytes),
    }


def make_optimizer(
    model: nn.Module, training: dict[str, Any]
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1.0e-3)),
        weight_decay=float(training.get("weight_decay", 1.0e-6)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(training.get("epochs", 1000)), 1),
        eta_min=float(training.get("eta_min", 1.0e-5)),
    )
    return optimizer, scheduler


def maybe_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    resume: bool,
) -> tuple[int, list[dict[str, float]]]:
    if not resume or not path.exists():
        return 1, []
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return int(state["epoch"]) + 1, list(state.get("history", []))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def train_grid_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    training: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    resume: bool,
) -> tuple[np.ndarray, list[dict[str, float]], float, float]:
    optimizer, scheduler = make_optimizer(model, training)
    checkpoint = out_dir / "checkpoint_last.pt"
    start_epoch, history = maybe_resume(
        checkpoint, model, optimizer, scheduler, device, resume
    )
    epochs = int(training["epochs"])
    grad_clip = float(training.get("grad_clip", 0.0))
    save_every = int(training.get("save_every", 50))
    log_every = int(training.get("log_every", 10))
    amp = bool(training.get("bf16_amp", False)) and device.type == "cuda"
    wall = time.perf_counter()
    log_wall = wall
    last_log_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for inputs, target in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                prediction = model(inputs)
                loss = torch.mean((prediction - target).square())
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or Inf.")
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * inputs.shape[0]
            count += inputs.shape[0]
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_normalized_mse": loss_sum / max(count, 1),
        }
        history.append(row)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            now = time.perf_counter()
            interval_epochs = max(epoch - last_log_epoch, 1)
            print(
                f"[epoch {epoch:5d}/{epochs}] mse={row['train_normalized_mse']:.6e} "
                f"lr={row['lr']:.3e} interval={now - log_wall:.2f}s "
                f"sec/epoch={(now - log_wall) / interval_epochs:.2f} "
                f"elapsed={now - wall:.2f}s",
                flush=True,
            )
            log_wall = now
            last_log_epoch = epoch
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, history)

    training_time = time.perf_counter() - wall
    model.eval()
    predictions: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_wall = time.perf_counter()
    with torch.inference_mode():
        for inputs, _ in test_loader:
            prediction = model(inputs.to(device, non_blocking=True))
            predictions.append(prediction[:, 0].float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - inference_wall
    return np.concatenate(predictions, axis=0), history, training_time, inference_time


def train_deeponet(
    model: DeepONet,
    train_loader: DataLoader,
    test_branch: np.ndarray,
    coordinates: np.ndarray,
    training: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    resume: bool,
) -> tuple[np.ndarray, list[dict[str, float]], float, float]:
    optimizer, scheduler = make_optimizer(model, training)
    checkpoint = out_dir / "checkpoint_last.pt"
    start_epoch, history = maybe_resume(
        checkpoint, model, optimizer, scheduler, device, resume
    )
    epochs = int(training["epochs"])
    query_count = coordinates.shape[0]
    train_queries = min(int(training.get("train_query_points", 4096)), query_count)
    query_batch = int(training.get("query_batch_size", 4096))
    grad_clip = float(training.get("grad_clip", 0.0))
    save_every = int(training.get("save_every", 50))
    log_every = int(training.get("log_every", 10))
    coords = torch.from_numpy(coordinates).to(device)
    wall = time.perf_counter()
    log_wall = wall
    last_log_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for branch, target in train_loader:
            ids = torch.randperm(query_count, device=device)[:train_queries]
            branch = branch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(branch, coords[ids])
            loss = torch.mean((prediction - target[:, ids]).square())
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or Inf.")
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * branch.shape[0]
            count += branch.shape[0]
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_normalized_mse": loss_sum / max(count, 1),
        }
        history.append(row)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            now = time.perf_counter()
            interval_epochs = max(epoch - last_log_epoch, 1)
            print(
                f"[epoch {epoch:5d}/{epochs}] mse={row['train_normalized_mse']:.6e} "
                f"lr={row['lr']:.3e} interval={now - log_wall:.2f}s "
                f"sec/epoch={(now - log_wall) / interval_epochs:.2f} "
                f"elapsed={now - wall:.2f}s",
                flush=True,
            )
            log_wall = now
            last_log_epoch = epoch
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, history)

    training_time = time.perf_counter() - wall
    model.eval()
    sample_batch = int(training.get("inference_sample_batch_size", 8))
    output = np.empty((test_branch.shape[0], query_count), dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_wall = time.perf_counter()
    with torch.inference_mode():
        for sample_start in range(0, test_branch.shape[0], sample_batch):
            sample_end = min(sample_start + sample_batch, test_branch.shape[0])
            branch = torch.from_numpy(test_branch[sample_start:sample_end]).to(device)
            for query_start in range(0, query_count, query_batch):
                query_end = min(query_start + query_batch, query_count)
                pred = model(branch, coords[query_start:query_end])
                output[sample_start:sample_end, query_start:query_end] = (
                    pred.float().cpu().numpy()
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - inference_wall
    return output, history, training_time, inference_time


def randomized_pod(
    train_fields: np.ndarray,
    maximum_rank: int,
    oversampling: int,
    energy: float,
    seed: int,
    sample_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n = train_fields.shape[0]
    points = int(np.prod(train_fields.shape[1:]))
    maximum_rank = min(maximum_rank, n - 1)
    q_rank = min(maximum_rank + oversampling, n - 1)
    mean = np.zeros(points, dtype=np.float64)
    for start in range(0, n, sample_chunk):
        block = train_fields[start : start + sample_chunk].reshape(-1, points)
        mean += block.sum(axis=0, dtype=np.float64)
    mean = (mean / n).astype(np.float32)

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((points, q_rank), dtype=np.float32)
    sketch = np.empty((n, q_rank), dtype=np.float32)
    total_energy = 0.0
    for start in range(0, n, sample_chunk):
        block = train_fields[start : start + sample_chunk].reshape(-1, points) - mean
        sketch[start : start + block.shape[0]] = block @ omega
        total_energy += float(np.sum(np.square(block.astype(np.float64))))
    q, _ = np.linalg.qr(sketch, mode="reduced")
    compressed = np.zeros((q_rank, points), dtype=np.float32)
    for start in range(0, n, sample_chunk):
        block = train_fields[start : start + sample_chunk].reshape(-1, points) - mean
        compressed += q[start : start + block.shape[0]].T @ block
    _, singular_values, vt = np.linalg.svd(compressed, full_matrices=False)
    cumulative = np.cumsum(np.square(singular_values.astype(np.float64))) / max(
        total_energy, 1.0e-30
    )
    meeting = np.flatnonzero(cumulative >= energy)
    rank = int(meeting[0] + 1) if meeting.size else min(maximum_rank, vt.shape[0])
    rank = min(rank, maximum_rank)
    captured_energy = float(cumulative[rank - 1])
    if captured_energy < energy:
        warnings.warn(
            f"maximum_rank={maximum_rank} captures {captured_energy:.6f}, "
            f"below requested POD energy {energy:.6f}.",
            RuntimeWarning,
        )
    basis = np.ascontiguousarray(vt[:rank], dtype=np.float32)
    coefficients = np.empty((n, rank), dtype=np.float32)
    for start in range(0, n, sample_chunk):
        block = train_fields[start : start + sample_chunk].reshape(-1, points) - mean
        coefficients[start : start + block.shape[0]] = block @ basis.T
    details = {
        "rank": rank,
        "requested_energy": float(energy),
        "captured_energy": captured_energy,
    }
    return mean, basis, coefficients, details


def train_pod(
    model: PODCoefficientNet,
    branch_train: np.ndarray,
    coefficients: np.ndarray,
    branch_test: np.ndarray,
    field_mean: np.ndarray,
    basis: np.ndarray,
    training: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    seed: int,
    resume: bool,
) -> tuple[np.ndarray, list[dict[str, float]], float, float]:
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(branch_train), torch.from_numpy(coefficients)
    )
    loader = make_loader(
        dataset,
        int(training.get("batch_size", 8)),
        True,
        int(training.get("num_workers", 0)),
        seed + 17,
        device,
    )
    optimizer, scheduler = make_optimizer(model, training)
    checkpoint = out_dir / "checkpoint_last.pt"
    start_epoch, history = maybe_resume(
        checkpoint, model, optimizer, scheduler, device, resume
    )
    epochs = int(training["epochs"])
    grad_clip = float(training.get("grad_clip", 0.0))
    save_every = int(training.get("save_every", 50))
    log_every = int(training.get("log_every", 10))
    wall = time.perf_counter()
    log_wall = wall
    last_log_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for branch, target in loader:
            branch = branch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(branch)
            loss = torch.mean((prediction - target).square())
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or Inf.")
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * branch.shape[0]
            count += branch.shape[0]
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_coefficient_mse": loss_sum / max(count, 1),
        }
        history.append(row)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            now = time.perf_counter()
            interval_epochs = max(epoch - last_log_epoch, 1)
            print(
                f"[epoch {epoch:5d}/{epochs}] coefficient_mse="
                f"{row['train_coefficient_mse']:.6e} lr={row['lr']:.3e} "
                f"interval={now - log_wall:.2f}s "
                f"sec/epoch={(now - log_wall) / interval_epochs:.2f} "
                f"elapsed={now - wall:.2f}s",
                flush=True,
            )
            log_wall = now
            last_log_epoch = epoch
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, history)

    training_time = time.perf_counter() - wall
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_wall = time.perf_counter()
    with torch.inference_mode():
        predicted_coefficients = model(
            torch.from_numpy(branch_test).to(device)
        ).float().cpu().numpy()
    fields = predicted_coefficients @ basis + field_mean[None, :]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - inference_wall
    return fields, history, training_time, inference_time


def solution_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
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


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def build_grid_model(name: str, config: dict[str, Any]) -> nn.Module:
    if name == "fno":
        fno_config = dict(config)
        fno_config.pop("nx_input", None)
        return FNO2dSolutionOperator(**fno_config)
    if name == "cno":
        return CNO(dim=2, **config)
    raise ValueError(f"Unsupported grid model {name}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    seed_everything(seed, deterministic)
    training = dict(config["training"])
    model_cfg = dict(config["model"])
    model_name = str(model_cfg.pop("name")).lower()
    device = torch.device(training.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    data_path = Path(config["data"]["path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    data = load_data(data_path.resolve())
    train_indices = np.arange(0, 3000, dtype=np.int64)
    test_indices = np.arange(3000, 3100, dtype=np.int64)

    experiment = str(config["experiment"])
    out_dir = PROJECT_ROOT / "results" / "burgers" / model_name / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    # Match the mesh-score preprocessing: reconstruct both input functions on
    # 256 sensors, then linearly interpolate them to the 101-point operator grid.
    grid_initial = np.stack(
        [np.interp(data.query_x, data.sensor_x, row) for row in data.initial_condition_sensors],
        axis=0,
    ).astype(np.float32)
    grid_forcing = np.stack(
        [np.interp(data.query_x, data.sensor_x, row) for row in data.forcing_sensors],
        axis=0,
    ).astype(np.float32)
    u0_norm = scalar_normalizer(grid_initial[train_indices])
    forcing_norm = scalar_normalizer(grid_forcing[train_indices])
    # The mesh-score experiments optimize raw-target MSE. Keep the same
    # convention; only the target meaning changes from score to solution.
    target_norm = ScalarNormalizer(0.0, 1.0)
    initial = u0_norm.encode(grid_initial)
    forcing = forcing_norm.encode(grid_forcing)
    target = target_norm.encode(data.solution)
    xn, tn, coordinates = normalized_coordinates(data.query_x, data.query_t)
    grid_xn = xn
    if model_name == "fno":
        # The original Burgers FNO normalizes by the full periodic length 2,
        # whereas CNO/DeepONet use the sampled coordinate min/max.
        grid_xn = (
            2.0 * (data.query_x - float(data.query_x.min())) / 2.0 - 1.0
        ).astype(np.float32)
    x_channel = np.broadcast_to(
        grid_xn[:, None], (grid_xn.size, tn.size)
    ).astype(np.float32)
    t_channel = np.broadcast_to(tn[None, :], (xn.size, tn.size)).astype(np.float32)
    branch_raw = np.concatenate(
        (data.initial_condition_sensors, data.forcing_sensors), axis=1
    ).astype(np.float32)
    branch_mean = branch_raw[train_indices].mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    branch_std = np.maximum(
        branch_raw[train_indices].std(axis=0, dtype=np.float64).astype(np.float32),
        1.0e-6,
    )
    branch = np.asarray((branch_raw - branch_mean) / branch_std, dtype=np.float32)
    np.savez_compressed(
        out_dir / "branch_normalization.npz",
        sensor_x=data.sensor_x,
        mean=branch_mean,
        std=branch_std,
    )

    pod_details: dict[str, Any] | None = None
    if model_name in {"fno", "cno"}:
        model = build_grid_model(model_name, model_cfg).to(device)
        counts = parameter_counts(model)
        print(
            f"[parameters] tensor elements={counts['trainable_tensor_elements']:,} | "
            f"equivalent real parameters={counts['real_trainable_parameters']:,} | "
            f"complex elements={counts['complex_trainable_elements']:,}"
        )
        train_dataset = GridDataset(
            initial, forcing, target, train_indices, x_channel, t_channel
        )
        test_dataset = GridDataset(
            initial, forcing, target, test_indices, x_channel, t_channel
        )
        train_loader = make_loader(
            train_dataset,
            int(training.get("batch_size", 16)),
            True,
            int(training.get("num_workers", 0)),
            seed + 1,
            device,
        )
        test_loader = make_loader(
            test_dataset,
            int(training.get("inference_sample_batch_size", 8)),
            False,
            int(training.get("num_workers", 0)),
            seed + 2,
            device,
        )
        normalized_prediction, history, training_time, inference_time = train_grid_model(
            model,
            train_loader,
            test_loader,
            training,
            device,
            out_dir,
            args.resume,
        )
    elif model_name == "deeponet":
        model = DeepONet(
            branch_dim=branch.shape[1], coordinate_dim=2, **model_cfg
        ).to(device)
        counts = parameter_counts(model)
        print(
            f"[parameters] tensor elements={counts['trainable_tensor_elements']:,} | "
            f"equivalent real parameters={counts['real_trainable_parameters']:,} | "
            f"complex elements={counts['complex_trainable_elements']:,}"
        )
        dataset = BranchDataset(branch, target, train_indices)
        loader = make_loader(
            dataset,
            int(training.get("batch_size", 16)),
            True,
            int(training.get("num_workers", 0)),
            seed + 1,
            device,
        )
        flat_prediction, history, training_time, inference_time = train_deeponet(
            model,
            loader,
            branch[test_indices],
            coordinates,
            training,
            device,
            out_dir,
            args.resume,
        )
        normalized_prediction = flat_prediction.reshape(
            test_indices.size, data.query_x.size, data.query_t.size
        )
    elif model_name == "pod_deeponet":
        pod_cfg = dict(config["pod"])
        field_mean, basis, coefficients, pod_details = randomized_pod(
            target[train_indices],
            int(pod_cfg.get("maximum_rank", 196)),
            int(pod_cfg.get("oversampling", 16)),
            float(pod_cfg.get("energy", 0.99)),
            seed + 101,
            int(pod_cfg.get("sample_chunk", 32)),
        )
        model = PODCoefficientNet(
            branch_dim=branch.shape[1], rank=basis.shape[0], **model_cfg
        ).to(device)
        counts = parameter_counts(model)
        print(
            f"[parameters] tensor elements={counts['trainable_tensor_elements']:,} | "
            f"equivalent real parameters={counts['real_trainable_parameters']:,} | "
            f"complex elements={counts['complex_trainable_elements']:,}"
        )
        flat_prediction, history, training_time, inference_time = train_pod(
            model,
            branch[train_indices],
            coefficients,
            branch[test_indices],
            field_mean,
            basis,
            training,
            device,
            out_dir,
            seed,
            args.resume,
        )
        normalized_prediction = flat_prediction.reshape(
            test_indices.size, data.query_x.size, data.query_t.size
        )
        np.savez_compressed(
            out_dir / "pod_basis.npz",
            field_mean=field_mean,
            basis=basis,
            **pod_details,
        )
    else:
        raise ValueError(f"Unsupported model.name: {model_name}")

    prediction = target_norm.decode(normalized_prediction)
    target_test = data.solution[test_indices]
    metrics = solution_metrics(prediction, target_test)
    final = {
        "problem": "burgers_direct_solution",
        "model": model_name,
        "experiment": experiment,
        "seed": seed,
        "training_samples": 3000,
        "test_samples": 100,
        "train_indices_zero_based": [0, 2999],
        "test_indices_zero_based": [3000, 3099],
        "source_attempt_id_test_min": int(data.source_ids[test_indices].min()),
        "source_attempt_id_test_max": int(data.source_ids[test_indices].max()),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "initial_learning_rate": float(training.get("learning_rate", 1.0e-3)),
        "eta_min": float(training.get("eta_min", 1.0e-5)),
        "weight_decay": float(training.get("weight_decay", 1.0e-6)),
        "epochs": int(training["epochs"]),
        "training_time_sec": training_time,
        "test_inference_time_sec": inference_time,
        "mean_inference_time_sec_per_sample": inference_time / int(test_indices.size),
        "inference_throughput_samples_per_sec": int(test_indices.size) / inference_time,
        "normalization": {
            "initial_condition": u0_norm.as_dict(),
            "forcing": forcing_norm.as_dict(),
            "solution": target_norm.as_dict(),
            "branch_featurewise": {
                "dimension": int(branch.shape[1]),
                "mean_file": "branch_normalization.npz",
                "std_file": "branch_normalization.npz",
            },
        },
        "parameter_count": counts,
        "test_metrics": metrics,
        "pod": pod_details,
    }
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)
    write_history(out_dir / "training_history.csv", history)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "model_config": model_cfg,
            "normalization": final["normalization"],
            "branch_mean": branch_mean,
            "branch_std": branch_std,
            "parameter_count": counts,
        },
        out_dir / "final_model.pt",
    )
    with h5py.File(out_dir / "predictions.h5", "w") as handle:
        handle.create_dataset("prediction", data=prediction, compression="lzf")
        handle.create_dataset("target", data=target_test, compression="lzf")
        handle.create_dataset("query_x", data=data.query_x)
        handle.create_dataset("query_t", data=data.query_t)
        handle.create_dataset("source_attempt_id", data=data.source_ids[test_indices])
        for key, value in metrics.items():
            handle.attrs[key] = value
    print(json.dumps(final, indent=2))
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
