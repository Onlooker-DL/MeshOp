#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the 2-D FNO for Burgers refinement-score regression.

Paths are resolved from the project root. With the default configuration:
    python src/training/train_burgers_fno.py
reads data/burgers/burgers_5100.mat and writes result/burgers_fno/.
Use --data or --out-dir only when overriding these defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CODE_VERSION = "FNO2D_BURGERS_V1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "burgers" / "burgers_5100.mat"
DEFAULT_OUT = PROJECT_ROOT / "result" / "burgers_fno"
DEFAULT_FIGURES = PROJECT_ROOT / "figures" / "burgers_fno"


# ============================================================================
# Reproducibility and device helpers
# ============================================================================

def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


# ============================================================================
# Dataset loading
# ============================================================================

@dataclass
class DatasetArrays:
    ic_coefficients: np.ndarray
    target: np.ndarray
    query_x: np.ndarray
    query_t: np.ndarray
    grf_modes: np.ndarray
    grf_spectral_std: np.ndarray
    forcing_input: np.ndarray
    source_attempt_id: np.ndarray
    completed_samples: int


def read_h5_vector(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle:
        raise KeyError(f'Missing field "{name}" in MAT file.')
    return np.asarray(handle[name]).reshape(-1)


def read_h5_scalar(
    handle: h5py.File,
    name: str,
    default: int,
) -> int:
    if name not in handle:
        return int(default)

    values = np.asarray(handle[name]).reshape(-1)
    if values.size == 0:
        return int(default)

    return int(round(float(values[0])))


def canonicalize_coefficients(
    raw: np.ndarray,
    coefficient_dimension: int,
) -> np.ndarray:
    raw = np.asarray(raw)

    if raw.ndim != 2:
        raise ValueError(
            f"ic_coefficients must be 2-D, got shape {raw.shape}."
        )

    if raw.shape[1] == coefficient_dimension:
        coefficients = raw
    elif raw.shape[0] == coefficient_dimension:
        coefficients = raw.T
    else:
        raise ValueError(
            "Cannot identify the coefficient axis in "
            f"ic_coefficients shape {raw.shape}; expected coefficient "
            f"dimension {coefficient_dimension}."
        )

    return np.ascontiguousarray(coefficients, dtype=np.float32)


def canonicalize_target(
    raw: np.ndarray,
    n_samples: int,
    nx: int,
    nt: int,
    hdf5_layout: bool,
) -> np.ndarray:
    """
Train the 2-D FNO for Burgers refinement-score regression.

Paths are resolved from the project root. With the default configuration:
    python src/training/train_burgers_fno.py
reads data/burgers/burgers_5100.mat and writes result/burgers_fno/.
Use --data or --out-dir only when overriding these defaults.
"""
    raw = np.asarray(raw)

    if raw.ndim != 3:
        raise ValueError(
            f"target_score must be 3-D, got shape {raw.shape}."
        )

    if hdf5_layout:
        expected = (n_samples, nt, nx)
        if raw.shape != expected:
            raise ValueError(
                "For MATLAB v7.3, expected h5py target_score shape "
                f"{expected} = [sample,t,x], got {raw.shape}."
            )
        target = np.transpose(raw, (0, 2, 1))
        source_order = "[sample,t,x] from MATLAB v7.3/HDF5"
    else:
        expected = (nx, nt, n_samples)
        if raw.shape != expected:
            raise ValueError(
                "For a legacy MAT file, expected target_score shape "
                f"{expected} = [x,t,sample], got {raw.shape}."
            )
        target = np.transpose(raw, (2, 0, 1))
        source_order = "[x,t,sample] from legacy MAT"

    if target.shape != (n_samples, nx, nt):
        raise RuntimeError(
            f"Canonical target shape is {target.shape}; expected "
            f"({n_samples},{nx},{nt})."
        )

    print(
        f"[data] target orientation: {source_order} -> "
        "[sample,x,t]"
    )
    return np.ascontiguousarray(target, dtype=np.float32)


def load_mat_v73(path: Path) -> DatasetArrays:
    with h5py.File(path, "r") as handle:
        required = [
            "ic_coefficients",
            "grf_modes",
            "grf_spectral_std",
            "query_x",
            "query_t",
            "target_score",
            "forcing_input",
        ]
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"Missing MAT fields: {missing}")

        query_x = read_h5_vector(handle, "query_x").astype(np.float32)
        query_t = read_h5_vector(handle, "query_t").astype(np.float32)
        grf_modes = read_h5_vector(
            handle, "grf_modes"
        ).astype(np.float32)
        grf_spectral_std = read_h5_vector(
            handle, "grf_spectral_std"
        ).astype(np.float32)

        if grf_modes.size != grf_spectral_std.size:
            raise ValueError(
                "grf_modes and grf_spectral_std have different lengths."
            )

        coefficient_dimension = 2 + 2 * int(grf_modes.size)
        coefficients = canonicalize_coefficients(
            np.asarray(handle["ic_coefficients"], dtype=np.float32),
            coefficient_dimension,
        )
        n_samples = int(coefficients.shape[0])
        if "source_attempt_id" in handle:
            source_attempt_id = read_h5_vector(
                handle, "source_attempt_id"
            ).astype(np.int64)
        else:
            source_attempt_id = np.arange(
                1, n_samples + 1, dtype=np.int64
            )
        if source_attempt_id.size != n_samples:
            raise ValueError(
                "source_attempt_id length does not match the sample count."
            )

        target = canonicalize_target(
            np.asarray(handle["target_score"], dtype=np.float32),
            n_samples=n_samples,
            nx=int(query_x.size),
            nt=int(query_t.size),
            hdf5_layout=True,
        )
        forcing_raw=np.asarray(handle["forcing_input"],dtype=np.float32)
        forcing_input=forcing_raw if forcing_raw.shape==(n_samples,query_x.size) else forcing_raw.T
        if forcing_input.shape!=(n_samples,query_x.size):
            raise ValueError(f"forcing_input has incompatible shape {forcing_raw.shape}.")

        completed = read_h5_scalar(
            handle,
            "completedSamples",
            n_samples,
        )
        completed = min(
            completed,
            n_samples,
            target.shape[0],
        )

    return DatasetArrays(
        ic_coefficients=np.ascontiguousarray(
            coefficients[:completed],
            dtype=np.float32,
        ),
        target=np.ascontiguousarray(
            target[:completed],
            dtype=np.float32,
        ),
        query_x=np.ascontiguousarray(query_x, dtype=np.float32),
        query_t=np.ascontiguousarray(query_t, dtype=np.float32),
        grf_modes=np.ascontiguousarray(grf_modes, dtype=np.float32),
        grf_spectral_std=np.ascontiguousarray(
            grf_spectral_std,
            dtype=np.float32,
        ),
        forcing_input=np.ascontiguousarray(forcing_input[:completed],dtype=np.float32),
        source_attempt_id=np.ascontiguousarray(
            source_attempt_id[:completed], dtype=np.int64
        ),
        completed_samples=int(completed),
    )


def load_mat_legacy(path: Path) -> DatasetArrays:
    loaded = sio.loadmat(
        path,
        squeeze_me=False,
        struct_as_record=False,
    )

    required = [
        "ic_coefficients",
        "grf_modes",
        "grf_spectral_std",
        "query_x",
        "query_t",
        "target_score",
        "forcing_input",
    ]
    missing = [name for name in required if name not in loaded]
    if missing:
        raise KeyError(f"Missing MAT fields: {missing}")

    query_x = np.asarray(
        loaded["query_x"]
    ).reshape(-1).astype(np.float32)
    query_t = np.asarray(
        loaded["query_t"]
    ).reshape(-1).astype(np.float32)
    grf_modes = np.asarray(
        loaded["grf_modes"]
    ).reshape(-1).astype(np.float32)
    grf_spectral_std = np.asarray(
        loaded["grf_spectral_std"]
    ).reshape(-1).astype(np.float32)

    coefficient_dimension = 2 + 2 * int(grf_modes.size)
    coefficients = canonicalize_coefficients(
        np.asarray(loaded["ic_coefficients"], dtype=np.float32),
        coefficient_dimension,
    )
    n_samples = int(coefficients.shape[0])
    if "source_attempt_id" in loaded:
        source_attempt_id = np.asarray(
            loaded["source_attempt_id"]
        ).reshape(-1).astype(np.int64)
    else:
        source_attempt_id = np.arange(
            1, n_samples + 1, dtype=np.int64
        )
    if source_attempt_id.size != n_samples:
        raise ValueError(
            "source_attempt_id length does not match the sample count."
        )

    target = canonicalize_target(
        np.asarray(loaded["target_score"], dtype=np.float32),
        n_samples=n_samples,
        nx=int(query_x.size),
        nt=int(query_t.size),
        hdf5_layout=False,
    )
    forcing_raw=np.asarray(loaded["forcing_input"],dtype=np.float32)
    forcing_input=forcing_raw.T if forcing_raw.shape==(query_x.size,n_samples) else forcing_raw
    if forcing_input.shape!=(n_samples,query_x.size):
        raise ValueError(f"forcing_input has incompatible shape {forcing_raw.shape}.")

    completed = n_samples
    if "completedSamples" in loaded:
        completed = min(
            completed,
            int(
                round(
                    float(
                        np.asarray(
                            loaded["completedSamples"]
                        ).reshape(-1)[0]
                    )
                )
            ),
        )
    completed = min(completed, target.shape[0])

    return DatasetArrays(
        ic_coefficients=np.ascontiguousarray(
            coefficients[:completed],
            dtype=np.float32,
        ),
        target=np.ascontiguousarray(
            target[:completed],
            dtype=np.float32,
        ),
        query_x=np.ascontiguousarray(query_x, dtype=np.float32),
        query_t=np.ascontiguousarray(query_t, dtype=np.float32),
        grf_modes=np.ascontiguousarray(grf_modes, dtype=np.float32),
        grf_spectral_std=np.ascontiguousarray(
            grf_spectral_std,
            dtype=np.float32,
        ),
        forcing_input=np.ascontiguousarray(forcing_input[:completed],dtype=np.float32),
        source_attempt_id=np.ascontiguousarray(
            source_attempt_id[:completed], dtype=np.int64
        ),
        completed_samples=int(completed),
    )


def load_dataset(path: Path) -> DatasetArrays:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        data = load_mat_v73(path)
        source = "MATLAB v7.3/HDF5"
    except OSError:
        data = load_mat_legacy(path)
        source = "legacy MAT"

    if data.ic_coefficients.shape[0] != data.target.shape[0]:
        raise ValueError(
            "Sample mismatch between ic_coefficients and target_score."
        )
    if not np.all(np.isfinite(data.ic_coefficients)):
        raise ValueError("ic_coefficients contains NaN or Inf.")
    if not np.all(np.isfinite(data.target)):
        raise ValueError("target_score contains NaN or Inf.")
    if not np.all(np.isfinite(data.grf_spectral_std)):
        raise ValueError("grf_spectral_std contains NaN or Inf.")
    if not np.all(np.isfinite(data.forcing_input)):
        raise ValueError("forcing_input contains NaN or Inf.")
    if data.source_attempt_id.shape[0] != data.completed_samples:
        raise ValueError("source_attempt_id has the wrong length.")

    print(f"[data] source          : {source}")
    print(f"[data] file            : {path}")
    print(f"[data] completed       : {data.completed_samples}")
    print(f"[data] coefficients    : {data.ic_coefficients.shape}")
    print(f"[data] target          : {data.target.shape}")
    print(
        f"[data] source attempts : "
        f"[{int(data.source_attempt_id.min())}, "
        f"{int(data.source_attempt_id.max())}]"
    )
    print(
        f"[data] query grid      : "
        f"{data.query_x.size} x {data.query_t.size}"
    )
    print(
        f"[data] target range    : "
        f"[{float(data.target.min()):.3f}, "
        f"{float(data.target.max()):.3f}]"
    )

    return data


# ============================================================================
# Initial-condition reconstruction and exact split
# ============================================================================

def reconstruct_u0_at_points(
    data: DatasetArrays,
    x_points: np.ndarray,
) -> np.ndarray:
    """
Train the 2-D FNO for Burgers refinement-score regression.

Paths are resolved from the project root. With the default configuration:
    python src/training/train_burgers_fno.py
reads data/burgers/burgers_5100.mat and writes result/burgers_fno/.
Use --data or --out-dir only when overriding these defaults.
"""
    coefficients = np.asarray(
        data.ic_coefficients,
        dtype=np.float64,
    )
    modes = np.asarray(
        data.grf_modes,
        dtype=np.float64,
    ).reshape(-1)
    spectral_std = np.asarray(
        data.grf_spectral_std,
        dtype=np.float64,
    ).reshape(-1)
    x_points = np.asarray(
        x_points,
        dtype=np.float64,
    ).reshape(-1)

    k_count = int(modes.size)
    expected_dimension = 2 + 2 * k_count
    if coefficients.shape[1] != expected_dimension:
        raise ValueError(
            f"Coefficient dimension {coefficients.shape[1]} does not "
            f"match expected {expected_dimension}."
        )

    amplitude = coefficients[:, 0:1]
    shift = coefficients[:, 1:2]
    xi_cos = coefficients[:, 2 : 2 + k_count]
    xi_sin = coefficients[
        :,
        2 + k_count : 2 + 2 * k_count,
    ]

    phase = np.pi * np.outer(x_points, modes)
    grf = (
        (xi_cos * spectral_std[None, :]) @ np.cos(phase).T
        + (xi_sin * spectral_std[None, :]) @ np.sin(phase).T
    )
    backbone = -amplitude * np.sin(
        np.pi * (x_points[None, :] + shift)
    )
    u0 = backbone + grf

    if not np.all(np.isfinite(u0)):
        raise ValueError(
            "Exactly reconstructed U0 contains NaN or Inf."
        )

    return np.ascontiguousarray(u0, dtype=np.float32)


def reconstruct_u0_on_discrete_grid(
    data: DatasetArrays,
    nx_input: int,
    xmin: float = -1.0,
    xmax: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if nx_input < 4:
        raise ValueError("--nx-input must be at least four.")

    coefficients = np.asarray(
        data.ic_coefficients,
        dtype=np.float64,
    )
    modes = np.asarray(
        data.grf_modes,
        dtype=np.float64,
    ).reshape(-1)
    spectral_std = np.asarray(
        data.grf_spectral_std,
        dtype=np.float64,
    ).reshape(-1)

    k_count = int(modes.size)
    expected_dimension = 2 + 2 * k_count
    if coefficients.shape[1] != expected_dimension:
        raise ValueError(
            f"Coefficient dimension {coefficients.shape[1]} does not "
            f"match expected {expected_dimension}."
        )

    x_input = np.linspace(
        xmin,
        xmax,
        num=int(nx_input),
        endpoint=False,
        dtype=np.float64,
    )

    amplitude = coefficients[:, 0:1]
    shift = coefficients[:, 1:2]
    xi_cos = coefficients[:, 2 : 2 + k_count]
    xi_sin = coefficients[
        :,
        2 + k_count : 2 + 2 * k_count,
    ]

    phase = np.pi * np.outer(x_input, modes)
    cos_basis = np.cos(phase)
    sin_basis = np.sin(phase)

    weighted_cos = xi_cos * spectral_std[None, :]
    weighted_sin = xi_sin * spectral_std[None, :]

    grf = (
        weighted_cos @ cos_basis.T
        + weighted_sin @ sin_basis.T
    )
    backbone = -amplitude * np.sin(
        np.pi * (x_input[None, :] + shift)
    )
    u0 = backbone + grf

    if not np.all(np.isfinite(u0)):
        raise ValueError("Reconstructed U0 contains NaN or Inf.")

    return (
        np.ascontiguousarray(x_input, dtype=np.float32),
        np.ascontiguousarray(u0, dtype=np.float32),
    )


def periodic_interpolate_u0(
    u0: np.ndarray,
    x_input: np.ndarray,
    query_x: np.ndarray,
    xmin: float = -1.0,
    xmax: float = 1.0,
) -> np.ndarray:
    if u0.ndim != 2:
        raise ValueError("u0 must have shape [sample,Nx_input].")

    x_input_double = np.asarray(
        x_input,
        dtype=np.float64,
    ).reshape(-1)
    query_double = np.asarray(
        query_x,
        dtype=np.float64,
    ).reshape(-1)

    if x_input_double.size != u0.shape[1]:
        raise ValueError(
            "x_input length and the U0 sensor dimension differ."
        )

    period = float(xmax - xmin)
    expected = (
        xmin
        + period
        * np.arange(x_input_double.size, dtype=np.float64)
        / x_input_double.size
    )
    if not np.allclose(
        x_input_double,
        expected,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "x_input must be a uniform periodic grid on [xmin,xmax)."
        )

    wrapped = xmin + np.mod(query_double - xmin, period)
    position = (
        (wrapped - xmin)
        * x_input_double.size
        / period
    )
    floor_position = np.floor(position)
    left = floor_position.astype(np.int64) % x_input_double.size
    right = (left + 1) % x_input_double.size
    alpha = (position - floor_position).astype(np.float32)

    interpolated = (
        (1.0 - alpha[None, :]) * u0[:, left]
        + alpha[None, :] * u0[:, right]
    )

    return np.ascontiguousarray(
        interpolated,
        dtype=np.float32,
    )


def make_exact_split(
    n_samples: int,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if train_samples + test_samples != n_samples:
        raise ValueError(
            "Exact train/test counts must equal the number of loaded "
            f"samples: train={train_samples}, test={test_samples}, "
            f"loaded={n_samples}."
        )

    if train_samples < 1 or test_samples < 1:
        raise ValueError(
            "Training and test sample counts must be positive."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_samples).astype(np.int64)
    train_indices = permutation[:train_samples]
    test_indices = permutation[train_samples:]

    return train_indices, test_indices


def load_or_create_split(
    out_dir: Path,
    n_samples: int,
    train_samples: int,
    test_samples: int,
    seed: int,
    reuse_existing: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    split_path = out_dir / "split_indices.npz"

    if reuse_existing and split_path.exists():
        loaded = np.load(split_path)
        train_indices = np.asarray(
            loaded["train_indices"],
            dtype=np.int64,
        ).reshape(-1)
        test_indices = np.asarray(
            loaded["test_indices"],
            dtype=np.int64,
        ).reshape(-1)
    else:
        train_indices, test_indices = make_exact_split(
            n_samples,
            train_samples,
            test_samples,
            seed,
        )
        np.savez(
            split_path,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    if train_indices.size != train_samples:
        raise ValueError(
            f"Saved training split has {train_indices.size} samples, "
            f"expected {train_samples}."
        )
    if test_indices.size != test_samples:
        raise ValueError(
            f"Saved test split has {test_indices.size} samples, "
            f"expected {test_samples}."
        )
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Training and test indices overlap.")
    if np.unique(
        np.concatenate((train_indices, test_indices))
    ).size != n_samples:
        raise ValueError(
            "The split does not cover every loaded sample exactly once."
        )

    return train_indices, test_indices


# ============================================================================
# Score weighting and PyTorch dataset
# ============================================================================

def compute_score_weights(
    target: np.ndarray,
    train_indices: np.ndarray,
    max_score: int,
    power: float,
    chunk_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    histogram = np.zeros(max_score + 1, dtype=np.int64)

    for start in range(0, train_indices.size, chunk_size):
        ids = train_indices[start : start + chunk_size]
        values = np.rint(target[ids]).astype(
            np.int16,
            copy=False,
        )
        values = np.clip(values, 0, max_score)
        histogram += np.bincount(
            values.reshape(-1),
            minlength=max_score + 1,
        )

    if power <= 0:
        weights = np.ones(
            max_score + 1,
            dtype=np.float32,
        )
    else:
        frequency = histogram.astype(np.float64)
        positive = frequency > 0
        reference = (
            float(np.mean(frequency[positive]))
            if np.any(positive)
            else 1.0
        )

        weights_double = np.ones(
            max_score + 1,
            dtype=np.float64,
        )
        weights_double[positive] = (
            reference / frequency[positive]
        ) ** power
        weights_double[positive] /= np.mean(
            weights_double[positive]
        )
        weights = weights_double.astype(np.float32)

    return histogram, weights


class BurgersScoreDataset(Dataset):
    def __init__(
        self,
        u0_query: np.ndarray,
        forcing_query: np.ndarray,
        target: np.ndarray,
        indices: np.ndarray,
        query_x: np.ndarray,
        query_t: np.ndarray,
        u_mean: float,
        u_std: float,
        forcing_mean: float,
        forcing_std: float,
    ) -> None:
        self.u0_query = u0_query
        self.forcing_query = forcing_query
        self.target = target
        self.indices = np.asarray(indices, dtype=np.int64)
        self.u_mean = float(u_mean)
        self.u_std = float(max(u_std, 1.0e-8))
        self.forcing_mean=float(forcing_mean)
        self.forcing_std=float(max(forcing_std,1.0e-8))

        x_min = float(np.min(query_x))
        x_period = 2.0
        x_normalized = (
            2.0
            * (np.asarray(query_x) - x_min)
            / x_period
            - 1.0
        )

        t_min = float(np.min(query_t))
        t_max = float(np.max(query_t))
        t_normalized = (
            2.0
            * (np.asarray(query_t) - t_min)
            / max(t_max - t_min, 1.0e-12)
            - 1.0
        )

        xx, tt = np.meshgrid(
            x_normalized.astype(np.float32),
            t_normalized.astype(np.float32),
            indexing="ij",
        )
        self.x_channel = torch.from_numpy(xx)
        self.t_channel = torch.from_numpy(tt)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(
        self,
        item: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])

        u_line = (
            self.u0_query[index] - self.u_mean
        ) / self.u_std
        u_field = np.repeat(
            u_line[:, None],
            self.target.shape[2],
            axis=1,
        )
        forcing_line=(self.forcing_query[index]-self.forcing_mean)/self.forcing_std
        forcing_field=np.repeat(forcing_line[:,None],self.target.shape[2],axis=1)

        input_tensor = torch.stack(
            (
                torch.from_numpy(
                    np.asarray(u_field, dtype=np.float32)
                ),
                torch.from_numpy(np.asarray(forcing_field,dtype=np.float32)),
                self.x_channel,
                self.t_channel,
            ),
            dim=0,
        )
        target_tensor = torch.from_numpy(
            np.asarray(
                self.target[index],
                dtype=np.float32,
            )
        ).unsqueeze(0)

        return input_tensor, target_tensor, index


# ============================================================================
# FNO model
# ============================================================================

class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_t: int,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_t = int(modes_t)

        scale = 1.0 / math.sqrt(
            max(in_channels * out_channels, 1)
        )
        shape = (
            in_channels,
            out_channels,
            modes_x,
            modes_t,
        )

        self.weight_positive = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def complex_multiply(
        input_fourier: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bixy,ioxy->boxy",
            input_fourier,
            weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, nt = x.shape
        input_dtype = x.dtype

        # FFTs are always performed in float32/complex64. This avoids
        # half-precision cuFFT restrictions on the non-power-of-two
        # 101 x (101 + padding) grid.
        with torch.amp.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            x_fourier = torch.fft.rfft2(
                x.float(),
                norm="ortho",
            )
            nt_rfft = x_fourier.shape[-1]

            active_modes_x = min(
                self.modes_x,
                nx // 2,
            )
            active_modes_t = min(
                self.modes_t,
                nt_rfft,
            )

            output_fourier = torch.zeros(
                batch,
                self.out_channels,
                nx,
                nt_rfft,
                dtype=torch.cfloat,
                device=x.device,
            )

            if active_modes_x > 0 and active_modes_t > 0:
                output_fourier[
                    :,
                    :,
                    :active_modes_x,
                    :active_modes_t,
                ] = self.complex_multiply(
                    x_fourier[
                        :,
                        :,
                        :active_modes_x,
                        :active_modes_t,
                    ],
                    self.weight_positive[
                        :,
                        :,
                        :active_modes_x,
                        :active_modes_t,
                    ],
                )

                output_fourier[
                    :,
                    :,
                    -active_modes_x:,
                    :active_modes_t,
                ] = self.complex_multiply(
                    x_fourier[
                        :,
                        :,
                        -active_modes_x:,
                        :active_modes_t,
                    ],
                    self.weight_negative[
                        :,
                        :,
                        :active_modes_x,
                        :active_modes_t,
                    ],
                )

            output = torch.fft.irfft2(
                output_fourier,
                s=(nx, nt),
                norm="ortho",
            )

        return output.to(dtype=input_dtype)


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_t: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes_x,
            modes_t,
        )
        self.local = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )
        self.normalization = nn.GroupNorm(1, width)
        self.dropout = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.spectral(x) + self.local(x)
        output = self.normalization(output)
        output = F.gelu(output)
        return self.dropout(output)


class FNO2dScoreRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        modes_x: int,
        modes_t: int,
        layers: int,
        projection_width: int,
        padding_t: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.padding_t = int(max(padding_t, 0))
        self.lifting = nn.Conv2d(
            in_channels,
            width,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    width,
                    modes_x,
                    modes_t,
                    dropout,
                )
                for _ in range(layers)
            ]
        )
        self.projection_one = nn.Conv2d(
            width,
            projection_width,
            kernel_size=1,
        )
        self.projection_two = nn.Conv2d(
            projection_width,
            1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is periodic, but time is not. Pad only time.
        if self.padding_t > 0:
            x = F.pad(
                x,
                (0, self.padding_t, 0, 0),
            )

        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)

        x = F.gelu(self.projection_one(x))
        x = self.projection_two(x)

        if self.padding_t > 0:
            x = x[..., :-self.padding_t]

        return x


# ============================================================================
# Loss and metrics
# ============================================================================

class AsymmetricMSEScoreLoss(nn.Module):
    def __init__(
        self,
        score_weights: Sequence[float],
        max_score: int,
        under_weight: float,
        bound_penalty: float,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "score_weights",
            torch.as_tensor(
                score_weights,
                dtype=torch.float32,
            ),
        )
        self.max_score = float(max_score)
        self.under_weight = float(under_weight)
        self.bound_penalty = float(bound_penalty)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        error = prediction - target

        asymmetric_factor = torch.where(
            error < 0,
            torch.full_like(
                error,
                self.under_weight,
            ),
            torch.ones_like(error),
        )

        bins = torch.clamp(
            torch.round(target).long(),
            min=0,
            max=self.score_weights.numel() - 1,
        )
        score_weight = self.score_weights[bins]

        fitting_loss = torch.mean(
            score_weight
            * asymmetric_factor
            * error.square()
        )

        lower_violation = F.relu(-prediction).square()
        upper_violation = F.relu(
            prediction - self.max_score
        ).square()
        range_loss = torch.mean(
            lower_violation + upper_violation
        )

        return (
            fitting_loss
            + self.bound_penalty * range_loss
        )



def score_to_generation_torch(
    score: torch.Tensor,
    max_score: int,
    threshold: float,
) -> torch.Tensor:
    """
Train the 2-D FNO for Burgers refinement-score regression.

Paths are resolved from the project root. With the default configuration:
    python src/training/train_burgers_fno.py
reads data/burgers/burgers_5100.mat and writes result/burgers_fno/.
Use --data or --out-dir only when overriding these defaults.
"""
    score_clamped = torch.clamp(
        score,
        0.0,
        float(max_score),
    )
    generation = torch.floor(
        score_clamped
        + (1.0 - float(threshold))
        + 1.0e-6
    )
    return torch.clamp(
        generation,
        0.0,
        float(max_score),
    )


def score_to_generation_numpy(
    score: np.ndarray,
    max_score: int,
    threshold: float,
) -> np.ndarray:
    """
Train the 2-D FNO for Burgers refinement-score regression.

Paths are resolved from the project root. With the default configuration:
    python src/training/train_burgers_fno.py
reads data/burgers/burgers_5100.mat and writes result/burgers_fno/.
Use --data or --out-dir only when overriding these defaults.
"""
    score_clamped = np.clip(
        np.asarray(score),
        0.0,
        float(max_score),
    )
    generation = np.floor(
        score_clamped
        + (1.0 - float(threshold))
        + 1.0e-6
    )
    return np.clip(
        generation,
        0,
        max_score,
    ).astype(np.uint8)


@dataclass
class MetricTotals:
    count: int = 0
    absolute_sum: float = 0.0
    squared_sum: float = 0.0
    rounded_correct: int = 0
    within_half: int = 0
    within_one: int = 0
    under_count: int = 0
    over_count: int = 0
    ceil_absolute_sum: float = 0.0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        max_score: int,
        generation_threshold: float,
    ) -> None:
        prediction_clamped = torch.clamp(
            prediction.detach(),
            0.0,
            float(max_score),
        )
        target_detached = target.detach()
        difference = prediction_clamped - target_detached
        count = int(difference.numel())

        prediction_generation = score_to_generation_torch(
            prediction_clamped,
            max_score=max_score,
            threshold=generation_threshold,
        )
        target_generation = score_to_generation_torch(
            target_detached,
            max_score=max_score,
            threshold=generation_threshold,
        )

        self.count += count
        self.absolute_sum += float(
            torch.sum(torch.abs(difference)).item()
        )
        self.squared_sum += float(
            torch.sum(difference.square()).item()
        )
        self.rounded_correct += int(
            torch.sum(
                prediction_generation == target_generation
            ).item()
        )
        self.within_half += int(
            torch.sum(
                torch.abs(difference) <= 0.5
            ).item()
        )
        self.within_one += int(
            torch.sum(
                torch.abs(difference) <= 1.0
            ).item()
        )
        self.under_count += int(
            torch.sum(
                prediction_generation < target_generation
            ).item()
        )
        self.over_count += int(
            torch.sum(
                prediction_generation > target_generation
            ).item()
        )
        self.ceil_absolute_sum += float(
            torch.sum(
                torch.abs(
                    prediction_generation - target_generation
                )
            ).item()
        )

    def finalize(self) -> Dict[str, float]:
        denominator = max(self.count, 1)

        return {
            "mae": self.absolute_sum / denominator,
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
                self.ceil_absolute_sum / denominator
            ),
        }


# ============================================================================
# DataLoader, training and evaluation
# ============================================================================

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def use_bf16_amp_requested(
    requested: bool,
    device: torch.device,
) -> bool:
    if not requested:
        return False
    if device.type != "cuda":
        return False
    return bool(torch.cuda.is_bf16_supported())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_score: int,
    generation_threshold: float,
    grad_clip: float,
    use_bf16_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    model.train()

    loss_sum = 0.0
    sample_count = 0
    totals = MetricTotals()

    for input_tensor, target, _ in loader:
        input_tensor = input_tensor.to(
            device,
            non_blocking=True,
        )
        target = target.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16_amp,
        ):
            prediction = model(input_tensor)
            loss = criterion(prediction, target)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Training loss became NaN or Inf."
            )

        # No GradScaler is used because the Fourier parameters are complex64.
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

        optimizer.step()

        batch_size = int(input_tensor.shape[0])
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        totals.update(
            prediction,
            target,
            max_score,
            generation_threshold,
        )

    return (
        loss_sum / max(sample_count, 1),
        totals.finalize(),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_score: int,
    generation_threshold: float,
    use_bf16_amp: bool,
    collect_predictions: bool,
) -> Tuple[
    float,
    Dict[str, float],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    model.eval()

    loss_sum = 0.0
    sample_count = 0
    totals = MetricTotals()

    predictions = []
    original_indices = []

    for input_tensor, target, indices in loader:
        input_tensor = input_tensor.to(
            device,
            non_blocking=True,
        )
        target = target.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16_amp,
        ):
            prediction = model(input_tensor)
            loss = criterion(prediction, target)

        batch_size = int(input_tensor.shape[0])
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        totals.update(
            prediction,
            target,
            max_score,
            generation_threshold,
        )

        if collect_predictions:
            predictions.append(
                torch.clamp(
                    prediction,
                    0.0,
                    float(max_score),
                )
                .squeeze(1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            original_indices.append(
                np.asarray(indices, dtype=np.int64)
            )

    prediction_array = (
        np.concatenate(predictions, axis=0)
        if collect_predictions
        else None
    )
    index_array = (
        np.concatenate(original_indices, axis=0)
        if collect_predictions
        else None
    )

    return (
        loss_sum / max(sample_count, 1),
        totals.finalize(),
        prediction_array,
        index_array,
    )


@torch.inference_mode()
def measure_inference_time(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_bf16_amp: bool,
    repeats: int,
) -> Dict[str, float]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
        seed=0,
    )

    model.eval()

    # Warm-up.
    for warmup_id, (input_tensor, _, _) in enumerate(loader):
        input_tensor = input_tensor.to(
            device,
            non_blocking=True,
        )
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16_amp,
        ):
            output = model(input_tensor)
        _ = output.detach().cpu().numpy()
        if warmup_id >= 2:
            break

    synchronize(device)

    repeat_times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()

        for input_tensor, _, _ in loader:
            input_tensor = input_tensor.to(
                device,
                non_blocking=True,
            )
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16_amp,
            ):
                output = model(input_tensor)
            _ = output.detach().cpu().numpy()

        synchronize(device)
        repeat_times.append(
            time.perf_counter() - start
        )

    median_total = float(np.median(repeat_times))
    n_samples = len(dataset)

    return {
        "total_test_inference_time_sec": median_total,
        "mean_inference_time_sec_per_sample": (
            median_total / max(n_samples, 1)
        ),
        "throughput_samples_per_sec": (
            n_samples / max(median_total, 1.0e-12)
        ),
        "timing_repeats": int(repeats),
        "timed_samples": int(n_samples),
    }


# ============================================================================
# Checkpoints and history
# ============================================================================

def checkpoint_state(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    training_time_sec: float,
    arguments: argparse.Namespace,
    model_config: Dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    u_mean: float,
    u_std: float,
    score_histogram: np.ndarray,
    score_weights: np.ndarray,
    x_input: np.ndarray,
    query_x: np.ndarray,
    query_t: np.ndarray,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "code_version": CODE_VERSION,
        "epoch": int(epoch),
        "training_time_sec": float(training_time_sec),
        "checkpoint_selection": (
            "fixed final epoch; test monitored periodically but not used for model selection"
        ),
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "arguments": {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(arguments).items()
        },
        "train_indices": torch.from_numpy(
            train_indices.astype(np.int64)
        ),
        "test_indices": torch.from_numpy(
            test_indices.astype(np.int64)
        ),
        "u_mean": float(u_mean),
        "u_std": float(u_std),
        "score_histogram": torch.from_numpy(
            score_histogram.astype(np.int64)
        ),
        "score_weights": torch.from_numpy(
            score_weights.astype(np.float32)
        ),
        "x_input": torch.from_numpy(
            x_input.astype(np.float32)
        ),
        "query_x": torch.from_numpy(
            query_x.astype(np.float32)
        ),
        "query_t": torch.from_numpy(
            query_t.astype(np.float32)
        ),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    return state


def save_history_csv(
    path: Path,
    history: Iterable[Dict[str, Any]],
) -> None:
    rows = list(history)
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def read_history_csv(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []

    def read_float(
        row: Dict[str, str],
        key: str,
        default: float = math.nan,
    ) -> float:
        value = row.get(key, "")
        if value is None or value == "":
            return float(default)
        return float(value)

    rows = []
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "lr": read_float(row, "lr"),
                    "train_loss": read_float(row, "train_loss"),
                    "train_mae": read_float(row, "train_mae"),
                    "train_rmse": read_float(row, "train_rmse"),
                    "train_under_rate": read_float(
                        row,
                        "train_under_rate",
                    ),
                    "train_over_rate": read_float(
                        row,
                        "train_over_rate",
                    ),
                    "train_within_1": read_float(
                        row,
                        "train_within_1",
                    ),
                    "test_loss": read_float(row, "test_loss"),
                    "test_mae": read_float(row, "test_mae"),
                    "test_rmse": read_float(row, "test_rmse"),
                    "test_rounded_accuracy": read_float(
                        row,
                        "test_rounded_accuracy",
                    ),
                    "test_under_rate": read_float(
                        row,
                        "test_under_rate",
                    ),
                    "test_over_rate": read_float(
                        row,
                        "test_over_rate",
                    ),
                    "test_within_1": read_float(
                        row,
                        "test_within_1",
                    ),
                    "epoch_time_sec": read_float(
                        row,
                        "epoch_time_sec",
                    ),
                }
            )
    return rows


def plot_history(
    history: Sequence[Dict[str, Any]],
    out_dir: Path,
) -> None:
    if not history:
        return

    epochs = np.asarray(
        [row["epoch"] for row in history]
    )

    fig = plt.figure(figsize=(7.5, 5.0))
    plt.semilogy(
        epochs,
        [row["train_loss"] for row in history],
    )
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(
        out_dir / "training_loss_curve.png",
        dpi=190,
    )
    plt.close(fig)

    fig = plt.figure(figsize=(7.5, 5.0))
    plt.plot(
        epochs,
        [row["train_mae"] for row in history],
        label="MAE",
    )
    plt.plot(
        epochs,
        [row["train_rmse"] for row in history],
        label="RMSE",
    )
    plt.xlabel("epoch")
    plt.ylabel("continuous-score error")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(
        out_dir / "training_curve.png",
        dpi=190,
    )
    plt.close(fig)


# ============================================================================
# Plots and output
# ============================================================================

def save_score_figure(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    query_x: np.ndarray,
    query_t: np.ndarray,
    source_sample_matlab: int,
    max_score: int,
) -> None:
    absolute_error = np.abs(
        prediction - target
    )

    extent = [
        float(query_x.min()),
        float(query_x.max()),
        float(query_t.min()),
        float(query_t.max()),
    ]

    fig = plt.figure(figsize=(15.0, 4.4))

    axis_one = fig.add_subplot(1, 3, 1)
    image_one = axis_one.imshow(
        target.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0.0,
        vmax=float(max_score),
    )
    axis_one.set_xlabel("x")
    axis_one.set_ylabel("t")
    axis_one.set_title("target score")
    fig.colorbar(image_one, ax=axis_one)

    axis_two = fig.add_subplot(1, 3, 2)
    image_two = axis_two.imshow(
        prediction.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0.0,
        vmax=float(max_score),
    )
    axis_two.set_xlabel("x")
    axis_two.set_ylabel("t")
    axis_two.set_title("FNO prediction")
    fig.colorbar(image_two, ax=axis_two)

    axis_three = fig.add_subplot(1, 3, 3)
    image_three = axis_three.imshow(
        absolute_error.T,
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    axis_three.set_xlabel("x")
    axis_three.set_ylabel("t")
    axis_three.set_title(
        f"|error|, MAE={float(absolute_error.mean()):.3e}"
    )
    fig.colorbar(image_three, ax=axis_three)

    fig.suptitle(
        f"MATLAB dataset sample {source_sample_matlab}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def per_sample_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    max_score: int,
    generation_threshold: float,
) -> Dict[str, np.ndarray]:
    difference = prediction - target

    mae = np.mean(
        np.abs(difference),
        axis=(1, 2),
    )
    rmse = np.sqrt(
        np.mean(
            difference * difference,
            axis=(1, 2),
        )
    )

    prediction_norm = np.linalg.norm(
        difference.reshape(
            difference.shape[0],
            -1,
        ),
        axis=1,
    )
    target_norm = np.linalg.norm(
        target.reshape(
            target.shape[0],
            -1,
        ),
        axis=1,
    )
    relative_l2 = prediction_norm / np.maximum(
        target_norm,
        1.0e-12,
    )

    prediction_generation = score_to_generation_numpy(
        prediction,
        max_score=max_score,
        threshold=generation_threshold,
    )
    target_generation = score_to_generation_numpy(
        target,
        max_score=max_score,
        threshold=generation_threshold,
    )

    under_rate = np.mean(
        prediction_generation < target_generation,
        axis=(1, 2),
    )
    over_rate = np.mean(
        prediction_generation > target_generation,
        axis=(1, 2),
    )
    within_one = np.mean(
        np.abs(difference) <= 1.0,
        axis=(1, 2),
    )

    return {
        "mae": mae.astype(np.float64),
        "rmse": rmse.astype(np.float64),
        "relative_l2": relative_l2.astype(np.float64),
        "under_rate": under_rate.astype(np.float64),
        "over_rate": over_rate.astype(np.float64),
        "within_one": within_one.astype(np.float64),
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FNO2d for 101x101 Burgers continuous refinement-score "
            "regression with 5000 training and 100 held-out test samples."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Figure directory; defaults to figures/burgers_fno/<run-name>.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help=(
            "Optional existing split_indices.npz. Use this to compare the "
            "axis-fixed model on exactly the same 100 test samples."
        ),
    )

    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--nx-input",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--modes-x",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--modes-t",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--projection-width",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--padding-t",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--max-score",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--generation-threshold",
        type=float,
        default=0.50,
        help=(
            "Fractional threshold for continuous score -> integer generation: "
            "fraction < threshold rounds down, fraction >= threshold rounds up. "
            "This affects metrics and exported generations only, not training loss."
        ),
    )
    parser.add_argument(
        "--under-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--score-balance-power",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--bound-penalty",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--bf16-amp",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=50,
        help=(
            "Evaluate and print metrics on the test set every N epochs. "
            "Use 0 to disable periodic test evaluation."
        ),
    )
    parser.add_argument(
        "--save-examples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.train_samples < 1 or args.test_samples < 1:
        raise ValueError(
            "Train/test sample counts must be positive."
        )
    if args.nx_input < 4:
        raise ValueError("--nx-input must be at least four.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError(
            "Epochs and batch size must be positive."
        )
    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )
    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be positive."
        )
    if args.eta_min < 0:
        raise ValueError(
            "--eta-min cannot be negative."
        )
    if args.eta_min > args.learning_rate:
        raise ValueError(
            "--eta-min cannot exceed --learning-rate."
        )
    if min(
        args.width,
        args.modes_x,
        args.modes_t,
        args.layers,
        args.projection_width,
    ) < 1:
        raise ValueError(
            "FNO architecture parameters must be positive."
        )
    if args.padding_t < 0:
        raise ValueError(
            "--padding-t cannot be negative."
        )
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(
            "--dropout must lie in [0,1)."
        )
    if args.max_score < 1:
        raise ValueError(
            "--max-score must be positive."
        )
    if not 0.0 < args.generation_threshold < 1.0:
        raise ValueError(
            "--generation-threshold must lie strictly between 0 and 1."
        )
    if args.under_weight <= 0:
        raise ValueError(
            "--under-weight must be positive."
        )
    if args.score_balance_power < 0:
        raise ValueError(
            "--score-balance-power cannot be negative."
        )
    if args.bound_penalty < 0:
        raise ValueError(
            "--bound-penalty cannot be negative."
        )
    if args.save_every < 0:
        raise ValueError(
            "--save-every cannot be negative."
        )
    if args.test_every < 0:
        raise ValueError(
            "--test-every cannot be negative."
        )
    if args.save_examples < 0:
        raise ValueError(
            "--save-examples cannot be negative."
        )
    if args.timing_repeats < 1:
        raise ValueError(
            "--timing-repeats must be positive."
        )
    if args.resume and args.evaluate_only:
        raise ValueError(
            "--resume and --evaluate-only cannot be combined."
        )

    return args


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    overall_start = time.perf_counter()
    args = parse_args()

    if args.fig_dir is None:
        args.fig_dir = DEFAULT_FIGURES / args.out_dir.name

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(
        args.seed,
        args.deterministic,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_bf16_amp = use_bf16_amp_requested(
        bool(args.bf16_amp),
        device,
    )
    if args.bf16_amp and not use_bf16_amp:
        print(
            "[precision] BF16 was requested but is unavailable; "
            "using float32."
        )

    print("=" * 76)
    print("Burgers continuous refinement-score FNO")
    print("=" * 76)
    print(f"code version      : {CODE_VERSION}")
    print(f"device            : {device}")
    print(
        f"precision         : "
        f"{'BF16 autocast without GradScaler' if use_bf16_amp else 'float32'}"
    )
    print(f"data              : {args.data}")
    print(f"output            : {args.out_dir}")
    print(f"figures           : {args.fig_dir}")
    print(f"generation thresh : {args.generation_threshold:.3f}")
    print("=" * 76)

    data = load_dataset(args.data)
    n_samples = int(data.ic_coefficients.shape[0])

    amplitudes = np.asarray(
        data.ic_coefficients[:, 0],
        dtype=np.float64,
    )
    print(
        f"[input distribution] sine amplitude a: "
        f"min={amplitudes.min():.6g}, "
        f"mean={amplitudes.mean():.6g}, "
        f"max={amplitudes.max():.6g}"
    )
    print(
        "[input distribution] GRF spectrum is read directly from "
        "grf_spectral_std in the MAT file (compatible with grfPower=2.5)."
    )

    if args.split_file is None and (
        args.train_samples + args.test_samples != n_samples
    ):
        raise ValueError(
            "The requested exact split does not match the loaded data: "
            f"{args.train_samples}+{args.test_samples}!={n_samples}."
        )

    observed_min = float(data.target.min())
    observed_max = float(data.target.max())
    if observed_min < -1.0e-6:
        raise ValueError(
            f"Target contains a negative score: {observed_min}."
        )
    if observed_max > args.max_score + 1.0e-6:
        raise ValueError(
            f"Target maximum {observed_max} exceeds "
            f"--max-score={args.max_score}."
        )

    if args.split_file is not None:
        split_path = args.split_file.expanduser()
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )
        split_data = np.load(split_path)
        train_key = (
            "train_indices"
            if "train_indices" in split_data
            else "fit_indices"
        )
        if train_key not in split_data or "test_indices" not in split_data:
            raise KeyError(
                "Split file must contain train_indices (or fit_indices) "
                "and test_indices."
            )
        train_indices = np.asarray(
            split_data[train_key], dtype=np.int64
        ).reshape(-1)
        test_indices = np.asarray(
            split_data["test_indices"], dtype=np.int64
        ).reshape(-1)
        if train_indices.size != args.train_samples:
            raise ValueError(
                f"Split file has {train_indices.size} training samples; "
                f"expected {args.train_samples}."
            )
        if test_indices.size != args.test_samples:
            raise ValueError(
                f"Split file has {test_indices.size} test samples; "
                f"expected {args.test_samples}."
            )
        if np.intersect1d(train_indices, test_indices).size:
            raise ValueError("Split file has overlapping train/test sets.")
        all_indices = np.concatenate((train_indices, test_indices))
        if all_indices.size == 0 or int(all_indices.min()) < 0:
            raise ValueError(
                "Split file contains a negative sample index."
            )
        if int(all_indices.max()) >= n_samples:
            raise ValueError(
                f"Split file index {int(all_indices.max())} exceeds the "
                f"loaded data range 0,...,{n_samples - 1}."
            )
        np.savez(
            args.out_dir / "split_indices.npz",
            train_indices=train_indices,
            test_indices=test_indices,
        )
        print(f"[split] loaded from {split_path}")
    else:
        train_indices, test_indices = load_or_create_split(
            out_dir=args.out_dir,
            n_samples=n_samples,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            seed=args.seed,
            reuse_existing=bool(
                args.resume or args.evaluate_only
            ),
        )
    print(
        f"[split] training={train_indices.size}, "
        f"held-out test={test_indices.size}, loaded={n_samples}, "
        f"unused={n_samples - train_indices.size - test_indices.size}"
    )

    # Keep the dense sensor-grid values for downstream MATLAB export,
    # but feed the FNO the exact Fourier reconstruction at query_x.
    x_input, u0_discrete = reconstruct_u0_on_discrete_grid(
        data,
        nx_input=args.nx_input,
        xmin=-1.0,
        xmax=1.0,
    )
    u0_query = reconstruct_u0_at_points(
        data,
        data.query_x,
    )
    forcing_query=np.ascontiguousarray(data.forcing_input,dtype=np.float32)
    forcing_discrete=periodic_interpolate_u0(
        forcing_query,data.query_x,x_input,xmin=-1.0,xmax=1.0
    )

    print(
        f"[input] x_input={x_input.shape}, "
        f"U0_discrete={u0_discrete.shape}, "
        f"U0_query={u0_query.shape}"
    )

    u_mean = float(
        np.mean(
            u0_query[train_indices],
            dtype=np.float64,
        )
    )
    u_std = float(
        np.std(
            u0_query[train_indices],
            dtype=np.float64,
        )
    )
    u_std = max(u_std, 1.0e-8)
    forcing_mean=float(np.mean(forcing_query[train_indices],dtype=np.float64))
    forcing_std=max(float(np.std(forcing_query[train_indices],dtype=np.float64)),1.0e-8)

    score_histogram, score_weights = compute_score_weights(
        data.target,
        train_indices,
        max_score=args.max_score,
        power=args.score_balance_power,
    )

    print(
        f"[normalization] U0 mean={u_mean:.6e}, "
        f"std={u_std:.6e}"
    )
    print(f"[normalization] forcing mean={forcing_mean:.6e}, std={forcing_std:.6e}")
    print(
        f"[targets] histogram 0..{args.max_score}: "
        f"{score_histogram.tolist()}"
    )
    print(
        f"[loss] score weights: {score_weights.tolist()}"
    )

    train_set = BurgersScoreDataset(
        u0_query,
        forcing_query,
        data.target,
        train_indices,
        data.query_x,
        data.query_t,
        u_mean,
        u_std,
        forcing_mean,
        forcing_std,
    )
    test_set = BurgersScoreDataset(
        u0_query,
        forcing_query,
        data.target,
        test_indices,
        data.query_x,
        data.query_t,
        u_mean,
        u_std,
        forcing_mean,
        forcing_std,
    )

    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed + 1,
    )
    test_loader = make_loader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed + 2,
    )

    model_config = {
        "in_channels": 4,
        "width": int(args.width),
        "modes_x": int(args.modes_x),
        "modes_t": int(args.modes_t),
        "layers": int(args.layers),
        "projection_width": int(
            args.projection_width
        ),
        "padding_t": int(args.padding_t),
        "dropout": float(args.dropout),
    }

    final_checkpoint = (
        args.out_dir / "final_fno_score_model.pt"
    )
    last_checkpoint = (
        args.out_dir / "last_fno_score_model.pt"
    )

    if args.evaluate_only:
        checkpoint = safe_torch_load(
            final_checkpoint,
            device,
        )
        model_config = dict(
            checkpoint["model_config"]
        )

    model = FNO2dScoreRegressor(
        **model_config
    ).to(device)

    print(model)
    print(
        f"[model] trainable parameters: "
        f"{count_parameters(model):,}"
    )

    criterion = AsymmetricMSEScoreLoss(
        score_weights=score_weights,
        max_score=args.max_score,
        under_weight=args.under_weight,
        bound_penalty=args.bound_penalty,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.eta_min,
    )

    history_path = (
        args.out_dir / "training_history.csv"
    )
    periodic_test_path = (
        args.out_dir / "periodic_test_metrics.csv"
    )
    history: list[Dict[str, Any]] = []
    periodic_test_history: list[Dict[str, Any]] = []
    training_time_sec = 0.0
    start_epoch = 1

    if args.evaluate_only:
        checkpoint = safe_torch_load(
            final_checkpoint,
            device,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        training_time_sec = float(
            checkpoint.get(
                "training_time_sec",
                0.0,
            )
        )
    else:
        if args.resume:
            checkpoint = safe_torch_load(
                last_checkpoint,
                device,
            )
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )
            start_epoch = int(
                checkpoint["epoch"]
            ) + 1
            training_time_sec = float(
                checkpoint.get(
                    "training_time_sec",
                    0.0,
                )
            )
            history = read_history_csv(
                history_path
            )
            periodic_test_history = [
                {
                    "epoch": int(row["epoch"]),
                    "test_loss": float(row["test_loss"]),
                    "test_mae": float(row["test_mae"]),
                    "test_rmse": float(row["test_rmse"]),
                    "test_rounded_accuracy": float(
                        row["test_rounded_accuracy"]
                    ),
                    "test_under_rate": float(
                        row["test_under_rate"]
                    ),
                    "test_over_rate": float(
                        row["test_over_rate"]
                    ),
                    "test_within_1": float(
                        row["test_within_1"]
                    ),
                }
                for row in history
                if math.isfinite(float(row["test_loss"]))
            ]
            print(
                f"[resume] starting from epoch "
                f"{start_epoch}"
            )

        with (
            args.out_dir / "run_config.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "code_version": CODE_VERSION,
                    "arguments": {
                        key: (
                            str(value)
                            if isinstance(value, Path)
                            else value
                        )
                        for key, value in vars(args).items()
                    },
                    "model_config": model_config,
                    "split": {
                        "training_samples": int(
                            train_indices.size
                        ),
                        "held_out_test_samples": int(
                            test_indices.size
                        ),
                    },
                    "normalization": {
                        "u_mean": u_mean,
                        "u_std": u_std,
                    },
                    "score_histogram": (
                        score_histogram.tolist()
                    ),
                    "score_weights": (
                        score_weights.tolist()
                    ),
                    "generation_rule": {
                        "threshold": float(args.generation_threshold),
                        "minimum_generation": 0,
                        "maximum_generation": int(args.max_score),
                        "used_in_training_loss": False,
                    },
                    "checkpoint_selection": (
                        "fixed final epoch; test monitored periodically "
                        "but not used for model selection"
                    ),
                    "test_monitoring": {
                        "test_every_epochs": int(args.test_every),
                        "used_for_early_stopping": False,
                        "used_for_checkpoint_selection": False,
                    },
                },
                file,
                indent=2,
            )

        if start_epoch <= args.epochs:
            print(
                "\n"
                + "=" * 76
                + "\nTraining: fixed epoch schedule; "
                f"test metrics printed every {args.test_every} epochs\n"
                + "=" * 76
            )

            synchronize(device)
            training_start = time.perf_counter()

            for epoch in range(
                start_epoch,
                args.epochs + 1,
            ):
                epoch_start = time.perf_counter()

                train_loss, train_metrics = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    max_score=args.max_score,
                    generation_threshold=args.generation_threshold,
                    grad_clip=args.grad_clip,
                    use_bf16_amp=use_bf16_amp,
                )
                scheduler.step()

                epoch_time = (
                    time.perf_counter()
                    - epoch_start
                )
                learning_rate = float(
                    optimizer.param_groups[0]["lr"]
                )

                periodic_test_due = (
                    args.test_every > 0
                    and (
                        epoch % args.test_every == 0
                        or epoch == args.epochs
                    )
                )

                test_loss_epoch = math.nan
                test_metrics_epoch: Dict[str, float] = {
                    "mae": math.nan,
                    "rmse": math.nan,
                    "rounded_accuracy": math.nan,
                    "under_refinement_rate": math.nan,
                    "over_refinement_rate": math.nan,
                    "within_1": math.nan,
                }

                if periodic_test_due:
                    (
                        test_loss_epoch,
                        test_metrics_epoch,
                        _,
                        _,
                    ) = evaluate(
                        model=model,
                        loader=test_loader,
                        criterion=criterion,
                        device=device,
                        max_score=args.max_score,
                        generation_threshold=args.generation_threshold,
                        use_bf16_amp=use_bf16_amp,
                        collect_predictions=False,
                    )

                row = {
                    "epoch": epoch,
                    "lr": learning_rate,
                    "train_loss": train_loss,
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "train_under_rate": train_metrics[
                        "under_refinement_rate"
                    ],
                    "train_over_rate": train_metrics[
                        "over_refinement_rate"
                    ],
                    "train_within_1": train_metrics[
                        "within_1"
                    ],
                    "test_loss": test_loss_epoch,
                    "test_mae": test_metrics_epoch["mae"],
                    "test_rmse": test_metrics_epoch["rmse"],
                    "test_rounded_accuracy": (
                        test_metrics_epoch["rounded_accuracy"]
                    ),
                    "test_under_rate": test_metrics_epoch[
                        "under_refinement_rate"
                    ],
                    "test_over_rate": test_metrics_epoch[
                        "over_refinement_rate"
                    ],
                    "test_within_1": test_metrics_epoch[
                        "within_1"
                    ],
                    "epoch_time_sec": epoch_time,
                }
                history.append(row)
                save_history_csv(
                    history_path,
                    history,
                )

                print(
                    f"epoch {epoch:04d}/{args.epochs} | "
                    f"train loss {train_loss:.5e} | "
                    f"MAE {train_metrics['mae']:.5e} | "
                    f"RMSE {train_metrics['rmse']:.5e} | "
                    f"under "
                    f"{100.0 * train_metrics['under_refinement_rate']:.3f}% | "
                    f"within1 "
                    f"{100.0 * train_metrics['within_1']:.3f}% | "
                    f"lr {learning_rate:.3e} | "
                    f"{epoch_time:.2f} s"
                )

                if periodic_test_due:
                    periodic_row = {
                        "epoch": epoch,
                        "test_loss": test_loss_epoch,
                        "test_mae": test_metrics_epoch["mae"],
                        "test_rmse": test_metrics_epoch["rmse"],
                        "test_rounded_accuracy": (
                            test_metrics_epoch["rounded_accuracy"]
                        ),
                        "test_under_rate": test_metrics_epoch[
                            "under_refinement_rate"
                        ],
                        "test_over_rate": test_metrics_epoch[
                            "over_refinement_rate"
                        ],
                        "test_within_1": test_metrics_epoch[
                            "within_1"
                        ],
                    }
                    periodic_test_history.append(periodic_row)
                    save_history_csv(
                        periodic_test_path,
                        periodic_test_history,
                    )

                    print(
                        f"  [test @ epoch {epoch:04d}] "
                        f"loss {test_loss_epoch:.5e} | "
                        f"MAE {test_metrics_epoch['mae']:.5e} | "
                        f"RMSE {test_metrics_epoch['rmse']:.5e} | "
                        f"rounded acc "
                        f"{100.0 * test_metrics_epoch['rounded_accuracy']:.3f}% | "
                        f"within1 "
                        f"{100.0 * test_metrics_epoch['within_1']:.3f}% | "
                        f"under "
                        f"{100.0 * test_metrics_epoch['under_refinement_rate']:.3f}% | "
                        f"over "
                        f"{100.0 * test_metrics_epoch['over_refinement_rate']:.3f}%"
                    )

                if args.save_every > 0 and (
                    epoch % args.save_every == 0
                    or epoch == args.epochs
                ):
                    elapsed_training = (
                        training_time_sec
                        + time.perf_counter()
                        - training_start
                    )
                    torch.save(
                        checkpoint_state(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            training_time_sec=elapsed_training,
                            arguments=args,
                            model_config=model_config,
                            train_indices=train_indices,
                            test_indices=test_indices,
                            u_mean=u_mean,
                            u_std=u_std,
                            score_histogram=score_histogram,
                            score_weights=score_weights,
                            x_input=x_input,
                            query_x=data.query_x,
                            query_t=data.query_t,
                        ),
                        last_checkpoint,
                    )

            synchronize(device)
            training_time_sec += (
                time.perf_counter()
                - training_start
            )

        final_epoch = min(
            args.epochs,
            max(
                start_epoch - 1,
                args.epochs,
            ),
        )
        torch.save(
            checkpoint_state(
                model=model,
                optimizer=None,
                scheduler=None,
                epoch=final_epoch,
                training_time_sec=training_time_sec,
                arguments=args,
                model_config=model_config,
                train_indices=train_indices,
                test_indices=test_indices,
                u_mean=u_mean,
                u_std=u_std,
                score_histogram=score_histogram,
                score_weights=score_weights,
                x_input=x_input,
                query_x=data.query_x,
                query_t=data.query_t,
            ),
            final_checkpoint,
        )

    plot_history(
        history,
        args.fig_dir,
    )

    # Evaluate the final model on the complete training set for a direct
    # generalization-gap diagnostic. This is reporting only and does not
    # select the checkpoint.
    train_eval_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed + 3,
    )
    train_eval_loss, train_eval_metrics, _, _ = evaluate(
        model=model,
        loader=train_eval_loader,
        criterion=criterion,
        device=device,
        max_score=args.max_score,
        generation_threshold=args.generation_threshold,
        use_bf16_amp=use_bf16_amp,
        collect_predictions=False,
    )

    # Final test evaluation. The same set may already have been monitored
    # every args.test_every epochs; it is not used for checkpoint selection.
    test_loss, test_metrics, test_prediction, test_order = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        max_score=args.max_score,
        generation_threshold=args.generation_threshold,
        use_bf16_amp=use_bf16_amp,
        collect_predictions=True,
    )

    if test_prediction is None or test_order is None:
        raise RuntimeError(
            "Internal error: test predictions were not collected."
        )

    target_test = np.ascontiguousarray(
        data.target[test_order],
        dtype=np.float32,
    )
    u0_test = np.ascontiguousarray(
        u0_discrete[test_order],
        dtype=np.float32,
    )
    forcing_test=np.ascontiguousarray(forcing_discrete[test_order],dtype=np.float32)
    coefficient_test = np.ascontiguousarray(
        data.ic_coefficients[test_order],
        dtype=np.float32,
    )

    timing = measure_inference_time(
        model=model,
        dataset=test_set,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_bf16_amp=use_bf16_amp,
        repeats=args.timing_repeats,
    )

    sample_metrics = per_sample_metrics(
        test_prediction,
        target_test,
        args.max_score,
        args.generation_threshold,
    )
    best_local = int(
        np.argmin(sample_metrics["mae"])
    )
    worst_local = int(
        np.argmax(sample_metrics["mae"])
    )

    with (
        args.out_dir
        / "per_sample_test_metrics.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
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

        for local_index in range(test_order.size):
            writer.writerow(
                [
                    local_index + 1,
                    int(test_order[local_index]),
                    int(test_order[local_index]) + 1,
                    int(data.source_attempt_id[test_order[local_index]]),
                    float(
                        sample_metrics["mae"][
                            local_index
                        ]
                    ),
                    float(
                        sample_metrics["rmse"][
                            local_index
                        ]
                    ),
                    float(
                        sample_metrics["relative_l2"][
                            local_index
                        ]
                    ),
                    float(
                        sample_metrics["under_rate"][
                            local_index
                        ]
                    ),
                    float(
                        sample_metrics["over_rate"][
                            local_index
                        ]
                    ),
                    float(
                        sample_metrics["within_one"][
                            local_index
                        ]
                    ),
                ]
            )

    final_report = {
        "code_version": CODE_VERSION,
        "method": "FNO2d continuous refinement-score regression",
        "split": {
            "training_samples": int(train_indices.size),
            "held_out_test_samples": int(test_indices.size),
        },
        "checkpoint_selection": (
            "fixed final epoch; test monitored periodically but not used for model selection"
        ),
        "training_time_sec": float(training_time_sec),
        "final_training_set_loss": float(train_eval_loss),
        "final_training_set_metrics": {
            key: float(value)
            for key, value in train_eval_metrics.items()
        },
        "test_loss": float(test_loss),
        "test_metrics": {
            key: float(value)
            for key, value in test_metrics.items()
        },
        "generation_rule": {
            "threshold": float(args.generation_threshold),
            "minimum_generation": 0,
            "maximum_generation": int(args.max_score),
            "used_in_training_loss": False,
        },
        "per_sample_relative_l2": {
            "mean": float(
                np.mean(
                    sample_metrics["relative_l2"]
                )
            ),
            "median": float(
                np.median(
                    sample_metrics["relative_l2"]
                )
            ),
            "worst": float(
                np.max(
                    sample_metrics["relative_l2"]
                )
            ),
        },
        "timing": timing,
        "trainable_parameters": int(
            count_parameters(model)
        ),
        "device": str(device),
        "precision": (
            "bf16_autocast_without_grad_scaler"
            if use_bf16_amp
            else "float32"
        ),
        "best_mae_sample_matlab": int(
            test_order[best_local]
        ) + 1,
        "worst_mae_sample_matlab": int(
            test_order[worst_local]
        ) + 1,
        "best_mae_source_attempt_id": int(
            data.source_attempt_id[test_order[best_local]]
        ),
        "worst_mae_source_attempt_id": int(
            data.source_attempt_id[test_order[worst_local]]
        ),
    }

    with (
        args.out_dir / "final_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_report,
            file,
            indent=2,
        )

    pred_generation = score_to_generation_numpy(
        test_prediction,
        max_score=args.max_score,
        threshold=args.generation_threshold,
    )
    target_generation = score_to_generation_numpy(
        target_test,
        max_score=args.max_score,
        threshold=args.generation_threshold,
    )

    prediction_mat = (
        args.out_dir
        / "fno_burgers_continuous_score_test_predictions.mat"
    )
    sio.savemat(
        prediction_mat,
        {
            "pred_score": np.transpose(
                test_prediction,
                (1, 2, 0),
            ),
            "target_score_test": np.transpose(
                target_test,
                (1, 2, 0),
            ),
            "pred_generation": np.transpose(
                pred_generation,
                (1, 2, 0),
            ),
            "target_generation": np.transpose(
                target_generation,
                (1, 2, 0),
            ),
            # Backward-compatible field name. Values use the configured
            # threshold rule and are no longer a mathematical ceiling.
            "pred_generation_ceil": np.transpose(
                pred_generation,
                (1, 2, 0),
            ),
            "generation_threshold": np.asarray(
                [[args.generation_threshold]],
                dtype=np.float64,
            ),
            "U0_test": u0_test.T,
            "F_test": forcing_test.T,
            "x_input": x_input.reshape(-1, 1),
            "query_x": data.query_x.reshape(-1, 1),
            "query_t": data.query_t.reshape(-1, 1),
            "ic_coefficients_test": coefficient_test.T,
            "params_test": coefficient_test,
            "grf_modes": data.grf_modes.reshape(-1, 1),
            "grf_spectral_std": (
                data.grf_spectral_std.reshape(-1, 1)
            ),
            "original_indices": (
                test_order.astype(np.int64) + 1
            ).reshape(-1, 1),
            "source_attempt_id_test": data.source_attempt_id[
                test_order
            ].astype(np.int64).reshape(-1, 1),
            "train_indices": (
                train_indices.astype(np.int64) + 1
            ).reshape(-1, 1),
            "test_indices": (
                test_indices.astype(np.int64) + 1
            ).reshape(-1, 1),
            "max_score": np.asarray(
                [[args.max_score]],
                dtype=np.int32,
            ),
            "u_mean": np.asarray(
                [[u_mean]],
                dtype=np.float32,
            ),
            "u_std": np.asarray(
                [[u_std]],
                dtype=np.float32,
            ),
            "forcing_mean": np.asarray([[forcing_mean]],dtype=np.float64),
            "forcing_std": np.asarray([[forcing_std]],dtype=np.float64),
            "test_mae": np.asarray(
                [[test_metrics["mae"]]],
                dtype=np.float64,
            ),
            "test_rmse": np.asarray(
                [[test_metrics["rmse"]]],
                dtype=np.float64,
            ),
            "test_under_refinement_rate": np.asarray(
                [[
                    test_metrics[
                        "under_refinement_rate"
                    ]
                ]],
                dtype=np.float64,
            ),
            "test_over_refinement_rate": np.asarray(
                [[
                    test_metrics[
                        "over_refinement_rate"
                    ]
                ]],
                dtype=np.float64,
            ),
            "test_within_one_rate": np.asarray(
                [[test_metrics["within_1"]]],
                dtype=np.float64,
            ),
            "training_time_sec": np.asarray(
                [[training_time_sec]],
                dtype=np.float64,
            ),
            "test_inference_time_sec": np.asarray(
                [[
                    timing[
                        "total_test_inference_time_sec"
                    ]
                ]],
                dtype=np.float64,
            ),
            "mean_inference_time_sec_per_sample": np.asarray(
                [[
                    timing[
                        "mean_inference_time_sec_per_sample"
                    ]
                ]],
                dtype=np.float64,
            ),
            "per_sample_mae": sample_metrics[
                "mae"
            ].reshape(-1, 1),
            "per_sample_rmse": sample_metrics[
                "rmse"
            ].reshape(-1, 1),
            "per_sample_relative_l2": sample_metrics[
                "relative_l2"
            ].reshape(-1, 1),
            "per_sample_under_rate": sample_metrics[
                "under_rate"
            ].reshape(-1, 1),
            "per_sample_over_rate": sample_metrics[
                "over_rate"
            ].reshape(-1, 1),
        },
        do_compression=True,
    )

    save_score_figure(
        args.fig_dir / "best_prediction.png",
        target_test[best_local],
        test_prediction[best_local],
        data.query_x,
        data.query_t,
        int(test_order[best_local]) + 1,
        args.max_score,
    )
    save_score_figure(
        args.fig_dir / "worst_prediction.png",
        target_test[worst_local],
        test_prediction[worst_local],
        data.query_x,
        data.query_t,
        int(test_order[worst_local]) + 1,
        args.max_score,
    )

    random_count = min(
        args.save_examples,
        test_order.size,
    )
    if random_count > 0:
        rng = np.random.default_rng(
            args.seed + 30
        )
        random_local_indices = np.sort(
            rng.choice(
                np.arange(test_order.size),
                size=random_count,
                replace=False,
            )
        )

        for plot_order, local_index in enumerate(
            random_local_indices,
            start=1,
        ):
            local_index = int(local_index)
            source_sample = (
                int(test_order[local_index]) + 1
            )
            save_score_figure(
                args.fig_dir
                / (
                    f"random_{plot_order:02d}_"
                    f"sample_{source_sample:04d}.png"
                ),
                target_test[local_index],
                test_prediction[local_index],
                data.query_x,
                data.query_t,
                source_sample,
                args.max_score,
            )

    total_wall_time = (
        time.perf_counter() - overall_start
    )

    print("\n" + "=" * 76)
    print("Continuous-score FNO training/evaluation finished")
    print("=" * 76)
    print(f"training samples              : {train_indices.size}")
    print(f"held-out test samples         : {test_indices.size}")
    print(
        f"final training-set MAE        : "
        f"{train_eval_metrics['mae']:.6e}"
    )
    print(f"test continuous-score MAE     : {test_metrics['mae']:.6e}")
    print(
        f"MAE generalization ratio      : "
        f"{test_metrics['mae'] / max(train_eval_metrics['mae'], 1.0e-12):.3f}"
    )
    print(f"test continuous-score RMSE    : {test_metrics['rmse']:.6e}")
    print(
        f"generation accuracy (threshold): "
        f"{100.0 * test_metrics['rounded_accuracy']:.4f}%"
    )
    print(
        f"within one score level        : "
        f"{100.0 * test_metrics['within_1']:.4f}%"
    )
    print(
        f"under-refinement rate         : "
        f"{100.0 * test_metrics['under_refinement_rate']:.4f}%"
    )
    print(
        f"over-refinement rate          : "
        f"{100.0 * test_metrics['over_refinement_rate']:.4f}%"
    )
    print(
        f"mean per-sample relative L2   : "
        f"{np.mean(sample_metrics['relative_l2']):.6e}"
    )
    print(
        f"inference time/sample         : "
        f"{timing['mean_inference_time_sec_per_sample']:.6e} s"
    )
    print(f"training time                 : {training_time_sec:.3f} s")
    print(f"total wall time               : {total_wall_time:.3f} s")
    print(f"checkpoint                    : {final_checkpoint}")
    print(f"prediction MAT                : {prediction_mat}")
    print("=" * 76)


if __name__ == "__main__":
    main()
