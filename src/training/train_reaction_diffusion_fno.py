#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the 3-D FNO for reaction-diffusion refinement-score regression.

Paths are resolved from the project root. With the default configuration:
    python src/training/train_reaction_diffusion_fno.py
reads data/reaction_diffusion/reaction_diffusion_5000.mat and writes
result/reaction_diffusion_fno/. Use --data or --out-dir only to override.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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


CODE_VERSION = "FNO3D_REACTION_DIFFUSION_V1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    PROJECT_ROOT / "data" / "reaction_diffusion" / "reaction_diffusion_5000.mat"
)
DEFAULT_OUT = PROJECT_ROOT / "result" / "reaction_diffusion_fno"
DEFAULT_FIGURES = PROJECT_ROOT / "figures" / "reaction_diffusion_fno"


# ============================================================================
# Utilities
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
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


# ============================================================================
# Dataset
# ============================================================================

@dataclass
class DatasetArrays:
    boundary: np.ndarray          # [N,x,y], float32
    target: np.ndarray            # [N,x,y,z], preferably uint8
    query_x: np.ndarray
    query_y: np.ndarray
    query_s: np.ndarray
    query_z: np.ndarray
    source_attempt_id: np.ndarray
    grf_modes: np.ndarray
    grf_spectral_std: np.ndarray
    grf_xi_cos: np.ndarray        # [N,nModes]
    grf_xi_sin: np.ndarray        # [N,nModes]
    completed_samples: int


def h5vec(h: h5py.File, name: str) -> np.ndarray:
    if name not in h:
        raise KeyError(f'Missing MAT field "{name}".')
    return np.asarray(h[name]).reshape(-1)


def h5scalar(h: h5py.File, name: str, default: int) -> int:
    if name not in h:
        return default
    a = np.asarray(h[name]).reshape(-1)
    return default if a.size == 0 else int(round(float(a[0])))


def canonical_modes(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"grf_modes must be 2-D, got {a.shape}")
    if a.shape[1] == 2:
        out = a
    elif a.shape[0] == 2:
        out = a.T
    else:
        raise ValueError(f"Cannot identify mode columns in {a.shape}")
    return np.ascontiguousarray(out, dtype=np.int16)


def canonical_sample_mode(
    a: np.ndarray,
    n: int,
    nmodes: int,
    hdf5_layout: bool,
    name: str,
) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {a.shape}")
    if hdf5_layout:
        if a.shape == (n, nmodes):
            out = a
        elif a.shape == (nmodes, n):
            out = a.T
        else:
            raise ValueError(f"{name} has incompatible shape {a.shape}")
    else:
        if a.shape == (nmodes, n):
            out = a.T
        elif a.shape == (n, nmodes):
            out = a
        else:
            raise ValueError(f"{name} has incompatible shape {a.shape}")
    return np.ascontiguousarray(out, dtype=np.float32)


def canonical_boundary(
    a: np.ndarray,
    n: int,
    nx: int,
    ny: int,
    hdf5_layout: bool,
) -> np.ndarray:
    """
Train the 3-D FNO for reaction-diffusion refinement-score regression.

Paths are resolved from the project root. With the default configuration:
    python src/training/train_reaction_diffusion_fno.py
reads data/reaction_diffusion/reaction_diffusion_5000.mat and writes
result/reaction_diffusion_fno/. Use --data or --out-dir only to override.
"""
    a = np.asarray(a)
    if a.ndim != 3:
        raise ValueError(f"boundary_input must be 3-D, got {a.shape}")
    if hdf5_layout:
        if a.shape != (n, ny, nx):
            raise ValueError(
                f"Expected v7.3 boundary shape {(n,ny,nx)}, got {a.shape}"
            )
        out = np.transpose(a, (0, 2, 1))
        print("[data] boundary: [sample,y,x] -> [sample,x,y]")
    else:
        if a.shape != (nx, ny, n):
            raise ValueError(
                f"Expected legacy boundary shape {(nx,ny,n)}, got {a.shape}"
            )
        out = np.transpose(a, (2, 0, 1))
        print("[data] boundary: [x,y,sample] -> [sample,x,y]")
    return np.ascontiguousarray(out, dtype=np.float32)


def canonical_target(
    a: np.ndarray,
    n: int,
    nx: int,
    ny: int,
    nz: int,
    hdf5_layout: bool,
) -> np.ndarray:
    """
Train the 3-D FNO for reaction-diffusion refinement-score regression.

Paths are resolved from the project root. With the default configuration:
    python src/training/train_reaction_diffusion_fno.py
reads data/reaction_diffusion/reaction_diffusion_5000.mat and writes
result/reaction_diffusion_fno/. Use --data or --out-dir only to override.
"""
    a = np.asarray(a)
    if a.ndim != 4:
        raise ValueError(f"target_score must be 4-D, got {a.shape}")
    if hdf5_layout:
        if a.shape != (n, nz, ny, nx):
            raise ValueError(
                f"Expected v7.3 target shape {(n,nz,ny,nx)}, got {a.shape}"
            )
        out = np.transpose(a, (0, 3, 2, 1))
        print("[data] target: [sample,z,y,x] -> [sample,x,y,z]")
    else:
        if a.shape != (nx, ny, nz, n):
            raise ValueError(
                f"Expected legacy target shape {(nx,ny,nz,n)}, got {a.shape}"
            )
        out = np.transpose(a, (3, 0, 1, 2))
        print("[data] target: [x,y,z,sample] -> [sample,x,y,z]")
    # Keep uint8 in RAM.  Casting the whole 5000-sample tensor to float32
    # would multiply RAM usage by four.
    return out if np.issubdtype(out.dtype, np.integer) else out.astype(np.float32, copy=False)


def load_v73(path: Path) -> DatasetArrays:
    with h5py.File(path, "r") as h:
        required = [
            "boundary_input", "target_score",
            "query_x", "query_y", "query_z",
            "grf_modes", "grf_spectral_std",
            "grf_xi_cos", "grf_xi_sin",
        ]
        missing = [k for k in required if k not in h]
        if missing:
            raise KeyError(f"Missing MAT fields: {missing}")

        qx = h5vec(h, "query_x").astype(np.float32)
        qy = h5vec(h, "query_y").astype(np.float32)
        qz = h5vec(h, "query_z").astype(np.float32)
        qs = (
            h5vec(h, "query_s").astype(np.float32)
            if "query_s" in h
            else np.linspace(0, 1, qz.size, dtype=np.float32)
        )

        n = int(h["boundary_input"].shape[0])
        boundary = canonical_boundary(
            np.asarray(h["boundary_input"]), n, qx.size, qy.size, True
        )
        target = canonical_target(
            np.asarray(h["target_score"]), n, qx.size, qy.size, qz.size, True
        )

        modes = canonical_modes(np.asarray(h["grf_modes"]))
        spectral_std = h5vec(h, "grf_spectral_std").astype(np.float32)
        if modes.shape[0] != spectral_std.size:
            raise ValueError("grf_modes/grf_spectral_std size mismatch.")

        xi_cos = canonical_sample_mode(
            np.asarray(h["grf_xi_cos"]), n, modes.shape[0], True, "grf_xi_cos"
        )
        xi_sin = canonical_sample_mode(
            np.asarray(h["grf_xi_sin"]), n, modes.shape[0], True, "grf_xi_sin"
        )

        source_attempt = (
            h5vec(h, "source_attempt_id").astype(np.int64)
            if "source_attempt_id" in h
            else np.arange(1, n + 1, dtype=np.int64)
        )
        completed = min(
            h5scalar(h, "completedSamples", n),
            n, target.shape[0], xi_cos.shape[0], xi_sin.shape[0],
            source_attempt.size,
        )

    return DatasetArrays(
        boundary=boundary[:completed],
        target=target[:completed],
        query_x=np.ascontiguousarray(qx),
        query_y=np.ascontiguousarray(qy),
        query_s=np.ascontiguousarray(qs),
        query_z=np.ascontiguousarray(qz),
        source_attempt_id=np.ascontiguousarray(source_attempt[:completed]),
        grf_modes=modes,
        grf_spectral_std=np.ascontiguousarray(spectral_std),
        grf_xi_cos=np.ascontiguousarray(xi_cos[:completed]),
        grf_xi_sin=np.ascontiguousarray(xi_sin[:completed]),
        completed_samples=int(completed),
    )


def load_legacy(path: Path) -> DatasetArrays:
    d = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    required = [
        "boundary_input", "target_score",
        "query_x", "query_y", "query_z",
        "grf_modes", "grf_spectral_std",
        "grf_xi_cos", "grf_xi_sin",
    ]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"Missing MAT fields: {missing}")

    qx = np.asarray(d["query_x"]).reshape(-1).astype(np.float32)
    qy = np.asarray(d["query_y"]).reshape(-1).astype(np.float32)
    qz = np.asarray(d["query_z"]).reshape(-1).astype(np.float32)
    qs = (
        np.asarray(d["query_s"]).reshape(-1).astype(np.float32)
        if "query_s" in d
        else np.linspace(0, 1, qz.size, dtype=np.float32)
    )

    n = int(np.asarray(d["boundary_input"]).shape[2])
    boundary = canonical_boundary(d["boundary_input"], n, qx.size, qy.size, False)
    target = canonical_target(d["target_score"], n, qx.size, qy.size, qz.size, False)

    modes = canonical_modes(d["grf_modes"])
    spectral_std = np.asarray(d["grf_spectral_std"]).reshape(-1).astype(np.float32)
    xi_cos = canonical_sample_mode(
        d["grf_xi_cos"], n, modes.shape[0], False, "grf_xi_cos"
    )
    xi_sin = canonical_sample_mode(
        d["grf_xi_sin"], n, modes.shape[0], False, "grf_xi_sin"
    )

    source_attempt = (
        np.asarray(d["source_attempt_id"]).reshape(-1).astype(np.int64)
        if "source_attempt_id" in d
        else np.arange(1, n + 1, dtype=np.int64)
    )
    completed = n
    if "completedSamples" in d:
        completed = min(
            completed,
            int(round(float(np.asarray(d["completedSamples"]).reshape(-1)[0]))),
        )

    return DatasetArrays(
        boundary=boundary[:completed],
        target=target[:completed],
        query_x=qx, query_y=qy, query_s=qs, query_z=qz,
        source_attempt_id=source_attempt[:completed],
        grf_modes=modes,
        grf_spectral_std=spectral_std,
        grf_xi_cos=xi_cos[:completed],
        grf_xi_sin=xi_sin[:completed],
        completed_samples=int(completed),
    )


def load_dataset(path: Path) -> DatasetArrays:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        data = load_v73(path)
        source = "MATLAB v7.3/HDF5"
    except OSError:
        data = load_legacy(path)
        source = "legacy MAT"

    if not np.all(np.isfinite(data.boundary)):
        raise ValueError("boundary_input contains NaN/Inf.")
    if np.issubdtype(data.target.dtype, np.floating):
        if not np.all(np.isfinite(data.target)):
            raise ValueError("target_score contains NaN/Inf.")

    print(f"[data] source     : {source}")
    print(f"[data] file       : {path}")
    print(f"[data] completed  : {data.completed_samples}")
    print(f"[data] boundary   : {data.boundary.shape}, {data.boundary.dtype}")
    print(f"[data] target     : {data.target.shape}, {data.target.dtype}")
    print(
        f"[data] z first 10 : "
        + " ".join(f"{v:.6g}" for v in data.query_z[:10])
    )
    print(
        f"[data] score range: [{float(np.min(data.target)):.3f}, "
        f"{float(np.max(data.target)):.3f}]"
    )
    return data


# ============================================================================
# Split, weighting, Dataset
# ============================================================================

def make_exact_split(
    n: int, ntrain: int, ntest: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if ntrain + ntest != n:
        raise ValueError(
            f"Exact split requires train+test=loaded: "
            f"{ntrain}+{ntest}!={n}."
        )
    rng = np.random.default_rng(seed)
    p = rng.permutation(n).astype(np.int64)
    return p[:ntrain], p[ntrain:]


def load_or_create_split(
    out_dir: Path,
    n: int,
    ntrain: int,
    ntest: int,
    seed: int,
    reuse: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    path = out_dir / "split_indices.npz"
    if reuse and path.exists():
        s = np.load(path)
        train = np.asarray(s["train_indices"], dtype=np.int64).reshape(-1)
        test = np.asarray(s["test_indices"], dtype=np.int64).reshape(-1)
    else:
        train, test = make_exact_split(n, ntrain, ntest, seed)
        np.savez(path, train_indices=train, test_indices=test)

    if train.size != ntrain or test.size != ntest:
        raise ValueError("Saved split has wrong sizes.")
    if np.intersect1d(train, test).size:
        raise ValueError("Train/test split overlaps.")
    return train, test


def score_weights(
    target: np.ndarray,
    train_indices: np.ndarray,
    max_score: int,
    power: float,
    chunk: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    hist = np.zeros(max_score + 1, dtype=np.int64)
    for i in range(0, train_indices.size, chunk):
        v = np.asarray(target[train_indices[i:i+chunk]])
        if np.issubdtype(v.dtype, np.floating):
            v = np.rint(v)
        v = np.clip(v.astype(np.int16, copy=False), 0, max_score)
        hist += np.bincount(v.reshape(-1), minlength=max_score + 1)

    if power <= 0:
        return hist, np.ones(max_score + 1, dtype=np.float32)

    f = hist.astype(np.float64)
    pos = f > 0
    ref = float(np.mean(f[pos])) if np.any(pos) else 1.0
    w = np.ones(max_score + 1, dtype=np.float64)
    w[pos] = (ref / f[pos]) ** power
    w[pos] /= np.mean(w[pos])
    return hist, w.astype(np.float32)


class RDScoreDataset(Dataset):
    def __init__(
        self,
        boundary: np.ndarray,
        target: np.ndarray,
        indices: np.ndarray,
        mean: float,
        std: float,
    ) -> None:
        self.boundary = boundary
        self.target = target
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = float(mean)
        self.std = float(max(std, 1e-8))

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        g = (
            np.asarray(self.boundary[idx], dtype=np.float32) - self.mean
        ) / self.std
        y = np.asarray(self.target[idx], dtype=np.float32)
        return (
            torch.from_numpy(g).unsqueeze(0),
            torch.from_numpy(y).unsqueeze(0),
            idx,
        )


def coordinate_channels(
    data: DatasetArrays,
    device: torch.device,
) -> torch.Tensor:
    x = 2 * data.query_x.astype(np.float32) - 1
    y = 2 * data.query_y.astype(np.float32) - 1
    z = 2 * data.query_z.astype(np.float32) - 1

    xx, yy, zz = np.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )

    a = np.stack(
        (xx, yy, zz),
        axis=0,
    ).astype(np.float32)

    return (
        torch.from_numpy(
            np.ascontiguousarray(a)
        )
        .unsqueeze(0)
        .to(device)
    )


def volume_input(
    boundary: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    # boundary [B,1,nx,ny], coords [1,3,nx,ny,nz]
    b = boundary.shape[0]
    nz = coords.shape[-1]
    g = boundary.unsqueeze(-1).expand(-1, -1, -1, -1, nz)
    c = coords.expand(b, -1, -1, -1, -1)
    return torch.cat((g, c), dim=1)


# ============================================================================
# FNO3d
# ============================================================================

class SpectralConv3d(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        mx: int,
        my: int,
        mz: int,
    ) -> None:
        super().__init__()
        self.mx, self.my, self.mz = mx, my, mz
        scale = 1.0 / math.sqrt(max(cin * cout, 1))
        shape = (cin, cout, mx, my, mz)
        self.wpp = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.wnp = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.wpn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.wnn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def mul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", a, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, nx, ny, nz = x.shape
        dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfftn(
                x.float(), dim=(-3, -2, -1), norm="ortho"
            )
            nzr = xf.shape[-1]
            mx = min(self.mx, nx // 2)
            my = min(self.my, ny // 2)
            mz = min(self.mz, nzr)

            out = torch.zeros(
                b, self.wpp.shape[1], nx, ny, nzr,
                device=x.device, dtype=torch.cfloat,
            )
            if mx and my and mz:
                out[:, :, :mx, :my, :mz] = self.mul(
                    xf[:, :, :mx, :my, :mz],
                    self.wpp[:, :, :mx, :my, :mz],
                )
                out[:, :, -mx:, :my, :mz] = self.mul(
                    xf[:, :, -mx:, :my, :mz],
                    self.wnp[:, :, :mx, :my, :mz],
                )
                out[:, :, :mx, -my:, :mz] = self.mul(
                    xf[:, :, :mx, -my:, :mz],
                    self.wpn[:, :, :mx, :my, :mz],
                )
                out[:, :, -mx:, -my:, :mz] = self.mul(
                    xf[:, :, -mx:, -my:, :mz],
                    self.wnn[:, :, :mx, :my, :mz],
                )
            y = torch.fft.irfftn(
                out, s=(nx, ny, nz), dim=(-3, -2, -1), norm="ortho"
            )
        return y.to(dtype=dtype)


class FNOBlock3d(nn.Module):
    def __init__(
        self,
        width: int,
        mx: int,
        my: int,
        mz: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, mx, my, mz)
        self.local = nn.Conv3d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.local(x)
        return self.drop(F.gelu(self.norm(y)))


class FNO3dScore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        modes_x: int,
        modes_y: int,
        modes_z: int,
        layers: int,
        projection_width: int,
        padding: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.padding = max(int(padding), 0)
        self.lift = nn.Conv3d(in_channels, width, 1)
        self.blocks = nn.ModuleList([
            FNOBlock3d(
                width, modes_x, modes_y, modes_z, dropout
            )
            for _ in range(layers)
        ])
        self.p1 = nn.Conv3d(width, projection_width, 1)
        self.p2 = nn.Conv3d(projection_width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding:
            p = self.padding
            x = F.pad(x, (0, p, 0, p, 0, p))
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = self.p2(F.gelu(self.p1(x)))
        if self.padding:
            p = self.padding
            x = x[..., :-p, :-p, :-p]
        return x


# ============================================================================
# Loss and metrics
# ============================================================================

class ScoreLoss(nn.Module):
    def __init__(
        self,
        weights: Sequence[float],
        max_score: int,
        under_weight: float,
        bound_penalty: float,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "weights", torch.as_tensor(weights, dtype=torch.float32)
        )
        self.max_score = float(max_score)
        self.under_weight = float(under_weight)
        self.bound_penalty = float(bound_penalty)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e = pred - target
        asym = torch.where(
            e < 0,
            torch.full_like(e, self.under_weight),
            torch.ones_like(e),
        )
        bins = torch.clamp(
            torch.round(target).long(), 0, self.weights.numel() - 1
        )
        fit = torch.mean(self.weights[bins] * asym * e.square())
        bounds = torch.mean(
            F.relu(-pred).square() + F.relu(pred - self.max_score).square()
        )
        return fit + self.bound_penalty * bounds


def gen_torch(
    score: torch.Tensor, max_score: int, threshold: float
) -> torch.Tensor:
    s = torch.clamp(score, 0.0, float(max_score))
    g = torch.floor(s + 1.0 - float(threshold) + 1e-6)
    return torch.clamp(g, 0.0, float(max_score))


def gen_numpy(
    score: np.ndarray, max_score: int, threshold: float
) -> np.ndarray:
    s = np.clip(score, 0.0, float(max_score))
    return np.clip(
        np.floor(s + 1.0 - float(threshold) + 1e-6),
        0, max_score,
    ).astype(np.uint8)


@dataclass
class Totals:
    n: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    correct: int = 0
    within1: int = 0
    under: int = 0
    over: int = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_score: int,
        threshold: float,
    ) -> None:
        p = torch.clamp(pred.detach(), 0.0, float(max_score))
        t = target.detach()
        d = p - t
        pg = gen_torch(p, max_score, threshold)
        tg = gen_torch(t, max_score, threshold)
        n = d.numel()
        self.n += n
        self.abs_sum += torch.sum(torch.abs(d)).item()
        self.sq_sum += torch.sum(d.square()).item()
        self.correct += torch.sum(pg == tg).item()
        self.within1 += torch.sum(torch.abs(d) <= 1.0).item()
        self.under += torch.sum(pg < tg).item()
        self.over += torch.sum(pg > tg).item()

    def final(self) -> Dict[str, float]:
        n = max(self.n, 1)
        return {
            "mae": self.abs_sum / n,
            "rmse": math.sqrt(self.sq_sum / n),
            "rounded_accuracy": self.correct / n,
            "within_1": self.within1 / n,
            "under_refinement_rate": self.under / n,
            "over_refinement_rate": self.over / n,
        }


def make_loader(
    ds: Dataset,
    batch: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        drop_last=False,
    )


def use_bf16(requested: bool, device: torch.device) -> bool:
    return bool(
        requested
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    coords: torch.Tensor,
    device: torch.device,
    max_score: int,
    threshold: float,
    grad_clip: float,
    amp: bool,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    loss_sum = 0.0
    nsamp = 0
    totals = Totals()

    for boundary, target, _ in loader:
        boundary = boundary.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        inp = volume_input(boundary, coords)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            pred = model(inp)
            loss = criterion(pred, target)

        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is NaN/Inf.")

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        b = boundary.shape[0]
        loss_sum += loss.item() * b
        nsamp += b
        totals.update(pred, target, max_score, threshold)

    return loss_sum / max(nsamp, 1), totals.final()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    coords: torch.Tensor,
    device: torch.device,
    max_score: int,
    threshold: float,
    amp: bool,
    collect: bool,
):
    model.eval()
    loss_sum = 0.0
    nsamp = 0
    totals = Totals()
    preds, ids = [], []

    for boundary, target, index in loader:
        boundary = boundary.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        inp = volume_input(boundary, coords)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            pred = model(inp)
            loss = criterion(pred, target)

        b = boundary.shape[0]
        loss_sum += loss.item() * b
        nsamp += b
        totals.update(pred, target, max_score, threshold)

        if collect:
            preds.append(
                torch.clamp(pred, 0.0, float(max_score))
                .squeeze(1).float().cpu().numpy()
            )
            ids.append(np.asarray(index, dtype=np.int64))

    return (
        loss_sum / max(nsamp, 1),
        totals.final(),
        np.concatenate(preds) if collect else None,
        np.concatenate(ids) if collect else None,
    )


@torch.inference_mode()
def inference_timing(
    model: nn.Module,
    ds: Dataset,
    coords: torch.Tensor,
    device: torch.device,
    batch: int,
    workers: int,
    amp: bool,
    repeats: int,
) -> Dict[str, float]:
    loader = make_loader(ds, batch, False, workers, device, 0)
    model.eval()

    for i, (boundary, _, _) in enumerate(loader):
        boundary = boundary.to(device, non_blocking=True)
        inp = volume_input(boundary, coords)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp
        ):
            out = model(inp)
        _ = out.float().cpu().numpy()
        if i >= 1:
            break
    synchronize(device)

    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        for boundary, _, _ in loader:
            boundary = boundary.to(device, non_blocking=True)
            inp = volume_input(boundary, coords)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp
            ):
                out = model(inp)
            _ = out.float().cpu().numpy()
        synchronize(device)
        times.append(time.perf_counter() - t0)

    total = float(np.median(times))
    return {
        "total_test_inference_time_sec": total,
        "mean_inference_time_sec_per_sample": total / max(len(ds), 1),
        "throughput_samples_per_sec": len(ds) / max(total, 1e-12),
        "timing_repeats": repeats,
    }


# ============================================================================
# Saving / plotting
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_mib(model: nn.Module) -> float:
    return sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1024**2


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_history(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = {}
            for k, v in r.items():
                if k == "epoch":
                    row[k] = int(v)
                else:
                    row[k] = float(v) if v not in ("", None) else math.nan
            out.append(row)
    return out


def plot_history(rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    e = [r["epoch"] for r in rows]

    fig = plt.figure(figsize=(7.5, 5))
    plt.semilogy(e, [r["train_loss"] for r in rows])
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss_curve.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7.5, 5))
    plt.plot(e, [r["train_mae"] for r in rows], label="MAE")
    plt.plot(e, [r["train_rmse"] for r in rows], label="RMSE")
    plt.xlabel("epoch")
    plt.ylabel("continuous-score error")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    training_time: float,
    args: argparse.Namespace,
    model_config: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mean: float,
    std: float,
    hist: np.ndarray,
    weights: np.ndarray,
    data: DatasetArrays,
) -> Dict[str, Any]:
    state = {
        "code_version": CODE_VERSION,
        "epoch": epoch,
        "training_time_sec": training_time,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "train_indices": torch.from_numpy(train_idx.astype(np.int64)),
        "test_indices": torch.from_numpy(test_idx.astype(np.int64)),
        "boundary_mean": mean,
        "boundary_std": std,
        "score_histogram": torch.from_numpy(hist.astype(np.int64)),
        "score_weights": torch.from_numpy(weights.astype(np.float32)),
        "query_x": torch.from_numpy(data.query_x),
        "query_y": torch.from_numpy(data.query_y),
        "query_s": torch.from_numpy(data.query_s),
        "query_z": torch.from_numpy(data.query_z),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    return state


def sample_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    max_score: int,
    threshold: float,
) -> Dict[str, np.ndarray]:
    d = pred - target
    axes = (1, 2, 3)
    mae = np.mean(np.abs(d), axis=axes)
    rmse = np.sqrt(np.mean(d * d, axis=axes))
    rel = np.linalg.norm(d.reshape(d.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(target.reshape(target.shape[0], -1), axis=1), 1e-12
    )
    pg = gen_numpy(pred, max_score, threshold)
    tg = gen_numpy(target, max_score, threshold)
    return {
        "mae": mae,
        "rmse": rmse,
        "relative_l2": rel,
        "under": np.mean(pg < tg, axis=axes),
        "over": np.mean(pg > tg, axis=axes),
        "within1": np.mean(np.abs(d) <= 1.0, axis=axes),
    }


def nearest(a: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(a.astype(np.float64) - value)))


def save_prediction_figure(
    path: Path,
    target: np.ndarray,
    pred: np.ndarray,
    data: DatasetArrays,
    sample_one_based: int,
    max_score: int,
) -> None:
    err = np.abs(pred - target)
    iz0 = 0
    ize = nearest(data.query_z, 0.02)
    iym = nearest(data.query_y, 0.5)

    fig, ax = plt.subplots(3, 3, figsize=(13.5, 11.5), constrained_layout=True)
    for row, iz in enumerate([iz0, ize]):
        arrs = [target[:, :, iz], pred[:, :, iz], err[:, :, iz]]
        names = [
            f"target z={data.query_z[iz]:.4g}",
            f"prediction z={data.query_z[iz]:.4g}",
            f"|error| z={data.query_z[iz]:.4g}",
        ]
        for j, a in enumerate(arrs):
            im = ax[row, j].imshow(
                a.T,
                origin="lower",
                extent=[0, 1, 0, 1],
                vmin=0 if j < 2 else None,
                vmax=max_score if j < 2 else None,
                aspect="equal",
            )
            ax[row, j].set_xlabel("x")
            ax[row, j].set_ylabel("y")
            ax[row, j].set_title(names[j])
            fig.colorbar(im, ax=ax[row, j])

    arrs = [target[:, iym, :], pred[:, iym, :], err[:, iym, :]]
    names = ["target vertical", "prediction vertical", "|error| vertical"]
    for j, a in enumerate(arrs):
        im = ax[2, j].pcolormesh(
            data.query_x,
            data.query_z,
            a.T,
            shading="auto",
            vmin=0 if j < 2 else None,
            vmax=max_score if j < 2 else None,
        )
        ax[2, j].set_xlabel("x")
        ax[2, j].set_ylabel("physical z")
        ax[2, j].set_title(names[j] + f", y={data.query_y[iym]:.3f}")
        fig.colorbar(im, ax=ax[2, j])

    fig.suptitle(f"MATLAB sample {sample_one_based}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ============================================================================
# Arguments
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-D FNO for reaction-diffusion AFEM score regression."
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fig-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=None)
    p.add_argument("--seed", type=int, default=900)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--train-samples", type=int, default=4900)
    p.add_argument("--test-samples", type=int, default=100)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eta-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--width", type=int, default=24)
    p.add_argument("--modes-x", type=int, default=12)
    p.add_argument("--modes-y", type=int, default=12)
    p.add_argument("--modes-z", type=int, default=12)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--projection-width", type=int, default=64)
    p.add_argument("--padding", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.03)

    p.add_argument("--max-score", type=int, default=10)
    p.add_argument("--generation-threshold", type=float, default=0.50)
    p.add_argument("--under-weight", type=float, default=1.0)
    p.add_argument("--score-balance-power", type=float, default=0.0)
    p.add_argument("--bound-penalty", type=float, default=0.0)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bf16-amp", type=int, choices=[0, 1], default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--test-every", type=int, default=20)
    p.add_argument("--save-examples", type=int, default=5)
    p.add_argument("--timing-repeats", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")

    a = p.parse_args()

    if a.train_samples < 1 or a.test_samples < 1:
        raise ValueError("train/test counts must be positive.")
    if a.epochs < 1 or a.batch_size < 1:
        raise ValueError("epochs/batch size must be positive.")
    if min(
        a.width, a.modes_x, a.modes_y, a.modes_z,
        a.layers, a.projection_width,
    ) < 1:
        raise ValueError("FNO architecture values must be positive.")
    if not 0 < a.generation_threshold < 1:
        raise ValueError("generation threshold must lie in (0,1).")
    if a.resume and a.evaluate_only:
        raise ValueError("--resume and --evaluate-only cannot be combined.")
    return a


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    wall0 = time.perf_counter()
    args = parse_args()
    if args.fig_dir is None:
        args.fig_dir = DEFAULT_FIGURES / args.out_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed, args.deterministic)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp = use_bf16(bool(args.bf16_amp), device)

    print("=" * 78)
    print("Reaction-diffusion 3-D FNO refinement-score regression")
    print("=" * 78)
    print(f"code version : {CODE_VERSION}")
    print(f"device       : {device}")
    print(f"precision    : {'BF16 autocast; FFT float32' if amp else 'float32'}")
    print(f"data         : {args.data}")
    print(f"out          : {args.out_dir}")
    print(f"figures      : {args.fig_dir}")
    print("=" * 78)

    data = load_dataset(args.data)
    n = data.completed_samples

    if args.split_file is None and args.train_samples + args.test_samples != n:
        raise ValueError(
            f"Requested split {args.train_samples}+{args.test_samples}!={n}. "
            "For the current 5000-sample file use 4900 train + 100 test."
        )
    if float(np.max(data.target)) > args.max_score + 1e-6:
        raise ValueError(
            f"Target max {np.max(data.target)} exceeds --max-score={args.max_score}."
        )

    if args.split_file is not None:
        s = np.load(args.split_file)
        key = "train_indices" if "train_indices" in s else "fit_indices"
        if key not in s or "test_indices" not in s:
            raise KeyError(
                "Split file must contain train_indices (or fit_indices) "
                "and test_indices."
            )
        train_idx = np.asarray(s[key], dtype=np.int64).reshape(-1)
        test_idx = np.asarray(s["test_indices"], dtype=np.int64).reshape(-1)
        if train_idx.size != args.train_samples or test_idx.size != args.test_samples:
            raise ValueError("External split sizes do not match arguments.")
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError("External split has overlapping train/test sets.")
        all_idx = np.concatenate((train_idx, test_idx))
        if all_idx.size == 0 or int(all_idx.min()) < 0 or int(all_idx.max()) >= n:
            raise ValueError(
                f"External split indices must lie in 0,...,{n - 1}."
            )
        np.savez(
            args.out_dir / "split_indices.npz",
            train_indices=train_idx,
            test_indices=test_idx,
        )
    else:
        train_idx, test_idx = load_or_create_split(
            args.out_dir, n,
            args.train_samples, args.test_samples,
            args.seed,
            reuse=(args.resume or args.evaluate_only),
        )

    print(
        f"[split] train={train_idx.size}, test={test_idx.size}, loaded={n}, "
        f"unused={n - train_idx.size - test_idx.size}"
    )

    # Train-only normalization.
    train_boundary = np.asarray(data.boundary[train_idx], dtype=np.float32)
    bmean = float(np.mean(train_boundary, dtype=np.float64))
    bstd = max(float(np.std(train_boundary, dtype=np.float64)), 1e-8)
    del train_boundary

    hist, weights = score_weights(
        data.target, train_idx, args.max_score, args.score_balance_power
    )
    print(f"[normalization] boundary mean={bmean:.6e}, std={bstd:.6e}")
    print(f"[targets] histogram={hist.tolist()}")
    print(f"[loss] weights={weights.tolist()}")

    train_ds = RDScoreDataset(data.boundary, data.target, train_idx, bmean, bstd)
    test_ds = RDScoreDataset(data.boundary, data.target, test_idx, bmean, bstd)
    train_loader = make_loader(
        train_ds, args.batch_size, True, args.num_workers, device, args.seed + 1
    )
    test_loader = make_loader(
        test_ds, args.batch_size, False, args.num_workers, device, args.seed + 2
    )
    coords = coordinate_channels(data, device)

    model_config = {
        "in_channels": 4,
        "width": args.width,
        "modes_x": args.modes_x,
        "modes_y": args.modes_y,
        "modes_z": args.modes_z,
        "layers": args.layers,
        "projection_width": args.projection_width,
        "padding": args.padding,
        "dropout": args.dropout,
    }

    final_ckpt = args.out_dir / "final_fno3d_score_model.pt"
    last_ckpt = args.out_dir / "last_fno3d_score_model.pt"

    if args.evaluate_only:
        ck = safe_torch_load(final_ckpt, device)
        model_config = dict(ck["model_config"])

    model = FNO3dScore(**model_config).to(device)
    print(model)
    print(f"[model] trainable parameters: {count_parameters(model):,}")
    print(f"[model] parameter storage: {parameter_mib(model):.1f} MiB")

    criterion = ScoreLoss(
        weights, args.max_score, args.under_weight, args.bound_penalty
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )

    history_path = args.out_dir / "training_history.csv"
    test_history_path = args.out_dir / "periodic_test_metrics.csv"
    history: list[Dict[str, Any]] = []
    test_history: list[Dict[str, Any]] = []
    training_time = 0.0
    start_epoch = 1

    if args.evaluate_only:
        ck = safe_torch_load(final_ckpt, device)
        model.load_state_dict(ck["model_state_dict"])
        training_time = float(ck.get("training_time_sec", 0.0))
    else:
        if args.resume:
            ck = safe_torch_load(last_ckpt, device)
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            start_epoch = int(ck["epoch"]) + 1
            training_time = float(ck.get("training_time_sec", 0.0))
            history = load_history(history_path)
            print(f"[resume] start epoch {start_epoch}")

        with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "code_version": CODE_VERSION,
                    "arguments": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    "model_config": model_config,
                    "input_channels": [
                        "normalized g_D repeated in third direction",
                        "x", "y", "computational s", "physical z=s^2",
                    ],
                    "boundary_mean": bmean,
                    "boundary_std": bstd,
                    "score_histogram": hist.tolist(),
                    "score_weights": weights.tolist(),
                },
                f,
                indent=2,
            )

        if start_epoch <= args.epochs:
            synchronize(device)
            ttrain = time.perf_counter()

            for epoch in range(start_epoch, args.epochs + 1):
                t0 = time.perf_counter()
                train_loss, tm = train_epoch(
                    model, train_loader, optimizer, criterion, coords,
                    device, args.max_score, args.generation_threshold,
                    args.grad_clip, amp,
                )
                scheduler.step()
                epoch_time = time.perf_counter() - t0
                lr = optimizer.param_groups[0]["lr"]

                due = (
                    args.test_every > 0
                    and (epoch % args.test_every == 0 or epoch == args.epochs)
                )
                test_loss = math.nan
                em = {
                    "mae": math.nan,
                    "rmse": math.nan,
                    "rounded_accuracy": math.nan,
                    "within_1": math.nan,
                    "under_refinement_rate": math.nan,
                    "over_refinement_rate": math.nan,
                }
                if due:
                    test_loss, em, _, _ = evaluate(
                        model, test_loader, criterion, coords, device,
                        args.max_score, args.generation_threshold, amp, False,
                    )

                row = {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_loss,
                    "train_mae": tm["mae"],
                    "train_rmse": tm["rmse"],
                    "train_under_rate": tm["under_refinement_rate"],
                    "train_over_rate": tm["over_refinement_rate"],
                    "train_within_1": tm["within_1"],
                    "test_loss": test_loss,
                    "test_mae": em["mae"],
                    "test_rmse": em["rmse"],
                    "test_rounded_accuracy": em["rounded_accuracy"],
                    "test_under_rate": em["under_refinement_rate"],
                    "test_over_rate": em["over_refinement_rate"],
                    "test_within_1": em["within_1"],
                    "epoch_time_sec": epoch_time,
                }
                history.append(row)
                save_csv(history_path, history)

                print(
                    f"epoch {epoch:04d}/{args.epochs} | "
                    f"train loss {train_loss:.5e} | "
                    f"MAE {tm['mae']:.5e} | RMSE {tm['rmse']:.5e} | "
                    f"under {100*tm['under_refinement_rate']:.3f}% | "
                    f"within1 {100*tm['within_1']:.3f}% | "
                    f"lr {lr:.3e} | {epoch_time:.2f}s"
                )

                if due:
                    tr = {
                        "epoch": epoch,
                        "test_loss": test_loss,
                        "test_mae": em["mae"],
                        "test_rmse": em["rmse"],
                        "test_rounded_accuracy": em["rounded_accuracy"],
                        "test_under_rate": em["under_refinement_rate"],
                        "test_over_rate": em["over_refinement_rate"],
                        "test_within_1": em["within_1"],
                    }
                    test_history.append(tr)
                    save_csv(test_history_path, test_history)
                    print(
                        f"  [test] loss {test_loss:.5e} | "
                        f"MAE {em['mae']:.5e} | RMSE {em['rmse']:.5e} | "
                        f"gen acc {100*em['rounded_accuracy']:.3f}% | "
                        f"under {100*em['under_refinement_rate']:.3f}% | "
                        f"over {100*em['over_refinement_rate']:.3f}%"
                    )

                if args.save_every > 0 and (
                    epoch % args.save_every == 0 or epoch == args.epochs
                ):
                    elapsed = training_time + time.perf_counter() - ttrain
                    torch.save(
                        checkpoint(
                            model, optimizer, scheduler, epoch, elapsed,
                            args, model_config, train_idx, test_idx,
                            bmean, bstd, hist, weights, data,
                        ),
                        last_ckpt,
                    )

            synchronize(device)
            training_time += time.perf_counter() - ttrain

        torch.save(
            checkpoint(
                model, None, None, args.epochs, training_time,
                args, model_config, train_idx, test_idx,
                bmean, bstd, hist, weights, data,
            ),
            final_ckpt,
        )

    plot_history(history, args.fig_dir)

    # Final train metrics (reporting only).
    train_eval_loader = make_loader(
        train_ds, args.batch_size, False, args.num_workers, device, args.seed + 3
    )
    train_eval_loss, train_metrics, _, _ = evaluate(
        model, train_eval_loader, criterion, coords, device,
        args.max_score, args.generation_threshold, amp, False,
    )

    test_loss, test_metrics, pred, order = evaluate(
        model, test_loader, criterion, coords, device,
        args.max_score, args.generation_threshold, amp, True,
    )
    if pred is None or order is None:
        raise RuntimeError("Test predictions were not collected.")

    target_test = np.asarray(data.target[order], dtype=np.float32)
    boundary_test = np.asarray(data.boundary[order], dtype=np.float32)

    timing = inference_timing(
        model, test_ds, coords, device, args.batch_size,
        args.num_workers, amp, args.timing_repeats,
    )
    sm = sample_metrics(
        pred, target_test, args.max_score, args.generation_threshold
    )
    best = int(np.argmin(sm["mae"]))
    worst = int(np.argmax(sm["mae"]))

    with (args.out_dir / "per_sample_test_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "test_order", "dataset_index_zero_based",
            "dataset_index_matlab_one_based", "source_attempt_id",
            "mae", "rmse", "relative_l2",
            "under_refinement_rate", "over_refinement_rate",
            "within_one_rate",
        ])
        for i in range(order.size):
            w.writerow([
                i + 1, int(order[i]), int(order[i]) + 1,
                int(data.source_attempt_id[order[i]]),
                float(sm["mae"][i]), float(sm["rmse"][i]),
                float(sm["relative_l2"][i]),
                float(sm["under"][i]), float(sm["over"][i]),
                float(sm["within1"][i]),
            ])

    report = {
        "code_version": CODE_VERSION,
        "method": "FNO3d continuous refinement-score regression",
        "operator": "g_D(x,y) -> score(x,y,z)",
        "coordinate_mapping": "uniform s with physical z=s^2",
        "training_samples": int(train_idx.size),
        "test_samples": int(test_idx.size),
        "training_time_sec": training_time,
        "final_training_loss": train_eval_loss,
        "final_training_metrics": train_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "mean_per_sample_relative_l2": float(np.mean(sm["relative_l2"])),
        "median_per_sample_relative_l2": float(np.median(sm["relative_l2"])),
        "timing": timing,
        "trainable_parameters": count_parameters(model),
        "parameter_storage_mib": parameter_mib(model),
        "best_sample_matlab": int(order[best]) + 1,
        "worst_sample_matlab": int(order[worst]) + 1,
    }
    with (args.out_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pg = gen_numpy(pred, args.max_score, args.generation_threshold)
    tg = gen_numpy(target_test, args.max_score, args.generation_threshold)

    prediction_mat = (
        args.out_dir / "fno_reactiondiffusion_3d_score_test_predictions.mat"
    )
    sio.savemat(
        prediction_mat,
        {
            # Python [sample,x,y,z] -> MATLAB [x,y,z,sample]
            "pred_score": np.transpose(pred, (1, 2, 3, 0)),
            "target_score_test": np.transpose(target_test, (1, 2, 3, 0)),
            "pred_generation": np.transpose(pg, (1, 2, 3, 0)),
            "target_generation": np.transpose(tg, (1, 2, 3, 0)),
            "boundary_test": np.transpose(boundary_test, (1, 2, 0)),
            "query_x": data.query_x[:, None],
            "query_y": data.query_y[:, None],
            "query_s": data.query_s[:, None],
            "query_z": data.query_z[:, None],
            "grf_modes": data.grf_modes,
            "grf_spectral_std": data.grf_spectral_std[:, None],
            "grf_xi_cos_test": data.grf_xi_cos[order].T,
            "grf_xi_sin_test": data.grf_xi_sin[order].T,
            "original_indices": (order + 1)[:, None],
            "source_attempt_id_test": data.source_attempt_id[order][:, None],
            "train_indices": (train_idx + 1)[:, None],
            "test_indices": (test_idx + 1)[:, None],
            "generation_threshold": np.array(
                [[args.generation_threshold]], dtype=np.float64
            ),
            "max_score": np.array([[args.max_score]], dtype=np.int32),
            "boundary_mean": np.array([[bmean]], dtype=np.float32),
            "boundary_std": np.array([[bstd]], dtype=np.float32),
            "per_sample_mae": sm["mae"][:, None],
            "per_sample_rmse": sm["rmse"][:, None],
            "per_sample_relative_l2": sm["relative_l2"][:, None],
            "per_sample_under_rate": sm["under"][:, None],
            "per_sample_over_rate": sm["over"][:, None],
        },
        do_compression=True,
    )

    save_prediction_figure(
        args.fig_dir / "best_prediction.png",
        target_test[best], pred[best], data, int(order[best]) + 1, args.max_score,
    )
    save_prediction_figure(
        args.fig_dir / "worst_prediction.png",
        target_test[worst], pred[worst], data, int(order[worst]) + 1, args.max_score,
    )

    rng = np.random.default_rng(args.seed + 30)
    nplot = min(args.save_examples, order.size)
    if nplot:
        selected = np.sort(rng.choice(order.size, nplot, replace=False))
        for j, i in enumerate(selected, 1):
            save_prediction_figure(
                args.fig_dir / f"random_{j:02d}_sample_{int(order[i])+1:04d}.png",
                target_test[i], pred[i], data, int(order[i]) + 1, args.max_score,
            )

    print("\n" + "=" * 78)
    print("Finished")
    print("=" * 78)
    print(f"train samples        : {train_idx.size}")
    print(f"test samples         : {test_idx.size}")
    print(f"train MAE            : {train_metrics['mae']:.6e}")
    print(f"test MAE             : {test_metrics['mae']:.6e}")
    print(f"test RMSE            : {test_metrics['rmse']:.6e}")
    print(f"generation accuracy  : {100*test_metrics['rounded_accuracy']:.4f}%")
    print(f"within one level     : {100*test_metrics['within_1']:.4f}%")
    print(f"under-refinement     : {100*test_metrics['under_refinement_rate']:.4f}%")
    print(f"over-refinement      : {100*test_metrics['over_refinement_rate']:.4f}%")
    print(
        f"inference/sample     : "
        f"{timing['mean_inference_time_sec_per_sample']:.6e} s"
    )
    print(f"training time        : {training_time:.3f} s")
    print(f"total wall time      : {time.perf_counter() - wall0:.3f} s")
    print(f"checkpoint           : {final_ckpt}")
    print(f"prediction MAT       : {prediction_mat}")
    print("=" * 78)


if __name__ == "__main__":
    main()
