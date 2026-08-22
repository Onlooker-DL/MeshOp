from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BurgersMetadata:
    samples: int
    train_samples: int
    test_samples: int
    nx: int
    nt: int
    log_h_min: float
    log_h_max: float
    input_mean: np.ndarray
    input_std: np.ndarray


def _scalar_attr(handle: h5py.File, name: str, default: float | None = None) -> float:
    if name not in handle.attrs:
        if default is None:
            raise KeyError(f"Missing root attribute {name!r}")
        return float(default)
    return float(np.asarray(handle.attrs[name]).reshape(-1)[0])


def read_metadata(path: str | Path) -> BurgersMetadata:
    with h5py.File(Path(path), "r") as handle:
        return BurgersMetadata(
            samples=int(handle["input"].shape[0]),
            train_samples=int(_scalar_attr(handle, "train_samples")),
            test_samples=int(_scalar_attr(handle, "test_samples")),
            nx=int(handle["input"].shape[-1]),
            nt=int(handle["input"].shape[-2]),
            log_h_min=_scalar_attr(handle, "log_h_min"),
            log_h_max=_scalar_attr(handle, "log_h_max"),
            input_mean=np.asarray(handle["input_channel_mean"], dtype=np.float32),
            input_std=np.asarray(handle["input_channel_std"], dtype=np.float32),
        )


class BurgersMeshSizeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Lazy HDF5 reader; every worker owns its own read-only handle."""

    def __init__(self, path: str | Path, split: str) -> None:
        self.path = Path(path)
        self.metadata = read_metadata(self.path)
        self.input_mean = self.metadata.input_mean.reshape(4, 1, 1)
        self.input_std = self.metadata.input_std.reshape(4, 1, 1)
        with h5py.File(self.path, "r") as handle:
            self.source_ids = np.asarray(handle["source_dataset_index"]).reshape(-1)
        if split == "train":
            self.indices = np.arange(self.metadata.train_samples, dtype=np.int64)
        elif split == "test":
            first = self.metadata.train_samples
            self.indices = np.arange(first, first + self.metadata.test_samples, dtype=np.int64)
        else:
            raise ValueError("split must be 'train' or 'test'")
        self._handle: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        handle = self._file()
        x = np.asarray(handle["input"][index], dtype=np.float32)
        x = (x - self.input_mean) / self.input_std
        y = np.asarray(handle["target_normalized_log_h"][index], dtype=np.float32)[None]
        source_id = int(self.source_ids[index])
        return torch.from_numpy(x), torch.from_numpy(y), source_id

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.close()
