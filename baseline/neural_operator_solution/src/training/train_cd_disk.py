from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.physical_l2 import (  # noqa: E402
    cd_disk_quadrature_weights,
    solution_metrics,
)
from src.models import DeepONet, PODCoefficientNet  # noqa: E402
from src.training.train_burgers import (  # noqa: E402
    BranchDataset,
    build_grid_model,
    make_loader,
    parameter_counts,
    randomized_pod,
    seed_everything,
    train_deeponet,
    train_grid_model,
    train_pod,
    write_history,
)


class CDGridDataset(Dataset):
    def __init__(self, forcing: np.ndarray, target: np.ndarray, beta: np.ndarray, indices: np.ndarray, x: np.ndarray, y: np.ndarray, forcing_mean: float, forcing_std: float) -> None:
        self.forcing = forcing
        self.target = target
        self.beta = beta
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x = x
        self.y = y
        self.forcing_mean = float(forcing_mean)
        self.forcing_std = float(forcing_std)

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(self.indices[item])
        f = ((self.forcing[i].T - self.forcing_mean) / self.forcing_std).astype(np.float32)
        cb, sb = np.cos(self.beta[i]), np.sin(self.beta[i])
        inputs = np.stack((f, self.x, self.y, np.full_like(self.x, cb), np.full_like(self.x, sb)))
        return torch.from_numpy(inputs), torch.from_numpy(self.target[i].T[None].astype(np.float32))


def load_data(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h:
        completed_attr = np.asarray(h.attrs.get("completed_samples", 0)).reshape(-1)
        completed = int(completed_attr[0]) if completed_attr.size else 0
        if completed < 3100:
            raise ValueError(f"CD spectral file has only {completed}/3100 completed samples.")
        out = {
            "solution": np.asarray(h["solution"][:3100], dtype=np.float32),
            "forcing": np.asarray(h["forcing"][:3100], dtype=np.float32),
            "r": np.asarray(h["r"]).reshape(-1).astype(np.float32),
            "theta": np.asarray(h["theta"]).reshape(-1).astype(np.float32),
            "beta": np.asarray(h["beta_angle"]).reshape(-1)[:3100].astype(np.float32),
            "source_id": np.asarray(h["source_sample_id"]).reshape(-1)[:3100].astype(np.int64),
        }
    if out["solution"].shape != out["forcing"].shape:
        raise ValueError("CD solution and forcing shapes differ.")
    return out


def branch_inputs(forcing: np.ndarray, beta: np.ndarray, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    rs = np.linspace(0.0, 1.0, 16)
    ts = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    ri = np.asarray([np.argmin(np.abs(r - value)) for value in rs])
    ti = np.asarray([np.argmin(np.abs(theta - value)) for value in ts])
    sensors = forcing[:, ri][:, :, ti].reshape(forcing.shape[0], -1)
    return np.concatenate((sensors, np.cos(beta)[:, None], np.sin(beta)[:, None]), axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct CD-disk solution operator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config.get("seed", 42))
    seed_everything(seed, bool(config.get("deterministic", False)))
    model_cfg = dict(config["model"])
    name = str(model_cfg.pop("name"))
    training = dict(config["training"])
    device = torch.device(training.get("device", "cuda:0"))
    data = load_data(ROOT / config["data"]["path"])
    ntrain, ntest = int(config["data"]["train_samples"]), int(config["data"]["test_samples"])
    train_ids = np.arange(ntrain, dtype=np.int64)
    test_ids = np.arange(3100 - ntest, 3100, dtype=np.int64)
    target = data["solution"]
    forcing = data["forcing"]
    r, theta, beta = data["r"], data["theta"], data["beta"]
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    x_grid = (rr * np.cos(tt)).T.astype(np.float32)
    y_grid = (rr * np.sin(tt)).T.astype(np.float32)
    coordinates = np.stack((rr.T.reshape(-1), tt.T.reshape(-1)), axis=1).astype(np.float32)
    branch_raw = branch_inputs(forcing, beta, r, theta)
    branch_mean = branch_raw[train_ids].mean(axis=0)
    branch_std = np.maximum(branch_raw[train_ids].std(axis=0), 1.0e-6)
    branch = ((branch_raw - branch_mean) / branch_std).astype(np.float32)
    out_dir = ROOT / "results" / "cd_disk" / name / str(config["experiment"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    np.savez_compressed(out_dir / "branch_normalization.npz", mean=branch_mean, std=branch_std)

    pod_details = None
    if name in {"fno", "cno"}:
        model = build_grid_model(name, model_cfg).to(device)
        fm = float(forcing[train_ids].mean(dtype=np.float64))
        fs = max(float(forcing[train_ids].std(dtype=np.float64)), 1.0e-6)
        train_set = CDGridDataset(forcing, target, beta, train_ids, x_grid, y_grid, fm, fs)
        test_set = CDGridDataset(forcing, target, beta, test_ids, x_grid, y_grid, fm, fs)
        train_loader = make_loader(train_set, int(training["batch_size"]), True, int(training.get("num_workers", 0)), seed + 1, device)
        test_loader = make_loader(test_set, int(training.get("inference_sample_batch_size", 8)), False, int(training.get("num_workers", 0)), seed + 2, device)
        prediction, history, elapsed, inference_time = train_grid_model(model, train_loader, test_loader, training, device, out_dir, args.resume)
        prediction = prediction.transpose(0, 2, 1)
    elif name == "deeponet":
        model = DeepONet(branch_dim=branch.shape[1], coordinate_dim=2, **model_cfg).to(device)
        loader = make_loader(BranchDataset(branch, target.transpose(0, 2, 1), train_ids), int(training["batch_size"]), True, int(training.get("num_workers", 0)), seed + 1, device)
        flat, history, elapsed, inference_time = train_deeponet(model, loader, branch[test_ids], coordinates, training, device, out_dir, args.resume)
        prediction = flat.reshape(ntest, theta.size, r.size).transpose(0, 2, 1)
    elif name == "pod_deeponet":
        pod_cfg = dict(config["pod"])
        mean, basis, coeff, pod_details = randomized_pod(target[train_ids], int(pod_cfg["maximum_rank"]), int(pod_cfg.get("oversampling", 16)), float(pod_cfg["energy"]), seed + 101, int(pod_cfg.get("sample_chunk", 32)))
        model = PODCoefficientNet(branch_dim=branch.shape[1], rank=basis.shape[0], **model_cfg).to(device)
        flat, history, elapsed, inference_time = train_pod(model, branch[train_ids], coeff, branch[test_ids], mean, basis, training, device, out_dir, seed, args.resume)
        prediction = flat.reshape(ntest, r.size, theta.size)
        np.savez_compressed(out_dir / "pod_basis.npz", field_mean=mean, basis=basis, **pod_details)
    else:
        raise ValueError(f"Unsupported model {name}")

    counts = parameter_counts(model)
    quadrature_weights = cd_disk_quadrature_weights(r, theta)
    metrics = solution_metrics(prediction, target[test_ids], quadrature_weights)
    print(f"[parameters] equivalent real parameters={counts['real_trainable_parameters']:,}")
    with h5py.File(out_dir / "predictions.h5", "w") as h:
        h.create_dataset("prediction", data=prediction, compression="gzip", compression_opts=1)
        h.create_dataset("target", data=target[test_ids], compression="gzip", compression_opts=1)
        h.create_dataset("r", data=r)
        h.create_dataset("theta", data=theta)
        h.create_dataset("source_sample_id", data=data["source_id"][test_ids])
        h.attrs["relative_l2_definition"] = (
            "mean of per-sample physical-domain quadrature relative L2 errors"
        )
    final = {
        "problem": "cd_disk",
        "model": name,
        "training_samples": ntrain,
        "test_samples": ntest,
        "split": "samples 1:3000 train, 3001:3100 test",
        "training_time_sec": elapsed,
        "test_inference_time_sec": inference_time,
        "mean_inference_time_sec_per_sample": inference_time / ntest,
        "inference_throughput_samples_per_sec": ntest / inference_time,
        "test_metrics": metrics,
        "relative_l2_evaluation": {
            "definition": "mean of per-sample physical-domain relative L2 errors",
            "quadrature": "polar control-volume r dr tensor periodic trapezoidal dtheta",
            "legacy_metric": "test_metrics.mean_relative_discrete_l2",
        },
        "parameter_count": counts,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "learning_rate": float(training["learning_rate"]),
        "eta_min": float(training["eta_min"]),
        "weight_decay": float(training["weight_decay"]),
    }
    if pod_details is not None:
        final["pod"] = pod_details
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)
    write_history(out_dir / "history.csv", history)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
