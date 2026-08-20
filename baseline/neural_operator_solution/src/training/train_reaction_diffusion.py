from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.physical_l2 import (  # noqa: E402
    reaction_diffusion_quadrature_weights,
    solution_metrics,
)
from src.models import CNO, DeepONet, FNO3dSolutionOperator, PODCoefficientNet  # noqa: E402
from src.training.train_burgers import (  # noqa: E402
    make_loader,
    make_optimizer,
    maybe_resume,
    parameter_counts,
    save_checkpoint,
    seed_everything,
    train_pod,
    write_history,
)


class RDGridDataset(Dataset):
    def __init__(self, path: Path, boundary: np.ndarray, indices: np.ndarray, coordinates: np.ndarray, mean: float, std: float) -> None:
        self.path = str(path)
        self.boundary = boundary
        self.indices = np.asarray(indices, dtype=np.int64)
        self.coordinates = coordinates
        self.mean, self.std = float(mean), float(std)
        self._file: h5py.File | None = None

    def __len__(self) -> int:
        return self.indices.size

    def _solution(self, index: int) -> np.ndarray:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return np.asarray(self._file["solution"][index], dtype=np.float32)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        b = ((self.boundary[index] - self.mean) / self.std).astype(np.float32)
        field = np.broadcast_to(b[:, :, None], self.coordinates.shape[1:])
        inputs = np.concatenate((field[None], self.coordinates), axis=0).copy()
        target = self._solution(index)[None]
        return torch.from_numpy(inputs), torch.from_numpy(target)


class RDPointDataset(Dataset):
    def __init__(self, path: Path, branch: np.ndarray, indices: np.ndarray) -> None:
        self.path = str(path)
        self.branch = branch
        self.indices = np.asarray(indices, dtype=np.int64)
        self._file: h5py.File | None = None

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        target = np.asarray(self._file["solution"][index], dtype=np.float32).reshape(-1)
        return torch.from_numpy(self.branch[index]), torch.from_numpy(target)


def load_metadata(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h:
        completed_attr = np.asarray(h.attrs.get("completed_samples", 0)).reshape(-1)
        completed = int(completed_attr[0]) if completed_attr.size else 0
        if completed < 3100:
            raise ValueError(f"Reaction-diffusion file has only {completed}/3100 completed samples.")
        return {
            "boundary": np.asarray(h["boundary"][:3100], dtype=np.float32),
            "x": np.asarray(h["x"]).reshape(-1).astype(np.float32),
            "y": np.asarray(h["y"]).reshape(-1).astype(np.float32),
            "z": np.asarray(h["z"]).reshape(-1).astype(np.float32),
            "source_id": np.asarray(h["source_sample_id"]).reshape(-1)[:3100].astype(np.int64),
        }


def stream_pod(path: Path, ntrain: int, maximum_rank: int, oversampling: int, energy: float, seed: int, chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    with h5py.File(path, "r") as h:
        ds = h["solution"]
        points = int(np.prod(ds.shape[1:]))
        qrank = min(maximum_rank + oversampling, ntrain - 1)
        mean = np.zeros(points, dtype=np.float64)
        for start in range(0, ntrain, chunk):
            block = np.asarray(ds[start : min(start + chunk, ntrain)], dtype=np.float32).reshape(-1, points)
            mean += block.sum(axis=0, dtype=np.float64)
        mean = (mean / ntrain).astype(np.float32)
        rng = np.random.default_rng(seed)
        omega = rng.standard_normal((points, qrank), dtype=np.float32)
        sketch = np.empty((ntrain, qrank), dtype=np.float32)
        total_energy = 0.0
        for start in range(0, ntrain, chunk):
            block = np.asarray(ds[start : min(start + chunk, ntrain)], dtype=np.float32).reshape(-1, points) - mean
            sketch[start : start + block.shape[0]] = block @ omega
            total_energy += float(np.sum(np.square(block.astype(np.float64))))
        q, _ = np.linalg.qr(sketch, mode="reduced")
        compressed = np.zeros((qrank, points), dtype=np.float32)
        for start in range(0, ntrain, chunk):
            block = np.asarray(ds[start : min(start + chunk, ntrain)], dtype=np.float32).reshape(-1, points) - mean
            compressed += q[start : start + block.shape[0]].T @ block
        _, singular, vt = np.linalg.svd(compressed, full_matrices=False)
        cumulative = np.cumsum(np.square(singular.astype(np.float64))) / max(total_energy, 1.0e-30)
        meeting = np.flatnonzero(cumulative >= energy)
        rank = min(int(meeting[0] + 1) if meeting.size else maximum_rank, maximum_rank)
        basis = np.ascontiguousarray(vt[:rank], dtype=np.float32)
        coefficients = np.empty((ntrain, rank), dtype=np.float32)
        for start in range(0, ntrain, chunk):
            block = np.asarray(ds[start : min(start + chunk, ntrain)], dtype=np.float32).reshape(-1, points) - mean
            coefficients[start : start + block.shape[0]] = block @ basis.T
    return mean, basis, coefficients, {"rank": rank, "requested_energy": energy, "captured_energy": float(cumulative[rank - 1])}


def train_deep_stream(model: DeepONet, loader: torch.utils.data.DataLoader, test_branch: np.ndarray, coordinates: np.ndarray, training: dict[str, Any], device: torch.device, out_dir: Path, resume: bool) -> tuple[np.ndarray, list[dict[str, float]], float, float]:
    optimizer, scheduler = make_optimizer(model, training)
    checkpoint = out_dir / "checkpoint_last.pt"
    start_epoch, history = maybe_resume(checkpoint, model, optimizer, scheduler, device, resume)
    epochs = int(training["epochs"])
    train_queries = min(int(training.get("train_query_points", 8192)), coordinates.shape[0])
    query_batch = int(training.get("query_batch_size", 16384))
    coords = torch.from_numpy(coordinates).to(device)
    wall = time.perf_counter()
    log_wall = wall
    last_log_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for branch, target in loader:
            ids = torch.randperm(coords.shape[0], device=device)[:train_queries]
            branch, target = branch.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(branch, coords[ids]) - target[:, ids]).square())
            loss.backward()
            if float(training.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
            optimizer.step()
            loss_sum += float(loss.detach()) * branch.shape[0]
            count += branch.shape[0]
        scheduler.step()
        row = {"epoch": float(epoch), "lr": float(optimizer.param_groups[0]["lr"]), "train_mse": loss_sum / max(count, 1)}
        history.append(row)
        if epoch == 1 or epoch % int(training.get("log_every", 10)) == 0 or epoch == epochs:
            now = time.perf_counter()
            interval_epochs = max(epoch - last_log_epoch, 1)
            print(
                f"[epoch {epoch:5d}/{epochs}] mse={row['train_mse']:.6e} "
                f"lr={row['lr']:.3e} interval={now - log_wall:.2f}s "
                f"sec/epoch={(now - log_wall) / interval_epochs:.2f} "
                f"elapsed={now - wall:.2f}s",
                flush=True,
            )
            log_wall = now
            last_log_epoch = epoch
        if epoch % int(training.get("save_every", 50)) == 0 or epoch == epochs:
            save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, history)
    elapsed = time.perf_counter() - wall
    output = np.empty((test_branch.shape[0], coordinates.shape[0]), dtype=np.float32)
    sample_batch = int(training.get("inference_sample_batch_size", 2))
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_wall = time.perf_counter()
    with torch.inference_mode():
        for ss in range(0, test_branch.shape[0], sample_batch):
            se = min(ss + sample_batch, test_branch.shape[0])
            branch = torch.from_numpy(test_branch[ss:se]).to(device)
            for qs in range(0, coordinates.shape[0], query_batch):
                qe = min(qs + query_batch, coordinates.shape[0])
                output[ss:se, qs:qe] = model(branch, coords[qs:qe]).float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - inference_wall
    return output, history, elapsed, inference_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct reaction-diffusion solution operator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config.get("seed", 42))
    seed_everything(seed, bool(config.get("deterministic", False)))
    training, model_cfg = dict(config["training"]), dict(config["model"])
    name = str(model_cfg.pop("name"))
    device = torch.device(training.get("device", "cuda:0"))
    path = ROOT / config["data"]["path"]
    data = load_metadata(path)
    ntrain, ntest = int(config["data"]["train_samples"]), int(config["data"]["test_samples"])
    train_ids, test_ids = np.arange(ntrain), np.arange(3100 - ntest, 3100)
    x, y, z = data["x"], data["y"], data["z"]
    xn, yn, zn = 2 * x - 1, 2 * y - 1, 2 * z - 1
    xx, yy, zz = np.meshgrid(xn, yn, zn, indexing="ij")
    coordinate_channels = np.stack((xx, yy, zz)).astype(np.float32)
    point_coordinates = np.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), axis=1).astype(np.float32)
    branch_raw = data["boundary"].reshape(3100, -1)
    branch_mean = branch_raw[train_ids].mean(axis=0)
    branch_std = np.maximum(branch_raw[train_ids].std(axis=0), 1.0e-6)
    branch = ((branch_raw - branch_mean) / branch_std).astype(np.float32)
    out_dir = ROOT / "results" / "reaction_diffusion" / name / str(config["experiment"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    np.savez_compressed(out_dir / "branch_normalization.npz", mean=branch_mean, std=branch_std)

    pod_details = None
    if name in {"fno", "cno"}:
        if name == "fno":
            model = FNO3dSolutionOperator(**model_cfg)
        else:
            dim = int(model_cfg.pop("dim", 3))
            model = CNO(dim=dim, **model_cfg)
        model = model.to(device)
        bm = float(data["boundary"][train_ids].mean(dtype=np.float64))
        bs = max(float(data["boundary"][train_ids].std(dtype=np.float64)), 1.0e-6)
        train_set = RDGridDataset(path, data["boundary"], train_ids, coordinate_channels, bm, bs)
        test_set = RDGridDataset(path, data["boundary"], test_ids, coordinate_channels, bm, bs)
        train_loader = make_loader(train_set, int(training["batch_size"]), True, int(training.get("num_workers", 0)), seed + 1, device)
        test_loader = make_loader(test_set, int(training.get("inference_sample_batch_size", 1)), False, int(training.get("num_workers", 0)), seed + 2, device)
        from src.training.train_burgers import train_grid_model
        prediction, history, elapsed, inference_time = train_grid_model(model, train_loader, test_loader, training, device, out_dir, args.resume)
    elif name == "deeponet":
        model = DeepONet(branch_dim=branch.shape[1], coordinate_dim=3, **model_cfg).to(device)
        loader = make_loader(RDPointDataset(path, branch, train_ids), int(training["batch_size"]), True, int(training.get("num_workers", 0)), seed + 1, device)
        flat, history, elapsed, inference_time = train_deep_stream(model, loader, branch[test_ids], point_coordinates, training, device, out_dir, args.resume)
        prediction = flat.reshape(ntest, x.size, y.size, z.size)
    elif name == "pod_deeponet":
        pod = dict(config["pod"])
        mean, basis, coefficients, pod_details = stream_pod(path, ntrain, int(pod["maximum_rank"]), int(pod.get("oversampling", 16)), float(pod["energy"]), seed + 101, int(pod.get("sample_chunk", 4)))
        model = PODCoefficientNet(branch_dim=branch.shape[1], rank=basis.shape[0], **model_cfg).to(device)
        flat, history, elapsed, inference_time = train_pod(model, branch[train_ids], coefficients, branch[test_ids], mean, basis, training, device, out_dir, seed, args.resume)
        prediction = flat.reshape(ntest, x.size, y.size, z.size)
        np.savez_compressed(out_dir / "pod_basis.npz", field_mean=mean, basis=basis, **pod_details)
    else:
        raise ValueError(f"Unsupported model {name}")

    with h5py.File(path, "r") as h:
        target = np.asarray(h["solution"][test_ids], dtype=np.float32)
    counts = parameter_counts(model)
    quadrature_weights = reaction_diffusion_quadrature_weights(x, y, z)
    test_metrics = solution_metrics(prediction, target, quadrature_weights)
    print(f"[parameters] equivalent real parameters={counts['real_trainable_parameters']:,}")
    with h5py.File(out_dir / "predictions.h5", "w") as h:
        h.create_dataset("prediction", data=prediction, compression="gzip", compression_opts=1)
        h.create_dataset("target", data=target, compression="gzip", compression_opts=1)
        h.create_dataset("x", data=x); h.create_dataset("y", data=y); h.create_dataset("z", data=z)
        h.create_dataset("source_sample_id", data=data["source_id"][test_ids])
        h.attrs["relative_l2_definition"] = (
            "mean of per-sample physical-domain quadrature relative L2 errors"
        )
    final = {"problem": "reaction_diffusion", "model": name, "training_samples": ntrain, "test_samples": ntest, "split": "samples 1:3000 train, 3001:3100 test", "training_time_sec": elapsed, "test_inference_time_sec": inference_time, "mean_inference_time_sec_per_sample": inference_time / ntest, "inference_throughput_samples_per_sec": ntest / inference_time, "test_metrics": test_metrics, "relative_l2_evaluation": {"definition": "mean of per-sample physical-domain relative L2 errors", "quadrature": "tensor-product trapezoidal dx dy dz on the stored nonuniform z grid", "legacy_metric": "test_metrics.mean_relative_discrete_l2"}, "parameter_count": counts, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "learning_rate": float(training["learning_rate"]), "eta_min": float(training["eta_min"]), "weight_decay": float(training["weight_decay"])}
    if pod_details is not None:
        final["pod"] = pod_details
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)
    write_history(out_dir / "history.csv", history)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
