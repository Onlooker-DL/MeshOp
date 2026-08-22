from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

# Support direct execution as ``python src/training/train_burgers_unet.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import BurgersMeshSizeDataset, read_metadata
from src.models.unet import UNet2d, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Burgers mesh-size U-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise TypeError("Configuration root must be a mapping")
    return cfg


def make_loader(dataset: BurgersMeshSizeDataset, cfg: dict, shuffle: bool) -> DataLoader:
    workers = int(cfg["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"] if shuffle else cfg["inference_batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def predict_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for x, y, source_id in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            prediction = model(x)
        predictions.append(prediction.float().cpu().numpy()[:, 0])
        targets.append(y.numpy()[:, 0])
        source_ids.append(np.asarray(source_id))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(source_ids).astype(np.int64),
        elapsed,
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = PROJECT_ROOT
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    seed_everything(seed, bool(cfg.get("deterministic", False)))

    data_path = project_root / cfg["data"]["path"]
    metadata = read_metadata(data_path)
    train_dataset = BurgersMeshSizeDataset(data_path, "train")
    test_dataset = BurgersMeshSizeDataset(data_path, "test")
    train_cfg = cfg["training"]
    train_loader = make_loader(train_dataset, train_cfg, shuffle=True)
    test_loader = make_loader(test_dataset, train_cfg, shuffle=False)

    device = torch.device(args.device or train_cfg["device"])
    model_cfg = cfg["model"]
    configured_channels = model_cfg.get("channels")
    channels = (
        tuple(int(value) for value in configured_channels)
        if configured_channels is not None
        else None
    )
    model = UNet2d(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg.get("base_channels", 32)),
        channels=channels,
    ).to(device)
    parameter_count = count_trainable_parameters(model)
    print(f"[parameters] trainable real parameters={parameter_count:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(train_cfg["eta_min"])
    )
    criterion = nn.MSELoss()

    result_dir = project_root / "results" / "burgers" / "unet" / cfg["experiment"]
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "checkpoint.pt"
    final_model_path = result_dir / "final_model.pt"
    start_epoch = 0
    train_seconds = 0.0
    history: list[dict[str, float]] = []
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        train_seconds = float(checkpoint.get("training_time_sec", 0.0))
        history = list(checkpoint.get("history", []))
        print(f"Resuming after epoch {start_epoch}", flush=True)

    use_bf16 = bool(train_cfg.get("bf16_amp", False)) and device.type == "cuda"
    log_every = int(train_cfg["log_every"])
    save_every = int(train_cfg["save_every"])
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                prediction = model(x)
                loss = criterion(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            optimizer.step()
            running_loss += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        train_seconds += elapsed
        epoch_loss = running_loss / max(seen, 1)
        history.append({
            "epoch": float(epoch),
            "train_mse": epoch_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": elapsed,
        })
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"epoch {epoch:04d}/{epochs} | mse={epoch_loss:.8e} | "
                f"lr={optimizer.param_groups[0]['lr']:.3e} | time={elapsed:.2f}s",
                flush=True,
            )
        if epoch % save_every == 0 or epoch == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "training_time_sec": train_seconds,
                    "history": history,
                    "config": cfg,
                },
                checkpoint_path,
            )

    torch.save({"model": model.state_dict(), "config": cfg}, final_model_path)
    prediction, target, source_ids, inference_seconds = predict_test(
        model, test_loader, device, use_bf16
    )
    prediction_clipped = np.clip(prediction, 0.0, 1.0)
    log_range = metadata.log_h_max - metadata.log_h_min
    predicted_log_h = metadata.log_h_min + prediction_clipped * log_range
    target_log_h = metadata.log_h_min + target * log_range

    prediction_path = result_dir / "predicted_mesh_size_test100.h5"
    with h5py.File(prediction_path, "w") as handle:
        handle.create_dataset("predicted_normalized_log_h", data=prediction_clipped, compression="gzip")
        handle.create_dataset("target_normalized_log_h", data=target, compression="gzip")
        handle.create_dataset("predicted_log_h", data=predicted_log_h, compression="gzip")
        handle.create_dataset("target_log_h", data=target_log_h, compression="gzip")
        handle.create_dataset("source_dataset_index", data=source_ids)
        handle.attrs["log_h_min"] = metadata.log_h_min
        handle.attrs["log_h_max"] = metadata.log_h_max

    error = prediction_clipped - target
    metrics = {
        "problem": "burgers_continuous_mesh_size",
        "model": "unet",
        "experiment": cfg["experiment"],
        "seed": seed,
        "training_samples": metadata.train_samples,
        "test_samples": metadata.test_samples,
        "source_test_min": int(source_ids.min()),
        "source_test_max": int(source_ids.max()),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "initial_learning_rate": float(train_cfg["learning_rate"]),
        "eta_min": float(train_cfg["eta_min"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "epochs": epochs,
        "training_time_sec": train_seconds,
        "test_inference_time_sec": inference_seconds,
        "mean_inference_time_sec_per_sample": inference_seconds / metadata.test_samples,
        "trainable_real_parameters": parameter_count,
        "test_normalized_log_h_mse": float(np.mean(error**2)),
        "test_normalized_log_h_mae": float(np.mean(np.abs(error))),
        "prediction_file": str(prediction_path.resolve()),
    }
    (result_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (result_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
