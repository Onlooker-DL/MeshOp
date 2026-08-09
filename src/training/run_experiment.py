from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("problem", "experiment", "model", "data", "training", "score"):
        if key not in config:
            raise KeyError(f"Configuration is missing '{key}'.")
    return config


def expected_split(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    ntrain = int(data_cfg["train_samples"])
    ntest = int(data_cfg["test_samples"])
    split_mode = str(data_cfg.get("split_mode", "random_exact"))

    if ntrain < 1 or ntest < 1:
        raise ValueError("train_samples and test_samples must be positive.")

    if split_mode == "prefix_train_tail_test":
        total_samples = int(data_cfg["total_samples"])
        if total_samples < ntrain + ntest:
            raise ValueError(
                "prefix_train_tail_test requires total_samples >= "
                "train_samples + test_samples."
            )
        train_indices = np.arange(ntrain, dtype=np.int64)
        test_indices = np.arange(
            total_samples - ntest, total_samples, dtype=np.int64
        )
        if np.intersect1d(train_indices, test_indices).size:
            raise ValueError("The prefix training set overlaps the tail test set.")
        return train_indices, test_indices

    if split_mode == "random_exact":
        permutation = np.random.default_rng(
            int(config.get("seed", 900))
        ).permutation(ntrain + ntest)
        return (
            permutation[:ntrain].astype(np.int64),
            permutation[ntrain:].astype(np.int64),
        )

    raise ValueError(f"Unsupported data.split_mode: {split_mode}")


def ensure_split(config: dict[str, Any], path: Path) -> None:
    expected_train, expected_test = expected_split(config)
    if path.exists():
        split = np.load(path)
        saved_train = np.asarray(split["train_indices"], dtype=np.int64).reshape(-1)
        saved_test = np.asarray(split["test_indices"], dtype=np.int64).reshape(-1)
        if not np.array_equal(saved_train, expected_train) or not np.array_equal(
            saved_test, expected_test
        ):
            raise ValueError(
                f"Existing shared split does not match the configured policy: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_indices=expected_train,
        test_indices=expected_test,
    )


def split_filename(config: dict[str, Any]) -> str:
    problem = str(config["problem"])
    data_cfg = config["data"]
    ntrain = int(data_cfg["train_samples"])
    ntest = int(data_cfg["test_samples"])
    split_mode = str(data_cfg.get("split_mode", "random_exact"))
    if split_mode == "prefix_train_tail_test":
        total = int(data_cfg["total_samples"])
        return f"{problem}_{total}data_first{ntrain}train_last{ntest}test.npz"
    return (
        f"{problem}_{ntrain}train_{ntest}test_"
        f"seed{config.get('seed', 900)}.npz"
    )


def _append(command: list[str], flag: str, value: Any) -> None:
    command.extend((flag, str(value)))


def run_fno(
    config: dict[str, Any], out_dir: Path, figure_dir: Path, split_file: Path
) -> Path:
    problem = str(config["problem"])
    script = (
        PROJECT_ROOT / "src" / "training" / "train_burgers_fno.py"
        if problem == "burgers"
        else PROJECT_ROOT / "src" / "training" / "train_reaction_diffusion_fno.py"
    )
    data_cfg, model_cfg = config["data"], config["model"]
    train_cfg, score_cfg = config["training"], config["score"]
    command = [
        sys.executable,
        str(script),
        "--data",
        str(PROJECT_ROOT / str(data_cfg["path"])),
        "--out-dir",
        str(out_dir),
        "--fig-dir",
        str(figure_dir),
        "--split-file",
        str(split_file),
    ]
    common = {
        "--seed": config.get("seed", 900),
        "--train-samples": data_cfg["train_samples"],
        "--test-samples": data_cfg["test_samples"],
        "--epochs": train_cfg["epochs"],
        "--batch-size": train_cfg["batch_size"],
        "--num-workers": train_cfg.get("num_workers", 4),
        "--learning-rate": train_cfg.get("learning_rate", 1.0e-3),
        "--eta-min": train_cfg.get("eta_min", 1.0e-5),
        "--weight-decay": train_cfg.get("weight_decay", 1.0e-6),
        "--grad-clip": train_cfg.get("grad_clip", 1.0),
        "--max-score": score_cfg["maximum"],
        "--generation-threshold": score_cfg["generation_threshold"],
        "--under-weight": 1.0,
        "--score-balance-power": 0.0,
        "--bound-penalty": 0.0,
        "--device": train_cfg.get("device", "cuda:0"),
        "--bf16-amp": int(bool(train_cfg.get("bf16_amp", False))),
        "--save-every": train_cfg.get("save_every", 10),
        "--test-every": train_cfg.get("test_every", 50),
        "--timing-repeats": train_cfg.get("timing_repeats", 3),
    }
    if str(train_cfg.get("loss", "mse")) == "weighted":
        common["--under-weight"] = score_cfg.get("under_weight", 2.0)
        common["--score-balance-power"] = score_cfg.get("balance_power", 0.25)
        common["--bound-penalty"] = score_cfg.get("bound_penalty", 0.01)
    for flag, value in common.items():
        _append(command, flag, value)

    architecture = (
        {
            "--nx-input": model_cfg.get("nx_input", 256),
            "--width": model_cfg.get("width", 64),
            "--modes-x": model_cfg.get("modes_x", 16),
            "--modes-t": model_cfg.get("modes_t", 16),
            "--layers": model_cfg.get("layers", 4),
            "--projection-width": model_cfg.get("projection_width", 128),
            "--padding-t": model_cfg.get("padding_t", 9),
            "--dropout": model_cfg.get("dropout", 0.03),
        }
        if problem == "burgers"
        else {
            "--width": model_cfg.get("width", 24),
            "--modes-x": model_cfg.get("modes_x", 12),
            "--modes-y": model_cfg.get("modes_y", 12),
            "--modes-z": model_cfg.get("modes_z", 12),
            "--layers": model_cfg.get("layers", 4),
            "--projection-width": model_cfg.get("projection_width", 64),
            "--padding": model_cfg.get("padding", 4),
            "--dropout": model_cfg.get("dropout", 0.03),
        }
    )
    for flag, value in architecture.items():
        _append(command, flag, value)
    if bool(config.get("deterministic", False)):
        command.append("--deterministic")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    legacy_name = (
        "fno_burgers_continuous_score_test_predictions.mat"
        if problem == "burgers"
        else "fno_reactiondiffusion_3d_score_test_predictions.mat"
    )
    canonical = out_dir / "predictions.mat"
    shutil.copy2(out_dir / legacy_name, canonical)
    return canonical


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    problem = str(config["problem"])
    model_name = str(config["model"]["name"])
    experiment = str(config["experiment"])
    out_dir = PROJECT_ROOT / "result" / "operators" / problem / model_name / experiment
    figure_dir = PROJECT_ROOT / "figures" / "operators" / problem / model_name / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    split_file = PROJECT_ROOT / "result" / "splits" / split_filename(config)
    ensure_split(config, split_file)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    if model_name == "fno":
        prediction_path = run_fno(config, out_dir, figure_dir, split_file)
    else:
        from src.training.train_backbone import train

        prediction_path = train(config, PROJECT_ROOT, out_dir, split_file)
    print(f"Canonical prediction export: {prediction_path}")
    return prediction_path
