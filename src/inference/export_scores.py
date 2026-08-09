from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from src.data import OperatorData


def score_to_generation(score: np.ndarray, maximum: int, threshold: float) -> np.ndarray:
    clipped = np.clip(score, 0.0, float(maximum))
    return np.clip(
        np.floor(clipped + 1.0 - threshold + 1.0e-6), 0, maximum
    ).astype(np.int16)


def export_predictions(
    path: Path,
    data: OperatorData,
    predictions: np.ndarray,
    test_indices: np.ndarray,
    train_indices: np.ndarray,
    maximum: int,
    threshold: float,
    inference_time_sec: float,
    extras: dict[str, Any] | None = None,
) -> None:
    """Write the same MATLAB-facing schema for every neural operator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    target = np.asarray(data.target[test_indices], dtype=np.float32)
    predictions = np.clip(np.asarray(predictions, dtype=np.float32), 0.0, maximum)
    pred_generation = score_to_generation(predictions, maximum, threshold)
    target_generation = score_to_generation(target, maximum, threshold)

    payload: dict[str, Any] = {
        "original_indices": (test_indices + 1)[:, None],
        "source_attempt_id_test": data.source_attempt_id[test_indices][:, None],
        "train_indices": (np.asarray(train_indices) + 1)[:, None],
        "test_indices": (test_indices + 1)[:, None],
        "generation_threshold": np.array([[threshold]], dtype=np.float64),
        "max_score": np.array([[maximum]], dtype=np.int32),
        "test_inference_time_sec": np.array([[inference_time_sec]], dtype=np.float64),
    }

    if data.problem == "burgers":
        payload.update(
            {
                "pred_score": np.transpose(predictions, (1, 2, 0)),
                "target_score_test": np.transpose(target, (1, 2, 0)),
                "pred_generation": np.transpose(pred_generation, (1, 2, 0)),
                "target_generation": np.transpose(target_generation, (1, 2, 0)),
                "U0_test": data.metadata["U0"][test_indices].T,
                "F_test": data.metadata["F"][test_indices].T,
                "x_input": data.metadata["x_input"][:, None],
                "query_x": data.metadata["query_x"][:, None],
                "query_t": data.metadata["query_t"][:, None],
            }
        )
    elif data.problem == "reaction_diffusion":
        payload.update(
            {
                "pred_score": np.transpose(predictions, (1, 2, 3, 0)),
                "target_score_test": np.transpose(target, (1, 2, 3, 0)),
                "pred_generation": np.transpose(pred_generation, (1, 2, 3, 0)),
                "target_generation": np.transpose(target_generation, (1, 2, 3, 0)),
                "boundary_test": np.transpose(
                    data.metadata["boundary"][test_indices], (1, 2, 0)
                ),
                "query_x": data.metadata["query_x"][:, None],
                "query_y": data.metadata["query_y"][:, None],
                "query_s": data.metadata["query_s"][:, None],
                "query_z": data.metadata["query_z"][:, None],
                "grf_modes": data.metadata["grf_modes"],
                "grf_spectral_std": data.metadata["grf_spectral_std"][:, None],
                "grf_xi_cos_test": data.metadata["grf_xi_cos"][test_indices].T,
                "grf_xi_sin_test": data.metadata["grf_xi_sin"][test_indices].T,
            }
        )
    else:
        raise ValueError(data.problem)

    if extras:
        payload.update(extras)
    sio.savemat(path, payload, do_compression=True)

