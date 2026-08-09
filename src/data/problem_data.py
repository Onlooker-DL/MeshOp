from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class OperatorData:
    problem: str
    branch: np.ndarray
    target: np.ndarray
    coordinates: np.ndarray
    grid_shape: tuple[int, ...]
    source_attempt_id: np.ndarray
    metadata: dict[str, Any]


def load_problem_data(problem: str, path: Path) -> OperatorData:
    """Load one MATLAB data set in a model-independent representation."""
    if problem == "burgers":
        from src.training.train_burgers_fno import (
            load_dataset,
            periodic_interpolate_u0,
            reconstruct_u0_on_discrete_grid,
        )

        raw = load_dataset(path)
        x_input, u0 = reconstruct_u0_on_discrete_grid(raw, 256)
        forcing = periodic_interpolate_u0(
            np.asarray(raw.forcing_input, dtype=np.float32),
            raw.query_x,
            x_input,
            xmin=-1.0,
            xmax=1.0,
        )
        branch = np.concatenate((u0, forcing), axis=1)
        xx, tt = np.meshgrid(raw.query_x, raw.query_t, indexing="ij")
        coordinates = np.stack((xx, tt), axis=-1).reshape(-1, 2)
        return OperatorData(
            problem=problem,
            branch=np.ascontiguousarray(branch, dtype=np.float32),
            target=np.ascontiguousarray(raw.target, dtype=np.float32),
            coordinates=np.ascontiguousarray(coordinates, dtype=np.float32),
            grid_shape=(raw.query_x.size, raw.query_t.size),
            source_attempt_id=raw.source_attempt_id,
            metadata={
                "query_x": raw.query_x,
                "query_t": raw.query_t,
                "x_input": x_input,
                "U0": u0,
                "F": forcing,
            },
        )

    if problem == "reaction_diffusion":
        from src.training.train_reaction_diffusion_fno import load_dataset

        raw = load_dataset(path)
        branch = raw.boundary.reshape(raw.boundary.shape[0], -1)
        xx, yy, zz = np.meshgrid(raw.query_x,raw.query_y,raw.query_z,indexing="ij",)

        coordinates = np.stack((xx, yy, zz),axis=-1,).reshape(-1, 3)
        return OperatorData(
            problem=problem,
            branch=np.ascontiguousarray(branch, dtype=np.float32),
            target=np.asarray(raw.target),
            coordinates=np.ascontiguousarray(coordinates, dtype=np.float32),
            grid_shape=(raw.query_x.size, raw.query_y.size, raw.query_z.size),
            source_attempt_id=raw.source_attempt_id,
            metadata={
                "query_x": raw.query_x,
                "query_y": raw.query_y,
                "query_s": raw.query_s,
                "query_z": raw.query_z,
                "boundary": raw.boundary,
                "grf_modes": raw.grf_modes,
                "grf_spectral_std": raw.grf_spectral_std,
                "grf_xi_cos": raw.grf_xi_cos,
                "grf_xi_sin": raw.grf_xi_sin,
            },
        )

    raise ValueError(f"Unsupported problem: {problem}")
