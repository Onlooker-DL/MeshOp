"""Physical-domain relative L2 metrics on the stored query grids.

The Direct baselines predict nodal values on fixed tensor-product grids. This
module supplies one shared evaluator for training-time reports and checkpoint
re-evaluation. The reported ``mean_relative_l2`` is the mean of per-sample
ratios after physical quadrature; the former flattened-array metric is retained
under ``*_relative_discrete_l2`` keys.
"""

from __future__ import annotations

import numpy as np


def _coordinate_vector(values: np.ndarray, name: str) -> np.ndarray:
    coordinates = np.asarray(values, dtype=np.float64).reshape(-1)
    if coordinates.size < 2:
        raise ValueError(f"{name} must contain at least two coordinates.")
    if not np.all(np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return coordinates


def trapezoidal_nodal_weights(values: np.ndarray, name: str) -> np.ndarray:
    """Nodal weights for the composite trapezoidal rule on a nonuniform grid."""

    coordinates = _coordinate_vector(values, name)
    weights = np.empty_like(coordinates)
    weights[0] = 0.5 * (coordinates[1] - coordinates[0])
    weights[-1] = 0.5 * (coordinates[-1] - coordinates[-2])
    weights[1:-1] = 0.5 * (coordinates[2:] - coordinates[:-2])
    return weights


def periodic_nodal_weights(
    values: np.ndarray, name: str, period: float | None = None
) -> np.ndarray:
    """Periodic nodal quadrature weights, including the wrap-around cell."""

    coordinates = _coordinate_vector(values, name)
    differences = np.diff(coordinates)
    if period is None:
        period = float(coordinates[-1] - coordinates[0] + np.median(differences))
    period = float(period)
    wrap = coordinates[0] + period - coordinates[-1]
    if not np.isfinite(period) or wrap <= 0:
        raise ValueError(f"Invalid period {period} for {name}.")
    previous_gaps = np.concatenate(([wrap], differences))
    next_gaps = np.concatenate((differences, [wrap]))
    return 0.5 * (previous_gaps + next_gaps)


def tensor_product_weights(*axis_weights: np.ndarray) -> np.ndarray:
    if not axis_weights:
        raise ValueError("At least one axis weight vector is required.")
    result = np.asarray(axis_weights[0], dtype=np.float64)
    for weights in axis_weights[1:]:
        result = np.multiply.outer(result, np.asarray(weights, dtype=np.float64))
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Quadrature weights must be finite and positive.")
    return result


def burgers_quadrature_weights(query_x: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Physical dx dt weights; x is periodic and t includes both endpoints."""

    wx = periodic_nodal_weights(query_x, "query_x")
    wt = trapezoidal_nodal_weights(query_t, "query_t")
    return tensor_product_weights(wx, wt)


def cd_disk_quadrature_weights(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Physical r dr dtheta weights on the boundary-clustered polar grid.

    The stored radial samples omit r=0. Radial Voronoi cells are therefore
    bounded by 0 and 1, with midpoint interfaces, and the factor r is integrated
    exactly over every annular cell.
    """

    radius = _coordinate_vector(r, "r")
    if radius[0] <= 0 or radius[-1] > 1.0 + 1.0e-8:
        raise ValueError("The CD radial grid must lie in (0, 1].")
    edges = np.empty(radius.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (radius[:-1] + radius[1:])
    radial_weights = 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)
    angular_weights = periodic_nodal_weights(theta, "theta", period=2.0 * np.pi)
    return tensor_product_weights(radial_weights, angular_weights)


def reaction_diffusion_quadrature_weights(
    query_x: np.ndarray, query_y: np.ndarray, query_z: np.ndarray
) -> np.ndarray:
    """Physical dx dy dz weights, including the nonuniform z=s^2 spacing."""

    return tensor_product_weights(
        trapezoidal_nodal_weights(query_x, "query_x"),
        trapezoidal_nodal_weights(query_y, "query_y"),
        trapezoidal_nodal_weights(query_z, "query_z"),
    )


def _validate_fields(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(prediction)
    truth = np.asarray(target)
    quadrature = np.asarray(weights, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim < 2:
        raise ValueError(
            f"Prediction/target shapes must match and include a sample axis: "
            f"{predicted.shape} versus {truth.shape}."
        )
    if predicted.shape[1:] != quadrature.shape:
        raise ValueError(
            f"Quadrature shape {quadrature.shape} does not match field shape "
            f"{predicted.shape[1:]}."
        )
    if not np.all(np.isfinite(quadrature)) or np.any(quadrature <= 0):
        raise ValueError("Quadrature weights must be finite and positive.")
    return predicted, truth, quadrature


def per_sample_relative_l2(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-quadrature and legacy discrete relative errors."""

    predicted, truth, quadrature = _validate_fields(prediction, target, weights)
    physical = np.empty(predicted.shape[0], dtype=np.float64)
    discrete = np.empty_like(physical)
    for sample_id in range(predicted.shape[0]):
        sample_target = truth[sample_id].astype(np.float64, copy=False)
        difference = predicted[sample_id].astype(np.float64, copy=False) - sample_target
        error_squared = float(np.sum(quadrature * difference * difference))
        reference_squared = float(np.sum(quadrature * sample_target * sample_target))
        physical[sample_id] = np.sqrt(
            max(error_squared, 0.0) / max(reference_squared, 1.0e-30)
        )
        discrete[sample_id] = np.linalg.norm(difference.reshape(-1)) / max(
            np.linalg.norm(sample_target.reshape(-1)), 1.0e-12
        )
    return physical, discrete


def solution_metrics(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Aggregate errors using one physical evaluator for every Direct model."""

    predicted, truth, quadrature = _validate_fields(prediction, target, weights)
    physical, discrete = per_sample_relative_l2(predicted, truth, quadrature)
    squared = 0.0
    absolute = 0.0
    points = 0
    for sample_id in range(predicted.shape[0]):
        difference = predicted[sample_id].astype(np.float64, copy=False) - truth[
            sample_id
        ].astype(np.float64, copy=False)
        squared += float(np.sum(difference * difference))
        absolute += float(np.sum(np.abs(difference)))
        points += difference.size
    return {
        "mse": squared / points,
        "rmse": float(np.sqrt(squared / points)),
        "mae": absolute / points,
        "mean_relative_l2": float(np.mean(physical)),
        "median_relative_l2": float(np.median(physical)),
        "maximum_relative_l2": float(np.max(physical)),
        "mean_relative_discrete_l2": float(np.mean(discrete)),
        "median_relative_discrete_l2": float(np.median(discrete)),
        "maximum_relative_discrete_l2": float(np.max(discrete)),
    }

