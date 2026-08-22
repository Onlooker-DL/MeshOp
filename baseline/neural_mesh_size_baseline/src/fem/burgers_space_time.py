from __future__ import annotations

import time
from dataclasses import dataclass

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _sample_rows(dataset: h5py.Dataset, expected_samples: int) -> np.ndarray:
    values = np.asarray(dataset, dtype=np.float64)
    if values.shape[0] == expected_samples:
        return values
    if values.ndim == 2 and values.shape[1] == expected_samples:
        return values.T
    raise ValueError(f"Cannot identify sample axis in {dataset.name} with shape {values.shape}")


@dataclass(frozen=True)
class BurgersFourierInput:
    ic_coefficients: np.ndarray
    grf_modes: np.ndarray
    grf_std: np.ndarray
    forcing_cos: np.ndarray
    forcing_sin: np.ndarray
    forcing_modes: np.ndarray
    forcing_std: np.ndarray
    source_ids: np.ndarray

    @classmethod
    def from_h5(cls, handle: h5py.File) -> "BurgersFourierInput":
        count = int(handle["input"].shape[0])
        return cls(
            ic_coefficients=_sample_rows(handle["ic_coefficients"], count),
            grf_modes=np.asarray(handle["grf_modes"]).reshape(-1).astype(np.float64),
            grf_std=np.asarray(handle["grf_spectral_std"]).reshape(-1).astype(np.float64),
            forcing_cos=_sample_rows(handle["forcing_xi_cos"], count),
            forcing_sin=_sample_rows(handle["forcing_xi_sin"], count),
            forcing_modes=np.asarray(handle["forcing_modes"]).reshape(-1).astype(np.float64),
            forcing_std=np.asarray(handle["forcing_spectral_std"]).reshape(-1).astype(np.float64),
            source_ids=np.asarray(handle["source_dataset_index"]).reshape(-1).astype(np.int64),
        )

    @staticmethod
    def _series(
        x: np.ndarray,
        cosine: np.ndarray,
        sine: np.ndarray,
        modes: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        angles = np.pi * x[:, None] * modes[None, :]
        return np.cos(angles) @ (std * cosine) + np.sin(angles) @ (std * sine)

    def initial_condition(self, row: int, x: np.ndarray) -> np.ndarray:
        coeff = self.ic_coefficients[row]
        number = self.grf_modes.size
        return -coeff[0] * np.sin(np.pi * (np.asarray(x) + coeff[1])) + self._series(
            x,
            coeff[2 : 2 + number],
            coeff[2 + number : 2 + 2 * number],
            self.grf_modes,
            self.grf_std,
        )

    def forcing(self, row: int, x: np.ndarray) -> np.ndarray:
        return self._series(
            x,
            self.forcing_cos[row],
            self.forcing_sin[row],
            self.forcing_modes,
            self.forcing_std,
        )


@dataclass
class BurgersSolveResult:
    solution: np.ndarray
    dof: int
    elements: int
    converged: bool
    newton_iterations: int
    initial_residual: float
    final_residual: float
    assembly_time_sec: float
    linear_solve_time_sec: float
    total_solve_time_sec: float


@dataclass
class _Geometry:
    reduced_elements: np.ndarray
    area: np.ndarray
    dphidx: np.ndarray
    dphidt: np.ndarray
    forcing_q: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    reduced_nodes: int


def _periodic_reduction(nodes: np.ndarray, tolerance: float = 1.0e-9) -> tuple[np.ndarray, np.ndarray]:
    left = np.flatnonzero(np.abs(nodes[:, 0] + 1.0) <= tolerance)
    right = np.flatnonzero(np.abs(nodes[:, 0] - 1.0) <= tolerance)
    if left.size != right.size:
        raise ValueError(
            f"Periodic boundary mismatch: {left.size} left nodes and {right.size} right nodes"
        )
    left = left[np.argsort(nodes[left, 1])]
    right = right[np.argsort(nodes[right, 1])]
    if left.size and np.max(np.abs(nodes[left, 1] - nodes[right, 1])) > 10 * tolerance:
        raise ValueError("Gmsh periodic left/right nodes do not match in time")
    representative = np.arange(nodes.shape[0], dtype=np.int64)
    representative[right] = left
    _, full_to_reduced = np.unique(representative, return_inverse=True)
    reduced_representative = np.full(full_to_reduced.max() + 1, -1, dtype=np.int64)
    for full, reduced in enumerate(full_to_reduced):
        if reduced_representative[reduced] < 0 or representative[full] == full:
            reduced_representative[reduced] = representative[full]
    return full_to_reduced, reduced_representative


def _precompute_geometry(
    nodes: np.ndarray,
    triangles: np.ndarray,
    full_to_reduced: np.ndarray,
    forcing,
) -> _Geometry:
    p1 = nodes[triangles[:, 0]]
    p2 = nodes[triangles[:, 1]]
    p3 = nodes[triangles[:, 2]]
    det = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (
        p3[:, 0] - p1[:, 0]
    ) * (p2[:, 1] - p1[:, 1])
    area = 0.5 * np.abs(det)
    if np.any(area <= 1.0e-15):
        raise ValueError("Degenerate FEM triangle")
    dphidx = np.column_stack(
        ((p2[:, 1] - p3[:, 1]) / det, (p3[:, 1] - p1[:, 1]) / det, (p1[:, 1] - p2[:, 1]) / det)
    )
    dphidt = np.column_stack(
        ((p3[:, 0] - p2[:, 0]) / det, (p1[:, 0] - p3[:, 0]) / det, (p2[:, 0] - p1[:, 0]) / det)
    )
    barycentric = np.asarray(
        ((1 / 6, 1 / 6, 2 / 3), (1 / 6, 2 / 3, 1 / 6), (2 / 3, 1 / 6, 1 / 6)),
        dtype=np.float64,
    )
    xq = np.column_stack(
        tuple(
            phi[0] * p1[:, 0] + phi[1] * p2[:, 0] + phi[2] * p3[:, 0]
            for phi in barycentric
        )
    )
    reduced_elements = full_to_reduced[triangles]
    rows = np.repeat(reduced_elements, 3, axis=1)
    cols = np.tile(reduced_elements, (1, 3))
    return _Geometry(
        reduced_elements=reduced_elements,
        area=area,
        dphidx=dphidx,
        dphidt=dphidt,
        forcing_q=forcing(xq.ravel()).reshape(xq.shape),
        rows=rows,
        cols=cols,
        reduced_nodes=int(full_to_reduced.max()) + 1,
    )


def _assemble(geometry: _Geometry, u: np.ndarray, viscosity: float, jacobian: bool):
    elem = geometry.reduced_elements
    ue = u[elem]
    ux = np.sum(ue * geometry.dphidx, axis=1)
    ut = np.sum(ue * geometry.dphidt, axis=1)
    barycentric = np.asarray(
        ((1 / 6, 1 / 6, 2 / 3), (1 / 6, 2 / 3, 1 / 6), (2 / 3, 1 / 6, 1 / 6)),
        dtype=np.float64,
    )
    uq = ue @ barycentric.T
    local_residual = np.zeros((elem.shape[0], 3), dtype=np.float64)
    local_jacobian = np.zeros((elem.shape[0], 9), dtype=np.float64) if jacobian else None
    for q, phi in enumerate(barycentric):
        strong = ut + uq[:, q] * ux - geometry.forcing_q[:, q]
        weight = geometry.area / 3.0
        for a in range(3):
            local_residual[:, a] += weight * (
                strong * phi[a] + viscosity * ux * geometry.dphidx[:, a]
            )
            if jacobian:
                for b in range(3):
                    column = 3 * a + b
                    dstrong = (
                        geometry.dphidt[:, b]
                        + phi[b] * ux
                        + uq[:, q] * geometry.dphidx[:, b]
                    )
                    local_jacobian[:, column] += weight * (
                        dstrong * phi[a]
                        + viscosity * geometry.dphidx[:, b] * geometry.dphidx[:, a]
                    )
    residual = np.bincount(
        elem.ravel(), weights=local_residual.ravel(), minlength=geometry.reduced_nodes
    )
    if not jacobian:
        return residual
    matrix = sparse.coo_matrix(
        (local_jacobian.ravel(), (geometry.rows.ravel(), geometry.cols.ravel())),
        shape=(geometry.reduced_nodes, geometry.reduced_nodes),
    ).tocsr()
    return residual, matrix


def solve_burgers(
    nodes: np.ndarray,
    triangles: np.ndarray,
    inputs: BurgersFourierInput,
    row: int,
    *,
    viscosity: float = 5.0e-3,
    maximum_iterations: int = 40,
    tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-6,
    maximum_elements: int = 3_000_000,
) -> BurgersSolveResult:
    nodes = np.asarray(nodes, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.shape[0] > maximum_elements:
        raise RuntimeError(f"Element count {triangles.shape[0]} exceeds {maximum_elements}")
    total_start = time.perf_counter()
    assembly_seconds = 0.0
    linear_seconds = 0.0
    setup_start = time.perf_counter()
    full_to_reduced, representatives = _periodic_reduction(nodes)
    geometry = _precompute_geometry(
        nodes, triangles, full_to_reduced, lambda x: inputs.forcing(row, x)
    )
    reduced_coordinates = nodes[representatives]
    fixed = np.flatnonzero(np.abs(reduced_coordinates[:, 1]) <= 1.0e-9)
    fixed_mask = np.zeros(geometry.reduced_nodes, dtype=bool)
    fixed_mask[fixed] = True
    free = np.flatnonzero(~fixed_mask)
    u = inputs.initial_condition(row, reduced_coordinates[:, 0])
    boundary = inputs.initial_condition(row, reduced_coordinates[fixed, 0])
    u[fixed] = boundary
    assembly_seconds += time.perf_counter() - setup_start

    start = time.perf_counter()
    residual, jacobian = _assemble(geometry, u, viscosity, True)
    assembly_seconds += time.perf_counter() - start
    initial_residual = np.linalg.norm(residual[free]) / np.sqrt(max(free.size, 1))
    target = max(tolerance, relative_tolerance * initial_residual)
    final_residual = initial_residual
    converged = False
    iteration = 0

    for iteration in range(maximum_iterations + 1):
        final_residual = np.linalg.norm(residual[free]) / np.sqrt(max(free.size, 1))
        if final_residual <= target:
            converged = True
            break
        if iteration == maximum_iterations:
            break
        matrix = jacobian[free][:, free]
        rhs = -residual[free]
        base_norm = np.linalg.norm(residual[free])
        accepted = False
        best_norm = base_norm
        best_u: np.ndarray | None = None
        matrix_scale = max(1.0, sparse.linalg.norm(matrix, 1))
        base_mu = max(1.0e-12 * matrix_scale, np.finfo(np.float64).eps * matrix_scale)
        for regularization_try in range(6):
            mu = 0.0 if regularization_try == 0 else base_mu * 10 ** (regularization_try - 1)
            trial_matrix = matrix if mu == 0 else matrix + mu * sparse.eye(matrix.shape[0])
            solve_start = time.perf_counter()
            try:
                update = spsolve(trial_matrix, rhs)
            except Exception:
                linear_seconds += time.perf_counter() - solve_start
                continue
            linear_seconds += time.perf_counter() - solve_start
            if not np.all(np.isfinite(update)):
                continue
            maximum_update = 2.0 * max(1.0, float(np.max(np.abs(u[free]))))
            update_inf = float(np.max(np.abs(update)))
            if update_inf > maximum_update:
                update *= maximum_update / update_inf
            for line_search in range(17):
                step = 2.0 ** (-line_search)
                trial = u.copy()
                trial[free] = u[free] + step * update
                trial[fixed] = boundary
                assemble_start = time.perf_counter()
                trial_residual = _assemble(geometry, trial, viscosity, False)
                assembly_seconds += time.perf_counter() - assemble_start
                trial_norm = np.linalg.norm(trial_residual[free])
                if np.isfinite(trial_norm) and trial_norm < best_norm:
                    best_norm = trial_norm
                    best_u = trial
                if np.isfinite(trial_norm) and trial_norm <= (1.0 - 1.0e-4 * step) * base_norm:
                    u = trial
                    residual = trial_residual
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            if best_u is None or best_norm >= base_norm * (1.0 - 1.0e-12):
                break
            u = best_u
            assemble_start = time.perf_counter()
            residual = _assemble(geometry, u, viscosity, False)
            assembly_seconds += time.perf_counter() - assemble_start
        assemble_start = time.perf_counter()
        residual, jacobian = _assemble(geometry, u, viscosity, True)
        assembly_seconds += time.perf_counter() - assemble_start

    full_solution = u[full_to_reduced]
    return BurgersSolveResult(
        solution=full_solution,
        dof=int(free.size),
        elements=int(triangles.shape[0]),
        converged=converged,
        newton_iterations=int(iteration),
        initial_residual=float(initial_residual),
        final_residual=float(final_residual),
        assembly_time_sec=float(assembly_seconds),
        linear_solve_time_sec=float(linear_seconds),
        total_solve_time_sec=float(time.perf_counter() - total_start),
    )

