from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib.tri as mtri
import numpy as np
import yaml

# Support direct execution as ``python src/evaluation/evaluate_burgers.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fem import BurgersFourierInput, solve_burgers
from src.meshing import GmshSizeFieldMesher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct 100 Gmsh meshes and solve the Burgers FEM problems"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-first-meshes", type=int, default=3)
    parser.add_argument(
        "--save-mesh-test-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Exact one-based test IDs whose raw meshes are saved. "
            "When supplied, this overrides --save-first-meshes."
        ),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def interpolate_solution(
    nodes: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    interpolator = mtri.LinearTriInterpolator(triangulation, values)
    x_grid, t_grid = np.meshgrid(x, t)
    result = np.ma.asarray(interpolator(x_grid, t_grid))
    if np.any(np.ma.getmaskarray(result)):
        x_inside = np.clip(x_grid, -1.0 + 1.0e-10, 1.0 - 1.0e-10)
        t_inside = np.clip(t_grid, 1.0e-12, 1.0 - 1.0e-12)
        fallback = np.ma.asarray(interpolator(x_inside, t_inside))
        result = np.ma.where(np.ma.getmaskarray(result), fallback, result)
    if np.any(np.ma.getmaskarray(result)):
        raise RuntimeError("Some reference-grid points could not be located in the Gmsh mesh")
    return np.asarray(result, dtype=np.float64)


def physical_metrics(prediction: np.ndarray, reference: np.ndarray, x: np.ndarray, t: np.ndarray):
    if prediction.shape != reference.shape or prediction.shape != (t.size, x.size):
        raise ValueError("Prediction/reference axes do not match [t,x]")
    wx = np.full(x.size, 2.0 / x.size, dtype=np.float64)
    wt = np.empty(t.size, dtype=np.float64)
    wt[0] = 0.5 * (t[1] - t[0])
    wt[-1] = 0.5 * (t[-1] - t[-2])
    wt[1:-1] = 0.5 * (t[2:] - t[:-2])
    weights = wt[:, None] * wx[None, :]
    difference = prediction - reference
    relative_l2 = np.sqrt(np.sum(weights * difference**2) / np.sum(weights * reference**2))
    return float(relative_l2), float(np.mean(difference**2)), float(np.sqrt(np.mean(difference**2)))


def initialize_progress(path: Path, samples: int, nt: int, nx: int) -> h5py.File:
    handle = h5py.File(path, "w")
    handle.create_dataset("fem_solution", (samples, nt, nx), dtype="f4", compression="gzip")
    for name in (
        "relative_l2",
        "mse",
        "rmse",
        "dof",
        "elements",
        "mesh_time_sec",
        "mesh_background_field_time_sec",
        "mesh_gmsh_generate_time_sec",
        "mesh_extraction_time_sec",
        "assembly_time_sec",
        "linear_solve_time_sec",
        "fem_solve_time_sec",
        "postprocess_time_sec",
        "inference_time_sec",
        "total_online_time_sec",
        "newton_iterations",
        "newton_final_residual",
        "newton_converged",
        "source_dataset_index",
    ):
        dtype = "f8"
        if name in {"dof", "elements", "newton_iterations", "source_dataset_index"}:
            dtype = "i8"
        elif name == "newton_converged":
            dtype = "u1"
        handle.create_dataset(name, (samples,), dtype=dtype)
    handle.attrs["completed_samples"] = 0
    handle.flush()
    return handle


def write_csv(path: Path, handle: h5py.File, completed: int) -> None:
    fields = [
        "test_id",
        "source_dataset_index",
        "relative_l2",
        "mse",
        "rmse",
        "dof",
        "elements",
        "mesh_time_sec",
        "mesh_background_field_time_sec",
        "mesh_gmsh_generate_time_sec",
        "mesh_extraction_time_sec",
        "assembly_time_sec",
        "linear_solve_time_sec",
        "fem_solve_time_sec",
        "postprocess_time_sec",
        "inference_time_sec",
        "total_online_time_sec",
        "newton_iterations",
        "newton_final_residual",
        "newton_converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(completed):
            row = {"test_id": index + 1}
            for field in fields[1:]:
                value = handle[field][index]
                row[field] = value.item() if hasattr(value, "item") else value
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = PROJECT_ROOT
    cfg = load_yaml(config_path)
    data_path = project_root / cfg["data"]["path"]
    result_dir = project_root / "results" / "burgers" / "unet" / cfg["experiment"]
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = (
        Path(args.predictions).resolve()
        if args.predictions
        else result_dir / "predicted_mesh_size_test100.h5"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing trained prediction file: {prediction_path}")

    with h5py.File(data_path, "r") as data_handle, h5py.File(prediction_path, "r") as prediction_handle:
        train_samples = int(np.asarray(data_handle.attrs["train_samples"]).reshape(-1)[0])
        test_samples = int(np.asarray(data_handle.attrs["test_samples"]).reshape(-1)[0])
        completed_reference = int(
            np.asarray(data_handle.attrs["completed_reference_samples"]).reshape(-1)[0]
        )
        if completed_reference != test_samples:
            raise RuntimeError(
                f"Only {completed_reference}/{test_samples} spectral references are complete"
            )
        x = np.asarray(data_handle["query_x"]).reshape(-1).astype(np.float64)
        t = np.asarray(data_handle["query_t"]).reshape(-1).astype(np.float64)
        reference = np.asarray(data_handle["reference_solution_test"], dtype=np.float64)
        if reference.shape != (test_samples, t.size, x.size):
            raise ValueError(f"Unexpected reference_solution_test shape {reference.shape}")
        predicted_log_h = np.asarray(prediction_handle["predicted_log_h"], dtype=np.float64)
        prediction_source_ids = np.asarray(
            prediction_handle["source_dataset_index"]
        ).reshape(-1)
        data_source_ids = np.asarray(data_handle["source_dataset_index"]).reshape(-1)
        expected_source_ids = data_source_ids[train_samples : train_samples + test_samples]
        if not np.array_equal(prediction_source_ids, expected_source_ids):
            raise ValueError("Prediction source IDs do not match data test rows")
        inputs = BurgersFourierInput.from_h5(data_handle)

    training_metrics_path = result_dir / "final_metrics.json"
    inference_per_sample = 0.0
    if training_metrics_path.is_file():
        training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
        inference_per_sample = float(training_metrics["mean_inference_time_sec_per_sample"])

    evaluation_cfg = cfg["evaluation"]
    if args.save_mesh_test_ids is None:
        if args.save_first_meshes < 0:
            raise ValueError("--save-first-meshes cannot be negative")
        saved_mesh_test_ids = set(
            range(1, min(args.save_first_meshes, test_samples) + 1)
        )
    else:
        saved_mesh_test_ids = set(args.save_mesh_test_ids)
        invalid_ids = sorted(
            test_id
            for test_id in saved_mesh_test_ids
            if test_id < 1 or test_id > test_samples
        )
        if invalid_ids:
            raise ValueError(
                f"Mesh test IDs must lie in [1,{test_samples}]; got {invalid_ids}"
            )
    mesher = GmshSizeFieldMesher(
        x,
        t,
        size_scale=float(evaluation_cfg["size_scale"]),
        minimum_size=float(evaluation_cfg["minimum_size"]),
        maximum_size=float(evaluation_cfg["maximum_size"]),
        algorithm=int(evaluation_cfg["gmsh_algorithm"]),
        coordinate_mode=str(evaluation_cfg.get("coordinate_mode", "physical_xt")),
    )
    coordinate_mode = str(
        evaluation_cfg.get("coordinate_mode", "physical_xt")
    ).strip().lower()
    output_suffix = "_normalized_xt" if coordinate_mode == "normalized_xt" else ""
    progress_path = result_dir / f"fem_evaluation_test100{output_suffix}.h5"
    if args.overwrite and progress_path.exists():
        progress_path.unlink()
    if progress_path.exists():
        progress = h5py.File(progress_path, "r+")
    else:
        progress = initialize_progress(progress_path, test_samples, t.size, x.size)
    completed = int(progress.attrs["completed_samples"])
    mesh_dir = result_dir / f"meshes{output_suffix}"
    mesh_dir.mkdir(exist_ok=True)
    try:
        for local_index in range(completed, test_samples):
            row = train_samples + local_index
            source_id = int(expected_source_ids[local_index])
            mesh = mesher.generate(predicted_log_h[local_index])
            fem = solve_burgers(
                mesh.nodes,
                mesh.triangles,
                inputs,
                row,
                maximum_iterations=int(evaluation_cfg["newton_max_iterations"]),
                tolerance=float(evaluation_cfg["newton_tolerance"]),
                relative_tolerance=float(evaluation_cfg["newton_relative_tolerance"]),
                maximum_elements=int(evaluation_cfg["maximum_elements"]),
            )
            postprocess_start = time.perf_counter()
            grid_solution = interpolate_solution(
                mesh.nodes, mesh.triangles, fem.solution, x, t
            )
            relative_l2, mse, rmse = physical_metrics(
                grid_solution, reference[local_index], x, t
            )
            postprocess_time = time.perf_counter() - postprocess_start
            total_online = inference_per_sample + mesh.generation_time_sec + fem.total_solve_time_sec
            values = {
                "relative_l2": relative_l2,
                "mse": mse,
                "rmse": rmse,
                "dof": fem.dof,
                "elements": fem.elements,
                "mesh_time_sec": mesh.generation_time_sec,
                "mesh_background_field_time_sec": mesh.background_field_time_sec,
                "mesh_gmsh_generate_time_sec": mesh.gmsh_generate_time_sec,
                "mesh_extraction_time_sec": mesh.extraction_time_sec,
                "assembly_time_sec": fem.assembly_time_sec,
                "linear_solve_time_sec": fem.linear_solve_time_sec,
                "fem_solve_time_sec": fem.total_solve_time_sec,
                "postprocess_time_sec": postprocess_time,
                "inference_time_sec": inference_per_sample,
                "total_online_time_sec": total_online,
                "newton_iterations": fem.newton_iterations,
                "newton_final_residual": fem.final_residual,
                "newton_converged": int(fem.converged),
                "source_dataset_index": source_id,
            }
            progress["fem_solution"][local_index] = grid_solution.astype(np.float32)
            for name, value in values.items():
                progress[name][local_index] = value
            progress.attrs["completed_samples"] = local_index + 1
            progress.flush()
            test_id = local_index + 1
            if test_id in saved_mesh_test_ids:
                np.savez_compressed(
                    mesh_dir / f"test_{test_id:03d}_source_{source_id}.npz",
                    nodes=mesh.nodes,
                    triangles=mesh.triangles,
                    solution=fem.solution,
                    fem_solution_grid=grid_solution.astype(np.float32),
                    reference_solution=reference[local_index].astype(np.float32),
                    predicted_log_h=predicted_log_h[local_index],
                    relative_l2=np.asarray(relative_l2),
                    dof=np.asarray(fem.dof),
                    elements=np.asarray(fem.elements),
                    inference_time_sec=np.asarray(inference_per_sample),
                    mesh_time_sec=np.asarray(mesh.generation_time_sec),
                    mesh_background_field_time_sec=np.asarray(
                        mesh.background_field_time_sec
                    ),
                    mesh_gmsh_generate_time_sec=np.asarray(
                        mesh.gmsh_generate_time_sec
                    ),
                    mesh_extraction_time_sec=np.asarray(mesh.extraction_time_sec),
                    fem_solve_time_sec=np.asarray(fem.total_solve_time_sec),
                    postprocess_time_sec=np.asarray(postprocess_time),
                    total_online_time_sec=np.asarray(total_online),
                )
            write_csv(
                result_dir / f"per_sample_fem_metrics{output_suffix}.csv",
                progress,
                local_index + 1,
            )
            print(
                f"[{local_index + 1:03d}/{test_samples}] source={source_id} | "
                f"dof={fem.dof:,} elems={fem.elements:,} | relL2={relative_l2:.6e} | "
                f"mesh={mesh.generation_time_sec:.3f}s solve={fem.total_solve_time_sec:.3f}s "
                f"converged={fem.converged}",
                flush=True,
            )

        summary = {
            "problem": "burgers",
            "method": "adapted_chan_unet_continuous_size_gmsh",
            "samples": test_samples,
            "source_id_min": int(np.min(progress["source_dataset_index"][:])),
            "source_id_max": int(np.max(progress["source_dataset_index"][:])),
            "mean_relative_l2": float(np.mean(progress["relative_l2"][:])),
            "median_relative_l2": float(np.median(progress["relative_l2"][:])),
            "maximum_relative_l2": float(np.max(progress["relative_l2"][:])),
            "mean_dof": float(np.mean(progress["dof"][:])),
            "mean_elements": float(np.mean(progress["elements"][:])),
            "mean_inference_time_sec_per_sample": inference_per_sample,
            "mean_mesh_time_sec": float(np.mean(progress["mesh_time_sec"][:])),
            "mean_mesh_background_field_time_sec": float(
                np.mean(progress["mesh_background_field_time_sec"][:])
            ),
            "mean_mesh_gmsh_generate_time_sec": float(
                np.mean(progress["mesh_gmsh_generate_time_sec"][:])
            ),
            "mean_mesh_extraction_time_sec": float(
                np.mean(progress["mesh_extraction_time_sec"][:])
            ),
            "mean_fem_solve_time_sec": float(np.mean(progress["fem_solve_time_sec"][:])),
            "mean_total_online_time_sec": float(np.mean(progress["total_online_time_sec"][:])),
            "mean_postprocess_time_sec_excluded_from_online": float(
                np.mean(progress["postprocess_time_sec"][:])
            ),
            "newton_converged_samples": int(np.sum(progress["newton_converged"][:])),
            "relative_l2_definition": "per-sample physical-domain relative L2",
            "quadrature": "periodic trapezoidal x tensor nonperiodic trapezoidal t",
            "size_scale": float(evaluation_cfg["size_scale"]),
            "timing_definition": {
                "total_online_time_sec": (
                    "model inference + complete mesh creation + FEM solve"
                ),
                "mesh_time_sec": (
                    "Gmsh initialization, background-field construction, "
                    "mesh generation, node/triangle extraction, and finalization"
                ),
                "fem_solve_time_sec": (
                    "periodic reduction, FEM geometry/source setup, nonlinear "
                    "assembly, damped Newton, and sparse linear solves"
                ),
                "excluded": (
                    "training, reference generation, reference-grid interpolation, "
                    "error evaluation, file saving, and plotting"
                ),
            },
        }
        summary["coordinate_mode"] = coordinate_mode
        (result_dir / f"fem_final_metrics{output_suffix}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        progress.close()


if __name__ == "__main__":
    main()
