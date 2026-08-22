from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.io import loadmat, savemat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.meshing import GmshSizeFieldMesher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export U-Net/Gmsh Burgers meshes for the MATLAB FEM evaluator"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-ids", type=int, nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    data_path = PROJECT_ROOT / cfg["data"]["path"]
    result_dir = PROJECT_ROOT / "results" / "burgers" / "unet" / cfg["experiment"]
    evaluation_cfg = cfg["evaluation"]
    coordinate_mode = str(
        evaluation_cfg.get("coordinate_mode", "physical_xt")
    ).strip().lower()
    directory_name = (
        "gmsh_meshes_matlab_normalized_xt"
        if coordinate_mode == "normalized_xt"
        else "gmsh_meshes_matlab"
    )
    if coordinate_mode == "normalized_xt":
        mesh_generation_domain = "[0,1]x[0,1]"
        coordinate_map = "xi=(x+1)/2, tau=t; return x=2*xi-1"
        gmsh_initial_size = 1.0 / 32.0
    else:
        mesh_generation_domain = "[-1,1]x[0,1]"
        coordinate_map = "identity in (x,t)"
        gmsh_initial_size = np.sqrt(2.0) / 32.0
    prediction_path = (
        Path(args.predictions).resolve()
        if args.predictions
        else result_dir / "predicted_mesh_size_test100.h5"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else result_dir / directory_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as data_handle, h5py.File(
        prediction_path, "r"
    ) as prediction_handle:
        train_samples = int(scalar(data_handle.attrs["train_samples"]))
        test_samples = int(scalar(data_handle.attrs["test_samples"]))
        x = np.asarray(data_handle["query_x"]).reshape(-1).astype(np.float64)
        t = np.asarray(data_handle["query_t"]).reshape(-1).astype(np.float64)
        source_ids = np.asarray(data_handle["source_dataset_index"]).reshape(-1)
        test_source_ids = source_ids[train_samples : train_samples + test_samples].astype(
            np.int64
        )
        predicted_log_h = np.asarray(
            prediction_handle["predicted_log_h"], dtype=np.float64
        )
        predicted_source_ids = np.asarray(
            prediction_handle["source_dataset_index"]
        ).reshape(-1)
    if predicted_log_h.shape != (test_samples, t.size, x.size):
        raise ValueError(f"Unexpected predicted_log_h shape {predicted_log_h.shape}")
    if not np.array_equal(predicted_source_ids, test_source_ids):
        raise ValueError("Prediction and dataset test source IDs differ")

    if args.test_ids is None:
        test_ids = list(range(1, test_samples + 1))
    else:
        test_ids = sorted(set(args.test_ids))
        invalid = [test_id for test_id in test_ids if not 1 <= test_id <= test_samples]
        if invalid:
            raise ValueError(f"test IDs must lie in [1,{test_samples}]: {invalid}")

    training_metrics_path = result_dir / "final_metrics.json"
    if not training_metrics_path.is_file():
        raise FileNotFoundError(f"Missing training metrics: {training_metrics_path}")
    training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    inference_per_sample = float(training_metrics["mean_inference_time_sec_per_sample"])
    mesher = GmshSizeFieldMesher(
        x,
        t,
        size_scale=float(evaluation_cfg["size_scale"]),
        minimum_size=float(evaluation_cfg["minimum_size"]),
        maximum_size=float(evaluation_cfg["maximum_size"]),
        algorithm=int(evaluation_cfg["gmsh_algorithm"]),
        coordinate_mode=coordinate_mode,
    )

    rows: list[dict[str, float | int | str]] = []
    for test_id in test_ids:
        local_index = test_id - 1
        source_id = int(test_source_ids[local_index])
        mesh_path = output_dir / f"test_{test_id:03d}_source_{source_id}.mat"
        if mesh_path.exists() and not args.overwrite:
            existing = loadmat(mesh_path, variable_names=[
                "nodes", "triangles", "inference_time_sec", "mesh_time_sec",
                "mesh_background_field_time_sec", "mesh_gmsh_generate_time_sec",
                "mesh_extraction_time_sec", "coordinate_mode",
            ])
            existing_mode = str(np.asarray(existing.get("coordinate_mode", ""))
                                .reshape(-1)[0]).strip()
            if existing_mode != coordinate_mode:
                raise ValueError(
                    f"Existing {mesh_path} has coordinate_mode={existing_mode!r}; "
                    f"expected {coordinate_mode!r}. Use --overwrite."
                )
            row = {
                "test_id": test_id,
                "source_dataset_index": source_id,
                "nodes": int(existing["nodes"].shape[0]),
                "elements": int(existing["triangles"].shape[0]),
                "inference_time_sec": scalar(existing["inference_time_sec"]),
                "mesh_time_sec": scalar(existing["mesh_time_sec"]),
                "mesh_background_field_time_sec": scalar(
                    existing["mesh_background_field_time_sec"]
                ),
                "mesh_gmsh_generate_time_sec": scalar(
                    existing["mesh_gmsh_generate_time_sec"]
                ),
                "mesh_extraction_time_sec": scalar(existing["mesh_extraction_time_sec"]),
                "mesh_file": str(mesh_path.resolve()),
            }
            rows.append(row)
            print(f"[{test_id:03d}/{test_samples}] existing {mesh_path.name}", flush=True)
            continue

        physical_log_h = predicted_log_h[local_index]
        gmsh_log_h = mesher.mesh_log_h(physical_log_h)
        mesh = mesher.generate(physical_log_h)
        # MATLAB uses one-based connectivity.
        triangles_matlab = mesh.triangles.astype(np.int64) + 1
        savemat(
            mesh_path,
            {
                "nodes": mesh.nodes,
                "triangles": triangles_matlab,
                # Stored as [t,x], so MATLAB imagesc(query_x,query_t,field)
                # displays the same orientation as the Python/Gmsh input.
                "predicted_log_h": gmsh_log_h,
                "predicted_log_h_physical_area_equivalent": physical_log_h,
                "query_x": x.reshape(-1, 1),
                "query_x_normalized": (0.5 * (x + 1.0)).reshape(-1, 1),
                "query_t": t.reshape(-1, 1),
                "test_id": np.asarray([[test_id]], dtype=np.int32),
                "source_dataset_index": np.asarray([[source_id]], dtype=np.int32),
                "inference_time_sec": np.asarray([[inference_per_sample]]),
                "mesh_time_sec": np.asarray([[mesh.generation_time_sec]]),
                "mesh_background_field_time_sec": np.asarray(
                    [[mesh.background_field_time_sec]]
                ),
                "mesh_gmsh_generate_time_sec": np.asarray(
                    [[mesh.gmsh_generate_time_sec]]
                ),
                "mesh_extraction_time_sec": np.asarray([[mesh.extraction_time_sec]]),
                "size_scale": np.asarray([[float(evaluation_cfg["size_scale"])]]),
                "coordinate_mode": coordinate_mode,
                "mesh_generation_domain": mesh_generation_domain,
                "returned_node_domain": "[-1,1]x[0,1]",
            },
            do_compression=True,
        )
        row = {
            "test_id": test_id,
            "source_dataset_index": source_id,
            "nodes": int(mesh.nodes.shape[0]),
            "elements": int(mesh.triangles.shape[0]),
            "inference_time_sec": inference_per_sample,
            "mesh_time_sec": mesh.generation_time_sec,
            "mesh_background_field_time_sec": mesh.background_field_time_sec,
            "mesh_gmsh_generate_time_sec": mesh.gmsh_generate_time_sec,
            "mesh_extraction_time_sec": mesh.extraction_time_sec,
            "mesh_file": str(mesh_path.resolve()),
        }
        rows.append(row)
        print(
            f"[{test_id:03d}/{test_samples}] source={source_id} | "
            f"nodes={mesh.nodes.shape[0]:,} elems={mesh.triangles.shape[0]:,} | "
            f"mesh={mesh.generation_time_sec:.3f}s",
            flush=True,
        )

    manifest = output_dir / "mesh_export_metrics.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "problem": "burgers",
        "method": "unet_continuous_mesh_size_gmsh",
        "coordinate_mode": coordinate_mode,
        "mesh_generation_domain": mesh_generation_domain,
        "returned_node_domain": "[-1,1]x[0,1]",
        "coordinate_map": coordinate_map,
        "gmsh_initial_size": gmsh_initial_size,
        "exported_test_ids": test_ids,
        "inference_time_sec_per_sample": inference_per_sample,
        "mesh_time_definition": (
            "Gmsh initialization + background field + core generation + "
            "node/triangle extraction + finalization"
        ),
        "matlab_connectivity": "triangles are one-based",
        "predicted_log_h_axis_order": "[t,x] in normalized (xi,tau) coordinates",
    }
    (output_dir / "mesh_export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
