#!/usr/bin/env python3
"""Train/export one CD-disk score operator (polar grid 64x128).

Usage:
    python experiments/cd_disk/operator/run_operator.py \
        --config experiments/cd_disk/operator/configs/operators/fno_b3000.yaml

Supported models: fno, cno, deeponet, pod_deeponet.  All models predict the
refinement score on the shared 64x128 polar query grid (r clustered toward 1).

- fno/cno : grid models, input (B, C, theta, r) so the periodic theta axis
            is the FNO x-dimension.
- deeponet : branch = coarse 16x16 forcing sensors + (cos,sin) beta;
             trunk = (r, theta) query coordinates.
- pod_deeponet : branch as deeponet; POD basis computed from the training
             targets; the network predicts POD coefficients.

Parameter counts are reported in REAL degrees of freedom (complex weights
count as two reals); the complex-DOF count is also printed.  final_metrics.json
carries the same score metrics as the Burgers/Reaction-Diffusion training
scripts (mae/rmse/rounded_accuracy/within_0p5/within_1/under/over/ceil mae)
plus inference timing and storage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

COARSE_R = 16   # coarse branch sensor grid: 16 x 16 = 256 forcing sensors
COARSE_T = 16


def load_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_cd_dataset(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        target = np.transpose(
            np.asarray(f["target_score"][:]), (2, 1, 0)
        ).astype(np.float32)                     # (nr, nth, N)
        forcing = np.transpose(
            np.asarray(f["forcing_query"][:]), (2, 1, 0)
        ).astype(np.float32)                     # (nr, nth, N)
        beta = np.asarray(f["beta_angle"][:]).ravel().astype(np.float32)
        r = np.asarray(f["rVec"][:]).ravel().astype(np.float32)
        th = np.asarray(f["thetaVec"][:]).ravel().astype(np.float32)
    nr, nth, N = target.shape
    RR, TT = np.meshgrid(r, th, indexing="ij")
    x = RR * np.cos(TT)
    y = RR * np.sin(TT)
    cb = np.cos(beta)
    sb = np.sin(beta)
    branch = np.stack(
        [
            forcing,
            np.broadcast_to(x[..., None], (nr, nth, N)),
            np.broadcast_to(y[..., None], (nr, nth, N)),
            np.broadcast_to(cb[None, None, :], (nr, nth, N)),
            np.broadcast_to(sb[None, None, :], (nr, nth, N)),
        ],
        axis=0,
    ).astype(np.float32)                         # (5, nr, nth, N)

    rs = np.linspace(0.0, 1.0, COARSE_R).astype(np.float32)
    ts = np.linspace(0.0, 2 * np.pi, COARSE_T, endpoint=False).astype(np.float32)
    f_coarse = np.empty((COARSE_R, COARSE_T, N), dtype=np.float32)
    for i in range(COARSE_R):
        ri = int(np.argmin(np.abs(r - rs[i])))
        for j in range(COARSE_T):
            tj = int(np.argmin(np.abs(th - ts[j])))
            f_coarse[i, j, :] = forcing[ri, tj, :]
    branch_sensors = np.concatenate(
        [f_coarse.reshape(COARSE_R * COARSE_T, N).T,
         cb[:, None], sb[:, None]], axis=1
    ).astype(np.float32)                         # (N, 258)

    r_flat = np.broadcast_to(RR[..., None], (nr, nth, N)).reshape(-1, N).T.astype(np.float32)
    t_flat = np.broadcast_to(TT[..., None], (nr, nth, N)).reshape(-1, N).T.astype(np.float32)
    coords = np.stack([r_flat, t_flat], axis=-1)  # (N, nQ, 2)
    coords = coords[0]                            # identical for all samples
    return branch, target, r, th, branch_sensors, coords


def count_real_parameters(model) -> int:
    """Count trainable parameters in real DOF (complex weights = 2 reals)."""
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if p.dtype.is_complex:
            n *= 2
        total += n
    return total


def parameter_storage_mib(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (2**20)


def build_model(cfg: dict, rank: int | None = None):
    import torch  # noqa: F401

    name = cfg["model"]["name"]
    m = cfg["model"]
    if name == "fno":
        from src.training.train_burgers_fno import FNO2dScoreRegressor

        return FNO2dScoreRegressor(
            in_channels=int(m["in_channels"]),
            width=int(m["width"]),
            modes_x=int(m["modes_x"]),
            modes_t=int(m["modes_t"]),
            layers=int(m["layers"]),
            projection_width=int(m["projection_width"]),
            padding_t=int(m.get("padding_t", 0)),
            dropout=float(m.get("dropout", 0.0)),
        )
    if name == "cno":
        from src.models.cno import CNO

        return CNO(
            dim=2,
            in_channels=int(m["in_channels"]),
            out_channels=int(m["out_channels"]),
            width=int(m["width"]),
            levels=int(m["levels"]),
            n_res_bottleneck=int(m["n_res_bottleneck"]),
            n_res_intermediate=int(m["n_res_intermediate"]),
            kernel_size=int(m.get("kernel_size", 3)),
            activation=str(m.get("activation", "gelu")),
            dropout=float(m.get("dropout", 0.0)),
        )
    if name == "deeponet":
        from src.models.deeponet import DeepONet

        return DeepONet(
            branch_dim=int(m["branch_dim"]),
            coordinate_dim=2,
            latent_dim=int(m["latent_dim"]),
            hidden_dim=int(m["hidden_dim"]),
            branch_depth=int(m["branch_depth"]),
            trunk_depth=int(m["trunk_depth"]),
            activation=str(m.get("activation", "gelu")),
            dropout=float(m.get("dropout", 0.0)),
        )
    if name == "pod_deeponet":
        if rank is None:
            raise ValueError("pod_deeponet requires a POD rank.")
        from src.models.pod_deeponet import PODCoefficientNet

        return PODCoefficientNet(
            branch_dim=int(m["branch_dim"]),
            rank=int(rank),
            hidden_dim=int(m["hidden_dim"]),
            depth=int(m["depth"]),
            activation=str(m.get("activation", "gelu")),
            dropout=float(m.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported model: {name}")


def compute_pod_basis(target_flat: np.ndarray, energy: float, max_rank: int):
    """SVD of the training targets; returns (basis, coefficient target)."""
    U, S, Vt = np.linalg.svd(target_flat, full_matrices=False)
    var = S**2
    total = var.sum()
    cum = np.cumsum(var) / max(total, 1e-30)
    rank = int(np.searchsorted(cum, energy) + 1)
    rank = max(1, min(rank, max_rank, Vt.shape[0]))
    basis = np.ascontiguousarray(Vt[:rank].T, dtype=np.float32)   # (nQ, rank)
    coeffs = np.ascontiguousarray(target_flat @ basis, dtype=np.float32)  # (N, rank)
    return basis, coeffs, rank


def score_generation_torch(score, max_score, threshold):
    """Same score-to-generation rule as the MATLAB FEM code:
    floor(s + 1 - threshold), clamped to [0, max_score]."""
    import torch
    s = torch.clamp(score, 0.0, float(max_score))
    return torch.clamp(torch.floor(s + 1.0 - threshold), 0, int(max_score))


def compute_score_metrics(pred_grid, target_grid, max_score, threshold):
    """Same metric definitions as the Burgers training script."""
    import torch
    p = torch.clamp(pred_grid.detach(), 0.0, float(max_score))
    t = target_grid.detach()
    d = p - t
    gp = score_generation_torch(p, max_score, threshold)
    gt = score_generation_torch(t, max_score, threshold)
    return {
        "mae": float(d.abs().mean().item()),
        "rmse": float(torch.sqrt(d.square().mean()).item()),
        "rounded_accuracy": float((gp == gt).float().mean().item()),
        "within_0p5": float((d.abs() <= 0.5).float().mean().item()),
        "within_1": float((d.abs() <= 1.0).float().mean().item()),
        "under_refinement_rate": float((gp < gt).float().mean().item()),
        "over_refinement_rate": float((gp > gt).float().mean().item()),
        "ceil_level_mae": float((gp - gt).abs().float().mean().item()),
    }


def predict_scores_torch(model, tensors, ids, device):
    """Predicted score grid (len(ids), nth, nr) for the given samples."""
    import torch
    import torch.nn.functional as F

    kind = tensors["kind"]
    nth, nr = tensors["nth"], tensors["nr"]
    if kind == "grid":
        xb = torch.from_numpy(tensors["X"][ids]).to(device)
        o = model(xb).squeeze(1)
        if o.shape[-2:] != (nth, nr):
            o = F.interpolate(
                o.unsqueeze(1), size=(nth, nr),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        return o
    if kind == "point":
        bb = torch.from_numpy(tensors["branch"][ids]).to(device)
        cb = torch.from_numpy(tensors["coords"]).to(device)
        o = model(bb, cb)
        return o.reshape(len(ids), nth, nr)
    if kind == "pod":
        bb = torch.from_numpy(tensors["branch"][ids]).to(device)
        coeff = model(bb)
        basis = torch.from_numpy(tensors["basis"].T).to(device)
        o = coeff @ basis
        return o.reshape(len(ids), nth, nr)
    raise ValueError(kind)


def measure_inference_time(model, tensors, device, idx, batch_size=64, repeats=3):
    import torch

    device = torch.device(device)
    model.eval()
    idx = np.asarray(idx)

    def run_once():
        start = time.perf_counter()
        with torch.no_grad():
            for s in range(0, len(idx), batch_size):
                ids = idx[s : s + batch_size]
                predict_scores_torch(model, tensors, ids, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - start

    for _ in range(2):
        run_once()
    times = [run_once() for _ in range(repeats)]
    mean_total = float(np.mean(times))
    n = max(len(idx), 1)
    return {
        "total_test_inference_time_sec": mean_total,
        "mean_inference_time_sec_per_sample": mean_total / n,
        "throughput_samples_per_sec": n / mean_total,
        "timing_repeats": repeats,
    }


def save_checkpoint(path: Path, model, optim, scheduler, epoch: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optim.state_dict(),
         "scheduler": scheduler.state_dict(), "epoch": int(epoch)},
        path,
    )


def load_checkpoint(path: Path, model, optim, scheduler, device) -> int:
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["epoch"])

def main() -> None:
    ap = argparse.ArgumentParser(description="Train/export one CD-disk operator.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoint_last.pt if it exists")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    cfg = load_config(args.config)
    dcfg, mcfg, tcfg, score_cfg = cfg["data"], cfg["model"], cfg["training"], cfg["score"]
    problem, experiment = cfg["problem"], cfg["experiment"]
    name = mcfg["name"]

    branch, target, r, th, branch_sensors, coords = load_cd_dataset(ROOT / dcfg["path"])
    nr, nth, N = target.shape
    nQ = nr * nth
    total = int(dcfg["total_samples"])
    ntrain = int(dcfg["train_samples"])
    ntest = int(dcfg["test_samples"])
    assert ntrain + ntest <= total
    train_idx = np.arange(ntrain)
    test_idx = np.arange(total - ntest, total)

    target_grid = np.ascontiguousarray(np.transpose(target, (2, 1, 0)), dtype=np.float32)  # (N,nth,nr)
    target_flat = target_grid.reshape(N, nQ)

    # ---------------- model-specific tensors ----------------
    if name in ("fno", "cno"):
        X = np.transpose(branch, (3, 0, 2, 1)).astype(np.float32)  # (N,C,nth,nr)
        X = np.ascontiguousarray(X)
        model = build_model(cfg)
        tensors = {"kind": "grid", "X": X, "nth": nth, "nr": nr}
    elif name == "deeponet":
        model = build_model(cfg)
        tensors = {
            "kind": "point",
            "branch": np.ascontiguousarray(branch_sensors, dtype=np.float32),
            "coords": np.ascontiguousarray(coords, dtype=np.float32),
            "nth": nth, "nr": nr,
        }
    elif name == "pod_deeponet":
        pod_cfg = cfg.get("pod", {})
        basis, coeffs, rank = compute_pod_basis(
            target_flat[train_idx],
            float(pod_cfg.get("energy", 0.99)),
            int(pod_cfg.get("maximum_rank", 196)),
        )
        model = build_model(cfg, rank=rank)
        tensors = {
            "kind": "pod",
            "branch": np.ascontiguousarray(branch_sensors, dtype=np.float32),
            "basis": basis,
            "nth": nth, "nr": nr,
        }
        print(f"[cd_disk] POD rank={rank} (energy={pod_cfg.get('energy', 0.99)})")
    else:
        raise ValueError(f"Unsupported model: {name}")

    model = model.to(tcfg["device"])
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    max_score = int(score_cfg["maximum"])
    threshold = float(score_cfg.get("generation_threshold", 0.5))
    epochs = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=max(epochs, 1),
        eta_min=float(tcfg.get("eta_min", 1.0e-5)),
    )
    batch_size = int(tcfg["batch_size"])
    log_every = int(tcfg.get("log_every", 10))
    complex_dof = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = count_real_parameters(model)
    print(
        f"[cd_disk] model={name} "
        f"trainable_parameters_real={total_params:,} "
        f"(complex_dof={complex_dof:,})"
    )

    def forward_batch(model, ids, train: bool):
        with torch.set_grad_enabled(train):
            out = predict_scores_torch(model, tensors, ids, tcfg["device"])
        yb = torch.from_numpy(target_grid[ids]).to(tcfg["device"])
        return out, yb

    def run_epoch(idx, train: bool):
        model.train(train)
        perm = np.random.permutation(idx) if train else idx
        total_loss = 0.0
        nbatch = 0
        for s in range(0, len(perm), batch_size):
            ids = perm[s : s + batch_size]
            if train:
                optim.zero_grad()
            out, yb = forward_batch(model, ids, train)
            loss = F.mse_loss(out, yb)
            if train:
                loss.backward()
                if tcfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(tcfg["grad_clip"])
                    )
                optim.step()
            total_loss += float(loss.detach().cpu()) * len(ids)
            nbatch += len(ids)
        return total_loss / max(nbatch, 1)

    out_dir = ROOT / "result" / "operators" / problem / name / experiment
    ckpt_path = out_dir / "checkpoint_last.pt"
    start_epoch = 1
    if args.resume and ckpt_path.exists():
        start_epoch = load_checkpoint(
            ckpt_path, model, optim, scheduler, tcfg["device"]
        ) + 1
        print(f"[cd_disk] resumed from epoch {start_epoch}")

    save_every = int(tcfg.get("save_every", 50))
    train_loss = float("nan")
    test_loss = float("nan")
    torch.manual_seed(int(cfg.get("seed", 42)))
    t_start = time.time()
    for ep in range(start_epoch, epochs + 1):
        train_loss = run_epoch(train_idx, True)
        scheduler.step()
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            test_loss = run_epoch(test_idx, False)
            print(
                f"[cd_disk] epoch {ep}/{epochs} train_mse={train_loss:.6e} "
                f"test_mse={test_loss:.6e} ({time.time()-t_start:.0f}s)"
            )
        if ep % save_every == 0 or ep == epochs:
            save_checkpoint(ckpt_path, model, optim, scheduler, ep)
    training_time_sec = time.time() - t_start

    # ---------------- final evaluation (same metrics as burgers/RD) ----------------
    model.eval()
    with torch.no_grad():
        train_metrics_sum = {k: 0.0 for k in
            ("mae", "rmse", "rounded_accuracy", "within_0p5", "within_1",
             "under_refinement_rate", "over_refinement_rate", "ceil_level_mae")}
        train_count = 0
        train_mse_sum = 0.0
        for s in range(0, len(train_idx), 64):
            ids = train_idx[s : s + 64]
            out, yb = forward_batch(model, ids, False)
            m = compute_score_metrics(out, yb, max_score, threshold)
            for k in train_metrics_sum:
                train_metrics_sum[k] += m[k] * len(ids)
            train_mse_sum += float(F.mse_loss(out, yb).item()) * len(ids)
            train_count += len(ids)
        train_loss = train_mse_sum / max(train_count, 1)
        train_metrics = {k: v / max(train_count, 1) for k, v in train_metrics_sum.items()}

        test_metrics_sum = {k: 0.0 for k in train_metrics_sum}
        test_count = 0
        test_mse_sum = 0.0
        for s in range(0, len(test_idx), 64):
            ids = test_idx[s : s + 64]
            out, yb = forward_batch(model, ids, False)
            m = compute_score_metrics(out, yb, max_score, threshold)
            for k in test_metrics_sum:
                test_metrics_sum[k] += m[k] * len(ids)
            test_mse_sum += float(F.mse_loss(out, yb).item()) * len(ids)
            test_count += len(ids)
        test_loss = test_mse_sum / max(test_count, 1)
        test_metrics = {k: v / max(test_count, 1) for k, v in test_metrics_sum.items()}

    timing = measure_inference_time(model, tensors, tcfg["device"], test_idx)

    # ---------------- export predictions.mat (v7.3/HDF5) ----------------
    # Export ONLY the held-out test set (the last ntest samples), with the
    # actual dataset source ids, so downstream FEM evaluates exactly the
    # same 100 test instances used for the reported test metrics.
    model.eval()
    ntest_export = len(test_idx)
    with torch.no_grad():
        pred = torch.zeros(ntest_export, nth, nr, dtype=torch.float32, device=tcfg["device"])
        for s in range(0, ntest_export, 64):
            ids = test_idx[s : s + 64]
            pred[s : s + len(ids)] = predict_scores_torch(
                model, tensors, ids, tcfg["device"])
    pred_np = pred.cpu().numpy()                       # (ntest, nth, nr)
    pred_grid = np.clip(np.transpose(pred_np, (2, 1, 0)), 0.0, float(max_score))

    out_dir = ROOT / "result" / "operators" / problem / name / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.mat"

    import h5py

    with h5py.File(out_path, "w") as f:
        f.create_dataset("pred_score", data=np.ascontiguousarray(np.transpose(pred_grid, (2, 1, 0))))
        f.create_dataset("target_score_test", data=np.ascontiguousarray(np.transpose(target[:, :, test_idx], (2, 1, 0))))
        f.create_dataset("query_r", data=r[np.newaxis])
        f.create_dataset("query_theta", data=th[np.newaxis])
        f.create_dataset("source_attempt_id_test", data=(test_idx + 1)[np.newaxis])
        f.create_dataset("max_score", data=np.asarray([max_score], dtype=np.float32))

    method_text = {
        "fno": "FNO2d continuous refinement-score regression (CD disk)",
        "cno": "CNO refinement-score regression (CD disk)",
        "deeponet": "DeepONet refinement-score regression (CD disk)",
        "pod_deeponet": "POD-DeepONet refinement-score regression (CD disk)",
    }[name]

    metrics = {
        "code_version": f"{name.upper()}_CD_DISK_V1",
        "method": method_text,
        "problem": problem,
        "model": name,
        "training_samples": ntrain,
        "test_samples": ntest,
        "training_time_sec": training_time_sec,
        "final_training_loss": train_loss,
        "final_training_metrics": train_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "timing": timing,
        "trainable_parameters_real": total_params,
        "trainable_parameters": complex_dof,
        "parameter_storage_mib": parameter_storage_mib(model),
        "epochs": epochs,
        "grid": [nr, nth],
    }
    if tensors["kind"] == "pod":
        metrics["pod_rank"] = int(rank)
    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[cd_disk] exported {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
