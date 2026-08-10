# Experiments — operator training

This directory contains the two experiment suites of the paper:

- `burgers/` — space–time Burgers (1-D space + time, 5100 samples)
- `reaction_diffusion_accu/` — accuracy-stopped 3-D steady reaction–diffusion (up to 3100 samples)

Each suite follows the same two-stage pipeline:

1. **Operator training** (Python) — `operator/`
2. **FEM evaluation** (MATLAB) — `fem/`

This README documents step 1 (operator training). For the full pipeline
(including data download) and FEM evaluation, see the repository root `README.md`.

## Layout

```
experiments/
  burgers/
    operator/run_operator.py
    operator/configs/operators/*.yaml
    fem/burgers_fem_config.m
    fem/run_burgers_fem.m
  reaction_diffusion_accu/
    operator/run_operator.py
    operator/configs/operators/*.yaml
    fem/reaction_diffusion_accu_fem_config.m
    fem/run_reaction_diffusion_accu_fem.m
```

## Prerequisites

- Python 3.12 with PyTorch (CUDA build), `numpy`, `scipy`, `h5py`,
  `matplotlib`, `pyyaml`.
- MATLAB (a recent release).
- The data files referenced by the operator configs (download from the GitHub Release; see `data/README.md`).



## Training a single operator

Every operator is described by exactly one YAML file under
`experiments/<problem>/operator/configs/operators/`. Training and prediction
export are driven by the same entry point:

```bash
python experiments/<problem>/operator/run_operator.py \
  --config experiments/<problem>/operator/configs/operators/<config>.yaml
```

Recommended launch pattern (background, detached, logged). Replace
`/path/to/MeshOp-Learning-Mesh-Refinement-Operators-for-Adaptive-PDE-Solvers` with your repository location and `python`
with your environment's interpreter:

```bash
cd /path/to/MeshOp-Learning-Mesh-Refinement-Operators-for-Adaptive-PDE-Solvers

mkdir -p .runtime/tmp logs
export TMPDIR="$PWD/.runtime/tmp"

# Select the physical GPU. training.device: cuda:0 in the YAML then refers to
# the first *visible* GPU, so it does not need to change.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# FNO: 1000 training samples (reaction-diffusion)
nohup python -u experiments/reaction_diffusion_accu/operator/run_operator.py \
  --config experiments/reaction_diffusion_accu/operator/configs/operators/fno_b1000.yaml \
  > logs/reaction_diffusion_accu_fno_b1000.log 2>&1 &
echo "FNO b1000 PID=$!"

# FNO: 2000 training samples (Burgers)
nohup python -u experiments/burgers/operator/run_operator.py \
  --config experiments/burgers/operator/configs/operators/fno_b2000.yaml \
  > logs/burgers_fno_b2000.log 2>&1 &
echo "FNO b2000 PID=$!"
```

Notes:

- `nohup ... &` detaches the job, so it keeps running after you log out. All
  console output goes to the log file under `logs/`.
- The GPU is selected with `CUDA_VISIBLE_DEVICES`; the YAML field
  `training.device: cuda:0` refers to the first *visible* GPU.
- To run both example commands at the same time on different GPUs, set a
  different `CUDA_VISIBLE_DEVICES` value for each job (e.g. `2` and `3`).
- When you inline the environment variables instead of using `export`, keep a
  space before the line-continuation backslash:
  `CUDA_VISIBLE_DEVICES=1 \` (a missing space, `CUDA_VISIBLE_DEVICES=1\`,
  makes the shell treat `nohup` as part of the variable value).
- The two commands above are minimal examples; the shipped pretrained exports under `result/operators/` correspond to the b3000 variants.

## Configurations

Supported models: `fno`, `cno`, `deeponet`, `deeponet_multi` (multi-branch
DeepONet), `pod_deeponet`. The tables below list every shipped
config. "Output experiment" is the sub-directory created under
`result/operators/<problem>/<model>/`.
FNO parameter counts are reported in complex-valued parameters; each complex parameter corresponds to two real-valued scalars.

### Burgers (`data/burgers/burgers_5100.mat`, 5100 samples)

| Config | Model | Train samples | Output experiment |
| --- | --- | --- | --- |
| `fno_b1000.yaml` | FNO | 1000 | `b1000_mse` |
| `fno_b2000.yaml` | FNO | 2000 | `b2000_mse` |
| `fno_b3000.yaml` | FNO | 3000 | `b3000_mse` |
| `cno_b1000.yaml` | CNO | 1000 | `b1000_mse` |
| `cno_b3000.yaml` | CNO | 3000 | `b3000_mse` |
| `deeponet_b2000.yaml` | DeepONet | 2000 | `b2000_mse` |
| `deeponet_b3000.yaml` | DeepONet | 3000 | `b3000_mse` |
| `deeponet_multi_b3000.yaml` | DeepONet multi-branch | 3000 | `b3000_mse` |
| `pod_deeponet_b2000.yaml` | POD-DeepONet | 2000 | `b2000_mse` |
| `pod_deeponet_b3000.yaml` | POD-DeepONet | 3000 | `b3000_mse` |

### Reaction–diffusion (accuracy-stopped)

| Config | Model | Train samples | Output experiment |
| --- | --- | --- | --- |
| `fno_b1000.yaml` | FNO | 1000 | `rd_accu_b1000_mse` |
| `fno_b2000.yaml` | FNO | 2000 | `rd_accu_b2000_mse` |
| `fno_b3000.yaml` | FNO | 3000 | `rd_accu_b3000_mse` |
| `cno_b3000.yaml` | CNO | 3000 | `rd_accu_b3000_mse` |
| `deeponet_b3000.yaml` | DeepONet | 3000 | `rd_accu_b3000_mse` |
| `pod_deeponet_b3000.yaml` | POD-DeepONet | 3000 | `rd_accu_b3000_mse` |

> Output folders are `result/operators/<problem>/<model>/<experiment>/`, so
> different models never overwrite each other. Re-running the same config
> does overwrite its own `predictions.mat`; back it up if needed.

## Outputs

Each run writes to `result/operators/<problem>/<model>/<experiment>/`:

- `predictions.mat` — canonical test-set score predictions consumed by the FEM stage
- `final_<model>_score_model.pt` / `last_<model>_score_model.pt` (FNO), or `final_model.pt` (CNO/DeepONet/POD-DeepONet) — trained weights
- `resolved_config.json`, `run_config.json` — exact configuration used
- `final_metrics.json`, `training_history.csv`, `per_sample_test_metrics.csv`,
  `periodic_test_metrics.csv`
- console log under `logs/`

## Monitoring and troubleshooting

- `jobs -l` — shows running jobs (`Running`) or the exit status of finished ones.
- `tail -f logs/<name>.log` — follow training progress.
- `nvidia-smi` — check GPU utilization.
- Exit code 2 usually means the Python entry point could not be opened
  (`can't open file 'experiments/...'`) because the command was launched from
  the wrong working directory — all paths are relative to the repository
  root, so `cd` into the repo first.
- Training is long: e.g. the RD-accu FNO b3000 run takes about 19 hours on an
  RTX A6000 (1000 epochs). If a run is slow, make sure
  `training.bf16_amp: true` is set and increase `training.batch_size` up to
  what GPU memory allows.
- Re-running an already trained experiment overwrites the existing
  `result/operators/...` folder; back it up if you still need the old
  predictions.

## Next step: FEM evaluation

After training, set the trained operator in the FEM config
(`experiments/<problem>/fem/*_fem_config.m`, fields `operatorName` and
`operatorExperiment`), then run from MATLAB:

```matlab
run_burgers_fem
run_reaction_diffusion_accu_fem
```
