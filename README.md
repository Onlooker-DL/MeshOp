# MeshOp: Operator Learning for Adaptive Mesh Refinement in Parametric PDEs

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

Official implementation of *MeshOp: Operator Learning for Adaptive Mesh Refinement in Parametric PDEs*.

This project learns a neural operator that maps the functional inputs of a PDE
instance (initial condition, forcing term, boundary data, physical parameters)
directly to a spatial refinement-demand field. The predicted field is converted
into a conforming hierarchical mesh through clipping, quantization, and
newest-vertex bisection (NVB), and is used as an adaptive initial mesh for a
conventional finite-element solve. Optional residual-based AFEM cycles can be
continued from the operator-generated mesh when additional accuracy is needed.

Four operator backbones are supported: FNO, CNO, DeepONet (including a
multi-branch variant), and POD-DeepONet. Finite-element
computations run in MATLAB; operator training and inference run in Python
(PyTorch). Direct-solution versions of the four neural operators are provided
as comparison baselines under `baseline/neural_operator_solution/`.

## Repository structure

```text
setup.m                         MATLAB path/environment initialization
requirements.txt                Python dependencies
README.md                       Main documentation
VALIDATION.md                   Validation and reproducibility checks
LICENSE                         BSD 3-Clause License

data/
  README.md                     Data download and placement instructions

baseline/
  neural_operator_solution/    Direct PDE-solution operator baselines

examples/
  burgers_fem_single_sample/    Minimal MATLAB example for a one-sample FEM solve

experiments/
  burgers/                      Space-time Burgers experiments
    operator/                   Python operator training/inference + configs
    fem/                        MATLAB FEM evaluation entry points
  reaction_diffusion_accu/      Accuracy-controlled 3-D reaction-diffusion experiments
    operator/                   Python operator training/inference + configs
    fem/                        MATLAB FEM evaluation entry points
  cd_disk/                      Steady convection-diffusion on the unit disk
    operator/                   Python operator training/inference + configs
    fem/                        MATLAB FEM evaluation entry points

src/
  data/                         Data loading helpers
  fem/                          FEM implementations shared by experiments
  inference/                    Score export and inference helpers
  models/                       Neural-operator architectures
  training/                     Training and experiment runner

result/                         Selected trained exports and numerical results
figures/                        Generated figures and mesh visualizations
logs/                           Training and experiment logs
tools/                          Auxiliary utilities and helper scripts
```

## Requirements

### MATLAB

- A recent MATLAB release, used for mesh construction and all finite-element
  solves.

### Python

- Python 3.12
- PyTorch with a CUDA build (experiments in the paper used one NVIDIA RTX A6000,
  48 GB)
- `numpy`,`scipy`, `h5py`, `matplotlib`, `pyyaml`

Example:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install numpy scipy h5py matplotlib pyyaml
pip install torch --index-url https://download.pytorch.org/whl/cu121  # adjust to your CUDA version
```

## Reproduction pipeline

The end-to-end pipeline is: **data download -> operator training
(Python) -> FEM evaluation (MATLAB)**.

### 0. Initialize the MATLAB environment

```matlab
cd /path/to/this/repo
setup;            % or: setup(false) to keep the current path
```

### 1. Data

The data sets are provided as GitHub Release assets (they are too large to store
in the repository itself). Download `burgers_5100.mat`,
`reaction_diffusion_accu_3100.mat`, and `cd_disk_3100.mat` from the latest
Release and place them as:

```
data/burgers/burgers_5100.mat
data/reaction_diffusion_accu/reaction_diffusion_accu_3100.mat
data/cd_disk/cd_disk_3100.mat
```

See `data/README.md` for detailed instructions.
### 2. Operator training (Python)

Each operator configuration is a YAML file under
`experiments/<problem>/operator/configs/operators/`. Train and export
predictions with:

```bash
python experiments/burgers/operator/run_operator.py \
  --config experiments/burgers/operator/configs/operators/fno_b2000.yaml

python experiments/reaction_diffusion_accu/operator/run_operator.py \
  --config experiments/reaction_diffusion_accu/operator/configs/operators/fno_b3000.yaml

python experiments/cd_disk/operator/run_operator.py \
  --config experiments/cd_disk/operator/configs/operators/fno_b3000.yaml
```

Outputs are written to
`result/operators/<problem>/<model>/<experiment>/`:

- `predictions.mat` — test-set score predictions consumed by the FEM stage
- `resolved_config.json` — the exact configuration used
- checkpoints and training logs (see `logs/`)

Full training instructions and the list of all configurations are in
`experiments/README.md`. Selected trained operator exports are included under `result/operators/` so the
FEM stage can be reproduced without re-training — see `result/README.md`.

### Direct-solution neural-operator baselines

For comparison with MeshOp's mesh-refinement-score prediction, the repository
also includes FNO, CNO, DeepONet, and POD-DeepONet baselines that predict the
PDE solution directly. They cover Burgers, disk convection-diffusion, and
three-dimensional reaction-diffusion, using 3000 samples for training and 100
held-out samples for testing. Burgers uses source samples 1--3000 for training
and 5001--5100 for testing, matching MeshOp's held-out Burgers test set.
CD-disk and reaction-diffusion use source samples 1--3000 for training and
3001--3100 for testing.

The baseline reuses the three original MAT files from the same GitHub Release.
Included MATLAB programs generate the required spectral or exact reference
solutions locally; the generated HDF5 files are not committed. See
[`baseline/neural_operator_solution/README.md`](baseline/neural_operator_solution/README.md)
for data preparation, GPU launch commands, configurations, metrics, and output
locations.

### 3. FEM evaluation (MATLAB)

Edit the experiment configuration file
`experiments/<problem>/fem/*_fem_config.m` (operator name/experiment, sample
counts, thresholds, which baselines to run), then run the corresponding entry
point from MATLAB:

```matlab
run_burgers_fem
run_reaction_diffusion_accu_fem
run_cd_disk_fem
```

Each entry point resolves the trained `predictions.mat` and the data set
automatically from `result/operators/...` and `data/...`, and writes:

- `result/fem/<problem>/...` — numerical results (errors, DOFs, timings)
- `figures/fem/<problem>/...` — mesh and convergence figures.


## Reproducibility notes

- Random seeds: all neural-operator configs under `experiments/*/operator/configs/operators/` use `seed: 42`.
- The data split is deterministic (`prefix_train_tail_test`: first N samples
  for training, final 100 samples for testing).
- Paper experiments ran on two Intel Xeon Gold 6240 CPUs, 187 GiB RAM, and one
  NVIDIA RTX A6000 (48 GB). Neural operators were trained on the GPU; mesh
  construction and finite-element solves ran on the CPU in MATLAB.
- Set `deterministic: optional` in a YAML config.

## License

This project is released under the BSD 3-Clause License. See
[LICENSE](LICENSE).
