# Direct-solution neural-operator baselines

This directory contains comparison baselines that predict the PDE solution
directly, rather than predicting the mesh-refinement score used by MeshOp. The
same four operator families are included: FNO, CNO, DeepONet, and
POD-DeepONet.

Three problems are supported:

- space-time Burgers;
- convection-diffusion on the unit disk (`cd_disk`);
- three-dimensional reaction-diffusion.

Each problem uses 3000 training samples and 100 held-out test samples. Burgers
uses source samples 1--3000 for training and 5001--5100 for testing so that its
test set matches the MeshOp Burgers experiment. CD-disk and reaction-diffusion
use source samples 1--3000 for training and 3001--3100 for testing. The
generated reference files and all training outputs are intentionally excluded
from Git.

## Layout

```text
data/                          MATLAB reference-data generators
experiments/<problem>/configs  Model and training YAML files
scripts/<problem>/             Data-generation and run-all launchers
src/models/                    FNO, CNO, DeepONet, and POD-DeepONet
src/training/                  Problem-specific training entry points
```

## Requirements

- MATLAB with the functions required by the included reference solvers;
- Python 3.12;
- PyTorch with a CUDA build;
- NumPy, SciPy, h5py, and PyYAML.

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare the reference data

Download the original MeshOp grid data from the repository's
[latest GitHub Release](https://github.com/Onlooker-DL/MeshOp/releases/latest).
Place the three MAT files under this baseline and then run the included MATLAB
generators. Exact paths, commands, solver descriptions, and output filenames
are documented in [`data/README.md`](data/README.md).

The generators produce:

```text
data/burgers/burgers_spectral_3100.h5
data/cd_disk/cd_disk_spectral_3100.h5
data/reaction_diffusion/reaction_diffusion_exact_3100.h5
```

These HDF5 files are training inputs and are not committed to the repository.

## Train one model

Run all commands from `baseline/neural_operator_solution`. The following
pattern follows the main experiment launch convention: it selects one physical
GPU, detaches the process, and sends all console output to a log file.

### Burgers FNO

```bash
cd /path/to/MeshOp/baseline/neural_operator_solution

mkdir -p .runtime/tmp logs/burgers
export TMPDIR="$PWD/.runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

nohup python -u src/training/train_burgers.py \
  --config experiments/burgers/configs/fno.yaml \
  > logs/burgers/fno_b3000_spectral_solution.log 2>&1 &

echo "Burgers FNO PID=$!"
```

### CD-disk FNO

```bash
cd /path/to/MeshOp/baseline/neural_operator_solution

mkdir -p .runtime/tmp logs/cd_disk
export TMPDIR="$PWD/.runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

nohup python -u src/training/train_cd_disk.py \
  --config experiments/cd_disk/configs/fno.yaml \
  > logs/cd_disk/fno_b3000_spectral_solution.log 2>&1 &

echo "CD FNO PID=$!"
```

### Reaction-diffusion FNO

```bash
cd /path/to/MeshOp/baseline/neural_operator_solution

mkdir -p .runtime/tmp logs/reaction_diffusion
export TMPDIR="$PWD/.runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

nohup python -u src/training/train_reaction_diffusion.py \
  --config experiments/reaction_diffusion/configs/fno.yaml \
  > logs/reaction_diffusion/fno_b3000_exact_solution.log 2>&1 &

echo "RD FNO PID=$!"
```

The YAML field `training.device: cuda:0` means the first GPU visible to the
process. For example, after `CUDA_VISIBLE_DEVICES=3`, `cuda:0` is physical GPU
3.

To train CNO, DeepONet, or POD-DeepONet, replace `fno.yaml` with
`cno.yaml`, `deeponet.yaml`, or `pod_deeponet.yaml`. To run all four models
sequentially on one GPU, use:

```bash
bash scripts/burgers/run_all.sh
bash scripts/cd_disk/run_all.sh
bash scripts/reaction_diffusion/run_all.sh
```

Add `--resume` to a single-model command to continue from
`checkpoint_last.pt` in the same output directory.

## Settings and outputs

The model scales follow the corresponding MeshOp experiments as closely as
possible. Every configuration uses raw-target MSE with AdamW,
CosineAnnealingLR, initial learning rate `1e-3`, minimum learning rate `1e-5`,
weight decay `1e-6`, and gradient clipping `1.0`.

Training progress is printed every 10 epochs with the loss, learning rate,
interval time, mean seconds per epoch, and elapsed training time. Each run also
reports the trainable parameter count; complex FNO weights are additionally
reported as an equivalent number of real parameters.

Outputs are written under:

```text
results/<problem>/<model>/<experiment>/
```

The directory contains the resolved configuration, checkpoint, training
history, test predictions, and `final_metrics.json`. Final metrics include test
MSE/RMSE/MAE, relative L2 errors, total training time, test inference time,
per-sample inference time, optimizer settings, and parameter counts.
