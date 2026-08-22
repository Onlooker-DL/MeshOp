# Neural mesh-size baseline

This standalone, problem-extensible project implements a method-level
continuous mesh-size baseline for MeshOp. Burgers is the first experiment:

```text
data/burgers/                 source MAT, conversion, and test reference
experiments/burgers/configs/  reproducible experiment settings
src/models/                   image-to-image U-Net
src/meshing/                  continuous size field -> periodic Gmsh mesh
src/fem/matlab/               MeshOp-matched MATLAB Burgers FEM evaluation
src/evaluation/               optional all-Python diagnostic evaluator
scripts/burgers/              data and end-to-end launchers
```

The baseline is an adaptation rather than a verbatim reproduction of the
linear-elasticity quad-meshing paper: it predicts a continuous scalar size
field but uses periodic triangular Gmsh meshes and the Burgers space--time FEM.
The distinction from a backbone ablation is essential:

- MeshOp predicts refinement generation and realizes a hierarchical NVB mesh.
- This baseline predicts continuous `log(h)` and asks Gmsh to remesh globally.

Both use the same AFEM-derived target information, random inputs, 3000/100
split, FEM weak form, spectral reference, and physical-domain relative L2
evaluator.

## 1. Environment

Python 3.10 or newer is recommended:

```bash
pip install -r requirements.txt
```

The Python `gmsh` wheel may require system OpenGL libraries on a headless
Linux server (commonly `libGLU.so.1`). MATLAB is used for deterministic data
conversion and for the reported periodic P1 space--time FEM evaluation.

## 2. Data generation

Manually copy the existing MeshOp data file to:

```text
data/burgers/burgers_5100.mat
```

In MATLAB:

```matlab
cd('/path/to/neural_mesh_size_baseline/data/burgers')
generate_burgers_mesh_size_data
```

If a parallel pool cannot start:

```matlab
generate_burgers_mesh_size_data('UseParallel',false)
```

The output `burgers_mesh_size_3100.h5` contains:

- rows 1--3000: source samples 1--3000;
- rows 3001--3100: source samples 5001--5100;
- four U-Net input channels `(u0, f, x, t)`;
- AFEM generation and its equivalent `log(h)` target;
- exact Fourier random coefficients for FEM boundary/source evaluation;
- spectral ETDRK4 solutions for only the 100 test rows.

The conversion is

```text
A0 = ((2/32)*(1/32))/2 = 1/1024
h  = sqrt(2*A0) * 2^(-generation/2).
```

The original MAT and generated HDF5 are ignored by Git. The generator writes
the conversion once and resumes spectral reference generation by completed
batch.

## 3. Train U-Net

From the project root:

```bash
mkdir -p logs/burgers
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

nohup python -u src/training/train_burgers_unet.py \
  --config experiments/burgers/configs/unet.yaml \
  > logs/burgers/unet_b3000_mesh_size.log 2>&1 &

echo "U-Net PID=$!"
```

Add `--resume` to continue `checkpoint.pt`. The parameter-matched U-Net widths
are 3--6--12--24--27, giving 48,064 trainable real parameters versus 48,190
for the Burgers FNO. Training uses raw normalized-log-size MSE, AdamW,
CosineAnnealingLR, learning rate `1e-3 -> 1e-5`, weight decay `1e-6`, seed 42,
1000 epochs, and batch size 16. Input normalization statistics are computed
from the 3000 training samples only.

Outputs are written under:

```text
results/burgers/unet/b3000_continuous_mesh_size_mse/
```

## 4. Export all 100 U-Net/Gmsh meshes

After training:

```bash
nohup python -u src/meshing/export_burgers_meshes.py \
  --config experiments/burgers/configs/unet.yaml \
  > logs/burgers/unet_b3000_gmsh_test100.log 2>&1 &

echo "Gmsh export PID=$!"
```

This reads `predicted_mesh_size_test100.h5`, maps
`xi=(x+1)/2, tau=t`, and constructs each periodic Gmsh mesh isotropically on
the normalized unit square. Returned nodes are mapped back with `x=2*xi-1`
before MATLAB FEM. One MAT file per sample is exported under
`gmsh_meshes_matlab_normalized_xt/`. Existing meshes are reused; pass `--overwrite` only
when they should be regenerated. Each MAT records the full mesh time and its
background-field, Gmsh-core and extraction components.

## 5. Solve and evaluate in MATLAB

From the project root, run either interactively or in batch mode:

```matlab
addpath('experiments/burgers/matlab');
run_burgers_matlab
```

```bash
matlab -batch "addpath('experiments/burgers/matlab'); run_burgers_matlab"
```

The MATLAB evaluation is resumable sample by sample. For every test it:

1. reconstructs the exact stored Fourier initial condition and forcing;
2. solves the common `32 x 32` periodic P1 space--time FEM problem;
3. interpolates that coarse FEM solution to the U-Net/Gmsh mesh as the
   Newton warm start;
4. solves `u_t + u u_x - nu u_xx = f` using MeshOp's Newton tolerances,
   regularization, update cap and backtracking line search;
5. builds the same `N=512`, `dt=2.5e-4` ETDRK4 reference and computes the
   same three-point-quadrature physical-domain relative L2 error;
6. records DOF, elements, Newton iterations and every timing component.

Two explicit online-time columns are reported:

```text
meshop_matched_online_sec
  = U-Net inference + complete Gmsh mesh creation
    + 32x32 coarse FEM solve + final FEM solve

strict_end_to_end_online_sec
  = meshop_matched_online_sec + coarse-to-Gmsh interpolation
```

The first follows MeshOp v1.1.1's current operator-path accounting; the
second is the stricter wall-clock definition. Spectral-reference work, error
integration, plotting and file I/O are excluded from both.

The main outputs under `matlab_fem_normalized_xt/` are
`per_sample_matlab_fem_metrics.csv`, `matlab_fem_results.mat`, and
`final_metrics_matlab.json`.

Tests 4 and 69 save full nodes, triangles, warm start, FEM solution and a PNG
by default. To evaluate only those two first:

```matlab
run_burgers_matlab('TestIds',[4 69])
```

To train and export Gmsh meshes sequentially in one foreground launcher:

```bash
bash scripts/burgers/run_all.sh
```

Then run the MATLAB command above. The older
`src/evaluation/evaluate_burgers.py` remains available only as a diagnostic
all-Python implementation; it is not the reported hybrid baseline.

The normalization is essential because space and time do not share physical
units. The original 32x32 macro grid becomes square in `(xi,tau)`, so its two
triangles are right isosceles there. The normalized scalar size is
`h_hat=(1/32)*2^(-generation/2)`. Mapping the mesh back preserves the physical
2:1 spatial-to-temporal macro-cell aspect ratio. The older physical-coordinate
isotropic result directories are intentionally left untouched.
