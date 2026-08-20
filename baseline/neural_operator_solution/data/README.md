# Baseline data preparation

The generated direct-solution labels are not stored in Git. Start from the
three original MeshOp grid-data files published in the repository's
[latest GitHub Release](https://github.com/Onlooker-DL/MeshOp/releases/latest):

- `burgers_5100.mat`;
- `cd_disk_3100.mat`;
- `reaction_diffusion_accu_3100.mat`.

## 1. Place the Release files

Copy or download the files to these exact locations:

```text
baseline/neural_operator_solution/data/burgers/burgers_5100.mat
baseline/neural_operator_solution/data/cd_disk/cd_disk_3100.mat
baseline/neural_operator_solution/data/reaction_diffusion/reaction_diffusion_accu_3100.mat
```

If the Release data have already been placed in the main MeshOp `data/`
directory, copy them from the repository root:

```bash
cd baseline/neural_operator_solution
mkdir -p data/burgers data/cd_disk data/reaction_diffusion

cp ../../data/burgers/burgers_5100.mat data/burgers/
cp ../../data/cd_disk/cd_disk_3100.mat data/cd_disk/
cp ../../data/reaction_diffusion_accu/reaction_diffusion_accu_3100.mat \
  data/reaction_diffusion/
```

## 2. Generate the reference solutions

Run MATLAB from each generator directory:

```matlab
cd('/path/to/MeshOp/baseline/neural_operator_solution/data/burgers')
generate_burgers_spectral_3100

cd('/path/to/MeshOp/baseline/neural_operator_solution/data/cd_disk')
generate_cd_disk_spectral_3100

cd('/path/to/MeshOp/baseline/neural_operator_solution/data/reaction_diffusion')
generate_reaction_diffusion_exact_3100
```

On a Linux server, the same generators can be launched from the baseline root:

```bash
mkdir -p logs/burgers logs/cd_disk logs/reaction_diffusion

nohup bash scripts/burgers/generate_data_matlab.sh \
  > logs/burgers/generate_burgers_spectral_3100.log 2>&1 &

nohup bash scripts/cd_disk/generate_data_matlab.sh \
  > logs/cd_disk/generate_cd_disk_spectral_3100.log 2>&1 &

nohup bash scripts/reaction_diffusion/generate_data_matlab.sh \
  > logs/reaction_diffusion/generate_reaction_diffusion_exact_3100.log 2>&1 &
```

The three outputs are:

```text
data/burgers/burgers_spectral_3100.h5
data/cd_disk/cd_disk_spectral_3100.h5
data/reaction_diffusion/reaction_diffusion_exact_3100.h5
```

## Reference definitions

- Burgers uses a 512-point Fourier pseudo-spectral ETDRK4 solver with
  `dt=2.5e-4`, viscosity `nu=0.005`, and 2/3 de-aliasing. The result is sampled
  on the original 101 by 101 query grid.
- CD-disk uses the Fourier-Chebyshev disk reference solver with
  `epsilon=1e-3`, 255 radial Chebyshev intervals, and angular modes
  `-128:128`. It writes the result on the original 64 by 128 query grid.
- Reaction-diffusion evaluates the discrete sine-series solution of
  `-epsilon^2 Delta(u) + u = 0`, with `epsilon=0.02`, on the original
  65 by 65 by 65 query grid.

All generators read the first 3100 source samples. Training always uses samples
1--3000 and testing uses samples 3001--3100. The HDF5 generators are resumable;
rerunning them continues an incomplete output unless overwrite mode is
explicitly requested.
