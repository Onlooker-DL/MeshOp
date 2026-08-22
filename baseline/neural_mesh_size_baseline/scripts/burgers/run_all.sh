#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs/burgers

python -u src/training/train_burgers_unet.py \
  --config experiments/burgers/configs/unet.yaml

python -u src/meshing/export_burgers_meshes.py \
  --config experiments/burgers/configs/unet.yaml

echo "Gmsh meshes exported. Next run MATLAB from the project root:"
echo "  matlab -batch \"addpath('experiments/burgers/matlab'); run_burgers_matlab\""
