#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python -m src.meshing.export_burgers_meshes \
  --config experiments/burgers/configs/unet.yaml "$@"
