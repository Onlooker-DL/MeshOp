#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/cd_disk
for model in fno cno deeponet pod_deeponet; do
  echo "===== ${model} ====="
  python -u src/training/train_cd_disk.py \
    --config "experiments/cd_disk/configs/${model}.yaml" \
    > "logs/cd_disk/${model}_b3000_spectral_solution.log" 2>&1
done
