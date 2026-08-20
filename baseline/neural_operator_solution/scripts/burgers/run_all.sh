#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/burgers

for model in fno cno deeponet pod_deeponet; do
  echo "===== ${model} ====="
  python -u src/training/train_burgers.py \
    --config "experiments/burgers/configs/${model}.yaml" \
    > "logs/burgers/${model}_b3000_spectral_solution.log" 2>&1
done
