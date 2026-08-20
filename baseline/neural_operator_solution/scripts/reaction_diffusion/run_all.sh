#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/reaction_diffusion
for model in fno cno deeponet pod_deeponet; do
  echo "===== ${model} ====="
  python -u src/training/train_reaction_diffusion.py \
    --config "experiments/reaction_diffusion/configs/${model}.yaml" \
    > "logs/reaction_diffusion/${model}_b3000_exact_solution.log" 2>&1
done
