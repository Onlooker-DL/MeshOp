#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p data/reaction_diffusion logs/reaction_diffusion
matlab -batch "cd('data/reaction_diffusion'); generate_reaction_diffusion_exact_3100"
