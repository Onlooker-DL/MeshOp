#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p data/burgers logs/burgers

matlab -batch "cd('data/burgers'); generate_burgers_spectral_3100"
