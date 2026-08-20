#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p data/cd_disk logs/cd_disk
matlab -batch "cd('data/cd_disk'); generate_cd_disk_spectral_3100"
