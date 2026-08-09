#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.training.run_experiment import run


parser = argparse.ArgumentParser(description="Train/export one Burgers score operator.")
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
run(args.config.resolve())
