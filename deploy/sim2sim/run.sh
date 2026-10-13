#!/usr/bin/env bash
set -euo pipefail
SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1
exec "${SRU_PYTHON:-python3}" "$SIM_DIR/../../training/scripts/nav/evaluate_unitree_mujoco.py" "$@"
