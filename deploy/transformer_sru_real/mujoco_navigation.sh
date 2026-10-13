#!/usr/bin/env bash
set -e
SRU_NAV_VIEW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1
# Pure local keyboard simulation: no ROS environment or hardware I/O.
exec "${SRU_PYTHON:-python3}" "$SRU_NAV_VIEW_DIR/mujoco_navigation.py" "$@"
