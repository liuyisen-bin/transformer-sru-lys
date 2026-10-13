#!/usr/bin/env bash
# Source from wrappers. Python must match the ROS installation's ABI.
set -e
SRU_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRU_ROS_SETUP="${SRU_ROS_SETUP:-/opt/ros/humble/setup.bash}"
SRU_UNITREE_SETUP="${SRU_UNITREE_SETUP:-$SRU_DEPLOY_DIR/../../ros/install/setup.bash}"
SRU_ADAPTER_SETUP="${SRU_ADAPTER_SETUP:-$SRU_DEPLOY_DIR/../../ros/install/setup.bash}"
for SRU_SETUP in "$SRU_ROS_SETUP" "$SRU_UNITREE_SETUP" "$SRU_ADAPTER_SETUP"; do
  if [[ ! -f "$SRU_SETUP" ]]; then
    echo "Missing setup file: $SRU_SETUP. See deploy/transformer_sru_real/README.md." >&2
    return 1
  fi
  source "$SRU_SETUP"
done
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
  if [[ ! "${SRU_NETWORK_INTERFACE:-}" =~ ^[[:alnum:]_.:-]+$ ]]; then
    echo 'Set SRU_NETWORK_INTERFACE to the robot network interface, or supply CYCLONEDDS_URI.' >&2
    return 1
  fi
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$SRU_NETWORK_INTERFACE\" /></Interfaces></General></Domain></CycloneDDS>"
fi
export OPENBLAS_NUM_THREADS=1
SRU_PYTHON="${SRU_PYTHON:-python3}"
