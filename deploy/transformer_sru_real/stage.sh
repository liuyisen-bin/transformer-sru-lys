#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
case "${1:-}" in
  prepare) SRU_STAGE=prepare ;;
  start-policy) SRU_STAGE=start_policy ;;
  damping) SRU_STAGE=damping ;;
  *) echo 'Usage: bash deploy/transformer_sru_real/stage.sh prepare|start-policy|damping' >&2; exit 2 ;;
esac
exec ros2 service call "/transformer_sru/$SRU_STAGE" std_srvs/srv/Trigger '{}'
