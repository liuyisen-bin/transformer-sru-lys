#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec ros2 service call /transformer_sru/stop std_srvs/srv/Trigger '{}'
