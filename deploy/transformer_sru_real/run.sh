#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$SRU_PYTHON" "$SRU_DEPLOY_DIR/controller.py" "$@"
