#!/usr/bin/env bash
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
confirmation_file=${GO2_PGTT_CONFIRMATION_FILE:-/home/unitree/go2_mapping/runtime/pgtt-field-confirmation}

exec env PYTHONPATH="$release_root" python3 -m mapping.robot.go2_mapping_locomotion.hardware_runner \
  --arm-hardware \
  --confirmation-file "$confirmation_file" \
  "$@"
