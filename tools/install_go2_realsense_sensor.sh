#!/usr/bin/env bash
# Install the mapping-only RealSense user unit without enabling or starting it.
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
unit_source="$release_root/tools/go2-mapping-realsense-sensor.service"
unit_dir=/home/unitree/.config/systemd/user
unit_target="$unit_dir/go2-mapping-realsense-sensor.service"

[[ -s "$unit_source" ]] || { echo 'RealSense sensor unit is missing' >&2; exit 3; }
[[ $(id -un) == unitree ]] || { echo 'sensor installer must run as unitree' >&2; exit 5; }
mkdir -p "$unit_dir"
install -m 0644 "$unit_source" "$unit_target"
systemctl --user daemon-reload
systemctl --user disable --now go2-mapping-realsense-sensor.service >/dev/null 2>&1 || true
printf 'GO2_REALSENSE_SENSOR_UNIT_READY unit=%s state=disabled\n' "$unit_target"
