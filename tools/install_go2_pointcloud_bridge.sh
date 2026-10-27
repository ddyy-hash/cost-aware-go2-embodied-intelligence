#!/usr/bin/env bash
# Install and start the read-only Unitree pointcloud bridge.
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
unit_source="$release_root/tools/go2-mapping-pointcloud-bridge.service"
unit_dir=/home/unitree/.config/systemd/user
unit_target="$unit_dir/go2-mapping-pointcloud-bridge.service"
[[ -s "$unit_source" ]] || { echo 'pointcloud bridge unit is missing' >&2; exit 3; }
[[ $(id -un) == unitree ]] || { echo 'pointcloud bridge installer must run as unitree' >&2; exit 5; }
mkdir -p "$unit_dir"
install -m 0644 "$unit_source" "$unit_target"
systemctl --user daemon-reload
systemctl --user disable --now go2-mapping-pointcloud-bridge.service >/dev/null 2>&1 || true
printf 'GO2_POINTCLOUD_BRIDGE_UNIT_READY service=%s state=disabled\n' "$unit_target"
