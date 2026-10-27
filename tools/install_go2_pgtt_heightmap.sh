#!/usr/bin/env bash
# Install and start the pointcloud-only PGTT heightmap service.
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
unit_source="$release_root/tools/go2-mapping-pgtt-heightmap.service"
unit_dir=/home/unitree/.config/systemd/user
unit_target="$unit_dir/go2-mapping-pgtt-heightmap.service"
[[ -s "$unit_source" ]] || { echo 'PGTT heightmap unit is missing' >&2; exit 3; }
[[ $(id -un) == unitree ]] || { echo 'PGTT heightmap installer must run as unitree' >&2; exit 5; }
mkdir -p "$unit_dir"
install -m 0644 "$unit_source" "$unit_target"
systemctl --user daemon-reload
systemctl --user disable --now go2-mapping-pgtt-heightmap.service >/dev/null 2>&1 || true
printf 'GO2_PGTT_HEIGHTMAP_UNIT_READY service=%s state=disabled\n' "$unit_target"
