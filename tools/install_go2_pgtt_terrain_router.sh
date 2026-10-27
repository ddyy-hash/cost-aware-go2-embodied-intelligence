#!/usr/bin/env bash
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
unit_dir=/home/unitree/.config/systemd/user
mkdir -p "$unit_dir"
install -m 0644 "$release_root/tools/go2-mapping-pgtt-terrain-router.service" "$unit_dir/go2-mapping-pgtt-terrain-router.service"
systemctl --user daemon-reload
systemctl --user enable go2-mapping-pgtt-terrain-router.service
# Reload the running monitor so a release update cannot leave old routing code
# alive behind an unchanged user unit.
systemctl --user restart go2-mapping-pgtt-terrain-router.service
systemctl --user --no-pager --plain status go2-mapping-pgtt-terrain-router.service
