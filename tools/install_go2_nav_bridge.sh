#!/usr/bin/env bash
# Maintenance-only installation of the Go2 navigation bridge user service.
set -Eeuo pipefail

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
unit_source="$release_root/tools/go2-mapping-nav-bridge.service"
launcher="$release_root/tools/start_go2_nav_bridge.sh"

[[ -f $unit_source && -f $launcher ]] || { echo 'navigation bridge release files are incomplete' >&2; exit 3; }
service_user=$(id -un)
[[ $service_user == unitree ]] || {
  echo "navigation bridge installer must run as unitree, not $service_user" >&2
  exit 5
}

runtime_dir=/home/unitree/go2_mapping/runtime
nav2_ws=${GO2_MAPPING_NAV2_WS:-$runtime_dir/nav2_ws}
unit_dir=/home/unitree/.config/systemd/user
unit_target="$unit_dir/go2-mapping-nav-bridge.service"
mkdir -p -- "$runtime_dir/navigation" "$nav2_ws" "$unit_dir"
chmod 0700 "$runtime_dir" "$runtime_dir/navigation" "$nav2_ws"

command -v colcon >/dev/null 2>&1 || {
  echo 'navigation bridge installer requires colcon for the ROS overlay build' >&2
  exit 6
}
if [[ -r /opt/ros/foxy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/foxy/setup.bash
  set -u
else
  echo 'navigation bridge installer requires /opt/ros/foxy/setup.bash' >&2
  exit 6
fi
if [[ -r /unitree/module/graph_pid_ws/install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /unitree/module/graph_pid_ws/install/setup.bash
  set -u
fi

# This robot's Foxy colcon exposes --log-base as a global option rather than a
# build-subcommand option.
# The effective build remains `colcon build`; only the global log option moves.
colcon --log-base "$nav2_ws/log" build \
  --merge-install \
  --base-paths "$release_root/mapping/robot/go2_mapping_nav_bridge" \
  --build-base "$nav2_ws/build" \
  --install-base "$nav2_ws/install"
[[ -r "$nav2_ws/install/local_setup.bash" ]] || {
  echo "navigation bridge ROS overlay was not installed: $nav2_ws/install/local_setup.bash" >&2
  exit 6
}

unit_tmp="$unit_target.$$.partial"
trap 'rm -f -- "$unit_tmp"' EXIT
cp -- "$unit_source" "$unit_tmp"
chmod 0644 "$unit_tmp"
mv -f -- "$unit_tmp" "$unit_target"

linger=$(loginctl show-user "$service_user" -p Linger --value)
if [[ $linger != yes ]]; then
  loginctl enable-linger "$service_user"
  linger=$(loginctl show-user "$service_user" -p Linger --value)
fi
[[ $linger == yes ]] || {
  echo "persistent user manager was not enabled for $service_user" >&2
  exit 5
}
systemctl --user daemon-reload
systemctl --user enable --now go2-mapping-nav-bridge.service
systemctl --user restart go2-mapping-nav-bridge.service
systemctl --user is-active --quiet go2-mapping-nav-bridge.service
printf 'GO2_NAV_BRIDGE_READY service=%s\n' "$unit_target"
