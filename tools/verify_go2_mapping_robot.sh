#!/usr/bin/env bash
# Read-only preflight for the Go2 mapping release. It does not
# start ROS nodes, alter services, change camera parameters, or use sudo.
set -Eeuo pipefail

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
report_dir=${GO2_MAPPING_REPORT_DIR:-"$release_root/reports"}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
report_file="$report_dir/robot-preflight-$timestamp.txt"

mkdir -p -- "$report_dir"
exec > >(tee "$report_file") 2>&1

pass_count=0
warn_count=0
fail_count=0

check_pass() {
  printf 'CHECK %s PASS %s\n' "$1" "$2"
  pass_count=$((pass_count + 1))
}

check_warn() {
  printf 'CHECK %s WARN %s\n' "$1" "$2"
  warn_count=$((warn_count + 1))
}

check_fail() {
  printf 'CHECK %s FAIL %s\n' "$1" "$2"
  fail_count=$((fail_count + 1))
}

# The active camera/follow stack may use a non-default ROS domain. Inspecting
# its environment is read-only and lets preflight query the same DDS graph.
detect_active_ros_domain() {
  local pid env_file candidate
  while read -r pid; do
    env_file="/proc/$pid/environ"
    [[ -r $env_file ]] || continue
    candidate=$(tr '\0' '\n' < "$env_file" | sed -n 's/^ROS_DOMAIN_ID=//p' | head -n1)
    if [[ -n $candidate ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(pgrep -f 'go2_person_follow|realsense2_camera' || true)
  return 1
}

detect_active_rmw() {
  local pid env_file candidate
  while read -r pid; do
    env_file="/proc/$pid/environ"
    [[ -r $env_file ]] || continue
    candidate=$(tr '\0' '\n' < "$env_file" | sed -n 's/^RMW_IMPLEMENTATION=//p' | head -n1)
    if [[ -n $candidate ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(pgrep -f 'go2_person_follow|realsense2_camera' || true)
  return 1
}

printf 'GO2_MAPPING_PREFLIGHT version=1 timestamp=%s release=%s\n' "$timestamp" "$release_root"
printf 'SYSTEM hostname=%s kernel=%s\n' "$(hostname)" "$(uname -r)"
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  printf 'OS id=%s version=%s\n' "${ID:-unknown}" "${VERSION_ID:-unknown}"
fi
if [[ -r /proc/device-tree/model ]]; then
  printf 'HARDWARE model=%s\n' "$(tr -d '\0' </proc/device-tree/model)"
fi

if [[ -d /home/unitree ]]; then
  check_pass home_unitree /home/unitree
else
  check_fail home_unitree missing
fi

available_gib=$(df -BG /home | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
printf 'STORAGE home_available_gib=%s\n' "${available_gib:-unknown}"
if [[ ${available_gib:-0} -ge 45 ]]; then
  check_pass mapping_storage "${available_gib}GiB available"
else
  check_fail mapping_storage "need at least 45GiB available before reserving 40GiB"
fi

for interface in eth0 wlan0; do
  if ip link show "$interface" >/dev/null 2>&1; then
    state=$(ip -br link show "$interface" | awk '{print $2}')
    printf 'NETWORK interface=%s state=%s\n' "$interface" "$state"
  else
    check_warn "network_$interface" missing
  fi
done

for setup_file in /opt/ros/foxy/setup.bash /unitree/module/graph_pid_ws/install/setup.bash; do
  if [[ -r $setup_file ]]; then
    # Unitree's ROS setup scripts reference unset AMENT variables on this image.
    set +u
    # shellcheck disable=SC1090
    source "$setup_file"
    set -u
    check_pass ros_setup "$setup_file"
  else
    check_fail ros_setup "missing $setup_file"
  fi
done

required_packages=(livox_ros_driver2 lio_sam_ros2 realsense2_camera)
for package_name in "${required_packages[@]}"; do
  package_prefix=$(ros2 pkg prefix "$package_name" 2>/dev/null || true)
  if [[ -n $package_prefix ]]; then
    printf 'ROS package=%s prefix=%s\n' "$package_name" "$package_prefix"
    check_pass "package_$package_name" present
  else
    check_fail "package_$package_name" missing
  fi
done

if lsusb 2>/dev/null | grep -qi 'Intel.*RealSense'; then
  check_pass realsense_usb detected
else
  check_warn realsense_usb not_detected
fi

follow_process=$(pgrep -af 'go2_person_follow|realsense2_camera' || true)
active_ros_domain=${ROS_DOMAIN_ID:-0}
active_rmw=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
if detected_domain=$(detect_active_ros_domain); then
  active_ros_domain=$detected_domain
fi
if detected_rmw=$(detect_active_rmw); then
  active_rmw=$detected_rmw
fi
printf 'ROS_QUERY domain=%s rmw=%s\n' "$active_ros_domain" "$active_rmw"
if [[ -n $follow_process ]]; then
  printf 'PROCESS camera_or_follow=%q\n' "$follow_process"
  check_warn camera_owner_active "mapping must use subscriber-only mode"
else
  check_pass camera_owner_idle "sensor-only mapping mode may own the camera"
fi

topic_list=$(ROS_DOMAIN_ID="$active_ros_domain" RMW_IMPLEMENTATION="$active_rmw" ros2 topic list 2>/dev/null || true)
if [[ -n $topic_list ]]; then
  printf 'ROS_TOPICS_BEGIN\n%s\nROS_TOPICS_END\n' "$topic_list"
  for pattern in 'livox|lidar' 'imu' 'camera|depth'; do
    if grep -Eqi "$pattern" <<<"$topic_list"; then
      check_pass "topic_${pattern//|/_}" observed
    else
      check_warn "topic_${pattern//|/_}" not_observed
    fi
  done
else
  check_warn ros_topics unavailable_or_empty
fi

if command -v tegrastats >/dev/null 2>&1; then
  printf 'JETSON tegrastats_available=true\n'
  check_pass tegrastats available
else
  check_warn tegrastats unavailable
fi

printf 'SUMMARY pass=%d warn=%d fail=%d report=%s\n' "$pass_count" "$warn_count" "$fail_count" "$report_file"
if [[ $fail_count -gt 0 ]]; then
  exit 2
fi
