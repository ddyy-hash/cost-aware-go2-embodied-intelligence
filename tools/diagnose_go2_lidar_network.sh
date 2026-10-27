#!/usr/bin/env bash
# Read-only diagnosis for the Go2 controller-managed lidar stream.
set -Eeuo pipefail

lidar_ip=${GO2_MAPPING_LIVOX_IP:-192.168.123.20}
interface=${GO2_MAPPING_LIVOX_INTERFACE:-eth0}
lidar_mode=${GO2_MAPPING_LIDAR_MODE:-controller_managed}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/.." && pwd)
liveness_probe="$release_root/mapping/robot/go2_mapping_lio/go2_mapping_lio/ros_topic_liveness.py"
capture=false
if [[ ${1:-} == --capture ]]; then
  capture=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--capture]" >&2
  exit 2
fi

[[ $lidar_ip =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || { echo "invalid lidar IP: $lidar_ip" >&2; exit 2; }
[[ $lidar_mode == controller_managed || $lidar_mode == direct_livox ]] || { echo "invalid lidar mode: $lidar_mode" >&2; exit 2; }
ip link show "$interface" >/dev/null 2>&1 || { echo "missing interface: $interface" >&2; exit 3; }
[[ -f $liveness_probe ]] || { echo "missing ROS topic liveness probe: $liveness_probe" >&2; exit 3; }

set +u
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

printf 'LIDAR_DIAG interface=%s mode=%s expected_ip=%s domain=%s\n' "$interface" "$lidar_mode" "$lidar_ip" "$ROS_DOMAIN_ID"
ip -br addr show dev "$interface"

if [[ $lidar_mode == controller_managed ]]; then
  # Go2 EDU exposes the MID-360 through controller DDS, not a separately
  # reachable raw-Livox IP. An ARP miss at .20 is therefore not a lidar fault.
  printf 'CHECK lidar_l2 NOT_APPLICABLE controller_managed_stream\n'
else
  printf 'NEIGHBOR before=%s\n' "$(ip neigh show "$lidar_ip" dev "$interface" || true)"
  ping -I "$interface" -c 2 -W 1 "$lidar_ip" || true
  printf 'NEIGHBOR after=%s\n' "$(ip neigh show "$lidar_ip" dev "$interface" || true)"
  if ip neigh show "$lidar_ip" dev "$interface" | grep -Eq 'REACHABLE|STALE|DELAY|PROBE'; then
    printf 'CHECK lidar_l2 PASS neighbor_present\n'
  else
    printf 'CHECK lidar_l2 FAIL no_arp_neighbor\n'
  fi
fi

printf 'LIVOX_PROCESSES\n'
pgrep -af 'livox_ros_driver2|livox_lidar_publisher' || true
printf 'LIVOX_PORTS\n'
ss -uapn | grep -E '(:5610[01]|:5620[01]|:5630[01]|:5640[01]|:5650[01])' || true

if ros2 topic info /livox/lidar 2>/dev/null; then
  if python3 "$liveness_probe" --topic /livox/lidar --message-type sensor_msgs/msg/PointCloud2 --timeout 5; then
    printf 'CHECK lidar_ros PASS live_sample\n'
  else
    printf 'CHECK lidar_ros FAIL no_live_sample\n'
  fi
else
  printf 'CHECK lidar_ros FAIL topic_absent\n'
fi

if [[ $capture == true ]]; then
  if sudo -n true 2>/dev/null; then
    printf 'RTPS_CAPTURE_BEGIN\n'
    if [[ $lidar_mode == controller_managed ]]; then
      sudo -n timeout 5s tcpdump -nn -i "$interface" -c 40 'udp port 7401' || true
    else
      sudo -n timeout 5s tcpdump -nn -i "$interface" -c 40 "host $lidar_ip and udp" || true
    fi
    printf 'RTPS_CAPTURE_END\n'
  else
    printf 'CHECK packet_capture WARN sudo_noninteractive_unavailable\n'
  fi
fi
