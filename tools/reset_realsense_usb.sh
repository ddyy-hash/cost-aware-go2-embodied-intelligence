#!/usr/bin/env bash
set -eo pipefail

device_path="${1:-/sys/bus/usb/devices/2-1}"

if [ ! -d "$device_path" ]; then
  echo "device path not found: $device_path" >&2
  exit 1
fi

vendor="$(cat "$device_path/idVendor")"
product="$(cat "$device_path/idProduct")"

if [ "$vendor" != "8086" ] || [ "$product" != "0b3a" ]; then
  echo "refusing to reset non-D435i device: $vendor:$product at $device_path" >&2
  exit 1
fi

echo 0 >"$device_path/authorized"
sleep 3
echo 1 >"$device_path/authorized"
sleep 5

lsusb | grep -i "8086:0b3a"
