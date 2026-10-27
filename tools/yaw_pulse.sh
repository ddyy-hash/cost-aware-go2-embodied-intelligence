#!/usr/bin/env bash
set -euo pipefail

interface="${1:-eth0}"
vyaw="${2:-0.20}"
duration_s="${3:-0.70}"
bridge="${4:-/home/unitree/go2_person_follow/bin/go2_sport_bridge}"

{
  printf 'balance\n'
  sleep 0.2
  printf 'joystick off\n'
  sleep 0.2
  printf 'move 0 0 %s\n' "$vyaw"
  sleep "$duration_s"
  printf 'stop\n'
  sleep 0.2
  printf 'joystick on\n'
  sleep 0.2
  printf 'quit\n'
} | "$bridge" --interface "$interface"
