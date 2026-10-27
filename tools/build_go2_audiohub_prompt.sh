#!/usr/bin/env bash
set -eo pipefail

source_dir="${1:-/home/unitree/go2_follow_ws/src/go2_person_follow}"
output_dir="${2:-/home/unitree/go2_person_follow/bin}"

mkdir -p "$output_dir"

g++ -std=c++14 -O2 \
  "$source_dir/cpp/go2_audiohub_prompt.cpp" \
  -o "$output_dir/go2_audiohub_prompt" \
  -I/usr/local/include \
  -I/usr/local/include/ddscxx \
  -I/usr/local/include/iceoryx/v2.0.2 \
  -L/usr/local/lib \
  -lunitree_go2_sdk -lunitree_go2_idl_cpp -lunitree_ros2_idl_cpp \
  -lddsc -lddscxx -lrt -lpthread

ls -lh "$output_dir/go2_audiohub_prompt"
