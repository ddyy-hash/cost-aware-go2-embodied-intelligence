#!/usr/bin/env bash
set -u
export PYTHONPATH=/mnt/c/Users/A/Desktop/DOG_ROBOTIC
mkdir -p output/video
for d in mapping-data/inbox/lidar-imu-rgbd/segment-*; do
  [ -d "$d" ] || continue
  id=${d##*/}
  out="output/video/${id}-color.mp4"
  [ -s "$out" ] && continue
  /home/a/micromamba/bin/micromamba run -n go2-mapping-pc python tools/convert_capture_color_to_mp4.py "$d" "$out" --fps 15 || rm -f "$out"
done
