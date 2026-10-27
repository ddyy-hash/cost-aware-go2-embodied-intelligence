# Go2 Mapping Capture

`go2_mapping_capture` is the robot-side durable storage boundary. ROS nodes
write each MCAP segment under `staging/`; only a file with an atomically written
SHA-256 manifest is published under `raw/` or `spool/`. Retention never deletes
unacknowledged data and quarantines checksum failures.

The capture wrapper is not installed or launched automatically. A live run
requires the mapping preflight and camera-ownership gate.
