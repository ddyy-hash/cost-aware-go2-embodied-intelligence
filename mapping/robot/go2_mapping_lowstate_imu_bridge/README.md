# Go2 LowState IMU Bridge

`go2_lowstate_imu_bridge` is a passive adapter from the Go2 native DDS
`rt/lowstate` topic to ROS 2 `sensor_msgs/msg/Imu`. It contains no Unitree
client, command publisher, or motion-control path.

Its default output is `/go2/lowstate/imu_raw`. Values stay in the Go2 native
IMU frame and use receipt time until the LiDAR-to-IMU
extrinsics, axis convention, and time residual have been versioned and
accepted. Do not remap it to `/dog_imu_raw` just to make an LIO launch pass.

The bridge is built into a release-specific workspace by
`tools/build_go2_lowstate_imu_bridge.sh` and launched by
`tools/start_go2_lowstate_imu_bridge.sh`.

The Unitree native point-cloud bridge is intentionally outside this CMake
package because the Unitree SDK and ROS Foxy use incompatible Cyclone DDS
ABIs. Its SDK-only reader and ROS-only publisher exchange a versioned local
socket protocol; launch it through `tools/start_go2_unitree_pointcloud_bridge.sh`.
