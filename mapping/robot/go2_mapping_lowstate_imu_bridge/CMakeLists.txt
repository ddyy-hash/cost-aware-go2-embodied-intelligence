cmake_minimum_required(VERSION 3.5)
project(go2_mapping_lowstate_imu_bridge)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(go2_lowstate_imu_bridge src/go2_lowstate_imu_bridge.cpp)
target_compile_features(go2_lowstate_imu_bridge PUBLIC cxx_std_14)
target_compile_options(go2_lowstate_imu_bridge BEFORE PRIVATE -I/usr/local/include)
target_include_directories(go2_lowstate_imu_bridge BEFORE PRIVATE
  /usr/local/include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
)
target_link_directories(go2_lowstate_imu_bridge PRIVATE /usr/local/lib)
target_link_libraries(go2_lowstate_imu_bridge
  unitree_go2_sdk
  unitree_go2_idl_cpp
  unitree_ros2_idl_cpp
  ddsc
  ddscxx
  rt
  pthread
)
ament_target_dependencies(go2_lowstate_imu_bridge rclcpp sensor_msgs)

add_executable(go2_native_pointcloud_bridge src/go2_native_pointcloud_bridge.cpp)
target_compile_features(go2_native_pointcloud_bridge PUBLIC cxx_std_14)
target_compile_options(go2_native_pointcloud_bridge BEFORE PRIVATE -I/usr/local/include)
target_include_directories(go2_native_pointcloud_bridge BEFORE PRIVATE
  /usr/local/include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
)
target_link_directories(go2_native_pointcloud_bridge PRIVATE /usr/local/lib)
target_link_libraries(go2_native_pointcloud_bridge
  unitree_go2_sdk
  unitree_go2_idl_cpp
  unitree_ros2_idl_cpp
  ddsc
  ddscxx
  rt
  pthread
)
ament_target_dependencies(go2_native_pointcloud_bridge rclcpp sensor_msgs)

install(TARGETS
  go2_lowstate_imu_bridge
  go2_native_pointcloud_bridge
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
