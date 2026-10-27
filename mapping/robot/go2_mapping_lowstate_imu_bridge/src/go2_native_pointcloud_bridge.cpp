#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "unitree/ros2_idl/PointCloud2_.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

namespace
{
constexpr char kDefaultNativePointCloudTopic[] = "rt/utlidar/cloud";

class NativePointCloudBridge final : public rclcpp::Node
{
public:
  NativePointCloudBridge()
  : Node("go2_native_pointcloud_bridge")
  {
    interface_ = declare_parameter<std::string>("interface", "eth0");
    source_topic_ = declare_parameter<std::string>("source_topic", kDefaultNativePointCloudTopic);
    output_topic_ = declare_parameter<std::string>("output_topic", "/go2/lidar/points_raw");
    fallback_frame_id_ = declare_parameter<std::string>("fallback_frame_id", "go2_lidar_raw");

    if (interface_.empty() || source_topic_.empty() || output_topic_.empty() || fallback_frame_id_.empty()) {
      throw std::invalid_argument("interface, source_topic, output_topic, and fallback_frame_id must be non-empty");
    }

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::QoS(2).reliable());
    unitree::robot::ChannelFactory::Instance()->Init(0, interface_);
    subscriber_.reset(new unitree::robot::ChannelSubscriber<sensor_msgs::msg::dds_::PointCloud2_>(source_topic_));
    subscriber_->InitChannel(
      std::bind(&NativePointCloudBridge::on_cloud, this, std::placeholders::_1), 1);

    RCLCPP_INFO(
      get_logger(),
      "Read-only native point-cloud bridge started: source=%s interface=%s output=%s",
      source_topic_.c_str(),
      interface_.c_str(),
      output_topic_.c_str());
  }

  ~NativePointCloudBridge() override
  {
    if (subscriber_) {
      subscriber_->CloseChannel();
    }
  }

private:
  void on_cloud(const void* raw_message)
  {
    const auto* native = static_cast<const sensor_msgs::msg::dds_::PointCloud2_*>(raw_message);
    const uint64_t expected_bytes = static_cast<uint64_t>(native->row_step()) * native->height();
    const uint64_t data_bytes = native->data().size();
    if (native->height() == 0 || native->width() == 0 || native->point_step() == 0 ||
        expected_bytes != data_bytes) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Dropping malformed native point cloud: height=%u width=%u point_step=%u row_step=%u data=%llu",
        native->height(),
        native->width(),
        native->point_step(),
        native->row_step(),
        static_cast<unsigned long long>(data_bytes));
      return;
    }

    sensor_msgs::msg::PointCloud2 cloud;
    const auto& native_header = native->header();
    const auto& native_stamp = native_header.stamp();
    if (native_stamp.sec() != 0 || native_stamp.nanosec() != 0) {
      cloud.header.stamp.sec = native_stamp.sec();
      cloud.header.stamp.nanosec = native_stamp.nanosec();
    } else {
      cloud.header.stamp = now();
    }
    cloud.header.frame_id = native_header.frame_id().empty() ? fallback_frame_id_ : native_header.frame_id();
    cloud.height = native->height();
    cloud.width = native->width();
    cloud.is_bigendian = native->is_bigendian();
    cloud.point_step = native->point_step();
    cloud.row_step = native->row_step();
    cloud.is_dense = native->is_dense();
    cloud.data.assign(native->data().begin(), native->data().end());
    cloud.fields.reserve(native->fields().size());
    for (const auto& native_field : native->fields()) {
      sensor_msgs::msg::PointField field;
      field.name = native_field.name();
      field.offset = native_field.offset();
      field.datatype = native_field.datatype();
      field.count = native_field.count();
      cloud.fields.push_back(std::move(field));
    }

    publisher_->publish(cloud);
    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "Point cloud: frame=%s points=%u fields=%zu bytes=%llu",
      cloud.header.frame_id.c_str(),
      cloud.width * cloud.height,
      cloud.fields.size(),
      static_cast<unsigned long long>(data_bytes));
  }

  std::string interface_;
  std::string source_topic_;
  std::string output_topic_;
  std::string fallback_frame_id_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  unitree::robot::ChannelSubscriberPtr<sensor_msgs::msg::dds_::PointCloud2_> subscriber_;
};
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<NativePointCloudBridge>());
  } catch (const std::exception& exc) {
    RCLCPP_FATAL(rclcpp::get_logger("go2_native_pointcloud_bridge"), "%s", exc.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
