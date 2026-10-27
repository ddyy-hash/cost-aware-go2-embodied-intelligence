#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#if __has_include("unitree/go2_idl/LowState_.hpp")
#include "unitree/go2_idl/LowState_.hpp"
#else
#include "unitree/idl/go2/LowState_.hpp"
#endif

#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

namespace
{
constexpr char kNativeLowStateTopic[] = "rt/lowstate";

class LowStateImuBridge final : public rclcpp::Node
{
public:
  LowStateImuBridge()
  : Node("go2_lowstate_imu_bridge")
  {
    interface_ = declare_parameter<std::string>("interface", "eth0");
    topic_ = declare_parameter<std::string>("topic", "/go2/lowstate/imu_raw");
    frame_id_ = declare_parameter<std::string>("frame_id", "go2_imu_raw");

    if (interface_.empty() || topic_.empty() || frame_id_.empty()) {
      throw std::invalid_argument("interface, topic, and frame_id must be non-empty");
    }

    publisher_ = create_publisher<sensor_msgs::msg::Imu>(topic_, rclcpp::QoS(200).reliable());
    unitree::robot::ChannelFactory::Instance()->Init(0, interface_);
    subscriber_.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(
      kNativeLowStateTopic));
    subscriber_->InitChannel(
      std::bind(&LowStateImuBridge::on_low_state, this, std::placeholders::_1), 1);

    RCLCPP_INFO(
      get_logger(),
      "Read-only native DDS bridge started: source=%s interface=%s output=%s frame=%s",
      kNativeLowStateTopic,
      interface_.c_str(),
      topic_.c_str(),
      frame_id_.c_str());
  }

  ~LowStateImuBridge() override
  {
    if (subscriber_) {
      subscriber_->CloseChannel();
    }
  }

private:
  void on_low_state(const void* raw_message)
  {
    const auto* low_state = static_cast<const unitree_go::msg::dds_::LowState_*>(raw_message);
    const auto& native_imu = low_state->imu_state();
    const auto& quaternion = native_imu.quaternion();

    sensor_msgs::msg::Imu imu;
    imu.header.stamp = now();
    imu.header.frame_id = frame_id_;

    // Unitree LowState encodes quaternion in scalar-first [w, x, y, z] order.
    // A malformed orientation is explicitly marked unavailable rather than
    // publishing a normalized-looking but false pose.
    const double norm = std::sqrt(
      static_cast<double>(quaternion[0]) * quaternion[0] +
      static_cast<double>(quaternion[1]) * quaternion[1] +
      static_cast<double>(quaternion[2]) * quaternion[2] +
      static_cast<double>(quaternion[3]) * quaternion[3]);
    if (std::isfinite(norm) && norm > 0.5 && norm < 1.5) {
      imu.orientation.w = quaternion[0] / norm;
      imu.orientation.x = quaternion[1] / norm;
      imu.orientation.y = quaternion[2] / norm;
      imu.orientation.z = quaternion[3] / norm;
    } else {
      imu.orientation_covariance[0] = -1.0;
    }

    const auto& gyroscope = native_imu.gyroscope();
    const auto& accelerometer = native_imu.accelerometer();
    imu.angular_velocity.x = gyroscope[0];
    imu.angular_velocity.y = gyroscope[1];
    imu.angular_velocity.z = gyroscope[2];
    imu.linear_acceleration.x = accelerometer[0];
    imu.linear_acceleration.y = accelerometer[1];
    imu.linear_acceleration.z = accelerometer[2];
    publisher_->publish(imu);
  }

  std::string interface_;
  std::string topic_;
  std::string frame_id_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> subscriber_;
};
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<LowStateImuBridge>();
    rclcpp::spin(node);
  } catch (const std::exception& exc) {
    RCLCPP_FATAL(rclcpp::get_logger("go2_lowstate_imu_bridge"), "%s", exc.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
