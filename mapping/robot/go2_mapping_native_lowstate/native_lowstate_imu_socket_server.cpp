#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include "unitree/go2_idl/LowState_.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

namespace
{
constexpr char kNativeLowStateTopic[] = "rt/lowstate";
constexpr char kDefaultSocketPath[] = "/home/unitree/go2_mapping/runtime/native_lowstate_imu.sock";
constexpr char kDefaultFrameId[] = "go2_imu_raw";
constexpr uint8_t kMagic[4] = {'G', '2', 'I', 'M'};
constexpr uint16_t kProtocolVersion = 1;
constexpr uint16_t kOrientationValid = 0x01;
constexpr uint16_t kMaxFrameIdLength = 512;

#pragma pack(push, 1)
struct FixedHeader {
  uint8_t magic[4];
  uint16_t version;
  uint16_t flags;
  int32_t stamp_sec;
  uint32_t stamp_nanosec;
  uint16_t frame_id_length;
  float orientation_w;
  float orientation_x;
  float orientation_y;
  float orientation_z;
  float angular_velocity_x;
  float angular_velocity_y;
  float angular_velocity_z;
  float linear_acceleration_x;
  float linear_acceleration_y;
  float linear_acceleration_z;
};
#pragma pack(pop)

static_assert(sizeof(FixedHeader) == 58, "G2IM fixed header layout changed");

std::string argument_value(int argc, char** argv, const std::string& key, const std::string& fallback)
{
  for (int index = 1; index + 1 < argc; ++index) {
    if (key == argv[index]) {
      return argv[index + 1];
    }
  }
  return fallback;
}

bool write_all(int fd, const void* buffer, size_t size)
{
  const auto* cursor = static_cast<const uint8_t*>(buffer);
  while (size > 0) {
    const ssize_t written = send(fd, cursor, size, MSG_NOSIGNAL);
    if (written > 0) {
      cursor += written;
      size -= static_cast<size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
  return true;
}

class NativeLowStateImuSocketServer
{
public:
  NativeLowStateImuSocketServer(std::string interface, std::string socket_path, std::string frame_id)
  : interface_(std::move(interface)), socket_path_(std::move(socket_path)), frame_id_(std::move(frame_id))
  {
    if (interface_.empty() || socket_path_.empty() || frame_id_.empty() ||
        socket_path_.size() >= sizeof(((sockaddr_un*)nullptr)->sun_path) ||
        frame_id_.size() > kMaxFrameIdLength) {
      throw std::invalid_argument("invalid native LowState IMU reader arguments");
    }
    unitree::robot::ChannelFactory::Instance()->Init(0, interface_);
    subscriber_.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(kNativeLowStateTopic));
    subscriber_->InitChannel(
      std::bind(&NativeLowStateImuSocketServer::on_low_state, this, std::placeholders::_1), 1);
    std::cerr << "native_lowstate_imu_socket_server source=" << kNativeLowStateTopic
              << " interface=" << interface_ << " socket=" << socket_path_
              << " frame=" << frame_id_ << std::endl;
  }

  ~NativeLowStateImuSocketServer()
  {
    if (subscriber_) {
      subscriber_->CloseChannel();
    }
    close_socket();
  }

private:
  bool connect_socket()
  {
    if (socket_fd_ >= 0) {
      return true;
    }
    const int candidate = socket(AF_UNIX, SOCK_STREAM, 0);
    if (candidate < 0) {
      return false;
    }
    timeval timeout{};
    timeout.tv_usec = 200000;
    setsockopt(candidate, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::snprintf(address.sun_path, sizeof(address.sun_path), "%s", socket_path_.c_str());
    if (connect(candidate, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
      close(candidate);
      return false;
    }
    socket_fd_ = candidate;
    return true;
  }

  void close_socket()
  {
    if (socket_fd_ >= 0) {
      close(socket_fd_);
      socket_fd_ = -1;
    }
  }

  bool send_imu(const unitree_go::msg::dds_::LowState_& low_state)
  {
    const auto& source_imu = low_state.imu_state();
    const auto& orientation = source_imu.quaternion();
    const auto& angular_velocity = source_imu.gyroscope();
    const auto& linear_acceleration = source_imu.accelerometer();

    FixedHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kProtocolVersion;
    header.frame_id_length = static_cast<uint16_t>(frame_id_.size());
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto stamp_sec = std::chrono::duration_cast<std::chrono::seconds>(now);
    const auto stamp_nanosec = std::chrono::duration_cast<std::chrono::nanoseconds>(now - stamp_sec);
    header.stamp_sec = static_cast<int32_t>(stamp_sec.count());
    header.stamp_nanosec = static_cast<uint32_t>(stamp_nanosec.count());
    header.orientation_w = orientation[0];
    header.orientation_x = orientation[1];
    header.orientation_y = orientation[2];
    header.orientation_z = orientation[3];
    header.angular_velocity_x = angular_velocity[0];
    header.angular_velocity_y = angular_velocity[1];
    header.angular_velocity_z = angular_velocity[2];
    header.linear_acceleration_x = linear_acceleration[0];
    header.linear_acceleration_y = linear_acceleration[1];
    header.linear_acceleration_z = linear_acceleration[2];

    const double orientation_norm = std::sqrt(
      static_cast<double>(orientation[0]) * orientation[0] +
      static_cast<double>(orientation[1]) * orientation[1] +
      static_cast<double>(orientation[2]) * orientation[2] +
      static_cast<double>(orientation[3]) * orientation[3]);
    const bool finite_values = std::isfinite(header.orientation_w) && std::isfinite(header.orientation_x) &&
      std::isfinite(header.orientation_y) && std::isfinite(header.orientation_z) &&
      std::isfinite(header.angular_velocity_x) && std::isfinite(header.angular_velocity_y) &&
      std::isfinite(header.angular_velocity_z) && std::isfinite(header.linear_acceleration_x) &&
      std::isfinite(header.linear_acceleration_y) && std::isfinite(header.linear_acceleration_z);
    if (!finite_values) {
      return false;
    }
    if (std::isfinite(orientation_norm) && orientation_norm > 0.5 && orientation_norm < 1.5) {
      header.flags |= kOrientationValid;
      header.orientation_w = static_cast<float>(orientation[0] / orientation_norm);
      header.orientation_x = static_cast<float>(orientation[1] / orientation_norm);
      header.orientation_y = static_cast<float>(orientation[2] / orientation_norm);
      header.orientation_z = static_cast<float>(orientation[3] / orientation_norm);
    }

    std::vector<uint8_t> body(sizeof(header) + frame_id_.size());
    std::memcpy(body.data(), &header, sizeof(header));
    std::memcpy(body.data() + sizeof(header), frame_id_.data(), frame_id_.size());
    const uint32_t body_length = static_cast<uint32_t>(body.size());
    if (!connect_socket() || !write_all(socket_fd_, &body_length, sizeof(body_length)) ||
        !write_all(socket_fd_, body.data(), body.size())) {
      close_socket();
      return false;
    }
    return true;
  }

  void on_low_state(const void* raw_message)
  {
    const auto* low_state = static_cast<const unitree_go::msg::dds_::LowState_*>(raw_message);
    const auto now = std::chrono::steady_clock::now();
    if (!send_imu(*low_state)) {
      if (now - last_report_ > std::chrono::seconds(2)) {
        std::cerr << "native_lowstate_imu_socket_server waiting for ROS socket receiver or dropped invalid IMU" << std::endl;
        last_report_ = now;
      }
      return;
    }
    if (now - last_report_ > std::chrono::seconds(1)) {
      std::cerr << "native_lowstate_imu_socket_server forwarded tick=" << low_state->tick() << std::endl;
      last_report_ = now;
    }
  }

  std::string interface_;
  std::string socket_path_;
  std::string frame_id_;
  int socket_fd_ = -1;
  std::chrono::steady_clock::time_point last_report_{};
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> subscriber_;
};
}  // namespace

int main(int argc, char** argv)
{
  try {
    NativeLowStateImuSocketServer server(
      argument_value(argc, argv, "--interface", "eth0"),
      argument_value(argc, argv, "--socket", kDefaultSocketPath),
      argument_value(argc, argv, "--frame-id", kDefaultFrameId));
    while (true) {
      sleep(1);
    }
  } catch (const std::exception& exc) {
    std::cerr << "native_lowstate_imu_socket_server error: " << exc.what() << std::endl;
    return 1;
  }
}
