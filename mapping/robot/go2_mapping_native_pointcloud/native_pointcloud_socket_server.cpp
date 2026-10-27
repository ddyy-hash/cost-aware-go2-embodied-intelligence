#include <cerrno>
#include <chrono>
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
#include <sys/un.h>
#include <unistd.h>

#include "unitree/ros2_idl/PointCloud2_.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

namespace
{
// Unitree's on-robot SDK declares this PointCloud2 stream explicitly.  The
// parent `rt/utlidar` name is not a data topic and will never invoke a reader.
constexpr char kDefaultPointCloudTopic[] = "rt/utlidar/cloud";
constexpr char kDefaultSocketPath[] = "/home/unitree/go2_mapping/runtime/native_pointcloud.sock";
constexpr uint8_t kMagic[4] = {'G', '2', 'P', 'C'};
constexpr uint16_t kProtocolVersion = 1;
constexpr uint16_t kMaxFields = 32;
constexpr uint16_t kMaxNameLength = 128;
constexpr uint16_t kMaxFrameIdLength = 512;
constexpr uint64_t kMaxPayloadBytes = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t kMaxFrameBodyBytes = kMaxPayloadBytes + 16ULL * 1024ULL;

#pragma pack(push, 1)
struct FixedHeader {
  uint8_t magic[4];
  uint16_t version;
  uint16_t reserved;
  int32_t stamp_sec;
  uint32_t stamp_nanosec;
  uint32_t height;
  uint32_t width;
  uint32_t point_step;
  uint32_t row_step;
  uint16_t field_count;
  uint8_t flags;
  uint16_t frame_id_length;
};
#pragma pack(pop)

static_assert(sizeof(FixedHeader) == 37, "G2PC fixed header layout changed");

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

template <typename T>
void append_scalar(std::vector<uint8_t>& buffer, const T& value)
{
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

void append_bytes(std::vector<uint8_t>& buffer, const void* data, size_t size)
{
  if (size == 0) {
    return;
  }
  const auto* bytes = static_cast<const uint8_t*>(data);
  buffer.insert(buffer.end(), bytes, bytes + size);
}

class NativePointCloudSocketServer
{
public:
  NativePointCloudSocketServer(std::string interface, std::string source_topic, std::string socket_path)
  : interface_(std::move(interface)), source_topic_(std::move(source_topic)), socket_path_(std::move(socket_path))
  {
    if (interface_.empty() || source_topic_.empty() || socket_path_.empty() ||
        socket_path_.size() >= sizeof(((sockaddr_un*)nullptr)->sun_path)) {
      throw std::invalid_argument("invalid native point-cloud reader arguments");
    }
    unitree::robot::ChannelFactory::Instance()->Init(0, interface_);
    subscriber_.reset(new unitree::robot::ChannelSubscriber<sensor_msgs::msg::dds_::PointCloud2_>(source_topic_));
    subscriber_->InitChannel(
      std::bind(&NativePointCloudSocketServer::on_cloud, this, std::placeholders::_1), 1);
    std::cerr << "native_pointcloud_socket_server source=" << source_topic_
              << " interface=" << interface_
              << " socket=" << socket_path_ << std::endl;
  }

  ~NativePointCloudSocketServer()
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

  bool valid_cloud(const sensor_msgs::msg::dds_::PointCloud2_& cloud) const
  {
    const uint64_t expected = static_cast<uint64_t>(cloud.row_step()) * cloud.height();
    const uint64_t minimum_row_step = static_cast<uint64_t>(cloud.width()) * cloud.point_step();
    return cloud.height() > 0 && cloud.width() > 0 && cloud.point_step() > 0 &&
           cloud.fields().size() <= kMaxFields && cloud.header().frame_id().size() <= kMaxFrameIdLength &&
           cloud.row_step() >= minimum_row_step && expected == cloud.data().size() && expected <= kMaxPayloadBytes;
  }

  bool send_cloud(const sensor_msgs::msg::dds_::PointCloud2_& cloud)
  {
    std::vector<uint8_t> body;
    body.reserve(sizeof(FixedHeader) + cloud.header().frame_id().size() + cloud.data().size() + 1024);
    FixedHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kProtocolVersion;
    header.stamp_sec = cloud.header().stamp().sec();
    header.stamp_nanosec = cloud.header().stamp().nanosec();
    header.height = cloud.height();
    header.width = cloud.width();
    header.point_step = cloud.point_step();
    header.row_step = cloud.row_step();
    header.field_count = static_cast<uint16_t>(cloud.fields().size());
    header.flags = static_cast<uint8_t>((cloud.is_bigendian() ? 0x01 : 0x00) | (cloud.is_dense() ? 0x02 : 0x00));
    header.frame_id_length = static_cast<uint16_t>(cloud.header().frame_id().size());
    append_scalar(body, header);
    append_bytes(body, cloud.header().frame_id().data(), header.frame_id_length);
    for (const auto& field : cloud.fields()) {
      if (field.name().size() > kMaxNameLength) {
        return false;
      }
      const uint16_t name_length = static_cast<uint16_t>(field.name().size());
      append_scalar(body, name_length);
      append_bytes(body, field.name().data(), name_length);
      append_scalar(body, field.offset());
      append_scalar(body, field.datatype());
      append_scalar(body, field.count());
    }
    append_bytes(body, cloud.data().data(), cloud.data().size());
    if (body.size() > kMaxFrameBodyBytes || body.size() > UINT32_MAX) {
      return false;
    }
    const uint32_t body_length = static_cast<uint32_t>(body.size());
    if (!connect_socket() || !write_all(socket_fd_, &body_length, sizeof(body_length)) ||
        !write_all(socket_fd_, body.data(), body.size())) {
      close_socket();
      return false;
    }
    return true;
  }

  void on_cloud(const void* raw_message)
  {
    const auto* cloud = static_cast<const sensor_msgs::msg::dds_::PointCloud2_*>(raw_message);
    const auto now = std::chrono::steady_clock::now();
    if (!valid_cloud(*cloud)) {
      if (now - last_report_ > std::chrono::seconds(2)) {
        std::cerr << "native_pointcloud_socket_server dropped malformed frame height=" << cloud->height()
                  << " width=" << cloud->width() << " point_step=" << cloud->point_step()
                  << " row_step=" << cloud->row_step() << " bytes=" << cloud->data().size() << std::endl;
        last_report_ = now;
      }
      return;
    }
    if (!send_cloud(*cloud)) {
      if (now - last_report_ > std::chrono::seconds(2)) {
        std::cerr << "native_pointcloud_socket_server waiting for ROS socket receiver" << std::endl;
        last_report_ = now;
      }
      return;
    }
    if (now - last_report_ > std::chrono::seconds(1)) {
      std::cerr << "native_pointcloud_socket_server forwarded points=" << cloud->width() * cloud->height()
                << " fields=" << cloud->fields().size() << " bytes=" << cloud->data().size() << std::endl;
      last_report_ = now;
    }
  }

  std::string interface_;
  std::string source_topic_;
  std::string socket_path_;
  int socket_fd_ = -1;
  std::chrono::steady_clock::time_point last_report_{};
  unitree::robot::ChannelSubscriberPtr<sensor_msgs::msg::dds_::PointCloud2_> subscriber_;
};
}  // namespace

int main(int argc, char** argv)
{
  try {
    NativePointCloudSocketServer server(
      argument_value(argc, argv, "--interface", "eth0"),
      argument_value(argc, argv, "--source-topic", kDefaultPointCloudTopic),
      argument_value(argc, argv, "--socket", kDefaultSocketPath));
    while (true) {
      sleep(1);
    }
  } catch (const std::exception& exc) {
    std::cerr << "native_pointcloud_socket_server error: " << exc.what() << std::endl;
    return 1;
  }
}
