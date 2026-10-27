#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

#if __has_include("unitree/go2_idl/LowState_.hpp")
#include "unitree/go2_idl/LowState_.hpp"
#else
#include "unitree/idl/go2/LowState_.hpp"
#endif
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"

#ifdef __UT_ROBOT_GO2_SPORT_CLIENT_HPP__
using Go2SportClient = unitree::robot::go2::SportClient;
#else
using Go2SportClient = unitree::robot::SportClient;
#endif

namespace {
constexpr char kConfirmation[] = "GO2_PGTT_FIELD_TEST_CONFIRMED";
constexpr double kLowStateTimeout = 0.10;
constexpr double kRouterTimeout = 0.25;
constexpr double kMaxHeightmapAgeMs = 250.0;
constexpr double kMaxSpeed = 0.12;
constexpr int kKeyA = 8;
std::atomic<bool> stop_requested{false};

struct Route {
  std::string state = "IDLE";
  bool executable = false;
  std::string policy = "03";
  double heightmap_age_ms = 1e9;
};

struct Request {
  std::string task;
  std::string controller;
  double max_speed_mps = 0.0;
  bool valid = false;
};

std::string arg(int argc, char** argv, const std::string& key, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) if (argv[i] == key) return argv[i + 1];
  return fallback;
}

bool finite(float value) { return std::isfinite(static_cast<double>(value)); }
void signal_handler(int) { stop_requested.store(true); }

Route read_route(const std::string& path) {
  struct stat info{};
  if (stat(path.c_str(), &info) != 0 || (std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() - info.st_mtime) > kRouterTimeout) return {};
  std::ifstream input(path);
  std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  Route route;
  std::smatch match;
  if (std::regex_search(text, match, std::regex(R"REGEX("state"\s*:\s*"([A-Z_]+)")REGEX"))) route.state = match[1].str();
  if (std::regex_search(text, match, std::regex(R"REGEX("executable"\s*:\s*(true|false))REGEX"))) route.executable = match[1].str() == "true";
  if (std::regex_search(text, match, std::regex(R"REGEX("policy_level"\s*:\s*"([0-9]+)")REGEX"))) route.policy = match[1].str();
  if (std::regex_search(text, match, std::regex(R"REGEX("heightmap_age_ms"\s*:\s*([-+0-9.eE]+))REGEX"))) route.heightmap_age_ms = std::stod(match[1].str());
  return route;
}

Request read_request(const std::string& path) {
  struct stat info{};
  if (stat(path.c_str(), &info) != 0 ||
      (std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() - info.st_mtime) > kRouterTimeout) {
    return {};
  }
  std::ifstream input(path);
  std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  Request request;
  std::smatch match;
  if (!std::regex_search(text, match, std::regex(R"REGEX("task"\s*:\s*"([A-Z_]+)")REGEX"))) return request;
  request.task = match[1].str();
  if (!std::regex_search(text, match, std::regex(R"REGEX("controller"\s*:\s*"([A-Za-z0-9._-]+)")REGEX"))) return request;
  request.controller = match[1].str();
  if (!std::regex_search(text, match, std::regex(R"REGEX("max_speed_mps"\s*:\s*([-+0-9.eE]+))REGEX"))) return request;
  request.max_speed_mps = std::stod(match[1].str());
  request.valid = std::isfinite(request.max_speed_mps) && request.max_speed_mps > 0.0 && request.max_speed_mps <= kMaxSpeed;
  return request;
}

bool request_matches_route(const Request& request, const Route& route) {
  if (!request.valid || request.controller != "pgtt-go2") return false;
  if (route.state == "STAIR_UP") return request.task == "STAIR_ASCEND";
  if (route.state == "STAIR_DOWN") return request.task == "STAIR_DESCEND";
  return route.state == "ROUGH_TERRAIN" && request.task == "ROUGH_TERRAIN";
}

bool route_fresh(const Route& route) {
  return std::isfinite(route.heightmap_age_ms) && route.heightmap_age_ms >= 0.0 && route.heightmap_age_ms <= kMaxHeightmapAgeMs;
}
bool qualified(const Route& route) { return route_fresh(route) && route.executable && (route.state == "STAIR_UP" || route.state == "STAIR_DOWN" || route.state == "ROUGH_TERRAIN"); }
int gait_for_route(const Route& route) {
  if (route.state == "STAIR_UP") return 3;
  if (route.state == "STAIR_DOWN") return 4;
  if (route.state == "ROUGH_TERRAIN") return 1;
  return 0;
}
#ifdef __UT_ROBOT_GO2_SPORT_CLIENT_HPP__
int vendor_check(Go2SportClient&) { return 0; }
int vendor_switch_gait(Go2SportClient&, int) { return 0; }
#else
int vendor_check(Go2SportClient& client) { int level = -1; return client.GetSpeedLevel(level) == 0 ? 0 : 1; }
int vendor_switch_gait(Go2SportClient& client, int gait) { return client.SwitchGait(gait); }
#endif
bool motor_health(const unitree_go::msg::dds_::LowState_& state) {
  constexpr int indices[] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
  for (const int index : indices) {
    const auto& motor = state.motor_state()[index];
    if (motor.lost() != 0 || !std::isfinite(static_cast<double>(motor.q())) ||
        !std::isfinite(static_cast<double>(motor.dq())) || motor.mode() == 0) return false;
  }
  return true;
}
float remote_float(const std::array<uint8_t, 40>& remote, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, remote.data() + offset, sizeof(value));
  return value;
}
}

int main(int argc, char** argv) {
  if (arg(argc, argv, "--confirmation") != kConfirmation) {
    std::cerr << "GO2_NATIVE_TERRAIN_EXECUTOR_BLOCKED confirmation_required" << std::endl;
    return 2;
  }
  const std::string network = arg(argc, argv, "--network", "eth0");
  const std::string router_path = arg(argc, argv, "--router-status", "/home/unitree/go2_mapping/runtime/terrain-router-status.json");
  const std::string request_path = arg(argc, argv, "--request-path", "/home/unitree/go2_mapping/runtime/terrain-control-request.json");
  const std::string status_path = arg(argc, argv, "--status-file", "/home/unitree/go2_mapping/runtime/native-terrain-executor-status.json");
  const double maximum_duration = std::stod(arg(argc, argv, "--maximum-duration", "180"));
  if (!(maximum_duration > 0.0 && maximum_duration <= 300.0)) return 2;
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  unitree::robot::ChannelFactory::Instance()->Init(0, network);
  Go2SportClient client(true);
  client.SetTimeout(0.25f);
  client.Init();
  client.WaitLeaseApplied();
  if (vendor_check(client) != 0) {
    std::cerr << "GO2_NATIVE_TERRAIN_EXECUTOR_BLOCKED reason=sport_capability_read_failed" << std::endl;
    return 3;
  }
  std::mutex state_mutex;
  unitree_go::msg::dds_::LowState_ latest;
  std::chrono::steady_clock::time_point latest_at{};
  unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_> subscriber("rt/lowstate");
  subscriber.InitChannel([&](const void* raw) { std::lock_guard<std::mutex> lock(state_mutex); latest = *static_cast<const unitree_go::msg::dds_::LowState_*>(raw); latest_at = std::chrono::steady_clock::now(); }, 1);

  // DDS discovery can take a few cycles after an explicit arm. Do not turn a
  // healthy dog into a fault lockout merely because the first LowState frame
  // has not arrived yet; once a frame exists, the normal 100 ms freshness gate
  // remains strict.
  for (int i = 0; i < 50; ++i) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      if (latest_at.time_since_epoch().count() != 0) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  const auto started = std::chrono::steady_clock::now();
  std::string active_state = "NATIVE_READY";
  int active_gait = -1;
  std::chrono::steady_clock::time_point gait_ready_at = std::chrono::steady_clock::time_point::min();
  bool moving = false;
  bool had_motion = false;
  std::string blocker;
  while (!stop_requested.load() && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < maximum_duration) {
    const auto loop_now = std::chrono::steady_clock::now();
    unitree_go::msg::dds_::LowState_ state;
    std::chrono::steady_clock::time_point received;
    { std::lock_guard<std::mutex> lock(state_mutex); state = latest; received = latest_at; }
    const double state_age = received.time_since_epoch().count() == 0 ? 999.0 : std::chrono::duration<double>(loop_now - received).count();
    const Route route = read_route(router_path);
    const Request request = read_request(request_path);
    if (state_age > kLowStateTimeout) {
      if (moving) client.StopMove();
      moving = false;
      active_state = "FAULT_LOCKOUT";
      blocker = "LOWSTATE_STALE";
      stop_requested.store(true);
    }
    else {
      const uint16_t keys = static_cast<uint16_t>(state.wireless_remote()[2]) | (static_cast<uint16_t>(state.wireless_remote()[3]) << 8);
      const bool emergency = (keys & (1u << kKeyA)) != 0;
      const float lx = remote_float(state.wireless_remote(), 4);
      const float rx = remote_float(state.wireless_remote(), 8);
      const float ly = remote_float(state.wireless_remote(), 20);
      if (emergency) {
        if (moving) client.StopMove();
        moving = false;
        active_state = "FAULT_LOCKOUT";
        blocker = "REMOTE_A_SOFTWARE_STOP";
        stop_requested.store(true);
      }
      else if (!motor_health(state)) {
        if (moving) client.StopMove();
        moving = false;
        active_state = "FAULT_LOCKOUT";
        blocker = "MOTOR_HEALTH";
        stop_requested.store(true);
      }
      else if (!route_fresh(route)) {
        if (moving) client.StopMove();
        moving = false;
        active_state = had_motion ? "RETURN_NATIVE" : "NATIVE_READY";
        blocker = "HEIGHTMAP_STALE";
        if (had_motion) stop_requested.store(true);
      }
      else if (!qualified(route) || !request_matches_route(request, route)) {
        if (moving) client.StopMove();
        moving = false;
        active_state = had_motion ? "RETURN_NATIVE" : "NATIVE_READY";
        blocker = "REQUEST_OR_ROUTE_NOT_READY";
        // A route downgrade after motion ends this short-lived lease session;
        // before the first command, remain armed and wait for the request.
        if (had_motion) stop_requested.store(true);
      }
      else if (!finite(lx) || !finite(ly) || !finite(rx)) {
        if (moving) client.StopMove();
        moving = false;
        active_state = "FAULT_LOCKOUT";
        blocker = "REMOTE_NONFINITE";
        stop_requested.store(true);
      }
      else {
        const int gait = gait_for_route(route);
        if (gait != active_gait) {
          if (moving) {
            client.StopMove();
            moving = false;
          }
          if (vendor_switch_gait(client, gait) == 0) {
            active_gait = gait;
            gait_ready_at = loop_now + std::chrono::milliseconds(100);
            active_state = "HANDOFF_PREP";
            blocker = "GAIT_SWITCH_SETTLING";
          } else {
            active_state = "FAULT_LOCKOUT";
            blocker = "GAIT_SWITCH_FAILED";
            stop_requested.store(true);
          }
        }
        const float vx = std::max(-static_cast<float>(kMaxSpeed), std::min(static_cast<float>(kMaxSpeed), ly * static_cast<float>(kMaxSpeed)));
        const float vy = std::max(-static_cast<float>(kMaxSpeed), std::min(static_cast<float>(kMaxSpeed), lx * static_cast<float>(kMaxSpeed)));
        if (!stop_requested.load() && std::chrono::steady_clock::now() < gait_ready_at) {
          active_state = "HANDOFF_PREP";
          blocker = "GAIT_SWITCH_SETTLING";
        } else if (!stop_requested.load() && std::chrono::steady_clock::now() >= gait_ready_at && client.Move(vx, vy, rx * 0.5f) == 0) {
          moving = true;
          had_motion = true;
          active_state = "ENHANCED_WALK";
          blocker.clear();
        } else if (!stop_requested.load()) {
          active_state = "FAULT_LOCKOUT";
          blocker = "MOVE_FAILED";
          stop_requested.store(true);
        }
      }
    }
    std::ofstream status(status_path, std::ios::trunc);
    status << "{\"state\":\"" << active_state << "\",\"blocker\":\"" << blocker
           << "\",\"transport_authorized\":" << (moving ? "true" : "false") << "}\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (moving) client.StopMove();
  // The installed legacy SDK has an ABI-unsafe SportClient destructor. Stop
  // before exiting and let its approximately one-second vendor lease TTL
  // expire; normal C++ teardown can otherwise crash after a safe handoff.
  std::_Exit(0);
}
