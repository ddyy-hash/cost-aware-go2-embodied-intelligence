#include <array>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>

#include "unitree/go2_idl/LowState_.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

namespace {
constexpr std::array<int, 12> PGTT_TO_UNITREE = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
std::atomic<int> samples{0};
unitree_go::msg::dds_::LowState_ latest;

void on_state(const void* raw) {
  latest = *static_cast<const unitree_go::msg::dds_::LowState_*>(raw);
  samples.fetch_add(1);
}
}

int main(int argc, char** argv) {
  const std::string network = argc > 1 ? argv[1] : "eth0";
  unitree::robot::ChannelFactory::Instance()->Init(0, network);
  unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_> subscriber("rt/lowstate");
  subscriber.InitChannel(on_state, 1);
  for (int i = 0; i < 250 && samples.load() < 20; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (samples.load() < 20) {
    std::cerr << "GO2_LOWSTATE_MOTOR_PROBE_BLOCKED reason=lowstate_not_advancing samples=" << samples.load() << std::endl;
    std::_Exit(2);
  }
  double max_abs_q = 0.0;
  double max_abs_dq = 0.0;
  uint64_t lost_samples = 0;
  for (const int index : PGTT_TO_UNITREE) {
    const auto& motor = latest.motor_state()[index];
    if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq()) || !std::isfinite(motor.temperature())) {
      std::cerr << "GO2_LOWSTATE_MOTOR_PROBE_BLOCKED reason=non_finite_motor_state index=" << index << std::endl;
      std::_Exit(3);
    }
    max_abs_q = std::max(max_abs_q, std::abs(static_cast<double>(motor.q())));
    max_abs_dq = std::max(max_abs_dq, std::abs(static_cast<double>(motor.dq())));
    if (motor.lost() != 0) {
      ++lost_samples;
      std::cerr << "GO2_LOWSTATE_MOTOR_LOST index=" << index
                << " lost=" << motor.lost() << " mode=" << static_cast<int>(motor.mode())
                << " temperature=" << static_cast<int>(motor.temperature()) << std::endl;
    }
  }
  std::cout << "GO2_LOWSTATE_MOTOR_PROBE_PASS samples=" << samples.load()
            << " tick=" << latest.tick() << " max_abs_q=" << max_abs_q
            << " max_abs_dq=" << max_abs_dq << " motors_with_lost_flag=" << lost_samples << std::endl;
  std::_Exit(lost_samples == 0 ? 0 : 4);
}
