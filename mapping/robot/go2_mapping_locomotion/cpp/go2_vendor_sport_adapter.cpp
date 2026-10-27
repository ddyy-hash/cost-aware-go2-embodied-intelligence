#include <cstdlib>
#include <iostream>
#include <string>

#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"

namespace {
constexpr char kConfirm[] = "I_UNDERSTAND_GO2_HIGH_LEVEL_MOTION";

std::string arg(int argc, char** argv, const std::string& name, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) return argv[i + 1];
  }
  return fallback;
}

bool confirmed(int argc, char** argv) { return arg(argc, argv, "--confirm") == kConfirm; }
int finish(int code) { std::_Exit(code < 0 ? 2 : code); }
}

int main(int argc, char** argv) {
  const std::string command = arg(argc, argv, "--command", "check");
  const std::string network = arg(argc, argv, "--network", "eth0");
  unitree::robot::ChannelFactory::Instance()->Init(0, network);
  unitree::robot::SportClient client;
  client.SetTimeout(3.0f);
  client.Init();

  if (command == "check") {
    int level = -1;
    const int32_t code = client.GetSpeedLevel(level);
    std::cout << "GO2_VENDOR_SPORT_ADAPTER_CHECK code=" << code
              << " speed_level=" << level << std::endl;
    return finish(code == 0 ? 0 : 3);
  }
  if (!confirmed(argc, argv)) {
    std::cerr << "GO2_VENDOR_SPORT_ADAPTER_BLOCKED confirmation_required" << std::endl;
    return finish(4);
  }
  if (command == "stop") {
    return finish(client.StopMove() == 0 ? 0 : 5);
  }
  if (command == "gait") {
    const int gait = std::stoi(arg(argc, argv, "--gait", "0"));
    return finish(client.SwitchGait(gait) == 0 ? 0 : 6);
  }
  if (command == "move") {
    const float vx = std::stof(arg(argc, argv, "--vx", "0"));
    const float vy = std::stof(arg(argc, argv, "--vy", "0"));
    const float vyaw = std::stof(arg(argc, argv, "--vyaw", "0"));
    return finish(client.Move(vx, vy, vyaw) == 0 ? 0 : 7);
  }
  std::cerr << "GO2_VENDOR_SPORT_ADAPTER_BLOCKED unsupported_command" << std::endl;
  return finish(8);
}
