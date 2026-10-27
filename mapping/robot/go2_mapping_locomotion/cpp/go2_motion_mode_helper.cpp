#include <iostream>
#include <stdexcept>
#include <string>

#include "unitree/robot/b2/motion_switcher/motion_switcher_client.hpp"
#include "unitree/robot/channel/channel_factory.hpp"

namespace {
constexpr char kMutationConfirmation[] = "I_UNDERSTAND_MOTION_MODE_WILL_CHANGE";

std::string argument(int argc, char** argv, const std::string& name, const std::string& fallback = "") {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return argv[index + 1];
  }
  return fallback;
}
}

int main(int argc, char** argv) {
  try {
    const std::string command = argument(argc, argv, "--command", "check");
    const std::string interface = argument(argc, argv, "--network", "eth0");
    unitree::robot::ChannelFactory::Instance()->Init(0, interface);
    unitree::robot::b2::MotionSwitcherClient client;
    client.SetTimeout(3.0f);
    client.Init();
    if (command == "check") {
      std::string form;
      std::string name;
      const int32_t code = client.CheckMode(form, name);
      std::cout << "MOTION_MODE code=" << code << " form=" << form << " name=" << name << std::endl;
      return code == 0 ? 0 : 3;
    }
    if (argument(argc, argv, "--confirm") != kMutationConfirmation) {
      throw std::invalid_argument("motion mode mutation requires the exact confirmation token");
    }
    if (command == "release") {
      const int32_t code = client.ReleaseMode();
      std::cout << "MOTION_MODE_RELEASE code=" << code << std::endl;
      return code == 0 ? 0 : 4;
    }
    if (command == "select") {
      const std::string mode = argument(argc, argv, "--mode");
      if (mode != "normal") throw std::invalid_argument("only normal mode restoration is allowed");
      const int32_t code = client.SelectMode(mode);
      std::cout << "MOTION_MODE_SELECT code=" << code << " name=" << mode << std::endl;
      return code == 0 ? 0 : 5;
    }
    throw std::invalid_argument("unsupported command");
  } catch (const std::exception& error) {
    std::cerr << "go2_motion_mode_helper: " << error.what() << std::endl;
    return 2;
  }
}
