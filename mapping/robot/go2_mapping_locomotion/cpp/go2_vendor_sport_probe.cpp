#include <cstdlib>
#include <iostream>
#include <string>

#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"

int main(int argc, char** argv) {
  const std::string network = argc > 1 ? argv[1] : "eth0";
  unitree::robot::ChannelFactory::Instance()->Init(0, network);
  unitree::robot::SportClient client;
  client.SetTimeout(3.0f);
  client.Init();

  int speed_level = -1;
  const int32_t code = client.GetSpeedLevel(speed_level);
  std::cout << "GO2_VENDOR_SPORT_READ_SPEED code=" << code
            << " level=" << speed_level << std::endl;
  // Avoid vendor DDS teardown on this legacy image; this probe owns no writer
  // and has made no motion request.
  std::_Exit(code == 0 ? 0 : 3);
}
