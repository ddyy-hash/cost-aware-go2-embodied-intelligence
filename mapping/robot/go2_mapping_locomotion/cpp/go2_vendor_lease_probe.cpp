#include <cstdlib>
#include <iostream>
#include <string>

#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"

int main(int argc, char** argv) {
  const std::string network = argc > 1 ? argv[1] : "eth0";
  unitree::robot::ChannelFactory::Instance()->Init(0, network);
  unitree::robot::SportClient client(true);
  client.SetTimeout(3.0f);
  client.Init();
  client.WaitLeaseApplied();
  int speed_level = -1;
  const int32_t code = client.GetSpeedLevel(speed_level);
  std::cout << "GO2_VENDOR_LEASE_PROBE code=" << code
            << " lease_applied=true speed_level=" << speed_level << std::endl;
  std::_Exit(code == 0 ? 0 : 3);
}
