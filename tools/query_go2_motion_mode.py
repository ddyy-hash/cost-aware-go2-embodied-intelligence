"""Read the current Unitree MotionSwitcher mode without changing it."""
import os

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


ChannelFactoryInitialize(0, "eth0")
client = MotionSwitcherClient()
client.SetTimeout(3.0)
client.Init()
print(client.CheckMode(), flush=True)
# CycloneDDS teardown in the vendor Python SDK can segfault after a successful
# read. This helper owns no publisher and exits without running that teardown.
os._exit(0)
