import argparse
import time

import rclpy
from std_msgs.msg import String


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["enter", "exit", "toggle"])
    parser.add_argument("--topic", default="/go2_person_follow/command")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--discovery-wait-s", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("go2_follow_command_sender")
    publisher = node.create_publisher(String, args.topic, 10)
    message = String()
    message.data = args.command

    deadline_s = time.monotonic() + max(0.0, args.discovery_wait_s)
    while time.monotonic() < deadline_s:
        rclpy.spin_once(node, timeout_sec=0.05)

    period_s = 1.0 / max(0.1, args.rate_hz)
    for _ in range(max(1, args.count)):
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(period_s)

    print(f"sent_command={args.command}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
