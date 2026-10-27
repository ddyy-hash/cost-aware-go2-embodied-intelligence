import argparse
import time

import rclpy
from std_msgs.msg import String


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/go2_person_follow/status")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("go2_follow_status_reader")
    messages = []

    def on_message(message: String) -> None:
        messages.append(message.data)
        print(message.data)

    node.create_subscription(String, args.topic, on_message, 10)
    deadline_s = time.monotonic() + max(0.1, args.timeout_s)
    while rclpy.ok() and len(messages) < args.count and time.monotonic() < deadline_s:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()
    if not messages:
        raise RuntimeError("no follow status messages received")


if __name__ == "__main__":
    main()
