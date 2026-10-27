import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ColorFrameCollector(Node):
    def __init__(self, args):
        super().__init__("go2_realsense_calibration_collector")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.count = int(args.count)
        self.interval_s = float(args.interval_s)
        self.saved = 0
        self.last_save_s = 0.0
        self.start_s = time.monotonic()
        self.timeout_s = float(args.timeout_s)
        self.create_subscription(Image, args.topic, self._on_image, qos_profile_sensor_data)

    @property
    def done(self) -> bool:
        return self.saved >= self.count or time.monotonic() - self.start_s > self.timeout_s

    def _on_image(self, msg: Image) -> None:
        now_s = time.monotonic()
        if now_s - self.last_save_s < self.interval_s:
            return

        image = self._decode_image(msg)
        if image is None:
            self.get_logger().warning(f"unsupported encoding: {msg.encoding}")
            return

        path = self.output_dir / f"calib_{self.saved:05d}.jpg"
        cv2.imwrite(str(path), image)
        self.saved += 1
        self.last_save_s = now_s
        if self.saved % 10 == 0 or self.saved == self.count:
            self.get_logger().info(f"saved {self.saved}/{self.count} calibration frames")

    @staticmethod
    def _decode_image(msg: Image):
        if msg.encoding == "rgb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if msg.encoding == "bgr8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return arr.copy()
        if msg.encoding == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/color/image_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--interval-s", type=float, default=0.10)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    rclpy.init()
    node = ColorFrameCollector(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        print(f"saved_frames={node.saved}")
        if node.saved <= 0:
            raise RuntimeError("no calibration frames were saved")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
