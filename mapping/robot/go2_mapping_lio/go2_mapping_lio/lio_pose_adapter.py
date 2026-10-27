#!/usr/bin/env python3
"""Expose a pose-qualified LIO-SAM odometry stream under Go2 mapping topics.

The adapter never controls the robot. It only republishes a validated LIO pose
and explicitly announces that the deployed MID-360 extrinsic is provisional.
"""
from __future__ import annotations

import argparse
import json
import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class LioPoseAdapter(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("go2_lio_pose_adapter")
        self._calibration_id = args.calibration_id
        self._source_topic = args.source_topic
        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, args.pose_topic, 20)
        self._quality_pub = self.create_publisher(String, args.quality_topic, 10)
        self.create_subscription(Odometry, args.source_topic, self._on_odometry, 40)

    def _on_odometry(self, message: Odometry) -> None:
        pose = message.pose.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        quaternion_norm = math.sqrt(sum(component * component for component in values[3:]))
        if not all(math.isfinite(component) for component in values) or quaternion_norm < 0.99 or quaternion_norm > 1.01:
            self.get_logger().warn("discarded invalid LIO odometry pose")
            return

        result = PoseWithCovarianceStamped()
        result.header = message.header
        result.pose = message.pose
        self._pose_pub.publish(result)
        quality = {
            "state": "PREVIEW_UNCALIBRATED",
            "calibration_id": self._calibration_id,
            "source_topic": self._source_topic,
            "reason": "LiDAR-to-IMU extrinsic and time offset require measured acceptance",
        }
        notice = String()
        notice.data = json.dumps(quality, separators=(",", ":"), sort_keys=True)
        self._quality_pub.publish(notice)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-topic", default="/lio_sam_ros2/mapping/odometry")
    parser.add_argument("--pose-topic", default="/go2/lio/pose")
    parser.add_argument("--quality-topic", default="/go2/lio/quality")
    parser.add_argument("--calibration-id", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = LioPoseAdapter(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
