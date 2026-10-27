#!/usr/bin/env python3
"""Publish a typed Livox CustomMsg stream for the vendor LIO front end only."""
from __future__ import annotations

import argparse
import copy

import rclpy
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from livox_custom_conversion import ConversionError, PointFieldSpec, convert_unitree_pointcloud


class LivoxCustomPublisher(Node):
    def __init__(self, input_topic: str, output_topic: str, lidar_id: int) -> None:
        super().__init__("go2_livox_custom_publisher")
        self._lidar_id = lidar_id
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = self.create_publisher(CustomMsg, output_topic, qos)
        self.create_subscription(PointCloud2, input_topic, self._on_cloud, qos)
        self.get_logger().info("Livox CustomMsg adapter input=%s output=%s lidar_id=%d" % (input_topic, output_topic, lidar_id))

    def _on_cloud(self, message: PointCloud2) -> None:
        try:
            converted = convert_unitree_pointcloud(
                bytes(message.data),
                point_step=message.point_step,
                point_count=message.width * message.height,
                fields=tuple(PointFieldSpec(field.name, field.offset, field.datatype, field.count) for field in message.fields),
            )
        except ConversionError as exc:
            self.get_logger().error("refused malformed Unitree PointCloud2 for Livox LIO: %s" % exc)
            return

        output = CustomMsg()
        output.header = copy.deepcopy(message.header)
        output.timebase = output.header.stamp.sec * 1_000_000_000 + output.header.stamp.nanosec
        output.point_num = len(converted)
        output.lidar_id = self._lidar_id
        output.rsvd = [0, 0, 0]
        for item in converted:
            point = CustomPoint()
            point.offset_time = item.offset_time
            point.x = item.x
            point.y = item.y
            point.z = item.z
            point.reflectivity = item.reflectivity
            point.tag = item.tag
            point.line = item.line
            output.points.append(point)
        self._publisher.publish(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", default="/go2/lio/lidar")
    parser.add_argument("--output-topic", default="/go2/lio/livox")
    parser.add_argument("--lidar-id", type=int, default=0)
    args = parser.parse_args()
    if args.lidar_id < 0 or args.lidar_id > 255:
        parser.error("--lidar-id must be in [0, 255]")
    rclpy.init()
    node = LivoxCustomPublisher(args.input_topic, args.output_topic, args.lidar_id)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
