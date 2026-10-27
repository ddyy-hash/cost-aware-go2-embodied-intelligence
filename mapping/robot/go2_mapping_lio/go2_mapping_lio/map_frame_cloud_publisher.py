#!/usr/bin/env python3
"""Publish LIO registered clouds in ``map`` only with a matching TF transform."""
from __future__ import annotations

import argparse

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from map_frame_cloud import (
    PointCloudTransformError,
    PointFieldSpec,
    RigidTransform,
    transform_pointcloud_to_map,
)


def _rigid_transform(message: TransformStamped) -> RigidTransform:
    rotation = message.transform.rotation
    translation = message.transform.translation
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    return RigidTransform(
        rotation=(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        translation=(translation.x, translation.y, translation.z),
    )


class MapFrameCloudPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("go2_map_frame_cloud_publisher")
        self._map_frame = args.map_frame
        self._lookup_timeout = Duration(seconds=args.lookup_timeout_s)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = self.create_publisher(PointCloud2, args.output_topic, qos)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)
        self.create_subscription(PointCloud2, args.input_topic, self._on_cloud, qos)
        self.get_logger().info(
            "map-frame cloud publisher input=%s output=%s required_target_frame=%s"
            % (args.input_topic, args.output_topic, args.map_frame)
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        source_frame = message.header.frame_id.strip()
        if not source_frame:
            self.get_logger().error("refused registered cloud with empty frame_id")
            return
        if message.is_bigendian or message.height != 1 or message.row_step != message.width * message.point_step:
            self.get_logger().error("refused registered cloud with unsupported PointCloud2 byte layout")
            return
        if source_frame == self._map_frame:
            if self._map_frame != "odom":
                self.get_logger().error("refused registered cloud already labelled global map frame")
                return
            # LIO is explicitly configured in the local odom scope; no TF
            # lookup is needed when the source and destination are identical.
            self._publisher.publish(message)
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                source_frame,
                message.header.stamp,
                timeout=self._lookup_timeout,
            )
            data = transform_pointcloud_to_map(
                bytes(message.data),
                point_step=message.point_step,
                point_count=message.width * message.height,
                fields=tuple(
                    PointFieldSpec(field.name, field.offset, field.datatype, field.count) for field in message.fields
                ),
                map_from_odom=_rigid_transform(transform),
            )
        except (PointCloudTransformError, TransformException) as exc:
            self.get_logger().warn("refused local registered cloud without verified map transform: %s" % exc)
            return
        output = PointCloud2()
        output.header = message.header
        output.header.frame_id = self._map_frame
        output.height = message.height
        output.width = message.width
        output.fields = message.fields
        output.is_bigendian = message.is_bigendian
        output.point_step = message.point_step
        output.row_step = message.row_step
        output.data = data
        output.is_dense = message.is_dense
        self._publisher.publish(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-topic", default="/lio_sam_ros2/mapping/cloud_registered")
    parser.add_argument("--output-topic", default="/go2/mapping/map_cloud")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--lookup-timeout-s", type=float, default=0.15)
    args = parser.parse_args()
    if args.lookup_timeout_s <= 0.0:
        parser.error("--lookup-timeout-s must be positive")
    rclpy.init(args=None)
    node = MapFrameCloudPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
