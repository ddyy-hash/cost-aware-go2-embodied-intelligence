"""ROS 2 publisher for the PGTT local elevation heightmap."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .heightmap import HeightmapError, build_pgtt_heightmap, build_stair_probe_profile, pointcloud_xyz, transform_lidar_points


def create_node(args: Any):
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2, PointField
    except ImportError as exc:
        raise RuntimeError("ROS 2 sensor_msgs/rclpy are required") from exc

    class PGTTHeightmapNode(Node):
        def __init__(self):
            super().__init__("go2_pgtt_heightmap")
            qos = QoSProfile(depth=2)
            qos.reliability = ReliabilityPolicy.RELIABLE
            qos.durability = DurabilityPolicy.VOLATILE
            self.publisher = self.create_publisher(PointCloud2, args.output_topic, 2)
            self.subscription = self.create_subscription(
                PointCloud2, args.input_topic, self.on_cloud, qos
            )
            self.received = 0
            self.published = 0

        def on_cloud(self, message: Any) -> None:
            try:
                points = pointcloud_xyz(message)
                transformed = transform_lidar_points(
                    points,
                    args.sensor_frame_transform_id,
                    source_frame_id=message.header.frame_id,
                )
                heightmap = build_pgtt_heightmap(
                    transformed,
                    spacing_m=args.spacing_m,
                    sensor_height_m=args.sensor_height_m,
                )
            except (HeightmapError, TypeError, ValueError) as exc:
                self.get_logger().warning("heightmap frame rejected: %s" % exc)
                return
            output = PointCloud2()
            output.header = message.header
            output.header.frame_id = args.frame_id
            output.height = 11
            output.width = 9
            output.is_bigendian = False
            output.is_dense = True
            output.point_step = 12
            output.row_step = output.point_step * output.width
            output.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            origin_x = -(11 - 1) * args.spacing_m / 2.0
            origin_y = -(9 - 1) * args.spacing_m / 2.0
            payload = bytearray()
            for row in range(11):
                for col in range(9):
                    payload.extend(struct.pack("<fff", origin_x + row * args.spacing_m, origin_y + col * args.spacing_m, float(heightmap[row, col])))
            output.data = payload
            self.publisher.publish(output)
            if args.output_file:
                temporary = Path(str(args.output_file) + ".partial")
                temporary.parent.mkdir(parents=True, exist_ok=True)
                with temporary.open("wb") as target:
                    np.asarray(heightmap, dtype="<f4").tofile(target)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, args.output_file)
            if args.stair_probe_file:
                probe = build_stair_probe_profile(transformed)
                temporary_probe = Path(str(args.stair_probe_file) + ".partial")
                temporary_probe.parent.mkdir(parents=True, exist_ok=True)
                with temporary_probe.open("wb") as target:
                    np.asarray(probe, dtype="<f4").tofile(target)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary_probe, args.stair_probe_file)
            self.received += 1
            self.published += 1
            self.get_logger().debug("heightmap transform=%s", args.sensor_frame_transform_id)

    return PGTTHeightmapNode, rclpy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-topic", default="/livox/lidar")
    parser.add_argument("--output-topic", default="/rt/elevation_heightmap")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--spacing-m", type=float, default=0.1)
    parser.add_argument("--sensor-height-m", type=float, default=0.6)
    parser.add_argument("--sensor-frame-transform-id", default="AUTO_GO2_LIDAR_FRAME_v1")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--stair-probe-file", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.spacing_m <= 0 or args.sensor_height_m < 0:
        parser.error("spacing must be positive and sensor height non-negative")
    Node, rclpy = create_node(args)
    rclpy.init(args=None)
    node = Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
