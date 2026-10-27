#!/usr/bin/env python3
"""ROS-only endpoint for the native Unitree PointCloud2 socket stream."""
from __future__ import annotations

import argparse
import os
import queue
import socket
import stat
import struct
import threading
import time
from array import array
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

from lio_clock_alignment import LioCloudClockAligner
from native_pointcloud_protocol import MAX_PAYLOAD_BYTES, NativePointCloudFrame, ProtocolError, decode_frame_body


LENGTH_PREFIX = struct.Struct("<I")


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("native reader closed socket during frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class NativePointCloudRosPublisher(Node):
    def __init__(
        self,
        socket_path: Path,
        output_topic: str,
        lio_output_topic: str,
        fallback_frame_id: str,
        lio_stamp_guard_ms: float,
    ) -> None:
        super().__init__("go2_native_pointcloud_publisher")
        self._socket_path = socket_path
        self._fallback_frame_id = fallback_frame_id
        self._raw_publisher = self.create_publisher(
            PointCloud2,
            output_topic,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._lio_publisher = None
        if lio_output_topic:
            self._lio_publisher = self.create_publisher(
                PointCloud2,
                lio_output_topic,
                QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
            )
        self._lio_clock_aligner = LioCloudClockAligner(guard_ms=lio_stamp_guard_ms)
        self._frames = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._server_error: Optional[str] = None
        self._thread = threading.Thread(target=self._serve, name="go2-native-pointcloud-socket", daemon=True)
        self._thread.start()
        self.create_timer(0.002, self._publish_available)
        self.get_logger().info(
            "native point-cloud socket=%s raw_output=%s lio_output=%s guard_ms=%.1f"
            % (socket_path, output_topic, lio_output_topic or "disabled", lio_stamp_guard_ms)
        )

    def destroy_node(self) -> bool:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return super().destroy_node()

    def _prepare_socket_path(self) -> None:
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._socket_path.exists():
            return
        if not stat.S_ISSOCK(self._socket_path.lstat().st_mode):
            raise RuntimeError("refusing to remove non-socket path: %s" % self._socket_path)
        self._socket_path.unlink()

    def _enqueue(self, frame: NativePointCloudFrame, arrival_stamp_ns: int) -> None:
        try:
            self._frames.put_nowait((frame, arrival_stamp_ns))
        except queue.Full:
            self._frames.get_nowait()
            self._frames.put_nowait((frame, arrival_stamp_ns))

    def _serve(self) -> None:
        try:
            self._prepare_socket_path()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(self._socket_path))
                os.chmod(self._socket_path, 0o600)
                server.listen(1)
                server.settimeout(0.5)
                while not self._stop.is_set():
                    try:
                        connection, _ = server.accept()
                    except socket.timeout:
                        continue
                    with connection:
                        connection.settimeout(2.0)
                        while not self._stop.is_set():
                            try:
                                body_length = LENGTH_PREFIX.unpack(recv_exact(connection, LENGTH_PREFIX.size))[0]
                                if body_length > MAX_PAYLOAD_BYTES + 16 * 1024:
                                    raise ProtocolError("native frame body exceeds protocol limit")
                                frame = decode_frame_body(recv_exact(connection, body_length))
                                self._enqueue(frame, time.time_ns())
                            except (ConnectionError, OSError, ProtocolError, UnicodeError) as exc:
                                if not self._stop.is_set():
                                    self._server_error = str(exc)
                                break
        except Exception as exc:
            self._server_error = str(exc)
        finally:
            try:
                if self._socket_path.exists() and stat.S_ISSOCK(self._socket_path.lstat().st_mode):
                    self._socket_path.unlink()
            except OSError:
                pass

    @staticmethod
    def _stamp_ns(frame: NativePointCloudFrame) -> int:
        return frame.stamp_sec * 1_000_000_000 + frame.stamp_nanosec

    def _message_from_frame(self, frame: NativePointCloudFrame, stamp_ns: int) -> PointCloud2:
        message = PointCloud2()
        message.header.stamp.sec = stamp_ns // 1_000_000_000
        message.header.stamp.nanosec = stamp_ns % 1_000_000_000
        message.header.frame_id = frame.frame_id or self._fallback_frame_id
        message.height = frame.height
        message.width = frame.width
        message.is_bigendian = frame.is_bigendian
        message.point_step = frame.point_step
        message.row_step = frame.row_step
        message.is_dense = frame.is_dense
        message.data = array("B", frame.data)
        for native_field in frame.fields:
            field = PointField()
            field.name = native_field.name
            field.offset = native_field.offset
            field.datatype = native_field.datatype
            field.count = native_field.count
            message.fields.append(field)
        return message

    def _publish_available(self) -> None:
        if self._server_error:
            self.get_logger().warn("native point-cloud socket: %s" % self._server_error)
            self._server_error = None
        latest = None
        while True:
            try:
                latest = self._frames.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        frame, arrival_stamp_ns = latest
        raw_stamp_ns = self._stamp_ns(frame) or arrival_stamp_ns
        self._raw_publisher.publish(self._message_from_frame(frame, raw_stamp_ns))
        if self._lio_publisher is not None:
            lio_stamp_ns = self._lio_clock_aligner.align(self._stamp_ns(frame), arrival_stamp_ns)
            self._lio_publisher.publish(self._message_from_frame(frame, lio_stamp_ns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output-topic", default="/livox/lidar")
    parser.add_argument("--lio-output-topic", default="/go2/lio/lidar")
    parser.add_argument("--fallback-frame-id", default="go2_lidar_raw")
    parser.add_argument("--lio-stamp-guard-ms", type=float, default=160.0)
    args = parser.parse_args()
    if args.lio_stamp_guard_ms < 0:
        parser.error("--lio-stamp-guard-ms must be non-negative")
    rclpy.init()
    node = NativePointCloudRosPublisher(
        args.socket,
        args.output_topic,
        args.lio_output_topic,
        args.fallback_frame_id,
        args.lio_stamp_guard_ms,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
