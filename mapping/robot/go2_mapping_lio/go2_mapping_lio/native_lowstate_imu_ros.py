#!/usr/bin/env python3
"""ROS-only endpoint for the native Unitree LowState IMU socket stream."""
from __future__ import annotations

import argparse
import os
import queue
import socket
import stat
import struct
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu

from native_lowstate_imu_protocol import FIXED_HEADER, NativeImuFrame, ProtocolError, decode_frame_body


LENGTH_PREFIX = struct.Struct("<I")
MAX_BODY_BYTES = FIXED_HEADER.size + 512


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("native LowState reader closed socket during frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class NativeLowStateImuRosPublisher(Node):
    def __init__(self, socket_path: Path, output_topic: str, fallback_frame_id: str) -> None:
        super().__init__("go2_native_lowstate_imu_publisher")
        self._socket_path = socket_path
        self._fallback_frame_id = fallback_frame_id
        self._publisher = self.create_publisher(
            Imu,
            output_topic,
            QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._frames = queue.Queue(maxsize=2048)
        self._dropped_frames = 0
        self._received_frames = 0
        self._published_frames = 0
        self._last_report = time.monotonic()
        self._stop = threading.Event()
        self._server_error: Optional[str] = None
        self._thread = threading.Thread(target=self._serve, name="go2-native-lowstate-socket", daemon=True)
        self._thread.start()
        self.create_timer(0.001, self._publish_available)
        self.get_logger().info("native LowState IMU socket=%s output=%s" % (socket_path, output_topic))

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

    def _enqueue(self, frame: NativeImuFrame) -> None:
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            self._frames.get_nowait()
            self._frames.put_nowait(frame)
            self._dropped_frames += 1
        self._received_frames += 1

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
                                if body_length > MAX_BODY_BYTES:
                                    raise ProtocolError("native IMU frame body exceeds protocol limit")
                                self._enqueue(decode_frame_body(recv_exact(connection, body_length)))
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

    def _publish_available(self) -> None:
        if self._server_error:
            self.get_logger().warn("native LowState IMU socket: %s" % self._server_error)
            self._server_error = None
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                break
            message = Imu()
            if frame.stamp_sec or frame.stamp_nanosec:
                message.header.stamp.sec = frame.stamp_sec
                message.header.stamp.nanosec = frame.stamp_nanosec
            else:
                message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = frame.frame_id or self._fallback_frame_id
            message.orientation.w, message.orientation.x, message.orientation.y, message.orientation.z = frame.orientation
            message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = frame.angular_velocity
            message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = frame.linear_acceleration
            if not frame.orientation_valid:
                message.orientation_covariance[0] = -1.0
            self._publisher.publish(message)
            self._published_frames += 1
        now = time.monotonic()
        if now - self._last_report >= 1.0 and self._received_frames:
            self.get_logger().info(
                "native LowState IMU received=%d published=%d dropped=%d"
                % (self._received_frames, self._published_frames, self._dropped_frames)
            )
            self._received_frames = 0
            self._published_frames = 0
            self._dropped_frames = 0
            self._last_report = now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output-topic", default="/dog_imu_raw")
    parser.add_argument("--fallback-frame-id", default="go2_imu_raw")
    args = parser.parse_args()
    rclpy.init()
    node = NativeLowStateImuRosPublisher(args.socket, args.output_topic, args.fallback_frame_id)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
