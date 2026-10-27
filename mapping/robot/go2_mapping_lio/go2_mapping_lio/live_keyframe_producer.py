#!/usr/bin/env python3
"""Publish only pose-qualified LiDAR keyframes to the PC live mapper."""
from __future__ import annotations

import argparse
import base64
from collections import deque
import hashlib
import json
import math
import os
import queue
import threading
import time
from typing import Deque, Optional
import uuid
import zlib

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2

from sequence_state import KeyframeSequenceState
try:
    from .bandwidth import DualQueueBudget, PreviewDemandStore, TokenBucket
except ImportError:  # Direct execution on the Go2 release directory.
    from bandwidth import DualQueueBudget, PreviewDemandStore, TokenBucket

try:
    from .detail_surface_source import (
        PayloadLimitExceeded,
        PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        adaptive_detail_hz,
        build_live_keyframe_message,
        build_surface_source_status_message,
        deduplicate_surface_cells,
        normalize_sensor_frame_transform_id,
    )
except ImportError:  # Direct execution on the Go2 release directory.
    from detail_surface_source import (
        PayloadLimitExceeded,
        PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        adaptive_detail_hz,
        build_live_keyframe_message,
        build_surface_source_status_message,
        deduplicate_surface_cells,
        normalize_sensor_frame_transform_id,
    )

try:
    from .texture_keyframe_source import TextureQualitySelector, build_texture_keyframe_message
except ImportError:  # Direct execution on the Go2 release directory.
    from texture_keyframe_source import TextureQualitySelector, build_texture_keyframe_message

try:
    import websocket
except ImportError:
    websocket = None

try:
    import cv2
except ImportError:
    cv2 = None


class Producer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("go2_live_keyframe_producer")
        self.args = args
        self.sensor_frame_transform_id = normalize_sensor_frame_transform_id(args.sensor_frame_transform_id)
        self.pose: Optional[PoseWithCovarianceStamped] = None
        self._pose_history: Deque[tuple[int, np.ndarray]] = deque(maxlen=120)
        self._pose_arrival_history: Deque[tuple[float, int, np.ndarray]] = deque(maxlen=120)
        self.imu: Optional[Imu] = None
        preview_bootstrap_mbps = float(os.environ.get("GO2_MAPPING_PREVIEW_BOOTSTRAP_MBPS", "1.0"))
        if not math.isfinite(preview_bootstrap_mbps) or preview_bootstrap_mbps <= 0:
            raise ValueError("GO2_MAPPING_PREVIEW_BOOTSTRAP_MBPS must be finite and positive")
        preview_budget = DualQueueBudget(
            args.bootstrap_total_mbps,
            preview_demand_mbps=preview_bootstrap_mbps,
        )
        self.preview_fallback_rate = preview_budget.preview_bytes_per_second
        self.preview_bucket = TokenBucket(preview_budget.preview_bytes_per_second, burst_seconds=0.25)
        self.preview_demand_store = PreviewDemandStore(args.preview_demand_state) if args.preview_demand_state else None
        self.sequence_state = KeyframeSequenceState(args.sequence_state) if args.sequence_state else None
        self.seq = self.sequence_state.load() if self.sequence_state else 0
        self._next_sequence = self.seq
        self.producer_instance_id = uuid.uuid4().hex
        source_configuration = {
            "surface_cell_m": args.surface_cell_m,
            "detail_payload_limit_bytes": args.detail_payload_limit_bytes,
            "lidar_topic": args.lidar_topic,
            "lidar_min_range_m": args.lidar_min_range_m,
            "lidar_max_range_m": args.lidar_max_range_m,
            "points_in_map_frame": args.points_in_map_frame,
            "sensor_frame_transform_id": self.sensor_frame_transform_id,
        }
        self.source_config_sha256 = hashlib.sha256(
            json.dumps(source_configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.status_sequence = 0
        self._latest_surface_metrics: Optional[dict] = None
        self._detail_queued_times: Deque[float] = deque(maxlen=64)
        self.last_attempt = 0.0
        self.last_camera_attempt = 0.0
        self.last_ack_rtt_ms: Optional[float] = None
        self._socket = None
        self._camera_socket = None
        self._texture_socket = None
        self._delivery_queue: queue.Queue[dict] = queue.Queue(maxsize=args.preview_queue_size)
        self._camera_delivery_queue: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._texture_delivery_queue: queue.Queue[dict] = queue.Queue(maxsize=args.texture_queue_size)
        self._delivery_stop = threading.Event()
        self._delivery_thread = threading.Thread(
            target=self._delivery_loop,
            name="go2-live-preview-delivery",
            daemon=True,
        )
        self._camera_delivery_thread = threading.Thread(
            target=self._camera_delivery_loop,
            name="go2-live-camera-delivery",
            daemon=True,
        )
        self._camera_sampling_thread = threading.Thread(
            target=self._camera_sampling_loop,
            name="go2-live-camera-sampling",
            daemon=True,
        )
        self._texture_delivery_thread = threading.Thread(
            target=self._texture_delivery_loop,
            name="go2-live-texture-delivery",
            daemon=True,
        )
        self._texture_sampling_thread = threading.Thread(
            target=self._texture_sampling_loop,
            name="go2-live-texture-sampling",
            daemon=True,
        )
        self.camera_color: Optional[Image] = None
        self.camera_depth: Optional[Image] = None
        self.camera_info: Optional[CameraInfo] = None
        self.camera_depth_info: Optional[CameraInfo] = None
        self.camera_color_received_at: Optional[float] = None
        self.camera_depth_received_at: Optional[float] = None
        self.camera_sequence = 0
        self.camera_instance_id = uuid.uuid4().hex
        self.texture_sequence = 0
        self.texture_producer_epoch = time.time_ns()
        texture_configuration = {
            "source_topic": args.camera_color_topic,
            "depth_topic": args.camera_depth_topic if args.camera_depth_aligned else None,
            "maximum_hz": args.texture_max_hz,
            "jpeg_quality": args.texture_jpeg_quality,
            "pose_association_clock": "subscriber_monotonic",
        }
        self.texture_source_config_sha256 = hashlib.sha256(
            json.dumps(texture_configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.texture_selector = TextureQualitySelector(
            minimum_sharpness=args.texture_minimum_sharpness,
            minimum_luma=args.texture_minimum_luma,
            maximum_luma=args.texture_maximum_luma,
            maximum_clipped_fraction=args.texture_maximum_clipped_fraction,
            minimum_translation_m=args.texture_minimum_translation_m,
            minimum_rotation_deg=args.texture_minimum_rotation_deg,
        )
        self.latest_cloud_timestamp_ns = 0
        self.camera_unavailable_logged = False
        self._input_counts = {
            "cloud": 0,
            "pose": 0,
            "imu": 0,
            "color": 0,
            "depth": 0,
            "delivered": 0,
            "texture_delivered": 0,
        }
        self._last_gate = "AWAITING_INPUT"
        self.create_subscription(PoseWithCovarianceStamped, args.pose_topic, self.on_pose, 10)
        self.create_subscription(Imu, args.imu_topic, self.on_imu, 100)
        self.create_subscription(PointCloud2, args.lidar_topic, self.on_cloud, 2)
        self.create_timer(5.0, self._report_input_health)
        self.create_timer(1.0, self._queue_surface_source_status)
        if args.camera_enabled:
            self.create_subscription(Image, args.camera_color_topic, self.on_camera_color, qos_profile_sensor_data)
            self.create_subscription(Image, args.camera_depth_topic, self.on_camera_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.camera_info_topic, self.on_camera_info, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.camera_depth_info_topic, self.on_camera_depth_info, qos_profile_sensor_data)
            if cv2 is None:
                self.get_logger().warn("camera assist is disabled because python3-opencv is unavailable")
        self._delivery_thread.start()
        if args.camera_enabled:
            self._camera_delivery_thread.start()
            self._camera_sampling_thread.start()
        if args.texture_enabled:
            self._texture_delivery_thread.start()
            self._texture_sampling_thread.start()

    def destroy_node(self) -> bool:
        self._delivery_stop.set()
        self._close_socket()
        self._close_camera_socket()
        self._close_texture_socket()
        self._delivery_thread.join(timeout=2.0)
        if self._camera_sampling_thread.is_alive():
            self._camera_sampling_thread.join(timeout=2.0)
        if self._camera_delivery_thread.is_alive():
            self._camera_delivery_thread.join(timeout=2.0)
        if self._texture_sampling_thread.is_alive():
            self._texture_sampling_thread.join(timeout=2.0)
        if self._texture_delivery_thread.is_alive():
            self._texture_delivery_thread.join(timeout=2.0)
        return super().destroy_node()

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _close_camera_socket(self) -> None:
        if self._camera_socket is not None:
            try:
                self._camera_socket.close()
            except Exception:
                pass
            self._camera_socket = None

    def _close_texture_socket(self) -> None:
        if self._texture_socket is not None:
            try:
                self._texture_socket.close()
            except Exception:
                pass
            self._texture_socket = None

    def _connect(self) -> None:
        if websocket is None:
            raise RuntimeError("websocket-client is required; install websocket-client on the robot")
        headers = []
        if self.args.token:
            headers.append("X-Go2-Live-Token: %s" % self.args.token)
        self._socket = websocket.create_connection(self.args.endpoint, timeout=10, header=headers)

    def _connect_camera(self) -> None:
        if websocket is None:
            raise RuntimeError("websocket-client is required; install websocket-client on the robot")
        headers = []
        if self.args.token:
            headers.append("X-Go2-Live-Token: %s" % self.args.token)
        self._camera_socket = websocket.create_connection(self.args.endpoint, timeout=10, header=headers)

    def _connect_texture(self) -> None:
        if websocket is None:
            raise RuntimeError("websocket-client is required; install websocket-client on the robot")
        headers = []
        if self.args.token:
            headers.append("X-Go2-Live-Token: %s" % self.args.token)
        self._texture_socket = websocket.create_connection(self.args.texture_endpoint, timeout=10, header=headers)

    def _resynchronize_sequence(self, acknowledgement: dict) -> bool:
        candidate = acknowledgement.get("next_sequence")
        if isinstance(candidate, bool):
            return False
        try:
            next_sequence = int(candidate)
        except (TypeError, ValueError):
            return False
        if next_sequence <= self.seq:
            return False
        self.seq = next_sequence - 1
        self._next_sequence = self.seq
        if self.sequence_state:
            self.sequence_state.store(self.seq)
        self.get_logger().warn("live keyframe sequence resynchronized to %s" % next_sequence)
        return True

    def _publish_preview_demand(self, payload: dict, *, cadence_hz: float) -> None:
        intended_bytes = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        requested_rate = intended_bytes * cadence_hz
        if self.preview_demand_store is not None:
            self.preview_demand_store.publish(
                required_mbps=requested_rate * 8.0 / 1_000_000.0,
                payload_bytes=intended_bytes,
                cadence_hz=cadence_hz,
            )

    @staticmethod
    def _correlation(payload: dict) -> tuple:
        message_type = payload.get("message_type")
        common = (
            payload.get("building_id"),
            payload.get("session_id"),
            payload.get("map_id"),
            payload.get("producer_instance_id"),
        )
        if message_type == "live_keyframe":
            return ("live_keyframe_ack", *common, payload.get("keyframe_sequence"))
        if message_type == "surface_source_status":
            return ("surface_source_status_ack", *common, payload.get("status_sequence"))
        if message_type == "camera_preview":
            return ("camera_preview_ack", *common, payload.get("camera", {}).get("sequence"))
        if message_type == "texture_keyframe":
            return (
                "texture_keyframe_ack",
                *common,
                payload.get("producer_epoch"),
                payload.get("texture_sequence"),
                payload.get("content_sha256"),
            )
        raise RuntimeError("unsupported typed live request")

    @staticmethod
    def _ack_correlation(acknowledgement: dict) -> tuple:
        message_type = acknowledgement.get("message_type")
        common = (
            acknowledgement.get("building_id"),
            acknowledgement.get("session_id"),
            acknowledgement.get("map_id"),
            acknowledgement.get("producer_instance_id"),
        )
        if message_type == "live_keyframe_ack":
            return (message_type, *common, acknowledgement.get("keyframe_sequence"))
        if message_type == "surface_source_status_ack":
            return (message_type, *common, acknowledgement.get("status_sequence"))
        if message_type == "camera_preview_ack":
            return (message_type, *common, acknowledgement.get("camera_sequence"))
        if message_type == "texture_keyframe_ack":
            return (
                message_type,
                *common,
                acknowledgement.get("producer_epoch"),
                acknowledgement.get("texture_sequence"),
                acknowledgement.get("content_sha256"),
            )
        raise RuntimeError("live acknowledgement type is invalid")

    def _send_payload(self, payload: dict) -> tuple[tuple, float]:
        message_type = payload.get("message_type")
        sequence = payload.get("_sequence")
        if message_type == "live_keyframe":
            if sequence is None:
                self._next_sequence += 1
                sequence = self._next_sequence
                payload["_sequence"] = sequence
            if not isinstance(sequence, int):
                raise RuntimeError("live keyframe sequence is invalid")
            wire_payload = {
                **{key: value for key, value in payload.items() if not key.startswith("_")},
                "keyframe_sequence": sequence,
                "previous_ack_rtt_ms": self.last_ack_rtt_ms,
            }
        elif message_type in {"surface_source_status", "camera_preview"}:
            wire_payload = dict(payload)
        else:
            raise RuntimeError("unsupported typed live request")
        # Preview owns its complete RGB-D + point payload. The high-fidelity
        # uploader observes the same demand and receives only the remainder;
        # never cap preview to a percentage of the measured link here.
        intended_bytes = len(json.dumps(wire_payload, separators=(",", ":")).encode("utf-8"))
        if message_type in {"live_keyframe", "camera_preview"}:
            requested_rate = intended_bytes * float(payload["_detail_cadence_hz"])
            self.preview_bucket.set_rate(max(self.preview_fallback_rate, requested_rate))
            wire_payload.pop("_detail_cadence_hz", None)
        body = json.dumps(wire_payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        body_bytes = len(body.encode("utf-8"))
        if message_type in {"live_keyframe", "camera_preview"} and body_bytes > self.args.detail_payload_limit_bytes:
            raise PayloadLimitExceeded(body_bytes, self.args.detail_payload_limit_bytes)
        delay = self.preview_bucket.delay_for(body_bytes)
        if delay:
            time.sleep(delay)
        if self._socket is None:
            self._connect()
        sent_at = time.monotonic()
        self._socket.send(body)
        return self._correlation(wire_payload), sent_at

    def _discard_queued_payloads(self) -> None:
        while True:
            try:
                self._delivery_queue.get_nowait()
            except queue.Empty:
                return

    def _enqueue_latest(self, payload: dict) -> None:
        while True:
            try:
                self._delivery_queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._delivery_queue.get_nowait()
                except queue.Empty:
                    continue

    def _enqueue_latest_camera(self, payload: dict) -> None:
        while True:
            try:
                self._camera_delivery_queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._camera_delivery_queue.get_nowait()
                except queue.Empty:
                    continue

    def _camera_sampling_loop(self) -> None:
        period_s = 1.0 / self.args.camera_max_hz
        next_sample = time.monotonic() + period_s
        while not self._delivery_stop.is_set():
            if self._delivery_stop.wait(max(0.0, next_sample - time.monotonic())):
                return
            self._queue_camera_preview()
            next_sample = max(next_sample + period_s, time.monotonic())

    def _camera_delivery_loop(self) -> None:
        """Send latest-only camera frames through a bounded independent ACK window."""
        camera_pending: Deque[tuple[tuple, float]] = deque()
        # DERP/Tailscale RTT can exceed 20 ms; use the configured ACK budget
        # so a valid camera frame is not discarded as a transient timeout.
        # DERP/Tailscale RTT plus Dashboard scheduling can exceed 100 ms.
        # Use the configured ACK budget for the socket itself; the explicit
        # deadline below still bounds a stuck pending frame.
        receive_timeout_s = self.args.camera_ack_timeout_s
        while not self._delivery_stop.is_set():
            try:
                while len(camera_pending) < self.args.camera_delivery_window:
                    try:
                        payload = self._camera_delivery_queue.get(
                            timeout=receive_timeout_s if not camera_pending else 0.0
                        )
                    except queue.Empty:
                        break
                    wire_payload = {key: value for key, value in payload.items() if not key.startswith("_")}
                    body = json.dumps(wire_payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
                    body_bytes = len(body.encode("utf-8"))
                    if body_bytes > self.args.detail_payload_limit_bytes:
                        raise PayloadLimitExceeded(body_bytes, self.args.detail_payload_limit_bytes)
                    if self._camera_socket is None:
                        self._connect_camera()
                        self._camera_socket.settimeout(receive_timeout_s)
                    expected = self._correlation(wire_payload)
                    sent_at = time.monotonic()
                    self._camera_socket.send(body)
                    camera_pending.append((expected, sent_at))
                if not camera_pending:
                    continue
                if time.monotonic() - camera_pending[0][1] > self.args.camera_ack_timeout_s:
                    raise TimeoutError("camera acknowledgement deadline exceeded")
                acknowledgement = json.loads(self._camera_socket.recv())
                acknowledgement_correlation = self._ack_correlation(acknowledgement)
                matched_index = next(
                    (index for index, item in enumerate(camera_pending) if item[0] == acknowledgement_correlation),
                    None,
                )
                if matched_index is None:
                    raise RuntimeError("camera acknowledgement correlation does not match a pending frame")
                camera_pending.rotate(-matched_index)
                camera_pending.popleft()
                camera_pending.rotate(matched_index)
                if acknowledgement.get("accepted") is not True:
                    raise RuntimeError("camera preview rejected: %s" % acknowledgement)
            except Exception as exc:
                if websocket is not None and isinstance(exc, websocket.WebSocketTimeoutException):
                    continue
                self._close_camera_socket()
                camera_pending.clear()
                if not self._delivery_stop.is_set():
                    self.get_logger().warn("camera delivery reset: %s" % exc)
                    time.sleep(0.2)

    def _texture_sampling_loop(self) -> None:
        period_s = 1.0 / self.args.texture_max_hz
        next_sample = time.monotonic() + period_s
        while not self._delivery_stop.is_set():
            if self._delivery_stop.wait(max(0.0, next_sample - time.monotonic())):
                return
            self._queue_texture_keyframe()
            next_sample = max(next_sample + period_s, time.monotonic())

    def _texture_delivery_loop(self) -> None:
        """Replay every numbered texture frame until its independent WAL ACK arrives."""
        pending: Deque[tuple[dict, tuple, float]] = deque()
        retry: Deque[dict] = deque()
        current_payload: Optional[dict] = None
        while not self._delivery_stop.is_set():
            try:
                while len(pending) < self.args.texture_delivery_window:
                    if retry:
                        current_payload = retry.popleft()
                    else:
                        try:
                            current_payload = self._texture_delivery_queue.get(
                                timeout=0.1 if not pending else 0.0
                            )
                        except queue.Empty:
                            break
                    body = json.dumps(current_payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
                    body_bytes = len(body.encode("utf-8"))
                    if body_bytes > self.args.texture_payload_limit_bytes:
                        raise PayloadLimitExceeded(body_bytes, self.args.texture_payload_limit_bytes)
                    if self._texture_socket is None:
                        self._connect_texture()
                    expected = self._correlation(current_payload)
                    sent_at = time.monotonic()
                    self._texture_socket.send(body)
                    pending.append((current_payload, expected, sent_at))
                    current_payload = None
                if not pending:
                    continue
                acknowledgement = json.loads(self._texture_socket.recv())
                actual = self._ack_correlation(acknowledgement)
                matched_index = next(
                    (index for index, item in enumerate(pending) if item[1] == actual),
                    None,
                )
                if matched_index is None:
                    raise RuntimeError("texture acknowledgement correlation does not match a pending frame")
                pending.rotate(-matched_index)
                payload, _, _ = pending.popleft()
                pending.rotate(matched_index)
                if acknowledgement.get("accepted") is not True:
                    pending.appendleft((payload, actual, time.monotonic()))
                    raise RuntimeError("texture keyframe rejected: %s" % acknowledgement)
                disposition = acknowledgement.get("disposition")
                if disposition != "DUPLICATE_INTEGRATED" and not acknowledgement.get("wal_record_sha256"):
                    raise RuntimeError("texture WAL acknowledgement is missing its record identity")
                self._input_counts["texture_delivered"] += 1
            except Exception as exc:
                self._close_texture_socket()
                replay: Deque[dict] = deque(item[0] for item in pending)
                if current_payload is not None:
                    replay.append(current_payload)
                replay.extend(retry)
                retry = replay
                pending.clear()
                current_payload = None
                if not self._delivery_stop.is_set():
                    self.get_logger().warn("texture delivery reset: %s" % exc)
                    time.sleep(0.2)

    def _delivery_loop(self) -> None:
        """Keep a bounded ACK window without letting a slow ACK block ROS callbacks."""
        pending: Deque[tuple[dict, tuple, float]] = deque()
        retry: Deque[dict] = deque()
        while not self._delivery_stop.is_set():
            current_payload: Optional[dict] = None
            try:
                while len(pending) < self.args.delivery_window:
                    if retry:
                        current_payload = retry.popleft()
                    else:
                        try:
                            current_payload = self._delivery_queue.get(timeout=0.1 if not pending else 0.0)
                        except queue.Empty:
                            break
                    correlation, sent_at = self._send_payload(current_payload)
                    pending.append((current_payload, correlation, sent_at))
                    current_payload = None
                if not pending:
                    continue
                acknowledgement = json.loads(self._socket.recv())
                acknowledgement_correlation = self._ack_correlation(acknowledgement)
                matched_index = next(
                    (index for index, item in enumerate(pending) if item[1] == acknowledgement_correlation),
                    None,
                )
                if matched_index is None:
                    raise RuntimeError("live acknowledgement correlation does not match a pending request")
                pending.rotate(-matched_index)
                payload, correlation, sent_at = pending.popleft()
                pending.rotate(matched_index)
                accepted = acknowledgement.get("accepted") is True
                status_duplicate = (
                    correlation[0] == "surface_source_status_ack"
                    and acknowledgement.get("error_code") == "STATUS_NOT_NEWER"
                )
                if not accepted and not status_duplicate:
                    if correlation[0] == "live_keyframe_ack" and self._resynchronize_sequence(acknowledgement):
                        # A window can have several older acknowledgements in
                        # the socket buffer. Reconnect before sending the new
                        # sequence so those replies cannot trigger a second,
                        # stale resynchronization.
                        self._close_socket()
                        pending.clear()
                        retry.clear()
                        self._discard_queued_payloads()
                        self._last_gate = "SEQUENCE_RESYNCHRONIZING"
                        continue
                    # A receiver backlog is a flow-control signal, not a
                    # license to lose this RGB-D/LiDAR tuple.  Put the exact
                    # frame back at the head so the exception path reconnects
                    # and replays the complete ordered window.
                    pending.appendleft((payload, correlation, sent_at))
                    raise RuntimeError("live request rejected: %s" % acknowledgement.get("error", acknowledgement))
                self.last_ack_rtt_ms = round((time.monotonic() - sent_at) * 1000.0, 1)
                if correlation[0] == "live_keyframe_ack":
                    sequence = int(correlation[-1])
                    self.seq = max(self.seq, sequence)
                    if self.sequence_state:
                        self.sequence_state.store(self.seq)
                    self._input_counts["delivered"] += 1
                else:
                    self._last_gate = "STATUS_ACKED"
                self._last_gate = "DELIVERED"
            except Exception as exc:
                self._close_socket()
                # Retain the unacknowledged prefix. The receiver treats an
                # already persisted chunk as a duplicate, so replay is safe.
                replay: Deque[dict] = deque(item[0] for item in pending)
                if current_payload is not None:
                    replay.append(current_payload)
                replay.extend(retry)
                retry = replay
                pending.clear()
                self._last_gate = "DELIVERY_ERROR:%s" % type(exc).__name__
                if not self._delivery_stop.is_set():
                    self.get_logger().warn("live delivery window reset: %s" % exc)
                    time.sleep(0.2)

    def on_pose(self, message: PoseWithCovarianceStamped) -> None:
        self._input_counts["pose"] += 1
        self.pose = message
        timestamp_ns = self._timestamp_ns(message)
        pose = self._pose_matrix(message)
        if timestamp_ns > 0 and pose is not None:
            self._pose_history.append((timestamp_ns, pose))
            self._pose_arrival_history.append((time.monotonic(), timestamp_ns, pose))

    def on_imu(self, message: Imu) -> None:
        self._input_counts["imu"] += 1
        self.imu = message

    def on_camera_color(self, message: Image) -> None:
        self._input_counts["color"] += 1
        self.camera_color = message
        self.camera_color_received_at = time.monotonic()

    def on_camera_depth(self, message: Image) -> None:
        self._input_counts["depth"] += 1
        self.camera_depth = message
        self.camera_depth_received_at = time.monotonic()

    def on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def on_camera_depth_info(self, message: CameraInfo) -> None:
        self.camera_depth_info = message

    def _report_input_health(self) -> None:
        self.get_logger().info(
            "PREVIEW_INPUT_HEALTH counts=%s gate=%s" % (json.dumps(self._input_counts, sort_keys=True), self._last_gate)
        )

    def _queue_surface_source_status(self) -> None:
        metrics = self._latest_surface_metrics
        if metrics is None:
            return
        self.status_sequence += 1
        accepted_hz = 0.0
        if len(self._detail_queued_times) >= 2:
            elapsed = self._detail_queued_times[-1] - self._detail_queued_times[0]
            if elapsed > 0:
                accepted_hz = (len(self._detail_queued_times) - 1) / elapsed
        status = build_surface_source_status_message(
            building_id=self.args.building_id,
            session_id=self.args.session_id,
            map_id=self.args.map_id,
            producer_instance_id=self.producer_instance_id,
            status_sequence=self.status_sequence,
            timestamp_ns=time.time_ns(),
            source_state=metrics["source_state"],
            source_clipped=False,
            source_points_before_dedup=metrics["source_points_before_dedup"],
            source_points_after_dedup=metrics["source_points_after_dedup"],
            compressed_point_bytes=metrics["compressed_point_bytes"],
            websocket_message_bytes=metrics["websocket_message_bytes"],
            accepted_detail_hz=accepted_hz,
            source_config_sha256=self.source_config_sha256,
            blocker=metrics["blocker"],
        )
        self._enqueue_latest(status.payload)

    def _queue_camera_preview(self) -> None:
        now = time.monotonic()
        camera = self._camera_payload(self.latest_cloud_timestamp_ns, now)
        if camera is None:
            return
        payload = {
            "message_type": "camera_preview",
            "protocol_version": 3,
            "building_id": self.args.building_id,
            "session_id": self.args.session_id,
            "map_id": self.args.map_id,
            "producer_instance_id": self.producer_instance_id,
            "calibration_id": self.args.calibration_id,
            "camera": camera,
            "_detail_cadence_hz": self.args.camera_max_hz,
        }
        self._publish_preview_demand(payload, cadence_hz=self.args.camera_max_hz)
        self._enqueue_latest_camera(payload)

    def _queue_texture_keyframe(self) -> None:
        if not self.args.texture_enabled or cv2 is None:
            return
        if self._texture_delivery_queue.full():
            self._last_gate = "TEXTURE_QUEUE_BACKPRESSURE"
            return
        color_message = self.camera_color
        camera_info = self.camera_info
        color_received_at = self.camera_color_received_at
        if color_message is None or camera_info is None or color_received_at is None:
            return
        now = time.monotonic()
        if now - color_received_at > self.args.texture_max_frame_age_ms / 1000.0:
            self._last_gate = "TEXTURE_COLOR_STALE"
            return
        color = self._image_color_bgr(color_message)
        if color is None:
            self._last_gate = "TEXTURE_COLOR_UNSUPPORTED"
            return
        source_height, source_width = color.shape[:2]
        if source_width != int(color_message.width) or source_height != int(color_message.height):
            self._last_gate = "TEXTURE_SOURCE_IDENTITY_MISMATCH"
            return
        pose_match = self._pose_for_camera_arrival(color_received_at)
        if pose_match is None:
            self._last_gate = "TEXTURE_LIO_POSE_UNAVAILABLE"
            return
        pose_arrived_at, pose_timestamp_ns, pose = pose_match
        pose_skew_ns = int(round((color_received_at - pose_arrived_at) * 1_000_000_000.0))
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            self._last_gate = "TEXTURE_INTRINSICS_INVALID"
            return
        encoded_ok, jpeg = cv2.imencode(
            ".jpg",
            color,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.args.texture_jpeg_quality],
        )
        if not encoded_ok:
            self._last_gate = "TEXTURE_JPEG_ENCODING_FAILED"
            return
        independent_view_id = "%s:%s" % (self.texture_producer_epoch, self.texture_sequence)
        selection = self.texture_selector.evaluate(
            color,
            pose,
            independent_view_id=independent_view_id,
        )
        if not selection.accepted:
            self._last_gate = "TEXTURE_%s" % selection.blocker
            return

        aligned_depth = None
        if self.args.camera_depth_aligned:
            depth_message = self.camera_depth
            depth_received_at = self.camera_depth_received_at
            if depth_message is not None and depth_received_at is not None:
                depth = self._image_depth_u16(depth_message)
                depth_fresh = now - depth_received_at <= self.args.camera_depth_max_age_ms / 1000.0
                depth_paired = abs(depth_received_at - color_received_at) <= (
                    self.args.camera_max_arrival_skew_ms / 1000.0
                )
                if depth is not None and depth.shape == (source_height, source_width) and depth_fresh and depth_paired:
                    aligned_depth = depth

        frame_timestamp_ns = self._timestamp_ns(color_message)
        if frame_timestamp_ns <= 0 or pose_timestamp_ns <= 0:
            self._last_gate = "TEXTURE_TIMESTAMP_INVALID"
            return
        try:
            encoded = build_texture_keyframe_message(
                building_id=self.args.building_id,
                session_id=self.args.session_id,
                map_id=self.args.map_id,
                producer_instance_id=self.producer_instance_id,
                producer_epoch=self.texture_producer_epoch,
                texture_sequence=self.texture_sequence,
                timestamp_ns=frame_timestamp_ns,
                calibration_id=self.args.calibration_id,
                source_config_sha256=self.texture_source_config_sha256,
                source_topic=self.args.camera_color_topic,
                source_frame_id=color_message.header.frame_id or camera_info.header.frame_id,
                source_encoding=color_message.encoding.lower(),
                source_width=source_width,
                source_height=source_height,
                color_jpeg=jpeg.tobytes(),
                intrinsics={"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                lio_pose_timestamp_ns=pose_timestamp_ns,
                pose_skew_ns=pose_skew_ns,
                pose_association_clock="subscriber_monotonic",
                pose_map_from_lidar=pose,
                quality=selection.quality_payload(),
                aligned_depth_u16=aligned_depth,
                depth_scale_m=self.args.camera_depth_scale_m,
                payload_limit_bytes=self.args.texture_payload_limit_bytes,
            )
        except (PayloadLimitExceeded, ValueError) as exc:
            self._last_gate = "TEXTURE_ENCODING_BLOCKED:%s" % type(exc).__name__
            self.get_logger().warn("texture keyframe blocked: %s" % exc)
            return
        try:
            self._texture_delivery_queue.put_nowait(encoded.payload)
        except queue.Full:
            self._last_gate = "TEXTURE_QUEUE_BACKPRESSURE"
            return
        self.texture_sequence += 1
        self._last_gate = "TEXTURE_QUEUED"

    def _pose_for_camera_arrival(self, color_received_at: float) -> Optional[tuple[float, int, np.ndarray]]:
        if not self._pose_arrival_history:
            return None
        match = min(self._pose_arrival_history, key=lambda item: abs(item[0] - color_received_at))
        maximum_skew_s = self.args.texture_pose_max_skew_ms / 1000.0
        if abs(match[0] - color_received_at) > maximum_skew_s:
            return None
        return match

    @staticmethod
    def _cloud_xyz(message: PointCloud2) -> Optional[np.ndarray]:
        fields = {field.name: field for field in message.fields}
        required = ("x", "y", "z")
        if message.is_bigendian or message.point_step < 12 or not all(name in fields for name in required):
            return None
        if any(fields[name].offset + 4 > message.point_step for name in required):
            return None
        point_count = message.width * message.height
        if point_count <= 0 or len(message.data) < point_count * message.point_step:
            return None
        raw = np.frombuffer(bytes(message.data[: point_count * message.point_step]), dtype=np.uint8)
        raw = raw.reshape((point_count, message.point_step))
        xyz = np.column_stack(
            [raw[:, fields[name].offset : fields[name].offset + 4].copy().view("<f4").ravel() for name in required]
        )
        return xyz[np.isfinite(xyz).all(axis=1)]

    @staticmethod
    def _preview_range_filter(points: np.ndarray, minimum_m: float, maximum_m: float) -> np.ndarray:
        ranges = np.linalg.norm(points, axis=1)
        return points[(ranges >= minimum_m) & (ranges <= maximum_m)]

    @staticmethod
    def _pose_matrix(message: PoseWithCovarianceStamped) -> Optional[np.ndarray]:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        w, x, y, z = orientation.w, orientation.x, orientation.y, orientation.z
        components = (position.x, position.y, position.z, w, x, y, z)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if not all(math.isfinite(value) for value in components) or norm < 0.99 or norm > 1.01:
            return None
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = (position.x, position.y, position.z)
        pose[:3, :3] = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        return pose

    def _pose_for_cloud_timestamp(self, timestamp_ns: int) -> Optional[np.ndarray]:
        if timestamp_ns <= 0 or not self._pose_history:
            return None
        pose_timestamp_ns, pose = min(self._pose_history, key=lambda item: abs(item[0] - timestamp_ns))
        maximum_skew_ns = int(self.args.pose_cloud_max_skew_ms * 1_000_000.0)
        if abs(pose_timestamp_ns - timestamp_ns) > maximum_skew_ns:
            return None
        return pose

    @staticmethod
    def _timestamp_ns(message) -> int:
        stamp = message.header.stamp
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    @staticmethod
    def _image_color_bgr(message: Image) -> Optional[np.ndarray]:
        if message.height <= 0 or message.width <= 0 or message.step < message.width * 3:
            return None
        encoding = message.encoding.lower()
        if encoding not in ("bgr8", "rgb8"):
            return None
        expected = message.height * message.step
        if len(message.data) < expected:
            return None
        rows = np.frombuffer(bytes(message.data[:expected]), dtype=np.uint8).reshape((message.height, message.step))
        image = rows[:, : message.width * 3].reshape((message.height, message.width, 3))
        return image if encoding == "bgr8" else image[:, :, ::-1]

    @staticmethod
    def _image_depth_u16(message: Image) -> Optional[np.ndarray]:
        if message.height <= 0 or message.width <= 0 or message.step < message.width * 2:
            return None
        if message.encoding.lower() not in ("16uc1", "mono16", "z16"):
            return None
        expected = message.height * message.step
        if len(message.data) < expected:
            return None
        dtype = ">u2" if message.is_bigendian else "<u2"
        rows = np.frombuffer(bytes(message.data[:expected]), dtype=np.uint8).reshape((message.height, message.step))
        return rows[:, : message.width * 2].copy().view(dtype).reshape((message.height, message.width))

    def _camera_payload(self, lidar_timestamp_ns: int, now: float) -> Optional[dict]:
        if not self.args.camera_enabled or cv2 is None:
            return None
        color_message = self.camera_color
        depth_message = self.camera_depth
        camera_info = self.camera_info
        depth_info = self.camera_depth_info
        color_received_at = self.camera_color_received_at
        depth_received_at = self.camera_depth_received_at
        if color_message is None or camera_info is None or color_received_at is None:
            return None
        # Some RealSense startup frames carry a zero ROS header stamp even
        # though the callback contains a valid image. Preview transport is
        # gated by callback freshness, so use the host arrival clock as the
        # protocol timestamp for those frames; raw headers remain untouched
        # in the recorded bag for offline calibration.
        color_timestamp_ns = self._timestamp_ns(color_message)
        if color_timestamp_ns <= 0:
            color_timestamp_ns = time.time_ns()
        max_arrival_skew_s = self.args.camera_max_arrival_skew_ms / 1000.0
        # MID-360 and RealSense headers are observed to use different clock
        # domains on this Go2. For PREVIEW assistance only, use local callback
        # arrival freshness; preserve the original headers for offline timing
        # calibration rather than pretending they are synchronized.
        if now - color_received_at > max_arrival_skew_s:
            return None
        color = self._image_color_bgr(color_message)
        if color is None:
            if not self.camera_unavailable_logged:
                self.get_logger().warn("camera assist requires bgr8 or rgb8 color frames")
                self.camera_unavailable_logged = True
            return None
        source_height, source_width = color.shape[:2]
        scale = min(1.0, self.args.camera_max_width / source_width, self.args.camera_max_height / source_height)
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))
        if (target_width, target_height) != (source_width, source_height):
            color = cv2.resize(color, (target_width, target_height), interpolation=cv2.INTER_AREA)
        encoded, jpeg = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.camera_jpeg_quality])
        if not encoded or jpeg.nbytes > 320 * 1024:
            return None
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            return None

        payload = {
            "sequence": self.camera_sequence + 1,
            "timestamp_ns": color_timestamp_ns,
            "producer_instance_id": self.camera_instance_id,
            "lidar_timestamp_ns": lidar_timestamp_ns,
            "arrival_skew_ms": round((now - color_received_at) * 1000.0, 1),
            "color": {
                "jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                "width": target_width,
                "height": target_height,
            },
            "intrinsics": {
                "fx": fx * target_width / source_width,
                "fy": fy * target_height / source_height,
                "cx": cx * target_width / source_width,
                "cy": cy * target_height / source_height,
            },
        }
        if depth_message is not None and depth_received_at is not None and self._timestamp_ns(depth_message) > 0:
            depth_timestamp_ns = self._timestamp_ns(depth_message)
            depth = self._image_depth_u16(depth_message)
            depth_is_fresh = now - depth_received_at <= self.args.camera_depth_max_age_ms / 1000.0
            depth_matches_color = abs(depth_received_at - color_received_at) <= max_arrival_skew_s
            # Raw depth in the unaligned Depth view is measured entirely in
            # its own optical frame. It only needs to be fresh; requiring it
            # to be paired with the color callback would discard valid depth
            # whenever live transport delivery blocks this single-threaded preview.
            if depth is not None and (depth_matches_color if self.args.camera_depth_aligned else depth_is_fresh):
                depth_source_height, depth_source_width = depth.shape
                depth_target_width = None
                depth_target_height = None
                depth_intrinsics = None
                if self.args.camera_depth_aligned:
                    if depth.shape == (source_height, source_width):
                        depth_target_width, depth_target_height = target_width, target_height
                        depth_intrinsics = payload["intrinsics"]
                elif depth_info is not None:
                    depth_scale = min(
                        1.0,
                        self.args.camera_depth_max_width / depth_source_width,
                        self.args.camera_depth_max_height / depth_source_height,
                    )
                    depth_target_width = max(1, int(round(depth_source_width * depth_scale)))
                    depth_target_height = max(1, int(round(depth_source_height * depth_scale)))
                    depth_fx, depth_fy = float(depth_info.k[0]), float(depth_info.k[4])
                    depth_cx, depth_cy = float(depth_info.k[2]), float(depth_info.k[5])
                    if not all(math.isfinite(value) for value in (depth_fx, depth_fy, depth_cx, depth_cy)) or depth_fx <= 0 or depth_fy <= 0:
                        return payload
                    depth_intrinsics = {
                        "fx": depth_fx * depth_target_width / depth_source_width,
                        "fy": depth_fy * depth_target_height / depth_source_height,
                        "cx": depth_cx * depth_target_width / depth_source_width,
                        "cy": depth_cy * depth_target_height / depth_source_height,
                    }
                if depth_target_width is not None and depth_target_height is not None and depth_intrinsics is not None:
                    if (depth_target_width, depth_target_height) != (depth_source_width, depth_source_height):
                        depth = cv2.resize(depth, (depth_target_width, depth_target_height), interpolation=cv2.INTER_NEAREST)
                    payload["depth"] = {
                        # Depth stays bit-exact.  Its spatial redundancy makes
                        # lossless zlib substantially cheaper than sending raw
                        # uint16 bytes through base64 over DERP.
                        "u16_le_zlib_b64": base64.b64encode(
                            zlib.compress(depth.astype("<u2", copy=False).tobytes(), level=1)
                        ).decode("ascii"),
                        "width": depth_target_width,
                        "height": depth_target_height,
                        "scale_m": self.args.camera_depth_scale_m,
                        "aligned_to_color": self.args.camera_depth_aligned,
                        "intrinsics": depth_intrinsics,
                    }
        self.last_camera_attempt = now
        self.camera_sequence += 1
        return payload

    def on_cloud(self, message: PointCloud2) -> None:
        self._input_counts["cloud"] += 1
        now = time.monotonic()
        detail_hz = adaptive_detail_hz(
            target_hz=self.args.detail_target_hz,
            minimum_hz=self.args.detail_minimum_hz,
            queued_frames=self._delivery_queue.qsize(),
            queue_capacity=self._delivery_queue.maxsize,
        )
        if now - self.last_attempt < 1.0 / detail_hz:
            self._last_gate = "RATE_LIMITED"
            return
        if self.imu is None or self.pose is None:
            self._last_gate = "AWAITING_POSE_OR_IMU"
            return
        xyz = self._cloud_xyz(message)
        timestamp_ns = self._timestamp_ns(message)
        self.latest_cloud_timestamp_ns = timestamp_ns
        pose = self._pose_matrix(self.pose) if self.args.points_in_map_frame else self._pose_for_cloud_timestamp(timestamp_ns)
        if xyz is None or pose is None:
            self._last_gate = "INVALID_CLOUD_OR_TIME_MATCHED_POSE"
            return
        # Registered clouds are already in the global map frame.  Preserve the
        # same sensor-relative display range by measuring from the current
        # LiDAR origin, not from map origin.
        range_points = xyz - pose[:3, 3] if self.args.points_in_map_frame else xyz
        ranges = np.linalg.norm(range_points, axis=1)
        xyz = xyz[(ranges >= self.args.lidar_min_range_m) & (ranges <= self.args.lidar_max_range_m)]
        if len(xyz) < 100:
            self._last_gate = "INSUFFICIENT_POINTS_IN_PREVIEW_RANGE"
            return
        detail = deduplicate_surface_cells(xyz, cell_m=self.args.surface_cell_m)
        xyz = detail.points_xyz
        if timestamp_ns <= 0:
            self._last_gate = "INVALID_CLOUD_TIMESTAMP"
            return
        try:
            encoded = build_live_keyframe_message(
                building_id=self.args.building_id,
                session_id=self.args.session_id,
                map_id=self.args.map_id,
                producer_instance_id=self.producer_instance_id,
                keyframe_sequence=0,
                timestamp_ns=timestamp_ns,
                calibration_id=self.args.calibration_id,
                source_config_sha256=self.source_config_sha256,
                points_xyz=xyz,
                pose_map_from_lidar=pose,
                imu_quaternion_wxyz=(
                    self.imu.orientation.w,
                    self.imu.orientation.x,
                    self.imu.orientation.y,
                    self.imu.orientation.z,
                ),
                point_cloud_source=self.args.lidar_topic,
                motion_compensated=self.args.lidar_topic.endswith("/cloud_deskewed") or self.args.points_in_map_frame,
                points_in_map_frame=self.args.points_in_map_frame,
                sensor_frame_transform_id=self.sensor_frame_transform_id,
                payload_limit_bytes=self.args.detail_payload_limit_bytes,
                camera=None,
                registration_residual_m=None,
                calibrated_lio=False,
                previous_ack_rtt_ms=self.last_ack_rtt_ms,
            )
        except PayloadLimitExceeded as error:
            self.last_attempt = now
            self._latest_surface_metrics = {
                "source_state": "DETAIL_PAYLOAD_LIMIT_EXCEEDED",
                "source_points_before_dedup": detail.source_points,
                "source_points_after_dedup": detail.occupied_cells,
                "compressed_point_bytes": 0,
                "websocket_message_bytes": error.message_bytes,
                "blocker": str(error),
            }
            self._last_gate = "DETAIL_PAYLOAD_LIMIT_EXCEEDED"
            self.get_logger().error(
                "%s; source_points=%s occupied_%.1fmm_cells=%s; frame blocked without clipping"
                % (error, detail.source_points, self.args.surface_cell_m * 1000.0, detail.occupied_cells)
            )
            return
        payload = encoded.payload
        self._latest_surface_metrics = {
            "source_state": "SOURCE_READY",
            "source_points_before_dedup": detail.source_points,
            "source_points_after_dedup": detail.occupied_cells,
            "compressed_point_bytes": len(base64.b64decode(payload["points_f32_zlib_b64"])),
            "websocket_message_bytes": encoded.message_bytes,
            "blocker": None,
        }
        self.last_attempt = now
        self._detail_queued_times.append(now)
        self._publish_preview_demand(payload, cadence_hz=detail_hz)
        payload["_detail_cadence_hz"] = detail_hz
        self._enqueue_latest(payload)
        self._last_gate = "QUEUED"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.getenv(
            "GO2_MAPPING_LIVE_ENDPOINT",
            "ws://192.0.2.10:8765/ws/live/keyframe",
        ),
    )
    parser.add_argument(
        "--texture-endpoint",
        default=os.getenv(
            "GO2_MAPPING_TEXTURE_ENDPOINT",
            "ws://192.0.2.10:8765/ws/live/texture",
        ),
    )
    parser.add_argument("--token", default="")
    parser.add_argument("--map-id", default="go2-live")
    parser.add_argument("--building-id", default=os.getenv("GO2_MAPPING_BUILDING_ID", ""))
    parser.add_argument("--session-id", default=os.getenv("GO2_MAPPING_CAPTURE_SESSION_ID", ""))
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--lidar-topic", default="/lio_sam_ros2/deskew/cloud_deskewed")
    parser.add_argument("--points-in-map-frame", action="store_true")
    parser.add_argument(
        "--sensor-frame-transform-id",
        default=os.getenv("GO2_MAPPING_SENSOR_FRAME_TRANSFORM_ID", PROVISIONAL_GO2_UTLIDAR_PITCH180_V2),
    )
    parser.add_argument("--imu-topic", default="/dog_imu_raw")
    parser.add_argument("--pose-topic", default="/go2/lio/pose")
    parser.add_argument("--surface-cell-m", type=float, default=0.005)
    parser.add_argument("--detail-payload-limit-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--detail-target-hz", type=float, default=5.0)
    parser.add_argument("--detail-minimum-hz", type=float, default=1.0)
    parser.add_argument("--lidar-min-range-m", type=float, default=0.45)
    parser.add_argument("--lidar-max-range-m", type=float, default=12.0)
    parser.add_argument("--pose-cloud-max-skew-ms", type=float, default=100.0)
    parser.add_argument("--sequence-state", default="")
    parser.add_argument("--preview-demand-state", default="")
    parser.add_argument("--bootstrap-total-mbps", type=float, default=4.0)
    parser.add_argument("--delivery-window", type=int, default=1)
    parser.add_argument("--preview-queue-size", type=int, default=2)
    parser.add_argument("--camera-enabled", action="store_true")
    parser.add_argument("--camera-color-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--camera-depth-info-topic", default="/camera/depth/camera_info")
    parser.add_argument("--camera-depth-aligned", action="store_true")
    parser.add_argument("--camera-max-hz", type=float, default=15.0)
    parser.add_argument("--camera-max-width", type=int, default=320)
    parser.add_argument("--camera-max-height", type=int, default=240)
    parser.add_argument("--camera-depth-max-width", type=int, default=160)
    parser.add_argument("--camera-depth-max-height", type=int, default=120)
    parser.add_argument("--camera-jpeg-quality", type=int, default=72)
    parser.add_argument("--camera-max-arrival-skew-ms", type=float, default=120.0)
    parser.add_argument("--camera-depth-max-age-ms", type=float, default=1000.0)
    parser.add_argument("--camera-depth-scale-m", type=float, default=0.001)
    parser.add_argument("--camera-delivery-window", type=int, default=1)
    parser.add_argument("--camera-ack-timeout-s", type=float, default=2.0)
    parser.add_argument("--texture-enabled", action="store_true")
    parser.add_argument("--texture-max-hz", type=float, default=2.0)
    parser.add_argument("--texture-queue-size", type=int, default=8)
    parser.add_argument("--texture-delivery-window", type=int, default=2)
    parser.add_argument("--texture-payload-limit-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--texture-jpeg-quality", type=int, default=90)
    parser.add_argument("--texture-max-frame-age-ms", type=float, default=500.0)
    parser.add_argument("--texture-pose-max-skew-ms", type=float, default=150.0)
    parser.add_argument("--texture-minimum-sharpness", type=float, default=20.0)
    parser.add_argument("--texture-minimum-luma", type=float, default=20.0)
    parser.add_argument("--texture-maximum-luma", type=float, default=235.0)
    parser.add_argument("--texture-maximum-clipped-fraction", type=float, default=0.1)
    parser.add_argument("--texture-minimum-translation-m", type=float, default=0.08)
    parser.add_argument("--texture-minimum-rotation-deg", type=float, default=5.0)
    args = parser.parse_args()
    if not args.building_id:
        parser.error("--building-id or GO2_MAPPING_BUILDING_ID is required")
    if not args.session_id:
        parser.error("--session-id or GO2_MAPPING_CAPTURE_SESSION_ID is required")
    if not 0.001 <= args.surface_cell_m <= 0.02:
        parser.error("--surface-cell-m must be between 0.001 and 0.02")
    try:
        args.sensor_frame_transform_id = normalize_sensor_frame_transform_id(args.sensor_frame_transform_id)
    except ValueError as exc:
        parser.error(str(exc))
    if not 64 * 1024 <= args.detail_payload_limit_bytes <= 64 * 1024 * 1024:
        parser.error("--detail-payload-limit-bytes must be between 64 KiB and 64 MiB")
    if not 0 < args.detail_minimum_hz <= args.detail_target_hz <= 15.0:
        parser.error("detail cadence must satisfy 0 < minimum <= target <= 15 Hz")
    if not 0.1 <= args.lidar_min_range_m < args.lidar_max_range_m <= 80.0:
        parser.error("LiDAR preview range limits must satisfy 0.1 <= min < max <= 80")
    if not 1.0 <= args.pose_cloud_max_skew_ms <= 250.0:
        parser.error("--pose-cloud-max-skew-ms must be between 1 and 250")
    if not 0 < args.camera_max_hz <= 15.0:
        parser.error("--camera-max-hz must be in (0, 15]")
    if not 32 <= args.camera_max_width <= 640 or not 32 <= args.camera_max_height <= 640:
        parser.error("camera preview dimensions must be between 32 and 640")
    if not 32 <= args.camera_depth_max_width <= 640 or not 32 <= args.camera_depth_max_height <= 640:
        parser.error("depth preview dimensions must be between 32 and 640")
    if not 20 <= args.camera_jpeg_quality <= 95:
        parser.error("--camera-jpeg-quality must be between 20 and 95")
    if not 20 <= args.camera_max_arrival_skew_ms <= 250:
        parser.error("--camera-max-arrival-skew-ms must be between 20 and 250")
    if not 100 <= args.camera_depth_max_age_ms <= 2000:
        parser.error("--camera-depth-max-age-ms must be between 100 and 2000")
    if not 0 < args.camera_depth_scale_m <= 0.1:
        parser.error("--camera-depth-scale-m must be in (0, 0.1]")
    if not 1 <= args.camera_delivery_window <= 2:
        parser.error("--camera-delivery-window must be between 1 and 2")
    if not 0.5 <= args.camera_ack_timeout_s <= 10.0:
        parser.error("--camera-ack-timeout-s must be between 0.5 and 10 seconds")
    if args.texture_enabled and not args.camera_enabled:
        parser.error("--texture-enabled requires --camera-enabled; it subscribes to the same existing camera topics")
    if not 0 < args.texture_max_hz <= 2.0:
        parser.error("--texture-max-hz must be in (0, 2]")
    if not 1 <= args.texture_delivery_window <= args.texture_queue_size <= 128:
        parser.error("texture queue must satisfy 1 <= delivery window <= queue size <= 128")
    if not 64 * 1024 <= args.texture_payload_limit_bytes <= 4 * 1024 * 1024:
        parser.error("--texture-payload-limit-bytes must be between 64 KiB and 4 MiB")
    if not 20 <= args.texture_jpeg_quality <= 95:
        parser.error("--texture-jpeg-quality must be between 20 and 95")
    if not 20 <= args.texture_max_frame_age_ms <= 2000:
        parser.error("--texture-max-frame-age-ms must be between 20 and 2000")
    if not 1 <= args.texture_pose_max_skew_ms <= 250:
        parser.error("--texture-pose-max-skew-ms must be between 1 and 250")
    try:
        TextureQualitySelector(
            minimum_sharpness=args.texture_minimum_sharpness,
            minimum_luma=args.texture_minimum_luma,
            maximum_luma=args.texture_maximum_luma,
            maximum_clipped_fraction=args.texture_maximum_clipped_fraction,
            minimum_translation_m=args.texture_minimum_translation_m,
            minimum_rotation_deg=args.texture_minimum_rotation_deg,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.delivery_window <= 32:
        parser.error("--delivery-window must be between 1 and 32")
    if not args.delivery_window <= args.preview_queue_size <= 128:
        parser.error("--preview-queue-size must be at least the delivery window and at most 128")
    try:
        DualQueueBudget(args.bootstrap_total_mbps)
    except ValueError as exc:
        parser.error(str(exc))
    rclpy.init()
    node = Producer(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
