import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from go2_person_follow.detectors import create_detector
from go2_person_follow.image_utils import (
    Frame,
    depth_to_meters,
    horizontal_angle_from_box,
    image_msg_to_array,
    median_depth_in_box,
    stamp_to_seconds,
)


class FollowAlignmentCalibrator(Node):
    def __init__(self, args):
        super().__init__("go2_follow_alignment_calibrator")
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector = create_detector(
            args.detector_backend,
            args.onnx_model,
            args.engine,
            args.input_size,
            args.confidence,
            args.nms,
        )
        self.horizontal_fov_rad = math.radians(float(args.horizontal_fov_deg))
        self.latest_color: Optional[Frame] = None
        self.latest_depth: Optional[Frame] = None
        self.samples = []
        self.last_sample_s = 0.0
        self.start_s = time.monotonic()
        self.create_subscription(Image, args.color_topic, self._on_color, qos_profile_sensor_data)
        self.create_subscription(Image, args.depth_topic, self._on_depth, qos_profile_sensor_data)

    @property
    def done(self) -> bool:
        return len(self.samples) >= self.args.samples or time.monotonic() - self.start_s > self.args.timeout_s

    def _on_color(self, msg: Image) -> None:
        image = image_msg_to_array(msg)
        if image is None:
            self.get_logger().warning(f"unsupported color encoding: {msg.encoding}")
            return
        self.latest_color = Frame(image=image, stamp_s=stamp_to_seconds(msg), encoding=msg.encoding)

    def _on_depth(self, msg: Image) -> None:
        image = image_msg_to_array(msg)
        if image is None:
            self.get_logger().warning(f"unsupported depth encoding: {msg.encoding}")
            return
        self.latest_depth = Frame(image=image, stamp_s=stamp_to_seconds(msg), encoding=msg.encoding)

    def maybe_sample(self) -> None:
        now_s = time.monotonic()
        if now_s - self.last_sample_s < self.args.interval_s:
            return
        color = self.latest_color
        depth = self.latest_depth
        if color is None or depth is None:
            return
        if abs(color.stamp_s - depth.stamp_s) > self.args.depth_time_tolerance_s:
            return

        detections = self.detector.detect(color.image)
        depth_m = depth_to_meters(depth.image, depth.encoding)
        image_height, image_width = color.image.shape[:2]
        scale_x = depth_m.shape[1] / max(1, image_width)
        scale_y = depth_m.shape[0] / max(1, image_height)
        candidates = []
        for detection in detections:
            depth_box = np.array(
                [
                    detection.box[0] * scale_x,
                    detection.box[1] * scale_y,
                    detection.box[2] * scale_x,
                    detection.box[3] * scale_y,
                ],
                dtype=np.float32,
            )
            depth_value = median_depth_in_box(
                depth_m,
                depth_box,
                self.args.min_valid_depth_m,
                self.args.max_valid_depth_m,
            )
            if depth_value is None:
                continue
            angle_rad = horizontal_angle_from_box(detection.box, image_width, self.horizontal_fov_rad)
            if abs(angle_rad) > self.args.max_angle_rad:
                continue
            candidates.append((depth_value, abs(angle_rad), angle_rad, detection))

        if not candidates:
            self.last_sample_s = now_s
            return

        depth_value, _, angle_rad, detection = min(candidates, key=lambda item: (item[0], item[1]))
        sample = {
            "angle_rad": float(angle_rad),
            "depth_m": float(depth_value),
            "confidence": float(detection.confidence),
            "box": [float(value) for value in detection.box],
        }
        self.samples.append(sample)
        self.last_sample_s = now_s
        self._write_annotation(color.image, detection, len(self.samples) - 1)
        self.get_logger().info(
            "sample={} angle_rad={:.4f} depth_m={:.2f} confidence={:.3f}".format(
                len(self.samples),
                angle_rad,
                depth_value,
                detection.confidence,
            )
        )

    def _write_annotation(self, image, detection, index: int) -> None:
        annotated = image.copy()
        x1, y1, x2, y2 = [int(round(float(value))) for value in detection.box]
        x1 = max(0, min(annotated.shape[1] - 1, x1))
        x2 = max(0, min(annotated.shape[1] - 1, x2))
        y1 = max(0, min(annotated.shape[0] - 1, y1))
        y2 = max(0, min(annotated.shape[0] - 1, y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        center_x = int(round(0.5 * (x1 + x2)))
        cv2.line(annotated, (annotated.shape[1] // 2, 0), (annotated.shape[1] // 2, annotated.shape[0] - 1), (255, 0, 0), 1)
        cv2.line(annotated, (center_x, 0), (center_x, annotated.shape[0] - 1), (0, 255, 0), 1)
        cv2.imwrite(str(self.output_dir / f"alignment_{index:03d}.jpg"), annotated)

    def result(self) -> dict:
        if not self.samples:
            raise RuntimeError("no valid person samples were collected")
        angles = np.array([sample["angle_rad"] for sample in self.samples], dtype=np.float32)
        depths = np.array([sample["depth_m"] for sample in self.samples], dtype=np.float32)
        confidences = np.array([sample["confidence"] for sample in self.samples], dtype=np.float32)
        median_angle = float(np.median(angles))
        result = {
            "samples": len(self.samples),
            "median_angle_rad": median_angle,
            "mean_angle_rad": float(np.mean(angles)),
            "std_angle_rad": float(np.std(angles)),
            "recommended_camera_yaw_offset_rad": -median_angle,
            "median_depth_m": float(np.median(depths)),
            "mean_confidence": float(np.mean(confidences)),
            "output_dir": str(self.output_dir),
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image_rect_raw")
    parser.add_argument("--detector-backend", default="tensorrt_yolo")
    parser.add_argument("--onnx-model", default="")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--horizontal-fov-deg", type=float, default=69.0)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--interval-s", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--depth-time-tolerance-s", type=float, default=0.25)
    parser.add_argument("--min-valid-depth-m", type=float, default=0.25)
    parser.add_argument("--max-valid-depth-m", type=float, default=5.0)
    parser.add_argument("--max-angle-rad", type=float, default=0.65)
    parser.add_argument("--output-dir", default="/tmp/go2_follow_alignment")
    args = parser.parse_args()

    rclpy.init()
    node = FollowAlignmentCalibrator(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
            node.maybe_sample()
        result = node.result()
        result_path = Path(args.output_dir) / "alignment_result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="ascii")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
