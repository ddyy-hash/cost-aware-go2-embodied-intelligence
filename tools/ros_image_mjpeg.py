#!/usr/bin/env python3
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageStreamNode(Node):
    def __init__(self, topic: str, quality: int):
        super().__init__("go2_ros_image_mjpeg")
        self.quality = int(quality)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_count = 0
        self.last_stamp_s = 0.0
        self.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f"streaming topic: {topic}")

    def _on_image(self, msg: Image) -> None:
        image = self._decode(msg)
        if image is None:
            self.get_logger().warning(f"unsupported encoding: {msg.encoding}")
            return
        ok, jpg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self.lock:
            self.frame = jpg.tobytes()
            self.frame_count += 1
            self.last_stamp_s = time.time()

    @staticmethod
    def _decode(msg: Image):
        if msg.encoding == "rgb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if msg.encoding == "bgr8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return arr.copy()
        if msg.encoding == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if msg.encoding in ("16UC1", "mono16"):
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            scaled = cv2.convertScaleAbs(arr, alpha=255.0 / max(1, int(arr.max())))
            return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        return None

    def get_frame(self):
        with self.lock:
            return self.frame, self.frame_count, self.last_stamp_s


def make_handler(node: ImageStreamNode, fps: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Go2 ROS Camera</title></head>
<body style="margin:0;background:#111;display:grid;place-items:center;height:100vh">
<img src="/stream.mjpg" style="max-width:100vw;max-height:100vh;object-fit:contain">
</body>
</html>"""
                )
                return

            if self.path == "/snapshot.jpg":
                frame, _, _ = node.get_frame()
                if not frame:
                    self.send_error(503, "No frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                return

            if self.path == "/status":
                frame, count, last_stamp_s = node.get_frame()
                age_s = time.time() - last_stamp_s if last_stamp_s else -1.0
                body = f"frames={count} has_frame={frame is not None} age_s={age_s:.3f}\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                frame, _, _ = node.get_frame()
                if frame:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(1.0 / max(fps, 1.0))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/color/image_raw")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    rclpy.init()
    node = ImageStreamNode(args.topic, args.quality)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(node, args.fps))
        print(f"http://{args.host}:{args.port}/", flush=True)
        server.serve_forever()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
