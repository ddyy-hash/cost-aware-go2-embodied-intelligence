#!/usr/bin/env python3
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class Camera:
    def __init__(self, device, width, height, fps):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

    def start(self):
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def _loop(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        while self.running:
            ok, frame = cap.read()
            if ok:
                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with self.lock:
                        self.frame = jpg.tobytes()
            else:
                time.sleep(0.1)

        cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame


def make_handler(camera):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Go2 Camera</title></head>
<body style="margin:0;background:#111;display:grid;place-items:center;height:100vh">
<img src="/stream.mjpg" style="max-width:100vw;max-height:100vh;object-fit:contain">
</body>
</html>""")
                return

            if self.path == "/snapshot.jpg":
                frame = camera.get_frame()
                if not frame:
                    self.send_error(503, "No frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
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
                frame = camera.get_frame()
                if frame:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except BrokenPipeError:
                        break
                time.sleep(1 / max(camera.fps, 1))

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video4")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    camera = Camera(args.device, args.width, args.height, args.fps)
    camera.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(camera))
    server.serve_forever()


if __name__ == "__main__":
    main()
