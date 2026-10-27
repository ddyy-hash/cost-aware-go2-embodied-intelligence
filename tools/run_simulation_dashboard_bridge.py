"""Read-only synthetic simulation stream for isolated Dashboard validation."""
from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import time
import urllib.request


def points_blob(step: int, x: float, y: float, yaw: float, count: int = 512) -> str:
    values: list[float] = []
    c, s = math.cos(yaw), math.sin(yaw)
    for i in range(count):
        a = 2.0 * math.pi * i / count
        radius = 2.0 + 0.15 * math.sin(step * 0.12 + i * 0.07)
        local_x = radius * math.cos(a)
        local_y = radius * math.sin(a)
        values.extend((x + c * local_x - s * local_y, y + s * local_x + c * local_y, 0.15 * math.sin(a * 3.0)))
    return base64.b64encode(struct.pack("<" + "f" * len(values), *values)).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18772/api/live/keyframe")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--hz", type=float, default=8.0)
    args = parser.parse_args()
    started = time.time_ns()
    for sequence in range(args.steps):
        t = sequence / max(args.hz, 1.0)
        x, y = -0.8 + 0.25 * t, 0.5 * math.sin(t * 0.8)
        yaw = 0.12 * math.sin(t * 0.5)
        c, s = math.cos(yaw), math.sin(yaw)
        pose = [c, -s, 0.0, x, s, c, 0.0, y, 0.0, 0.0, 1.0, 0.35, 0.0, 0.0, 0.0, 1.0]
        payload = {
            "map_id": "simulation-map",
            "sequence": sequence,
            "timestamp_ns": started + int(t * 1e9),
            "calibration_id": "PROVISIONAL_SIMULATION",
            "points_f32_b64": points_blob(sequence, x, y, yaw),
            "pose_map_from_lidar": pose,
            "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "registration_residual_m": None,
            "calibrated_lio": False,
            "points_in_map_frame": True,
        }
        request = urllib.request.Request(
            args.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "X-Go2-Simulation": "true"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
        time.sleep(1.0 / max(args.hz, 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
