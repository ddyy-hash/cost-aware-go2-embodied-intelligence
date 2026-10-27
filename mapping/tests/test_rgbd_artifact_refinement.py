import json
import sqlite3
import struct

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256
from mapping.pc.go2_mapping_rgbd import ArtifactRgbdRefiner
from mapping.tests.test_mapping_quality import _calibration_v2


class _Writer:
    def __init__(self):
        self.data = bytearray(b"\x00\x01\x00\x00")
        self.origin = 4

    def align(self, size):
        self.data.extend(b"\0" * ((-(len(self.data) - self.origin)) % size))

    def pack(self, fmt, value, alignment):
        self.align(alignment)
        self.data.extend(struct.pack(fmt, value))

    def u8(self, value): self.pack("<B", value, 1)
    def u32(self, value): self.pack("<I", value, 4)
    def i32(self, value): self.pack("<i", value, 4)
    def f64(self, value): self.pack("<d", value, 8)

    def string(self, value):
        raw = value.encode("utf-8") + b"\0"
        self.u32(len(raw))
        self.data.extend(raw)

    def header(self, frame_id):
        self.i32(1)
        self.u32(0)
        self.string(frame_id)


def _image_payload(frame_id, encoding, image):
    value = np.asarray(image)
    height, width = value.shape[:2]
    writer = _Writer()
    writer.header(frame_id)
    writer.u32(height)
    writer.u32(width)
    writer.string(encoding)
    writer.u8(0)
    writer.u32(value[0].nbytes)
    raw = value.tobytes()
    writer.u32(len(raw))
    writer.data.extend(raw)
    return bytes(writer.data)


def _camera_contract(frame_id):
    contract = {
        "width": 20,
        "height": 20,
        "distortion_model": "plumb_bob",
        "k": [100.0, 0.0, 10.0, 0.0, 100.0, 10.0, 0.0, 0.0, 1.0],
        "d": [0.0] * 5,
        "r": np.eye(3).reshape(-1).tolist(),
        "p": [100.0, 0.0, 10.0, 0.0, 0.0, 100.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "rectified": False,
    }
    return frame_id, contract


def _camera_info_payload(frame_id, contract):
    writer = _Writer()
    writer.header(frame_id)
    writer.u32(contract["height"])
    writer.u32(contract["width"])
    writer.string(contract["distortion_model"])
    writer.u32(len(contract["d"]))
    for value in contract["d"]: writer.f64(value)
    for value in contract["k"]: writer.f64(value)
    for value in contract["r"]: writer.f64(value)
    for value in contract["p"]: writer.f64(value)
    writer.u32(0)
    writer.u32(0)
    for value in (0, 0, 0, 0): writer.u32(value)
    writer.u8(0)
    return bytes(writer.data)


def _manifest(path):
    payload = _calibration_v2()
    color_frame, color = _camera_contract("camera_color_optical_frame")
    depth_frame, depth = _camera_contract("camera_depth_optical_frame")
    payload["frame_chain"]["color_optical_frame_id"] = color_frame
    payload["frame_chain"]["depth_optical_frame_id"] = depth_frame
    payload["frame_chain"]["color_camera_info"] = color
    payload["frame_chain"]["depth_camera_info"] = depth
    payload["frame_chain"]["color_camera_info_sha256"] = canonical_sha256(color)
    payload["frame_chain"]["depth_camera_info_sha256"] = canonical_sha256(depth)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, color, depth


def _bag(path, color_contract, depth_contract):
    color = np.zeros((20, 20, 3), dtype=np.uint8)
    color[..., 0] = np.arange(20, dtype=np.uint8)[None, :] * 10
    color[..., 1] = 80
    color[..., 2] = 160
    depth = np.full((20, 20), 1000, dtype=np.uint16)
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
    )
    topics = (
        (1, "/camera/color/image_raw", "sensor_msgs/msg/Image"),
        (2, "/camera/depth/image_rect_raw", "sensor_msgs/msg/Image"),
        (3, "/camera/color/camera_info", "sensor_msgs/msg/CameraInfo"),
        (4, "/camera/depth/camera_info", "sensor_msgs/msg/CameraInfo"),
    )
    database.executemany("INSERT INTO topics VALUES (?, ?, ?)", topics)
    messages = (
        (1, 3, 1_000_000_000, _camera_info_payload("camera_color_optical_frame", color_contract)),
        (2, 4, 1_000_000_000, _camera_info_payload("camera_depth_optical_frame", depth_contract)),
        (3, 1, 1_000_000_000, _image_payload("camera_color_optical_frame", "rgb8", color)),
        (4, 2, 1_000_000_000, _image_payload("camera_depth_optical_frame", "16UC1", depth)),
    )
    database.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", messages)
    database.commit()
    database.close()


def _lidar_frame():
    points = np.asarray([
        ((u - 10) / 100.0, (v - 10) / 100.0, 1.0)
        for v in range(20)
        for u in range(20)
    ], dtype=np.float64)
    return {"timestamp_ns": 1_000_000_000, "points_xyz": points, "pose_map_from_lidar": np.eye(4)}


def test_rgbd_artifact_refinement_requires_measured_manifest(tmp_path):
    _, color, depth = _manifest(tmp_path / "unused.json")
    bag = tmp_path / "rgbd.db3"
    _bag(bag, color, depth)
    refiner = ArtifactRgbdRefiner(None, depth_scale_m=0.001)

    refiner.process_sqlite_bag(bag, artifact_sha256="a" * 64, lidar_frames=(_lidar_frame(),), measured_generation="pg-1")

    state = refiner.snapshot()
    assert state["state"] == "RGBD_CALIBRATION_BLOCKED"
    assert state["depth_correspondence_count"] == 0
    assert state["accepted_color_observations"] == 0
    assert "MEASURED_CALIBRATION_MANIFEST_MISSING" in state["blockers"]
    evidence = refiner.appearance_evidence(("0:0:0",))
    assert evidence.ready is False
    assert evidence.vertex_colors == ()
    assert "MEASURED_CALIBRATION_REQUIRED" in evidence.blockers


def test_rgbd_artifact_refinement_requires_verified_depth_scale_for_16uc1(tmp_path):
    manifest, color, depth = _manifest(tmp_path / "calibration.json")
    bag = tmp_path / "rgbd.db3"
    _bag(bag, color, depth)
    refiner = ArtifactRgbdRefiner(manifest)

    refiner.process_sqlite_bag(bag, artifact_sha256="a" * 64, lidar_frames=(_lidar_frame(),), measured_generation="pg-1")

    state = refiner.snapshot()
    assert state["depth_correspondence_count"] == 0
    assert state["accepted_color_observations"] == 0
    assert "VERIFIED_DEPTH_SCALE_REQUIRED_FOR_16UC1" in state["blockers"]
    evidence = refiner.appearance_evidence(("0:0:0",))
    assert evidence.ready is False
    assert "VERIFIED_DEPTH_SCALE_REQUIRED_FOR_16UC1" in evidence.blockers


def test_rgbd_artifact_refinement_reports_real_depth_and_texture_evidence(tmp_path):
    manifest, color, depth = _manifest(tmp_path / "calibration.json")
    bag = tmp_path / "rgbd.db3"
    _bag(bag, color, depth)
    refiner = ArtifactRgbdRefiner(manifest, depth_scale_m=0.001)
    before = np.array(_lidar_frame()["points_xyz"], copy=True)
    lidar = _lidar_frame()

    refiner.process_sqlite_bag(bag, artifact_sha256="a" * 64, lidar_frames=(lidar,), measured_generation="pg-1")

    state = refiner.snapshot()
    assert state["state"] == "CALIBRATED_RGBD_ACTIVE"
    assert state["accepted_depth_frames"] == 1
    assert state["depth_correspondence_count"] == 400
    assert state["depth_residual_median_m"] == 0.0
    assert state["depth_residual_p95_m"] == 0.0
    assert state["accepted_color_observations"] == 400
    assert state["texture"]["surface_coverage"] == 0.0
    assert state["texture"]["delivery_authorized"] is False
    assert np.array_equal(lidar["points_xyz"], before)

    observed = tuple(sorted(refiner._observed_vertices))
    requested = (observed[-1], observed[0])
    evidence = refiner.appearance_evidence(requested)
    assert evidence.ready is True
    assert tuple(vertex.vertex_id for vertex in evidence.vertex_colors) == requested
    assert len(evidence.physical_depth_validation_revision) == 64
    assert evidence.source_frame_sha256 == tuple(sorted(evidence.source_frame_sha256))
    assert evidence.calibration_sha256 == state["calibration_sha256"]

    missing = refiner.appearance_evidence(("not:a:measured-vertex",))
    assert missing.ready is False
    assert "NO_MATCHING_COLOR_VERTICES" in missing.blockers


def test_refiner_exposes_assistance_telemetry_without_requiring_the_worker():
    refiner = ArtifactRgbdRefiner(None)

    state = refiner.snapshot()

    assert state["depth_anything"]["state"] == "DEPTH_ANYTHING_DISABLED"
    assert state["depth_anything"]["augmentation_reference"] is None
    assert "DEPTH_ANYTHING_PROVIDER_UNAVAILABLE" in state["depth_anything"]["blockers"]


def test_refiner_queues_only_accepted_physical_correspondences_for_assistance(tmp_path):
    class RecordingWorker:
        model_revision_id = "e" * 64
        assistance_config_sha256 = "f" * 64

        def __init__(self):
            self.items = []

        def submit(self, item):
            self.items.append(item)
            return True

        def snapshot(self):
            return {"state": "DEPTH_ANYTHING_PENDING", "augmentation_reference": None, "blockers": []}

    manifest, color, depth = _manifest(tmp_path / "calibration.json")
    bag = tmp_path / "rgbd.db3"
    _bag(bag, color, depth)
    worker = RecordingWorker()
    refiner = ArtifactRgbdRefiner(manifest, depth_scale_m=0.001, depth_anything_worker=worker)

    refiner.process_sqlite_bag(
        bag,
        artifact_sha256="a" * 64,
        lidar_frames=(_lidar_frame(),),
        measured_generation="b" * 64,
    )

    assert len(worker.items) == 1
    item = worker.items[0]
    assert item.source_generation_id == "b" * 64
    assert item.source_revision_id == "a" * 64
    assert item.frame_sha256 == item.anchors[0].frame_sha256
    assert len(item.anchors) == 400
    assert item.rectified_rgb.shape == (20, 20, 3)
