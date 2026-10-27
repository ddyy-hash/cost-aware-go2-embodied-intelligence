import hashlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np

import pytest
import zstandard

from mapping.pc.go2_mapping_calibration.bag import (
    BagEvidenceError,
    CalibrationBag,
    CalibrationCapture,
    CalibrationTopicMap,
)
from mapping.pc.go2_mapping_calibration.workflow import (
    CALIBRATION_BLOCKED_EXIT_CODE,
    run_calibration_workflow,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import (
    canonical_sha256,
    parse_calibration_manifest,
)


def _make_bag(path, topics, *, duration_ns=31_000_000_000):
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE topics (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            serialization_format TEXT NOT NULL DEFAULT 'cdr',
            offered_qos_profiles TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            topic_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            data BLOB NOT NULL
        );
        """
    )
    for topic_id, (name, type_name) in enumerate(topics, start=1):
        database.execute(
            "INSERT INTO topics (id, name, type) VALUES (?, ?, ?)",
            (topic_id, name, type_name),
        )
        database.execute(
            "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
            (topic_id, 1_000_000_000, b"first"),
        )
        database.execute(
            "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
            (topic_id, 1_000_000_000 + duration_ns, b"last"),
        )
    database.commit()
    database.close()
    return path


def _required_topics():
    mapping = CalibrationTopicMap.defaults()
    return [
        (mapping.lidar, "sensor_msgs/msg/PointCloud2"),
        (mapping.imu, "sensor_msgs/msg/Imu"),
        (mapping.lio_pose, "geometry_msgs/msg/PoseWithCovarianceStamped"),
        (mapping.color, "sensor_msgs/msg/Image"),
        (mapping.depth, "sensor_msgs/msg/Image"),
        (mapping.color_camera_info, "sensor_msgs/msg/CameraInfo"),
        (mapping.depth_camera_info, "sensor_msgs/msg/CameraInfo"),
        (mapping.tf, "tf2_msgs/msg/TFMessage"),
        (mapping.tf_static, "tf2_msgs/msg/TFMessage"),
    ]


def _runtime_manifest(path, *, robot_serial="go2-test", camera_serial="d435-test"):
    payload = {
        "schema_version": 1,
        "robot": {"model": "go2_edu", "serial": robot_serial, "identity_source": "operator", "factory_serial_verified": True},
        "camera": {"model": "Intel RealSense D435I", "serial": camera_serial, "firmware": "5.15.1.55", "identity_source": "rs-enumerate-devices"},
        "release_sha256": "a" * 64,
        "capture_mode": "calibration",
        "started_at_utc": "2026-08-02T12:00:00Z",
        "finished_at_utc": "2026-08-02T12:01:00Z",
        "ros_domain_id": 42,
        "topics": {name: type_name for name, type_name in _required_topics()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_calibration_bag_reads_zstd_read_only_and_hashes_published_bytes(tmp_path):
    source = _make_bag(tmp_path / "capture.db3", _required_topics())
    published = tmp_path / "capture.db3.zstd"
    published.write_bytes(zstandard.ZstdCompressor().compress(source.read_bytes()))
    before = published.read_bytes()

    with CalibrationBag(published) as bag:
        inventory = bag.inventory()
        materialized = bag.sqlite_path
        assert materialized != published
        assert materialized.exists()

    assert not materialized.exists()
    assert published.read_bytes() == before
    assert inventory.artifact_sha256 == hashlib.sha256(before).hexdigest()
    assert inventory.missing_streams == ()
    assert inventory.duration_s == pytest.approx(31.0)
    assert inventory.message_counts["lidar"] == 2


def test_calibration_bag_fails_closed_on_missing_or_wrong_type_stream(tmp_path):
    topics = _required_topics()
    topics = [item for item in topics if item[0] != "/tf_static"]
    topics[0] = (topics[0][0], "sensor_msgs/msg/LaserScan")
    bag_path = _make_bag(tmp_path / "incomplete.db3", topics)

    with CalibrationBag(bag_path) as bag:
        inventory = bag.inventory()
        with pytest.raises(BagEvidenceError, match="lidar.*tf_static"):
            bag.require_complete_inventory()

    assert inventory.missing_streams == ("lidar", "tf_static")


def test_workflow_writes_diagnostics_but_removes_stale_manifest_when_evidence_is_blocked(tmp_path):
    incomplete = _make_bag(tmp_path / "incomplete.db3", _required_topics()[:-1])
    diagnostics = tmp_path / "diagnostics.json"
    manifest = tmp_path / "calibration.json"
    manifest.write_text('{"stale": true}', encoding="utf-8")

    result = run_calibration_workflow(
        estimation_paths=[incomplete, incomplete, incomplete],
        holdout_path=incomplete,
        output_manifest=manifest,
        output_diagnostics=diagnostics,
        robot_serial="go2-unit-test",
        camera_serial="d435i-unit-test",
    )

    report = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert result.exit_code == CALIBRATION_BLOCKED_EXIT_CODE
    assert result.accepted is False
    assert not manifest.exists()
    assert report["accepted"] is False
    assert report["quality_level"] == "PREVIEW_UNCALIBRATED"
    assert "CAPTURE_ARTIFACTS_NOT_UNIQUE" in report["blockers"]
    assert any("MISSING_REQUIRED_STREAM" in reason for reason in report["blockers"])


def test_workflow_never_treats_preview_jpeg_as_calibration_evidence(tmp_path):
    preview = tmp_path / "preview.jpg"
    preview.write_bytes(b"not authoritative")

    with pytest.raises(BagEvidenceError, match="db3"):
        CalibrationBag(preview)


def test_segmented_capture_directory_has_one_order_independent_content_identity(tmp_path):
    capture = tmp_path / "segment-20260802T120000Z"
    capture.mkdir()
    first = _make_bag(capture / "segment_0.db3", _required_topics(), duration_ns=5_000_000_000)
    second = _make_bag(capture / "segment_1.db3", _required_topics(), duration_ns=5_000_000_000)
    second_bytes = second.read_bytes()
    second.unlink()
    compressed = capture / "segment_1.db3.zstd"
    compressed.write_bytes(zstandard.ZstdCompressor().compress(second_bytes))
    sidecar = _runtime_manifest(capture / "capture.runtime.json")

    with CalibrationCapture(capture) as evidence:
        inventory = evidence.inventory()
        lidar_messages = list(evidence.iter_messages("lidar"))

    assert inventory.part_count == 2
    assert inventory.message_counts["lidar"] == 4
    assert len(lidar_messages) == 4
    assert inventory.missing_streams == ()
    assert len(inventory.artifact_sha256) == 64
    assert [item.relative_path for item in inventory.parts] == ["segment_0.db3", "segment_1.db3.zstd"]
    assert inventory.runtime_manifest.robot_serial == "go2-test"
    assert inventory.runtime_manifest.camera_serial == "d435-test"
    assert inventory.runtime_manifest_sha256 == hashlib.sha256(sidecar.read_bytes()).hexdigest()


def test_segmented_capture_materializes_parts_with_a_bounded_worker_pool(monkeypatch, tmp_path):
    from mapping.pc.go2_mapping_calibration import bag as bag_module

    capture = tmp_path / "capture"
    capture.mkdir()
    for index in range(3):
        raw = _make_bag(capture / f"source-{index}.db3", _required_topics())
        compressed = capture / f"segment_{index}.db3.zstd"
        compressed.write_bytes(zstandard.ZstdCompressor().compress(raw.read_bytes()))
        raw.unlink()

    requested_workers = []

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            requested_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, values):
            return [function(value) for value in values]

    monkeypatch.setattr(bag_module, "ThreadPoolExecutor", ImmediateExecutor)

    with CalibrationCapture(capture, materialization_workers=3) as evidence:
        messages = list(evidence.iter_messages("lidar"))

    assert requested_workers == [3]
    assert len(messages) == 6
    assert [timestamp for timestamp, _ in messages] == sorted(
        timestamp for timestamp, _ in messages
    )


def test_capture_without_runtime_sidecar_remains_readable_but_not_production_complete(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    _make_bag(capture / "part.db3", _required_topics())

    with CalibrationCapture(capture) as evidence:
        inventory = evidence.inventory()

    assert inventory.runtime_manifest is None
    assert "runtime_manifest" in inventory.missing_evidence


def test_capture_runtime_sidecar_rejects_unknown_fields_and_topic_contract_mismatch(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    _make_bag(capture / "part.db3", _required_topics())
    sidecar = _runtime_manifest(capture / "capture.runtime.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with CalibrationCapture(capture) as evidence:
        with pytest.raises(BagEvidenceError, match="runtime manifest"):
            evidence.inventory()


def test_segmented_capture_rejects_hidden_staging_parts_and_empty_directories(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BagEvidenceError, match="no published"):
        CalibrationCapture(empty)
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    _make_bag(hidden / ".partial.db3", _required_topics())
    with pytest.raises(BagEvidenceError, match="no published"):
        CalibrationCapture(hidden)


def test_workflow_publishes_only_after_independent_lidar_imu_and_excluded_holdout_pass(monkeypatch, tmp_path):
    from mapping.pc.go2_mapping_calibration import workflow

    hashes = [character * 64 for character in "1234"]
    paths = [tmp_path / f"capture-{index}" for index in range(4)]
    camera_contract = {
        "width": 424,
        "height": 240,
        "distortion_model": "plumb_bob",
        "k": [300.0, 0.0, 212.0, 0.0, 300.0, 120.0, 0.0, 0.0, 1.0],
        "d": [0.0] * 5,
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [300.0, 0.0, 212.0, 0.0, 0.0, 300.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "rectified": False,
    }
    depth_contract = {**camera_contract, "k": [301.0, 0.0, 211.0, 0.0, 301.0, 119.0, 0.0, 0.0, 1.0]}

    def fit(digest):
        return SimpleNamespace(
            slope=1.0,
            offset_ns=0.0,
            sample_count=200,
            span_s=30.0,
            inlier_fraction=0.99,
            median_absolute_residual_ms=0.1,
            p95_residual_ms=0.2,
            maximum_prediction_stddev_ms=0.1,
            estimator_version="test-clock",
            sha256=digest,
        )

    analyses = []
    inventories = {}
    for index, (path, digest) in enumerate(zip(paths, hashes)):
        runtime = SimpleNamespace(
            robot_serial="go2-test",
            camera_serial="d435-test",
            robot_factory_serial_verified=True,
        )
        inventory = SimpleNamespace(
            artifact_sha256=digest,
            missing_streams=(),
            missing_evidence=(),
            runtime_manifest=runtime,
            as_dict=lambda digest=digest: {
                "artifact_sha256": digest,
                "missing_streams": [],
                "missing_evidence": [],
            },
        )
        inventories[path] = inventory
        analyses.append(
            SimpleNamespace(
                artifact_sha256=digest,
                inventory=inventory,
                accepted=True,
                blockers=(),
                clock_fits={
                    "lidar": fit("5678"[index] * 64),
                    "color": fit("9abc"[index] * 64),
                    "depth": fit("def0"[index] * 64),
                },
                board_detections=30,
                accepted_board_detections=30,
                accepted_lidar_planes=30,
                motion=SimpleNamespace(
                    cumulative_rotation_deg=45.0,
                    rotation_axes=3,
                    translation_m=1.0,
                    angular_speed_duration_s=2.0,
                    lio_epoch_count=1,
                    robust_translation_span_m=0.8,
                    robust_rotation_span_deg=42.0,
                    reset_reasons=(),
                ),
                color_camera_info=SimpleNamespace(frame_id="camera_color_optical_frame"),
                depth_camera_info=SimpleNamespace(frame_id="camera_depth_optical_frame"),
                color_camera_identity={"sha256": "a" * 64, "contract": camera_contract},
                depth_camera_identity={"sha256": "b" * 64, "contract": depth_contract},
                tf_static_records=(),
                lidar_records=((0, 0, "livox_frame", np.empty((0, 3))),),
            )
        )

    class FakeCapture:
        def __init__(self, path, topics=None):
            self.inventory_value = inventories[path]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def inventory(self):
            return self.inventory_value

    monkeypatch.setattr(workflow, "CalibrationCapture", FakeCapture)
    analysis_calls = []

    def analyze(path, topics=None, *, initial_camera_from_lidar=None):
        analysis_calls.append(initial_camera_from_lidar)
        return analyses[paths.index(type(paths[0])(path))]

    monkeypatch.setattr(workflow, "analyze_capture", analyze)
    solver_inputs = []

    def solve(captures):
        solver_inputs.extend(captures)
        return SimpleNamespace(
            accepted=True,
            blockers=(),
            camera_from_lidar=np.eye(4),
            delta_camera_lidar_ms=0.0,
            translation_repeatability_mm=1.0,
            rotation_repeatability_deg=0.1,
            timing_repeatability_ms=0.5,
            solver_version="test-camera-lidar-v1",
        )

    monkeypatch.setattr(workflow, "solve_three_estimation_captures", solve)
    monkeypatch.setattr(workflow, "resolve_frame_transform", lambda *args, **kwargs: np.eye(4))
    class Holdout(SimpleNamespace):
        def manifest_payload(self, artifact_sha256):
            return {
                "artifact_sha256": artifact_sha256,
                "accepted_board_poses": self.accepted_board_poses,
                "correspondences_per_pose": list(self.correspondences_per_pose),
                "total_correspondences": self.total_correspondences,
                "validated_range_min_m": self.validated_range_min_m,
                "validated_range_max_m": self.validated_range_max_m,
                "camera_reprojection_median_px": self.camera_reprojection_median_px,
                "camera_reprojection_p95_px": self.camera_reprojection_p95_px,
                "depth_median_absolute_residual_m": self.depth_median_absolute_residual_m,
                "depth_p95_absolute_residual_m": self.depth_p95_absolute_residual_m,
                "absolute_mean_signed_bias_m": self.absolute_mean_signed_bias_m,
                "occlusion_conflict_fraction": self.occlusion_conflict_fraction,
                "free_space_violation_fraction": self.free_space_violation_fraction,
                "t_d_c_validated": self.t_d_c_validated,
            }

    holdout = Holdout(
        accepted=True,
        blockers=(),
        accepted_board_poses=3,
        correspondences_per_pose=(4000, 4000, 4000),
        total_correspondences=12000,
        validated_range_min_m=0.6,
        validated_range_max_m=3.9,
        camera_reprojection_median_px=0.2,
        camera_reprojection_p95_px=0.4,
        depth_median_absolute_residual_m=0.01,
        depth_p95_absolute_residual_m=0.03,
        absolute_mean_signed_bias_m=0.005,
        occlusion_conflict_fraction=0.01,
        free_space_violation_fraction=0.005,
        t_d_c_validated=True,
    )
    monkeypatch.setattr(workflow, "evaluate_holdout_capture", lambda analysis, **kwargs: holdout)

    runs = []
    for index, digest in enumerate(hashes[:3]):
        transform = np.eye(4)
        transform[0, 3] = index * 0.001
        row_major = transform.reshape(-1).tolist()
        runs.append({
            "source_capture_sha256": digest,
            "result_artifact_sha256": f"{index + 5}" * 64,
            "transform": row_major,
            "transform_sha256": canonical_sha256({"row_major": row_major}),
            "timing_offset_ms": index * 0.2,
            "translation_repeatability_mm": 1.0,
            "rotation_repeatability_deg": 0.1,
            "timing_uncertainty_ms": 0.5,
            "accepted": True,
        })
    lidar_imu_path = tmp_path / "lidar-imu.json"
    lidar_imu_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "independent_raw",
        "robot_serial": "go2-test",
        "lidar_frame_id": "livox_frame",
        "imu_frame_id": "imu_link",
        "provenance": {"raw_lidar_topic": "/livox/lidar", "raw_imu_topic": "/livox/imu", "uses_vendor_extrinsic": False},
        "solver": {"name": "li-init", "version": "2024.1", "implementation_sha256": "e" * 64},
        "runs": runs,
    }), encoding="utf-8")
    manifest = tmp_path / "calibration.json"
    diagnostics = tmp_path / "diagnostics.json"

    result = run_calibration_workflow(
        estimation_paths=paths[:3],
        holdout_path=paths[3],
        lidar_imu_evidence_path=lidar_imu_path,
        output_manifest=manifest,
        output_diagnostics=diagnostics,
        robot_serial="go2-test",
        camera_serial="d435-test",
    )

    assert result.accepted is True
    assert result.exit_code == 0
    assert len(analysis_calls) == 4
    expected_seed = workflow.diagnostic_camera_from_lidar_seed()
    assert all(value == pytest.approx(expected_seed) for value in analysis_calls)
    assert all(not value.flags.writeable for value in analysis_calls)
    assert solver_inputs == analyses[:3]
    assert analyses[3] not in solver_inputs
    parsed = parse_calibration_manifest(json.loads(manifest.read_text(encoding="utf-8")))
    assert parsed.status == "measured"
    assert parsed.lidar_imu_evidence.source_capture_sha256 == tuple(hashes[:3])
    report = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert report["quality_level"] == "CALIBRATED_DENSE"
    assert report["captures"][0]["analysis"]["motion"] == {
        "angular_speed_duration_s": 2.0,
        "cumulative_rotation_deg": 45.0,
        "lio_epoch_count": 1,
        "reset_reasons": [],
        "robust_rotation_span_deg": 42.0,
        "robust_translation_span_m": 0.8,
        "rotation_axes": 3,
        "translation_m": 1.0,
    }
