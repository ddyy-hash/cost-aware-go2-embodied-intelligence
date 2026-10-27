import hashlib
import sqlite3

import zstandard

from mapping.pc.go2_mapping_quality.bootstrap_session_evidence import build_evidence


def _bag(path, topics):
    database = sqlite3.connect(path)
    database.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT)")
    for index, topic in enumerate(topics, start=1):
        database.execute("INSERT INTO topics VALUES (?, ?, 'test/type')", (index, topic))
    database.commit()
    database.close()


def test_bootstrap_uses_only_published_bags_and_reports_missing_dynamic_tf(tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    bag = root / "segment-001.db3"
    _bag(
        bag,
        [
            "/livox/lidar", "/dog_imu_raw", "/go2/lio/pose", "/camera/color/image_raw",
            "/camera/depth/image_rect_raw", "/camera/color/camera_info", "/camera/depth/camera_info", "/tf_static",
        ],
    )
    staging = root / ".progressive" / "session" / "ignored.db3"
    staging.parent.mkdir(parents=True)
    _bag(staging, ["/tf"])

    evidence = build_evidence(root, "cal-001")

    assert evidence["raw_capture"]["artifact_count"] == 1
    assert evidence["raw_capture"]["artifact_sha256"] == [hashlib.sha256(bag.read_bytes()).hexdigest()]
    assert "tf_static" in evidence["raw_capture"]["required_streams"]
    assert "tf" not in evidence["raw_capture"]["required_streams"]


def test_bootstrap_reads_verified_zstd_bags_for_one_selected_session(tmp_path):
    inbox = tmp_path / "inbox"
    session = inbox / "lidar-imu-rgbd" / "segment-001"
    source = tmp_path / "source.db3"
    _bag(
        source,
        [
            "/livox/lidar", "/dog_imu_raw", "/go2/lio/pose", "/camera/color/image_raw",
            "/camera/depth/image_rect_raw", "/camera/color/camera_info", "/camera/depth/camera_info",
            "/tf", "/tf_static",
        ],
    )
    compressed = session / "segment-001_0.db3.zstd"
    compressed.parent.mkdir(parents=True)
    with source.open("rb") as source_file, compressed.open("wb") as destination:
        zstandard.ZstdCompressor().copy_stream(source_file, destination)

    evidence = build_evidence(session, "cal-001")

    assert evidence["raw_capture"]["artifact_count"] == 1
    assert evidence["raw_capture"]["artifact_sha256"] == [hashlib.sha256(compressed.read_bytes()).hexdigest()]
    assert evidence["raw_capture"]["artifact_paths"][0]["relative_path"] == "segment-001_0.db3.zstd"
    assert set(evidence["raw_capture"]["required_streams"]) == {
        "lidar", "imu", "lio_pose", "rgb", "depth", "camera_info", "tf", "tf_static",
    }
