import hashlib
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import zstandard

from mapping.pc.go2_mapping_live.sensor_frame import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
from mapping.pc.go2_mapping_reconstruction.offline import (
    PREVIEW_UNCALIBRATED,
    RECORDED_LIO_SESSION_ODOM_CLOUD,
    derive_capture,
)


def _load_cli_module():
    path = Path(__file__).parents[2] / "tools" / "derive_go2_reconstruction.py"
    spec = importlib.util.spec_from_file_location("derive_go2_reconstruction_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_compressed_part(path: Path, marker: bytes) -> None:
    with path.open("wb") as destination:
        with zstandard.ZstdCompressor().stream_writer(destination) as writer:
            writer.write(marker)


class _RecordingMapper:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.finalize_calls = 0
        registered = kwargs.get("source_mode") in {
            "RECORDED_LIO_REGISTERED_CLOUD",
            "RECORDED_LIO_SESSION_ODOM_CLOUD",
        }
        self._surface = {
            "source_layer": "LIO_REGISTERED_LIDAR_OBSERVED" if registered else "LIDAR_MEASURED",
            "geometry_authority": False,
            "point_authority": "LIO_REGISTERED_LIDAR_OBSERVED" if registered else "LIDAR_MEASURED",
            "source_vertex_ids": ["0:0:0", "1:0:0", "0:1:0"],
            "positions": [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
            "normals": [0.0, 0.0, 1.0] * 3,
            "triangles": [0, 1, 2],
            "pose_generation": "a" * 64,
            "reconstruction_phase": "IDLE",
            "artifact_sha256": [],
            "sensor_frame_transform_id": PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
            "mapping_source_mode": kwargs.get("source_mode", "RAW_LIDAR_WITH_RECORDED_LIO"),
            "sensor_frame_transform_applied": not registered,
            "quality": PREVIEW_UNCALIBRATED,
        }
        self.__class__.instances.append(self)

    def ingest_sqlite_bag(self, path, *, artifact_sha256, source_artifact_path):
        path = Path(path)
        source_artifact_path = Path(source_artifact_path)
        assert path.suffix == ".db3"
        assert path != source_artifact_path
        assert source_artifact_path.suffixes[-2:] == [".db3", ".zstd"]
        assert path.is_file()
        self.calls.append((path, source_artifact_path, artifact_sha256))
        self._surface["artifact_sha256"] = [*self._surface["artifact_sha256"], artifact_sha256]
        return {"duplicate": False, "integrated_frames": 1, "integrated_points": 3}

    def snapshot(self):
        return {
            "pose_generation": "a" * 64,
            "optimized_pose_generation": "a" * 64,
            "sensor_frame_transform_id": self.kwargs["sensor_frame_transform_id"],
            "artifacts": len(self.calls),
            "frames": len(self.calls),
            "source_points": len(self.calls) * 3,
            "skipped_pose_frames": 0,
            "decode_errors": 0,
            "points": len(self.calls) * 3,
            "reconstruction_phase": "IDLE",
        }

    def measured_surface_snapshot(self):
        return dict(self._surface)

    def finalize_deferred_batch(self):
        self.finalize_calls += 1


class _NoSurfaceMapper(_RecordingMapper):
    def measured_surface_snapshot(self):
        return None


def test_derive_capture_is_content_addressed_and_keeps_each_part_ephemeral(tmp_path):
    source = tmp_path / "capture"
    output = tmp_path / "derived"
    source.mkdir()
    for index in (10, 2, 0):
        _write_compressed_part(source / f"segment-20260807T000000Z_{index}.db3.zstd", str(index).encode())
    (output / "legacy").mkdir(parents=True)
    (output / "legacy" / "CURRENT").write_text("old-generation\n", encoding="utf-8")
    _RecordingMapper.instances.clear()

    result = derive_capture(
        source,
        output,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        mapper_factory=_RecordingMapper,
    )

    assert result.manifest_path.is_file()
    assert result.surface_json_path.is_file()
    assert result.ply_path.is_file()
    assert result.output_root.is_relative_to(output.resolve())
    assert not result.output_root.is_relative_to(source.resolve())
    assert (output / "legacy" / "CURRENT").read_text(encoding="utf-8") == "old-generation\n"
    assert [call[1].name for call in _RecordingMapper.instances[0].calls] == [
        "segment-20260807T000000Z_0.db3.zstd",
        "segment-20260807T000000Z_2.db3.zstd",
        "segment-20260807T000000Z_10.db3.zstd",
    ]
    assert _RecordingMapper.instances[0].kwargs["prewarm_surface_after_ingest"] is False
    assert _RecordingMapper.instances[0].kwargs["defer_pose_graph_optimization"] is True
    assert _RecordingMapper.instances[0].finalize_calls == 1
    assert all(not call[0].exists() for call in _RecordingMapper.instances[0].calls)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == PREVIEW_UNCALIBRATED
    assert manifest["quality"] == PREVIEW_UNCALIBRATED
    assert manifest["geometry_authority"] is False
    assert manifest["volume_filled"] is False
    assert manifest["watertight"] is False
    assert manifest["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
    assert manifest["derived_identity"]["fusion_config_sha256"]
    assert manifest["derived_identity"]["pose_generation"] == "a" * 64
    assert manifest["derived_identity"]["sensor_frame_transform_id"] == PROVISIONAL_GO2_UTLIDAR_PITCH180_V2
    assert len(manifest["source_artifacts"]) == 3
    assert {item["sha256"] for item in manifest["source_artifacts"]} == {
        hashlib.sha256(
            (source / f"segment-20260807T000000Z_{index}.db3.zstd").read_bytes()
        ).hexdigest()
        for index in (10, 2, 0)
    }
    assert manifest["surface"]["vertex_count"] == 3
    assert manifest["surface"]["triangle_count"] == 1


def test_derive_capture_reports_ordered_part_and_phase_progress(tmp_path):
    source = tmp_path / "capture"
    output = tmp_path / "derived"
    source.mkdir()
    _write_compressed_part(source / "segment-20260807T000000Z_0.db3.zstd", b"part")
    events = []

    derive_capture(
        source,
        output,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        mapper_factory=_RecordingMapper,
        progress_callback=events.append,
    )

    assert [event["stage"] for event in events] == [
        "part_ready", "part_ingested", "reintegration", "surfacing"
    ]


def test_derive_capture_rejects_an_output_that_can_overlap_raw_capture(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    _write_compressed_part(source / "segment_0.db3.zstd", b"part")

    with pytest.raises(ValueError, match="independent output root"):
        derive_capture(
            source,
            source / "derived",
            sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
            mapper_factory=_RecordingMapper,
        )


def test_derive_capture_does_not_replace_an_existing_content_addressed_generation(tmp_path):
    source = tmp_path / "capture"
    output = tmp_path / "derived"
    source.mkdir()
    _write_compressed_part(source / "segment_0.db3.zstd", b"part")
    _RecordingMapper.instances.clear()

    first = derive_capture(
        source,
        output,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        mapper_factory=_RecordingMapper,
    )
    before = first.manifest_path.read_bytes()
    second = derive_capture(
        source,
        output,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        mapper_factory=_RecordingMapper,
    )

    assert second.output_root == first.output_root
    assert second.manifest_path.read_bytes() == before


def test_derive_capture_binds_registered_cloud_source_mode_into_its_generation_identity(tmp_path):
    source = tmp_path / "capture"
    output = tmp_path / "derived"
    source.mkdir()
    _write_compressed_part(source / "segment_0.db3.zstd", b"part")
    _RecordingMapper.instances.clear()

    result = derive_capture(
        source,
        output,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        source_mode="RECORDED_LIO_REGISTERED_CLOUD",
        mapper_factory=_RecordingMapper,
    )

    assert _RecordingMapper.instances[0].kwargs["source_mode"] == "RECORDED_LIO_REGISTERED_CLOUD"
    assert result.manifest["derived_identity"]["mapping_source_mode"] == "RECORDED_LIO_REGISTERED_CLOUD"
    assert result.manifest["mapping_source_mode"] == "RECORDED_LIO_REGISTERED_CLOUD"
    assert result.manifest["source_layer"] == "LIO_REGISTERED_LIDAR_OBSERVED"
    assert result.manifest["point_authority"] == "LIO_REGISTERED_LIDAR_OBSERVED"
    assert result.manifest["sensor_frame_transform_applied"] is False


def test_derive_capture_marks_a_registered_odom_replay_as_local_observed_only(tmp_path):
    source = tmp_path / "capture"
    output = tmp_path / "derived"
    source.mkdir()
    _write_compressed_part(source / "segment_0.db3.zstd", b"part")
    _RecordingMapper.instances.clear()

    result = derive_capture(
        source,
        output,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        source_mode=RECORDED_LIO_SESSION_ODOM_CLOUD,
        mapper_factory=_RecordingMapper,
    )

    assert _RecordingMapper.instances[0].kwargs["source_mode"] == RECORDED_LIO_SESSION_ODOM_CLOUD
    assert result.manifest["derived_identity"]["mapping_source_mode"] == RECORDED_LIO_SESSION_ODOM_CLOUD
    assert result.manifest["coordinate_scope"] == "SESSION_ODOM_LOCAL"
    assert result.manifest["coordinate_frame"] == "odom"
    assert result.manifest["room_or_building"] is False
    assert result.manifest["texture_authorized"] is False
    assert result.manifest["sensor_frame_transform_applied"] is False
    surface = json.loads(result.surface_json_path.read_text(encoding="utf-8"))
    assert surface["coordinate_scope"] == "SESSION_ODOM_LOCAL"
    assert surface["room_or_building"] is False
    assert surface["texture_authorized"] is False


def test_derive_capture_reuses_a_completed_generation_without_a_surface(tmp_path):
    source = tmp_path / "capture"
    output = tmp_path / "derived"
    source.mkdir()
    _write_compressed_part(source / "segment_0.db3.zstd", b"part")
    _NoSurfaceMapper.instances.clear()

    first = derive_capture(
        source,
        output,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        mapper_factory=_NoSurfaceMapper,
    )
    before = first.manifest_path.read_bytes()
    second = derive_capture(
        source,
        output,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        mapper_factory=_NoSurfaceMapper,
    )

    assert second.output_root == first.output_root
    assert second.surface_json_path is None
    assert second.ply_path is None
    assert second.manifest_path.read_bytes() == before


def test_offline_derivation_cli_exposes_explicit_capture_and_output_contract(capsys):
    module = _load_cli_module()

    parser = module.build_parser()
    options = {action.dest for action in parser._actions}

    assert {"input_dir", "output_root", "sensor_frame_transform_id", "max_parts", "source_mode"} <= options
    assert "ssh" not in Path(module.__file__).read_text(encoding="utf-8").lower()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["--help"])
    assert raised.value.code == 0
    assert "PREVIEW_UNCALIBRATED" in capsys.readouterr().out
