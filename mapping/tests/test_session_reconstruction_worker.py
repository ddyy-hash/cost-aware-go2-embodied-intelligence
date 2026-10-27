import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import zstandard

from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore
from mapping.pc.go2_mapping_appearance import measured_surface_identity
from mapping.pc.go2_mapping_dashboard.derived_surface import load_derived_surface_preview
from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config
from mapping.pc.go2_mapping_reconstruction.session_worker import SessionReconstructionWorker


TRANSFORM_ID = "PROVISIONAL_GO2_UTLIDAR_PITCH180_v2"
CAPTURE_LABEL = "lidar-imu-rgbd"
ROOT = Path(__file__).resolve().parents[2]


def _write_verified_part(inbox_root: Path, session_id: str, name: str, marker: bytes) -> tuple[Path, str]:
    artifact = inbox_root / CAPTURE_LABEL / session_id / name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("wb") as destination:
        with zstandard.ZstdCompressor().stream_writer(destination) as compressor:
            compressor.write(marker)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    ArtifactLifecycleStore(inbox_root).record_verified(
        session_id,
        digest,
        artifact.relative_to(inbox_root).as_posix(),
        artifact.stat().st_size,
    )
    return artifact, digest


class _RecordingMapper:
    instances = []

    def __init__(self, *, sensor_frame_transform_id: str, **_kwargs) -> None:
        self.sensor_frame_transform_id = sensor_frame_transform_id
        self.calls: list[tuple[Path, Path, str]] = []
        self.__class__.instances.append(self)

    def ingest_sqlite_bag(self, sqlite_path, *, artifact_sha256, source_artifact_path):
        sqlite_path = Path(sqlite_path)
        source_artifact_path = Path(source_artifact_path)
        assert sqlite_path.suffix == ".db3"
        assert sqlite_path.is_file()
        assert source_artifact_path.suffixes[-2:] == [".db3", ".zstd"]
        self.calls.append((sqlite_path, source_artifact_path, artifact_sha256))
        return {
            "duplicate": False,
            "integrated_frames": 1,
            "integrated_points": 3,
            "skipped_pose_frames": 0,
            "decode_errors": 0,
        }

    def snapshot(self):
        return {
            "pose_generation": "b" * 64,
            "optimized_pose_generation": "b" * 64,
            "sensor_frame_transform_id": self.sensor_frame_transform_id,
            "artifacts": len(self.calls),
            "frames": len(self.calls),
            "source_points": len(self.calls) * 3,
            "points": len(self.calls) * 3,
            "skipped_pose_frames": 0,
            "decode_errors": 0,
            "reconstruction_phase": "IDLE",
            "accepted_loop_closures": 0,
            "raw_source_artifacts": len(self.calls),
        }

    def measured_surface_snapshot(self):
        if not self.calls:
            return None
        source_hashes = sorted(call[2] for call in self.calls)
        positions = np.asarray(
            [[0.0, 0.0, -0.5], [1.0, 0.0, -0.5], [0.0, 0.0, 0.5]],
            dtype=np.float64,
        )
        normals = np.asarray(
            [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
        identity = measured_surface_identity(
            positions=positions,
            normals=normals,
            triangles=triangles,
            pose_generation="b" * 64,
            artifact_sha256=source_hashes,
            fusion_config_sha256=load_reconstruction_config().fusion_sha256,
        )
        return {
            "source_layer": "LIDAR_MEASURED",
            "geometry_authority": False,
            "point_authority": "LIDAR_MEASURED",
            "source_vertex_ids": ["0:0:-1", "1:0:-1", "0:0:1"],
            "positions": positions.reshape(-1).tolist(),
            "normals": normals.reshape(-1).tolist(),
            "triangles": triangles.reshape(-1).tolist(),
            "artifact_sha256": source_hashes,
            "pose_generation": "b" * 64,
            "measured_surface_geometry_revision_id": identity.revision_id,
            "reconstruction_phase": "IDLE",
            "sensor_frame_transform_id": self.sensor_frame_transform_id,
            "quality": "PREVIEW_UNCALIBRATED",
        }


def _worker(tmp_path, session_id, **kwargs):
    return SessionReconstructionWorker(
        session_id=session_id,
        inbox_root=tmp_path / "inbox",
        output_root=tmp_path / "derived",
        status_path=tmp_path / "runtime" / "automatic-reconstruction-current.json",
        sensor_frame_transform_id=TRANSFORM_ID,
        mapper_factory=_RecordingMapper,
        **kwargs,
    )


def test_automatic_worker_accepts_room_local_odom_source_mode(tmp_path):
    worker = _worker(
        tmp_path,
        "segment-20260812T083241Z",
        source_mode="RECORDED_LIO_SESSION_ODOM_CLOUD",
    )
    assert worker.source_mode == "RECORDED_LIO_SESSION_ODOM_CLOUD"


def test_worker_incrementally_publishes_only_current_verified_closed_parts_without_authorizing_release(tmp_path):
    session_id = "segment-20260812T083241Z"
    inbox_root = tmp_path / "inbox"
    part, digest = _write_verified_part(
        inbox_root,
        session_id,
        f"{session_id}_0.db3.zstd",
        b"current-session",
    )
    other_part, _other_digest = _write_verified_part(
        inbox_root,
        "segment-20260812T083242Z",
        "segment-20260812T083242Z_0.db3.zstd",
        b"other-session",
    )
    _RecordingMapper.instances.clear()

    worker = _worker(tmp_path, session_id)
    result = worker.process_once()

    assert result["state"] == "INCREMENTAL_SURFACE_AVAILABLE"
    assert len(_RecordingMapper.instances) == 1
    assert _RecordingMapper.instances[0].calls == [
        (_RecordingMapper.instances[0].calls[0][0], part, digest)
    ]
    assert _RecordingMapper.instances[0].calls[0][1] != other_part
    lifecycle = ArtifactLifecycleStore(inbox_root).status(session_id, digest)
    assert lifecycle["state"] == "SHA256_VERIFIED"
    assert lifecycle["release_authorized"] is False

    status_path = tmp_path / "runtime" / "automatic-reconstruction-current.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["session_id"] == session_id
    assert status["mode"] == "high_fidelity_incremental_reconstruction"
    assert status["finalized"] is False
    assert status["remaining_verified_artifacts"] == 0
    assert status["final_manifest_state"] == "ABSENT"
    assert status["surface_json_relative_path"] == Path(
        status["surface_json_path"]
    ).relative_to(tmp_path / "derived").as_posix()
    preview = load_derived_surface_preview(status["surface_json_path"])
    assert preview.status["surface_sha256"] == status["surface_sha256"]
    assert preview.status["derived_cache_key"] == status["derived_cache_key"]
    assert not list((tmp_path / "derived").rglob("*.partial"))
    assert not list(status_path.parent.glob("*.partial"))


def test_worker_rejects_a_lifecycle_record_that_does_not_describe_its_verified_file(tmp_path):
    session_id = "segment-20260812T083241Z"
    inbox_root = tmp_path / "inbox"
    _part, digest = _write_verified_part(
        inbox_root,
        session_id,
        f"{session_id}_0.db3.zstd",
        b"tampered-lifecycle",
    )
    registry = ArtifactLifecycleStore(inbox_root).registry_path(session_id, digest)
    record = json.loads(registry.read_text(encoding="utf-8"))
    record["relative_path"] = f"{CAPTURE_LABEL}/other-session/{session_id}_0.db3.zstd"
    registry.write_text(json.dumps(record), encoding="utf-8")
    _RecordingMapper.instances.clear()

    result = _worker(tmp_path, session_id).process_once()

    assert result["state"] == "WAITING_FOR_VERIFIED_ARTIFACTS"
    assert _RecordingMapper.instances[0].calls == []
    assert result["invalid_artifacts"] == 1


def test_worker_waits_for_a_complete_verified_manifest_before_running_final_replay(tmp_path):
    session_id = "segment-20260812T083241Z"
    inbox_root = tmp_path / "inbox"
    part, digest = _write_verified_part(
        inbox_root,
        session_id,
        f"{session_id}_0.db3.zstd",
        b"final-replay",
    )
    manifest = inbox_root / CAPTURE_LABEL / f"{session_id}.sha256"
    manifest.write_text(
        f"{'f' * 64}  /home/unitree/go2-mapping-data/{CAPTURE_LABEL}/{session_id}/{part.name}\n",
        encoding="utf-8",
    )
    calls = []
    _RecordingMapper.instances.clear()
    worker = _worker(
        tmp_path,
        session_id,
        final_replayer=lambda *_args, **_kwargs: calls.append(True),
    )

    incomplete = worker.process_once()

    assert incomplete["final_manifest_state"] == "INVALID"
    assert calls == []

    manifest.write_text(
        f"{digest}  /home/unitree/go2-mapping-data/{CAPTURE_LABEL}/{session_id}/{part.name}\n",
        encoding="utf-8",
    )

    def final_replayer(*_args, **_kwargs):
        calls.append(True)
        status = json.loads((tmp_path / "runtime" / "automatic-reconstruction-current.json").read_text(encoding="utf-8"))
        surface_path = Path(status["surface_json_path"])
        return SimpleNamespace(
            output_root=surface_path.parent,
            manifest_path=surface_path.with_name("manifest.json"),
            surface_json_path=surface_path,
        )

    worker._final_replayer = final_replayer
    finalized = worker.process_once()

    assert calls == [True]
    assert finalized["state"] == "FINAL_SURFACE_AVAILABLE"
    assert finalized["mode"] == "high_fidelity_batch_reconstruction"
    assert finalized["finalized"] is True


def test_worker_processes_bounded_incremental_parts_before_validating_the_final_manifest(tmp_path):
    session_id = "segment-20260812T083241Z"
    inbox_root = tmp_path / "inbox"
    first_part, first_digest = _write_verified_part(
        inbox_root,
        session_id,
        f"{session_id}_0.db3.zstd",
        b"first-bounded-part",
    )
    second_part, second_digest = _write_verified_part(
        inbox_root,
        session_id,
        f"{session_id}_1.db3.zstd",
        b"second-bounded-part",
    )
    manifest = inbox_root / CAPTURE_LABEL / f"{session_id}.sha256"
    manifest.write_text(
        "\n".join(
            (
                f"{first_digest}  /home/unitree/go2-mapping-data/{CAPTURE_LABEL}/{session_id}/{first_part.name}",
                f"{second_digest}  /home/unitree/go2-mapping-data/{CAPTURE_LABEL}/{session_id}/{second_part.name}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    final_replay_calls = []
    _RecordingMapper.instances.clear()

    worker = _worker(
        tmp_path,
        session_id,
        max_artifacts_per_pass=1,
        final_replayer=lambda *_args, **_kwargs: final_replay_calls.append(True),
    )

    first_pass = worker.process_once()

    assert [call[1] for call in _RecordingMapper.instances[0].calls] == [first_part]
    assert first_pass["final_manifest_state"] == "PRESENT_WAITING_FOR_INCREMENTAL_INGEST"
    assert first_pass["remaining_verified_artifacts"] == 1
    assert final_replay_calls == []

    def final_replayer(*_args, **_kwargs):
        final_replay_calls.append(True)
        status = json.loads((tmp_path / "runtime" / "automatic-reconstruction-current.json").read_text(encoding="utf-8"))
        surface_path = Path(status["surface_json_path"])
        return SimpleNamespace(
            output_root=surface_path.parent,
            manifest_path=surface_path.with_name("manifest.json"),
            surface_json_path=surface_path,
        )

    worker._final_replayer = final_replayer
    second_pass = worker.process_once()

    assert [call[1] for call in _RecordingMapper.instances[0].calls] == [first_part, second_part]
    assert second_pass["final_manifest_state"] == "VERIFIED"
    assert second_pass["remaining_verified_artifacts"] == 0
    assert final_replay_calls == [True]


def test_worker_can_finalize_a_closed_verified_session_without_waiting_for_incremental_ingest(tmp_path):
    session_id = "segment-20260812T083241Z"
    inbox_root = tmp_path / "inbox"
    first_part, first_digest = _write_verified_part(
        inbox_root,
        session_id,
        f"{session_id}_0.db3.zstd",
        b"first-final-part",
    )
    second_part, second_digest = _write_verified_part(
        inbox_root,
        session_id,
        f"{session_id}_1.db3.zstd",
        b"second-final-part",
    )
    _RecordingMapper.instances.clear()
    worker = _worker(tmp_path, session_id, max_artifacts_per_pass=1)

    incremental = worker.process_once()

    assert incremental["state"] == "INCREMENTAL_SURFACE_AVAILABLE"
    assert incremental["remaining_verified_artifacts"] == 1
    manifest = inbox_root / CAPTURE_LABEL / f"{session_id}.sha256"
    manifest.write_text(
        "\n".join(
            (
                f"{first_digest}  /home/unitree/go2-mapping-data/{CAPTURE_LABEL}/{session_id}/{first_part.name}",
                f"{second_digest}  /home/unitree/go2-mapping-data/{CAPTURE_LABEL}/{session_id}/{second_part.name}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    replay_calls = []

    def final_replayer(source_root, output_root, **kwargs):
        replay_calls.append((Path(source_root), Path(output_root), kwargs))
        return SimpleNamespace(surface_json_path=Path(incremental["surface_json_path"]))

    writes = []
    original_write_status = worker._write_status

    def record_write(status):
        writes.append(dict(status))
        original_write_status(status)

    worker._final_replayer = final_replayer
    worker._write_status = record_write
    finalized = worker.finalize_verified_session()

    assert replay_calls == [
        (
            inbox_root / CAPTURE_LABEL / session_id,
            tmp_path / "derived",
            {
                "sensor_frame_transform_id": TRANSFORM_ID,
                "work_root": Path("/tmp/go2-mapping-reconstruction") / session_id,
            },
        )
    ]
    assert finalized["state"] == "FINAL_SURFACE_AVAILABLE"
    assert finalized["mode"] == "high_fidelity_batch_reconstruction"
    assert finalized["finalized"] is True
    assert finalized["final_manifest_state"] == "VERIFIED"
    assert finalized["remaining_verified_artifacts"] == 0
    assert len(writes) == 1
    assert writes[0]["final_manifest_state"] == "VERIFIED"
    assert writes[0]["remaining_verified_artifacts"] == 0
    persisted = json.loads((tmp_path / "runtime" / "automatic-reconstruction-current.json").read_text(encoding="utf-8"))
    assert persisted["state"] == "FINAL_SURFACE_AVAILABLE"
    assert persisted["final_manifest_state"] == "VERIFIED"
    assert persisted["remaining_verified_artifacts"] == 0
    assert ArtifactLifecycleStore(inbox_root).status(session_id, first_digest)["release_authorized"] is False
    assert ArtifactLifecycleStore(inbox_root).status(session_id, second_digest)["release_authorized"] is False


def test_worker_marks_recorded_lio_only_final_replay_without_promoting_geometry(tmp_path):
    session_id = "segment-20260812T083241Z"
    inbox_root = tmp_path / "inbox"
    first_part, first_digest = _write_verified_part(
        inbox_root,
        session_id,
        f"{session_id}_0.db3.zstd",
        b"recorded-lio-final-part",
    )
    worker = _worker(tmp_path, session_id, final_global_optimization=False)
    incremental = worker.process_once()
    manifest = inbox_root / CAPTURE_LABEL / f"{session_id}.sha256"
    manifest.write_text(
        f"{first_digest}  /home/unitree/go2-mapping-data/{CAPTURE_LABEL}/{session_id}/{first_part.name}\n",
        encoding="utf-8",
    )
    replay_calls = []

    def final_replayer(source_root, output_root, **kwargs):
        replay_calls.append(kwargs)
        return SimpleNamespace(surface_json_path=Path(incremental["surface_json_path"]))

    worker._final_replayer = final_replayer
    finalized = worker.finalize_verified_session()

    assert replay_calls == [{
        "sensor_frame_transform_id": TRANSFORM_ID,
            "work_root": Path("/tmp/go2-mapping-reconstruction") / session_id,
        "global_optimization": False,
    }]
    assert finalized["pose_graph_optimization"] == "RECORDED_LIO_POSES_ONLY"
    assert finalized["geometry_authority"] is False
    assert finalized["point_authority"] == "LIDAR_MEASURED"

    status_path = tmp_path / "runtime" / "automatic-reconstruction-current.json"
    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    legacy_pointer = dict(persisted)
    legacy_pointer.pop("surface_json_relative_path")
    status_path.write_text(json.dumps(legacy_pointer), encoding="utf-8")

    resumed = _worker(tmp_path, session_id, final_global_optimization=False)

    def final_replay_must_not_run(*_args, **_kwargs):
        raise AssertionError("a finalized pointer must be reused")

    resumed._final_replayer = final_replay_must_not_run
    migrated = resumed.finalize_verified_session()

    assert migrated["surface_json_relative_path"] == persisted["surface_json_relative_path"]
    assert json.loads(status_path.read_text(encoding="utf-8"))["surface_json_relative_path"] == (
        persisted["surface_json_relative_path"]
    )

    migrated["surface_json_path"] = r"C:\stale-host\derived\measured-surface.json"
    status_path.write_text(json.dumps(migrated), encoding="utf-8")

    resumed = _worker(tmp_path, session_id, final_global_optimization=False)

    def stale_host_final_replay_must_not_run(*_args, **_kwargs):
        raise AssertionError("a portable finalized pointer must be reused")

    resumed._final_replayer = stale_host_final_replay_must_not_run
    reused = resumed.finalize_verified_session()

    assert reused["surface_json_relative_path"] == persisted["surface_json_relative_path"]
    assert reused["state"] == "FINAL_SURFACE_AVAILABLE"


def test_automatic_reconstruction_tool_runs_as_a_direct_script_from_the_workspace_root():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_go2_automatic_reconstruction.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--finalize-verified-session" in result.stdout
