import hashlib
import json

import pytest

from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore


def test_artifact_is_releasable_only_after_checksum_verification_and_reconstruction_ingest(tmp_path):
    root = tmp_path / "inbox"
    artifact = root / "lidar-imu-rgbd" / "segment-a" / "segment-a_0.db3.zstd"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"verified-raw-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    store = ArtifactLifecycleStore(root)

    verified = store.record_verified(
        session_id="segment-a",
        artifact_id=digest,
        relative_path=artifact.relative_to(root).as_posix(),
        size_bytes=artifact.stat().st_size,
    )

    assert verified["state"] == "SHA256_VERIFIED"
    assert verified["release_authorized"] is False
    reconstructed = store.mark_reconstructed(
        artifact,
        result={"duplicate": False, "integrated_frames": 2, "integrated_points": 400},
    )
    assert reconstructed["state"] == "PC_RECONSTRUCTION_INGESTED"
    assert reconstructed["release_authorized"] is True
    assert store.status("segment-a", digest)["artifact_sha256"] == digest

    repeated_seal = store.record_verified(
        "segment-a",
        digest,
        artifact.relative_to(root).as_posix(),
        artifact.stat().st_size,
    )
    assert repeated_seal["state"] == "PC_RECONSTRUCTION_INGESTED"
    assert repeated_seal["release_authorized"] is True


def test_artifact_lifecycle_fails_closed_for_missing_or_modified_pc_artifacts(tmp_path):
    root = tmp_path / "inbox"
    artifact = root / "lidar" / "segment.db3.zstd"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"first")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    store = ArtifactLifecycleStore(root)
    store.record_verified("segment-a", digest, artifact.relative_to(root).as_posix(), artifact.stat().st_size)
    artifact.write_bytes(b"modified")

    with pytest.raises(ValueError, match="checksum"):
        store.mark_reconstructed(artifact, result={"integrated_frames": 1, "integrated_points": 1})

    assert store.status("segment-a", digest)["release_authorized"] is False


def test_lifecycle_registry_is_atomic_json_and_rejects_unsafe_identity(tmp_path):
    store = ArtifactLifecycleStore(tmp_path)
    with pytest.raises(ValueError, match="session_id"):
        store.record_verified("../unsafe", "a" * 64, "lidar/part.db3.zstd", 1)

    path = store.registry_path("segment-a", "a" * 64)
    store.record_verified("segment-a", "a" * 64, "lidar/part.db3.zstd", 1)
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "SHA256_VERIFIED"
    assert not list(path.parent.glob("*.tmp"))
