import json

import numpy as np
import pytest

from mapping.pc.go2_mapping_neural import (
    GaussianBudget,
    NeuralDatasetExporter,
    NeuralFrame,
    NeuralTrainingError,
    NeuralTrainingJob,
)


HASH = "a" * 64


def _frame():
    return NeuralFrame(
        frame_id="frame-1",
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        camera_from_map=np.eye(4),
        camera_matrix=np.asarray([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]),
        calibration_id="cal-1",
        camera_info_sha256=HASH,
        source_artifact_sha256=HASH,
        rectified=True,
        depth_validation_accepted=True,
        depth_weight=1.0,
    )


def _dataset(tmp_path):
    seed = tmp_path / "measured.ply"
    seed.write_text("ply\n", encoding="ascii")
    return NeuralDatasetExporter(tmp_path / "datasets", GaussianBudget(1024, 64)).export(
        [_frame()], seed, {"eligible_for": {"neural_geometry": True, "texture": True}},
        calibration_id="cal-1", derived_cache_key=HASH,
    )


def test_neural_job_fails_closed_before_calibrated_dataset_authority(tmp_path):
    job = NeuralTrainingJob(tmp_path / "jobs")

    state = job.prepare(tmp_path / "missing.json", {"eligible_for": {}}, calibration_id="cal-1", pose_generation="pg-1")

    assert state["state"] == "BLOCKED"
    assert "NEURAL_DATASET_MANIFEST_MISSING" in state["blockers"]
    with pytest.raises(NeuralTrainingError):
        job.command()


def test_neural_job_exports_backend_command_and_hash_bound_checkpoint(tmp_path):
    dataset = _dataset(tmp_path)
    quality = {"eligible_for": {"neural_geometry": True, "texture": True}}
    job = NeuralTrainingJob(tmp_path / "jobs", backend="gsplat")

    state = job.prepare(dataset.manifest_path, quality, calibration_id="cal-1", pose_generation="pg-42")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"tiny-checkpoint")
    record = job.record_checkpoint(100, checkpoint)

    assert state["state"] == "READY"
    assert job.command(max_steps=1000)[0:3] == ["python", "-m", "gsplat"]
    assert record.dataset_manifest_sha256 == state["dataset_manifest_sha256"]
    assert record.pose_generation == "pg-42"
    metadata = json.loads((tmp_path / "jobs" / "checkpoint-00000100.json").read_text(encoding="utf-8"))
    assert metadata["geometry_authority"] is False
    assert metadata["checkpoint_sha256"] == record.sha256


def test_nerfstudio_backend_remains_presentation_only(tmp_path):
    dataset = _dataset(tmp_path)
    job = NeuralTrainingJob(tmp_path / "jobs", backend="nerfstudio")
    state = job.prepare(dataset.manifest_path, {"eligible_for": {"neural_geometry": True, "texture": True}}, calibration_id="cal-1", pose_generation="pg-1")

    assert state["geometry_authority"] is False
    assert job.command(max_steps=10)[:2] == ["ns-train", "nerfacto"]
