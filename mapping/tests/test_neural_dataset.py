import json

import numpy as np
import pytest

from mapping.pc.go2_mapping_neural import (
    GaussianBudget,
    NeuralDatasetExporter,
    NeuralFrame,
)


HASH = "a" * 64


def _frame(frame_id="frame-1", *, accepted=True, calibration_id="cal-1"):
    return NeuralFrame(
        frame_id=frame_id,
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        camera_from_map=np.eye(4),
        camera_matrix=np.asarray([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]),
        calibration_id=calibration_id,
        camera_info_sha256=HASH,
        source_artifact_sha256=HASH,
        rectified=True,
        depth_validation_accepted=accepted,
        depth_weight=1.0,
    )


def _quality():
    return {"eligible_for": {"neural_geometry": True, "texture": True}}


def test_neural_dataset_contains_only_accepted_calibrated_frames(tmp_path):
    seed = tmp_path / "measured.ply"
    seed.write_text("ply\n", encoding="ascii")
    frames = [_frame("accepted"), _frame("bad-depth", accepted=False), _frame("bad-cal", calibration_id="other")]

    result = NeuralDatasetExporter(tmp_path / "datasets").export(
        frames,
        seed,
        _quality(),
        calibration_id="cal-1",
        derived_cache_key=HASH,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.accepted_frame_count == 1
    assert result.rejected_frame_count == 2
    assert result.geometry_authority is False
    assert manifest["adapters"] == ["gsplat", "nerfstudio"]
    assert manifest["measured_seed_authority"] == "LIDAR_MEASURED"


def test_neural_dataset_export_requires_appearance_authority(tmp_path):
    seed = tmp_path / "measured.ply"
    seed.write_text("ply\n", encoding="ascii")

    with pytest.raises(PermissionError, match="appearance"):
        NeuralDatasetExporter(tmp_path / "datasets").export(
            [_frame()],
            seed,
            {"eligible_for": {"neural_geometry": False, "texture": False}},
            calibration_id="cal-1",
            derived_cache_key=HASH,
        )


def test_neural_dataset_respects_gaussian_vram_budget(tmp_path):
    seed = tmp_path / "measured.ply"
    seed.write_text("ply\n", encoding="ascii")
    exporter = NeuralDatasetExporter(tmp_path / "datasets", GaussianBudget(1024, 64))

    with pytest.raises(MemoryError, match="VRAM"):
        exporter.export(
            [_frame()],
            seed,
            _quality(),
            calibration_id="cal-1",
            derived_cache_key=HASH,
            requested_gaussians=17,
        )
