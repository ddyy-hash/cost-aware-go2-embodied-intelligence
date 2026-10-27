import hashlib

import numpy as np

from mapping.pc.go2_mapping_rgbd import depth_anything_runtime
from mapping.pc.go2_mapping_rgbd.depth_anything import RelativeDepthFrame
from mapping.pc.go2_mapping_rgbd.depth_anything_runtime import (
    DepthAnythingProcessor,
    create_depth_anything_worker_from_environment,
)
from mapping.pc.go2_mapping_rgbd.depth_anything_worker import DepthAnythingWorkItem
from mapping.pc.go2_mapping_rgbd.depth_validation import AcceptedDepthCorrespondence


HASHES = tuple(value * 64 for value in "abcdef12")


def _correspondences(width=80, height=60):
    records = []
    index = 0
    for row in range(6):
        for column in range(8):
            for offset in range(6):
                x = column * 10 + 1 + offset
                y = row * 10 + 2
                model = 1.0 + x / width + y / height
                depth = 1.0 / (1.25 * model + 0.02)
                records.append(
                    AcceptedDepthCorrespondence(
                        lidar_point_index=index,
                        depth_pixel=(x, y),
                        color_pixel=(x, y),
                        point_camera_xyz=(0.0, 0.0, depth),
                        observed_metric_depth_m=depth,
                        residual_class="CONSENSUS",
                        accepted_region_id=f"cell-{row * 8 + column}",
                        frame_sha256=HASHES[0],
                        calibration_sha256=HASHES[2],
                        measured_generation=HASHES[3],
                        timestamp_ns=1_000,
                    )
                )
                index += 1
    return tuple(records)


def _item():
    return DepthAnythingWorkItem(
        frame_sha256=HASHES[0],
        source_generation_id=HASHES[3],
        source_revision_id=HASHES[1],
        calibration_sha256=HASHES[2],
        physical_depth_validation_revision=HASHES[4],
        model_revision_id=HASHES[5],
        assistance_config_sha256=HASHES[6],
        rectified_rgb=np.full((60, 80, 3), 127, dtype=np.uint8),
        anchors=_correspondences(),
    )


def test_runtime_factory_requires_explicit_local_verified_configuration(tmp_path):
    assert create_depth_anything_worker_from_environment({}, augmentation_store=object()) is None

    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"verified-local-weights")
    common = {
        "GO2_MAPPING_DEPTH_ANYTHING_ENABLED": "1",
        "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_PATH": str(weights),
        "GO2_MAPPING_DEPTH_ANYTHING_DEVICE": "cpu",
        "GO2_MAPPING_DEPTH_ANYTHING_PRECISION": "float32",
    }
    assert create_depth_anything_worker_from_environment(
        {**common, "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_SHA256": "0" * 64},
        augmentation_store=object(),
    ) is None
    assert create_depth_anything_worker_from_environment(
        {**common, "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_SHA256": "a" * 64},
        augmentation_store=object(),
    ) is None


def test_runtime_factory_selects_explicit_da3_backend(tmp_path, monkeypatch):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified-da3-weights")
    digest = hashlib.sha256(b"not-the-tree-digest").hexdigest()
    observed = {}

    class Provider:
        model_revision_id = HASHES[5]

        def __init__(self, **kwargs):
            observed.update(kwargs)

        def infer(self, rectified_rgb, *, frame_sha256):
            return RelativeDepthFrame(
                np.ones(rectified_rgb.shape[:2], dtype=np.float32),
                self.model_revision_id,
                "da3-local-v1",
                frame_sha256,
            )

    monkeypatch.setattr(depth_anything_runtime, "DepthAnything3Provider", Provider)
    worker = create_depth_anything_worker_from_environment(
        {
            "GO2_MAPPING_DEPTH_ANYTHING_ENABLED": "1",
            "GO2_MAPPING_DEPTH_ANYTHING_BACKEND": "da3",
            "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_PATH": str(weights),
            "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_SHA256": digest,
            "GO2_MAPPING_DEPTH_ANYTHING_DEVICE": "cpu",
            "GO2_MAPPING_DEPTH_ANYTHING_PRECISION": "float32",
        },
        augmentation_store=object(),
    )

    assert worker is not None
    assert observed["weights_path"] == str(weights)
    assert observed["weights_sha256"] == digest


def test_runtime_worker_executes_full_metric_validation_without_claiming_single_frame_surface(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"verified-local-weights")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()

    def provider_factory(**kwargs):
        assert kwargs["weights_path"] == str(weights)
        assert kwargs["weights_sha256"] == digest

        class Provider:
            model_revision_id = HASHES[5]

            def infer(self, rectified_rgb, *, frame_sha256):
                y, x = np.indices(rectified_rgb.shape[:2])
                model = 1.0 + x / rectified_rgb.shape[1] + y / rectified_rgb.shape[0]
                return RelativeDepthFrame(
                    model,
                    self.model_revision_id,
                    "local-only-v1",
                    frame_sha256,
                    np.full(model.shape, 0.8, dtype=np.float32),
                )

        return Provider()

    worker = create_depth_anything_worker_from_environment(
        {
            "GO2_MAPPING_DEPTH_ANYTHING_ENABLED": "1",
            "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_PATH": str(weights),
            "GO2_MAPPING_DEPTH_ANYTHING_WEIGHTS_SHA256": digest,
            "GO2_MAPPING_DEPTH_ANYTHING_DEVICE": "cpu",
            "GO2_MAPPING_DEPTH_ANYTHING_PRECISION": "float32",
        },
        augmentation_store=object(),
        provider_factory=provider_factory,
        assistance_config_sha256=HASHES[6],
    )

    assert worker is not None
    assert isinstance(worker._processor, DepthAnythingProcessor)
    assert worker.submit(_item()) is True
    assert worker.run_once() is False
    snapshot = worker.snapshot()
    metrics = snapshot["validation_metrics"]
    assert snapshot["fit_accepted"] is True
    assert snapshot["augmentation_reference"] is None
    assert metrics["surviving_anchor_count"] == 288
    assert metrics["occupied_cells"] == 48
    assert metrics["total_cells"] == 48
    assert metrics["quadrants"] == [True, True, True, True]
    assert metrics["held_out_median_m"] <= 0.025
    assert metrics["held_out_p95_m"] <= 0.060
    assert metrics["relative_scale_stddev"] <= 0.02
    assert metrics["maximum_uncertainty_m"] <= 0.030
    assert np.isclose(metrics["model_confidence_p10"], 0.8)
    assert np.isclose(metrics["model_confidence_median"], 0.8)
    assert "MULTI_FRAME_BOUNDARY_EVIDENCE_REQUIRED" in snapshot["blockers"]
