import numpy as np
import pytest

from mapping.pc.go2_mapping_rgbd.depth_anything import RelativeDepthFrame
from mapping.pc.go2_mapping_rgbd.depth_anything_worker import (
    DepthAnythingWorkItem,
    DepthAnythingWorker,
    DepthAnythingWorkResult,
)


HASHES = tuple(value * 64 for value in ("a", "b", "c", "d", "e", "f", "1", "2", "3"))


class _Provider:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.frames = []

    def infer(self, rectified_rgb, *, frame_sha256):
        self.frames.append(frame_sha256)
        if self.fail:
            raise RuntimeError("inference failed")
        inverse = np.ones(rectified_rgb.shape[:2], dtype=np.float32)
        return RelativeDepthFrame(inverse, HASHES[4], "preprocess-v1", frame_sha256)


def test_worker_exposes_the_constructed_provider_as_read_only():
    provider = _Provider()
    worker = DepthAnythingWorker(provider, lambda item, relative: None)

    assert worker.provider is provider
    with pytest.raises(AttributeError):
        worker.provider = _Provider()
    assert worker.provider is provider
    assert DepthAnythingWorker(None, lambda item, relative: None).provider is None


def _item(frame=HASHES[6], generation=HASHES[0], anchors=(object(),)):
    return DepthAnythingWorkItem(
        frame_sha256=frame,
        source_generation_id=generation,
        source_revision_id=HASHES[1],
        calibration_sha256=HASHES[2],
        physical_depth_validation_revision=HASHES[3],
        model_revision_id=HASHES[4],
        assistance_config_sha256=HASHES[5],
        rectified_rgb=np.full((4, 5, 3), 127, dtype=np.uint8),
        anchors=anchors,
    )


def test_worker_coalesces_to_latest_frame_and_replay_is_idempotent():
    provider = _Provider()
    processed = []

    def process(item, relative):
        processed.append((item.frame_sha256, relative.frame_sha256))
        return DepthAnythingWorkResult(
            accepted=True,
            fit_accepted=True,
            assistance_coverage=0.25,
            blockers=(),
            augmentation_reference={"revision_id": HASHES[8]},
            validation_metrics={"surviving_anchor_count": 300},
        )

    worker = DepthAnythingWorker(provider, process)
    assert worker.submit(_item(frame=HASHES[6])) is True
    assert worker.submit(_item(frame=HASHES[7])) is True
    assert worker.run_once() is True
    assert provider.frames == [HASHES[7]]
    assert processed == [(HASHES[7], HASHES[7])]
    assert worker.submit(_item(frame=HASHES[7])) is False
    state = worker.snapshot()
    assert state["queue_capacity"] == 1
    assert state["coalesced_frames"] == 1
    assert state["processed_frames"] == 1
    assert state["idempotent_replays"] == 1
    assert state["assistance_coverage"] == 0.25
    assert state["augmentation_reference"] == {"revision_id": HASHES[8]}
    assert state["validation_metrics"] == {"surviving_anchor_count": 300}


def test_worker_drops_before_inference_without_lidar_anchor_evidence():
    provider = _Provider()
    worker = DepthAnythingWorker(provider, lambda item, relative: None)

    assert worker.submit(_item(anchors=())) is False
    assert worker.run_once() is False
    assert provider.frames == []
    assert "ACCEPTED_LIDAR_ANCHORS_REQUIRED" in worker.snapshot()["blockers"]


def test_generation_change_invalidates_pending_and_published_reference():
    provider = _Provider()
    worker = DepthAnythingWorker(
        provider,
        lambda item, relative: DepthAnythingWorkResult(
            True, True, 0.5, (), {"revision_id": HASHES[8]}
        ),
    )
    worker.submit(_item())
    worker.run_once()
    assert worker.snapshot()["augmentation_reference"] is not None

    worker.submit(_item(frame=HASHES[7], generation=HASHES[8]))

    state = worker.snapshot()
    assert state["source_generation_id"] == HASHES[8]
    assert state["augmentation_reference"] is None
    assert state["identity_invalidations"] == 1


def test_provider_and_processor_failures_are_isolated_and_visible():
    provider = _Provider(fail=True)
    worker = DepthAnythingWorker(provider, lambda item, relative: None)
    worker.submit(_item())

    assert worker.run_once() is False
    state = worker.snapshot()
    assert state["failed_frames"] == 1
    assert state["pending_frames"] == 0
    assert state["last_error"].startswith("RuntimeError:")
    assert "DEPTH_ANYTHING_INFERENCE_FAILED" in state["blockers"]


def test_unavailable_worker_is_diagnostic_only():
    worker = DepthAnythingWorker(None, lambda item, relative: None, enabled=False)

    assert worker.submit(_item()) is False
    assert worker.run_once() is False
    state = worker.snapshot()
    assert state["state"] == "DEPTH_ANYTHING_DISABLED"
    assert "DEPTH_ANYTHING_PROVIDER_UNAVAILABLE" in state["blockers"]
