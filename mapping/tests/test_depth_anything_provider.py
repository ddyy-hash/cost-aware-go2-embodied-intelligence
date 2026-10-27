from pathlib import Path
import hashlib
import sys
import threading
from types import ModuleType

import numpy as np
import pytest
import torch

from mapping.pc.go2_mapping_quality.reconstruction_config import load_reconstruction_config
from mapping.pc.go2_mapping_rgbd import depth_anything
from mapping.pc.go2_mapping_rgbd.depth_anything import (
    DA3_SOURCE_REVISION,
    DepthAnything3Provider,
    DepthAnythingError,
    RelativeDepthFrame,
    TransformersDepthAnythingProvider,
)


def test_model_inputs_match_model_precision_without_casting_integer_tensors():
    assert hasattr(depth_anything, "_move_model_inputs")
    moved = depth_anything._move_model_inputs(
        {
            "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
            "mask": torch.ones((1, 2), dtype=torch.int64),
        },
        device="cpu",
        dtype=torch.float16,
    )

    assert moved["pixel_values"].dtype == torch.float16
    assert moved["mask"].dtype == torch.int64


def test_low_precision_cuda_inference_uses_autocast():
    assert depth_anything._requires_autocast("cuda", torch.float16, torch) is True
    assert depth_anything._requires_autocast("cuda:0", torch.bfloat16, torch) is True
    assert depth_anything._requires_autocast("cuda", torch.float32, torch) is False
    assert depth_anything._requires_autocast("cpu", torch.float16, torch) is False


def test_depth_anything_configuration_is_versioned_and_fail_closed():
    gates = load_reconstruction_config().appearance.depth_anything

    assert gates.schema_version == 1
    assert gates.minimum_anchor_count == 256
    assert (gates.grid_columns, gates.grid_rows) == (8, 6)
    assert gates.holdout_fraction == pytest.approx(0.2)
    assert gates.minimum_holdout_count >= 64
    assert gates.huber_delta_inverse_m == pytest.approx(0.05)
    assert gates.maximum_irls_iterations == 50
    assert gates.parameter_relative_tolerance == pytest.approx(1e-9)
    assert gates.maximum_holdout_median_residual_m == pytest.approx(0.025)
    assert gates.maximum_holdout_p95_residual_m == pytest.approx(0.06)
    assert gates.maximum_holdout_absolute_bias_m == pytest.approx(0.015)
    assert gates.bootstrap_replicates == 1000
    assert gates.maximum_scale_stddev_fraction == pytest.approx(0.02)
    assert gates.maximum_uncertainty_m == pytest.approx(0.03)
    assert gates.maximum_boundary_distance_m == pytest.approx(0.15)
    assert gates.model_name
    assert len(gates.weights_sha256) == 64
    assert gates.weights_sha256 == "37a91a613d005c92f71e6b0caf2790c112ab06a7794fd2fa275521be14510137"
    assert gates.preprocessing_revision
    assert gates.input_size == (518, 518)
    assert gates.precision in {"float16", "bfloat16", "float32"}


def test_provider_is_lazy_local_only_and_returns_immutable_positive_inverse_depth(tmp_path: Path):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"verified local weights")
    observed = {}

    def loader(provider):
        observed["loaded"] = True
        observed["local_files_only"] = provider.local_files_only

        def infer(image):
            return np.full(image.shape[:2], 2.0, dtype=np.float32)

        return infer

    provider = TransformersDepthAnythingProvider(
        weights_path=weights,
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        model_name="test-depth-anything",
        preprocessing_revision="preprocess-v1",
        input_size=(518, 518),
        precision="float32",
        model_loader=loader,
    )
    assert observed == {}

    frame = provider.infer(np.zeros((3, 4, 3), dtype=np.uint8), frame_sha256="a" * 64)

    assert observed == {"loaded": True, "local_files_only": True}
    assert frame.inverse_depth.shape == (3, 4)
    assert frame.inverse_depth.dtype == np.float32
    assert np.all(frame.inverse_depth == 2.0)
    assert frame.inverse_depth.flags.writeable is False
    assert frame.model_revision_id == provider.model_revision_id
    assert frame.preprocessing_revision == "preprocess-v1"
    assert frame.frame_sha256 == "a" * 64
    with pytest.raises(ValueError):
        frame.inverse_depth[0, 0] = 3.0


def test_provider_rejects_weight_mismatch_bad_input_and_non_positive_output(tmp_path: Path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")

    with pytest.raises(DepthAnythingError, match="weights SHA-256"):
        TransformersDepthAnythingProvider(
            weights_path=weights,
            weights_sha256="b" * 64,
            model_loader=lambda _provider: lambda image: np.ones(image.shape[:2]),
        )

    provider = TransformersDepthAnythingProvider(
        weights_path=weights,
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        model_loader=lambda _provider: lambda image: np.zeros(image.shape[:2]),
    )
    with pytest.raises(ValueError, match="RGB"):
        provider.infer(np.zeros((3, 4), dtype=np.uint8), frame_sha256="a" * 64)
    with pytest.raises(ValueError, match="positive"):
        provider.infer(np.zeros((3, 4, 3), dtype=np.uint8), frame_sha256="a" * 64)


def test_relative_depth_frame_rejects_non_finite_and_bad_identity():
    with pytest.raises(ValueError, match="finite positive"):
        RelativeDepthFrame(np.array([[np.nan]], dtype=np.float32), "a" * 64, "prep", "b" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        RelativeDepthFrame(np.ones((1, 1), dtype=np.float32), "not-a-revision", "prep", "b" * 64)


def test_da3_provider_is_lazy_local_only_and_preserves_confidence(tmp_path: Path):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified da3 weights")
    digest = depth_anything._sha256_path(weights)
    observed = {}

    class Prediction:
        depth = np.full((1, 3, 4), 0.5, dtype=np.float32)
        conf = np.full((1, 3, 4), 0.75, dtype=np.float32)

    def loader(provider):
        observed["loaded"] = True
        observed["weights_path"] = provider.weights_path
        observed["local_files_only"] = provider.local_files_only

        def infer(images, **kwargs):
            observed["image_count"] = len(images)
            observed["kwargs"] = kwargs
            return Prediction()

        return infer

    provider = DepthAnything3Provider(
        weights_path=weights,
        weights_sha256=digest,
        model_name="DA3-BASE",
        preprocessing_revision="da3-preprocess-v1",
        input_size=(504, 504),
        precision="float32",
        model_loader=loader,
    )
    assert observed == {}

    frame = provider.infer(np.zeros((3, 4, 3), dtype=np.uint8), frame_sha256="a" * 64)

    assert np.all(frame.inverse_depth == 2.0)
    assert np.all(frame.model_confidence == 0.75)
    assert frame.model_confidence.flags.writeable is False
    assert observed["local_files_only"] is True
    assert observed["image_count"] == 1
    assert observed["kwargs"] == {
        "process_res": 504,
        "process_res_method": "upper_bound_resize",
        "ref_view_strategy": "middle",
    }


def test_da3_single_view_resizes_processing_resolution_back_to_source_pixels(tmp_path: Path):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified da3 weights")
    digest = depth_anything._sha256_path(weights)

    class Prediction:
        depth = np.arange(1, 25, dtype=np.float32).reshape(1, 4, 6)
        conf = np.arange(24, dtype=np.float32).reshape(1, 4, 6) / 10.0

    provider = DepthAnything3Provider(
        weights,
        weights_sha256=digest,
        model_loader=lambda _provider: lambda _images, **_kwargs: Prediction(),
    )

    frame = provider.infer(
        np.zeros((2, 3, 3), dtype=np.uint8),
        frame_sha256="a" * 64,
    )

    resized_depth = np.asarray(
        [[6.0, 7.785714, 9.571428], [15.428571, 17.214285, 19.0]],
        dtype=np.float32,
    )
    resized_confidence = np.asarray(
        [[0.5, 0.67857146, 0.85714287], [1.4428571, 1.6214286, 1.8000001]],
        dtype=np.float32,
    )
    assert frame.inverse_depth.shape == (2, 3)
    assert frame.inverse_depth.dtype == np.float32
    assert frame.model_confidence.shape == (2, 3)
    np.testing.assert_allclose(frame.inverse_depth, np.reciprocal(resized_depth), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(frame.model_confidence, resized_confidence, rtol=1e-5, atol=1e-6)
    assert np.isfinite(frame.inverse_depth).all()
    assert (frame.inverse_depth > 0).all()
    assert np.isfinite(frame.model_confidence).all()
    assert (frame.model_confidence >= 0).all()
    assert frame.inverse_depth.flags.writeable is False
    assert frame.model_confidence.flags.writeable is False


def test_da3_multiview_resizes_each_processing_view_back_to_source_pixels(tmp_path: Path):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified da3 weights")
    digest = depth_anything._sha256_path(weights)

    class Prediction:
        depth = np.stack([np.full((4, 6), value, dtype=np.float32) for value in (1.0, 2.0, 3.0)])
        conf = np.stack([np.full((4, 6), value, dtype=np.float32) for value in (0.25, 0.5, 0.75)])

    provider = DepthAnything3Provider(
        weights,
        weights_sha256=digest,
        model_loader=lambda _provider: lambda _images, **_kwargs: Prediction(),
    )
    images = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)]
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    identities = tuple("abcdef"[index] * 64 for index in range(3))

    result = provider.infer_multiview(
        images,
        frame_sha256s=identities,
        intrinsics=intrinsics,
        extrinsics_world_to_camera=extrinsics,
    )

    assert result.depth_m.shape == (3, 2, 3)
    assert result.depth_m.dtype == np.float32
    assert result.model_confidence.shape == (3, 2, 3)
    np.testing.assert_allclose(result.depth_m[:, 0, 0], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(result.model_confidence[:, 0, 0], (0.25, 0.5, 0.75))
    assert np.isfinite(result.depth_m).all()
    assert (result.depth_m > 0).all()
    assert np.isfinite(result.model_confidence).all()
    assert (result.model_confidence >= 0).all()
    assert result.depth_m.flags.writeable is False
    assert result.model_confidence.flags.writeable is False


def test_da3_provider_rejects_invalid_depth_and_confidence(tmp_path: Path):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified da3 weights")
    digest = depth_anything._sha256_path(weights)

    class Prediction:
        depth = np.ones((1, 2, 2), dtype=np.float32)
        conf = np.full((1, 2, 2), np.nan, dtype=np.float32)

    provider = DepthAnything3Provider(
        weights_path=weights,
        weights_sha256=digest,
        model_loader=lambda _provider: lambda _images, **_kwargs: Prediction(),
    )
    with pytest.raises(ValueError, match="confidence"):
        provider.infer(np.zeros((2, 2, 3), dtype=np.uint8), frame_sha256="a" * 64)


def test_da3_multiview_adapter_passes_known_cameras_and_preserves_all_view_identities(tmp_path: Path):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified da3 base weights")
    digest = depth_anything._sha256_path(weights)
    observed = {}

    class Prediction:
        depth = np.stack([np.full((3, 4), 1.0 + index, dtype=np.float32) for index in range(5)])
        conf = np.full((5, 3, 4), 0.8, dtype=np.float32)

    def loader(_provider):
        def infer(images, **kwargs):
            observed["images"] = images
            observed["kwargs"] = kwargs
            return Prediction()

        return infer

    provider = DepthAnything3Provider(
        weights,
        weights_sha256=digest,
        model_name="DA3-BASE",
        source_revision=DA3_SOURCE_REVISION,
        gpu_memory_budget_gib=9.5,
        model_loader=loader,
    )
    images = [np.full((3, 4, 3), index, dtype=np.uint8) for index in range(5)]
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 5, axis=0)
    intrinsics[:, 0, 0] = 300.0
    intrinsics[:, 1, 1] = 300.0
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 5, axis=0)
    extrinsics[:, 0, 3] = np.linspace(0.0, 0.4, 5)
    identities = tuple(hashlib.sha256(f"frame-{index}".encode()).hexdigest() for index in range(5))

    result = provider.infer_multiview(
        images,
        frame_sha256s=identities,
        intrinsics=intrinsics,
        extrinsics_world_to_camera=extrinsics,
    )

    assert result.frame_sha256s == identities
    assert result.depth_m.shape == (5, 3, 4)
    assert result.depth_m.flags.writeable is False
    assert result.model_confidence.flags.writeable is False
    assert observed["kwargs"]["process_res"] == 504
    assert observed["kwargs"]["ref_view_strategy"] == "middle"
    assert observed["kwargs"]["align_to_input_ext_scale"] is True
    assert np.array_equal(observed["kwargs"]["intrinsics"], intrinsics)
    assert np.array_equal(observed["kwargs"]["extrinsics"], extrinsics)
    assert provider.source_revision == DA3_SOURCE_REVISION
    assert provider.gpu_memory_budget_bytes == int(9.5 * 1024**3)


def test_da3_provider_fails_closed_for_unpinned_source_or_non_apache_base_model(tmp_path: Path):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified da3 weights")
    digest = depth_anything._sha256_path(weights)

    with pytest.raises(ValueError, match="source revision"):
        DepthAnything3Provider(weights, weights_sha256=digest, source_revision="0" * 40)
    with pytest.raises(ValueError, match="DA3-BASE"):
        DepthAnything3Provider(weights, weights_sha256=digest, model_name="DA3-LARGE")


@pytest.mark.parametrize("precision", ("float16", "bfloat16"))
def test_da3_low_precision_keeps_official_model_weights_in_their_expected_dtype(
    monkeypatch,
    tmp_path: Path,
    precision: str,
):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified da3 weights")
    digest = depth_anything._sha256_path(weights)
    to_calls = []

    class Model:
        def to(self, **kwargs):
            to_calls.append(kwargs)
            return self

        def eval(self):
            return self

    model = Model()

    class OfficialApi:
        @staticmethod
        def from_pretrained(path):
            assert path == str(weights)
            return model

    package = ModuleType("depth_anything_3")
    package.__path__ = []
    api = ModuleType("depth_anything_3.api")
    api.DepthAnything3 = OfficialApi
    monkeypatch.setitem(sys.modules, "depth_anything_3", package)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    provider = DepthAnything3Provider(
        weights,
        weights_sha256=digest,
        device="cuda",
        precision=precision,
    )

    provider._load()

    assert to_calls == [{"device": "cuda"}]


def test_da3_single_and_multiview_inference_share_one_model_lock(tmp_path: Path):
    weights = tmp_path / "da3-local"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"verified da3 weights")
    digest = depth_anything._sha256_path(weights)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    guard = threading.Lock()
    active_calls = 0
    peak_active_calls = 0
    call_count = 0

    class Prediction:
        def __init__(self, view_count):
            self.depth = np.ones((view_count, 2, 3), dtype=np.float32)
            self.conf = np.ones((view_count, 2, 3), dtype=np.float32)

    def loader(_provider):
        def infer(images, **_kwargs):
            nonlocal active_calls, peak_active_calls, call_count
            with guard:
                call_count += 1
                current_call = call_count
                active_calls += 1
                peak_active_calls = max(peak_active_calls, active_calls)
                (first_entered if current_call == 1 else second_entered).set()
            try:
                if current_call == 1:
                    assert release_first.wait(timeout=2.0)
                return Prediction(len(images))
            finally:
                with guard:
                    active_calls -= 1

        return infer

    provider = DepthAnything3Provider(
        weights,
        weights_sha256=digest,
        model_loader=loader,
    )
    errors = []
    results = []

    def infer_single():
        try:
            results.append(
                provider.infer(
                    np.zeros((2, 3, 3), dtype=np.uint8),
                    frame_sha256="d" * 64,
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def infer_multiview():
        try:
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0)
            extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
            results.append(
                provider.infer_multiview(
                    [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)],
                    frame_sha256s=("a" * 64, "b" * 64, "c" * 64),
                    intrinsics=intrinsics,
                    extrinsics_world_to_camera=extrinsics,
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    single_thread = threading.Thread(target=infer_single)
    multiview_thread = threading.Thread(target=infer_multiview)
    single_thread.start()
    assert first_entered.wait(timeout=1.0)
    multiview_thread.start()
    second_entered_before_release = second_entered.wait(timeout=0.2)
    release_first.set()
    single_thread.join(timeout=2.0)
    multiview_thread.join(timeout=2.0)

    assert not single_thread.is_alive()
    assert not multiview_thread.is_alive()
    assert errors == []
    assert len(results) == 2
    assert second_entered_before_release is False
    assert peak_active_calls == 1
