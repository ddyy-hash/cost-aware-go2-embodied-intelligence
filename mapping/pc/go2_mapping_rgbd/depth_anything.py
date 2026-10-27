"""Local-only relative inverse-depth inference for bounded RGB-D assistance."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256


_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}", re.ASCII)
DA3_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_MODEL_LICENSE = "Apache-2.0"


class DepthAnythingError(RuntimeError):
    """Raised when local model identity, loading, or inference fails closed."""


def _sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise DepthAnythingError(f"local Depth Anything weights do not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise DepthAnythingError(f"local Depth Anything weights directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        payload = item.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class RelativeDepthFrame:
    inverse_depth: np.ndarray
    model_revision_id: str
    preprocessing_revision: str
    frame_sha256: str
    model_confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.inverse_depth, dtype=np.float32)
        if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("inverse_depth must be a non-empty finite positive H by W array")
        values = np.array(values, dtype=np.float32, order="C", copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "inverse_depth", values)
        object.__setattr__(self, "model_revision_id", _sha256(self.model_revision_id, "model_revision_id"))
        if not isinstance(self.preprocessing_revision, str) or not self.preprocessing_revision:
            raise ValueError("preprocessing_revision must be non-empty")
        object.__setattr__(self, "frame_sha256", _sha256(self.frame_sha256, "frame_sha256"))
        if self.model_confidence is not None:
            confidence = np.asarray(self.model_confidence, dtype=np.float32)
            if confidence.shape != values.shape or not np.all(np.isfinite(confidence)) or np.any(confidence < 0.0):
                raise ValueError("model_confidence must be a finite non-negative array matching inverse_depth")
            confidence = np.array(confidence, dtype=np.float32, order="C", copy=True)
            confidence.setflags(write=False)
            object.__setattr__(self, "model_confidence", confidence)


@dataclass(frozen=True)
class MultiViewDepthFrame:
    depth_m: np.ndarray
    model_revision_id: str
    preprocessing_revision: str
    frame_sha256s: tuple[str, ...]
    intrinsics: np.ndarray
    extrinsics_world_to_camera: np.ndarray
    model_confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m, dtype=np.float32)
        if depth.ndim != 3 or depth.shape[0] < 3 or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError("multi-view depth must be finite, positive, and contain at least three views")
        hashes = tuple(_sha256(value, "frame_sha256") for value in self.frame_sha256s)
        if len(hashes) != len(depth) or len(set(hashes)) != len(hashes):
            raise ValueError("multi-view frame identities must be unique and match depth")
        intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        extrinsics = np.asarray(self.extrinsics_world_to_camera, dtype=np.float32)
        if intrinsics.shape != (len(depth), 3, 3) or extrinsics.shape != (len(depth), 4, 4):
            raise ValueError("multi-view camera arrays do not match depth")
        if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
            raise ValueError("multi-view camera arrays must be finite")
        confidence = None
        if self.model_confidence is not None:
            confidence = np.asarray(self.model_confidence, dtype=np.float32)
            if confidence.shape != depth.shape or not np.isfinite(confidence).all() or np.any(confidence < 0):
                raise ValueError("multi-view confidence must be finite, non-negative, and match depth")
        for name, value in (
            ("depth_m", depth),
            ("intrinsics", intrinsics),
            ("extrinsics_world_to_camera", extrinsics),
        ):
            immutable = np.array(value, copy=True, order="C")
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        if confidence is not None:
            immutable_confidence = np.array(confidence, copy=True, order="C")
            immutable_confidence.setflags(write=False)
            object.__setattr__(self, "model_confidence", immutable_confidence)
        object.__setattr__(self, "frame_sha256s", hashes)
        object.__setattr__(self, "model_revision_id", _sha256(self.model_revision_id, "model_revision_id"))
        if not isinstance(self.preprocessing_revision, str) or not self.preprocessing_revision:
            raise ValueError("preprocessing_revision must be non-empty")


class DepthAnythingProvider(Protocol):
    def infer(self, rectified_rgb: np.ndarray, *, frame_sha256: str) -> RelativeDepthFrame: ...


ModelLoader = Callable[["TransformersDepthAnythingProvider"], Callable[[np.ndarray], np.ndarray]]
DA3ModelLoader = Callable[["DepthAnything3Provider"], Callable[..., Any]]


def _move_model_inputs(inputs: dict, *, device: str, dtype) -> dict:
    return {
        name: tensor.to(device=device, dtype=dtype) if tensor.is_floating_point() else tensor.to(device=device)
        for name, tensor in inputs.items()
    }


def _requires_autocast(device: str, dtype, torch_module) -> bool:
    return device.startswith("cuda") and dtype in {torch_module.float16, torch_module.bfloat16}


def _resize_da3_prediction(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float32)
    if source.shape[1:] == target_shape:
        return source
    try:
        import torch
    except ImportError as exc:
        raise DepthAnythingError("PyTorch is required to resize Depth Anything 3 predictions") from exc
    with torch.inference_mode():
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(source).unsqueeze(1),
            size=target_shape,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(1)
    return np.ascontiguousarray(resized.numpy(), dtype=np.float32)


class TransformersDepthAnythingProvider:
    """Lazily loads verified operator-provided weights with no network fallback."""

    local_files_only = True

    def __init__(
        self,
        weights_path: str | Path,
        *,
        weights_sha256: str,
        model_name: str = "Depth-Anything-V2-Small-hf",
        preprocessing_revision: str = "depth-anything-preprocess-v1",
        input_size: tuple[int, int] = (518, 518),
        device: str = "cpu",
        precision: str = "float32",
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.weights_sha256 = _sha256(weights_sha256, "weights_sha256")
        observed_digest = _sha256_path(self.weights_path)
        if observed_digest != self.weights_sha256:
            raise DepthAnythingError(
                f"Depth Anything weights SHA-256 mismatch: expected {self.weights_sha256}, got {observed_digest}"
            )
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be non-empty")
        if not isinstance(preprocessing_revision, str) or not preprocessing_revision:
            raise ValueError("preprocessing_revision must be non-empty")
        if (
            not isinstance(input_size, tuple)
            or len(input_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in input_size)
        ):
            raise ValueError("input_size must be a positive (width, height) tuple")
        if precision not in {"float16", "bfloat16", "float32"}:
            raise ValueError("precision must be float16, bfloat16, or float32")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be non-empty")
        self.model_name = model_name
        self.preprocessing_revision = preprocessing_revision
        self.input_size = input_size
        self.device = device
        self.precision = precision
        self.model_revision_id = canonical_sha256(
            {
                "schema_version": 1,
                "model_name": model_name,
                "weights_sha256": self.weights_sha256,
                "preprocessing_revision": preprocessing_revision,
                "input_size": list(input_size),
                "device": device,
                "precision": precision,
                "local_files_only": True,
            }
        )
        self._model_loader = model_loader or self._load_transformers
        self._inference: Callable[[np.ndarray], np.ndarray] | None = None
        self._load_lock = threading.Lock()

    def _load_transformers(self, _provider: "TransformersDepthAnythingProvider") -> Callable[[np.ndarray], np.ndarray]:
        if not self.weights_path.is_dir():
            raise DepthAnythingError("Transformers loading requires a verified local weights directory")
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise DepthAnythingError("local Depth Anything dependencies are unavailable") from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise DepthAnythingError("configured CUDA device is unavailable")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.precision]
        try:
            processor = AutoImageProcessor.from_pretrained(
                str(self.weights_path),
                local_files_only=True,
                use_fast=False,
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                str(self.weights_path),
                local_files_only=True,
                torch_dtype=dtype,
            ).to(self.device)
            model.eval()
        except Exception as exc:
            raise DepthAnythingError(f"cannot load verified local Depth Anything weights: {exc}") from exc

        def run(image: np.ndarray) -> np.ndarray:
            try:
                inputs = processor(images=image, return_tensors="pt")
                inputs = _move_model_inputs(inputs, device=self.device, dtype=dtype)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=dtype)
                    if _requires_autocast(self.device, dtype, torch)
                    else contextlib.nullcontext()
                )
                with torch.inference_mode(), autocast:
                    prediction = model(**inputs).predicted_depth
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=image.shape[:2],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                return prediction.detach().to(device="cpu", dtype=torch.float32).numpy()
            except Exception as exc:
                raise DepthAnythingError(f"Depth Anything inference failed: {exc}") from exc

        return run

    def _load(self) -> Callable[[np.ndarray], np.ndarray]:
        if self._inference is None:
            with self._load_lock:
                if self._inference is None:
                    inference = self._model_loader(self)
                    if not callable(inference):
                        raise DepthAnythingError("model loader did not return an inference callable")
                    self._inference = inference
        return self._inference

    def infer(self, rectified_rgb: np.ndarray, *, frame_sha256: str) -> RelativeDepthFrame:
        frame_digest = _sha256(frame_sha256, "frame_sha256")
        image = np.asarray(rectified_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0 or image.dtype != np.uint8:
            raise ValueError("rectified_rgb must be a non-empty uint8 H by W by 3 RGB array")
        image = np.array(image, dtype=np.uint8, order="C", copy=True)
        try:
            inverse_depth = self._load()(image)
        except DepthAnythingError:
            raise
        except Exception as exc:
            raise DepthAnythingError(f"Depth Anything inference failed: {exc}") from exc
        values = np.asarray(inverse_depth, dtype=np.float32)
        if values.shape != image.shape[:2]:
            raise ValueError("Depth Anything output shape does not match the rectified RGB frame")
        return RelativeDepthFrame(
            inverse_depth=values,
            model_revision_id=self.model_revision_id,
            preprocessing_revision=self.preprocessing_revision,
            frame_sha256=frame_digest,
        )


class DepthAnything3Provider:
    """Lazily runs the official DA3 API from a verified local model directory."""

    local_files_only = True

    def __init__(
        self,
        weights_path: str | Path,
        *,
        weights_sha256: str,
        model_name: str = "DA3-BASE",
        preprocessing_revision: str = "depth-anything-3-preprocess-v1",
        input_size: tuple[int, int] = (504, 504),
        device: str = "cpu",
        precision: str = "float32",
        source_revision: str = DA3_SOURCE_REVISION,
        gpu_memory_budget_gib: float = 9.5,
        inference_timeout_s: float = 30.0,
        model_loader: DA3ModelLoader | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_dir():
            raise DepthAnythingError("Depth Anything 3 requires a verified local model directory")
        self.weights_sha256 = _sha256(weights_sha256, "weights_sha256")
        observed_digest = _sha256_path(self.weights_path)
        if observed_digest != self.weights_sha256:
            raise DepthAnythingError(
                f"Depth Anything weights SHA-256 mismatch: expected {self.weights_sha256}, got {observed_digest}"
            )
        if model_name != "DA3-BASE":
            raise ValueError("only Apache-2.0 DA3-BASE is authorized for surface assistance")
        if source_revision != DA3_SOURCE_REVISION or _GIT_COMMIT.fullmatch(source_revision) is None:
            raise ValueError("Depth Anything 3 source revision is not the pinned commit")
        if not isinstance(preprocessing_revision, str) or not preprocessing_revision:
            raise ValueError("preprocessing_revision must be non-empty")
        if (
            not isinstance(input_size, tuple)
            or len(input_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in input_size)
        ):
            raise ValueError("input_size must be a positive (width, height) tuple")
        if precision not in {"float16", "bfloat16", "float32"}:
            raise ValueError("precision must be float16, bfloat16, or float32")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be non-empty")
        if (
            isinstance(gpu_memory_budget_gib, bool)
            or not np.isfinite(gpu_memory_budget_gib)
            or not 0 < float(gpu_memory_budget_gib) <= 9.5
        ):
            raise ValueError("gpu_memory_budget_gib must be finite and in (0, 9.5]")
        if isinstance(inference_timeout_s, bool) or not np.isfinite(inference_timeout_s) or inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be finite and positive")
        self.model_name = model_name
        self.preprocessing_revision = preprocessing_revision
        self.input_size = input_size
        self.device = device
        self.precision = precision
        self.source_revision = source_revision
        self.model_license = DA3_MODEL_LICENSE
        self.gpu_memory_budget_bytes = int(float(gpu_memory_budget_gib) * 1024**3)
        self.inference_timeout_s = float(inference_timeout_s)
        self.model_revision_id = canonical_sha256(
            {
                "schema_version": 2,
                "backend": "depth-anything-3-official-api",
                "model_name": model_name,
                "model_license": self.model_license,
                "source_revision": self.source_revision,
                "weights_sha256": self.weights_sha256,
                "preprocessing_revision": preprocessing_revision,
                "input_size": list(input_size),
                "process_res_method": "upper_bound_resize",
                "ref_view_strategy": "middle",
                "device": device,
                "precision": precision,
                "gpu_memory_budget_bytes": self.gpu_memory_budget_bytes,
                "inference_timeout_s": self.inference_timeout_s,
                "local_files_only": True,
            }
        )
        self._model_loader = model_loader or self._load_da3
        self._inference: Callable[..., Any] | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load_da3(self, _provider: "DepthAnything3Provider") -> Callable[..., Any]:
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise DepthAnythingError("local Depth Anything 3 dependencies are unavailable") from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise DepthAnythingError("configured CUDA device is unavailable")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.precision]
        try:
            model = DepthAnything3.from_pretrained(str(self.weights_path))
            model = model.to(device=self.device)
            model.eval()
        except Exception as exc:
            raise DepthAnythingError(f"cannot load verified local Depth Anything 3 weights: {exc}") from exc

        def run(images: list[np.ndarray], **kwargs: Any) -> Any:
            autocast = (
                torch.autocast(device_type="cuda", dtype=dtype)
                if _requires_autocast(self.device, dtype, torch)
                else contextlib.nullcontext()
            )
            try:
                if self.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(self.device)
                started = time.perf_counter()
                with torch.inference_mode(), autocast:
                    prediction = model.inference(image=images, **kwargs)
                elapsed = time.perf_counter() - started
                if elapsed > self.inference_timeout_s:
                    raise DepthAnythingError(
                        f"Depth Anything 3 inference exceeded {self.inference_timeout_s:.3f}s"
                    )
                if self.device.startswith("cuda"):
                    peak = int(torch.cuda.max_memory_reserved(self.device))
                    if peak > self.gpu_memory_budget_bytes:
                        raise DepthAnythingError(
                            f"Depth Anything 3 reserved {peak} bytes above the session budget"
                        )
                return prediction
            except Exception as exc:
                raise DepthAnythingError(f"Depth Anything 3 inference failed: {exc}") from exc

        return run

    def _load(self) -> Callable[..., Any]:
        if self._inference is None:
            with self._load_lock:
                if self._inference is None:
                    inference = self._model_loader(self)
                    if not callable(inference):
                        raise DepthAnythingError("model loader did not return an inference callable")
                    self._inference = inference
        return self._inference

    def infer(self, rectified_rgb: np.ndarray, *, frame_sha256: str) -> RelativeDepthFrame:
        frame_digest = _sha256(frame_sha256, "frame_sha256")
        image = np.asarray(rectified_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0 or image.dtype != np.uint8:
            raise ValueError("rectified_rgb must be a non-empty uint8 H by W by 3 RGB array")
        image = np.array(image, dtype=np.uint8, order="C", copy=True)
        try:
            with self._inference_lock:
                prediction = self._load()(
                    [image],
                    process_res=max(self.input_size),
                    process_res_method="upper_bound_resize",
                    ref_view_strategy="middle",
                )
        except DepthAnythingError:
            raise
        except Exception as exc:
            raise DepthAnythingError(f"Depth Anything 3 inference failed: {exc}") from exc

        depth = np.asarray(getattr(prediction, "depth", None), dtype=np.float32)
        if depth.ndim != 3 or depth.shape[0] != 1 or not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
            raise ValueError("Depth Anything 3 depth must be finite, positive, and contain one view")
        confidence_value = getattr(prediction, "conf", None)
        confidence = None
        if confidence_value is not None:
            confidence_batch = np.asarray(confidence_value, dtype=np.float32)
            if (
                confidence_batch.shape != depth.shape
                or not np.all(np.isfinite(confidence_batch))
                or np.any(confidence_batch < 0.0)
            ):
                raise ValueError("Depth Anything 3 confidence must be finite, non-negative, and match depth")
            confidence = _resize_da3_prediction(confidence_batch, image.shape[:2])[0]
        depth = _resize_da3_prediction(depth, image.shape[:2])
        return RelativeDepthFrame(
            inverse_depth=np.reciprocal(depth[0], dtype=np.float32),
            model_revision_id=self.model_revision_id,
            preprocessing_revision=self.preprocessing_revision,
            frame_sha256=frame_digest,
            model_confidence=confidence,
        )

    def infer_multiview(
        self,
        rectified_rgbs: Sequence[np.ndarray],
        *,
        frame_sha256s: Sequence[str],
        intrinsics: np.ndarray,
        extrinsics_world_to_camera: np.ndarray,
    ) -> MultiViewDepthFrame:
        """Run pose-conditioned DA3 on one bounded, identity-complete view window."""
        images = [np.asarray(image) for image in rectified_rgbs]
        hashes = tuple(_sha256(value, "frame_sha256") for value in frame_sha256s)
        if not 3 <= len(images) <= 5 or len(hashes) != len(images) or len(set(hashes)) != len(hashes):
            raise ValueError("DA3 multi-view inference requires three to five unique frames")
        expected_shape = images[0].shape
        if (
            len(expected_shape) != 3
            or expected_shape[2] != 3
            or any(image.shape != expected_shape or image.dtype != np.uint8 for image in images)
        ):
            raise ValueError("DA3 multi-view RGB inputs must be equal-sized uint8 H by W by 3 arrays")
        exact_images = [np.array(image, dtype=np.uint8, order="C", copy=True) for image in images]
        camera_intrinsics = np.asarray(intrinsics, dtype=np.float32)
        camera_extrinsics = np.asarray(extrinsics_world_to_camera, dtype=np.float32)
        count = len(exact_images)
        if camera_intrinsics.shape != (count, 3, 3) or camera_extrinsics.shape != (count, 4, 4):
            raise ValueError("DA3 multi-view camera arrays have invalid shapes")
        if (
            not np.isfinite(camera_intrinsics).all()
            or not np.isfinite(camera_extrinsics).all()
            or np.any(camera_intrinsics[:, 0, 0] <= 0)
            or np.any(camera_intrinsics[:, 1, 1] <= 0)
            or not np.allclose(camera_extrinsics[:, 3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
        ):
            raise ValueError("DA3 multi-view camera arrays must be finite calibrated transforms")
        try:
            with self._inference_lock:
                prediction = self._load()(
                    exact_images,
                    extrinsics=np.array(camera_extrinsics, copy=True),
                    intrinsics=np.array(camera_intrinsics, copy=True),
                    align_to_input_ext_scale=True,
                    process_res=max(self.input_size),
                    process_res_method="upper_bound_resize",
                    ref_view_strategy="middle",
                )
        except DepthAnythingError:
            raise
        except Exception as exc:
            raise DepthAnythingError(f"Depth Anything 3 multi-view inference failed: {exc}") from exc
        depth = np.asarray(getattr(prediction, "depth", None), dtype=np.float32)
        if depth.ndim != 3 or depth.shape[0] != count or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError("Depth Anything 3 multi-view depth must contain every finite positive view")
        confidence_value = getattr(prediction, "conf", None)
        confidence = None if confidence_value is None else np.asarray(confidence_value, dtype=np.float32)
        if confidence is not None and (
            confidence.shape != depth.shape or not np.isfinite(confidence).all() or np.any(confidence < 0)
        ):
            raise ValueError("Depth Anything 3 multi-view confidence must be finite and match depth")
        depth = _resize_da3_prediction(depth, expected_shape[:2])
        if confidence is not None:
            confidence = _resize_da3_prediction(confidence, expected_shape[:2])
        return MultiViewDepthFrame(
            depth_m=depth,
            model_revision_id=self.model_revision_id,
            preprocessing_revision=self.preprocessing_revision,
            frame_sha256s=hashes,
            intrinsics=camera_intrinsics,
            extrinsics_world_to_camera=camera_extrinsics,
            model_confidence=confidence,
        )
