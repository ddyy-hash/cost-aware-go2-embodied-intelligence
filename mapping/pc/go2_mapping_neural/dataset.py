"""Backend-neutral calibrated dataset export for gsplat and Nerfstudio."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from .gaussian_submap import GaussianBudget


@dataclass(frozen=True)
class NeuralFrame:
    frame_id: str
    image_bgr: np.ndarray
    camera_from_map: np.ndarray
    camera_matrix: np.ndarray
    calibration_id: str
    camera_info_sha256: str
    source_artifact_sha256: str
    rectified: bool
    depth_validation_accepted: bool
    depth_weight: float
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class NeuralDataset:
    path: Path
    manifest_path: Path
    accepted_frame_count: int
    rejected_frame_count: int
    geometry_authority: bool
    adapters: tuple[str, ...]


class NeuralDatasetExporter:
    def __init__(self, root: str | Path, gaussian_budget: GaussianBudget | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.gaussian_budget = gaussian_budget

    def export(
        self,
        frames: list[NeuralFrame] | tuple[NeuralFrame, ...],
        measured_seed_ply: str | Path,
        quality: dict,
        *,
        calibration_id: str,
        derived_cache_key: str,
        requested_gaussians: int = 0,
    ) -> NeuralDataset:
        if len(derived_cache_key) != 64:
            raise ValueError("derived_cache_key must be a SHA-256 identity")
        eligible = quality.get("eligible_for", {})
        if not eligible.get("neural_geometry") or not eligible.get("texture"):
            raise PermissionError("neural export requires calibrated appearance authorization")
        seed = Path(measured_seed_ply)
        if not seed.is_file():
            raise FileNotFoundError(seed)
        if requested_gaussians < 0:
            raise ValueError("requested_gaussians cannot be negative")
        if self.gaussian_budget is not None and (
            requested_gaussians * self.gaussian_budget.bytes_per_gaussian > self.gaussian_budget.max_vram_bytes
        ):
            raise MemoryError("requested 3DGS job exceeds the configured VRAM budget")

        accepted = [frame for frame in frames if self._accepted(frame, calibration_id)]
        rejected_count = len(frames) - len(accepted)
        if not accepted:
            raise PermissionError("no calibrated depth-validated frames are eligible for neural export")
        final = self.root / f"neural-{derived_cache_key[:16]}"
        if final.exists():
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            return NeuralDataset(final, final / "manifest.json", len(manifest["frames"]), rejected_count, False, tuple(manifest["adapters"]))

        temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=self.root))
        try:
            images = temporary / "images"
            masks = temporary / "masks"
            images.mkdir()
            masks.mkdir()
            records = []
            for index, frame in enumerate(accepted):
                image_name = f"{index:06d}.png"
                if not cv2.imwrite(str(images / image_name), np.asarray(frame.image_bgr)):
                    raise OSError(f"failed to encode {image_name}")
                mask_name = None
                if frame.mask is not None:
                    mask_name = image_name
                    if not cv2.imwrite(str(masks / mask_name), np.asarray(frame.mask, dtype=np.uint8)):
                        raise OSError(f"failed to encode mask {mask_name}")
                records.append(
                    {
                        "frame_id": frame.frame_id,
                        "image": f"images/{image_name}",
                        "mask": None if mask_name is None else f"masks/{mask_name}",
                        "camera_from_map": np.asarray(frame.camera_from_map, dtype=np.float64).reshape(-1).tolist(),
                        "camera_matrix": np.asarray(frame.camera_matrix, dtype=np.float64).reshape(-1).tolist(),
                        "depth_weight": float(frame.depth_weight),
                        "camera_info_sha256": frame.camera_info_sha256.lower(),
                        "source_artifact_sha256": frame.source_artifact_sha256.lower(),
                    }
                )
            seed_target = temporary / "lidar_measured_seed.ply"
            shutil.copyfile(seed, seed_target)
            seed_hash = hashlib.sha256(seed_target.read_bytes()).hexdigest()
            manifest = {
                "schema_version": 1,
                "dataset_type": "CALIBRATED_PRESENTATION_REFINEMENT",
                "geometry_authority": False,
                "measured_seed_authority": "LIDAR_MEASURED",
                "measured_seed_ply": seed_target.name,
                "measured_seed_sha256": seed_hash,
                "calibration_id": calibration_id,
                "derived_cache_key": derived_cache_key,
                "adapters": ["gsplat", "nerfstudio"],
                "requested_gaussians": requested_gaussians,
                "frames": records,
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, final)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return NeuralDataset(final, final / "manifest.json", len(accepted), rejected_count, False, ("gsplat", "nerfstudio"))

    @staticmethod
    def _accepted(frame: NeuralFrame, calibration_id: str) -> bool:
        image = np.asarray(frame.image_bgr)
        pose = np.asarray(frame.camera_from_map, dtype=np.float64)
        camera = np.asarray(frame.camera_matrix, dtype=np.float64)
        return bool(
            frame.frame_id
            and frame.calibration_id == calibration_id
            and frame.rectified
            and frame.depth_validation_accepted
            and image.ndim == 3
            and image.shape[2] == 3
            and image.dtype == np.uint8
            and pose.shape == (4, 4)
            and np.all(np.isfinite(pose))
            and np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
            and camera.shape == (3, 3)
            and np.all(np.isfinite(camera))
            and camera[0, 0] > 0.0
            and camera[1, 1] > 0.0
            and len(frame.camera_info_sha256) == 64
            and len(frame.source_artifact_sha256) == 64
            and math.isfinite(frame.depth_weight)
            and frame.depth_weight > 0.0
        )
