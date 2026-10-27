"""Fail-closed progressive gsplat/Nerfstudio training job contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


class NeuralTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class NeuralCheckpoint:
    step: int
    path: Path
    sha256: str
    dataset_manifest_sha256: str
    calibration_id: str
    pose_generation: str


class NeuralTrainingJob:
    """Tracks immutable presentation-only checkpoints; never owns geometry."""

    def __init__(self, root: str | Path, *, backend: str = "gsplat") -> None:
        if backend not in {"gsplat", "nerfstudio"}:
            raise ValueError("backend must be gsplat or nerfstudio")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self._dataset_manifest: Path | None = None
        self._dataset_sha256: str | None = None
        self._calibration_id: str | None = None
        self._pose_generation: str | None = None
        self._checkpoints: list[NeuralCheckpoint] = []
        self._blockers: list[str] = []

    def prepare(
        self,
        dataset_manifest: str | Path,
        quality: Mapping[str, Any],
        *,
        calibration_id: str,
        pose_generation: str,
    ) -> dict[str, Any]:
        path = Path(dataset_manifest)
        blockers: list[str] = []
        if not path.is_file():
            blockers.append("NEURAL_DATASET_MANIFEST_MISSING")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
            blockers.append("NEURAL_DATASET_MANIFEST_INVALID")
        if isinstance(payload, dict):
            if payload.get("dataset_type") != "CALIBRATED_PRESENTATION_REFINEMENT":
                blockers.append("NEURAL_DATASET_TYPE_UNSUPPORTED")
            if payload.get("geometry_authority") is not False:
                blockers.append("NEURAL_GEOMETRY_AUTHORITY_MUST_REMAIN_FALSE")
            if payload.get("measured_seed_authority") != "LIDAR_MEASURED":
                blockers.append("LIDAR_MEASURED_SEED_REQUIRED")
            if payload.get("calibration_id") != calibration_id:
                blockers.append("NEURAL_DATASET_CALIBRATION_ID_MISMATCH")
            if not isinstance(payload.get("frames"), list) or not payload["frames"]:
                blockers.append("NEURAL_DATASET_HAS_NO_ELIGIBLE_FRAMES")
            for frame in payload.get("frames", ()):
                if not isinstance(frame, dict) or not self._sha256(frame.get("source_artifact_sha256")):
                    blockers.append("NEURAL_FRAME_SOURCE_ARTIFACT_HASH_MISSING")
                    break
        eligible = quality.get("eligible_for", {}) if isinstance(quality, Mapping) else {}
        if not eligible.get("neural_geometry") or not eligible.get("texture"):
            blockers.append("CALIBRATED_APPEARANCE_AUTHORIZATION_REQUIRED")
        if not calibration_id or not pose_generation:
            blockers.append("CALIBRATION_AND_POSE_GENERATION_REQUIRED")
        self._blockers = list(dict.fromkeys(blockers))
        if self._blockers:
            return self.snapshot()
        digest = self._file_sha256(path)
        self._dataset_manifest = path.resolve()
        self._dataset_sha256 = digest
        self._calibration_id = calibration_id
        self._pose_generation = pose_generation
        return self.snapshot()

    def command(self, *, max_steps: int = 30_000) -> list[str]:
        if self._blockers or self._dataset_manifest is None:
            raise NeuralTrainingError("neural job is not prepared")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.backend == "gsplat":
            return ["python", "-m", "gsplat", "--data", str(self._dataset_manifest), "--max-steps", str(max_steps)]
        return ["ns-train", "nerfacto", "--data", str(self._dataset_manifest), "--max-num-iterations", str(max_steps)]

    def record_checkpoint(self, step: int, checkpoint: str | Path) -> NeuralCheckpoint:
        if self._blockers or self._dataset_manifest is None or self._dataset_sha256 is None:
            raise NeuralTrainingError("neural job is not prepared")
        if step <= 0:
            raise ValueError("checkpoint step must be positive")
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = self._file_sha256(path)
        record = NeuralCheckpoint(
            step=step,
            path=path.resolve(),
            sha256=digest,
            dataset_manifest_sha256=self._dataset_sha256,
            calibration_id=str(self._calibration_id),
            pose_generation=str(self._pose_generation),
        )
        self._checkpoints = [item for item in self._checkpoints if item.step != step]
        self._checkpoints.append(record)
        self._checkpoints.sort(key=lambda item: item.step)
        metadata = {
            "schema_version": 1,
            "backend": self.backend,
            "step": step,
            "checkpoint": str(record.path),
            "checkpoint_sha256": digest,
            "dataset_manifest_sha256": record.dataset_manifest_sha256,
            "calibration_id": record.calibration_id,
            "pose_generation": record.pose_generation,
            "geometry_authority": False,
        }
        target = self.root / f"checkpoint-{step:08d}.json"
        descriptor_fd, descriptor_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=self.root)
        os.close(descriptor_fd)
        temporary = Path(descriptor_name)
        try:
            temporary.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return record

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": "READY" if self._dataset_manifest is not None and not self._blockers else "BLOCKED",
            "backend": self.backend,
            "dataset_manifest": None if self._dataset_manifest is None else str(self._dataset_manifest),
            "dataset_manifest_sha256": self._dataset_sha256,
            "calibration_id": self._calibration_id,
            "pose_generation": self._pose_generation,
            "geometry_authority": False,
            "checkpoints": [
                {"step": item.step, "path": str(item.path), "sha256": item.sha256}
                for item in self._checkpoints
            ],
            "blockers": list(self._blockers),
        }

    @staticmethod
    def _sha256(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
