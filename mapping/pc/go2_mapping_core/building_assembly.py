"""Immutable assembly generations for observed building surfaces."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import uuid
from typing import Any, Optional, Sequence

import numpy as np

from mapping.pc.go2_mapping_live.building_project import BuildingProject


_IDENTIFIER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or any(char not in _IDENTIFIER_CHARS for char in value):
        raise ValueError(f"{name} is invalid")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _rigid_transform(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != 16:
        raise ValueError("transform_building_from_submap must contain 16 values")
    try:
        matrix = np.asarray(tuple(float(value) for value in values), dtype=np.float64).reshape((4, 4))
    except (TypeError, ValueError) as exc:
        raise ValueError("transform_building_from_submap must be a finite rigid transform") from exc
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("transform_building_from_submap must be a finite rigid transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise ValueError("transform_building_from_submap must be rigid")
    return tuple(float(value) for value in matrix.reshape(-1))


def _is_identity(values: Sequence[float]) -> bool:
    return bool(np.allclose(np.asarray(values, dtype=np.float64), np.asarray(_IDENTITY), atol=1e-8))


@dataclass(frozen=True)
class AssemblyMember:
    building_id: str
    component_id: str
    session_id: str
    submap_id: str
    generation_id: str
    transform_building_from_submap: tuple[float, ...]
    source_decision_id: Optional[str] = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.building_id, "building_id"),
            (self.component_id, "component_id"),
            (self.session_id, "session_id"),
            (self.submap_id, "submap_id"),
        ):
            _identifier(value, name)
        _sha256(self.generation_id, "generation_id")
        object.__setattr__(
            self,
            "transform_building_from_submap",
            _rigid_transform(self.transform_building_from_submap),
        )
        if self.source_decision_id is not None:
            _sha256(self.source_decision_id, "source_decision_id")

    def payload(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "component_id": self.component_id,
            "session_id": self.session_id,
            "submap_id": self.submap_id,
            "generation_id": self.generation_id,
            "transform_building_from_submap": list(self.transform_building_from_submap),
            "source_decision_id": self.source_decision_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AssemblyMember":
        return cls(
            building_id=payload["building_id"],
            component_id=payload["component_id"],
            session_id=payload["session_id"],
            submap_id=payload["submap_id"],
            generation_id=payload["generation_id"],
            transform_building_from_submap=tuple(payload["transform_building_from_submap"]),
            source_decision_id=payload.get("source_decision_id"),
        )


@dataclass(frozen=True)
class BuildingAssembly:
    assembly_id: str
    building_id: str
    algorithm_version: str
    members: tuple[AssemblyMember, ...]
    path: Path


class BuildingAssemblyStore:
    """Publishes immutable placements without changing measured geometry."""

    def __init__(self, root: str | Path, *, project: BuildingProject, relocalization=None) -> None:
        self.root = Path(root)
        self.assemblies_root = self.root / "assemblies"
        self.current_path = self.root / "CURRENT"
        self.assemblies_root.mkdir(parents=True, exist_ok=True)
        self.project = project
        self.relocalization = relocalization
        self._lock = threading.RLock()

    def publish(self, members: Sequence[AssemblyMember], *, algorithm_version: str) -> BuildingAssembly:
        algorithm_version = _identifier(algorithm_version, "algorithm_version")
        with self._lock:
            normalized, core = self._validated_core(tuple(members), algorithm_version)
            assembly_id = hashlib.sha256(_canonical(core).encode("utf-8")).hexdigest()
            manifest = {**core, "assembly_id": assembly_id}
            destination = self.assemblies_root / assembly_id
            if destination.exists():
                existing = self.load(assembly_id)
                if existing.members != normalized or existing.algorithm_version != algorithm_version:
                    raise ValueError("immutable assembly identity conflicts with existing content")
            else:
                temporary = self.assemblies_root / f".partial-{uuid.uuid4().hex}"
                temporary.mkdir()
                try:
                    self._write_json(temporary / "manifest.json", manifest)
                    os.replace(temporary, destination)
                finally:
                    if temporary.exists():
                        for child in temporary.iterdir():
                            child.unlink()
                        temporary.rmdir()
            self._write_text_atomic(self.current_path, assembly_id + "\n")
            return BuildingAssembly(assembly_id, self.project.building_id, algorithm_version, normalized, destination)

    def load(self, assembly_id: str) -> BuildingAssembly:
        assembly_id = _sha256(assembly_id, "assembly_id")
        with self._lock:
            path = self.assemblies_root / assembly_id
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("assembly manifest is unreadable") from exc
            if not isinstance(manifest, dict) or manifest.get("assembly_id") != assembly_id:
                raise ValueError("assembly manifest identity is invalid")
            core = dict(manifest)
            core.pop("assembly_id", None)
            if hashlib.sha256(_canonical(core).encode("utf-8")).hexdigest() != assembly_id:
                raise ValueError("assembly manifest hash is invalid")
            try:
                algorithm_version = manifest["algorithm_version"]
                members = tuple(AssemblyMember.from_payload(payload) for payload in manifest["members"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("assembly manifest members are invalid") from exc
            normalized, expected_core = self._validated_core(members, algorithm_version)
            if expected_core != core or normalized != members:
                raise ValueError("assembly manifest provenance is invalid")
            return BuildingAssembly(assembly_id, self.project.building_id, algorithm_version, members, path)

    def current(self) -> Optional[BuildingAssembly]:
        with self._lock:
            if not self.current_path.exists():
                return None
            try:
                assembly_id = self.current_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                raise ValueError("CURRENT pointer is unreadable") from exc
            if len(assembly_id) != 64 or any(char not in "0123456789abcdef" for char in assembly_id):
                raise ValueError("CURRENT pointer is invalid")
            return self.load(assembly_id)

    def _validated_core(
        self,
        members: tuple[AssemblyMember, ...],
        algorithm_version: str,
    ) -> tuple[tuple[AssemblyMember, ...], dict[str, Any]]:
        _identifier(algorithm_version, "algorithm_version")
        if not members:
            raise ValueError("building assembly requires at least one member")
        seen_components = set()
        seen_sessions = set()
        seen_submaps = set()
        validated: list[tuple[AssemblyMember, dict[str, Any], dict[str, Any]]] = []
        root_component_id = self.project.snapshot().get("root_component_id")
        for member in members:
            if member.building_id != self.project.building_id:
                raise ValueError("assembly member building provenance does not match project")
            if member.component_id in seen_components or member.session_id in seen_sessions or member.submap_id in seen_submaps:
                raise ValueError("assembly member identity is duplicated")
            seen_components.add(member.component_id)
            seen_sessions.add(member.session_id)
            seen_submaps.add(member.submap_id)
            try:
                sealed = self.project.sealed_generation(member.submap_id, member.generation_id)
                submap = self.project.submap(member.submap_id)
                component = self.project.component(member.component_id)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("assembly member must reference an exact sealed generation") from exc
            if (
                submap["component_id"] != member.component_id
                or submap["session_id"] != member.session_id
                or sealed["building_id"] != member.building_id
                or sealed["session_id"] != member.session_id
            ):
                raise ValueError("assembly member provenance does not match sealed generation")
            if component["building_id"] != self.project.building_id or component["session_id"] != member.session_id:
                raise ValueError("assembly member provenance does not match component")
            if component["component_id"] == root_component_id:
                if member.source_decision_id is not None or not _is_identity(member.transform_building_from_submap):
                    raise ValueError("ROOT assembly member must use identity placement without a source decision")
            elif member.source_decision_id is None:
                raise ValueError("non-ROOT assembly member requires authoritative relocalization")
            validated.append((member, sealed, component))

        roots = [entry for entry in validated if entry[2]["component_id"] == root_component_id]
        if len(roots) != 1:
            raise ValueError("assembly requires exactly one ROOT member")
        if roots[0][0].component_id != root_component_id:
            raise ValueError("assembly ROOT component is invalid")

        by_generation = {(member.submap_id, member.generation_id): member for member, _, _ in validated}
        for member, _, component in validated:
            if component["component_id"] == root_component_id:
                continue
            if self.relocalization is None:
                raise ValueError("authoritative relocalization store is required for attached members")
            decision = self.relocalization.authoritative_decision(self.project.building_id, member.session_id)
            if decision is None or decision.decision_id != member.source_decision_id:
                raise ValueError("assembly member lacks the authoritative relocalization decision")
            target_key = (decision.accepted_target_submap_id, decision.accepted_target_generation_id)
            if target_key not in by_generation or target_key == (member.submap_id, member.generation_id):
                raise ValueError("authoritative relocalization target is not in this assembly")

        normalized = tuple(
            member
            for member, _, _ in sorted(
                validated,
                key=lambda entry: (0 if entry[2]["component_id"] == root_component_id else 1, entry[0].component_id),
            )
        )
        member_payloads = []
        for member in normalized:
            entry = next(item for item in validated if item[0] == member)
            sealed = entry[1]
            component = entry[2]
            member_payloads.append({
                **member.payload(),
                "floor_id": sealed["floor_id"],
                "zone_id": sealed["zone_id"],
                "attachment_state": "ROOT" if component["component_id"] == root_component_id else "ATTACHED",
                "geometry_authority": "OBSERVED_SURFACE",
            })
        core = {
            "schema_version": 1,
            "building_id": self.project.building_id,
            "algorithm_version": algorithm_version,
            "geometry_authority": "OBSERVED_SURFACE",
            "volume_filled": False,
            "watertight": False,
            "members": member_payloads,
        }
        return normalized, core

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            output.flush()
            os.fsync(output.fileno())

    @staticmethod
    def _write_text_atomic(path: Path, value: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                output.write(value)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
