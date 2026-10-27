from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mapping.pc.go2_mapping_appearance import derived_provisional_workflow
from mapping.pc.go2_mapping_appearance.derived_provisional_preview import (
    ProvisionalProjectionEvidence,
    load_provisional_surface_preview,
)
from mapping.pc.go2_mapping_live.sensor_frame import PROVISIONAL_GO2_UTLIDAR_PITCH180_V2


SESSION_ID = "segment-20260812T083241Z"
SURFACE_SHA256 = "a" * 64
GEOMETRY_REVISION_ID = "b" * 64
OVERRIDE_SHA256 = "c" * 64
ARTIFACT_SHA256 = "d" * 64
CAMERA_INFO_SHA256 = "e" * 64
COLOR_FRAME_SHA256 = "f" * 64
ROOT = Path(__file__).resolve().parents[2]


def _evidence() -> ProvisionalProjectionEvidence:
    return ProvisionalProjectionEvidence(
        session_id=SESSION_ID,
        source_surface_sha256=SURFACE_SHA256,
        measured_surface_geometry_revision_id=GEOMETRY_REVISION_ID,
        sensor_frame_transform_id=PROVISIONAL_GO2_UTLIDAR_PITCH180_V2,
        voxel_size_m=0.02,
        source_artifact_sha256=(ARTIFACT_SHA256,),
        source_vertex_ids=("0:0:-1",),
        rgb=np.asarray([[12, 34, 56]], dtype=np.uint8),
        validity=np.asarray([True], dtype=np.bool_),
        calibration_id="PROVISIONAL_AUTOCAL_PENDING_v1",
        operator_override_sha256=OVERRIDE_SHA256,
        runtime_projection_authorized=True,
        measurement_authority=False,
        color_camera_info_sha256=CAMERA_INFO_SHA256,
        color_source_frame_sha256=(COLOR_FRAME_SHA256,),
        lidar_frame_count=1,
        color_frame_count=1,
        paired_frame_count=1,
        projected_point_count=1,
        matched_surface_point_count=1,
    )


def test_workflow_publishes_only_to_the_dedicated_preview_root_and_preserves_active_pointer(tmp_path, monkeypatch):
    status_path = tmp_path / "pc-stack" / "automatic-reconstruction-current.json"
    status_path.parent.mkdir()
    pointer_bytes = b'{"active":"immutable"}\n'
    status_path.write_bytes(pointer_bytes)
    output_root = tmp_path / "output" / "reconstruction" / "derived"
    output_root.mkdir(parents=True)
    capture_root = tmp_path / "mapping-data" / "inbox" / "lidar-imu-rgbd"
    (capture_root / SESSION_ID).mkdir(parents=True)
    override_path = tmp_path / "calibration-work" / "formal-operator-runtime-override.json"
    override_path.parent.mkdir()
    override_path.write_text("{}\n", encoding="utf-8")
    preview_root = tmp_path / "output" / "reconstruction" / "derived-provisional-appearance"

    binding = SimpleNamespace(session_id=SESSION_ID)
    projection = object()
    calls = {}

    monkeypatch.setattr(
        derived_provisional_workflow,
        "load_active_derived_surface_binding",
        lambda *, status_path, output_root: calls.setdefault("binding", (Path(status_path), Path(output_root))) and binding,
    )
    monkeypatch.setattr(
        derived_provisional_workflow,
        "load_operator_forced_runtime_projection",
        lambda path: calls.setdefault("override", Path(path)) and projection,
    )

    def build(*, binding, source_root, projection, max_pair_delta_ns, work_root):
        calls["build"] = (binding, Path(source_root), projection, max_pair_delta_ns, work_root)
        return _evidence()

    monkeypatch.setattr(derived_provisional_workflow, "build_provisional_projection_evidence", build)

    result = derived_provisional_workflow.build_active_provisional_surface_preview(
        status_path=status_path,
        output_root=output_root,
        capture_root=capture_root,
        operator_override_path=override_path,
        preview_root=preview_root,
    )

    assert calls["binding"] == (status_path.resolve(), output_root.resolve())
    assert calls["override"] == override_path.resolve()
    assert calls["build"] == (
        binding,
        (capture_root / SESSION_ID).resolve(),
        projection,
        100_000_000,
        None,
    )
    assert status_path.read_bytes() == pointer_bytes
    assert result["pointer_bytes_unchanged"] is True
    assert result["pointer_sha256"] == hashlib.sha256(pointer_bytes).hexdigest()
    assert result["delivery_authorized"] is False
    assert result["measurement_authority"] is False
    assert result["final_store_eligible"] is False
    assert result["preview_root"] == str(preview_root.resolve())

    published = load_provisional_surface_preview(
        result["preview_manifest_path"],
        expected_surface_sha256=SURFACE_SHA256,
        expected_geometry_revision_id=GEOMETRY_REVISION_ID,
    )
    assert published.root_path.is_relative_to(preview_root.resolve())
    assert not any(output_root.iterdir())


def test_workflow_rejects_a_preview_root_inside_the_formal_derived_surface_store(tmp_path):
    status_path = tmp_path / "pc-stack" / "automatic-reconstruction-current.json"
    status_path.parent.mkdir()
    status_path.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "output" / "reconstruction" / "derived"
    output_root.mkdir(parents=True)
    capture_root = tmp_path / "mapping-data" / "inbox" / "lidar-imu-rgbd"
    override_path = tmp_path / "calibration-work" / "formal-operator-runtime-override.json"

    with pytest.raises(ValueError, match="separate from the formal derived surface store"):
        derived_provisional_workflow.build_active_provisional_surface_preview(
            status_path=status_path,
            output_root=output_root,
            capture_root=capture_root,
            operator_override_path=override_path,
            preview_root=output_root / "derived-provisional-appearance",
        )


def test_preview_builder_tool_runs_directly_from_the_workspace_root():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_go2_derived_provisional_preview.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--status-path" in result.stdout
    assert "--capture-root" in result.stdout
    assert "--operator-override-path" in result.stdout
    assert "--preview-root" in result.stdout
