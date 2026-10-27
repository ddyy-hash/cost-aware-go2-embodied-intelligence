import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mapping.pc.go2_mapping_dashboard import server


H = tuple(char * 64 for char in "abcdef")


def _result(root: Path, name: str):
    glb = root / name
    provenance = glb.with_suffix(".glb.provenance.json")
    glb.parent.mkdir(parents=True, exist_ok=True)
    glb.write_bytes(b"glTF")
    provenance.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        glb_path=glb, provenance_path=provenance, provenance_sha256=H[5],
        assembly_id=None, building_appearance_revision_id=None,
    )


def test_textured_tile_endpoint_passes_only_exact_identities_and_returns_paired_artifacts(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(server, "_textured_export_root", tmp_path)
    monkeypatch.setattr(server, "_appearance_export_sources", lambda: "SOURCES")

    def exporter(**kwargs):
        calls.append(kwargs)
        return _result(tmp_path, Path(kwargs["output_path"]).name)

    monkeypatch.setattr(server, "export_local_tile_glb", exporter)
    response = server.export_textured_tile({
        "tile_revision_id": H[0],
        "appearance_revision_id": H[1],
        "lod_id": H[2],
        "rgbd_augmented_revision_id": None,
    })

    assert calls[0]["sources"] == "SOURCES"
    assert calls[0]["tile_revision_id"] == H[0]
    assert calls[0]["appearance_revision_id"] == H[1]
    assert calls[0]["lod_id"] == H[2]
    assert Path(calls[0]["output_path"]).parent == tmp_path
    assert response["artifact_kind"] == "TEXTURED_TILE_GLB"
    assert response["glb_url"].endswith(response["glb_name"])
    assert response["provenance_url"].endswith(response["provenance_name"])
    assert response["provenance_sha256"] == H[5]


def test_textured_building_endpoint_requires_complete_explicit_lod_map(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_textured_export_root", tmp_path)
    monkeypatch.setattr(server, "_appearance_export_sources", lambda: "SOURCES")

    def exporter(**kwargs):
        calls.append(kwargs)
        result = _result(tmp_path, Path(kwargs["output_path"]).name)
        return SimpleNamespace(**{**result.__dict__, "assembly_id": H[0], "building_appearance_revision_id": H[1]})

    monkeypatch.setattr(server, "export_building_glb", exporter)
    response = server.export_textured_building({
        "assembly_id": H[0],
        "building_appearance_revision_id": H[1],
        "lod_by_member": {H[2]: H[3], H[4]: H[5]},
    })

    assert calls[0]["lod_by_member"] == {H[2]: H[3], H[4]: H[5]}
    assert response["artifact_kind"] == "TEXTURED_BUILDING_GLB"
    assert response["assembly_id"] == H[0]
    assert response["building_appearance_revision_id"] == H[1]


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: server.export_textured_tile({"tile_revision_id": "BAD", "appearance_revision_id": H[1], "lod_id": H[2]}),
        lambda: server.export_textured_building({"assembly_id": H[0], "building_appearance_revision_id": H[1], "lod_by_member": {}}),
        lambda: server.export_textured_building({"assembly_id": H[0], "building_appearance_revision_id": H[1], "lod_by_member": {"../escape": H[2]}}),
    ],
)
def test_textured_export_payloads_fail_closed(invoke):
    with pytest.raises(HTTPException) as error:
        invoke()
    assert error.value.status_code == 422


def test_exporter_failures_do_not_claim_an_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_textured_export_root", tmp_path)
    monkeypatch.setattr(server, "_appearance_export_sources", lambda: "SOURCES")
    monkeypatch.setattr(server, "export_local_tile_glb", lambda **_kwargs: (_ for _ in ()).throw(ValueError("unauthorized appearance")))
    with pytest.raises(HTTPException) as error:
        server.export_textured_tile({"tile_revision_id": H[0], "appearance_revision_id": H[1], "lod_id": H[2]})
    assert error.value.status_code == 409
    assert not list(tmp_path.iterdir())


def test_textured_artifact_resolver_is_bounded_to_owned_root(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_textured_export_root", tmp_path)
    path = tmp_path / f"tile-{H[0]}-{H[1]}-{H[2]}.glb"
    path.write_bytes(b"glTF")
    response = server.reconstruction_artifact(path.name)
    assert Path(response.path) == path
    with pytest.raises(HTTPException):
        server.reconstruction_artifact("../outside.glb")


def test_operator_tool_has_unambiguous_measured_tile_and_building_parameter_sets():
    tool = (Path(__file__).resolve().parents[2] / "tools" / "export_go2_reconstruction.ps1").read_text(encoding="utf-8")
    for name in ("Measured", "TexturedTile", "TexturedBuilding"):
        assert f"ParameterSetName = '{name}'" in tool
    assert "/api/reconstruction/export-textured-tile" in tool
    assert "/api/reconstruction/export-textured-building" in tool
    assert "Invoke-RestMethod" in tool
    assert "ssh" not in tool.lower()
