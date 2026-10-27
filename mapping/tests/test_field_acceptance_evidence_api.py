import json
from types import SimpleNamespace

from mapping.pc.go2_mapping_dashboard import server


H = tuple(value * 64 for value in "abcdef12")


def _member(index=0):
    return {
        "component_id": f"component-{index}",
        "session_id": f"session-{index}",
        "submap_id": f"submap-{index}",
        "source_generation_id": H[4 + index],
        "placement_revision_id": H[3 + index],
        "tile_revision_id": H[index],
        "tile_appearance_revision_id": H[1 + index],
        "authorized_lod_ids": [H[6 + index]],
        "delivery_authorization": {
            "measured_surface_coverage_ppm": 850_000 - index,
            "minimum_independent_view_count": 3,
            "delivery_authorized": True,
        },
    }


class _Pipeline:
    def __init__(self, published):
        self.published = published

    def snapshot(self):
        return {"building_id": "building-a", "meshing_error": None}

    def current_building_appearance(self, building_id):
        assert building_id == "building-a"
        return self.published


def _physical_snapshot():
    return {
        "accepted_depth_reports": [
            {
                "tiles": [
                    {
                        "correspondence_count": 120,
                        "median_absolute_residual_m": 0.010,
                        "p95_absolute_residual_m": 0.030,
                        "mean_signed_residual_m": -0.005,
                        "occlusion_conflict_fraction": 0.010,
                        "free_space_violation_fraction": 0.004,
                    },
                    {
                        "correspondence_count": 150,
                        "median_absolute_residual_m": 0.020,
                        "p95_absolute_residual_m": 0.050,
                        "mean_signed_residual_m": 0.012,
                        "occlusion_conflict_fraction": 0.015,
                        "free_space_violation_fraction": 0.008,
                    },
                ]
            }
        ],
        "rejected_depth_frames": 0,
        "depth_anything": {
            "failed_frames": 0,
            "fit_accepted": True,
            "validation_metrics": {
                "surviving_anchor_count": 300,
                "occupied_cells": 48,
                "total_cells": 48,
                "quadrants": [True, True, True, True],
                "held_out_median_m": 0.010,
                "held_out_p95_m": 0.030,
                "held_out_signed_bias_m": 0.004,
                "scale": 1.2,
                "relative_scale_stddev": 0.01,
                "maximum_uncertainty_m": 0.02,
            },
        },
    }


def test_field_acceptance_evidence_fails_closed_without_physical_building(monkeypatch):
    monkeypatch.setattr(server, "_live_surface_pipeline", None)
    response = server.field_acceptance_evidence()
    payload = json.loads(response.body)

    assert response.headers["cache-control"] == "no-store"
    assert payload["available"] is False
    assert "BUILDING_PIPELINE_UNAVAILABLE" in payload["blockers"]
    assert "ACCEPTED_PHYSICAL_DEPTH_EVIDENCE_UNAVAILABLE" in payload["blockers"]


def test_field_acceptance_evidence_aggregates_exact_worst_case_and_export_identity(monkeypatch):
    members = [_member()]
    manifest = {
        "assembly_id": H[2],
        "building_appearance_revision_id": H[7],
        "members": members,
    }
    published = SimpleNamespace(
        manifest=manifest,
        building_appearance_revision_id=H[7],
    )
    monkeypatch.setattr(server, "_live_surface_pipeline", _Pipeline(published))
    monkeypatch.setattr(server.rgbd_refiner, "field_acceptance_snapshot", _physical_snapshot)
    monkeypatch.setattr(
        server,
        "artifact_ingestor",
        SimpleNamespace(snapshot=lambda: {"errors": 0}),
    )
    monkeypatch.setattr(
        server.dense_mapper,
        "snapshot",
        lambda: {"skipped_pose_frames": 0, "decode_errors": 0, "last_surface_error": None},
    )

    response = server.field_acceptance_evidence()
    payload = json.loads(response.body)

    assert payload["available"] is True
    assert payload["blockers"] == []
    assert payload["surface"]["coverage_ppm"] == 850_000
    assert payload["surface"]["independent_color_views"] == 3
    assert payload["physical_depth"] == {
        "correspondence_count": 270,
        "median_m": 0.020,
        "p95_m": 0.050,
        "signed_bias_m": 0.012,
        "occlusion_fraction": 0.015,
        "free_space_fraction": 0.008,
    }
    assert payload["identity"]["building_appearance_revision_id"] == H[7]
    assert payload["exports"]["local_request"] == {
        "tile_revision_id": H[0],
        "appearance_revision_id": H[1],
        "lod_id": H[6],
        "rgbd_augmented_revision_id": None,
    }
    assert payload["exports"]["building_request"] == {
        "assembly_id": H[2],
        "building_appearance_revision_id": H[7],
        "lod_by_member": {H[0]: H[6]},
    }
