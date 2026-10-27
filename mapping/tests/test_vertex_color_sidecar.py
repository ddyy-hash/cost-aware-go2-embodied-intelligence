import hashlib
import struct

import pytest

from mapping.pc.go2_mapping_appearance.vertex_color import (
    build_historical_vertex_color_sidecar,
    parse_vertex_color_v1,
)
from mapping.pc.go2_mapping_quality.reconstruction_config import canonical_sha256
from mapping.pc.go2_mapping_rgbd.texture import VertexColor


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def _color(index, *, accepted=True, views=2, rgb=(0.0, 0.5, 1.0)):
    return VertexColor(
        vertex_id=f"{index}:0:0",
        mean_linear_rgb=rgb,
        channel_stddev=(0.01, 0.01, 0.01),
        accepted_count=views,
        rejected_count=0,
        independent_views=views,
        source_frame_sha256=(HASH_F, HASH_E),
        accepted_for_texture=accepted,
    )


def _build(colors):
    return build_historical_vertex_color_sidecar(
        geometry_revision_id=HASH_A,
        pose_generation=HASH_B,
        calibration_id="go2-d435i-v1",
        calibration_sha256=HASH_C,
        physical_depth_validation_revision=HASH_D,
        appearance_config_sha256=HASH_E,
        vertex_colors=colors,
    )


def test_v1_binary_header_records_and_revision_are_content_addressed():
    sidecar = _build([_color(0), _color(1, accepted=False, views=1)])
    magic, version, header_bytes, count = struct.unpack_from("<8sHHI", sidecar.binary, 0)
    parsed = parse_vertex_color_v1(sidecar.binary)

    assert (magic, version, header_bytes, count) == (b"G2VCOLR1", 1, 80, 2)
    assert sidecar.binary[16:48].hex() == HASH_A
    assert sidecar.binary[48:80].hex() == sidecar.manifest["appearance_revision_id"]
    assert sidecar.binary[80:] == bytes((0, 188, 255, 255, 0, 0, 0, 0))
    assert parsed.vertex_count == 2
    assert parsed.geometry_revision_id == HASH_A
    assert parsed.appearance_revision_id == sidecar.manifest["appearance_revision_id"]
    assert parsed.rgb.tolist() == [[0, 188, 255], [0, 0, 0]]
    assert parsed.validity.tolist() == [255, 0]
    assert sidecar.manifest["vertex_color"]["content_sha256"] == hashlib.sha256(sidecar.binary).hexdigest()
    assert sidecar.manifest["revision_core"]["vertex_color_payload_sha256"] == hashlib.sha256(
        sidecar.binary[80:]
    ).hexdigest()
    assert sidecar.manifest["appearance_revision_id"] == canonical_sha256(
        sidecar.manifest["revision_core"]
    )
    assert sidecar.manifest["source_frame_sha256"] == [HASH_E, HASH_F]


def test_delivery_authorization_requires_80_percent_and_two_views():
    accepted = _build([_color(index) for index in range(4)] + [_color(4, accepted=False, views=1)])
    low_coverage = _build(
        [_color(index) for index in range(3)]
        + [_color(3, accepted=False), _color(4, accepted=False)]
    )
    one_view = _build([_color(index, views=1) for index in range(5)])

    assert accepted.manifest["coverage_ppm"] == 800_000
    assert accepted.manifest["minimum_independent_accepted_views"] == 2
    assert accepted.manifest["delivery_authorized"] is True
    assert accepted.manifest["blockers"] == []
    assert low_coverage.manifest["coverage_ppm"] == 600_000
    assert low_coverage.manifest["delivery_authorized"] is False
    assert "VERTEX_COLOR_COVERAGE_BELOW_800000_PPM" in low_coverage.manifest["blockers"]
    assert one_view.manifest["delivery_authorized"] is False
    assert "VERTEX_COLOR_INDEPENDENT_VIEWS_BELOW_2" in one_view.manifest["blockers"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: b"BADMAGIC" + value[8:],
        lambda value: value[:8] + struct.pack("<H", 2) + value[10:],
        lambda value: value[:10] + struct.pack("<H", 79) + value[12:],
        lambda value: value[:-1],
        lambda value: value + b"\0",
        lambda value: value[:83] + b"\x01" + value[84:],
    ],
)
def test_parser_rejects_corrupt_or_noncanonical_payloads(mutate):
    payload = _build([_color(0)]).binary

    with pytest.raises(ValueError):
        parse_vertex_color_v1(mutate(payload))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"geometry_revision_id": "bad"}, "geometry_revision_id"),
        ({"pose_generation": "bad"}, "pose_generation"),
        ({"calibration_id": ""}, "calibration_id"),
        ({"calibration_sha256": "bad"}, "calibration_sha256"),
        ({"vertex_colors": []}, "vertex_colors"),
    ],
)
def test_builder_rejects_missing_identity_or_vertices(override, message):
    values = {
        "geometry_revision_id": HASH_A,
        "pose_generation": HASH_B,
        "calibration_id": "go2-d435i-v1",
        "calibration_sha256": HASH_C,
        "physical_depth_validation_revision": HASH_D,
        "appearance_config_sha256": HASH_E,
        "vertex_colors": [_color(0)],
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        build_historical_vertex_color_sidecar(**values)
