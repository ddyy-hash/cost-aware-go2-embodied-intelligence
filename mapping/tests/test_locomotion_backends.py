from mapping.pc.go2_mapping_locomotion.backends import available_backends, backend_manifest


def test_external_projects_are_registered_as_explicit_adapters():
    names = {item["id"] for item in available_backends()}
    assert {"pgtt-go2", "unitree-rl-gym", "walk-these-ways-go2"} <= names
    assert "genesis-go2" not in names


def test_pgtt_adapter_declares_hardware_and_perception_contract():
    item = backend_manifest("pgtt-go2")
    assert item["hardware"] == ["unitree_go2"]
    assert "heightmap" in item["inputs"]
    assert item["direct_motor_output"] is False
    assert item["status"] == "ADAPTER_REQUIRED"


def test_genesis_backend_is_disabled_from_runtime_registry():
    import pytest

    with pytest.raises(KeyError, match="unknown locomotion backend"):
        backend_manifest("genesis-go2")
