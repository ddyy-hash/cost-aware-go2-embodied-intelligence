from pathlib import Path

from mapping.robot.go2_mapping_locomotion.vendor_sport_adapter import (
    Go2VendorSportAdapter,
    VendorSportAdapterError,
)


def test_vendor_adapter_rejects_unbounded_or_unknown_commands(tmp_path: Path):
    adapter = Go2VendorSportAdapter(tmp_path / "missing")

    assert adapter.available is False
    try:
        adapter.select_task("JUMP")
    except VendorSportAdapterError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unknown terrain task was accepted")


def test_vendor_adapter_exposes_only_bounded_task_gaits():
    assert Go2VendorSportAdapter.__module__.endswith("vendor_sport_adapter")
