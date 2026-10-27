import json

from mapping.pc.go2_mapping_calibration.__main__ import build_parser, main


def test_calibration_cli_requires_three_explicit_estimation_slots():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--estimation", "one",
            "--estimation", "two",
            "--estimation", "three",
            "--holdout", "holdout",
            "--lidar-imu-evidence", "lidar-imu.json",
            "--output-manifest", "manifest.json",
            "--output-diagnostics", "diagnostics.json",
            "--robot-serial", "go2",
            "--camera-serial", "d435",
        ]
    )
    assert [str(item) for item in args.estimation] == ["one", "two", "three"]
    assert str(args.lidar_imu_evidence) == "lidar-imu.json"


def test_calibration_cli_returns_blocked_exit_and_machine_report_for_missing_bags(tmp_path):
    manifest = tmp_path / "manifest.json"
    diagnostics = tmp_path / "diagnostics.json"
    result = main(
        [
            "--estimation", str(tmp_path / "one.db3"),
            "--estimation", str(tmp_path / "two.db3"),
            "--estimation", str(tmp_path / "three.db3"),
            "--holdout", str(tmp_path / "holdout.db3"),
            "--lidar-imu-evidence", str(tmp_path / "lidar-imu.json"),
            "--output-manifest", str(manifest),
            "--output-diagnostics", str(diagnostics),
            "--robot-serial", "go2",
            "--camera-serial", "d435",
        ]
    )

    assert result == 7
    assert json.loads(diagnostics.read_text(encoding="utf-8"))["quality_level"] == "PREVIEW_UNCALIBRATED"
    assert not manifest.exists()
