import json
from pathlib import Path

import tools.analyze_go2_joint_calibration as calibration_tool

from tools.analyze_go2_joint_calibration import (
    _analyze_captures,
    _atomic_json,
    _capture_payload,
    _stage_capture_tasks,
)


class _Manifest:
    robot_factory_serial_verified = False


class _Inventory:
    artifact_bytes = 12
    part_count = 2
    duration_s = 3.5
    message_counts = {"color": 1}
    runtime_manifest = _Manifest()


class _Motion:
    lio_epoch_count = 2
    calibration_rotation_span_deg = 12.0
    calibration_translation_span_m = 0.2
    angular_speed_duration_s = 0.7


class _Analysis:
    artifact_sha256 = "a" * 64
    inventory = _Inventory()
    accepted_board_detections = 5
    accepted_lidar_planes = 1
    motion = _Motion()
    blockers = ("BLOCKED",)
    lidar_rejection_reasons = ("NO_BOARD",)


def test_capture_payload_preserves_evidence_summary_without_transform():
    payload = _capture_payload("capture-a", _Analysis())

    assert payload["capture_id"] == "capture-a"
    assert payload["artifact_sha256"] == "a" * 64
    assert payload["factory_serial_verified"] is False
    assert "camera_from_lidar" not in payload


def test_atomic_json_writes_a_complete_json_document(tmp_path):
    path = tmp_path / "report.json"

    _atomic_json(path, {"analysis_kind": "JOINT_DIAGNOSTIC_ONLY"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "analysis_kind": "JOINT_DIAGNOSTIC_ONLY"
    }


def test_parallel_capture_analysis_preserves_order_and_bounds_opencv_threads():
    submitted_workers = []
    received = []

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            submitted_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, worker, tasks, thread_counts, incomplete_flags, relaxed_flags, association_limits):
            return [
                worker(task, threads, incomplete, relaxed, association_limit)
                for task, threads, incomplete, relaxed, association_limit in zip(
                    tasks, thread_counts, incomplete_flags, relaxed_flags, association_limits
                )
            ]

    def worker(task, opencv_threads, allow_incomplete, operator_relaxed, association_limit):
        received.append((task, opencv_threads, allow_incomplete, operator_relaxed, association_limit))
        return task[0]

    result = _analyze_captures(
        [
            ("estimation_1", Path("one")),
            ("estimation_2", Path("two")),
            ("estimation_3", Path("three")),
            ("holdout", Path("four")),
        ],
        workers=4,
        cpu_count=20,
        executor_factory=ImmediateExecutor,
        worker=worker,
        allow_incomplete=True,
        operator_relaxed=True,
        max_lidar_associations=120,
    )

    assert submitted_workers == [4]
    assert received == [
        (("estimation_1", Path("one")), 5, True, True, 120),
        (("estimation_2", Path("two")), 5, True, True, 120),
        (("estimation_3", Path("three")), 5, True, True, 120),
        (("holdout", Path("four")), 5, True, True, 120),
    ]
    assert result == ["estimation_1", "estimation_2", "estimation_3", "holdout"]


def test_staged_capture_tasks_preserve_capture_ids_and_source_provenance(tmp_path):
    source_one = tmp_path / "source-one"
    source_two = tmp_path / "source-two"
    staging_root = tmp_path / "staging"
    calls = []

    class Staged:
        def __init__(self, source, staged, identity, part_count):
            self.source_path = source
            self.staged_path = staged
            self.source_parts_sha256 = identity
            self.parts = tuple(range(part_count))
            self.manifest_path = staged / "diagnostic-staging.json"

    def stager(source, root, *, workers):
        calls.append((source, root, workers))
        return Staged(
            source,
            root / source.name,
            "a" * 64 if source == source_one else "b" * 64,
            2,
        )

    staged_tasks, staging_report = _stage_capture_tasks(
        [("estimation_1", source_one), ("holdout", source_two)],
        staging_root=staging_root,
        workers=20,
        stager=stager,
    )

    assert calls == [
        (source_one, staging_root, 20),
        (source_two, staging_root, 20),
    ]
    assert staged_tasks == [
        ("estimation_1", staging_root / "source-one"),
        ("holdout", staging_root / "source-two"),
    ]
    assert staging_report == {
        "kind": "READ_ONLY_UNCOMPRESSED_DIAGNOSTIC_STAGING",
        "staging_root": str(staging_root),
        "captures": [
            {
                "capture_id": "estimation_1",
                "source_path": str(source_one),
                "staged_path": str(staging_root / "source-one"),
                "source_parts_sha256": "a" * 64,
                "source_part_count": 2,
                "manifest_path": str(
                    staging_root / "source-one" / "diagnostic-staging.json"
                ),
            },
            {
                "capture_id": "holdout",
                "source_path": str(source_two),
                "staged_path": str(staging_root / "source-two"),
                "source_parts_sha256": "b" * 64,
                "source_part_count": 2,
                "manifest_path": str(
                    staging_root / "source-two" / "diagnostic-staging.json"
                ),
            },
        ],
    }


def test_capture_worker_assigns_its_bounded_cpu_budget_to_color_detection(monkeypatch):
    configured_threads = []
    analysis_kwargs = {}

    monkeypatch.setattr(
        calibration_tool.cv2,
        "setNumThreads",
        lambda count: configured_threads.append(count),
    )

    def analyze(_path, **kwargs):
        analysis_kwargs.update(kwargs)
        return _Analysis()

    monkeypatch.setattr(calibration_tool, "analyze_capture", analyze)
    monkeypatch.setattr(
        calibration_tool,
        "joint_capture_from_analysis",
        lambda capture_id, analysis: (capture_id, analysis),
    )

    _payload, capture = calibration_tool._analyze_capture_task(
        ("estimation_1", Path("capture")), 5, True, True, 120
    )

    assert configured_threads == [1]
    assert analysis_kwargs["materialization_workers"] == 1
    assert analysis_kwargs["target_detection_workers"] == 5
    assert analysis_kwargs["allow_incomplete"] is True
    assert analysis_kwargs["operator_relaxed_lidar_planes"] is True
    assert analysis_kwargs["max_lidar_associations"] == 120
    assert capture[0] == "estimation_1"


def test_main_excludes_named_holdout_from_joint_fit(monkeypatch, tmp_path):
    captured_fit_inputs = []
    evaluated_holdouts = []

    class Joint:
        status = "JOINT_DIAGNOSTIC_PROVISIONAL"

        def as_dict(self):
            return {"status": self.status}

    monkeypatch.setattr(
        calibration_tool,
        "_analyze_captures",
        lambda *_args, **_kwargs: [
            ({"capture_id": "estimation_1"}, "E1"),
            ({"capture_id": "estimation_2"}, "E2"),
            ({"capture_id": "estimation_3"}, "E3"),
            ({"capture_id": "holdout"}, "H"),
        ],
    )

    def solve(captures, *, operator_relaxed):
        captured_fit_inputs.append((tuple(captures), operator_relaxed))
        return Joint()

    monkeypatch.setattr(calibration_tool, "solve_joint_diagnostic", solve)
    monkeypatch.setattr(
        calibration_tool,
        "_holdout_diagnostic",
        lambda capture, _joint: evaluated_holdouts.append(capture) or {"capture": capture},
        raising=False,
    )

    assert calibration_tool.main(
        [
            "--capture", "estimation_1=one",
            "--capture", "estimation_2=two",
            "--capture", "estimation_3=three",
            "--capture", "holdout=four",
            "--allow-incomplete",
            "--operator-relaxed",
            "--output", str(tmp_path / "report.json"),
        ]
    ) == 0

    assert captured_fit_inputs == [(("E1", "E2", "E3"), True)]
    assert evaluated_holdouts == ["H"]
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["holdout_diagnostics"] == [
        {"capture": "H"}
    ]
