import json

from tools.build_offline_acceptance_report import load


def test_load_accepts_nested_mobilevla_classification(tmp_path):
    path = tmp_path / "mobilevla.json"
    path.write_text(json.dumps({"artifact_type": "mobilevla_gpu_inference_audit", "classification": {"state": "ACCEPTED"}}), encoding="utf-8")
    report = load(path)
    assert report["available"] is True
    assert report["accepted"] is True


def test_load_fails_closed_for_invalid_artifact(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("not-json", encoding="utf-8")
    report = load(path)
    assert report["available"] is False
    assert report["accepted"] is False
