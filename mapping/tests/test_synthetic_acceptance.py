import json

from mapping.pc.synthetic_acceptance import run_synthetic_acceptance


def test_synthetic_acceptance_exercises_resume_verify_commit_and_exact_digest_cleanup(tmp_path):
    report = run_synthetic_acceptance(tmp_path, chunk_size=1024)

    assert report["label"] == "SYNTHETIC_PASS"
    assert report["production_eligible"] is False
    assert report["quality_level"] == "PREVIEW_UNCALIBRATED"
    assert report["resume_duplicate_observed"] is True
    assert report["artifact_sha256_verified"] is True
    assert report["reconstruction_committed"] is True
    assert report["filename_only_delete_rejected"] is True
    assert report["exact_digest_delete_authorized"] is True
    assert len(report["artifact_sha256"]) == 64
    assert json.loads((tmp_path / "synthetic-acceptance.json").read_text(encoding="utf-8")) == report
