import json
from pathlib import Path

import pytest

from mapping.pc.go2_mapping_vla.offline import (
    MobileVLADecisionError,
    OfflineVLAAdapter,
    decision_to_navigation_payload,
    parse_mobilevla_decision,
    parse_mobilevla_action_proposal,
    classify_model_response,
)
from mapping.pc.go2_mapping_assets import build_model_manifest, verify_tree_identity


def test_parse_mobilevla_decision_accepts_only_structured_high_level_intent():
    decision = parse_mobilevla_decision(
        json.dumps(
            {
                "intent": "inspect_room",
                "target": "room_a",
                "confidence": 0.91,
                "reasoning_summary": "Room perimeter is the next assisted task.",
                "constraints": {"max_speed_mps": 0.2, "require_operator_confirmation": True},
            }
        )
    )

    assert decision.intent == "inspect_room"
    assert decision.target == "room_a"
    assert decision.confidence == pytest.approx(0.91)
    assert decision.requires_operator_confirmation is True
    assert decision.allow_navigation is False


def test_parse_mobilevla_decision_rejects_low_level_action_fields():
    with pytest.raises(MobileVLADecisionError, match="low-level action"):
        parse_mobilevla_decision(
            '{"intent":"inspect_room","target":"room_a","cmd_vel":{"x":0.1},"confidence":0.9}'
        )


def test_parse_mobilevla_decision_rejects_unsafe_or_unknown_intents():
    with pytest.raises(MobileVLADecisionError, match="intent"):
        parse_mobilevla_decision('{"intent":"walk_anywhere","target":"room_a","confidence":0.9}')


def test_decision_to_navigation_payload_is_assisted_and_never_activates_motion():
    decision = parse_mobilevla_decision(
        '{"intent":"inspect_room","target":"room_a","confidence":0.95,"constraints":{"max_speed_mps":0.15}}'
    )
    payload = decision_to_navigation_payload(
        decision,
        mission_id="mission-1",
        task_id="task-room-a",
        task_revision=2,
        waypoints=[[1.0, 1.0], [2.0, 1.0]],
        geofence={"frame": "SESSION_ODOM_LOCAL", "polygon": [[0, 0], [3, 0], [3, 3], [0, 3]]},
    )

    assert payload["mode"] == "ASSISTED"
    assert payload["task_status"] == "READY"
    assert payload["operator_confirmation_required"] is True
    assert payload["max_speed_mps"] == pytest.approx(0.15)
    assert "cmd_vel" not in payload
    assert "joint_targets" not in payload


def test_offline_adapter_fails_closed_when_mobilevla_checkpoint_is_not_staged(tmp_path):
    adapter = OfflineVLAAdapter(tmp_path / "missing-checkpoint")

    assert adapter.status()["state"] == "CHECKPOINT_UNAVAILABLE"
    with pytest.raises(MobileVLADecisionError, match="checkpoint"):
        adapter.decide("inspect room A", observation={})


def test_mobilevla_manifest_records_the_verified_staged_upstream_checkpoint():
    manifest = json.loads(
        Path("mapping/config/mobilevla_r1_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert manifest["upstream_commit"] == "ff9062c143d6b50c05d913e6be15d36634a7a460"
    assert manifest["huggingface_repo"] == "AIGeeksGroup/MobileVLA-R1"
    assert manifest["checkpoint_parts"] == 26
    assert manifest["checkpoint_archive_bytes"] == 26871863199
    assert manifest["checkpoint_archive_sha256"] == "90a356b75dada4be196462926082a5e36a3d9a77c438ff412bb1e7db98e5a966"
    assert manifest["checkpoint_state"] == "READY"
    assert manifest["local_rl_directory_sha256"] == "b44e41fdb7d4812494660e72b80a8f3504cadd89c58d953fad27e648f94a79d6"
    assert manifest["local_rl_file_count"] == 17
    assert manifest["offline_only"] is True
    assert manifest["action_boundary"] == "ASSISTED_NAV2_INTENT_ONLY"


def test_mobilevla_manifest_distinguishes_navigation_ready_and_loadable_variants():
    manifest = json.loads(Path("mapping/config/mobilevla_r1_manifest.json").read_text(encoding="utf-8"))

    assert manifest["variant_capabilities"]["rl"] == {
        "loadable": True,
        "structured_navigation": True,
    }
    assert manifest["variant_capabilities"]["sft"] == {
        "loadable": True,
        "structured_navigation": False,
    }


def test_mobilevla_import_is_local_only_and_records_an_integrity_manifest():
    importer = Path("tools/import_mobilevla_offline.ps1").read_text(encoding="utf-8")

    assert "Expand-Archive" in importer
    assert "Get-FileHash" in importer
    assert "local_files_only" in importer
    assert "Invoke-WebRequest" not in importer
    assert "curl" not in importer.lower()
    assert "MOBILEVLA_OFFLINE_IMPORT_READY" in importer


def test_mobilevla_checkpoint_downloader_is_resumable_and_validates_the_local_zip():
    downloader = Path("tools/download_mobilevla_checkpoint.sh").read_text(encoding="utf-8")

    assert "weight.zip.part-" in downloader
    assert "--continue-at" in downloader
    assert "weight.zip" in downloader
    assert "unzip -t" in downloader
    assert "zipfile" in downloader
    assert "testzip" in downloader
    assert "AIGeeksGroup/MobileVLA-R1" in downloader
    assert "huggingface.co" in downloader


def test_mobilevla_runtime_lock_records_the_tested_inference_environment():
    requirements = Path("mapping/pc/mobilevla_requirements.txt").read_text(encoding="utf-8")
    manifest = json.loads(Path("mapping/config/mobilevla_r1_manifest.json").read_text(encoding="utf-8"))

    assert "torch==2.7.1+cu128" in requirements
    assert "transformers==4.53.2" in requirements
    assert "peft==0.9.0" in requirements
    assert "decord==0.6.0" in requirements
    assert manifest["verified_runtime"]["torch"] == "2.7.1+cu128"
    assert manifest["verified_runtime"]["transformers"] == "4.53.2"
    assert manifest["verified_runtime"]["local_only"] is True


def test_official_mobilevla_backend_uses_local_only_prompt_and_fails_closed_without_source():
    from mapping.pc.go2_mapping_vla.official import (
        OfficialMobileVLABackend,
        build_navigation_prompt,
    )

    prompt = build_navigation_prompt("inspect room A", frame_count=1)
    assert "inspect room A" in prompt
    assert "local" in prompt.lower()
    assert "velocity" in prompt.lower()

    backend = OfficialMobileVLABackend(
        model_path="/missing/model",
        source_root="/missing/source",
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="local MobileVLA source/model"):
        backend.load()


def test_mobilevla_navigation_prompt_requires_structured_semantic_json():
    from mapping.pc.go2_mapping_vla.official import build_navigation_prompt

    prompt = build_navigation_prompt("inspect room A")
    assert "JSON object" in prompt
    assert '"intent"' in prompt
    assert '"confidence"' in prompt
    assert "operator confirmation" in prompt.lower()


def test_model_response_classification_records_acceptance_or_fail_closed_reason():
    accepted = classify_model_response(
        '{"intent":"inspect_room","target":"room","confidence":0.8,'
        '"constraints":{"require_operator_confirmation":true}}'
    )
    assert accepted["state"] == "ACCEPTED"
    assert accepted["decision"]["intent"] == "inspect_room"
    assert accepted["executable"] is False

    rejected = classify_model_response("personality, the robot")
    assert rejected["state"] == "REJECTED"
    assert rejected["reason"] == "MobileVLA response must be JSON"
    assert rejected["executable"] is False


def test_official_mobilevla_backend_preserves_checkpoint_variant_identity():
    from mapping.pc.go2_mapping_vla.official import model_name_for_variant

    assert model_name_for_variant("sft") == "mobilevla-r1-sft"
    assert model_name_for_variant("rl") == "mobilevla-r1-rl"
    with pytest.raises(ValueError, match="variant"):
        model_name_for_variant("unknown")


def test_mobilevla_offline_cli_is_available_for_reproducible_local_inference():
    cli = Path("tools/run_mobilevla_offline_inference.py").read_text(encoding="utf-8")

    assert "--model-path" in cli
    assert "--source-root" in cli
    assert "--image" in cli
    assert "--variant" in cli
    assert 'default="rl"' in cli
    assert "args.variant" in cli
    assert "identity_key" in cli
    assert "TRANSFORMERS_OFFLINE" in cli
    assert "local_files_only" in cli
    assert "mobilevla_r1_manifest.json" in cli
    assert "verify_tree_identity" in cli
    assert "MOBILEVLA_INFERENCE_RESULT" in cli
    assert "variant=args.variant" in cli
    assert "--load-in-4bit" in cli

    backend = Path("mapping/pc/go2_mapping_vla/official.py").read_text(encoding="utf-8")
    assert "attention_mask" in backend
    assert "temperature=0.0" not in backend


def test_compact_mobilevla_tree_identity_detects_tampering(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    manifest = build_model_manifest(model, model_id="mobilevla-test", revision="rev")

    assert verify_tree_identity(
        model,
        digest=manifest["directory_sha256"],
        file_count=manifest["file_count"],
        total_bytes=manifest["total_bytes"],
    ) == manifest["directory_sha256"]
    (model / "config.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tree identity mismatch"):
        verify_tree_identity(
            model,
            digest=manifest["directory_sha256"],
            file_count=manifest["file_count"],
            total_bytes=manifest["total_bytes"],
        )


def test_real_model_text_must_be_structured_before_it_can_become_a_navigation_intent():
    assert "person walking in the park" not in {"inspect_room", "go_to_room"}
    with pytest.raises(MobileVLADecisionError, match="JSON"):
        parse_mobilevla_decision("person walking in the park")


def test_rl_model_text_is_recorded_as_a_non_executable_action_proposal():
    proposal = parse_mobilevla_action_proposal("The next action is move forward 25 cm.")

    assert proposal.action == "move_forward"
    assert proposal.distance_m == pytest.approx(0.25)
    assert proposal.executable is False
    assert proposal.requires_operator_confirmation is True


def test_action_proposal_rejects_unknown_or_unsafe_text():
    with pytest.raises(MobileVLADecisionError, match="action proposal"):
        parse_mobilevla_action_proposal("run the motors at full speed")
