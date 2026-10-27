import json
import os
import time

import pytest

from mapping.pc.temp_recovery import RecoveryDecision, recover_abandoned_temp


def _age(path, seconds=7200):
    timestamp = time.time() - seconds
    os.utime(path, (timestamp, timestamp))


def test_young_temp_state_is_never_recovered(tmp_path):
    candidate = tmp_path / ".dense-tmp" / "active.sqlite"
    candidate.parent.mkdir()
    candidate.write_bytes(b"sqlite")

    result = recover_abandoned_temp(candidate, root=tmp_path, minimum_age_s=3600)

    assert result.decision is RecoveryDecision.TOO_YOUNG
    assert candidate.exists()


def test_live_owner_or_journal_reference_blocks_recovery(tmp_path):
    candidate = tmp_path / ".dense-tmp" / "active.sqlite"
    candidate.parent.mkdir()
    candidate.write_bytes(b"sqlite")
    _age(candidate)
    owner = candidate.with_suffix(candidate.suffix + ".owner.json")
    owner.write_text(json.dumps({"pid": 123, "process_identity": "start-token"}), encoding="utf-8")

    live = recover_abandoned_temp(
        candidate,
        root=tmp_path,
        minimum_age_s=3600,
        process_is_live=lambda pid, identity: (pid, identity) == (123, "start-token"),
    )
    referenced = recover_abandoned_temp(
        candidate,
        root=tmp_path,
        minimum_age_s=3600,
        active_journal_paths={candidate.resolve()},
    )

    assert live.decision is RecoveryDecision.ACTIVE_OWNER
    assert referenced.decision is RecoveryDecision.JOURNAL_REFERENCED
    assert candidate.exists()


def test_existing_recovery_lock_blocks_recovery(tmp_path):
    candidate = tmp_path / ".dense-tmp" / "active.sqlite"
    candidate.parent.mkdir()
    candidate.write_bytes(b"sqlite")
    _age(candidate)
    candidate.with_suffix(candidate.suffix + ".recovery.lock").write_text("held", encoding="utf-8")

    result = recover_abandoned_temp(candidate, root=tmp_path, minimum_age_s=3600)

    assert result.decision is RecoveryDecision.LOCKED
    assert candidate.exists()


def test_stale_unowned_temp_is_moved_to_quarantine(tmp_path):
    candidate = tmp_path / ".dense-tmp" / "abandoned.sqlite"
    candidate.parent.mkdir()
    candidate.write_bytes(b"sqlite")
    _age(candidate)

    result = recover_abandoned_temp(candidate, root=tmp_path, minimum_age_s=3600)

    assert result.decision is RecoveryDecision.QUARANTINED
    assert not candidate.exists()
    assert result.quarantine_path is not None
    assert result.quarantine_path.read_bytes() == b"sqlite"
    assert result.quarantine_path.is_relative_to(tmp_path / ".dense-quarantine")


def test_recovery_rejects_a_candidate_outside_the_declared_root(tmp_path):
    outside = tmp_path.parent / "outside.sqlite"
    outside.write_bytes(b"do-not-touch")
    _age(outside)

    with pytest.raises(ValueError, match="outside"):
        recover_abandoned_temp(outside, root=tmp_path, minimum_age_s=3600)

    assert outside.read_bytes() == b"do-not-touch"
