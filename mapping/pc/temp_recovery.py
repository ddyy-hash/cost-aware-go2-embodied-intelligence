"""Conservative quarantine for abandoned dense reconstruction state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import time
from typing import Callable, Iterable


class RecoveryDecision(str, Enum):
    TOO_YOUNG = "TOO_YOUNG"
    ACTIVE_OWNER = "ACTIVE_OWNER"
    JOURNAL_REFERENCED = "JOURNAL_REFERENCED"
    LOCKED = "LOCKED"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True)
class RecoveryResult:
    candidate: Path
    decision: RecoveryDecision
    quarantine_path: Path | None = None


def _default_process_is_live(pid: int, _identity: str) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def recover_abandoned_temp(
    candidate: str | Path,
    *,
    root: str | Path,
    minimum_age_s: float,
    active_journal_paths: Iterable[Path] = (),
    process_is_live: Callable[[int, str], bool] = _default_process_is_live,
    now: float | None = None,
) -> RecoveryResult:
    """Move one proven-abandoned file into quarantine; never delete it."""

    if minimum_age_s < 0:
        raise ValueError("minimum_age_s must be non-negative")
    resolved_root = Path(root).resolve()
    path = Path(candidate).resolve()
    try:
        relative = path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("candidate is outside the declared recovery root") from exc
    if not path.is_file():
        raise ValueError("candidate must be an existing regular file")
    if relative.parts and relative.parts[0] == ".dense-quarantine":
        raise ValueError("candidate is already inside quarantine")

    current_time = time.time() if now is None else float(now)
    if current_time - path.stat().st_mtime < minimum_age_s:
        return RecoveryResult(path, RecoveryDecision.TOO_YOUNG)

    references = {Path(item).resolve() for item in active_journal_paths}
    if path in references:
        return RecoveryResult(path, RecoveryDecision.JOURNAL_REFERENCED)

    owner_path = path.with_suffix(path.suffix + ".owner.json")
    if owner_path.is_file():
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            pid = int(owner["pid"])
            identity = str(owner["process_identity"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return RecoveryResult(path, RecoveryDecision.ACTIVE_OWNER)
        if process_is_live(pid, identity):
            return RecoveryResult(path, RecoveryDecision.ACTIVE_OWNER)

    lock_path = path.with_suffix(path.suffix + ".recovery.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return RecoveryResult(path, RecoveryDecision.LOCKED)

    try:
        with os.fdopen(lock_fd, "w", encoding="ascii") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        quarantine_root = resolved_root / ".dense-quarantine"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        suffix = f".{time.time_ns()}.quarantine"
        target = quarantine_root / ("__".join(relative.parts) + suffix)
        os.replace(path, target)
        return RecoveryResult(path, RecoveryDecision.QUARANTINED, target)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
