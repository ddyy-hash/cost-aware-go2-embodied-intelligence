"""Stable machine-readable contracts for mapping entrypoint results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
import json
import os
from pathlib import Path
from typing import Iterable


class ExitCode(IntEnum):
    SUCCESS = 0
    INVALID_CONFIGURATION = 2
    DEPENDENCY_MISSING = 3
    ENDPOINT_UNAVAILABLE = 4
    SENSOR_PREFLIGHT = 5
    DATA_INTEGRITY = 6
    QUALITY_GATE = 7
    STATE_CONFLICT = 8
    INTERNAL_FAILURE = 9


class OperationalState(str, Enum):
    OFFLINE = "OFFLINE"
    CONNECTED = "CONNECTED"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    PREVIEW_READY = "PREVIEW_READY"
    CAPTURE_READY = "CAPTURE_READY"
    RECORDING = "RECORDING"
    FINALIZING = "FINALIZING"
    UPLOADING = "UPLOADING"
    RECONSTRUCTING = "RECONSTRUCTING"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    QUALITY_BLOCKED = "QUALITY_BLOCKED"
    DELIVERABLE_READY = "DELIVERABLE_READY"
    FAULTED = "FAULTED"


@dataclass(frozen=True)
class CheckRecord:
    name: str
    passed: bool
    blocking: bool = True
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("check name must be non-empty")
        if not isinstance(self.passed, bool) or not isinstance(self.blocking, bool):
            raise ValueError("check passed and blocking must be booleans")
        if not isinstance(self.detail, str):
            raise ValueError("check detail must be a string")


def _strings(values: Iterable[str], field: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{field} must contain non-empty strings")
    return result


@dataclass(frozen=True)
class OperationResult:
    command: str
    state_before: OperationalState
    state_after: OperationalState
    exit_code: ExitCode
    checks: tuple[CheckRecord, ...] = ()
    artifacts: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    log_paths: tuple[str, ...] = ()
    release_hash: str | None = None
    config_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("command must be non-empty")
        if not isinstance(self.state_before, OperationalState):
            raise ValueError("state_before must be an OperationalState")
        if not isinstance(self.state_after, OperationalState):
            raise ValueError("state_after must be an OperationalState")
        if not isinstance(self.exit_code, ExitCode):
            raise ValueError("exit_code must be an ExitCode")
        if any(not isinstance(check, CheckRecord) for check in self.checks):
            raise ValueError("checks must contain CheckRecord values")
        object.__setattr__(self, "artifacts", _strings(self.artifacts, "artifacts"))
        object.__setattr__(self, "blockers", _strings(self.blockers, "blockers"))
        object.__setattr__(self, "log_paths", _strings(self.log_paths, "log_paths"))
        for field in ("release_hash", "config_hash"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} must be null or a non-empty string")
        failed_blocking = any(not check.passed and check.blocking for check in self.checks)
        if self.exit_code is ExitCode.SUCCESS and failed_blocking:
            raise ValueError("successful operation requires every blocking postcondition to pass")
        if self.exit_code is not ExitCode.SUCCESS and not self.blockers:
            raise ValueError("failed operation requires at least one machine-readable blocker")

    @classmethod
    def success(cls, command: str, state: OperationalState) -> "OperationResult":
        return cls(command, state, state, ExitCode.SUCCESS)

    @classmethod
    def failure(
        cls,
        command: str,
        state_before: OperationalState,
        state_after: OperationalState,
        exit_code: ExitCode,
        *,
        blocker: str,
    ) -> "OperationResult":
        if exit_code is ExitCode.SUCCESS:
            raise ValueError("failure requires a non-success exit code")
        return cls(
            command=command,
            state_before=state_before,
            state_after=state_after,
            exit_code=exit_code,
            blockers=(blocker,),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "command": self.command,
            "exit_code": int(self.exit_code),
            "state_before": self.state_before.value,
            "state_after": self.state_after.value,
            "checks": [asdict(check) for check in self.checks],
            "artifacts": list(self.artifacts),
            "blockers": list(self.blockers),
            "log_paths": list(self.log_paths),
            "release_hash": self.release_hash,
            "config_hash": self.config_hash,
        }


def write_result_atomic(path: str | Path, result: OperationResult) -> None:
    """Durably replace a result without exposing partial JSON to readers."""

    if not isinstance(result, OperationResult):
        raise TypeError("result must be an OperationResult")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    payload = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
