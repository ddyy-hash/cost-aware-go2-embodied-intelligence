"""Bounded wrapper for the matched Go2 native SportClient adapter."""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any


CONFIRMATION = "I_UNDERSTAND_GO2_HIGH_LEVEL_MOTION"
TASK_GAITS = {"STAIR_ASCEND": 3, "STAIR_DESCEND": 3, "ROUGH_TERRAIN": 4}


class VendorSportAdapterError(RuntimeError):
    pass


class Go2VendorSportAdapter:
    def __init__(self, binary: Path, *, network: str = "eth0") -> None:
        self.binary = Path(binary)
        self.network = network

    @property
    def available(self) -> bool:
        return self.binary.is_file()

    def _run(self, args: list[str], *, confirm: bool = False) -> str:
        if not self.available:
            raise VendorSportAdapterError("vendor SportClient adapter is unavailable")
        command = [str(self.binary), "--network", self.network, *args]
        if confirm:
            command += ["--confirm", CONFIRMATION]
        result = subprocess.run(command, text=True, capture_output=True, timeout=8, check=False)
        output = (result.stdout + result.stderr).strip()
        if result.returncode:
            raise VendorSportAdapterError(output or "vendor SportClient command failed")
        return output

    def check(self) -> str:
        return self._run(["--command", "check"])

    def stop(self) -> str:
        return self._run(["--command", "stop"], confirm=True)

    def select_task(self, task: str) -> str:
        if task not in TASK_GAITS:
            raise VendorSportAdapterError("unsupported terrain task")
        return self._run(["--command", "gait", "--gait", str(TASK_GAITS[task])], confirm=True)

    def move(self, *, vx: Any, vy: Any = 0.0, vyaw: Any = 0.0, max_speed_mps: float = 0.12) -> str:
        values = [float(vx), float(vy), float(vyaw)]
        if not all(value == value and abs(value) != float("inf") for value in values):
            raise VendorSportAdapterError("velocity must be finite")
        if abs(values[0]) > max_speed_mps or abs(values[1]) > max_speed_mps or abs(values[2]) > 1.0:
            raise VendorSportAdapterError("velocity exceeds hardware envelope")
        return self._run(
            ["--command", "move", "--vx", str(values[0]), "--vy", str(values[1]), "--vyaw", str(values[2])],
            confirm=True,
        )


__all__ = ["CONFIRMATION", "Go2VendorSportAdapter", "TASK_GAITS", "VendorSportAdapterError"]
