"""Durable keyframe sequence state for live preview producer restarts."""
from __future__ import annotations

import os
from pathlib import Path


class KeyframeSequenceState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> int:
        try:
            value = int(self.path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return 0
        return value if value >= 0 else 0

    def store(self, sequence: int) -> None:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(".%s.%s.tmp" % (self.path.name, os.getpid()))
        try:
            temporary.write_text("%d\n" % sequence, encoding="ascii")
            os.replace(temporary, self.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
