from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict


class SubmapStore:
    def __init__(self, root: Path, max_active_bytes: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_active_bytes = max_active_bytes
        self._active: "OrderedDict[str, bytes]" = OrderedDict()
        self._metadata: Dict[str, Dict[str, object]] = {}

    def add(self, submap_id: str, payload: bytes, sealed: bool) -> None:
        if submap_id in self._metadata:
            raise ValueError("submaps are immutable once written")
        self._active[submap_id] = payload
        self._metadata[submap_id] = {"sealed": sealed, "size_bytes": len(payload), "evicted": False}
        self._evict_if_needed()

    def is_evicted(self, submap_id: str) -> bool:
        return bool(self._metadata[submap_id]["evicted"])

    def metadata(self, submap_id: str) -> Dict[str, object]:
        return dict(self._metadata[submap_id])

    def load(self, submap_id: str) -> bytes:
        if submap_id in self._active:
            self._active.move_to_end(submap_id)
            return self._active[submap_id]
        payload = (self.root / f"{submap_id}.bin").read_bytes()
        self._active[submap_id] = payload
        self._metadata[submap_id]["evicted"] = False
        self._evict_if_needed(protect=submap_id)
        return payload

    def _evict_if_needed(self, protect: str = "") -> None:
        while sum(map(len, self._active.values())) > self.max_active_bytes:
            candidate = next((key for key in self._active if key != protect and self._metadata[key]["sealed"]), None)
            if candidate is None:
                raise MemoryError("active submap budget exceeded without an evictable sealed submap")
            payload = self._active.pop(candidate)
            (self.root / f"{candidate}.bin").write_bytes(payload)
            (self.root / f"{candidate}.json").write_text(json.dumps(self._metadata[candidate], sort_keys=True), encoding="utf-8")
            self._metadata[candidate]["evicted"] = True
