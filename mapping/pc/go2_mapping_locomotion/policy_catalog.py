"""Verified PGTT policy catalog and conservative terrain-level selection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class PolicyCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class PGTTPolicyAsset:
    level: str
    run: int
    path: Path
    sha256: str
    input_size: int
    action_size: int
    executable: bool = False

    def verify(self) -> None:
        if not self.path.is_file():
            raise PolicyCatalogError("PGTT policy file is missing: %s" % self.path)
        actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if actual != self.sha256:
            raise PolicyCatalogError("PGTT policy identity mismatch for level%s" % self.level)
        if (self.input_size, self.action_size) != (153, 12):
            raise PolicyCatalogError("PGTT policy tensor contract is unsupported")
        if self.executable:
            raise PolicyCatalogError("catalog assets cannot authorize actuation")


def load_policy_catalog(path: str | Path) -> dict[str, Any]:
    catalog_path = Path(path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyCatalogError("PGTT policy catalog is unreadable") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("policies"), list):
        raise PolicyCatalogError("PGTT policy catalog schema is unsupported")
    return payload


def policy_asset(catalog_path: str | Path, level: str, *, root: str | Path | None = None) -> PGTTPolicyAsset:
    catalog_file = Path(catalog_path)
    catalog = load_policy_catalog(catalog_file)
    for item in catalog["policies"]:
        if item.get("level") == level and item.get("run") == 0:
            policy_path = Path(item["path"])
            if root is not None and not policy_path.is_absolute():
                policy_path = Path(root) / policy_path
                # Catalogs are shared by the PC workspace and robot release.
                # The release keeps the same asset under mapping/robot.
                if not policy_path.is_file() and str(item["path"]).startswith("output/robot-release/"):
                    release_path = Path(root) / str(item["path"]).replace(
                        "output/robot-release/", "mapping/robot/", 1
                    )
                    if release_path.is_file():
                        policy_path = release_path
            asset = PGTTPolicyAsset(
                level=level,
                run=0,
                path=policy_path,
                sha256=str(item["sha256"]),
                input_size=int(item["input_size"]),
                action_size=int(item["action_size"]),
            )
            asset.verify()
            return asset
    raise PolicyCatalogError("PGTT policy level is not cataloged: %s" % level)


def select_policy_level(*, terrain_state: str, risk_level: str = "low") -> str:
    if terrain_state not in {"STAIR_UP", "STAIR_DOWN", "ROUGH_TERRAIN"}:
        raise PolicyCatalogError("terrain state is not supported")
    if risk_level not in {"low", "medium", "high"}:
        raise PolicyCatalogError("risk level is not supported")
    if terrain_state in {"STAIR_UP", "STAIR_DOWN"}:
        return {"low": "03", "medium": "07", "high": "13"}[risk_level]
    return {"low": "07", "medium": "10", "high": "17"}[risk_level]


__all__ = ["PGTTPolicyAsset", "PolicyCatalogError", "load_policy_catalog", "policy_asset", "select_policy_level"]
