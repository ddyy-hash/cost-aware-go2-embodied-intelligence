"""Integrity manifests for locally staged mapping assets."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    if not root.is_dir():
        raise ValueError(f"model directory does not exist: {root}")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory is empty: {root}")
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for item in files:
        relative = item.relative_to(root).as_posix()
        relative_bytes = relative.encode("utf-8")
        payload = item.read_bytes()
        file_digest = hashlib.sha256(payload).hexdigest()
        digest.update(len(relative_bytes).to_bytes(4, "little"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        entries.append({"path": relative, "bytes": len(payload), "sha256": file_digest})
    return digest.hexdigest(), entries


def build_model_manifest(root: str | Path, *, model_id: str, revision: str) -> dict[str, Any]:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be non-empty")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be non-empty")
    model_root = Path(root)
    directory_sha256, files = _tree_digest(model_root)
    return {
        "schema_version": 1,
        "asset_type": "pc_model_weights",
        "model_id": model_id,
        "source_revision": revision,
        "relative_path": model_root.as_posix(),
        "directory_sha256": directory_sha256,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def verify_model_manifest(root: str | Path, manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise ValueError("model manifest schema_version must be 1")
    expected_digest = manifest.get("directory_sha256")
    if not isinstance(expected_digest, str) or _SHA256.fullmatch(expected_digest) is None:
        raise ValueError("model manifest directory_sha256 must be lowercase SHA-256")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("model manifest files must be a non-empty list")
    actual_digest, actual_files = _tree_digest(Path(root))
    expected_by_path = {item.get("path"): item for item in expected_files if isinstance(item, Mapping)}
    actual_by_path = {item["path"]: item for item in actual_files}
    if set(expected_by_path) != set(actual_by_path):
        raise ValueError("model file list mismatch")
    for path, expected in expected_by_path.items():
        if not isinstance(path, str) or not _SHA256.fullmatch(str(expected.get("sha256", ""))):
            raise ValueError(f"invalid model file entry: {path}")
        actual = actual_by_path[path]
        if expected["sha256"] != actual["sha256"] or expected.get("bytes") != actual["bytes"]:
            raise ValueError(f"model file digest mismatch: {path}")
    if actual_digest != expected_digest:
        raise ValueError(f"model directory digest mismatch: expected {expected_digest}, got {actual_digest}")
    if manifest.get("file_count") != len(actual_files) or manifest.get("total_bytes") != sum(
        item["bytes"] for item in actual_files
    ):
        raise ValueError("model manifest size metadata mismatch")
    return actual_digest


def verify_tree_identity(root: str | Path, *, digest: str, file_count: int, total_bytes: int) -> str:
    """Verify a model tree using compact identity metadata without a file list."""
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("tree digest must be lowercase SHA-256")
    actual_digest, files = _tree_digest(Path(root))
    actual_bytes = sum(item["bytes"] for item in files)
    if actual_digest != digest or len(files) != file_count or actual_bytes != total_bytes:
        raise ValueError(
            f"tree identity mismatch: expected {digest}/{file_count}/{total_bytes}, "
            f"got {actual_digest}/{len(files)}/{actual_bytes}"
        )
    return actual_digest
