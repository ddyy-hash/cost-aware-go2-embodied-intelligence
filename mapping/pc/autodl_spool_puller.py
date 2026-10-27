#!/usr/bin/env python3
"""Pull verified AutoDL artifacts into the PC inbox and relay reconstruction ACKs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Optional
from urllib.parse import quote, urlencode, urlsplit

try:
    from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore
except ImportError:
    from artifact_lifecycle import ArtifactLifecycleStore


_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_BYTES = 1024 * 1024


class SpoolError(RuntimeError):
    pass


def _endpoint(value: str):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an http or https URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain credentials, query, or fragment")
    return parsed


def _connection(parsed):
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=300)
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=300)


def _path(parsed, suffix: str) -> str:
    return f"{parsed.path.rstrip('/')}{suffix}"


def _headers(token: str) -> dict[str, str]:
    if not token or "\r" in token or "\n" in token:
        raise ValueError("bulk authentication token is invalid")
    return {"Authorization": f"Bearer {token}"}


def _json_request(parsed, token: str, method: str, suffix: str, payload: Optional[dict] = None) -> dict:
    body = b"" if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = _headers(token)
    if payload is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
    connection = _connection(parsed)
    try:
        connection.request(method, _path(parsed, suffix), body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpoolError(f"spool returned invalid JSON with HTTP {response.status}") from exc
        if response.status != 200:
            raise SpoolError(f"spool rejected {method} {suffix} with HTTP {response.status}: {result}")
        if not isinstance(result, dict):
            raise SpoolError("spool returned a non-object JSON response")
        return result
    finally:
        connection.close()


def _identity(status: dict) -> tuple[str, str, PurePosixPath, int]:
    session_id = str(status.get("session_id", ""))
    artifact_id = str(status.get("artifact_id", ""))
    relative = PurePosixPath(str(status.get("relative_path", "")))
    try:
        size_bytes = int(status.get("size_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise SpoolError("spool artifact size is invalid") from exc
    if not _COMPONENT.fullmatch(session_id) or not _DIGEST.fullmatch(artifact_id):
        raise SpoolError("spool artifact identity is unsafe")
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or size_bytes < 0:
        raise SpoolError("spool artifact path or size is unsafe")
    return session_id, artifact_id, relative, size_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _download_one(parsed, token: str, root: Path, status: dict) -> dict:
    session_id, artifact_id, relative, size_bytes = _identity(status)
    target = root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_file() and target.stat().st_size == size_bytes and _sha256(target) == artifact_id:
            return {"status": status, "path": target, "bytes": 0, "duplicate": True}
        raise SpoolError(f"local artifact path conflicts with verified data: {relative}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.partial")
    connection = _connection(parsed)
    digest = hashlib.sha256()
    received = 0
    try:
        suffix = f"/progressive/artifacts/{quote(session_id)}/{quote(artifact_id)}/content"
        connection.request("GET", _path(parsed, suffix), headers=_headers(token))
        response = connection.getresponse()
        if response.status != 200:
            raw = response.read()
            raise SpoolError(f"artifact download failed with HTTP {response.status}: {raw[:256]!r}")
        if response.getheader("X-Artifact-SHA256") != artifact_id:
            raise SpoolError("artifact download identity header does not match")
        with temporary.open("xb") as output:
            while block := response.read(_BLOCK_BYTES):
                output.write(block)
                digest.update(block)
                received += len(block)
            output.flush()
            os.fsync(output.fileno())
        if received != size_bytes or digest.hexdigest() != artifact_id:
            raise SpoolError("downloaded artifact failed size or SHA-256 verification")
        os.replace(temporary, target)
        return {"status": status, "path": target, "bytes": received, "duplicate": False}
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)


def pull_once(endpoint: str, token: str, root: Path, *, workers: int = 4) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    parsed = _endpoint(endpoint)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lifecycle = ArtifactLifecycleStore(root)
    listing = _json_request(
        parsed,
        token,
        "GET",
        "/progressive/artifacts?" + urlencode({"state": "SHA256_VERIFIED"}),
    )
    artifacts = listing.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise SpoolError("spool artifact listing is invalid")
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="autodl-pull") as executor:
        futures = [executor.submit(_download_one, parsed, token, root, status) for status in artifacts]
        for future in as_completed(futures):
            results.append(future.result())
    transferred = 0
    acknowledged = 0
    for result in results:
        status = result["status"]
        session_id, artifact_id, relative, size_bytes = _identity(status)
        lifecycle.record_verified(session_id, artifact_id, relative.as_posix(), size_bytes)
        transferred += int(result["bytes"])
        local = lifecycle.status(session_id, artifact_id)
        if local.get("state") != "PC_RECONSTRUCTION_INGESTED" or local.get("release_authorized") is not True:
            continue
        reconstruction = {
            "integrated_frames": int(local.get("integrated_frames", 0)),
            "integrated_points": int(local.get("integrated_points", 0)),
            "duplicate": bool(local.get("duplicate_ingest", False)),
        }
        suffix = f"/progressive/artifacts/{quote(session_id)}/{quote(artifact_id)}/reconstructed"
        remote = _json_request(parsed, token, "POST", suffix, reconstruction)
        if remote.get("release_authorized") is not True:
            raise SpoolError("remote reconstruction ACK did not authorize release")
        _json_request(
            parsed,
            token,
            "DELETE",
            f"/progressive/artifacts/{quote(session_id)}/{quote(artifact_id)}/content",
        )
        acknowledged += 1
    elapsed = max(time.monotonic() - started, 1e-6)
    return {
        "state": "ACTIVE",
        "listed_artifacts": len(artifacts),
        "downloaded_artifacts": sum(not item["duplicate"] for item in results),
        "downloaded_bytes": transferred,
        "download_goodput_mbps": round(transferred * 8.0 / elapsed / 1_000_000.0, 3),
        "reconstruction_acks": acknowledged,
        "updated_at_ms": int(time.time() * 1000),
    }


def _write_status(path: Optional[Path], payload: dict) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="ascii").strip()
    while True:
        try:
            result = pull_once(args.endpoint, token, args.root, workers=args.workers)
        except Exception as exc:
            result = {"state": "ERROR", "error": str(exc), "updated_at_ms": int(time.time() * 1000)}
            _write_status(args.status_path, result)
            if args.once:
                raise
        else:
            _write_status(args.status_path, result)
            if args.once:
                print(json.dumps(result, sort_keys=True))
                return 0
        time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
