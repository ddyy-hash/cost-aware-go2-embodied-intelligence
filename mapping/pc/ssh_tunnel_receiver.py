"""Tailnet-bound receiver for finalized and progressive Go2 artifacts.

The robot reaches this listener directly over its Tailscale interface. Tailscale
uses an encrypted WireGuard path when the peers can connect directly and
automatically falls back to DERP when they cannot. High-fidelity files are
received as independently checksummed chunks and are published only after the
reassembled artifact passes its own SHA-256 check.

SSH runs separately from this receiver and is used only by
deployment, maintenance, or explicit file-transfer tooling.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Deque, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

try:
    from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore
except ImportError:  # Direct script execution keeps mapping/pc on sys.path.
    from artifact_lifecycle import ArtifactLifecycleStore


_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024


class Receiver(BaseHTTPRequestHandler):
    root = Path(".")
    status_path: Optional[Path] = None
    allowed_clients = frozenset({"127.0.0.1", "::1"})
    _lifecycle_store: Optional[ArtifactLifecycleStore] = None
    _lifecycle_root: Optional[Path] = None
    _status_lock = threading.Lock()
    _status_root: Optional[Path] = None
    _arrivals: Deque[Tuple[float, int]] = deque()
    _accepted_chunk_count = 0
    _verified_artifact_count = 0
    _verified_artifact_bytes = 0
    _last_verified_at: Optional[float] = None
    _last_allocation: Optional[dict] = None

    def log_message(self, fmt, *args):
        print("[receiver] " + fmt % args, flush=True)

    def do_GET(self):
        if not self._client_is_allowed():
            return
        path = urlsplit(self.path).path
        if path == "/health":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "transport": "tailnet-direct-or-loopback",
                    "progressive_protocol": "sha256-chunk-v1",
                    "capacity_probe": True,
                },
            )
        elif path == "/progressive/artifacts":
            self._artifact_list(urlsplit(self.path).query)
        elif path.startswith("/progressive/artifacts/") and path.endswith("/status"):
            self._artifact_status(path)
        elif path.startswith("/progressive/artifacts/") and path.endswith("/content"):
            self._artifact_content(path)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_PUT(self):
        if not self._client_is_allowed():
            return
        path = urlsplit(self.path).path
        if path.startswith("/segments/"):
            self._put_finalized_segment(path)
            return
        if path.startswith("/progressive/chunks/"):
            self._put_progressive_chunk(path)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown upload path"})

    def do_POST(self):
        if not self._client_is_allowed():
            return
        path = urlsplit(self.path).path
        if path.startswith("/progressive/probe/"):
            self._capacity_probe(path)
            return
        if path.startswith("/progressive/artifacts/"):
            if path.endswith("/reconstructed"):
                self._mark_reconstructed(path)
            else:
                self._seal_progressive_artifact(path)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown upload path"})

    def _capacity_probe(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) != 3 or parts[:2] != ["progressive", "probe"] or not self._safe_component(parts[2]):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsafe probe identity"})
            return
        length = self._content_length(maximum=256 * 1024)
        if length != 256 * 1024:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "capacity probe must be exactly 256 KiB"})
            return
        expected = self.headers.get("X-Probe-SHA256", "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "X-Probe-SHA256 is required"})
            return
        digest = hashlib.sha256()
        remaining = length
        while remaining:
            block = self.rfile.read(min(1024 * 1024, remaining))
            if not block:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "truncated capacity probe"})
                return
            digest.update(block)
            remaining -= len(block)
        actual = digest.hexdigest()
        if actual != expected:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "capacity probe checksum mismatch"})
            return
        self._json(HTTPStatus.OK, {"probe_bytes": length, "probe_sha256": actual, "throttled": False})

    def do_DELETE(self):
        if not self._client_is_allowed():
            return
        path = urlsplit(self.path).path
        if path.startswith("/progressive/artifacts/") and path.endswith("/content"):
            self._delete_reconstructed_content(path)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown deletion path"})

    def _put_finalized_segment(self, path: str) -> None:
        relative = self._safe_relative(path.removeprefix("/segments/"))
        if relative is None:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsafe path"})
            return
        length = self._content_length()
        if length is None:
            return
        target = self.root / Path(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{time.time_ns()}.partial")
        actual = self._receive_to_file(partial, length)
        if actual is None:
            return
        expected = self.headers.get("X-SHA256")
        if expected and expected.lower() != actual:
            partial.unlink(missing_ok=True)
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "checksum mismatch", "actual": actual})
            return
        os.replace(partial, target)
        self._record_high_fidelity(length, verified_artifact_bytes=length)
        self._json(HTTPStatus.CREATED, {"path": str(relative), "sha256": actual, "bytes": length})

    def _put_progressive_chunk(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) != 5 or parts[:2] != ["progressive", "chunks"]:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown progressive chunk path"})
            return
        session_id, artifact_id, chunk_number = parts[2:]
        if (
            not self._safe_component(session_id)
            or not re.fullmatch(r"[0-9a-f]{64}", artifact_id)
            or not chunk_number.isdigit()
        ):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsafe progressive artifact identity"})
            return
        chunk_index = int(chunk_number)
        if chunk_index > 9_999_999:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "chunk index is too large"})
            return
        length = self._content_length(maximum=_MAX_CHUNK_BYTES)
        if length is None:
            return
        expected = self.headers.get("X-Chunk-SHA256", "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "X-Chunk-SHA256 is required"})
            return

        directory = self.root / ".progressive" / session_id / artifact_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{chunk_index:08d}.part"
        incoming = directory / f".{chunk_index:08d}.{time.time_ns()}.incoming"
        actual = self._receive_to_file(incoming, length)
        if actual is None:
            return
        if actual != expected:
            incoming.unlink(missing_ok=True)
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "chunk checksum mismatch", "actual": actual})
            return
        if target.exists():
            existing = self._sha256(target)
            incoming.unlink(missing_ok=True)
            if existing != expected:
                self._json(HTTPStatus.CONFLICT, {"error": "chunk identity conflicts with retained data"})
                return
            self._json(
                HTTPStatus.ALREADY_REPORTED,
                {"chunk_sha256": actual, "bytes": length, "duplicate": True},
            )
            return
        os.replace(incoming, target)
        self._record_high_fidelity(length, verified_artifact_bytes=0, allocation=self._bandwidth_allocation_headers())
        self._json(HTTPStatus.CREATED, {"chunk_sha256": actual, "bytes": length, "duplicate": False})

    def _seal_progressive_artifact(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) != 5 or parts[:2] != ["progressive", "artifacts"] or parts[4] != "seal":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown progressive artifact path"})
            return
        session_id, artifact_id = parts[2:4]
        if not self._safe_component(session_id) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsafe progressive artifact identity"})
            return
        length = self._content_length(maximum=_MAX_MANIFEST_BYTES)
        if length is None:
            return
        raw = self.rfile.read(length)
        if len(raw) != length:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "truncated manifest"})
            return
        try:
            manifest = json.loads(raw.decode("utf-8"))
            relative = self._safe_relative(str(manifest["relative_path"]))
            expected_sha256 = str(manifest["artifact_sha256"]).lower()
            size_bytes = int(manifest["size_bytes"])
            chunk_count = int(manifest["chunk_count"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid artifact manifest"})
            return
        if relative is None or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or artifact_id != expected_sha256:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid artifact identity"})
            return
        if size_bytes < 0 or not 0 < chunk_count <= 100_000:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid artifact size or chunk count"})
            return

        chunks = self.root / ".progressive" / session_id / artifact_id
        target = self.root / Path(*relative.parts)
        if target.exists():
            existing = self._sha256(target)
            if target.stat().st_size == size_bytes and existing == expected_sha256:
                self._lifecycle().record_verified(
                    session_id,
                    artifact_id,
                    relative.as_posix(),
                    size_bytes,
                )
                self._json(
                    HTTPStatus.ALREADY_REPORTED,
                    {"path": str(relative), "artifact_sha256": existing, "bytes": size_bytes, "duplicate": True},
                )
                return
            self._json(HTTPStatus.CONFLICT, {"error": "artifact path conflicts with different verified data"})
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{time.time_ns()}.partial")
        digest = hashlib.sha256()
        written = 0
        try:
            with partial.open("xb") as output:
                for index in range(chunk_count):
                    source = chunks / f"{index:08d}.part"
                    if not source.is_file():
                        raise FileNotFoundError(source.name)
                    with source.open("rb") as input_file:
                        while block := input_file.read(1024 * 1024):
                            output.write(block)
                            digest.update(block)
                            written += len(block)
                output.flush()
                os.fsync(output.fileno())
        except FileNotFoundError as exc:
            partial.unlink(missing_ok=True)
            self._json(HTTPStatus.CONFLICT, {"error": f"missing chunk {exc.args[0]}"})
            return
        actual = digest.hexdigest()
        if written != size_bytes or actual != expected_sha256:
            partial.unlink(missing_ok=True)
            self._json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "artifact checksum mismatch", "actual": actual, "bytes": written},
            )
            return
        os.replace(partial, target)
        self._lifecycle().record_verified(
            session_id,
            artifact_id,
            relative.as_posix(),
            written,
        )
        for source in chunks.glob("*"):
            source.unlink(missing_ok=True)
        chunks.rmdir()
        try:
            chunks.parent.rmdir()
        except OSError:
            pass
        self._record_high_fidelity(0, verified_artifact_bytes=written)
        self._json(
            HTTPStatus.CREATED,
            {"path": str(relative), "artifact_sha256": actual, "bytes": written, "duplicate": False},
        )

    def _artifact_status(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) != 5 or parts[:2] != ["progressive", "artifacts"] or parts[4] != "status":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown lifecycle path"})
            return
        session_id, artifact_id = parts[2:4]
        if not self._safe_component(session_id) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsafe lifecycle identity"})
            return
        status = self._lifecycle().status(session_id, artifact_id)
        if status.get("state") == "UNKNOWN":
            self._json(HTTPStatus.NOT_FOUND, status)
            return
        self._json(HTTPStatus.OK, status)

    def _artifact_list(self, query: str) -> None:
        requested = parse_qs(query).get("state", [""])[0]
        allowed_states = {"SHA256_VERIFIED", "PC_RECONSTRUCTION_INGESTED"}
        if requested and requested not in allowed_states:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsupported lifecycle state"})
            return
        artifacts = []
        registry_root = self.root / ".artifact-lifecycle"
        for candidate in sorted(registry_root.glob("*/*.json")):
            try:
                status = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
            if not isinstance(status, dict) or status.get("state") not in allowed_states:
                continue
            if requested and status.get("state") != requested:
                continue
            artifacts.append(status)
        self._json(HTTPStatus.OK, {"artifacts": artifacts})

    def _artifact_content(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) != 5 or parts[:2] != ["progressive", "artifacts"] or parts[4] != "content":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown artifact content path"})
            return
        session_id, artifact_id = parts[2:4]
        if not self._safe_component(session_id) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsafe artifact identity"})
            return
        status = self._lifecycle().status(session_id, artifact_id)
        if status.get("state") not in {"SHA256_VERIFIED", "PC_RECONSTRUCTION_INGESTED"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "artifact is not published"})
            return
        try:
            relative = self._safe_relative(str(status["relative_path"]))
            expected = str(status["artifact_sha256"]).lower()
            size_bytes = int(status["size_bytes"])
        except (KeyError, TypeError, ValueError):
            self._json(HTTPStatus.CONFLICT, {"error": "invalid lifecycle record"})
            return
        if relative is None or not re.fullmatch(r"[0-9a-f]{64}", expected) or size_bytes < 0:
            self._json(HTTPStatus.CONFLICT, {"error": "invalid lifecycle identity"})
            return
        target = self.root / Path(*relative.parts)
        if not target.is_file() or target.stat().st_size != size_bytes:
            self._json(HTTPStatus.CONFLICT, {"error": "published artifact is missing or truncated"})
            return
        if self._sha256(target) != expected:
            self._json(HTTPStatus.CONFLICT, {"error": "published artifact checksum changed"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size_bytes))
        self.send_header("X-Artifact-SHA256", expected)
        self.send_header("X-Artifact-Relative-Path", relative.as_posix())
        self.end_headers()
        with target.open("rb") as source:
            while block := source.read(1024 * 1024):
                self.wfile.write(block)

    def _mark_reconstructed(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) != 5 or parts[:2] != ["progressive", "artifacts"] or parts[4] != "reconstructed":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown reconstruction path"})
            return
        session_id, artifact_id = parts[2:4]
        if not self._safe_component(session_id) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsafe artifact identity"})
            return
        length = self._content_length(maximum=_MAX_MANIFEST_BYTES)
        if length is None:
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = payload if isinstance(payload, dict) else {}
            status = self._lifecycle().status(session_id, artifact_id)
            relative = self._safe_relative(str(status["relative_path"]))
            if relative is None:
                raise ValueError("invalid lifecycle relative path")
            target = self.root / Path(*relative.parts)
            updated = self._lifecycle().mark_reconstructed(target, result=result)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, LookupError) as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            return
        self._json(HTTPStatus.OK, updated)

    def _delete_reconstructed_content(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) != 5 or parts[:2] != ["progressive", "artifacts"] or parts[4] != "content":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown artifact deletion path"})
            return
        session_id, artifact_id = parts[2:4]
        if not self._safe_component(session_id) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "unsafe artifact identity"})
            return
        status = self._lifecycle().status(session_id, artifact_id)
        if status.get("state") != "PC_RECONSTRUCTION_INGESTED" or status.get("release_authorized") is not True:
            self._json(HTTPStatus.CONFLICT, {"error": "reconstruction has not authorized release"})
            return
        relative = self._safe_relative(str(status.get("relative_path", "")))
        if relative is None:
            self._json(HTTPStatus.CONFLICT, {"error": "invalid lifecycle relative path"})
            return
        target = self.root / Path(*relative.parts)
        deleted = False
        if target.exists():
            if not target.is_file():
                self._json(HTTPStatus.CONFLICT, {"error": "artifact content path is not a file"})
                return
            if target.stat().st_size != int(status.get("size_bytes", -1)) or self._sha256(target) != artifact_id:
                self._json(HTTPStatus.CONFLICT, {"error": "artifact content no longer matches lifecycle identity"})
                return
            target.unlink()
            deleted = True
        self._json(
            HTTPStatus.OK,
            {"artifact_id": artifact_id, "content_deleted": deleted, "release_authorized": True},
        )

    def _client_is_allowed(self) -> bool:
        client = str(self.client_address[0])
        if client in type(self).allowed_clients:
            return True
        self._json(HTTPStatus.FORBIDDEN, {"error": "client is outside the receiver allowlist"})
        return False

    @classmethod
    def _lifecycle(cls) -> ArtifactLifecycleStore:
        root = cls.root.resolve()
        if cls._lifecycle_store is None or cls._lifecycle_root != root:
            cls._lifecycle_store = ArtifactLifecycleStore(root)
            cls._lifecycle_root = root
        return cls._lifecycle_store

    def _content_length(self, maximum: Optional[int] = None) -> Optional[int]:
        try:
            length = int(self.headers["Content-Length"])
        except (KeyError, ValueError):
            self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length required"})
            return None
        if length < 0 or (maximum is not None and length > maximum):
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body is too large"})
            return None
        return length

    def _bandwidth_allocation_headers(self) -> Optional[dict]:
        names = {
            "total_mbps": "X-Go2-Total-Bandwidth-Mbps",
            "preview_mbps": "X-Go2-Preview-Bandwidth-Mbps",
            "high_fidelity_mbps": "X-Go2-High-Fidelity-Bandwidth-Mbps",
        }
        allocation = {}
        try:
            for key, name in names.items():
                value = float(self.headers[name])
                if value <= 0:
                    return None
                allocation[key] = value
        except (KeyError, TypeError, ValueError):
            return None
        try:
            observed_capacity_mbps = float(self.headers.get("X-Go2-Observed-Capacity-Mbps", ""))
            if observed_capacity_mbps > 0:
                allocation["observed_capacity_mbps"] = observed_capacity_mbps
        except (TypeError, ValueError):
            pass
        return allocation

    def _receive_to_file(self, target: Path, length: int) -> Optional[str]:
        digest = hashlib.sha256()
        remaining = length
        try:
            with target.open("xb") as output:
                while remaining:
                    block = self.rfile.read(min(1024 * 1024, remaining))
                    if not block:
                        raise EOFError
                    output.write(block)
                    digest.update(block)
                    remaining -= len(block)
                output.flush()
                os.fsync(output.fileno())
        except EOFError:
            target.unlink(missing_ok=True)
            self._json(HTTPStatus.BAD_REQUEST, {"error": "truncated upload"})
            return None
        return digest.hexdigest()

    @staticmethod
    def _safe_relative(value: str) -> Optional[PurePosixPath]:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            return None
        return relative

    @staticmethod
    def _safe_component(value: str) -> bool:
        return bool(_COMPONENT.fullmatch(value))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _record_high_fidelity(cls, received_bytes: int, verified_artifact_bytes: int, allocation: Optional[dict] = None) -> None:
        now = time.monotonic()
        root = cls.root.resolve()
        with cls._status_lock:
            if cls._status_root != root:
                cls._status_root = root
                cls._arrivals.clear()
                cls._accepted_chunk_count = 0
                cls._verified_artifact_count = 0
                cls._verified_artifact_bytes = 0
                cls._last_verified_at = None
                cls._last_allocation = None
            if allocation is not None:
                cls._last_allocation = allocation
            else:
                allocation = cls._last_allocation
            if received_bytes:
                cls._arrivals.append((now, received_bytes))
                cls._accepted_chunk_count += 1
            while cls._arrivals and now - cls._arrivals[0][0] > 5.0:
                cls._arrivals.popleft()
            if verified_artifact_bytes:
                cls._verified_artifact_count += 1
                cls._verified_artifact_bytes += verified_artifact_bytes
                cls._last_verified_at = now
            first = cls._arrivals[0][0] if cls._arrivals else now
            window = max(1.0, now - first)
            goodput_mbps = sum(size for _, size in cls._arrivals) * 8.0 / window / 1_000_000.0
            age_ms = None if cls._last_verified_at is None else int((now - cls._last_verified_at) * 1000.0)
            status = {
                "schema_version": 1,
                "high_fidelity": {
                    "state": "CHECKSUM_VERIFIED" if cls._verified_artifact_count else "RECEIVING_CHUNKS",
                    "app_goodput_mbps": round(goodput_mbps, 3),
                    "accepted_chunks": cls._accepted_chunk_count,
                    "verified_artifacts": cls._verified_artifact_count,
                    "verified_bytes": cls._verified_artifact_bytes,
                    "last_verified_age_ms": age_ms,
                    "updated_at_ms": int(time.time() * 1000),
                    "total_budget_mbps": None if allocation is None else allocation.get("total_mbps"),
                    "preview_budget_mbps": None if allocation is None else allocation.get("preview_mbps"),
                    "high_fidelity_budget_mbps": None if allocation is None else allocation.get("high_fidelity_mbps"),
                    "observed_capacity_mbps": None if allocation is None else allocation.get("observed_capacity_mbps"),
                },
            }
            destination = cls.status_path or root / ".progressive-status.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{time.time_ns()}.tmp")
            temporary.write_text(json.dumps(status, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            os.replace(temporary, destination)

    def _json(self, status: HTTPStatus, body: dict) -> None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", action="append", dest="bind_addresses")
    parser.add_argument("--allow-client", action="append", dest="allowed_clients")
    args = parser.parse_args()
    Receiver.root = args.root.resolve()
    Receiver.status_path = args.status_path.resolve() if args.status_path else None
    Receiver.allowed_clients = frozenset(args.allowed_clients or ["127.0.0.1", "::1"])
    Receiver.root.mkdir(parents=True, exist_ok=True)
    bind_addresses = list(dict.fromkeys(args.bind_addresses or ["127.0.0.1"]))
    servers = [ThreadingHTTPServer((address, args.port), Receiver) for address in bind_addresses]
    threads = []
    try:
        for server in servers[1:]:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        servers[0].serve_forever()
    finally:
        for server in servers[1:]:
            server.shutdown()
        for server in servers:
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
