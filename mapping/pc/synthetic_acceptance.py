"""Deterministic non-production acceptance of the progressive transfer lifecycle."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.client
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from typing import Iterator

from mapping.pc.ssh_tunnel_receiver import Receiver


def _request(port: int, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, json.loads(raw.decode("utf-8"))


@contextmanager
def _receiver(root: Path) -> Iterator[int]:
    previous_root = Receiver.root
    previous_status = Receiver.status_path
    previous_clients = Receiver.allowed_clients
    Receiver.root = root
    Receiver.status_path = root / ".synthetic-transfer-status.json"
    Receiver.allowed_clients = frozenset({"127.0.0.1", "::1"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root
        Receiver.status_path = previous_status
        Receiver.allowed_clients = previous_clients


def _put_chunk(port: int, session_id: str, artifact_id: str, index: int, chunk: bytes):
    digest = hashlib.sha256(chunk).hexdigest()
    return _request(
        port,
        "PUT",
        f"/progressive/chunks/{session_id}/{artifact_id}/{index}",
        chunk,
        {"Content-Length": str(len(chunk)), "X-Chunk-SHA256": digest},
    )


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_synthetic_acceptance(root: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> dict[str, object]:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    output_root = Path(root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    session_id = "synthetic-p0"
    seed = bytes(range(251))
    artifact = (seed * ((chunk_size * 2 + 4096) // len(seed) + 1))[: chunk_size * 2 + 4096]
    artifact_id = hashlib.sha256(artifact).hexdigest()
    chunks = [artifact[offset : offset + chunk_size] for offset in range(0, len(artifact), chunk_size)]

    with _receiver(output_root) as port:
        first_status, _ = _put_chunk(port, session_id, artifact_id, 0, chunks[0])
        if first_status != 201:
            raise RuntimeError(f"initial synthetic chunk failed with HTTP {first_status}")

    with _receiver(output_root) as port:
        duplicate_status, duplicate = _put_chunk(port, session_id, artifact_id, 0, chunks[0])
        if duplicate_status != 208 or duplicate.get("duplicate") is not True:
            raise RuntimeError("resumed transfer did not observe retained chunk zero")
        for index, chunk in enumerate(chunks[1:], start=1):
            status, _ = _put_chunk(port, session_id, artifact_id, index, chunk)
            if status != 201:
                raise RuntimeError(f"synthetic chunk {index} failed with HTTP {status}")

        relative_path = "synthetic/raw/session.db3.zstd"
        manifest = json.dumps(
            {
                "relative_path": relative_path,
                "artifact_sha256": artifact_id,
                "size_bytes": len(artifact),
                "chunk_count": len(chunks),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        seal_status, sealed = _request(
            port,
            "POST",
            f"/progressive/artifacts/{session_id}/{artifact_id}/seal",
            manifest,
            {"Content-Length": str(len(manifest)), "Content-Type": "application/json"},
        )
        verified_path = output_root / relative_path
        verified = (
            seal_status == 201
            and sealed.get("artifact_sha256") == artifact_id
            and verified_path.is_file()
            and hashlib.sha256(verified_path.read_bytes()).hexdigest() == artifact_id
        )
        reconstruction = json.dumps({"integrated_frames": 3, "integrated_points": 4096}).encode("utf-8")
        reconstructed_status, reconstructed = _request(
            port,
            "POST",
            f"/progressive/artifacts/{session_id}/{artifact_id}/reconstructed",
            reconstruction,
            {"Content-Length": str(len(reconstruction)), "Content-Type": "application/json"},
        )
        filename_status, _ = _request(
            port,
            "DELETE",
            f"/progressive/artifacts/{session_id}/session.db3.zstd/content",
        )
        delete_status, deleted = _request(
            port,
            "DELETE",
            f"/progressive/artifacts/{session_id}/{artifact_id}/content",
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "label": "SYNTHETIC_PASS",
        "production_eligible": False,
        "quality_level": "PREVIEW_UNCALIBRATED",
        "session_id": session_id,
        "artifact_sha256": artifact_id,
        "artifact_size_bytes": len(artifact),
        "chunk_size_bytes": chunk_size,
        "chunk_count": len(chunks),
        "resume_duplicate_observed": duplicate_status == 208,
        "artifact_sha256_verified": verified,
        "reconstruction_committed": reconstructed_status == 200
        and reconstructed.get("release_authorized") is True,
        "filename_only_delete_rejected": filename_status in {400, 404},
        "exact_digest_delete_authorized": delete_status == 200
        and deleted.get("release_authorized") is True
        and deleted.get("content_deleted") is True,
    }
    if not all(
        report[key]
        for key in (
            "resume_duplicate_observed",
            "artifact_sha256_verified",
            "reconstruction_committed",
            "filename_only_delete_rejected",
            "exact_digest_delete_authorized",
        )
    ):
        raise RuntimeError(f"synthetic acceptance failed: {report}")
    _write_atomic(output_root / "synthetic-acceptance.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(run_synthetic_acceptance(args.root, chunk_size=args.chunk_size), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
