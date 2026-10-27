import hashlib
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

from mapping.pc.ssh_tunnel_receiver import Receiver


def _request(port, method, path, body=b"", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def test_progressive_receiver_accepts_idempotent_chunks_and_publishes_only_a_verified_artifact(tmp_path):
    previous_root = Receiver.root
    previous_status_path = getattr(Receiver, "status_path", None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    Receiver.root = tmp_path
    Receiver.status_path = tmp_path / ".progressive-status.json"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = b"verified raw lidar bytes"
        digest = hashlib.sha256(body).hexdigest()
        headers = {"Content-Length": str(len(body)), "X-Chunk-SHA256": digest}
        path = f"/progressive/chunks/session-001/{digest}/0"

        status, received = _request(server.server_port, "PUT", path, body, headers)
        assert status == 201
        assert received["chunk_sha256"] == digest

        status, replay = _request(server.server_port, "PUT", path, body, headers)
        assert status == 208
        assert replay["duplicate"] is True

        manifest = json.dumps(
            {
                "relative_path": "lidar/segment-000.db3",
                "artifact_sha256": digest,
                "size_bytes": len(body),
                "chunk_count": 1,
            }
        ).encode("utf-8")
        status, sealed = _request(
            server.server_port,
            "POST",
            f"/progressive/artifacts/session-001/{digest}/seal",
            manifest,
            {"Content-Length": str(len(manifest)), "Content-Type": "application/json"},
        )

        assert status == 201
        assert sealed["artifact_sha256"] == digest
        assert (tmp_path / "lidar" / "segment-000.db3").read_bytes() == body
        assert not (tmp_path / ".progressive" / "session-001" / digest).exists()
        assert json.loads(Receiver.status_path.read_text(encoding="utf-8"))["high_fidelity"]["verified_artifacts"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root
        Receiver.status_path = previous_status_path


def test_progressive_receiver_health_advertises_capacity_probe(tmp_path):
    previous_root = Receiver.root
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    Receiver.root = tmp_path
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result = _request(server.server_port, "GET", "/health")
        assert status == 200
        assert result["progressive_protocol"] == "sha256-chunk-v1"
        assert result["capacity_probe"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root


def test_progressive_receiver_rejects_non_digest_artifact_identity(tmp_path):
    previous_root = Receiver.root
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    Receiver.root = tmp_path
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = b"chunk"
        digest = hashlib.sha256(body).hexdigest()
        status, payload = _request(
            server.server_port,
            "PUT",
            "/progressive/chunks/session-001/not-a-digest/0",
            body,
            {"Content-Length": str(len(body)), "X-Chunk-SHA256": digest},
        )
        assert status == 400
        assert "identity" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root


def test_progressive_receiver_lists_downloads_and_acknowledges_reconstructed_artifacts(tmp_path):
    previous_root = Receiver.root
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    Receiver.root = tmp_path
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = b"immutable lidar artifact"
        digest = hashlib.sha256(body).hexdigest()
        chunk_path = f"/progressive/chunks/session-002/{digest}/0"
        status, _ = _request(
            server.server_port,
            "PUT",
            chunk_path,
            body,
            {"Content-Length": str(len(body)), "X-Chunk-SHA256": digest},
        )
        assert status == 201
        manifest = json.dumps(
            {
                "relative_path": "lidar/session-002/part.db3.zstd",
                "artifact_sha256": digest,
                "size_bytes": len(body),
                "chunk_count": 1,
            }
        ).encode("utf-8")
        status, _ = _request(
            server.server_port,
            "POST",
            f"/progressive/artifacts/session-002/{digest}/seal",
            manifest,
            {"Content-Length": str(len(manifest)), "Content-Type": "application/json"},
        )
        assert status == 201

        status, listing = _request(server.server_port, "GET", "/progressive/artifacts?state=SHA256_VERIFIED")
        assert status == 200
        assert listing["artifacts"][0]["artifact_sha256"] == digest

        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", f"/progressive/artifacts/session-002/{digest}/content")
        response = connection.getresponse()
        downloaded = response.read()
        assert response.status == 200
        assert response.getheader("X-Artifact-SHA256") == digest
        assert downloaded == body
        connection.close()

        reconstruction = json.dumps({"integrated_frames": 1, "integrated_points": 128}).encode("utf-8")
        status, acknowledged = _request(
            server.server_port,
            "POST",
            f"/progressive/artifacts/session-002/{digest}/reconstructed",
            reconstruction,
            {"Content-Length": str(len(reconstruction)), "Content-Type": "application/json"},
        )
        assert status == 200
        assert acknowledged["state"] == "PC_RECONSTRUCTION_INGESTED"
        assert acknowledged["release_authorized"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root


def test_progressive_receiver_accepts_256kib_capacity_probe_without_publishing_artifact(tmp_path):
    previous_root = Receiver.root
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    Receiver.root = tmp_path
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = bytes(256 * 1024)
        digest = hashlib.sha256(body).hexdigest()
        status, result = _request(
            server.server_port,
            "POST",
            "/progressive/probe/session-probe",
            body,
            {"Content-Length": str(len(body)), "X-Probe-SHA256": digest},
        )
        assert status == 200
        assert result["probe_bytes"] == len(body)
        assert not (tmp_path / ".progressive").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root
