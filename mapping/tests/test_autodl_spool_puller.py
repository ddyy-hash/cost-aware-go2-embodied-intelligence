import hashlib
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore
from mapping.pc.autodl_spool_puller import pull_once
from mapping.pc.ssh_tunnel_receiver import Receiver


def _request(port, method, path, body=b"", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def test_puller_verifies_publishes_acknowledges_and_releases_remote_content(tmp_path):
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    previous_root = Receiver.root
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    Receiver.root = remote_root
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = b"sealed lidar payload"
        digest = hashlib.sha256(body).hexdigest()
        session_id = "session-003"
        relative = "lidar/session-003/part.db3.zstd"
        status, _ = _request(
            server.server_port,
            "PUT",
            f"/progressive/chunks/{session_id}/{digest}/0",
            body,
            {"Content-Length": str(len(body)), "X-Chunk-SHA256": digest},
        )
        assert status == 201
        manifest = json.dumps(
            {
                "relative_path": relative,
                "artifact_sha256": digest,
                "size_bytes": len(body),
                "chunk_count": 1,
            }
        ).encode("utf-8")
        status, _ = _request(
            server.server_port,
            "POST",
            f"/progressive/artifacts/{session_id}/{digest}/seal",
            manifest,
            {"Content-Length": str(len(manifest))},
        )
        assert status == 201

        endpoint = f"http://127.0.0.1:{server.server_port}"
        first = pull_once(endpoint, "test-token", local_root, workers=2)
        local_path = local_root / relative
        assert first["downloaded_artifacts"] == 1
        assert local_path.read_bytes() == body

        lifecycle = ArtifactLifecycleStore(local_root)
        lifecycle.mark_reconstructed(
            local_path,
            result={"integrated_frames": 3, "integrated_points": 1024},
        )
        second = pull_once(endpoint, "test-token", local_root, workers=2)
        assert second["reconstruction_acks"] == 1
        remote_status = ArtifactLifecycleStore(remote_root).status(session_id, digest)
        assert remote_status["release_authorized"] is True
        assert not (remote_root / relative).exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Receiver.root = previous_root
