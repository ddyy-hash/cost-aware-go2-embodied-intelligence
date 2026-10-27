#!/usr/bin/env python3
"""Upload an immutable mapping artifact in rate-limited, resumable chunks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import quote, urlsplit

try:
    from .bandwidth import AdaptiveBandwidthController, BandwidthAllocationStore, PreviewDemandStore, TokenBucket, VerifiedAckGoodputWindow
except ImportError:  # Direct execution on the Go2 keeps the release self-contained.
    from bandwidth import AdaptiveBandwidthController, BandwidthAllocationStore, PreviewDemandStore, TokenBucket, VerifiedAckGoodputWindow


_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SEND_BLOCK_BYTES = 16 * 1024
_SOCKET_TIMEOUT_SECONDS = 300
PROBE_CHUNK_BYTES = 256 * 1024


class UploadError(RuntimeError):
    pass


def upload_artifact(
    endpoint: str,
    *,
    session_id: str,
    source: Path,
    relative_path: str,
    chunk_size_bytes: int,
    rate_mbps: float,
    bandwidth_state_path: Optional[Path] = None,
    preview_demand_state_path: Optional[Path] = None,
    total_ceiling_mbps: Optional[float] = None,
    route_id: Optional[str] = None,
    parallel_workers: int = 3,
    capacity_probe_interval_s: float = 30.0,
    await_reconstruction_s: float = 0.0,
    auth_token: Optional[str] = None,
    probe_chunk_bytes: int = PROBE_CHUNK_BYTES,
) -> Dict[str, object]:
    """Transfer ``source`` and atomically publish it only after full verification."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not _COMPONENT.fullmatch(session_id):
        raise ValueError("session_id is unsafe")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("relative_path is unsafe")
    if not 0 < chunk_size_bytes <= 64 * 1024 * 1024:
        raise ValueError("chunk_size_bytes must be between 1 byte and 64 MiB")
    if rate_mbps <= 0:
        raise ValueError("rate_mbps must be positive")
    if bandwidth_state_path is not None and (total_ceiling_mbps is None or total_ceiling_mbps <= 0):
        raise ValueError("total_ceiling_mbps must be positive when adaptive bandwidth is enabled")
    if not 1 <= parallel_workers <= 8:
        raise ValueError("parallel_workers must be between 1 and 8")
    if capacity_probe_interval_s <= 0:
        raise ValueError("capacity_probe_interval_s must be positive")
    if probe_chunk_bytes != PROBE_CHUNK_BYTES:
        raise ValueError("probe_chunk_bytes must be exactly 256 KiB")
    if await_reconstruction_s < 0:
        raise ValueError("await_reconstruction_s must not be negative")
    parsed = _endpoint(endpoint)
    route_id = route_id or f"endpoint:{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"
    auth_token = auth_token or os.environ.get("GO2_MAPPING_BULK_AUTH_TOKEN")
    size_bytes = source.stat().st_size
    artifact_sha256 = _sha256(source)
    artifact_id = artifact_sha256
    chunk_count = max(1, (size_bytes + chunk_size_bytes - 1) // chunk_size_bytes)
    controller = None
    allocation_store = None
    preview_demand_store = None
    if bandwidth_state_path is not None:
        allocation_store = BandwidthAllocationStore(Path(bandwidth_state_path))
        preview_demand_store = PreviewDemandStore(
            Path(preview_demand_state_path or Path(bandwidth_state_path).with_name("preview-demand.json"))
        )
        # Reserve a small preview floor until the first demand heartbeat arrives.
        # Later heartbeats replace this bootstrap value.
        minimum_total_mbps = float(os.environ.get("GO2_MAPPING_MIN_TOTAL_BANDWIDTH_MBPS", "4.0"))
        if not math.isfinite(minimum_total_mbps) or minimum_total_mbps <= 0:
            raise ValueError("GO2_MAPPING_MIN_TOTAL_BANDWIDTH_MBPS must be finite and positive")
        if minimum_total_mbps > float(total_ceiling_mbps):
            raise ValueError("GO2_MAPPING_MIN_TOTAL_BANDWIDTH_MBPS must not exceed total_ceiling_mbps")
        previous = allocation_store.read(expected_route_id=route_id)
        previous_total = previous.total_mbps if previous is not None else minimum_total_mbps
        preview_bootstrap_mbps = float(os.environ.get("GO2_MAPPING_PREVIEW_BOOTSTRAP_MBPS", "1.0"))
        if not math.isfinite(preview_bootstrap_mbps) or preview_bootstrap_mbps < 0:
            raise ValueError("GO2_MAPPING_PREVIEW_BOOTSTRAP_MBPS must be finite and non-negative")
        preview_demand = preview_demand_store.read()
        if preview_demand is None:
            preview_demand = preview_bootstrap_mbps
        controller = AdaptiveBandwidthController(
            ceiling_mbps=float(total_ceiling_mbps),
            floor_mbps=minimum_total_mbps,
            initial_total_mbps=max(previous_total, minimum_total_mbps),
            preview_demand_mbps=preview_demand,
        )
        initial_rate_mbps = controller.budget.high_fidelity_mbps
    else:
        initial_rate_mbps = rate_mbps
    bucket = TokenBucket(initial_rate_mbps * 1_000_000.0 / 8.0, burst_seconds=0.05)
    chunks = _read_chunks(source, chunk_size_bytes, chunk_count)
    controller_lock = threading.Lock()
    ack_window = VerifiedAckGoodputWindow(minimum_interval_s=5.0)
    probe_state = {"last_at": None, "observed_mbps": None, "samples": 0}

    def refresh_preview_demand() -> None:
        if controller is None or preview_demand_store is None:
            return
        required_mbps = preview_demand_store.read()
        # Keep the bootstrap floor during a heartbeat gap so bulk data cannot
        # consume the camera's real-time allocation.
        fallback = float(os.environ.get("GO2_MAPPING_PREVIEW_BOOTSTRAP_MBPS", "1.0"))
        controller.set_preview_demand(fallback if required_mbps is None else required_mbps)

    existing = _artifact_status(parsed, session_id, artifact_id, auth_token=auth_token)
    if _release_authorized(existing, relative.as_posix(), artifact_sha256, size_bytes):
        return _completed_response(existing, chunk_count, controller, rate_mbps, parallel_workers, probe_state, route_id)
    if await_reconstruction_s > 0 and _verified_identity(
        existing,
        relative.as_posix(),
        artifact_sha256,
        size_bytes,
    ):
        authorized = _wait_for_reconstruction(
            parsed,
            session_id,
            artifact_id,
            relative.as_posix(),
            artifact_sha256,
            size_bytes,
            timeout_s=await_reconstruction_s,
            auth_token=auth_token,
        )
        return _completed_response(authorized, chunk_count, controller, rate_mbps, parallel_workers, probe_state, route_id)

    def send_chunk(index: int, chunk: bytes, force_probe: bool = False) -> Dict[str, object]:
        now = time.monotonic()
        with controller_lock:
            refresh_preview_demand()
            probe = controller is not None and (
                force_probe
                or _capacity_probe_due(
                    last_probe_at=probe_state["last_at"],
                    now=now,
                    interval_s=capacity_probe_interval_s,
                )
            )
            if probe:
                probe_state["last_at"] = now
            allocation = controller.budget if controller is not None else None
            observed = probe_state["observed_mbps"]
        # A probe is separate from the artifact chunk that triggered it; both
        # requests are required before the artifact can be sealed.
        if probe:
            probe_started_at = time.monotonic()
            _put_probe(
                parsed,
                session_id,
                bytes(probe_chunk_bytes),
                allocation,
                auth_token=auth_token,
            )
            probe_elapsed_s = max(time.monotonic() - probe_started_at, 1e-6)
            if controller is not None and allocation_store is not None:
                observed_mbps = probe_chunk_bytes * 8.0 / probe_elapsed_s / 1_000_000.0
                with controller_lock:
                    budget = controller.observe_capacity_probe(
                        byte_count=probe_chunk_bytes,
                        elapsed_s=probe_elapsed_s,
                    )
                    probe_state["observed_mbps"] = observed_mbps
                    probe_state["samples"] += 1
                    allocation_store.publish(
                        budget,
                        observed_high_fidelity_mbps=observed_mbps,
                        sample_elapsed_s=probe_elapsed_s,
                        route_id=route_id,
                    )
                    bucket.set_rate(budget.high_fidelity_bytes_per_second)

        with controller_lock:
            refresh_preview_demand()
            current_allocation = controller.budget if controller is not None else None
            observed = probe_state["observed_mbps"]
            if current_allocation is not None and current_allocation.high_fidelity_mbps <= 0:
                raise UploadError("high-fidelity transfer is paused while real-time preview consumes the measured capacity")
        response, throttle_sleep_s, ack_rtt_ms = _put_chunk(
                parsed,
                session_id,
                artifact_id,
                index,
                chunk,
                bucket,
                current_allocation,
                observed,
                throttle=True,
                auth_token=auth_token,
            )
        if controller is not None and allocation_store is not None:
            sample = ack_window.note_verified(len(chunk))
            with controller_lock:
                if sample is not None:
                    budget = controller.observe_high_fidelity_transfer(
                        byte_count=sample.byte_count,
                        elapsed_s=sample.elapsed_s,
                    )
                    observed_mbps = sample.goodput_mbps
                    probe_state["observed_mbps"] = observed_mbps
                else:
                    budget = controller.budget
                    observed_mbps = float(probe_state["observed_mbps"] or 0.0)
                allocation_store.publish(
                    budget,
                    observed_high_fidelity_mbps=observed_mbps,
                    sample_elapsed_s=(sample.elapsed_s if sample is not None else 0.0),
                    route_id=route_id,
                    ack_rtt_ms=ack_rtt_ms,
                )
                bucket.set_rate(budget.high_fidelity_bytes_per_second)
        return response

    send_chunk(0, chunks[0], force_probe=True)
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers, thread_name_prefix="go2-upload") as executor:
            futures = [executor.submit(send_chunk, index, chunk) for index, chunk in enumerate(chunks[1:], start=1)]
            for future in as_completed(futures):
                future.result()

    manifest = {
        "relative_path": relative.as_posix(),
        "artifact_sha256": artifact_sha256,
        "size_bytes": size_bytes,
        "chunk_count": chunk_count,
    }
    response = _json_request(
        parsed,
        "POST",
        f"/progressive/artifacts/{quote(session_id)}/{quote(artifact_id)}/seal",
        manifest,
        auth_token=auth_token,
    )
    if await_reconstruction_s > 0 and relative.suffixes[-2:] == [".db3", ".zstd"]:
        response = _wait_for_reconstruction(
            parsed,
            session_id,
            artifact_id,
            relative.as_posix(),
            artifact_sha256,
            size_bytes,
            timeout_s=await_reconstruction_s,
            auth_token=auth_token,
        )
    return _completed_response(response, chunk_count, controller, rate_mbps, parallel_workers, probe_state, route_id)


def _endpoint(endpoint: str):
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint must be an http or https URL without embedded credentials")
    return parsed


def _connection(parsed) -> http.client.HTTPConnection:
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=_SOCKET_TIMEOUT_SECONDS)
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=_SOCKET_TIMEOUT_SECONDS)


def _auth_headers(auth_token: Optional[str]) -> Dict[str, str]:
    if not auth_token:
        return {}
    if "\r" in auth_token or "\n" in auth_token or len(auth_token) > 512:
        raise ValueError("auth_token contains invalid characters")
    return {"Authorization": f"Bearer {auth_token}"}


def _path(parsed, suffix: str) -> str:
    prefix = parsed.path.rstrip("/")
    return f"{prefix}{suffix}" or suffix


def _capacity_elapsed_s(wall_elapsed_s: float, throttle_sleep_s: float) -> float:
    if wall_elapsed_s <= 0 or throttle_sleep_s < 0:
        raise ValueError("transfer timing values are invalid")
    return max(wall_elapsed_s - min(throttle_sleep_s, wall_elapsed_s), 1e-3)


def _capacity_probe_due(*, last_probe_at: Optional[float], now: float, interval_s: float) -> bool:
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    return last_probe_at is None or now - last_probe_at >= interval_s


def _put_chunk(
    parsed,
    session_id: str,
    artifact_id: str,
    index: int,
    chunk: bytes,
    bucket: TokenBucket,
    allocation=None,
    observed_capacity_mbps: Optional[float] = None,
    throttle: bool = True,
    auth_token: Optional[str] = None,
) -> Tuple[Dict[str, object], float, float]:
    digest = hashlib.sha256(chunk).hexdigest()
    connection = _connection(parsed)
    try:
        connection.putrequest("PUT", _path(parsed, f"/progressive/chunks/{quote(session_id)}/{quote(artifact_id)}/{index}"))
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(len(chunk)))
        connection.putheader("X-Chunk-SHA256", digest)
        for name, value in _auth_headers(auth_token).items():
            connection.putheader(name, value)
        if allocation is not None:
            connection.putheader("X-Go2-Total-Bandwidth-Mbps", f"{allocation.total_mbps:.6f}")
            connection.putheader("X-Go2-Preview-Bandwidth-Mbps", f"{allocation.preview_mbps:.6f}")
            connection.putheader("X-Go2-High-Fidelity-Bandwidth-Mbps", f"{allocation.high_fidelity_mbps:.6f}")
        if observed_capacity_mbps is not None:
            connection.putheader("X-Go2-Observed-Capacity-Mbps", f"{observed_capacity_mbps:.6f}")
        connection.endheaders()
        throttle_sleep_s = 0.0
        for offset in range(0, len(chunk), _SEND_BLOCK_BYTES):
            block = chunk[offset : offset + _SEND_BLOCK_BYTES]
            delay = bucket.delay_for(len(block)) if throttle else 0.0
            if delay:
                sleep_started_at = time.monotonic()
                time.sleep(delay)
                throttle_sleep_s += time.monotonic() - sleep_started_at
            connection.send(block)
        ack_started_at = time.monotonic()
        response = _response_json(connection)
        return response, throttle_sleep_s, (time.monotonic() - ack_started_at) * 1000.0
    finally:
        connection.close()


def _put_probe(
    parsed,
    session_id: str,
    probe: bytes,
    allocation=None,
    *,
    auth_token: Optional[str] = None,
) -> Tuple[Dict[str, object], float]:
    if len(probe) != PROBE_CHUNK_BYTES:
        raise ValueError("capacity probe must be exactly 256 KiB")
    digest = hashlib.sha256(probe).hexdigest()
    connection = _connection(parsed)
    try:
        connection.putrequest("POST", _path(parsed, f"/progressive/probe/{quote(session_id)}"))
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(len(probe)))
        connection.putheader("X-Probe-SHA256", digest)
        for name, value in _auth_headers(auth_token).items():
            connection.putheader(name, value)
        connection.endheaders()
        connection.send(probe)
        return _response_json(connection), 0.0
    finally:
        connection.close()


def _json_request(
    parsed,
    method: str,
    suffix: str,
    payload: Dict[str, object],
    *,
    auth_token: Optional[str] = None,
) -> Dict[str, object]:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    connection = _connection(parsed)
    try:
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        headers.update(_auth_headers(auth_token))
        connection.request(
            method,
            _path(parsed, suffix),
            body=body,
            headers=headers,
        )
        return _response_json(connection)
    finally:
        connection.close()


def _artifact_status(
    parsed,
    session_id: str,
    artifact_id: str,
    *,
    auth_token: Optional[str] = None,
) -> Dict[str, object]:
    connection = _connection(parsed)
    try:
        connection.request(
            "GET",
            _path(parsed, f"/progressive/artifacts/{quote(session_id)}/{quote(artifact_id)}/status"),
            headers=_auth_headers(auth_token),
        )
        response = connection.getresponse()
        raw = response.read()
        if response.status == 404:
            return {"state": "UNKNOWN", "release_authorized": False}
        if response.status != 200:
            raise UploadError(f"receiver rejected lifecycle status with HTTP {response.status}")
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict):
            raise UploadError("receiver returned a non-object lifecycle status")
        return body
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UploadError("receiver returned invalid lifecycle JSON") from exc
    finally:
        connection.close()


def _release_authorized(status: Dict[str, object], relative_path: str, digest: str, size_bytes: int) -> bool:
    return bool(
        status.get("release_authorized") is True
        and status.get("state") == "PC_RECONSTRUCTION_INGESTED"
        and status.get("relative_path") == relative_path
        and status.get("artifact_sha256") == digest
        and status.get("size_bytes") == size_bytes
    )


def _verified_identity(status: Dict[str, object], relative_path: str, digest: str, size_bytes: int) -> bool:
    return bool(
        status.get("state") in {"SHA256_VERIFIED", "PC_RECONSTRUCTION_INGESTED"}
        and status.get("relative_path") == relative_path
        and status.get("artifact_sha256") == digest
        and status.get("size_bytes") == size_bytes
    )


def _wait_for_reconstruction(
    parsed,
    session_id: str,
    artifact_id: str,
    relative_path: str,
    digest: str,
    size_bytes: int,
    *,
    timeout_s: float,
    auth_token: Optional[str] = None,
) -> Dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while True:
        status = _artifact_status(parsed, session_id, artifact_id, auth_token=auth_token)
        if _release_authorized(status, relative_path, digest, size_bytes):
            return status
        if time.monotonic() >= deadline:
            raise UploadError("PC reconstruction did not authorize artifact release before timeout")
        time.sleep(min(2.0, max(0.05, deadline - time.monotonic())))


def _completed_response(
    response: Dict[str, object],
    chunk_count: int,
    controller,
    rate_mbps: float,
    parallel_workers: int,
    probe_state: Dict[str, object],
    route_id: str,
) -> Dict[str, object]:
    completed = dict(response)
    completed["chunk_count"] = chunk_count
    completed["parallel_workers"] = parallel_workers
    completed["capacity_probe"] = {
        "unthrottled": bool(probe_state["samples"]),
        "samples": probe_state["samples"],
        "observed_mbps": probe_state["observed_mbps"],
    }
    if controller is not None:
        completed["bandwidth"] = {
            "dynamic": True,
            "total_mbps": controller.budget.total_mbps,
            "preview_mbps": controller.budget.preview_mbps,
            "high_fidelity_mbps": controller.budget.high_fidelity_mbps,
            "preview_policy": controller.budget.preview_policy,
            "preview_demand_mbps": controller.budget.preview_demand_mbps,
            "preview_share": controller.budget.effective_preview_share,
            "high_fidelity_share": controller.budget.effective_high_fidelity_share,
            "route_id": route_id,
        }
    else:
        completed["bandwidth"] = {"dynamic": False, "high_fidelity_mbps": rate_mbps}
    return completed


def _read_chunks(source: Path, chunk_size_bytes: int, chunk_count: int) -> list[bytes]:
    chunks = []
    with source.open("rb") as input_file:
        for index in range(chunk_count):
            chunk = input_file.read(chunk_size_bytes)
            if index < chunk_count - 1 and len(chunk) != chunk_size_bytes:
                raise UploadError("source changed while uploading")
            chunks.append(chunk)
        if input_file.read(1):
            raise UploadError("source changed while uploading")
    return chunks


def _response_json(connection: http.client.HTTPConnection) -> Dict[str, object]:
    response = connection.getresponse()
    raw = response.read()
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UploadError(f"receiver returned invalid JSON with HTTP {response.status}") from exc
    # Artifact chunks use 201/208. The capacity probe uses 200 because it does
    # not create an artifact.
    if response.status not in (200, 201, 208):
        raise UploadError(f"receiver rejected artifact with HTTP {response.status}: {body.get('error', body)}")
    if not isinstance(body, dict):
        raise UploadError("receiver returned a non-object JSON body")
    return body


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload one immutable Go2 mapping artifact in verified chunks")
    parser.add_argument("source", type=Path)
    parser.add_argument("relative_path")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("GO2_MAPPING_HIGH_FIDELITY_URL", "http://192.0.2.10:8000"),
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--chunk-mib", type=int)
    parser.add_argument("--chunk-kib", type=int, default=256)
    parser.add_argument("--rate-mbps", type=float, required=True)
    parser.add_argument("--bandwidth-state", type=Path)
    parser.add_argument("--preview-demand-state", type=Path)
    parser.add_argument("--total-ceiling-mbps", type=float)
    parser.add_argument("--route-id", default=os.environ.get("GO2_MAPPING_ROUTE_ID"))
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--capacity-probe-interval", type=float, default=30.0)
    parser.add_argument("--probe-chunk-kib", type=int, default=256)
    parser.add_argument("--await-reconstruction", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--auth-token", default=os.environ.get("GO2_MAPPING_BULK_AUTH_TOKEN"))
    args = parser.parse_args(argv)
    if args.chunk_mib is not None and args.chunk_mib <= 0:
        parser.error("--chunk-mib must be positive")
    if args.chunk_kib <= 0:
        parser.error("--chunk-kib must be positive")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    result = upload_artifact(
        args.endpoint,
        session_id=args.session_id,
        source=args.source,
        relative_path=args.relative_path,
        chunk_size_bytes=(args.chunk_mib * 1024 * 1024 if args.chunk_mib is not None else args.chunk_kib * 1024),
        rate_mbps=args.rate_mbps,
        bandwidth_state_path=args.bandwidth_state,
        preview_demand_state_path=args.preview_demand_state,
        total_ceiling_mbps=args.total_ceiling_mbps,
        route_id=args.route_id,
        parallel_workers=args.parallel_workers,
        capacity_probe_interval_s=args.capacity_probe_interval,
        await_reconstruction_s=args.await_reconstruction,
        auth_token=args.auth_token,
        probe_chunk_bytes=args.probe_chunk_kib * 1024,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
