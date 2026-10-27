#!/usr/bin/env python3
"""Measure checksum-neutral TCP goodput over a selected Tailnet path."""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Iterable


HEADER_LIMIT = 4096
CHUNK_SIZE = 256 * 1024


def _read_line(connection: socket.socket) -> bytes:
    data = bytearray()
    while len(data) < HEADER_LIMIT:
        chunk = connection.recv(1)
        if not chunk:
            raise RuntimeError("connection closed before request header")
        if chunk == b"\n":
            return bytes(data)
        data.extend(chunk)
    raise RuntimeError("request header exceeds limit")


def _mbps(byte_count: int, elapsed_s: float) -> float:
    return byte_count * 8.0 / max(elapsed_s, 1e-9) / 1_000_000.0


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated bytes")
    return sizes


def run_server(bind: str, port: int, sessions: int, timeout_s: float) -> int:
    completed = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind, port))
        listener.listen(4)
        listener.settimeout(timeout_s)
        print(json.dumps({"event": "listening", "bind": bind, "port": port}), flush=True)
        while completed < sessions:
            connection, address = listener.accept()
            with connection:
                connection.settimeout(timeout_s)
                try:
                    request = json.loads(_read_line(connection).decode("ascii"))
                except (ConnectionError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
                    # Tailscale health checks can open and close a TCP socket
                    # without sending application data. They are not samples.
                    print(
                        json.dumps({"event": "ignored_connection", "peer": address[0], "reason": str(exc)}),
                        flush=True,
                    )
                    continue
                expected = int(request["bytes"])
                run = int(request["run"])
                if expected <= 0:
                    raise RuntimeError("requested byte count must be positive")

                received = 0
                started = time.perf_counter()
                while received < expected:
                    chunk = connection.recv(min(CHUNK_SIZE, expected - received))
                    if not chunk:
                        raise RuntimeError(
                            f"connection closed at {received} of {expected} bytes"
                        )
                    received += len(chunk)
                elapsed = time.perf_counter() - started
                result = {
                    "event": "received",
                    "run": run,
                    "bytes": received,
                    "elapsed_s": round(elapsed, 6),
                    "goodput_mbps": round(_mbps(received, elapsed), 3),
                    "peer": address[0],
                }
                connection.sendall((json.dumps(result) + "\n").encode("ascii"))
                print(json.dumps(result), flush=True)
                completed += 1
    return 0


def _runs(sizes: Iterable[int], repeats: int) -> Iterable[tuple[int, int]]:
    run = 0
    for _ in range(repeats):
        for size in sizes:
            run += 1
            yield run, size


def run_client(host: str, port: int, sizes: list[int], repeats: int, timeout_s: float) -> int:
    payload = bytes(CHUNK_SIZE)
    for run, size in _runs(sizes, repeats):
        started = time.perf_counter()
        with socket.create_connection((host, port), timeout=timeout_s) as connection:
            connection.settimeout(timeout_s)
            request = json.dumps({"run": run, "bytes": size}).encode("ascii") + b"\n"
            connection.sendall(request)
            remaining = size
            while remaining:
                count = min(len(payload), remaining)
                connection.sendall(payload[:count])
                remaining -= count
            response = json.loads(_read_line(connection).decode("ascii"))
        elapsed = time.perf_counter() - started
        result = {
            "event": "sent",
            "run": run,
            "bytes": size,
            "elapsed_s": round(elapsed, 6),
            "goodput_mbps": round(_mbps(size, elapsed), 3),
            "receiver_goodput_mbps": response["goodput_mbps"],
        }
        print(json.dumps(result), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("server")
    server.add_argument("--bind", required=True)
    server.add_argument("--port", type=int, default=19081)
    server.add_argument("--sessions", type=int, required=True)
    server.add_argument("--timeout-s", type=float, default=180.0)

    client = subparsers.add_parser("client")
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, default=19081)
    client.add_argument("--sizes", type=_parse_sizes, required=True)
    client.add_argument("--repeats", type=int, default=1)
    client.add_argument("--timeout-s", type=float, default=180.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "server":
        return run_server(args.bind, args.port, args.sessions, args.timeout_s)
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    return run_client(args.host, args.port, args.sizes, args.repeats, args.timeout_s)


if __name__ == "__main__":
    raise SystemExit(main())
