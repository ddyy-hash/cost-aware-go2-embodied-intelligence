#!/usr/bin/env python3
"""Measure HTTP POST ingress with exact Content-Length accounting."""

from __future__ import annotations

import argparse
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class IngressHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            expected = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            expected = 0
        if expected <= 0:
            self.send_error(411, "Content-Length required")
            return

        received = 0
        started = time.perf_counter()
        while received < expected:
            chunk = self.rfile.read(min(256 * 1024, expected - received))
            if not chunk:
                break
            received += len(chunk)
        elapsed = time.perf_counter() - started
        result = {
            "bytes": received,
            "expected": expected,
            "elapsed_s": round(elapsed, 6),
            "goodput_mbps": round(received * 8.0 / max(elapsed, 1e-9) / 1_000_000.0, 3),
            "peer": self.client_address[0],
            "path": self.path,
        }
        body = (json.dumps(result) + "\n").encode("ascii")
        self.send_response(200 if received == expected else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        print(json.dumps(result), flush=True)

    def log_message(self, *_args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6008)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), IngressHandler)
    print(json.dumps({"event": "listening", "bind": args.bind, "port": args.port}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
