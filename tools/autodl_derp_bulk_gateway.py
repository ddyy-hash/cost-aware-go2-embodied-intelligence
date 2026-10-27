#!/usr/bin/env python3
"""Multiplex authenticated bulk HTTP and DERP Upgrade on one AutoDL port.

The AutoDL HTTPS proxy exposes one public port. DERP remains a raw upgraded
stream to the loopback derper, while normal /progressive HTTP requests are
forwarded to the checksum-verifying spool receiver after Bearer validation.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
from pathlib import Path


MAX_HEADER_BYTES = 64 * 1024
CHUNK_BYTES = 256 * 1024


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(CHUNK_BYTES)
            if not data:
                return
            writer.write(data)
            await writer.drain()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def _headers(header: bytes) -> dict[bytes, bytes]:
    result = {}
    for line in header.split(b"\r\n")[1:]:
        if b":" not in line:
            continue
        name, value = line.split(b":", 1)
        result[name.strip().lower()] = value.strip()
    return result


async def _reject(writer: asyncio.StreamWriter, status: bytes, message: bytes) -> None:
    body = message + b"\n"
    writer.write(
        b"HTTP/1.1 " + status + b"\r\nContent-Type: text/plain\r\nContent-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\nConnection: close\r\n\r\n"
        + body
    )
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def _handle(
    client_r: asyncio.StreamReader,
    client_w: asyncio.StreamWriter,
    *,
    derp_backend: tuple[str, int],
    http_backend: tuple[str, int],
    token: bytes,
) -> None:
    backend_w = None
    try:
        header = await asyncio.wait_for(client_r.readuntil(b"\r\n\r\n"), timeout=15)
        if len(header) > MAX_HEADER_BYTES:
            await _reject(client_w, b"431 Request Header Fields Too Large", b"header too large")
            return
        first_line = header.split(b"\r\n", 1)[0]
        fields = _headers(header)
        is_derp = first_line.upper().startswith(b"GET ") and fields.get(b"upgrade", b"").lower() == b"derp"
        if is_derp:
            backend_r, backend_w = await asyncio.open_connection(*derp_backend)
            backend_w.write(header)
            await backend_w.drain()
            await asyncio.gather(_pipe(client_r, backend_w), _pipe(backend_r, client_w))
            return

        supplied = fields.get(b"authorization", b"")
        expected = b"Bearer " + token
        if not hmac.compare_digest(supplied, expected):
            await _reject(client_w, b"401 Unauthorized", b"bulk authentication required")
            return
        path = first_line.split(b" ", 2)[1] if len(first_line.split(b" ", 2)) >= 2 else b""
        if not (path.startswith(b"/progressive") or path == b"/health"):
            await _reject(client_w, b"404 Not Found", b"unknown bulk path")
            return
        length = int(fields.get(b"content-length", b"0") or b"0")
        if length < 0 or length > 64 * 1024 * 1024:
            await _reject(client_w, b"413 Request Entity Too Large", b"invalid content length")
            return
        backend_r, backend_w = await asyncio.open_connection(*http_backend)
        backend_w.write(header)
        await backend_w.drain()
        remaining = length
        while remaining:
            data = await client_r.read(min(CHUNK_BYTES, remaining))
            if not data:
                raise ConnectionError("truncated client request")
            backend_w.write(data)
            await backend_w.drain()
            remaining -= len(data)
        await _pipe(backend_r, client_w)
    except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError, OSError, ValueError):
        pass
    finally:
        for writer in (backend_w, client_w):
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass


async def _serve(args: argparse.Namespace) -> None:
    token = Path(args.token_file).read_bytes().strip()
    if not token or b"\r" in token or b"\n" in token:
        raise SystemExit("bulk token file is empty or invalid")
    server = await asyncio.start_server(
        lambda r, w: _handle(
            r,
            w,
            derp_backend=(args.bind, args.derp_port),
            http_backend=(args.bind, args.http_port),
            token=token,
        ),
        args.bind,
        args.listen_port,
    )
    print(
        f"gateway listening on {args.bind}:{args.listen_port}; "
        f"derp={args.derp_port} bulk={args.http_port}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=6006)
    parser.add_argument("--derp-port", type=int, default=6009)
    parser.add_argument("--http-port", type=int, default=6008)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
