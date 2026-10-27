#!/usr/bin/env python3
"""Temporary HTTP/DERP multiplexer for a single AutoDL public port."""

from __future__ import annotations

import asyncio
import sys
from typing import Tuple


LISTEN: Tuple[str, int] = ("127.0.0.1", 6006)
DERP: Tuple[str, int] = ("127.0.0.1", 6009)
HTTP: Tuple[str, int] = ("127.0.0.1", 6008)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(256 * 1024)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
    backend_w = None
    try:
        header = await client_r.readuntil(b"\r\n\r\n")
        first = header.split(b"\r\n", 1)[0].upper()
        is_derp = b"UPGRADE: DERP" in header.upper() and first.startswith(b"GET ")
        backend = DERP if is_derp else HTTP
        backend_r, backend_w = await asyncio.open_connection(*backend)
        backend_w.write(header)
        await backend_w.drain()
        if is_derp:
            await asyncio.gather(pipe(client_r, backend_w), pipe(backend_r, client_w))
        else:
            # POST bodies and normal HTTP responses remain byte-for-byte intact.
            await pipe(client_r, backend_w)
            await pipe(backend_r, client_w)
    except (asyncio.IncompleteReadError, ConnectionError, OSError):
        pass
    finally:
        for writer in (backend_w, client_w):
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass


async def main() -> None:
    server = await asyncio.start_server(handle, *LISTEN)
    print({"listen": LISTEN, "derp": DERP, "http": HTTP}, flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
