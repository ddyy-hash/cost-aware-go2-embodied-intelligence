"""Small HTTP and HTTPS CONNECT proxy for the wired Go2 link."""

import argparse
import select
import socket
import socketserver
import sys
from urllib.parse import urlsplit


BUFFER_SIZE = 65536
HEADER_LIMIT = 65536
DEFAULT_TIMEOUT_S = 15.0


class ProxyError(Exception):
    pass


class ThreadedTcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(DEFAULT_TIMEOUT_S)
        try:
            request_line, headers, leftover = self._read_request()
            method, target, version = self._parse_request_line(request_line)
            if method.upper() == "CONNECT":
                self._handle_connect(target, leftover)
            else:
                self._handle_http(method, target, version, headers, leftover)
        except Exception as exc:
            self._send_error(exc)

    def _read_request(self):
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = self.request.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > HEADER_LIMIT:
                raise ProxyError("request header is too large")

        header_bytes = bytes(data)
        header_end = header_bytes.find(b"\r\n\r\n")
        if header_end < 0:
            raise ProxyError("incomplete request header")

        header_text = header_bytes[:header_end].decode("iso-8859-1")
        leftover = header_bytes[header_end + 4 :]
        lines = header_text.split("\r\n")
        request_line = lines[0]
        headers = []
        for line in lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers.append((name.strip(), value.strip()))
        return request_line, headers, leftover

    @staticmethod
    def _parse_request_line(line):
        parts = line.split()
        if len(parts) != 3:
            raise ProxyError("invalid request line")
        return parts[0], parts[1], parts[2]

    def _handle_connect(self, target, leftover):
        host, port = self._split_host_port(target, 443)
        upstream = socket.create_connection((host, port), timeout=DEFAULT_TIMEOUT_S)
        with upstream:
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            if leftover:
                upstream.sendall(leftover)
            self._tunnel(self.request, upstream)

    def _handle_http(self, method, target, version, headers, leftover):
        parsed = urlsplit(target)
        if not parsed.scheme or not parsed.netloc:
            raise ProxyError("only absolute proxy URLs are supported")
        if parsed.scheme.lower() != "http":
            raise ProxyError("unsupported URL scheme")

        host, port = self._split_host_port(parsed.netloc, 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        upstream = socket.create_connection((host, port), timeout=DEFAULT_TIMEOUT_S)
        with upstream:
            upstream.settimeout(DEFAULT_TIMEOUT_S)
            outgoing = [f"{method} {path} {version}"]
            has_host = False
            for name, value in headers:
                lower_name = name.lower()
                if lower_name == "host":
                    has_host = True
                if lower_name in ("proxy-connection", "connection"):
                    continue
                outgoing.append(f"{name}: {value}")
            if not has_host:
                outgoing.append(f"Host: {parsed.netloc}")
            outgoing.append("Connection: close")
            outgoing.append("")
            outgoing.append("")
            upstream.sendall("\r\n".join(outgoing).encode("iso-8859-1"))
            if leftover:
                upstream.sendall(leftover)
            self._copy_response(upstream, self.request)

    @staticmethod
    def _split_host_port(value, default_port):
        if value.startswith("["):
            host, _, tail = value[1:].partition("]")
            if tail.startswith(":"):
                return host, int(tail[1:])
            return host, default_port
        if ":" in value:
            host, port_text = value.rsplit(":", 1)
            return host, int(port_text)
        return value, default_port

    @staticmethod
    def _copy_response(source, target):
        while True:
            data = source.recv(BUFFER_SIZE)
            if not data:
                return
            target.sendall(data)

    @staticmethod
    def _tunnel(left, right):
        sockets = [left, right]
        peers = {left: right, right: left}
        while True:
            readable, _, errored = select.select(sockets, [], sockets, DEFAULT_TIMEOUT_S)
            if errored or not readable:
                return
            for sock in readable:
                data = sock.recv(BUFFER_SIZE)
                other = peers[sock]
                if not data:
                    sockets.remove(sock)
                    try:
                        other.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    if not sockets:
                        return
                    continue
                other.sendall(data)

    def _send_error(self, exc):
        message = str(exc) or exc.__class__.__name__
        sys.stderr.write(f"proxy error from {self.client_address[0]}: {message}\n")
        sys.stderr.flush()
        try:
            body = message.encode("utf-8", errors="replace")
            response = (
                b"HTTP/1.1 502 Bad Gateway\r\n"
                + f"Content-Length: {len(body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
                + body
            )
            self.request.sendall(response)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3128)
    args = parser.parse_args()

    with ThreadedTcpServer((args.host, args.port), ProxyHandler) as server:
        print(f"proxy listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
