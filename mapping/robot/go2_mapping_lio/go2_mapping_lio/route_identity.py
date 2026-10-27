"""Stable identity for invalidating bandwidth estimates after route changes."""
from __future__ import annotations

import argparse
import json
import re
from typing import Iterable, Optional


def _clean(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:@-]+", "_", str(value or "unknown").strip())
    return cleaned[:200] or "unknown"


def route_identity(status_json: str, peer_ip: str, wifi_identity: str = "unknown") -> str:
    route = "unknown"
    try:
        payload = json.loads(status_json)
        peers = payload.get("Peer", {})
        if isinstance(peers, dict):
            for peer in peers.values():
                if not isinstance(peer, dict) or peer_ip not in peer.get("TailscaleIPs", []):
                    continue
                current = str(peer.get("CurAddr", "")).strip()
                relay = str(peer.get("Relay", "")).strip()
                if current:
                    route = f"direct:{_clean(current)}"
                elif relay:
                    route = f"derp:{_clean(relay)}"
                break
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return f"wifi:{_clean(wifi_identity)}|tailscale:{route}"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-ip", required=True)
    parser.add_argument("--wifi-identity", default="unknown")
    parser.add_argument("--status-file")
    args = parser.parse_args(argv)
    if args.status_file:
        with open(args.status_file, "r", encoding="utf-8") as source:
            status_json = source.read()
    else:
        import sys
        status_json = sys.stdin.read()
    print(route_identity(status_json, args.peer_ip, args.wifi_identity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
