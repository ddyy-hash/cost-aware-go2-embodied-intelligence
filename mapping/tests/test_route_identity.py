import json

from mapping.robot.go2_mapping_lio.go2_mapping_lio.route_identity import route_identity


def test_route_identity_distinguishes_direct_and_derp_paths():
    direct = {
        "Peer": {
            "node-key": {
                "TailscaleIPs": ["192.0.2.10"],
                "CurAddr": "203.0.113.8:41641",
                "Relay": "",
            }
        }
    }
    derp = {
        "Peer": {
            "node-key": {
                "TailscaleIPs": ["192.0.2.10"],
                "CurAddr": "",
                "Relay": "autodl-westb",
            }
        }
    }

    assert route_identity(json.dumps(direct), "192.0.2.10", "lab-wifi") == (
        "wifi:lab-wifi|tailscale:direct:203.0.113.8:41641"
    )
    assert route_identity(json.dumps(derp), "192.0.2.10", "lab-wifi") == (
        "wifi:lab-wifi|tailscale:derp:autodl-westb"
    )


def test_route_identity_is_single_line_and_fails_closed_when_peer_is_unknown():
    identity = route_identity("{}", "192.0.2.10", "bad\nssid")

    assert identity == "wifi:bad_ssid|tailscale:unknown"
    assert "\n" not in identity
