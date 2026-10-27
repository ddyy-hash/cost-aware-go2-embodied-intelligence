from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOTE = (ROOT / "tools" / "manage_autodl_derp.sh").read_text(encoding="utf-8")
PC = (ROOT / "tools" / "manage_autodl_derp.ps1").read_text(encoding="utf-8")
PC_STACK = (ROOT / "tools" / "start_go2_pc_stack.ps1").read_text(encoding="utf-8")
UPLINK = (ROOT / "mapping" / "config" / "uplink.yaml").read_text(encoding="utf-8")


def test_autodl_derper_is_loopback_only_and_verifies_tailnet_clients():
    assert "127.0.0.1:6006" in REMOTE
    assert "127.0.0.1:6009" in REMOTE
    assert "127.0.0.1:6008" in REMOTE
    assert "autodl_derp_bulk_gateway.py" in REMOTE
    assert "bulk-auth-token" in REMOTE
    assert "-verify-clients" in REMOTE
    assert "-socket=\"$tailscaled_socket\"" in REMOTE
    assert "-stun=false" in REMOTE
    assert "-http-port=-1" in REMOTE
    assert "--tun=userspace-networking" in REMOTE
    assert "attempt < 60" in REMOTE
    assert "did not restore its authorized state" in REMOTE


def test_autodl_manager_is_idempotent_and_has_a_bounded_rollback():
    assert "pid_is_live" in REMOTE
    assert '"/proc/$pid/cmdline"' in REMOTE
    assert "expected_executable" in REMOTE
    assert "start|status|stop|stop-all" in REMOTE
    assert "killall" not in REMOTE
    assert "pkill" not in REMOTE
    assert "rm -rf" not in REMOTE
    assert "Set-PreferredRegion 0" in PC
    assert "AUTODL_DERP_ROLLED_BACK" in PC


def test_autodl_manager_replaces_only_stale_owned_tailscaled_instances():
    assert "terminate_owned_tailscaled_instances" in REMOTE
    assert "for proc_dir in /proc/[0-9]*" in REMOTE
    assert '"--state=$tailscaled_state"' in REMOTE
    assert '"--socket=$tailscaled_socket"' in REMOTE
    assert REMOTE.index("terminate_owned_tailscaled_instances") < REMOTE.index(
        'rm -f "$tailscaled_socket"'
    )
    assert "killall" not in REMOTE
    assert "pkill" not in REMOTE


def test_pc_manager_discovers_the_current_public_endpoint_and_preserves_tls_upgrade():
    assert "101 Switching Protocols" in PC
    assert "Derp-Public-Key:" in PC
    assert "function Resolve-PublicDerpEndpoint" in PC
    assert "autodl-container-([a-z0-9]+)-([a-z0-9]+)" in PC
    assert "Substring($compactContainerId.Length - 12)" in PC
    assert "AUTODL_DERP_PUBLIC_ENDPOINT url=" in PC
    assert "[string]$AutoDLPublicUser = 'u692021'" in PC
    assert "[string]$PublicDerpHost = ''" in PC
    assert "OmitDefaultRegions = $false" in PC
    assert "127.0.0.1:${RemoteControlProxyPort}:127.0.0.1:${LocalProxyPort}" in PC
    assert "ServerAliveInterval=15" in PC
    assert "ServerAliveCountMax=3" in PC
    assert "[int]$PreferredDerpRegion = 0" in PC
    assert "if ($PreferredDerpRegion -gt 0)" in PC


def test_pc_manager_does_not_return_curl_timeout_after_a_valid_derp_upgrade():
    assert "$curlExitCode = $LASTEXITCODE" in PC
    assert "$global:LASTEXITCODE = 0" in PC
    assert PC.index("if ($response -notmatch '101 Switching Protocols'") < PC.index(
        "$global:LASTEXITCODE = 0"
    )


def test_pc_manager_reports_a_stale_tailnet_map_without_mutating_tailnet_policy():
    assert "AUTODL_DERP_MAP_MATCH" in PC
    assert "AUTODL_DERP_POLICY_FRAGMENT" in PC
    assert "tailscale.exe debug derp-map" in PC
    assert "Invoke-RestMethod" not in PC


def test_pc_manager_refuses_to_force_a_stale_autodl_derp_map():
    assert "Refusing to force region 901 while the Tailnet DERP map is stale" in PC
    assert PC.index("if (-not $script:tailnetMapMatches)") < PC.index(
        "Set-PreferredRegion $PreferredDerpRegion"
    )


def test_pc_stack_uses_the_runtime_discovered_autodl_bulk_endpoint():
    assert "[string]$BulkEndpoint = ''" in PC_STACK
    assert "AUTODL_DERP_PUBLIC_ENDPOINT url=" in PC_STACK
    assert "Could not read the runtime AutoDL public endpoint" in PC_STACK


def test_pc_stack_defers_autodl_preference_until_the_runtime_map_is_verified():
    assert "Deferred AutoDL DERP preference until runtime mapping verification completes" in PC_STACK
    assert "Current AutoDL region 901 map did not match the discovered endpoint" in PC_STACK
    assert "if ($PreferredDerpRegion -gt 0 -and $HighFidelityTransport -eq 'autodl')" in PC_STACK
    assert "^AUTODL_DERP_MAP_MATCH region=901 endpoint=" in PC_STACK


def test_uplink_configuration_marks_autodl_public_endpoint_as_runtime_discovered():
    assert "public_endpoint: runtime_discovered_from_autodl_container" in UPLINK
    assert "u692021-be4b-1c7352e2.westb.seetacloud.com" not in UPLINK


def test_deployment_scripts_do_not_embed_credentials():
    combined = REMOTE + PC
    assert "jTau" not in combined
    assert "password=" not in combined.lower()
    assert "sshpass" not in combined


def test_bulk_gateway_requires_authentication_and_preserves_derp_upgrade():
    gateway = (ROOT / "tools" / "autodl_derp_bulk_gateway.py").read_text(encoding="utf-8")
    assert "authorization" in gateway.lower()
    assert "compare_digest" in gateway
    assert 'fields.get(b"upgrade"' in gateway
    assert "asyncio.gather" in gateway
    assert "/progressive" in gateway
