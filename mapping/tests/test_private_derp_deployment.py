from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_private_derp_installer_uses_real_tls_client_verification_and_stun():
    remote = (ROOT / "tools/install_private_derp.sh").read_text(encoding="utf-8")

    assert "-certmode=letsencrypt" in remote
    assert "-verify-clients" in remote
    assert "-stun=true" in remote
    assert "-stun-port=3478" in remote
    assert "-a=:443" in remote
    assert "tailscaled.sock" in remote
    assert "systemd" in remote.lower()
    assert "rm -rf" not in remote


def test_private_derp_deployer_is_parameterized_and_does_not_force_relay_by_default():
    deployer = (ROOT / "tools/deploy_private_derp.ps1").read_text(encoding="utf-8")

    assert "[string]$Domain" in deployer
    assert "[string]$ServerHost" in deployer
    assert "[int]$RegionId" in deployer
    assert "OmitDefaultRegions" in deployer
    assert "false" in deployer.lower()
    assert "force-prefer-derp" not in deployer
    assert "sshpass" not in deployer
    assert "password=" not in deployer.lower()
