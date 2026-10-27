from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_derp_queue_uses_four_workers_eight_mib_segments_and_256kib_probes():
    config = (ROOT / "mapping/config/uplink.yaml").read_text(encoding="utf-8")
    watcher = (ROOT / "tools/stream_go2_high_fidelity_queue.sh").read_text(encoding="utf-8")
    uploader = (ROOT / "mapping/robot/go2_mapping_lio/go2_mapping_lio/progressive_artifact_uploader.py").read_text(encoding="utf-8")

    assert "parallel_workers: 4" in config
    assert "chunk_kib: 8192" in config
    assert "probe_chunk_kib: 256" in config
    assert "GO2_MAPPING_UPLOAD_CHUNK_KIB:-8192" in watcher
    assert "GO2_MAPPING_UPLOAD_WORKERS:-4" in watcher
    assert "PROBE_CHUNK_BYTES = 256 * 1024" in uploader
    assert "/progressive/probe" in uploader


def test_pc_high_fidelity_receiver_binds_only_to_tailnet_interface():
    session = (ROOT / "tools/start_go2_mapping_session.ps1").read_text(encoding="utf-8")
    stack = (ROOT / "tools/start_go2_pc_stack.ps1").read_text(encoding="utf-8")

    assert "start_go2_mapping_session_tailnet.ps1" in session
    assert "transport = 'TAILSCALE_DIRECT'" in session
    assert "'--bind', '127.0.0.1'" not in session
    assert "'--bind', $tailnetIp" in stack
    assert "'--bind', '127.0.0.1'" in stack
    assert "127.0.0.1:19000:127.0.0.1:8000" not in session
    assert "127.0.0.1:19000:127.0.0.1:8000" not in stack


def test_one_click_stacks_leave_tailscale_free_to_choose_direct_by_default():
    session = (ROOT / "tools/start_go2_mapping_session.ps1").read_text(encoding="utf-8")
    stack = (ROOT / "tools/start_go2_pc_stack.ps1").read_text(encoding="utf-8")

    assert "start_go2_mapping_session_tailnet.ps1" in session
    assert "[int]$PreferredDerpRegion = 0" in stack
    assert "Direct path remains preferred" in stack
    assert "[string]$HighFidelityTransport = 'direct'" in stack
    assert "if ($HighFidelityTransport -eq 'autodl')" in stack
