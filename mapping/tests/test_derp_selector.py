from mapping.pc.go2_mapping_network.derp_selector import (
    parse_netcheck_latencies,
    select_derp_region,
)


def test_netcheck_parser_reads_reachable_regions_and_ignores_noise():
    report = '''
    * UDP: true
    * DERP latency:
        - sfo: 143.1ms (San Francisco)
        - private-cn: 41.6ms (Private China)
        - broken: 1.2s (ignored)
    '''

    assert parse_netcheck_latencies(report) == {"sfo": 143.1, "private-cn": 41.6}


def test_selector_prefers_private_region_only_when_both_ends_measure_a_clear_gain():
    decision = select_derp_region(
        pc_latencies={"sfo": 143.0, "private-cn": 42.0},
        robot_latencies={"sfo": 120.0, "private-cn": 38.0},
        private_regions={"private-cn": 902},
        improvement_ratio=0.10,
        peer_is_direct=False,
    )

    assert decision.preferred_region_id == 902
    assert decision.preferred_region_code == "private-cn"
    assert decision.private_score_ms == 80.0
    assert decision.best_official_score_ms == 263.0
    assert decision.reason == "private_region_materially_faster"


def test_selector_keeps_tailscale_automatic_when_private_gain_is_too_small():
    decision = select_derp_region(
        pc_latencies={"sfo": 100.0, "private-cn": 96.0},
        robot_latencies={"sfo": 80.0, "private-cn": 79.0},
        private_regions={"private-cn": 902},
        improvement_ratio=0.10,
        peer_is_direct=False,
    )

    assert decision.preferred_region_id == 0
    assert decision.reason == "private_gain_below_threshold"


def test_selector_never_forces_a_relay_while_p2p_is_direct():
    decision = select_derp_region(
        pc_latencies={"private-cn": 20.0},
        robot_latencies={"private-cn": 20.0},
        private_regions={"private-cn": 902},
        peer_is_direct=True,
    )

    assert decision.preferred_region_id == 0
    assert decision.reason == "p2p_direct_preferred"


def test_selector_fails_closed_when_private_region_is_not_reachable_from_both_ends():
    decision = select_derp_region(
        pc_latencies={"sfo": 100.0, "private-cn": 20.0},
        robot_latencies={"sfo": 90.0},
        private_regions={"private-cn": 902},
        peer_is_direct=False,
    )

    assert decision.preferred_region_id == 0
    assert decision.reason == "private_region_not_reachable_from_both_ends"
