import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from pathlib import Path

from mapping.robot.go2_mapping_lio.go2_mapping_lio.bandwidth import (
    AdaptiveBandwidthController,
    BandwidthAllocationStore,
    DualQueueBudget,
    PreviewDemandStore,
    TokenBucket,
    VerifiedAckGoodputWindow,
)


ROOT = Path(__file__).resolve().parents[2]


def test_dual_queue_budget_has_no_fixed_preview_percentage():
    budget = DualQueueBudget(total_mbps=120.0)

    assert budget.preview_mbps == 0.0
    assert budget.high_fidelity_mbps == 120.0
    assert budget.preview_share_percent == 0.0
    assert budget.high_fidelity_share_percent == 100.0


def test_priority_budget_gives_preview_its_measured_demand_before_high_fidelity():
    budget = DualQueueBudget(total_mbps=12.0, preview_demand_mbps=3.25)

    assert budget.preview_mbps == pytest.approx(3.25)
    assert budget.high_fidelity_mbps == pytest.approx(8.75)
    assert budget.preview_policy == "realtime_preview_priority_high_fidelity_remaining_capacity"


def test_priority_budget_never_overcommits_the_link_when_preview_needs_all_capacity():
    budget = DualQueueBudget(total_mbps=2.0, preview_demand_mbps=4.0)

    assert budget.preview_mbps == pytest.approx(2.0)
    assert budget.high_fidelity_mbps == pytest.approx(0.0)


def test_preview_demand_store_expires_stale_preview_reservations(tmp_path):
    state = tmp_path / "preview-demand.json"
    store = PreviewDemandStore(state, max_age_s=5.0)

    store.publish(required_mbps=1.6, payload_bytes=100_000, cadence_hz=2.0, now=100.0)

    assert store.read(now=104.9) == pytest.approx(1.6)
    assert store.read(now=105.1) is None


def test_preview_demand_store_serializes_concurrent_camera_and_geometry_publishers(tmp_path):
    state = tmp_path / "preview-demand.json"
    store = PreviewDemandStore(state)

    def publish(worker: int) -> None:
        for sequence in range(100):
            store.publish(
                required_mbps=1.0 + worker,
                payload_bytes=10_000 + sequence,
                cadence_hz=5.0,
                now=100.0 + sequence,
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(publish, worker) for worker in range(4)]
        for future in futures:
            future.result()

    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["cadence_hz"] == 5.0
    assert list(tmp_path.glob(".preview-demand.json.*.tmp")) == []


def test_adaptive_controller_releases_expired_preview_demand_to_high_fidelity():
    controller = AdaptiveBandwidthController(
        ceiling_mbps=20.0,
        floor_mbps=1.0,
        initial_total_mbps=10.0,
        preview_demand_mbps=2.5,
    )

    controller.set_preview_demand(0.0)

    assert controller.budget.preview_mbps == 0.0
    assert controller.budget.high_fidelity_mbps == pytest.approx(10.0)


@pytest.mark.parametrize(
    "total_mbps",
    [
        30.0,
        15.0,
        10.0,
        7.5,
        5.0,
        2.0,
        0.66,
    ],
)
def test_dynamic_total_capacity_has_no_claim_without_preview_demand(total_mbps):
    budget = DualQueueBudget(total_mbps=total_mbps)

    assert budget.preview_mbps == pytest.approx(0.0)
    assert budget.high_fidelity_mbps == pytest.approx(total_mbps)
    assert budget.high_fidelity_share_percent == pytest.approx(100.0)


def test_dual_queue_budget_has_no_caller_override_for_the_authoritative_policy():
    with pytest.raises(TypeError):
        DualQueueBudget(total_mbps=120.0, preview_share=0.05)


def test_preview_token_bucket_waits_before_exceeding_its_reserved_rate():
    bucket = TokenBucket(rate_bytes_per_second=100.0, burst_seconds=1.0, now=0.0)

    assert bucket.delay_for(100, now=0.0) == 0.0
    assert bucket.delay_for(50, now=0.0) == 0.5


def test_shared_token_bucket_serializes_future_reservations_from_parallel_workers():
    bucket = TokenBucket(rate_bytes_per_second=100.0, burst_seconds=1.0, now=0.0)

    assert bucket.delay_for(100, now=0.0) == 0.0
    assert bucket.delay_for(50, now=0.0) == 0.5
    assert bucket.delay_for(50, now=0.0) == 1.0


def test_token_bucket_can_adopt_a_new_dynamic_rate_without_an_extra_burst():
    bucket = TokenBucket(rate_bytes_per_second=100.0, burst_seconds=1.0, now=0.0)
    assert bucket.delay_for(100, now=0.0) == 0.0

    bucket.set_rate(50.0, now=0.0)

    assert bucket.delay_for(50, now=0.0) == 1.0


def test_verified_ack_window_aggregates_all_parallel_worker_bytes():
    window = VerifiedAckGoodputWindow(minimum_interval_s=5.0, now=100.0)

    assert window.note_verified(1_000_000, now=102.0) is None
    assert window.note_verified(2_000_000, now=104.0) is None
    sample = window.note_verified(3_000_000, now=105.0)

    assert sample is not None
    assert sample.byte_count == 6_000_000
    assert sample.elapsed_s == 5.0
    assert sample.goodput_mbps == pytest.approx(9.6)


def test_verified_ack_window_never_publishes_before_the_minimum_interval():
    window = VerifiedAckGoodputWindow(minimum_interval_s=5.0, now=0.0)

    sample = window.note_verified(8_000_000, now=4.999)

    assert sample is None
    assert window.pending_bytes == 8_000_000


def test_adaptive_bandwidth_drops_fast_and_recomputes_remaining_capacity():
    controller = AdaptiveBandwidthController(
        ceiling_mbps=120.0,
        floor_mbps=1.0,
        initial_total_mbps=120.0,
    )

    budget = controller.observe_high_fidelity_transfer(byte_count=8_000_000, elapsed_s=8.0)

    assert budget.total_mbps == pytest.approx(7.2)
    assert budget.preview_mbps == pytest.approx(0.0)
    assert budget.high_fidelity_mbps == pytest.approx(7.2)


def test_adaptive_bandwidth_slow_starts_then_uses_small_probes_after_congestion():
    controller = AdaptiveBandwidthController(ceiling_mbps=120.0, floor_mbps=0.25)

    assert controller.budget.total_mbps == pytest.approx(0.25)
    assert controller.observe_high_fidelity_transfer(byte_count=237_500, elapsed_s=8.0).total_mbps == pytest.approx(0.5)
    assert controller.observe_high_fidelity_transfer(byte_count=475_000, elapsed_s=8.0).total_mbps == pytest.approx(1.0)
    assert controller.observe_high_fidelity_transfer(byte_count=200_000, elapsed_s=8.0).total_mbps == pytest.approx(0.25)
    assert controller.observe_high_fidelity_transfer(byte_count=237_500, elapsed_s=8.0).total_mbps == pytest.approx(0.275)


def test_adaptive_bandwidth_probes_up_by_segment_and_never_exceeds_ceiling():
    controller = AdaptiveBandwidthController(
        ceiling_mbps=20.0,
        floor_mbps=1.0,
        initial_total_mbps=10.0,
        probe_gain=0.10,
    )

    first = controller.observe_high_fidelity_transfer(byte_count=9_500_000, elapsed_s=8.0)
    second = controller.observe_high_fidelity_transfer(byte_count=10_450_000, elapsed_s=8.0)
    controller.observe_high_fidelity_transfer(byte_count=19_000_000, elapsed_s=8.0)

    assert first.total_mbps == pytest.approx(11.0)
    assert second.total_mbps == pytest.approx(12.1)
    assert controller.budget.total_mbps <= 20.0


def test_unthrottled_capacity_probe_immediately_adopts_a_safe_fraction_of_capacity():
    controller = AdaptiveBandwidthController(
        ceiling_mbps=120.0,
        floor_mbps=0.1,
        initial_total_mbps=0.1,
    )

    budget = controller.observe_capacity_probe(byte_count=256 * 1024, elapsed_s=0.2)
    observed_high_mbps = 256 * 1024 * 8.0 / 0.2 / 1_000_000.0

    assert budget.total_mbps == pytest.approx(observed_high_mbps * 0.90)
    assert budget.preview_mbps == pytest.approx(0.0)
    assert budget.high_fidelity_mbps == pytest.approx(budget.total_mbps)
    assert budget.high_fidelity_mbps == pytest.approx(budget.total_mbps - budget.preview_mbps)


def test_shared_bandwidth_state_is_atomic_readable_and_expires(tmp_path):
    state = tmp_path / "bandwidth-allocation.json"
    store = BandwidthAllocationStore(state, max_age_s=30.0)
    controller = AdaptiveBandwidthController(ceiling_mbps=120.0)
    budget = controller.observe_high_fidelity_transfer(byte_count=8_000_000, elapsed_s=8.0)

    store.publish(
        budget,
        observed_high_fidelity_mbps=8.0,
        sample_elapsed_s=8.0,
        route_id="wifi:lab|tailscale:direct",
        ack_rtt_ms=42.0,
        now=100.0,
    )

    loaded = store.read(now=120.0, expected_route_id="wifi:lab|tailscale:direct")
    assert loaded is not None
    assert loaded.total_mbps == pytest.approx(budget.total_mbps)
    assert store.read(now=120.0, expected_route_id="wifi:lab|tailscale:derp-901") is None
    assert store.read(now=131.0) is None
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["route_id"] == "wifi:lab|tailscale:direct"
    assert payload["ack_rtt_ms"] == 42.0
    assert payload["preview_policy"] == "realtime_preview_priority_high_fidelity_remaining_capacity"
    assert payload["preview_share"] == pytest.approx(budget.effective_preview_share)
    assert payload["high_fidelity_share"] == pytest.approx(budget.effective_high_fidelity_share)


def test_preview_bandwidth_module_has_no_identity_payload_fitter_or_queue_delay_contract():
    root = Path(__file__).resolve().parents[2]
    bandwidth_source = (
        root / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "bandwidth.py"
    ).read_text(encoding="utf-8")
    producer_source = (
        root / "mapping" / "robot" / "go2_mapping_lio" / "go2_mapping_lio" / "live_keyframe_producer.py"
    ).read_text(encoding="utf-8")

    assert "fit_preview_payload" not in bandwidth_source
    assert "fit_preview_payload" not in producer_source
    assert "--preview-max-queue-delay-s" not in producer_source


def test_uplink_profile_makes_the_dynamic_preview_reservation_explicit():
    profile = yaml.safe_load((ROOT / "mapping" / "config" / "uplink.yaml").read_text(encoding="utf-8"))

    assert profile["dual_queue"]["preview"]["allocation"] == "realtime_priority_measured_demand"
    assert "max_queue_delay_ms" not in profile["dual_queue"]["preview"]
    assert profile["dual_queue"]["high_fidelity"]["allocation"] == "remaining_measured_capacity"
    assert profile["dual_queue"]["probe_ceiling_mbps"] == 30
    # Capacity probing replaces the old 0.1 Mbps slow-start so the first
    # independent probe can immediately establish a useful budget.
    assert profile["dual_queue"]["bootstrap_total_mbps"] == 4.0
    assert profile["dual_queue"]["high_fidelity"]["measurement_chunk_kib"] == 8192
    assert profile["dual_queue"]["high_fidelity"]["probe_chunk_kib"] == 256
