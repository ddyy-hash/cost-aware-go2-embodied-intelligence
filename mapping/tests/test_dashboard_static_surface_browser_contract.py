from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "mapping" / "pc" / "go2_mapping_dashboard" / "static"


def _javascript_function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    async_start = start - len("async ")
    if async_start >= 0 and source[async_start:start] == "async ":
        start = async_start
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unterminated JavaScript function: {name}")


def test_browser_static_compatibility_artifact_is_verified_but_never_loaded_by_current_surface():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    refresh = _javascript_function(app, "refreshMeasuredSurfacePreview")

    assert "/static/compatibility/measured-surface-current.json" in app
    assert "loadStaticMeasuredSurfaceCompatibility" in app
    assert "validateStaticMeasuredSurfaceCompatibility" in app
    assert "await sha256Hex(surfaceBuffer)" in app
    assert "STATIC_COMPATIBILITY" in app
    assert "officialStatus.available === false" in refresh
    assert "loadStaticMeasuredSurfaceCompatibility" not in refresh
    assert "disposeMeasuredSurfacePreview();" in refresh
    assert "statusResponse.ok" in refresh
    assert "buildMeasuredSurfacePreview" in refresh
    assert "hidden_surface_policy" in app
    assert "TEXTURE_NOT_AUTHORIZED" in app


def test_browser_static_compatibility_contract_preserves_observed_only_read_only_provenance():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    validator = _javascript_function(app, "validateStaticMeasuredSurfaceCompatibility")

    for fragment in (
        'manifest.source_mode !== "DERIVED_READ_ONLY"',
        "manifest.read_only !== true",
        'manifest.point_authority !== "LIDAR_MEASURED"',
        "manifest.geometry_authority !== false",
        'manifest.hidden_surface_policy !== "OBSERVED_POINTS_ONLY"',
        'manifest.quality !== "PREVIEW_UNCALIBRATED"',
        "manifest.volume_filled !== false",
        "manifest.watertight !== false",
        "manifest.vertex_count",
        "manifest.triangle_count",
        "manifest.surface_sha256",
        "manifest.surface_relative_url",
    ):
        assert fragment in validator


def test_static_compatibility_loader_remains_content_addressed_without_current_view_fallback():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    loader = _javascript_function(app, "loadStaticMeasuredSurfaceCompatibility")
    refresh = _javascript_function(app, "refreshMeasuredSurfacePreview")

    assert "currentSurfaceState" in loader
    assert "currentSurfaceState?.derivedCacheKey === manifest.derived_cache_key" in loader
    assert "currentSurfaceState?.surfaceSha256 === manifest.surface_sha256" in loader
    assert "surface: null" in loader
    assert "loadStaticMeasuredSurfaceCompatibility" not in refresh
    assert "if (surfacePayload === null)" in refresh


def test_dynamic_read_only_surface_reuses_a_matching_content_address_before_payload_fetch():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    refresh = _javascript_function(app, "refreshMeasuredSurfacePreview")

    reuse_guard = 'if (status.source_mode === "DERIVED_READ_ONLY"'
    payload_fetch = 'fetch("/api/reconstruction/measured-surface"'

    assert reuse_guard in refresh
    assert "measuredSurfaceState?.derivedCacheKey === status.derived_cache_key" in refresh
    assert "measuredSurfaceState?.surfaceSha256 === status.surface_sha256" in refresh
    assert refresh.index(reuse_guard) < refresh.index(payload_fetch)


def test_dynamic_preview_accepts_only_explicitly_paired_lidar_observation_provenance():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    builder = _javascript_function(app, "buildMeasuredSurfacePreview")

    assert "const allowedObservationLayers = new Set([" in builder
    assert '["LIDAR_MEASURED", "LIDAR_MEASURED"]' in builder
    assert '["LIO_REGISTERED_LIDAR_OBSERVED", "LIO_REGISTERED_LIDAR_OBSERVED"]' in builder
    assert "allowedObservationLayers.has(`${surface.source_layer}:${surface.point_authority}`)" in builder


def test_explicit_local_surface_query_uses_only_the_isolated_odom_preview_endpoints():
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    refresh = _javascript_function(app, "refreshMeasuredSurfacePreview")
    visibility = _javascript_function(app, "updateSurfaceVisibility")

    assert 'get("surface") === "local"' in app
    assert 'fetch("/api/reconstruction/session-odom-local/status", { cache: "no-store" })' in refresh
    assert 'fetch("/api/reconstruction/session-odom-local", { cache: "no-store" })' in refresh
    assert "if (!sessionOdomLocalPreview && officialStatus.available === false)" in refresh
    assert 'status.source_mode === "SESSION_ODOM_LOCAL_READ_ONLY"' in app
    assert 'status.coordinate_scope !== "SESSION_ODOM_LOCAL"' in app
    assert 'status.coordinate_frame !== "odom"' in app
    assert "status.room_or_building !== false" in app
    assert "status.texture_authorized !== false" in app
    assert 'surfaceViewMode = "local"' in app
    assert "sessionOdomLocalPreview" in visibility
    assert '"SESSION_ODOM_LOCAL / NO TEXTURE AUTHORITY"' in app
    assert '"SESSION ODOM LOCAL / OBSERVED SURFACE / METERS"' in app
    assert '"SESSION ODOM LOCAL / READ-ONLY / NOT GLOBAL"' in app
    appearance = _javascript_function(app, "refreshBuildingAppearance")
    assert "sessionOdomLocalPreview" in appearance
    assert 'status.source_mode === "SESSION_ODOM_LOCAL_READ_ONLY"' in refresh
    assert '"SESSION_ODOM_LOCAL / NO TEXTURE AUTHORITY"' in refresh
    assert 'candidate.state.sourceMode === "SESSION_ODOM_LOCAL_READ_ONLY"' in refresh
    assert '"AWAITING SESSION ODOM LOCAL OBSERVED SURFACE / METERS"' in app
    assert '"SESSION ODOM LOCAL / AWAITING VERIFIED SURFACE"' in app
    assert '"SESSION ODOM LOCAL / NOT GLOBAL"' in app
    assert 'cameraAssist.hidden = sessionOdomLocalPreview' in app
