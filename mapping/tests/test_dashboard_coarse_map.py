from pathlib import Path


ROOT = Path(__file__).parents[1] / "pc" / "go2_mapping_dashboard"


def test_dashboard_contains_coarse_map_and_robot_verification_surface():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

    for marker in ("coarse-map-card", "coarse-map-svg", "coarse-task-list", "coarse-map-add-feature", "coarse-map-compile", "coarse-map-nav2-frontiers", "coarse-frontier-candidate", "coarse-stair-survey", "coarse-stair-survey-state", "coarse-mission-state", "coarse-navigation-state", "BUILDING FUNCTION MAP"):
        assert marker in index
    for marker in ("/api/maps/coarse/", "/api/maps/coarse/${encodeURIComponent(mapId)}/frontiers", "/api/stairs/survey/latest", "/api/coarse-map", "/api/coarse-map/tasks/", "/api/missions", "/api/robot/navigation/goal", "/api/robot/navigation/status", "task-go", "task-mode", "task-lease", "AUTONOMOUS"):
        assert marker in script
    assert "/static/coarse-map.js?v=coarse-map-editor-v8" in index
    assert ".coarse-map-card" in style
    assert ".coarse-task-panel" in style


def test_dashboard_exposes_a_versioned_coarse_map_editor_contract():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")

    for marker in (
        "coarse-map-tools",
        "coarse-map-snap",
        "coarse-map-finish",
        "coarse-map-json-import",
        "coarse-map-json-export",
        "coarse-map-history",
        "coarse-map-editor-status",
        "coarse-navigation-pause",
        "coarse-navigation-resume",
        "coarse-navigation-safety-stop",
    ):
        assert marker in index
    for marker in (
        "clientToMapPoint",
        "commitDraft",
        "saveCanonicalMap",
        "loadRevisionHistory",
        "loadRevisionIntoDraft",
        "/revisions/${encodeURIComponent(revisionNumber)}",
        "historical revision",
        "FileReader",
        "URL.createObjectURL",
        "/api/maps/coarse/",
        "pointerdown",
        "pointermove",
        "/api/missions/${encodeURIComponent(currentMission.mission_id)}",
        "body: action === \"safety-stop\"",
        "data-task-action=\"complete\"",
        "completeTask",
        "frontierCandidates",
        "previewNav2Frontiers",
        "data-frontier-id",
        "loadStairSurvey",
        "renderStairSurvey",
    ):
        assert marker in script


def test_dashboard_exposes_map_navigation_and_coverage_state_contract():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

    for marker in ("coarse-map-zoom-in", "coarse-map-zoom-out", "coarse-map-reset-view", "coarse-map-coverage-legend"):
        assert marker in index
    for marker in ("mapViewport", "wheel", "zoomIn", "zoomOut", "resetMapView", "UNSEEN", "OBSERVED", "VERIFIED", "QUARANTINED"):
        assert marker in script
    for marker in (".coarse-map-coverage-legend", ".coverage-unseen", ".coverage-observed", ".coverage-verified", ".coverage-quarantined", ".coarse-feature.coverage-observed"):
        assert marker in style


def test_dashboard_exposes_reconstruction_quality_and_export_workflow_contract():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "workflow-status.js").read_text(encoding="utf-8")
    for marker in (
        "workflow-segment-state",
        "workflow-quality-state",
        "workflow-coverage-state",
        "workflow-appearance-state",
        "workflow-pose-p95",
        "workflow-plane-p95",
        "workflow-clock-fit",
        "workflow-revisit-count",
        "workflow-export",
        "workflow-replay-button",
        "workflow-replay",
        "workflow-blockers",
    ):
        assert marker in index
    for marker in (
        "/api/reconstruction/segments",
        "/api/coverage/",
        "/api/appearance/measured-surface/status",
        "/api/reconstruction/field-acceptance-evidence",
        "/api/exports/glb",
        "/replay",
        "workflowReplay",
        "point_to_trajectory",
        "wall_plane",
        "clock_model",
        "423",
    ):
        assert marker in script


def test_dashboard_prefers_canonical_map_api_with_legacy_read_fallback():
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    load_map = script[script.index("async function loadMap"):script.index("async function rebuildTasks")]

    canonical = "fetch(`/api/maps/coarse/${encodeURIComponent(DEFAULT_MAP_ID)}`"
    legacy = 'fetch("/api/coarse-map", { cache: "no-store" })'
    assert canonical in load_map
    assert legacy in load_map
    assert load_map.index(canonical) < load_map.index(legacy)
    assert "canonicalResponse.ok" in load_map


def test_dashboard_task_mutations_use_canonical_api_before_legacy_fallback():
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")

    assert "coarseTaskRequest" in script
    assert "/api/maps/coarse/${encodeURIComponent(mapId)}/tasks/" in script
    assert "/api/coarse-map/tasks/" in script
    assert "response.status === 404 || response.status === 405" in script


def test_dashboard_exposes_a_focusable_coarse_workspace_and_draft_controls():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

    for marker in (
        'id="workspace-mode-observe"',
        'id="workspace-mode-coarse"',
        'id="coarse-map-load-current"',
        'id="coarse-map-discard-draft"',
        'id="coarse-map-save-revision"',
        'id="coarse-map-nav2-export"',
        'id="coarse-observation-reference"',
        'id="coarse-room-scan-start"',
        'id="coarse-room-scan-dialog"',
        'id="coarse-room-scan-token"',
    ):
        assert marker in index
    for marker in (
        "workspaceMode",
        "URLSearchParams",
        "localStorage",
        "draftBaseRevision",
        "draftHistory",
        "draftFuture",
        "saveDraftToStorage",
        "loadDraftFromStorage",
        "saveMapRevision",
        "exportNav2Map",
        "/api/maps/coarse/${encodeURIComponent(mapId)}/nav2-export",
        "coarse-map-save-revision",
        "dblclick",
        "event.key === \"Enter\"",
        "event.key === \"Escape\"",
        "undoDraft",
        "redoDraft",
        "expected_revision",
        "409",
        "/api/room-scan/capabilities",
        "/api/room-scan/start",
    ):
        assert marker in script
    for marker in (
        "data-workspace-mode",
        "workspace-mode-switch",
        'data-workspace-mode="coarse"',
        "coarse-observation-reference",
        "coarse-map-editor-shell",
        "min-height",
        "aspect-ratio",
    ):
        assert marker in style


def test_dashboard_exposes_room_scan_workflow_state_contract():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    for marker in (
        'id="coarse-scan-workflow"',
        'id="coarse-scan-phase"',
        'id="coarse-scan-session"',
        'id="coarse-scan-health"',
        'id="coarse-scan-stop"',
    ):
        assert marker in index
    for marker in (
        "roomScanRequestId",
        "roomScanPhase",
        "/api/room-scan/status/",
        "/api/room-scan/stop/",
        "STARTING",
        "RECORDING",
        "FINALIZING",
        "PREVIEW_READY",
        "FAILED",
        "setInterval",
    ):
        assert marker in script
    assert ".coarse-scan-workflow" in style


def test_dashboard_exposes_live_mapping_visualization_controls():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    coarse_script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    for marker in (
        'id="live-point-cloud"',
        'id="live-coverage"',
        'id="live-sensor-health"',
    ):
        assert marker in index
    for marker in (
        "live-point-cloud",
        "live-coverage",
        "live-sensor-health",
        "/api/snapshot",
        "refreshFullSnapshot",
        "liveRenderEnabled",
        "trajectory_delta",
        "robot_state",
    ):
        assert marker in script
    for marker in ("go2DashboardSnapshot", "go2:telemetry", "robotPosition"):
        assert marker in coarse_script
    assert ".live-mapping-controls" in style


def test_dashboard_exposes_measured_live_comparison_controls():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    for marker in ('id="scene-layer-measured"', 'id="scene-layer-live"', 'id="scene-layer-compare"', 'id="live-alignment"'):
        assert marker in index
    for marker in ("sceneLayerMode", "coarseReferenceGroup", "/api/maps/coarse/", "scene-layer-compare", "LIVE_REFERENCE_ONLY", "PENDING_ALIGNMENT"):
        assert marker in script
    assert ".scene-layer-controls" in style


def test_scene_layer_mode_is_the_only_point_cloud_visibility_owner():
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert script.count("cloud.visible =") == 2
    assert "cloud.visible = false;" in script
    assert "cloud.visible = liveRenderEnabled && (live || compare);" in script


def test_dashboard_exposes_coverage_audit_and_rescan_suggestion_contract():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    coarse_script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    for marker in ('id="coverage-audit-state"', 'id="coverage-audit-unseen"', 'id="coverage-audit-next"', 'id="coverage-audit-route"', 'id="coverage-audit-gate"', 'id="coverage-audit-focus"'):
        assert marker in index
    for marker in ("refreshCoverageAudit", "routeLength", "taskRoutePoints", "POLYGON_FALLBACK", "/api/coverage/", "UNSEEN", "READY FOR HUMAN REVIEW", "REVIEW ONLY", "coverage-audit-focus", "go2:coverage"):
        assert marker in script
    assert "go2:coverage" in coarse_script
    assert ".coverage-audit" in style
    for marker in ("rescanPreviewTask", "coarse-rescan-route", "waypoints", "RESCAN PREVIEW"):
        assert marker in coarse_script


def test_dashboard_exposes_route_safety_audit_and_navigation_gate_contract():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    for marker in (
        'id="coarse-route-audit"',
        'id="coarse-route-audit-state"',
        'id="coarse-route-audit-findings"',
    ):
        assert marker in index
    for marker in (
        "routeAudits",
        "refreshMissionRouteAudits",
        "/route-audit",
        "navigation_eligible",
        "routeAuditFindings",
        "task-route-safety",
        "AUDIT_PENDING",
    ):
        assert marker in script
    assert ".coarse-route-audit" in style


def test_dashboard_legacy_map_conversion_preserves_keep_out_zones():
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    conversion = script[script.index("function toV2"):script.index("function entityCollection")]
    assert "keep_out_zones: Array.isArray(payload?.keep_out_zones) ? payload.keep_out_zones : []" in conversion


def test_dashboard_exposes_mapping_review_evidence_summary():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    coarse_script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    for marker in (
        'id="mapping-review"',
        'id="mapping-review-state"',
        'id="mapping-review-coarse"',
        'id="mapping-review-observed"',
        'id="mapping-review-trajectory"',
        'id="mapping-review-reconstruction"',
        'id="mapping-review-coverage"',
        'id="mapping-review-difference"',
    ):
        assert marker in index
    for marker in (
        "updateMappingReview",
        "coarseMapExtent",
        "go2:coarse-map",
        "go2:coverage",
        "ALIGNMENT_PENDING",
        "NOT COMPUTED",
        "mapping-review",
    ):
        assert marker in script
    assert "go2:coarse-map" in coarse_script
    assert ".mapping-review" in style


def test_dashboard_exposes_read_only_robot_control_status_contract():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    for marker in (
        'id="robot-control-status"',
        'id="robot-control-state"',
        'id="robot-control-link"',
        'id="robot-control-locomotion"',
        'id="robot-control-navigation"',
        'id="robot-control-sensors"',
        'id="robot-control-gates"',
    ):
        assert marker in index
    for marker in (
        "refreshRobotControlStatus",
        "/api/robot/capabilities",
        "/api/robot/locomotion/status",
        "/api/robot/navigation/status",
        "ACTUATION_LOCKED",
        "GO2_LINK_UNVERIFIED",
    ):
        assert marker in script
    assert ".robot-control-status" in style


def test_dashboard_exposes_building_3d_and_blueprint_style_entity_controls():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    three = (ROOT / "static" / "coarse-map-3d-threejs.js").read_text(encoding="utf-8")
    for marker in ("coarse-building-floor-cards", "coarse-building-mission", "coarse-material-dialog", "coarse-entity-locked", "coarse-furniture-rotate-left"):
        assert marker in index
    for marker in ("renderBuildingMapsThreeJS", "transformEntityFromThree", "mergePolygonsOnSharedEdge", "data-material", "locked"):
        assert marker in script
    for marker in ("renderBuildingMapsThreeJS", "floorOffset", "entityTransformHandler", "furnitureGeometry", "setEnabled"):
        assert marker in three
    assert "/api/catalog/furniture" in script
    assert "/api/catalog/coarse-map-templates" in script
    assert "applyCoarseMapTemplate" in script
    assert "GLTFLoader" in three
    assert "data-gltf-assets-loaded" not in three
    assert "gltfAssetsLoaded" in three
    assert "loadPbrMaterial" in three
    assert "PBR_MATERIAL_SPECS" in three
    assert "roomTextureRepeat" in three
    assert "texture.clone()" in three
    assert "roomBounds.width / spec.realSizeM[0]" in three
    assert "roomBounds.height / spec.realSizeM[1]" in three
    assert "/api/catalog/pbr-materials" in script


def test_dashboard_exposes_local_three_transform_controls_for_furniture():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "coarse-map.js").read_text(encoding="utf-8")
    three = (ROOT / "static" / "coarse-map-3d-threejs.js").read_text(encoding="utf-8")
    style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    controls = ROOT / "static" / "three-examples" / "controls" / "TransformControls.js"

    for marker in (
        'id="coarse-three-transform-tools"',
        'id="coarse-three-transform-translate"',
        'id="coarse-three-transform-rotate"',
        'id="coarse-three-transform-space"',
        'id="coarse-three-transform-snap"',
    ):
        assert marker in index
    for marker in ("setThreeTransformMode", "setThreeTransformSpace", "setThreeTransformSnap"):
        assert marker in script
    for marker in (
        "TransformControls",
        "transformControls.attach",
        "transformControls?.detach",
        "setTranslationSnap",
        "setRotationSnap",
        "dragging-changed",
        "mouseUp",
        "rotation_rad",
    ):
        assert marker in three
    assert controls.is_file()
    assert "class TransformControls" in controls.read_text(encoding="utf-8")
    assert "furnitureDrag" not in three
    assert "dataset.transformAttached" in three
    assert "dataset.transformScreenX" in three
    assert "dataset.transformCommits" in three
    assert "dataset.transformAxis" in three
    assert ".coarse-three-transform-tools" in style
    assert "min-height: 2.75rem" in style.split(".coarse-three-transform-tools", 1)[1]


def test_threejs_initialization_is_serialized_across_async_renders():
    three = (ROOT / "static" / "coarse-map-3d-threejs.js").read_text(encoding="utf-8")

    assert "let threeInitPromise = null;" in three
    assert "threeInitPromise = createThreeJS(container);" in three
    assert "async function createThreeJS(container)" in three
