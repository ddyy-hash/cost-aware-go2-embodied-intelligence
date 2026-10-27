import * as THREE from "./vendor/three.module.js";
import { sha256Hex } from "./sha256.js";
import { updateEmptyStateChecklist } from "./empty-state.js";

const byId = (id) => document.getElementById(id);
const uiText = (value) => {
  const dictionary = {
    INITIALIZING: "正在初始化", AWAITING_KEYFRAME: "等待关键帧", STALE: "数据已过期",
    PREVIEW_UNCALIBRATED: "预览 / 未标定", LIVE: "实时", READ_ONLY: "只读",
    READ_ONLY_LIVE: "实时只读", UNAVAILABLE: "不可用", UNVERIFIED: "未验证",
    GO2_LINK_UNVERIFIED: "GO2 连接未验证", ACTUATION_LOCKED: "运动层已锁定",
    AUTHORIZED: "已授权", NAV_READY: "导航就绪", NAV_ACTIVE: "导航执行中",
    NAV_PAUSED: "导航已暂停", WAITING: "等待数据", WAITING_FOR_EVIDENCE: "等待证据",
    REFERENCE_OVERLAY_ONLY: "仅参考叠加", COMPARABLE_FRAME: "坐标可比较",
    POINTS_AVAILABLE_EXTENT_N_A: "已有点云 / 范围待定", NO_VERIFIED_TASKS: "暂无已验证任务",
    NO_LIVE_POINTS: "暂无实时点云", PENDING_ALIGNMENT: "等待对齐", LIVE_REFERENCE_ONLY: "仅实时参考",
    LIVE_ROOM_SCAN: "实时房间扫描", CALIBRATION_PENDING: "等待标定", CALIBRATED_LIO_ACCEPTED: "LIO 标定通过",
    OBSERVED_SURFACE: "观测表面", DENSE_LIDAR: "稠密 LiDAR", PREVIEW_ONLY: "仅预览",
    NOT_SURVEY_GRADE: "非测量级", UNCALIBRATED: "未标定", NOT_CALIBRATED: "未标定",
    NOT_AUTHORIZED: "未授权", CALIBRATION_REQUIRED: "需要标定", NO_COVERAGE: "暂无覆盖",
    WAITING_FOR_LIVE_DATA: "等待实时数据", WAITING_FOR_SURFACE: "等待表面数据",
    UNASSIGNED: "未分配", IDLE: "空闲", FINALIZING: "正在封存", STOPPED: "已停止",
    SESSION_IDLE: "会话空闲", RAW_RECORDING_ACTIVE: "原始录制中", RAW_RECORDING_IDLE: "原始录制空闲",
    RAW_RECORDING_STOPPED: "原始录制已停止", RAW_RECORDING_COMPLETE: "原始录制完成",
    NOT_GLOBAL: "非全局坐标", NOT_ROOM_OR_BUILDING: "非房间或楼栋坐标", NOT_ASSEMBLY_AUTHORITY: "非装配依据",
    NO_IMAGERY: "暂无影像", AWAITING_ALIGNED_RGB_D_KEYFRAME: "等待对齐 RGB-D 关键帧",
    CAMERA_PREVIEW_UNAVAILABLE: "相机预览不可用", DISTANCE_SELECT_TWO_POINTS: "距离：请选择两个点",
    SESSION_ARCHIVED: "会话已归档", CAMERA_KEYFRAME_DEFERRED: "相机关键帧已延后 / 优先显示预览",
    LIVE_RGB_DEPTH_DEFERRED: "实时彩色 / 深度已延后", LIVE_RGB_KEYFRAME: "实时彩色关键帧",
    LIVE_DEPTH_ALIGNED: "实时深度 / 已与彩色对齐", LIVE_DEPTH_CAMERA_RELATIVE: "实时深度 / 相机相对坐标",
    DEPTH_ALIGNED_READY: "深度：已对齐光学坐标 / 可测量", DEPTH_UNALIGNED_RELATIVE: "深度：未对齐光学坐标 / 仅相机相对",
    ARCHIVED_SESSION: "会话已归档", CAMERA_DEFERRED_PREVIEW: "相机已延后 / 优先显示预览",
    DEPTH_ALIGNED: "已对齐", DEPTH_UNALIGNED: "未对齐", TEXTURE_DEPTH_DEFERRED: "深度：已延后 / 优先显示预览",
    TEXTURE_NOT_AUTHORIZED: "纹理未授权", SOURCE_STATUS_STALE: "源状态已过期",
    DIFFERENCE_METRICS_NOT_COMPUTED: "差异指标尚未计算", ALIGNMENT_PENDING_NOT_COMPUTED: "等待对齐 / 尚未计算",
  };
  if (value === null || value === undefined || value === "") return value;
  return String(value).split(" / ").map((part) => dictionary[part] || part).join(" / ");
};
const STATIC_MEASURED_SURFACE_COMPATIBILITY_MANIFEST = "/static/compatibility/measured-surface-current.json";

const dashboardUrl = new URL(location.href);
if (dashboardUrl.searchParams.has("view")) {
  dashboardUrl.searchParams.delete("view");
  const query = dashboardUrl.searchParams.toString();
  const canonicalUrl = `${dashboardUrl.pathname}${query ? `?${query}` : ""}${dashboardUrl.hash}`;
  history.replaceState(null, "", canonicalUrl);
}
const sessionOdomLocalPreview = new URLSearchParams(location.search).get("surface") === "local";

// The observed triangle surface is primary; the live point layer is a
// diagnostic toggle and remains off until an operator explicitly enables it.
let liveRenderEnabled = false;
document.documentElement.dataset.diagnosticsOpen = "false";
document.title = "GO2 空间建图控制台";

const canvas = byId("scene");
const emptyState = byId("scene-empty");
const statusPanel = byId("status-panel");
const statusTitle = byId("status-title");
const statusMessage = byId("status-message");
const cameraStream = byId("camera-stream");
const cameraCanvas = byId("camera-view");
const cameraContext = cameraCanvas.getContext("2d", { alpha: false });
const cameraEmpty = byId("camera-empty");
const cameraAssist = byId("camera-assist");
const cameraRgbButton = byId("camera-rgb-view");
const cameraDepthButton = byId("camera-depth-view");
const clearMeasurementsButton = byId("clear-measurements");
const cameraFullscreenButton = byId("camera-fullscreen");
let cameraStreamReconnectAfter = Date.now() + 2000;
cameraStream.dataset.visible = "false";
cameraAssist.hidden = sessionOdomLocalPreview;
cameraStream.addEventListener("load", () => syncCameraMedia());
cameraStream.addEventListener("error", () => {
  cameraStreamReconnectAfter = 0;
  syncCameraMedia();
});
cameraStream.src = cameraStream.dataset.src;
const exportMeasuredButton = byId("export-measured");
const exportResult = byId("export-result");
const exportState = byId("export-state");
const exportPly = byId("export-ply");
const exportMetadata = byId("export-metadata");
const surfaceBuildingButton = byId("surface-view-building");
const surfaceLocalButton = byId("surface-view-local");
const surfaceFloorSelect = byId("surface-floor");
const surfaceControls = document.querySelector(".surface-controls");
const mapViewTopButton = byId("map-view-top");
const mapView3dButton = byId("map-view-3d");
const trajectoryCurrentButton = byId("trajectory-current");
const trajectoryHistoryButton = byId("trajectory-history");
const trajectorySessionSelect = byId("trajectory-session");
const trajectorySessionControl = document.querySelector(".trajectory-session-control");
const mapTextureStatus = byId("map-texture-status");
const geometryProvenance = byId("geometry-provenance");
const geometryQuality = byId("geometry-quality");
const geometryTransform = byId("geometry-transform");
const diagnosticsDrawer = byId("diagnostics-drawer");
const followRobotButton = byId("follow-robot");
const livePointCloudButton = byId("live-point-cloud");
const liveCoverageNode = byId("live-coverage");
const liveSensorHealthNode = byId("live-sensor-health");
const sceneLayerMeasuredButton = byId("scene-layer-measured");
const sceneLayerLiveButton = byId("scene-layer-live");
const sceneLayerCompareButton = byId("scene-layer-compare");
const liveAlignmentNode = byId("live-alignment");
const mappingReviewStateNode = byId("mapping-review-state");
const mappingReviewCoarseNode = byId("mapping-review-coarse");
const mappingReviewObservedNode = byId("mapping-review-observed");
const mappingReviewTrajectoryNode = byId("mapping-review-trajectory");
const mappingReviewReconstructionNode = byId("mapping-review-reconstruction");
const mappingReviewCoverageNode = byId("mapping-review-coverage");
const mappingReviewDifferenceNode = byId("mapping-review-difference");
const robotControlStateNode = byId("robot-control-state");
const robotControlLinkNode = byId("robot-control-link");
const robotControlLocomotionNode = byId("robot-control-locomotion");
const robotControlNavigationNode = byId("robot-control-navigation");
const robotControlSensorsNode = byId("robot-control-sensors");
const robotControlGatesNode = byId("robot-control-gates");
const coverageAuditStateNode = byId("coverage-audit-state");
const coverageAuditUnseenNode = byId("coverage-audit-unseen");
const coverageAuditNextNode = byId("coverage-audit-next");
const coverageAuditRouteNode = byId("coverage-audit-route");
const coverageAuditGateNode = byId("coverage-audit-gate");
const coverageAuditFocusButton = byId("coverage-audit-focus");
diagnosticsDrawer.addEventListener("toggle", () => {
  document.documentElement.dataset.diagnosticsOpen = String(diagnosticsDrawer.open);
  if (diagnosticsDrawer.open) refreshSnapshot(true);
  requestAnimationFrame(resize);
});

const surfaceTileWorker = new Worker("/static/surface-tile-worker.js", { type: "module" });
const surfaceTileWorkerRequests = new Map();
let surfaceTileWorkerSequence = 0;
if (surfaceTileWorker) {
  surfaceTileWorker.onmessage = (event) => {
    const request = surfaceTileWorkerRequests.get(event.data?.id);
    if (!request) return;
    surfaceTileWorkerRequests.delete(event.data.id);
    if (event.data.ok) request.resolve(event.data.result);
    else request.reject(new Error(event.data.error || "surface worker failed"));
  };
  surfaceTileWorker.onerror = (event) => {
    for (const request of surfaceTileWorkerRequests.values()) request.reject(new Error(event.message));
    surfaceTileWorkerRequests.clear();
  };
}

function parseSurfaceTileInWorker(buffer, expectedSha256) {
  if (!surfaceTileWorker) return Promise.reject(new Error("surface tile worker is unavailable"));
  const id = ++surfaceTileWorkerSequence;
  return new Promise((resolve, reject) => {
    surfaceTileWorkerRequests.set(id, { resolve, reject });
    surfaceTileWorker.postMessage({ id, operation: "parse_surface_tile", buffer, expectedSha256 }, [buffer]);
  });
}

function verifyBytesInWorker(buffer, expectedSha256) {
  if (!surfaceTileWorker) return Promise.reject(new Error("surface tile worker is unavailable"));
  const id = ++surfaceTileWorkerSequence;
  return new Promise((resolve, reject) => {
    surfaceTileWorkerRequests.set(id, { resolve, reject });
    surfaceTileWorker.postMessage({ id, operation: "verify_bytes", buffer, expectedSha256 }, [buffer]);
  });
}

const legacyPlyWorker = new Worker("/static/legacy-ply-worker.js", { type: "module" });
const legacyPlyWorkerRequests = new Map();
let legacyPlyWorkerSequence = 0;
legacyPlyWorker.onmessage = (event) => {
  const request = legacyPlyWorkerRequests.get(event.data?.id);
  if (!request) return;
  legacyPlyWorkerRequests.delete(event.data.id);
  if (event.data.ok) request.resolve(event.data.result);
  else request.reject(new Error(event.data.error || "legacy PLY worker failed"));
};
legacyPlyWorker.onerror = (event) => {
  for (const request of legacyPlyWorkerRequests.values()) request.reject(new Error(event.message));
  legacyPlyWorkerRequests.clear();
};

function parseLegacyPlyInWorker(buffer, status) {
  const id = ++legacyPlyWorkerSequence;
  return new Promise((resolve, reject) => {
    legacyPlyWorkerRequests.set(id, { resolve, reject });
    legacyPlyWorker.postMessage({
      id,
      buffer,
      expectedSha256: status.surface_sha256,
      expectedVertexCount: status.vertex_count,
      expectedTriangleCount: status.triangle_count,
    }, [buffer]);
  });
}
const metric = {
  quality: byId("quality"), indicator: byId("indicator"), points: byId("points"), sequence: byId("sequence"),
  uplink: byId("uplink"), ackRtt: byId("ack-rtt"), inputRate: byId("input-rate"), frameAge: byId("frame-age"),
  residual: byId("residual"), retries: byId("retries"), pose: byId("pose"), capture: byId("capture"),
  gate: byId("gate"), source: byId("source"), clock: byId("clock"), densePoints: byId("dense-points"),
  rawIngress: byId("raw-ingress"), verifiedArtifacts: byId("verified-artifacts"), sceneLabel: byId("scene-label"),
  reconstructedArtifacts: byId("reconstructed-artifacts"),
  sessionState: byId("session-state"), linkBudget: byId("link-budget"), previewBudget: byId("preview-budget"),
  highBudget: byId("high-budget"), observedCapacity: byId("observed-capacity"),
  sceneLegend: byId("scene-legend"),
  cameraState: byId("camera-state"), cameraPointA: byId("camera-point-a"), cameraPointB: byId("camera-point-b"),
  cameraDistance: byId("camera-distance"), textureState: byId("texture-state"),
  depthCorrespondences: byId("depth-correspondences"), depthResidualP95: byId("depth-residual-p95"),
  textureCoverage: byId("texture-coverage"), textureBlocker: byId("texture-blocker"),
  roomScanState: byId("room-scan-state"), roomScanFrames: byId("room-scan-frames"),
  roomScanSourcePoints: byId("room-scan-source-points"), roomScanTrajectory: byId("room-scan-trajectory"),
  roomScanExtent: byId("room-scan-extent"),
  buildingScanState: byId("building-scan-state"), buildingFloor: byId("building-floor"),
  buildingSubmap: byId("building-submap"), surfaceSurfels: byId("surface-surfels"),
  surfaceTriangles: byId("surface-triangles"), surfaceTiles: byId("surface-tiles"),
  surfaceUnresolved: byId("surface-unresolved"),
  appearanceState: byId("appearance-state"), coloredCoverage: byId("colored-coverage"),
  texturedCoverage: byId("textured-coverage"), appearanceViews: byId("appearance-views"),
  appearanceThresholds: byId("appearance-thresholds"), appearanceDelivery: byId("appearance-delivery"),
  physicalDepthAcceptance: byId("physical-depth-acceptance"), assistanceCoverage: byId("assistance-coverage"),
  appearanceDensity: byId("appearance-density"), appearanceLod: byId("appearance-lod"),
  appearanceRevision: byId("appearance-revision"), buildingAppearanceRevision: byId("building-appearance-revision"),
  appearanceBlocker: byId("appearance-blocker"),
  simulationBadge: byId("simulation-badge"),
};

let latestSnapshot = null;
let latestCameraStatus = null;
let latestCameraFrame = null;
let latestCameraSequence = -1;
let latestCameraIdentity = "";
let cameraLoadInFlight = false;
let measurements = [];
let cameraView = "color";

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x06100f);
scene.fog = new THREE.Fog(0x06100f, 14, 52);
const perspectiveCamera = new THREE.PerspectiveCamera(52, 1, 0.03, 400);
const orthographicCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.03, 400);
let mapProjectionMode = "top";
let camera = orthographicCamera;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.localClippingEnabled = true;
const currentFloorCutaway = new THREE.Plane(new THREE.Vector3(0, -1, 0), 2.0);

const grid = new THREE.GridHelper(24, 24, 0x2d655a, 0x173c35);
grid.position.y = -0.02;
scene.add(grid);

const cloudGeometry = new THREE.BufferGeometry();
const cloud = new THREE.Points(cloudGeometry, new THREE.PointsMaterial({
  size: 0.035, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: 0.92, depthWrite: false,
}));
cloud.visible = false;
scene.add(cloud);

const surfaceGroup = new THREE.Group();
surfaceGroup.name = "observed-building-surface";
scene.add(surfaceGroup);
const coarseReferenceGroup = new THREE.Group();
coarseReferenceGroup.name = "coarse-user-prior-reference";
coarseReferenceGroup.visible = false;
scene.add(coarseReferenceGroup);
let sceneLayerMode = "measured";
let coarseReferenceLoadedFor = "";
const surfaceMaterial = new THREE.MeshStandardMaterial({
  color: 0x9fd8cf,
  roughness: 0.82,
  metalness: 0.04,
  side: THREE.DoubleSide,
  transparent: true,
  opacity: 0.94,
});
surfaceMaterial.clippingPlanes = [currentFloorCutaway];
surfaceMaterial.userData = { sharedSurfaceMaterial: true };
scene.add(new THREE.HemisphereLight(0xdff7f1, 0x182b28, 1.35));
const surfaceLight = new THREE.DirectionalLight(0xffe6b0, 1.1);
surfaceLight.position.set(5, 9, 4);
scene.add(surfaceLight);

const tileMeshes = new Map();
const currentTileRevisions = new Map();
let measuredSurfaceMesh = null;
let measuredSurfaceState = null;
let measuredSurfaceSignature = "";
let measuredSurfaceInFlight = false;
let measuredSurfaceRetryAfterMs = 0;
let measuredSurfaceSourceStatus = null;
let measuredAppearanceRequestIdentity = "";
let measuredAppearanceState = "TEXTURE_NOT_AUTHORIZED";
let measuredAppearanceDetails = "";
let measuredAppearanceSurfaceActive = false;
let surfaceViewMode = "local";
if (sessionOdomLocalPreview) {
  surfaceViewMode = "local";
  surfaceBuildingButton.setAttribute("aria-pressed", "false");
  surfaceLocalButton.setAttribute("aria-pressed", "true");
}
if (new URLSearchParams(location.search).get("render") === "surface") {
  surfaceViewMode = "building";
  surfaceBuildingButton.setAttribute("aria-pressed", "true");
  surfaceLocalButton.setAttribute("aria-pressed", "false");
}
let activeSurfaceSubmap = null;
let surfaceManifestInFlight = false;
let surfaceManifestSignature = "";
let requestedSceneRevisionId = "";
let activeBuildingId = "";
let surfaceFramePending = false;
let buildingAppearanceObject = null;
let buildingAppearanceRevisionId = "";
let buildingAppearanceRequestIdentity = "";
let buildingAppearanceInFlight = false;
let buildingAppearanceRetryAfterMs = 0;
let buildingAppearanceLodRefreshAt = 0;
let appearanceTelemetryState = normalizeAppearanceTelemetry();

const trajectoryGeometry = new THREE.BufferGeometry();
const trajectory = new THREE.Line(trajectoryGeometry, new THREE.LineBasicMaterial({ color: 0xb8e96b, transparent: true, opacity: 0.94 }));
scene.add(trajectory);
const historicalTrajectoryGroup = new THREE.Group();
historicalTrajectoryGroup.name = "historical-trajectories";
historicalTrajectoryGroup.visible = false;
scene.add(historicalTrajectoryGroup);
let historyVisible = false;
let historicalTrajectoryInFlight = false;
let historicalTrajectoryRevision = "";
let historicalTrajectoryPayload = null;
let followRobotEnabled = true;
let automaticFloorSelection = true;
let coverageAuditInFlight = false;
let coverageAuditTask = null;
let latestCoarseMap = null;
let latestCoveragePayload = null;
let robotControlRequestInFlight = false;

function renderRobotControlStatus(status = {}) {
  const capabilities = status.capabilities || {};
  const locomotion = status.locomotion || {};
  const navigation = status.navigation || {};
  const sensorValues = Object.values(capabilities.sensors || {});
  const verifiedSensors = sensorValues.filter((sensor) => sensor?.verified === true).length;
  const sensorLabel = sensorValues.length ? `${verifiedSensors}/${sensorValues.length} VERIFIED` : "UNVERIFIED";
  const linkLabel = status.link || "GO2_LINK_UNVERIFIED";
  const locomotionLabel = locomotion.actuation_authorized === true ? "AUTHORIZED" : "ACTUATION_LOCKED";
  const navigationLabel = navigation.nav2?.bridge_state || navigation.nav2?.action_server_state || "UNVERIFIED";
  if (robotControlStateNode) {
    robotControlStateNode.textContent = uiText(status.error ? "UNAVAILABLE" : status.live ? "READ_ONLY_LIVE" : "READ_ONLY");
    robotControlStateNode.dataset.state = status.error ? "error" : status.live ? "ready" : "warning";
  }
  if (robotControlLinkNode) robotControlLinkNode.textContent = uiText(linkLabel);
  if (robotControlLocomotionNode) robotControlLocomotionNode.textContent = uiText(locomotionLabel);
  if (robotControlNavigationNode) robotControlNavigationNode.textContent = uiText(navigationLabel);
  if (robotControlSensorsNode) robotControlSensorsNode.textContent = sensorLabel.replace("VERIFIED", "已验证").replace("UNVERIFIED", "未验证");
  if (robotControlGatesNode) {
    robotControlGatesNode.textContent = status.error
      ? `运动层已锁定 / ${uiText(status.error)}`
      : `运动层已锁定 / ${uiText(locomotion.required_gates?.join(" / ") || "MOTION_LEASE / WATCHDOG / OPERATOR_CONFIRMATION")}`;
  }
}

async function refreshRobotControlStatus() {
  if (robotControlRequestInFlight || document.visibilityState === "hidden") return;
  robotControlRequestInFlight = true;
  try {
    const [capabilitiesResponse, locomotionResponse, navigationResponse] = await Promise.all([
      fetch("/api/robot/capabilities", { cache: "no-store" }),
      fetch("/api/robot/locomotion/status", { cache: "no-store" }),
      fetch("/api/robot/navigation/status", { cache: "no-store" }),
    ]);
    if (!capabilitiesResponse.ok || !locomotionResponse.ok || !navigationResponse.ok) {
      throw new Error(`STATUS HTTP ${capabilitiesResponse.status}/${locomotionResponse.status}/${navigationResponse.status}`);
    }
    const capabilities = await capabilitiesResponse.json();
    const locomotion = await locomotionResponse.json();
    const navigation = await navigationResponse.json();
    renderRobotControlStatus({
      capabilities,
      locomotion,
      navigation: { nav2: capabilities.navigation?.nav2 || {}, state: navigation.state },
      link: capabilities.capture?.tailnet_connectivity || "GO2_LINK_UNVERIFIED",
      live: liveSocketConnected && Date.now() - lastSocketMessageAt < 10000,
    });
  } catch (error) {
    renderRobotControlStatus({ error: String(error.message || error) });
  } finally {
    robotControlRequestInFlight = false;
  }
}

renderRobotControlStatus();

function coarseMapExtent(map) {
  const polygons = [...(map?.rooms || []), ...(map?.corridors || [])]
    .map((item) => item?.polygon)
    .filter((polygon) => Array.isArray(polygon) && polygon.length >= 3);
  const points = polygons.flat().filter((point) => Array.isArray(point) && point.length >= 2).map((point) => [Number(point[0]), Number(point[1])]);
  if (!points.length || points.some((point) => !point.every(Number.isFinite))) return null;
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  return [Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
}

function mappingExtentText(extent) {
  return Array.isArray(extent) && extent.length >= 2 && extent.slice(0, 2).every((value) => Number.isFinite(Number(value)))
    ? `${Number(extent[0]).toFixed(2)} x ${Number(extent[1]).toFixed(2)} 米`
    : "暂无";
}

function updateMappingReview(snapshot = latestSnapshot) {
  const live = snapshot?.live_map || {};
  const roomScan = snapshot?.room_scan || live.room_scan || {};
  const dense = snapshot?.dense_map || {};
  const building = snapshot?.building_scan || {};
  const coarseExtent = coarseMapExtent(latestCoarseMap);
  const observedExtent = Array.isArray(roomScan.extent_m) ? roomScan.extent_m : null;
  const trajectoryLength = Number(roomScan.trajectory_length_m);
  const hasObserved = observedExtent?.length === 3 || Number(live.points || dense.points || 0) > 0;
  const reconstruction = building.state || (snapshot?.session?.state === "FINALIZING" ? "FINALIZING" : "WAITING_FOR_SURFACE");
  const coverage = latestCoveragePayload?.coverage_percent;
  if (mappingReviewCoarseNode) mappingReviewCoarseNode.textContent = mappingExtentText(coarseExtent);
  if (mappingReviewObservedNode) mappingReviewObservedNode.textContent = observedExtent?.length === 3 ? mappingExtentText(observedExtent) + ` x ${Number(observedExtent[2]).toFixed(2)} 米` : hasObserved ? "已有点云 / 范围待定" : "暂无";
  if (mappingReviewTrajectoryNode) mappingReviewTrajectoryNode.textContent = Number.isFinite(trajectoryLength) ? `${trajectoryLength.toFixed(2)} 米` : Array.isArray(live.trajectory) && live.trajectory.length ? `${live.trajectory.length} 个姿态样本` : "暂无";
  if (mappingReviewReconstructionNode) mappingReviewReconstructionNode.textContent = uiText(String(reconstruction).toUpperCase());
  if (mappingReviewCoverageNode) mappingReviewCoverageNode.textContent = Number.isFinite(Number(coverage)) ? `${Number(coverage).toFixed(1)}%` : latestCoveragePayload ? "暂无已验证任务" : "暂无";
  const aligned = snapshot?.coordinate_frame && latestCoarseMap?.coordinate_frame && snapshot.coordinate_frame === latestCoarseMap.coordinate_frame && hasObserved;
  if (mappingReviewStateNode) {
    mappingReviewStateNode.textContent = aligned ? "坐标可比较" : latestCoarseMap && hasObserved ? "仅参考叠加" : "等待证据";
    mappingReviewStateNode.dataset.state = aligned ? "ready" : "warning";
  }
  if (mappingReviewDifferenceNode) mappingReviewDifferenceNode.textContent = aligned ? uiText("DIFFERENCE_METRICS_NOT_COMPUTED") : uiText("ALIGNMENT_PENDING_NOT_COMPUTED");
}

window.addEventListener("go2:coarse-map", (event) => {
  latestCoarseMap = event.detail || null;
  updateMappingReview();
});
window.addEventListener("go2:coverage", (event) => {
  latestCoveragePayload = event.detail || null;
  updateMappingReview();
});

function routeLength(waypoints) {
  if (!Array.isArray(waypoints) || waypoints.length < 2) return null;
  let total = 0;
  for (let index = 1; index < waypoints.length; index += 1) {
    const previous = waypoints[index - 1];
    const current = waypoints[index];
    if (!Array.isArray(previous) || !Array.isArray(current) || previous.length < 2 || current.length < 2) return null;
    const dx = Number(current[0]) - Number(previous[0]);
    const dy = Number(current[1]) - Number(previous[1]);
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return null;
    total += Math.hypot(dx, dy);
  }
  return total;
}

function taskRoutePoints(task) {
  if (Array.isArray(task?.waypoints) && task.waypoints.length >= 2) {
    return { points: task.waypoints, source: "WAYPOINTS" };
  }
  if (Array.isArray(task?.room_polygon) && task.room_polygon.length >= 2) {
    return { points: task.room_polygon, source: "POLYGON_FALLBACK" };
  }
  return { points: [], source: "MISSING" };
}

const robot = new THREE.Group();
const body = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 10), new THREE.MeshBasicMaterial({ color: 0xff7867 }));
const heading = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.16, 12), new THREE.MeshBasicMaterial({ color: 0xffc95c }));
heading.rotation.x = Math.PI / 2;
heading.position.z = 0.06;
robot.add(body, heading);
scene.add(robot);

const orbit = { theta: 0, phi: 0.005, radius: 10, target: new THREE.Vector3(), framed: false };
const obliqueOrbit = { theta: -0.72, phi: 1.05 };
let lastLegacySurfacePortrait = null;

function applyAssemblyTransform(point, metadata, rotationOnly = false) {
  const vector = new THREE.Vector3(Number(point[0]), Number(point[1]), Number(point[2]));
  if (![vector.x, vector.y, vector.z].every(Number.isFinite)) {
    throw new Error("surface tile contains a non-finite vector");
  }
  const values = metadata?.transform_building_from_submap;
  if (values == null) return vector;
  if (!Array.isArray(values) || values.length !== 16 || !values.every((value) => Number.isFinite(Number(value)))) {
    throw new Error("surface tile assembly transform is invalid");
  }
  const transform = new THREE.Matrix4();
  transform.set(...values.map(Number));
  if (rotationOnly) transform.setPosition(0, 0, 0);
  vector.applyMatrix4(transform);
  return rotationOnly ? vector.normalize() : vector;
}

const positionToScene = (point, elevation = 0) => new THREE.Vector3(point[0], point[2] + elevation, -point[1]);

function clearCoarseReference() {
  for (const child of [...coarseReferenceGroup.children]) {
    coarseReferenceGroup.remove(child);
    child.geometry.dispose();
    child.material.dispose();
  }
}

async function loadCoarseReference(mapId = "go2-demo-floor") {
  if (coarseReferenceLoadedFor === mapId) return;
  try {
    const response = await fetch(`/api/maps/coarse/${encodeURIComponent(mapId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`COARSE REFERENCE HTTP ${response.status}`);
    const payload = await response.json();
    const areas = [...(payload.rooms || []), ...(payload.corridors || []), ...(payload.stairs || []), ...(payload.keep_out_zones || [])];
    const staged = [];
    for (const area of areas) {
      if (!Array.isArray(area.polygon) || area.polygon.length < 3) continue;
      const points = area.polygon.map((point) => positionToScene([Number(point[0]), 0, Number(point[1])], 0.025));
      if (!points.every((point) => [point.x, point.y, point.z].every(Number.isFinite))) continue;
      const geometry = new THREE.BufferGeometry().setFromPoints(points.concat([points[0]]));
      const material = new THREE.LineBasicMaterial({ color: area.kind === "keep_out_zone" ? 0xff7867 : 0xffc95c, transparent: true, opacity: 0.85 });
      const line = new THREE.Line(geometry, material);
      line.userData = { source: "USER_PRIOR", area_id: area.id, kind: area.kind || "room" };
      staged.push(line);
    }
    clearCoarseReference();
    staged.forEach((line) => coarseReferenceGroup.add(line));
    coarseReferenceLoadedFor = mapId;
  } catch (_error) {
    clearCoarseReference();
    coarseReferenceLoadedFor = "";
  }
}

function applySceneLayerVisibility() {
  const compare = sceneLayerMode === "compare";
  const live = sceneLayerMode === "live";
  surfaceGroup.visible = !live;
  coarseReferenceGroup.visible = compare;
  cloud.visible = liveRenderEnabled && (live || compare);
  sceneLayerMeasuredButton?.setAttribute("aria-pressed", String(sceneLayerMode === "measured"));
  sceneLayerLiveButton?.setAttribute("aria-pressed", String(live));
  sceneLayerCompareButton?.setAttribute("aria-pressed", String(compare));
}

function setSceneLayerMode(mode) {
  sceneLayerMode = ["measured", "live", "compare"].includes(mode) ? mode : "measured";
  if (sceneLayerMode !== "measured") liveRenderEnabled = true;
  applySceneLayerVisibility();
  if (sceneLayerMode !== "measured") void refreshFullSnapshot();
  if (latestSnapshot) apply(latestSnapshot);
}

class StationaryPoseFilter {
  constructor({ stationaryRadiusM, releaseSamples, positionSmoothing, headingSmoothing }) {
    this.stationaryRadiusM = stationaryRadiusM;
    this.releaseSamples = releaseSamples;
    this.positionSmoothing = positionSmoothing;
    this.headingSmoothing = headingSmoothing;
    this.anchorPosition = null;
    this.displayPosition = null;
    this.displayHeadingXY = [1, 0];
    this.outsideSamples = 0;
    this.settleSamples = 0;
    this.isLocked = true;
  }

  static normalizedHeading(headingXY) {
    const x = Number(headingXY[0]);
    const y = Number(headingXY[1]);
    const length = Math.hypot(x, y);
    return Number.isFinite(length) && length > 1e-6 ? [x / length, y / length] : [1, 0];
  }

  static positionDistance(left, right) {
    return Math.hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2]);
  }

  update(rawPosition, rawHeadingXY) {
    const position = rawPosition.map((value) => Number.isFinite(Number(value)) ? Number(value) : 0);
    const headingXY = StationaryPoseFilter.normalizedHeading(rawHeadingXY);
    if (!this.displayPosition) {
      this.anchorPosition = [...position];
      this.displayPosition = [...position];
      this.displayHeadingXY = headingXY;
      return { position: [...this.displayPosition], headingXY: [...this.displayHeadingXY] };
    }

    if (this.isLocked) {
      if (StationaryPoseFilter.positionDistance(position, this.anchorPosition) <= this.stationaryRadiusM) {
        this.outsideSamples = 0;
        return { position: [...this.displayPosition], headingXY: [...this.displayHeadingXY] };
      }
      this.outsideSamples += 1;
      if (this.outsideSamples < this.releaseSamples) {
        return { position: [...this.displayPosition], headingXY: [...this.displayHeadingXY] };
      }
      this.isLocked = false;
      this.outsideSamples = 0;
    }

    const distanceBeforeSmoothing = StationaryPoseFilter.positionDistance(position, this.displayPosition);
    this.displayPosition = this.displayPosition.map(
      (value, index) => value + (position[index] - value) * this.positionSmoothing,
    );
    this.displayHeadingXY = StationaryPoseFilter.normalizedHeading(this.displayHeadingXY.map(
      (value, index) => value + (headingXY[index] - value) * this.headingSmoothing,
    ));
    this.settleSamples = distanceBeforeSmoothing <= this.stationaryRadiusM * 0.25
      ? this.settleSamples + 1
      : 0;
    if (this.settleSamples >= this.releaseSamples) {
      this.anchorPosition = [...position];
      this.displayPosition = [...position];
      this.displayHeadingXY = headingXY;
      this.settleSamples = 0;
      this.isLocked = true;
    }
    return { position: [...this.displayPosition], headingXY: [...this.displayHeadingXY] };
  }
}

const stationaryPoseFilter = new StationaryPoseFilter({
  stationaryRadiusM: 0.08,
  releaseSamples: 3,
  positionSmoothing: 0.35,
  headingSmoothing: 0.3,
});

function mapProjectionPreset(mode, obliqueTheta = -0.72, obliquePhi = 1.05) {
  if (mode === "top") return { mode, theta: 0, phi: 0.005 };
  if (mode !== "oblique") throw new Error("unknown map projection mode");
  const theta = Number.isFinite(obliqueTheta) ? obliqueTheta : -0.72;
  const sourcePhi = Number.isFinite(obliquePhi) ? obliquePhi : 1.05;
  const phi = Math.min(Math.PI - 0.18, Math.max(0.18, sourcePhi));
  return { mode, theta, phi };
}

function topDownPanDelta(deltaX, deltaY, viewportHeight, visibleHeight) {
  if (![deltaX, deltaY, viewportHeight, visibleHeight].every(Number.isFinite)
      || viewportHeight <= 0 || visibleHeight <= 0) {
    throw new Error("top-down pan inputs are invalid");
  }
  const metersPerPixel = visibleHeight / viewportHeight;
  return { x: -deltaX * metersPerPixel, z: -deltaY * metersPerPixel };
}

function mapImageryLabel(state) {
  if (state === "VALIDATED") return "PHOTO ATLAS";
  if (state === "PROVISIONAL") return "LIVE RGB";
  if (state === "VERTEX_COLOR") return "VERTEX COLOR";
  return uiText("NO_IMAGERY");
}

function renderMapImageryState(state) {
  mapTextureStatus.dataset.state = state;
  mapTextureStatus.textContent = mapImageryLabel(state);
}

function orthographicVisibleHeight() {
  return Math.max(1.2, orbit.radius * 0.82);
}

function updateCameraProjection(
  width = Math.max(canvas.clientWidth, 1),
  height = Math.max(canvas.clientHeight, 1),
) {
  const aspect = width / height;
  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();
  const halfHeight = orthographicVisibleHeight() / 2;
  orthographicCamera.left = -halfHeight * aspect;
  orthographicCamera.right = halfHeight * aspect;
  orthographicCamera.top = halfHeight;
  orthographicCamera.bottom = -halfHeight;
  orthographicCamera.updateProjectionMatrix();
}

function setMapProjectionMode(mode) {
  if (mapProjectionMode === "oblique") {
    obliqueOrbit.theta = orbit.theta;
    obliqueOrbit.phi = orbit.phi;
  }
  const preset = mapProjectionPreset(mode, obliqueOrbit.theta, obliqueOrbit.phi);
  mapProjectionMode = preset.mode;
  orbit.theta = preset.theta;
  orbit.phi = preset.phi;
  camera = mapProjectionMode === "top" ? orthographicCamera : perspectiveCamera;
  mapViewTopButton.setAttribute("aria-pressed", String(mapProjectionMode === "top"));
  mapView3dButton.setAttribute("aria-pressed", String(mapProjectionMode === "oblique"));
  canvas.dataset.projection = mapProjectionMode;
  updateCameraProjection();
  updateCamera();
}

function updateCamera() {
  const sinPhi = Math.sin(orbit.phi);
  camera.up.set(0, 1, 0);
  camera.position.set(
    orbit.target.x + orbit.radius * sinPhi * Math.sin(orbit.theta),
    orbit.target.y + orbit.radius * Math.cos(orbit.phi),
    orbit.target.z + orbit.radius * sinPhi * Math.cos(orbit.theta),
  );
  if (mapProjectionMode === "top") camera.up.set(0, 0, -1);
  camera.lookAt(orbit.target);
  updateCameraProjection();
}

function frameSubmap(points, position) {
  const bounds = new THREE.Box3();
  for (const point of points) bounds.expandByPoint(positionToScene(point));
  bounds.expandByPoint(positionToScene(position));
  if (bounds.isEmpty()) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  orbit.target.copy(center);
  orbit.radius = Math.max(4.5, Math.min(32, size.length() * 1.25));
  orbit.framed = true;
  updateCamera();
}

function frameSurface() {
  const bounds = new THREE.Box3();
  for (const child of surfaceGroup.children) {
    if (child.visible) bounds.expandByObject(child);
  }
  if (bounds.isEmpty()) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const legacySurfaceFrame = measuredSurfaceState?.sourceMode === "LEGACY_SESSION_READ_ONLY";
  if (legacySurfaceFrame) lastLegacySurfacePortrait = canvas.clientHeight > canvas.clientWidth;
  orbit.target.copy(center);
  if (legacySurfaceFrame && mapProjectionMode === "oblique") orbit.phi = 0.72;
  let targetRadius = legacySurfaceFrame ? size.length() * 0.88 : size.length() * 1.05;
  if (legacySurfaceFrame && canvas.clientHeight > canvas.clientWidth) {
    targetRadius = size.length() * 1.45;
  }
  orbit.radius = Math.max(1.2, Math.min(140, targetRadius));
  grid.scale.setScalar(legacySurfaceFrame ? 4 : 1);
  scene.fog.near = Math.max(14, orbit.radius * 0.55);
  scene.fog.far = Math.max(52, orbit.radius * 2.1);
  orbit.framed = true;
  surfaceFramePending = false;
  updateCamera();
}

const placementRevisionId = (metadata) => metadata.placement_revision_id || metadata.revision_id;

const tileKey = (metadata) => [
  metadata.building_id,
  metadata.floor_id,
  metadata.submap_id,
  ...(metadata.tile_id || []),
  metadata.assembly_id || "local",
  placementRevisionId(metadata),
].join("/");

function disposeTile(placementId) {
  const entry = tileMeshes.get(placementId);
  if (!entry) return;
  surfaceGroup.remove(entry.mesh);
  entry.mesh.geometry.dispose();
  if (entry.mesh.material !== surfaceMaterial) {
    entry.mesh.material.userData?.provisionalTexture?.dispose?.();
    entry.mesh.material.dispose();
  }
  tileMeshes.delete(placementId);
}

function disposeMeasuredSurfacePreview() {
  if (!measuredSurfaceMesh) return;
  measuredAppearanceRequestIdentity = "";
  measuredAppearanceSurfaceActive = false;
  surfaceGroup.remove(measuredSurfaceMesh);
  if (measuredSurfaceMesh.material !== surfaceMaterial) {
    measuredSurfaceMesh.material.dispose();
  }
  measuredSurfaceMesh.geometry.dispose();
  measuredSurfaceMesh = null;
  measuredSurfaceState = null;
  measuredSurfaceSignature = "";
}

function buildMeasuredSurfacePreview(surface) {
  const allowedObservationLayers = new Set([
    ["LIDAR_MEASURED", "LIDAR_MEASURED"],
    ["LIO_REGISTERED_LIDAR_OBSERVED", "LIO_REGISTERED_LIDAR_OBSERVED"],
  ].map(([sourceLayer, pointAuthority]) => `${sourceLayer}:${pointAuthority}`));
  if (!allowedObservationLayers.has(`${surface.source_layer}:${surface.point_authority}`)
      || surface.geometry_authority !== false) {
    throw new Error("measured surface preview authority is invalid");
  }
  const status = measuredSurfaceSourceStatus || {};
  const isSessionOdomLocal = status.source_mode === "SESSION_ODOM_LOCAL_READ_ONLY";
  if (isSessionOdomLocal && (
    status.read_only !== true
    || status.mapping_source_mode !== "RECORDED_LIO_SESSION_ODOM_CLOUD"
    || status.coordinate_scope !== "SESSION_ODOM_LOCAL"
    || status.coordinate_frame !== "odom"
    || status.room_or_building !== false
    || status.texture_authorized !== false
    || status.sensor_frame_transform_applied !== false
  )) {
    throw new Error("local odom surface preview authority is invalid");
  }
  if (sessionOdomLocalPreview !== isSessionOdomLocal) {
    throw new Error("surface endpoint does not match the requested coordinate scope");
  }
  const quality = surface.quality || status.quality;
  const sensorFrameTransformId = surface.sensor_frame_transform_id || status.sensor_frame_transform_id;
  if (quality !== "PREVIEW_UNCALIBRATED") {
    throw new Error("measured surface preview quality is invalid");
  }
  if (status.sensor_frame_transform_id
      && status.sensor_frame_transform_id !== sensorFrameTransformId) {
    throw new Error("measured surface preview transform identity is inconsistent");
  }
  const sourcePositions = Array.isArray(surface.positions) ? surface.positions : [];
  const sourceNormals = Array.isArray(surface.normals) ? surface.normals : [];
  const sourceTriangles = Array.isArray(surface.triangles) ? surface.triangles : [];
  if (sourcePositions.length === 0
      || sourcePositions.length % 3 !== 0
      || sourceNormals.length !== sourcePositions.length
      || sourceTriangles.length === 0
      || sourceTriangles.length % 3 !== 0) {
    throw new Error("measured surface preview arrays are invalid");
  }
  const vertexCount = sourcePositions.length / 3;
  const positions = new Float32Array(sourcePositions.length);
  const normals = new Float32Array(sourceNormals.length);
  for (let offset = 0; offset < sourcePositions.length; offset += 3) {
    const mapPosition = sourcePositions.slice(offset, offset + 3).map(Number);
    const mapNormal = sourceNormals.slice(offset, offset + 3).map(Number);
    if (![...mapPosition, ...mapNormal].every(Number.isFinite)) {
      throw new Error("measured surface preview contains a non-finite vector");
    }
    const scenePosition = positionToScene(mapPosition);
    const sceneNormal = positionToScene(mapNormal).normalize();
    positions.set(scenePosition.toArray(), offset);
    normals.set(sceneNormal.toArray(), offset);
  }
  const triangles = new Uint32Array(sourceTriangles.length);
  sourceTriangles.forEach((value, index) => {
    const vertexIndex = Number(value);
    if (!Number.isInteger(vertexIndex) || vertexIndex < 0 || vertexIndex >= vertexCount) {
      throw new Error("measured surface preview index is invalid");
    }
    triangles[index] = vertexIndex;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(new THREE.BufferAttribute(triangles, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const mesh = new THREE.Mesh(geometry, surfaceMaterial);
  mesh.name = "verified-artifact-measured-surface-preview";
  mesh.userData = {
    source_layer: surface.source_layer,
    geometry_authority: false,
    pose_generation: surface.pose_generation,
    measured_surface_geometry_revision_id: surface.measured_surface_geometry_revision_id || null,
  };
  return {
    mesh,
    state: {
      vertexCount,
      triangleCount: sourceTriangles.length / 3,
      artifactCount: Array.isArray(surface.artifact_sha256) ? surface.artifact_sha256.length : 0,
      reconstructionPhase: surface.reconstruction_phase || "UNKNOWN",
      poseGeneration: surface.pose_generation || "unknown",
      geometryRevisionId: surface.measured_surface_geometry_revision_id || null,
      quality,
      sensorFrameTransformId: sensorFrameTransformId || "SENSOR_FRAME_UNAVAILABLE",
      readOnly: (status.source_mode === "DERIVED_READ_ONLY"
        || status.source_mode === "SESSION_ODOM_LOCAL_READ_ONLY"
        || status.source_mode === "LEGACY_SESSION_READ_ONLY") && status.read_only === true,
      derivedCacheKey: status.derived_cache_key || null,
      sourceMode: status.source_mode || "LIVE_MEASURED_SURFACE",
      surfaceSha256: status.surface_sha256 || null,
      pointAuthority: surface.point_authority || status.point_authority || "UNKNOWN",
      mappingSourceMode: surface.mapping_source_mode || status.mapping_source_mode || "UNKNOWN",
      coordinateScope: surface.coordinate_scope || status.coordinate_scope || null,
      coordinateFrame: surface.coordinate_frame || status.coordinate_frame || null,
      roomOrBuilding: surface.room_or_building ?? status.room_or_building ?? null,
      textureAuthorized: surface.texture_authorized ?? status.texture_authorized ?? null,
    },
  };
}

function buildLegacySurfacePreview(parsed, status) {
  if (status.source_mode !== "LEGACY_SESSION_READ_ONLY" || status.read_only !== true
      || status.quality !== "PREVIEW_UNCALIBRATED" || status.geometry_authority !== false
      || status.hidden_surface_policy !== "LEGACY_RECONSTRUCTION_UNVERIFIED"
      || status.surface_encoding !== "PLY_BINARY_LITTLE_ENDIAN_OPEN3D_V1") {
    throw new Error("legacy surface preview authority is invalid");
  }
  if (!(parsed.positions instanceof Float32Array) || !(parsed.normals instanceof Float32Array)
      || !(parsed.colors instanceof Uint8Array) || !(parsed.indices instanceof Uint32Array)
      || parsed.vertexCount !== status.vertex_count || parsed.triangleCount !== status.triangle_count) {
    throw new Error("legacy surface preview arrays are invalid");
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(parsed.positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(parsed.normals, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(parsed.colors, 3, true));
  geometry.setIndex(new THREE.BufferAttribute(parsed.indices, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const material = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
  });
  material.userData = { legacyReadOnlyMaterial: true };
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "legacy-session-read-only-surface-preview";
  mesh.userData = {
    source_layer: "LEGACY_RECONSTRUCTION",
    geometry_authority: false,
    session_id: status.session_id,
    surface_sha256: status.surface_sha256,
  };
  return {
    mesh,
    state: {
      vertexCount: parsed.vertexCount,
      triangleCount: parsed.triangleCount,
      artifactCount: 1,
      reconstructionPhase: "LEGACY_IMPORT",
      poseGeneration: status.session_id,
      geometryRevisionId: null,
      quality: status.quality,
      sensorFrameTransformId: status.sensor_frame_transform_id,
      readOnly: true,
      derivedCacheKey: null,
      sourceMode: status.source_mode,
      surfaceSha256: status.surface_sha256,
      floorHeightOffsetM: status.floor_height_offset_m,
    },
  };
}

function bytesToHex(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function parseVertexColorV1(buffer, expectedGeometryRevisionId, expectedAppearanceRevisionId, expectedVertexCount) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 80) {
    throw new Error("vertex-color payload is shorter than its header");
  }
  const bytes = new Uint8Array(buffer);
  const magic = new TextDecoder("ascii", { fatal: true }).decode(bytes.subarray(0, 8));
  const view = new DataView(buffer);
  const version = view.getUint16(8, true);
  const headerBytes = view.getUint16(10, true);
  const vertexCount = view.getUint32(12, true);
  if (magic !== "G2PVCOL1" || version !== 1 || headerBytes !== 80) {
    throw new Error("vertex-color header is not RGB8_VALIDITY8_V1");
  }
  if (vertexCount <= 0 || buffer.byteLength !== 80 + vertexCount * 4) {
    throw new Error("vertex-color payload length does not match vertex count");
  }
  const geometryRevisionId = bytesToHex(bytes.subarray(16, 48));
  const appearanceRevisionId = bytesToHex(bytes.subarray(48, 80));
  if (geometryRevisionId !== expectedGeometryRevisionId
      || appearanceRevisionId !== expectedAppearanceRevisionId
      || vertexCount !== expectedVertexCount) {
    throw new Error("vertex-color identity does not match the measured surface");
  }
  const rgb = new Uint8Array(vertexCount * 3);
  const validity = new Uint8Array(vertexCount);
  for (let index = 0; index < vertexCount; index += 1) {
    const source = 80 + index * 4;
    const valid = bytes[source + 3];
    if (valid !== 0 && valid !== 255) throw new Error("vertex-color validity is invalid");
    if (valid === 0 && (bytes[source] !== 0 || bytes[source + 1] !== 0 || bytes[source + 2] !== 0)) {
      throw new Error("invalid vertex color must have zero RGB");
    }
    rgb.set(bytes.subarray(source, source + 3), index * 3);
    validity[index] = valid;
  }
  return { vertexCount, geometryRevisionId, appearanceRevisionId, rgb, validity };
}

function buildVertexColorMaterial(baseMaterial, directRgb = false) {
  if (directRgb) {
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      vertexColors: true,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    material.onBeforeCompile = (shader) => {
      shader.vertexShader = `attribute float appearanceValidity;\nvarying float vAppearanceValidity;\n${shader.vertexShader}`
        .replace("#include <color_vertex>", "#include <color_vertex>\nvAppearanceValidity = appearanceValidity;");
      shader.fragmentShader = `varying float vAppearanceValidity;\n${shader.fragmentShader}`
        .replace(
          "#include <color_fragment>",
          "#ifdef USE_COLOR\n  diffuseColor.rgb = mix(vec3(0.42, 0.48, 0.47), vColor, vAppearanceValidity);\n#endif",
        );
    };
    material.customProgramCacheKey = () => "go2-direct-rgb-validity-v1";
    material.needsUpdate = true;
    material.userData = { directRgbVertexColor: true };
    return material;
  }
  const material = baseMaterial.clone();
  material.vertexColors = true;
  material.userData = { ...material.userData, measuredAppearanceMaterial: true };
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = `attribute float appearanceValidity;\nvarying float vAppearanceValidity;\n${shader.vertexShader}`
      .replace("#include <color_vertex>", "#include <color_vertex>\nvAppearanceValidity = appearanceValidity;");
    shader.fragmentShader = `varying float vAppearanceValidity;\n${shader.fragmentShader}`
      .replace(
        "#include <color_fragment>",
        "#ifdef USE_COLOR\n  diffuseColor.rgb = mix(diffuseColor.rgb, vColor, vAppearanceValidity);\n#endif",
      );
  };
  material.customProgramCacheKey = () => "go2-measured-vertex-color-validity-v1";
  material.needsUpdate = true;
  return material;
}

function buildMeasuredAppearanceCandidate(mesh, parsed, baseMaterial, directRgb = false) {
  const position = mesh.geometry.getAttribute("position");
  if (!position || position.count !== parsed.vertexCount) {
    throw new Error("vertex-color count does not match measured geometry");
  }
  const geometry = mesh.geometry.clone();
  const linearRgb = new Float32Array(parsed.rgb.length);
  for (let index = 0; index < parsed.rgb.length; index += 1) {
    const srgb = parsed.rgb[index] / 255;
    linearRgb[index] = srgb <= 0.04045
      ? srgb / 12.92
      : ((srgb + 0.055) / 1.055) ** 2.4;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(linearRgb, 3));
  geometry.setAttribute("appearanceValidity", new THREE.BufferAttribute(parsed.validity, 1, true));
  return {
    geometry,
    material: buildVertexColorMaterial(baseMaterial, directRgb),
    geometryRevisionId: parsed.geometryRevisionId,
    appearanceRevisionId: parsed.appearanceRevisionId,
  };
}

function commitMeasuredAppearance(
  mesh, candidate, expectedGeometryRevisionId, requestIdentity, activeRequestIdentity, baseMaterial,
) {
  const currentGeometryRevisionId = mesh.userData?.measured_surface_geometry_revision_id;
  if (requestIdentity !== activeRequestIdentity
      || currentGeometryRevisionId !== expectedGeometryRevisionId
      || candidate.geometryRevisionId !== expectedGeometryRevisionId) {
    candidate.geometry.dispose();
    candidate.material.dispose();
    return false;
  }
  const previousGeometry = mesh.geometry;
  const previousMaterial = mesh.material;
  mesh.geometry = candidate.geometry;
  mesh.material = candidate.material;
  mesh.userData.appearance_revision_id = candidate.appearanceRevisionId;
  if (previousGeometry !== candidate.geometry) previousGeometry.dispose();
  if (previousMaterial !== baseMaterial && previousMaterial !== candidate.material) previousMaterial.dispose();
  return true;
}

function normalizeAppearanceTelemetry(input) {
  input = input || {};
  const exactStates = new Set([
    "TEXTURE_NOT_AUTHORIZED", "DEPTH_VALIDATING", "VERTEX_COLOR_READY",
    "ATLAS_BUILDING", "TEXTURED_READY", "TEXTURE_STALE", "TEXTURE_FAILED",
    "DIRECT_RGB_VERTEX_COLOR", "DIRECT_RGB_FAILED",
  ]);
  const validPpm = (value) => Number.isInteger(value) && value >= 0 && value <= 1000000;
  const validCount = (value) => Number.isInteger(value) && value >= 0;
  const coloredCoveragePpm = validPpm(input.coloredCoveragePpm) ? input.coloredCoveragePpm : null;
  const texturedCoveragePpm = validPpm(input.texturedCoveragePpm) ? input.texturedCoveragePpm : null;
  const minimumIndependentViewCount = validCount(input.minimumIndependentViewCount)
    ? input.minimumIndependentViewCount : null;
  const requiredCoveragePpm = 800000;
  const requiredIndependentViewCount = 2;
  const thresholdsExact = input.requiredCoveragePpm === 800000
    && input.requiredIndependentViewCount === 2;
  const deliveryGatePassed = thresholdsExact
    && input.deliveryAuthorized === true
    && texturedCoveragePpm !== null && texturedCoveragePpm >= 800000
    && minimumIndependentViewCount !== null && minimumIndependentViewCount >= 2;
  const blockers = [];
  if (Array.isArray(input.blockers)) {
    blockers.push(...input.blockers.filter((value) => typeof value === "string" && value));
  }
  if (typeof input.blocker === "string" && input.blocker) blockers.push(input.blocker);
  let state = exactStates.has(input.state) ? input.state : "TEXTURE_FAILED";
  if (!exactStates.has(input.state)) blockers.push("UNKNOWN_APPEARANCE_STATE");
  if (state === "TEXTURED_READY" && !deliveryGatePassed) {
    state = "TEXTURE_NOT_AUTHORIZED";
    if (!thresholdsExact) blockers.push("DELIVERY_THRESHOLD_IDENTITY_INVALID");
    if (texturedCoveragePpm === null || texturedCoveragePpm < 800000) {
      blockers.push("MEASURED_SURFACE_COVERAGE_REQUIRED");
    }
    if (minimumIndependentViewCount === null || minimumIndependentViewCount < 2) {
      blockers.push("INDEPENDENT_COLOR_VIEWS_REQUIRED");
    }
    if (input.deliveryAuthorized !== true) blockers.push("DELIVERY_AUTHORIZATION_REQUIRED");
  }
  const appearanceRevisionIds = Array.isArray(input.appearanceRevisionIds)
    ? [...new Set(input.appearanceRevisionIds.filter((value) => typeof value === "string" && value))]
    : [];
  const lodIds = Array.isArray(input.lodIds)
    ? [...new Set(input.lodIds.filter((value) => typeof value === "string" && value))]
    : [];
  const atlasDensityPolicies = Array.isArray(input.atlasDensityPolicies)
    ? input.atlasDensityPolicies.filter((value) => value && typeof value === "object")
    : [];
  return {
    state,
    coloredCoveragePpm,
    texturedCoveragePpm,
    minimumIndependentViewCount,
    requiredCoveragePpm,
    requiredIndependentViewCount,
    deliveryAuthorized: deliveryGatePassed,
    physicalDepthAccepted: input.physicalDepthAccepted === true,
    acceptedDepthFrames: validCount(input.acceptedDepthFrames) ? input.acceptedDepthFrames : null,
    assistanceCoverage: Number.isFinite(input.assistanceCoverage)
      && input.assistanceCoverage >= 0 && input.assistanceCoverage <= 1
      ? Number(input.assistanceCoverage) : null,
    atlasDensityPolicies,
    lodIds,
    appearanceRevisionIds,
    buildingAppearanceRevisionId: typeof input.buildingAppearanceRevisionId === "string"
      ? input.buildingAppearanceRevisionId : "",
    blocker: [...new Set(blockers)].join(" / "),
  };
}

function coveragePpmText(value) {
  return Number.isInteger(value) ? `${(value / 10000).toFixed(4)}%` : "N/A";
}

function atlasDensityText(policies) {
  const values = policies.map((policy) => {
    const selected = Number(policy.selected_mm_per_texel);
    const minimum = Number(policy.minimum_mm_per_texel);
    const maximum = Number(policy.maximum_mm_per_texel);
    if (Number.isFinite(selected)) return `${selected.toFixed(2)} mm/texel`;
    if (Number.isFinite(minimum) && Number.isFinite(maximum)) return `${minimum}-${maximum} mm/texel`;
    return null;
  }).filter(Boolean);
  return [...new Set(values)].join(" / ") || "N/A";
}

function exactIdentityText(values) {
  return values.length > 0 ? values.join(" / ") : "N/A";
}

function renderAppearanceTelemetry(input = {}) {
  const telemetry = normalizeAppearanceTelemetry(input);
  appearanceTelemetryState = telemetry;
  metric.appearanceState.textContent = telemetry.state;
  metric.appearanceState.dataset.state = telemetry.state;
  metric.coloredCoverage.textContent = coveragePpmText(telemetry.coloredCoveragePpm);
  metric.texturedCoverage.textContent = coveragePpmText(telemetry.texturedCoveragePpm);
  metric.appearanceViews.textContent = telemetry.minimumIndependentViewCount === null
    ? "N/A" : String(telemetry.minimumIndependentViewCount);
  metric.appearanceThresholds.textContent = `${coveragePpmText(telemetry.requiredCoveragePpm)} / ${telemetry.requiredIndependentViewCount} VIEWS`;
  metric.appearanceDelivery.textContent = telemetry.deliveryAuthorized ? "AUTHORIZED" : "BLOCKED";
  metric.physicalDepthAcceptance.textContent = telemetry.physicalDepthAccepted
    ? `ACCEPTED${telemetry.acceptedDepthFrames === null ? "" : ` / ${telemetry.acceptedDepthFrames} FRAMES`}`
    : "NOT ACCEPTED";
  metric.assistanceCoverage.textContent = telemetry.assistanceCoverage === null
    ? "N/A" : `${(telemetry.assistanceCoverage * 100).toFixed(2)}%`;
  metric.appearanceDensity.textContent = atlasDensityText(telemetry.atlasDensityPolicies);
  metric.appearanceLod.textContent = exactIdentityText(telemetry.lodIds);
  metric.appearanceRevision.textContent = exactIdentityText(telemetry.appearanceRevisionIds);
  metric.buildingAppearanceRevision.textContent = telemetry.buildingAppearanceRevisionId || "N/A";
  metric.appearanceBlocker.textContent = telemetry.blocker || "NONE";
  return telemetry;
}

function setMeasuredAppearanceState(state, details = "", telemetry = {}) {
  measuredAppearanceState = state;
  measuredAppearanceDetails = details;
  metric.textureState.textContent = details ? `${state} / ${details}` : state;
  metric.textureState.dataset.ready = String(
    state === "VERTEX_COLOR_READY" || state === "TEXTURED_READY" || state === "DIRECT_RGB_VERTEX_COLOR",
  );
  const reportsBlocker = state === "TEXTURE_NOT_AUTHORIZED"
    || state === "TEXTURE_STALE" || state === "TEXTURE_FAILED" || state === "DIRECT_RGB_FAILED";
  renderAppearanceTelemetry({
    ...telemetry,
    state,
    blocker: telemetry.blocker || (reportsBlocker ? details : ""),
  });
  if (state === "TEXTURED_READY") renderMapImageryState("VALIDATED");
  else if (state === "VERTEX_COLOR_READY") renderMapImageryState("VERTEX_COLOR");
  else if (state === "DIRECT_RGB_VERTEX_COLOR") renderMapImageryState("VERTEX_COLOR");
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

const LOWER_SHA256 = /^[0-9a-f]{64}$/;

function validateStaticMeasuredSurfaceCompatibility(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)
      || manifest.schema_version !== 1
      || manifest.compatibility_protocol !== "go2-static-verified-derived-surface-v1"
      || manifest.available !== true || manifest.blocker !== null
      || manifest.source_mode !== "DERIVED_READ_ONLY" || manifest.read_only !== true
      || manifest.point_authority !== "LIDAR_MEASURED" || manifest.geometry_authority !== false
      || manifest.hidden_surface_policy !== "OBSERVED_POINTS_ONLY"
      || manifest.quality !== "PREVIEW_UNCALIBRATED"
      || manifest.volume_filled !== false || manifest.watertight !== false
      || !LOWER_SHA256.test(manifest.derived_cache_key || "")
      || !LOWER_SHA256.test(manifest.surface_sha256 || "")
      || !LOWER_SHA256.test(manifest.ply_sha256 || "")
      || !LOWER_SHA256.test(manifest.metadata_sha256 || "")
      || !Number.isSafeInteger(manifest.surface_bytes) || manifest.surface_bytes <= 0
      || !Number.isSafeInteger(manifest.vertex_count) || manifest.vertex_count <= 0
      || !Number.isSafeInteger(manifest.triangle_count) || manifest.triangle_count <= 0
      || typeof manifest.session_id !== "string" || !/^segment-[0-9]{8}T[0-9]{6}Z$/.test(manifest.session_id)
      || typeof manifest.sensor_frame_transform_id !== "string" || !manifest.sensor_frame_transform_id
      || manifest.state !== "FINAL_SURFACE_AVAILABLE" || manifest.final_manifest_state !== "VERIFIED") {
    throw new Error("static measured-surface compatibility manifest is invalid");
  }
  const expectedSurfaceUrl = "/static/compatibility/measured-surfaces/"
    + `${manifest.derived_cache_key}/${manifest.surface_sha256}.json`;
  if (manifest.surface_relative_url !== expectedSurfaceUrl) {
    throw new Error("static measured-surface compatibility URL is invalid");
  }
  return manifest;
}

function validateStaticMeasuredSurfacePayload(surface, manifest) {
  if (!surface || typeof surface !== "object" || Array.isArray(surface)
      || surface.source_layer !== "LIDAR_MEASURED" || surface.point_authority !== "LIDAR_MEASURED"
      || surface.geometry_authority !== false || surface.quality !== "PREVIEW_UNCALIBRATED"
      || surface.status !== "PREVIEW_UNCALIBRATED"
      || surface.volume_filled !== false || surface.watertight !== false
      || surface.sensor_frame_transform_id !== manifest.sensor_frame_transform_id
      || !Array.isArray(surface.positions) || surface.positions.length !== manifest.vertex_count * 3
      || !Array.isArray(surface.normals) || surface.normals.length !== manifest.vertex_count * 3
      || !Array.isArray(surface.triangles) || surface.triangles.length !== manifest.triangle_count * 3
      || !Array.isArray(surface.source_vertex_ids) || surface.source_vertex_ids.length !== manifest.vertex_count) {
    throw new Error("static measured-surface compatibility payload is invalid");
  }
  return surface;
}

async function loadStaticMeasuredSurfaceCompatibility(currentSurfaceState = null) {
  const manifestResponse = await fetch(STATIC_MEASURED_SURFACE_COMPATIBILITY_MANIFEST, { cache: "no-store" });
  if (!manifestResponse.ok) throw new Error(`STATIC_COMPATIBILITY MANIFEST HTTP ${manifestResponse.status}`);
  const manifest = validateStaticMeasuredSurfaceCompatibility(await manifestResponse.json());
  if (currentSurfaceState?.derivedCacheKey === manifest.derived_cache_key
      && currentSurfaceState?.surfaceSha256 === manifest.surface_sha256) {
    return { manifest, surface: null, status: null };
  }
  const surfaceResponse = await fetch(manifest.surface_relative_url, { cache: "no-store" });
  if (!surfaceResponse.ok) throw new Error(`STATIC_COMPATIBILITY SURFACE HTTP ${surfaceResponse.status}`);
  const surfaceBuffer = await surfaceResponse.arrayBuffer();
  if (surfaceBuffer.byteLength !== manifest.surface_bytes
      || await sha256Hex(surfaceBuffer) !== manifest.surface_sha256) {
    throw new Error("STATIC_COMPATIBILITY surface checksum mismatch");
  }
  let surface;
  try {
    surface = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(surfaceBuffer));
  } catch (_error) {
    throw new Error("STATIC_COMPATIBILITY surface JSON is invalid");
  }
  return {
    manifest,
    surface: validateStaticMeasuredSurfacePayload(surface, manifest),
    status: {
      schema_version: 1,
      available: true,
      blocker: null,
      source_mode: "DERIVED_READ_ONLY",
      read_only: true,
      quality: manifest.quality,
      geometry_authority: false,
      sensor_frame_transform_id: manifest.sensor_frame_transform_id,
      derived_cache_key: manifest.derived_cache_key,
      surface_sha256: manifest.surface_sha256,
      ply_sha256: manifest.ply_sha256,
      metadata_sha256: manifest.metadata_sha256,
      vertex_count: manifest.vertex_count,
      triangle_count: manifest.triangle_count,
      compatibility_mode: "STATIC_COMPATIBILITY",
    },
  };
}

async function verifyBuildingAppearanceManifest(
  manifest, expectedRevisionId = null, skipCanonicalIdentity = false,
) {
  if (!manifest || manifest.schema_version !== 1
      || !LOWER_SHA256.test(manifest.building_appearance_revision_id || "")
      || !LOWER_SHA256.test(manifest.assembly_id || "")
      || manifest.geometry_authority !== "OBSERVED_SURFACE"
      || manifest.volume_filled !== false || manifest.watertight !== false
      || !Array.isArray(manifest.members) || manifest.members.length === 0) {
    throw new Error("Building appearance manifest is invalid");
  }
  if (!skipCanonicalIdentity) {
    const core = { ...manifest };
    delete core.building_appearance_revision_id;
    const digest = await sha256Hex(new TextEncoder().encode(canonicalJson(core)).buffer);
    if (digest !== manifest.building_appearance_revision_id) {
      throw new Error("Building appearance manifest identity mismatch");
    }
  }
  if (expectedRevisionId != null && manifest.building_appearance_revision_id !== expectedRevisionId) {
    throw new Error("Building appearance manifest revision does not match its status");
  }
  const tiles = new Set();
  for (const member of manifest.members) {
    const identityFields = [
      member.source_generation_id, member.placement_revision_id, member.tile_revision_id,
      member.tile_appearance_revision_id,
    ];
    const authorization = member.delivery_authorization;
    if (member.building_id !== manifest.building_id
        || identityFields.some((value) => !LOWER_SHA256.test(value || ""))
        || tiles.has(member.tile_revision_id)
        || !Array.isArray(member.authorized_lod_ids) || member.authorized_lod_ids.length === 0
        || member.authorized_lod_ids.some((value) => !LOWER_SHA256.test(value))
        || new Set(member.authorized_lod_ids).size !== member.authorized_lod_ids.length
        || !Array.isArray(member.distance_policy)
        || member.distance_policy.length !== member.authorized_lod_ids.length
        || !Array.isArray(member.transform_building_from_submap)
        || member.transform_building_from_submap.length !== 16
        || !member.transform_building_from_submap.every((value) => Number.isFinite(Number(value)))
        || authorization?.delivery_authorized !== true
        || !Number.isInteger(authorization.measured_surface_coverage_ppm)
        || !Number.isInteger(authorization.minimum_independent_view_count)
        || authorization.required_coverage_ppm !== 800000
        || authorization.required_independent_view_count !== 2
        || authorization.measured_surface_coverage_ppm < 800000
        || authorization.measured_surface_coverage_ppm > 1000000
        || authorization.minimum_independent_view_count < 2
        || !Array.isArray(authorization.blockers) || authorization.blockers.length !== 0
        || authorization.blockers.some((value) => typeof value !== "string" || !value)) {
      throw new Error("Building appearance member is invalid or unauthorized");
    }
    chooseAuthorizedLod(0, member);
    tiles.add(member.tile_revision_id);
  }
  return manifest;
}

async function verifyTileAppearanceManifest(
  manifest, member, skipCanonicalIdentity = false,
) {
  const memberAugmentationRevisionId = member.rgbd_augmented_revision_id ?? null;
  if (!manifest || manifest.schema_version !== 1
      || manifest.appearance_revision_id !== member.tile_appearance_revision_id
      || manifest.revision_core?.tile_revision_id !== member.tile_revision_id
      || manifest.revision_core?.source_generation_id !== member.source_generation_id
      || (manifest.revision_core?.rgbd_augmented_revision_id ?? null)
        !== memberAugmentationRevisionId
      || canonicalJson(manifest.delivery_authorization) !== canonicalJson(member.delivery_authorization)
      || manifest.publication_authorization?.building_authorized !== true
      || manifest.publication_authorization?.building_delivery_eligible !== true
      || !Array.isArray(manifest.lods) || manifest.lods.length === 0) {
    throw new Error("tile appearance manifest provenance is invalid");
  }
  if (!skipCanonicalIdentity) {
    const appearanceDigest = await sha256Hex(
      new TextEncoder().encode(canonicalJson(manifest.revision_core)).buffer,
    );
    if (appearanceDigest !== manifest.appearance_revision_id) {
      throw new Error("tile appearance revision identity mismatch");
    }
  }
  const lodIds = new Set();
  for (const lod of manifest.lods) {
    const artifacts = lod?.core?.artifacts;
    const expectedArtifactNames = new Set([
      "render-vertices.bin", "render-indices.bin", "render-uvs.bin",
      ...Array.from(
        { length: Number.isInteger(lod?.core?.atlas_count) ? lod.core.atlas_count : 0 },
        (_, index) => `atlas-${index}.png`,
      ),
    ]);
    const artifactNames = Array.isArray(artifacts)
      ? artifacts.map((artifact) => artifact?.name)
      : [];
    if (!lod || !LOWER_SHA256.test(lod.lod_id || "") || !lod.core
        || lod.core.tile_revision_id !== member.tile_revision_id
        || lod.core.appearance_revision_id !== member.tile_appearance_revision_id
        || (lod.core.rgbd_augmented_revision_id ?? null) !== memberAugmentationRevisionId
        || !Number.isInteger(lod.core.atlas_count) || lod.core.atlas_count < 1
        || !Array.isArray(artifacts)
        || artifacts.some((artifact) => !artifact
          || typeof artifact.name !== "string"
          || !LOWER_SHA256.test(artifact.sha256 || "")
          || !Number.isInteger(artifact.bytes) || artifact.bytes <= 0)
        || new Set(artifactNames).size !== artifactNames.length
        || artifactNames.length !== expectedArtifactNames.size
        || artifactNames.some((name) => !expectedArtifactNames.has(name))
        || !member.authorized_lod_ids.includes(lod.lod_id)) {
      throw new Error("tile appearance LOD provenance is invalid");
    }
    const lodDigest = skipCanonicalIdentity
      ? lod.lod_id
      : await sha256Hex(new TextEncoder().encode(canonicalJson(lod.core)).buffer);
    if (lodDigest !== lod.lod_id || lodIds.has(lod.lod_id)) {
      throw new Error("tile appearance LOD identity mismatch");
    }
    lodIds.add(lod.lod_id);
  }
  if (lodIds.size !== member.authorized_lod_ids.length
      || member.authorized_lod_ids.some((lodId) => !lodIds.has(lodId))) {
    throw new Error("tile appearance authorized LOD set is incomplete");
  }
  return manifest;
}

function parseAtlasHeaderV1(buffer, magic, recordBytes, tileRevisionId, appearanceRevisionId) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 80) {
    throw new Error("atlas binary header is truncated");
  }
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const actualMagic = new TextDecoder("ascii", { fatal: true }).decode(bytes.subarray(0, 8));
  const count = view.getUint32(12, true);
  if (actualMagic !== magic || view.getUint16(8, true) !== 1
      || view.getUint16(10, true) !== 80
      || buffer.byteLength !== 80 + count * recordBytes
      || bytesToHex(bytes.subarray(16, 48)) !== tileRevisionId
      || bytesToHex(bytes.subarray(48, 80)) !== appearanceRevisionId) {
    throw new Error("atlas binary header or identity is invalid");
  }
  return { bytes, view, count };
}

function parseRenderVerticesV1(buffer, tileRevisionId, appearanceRevisionId, sourceVertexCount = null) {
  const header = parseAtlasHeaderV1(
    buffer, "G2RVERT1", 28, tileRevisionId, appearanceRevisionId,
  );
  const positions = new Float32Array(header.count * 3);
  const normals = new Float32Array(header.count * 3);
  const sourceVertexIds = new Uint32Array(header.count);
  for (let index = 0; index < header.count; index += 1) {
    const source = 80 + index * 28;
    const target = index * 3;
    for (let axis = 0; axis < 3; axis += 1) {
      positions[target + axis] = header.view.getFloat32(source + axis * 4, true);
      normals[target + axis] = header.view.getFloat32(source + 12 + axis * 4, true);
    }
    const length = Math.hypot(normals[target], normals[target + 1], normals[target + 2]);
    const sourceVertexId = header.view.getUint32(source + 24, true);
    if (![positions[target], positions[target + 1], positions[target + 2], length].every(Number.isFinite)
        || Math.abs(length - 1) > 1e-3
        || (sourceVertexCount != null && sourceVertexId >= sourceVertexCount)) {
      throw new Error("render vertex record is invalid");
    }
    sourceVertexIds[index] = sourceVertexId;
  }
  return { count: header.count, positions, normals, sourceVertexIds };
}

function parseRenderIndicesV1(
  buffer, tileRevisionId, appearanceRevisionId, renderVertexCount, sourceTriangleCount = null,
) {
  const header = parseAtlasHeaderV1(
    buffer, "G2RIDX01", 16, tileRevisionId, appearanceRevisionId,
  );
  const triangles = new Uint32Array(header.count * 3);
  const sourceTriangleIds = new Uint32Array(header.count);
  for (let index = 0; index < header.count; index += 1) {
    const source = 80 + index * 16;
    for (let axis = 0; axis < 3; axis += 1) {
      const vertex = header.view.getUint32(source + axis * 4, true);
      if (vertex >= renderVertexCount) throw new Error("render index is out of range");
      triangles[index * 3 + axis] = vertex;
    }
    const sourceTriangleId = header.view.getUint32(source + 12, true);
    if (sourceTriangleCount != null && sourceTriangleId >= sourceTriangleCount) {
      throw new Error("source triangle identity is out of range");
    }
    sourceTriangleIds[index] = sourceTriangleId;
  }
  return { count: header.count, triangles, sourceTriangleIds };
}

function parseRenderUvsV1(buffer, tileRevisionId, appearanceRevisionId, atlasCount) {
  const header = parseAtlasHeaderV1(
    buffer, "G2RUV001", 12, tileRevisionId, appearanceRevisionId,
  );
  const uvs = new Float32Array(header.count * 2);
  const atlasIndices = new Uint16Array(header.count);
  const flags = new Uint16Array(header.count);
  for (let index = 0; index < header.count; index += 1) {
    const source = 80 + index * 12;
    const u = header.view.getFloat32(source, true);
    const v = header.view.getFloat32(source + 4, true);
    const atlas = header.view.getUint16(source + 8, true);
    const flag = header.view.getUint16(source + 10, true);
    const textured = flag === 1;
    if (![u, v].every(Number.isFinite) || u < 0 || u > 1 || v < 0 || v > 1
        || (flag !== 0 && flag !== 1)
        || (textured && (atlas === 65535 || atlas >= atlasCount))
        || (!textured && atlas !== 65535)) {
      throw new Error("render UV record is invalid");
    }
    uvs[index * 2] = u;
    uvs[index * 2 + 1] = v;
    atlasIndices[index] = atlas;
    flags[index] = flag;
  }
  return { count: header.count, uvs, atlasIndices, flags };
}

function validateAppearancePrimitiveSet(lodCore, parsed) {
  const primitives = lodCore?.primitives;
  if (!Number.isInteger(lodCore?.atlas_count) || lodCore.atlas_count < 1
      || !Number.isInteger(lodCore?.triangle_count)
      || lodCore.triangle_count !== parsed.indices.count
      || parsed.vertices.count !== parsed.uvs.count
      || !Array.isArray(primitives) || primitives.length === 0) {
    throw new Error("appearance primitive set is invalid");
  }
  const groups = { measured: [], augmented: [] };
  let expectedOffset = 0;
  for (let index = 0; index < primitives.length; index += 1) {
    const primitive = primitives[index];
    const textured = primitive.material === "VALIDATED_UV_ATLAS";
    const base = primitive.material === "BASE_MATERIAL";
    if (primitive.primitive_id !== index || primitive.index_offset !== expectedOffset
        || !Number.isInteger(primitive.index_count) || primitive.index_count <= 0
        || primitive.index_count % 3 !== 0
        || (!textured && !base)
        || (textured && (!Number.isInteger(primitive.atlas_index)
          || primitive.atlas_index < 0 || primitive.atlas_index >= lodCore.atlas_count))
        || (base && primitive.atlas_index !== 65535)
        || !Array.isArray(primitive.source_triangle_ids)
        || primitive.source_triangle_ids.length !== primitive.index_count / 3
        || !["LIDAR_MEASURED", "RGBD_AUGMENTED"].includes(primitive.source_layer)) {
      throw new Error("appearance primitive range or material is invalid");
    }
    expectedOffset += primitive.index_count;
    const firstTriangle = primitive.index_offset / 3;
    const triangleCount = primitive.index_count / 3;
    for (let triangleOffset = 0; triangleOffset < triangleCount; triangleOffset += 1) {
      const triangleIndex = firstTriangle + triangleOffset;
      if (primitive.source_triangle_ids[triangleOffset]
          !== parsed.indices.sourceTriangleIds[triangleIndex]) {
        throw new Error("appearance primitive source triangle identity is invalid");
      }
      for (let corner = 0; corner < 3; corner += 1) {
        const vertex = parsed.indices.triangles[triangleIndex * 3 + corner];
        const expectedFlag = textured ? 1 : 0;
        const expectedAtlas = textured ? primitive.atlas_index : 65535;
        if (parsed.uvs.flags[vertex] !== expectedFlag
            || parsed.uvs.atlasIndices[vertex] !== expectedAtlas) {
          throw new Error("appearance primitive UV material identity is invalid");
        }
      }
    }
    groups[primitive.source_layer === "RGBD_AUGMENTED" ? "augmented" : "measured"].push(primitive);
  }
  if (expectedOffset !== lodCore.triangle_count * 3
      || (groups.augmented.length > 0) !== Boolean(lodCore.rgbd_augmented_revision_id)) {
    throw new Error("appearance primitive coverage or augmentation is invalid");
  }
  return groups;
}

function chooseAuthorizedLod(distanceM, member) {
  const distance = Number(distanceM);
  if (!Number.isFinite(distance) || distance < 0
      || !Array.isArray(member?.authorized_lod_ids)
      || !Array.isArray(member?.distance_policy)
      || member.authorized_lod_ids.length === 0
      || member.authorized_lod_ids.length !== member.distance_policy.length) {
    throw new Error("Building appearance LOD policy is invalid");
  }
  let previousMaximum = 0;
  for (let index = 0; index < member.distance_policy.length; index += 1) {
    const interval = member.distance_policy[index];
    const minimum = Number(interval.minimum_distance_m);
    const maximum = interval.maximum_distance_m == null ? null : Number(interval.maximum_distance_m);
    if (!LOWER_SHA256.test(interval.lod_id || "")
        || interval.lod_id !== member.authorized_lod_ids[index]
        || minimum !== previousMaximum || minimum < 0
        || (maximum != null && (!Number.isFinite(maximum) || maximum <= minimum))
        || (index < member.distance_policy.length - 1 && maximum == null)
        || (index === member.distance_policy.length - 1 && maximum != null)) {
      throw new Error("Building appearance LOD intervals are invalid");
    }
    if (distance >= minimum && (maximum == null || distance < maximum)) return interval.lod_id;
    previousMaximum = maximum;
  }
  throw new Error("camera distance is outside the authorized LOD policy");
}

function authorizedLodFallbacks(member, requestedLodId) {
  const index = member.authorized_lod_ids.indexOf(requestedLodId);
  if (index < 0) throw new Error("requested LOD is unauthorized");
  return member.authorized_lod_ids.slice(index);
}

function updateLodDebounce(state, requestedLodId, nowMs, dwellMs) {
  if (requestedLodId === state.currentLodId) {
    return { commit: false, state: { ...state, pendingLodId: null, pendingSinceMs: 0 } };
  }
  if (state.pendingLodId !== requestedLodId) {
    return {
      commit: false,
      state: { ...state, pendingLodId: requestedLodId, pendingSinceMs: Number(nowMs) },
    };
  }
  if (Number(nowMs) - state.pendingSinceMs < Number(dwellMs)) {
    return { commit: false, state: { ...state } };
  }
  return {
    commit: true,
    lodId: requestedLodId,
    state: { currentLodId: requestedLodId, pendingLodId: null, pendingSinceMs: 0 },
  };
}

async function fetchVerifiedAppearanceArtifact(url, reference) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`appearance artifact ${reference.name} returned HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== reference.bytes || await sha256Hex(buffer) !== reference.sha256) {
    throw new Error(`appearance artifact ${reference.name} checksum mismatch`);
  }
  return buffer;
}

async function parseVerifiedManifestResponse(response, label) {
  const rawEtag = response.headers.get("etag") || "";
  const etag = rawEtag.replace(/^W\//, "").replace(/^"|"$/g, "");
  if (!LOWER_SHA256.test(etag)) throw new Error(label + " response ETag is invalid");
  const body = await response.arrayBuffer();
  const digest = await sha256Hex(body);
  if (digest !== etag) throw new Error(label + " manifest response checksum mismatch");
  try {
    return { manifest: JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(body)), digest };
  } catch (_error) {
    throw new Error(label + " response is not valid JSON");
  }
}

async function decodeAuthorizedAtlas(buffer) {
  if (typeof createImageBitmap !== "function") {
    throw new Error("appearance atlas decoding is unavailable");
  }
  const image = await createImageBitmap(new Blob([buffer], { type: "image/png" }));
  if (image.width <= 0 || image.height <= 0 || image.width > 4096 || image.height > 4096) {
    image.close?.();
    throw new Error("appearance atlas dimensions are invalid");
  }
  const bitmapCanvas = document.createElement("canvas");
  bitmapCanvas.width = image.width;
  bitmapCanvas.height = image.height;
  const context = bitmapCanvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    image.close?.();
    throw new Error("appearance atlas pixel verification is unavailable");
  }
  context.drawImage(image, 0, 0);
  const rgba = context.getImageData(0, 0, image.width, image.height).data;
  for (let offset = 3; offset < rgba.length; offset += 4) {
    if (rgba[offset] !== 0 && rgba[offset] !== 255) {
      image.close?.();
      throw new Error("appearance atlas alpha is not binary");
    }
  }
  const texture = new THREE.CanvasTexture(bitmapCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.flipY = false;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  image.close?.();
  return texture;
}

function buildAuthorizedAtlasMaterial(texture) {
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: texture,
    roughness: 0.78,
    metalness: 0.03,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  material.clippingPlanes = [currentFloorCutaway];
  material.userData = { authorizedAtlasMaterial: true };
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <map_fragment>",
      "#ifdef USE_MAP\n  vec4 sampledDiffuseColor = texture2D(map, vMapUv);\n  if (sampledDiffuseColor.a < 0.5) discard;\n  diffuseColor.rgb = sampledDiffuseColor.rgb;\n#endif",
    );
  };
  material.customProgramCacheKey = () => "go2-authorized-binary-alpha-atlas-v2";
  return material;
}

function buildAppearanceGeometry(parsed, member) {
  const positions = new Float32Array(parsed.vertices.count * 3);
  const normals = new Float32Array(parsed.vertices.count * 3);
  for (let index = 0; index < parsed.vertices.count; index += 1) {
    const offset = index * 3;
    const buildingPosition = applyAssemblyTransform([
      parsed.vertices.positions[offset], parsed.vertices.positions[offset + 1],
      parsed.vertices.positions[offset + 2],
    ], member);
    const buildingNormal = applyAssemblyTransform([
      parsed.vertices.normals[offset], parsed.vertices.normals[offset + 1],
      parsed.vertices.normals[offset + 2],
    ], member, true);
    positions.set(positionToScene(buildingPosition.toArray()).toArray(), offset);
    normals.set(positionToScene(buildingNormal.toArray()).normalize().toArray(), offset);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(parsed.uvs.uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(parsed.indices.triangles, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function buildAppearanceTileObject(member, lodCore, parsed, textures) {
  const groups = validateAppearancePrimitiveSet(lodCore, parsed);
  const root = new THREE.Group();
  root.name = `BUILDING_APPEARANCE/${member.tile_revision_id}`;
  root.userData = { member, lodId: lodCore.lod_id || null };
  const layers = [
    ["LIDAR_MEASURED", groups.measured],
    [`RGBD_AUGMENTED/${member.tile_revision_id}`, groups.augmented],
  ];
  for (const [name, primitives] of layers) {
    if (primitives.length === 0) continue;
    const layer = new THREE.Group();
    layer.name = name;
    layer.userData = { source_layer: name.startsWith("RGBD_AUGMENTED") ? "RGBD_AUGMENTED" : "LIDAR_MEASURED" };
    for (const primitive of primitives) {
      if (primitive.material === "BASE_MATERIAL" && primitive.source_layer === "LIDAR_MEASURED") {
        continue;
      }
      const geometry = buildAppearanceGeometry(parsed, member);
      geometry.setDrawRange(primitive.index_offset, primitive.index_count);
      const material = primitive.material === "VALIDATED_UV_ATLAS"
        ? buildAuthorizedAtlasMaterial(textures[primitive.atlas_index].clone())
        : surfaceMaterial;
      if (material.map) material.map.needsUpdate = true;
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `${primitive.source_layer}/${primitive.primitive_id}`;
      mesh.userData = { ...primitive, tile_revision_id: member.tile_revision_id };
      layer.add(mesh);
    }
    if (layer.children.length > 0) root.add(layer);
  }
  return root;
}

function disposeAppearanceObject(root) {
  if (!root) return;
  const disposedTextures = new Set();
  const disposedMaterials = new Set();
  const disposedGeometries = new Set();
  root.traverse((object) => {
    if (object.geometry && object.geometry !== undefined && !disposedGeometries.has(object.geometry)) {
      object.geometry.dispose();
      disposedGeometries.add(object.geometry);
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material || material.userData?.sharedSurfaceMaterial || disposedMaterials.has(material)) continue;
      if (material.map && !disposedTextures.has(material.map)) {
        material.map.dispose();
        disposedTextures.add(material.map);
      }
      material.dispose();
      disposedMaterials.add(material);
    }
  });
}

function commitBuildingAppearanceCandidate(
  parent, previous, candidate, requestIdentity, activeRequestIdentity,
) {
  if (requestIdentity !== activeRequestIdentity) {
    disposeAppearanceObject(candidate);
    return { committed: false, current: previous };
  }
  if (previous) parent.remove(previous);
  parent.add(candidate);
  disposeAppearanceObject(previous);
  return { committed: true, current: candidate };
}

function appearanceArtifactUrl(member, lodId, artifactName) {
  return `/api/building/appearance/tiles/${member.tile_revision_id}/${member.tile_appearance_revision_id}/${lodId}/${artifactName}`;
}

async function loadAppearanceTileLod(member, tileManifest, lodId) {
  const lod = tileManifest.lods.find((value) => value.lod_id === lodId);
  if (!lod || !member.authorized_lod_ids.includes(lodId)) {
    throw new Error("requested appearance LOD is unavailable or unauthorized");
  }
  const references = new Map(lod.core.artifacts.map((value) => [value.name, value]));
  const required = ["render-vertices.bin", "render-indices.bin", "render-uvs.bin"];
  const atlasNames = Array.from({ length: lod.core.atlas_count }, (_, index) => `atlas-${index}.png`);
  if ([...required, ...atlasNames].some((name) => !references.has(name))
      || [...references].filter(([name]) => name.startsWith("atlas-")).length !== atlasNames.length) {
    throw new Error("appearance LOD artifact set is incomplete");
  }
  const buffers = new Map(await Promise.all(
    [...required, ...atlasNames].map(async (name) => [
      name,
      await fetchVerifiedAppearanceArtifact(
        appearanceArtifactUrl(member, lodId, name), references.get(name),
      ),
    ]),
  ));
  const parsed = {
    vertices: parseRenderVerticesV1(
      buffers.get("render-vertices.bin"), member.tile_revision_id,
      member.tile_appearance_revision_id,
    ),
    indices: null,
    uvs: null,
  };
  parsed.indices = parseRenderIndicesV1(
    buffers.get("render-indices.bin"), member.tile_revision_id,
    member.tile_appearance_revision_id, parsed.vertices.count,
  );
  parsed.uvs = parseRenderUvsV1(
    buffers.get("render-uvs.bin"), member.tile_revision_id,
    member.tile_appearance_revision_id, lod.core.atlas_count,
  );
  if (parsed.vertices.count !== lod.core.vertex_count
      || parsed.indices.count !== lod.core.triangle_count
      || parsed.uvs.count !== lod.core.vertex_count) {
    throw new Error("appearance LOD binary counts do not match manifest");
  }
  const textures = [];
  try {
    for (const name of atlasNames) textures.push(await decodeAuthorizedAtlas(buffers.get(name)));
    const object = buildAppearanceTileObject(member, { ...lod.core, lod_id: lodId }, parsed, textures);
    object.userData = { member, tileManifest, lodId, lodState: { currentLodId: lodId, pendingLodId: null, pendingSinceMs: 0 } };
    return object;
  } finally {
    for (const texture of textures) texture.dispose();
  }
}

async function loadBuildingAppearanceCandidate(manifest, requestIdentity) {
  const candidate = new THREE.Group();
  candidate.name = `BUILDING_APPEARANCE/${manifest.building_appearance_revision_id}`;
  candidate.userData = { manifest, requestIdentity };
  try {
    for (const member of manifest.members) {
      const tileResponse = await fetch(
        `/api/building/appearance/tiles/${member.tile_revision_id}/${member.tile_appearance_revision_id}/manifest`,
        { cache: "force-cache" },
      );
      if (!tileResponse.ok) throw new Error(`tile appearance manifest returned HTTP ${tileResponse.status}`);
      const parsedTileManifest = await parseVerifiedManifestResponse(tileResponse, "tile appearance");
      const tileManifest = await verifyTileAppearanceManifest(
        parsedTileManifest.manifest, member, true,
      );
      const baseTile = tileMeshes.get(member.placement_revision_id)?.mesh;
      const distance = baseTile
        ? camera.position.distanceTo(baseTile.geometry.boundingBox?.clampPoint(camera.position, new THREE.Vector3()) || baseTile.position)
        : camera.position.length();
      const requested = chooseAuthorizedLod(distance, member);
      let tileObject = null;
      let lastError = null;
      for (const lodId of authorizedLodFallbacks(member, requested)) {
        try {
          tileObject = await loadAppearanceTileLod(member, tileManifest, lodId);
          break;
        } catch (error) {
          lastError = error;
        }
      }
      if (!tileObject) {
        if (!baseTile) throw lastError || new Error("appearance tile and base geometry are unavailable");
        tileObject = baseTile.clone();
        tileObject.geometry = baseTile.geometry.clone();
        tileObject.material = surfaceMaterial;
        tileObject.userData = { member, tileManifest, lodId: null, baseFallback: true };
      }
      tileObject.userData.floor_id = baseTile?.userData?.floor_id || null;
      candidate.add(tileObject);
    }
    if (candidate.children.length !== manifest.members.length) {
      throw new Error("Building appearance candidate member set is incomplete");
    }
    return candidate;
  } catch (error) {
    disposeAppearanceObject(candidate);
    throw error;
  }
}

function buildingAppearanceCandidateState(candidate, requestedState = null) {
  const children = Array.isArray(candidate?.children) ? candidate.children : [];
  const manifest = candidate?.userData?.manifest || {};
  const members = Array.isArray(manifest.members) ? manifest.members : [];
  const authorizations = members.map((member) => member.delivery_authorization || {});
  const fallbackCount = children.filter((child) => child?.userData?.baseFallback === true).length;
  if (members.length === 0) {
    return {
      state: requestedState || (
        children.length > 0 && fallbackCount === 0 ? "TEXTURED_READY" : "TEXTURE_FAILED"
      ),
      details: fallbackCount > 0
        ? `${fallbackCount} BASE FALLBACK / ${children.length} TILES`
        : `${children.length} AUTHORIZED TILES`,
    };
  }
  const requested = requestedState || (
    children.length > 0 && fallbackCount === 0 ? "TEXTURED_READY" : "TEXTURE_FAILED"
  );
  const coverageValues = authorizations
    .map((authorization) => authorization.measured_surface_coverage_ppm)
    .filter(Number.isInteger);
  const viewValues = authorizations
    .map((authorization) => authorization.minimum_independent_view_count)
    .filter(Number.isInteger);
  const requiredCoverageValues = authorizations.map((authorization) => authorization.required_coverage_ppm);
  const requiredViewValues = authorizations.map((authorization) => authorization.required_independent_view_count);
  const tileManifests = children.map((child) => child?.userData?.tileManifest).filter(Boolean);
  const telemetry = normalizeAppearanceTelemetry({
    state: requested,
    coloredCoveragePpm: coverageValues.length === members.length ? Math.min(...coverageValues) : null,
    texturedCoveragePpm: coverageValues.length === members.length ? Math.min(...coverageValues) : null,
    minimumIndependentViewCount: viewValues.length === members.length ? Math.min(...viewValues) : null,
    requiredCoveragePpm: requiredCoverageValues.length > 0
      && requiredCoverageValues.every((value) => value === requiredCoverageValues[0])
      ? requiredCoverageValues[0] : null,
    requiredIndependentViewCount: requiredViewValues.length > 0
      && requiredViewValues.every((value) => value === requiredViewValues[0])
      ? requiredViewValues[0] : null,
    deliveryAuthorized: authorizations.length === members.length && members.length > 0
      && authorizations.every((authorization) => authorization.delivery_authorized === true),
    physicalDepthAccepted: tileManifests.length > 0 && tileManifests.every(
      (tileManifest) => LOWER_SHA256.test(tileManifest.revision_core?.physical_depth_validation_revision || ""),
    ),
    assistanceCoverage: latestSnapshot?.depth_anything?.assistance_coverage,
    atlasDensityPolicies: tileManifests.map(
      (tileManifest) => tileManifest.revision_core?.atlas_density_policy,
    ).filter(Boolean),
    lodIds: children.length > 0
      ? children.map((child) => child?.userData?.lodId || "BASE_GEOMETRY")
      : members.flatMap((member) => member.authorized_lod_ids || []),
    appearanceRevisionIds: members.map((member) => member.tile_appearance_revision_id),
    buildingAppearanceRevisionId: manifest.building_appearance_revision_id,
    blocker: fallbackCount > 0 ? `BASE_GEOMETRY_FALLBACK:${fallbackCount}` : "",
  });
  return {
    ...telemetry,
    details: requested === "ATLAS_BUILDING"
      ? `${members.length} TILES`
      : fallbackCount > 0
        ? `${fallbackCount} BASE FALLBACK / ${children.length} TILES`
        : `${children.length} AUTHORIZED TILES`,
  };
}

async function refreshBuildingAppearance() {
  if (buildingAppearanceInFlight || Date.now() < buildingAppearanceRetryAfterMs
      || document.visibilityState === "hidden" || sessionOdomLocalPreview || surfaceViewMode !== "building"
      || measuredSurfaceState?.readOnly) return;
  buildingAppearanceInFlight = true;
  const requestIdentity = `${Date.now()}:${Math.random()}`;
  buildingAppearanceRequestIdentity = requestIdentity;
  try {
    const statusResponse = await fetch("/api/building/appearance/status", { cache: "no-store" });
    if (!statusResponse.ok) throw new Error(`BUILDING APPEARANCE STATUS HTTP ${statusResponse.status}`);
    const status = await statusResponse.json();
    if (status.available !== true) {
      setMeasuredAppearanceState(
        buildingAppearanceObject ? "TEXTURE_STALE" : "TEXTURE_NOT_AUTHORIZED",
        status.blocker || "BUILDING APPEARANCE UNAVAILABLE",
      );
      buildingAppearanceRetryAfterMs = Date.now() + 2000;
      return;
    }
    const currentResponse = await fetch("/api/building/appearance/manifest", { cache: "no-store" });
    if (currentResponse.status === 404) {
      setMeasuredAppearanceState(
        buildingAppearanceObject ? "TEXTURE_STALE" : "TEXTURE_NOT_AUTHORIZED",
        "BUILDING APPEARANCE UNAVAILABLE",
      );
      buildingAppearanceRetryAfterMs = Date.now() + 2000;
      return;
    }
    if (!currentResponse.ok) throw new Error(`BUILDING APPEARANCE HTTP ${currentResponse.status}`);
    const currentManifest = await parseVerifiedManifestResponse(
      currentResponse, "Building appearance CURRENT",
    );
    const current = await verifyBuildingAppearanceManifest(
      currentManifest.manifest, status.building_appearance_revision_id,
      true,
    );
    if (current.building_appearance_revision_id === buildingAppearanceRevisionId
        && buildingAppearanceObject) return;
    const stagingState = buildingAppearanceCandidateState(
      { children: [], userData: { manifest: current } }, "ATLAS_BUILDING",
    );
    setMeasuredAppearanceState(stagingState.state, stagingState.details, stagingState);
    const exactResponse = await fetch(
      `/api/building/appearance/revisions/${current.building_appearance_revision_id}/manifest`,
      { cache: "force-cache" },
    );
    if (!exactResponse.ok) throw new Error(`EXACT BUILDING APPEARANCE HTTP ${exactResponse.status}`);
    const exactManifest = await parseVerifiedManifestResponse(
      exactResponse, "exact Building appearance",
    );
    const exact = await verifyBuildingAppearanceManifest(
      exactManifest.manifest, current.building_appearance_revision_id,
      true,
    );
    if (currentManifest.digest !== exactManifest.digest) {
      throw new Error("Building appearance CURRENT changed before exact staging");
    }
    const candidate = await loadBuildingAppearanceCandidate(exact, requestIdentity);
    if (surfaceViewMode !== "building") buildingAppearanceRequestIdentity = "";
    const result = commitBuildingAppearanceCandidate(
      surfaceGroup, buildingAppearanceObject, candidate, requestIdentity,
      buildingAppearanceRequestIdentity,
    );
    if (!result.committed) {
      setMeasuredAppearanceState("TEXTURE_STALE");
      return;
    }
    buildingAppearanceObject = result.current;
    buildingAppearanceRevisionId = exact.building_appearance_revision_id;
    buildingAppearanceRetryAfterMs = 0;
    surfaceFramePending = true;
    orbit.framed = false;
    const candidateState = buildingAppearanceCandidateState(buildingAppearanceObject);
    setMeasuredAppearanceState(candidateState.state, candidateState.details, candidateState);
    updateSurfaceVisibility();
  } catch (error) {
    if (requestIdentity === buildingAppearanceRequestIdentity) {
      setMeasuredAppearanceState(
        buildingAppearanceObject ? "TEXTURE_STALE" : "TEXTURE_FAILED", error.message,
      );
      buildingAppearanceRetryAfterMs = Date.now() + 3000;
    }
  } finally {
    buildingAppearanceInFlight = false;
  }
}

async function replaceAppearanceTileLod(tileObject, requestedLodId, requestIdentity) {
  const { member, tileManifest } = tileObject.userData;
  let replacement = null;
  for (const lodId of authorizedLodFallbacks(member, requestedLodId)) {
    try {
      replacement = await loadAppearanceTileLod(member, tileManifest, lodId);
      break;
    } catch (_error) {
      // Continue only toward a coarser LOD frozen in this Building revision.
    }
  }
  if (!replacement || requestIdentity !== buildingAppearanceRequestIdentity
      || tileObject.parent !== buildingAppearanceObject) {
    disposeAppearanceObject(replacement);
    return false;
  }
  replacement.userData.floor_id = tileObject.userData.floor_id;
  const index = buildingAppearanceObject.children.indexOf(tileObject);
  buildingAppearanceObject.remove(tileObject);
  buildingAppearanceObject.add(replacement);
  if (index >= 0) {
    buildingAppearanceObject.children.splice(
      buildingAppearanceObject.children.indexOf(replacement), 1,
    );
    buildingAppearanceObject.children.splice(index, 0, replacement);
  }
  disposeAppearanceObject(tileObject);
  const candidateState = buildingAppearanceCandidateState(buildingAppearanceObject);
  setMeasuredAppearanceState(candidateState.state, candidateState.details, candidateState);
  return true;
}

function refreshBuildingAppearanceLods(nowMs = performance.now()) {
  if (!buildingAppearanceObject || surfaceViewMode !== "building"
      || nowMs < buildingAppearanceLodRefreshAt) return;
  buildingAppearanceLodRefreshAt = nowMs + 150;
  for (const tileObject of [...buildingAppearanceObject.children]) {
    const member = tileObject.userData.member;
    if (!member || tileObject.userData.baseFallback || tileObject.userData.lodLoading) continue;
    const bounds = new THREE.Box3().setFromObject(tileObject);
    if (bounds.isEmpty()) continue;
    const requested = chooseAuthorizedLod(bounds.distanceToPoint(camera.position), member);
    const transition = updateLodDebounce(
      tileObject.userData.lodState, requested, nowMs, 300,
    );
    tileObject.userData.lodState = transition.state;
    if (!transition.commit) continue;
    tileObject.userData.lodLoading = true;
    replaceAppearanceTileLod(
      tileObject, transition.lodId, buildingAppearanceRequestIdentity,
    ).finally(() => { tileObject.userData.lodLoading = false; });
  }
}

async function verifyAppearanceManifest(manifest) {
  if (!manifest || typeof manifest.manifest_sha256 !== "string"
      || !/^[0-9a-f]{64}$/.test(manifest.manifest_sha256)
      || typeof manifest.manifest_canonical_json !== "string") {
    throw new Error("appearance manifest checksum is missing");
  }
  const encoded = new TextEncoder().encode(manifest.manifest_canonical_json);
  const digest = await sha256Hex(encoded.buffer);
  if (digest !== manifest.manifest_sha256) throw new Error("appearance manifest checksum mismatch");
  let verified;
  try {
    verified = JSON.parse(manifest.manifest_canonical_json);
  } catch (_error) {
    throw new Error("appearance canonical manifest is invalid");
  }
  if (!verified || typeof verified !== "object" || Array.isArray(verified)) {
    throw new Error("appearance canonical manifest is invalid");
  }
  return verified;
}

async function loadMeasuredAppearance(mesh, geometryRevisionId, requestIdentity) {
  if (!geometryRevisionId) {
    setMeasuredAppearanceState("TEXTURE_NOT_AUTHORIZED", "GEOMETRY REVISION MISSING");
    return false;
  }
  try {
    const manifestResponse = await fetch(
      "/api/reconstruction/measured-surface/appearance",
      { cache: "no-store" },
    );
    if (!manifestResponse.ok) throw new Error(`MANIFEST HTTP ${manifestResponse.status}`);
    const manifestEnvelope = await manifestResponse.json();
    if (requestIdentity !== measuredAppearanceRequestIdentity || measuredSurfaceMesh !== mesh) {
      setMeasuredAppearanceState("TEXTURE_STALE");
      return false;
    }
    if (manifestEnvelope.state !== "VERTEX_COLOR_READY") {
      setMeasuredAppearanceState(
        "TEXTURE_NOT_AUTHORIZED",
        Array.isArray(manifestEnvelope.blockers) ? manifestEnvelope.blockers.join(", ") : "APPEARANCE BLOCKED",
        { blockers: manifestEnvelope.blockers },
      );
      return false;
    }
    const manifest = await verifyAppearanceManifest(manifestEnvelope);
    if (manifest.measured_surface_geometry_revision_id !== geometryRevisionId
        || typeof manifest.appearance_revision_id !== "string"
        || !/^[0-9a-f]{64}$/.test(manifest.appearance_revision_id)) {
      throw new Error("appearance manifest identity is stale");
    }
    const binaryResponse = await fetch(manifest.vertex_color.url, { cache: "force-cache" });
    if (!binaryResponse.ok) throw new Error(`VERTEX COLOR HTTP ${binaryResponse.status}`);
    const buffer = await binaryResponse.arrayBuffer();
    const digest = await sha256Hex(buffer);
    if (digest !== manifest.vertex_color.content_sha256) {
      throw new Error("vertex-color checksum mismatch");
    }
    const parsed = parseVertexColorV1(
      buffer,
      geometryRevisionId,
      manifest.appearance_revision_id,
      mesh.geometry.getAttribute("position").count,
    );
    const candidate = buildMeasuredAppearanceCandidate(mesh, parsed, surfaceMaterial);
    if (!commitMeasuredAppearance(
      mesh, candidate, geometryRevisionId, requestIdentity, measuredAppearanceRequestIdentity, surfaceMaterial,
    )) {
      setMeasuredAppearanceState("TEXTURE_STALE");
      return false;
    }
    setMeasuredAppearanceState(
      "VERTEX_COLOR_READY",
      `${(Number(manifest.surface_coverage || 0) * 100).toFixed(1)}% COVERAGE`,
      {
        coloredCoveragePpm: Number.isInteger(manifest.revision_core?.coverage_ppm)
          ? manifest.revision_core.coverage_ppm
          : Math.round(Number(manifest.surface_coverage || 0) * 1000000),
        physicalDepthAccepted: LOWER_SHA256.test(
          manifest.physical_depth_validation_revision || "",
        ),
        appearanceRevisionIds: [manifest.appearance_revision_id],
        assistanceCoverage: latestSnapshot?.depth_anything?.assistance_coverage,
      },
    );
    return true;
  } catch (error) {
    if (requestIdentity !== measuredAppearanceRequestIdentity || measuredSurfaceMesh !== mesh) {
      setMeasuredAppearanceState("TEXTURE_STALE");
    } else {
      setMeasuredAppearanceState("TEXTURE_FAILED", error.message);
    }
    return false;
  }
}

async function loadRoomDirectRgbAppearance(mesh, geometryRevisionId, requestIdentity) {
  try {
    const response = await fetch("/api/reconstruction/session-odom-local/appearance", { cache: "no-store" });
    if (!response.ok) throw new Error(`ROOM RGB MANIFEST HTTP ${response.status}`);
    const manifest = await response.json();
    if (manifest.measured_surface_geometry_revision_id !== geometryRevisionId) throw new Error("ROOM RGB geometry revision mismatch");
    const binaryResponse = await fetch("/api/reconstruction/session-odom-local/appearance/vertex-color.bin", { cache: "no-store" });
    if (!binaryResponse.ok) throw new Error(`ROOM RGB COLOR HTTP ${binaryResponse.status}`);
    const buffer = await binaryResponse.arrayBuffer();
    const parsed = parseVertexColorV1(buffer, geometryRevisionId, manifest.preview_revision_id, mesh.geometry.getAttribute("position").count);
    const candidate = buildMeasuredAppearanceCandidate(mesh, parsed, surfaceMaterial, true);
    if (requestIdentity !== measuredAppearanceRequestIdentity) return false;
    commitMeasuredAppearance(mesh, candidate, geometryRevisionId, requestIdentity, measuredAppearanceRequestIdentity, surfaceMaterial);
    setMeasuredAppearanceState("DIRECT_RGB_VERTEX_COLOR", `${(Number(manifest.surface_coverage || 0) * 100).toFixed(2)}% COVERAGE`, { blockers: manifest.blockers || [] });
    return true;
  } catch (error) {
    setMeasuredAppearanceState("DIRECT_RGB_FAILED", error.message, { blockers: ["DIRECT_RGB_REPLAY_FAILED"] });
    return false;
  }
}

function setLegacySessionPresentation() {
  document.documentElement.dataset.legacySession = "true";
  cameraAssist.hidden = true;
  cameraStream.dataset.visible = "false";
  cameraStream.removeAttribute("src");
  surfaceControls.hidden = false;
  for (const control of surfaceControls.children) {
    control.hidden = control !== mapTextureStatus;
  }
  followRobotButton.hidden = true;
  trajectory.visible = false;
  robot.visible = false;
  renderMapImageryState("VERTEX_COLOR");
}

async function refreshMeasuredSurfacePreview() {
  if (measuredSurfaceInFlight || Date.now() < measuredSurfaceRetryAfterMs
      || document.visibilityState === "hidden") return;
  const retainedReadOnlySurface = measuredSurfaceMesh && measuredSurfaceState.readOnly;
  const dense = latestSnapshot?.dense_map || {};
  const snapshotArtifactCount = Number(dense.artifacts || 0);
  const snapshotPhase = dense.reconstruction_phase || "UNKNOWN";
  if (!retainedReadOnlySurface && measuredSurfaceMesh
      && snapshotArtifactCount === measuredSurfaceState.artifactCount
      && snapshotPhase === measuredSurfaceState.reconstructionPhase) return;
  measuredSurfaceInFlight = true;
  try {
    const statusResponse = sessionOdomLocalPreview
      ? await fetch("/api/reconstruction/session-odom-local/status", { cache: "no-store" })
      : await fetch("/api/reconstruction/measured-surface/status", { cache: "no-store" });
    if (!statusResponse.ok) {
      disposeMeasuredSurfacePreview();
      measuredSurfaceSourceStatus = null;
      measuredSurfaceRetryAfterMs = Date.now() + 1000;
      updateSurfaceVisibility();
      if (latestSnapshot) apply(latestSnapshot);
      return;
    }
    const officialStatus = await statusResponse.json();
    let status = officialStatus;
    let surfacePayload = null;
    if (!sessionOdomLocalPreview && officialStatus.available === false) {
      disposeMeasuredSurfacePreview();
      measuredSurfaceSourceStatus = officialStatus;
      measuredSurfaceRetryAfterMs = Date.now() + 1000;
      updateSurfaceVisibility();
      if (latestSnapshot) apply(latestSnapshot);
      return;
    } else if (officialStatus.available !== true) {
      disposeMeasuredSurfacePreview();
      measuredSurfaceSourceStatus = officialStatus;
      measuredSurfaceRetryAfterMs = Date.now() + 1000;
      updateSurfaceVisibility();
      if (latestSnapshot) apply(latestSnapshot);
      return;
    }
    measuredSurfaceSourceStatus = status;
    if (status.source_mode === "SESSION_ODOM_LOCAL_READ_ONLY" && !sessionOdomLocalPreview && measuredAppearanceState !== "DIRECT_RGB_VERTEX_COLOR") {
      setMeasuredAppearanceState(
        "TEXTURE_NOT_AUTHORIZED",
        "SESSION_ODOM_LOCAL / NO TEXTURE AUTHORITY",
        { blockers: ["SESSION_ODOM_LOCAL", "TEXTURE_NOT_AUTHORIZED"] },
      );
      cameraAssist.hidden = true;
      cameraStream.dataset.visible = "false";
      cameraStream.removeAttribute("src");
      trajectory.visible = false;
      robot.visible = false;
    }
    if (status.source_mode === "LEGACY_SESSION_READ_ONLY") setLegacySessionPresentation();
    if (status.source_mode === "DERIVED_READ_ONLY"
        && measuredSurfaceMesh
        && measuredSurfaceState?.derivedCacheKey === status.derived_cache_key
        && measuredSurfaceState?.surfaceSha256 === status.surface_sha256) {
      measuredSurfaceRetryAfterMs = Date.now() + 1000;
      return;
    }
    if (status.source_mode === "SESSION_ODOM_LOCAL_READ_ONLY"
        && measuredSurfaceMesh
        && measuredSurfaceState?.derivedCacheKey === status.derived_cache_key
        && measuredSurfaceState?.surfaceSha256 === status.surface_sha256) {
      measuredSurfaceRetryAfterMs = Date.now() + 1000;
      return;
    }
    if (surfacePayload === null) {
      const response = sessionOdomLocalPreview
        ? await fetch("/api/reconstruction/session-odom-local", { cache: "no-store" })
        : await fetch("/api/reconstruction/measured-surface", { cache: "no-store" });
      if (!response.ok) {
        measuredSurfaceRetryAfterMs = Date.now() + 1000;
        return;
      }
      surfacePayload = status.source_mode === "LEGACY_SESSION_READ_ONLY"
        ? await parseLegacyPlyInWorker(await response.arrayBuffer(), status)
        : await response.json();
    }
    const candidate = status.source_mode === "LEGACY_SESSION_READ_ONLY"
      ? buildLegacySurfacePreview(surfacePayload, status)
      : buildMeasuredSurfacePreview(surfacePayload);
    measuredSurfaceRetryAfterMs = Date.now() + 1000;
    const candidateSignature = [
      candidate.state.poseGeneration,
      candidate.state.reconstructionPhase,
      candidate.state.artifactCount,
      candidate.state.vertexCount,
      candidate.state.triangleCount,
      candidate.state.surfaceSha256 || "live",
    ].join(":");
    if (candidateSignature === measuredSurfaceSignature) {
      candidate.mesh.geometry.dispose();
      return;
    }
    surfaceGroup.add(candidate.mesh);
    disposeMeasuredSurfacePreview();
    measuredSurfaceMesh = candidate.mesh;
    measuredSurfaceState = candidate.state;
    measuredSurfaceSignature = candidateSignature;
    if (candidate.state.readOnly) {
      surfaceViewMode = sessionOdomLocalPreview ? "local" : "building";
      surfaceBuildingButton.setAttribute("aria-pressed", String(surfaceViewMode === "building"));
      surfaceLocalButton.setAttribute("aria-pressed", String(surfaceViewMode === "local"));
      followRobotEnabled = false;
      followRobotButton.setAttribute("aria-pressed", "false");
    }
    measuredAppearanceSurfaceActive = true;
    measuredAppearanceRequestIdentity = `${candidateSignature}:${candidate.state.geometryRevisionId || "blocked"}`;
    if (candidate.state.sourceMode === "LEGACY_SESSION_READ_ONLY") {
      setMeasuredAppearanceState(
        "TEXTURE_NOT_AUTHORIZED",
        "LEGACY_VERTEX_COLOR_UNCALIBRATED",
        { coloredCoveragePpm: 1000000, blockers: ["LEGACY_SESSION_READ_ONLY"] },
      );
    } else if (sessionOdomLocalPreview || candidate.state.sourceMode === "SESSION_ODOM_LOCAL_READ_ONLY") {
      loadRoomDirectRgbAppearance(candidate.mesh, candidate.state.geometryRevisionId, measuredAppearanceRequestIdentity);
    } else if (status.compatibility_mode === "STATIC_COMPATIBILITY") {
      setMeasuredAppearanceState(
        "TEXTURE_NOT_AUTHORIZED",
        "DERIVED_SURFACE_READ_ONLY",
        { blockers: ["DERIVED_SURFACE_READ_ONLY"] },
      );
    } else {
      loadMeasuredAppearance(
        candidate.mesh, candidate.state.geometryRevisionId, measuredAppearanceRequestIdentity,
      );
    }
    surfaceFramePending = true;
    orbit.framed = false;
    updateSurfaceVisibility();
    if (latestSnapshot) apply(latestSnapshot);
  } catch (_error) {
    measuredSurfaceRetryAfterMs = Date.now() + 1000;
  } finally {
    measuredSurfaceInFlight = false;
  }
}

async function loadProvisionalProjection(assignment, triangleCount) {
  if (!assignment || assignment.texture_state !== "PROVISIONAL"
      || assignment.final_store_eligible !== false) return null;
  try {
    const [colorResponse, maskResponse] = await Promise.all([
      fetch(assignment.color_url, { cache: "force-cache" }),
      fetch(assignment.visibility_mask_url, { cache: "force-cache" }),
    ]);
    if (!colorResponse.ok || !maskResponse.ok) throw new Error("provisional texture artifact is unavailable");
    const [verifiedColor, verifiedMask] = await Promise.all([
      verifyBytesInWorker(await colorResponse.arrayBuffer(), assignment.color_content_sha256),
      verifyBytesInWorker(await maskResponse.arrayBuffer(), assignment.visibility_mask_sha256),
    ]);
    const visibility = new Uint8Array(verifiedMask.buffer);
    if (visibility.length !== triangleCount || visibility.some((value) => value !== 0 && value !== 1)) {
      throw new Error("provisional visibility mask is invalid");
    }
    const bitmap = await createImageBitmap(new Blob([verifiedColor.buffer], { type: "image/jpeg" }));
    const texture = new THREE.Texture(bitmap);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.flipY = false;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const matrixValues = assignment.camera_from_map_row_major;
    if (!Array.isArray(matrixValues) || matrixValues.length !== 16) throw new Error("provisional projection is invalid");
    const cameraFromMap = new THREE.Matrix4().set(...matrixValues.map(Number));
    const intrinsics = assignment.intrinsics || {};
    const projection = new THREE.Vector4(
      Number(intrinsics.fx), Number(intrinsics.fy), Number(intrinsics.cx), Number(intrinsics.cy),
    );
    const imageSize = new THREE.Vector2(Number(intrinsics.width), Number(intrinsics.height));
    if (![projection.x, projection.y, projection.z, projection.w, imageSize.x, imageSize.y].every(Number.isFinite)
        || projection.x <= 0 || projection.y <= 0
        || imageSize.x <= 0 || imageSize.y <= 0) throw new Error("provisional intrinsics are invalid");
    const material = surfaceMaterial.clone();
    material.color.set(0xffffff);
    material.userData = {
      provisionalTextureMaterial: true,
      provisionalTextureRevisionId: assignment.provisional_assignment_revision_id,
      provisionalTexture: texture,
    };
    material.onBeforeCompile = (shader) => {
      shader.uniforms.provisionalMap = { value: texture };
      shader.uniforms.cameraFromMap = { value: cameraFromMap };
      shader.uniforms.provisionalIntrinsics = { value: projection };
      shader.uniforms.provisionalImageSize = { value: imageSize };
      shader.vertexShader = `attribute vec3 mapPosition;\nattribute float provisionalVisibility;\nvarying vec3 vMapPosition;\nvarying float vProvisionalVisibility;\n${shader.vertexShader}`
        .replace(
          "#include <begin_vertex>",
          "#include <begin_vertex>\nvMapPosition = mapPosition;\nvProvisionalVisibility = provisionalVisibility;",
        );
      shader.fragmentShader = `uniform sampler2D provisionalMap;\nuniform mat4 cameraFromMap;\nuniform vec4 provisionalIntrinsics;\nuniform vec2 provisionalImageSize;\nvarying vec3 vMapPosition;\nvarying float vProvisionalVisibility;\n${shader.fragmentShader}`
        .replace(
          "#include <map_fragment>",
          `#include <map_fragment>
vec4 provisionalCameraPoint = cameraFromMap * vec4(vMapPosition, 1.0);
vec2 provisionalPixel = vec2(
  provisionalIntrinsics.x * provisionalCameraPoint.x / max(provisionalCameraPoint.z, 0.0001) + provisionalIntrinsics.z,
  provisionalIntrinsics.y * provisionalCameraPoint.y / max(provisionalCameraPoint.z, 0.0001) + provisionalIntrinsics.w
);
vec2 provisionalUv = vec2(provisionalPixel.x / provisionalImageSize.x, 1.0 - provisionalPixel.y / provisionalImageSize.y);
if (vProvisionalVisibility > 0.5 && provisionalCameraPoint.z > 0.0
    && all(greaterThanEqual(provisionalUv, vec2(0.0))) && all(lessThanEqual(provisionalUv, vec2(1.0)))) {
  diffuseColor.rgb = texture2D(provisionalMap, provisionalUv).rgb;
}`,
        );
    };
    material.customProgramCacheKey = () => `go2-provisional-projective-v1/${assignment.provisional_assignment_revision_id}`;
    return { material, visibility };
  } catch (_error) {
    return null;
  }
}

async function buildSurfaceTile(metadata) {
  const placementId = placementRevisionId(metadata);
  if (!placementId) throw new Error("surface tile revision identity is missing");
  const response = await fetch(metadata.binary_url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`surface tile ${metadata.revision_id} returned HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  const parsed = await parseSurfaceTileInWorker(buffer, metadata.content_sha256);
  const {
    binarySchemaVersion,
    positions: mapPositions,
    normals: mapNormals,
    indices,
    provenanceCounts,
    maximumTriangleUncertaintyM,
  } = parsed;
  const vertexCount = mapPositions.length / 3;
  const triangleCount = indices.length / 3;
  const provisional = await loadProvisionalProjection(metadata.provisional_texture, triangleCount);
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  for (let index = 0; index < vertexCount; index += 1) {
    const offset = index * 3;
    const buildingPosition = applyAssemblyTransform([
      mapPositions[offset],
      mapPositions[offset + 1],
      mapPositions[offset + 2],
    ], metadata);
    const scenePosition = positionToScene(buildingPosition.toArray());
    positions[offset] = scenePosition.x;
    positions[offset + 1] = scenePosition.y;
    positions[offset + 2] = scenePosition.z;
    const buildingNormal = applyAssemblyTransform([
      mapNormals[offset],
      mapNormals[offset + 1],
      mapNormals[offset + 2],
    ], metadata, true);
    const sceneNormal = positionToScene(buildingNormal.toArray()).normalize();
    normals[offset] = sceneNormal.x;
    normals[offset + 1] = sceneNormal.y;
    normals[offset + 2] = sceneNormal.z;
  }
  const geometry = new THREE.BufferGeometry();
  if (provisional) {
    const expandedPositions = new Float32Array(indices.length * 3);
    const expandedNormals = new Float32Array(indices.length * 3);
    const expandedMapPositions = new Float32Array(indices.length * 3);
    const expandedVisibility = new Float32Array(indices.length);
    for (let index = 0; index < indices.length; index += 1) {
      const source = indices[index] * 3;
      const target = index * 3;
      expandedPositions.set(positions.subarray(source, source + 3), target);
      expandedNormals.set(normals.subarray(source, source + 3), target);
      expandedMapPositions.set(mapPositions.subarray(source, source + 3), target);
      expandedVisibility[index] = provisional.visibility[Math.floor(index / 3)];
    }
    geometry.setAttribute("position", new THREE.BufferAttribute(expandedPositions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(expandedNormals, 3));
    geometry.setAttribute("mapPosition", new THREE.BufferAttribute(expandedMapPositions, 3));
    geometry.setAttribute("provisionalVisibility", new THREE.BufferAttribute(expandedVisibility, 1));
  } else {
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    geometry.setAttribute("mapPosition", new THREE.BufferAttribute(mapPositions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  }
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const mesh = new THREE.Mesh(geometry, provisional?.material || surfaceMaterial);
  mesh.name = `surface-tile-${placementId}`;
  mesh.userData = {
    ...metadata,
    binarySchemaVersion,
    provenanceCounts,
    maximumTriangleUncertaintyM,
  };
  return { mesh, metadata, placementId };
}

function updateFloorOptions(tiles) {
  const floors = [...new Set(tiles.map((tile) => tile.floor_id).filter(Boolean))].sort();
  const selected = surfaceFloorSelect.value || "all";
  const activeFloor = latestSnapshot?.building_scan?.active_floor_id;
  surfaceFloorSelect.replaceChildren(new Option("All", "all"));
  for (const floor of floors) surfaceFloorSelect.add(new Option(floor, floor));
  surfaceFloorSelect.value = automaticFloorSelection && floors.includes(activeFloor)
    ? activeFloor
    : floors.includes(selected) ? selected : "all";
}

function updateSurfaceVisibility() {
  let visibleAppearanceTiles = 0;
  if (buildingAppearanceObject) {
    buildingAppearanceObject.visible = surfaceViewMode === "building";
    for (const tileObject of buildingAppearanceObject.children) {
      const floorVisible = surfaceFloorSelect.value === "all"
        || tileObject.userData.floor_id === surfaceFloorSelect.value;
      tileObject.visible = buildingAppearanceObject.visible && floorVisible;
      if (tileObject.visible) visibleAppearanceTiles += 1;
    }
  }
  let visibleTileMeshes = 0;
  for (const { mesh, metadata } of tileMeshes.values()) {
    const floorVisible = surfaceFloorSelect.value === "all" || metadata.floor_id === surfaceFloorSelect.value;
    const aligned = metadata.attachment_state === "ROOT" || metadata.attachment_state === "ATTACHED";
    const assemblyPlaced = metadata.attachment_state === "ROOT"
      || Boolean(metadata.assembly_id && metadata.placement_revision_id);
    const scopeVisible = surfaceViewMode === "building"
      ? aligned && assemblyPlaced
      : !assemblyPlaced && metadata.submap_id === activeSurfaceSubmap;
    mesh.visible = floorVisible && scopeVisible;
    if (mesh.visible) visibleTileMeshes += 1;
  }
  if (measuredSurfaceMesh) {
    const derivedMeasuredSurface = measuredSurfaceState?.sourceMode === "DERIVED_READ_ONLY";
    const localMeasuredSurface = measuredSurfaceState?.sourceMode === "SESSION_ODOM_LOCAL_READ_ONLY";
    measuredSurfaceMesh.visible = (localMeasuredSurface && sessionOdomLocalPreview && surfaceViewMode === "local") || (
      surfaceViewMode === "building" && (
      derivedMeasuredSurface || (visibleAppearanceTiles === 0 && visibleTileMeshes === 0)
      )
    );
  }
  const surfaceReady = visibleAppearanceTiles > 0
    || visibleTileMeshes > 0 || measuredSurfaceMesh?.visible === true;
  applySceneLayerVisibility();

  // Update map texture status based on appearance state
  if (surfaceReady) {
    const qualityEvidence = latestSnapshot?.quality;
    const hasTexture = measuredAppearanceState === "TEXTURED_READY";
    const hasVertexColor = measuredAppearanceState === "VERTEX_COLOR_READY";
    const runtimeAuthority = qualityEvidence?.runtime_authority;

    if (hasTexture) {
      renderMapImageryState("VALIDATED");
    } else if (hasVertexColor) {
      renderMapImageryState("VERTEX_COLOR");
    } else if (runtimeAuthority === "MEASURED" && visibleAppearanceTiles === 0) {
      renderMapImageryState("PROVISIONAL");
    } else {
      renderMapImageryState("UNTEXTURED");
    }
  }

  if (surfaceReady && surfaceFramePending && !orbit.framed) frameSurface();
  return surfaceReady;
}

async function refreshSurfaceManifest() {
  if (surfaceManifestInFlight || document.visibilityState === "hidden"
      || measuredSurfaceState?.readOnly || sessionOdomLocalPreview) return;
  const requestedMode = surfaceViewMode;
  surfaceManifestInFlight = true;
  try {
    const manifestUrl = requestedMode === "building"
      ? "/api/building/render/manifest"
      : `/api/building/surface/manifest?scope=${requestedMode}`;
    const response = await fetch(manifestUrl, { cache: "no-store" });
    if (!response.ok) return;
    const manifest = await response.json();
    if (requestedMode !== surfaceViewMode) return;
    if (requestedMode !== "building" && manifest.scope !== requestedMode) return;
    if (requestedMode === "building" && requestedSceneRevisionId
        && manifest.scene_revision_id !== requestedSceneRevisionId) return;
    const sourceTiles = Array.isArray(manifest.tiles) ? manifest.tiles : [];
    const tiles = requestedMode === "building" ? sourceTiles.map((tile) => ({
      building_id: tile.building_id || manifest.building_id,
      floor_id: tile.floor_id,
      submap_id: tile.submap_id,
      tile_id: tile.tile_id,
      assembly_id: manifest.assembly_id,
      placement_revision_id: tile.placement_revision_id,
      transform_building_from_submap: tile.transform_building_from_submap,
      attachment_state: tile.attachment_state || "ROOT",
      revision_id: tile.geometry?.revision_id || tile.tile_revision_id,
      geometry_revision_id: tile.geometry?.geometry_revision_id,
      binary_url: tile.geometry?.binary_url,
      content_sha256: tile.geometry?.content_sha256,
      binary_schema_version: tile.geometry?.binary_schema_version,
      provisional_texture: tile.provisional_texture,
      validated_texture: tile.validated_texture,
      texture_composition: tile.texture_composition,
    })) : sourceTiles;
    if (requestedMode === "building") {
      activeBuildingId = manifest.building_id || activeBuildingId;
      const hasValidated = tiles.some((tile) => tile.validated_texture);
      const hasProvisional = tiles.some((tile) => tile.provisional_texture);
      const textureState = hasValidated ? "VALIDATED" : hasProvisional ? "PROVISIONAL" : "UNTEXTURED";
      renderMapImageryState(textureState);
    }
    if (tiles.length === 0 && requestedMode === "building") refreshMeasuredSurfacePreview();
    const signature = manifest.scene_revision_id || [
        requestedMode,
        manifest.assembly_id || "unassembled",
        ...tiles.map(placementRevisionId).sort(),
      ].join(":");
    activeSurfaceSubmap = manifest.building_scan?.active_submap_id || activeSurfaceSubmap;
    if (signature !== surfaceManifestSignature) {
      const desiredRevisions = new Set(tiles.map(placementRevisionId));
      const stagedTileMeshes = new Map();
      const pendingTiles = tiles.filter((metadata) => !tileMeshes.has(placementRevisionId(metadata)));
      const stagedResults = await Promise.allSettled(pendingTiles.map(buildSurfaceTile));
      for (const result of stagedResults) {
        if (result.status === "fulfilled") stagedTileMeshes.set(result.value.placementId, result.value);
      }
      const failed = stagedResults.find((result) => result.status === "rejected");
      if (failed) {
        for (const { mesh } of stagedTileMeshes.values()) mesh.geometry.dispose();
        throw new Error("surface tile assembly update failed", { cause: failed.reason });
      }
      if (requestedMode !== surfaceViewMode) {
        for (const { mesh } of stagedTileMeshes.values()) mesh.geometry.dispose();
        return;
      }
      for (const placementId of [...tileMeshes.keys()]) {
        if (!desiredRevisions.has(placementId)) disposeTile(placementId);
      }
      for (const [placementId, entry] of stagedTileMeshes) {
        surfaceGroup.add(entry.mesh);
        tileMeshes.set(placementId, entry);
      }
      currentTileRevisions.clear();
      for (const metadata of tiles) {
        currentTileRevisions.set(tileKey(metadata), placementRevisionId(metadata));
      }
      updateFloorOptions(tiles);
      requestedSceneRevisionId = "";
      surfaceFramePending = true;
      surfaceManifestSignature = signature;
      cloudSignature = "";
    }
    updateSurfaceVisibility();
    if (latestSnapshot) apply(latestSnapshot);
  } catch (_error) {
    // Retain the last complete immutable tile assembly on transport failure.
  } finally {
    surfaceManifestInFlight = false;
    if (requestedMode !== surfaceViewMode) refreshSurfaceManifest();
  }
}

function updatePointCloud(points, projectedColors = null) {
  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const point of points) { minZ = Math.min(minZ, point[2]); maxZ = Math.max(maxZ, point[2]); }
  const zSpan = Math.max(maxZ - minZ, 0.05);
  const low = new THREE.Color(0x3f93e8);
  const high = new THREE.Color(0x51e5d0);
  const color = new THREE.Color();
  points.forEach((point, index) => {
    const scenePoint = positionToScene(point);
    const offset = index * 3;
    positions[offset] = scenePoint.x;
    positions[offset + 1] = scenePoint.y;
    positions[offset + 2] = scenePoint.z;
    if (projectedColors && projectedColors[index]) {
      color.setRGB(projectedColors[index][0], projectedColors[index][1], projectedColors[index][2]);
    } else {
      color.lerpColors(low, high, (point[2] - minZ) / zSpan);
    }
    colors[offset] = color.r;
    colors[offset + 1] = color.g;
    colors[offset + 2] = color.b;
  });
  cloudGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  cloudGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  cloudGeometry.computeBoundingSphere();
}

function transformPoint(matrix, point) {
  return [
    matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
    matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
    matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
  ];
}

function inverseRigid(matrix) {
  const rotationTranspose = [
    matrix[0], matrix[4], matrix[8], 0,
    matrix[1], matrix[5], matrix[9], 0,
    matrix[2], matrix[6], matrix[10], 0,
    0, 0, 0, 1,
  ];
  const translation = [matrix[3], matrix[7], matrix[11]];
  rotationTranspose[3] = -(rotationTranspose[0] * translation[0] + rotationTranspose[1] * translation[1] + rotationTranspose[2] * translation[2]);
  rotationTranspose[7] = -(rotationTranspose[4] * translation[0] + rotationTranspose[5] * translation[1] + rotationTranspose[6] * translation[2]);
  rotationTranspose[11] = -(rotationTranspose[8] * translation[0] + rotationTranspose[9] * translation[1] + rotationTranspose[10] * translation[2]);
  return rotationTranspose;
}

function projectedPointColors(points, snapshot) {
  const camera = snapshot.camera || {};
  const live = snapshot.live_map || {};
  const frame = latestCameraFrame;
  const mapFromLidar = live.pose_map_from_lidar;
  const cameraFromLidar = camera.camera_from_lidar_row_major;
  if (!camera.texture_authorized || !frame || !frame.pixels || !frame.depth || !frame.depthAligned || !Array.isArray(mapFromLidar) || mapFromLidar.length !== 16 || !Array.isArray(cameraFromLidar) || cameraFromLidar.length !== 16) {
    return null;
  }
  const lidarFromMap = inverseRigid(mapFromLidar);
  const { fx, fy, cx, cy } = frame.intrinsics;
  const colors = new Array(points.length);
  for (let index = 0; index < points.length; index += 1) {
    const lidarPoint = transformPoint(lidarFromMap, points[index]);
    const cameraPoint = transformPoint(cameraFromLidar, lidarPoint);
    if (cameraPoint[2] <= 0.05) continue;
    const u = Math.round(fx * cameraPoint[0] / cameraPoint[2] + cx);
    const v = Math.round(fy * cameraPoint[1] / cameraPoint[2] + cy);
    if (u < 0 || v < 0 || u >= frame.width || v >= frame.height) continue;
    const depthM = frame.depth[(v * frame.width + u)] * frame.depthScaleM;
    if (!depthM || Math.abs(depthM - cameraPoint[2]) > Math.max(0.08, cameraPoint[2] * 0.08)) continue;
    const pixel = (v * frame.width + u) * 4;
    colors[index] = [frame.pixels[pixel] / 255, frame.pixels[pixel + 1] / 255, frame.pixels[pixel + 2] / 255];
  }
  return colors;
}

function base64Bytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function depthArray(value) {
  const bytes = base64Bytes(value);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Uint16Array(bytes.byteLength / 2);
  for (let index = 0; index < values.length; index += 1) values[index] = view.getUint16(index * 2, true);
  return values;
}

function renderCamera() {
  const frame = latestCameraFrame;
  if (!frame) {
    syncCameraMedia();
    return;
  }
  const showDepth = cameraView === "depth" && frame.depth;
  if (!showDepth) {
    syncCameraMedia();
    return;
  }
  const width = showDepth ? frame.depthWidth : frame.width;
  const height = showDepth ? frame.depthHeight : frame.height;
  cameraCanvas.width = width;
  cameraCanvas.height = height;
  cameraCanvas.parentElement.style.aspectRatio = `${width} / ${height}`;
  if (showDepth) {
    const image = cameraContext.createImageData(width, height);
    for (let index = 0; index < frame.depth.length; index += 1) {
      const meters = frame.depth[index] * frame.depthScaleM;
      const value = meters > 0.15 && meters < 8 ? Math.round(255 * (1 - Math.min(1, meters / 8))) : 0;
      const pixel = index * 4;
      image.data[pixel] = value >> 2;
      image.data[pixel + 1] = value;
      image.data[pixel + 2] = Math.min(255, value + 40);
      image.data[pixel + 3] = 255;
    }
    cameraContext.putImageData(image, 0, 0);
  } else {
    cameraContext.drawImage(frame.image, 0, 0, width, height);
  }
  cameraEmpty.dataset.ready = "true";
  for (let index = 0; index < measurements.length; index += 1) {
    const point = measurements[index];
    const color = index === 0 ? "#b8e96b" : "#ffc95c";
    cameraContext.strokeStyle = color;
    cameraContext.lineWidth = Math.max(1, frame.width / 240);
    cameraContext.beginPath();
    cameraContext.arc(point.u, point.v, Math.max(4, frame.width / 50), 0, Math.PI * 2);
    cameraContext.stroke();
    cameraContext.fillStyle = color;
    cameraContext.font = `${Math.max(12, frame.width / 24)}px Cascadia Mono`;
    cameraContext.fillText(index === 0 ? "A" : "B", point.u + 6, point.v - 6);
  }
  if (measurements.length === 2) {
    cameraContext.strokeStyle = "#ffc95c";
    cameraContext.lineWidth = Math.max(1, frame.width / 320);
    cameraContext.beginPath();
    cameraContext.moveTo(measurements[0].u, measurements[0].v);
    cameraContext.lineTo(measurements[1].u, measurements[1].v);
    cameraContext.stroke();
  }
  syncCameraMedia();
}

function ensureCameraStream(status) {
  const streamExpected = Number(status?.sequence) >= 0;
  const streamHasPixels = cameraStream.naturalWidth > 0 && cameraStream.naturalHeight > 0;
  if (!streamExpected || streamHasPixels || Date.now() < cameraStreamReconnectAfter) return;
  cameraStreamReconnectAfter = Date.now() + 2000;
  cameraStream.src = `${cameraStream.dataset.src}?reconnect=${Date.now()}`;
}

function syncCameraMedia() {
  const cameraStatus = latestCameraStatus || latestSnapshot?.camera || {};
  const streamHasPixels = cameraStream.naturalWidth > 0 && cameraStream.naturalHeight > 0;
  const streamAvailable = Number(cameraStatus.sequence) >= 0 && streamHasPixels;
  const showDepth = cameraView === "depth" && streamAvailable
    && Boolean(latestCameraFrame && latestCameraFrame.depth);
  const showStream = cameraView === "color" && streamAvailable;
  const showCanvas = showDepth;
  const depthAligned = Boolean(latestCameraFrame && latestCameraFrame.depthAligned);
  const cameraPreviewFps = Number(cameraStatus.preview_fps);
  const cameraDeferred = Boolean(cameraStatus.deferred_by_bandwidth);
  const depthDeferred = Boolean(cameraStatus.depth_deferred_by_bandwidth);
  const sessionStale = latestSnapshot?.session?.state === "STALE";

  cameraAssist.dataset.stale = String(sessionStale);
  cameraStream.dataset.visible = String(showStream);
  cameraCanvas.dataset.visible = String(showCanvas);
  cameraEmpty.dataset.ready = String(showStream || showCanvas);

  if (sessionStale && !streamAvailable) {
    cameraEmpty.textContent = uiText("SESSION_ARCHIVED");
  } else {
    cameraEmpty.textContent = cameraDeferred
      ? uiText("CAMERA_KEYFRAME_DEFERRED")
      : uiText("AWAITING_ALIGNED_RGB_D_KEYFRAME");
  }

  if (showStream && !showDepth) metric.cameraState.textContent = depthDeferred
    ? uiText("LIVE_RGB_DEPTH_DEFERRED")
    : uiText("LIVE_RGB_KEYFRAME");
  if (showDepth) {
    metric.cameraState.textContent = depthAligned
      ? uiText("LIVE_DEPTH_ALIGNED")
      : uiText("LIVE_DEPTH_CAMERA_RELATIVE");
    metric.textureState.textContent = depthAligned
      ? uiText("DEPTH_ALIGNED_READY")
      : uiText("DEPTH_UNALIGNED_RELATIVE");
    metric.textureState.dataset.ready = String(depthAligned);
  } else if (!showStream && !showCanvas) {
    if (sessionStale) {
      metric.cameraState.textContent = uiText("ARCHIVED_SESSION");
    } else {
      metric.cameraState.textContent = cameraDeferred
        ? uiText("CAMERA_DEFERRED_PREVIEW")
        : uiText("CAMERA_PREVIEW_UNAVAILABLE");
    }
  }
  if (showStream && !showDepth && depthDeferred) {
    metric.textureState.textContent = uiText("TEXTURE_DEPTH_DEFERRED");
    metric.textureState.dataset.ready = "false";
  }
  if ((showStream || showCanvas) && Number.isFinite(cameraPreviewFps)) {
    metric.cameraState.textContent += ` / ${cameraPreviewFps.toFixed(1)} FPS`;
  }
  cameraDepthButton.disabled = !Boolean(cameraStatus.distance_measurement_ready);
}

function depthAt(frame, u, v, width, height) {
  const samples = [];
  for (let row = Math.max(0, v - 2); row <= Math.min(height - 1, v + 2); row += 1) {
    for (let column = Math.max(0, u - 2); column <= Math.min(width - 1, u + 2); column += 1) {
      const depthM = frame.depth[row * width + column] * frame.depthScaleM;
      if (depthM > 0.1 && depthM < 12) samples.push(depthM);
    }
  }
  if (!samples.length) return null;
  samples.sort((left, right) => left - right);
  return samples[Math.floor(samples.length / 2)];
}

function updateMeasurementReadout() {
  const label = (point, name) => point ? `${name}: Z ${point.z.toFixed(2)} m / R ${point.range.toFixed(2)} m` : `${name}: --`;
  metric.cameraPointA.textContent = label(measurements[0], "A");
  metric.cameraPointB.textContent = label(measurements[1], "B");
  if (measurements.length === 2) {
    const [left, right] = measurements;
    const distance = Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
    metric.cameraDistance.textContent = `距离：${distance.toFixed(3)} 米`;
  } else {
    metric.cameraDistance.textContent = uiText("DISTANCE_SELECT_TWO_POINTS");
  }
  clearMeasurementsButton.disabled = measurements.length === 0;
}

function addMeasurement(event) {
  const frame = latestCameraFrame;
  if (!frame || !frame.depth) return;
  const useDepth = cameraView === "depth" || frame.depthAligned;
  if (!useDepth) return;
  const width = useDepth ? frame.depthWidth : frame.width;
  const height = useDepth ? frame.depthHeight : frame.height;
  const intrinsics = useDepth ? frame.depthIntrinsics : frame.intrinsics;
  const pixel = containedCanvasPixel(event, width, height);
  if (!pixel) return;
  const { u, v } = pixel;
  const z = depthAt(frame, u, v, width, height);
  if (z === null) return;
  const x = (u - intrinsics.cx) * z / intrinsics.fx;
  const y = (v - intrinsics.cy) * z / intrinsics.fy;
  const point = { u, v, x, y, z, range: Math.hypot(x, y, z) };
  measurements = measurements.length >= 2 ? [point] : [...measurements, point];
  renderCamera();
  updateMeasurementReadout();
}

function containedCanvasPixel(event, width, height) {
  const bounds = cameraCanvas.getBoundingClientRect();
  const scale = Math.min(bounds.width / width, bounds.height / height);
  const renderedWidth = width * scale;
  const renderedHeight = height * scale;
  const left = bounds.left + (bounds.width - renderedWidth) / 2;
  const top = bounds.top + (bounds.height - renderedHeight) / 2;
  const x = event.clientX - left;
  const y = event.clientY - top;
  if (x < 0 || y < 0 || x >= renderedWidth || y >= renderedHeight) return null;
  return {
    u: Math.min(width - 1, Math.floor(x / scale)),
    v: Math.min(height - 1, Math.floor(y / scale)),
  };
}

async function loadCameraFrame(identity = null) {
  if (cameraLoadInFlight || document.visibilityState === "hidden") return;
  if (identity !== null && (!identity || identity === latestCameraIdentity)) return;
  cameraLoadInFlight = true;
  try {
    const response = await fetch("/api/camera/frame", { cache: "no-store" });
    if (!response.ok) return;
    const frame = await response.json();
    const frameIdentity = `${frame.producer_instance_id || "legacy"}:${frame.sequence}`;
    if ((identity !== null && frameIdentity !== identity) || frameIdentity === latestCameraIdentity || !frame.color || !frame.color.jpeg_b64) return;
    const image = await createImageBitmap(new Blob([base64Bytes(frame.color.jpeg_b64)], { type: "image/jpeg" }));
    const buffer = document.createElement("canvas");
    buffer.width = frame.color.width;
    buffer.height = frame.color.height;
    const context = buffer.getContext("2d", { willReadFrequently: true });
    context.drawImage(image, 0, 0, buffer.width, buffer.height);
    const previousImage = latestCameraFrame?.image;
    latestCameraFrame = {
      sequence: frame.sequence,
      image,
      width: frame.color.width,
      height: frame.color.height,
      pixels: context.getImageData(0, 0, buffer.width, buffer.height).data,
      depth: frame.depth ? depthArray(frame.depth.u16_le_b64) : null,
      depthWidth: frame.depth ? frame.depth.width : null,
      depthHeight: frame.depth ? frame.depth.height : null,
      depthScaleM: frame.depth ? Number(frame.depth.scale_m) : null,
      depthAligned: Boolean(frame.depth && frame.depth.aligned_to_color),
      intrinsics: frame.intrinsics,
      depthIntrinsics: frame.depth ? frame.depth.intrinsics : null,
    };
    previousImage?.close();
    latestCameraSequence = frame.sequence;
    latestCameraIdentity = frameIdentity;
    cameraDepthButton.disabled = !latestCameraFrame.depth;
    if (cameraView === "depth" && !latestCameraFrame.depth) cameraView = "color";
    measurements = [];
    updateCameraControls();
    renderCamera();
    updateMeasurementReadout();
    cloudSignature = "";
    if (latestSnapshot) apply(latestSnapshot);
  } catch (_error) {
    // Camera assist is optional; preserve the LiDAR preview on a bad frame.
  } finally {
    cameraLoadInFlight = false;
  }
}

function updateTrajectory(track) {
  const positions = new Float32Array(track.length * 3);
  track.forEach((point, index) => {
    const scenePoint = positionToScene(point, 0.035);
    const offset = index * 3;
    positions[offset] = scenePoint.x;
    positions[offset + 1] = scenePoint.y;
    positions[offset + 2] = scenePoint.z;
  });
  trajectoryGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  trajectoryGeometry.computeBoundingSphere();
}

function disposeHistoricalTrajectories() {
  for (const child of [...historicalTrajectoryGroup.children]) {
    historicalTrajectoryGroup.remove(child);
    child.geometry.dispose();
    child.material.dispose();
  }
}

function renderHistoricalTrajectories(payload) {
  if (!payload || !Array.isArray(payload.trajectories)) return;
  historicalTrajectoryPayload = payload;
  const selectedSession = trajectorySessionSelect.value || "all";
  const sessions = [...new Set(payload.trajectories.map((item) => item.session_id).filter(Boolean))].sort();
  trajectorySessionSelect.replaceChildren(new Option("All", "all"));
  for (const sessionId of sessions) trajectorySessionSelect.add(new Option(sessionId, sessionId));
  trajectorySessionSelect.value = sessions.includes(selectedSession) ? selectedSession : "all";
  disposeHistoricalTrajectories();
  const activeFloor = surfaceFloorSelect.value;
  for (const item of payload.trajectories) {
    if (trajectorySessionSelect.value !== "all" && item.session_id !== trajectorySessionSelect.value) continue;
    if (activeFloor !== "all" && item.floor_id !== activeFloor) continue;
    const samples = Array.isArray(item.samples) ? item.samples : [];
    if (samples.length < 2) continue;
    const positions = new Float32Array(samples.length * 3);
    samples.forEach((sample, index) => {
      const point = positionToScene(sample.position_m, 0.025);
      positions.set(point.toArray(), index * 3);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color: 0x8ea39e, transparent: true, opacity: 0.58 });
    const line = new THREE.Line(geometry, material);
    line.userData = { session_id: item.session_id, floor_id: item.floor_id };
    historicalTrajectoryGroup.add(line);
  }
  historicalTrajectoryGroup.visible = historyVisible;
  historicalTrajectoryRevision = payload.revision_id || "";
}

async function refreshHistoricalTrajectories() {
  if (!historyVisible || !activeBuildingId || historicalTrajectoryInFlight) return;
  historicalTrajectoryInFlight = true;
  try {
    const query = new URLSearchParams({ building_id: activeBuildingId });
    if (surfaceFloorSelect.value !== "all") query.set("floor_id", surfaceFloorSelect.value);
    const response = await fetch(`/api/building/trajectories?${query}`, { cache: "no-store" });
    if (!response.ok) return;
    renderHistoricalTrajectories(await response.json());
  } catch (_error) {
    // Keep the last complete historical route set on a transient failure.
  } finally {
    historicalTrajectoryInFlight = false;
  }
}

const unavailable = (value) => value === null || value === undefined || !Number.isFinite(Number(value));
const densePreviewQuality = "DENSE_LIDAR_PREVIEW_UNCALIBRATED";
const valueText = (value, suffix = "", decimals = 0) => unavailable(value) ? "N/A" : `${Number(value).toFixed(decimals)}${suffix}`;
const metricText = (value, suffix = "", decimals = 0) => unavailable(value) ? "N/A" : `${Number(value).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;

function pointBounds(points) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    for (let axis = 0; axis < 3; axis += 1) {
      minimum[axis] = Math.min(minimum[axis], Number(point[axis]));
      maximum[axis] = Math.max(maximum[axis], Number(point[axis]));
    }
  }
  return minimum.every(Number.isFinite) && maximum.every(Number.isFinite)
    ? { minimum, maximum }
    : null;
}

function measuredLayerCoversPreview(measuredPoints, previewPoints) {
  const measuredBounds = pointBounds(measuredPoints);
  const previewBounds = pointBounds(previewPoints);
  if (!measuredBounds || !previewBounds) return false;
  const measuredExtent = measuredBounds.maximum.map((value, axis) => value - measuredBounds.minimum[axis]);
  const previewExtent = previewBounds.maximum.map((value, axis) => value - previewBounds.minimum[axis]);
  const measuredDiagonal = Math.hypot(...measuredExtent);
  const previewDiagonal = Math.hypot(...previewExtent);
  return previewDiagonal > 0.25 && measuredDiagonal / previewDiagonal >= 0.35;
}

let cloudSignature = "";
let framedMapMode = "";
let framedPointCount = 0;

function apply(snapshot) {
  latestSnapshot = snapshot;
  window.go2DashboardSnapshot = snapshot;
  window.dispatchEvent(new CustomEvent("go2:telemetry", { detail: snapshot }));
  const live = snapshot.live_map || {};
  const roomScan = snapshot.room_scan || live.room_scan || {};
  const buildingScan = snapshot.building_scan || {};
  const transport = snapshot.transport || {};
  const dense = snapshot.dense_map || {};
  const rgbd = dense.rgbd_refinement || {};
  const texture = rgbd.texture || {};
  const depthAnything = snapshot.depth_anything || rgbd.depth_anything || {};
  const highFidelity = transport.high_fidelity || {};
  const progressiveIngest = snapshot.progressive_ingest || {};
  const session = snapshot.session || {};
  updateMappingReview(snapshot);
  if (snapshot.camera) latestCameraStatus = { ...(latestCameraStatus || {}), ...snapshot.camera };
  const cameraAssist = latestCameraStatus || {};
  const cameraSequence = Number.isInteger(cameraAssist.sequence) ? cameraAssist.sequence : -1;
  const cameraIdentity = cameraSequence >= 0 ? `${cameraAssist.producer_instance_id || "legacy"}:${cameraSequence}` : "";
  const liveFresh = roomScan.state === "SCANNING"
    || (!unavailable(transport.last_frame_age_ms) && Number(transport.last_frame_age_ms) <= 15000);
  const hasDenseMap = liveRenderEnabled && Number(dense.points || 0) > 0
    && Array.isArray(dense.preview_points) && dense.preview_points.length > 0;
  const liveRoomMapReady = liveRenderEnabled && liveFresh && Number(live.points || 0) > 0;
  activeSurfaceSubmap = buildingScan.active_submap_id || activeSurfaceSubmap;
  if (automaticFloorSelection && buildingScan.active_floor_id
      && [...surfaceFloorSelect.options].some((option) => option.value === buildingScan.active_floor_id)) {
    surfaceFloorSelect.value = buildingScan.active_floor_id;
  }
  const surfaceReady = updateSurfaceVisibility();
  const measuredSurfaceReady = measuredSurfaceMesh?.visible === true;
  const localOdomSurfaceReady = measuredSurfaceReady
    && measuredSurfaceState.sourceMode === "SESSION_ODOM_LOCAL_READ_ONLY";
  const legacySurfaceReady = measuredSurfaceReady
    && measuredSurfaceState.sourceMode === "LEGACY_SESSION_READ_ONLY";
  geometryProvenance.hidden = !measuredSurfaceReady;
  if (measuredSurfaceReady) {
    const qualityEvidence = latestSnapshot?.quality;
    const runtimeAuthority = qualityEvidence?.runtime_authority;
    const displayQuality = (runtimeAuthority === "MEASURED")
      ? (qualityEvidence?.dense_quality || "DENSE_LIDAR_CALIBRATED")
      : (measuredSurfaceState.quality || "PREVIEW_UNCALIBRATED");
    geometryQuality.textContent = displayQuality;
    geometryTransform.textContent = measuredSurfaceState.sensorFrameTransformId;
    if (localOdomSurfaceReady) {
      geometryTransform.textContent = "ODOM / NO SENSOR TRANSFORM / LOCAL SESSION";
    } else if (measuredSurfaceState.mappingSourceMode === "RECORDED_LIO_REGISTERED_CLOUD") {
      geometryTransform.textContent = "RECORDED_ODOM / NO SENSOR TRANSFORM";
    }
  }
  const activeMap = liveRoomMapReady ? live : hasDenseMap ? dense : live;
  const mapLayer = surfaceReady ? "building-surface" : liveRoomMapReady ? "live-room" : hasDenseMap ? "dense-artifact" : "preview";
  // Dense mapper exposes both the complete candidate voxel layer and the
  // multi-observation measured surface layer.  Render the latter by default;
  // otherwise unstable/single-view voxels make a wall look like a solid fog.
  const measuredPoints = liveRenderEnabled && Array.isArray(activeMap.measured_preview_points)
    ? activeMap.measured_preview_points
    : [];
  const previewPoints = liveRenderEnabled && Array.isArray(activeMap.preview_points) ? activeMap.preview_points : [];
  // Keep the global dense extent visible until the measured layer has enough
  // spatial support. A handful of confirmed voxels must not collapse the map
  // to a tiny island after a Dashboard restart or early artifact ingest.
  const measuredCoverageReady = measuredPoints.length >= Math.max(1000, Math.ceil(Number(activeMap.points || 0) * 0.02))
    && measuredLayerCoversPreview(measuredPoints, previewPoints);
  const points = measuredCoverageReady
    ? measuredPoints
    : previewPoints;
  applySceneLayerVisibility();
  const liveExtent = Array.isArray(roomScan.extent_m) ? roomScan.extent_m : [];
  if (liveCoverageNode) {
    liveCoverageNode.textContent = points.length > 0
      ? `${metricText(points.length)} PTS / ${liveExtent.length === 3 ? liveExtent.map((value) => valueText(value, " m", 1)).join(" x ") : "EXTENT N/A"}`
      : "暂无实时点云";
  }
  if (liveSensorHealthNode) {
    const age = Number(roomScan.last_frame_age_ms ?? transport.last_frame_age_ms);
    liveSensorHealthNode.textContent = liveFresh
      ? `实时 / ${Number.isFinite(age) ? `${Math.round(age)} 毫秒` : "新鲜"}`
      : roomScan.state === "STALE" ? "数据已过期" : "等待数据";
  }
  if (liveAlignmentNode) {
    liveAlignmentNode.textContent = coarseReferenceGroup.children.length && points.length
      ? "仅实时参考"
      : "等待对齐";
  }
  const track = Array.isArray(live.trajectory) ? live.trajectory : [];
  const position = Array.isArray(live.position) ? live.position : [0, 0, 0];
  const headingXY = Array.isArray(live.heading_xy) ? live.heading_xy : [1, 0];
  const hasSpatialData = surfaceReady || points.length > 0
    || track.length > 0 || Boolean(session.session_id);
  const displayPose = stationaryPoseFilter.update(position, headingXY);
  const highFidelityActive = highFidelity.active === true;
  const qualityLabel = measuredSurfaceReady
    ? legacySurfaceReady ? "LEGACY SURFACE" : "DERIVED SURFACE"
    : surfaceReady ? buildingScan.quality : activeMap.quality || "AWAITING KEYFRAME";

  metric.quality.textContent = !measuredSurfaceReady && sessionOdomLocalPreview
    ? "SESSION ODOM LOCAL / AWAITING VERIFIED SURFACE"
    : localOdomSurfaceReady
    ? "SESSION ODOM LOCAL / READ-ONLY"
    : legacySurfaceReady
    ? `${qualityLabel} / READ-ONLY`
    : liveFresh ? `${uiText(qualityLabel)} / 实时` : hasSpatialData ? `${uiText(qualityLabel)} / 数据已过期` : uiText(qualityLabel);
  metric.indicator.className = activeMap.quality === "REFINED" && liveFresh
    ? "good"
    : activeMap.quality === densePreviewQuality || session.capture_active || liveFresh || highFidelityActive ? "warn" : "";
  metric.points.textContent = metricText(activeMap.points);
  metric.densePoints.textContent = metricText(dense.points);
  metric.sequence.textContent = live.last_sequence >= 0 ? metricText(live.last_sequence) : "暂无关键帧";
  metric.uplink.textContent = valueText(transport.app_goodput_mbps, " Mbps", 3);
  metric.rawIngress.textContent = highFidelityActive ? valueText(highFidelity.app_goodput_mbps, " Mbps", 3) : "空闲";
  metric.observedCapacity.textContent = highFidelityActive ? valueText(highFidelity.observed_capacity_mbps, " Mbps", 3) : "空闲";
  metric.linkBudget.textContent = highFidelityActive ? valueText(highFidelity.total_budget_mbps, " Mbps", 3) : "空闲";
  metric.previewBudget.textContent = highFidelityActive ? valueText(highFidelity.preview_budget_mbps, " Mbps", 3) : "空闲";
  metric.highBudget.textContent = highFidelityActive ? valueText(highFidelity.high_fidelity_budget_mbps, " Mbps", 3) : "空闲";
  metric.verifiedArtifacts.textContent = metricText(highFidelity.verified_artifacts);
  metric.reconstructedArtifacts.textContent = metricText(progressiveIngest.release_authorized_artifacts);
  metric.ackRtt.textContent = valueText(transport.ack_rtt_ms, " ms", 1);
  metric.inputRate.textContent = valueText(transport.keyframe_hz, " Hz", 2);
  metric.frameAge.textContent = valueText(transport.last_frame_age_ms, " ms");
  metric.residual.textContent = unavailable(live.residual_cm) ? "未标定" : valueText(live.residual_cm, " 厘米", 2);
  metric.depthCorrespondences.textContent = unavailable(rgbd.depth_correspondence_count)
    ? "未标定" : metricText(rgbd.depth_correspondence_count);
  metric.depthResidualP95.textContent = unavailable(rgbd.depth_residual_p95_m) ? "未标定" : valueText(
    unavailable(rgbd.depth_residual_p95_m) ? null : Number(rgbd.depth_residual_p95_m) * 1000,
    " mm",
    1,
  );
  metric.textureCoverage.textContent = unavailable(texture.surface_coverage)
    ? "未授权"
    : `${(Number(texture.surface_coverage) * 100).toFixed(1)}%`;
  metric.textureBlocker.textContent = rgbd.texture_authorization_blocker || "CALIBRATION REQUIRED";
  metric.roomScanState.textContent = uiText(roomScan.state || "WAITING_FOR_LIVE_DATA");
  metric.roomScanFrames.textContent = metricText(roomScan.frames);
  metric.roomScanSourcePoints.textContent = metricText(roomScan.source_points);
  metric.roomScanTrajectory.textContent = valueText(roomScan.trajectory_length_m, " 米", 2);
  const extent = Array.isArray(roomScan.extent_m) ? roomScan.extent_m : null;
  metric.roomScanExtent.textContent = extent && extent.length === 3
    ? `${valueText(extent[0], " m", 2)} x ${valueText(extent[1], " m", 2)} x ${valueText(extent[2], " m", 2)}`
    : "暂无覆盖";
  metric.buildingScanState.textContent = measuredSurfaceReady
    ? `${localOdomSurfaceReady ? "SESSION ODOM LOCAL" : legacySurfaceReady ? "LEGACY PREVIEW" : measuredSurfaceState.pointAuthority === "LIO_REGISTERED_LIDAR_OBSERVED" ? "REGISTERED OBSERVED PREVIEW" : "MEASURED PREVIEW"} / ${measuredSurfaceState.reconstructionPhase}`
    : uiText(buildingScan.state || "WAITING_FOR_SURFACE");
  metric.buildingFloor.textContent = localOdomSurfaceReady
    ? "NOT A BUILDING FLOOR"
    : legacySurfaceReady
    ? `+${Number(measuredSurfaceState.floorHeightOffsetM).toFixed(3)} M / UNVERIFIED`
    : buildingScan.active_floor_id || "未分配";
  metric.buildingSubmap.textContent = localOdomSurfaceReady
    ? `${measuredSurfaceState.coordinateFrame || "odom"} / CONTINUOUS SESSION`
    : legacySurfaceReady
    ? measuredSurfaceState.poseGeneration
    : buildingScan.active_submap_id || "未分配";
  metric.surfaceSurfels.textContent = measuredSurfaceReady
    ? metricText(measuredSurfaceState.vertexCount)
    : metricText(buildingScan.surfel_count);
  metric.surfaceTriangles.textContent = measuredSurfaceReady
    ? metricText(measuredSurfaceState.triangleCount)
    : metricText(buildingScan.triangle_count);
  metric.surfaceTiles.textContent = measuredSurfaceReady ? "0 / PREVIEW" : metricText(buildingScan.tile_count);
  metric.surfaceUnresolved.textContent = metricText(buildingScan.unresolved_surfel_count);
  metric.retries.textContent = unavailable(transport.duplicate_frames) ? "N/A" : metricText(transport.duplicate_frames);
  metric.pose.textContent = `X ${valueText(displayPose.position[0], "", 3)} / Y ${valueText(displayPose.position[1], "", 3)} / Z ${valueText(displayPose.position[2], "", 3)}`;
  const missionIdentity = session.mission_id
    ? ` / MISSION ${session.mission_id}${session.task_id ? ` / TASK ${session.task_id}` : ""}${session.lease_id ? ` / LEASE ${session.lease_id}` : ""}`
    : "";
  metric.sessionState.textContent = session.capture_active
    ? `${session.session_id} / ${String(session.mode || "").toUpperCase()}${missionIdentity} / HEARTBEAT ${valueText(session.heartbeat_age_ms, " ms")}`
    : session.state === "FINALIZING"
      ? `${session.session_id} / UPLOAD VERIFY / FINALIZING${missionIdentity} / HEARTBEAT ${valueText(session.heartbeat_age_ms, " ms")}`
      : session.state === "STOPPED"
      ? `${session.session_id || "SESSION"} / STOPPED${missionIdentity}`
      : session.state === "STALE"
        ? `${session.session_id || "SESSION"} / STALE${missionIdentity}`
        : "SESSION IDLE";
  metric.capture.textContent = session.capture_active
    ? `RAW RECORDING ACTIVE / LIMIT ${metricText(session.capture_limit_gib)} GiB`
    : session.state === "FINALIZING"
      ? "RAW RECORDING COMPLETE / UPLOAD VERIFY ACTIVE"
      : session.state === "STOPPED" ? "RAW RECORDING STOPPED" : "RAW RECORDING IDLE";
  metric.gate.textContent = localOdomSurfaceReady
    ? "SESSION ODOM LOCAL / READ-ONLY / NOT GLOBAL"
    : activeMap.quality === "REFINED"
    ? "CALIBRATED LIO ACCEPTED"
    : measuredSurfaceReady
      ? legacySurfaceReady
        ? "LEGACY RECONSTRUCTION / NOT GEOMETRY AUTHORITY"
        : "MEASURED SURFACE PREVIEW / NOT ASSEMBLY AUTHORITY"
      : surfaceReady
      ? "OBSERVED SURFACE / CALIBRATION PENDING"
      : liveRoomMapReady
      ? "LIVE ROOM SCAN / CALIBRATION PENDING"
      : hasDenseMap
        ? "DENSE LIDAR / CALIBRATION PENDING"
        : "PREVIEW ONLY / NOT SURVEY-GRADE";
  metric.sceneLabel.textContent = measuredSurfaceReady
    ? localOdomSurfaceReady ? "本地里程计 / 观测表面 / 米" : legacySurfaceReady ? "历史重建 / 米" : "观测表面 / 米"
    : surfaceReady
    ? `${surfaceViewMode === "building" ? "楼栋" : "本地子图"} 观测表面 / 米`
    : sessionOdomLocalPreview ? "等待本地里程计观测表面 / 米" : "等待观测三角表面 / 米";
  metric.sceneLegend.innerHTML = measuredSurfaceReady
    ? localOdomSurfaceReady
      ? '<span class="surface-key"></span> observed odom-frame triangles / no texture authority'
      : legacySurfaceReady
      ? '<span class="surface-key"></span> legacy uncalibrated triangles'
      : '<span class="surface-key"></span> measured artifact triangles'
    : surfaceReady
    ? '<span class="surface-key"></span> observed triangle surface'
    : '<span class="trail-key"></span> current route <span class="robot-key"></span> Go2';
  metric.source.textContent = measuredSurfaceReady
    ? localOdomSurfaceReady ? "本地里程计 / 只读 / 非房间或楼栋" : legacySurfaceReady ? "历史会话 / 只读" : "派生只读制品"
    : sessionOdomLocalPreview ? "本地里程计 / 非全局坐标" : snapshot.source || "等待已验证 LiDAR";
  const simulationStream = snapshot.live_map?.calibration_id === "PROVISIONAL_SIMULATION"
    || snapshot.source?.includes("SIMULATION");
  metric.simulationBadge.hidden = !simulationStream;
  metric.clock.textContent = new Date(snapshot.timestamp_ms || Date.now()).toLocaleTimeString();
  metric.cameraState.textContent = cameraAssist.deferred_by_bandwidth
    ? "相机延迟 / 带宽限制"
    : cameraAssist.state || "等待 RGB-D";
   if ((!measuredAppearanceSurfaceActive || (!measuredSurfaceReady && !sessionOdomLocalPreview))
       && !buildingAppearanceObject && !buildingAppearanceInFlight) {
     const surfaceCoverage = Number(texture.surface_coverage);
     const coloredCoveragePpm = Number.isFinite(surfaceCoverage)
       && surfaceCoverage >= 0 && surfaceCoverage <= 1
       ? Math.round(surfaceCoverage * 1000000) : null;
     const depthValidating = rgbd.state === "CALIBRATED_RGBD_ACTIVE"
       && (Number(rgbd.depth_frames || 0) > 0 || depthAnything.state === "DEPTH_ANYTHING_PENDING");
     const blocker = rgbd.texture_authorization_blocker
       || cameraAssist.texture_blocker
       || (Array.isArray(depthAnything.blockers) ? depthAnything.blockers[0] : null)
       || "APPEARANCE EVIDENCE UNAVAILABLE";
     const baselineTelemetry = renderAppearanceTelemetry({
       state: depthValidating ? "DEPTH_VALIDATING" : "TEXTURE_NOT_AUTHORIZED",
       coloredCoveragePpm,
       physicalDepthAccepted: Number(rgbd.accepted_depth_frames || 0) > 0,
       acceptedDepthFrames: Number.isInteger(rgbd.accepted_depth_frames)
         ? rgbd.accepted_depth_frames : null,
       assistanceCoverage: depthAnything.assistance_coverage,
       blocker,
     });
     measuredAppearanceState = baselineTelemetry.state;
     measuredAppearanceDetails = blocker;
     metric.textureState.textContent = `${baselineTelemetry.state} / ${blocker}`;
     metric.textureState.dataset.ready = "false";
    } else {
      renderAppearanceTelemetry({
        ...appearanceTelemetryState,
        assistanceCoverage: depthAnything.assistance_coverage,
      });
      metric.textureState.textContent = measuredAppearanceDetails
        ? `${measuredAppearanceState} / ${measuredAppearanceDetails}`
       : measuredAppearanceState;
     metric.textureState.dataset.ready = String(
       measuredAppearanceState === "VERTEX_COLOR_READY"
       || measuredAppearanceState === "TEXTURED_READY"
       || measuredAppearanceState === "DIRECT_RGB_VERTEX_COLOR",
     );
  }
  syncCameraMedia();
  const cameraFramePayloadNeeded = !sessionOdomLocalPreview && (cameraView === "depth"
    || cameraAssist.texture_authorized === true);
  if (cameraIdentity && cameraFramePayloadNeeded) loadCameraFrame(cameraIdentity);

  updateTrajectory(track);
  robot.position.copy(positionToScene(displayPose.position, 0.055));
  robot.rotation.y = Math.atan2(displayPose.headingXY[0], -displayPose.headingXY[1]);
  currentFloorCutaway.constant = robot.position.y + 1.8;
  if (followRobotEnabled) {
    orbit.target.lerp(robot.position, 0.35);
    orbit.radius = Math.min(orbit.radius, 8);
    orbit.framed = true;
    updateCamera();
  }
  emptyState.dataset.ready = String(hasSpatialData);
  updateEmptyStateChecklist(snapshot);
  if (!hasSpatialData) return;
  const cameraProjectionIdentity = cameraAssist.texture_authorized ? cameraIdentity : "untextured";
  const signature = mapLayer === "dense-artifact"
    ? `dense:${dense.artifacts || 0}:${dense.points || 0}:measured:${measuredPoints.length}:${measuredCoverageReady}:camera:${cameraProjectionIdentity}:${Boolean(cameraAssist.texture_authorized)}`
    : `preview:${live.last_sequence || -1}:${live.points || 0}:camera:${cameraProjectionIdentity}:${Boolean(cameraAssist.texture_authorized)}`;
  if (liveRenderEnabled && points.length > 0 && signature !== cloudSignature) {
    updatePointCloud(points, projectedPointColors(points, snapshot));
    cloudSignature = signature;
  }
  // Dense artifacts can arrive after the low-latency preview has already
  // framed the scene. Reframe only on a map-layer transition or substantial
  // dense growth, so incremental uploads remain visible without resetting a
  // user's manual orbit on every heartbeat.
  const mapMode = mapLayer;
  const denseGrowth = mapLayer === "dense-artifact" && framedPointCount > 0 && dense.points > framedPointCount * 1.35;
  if (!orbit.framed || framedMapMode !== mapMode || denseGrowth) {
    if (surfaceReady) frameSurface();
    else frameSubmap(points, displayPose.position);
    framedMapMode = mapMode;
    framedPointCount = Number(dense.points || activeMap.points || points.length || 0);
  }
}

function resize() {
  const width = Math.max(canvas.clientWidth, 1);
  const height = Math.max(canvas.clientHeight, 1);
  updateCameraProjection(width, height);
  renderer.setSize(width, height, false);
  const legacySurfacePortrait = height > width;
  if (measuredSurfaceState?.sourceMode === "LEGACY_SESSION_READ_ONLY"
      && lastLegacySurfacePortrait !== null
      && legacySurfacePortrait !== lastLegacySurfacePortrait) {
    lastLegacySurfacePortrait = legacySurfacePortrait;
    orbit.framed = false;
    surfaceFramePending = true;
    requestAnimationFrame(updateSurfaceVisibility);
  }
}

let activePointer = null;
let lastPointer = { x: 0, y: 0 };
canvas.addEventListener("pointerdown", (event) => {
  followRobotEnabled = false;
  followRobotButton.setAttribute("aria-pressed", "false");
  activePointer = event.pointerId;
  lastPointer = { x: event.clientX, y: event.clientY };
  canvas.dataset.dragging = "true";
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (event.pointerId !== activePointer) return;
  const deltaX = event.clientX - lastPointer.x;
  const deltaY = event.clientY - lastPointer.y;
  if (mapProjectionMode === "top") {
    const delta = topDownPanDelta(
      deltaX, deltaY, Math.max(canvas.clientHeight, 1), orthographicVisibleHeight(),
    );
    orbit.target.x += delta.x;
    orbit.target.z += delta.z;
  } else {
    orbit.theta -= deltaX * 0.008;
    orbit.phi = THREE.MathUtils.clamp(orbit.phi + deltaY * 0.008, 0.18, Math.PI - 0.18);
  }
  lastPointer = { x: event.clientX, y: event.clientY };
  updateCamera();
});
function releasePointer(event) {
  if (event.pointerId === activePointer) {
    activePointer = null;
    canvas.dataset.dragging = "false";
  }
}
canvas.addEventListener("pointerup", releasePointer);
canvas.addEventListener("pointercancel", releasePointer);
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  orbit.radius = THREE.MathUtils.clamp(orbit.radius * Math.exp(event.deltaY * 0.001), 1.2, 140);
  updateCamera();
}, { passive: false });
byId("reset-view").addEventListener("click", () => {
  followRobotEnabled = false;
  followRobotButton.setAttribute("aria-pressed", "false");
  orbit.framed = false;
});
mapViewTopButton.addEventListener("click", () => setMapProjectionMode("top"));
mapView3dButton.addEventListener("click", () => setMapProjectionMode("oblique"));
sceneLayerMeasuredButton?.addEventListener("click", () => setSceneLayerMode("measured"));
sceneLayerLiveButton?.addEventListener("click", () => setSceneLayerMode("live"));
sceneLayerCompareButton?.addEventListener("click", () => setSceneLayerMode("compare"));
followRobotButton.addEventListener("click", () => {
  followRobotEnabled = !followRobotEnabled;
  followRobotButton.setAttribute("aria-pressed", String(followRobotEnabled));
  if (followRobotEnabled) {
    automaticFloorSelection = true;
    orbit.target.copy(robot.position);
    orbit.radius = Math.min(orbit.radius, 8);
    orbit.framed = true;
    updateCamera();
  }
});
livePointCloudButton?.addEventListener("click", () => {
  liveRenderEnabled = !liveRenderEnabled;
  livePointCloudButton.setAttribute("aria-pressed", String(liveRenderEnabled));
  livePointCloudButton.textContent = liveRenderEnabled ? "Hide point cloud" : "Point cloud";
  cloudSignature = "";
  if (latestSnapshot) apply(latestSnapshot);
  if (liveRenderEnabled) void refreshFullSnapshot();
});
coverageAuditFocusButton?.addEventListener("click", () => {
  if (!coverageAuditTask) return;
  window.dispatchEvent(new CustomEvent("go2:coverage-focus", { detail: coverageAuditTask }));
});
function setSurfaceView(mode) {
  if (sessionOdomLocalPreview && mode !== "local") return;
  surfaceViewMode = mode;
  if (mode !== "building") buildingAppearanceRequestIdentity = "";
  surfaceBuildingButton.setAttribute("aria-pressed", String(mode === "building"));
  surfaceLocalButton.setAttribute("aria-pressed", String(mode === "local"));
  orbit.framed = false;
  surfaceFramePending = true;
  surfaceManifestSignature = "";
  refreshSurfaceManifest();
  if (mode === "building" && !sessionOdomLocalPreview) refreshBuildingAppearance();
  updateSurfaceVisibility();
  if (latestSnapshot) apply(latestSnapshot);
}
surfaceBuildingButton.addEventListener("click", () => setSurfaceView("building"));
surfaceLocalButton.addEventListener("click", () => setSurfaceView("local"));
surfaceFloorSelect.addEventListener("change", () => {
  automaticFloorSelection = false;
  orbit.framed = false;
  surfaceFramePending = true;
  updateSurfaceVisibility();
  refreshHistoricalTrajectories();
});
trajectoryCurrentButton.addEventListener("click", () => {
  trajectory.visible = !trajectory.visible;
  trajectoryCurrentButton.setAttribute("aria-pressed", String(trajectory.visible));
});
trajectoryHistoryButton.addEventListener("click", () => {
  historyVisible = !historyVisible;
  historicalTrajectoryGroup.visible = historyVisible;
  trajectoryHistoryButton.setAttribute("aria-pressed", String(historyVisible));
  trajectorySessionControl.hidden = !historyVisible;
  if (historyVisible) refreshHistoricalTrajectories();
});
trajectorySessionSelect.addEventListener("change", () => {
  if (historicalTrajectoryPayload) renderHistoricalTrajectories(historicalTrajectoryPayload);
});

async function publishMeasuredSurface() {
  exportMeasuredButton.disabled = true;
  exportResult.hidden = false;
  exportState.textContent = "EXPORTING STABLE GENERATION";
  exportState.dataset.state = "working";
  exportPly.hidden = true;
  exportMetadata.hidden = true;
  try {
    const response = await fetch("/api/reconstruction/publish-measured", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    if (payload.artifact_kind !== "MEASURED_SURFACE_PREVIEW") {
      throw new Error("UNEXPECTED RECONSTRUCTION ARTIFACT");
    }
    exportState.textContent = `${payload.quality_label} / ${payload.vertex_count.toLocaleString()} VERTICES`;
    exportState.dataset.state = payload.delivery_authorized ? "authorized" : "preview";
    exportPly.href = payload.ply_url;
    exportMetadata.href = payload.metadata_url;
    exportPly.hidden = false;
    exportMetadata.hidden = false;
  } catch (error) {
    exportState.textContent = `EXPORT BLOCKED / ${error.message}`;
    exportState.dataset.state = "blocked";
  } finally {
    exportMeasuredButton.disabled = false;
  }
}

exportMeasuredButton.addEventListener("click", publishMeasuredSurface);
cameraCanvas.addEventListener("pointerdown", (event) => {
  event.stopPropagation();
  addMeasurement(event);
});
byId("clear-measurements").addEventListener("click", () => {
  measurements = [];
  renderCamera();
  updateMeasurementReadout();
});
async function setCameraView(view) {
  if (view === "depth" && (!latestCameraFrame || !latestCameraFrame.depth)) {
    await loadCameraFrame(null);
  }
  if (view === "depth" && (!latestCameraFrame || !latestCameraFrame.depth)) return;
  cameraView = view;
  measurements = [];
  updateCameraControls();
  renderCamera();
  syncCameraMedia();
  updateMeasurementReadout();
}

function updateCameraControls() {
  cameraRgbButton.setAttribute("aria-pressed", String(cameraView === "color"));
  cameraDepthButton.setAttribute("aria-pressed", String(cameraView === "depth"));
}

cameraRgbButton.addEventListener("click", () => setCameraView("color"));
cameraDepthButton.addEventListener("click", () => setCameraView("depth"));
cameraFullscreenButton.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement === cameraAssist) {
      await document.exitFullscreen();
    } else {
      await cameraAssist.requestFullscreen();
    }
  } catch (_error) {
    // Some embedded browsers deny fullscreen; the camera window remains usable.
  }
});
document.addEventListener("fullscreenchange", () => {
  const expanded = document.fullscreenElement === cameraAssist;
  cameraFullscreenButton.title = expanded ? "Exit camera assist full screen" : "Open camera assist full screen";
  cameraFullscreenButton.setAttribute("aria-label", cameraFullscreenButton.title);
});
updateCameraControls();
updateMeasurementReadout();

let snapshotRequestInFlight = false;
let fullSnapshotRequestInFlight = false;
let cameraStatusRequestInFlight = false;
let liveSocketConnected = false;
let lastSocketMessageAt = 0;

async function refreshCameraStatus() {
  if (cameraStatusRequestInFlight || document.visibilityState === "hidden") return;
  cameraStatusRequestInFlight = true;
  try {
    const response = await fetch("/api/camera/status", { cache: "no-store" });
    if (!response.ok) return;
    latestCameraStatus = await response.json();
    ensureCameraStream(latestCameraStatus);
    syncCameraMedia();
  } catch (_error) {
    // The MJPEG stream keeps its own reconnecting request; retain the last
    // confirmed status until the next lightweight status probe succeeds.
  } finally {
    cameraStatusRequestInFlight = false;
  }
}

// Telemetry is intentionally point-free at 15 Hz; complete geometry remains
// on the slower snapshot and immutable tile paths.
async function refreshSnapshot(force = false) {
  if (!diagnosticsDrawer.open) return;
  const socketIsFresh = liveSocketConnected && Date.now() - lastSocketMessageAt < 10000;
  if ((!force && socketIsFresh) || snapshotRequestInFlight) return;

  snapshotRequestInFlight = true;
  try {
    const response = await fetch("/api/diagnostics", {cache:'no-store'});
    if (response.ok) apply(await response.json());
  } catch (_error) {
    // The WebSocket reconnect loop remains active when a one-off snapshot request fails.
  } finally {
    snapshotRequestInFlight = false;
  }
}

async function refreshFullSnapshot() {
  if (fullSnapshotRequestInFlight) return;
  fullSnapshotRequestInFlight = true;
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (response.ok) apply(await response.json());
  } catch (_error) {
    // Keep point-free telemetry when a full diagnostic snapshot is unavailable.
  } finally {
    fullSnapshotRequestInFlight = false;
  }
}

async function refreshCoverageAudit() {
  if (coverageAuditInFlight || document.visibilityState === "hidden") return;
  coverageAuditInFlight = true;
  try {
    const missionsResponse = await fetch("/api/missions", { cache: "no-store" });
    if (!missionsResponse.ok) throw new Error(`MISSIONS HTTP ${missionsResponse.status}`);
    const summary = (await missionsResponse.json()).missions?.[0] || null;
    if (!summary?.mission_id) {
      coverageAuditTask = null;
      if (coverageAuditStateNode) { coverageAuditStateNode.textContent = "NO MISSION"; coverageAuditStateNode.dataset.state = "unknown"; }
      if (coverageAuditUnseenNode) coverageAuditUnseenNode.textContent = "N/A";
      if (coverageAuditNextNode) coverageAuditNextNode.textContent = "N/A";
      if (coverageAuditRouteNode) coverageAuditRouteNode.textContent = "N/A";
      if (coverageAuditGateNode) coverageAuditGateNode.textContent = "REVIEW ONLY / NOT APPROVED";
      if (coverageAuditFocusButton) coverageAuditFocusButton.hidden = true;
      return;
    }
    const [coverageResponse, missionResponse] = await Promise.all([
      fetch(`/api/coverage/${encodeURIComponent(summary.mission_id)}`, { cache: "no-store" }),
      fetch(`/api/missions/${encodeURIComponent(summary.mission_id)}`, { cache: "no-store" }),
    ]);
    if (!coverageResponse.ok) throw new Error(`COVERAGE HTTP ${coverageResponse.status}`);
    const payload = await coverageResponse.json();
    const mission = missionResponse.ok ? await missionResponse.json() : null;
    const tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
    const unseen = tasks.filter((task) => String(task.coverage_state || "").toUpperCase() === "UNSEEN" || Number(task.coverage_percent || 0) < 100);
    const coverageTask = unseen[0] || null;
    const missionTask = coverageTask && Array.isArray(mission?.tasks)
      ? mission.tasks.find((task) => task.task_id === coverageTask.task_id)
      : null;
    coverageAuditTask = coverageTask ? { ...coverageTask, ...(missionTask || {}) } : null;
    const state = unseen.length ? "RESCAN_REQUIRED" : tasks.length ? "COVERAGE_COMPLETE" : "NO TASKS";
    if (coverageAuditStateNode) { coverageAuditStateNode.textContent = state; coverageAuditStateNode.dataset.state = unseen.length ? "warning" : "ready"; }
    if (coverageAuditUnseenNode) coverageAuditUnseenNode.textContent = `${unseen.length} TASKS`;
    if (coverageAuditNextNode) coverageAuditNextNode.textContent = coverageAuditTask ? `${coverageAuditTask.task_id || coverageAuditTask.room_id || "TASK"}` : "NONE";
    const routeInfo = taskRoutePoints(coverageAuditTask);
    const route = routeLength(routeInfo.points);
    if (coverageAuditRouteNode) coverageAuditRouteNode.textContent = route === null ? "NO ROUTE" : `${route.toFixed(2)} m / ${coverageAuditTask.estimated_minutes || "N/A"} min / ${routeInfo.source}`;
    if (coverageAuditGateNode) coverageAuditGateNode.textContent = coverageAuditTask
      ? `REVIEW ONLY / ${route === null ? "ROUTE INCOMPLETE" : routeInfo.source === "POLYGON_FALLBACK" ? "POLYGON FALLBACK / REVIEW" : "READY FOR HUMAN REVIEW"}`
      : "REVIEW ONLY / NO RESCAN REQUIRED";
    if (coverageAuditFocusButton) coverageAuditFocusButton.hidden = !coverageAuditTask;
    window.go2Coverage = payload;
    window.dispatchEvent(new CustomEvent("go2:coverage", { detail: { ...payload, next_task: coverageAuditTask } }));
  } catch (error) {
    if (coverageAuditStateNode) { coverageAuditStateNode.textContent = "UNAVAILABLE"; coverageAuditStateNode.dataset.state = "error"; }
    if (coverageAuditNextNode) coverageAuditNextNode.textContent = String(error.message);
    if (coverageAuditFocusButton) coverageAuditFocusButton.hidden = true;
  } finally {
    coverageAuditInFlight = false;
  }
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/live?protocol=3`);
  socket.onopen = () => {
    liveSocketConnected = true;
    lastSocketMessageAt = Date.now();
  };
  socket.onmessage = (event) => {
    lastSocketMessageAt = Date.now();
    const payload = JSON.parse(event.data);
    if (payload.message_type === "snapshot" || payload.message_type === "telemetry") {
      apply(payload.snapshot || payload);
    } else if (payload.message_type === "robot_state") {
      const previous = latestSnapshot || {};
      apply({
        ...previous,
        timestamp_ms: Number(payload.timestamp_ns || 0) / 1_000_000,
        session: { ...(previous.session || {}), ...(payload.session || {}) },
        live_map: {
          ...(previous.live_map || {}),
          position: payload.robot?.position || [0, 0, 0],
          heading_xy: payload.robot?.heading_xy || [1, 0],
        },
        building_scan: {
          ...(previous.building_scan || {}),
          state: payload.building_state,
          quality: payload.building_quality || previous.building_scan?.quality || "PREVIEW_UNCALIBRATED",
          active_floor_id: payload.active_floor_id,
          active_submap_id: payload.active_submap_id || previous.building_scan?.active_submap_id || null,
        },
      });
    } else if (payload.message_type === "trajectory_reset") {
      const previous = latestSnapshot || {};
      const points = Array.isArray(payload.points) ? payload.points : [];
      apply({
        ...previous,
        live_map: { ...(previous.live_map || {}), trajectory: points },
      });
    } else if (payload.message_type === "trajectory_delta") {
      const previous = latestSnapshot || {};
      const trajectory = Array.isArray(previous.live_map?.trajectory)
        ? [...previous.live_map.trajectory]
        : [];
      const startIndex = Number(payload.start_index);
      if (Number.isInteger(startIndex) && startIndex === trajectory.length && Array.isArray(payload.points)) {
        trajectory.push(...payload.points);
        apply({ ...previous, live_map: { ...(previous.live_map || {}), trajectory } });
      }
    } else if (payload.message_type === "tile_revision") {
      activeBuildingId = payload.building_id || activeBuildingId;
      requestedSceneRevisionId = payload.scene_revision_id || "";
      surfaceManifestSignature = "";
      refreshSurfaceManifest();
    }
  };
  socket.onerror = () => socket.close();
  socket.onclose = () => {
    liveSocketConnected = false;
    setTimeout(connect, 1000);
  };
}

function render() {
  requestAnimationFrame(render);
  refreshBuildingAppearanceLods();
  renderer.render(scene, camera);
}

setMapProjectionMode("top");
renderMapImageryState("UNTEXTURED");
void loadCoarseReference();
applySceneLayerVisibility();
new ResizeObserver(resize).observe(canvas);
resize();
updateCamera();
connect();
refreshCameraStatus();
setInterval(refreshCameraStatus, 1000);
refreshRobotControlStatus();
setInterval(refreshRobotControlStatus, 3000);
refreshSurfaceManifest();
setInterval(refreshSurfaceManifest, 2000);
refreshBuildingAppearance();
setInterval(refreshBuildingAppearance, 2000);
refreshMeasuredSurfacePreview();
setInterval(refreshMeasuredSurfacePreview, 1000);
refreshCoverageAudit();
setInterval(refreshCoverageAudit, 3000);
syncCameraMedia();
render();
