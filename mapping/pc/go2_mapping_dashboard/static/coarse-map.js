const byId = (id) => document.getElementById(id);
const svg = byId("coarse-map-svg");
const svg3d = byId("coarse-map-svg-3d");
const threejsContainer = byId("coarse-map-threejs-container");
const mapStage = document.querySelector(".coarse-map-stage");
const view2DButton = byId("coarse-view-2d");
const viewIsoButton = byId("coarse-view-iso");
const view3DButton = byId("coarse-view-3d");
const threeTransformTools = byId("coarse-three-transform-tools");
const threeTransformTranslate = byId("coarse-three-transform-translate");
const threeTransformRotate = byId("coarse-three-transform-rotate");
const threeTransformSpace = byId("coarse-three-transform-space");
const threeTransformSnap = byId("coarse-three-transform-snap");
const roomList = byId("coarse-room-list");
const taskList = byId("coarse-task-list");
const activeTask = byId("coarse-active-task");
const coverage = byId("coarse-coverage");
const authority = byId("coarse-authority");
const taskCount = byId("coarse-task-count");
const navigationState = byId("coarse-navigation-state");
const missionState = byId("coarse-mission-state");
const refreshButton = byId("coarse-map-refresh");
const rebuildButton = byId("coarse-map-rebuild");
const compileButton = byId("coarse-map-compile");
const importButton = byId("coarse-map-json-import");
const exportButton = byId("coarse-map-json-export");
const historyToggle = byId("coarse-map-history-toggle");
const historyPanel = byId("coarse-map-history");
const historyClose = byId("coarse-map-history-close");
const historyList = byId("coarse-map-history-list");
const importInput = byId("coarse-map-json-file");
const zoomInButton = byId("coarse-map-zoom-in");
const zoomOutButton = byId("coarse-map-zoom-out");
const resetViewButton = byId("coarse-map-reset-view");
const editorStatus = byId("coarse-map-editor-status");
const snapInput = byId("coarse-map-snap");
const finishButton = byId("coarse-map-finish");
const cancelButton = byId("coarse-map-cancel");
const entityForm = byId("coarse-entity-form");
const entityNameInput = byId("coarse-entity-name");
const entityKindInput = byId("coarse-entity-kind");
const entityLockedInput = byId("coarse-entity-locked");
const entityDeleteButton = byId("coarse-entity-delete");
const navPauseButton = byId("coarse-navigation-pause");
const navResumeButton = byId("coarse-navigation-resume");
const navCancelButton = byId("coarse-navigation-cancel");
const navSafetyButton = byId("coarse-navigation-safety-stop");
const navAcknowledgeButton = byId("coarse-navigation-acknowledge");
const workspaceModeObserveButton = byId("workspace-mode-observe");
const workspaceModeCoarseButton = byId("workspace-mode-coarse");
const workspaceModeState = byId("workspace-mode-state");
const observationReference = byId("coarse-observation-reference");
const loadCurrentButton = byId("coarse-map-load-current");
const discardDraftButton = byId("coarse-map-discard-draft");
const saveRevisionButton = byId("coarse-map-save-revision");
const nav2ExportButton = byId("coarse-map-nav2-export");
const nav2FrontierButton = byId("coarse-map-nav2-frontiers");
const draftStatus = byId("coarse-map-draft-status");
const draftConflict = byId("coarse-map-draft-conflict");
const reloadServerButton = byId("coarse-map-reload-server");
const keepDraftButton = byId("coarse-map-keep-draft");
const roomScanStartButton = byId("coarse-room-scan-start");
const roomScanDialog = byId("coarse-room-scan-dialog");
const roomScanForm = byId("coarse-room-scan-form");
const roomScanCancelButton = byId("coarse-room-scan-cancel");
const roomScanDuration = byId("coarse-room-scan-duration");
const roomScanToken = byId("coarse-room-scan-token");
const roomScanStatus = byId("coarse-room-scan-status");
const roomScanPhase = byId("coarse-scan-phase");
const roomScanSession = byId("coarse-scan-session");
const roomScanHealth = byId("coarse-scan-health");
const roomScanDetail = byId("coarse-scan-detail");
const roomScanStopButton = byId("coarse-scan-stop");
const stairSurveyState = byId("coarse-stair-survey-state");
const stairSurveyProfile = byId("coarse-stair-survey-profile");
const stairSurveySource = byId("coarse-stair-survey-source");
const stairSurveyBlockers = byId("coarse-stair-survey-blockers");
const stairSurveyRefresh = byId("coarse-stair-survey-refresh");
const buildingIdLabel = byId("coarse-building-id");
const floorIdLabel = byId("coarse-floor-id");
const spaceCountLabel = byId("coarse-space-count");
const buildingFloorCards = byId("coarse-building-floor-cards");
const buildingConnections = byId("coarse-building-connections");
const overviewSummary = byId("coarse-overview-summary");
const buildingMissionButton = byId("coarse-building-mission");
const floorSelect = byId("coarse-floor-select");
const floorAddButton = byId("coarse-floor-add");
const templateOpenButton = byId("coarse-template-open");
const templateDialog = byId("coarse-template-dialog");
const templateGrid = byId("coarse-template-grid");
const templateCancel = byId("coarse-template-cancel");
const connectionAddButton = byId("coarse-connection-add");
const connectionDialog = byId("coarse-connection-dialog");
const connectionForm = byId("coarse-connection-form");
const connectionKind = byId("coarse-connection-kind");
const connectionFrom = byId("coarse-connection-from");
const connectionTo = byId("coarse-connection-to");
const connectionLabel = byId("coarse-connection-label");
const connectionStatus = byId("coarse-connection-status");
const connectionCancel = byId("coarse-connection-cancel");
const connectionStairFields = byId("coarse-connection-stair-fields");
const connectionStepCount = byId("coarse-connection-step-count");
const connectionRiser = byId("coarse-connection-riser");
const connectionTread = byId("coarse-connection-tread");
const connectionWidth = byId("coarse-connection-width");
const connectionDirection = byId("coarse-connection-direction");
const connectionFromX = byId("coarse-connection-from-x");
const connectionFromY = byId("coarse-connection-from-y");
const connectionToX = byId("coarse-connection-to-x");
const connectionToY = byId("coarse-connection-to-y");
const floorDialog = byId("coarse-floor-dialog");
const floorForm = byId("coarse-floor-form");
const floorFormId = byId("coarse-floor-form-id");
const floorFormLabel = byId("coarse-floor-form-label");
const floorFormStatus = byId("coarse-floor-form-status");
const floorFormCancel = byId("coarse-floor-form-cancel");
const furnitureDialog = byId("coarse-furniture-dialog");
const furnitureCancel = byId("coarse-furniture-cancel");
const furnitureSearch = byId("coarse-furniture-search");
const furnitureCategory = byId("coarse-furniture-category");
const furnitureFields = document.querySelector(".coarse-furniture-fields");
const furnitureWidth = byId("coarse-furniture-width");
const furnitureDepth = byId("coarse-furniture-depth");
const furnitureHeight = byId("coarse-furniture-height");
const furnitureRotation = byId("coarse-furniture-rotation");
const rotateFurnitureLeft = byId("coarse-furniture-rotate-left");
const rotateFurnitureRight = byId("coarse-furniture-rotate-right");
const materialDialog = byId("coarse-material-dialog");
const materialCancel = byId("coarse-material-cancel");
const stairFields = byId("coarse-stair-fields");
const stairStepCount = byId("coarse-stair-step-count");
const stairRiser = byId("coarse-stair-riser");
const stairTread = byId("coarse-stair-tread");
const stairWidth = byId("coarse-stair-width");
const stairDirection = byId("coarse-stair-direction");
const stairEntryX = byId("coarse-stair-entry-x");
const stairEntryY = byId("coarse-stair-entry-y");
const stairExitX = byId("coarse-stair-exit-x");
const stairExitY = byId("coarse-stair-exit-y");

const FURNITURE_PRESETS = {
  sofa: { label: "沙发", asset_id: "local/sofa-v1", asset_source: "LOCAL_GLTF", asset_url: "/static/assets/furniture/unit-sofa.gltf", dimensions_m: [2.0, 0.85, 0.8] },
  table: { label: "桌子", asset_id: "cc0/poly-pizza/rAEBvfb1FT", asset_source: "LOCAL_GLTF", asset_url: "/static/assets/furniture/cc0/table-small-quaternius.glb", dimensions_m: [1.4, 0.8, 0.75] },
  bed: { label: "床", asset_id: "cc0/poly-pizza/sn8az3odMR", asset_source: "LOCAL_GLTF", asset_url: "/static/assets/furniture/cc0/bed-single-kenney.glb", dimensions_m: [2.0, 1.6, 0.55] },
  chair: { label: "椅子", asset_id: "cc0/poly-pizza/iMNqRzPwwe", asset_source: "LOCAL_GLTF", asset_url: "/static/assets/furniture/cc0/chair-quaternius.glb", dimensions_m: [0.55, 0.55, 0.9] },
  cabinet: { label: "柜子", asset_id: "cc0/poly-pizza/wOiMrnUuhe", asset_source: "LOCAL_GLTF", asset_url: "/static/assets/furniture/cc0/cabinet-creativetrio.glb", dimensions_m: [1.2, 0.45, 1.8] },
  plant: { label: "绿植", asset_id: "cc0/poly-pizza/bfLOqIV5uP", asset_source: "LOCAL_GLTF", asset_url: "/static/assets/furniture/cc0/houseplant-quaternius.glb", dimensions_m: [0.45, 0.45, 1.2] },
};
let pendingFurnitureType = null;
let furnitureCatalog = { ...FURNITURE_PRESETS };
let coarseMapTemplates = [];
let pbrMaterialCatalog = [];

function ensureEntityKindEditor() {
  const current = byId("coarse-entity-kind");
  if (!current || current.tagName === "SELECT") return current;
  const select = document.createElement("select");
  select.id = current.id;
  select.setAttribute("aria-label", "功能类型");
  [["room", "功能区"], ["landing", "平台"], ["stair", "楼梯间"], ["elevator_hall", "电梯厅"], ["corridor", "走廊"], ["equipment", "设备"], ["obstacle", "障碍物"], ["inspection_target", "检查目标"], ["door", "门"], ["anchor", "锚点"], ["keep_out_zone", "禁行区"]].forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  });
  current.replaceWith(select);
  return select;
}

const TASK_MODE_SEQUENCE = ["MANUAL", "ASSISTED", "AUTONOMOUS"];
const DEFAULT_MAP_ID = "go2-demo-floor";
const VIEWBOX = { width: 520, height: 320 };
const MIN_DRAW_POINTS = { room: 3, corridor: 3, stair: 3, landing: 3, elevator_hall: 3, wall: 2, keep_out_zone: 3, carpet: 3, threshold: 2 };
const ENTITY_CONFIG = {
  carpet: { collection: "carpets", geometry: "polygon", label: "地毯" },
  threshold: { collection: "thresholds", geometry: "line", label: "门槛条" },
  stair: { collection: "rooms", geometry: "polygon", label: "楼梯间" },
  landing: { collection: "rooms", geometry: "polygon", label: "平台" },
  elevator_hall: { collection: "rooms", geometry: "polygon", label: "电梯厅" },
  room: { collection: "rooms", geometry: "polygon", label: "功能区" },
  corridor: { collection: "corridors", geometry: "polygon", label: "走廊" },
  wall: { collection: "walls", geometry: "line", label: "墙线" },
  door: { collection: "doors", geometry: "point", label: "门" },
  equipment: { collection: "features", geometry: "point", label: "设备" },
  inspection_target: { collection: "features", geometry: "point", label: "检查目标" },
  anchor: { collection: "anchors", geometry: "point", label: "锚点" },
  keep_out_zone: { collection: "keep_out_zones", geometry: "polygon", label: "禁行区" },
};

let coarseMap = null;
let serverMap = null;
let draftMap = null;
let mapId = DEFAULT_MAP_ID;
let selectedRoomId = null;
let selectedEntity = null;
let selectedEntityIds = [];
let currentMission = null;
const routeAudits = new Map();
let routeAuditRequestKey = "";
let currentNavigation = null;
let robotPosition = null;
let robotPositionRenderQueued = false;
let frontierCandidates = [];
let rescanPreviewTask = null;
let editor = { tool: "select", draftPoints: [], dragging: null, saving: false };
let lastDrawClick = null;
let draftCommitInFlight = false;
let mapViewport = { scale: 1, offsetX: 0, offsetY: 0, dragging: null };
let viewMode = "2d";
let workspaceMode = new URLSearchParams(location.search).get("mode") === "coarse" ? "coarse" : "observe";
let draftBaseRevision = 1;
let draftDirty = false;
let draftInitialized = false;
let draftHistory = [];
let draftFuture = [];
let draftStorageTimer = null;
let draftStorageKey = "";
let draftConflictPayload = null;
let buildingIndex = null;

function applyDashboardTelemetry(snapshot = {}) {
  const position = snapshot.live_map?.position || snapshot.robot?.position;
  if (!Array.isArray(position) || position.length < 2) return;
  const x = Number(position[0]);
  const y = Number(position[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  if (robotPosition && Math.hypot(robotPosition[0] - x, robotPosition[1] - y) < 0.01) return;
  robotPosition = [x, y];
  if (coarseMap && workspaceMode === "coarse" && !robotPositionRenderQueued) {
    robotPositionRenderQueued = true;
    requestAnimationFrame(() => {
      robotPositionRenderQueued = false;
      renderMap();
    });
  }
}

window.addEventListener("go2:telemetry", (event) => applyDashboardTelemetry(event.detail));
if (window.go2DashboardSnapshot) applyDashboardTelemetry(window.go2DashboardSnapshot);
window.addEventListener("go2:coverage-focus", (event) => {
  const roomId = event.detail?.room_id;
  if (!roomId || !findEntity("room", roomId)) return;
  rescanPreviewTask = event.detail || null;
  selectEntity("room", roomId);
  document.querySelector("#coarse-map-card")?.scrollIntoView({ behavior: "smooth", block: "center" });
  setEditorStatus(`已定位补扫建议：${roomId}。仅供人工审核。`, "warning");
});
let roomScanRequestId = "";
let roomScanOperatorToken = "";
let roomScanPollTimer = null;

function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[character]));
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function taskStatusIcon(status) {
  if (status === "DONE") return "✓";
  if (status === "ACTIVE") return "▶";
  if (status === "PAUSED") return "Ⅱ";
  if (status === "QUARANTINED") return "!";
  return "•";
}

function currentMissionTaskFor(taskId) {
  return currentMission?.tasks?.find((task) => task.task_id === taskId) || null;
}

function routeAuditFor(taskId) {
  return routeAudits.get(String(taskId)) || null;
}

function routeAuditLabel(report) {
  if (!report) return "AUDIT_PENDING";
  if (report.navigation_eligible === true && report.state === "READY_FOR_HUMAN_REVIEW") return "READY_FOR_HUMAN_REVIEW";
  if (report.state === "BLOCKED") return "BLOCKED";
  if (report.route_source === "POLYGON_FALLBACK") return "REVIEW_ONLY / POLYGON_FALLBACK";
  return String(report.state || "REVIEW_REQUIRED");
}

function routeAuditFindings(report) {
  if (!report) return "Route audit is loading; GO remains disabled.";
  const findings = Array.isArray(report.findings) ? report.findings : [];
  if (!findings.length) return "No findings returned; human review is still required.";
  return findings.map((item) => `${item.code} [${item.severity}]`).join(" / ");
}

function renderRouteAuditSummary(taskId = null) {
  const stateNode = byId("coarse-route-audit-state");
  const findingsNode = byId("coarse-route-audit-findings");
  const task = taskId ? currentMissionTaskFor(taskId) : (currentMission?.tasks || [])[0];
  const report = task ? routeAuditFor(task.task_id) : null;
  if (stateNode) {
    stateNode.textContent = routeAuditLabel(report);
    stateNode.dataset.state = !report ? "unknown" : report.navigation_eligible === true ? "ready" : report.state === "BLOCKED" ? "error" : "warning";
  }
  if (findingsNode) findingsNode.textContent = task ? `${task.task_id} / ${routeAuditFindings(report)}` : "No mission task selected.";
}

async function refreshMissionRouteAudits() {
  if (!currentMission?.mission_id || !Array.isArray(currentMission.tasks)) {
    routeAudits.clear();
    routeAuditRequestKey = "";
    renderRouteAuditSummary();
    return;
  }
  const source = serverMap || coarseMap;
  const key = `${currentMission.mission_id}:${currentMission.tasks.map((task) => `${task.task_id}:${task.revision || 1}`).join(",")}:${source?.revision || 0}`;
  if (key === routeAuditRequestKey) return;
  routeAuditRequestKey = key;
  routeAudits.clear();
  renderLists();
  await Promise.all(currentMission.tasks.map(async (task) => {
    try {
      const response = await fetch(`/api/missions/${encodeURIComponent(currentMission.mission_id)}/tasks/${encodeURIComponent(task.task_id)}/route-audit`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(source ? { source_coarse_map: source } : {}),
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      routeAudits.set(String(task.task_id), await response.json());
    } catch (_error) {
      routeAudits.set(String(task.task_id), { state: "UNAVAILABLE", navigation_eligible: false, findings: [{ code: "AUDIT_UNAVAILABLE", severity: "BLOCKER" }] });
    }
  }));
  renderLists();
}

function toV2(payload) {
  if (payload?.schema_version === 2) {
    return {
      ...payload,
      rooms: Array.isArray(payload.rooms) ? payload.rooms : [],
      corridors: Array.isArray(payload.corridors) ? payload.corridors : [],
      walls: Array.isArray(payload.walls) ? payload.walls : [],
      doors: Array.isArray(payload.doors) ? payload.doors : [],
      features: Array.isArray(payload.features) ? payload.features : [],
      keep_out_zones: Array.isArray(payload.keep_out_zones) ? payload.keep_out_zones : [],
      anchors: Array.isArray(payload.anchors) ? payload.anchors : [],
      carpets: Array.isArray(payload.carpets) ? payload.carpets : [],
      thresholds: Array.isArray(payload.thresholds) ? payload.thresholds : [],
    };
  }
  const features = Array.isArray(payload?.features) ? payload.features : [];
  return {
    schema_version: 2,
    map_id: String(payload?.map_id || DEFAULT_MAP_ID),
    coordinate_frame: payload?.coordinate_frame || "SESSION_ODOM_LOCAL",
    building_id: "building-1",
    floor_id: String(payload?.map_id || DEFAULT_MAP_ID),
    rooms: Array.isArray(payload?.rooms) ? payload.rooms : [],
    corridors: [],
    walls: [],
    doors: features.filter((feature) => feature.kind === "door"),
    features: features.filter((feature) => !["door", "anchor"].includes(feature.kind)),
    keep_out_zones: Array.isArray(payload?.keep_out_zones) ? payload.keep_out_zones : [],
    anchors: features.filter((feature) => feature.kind === "anchor"),
    carpets: [],
    thresholds: [],
    provenance: "USER_PRIOR",
    geometry_authority: false,
    revision: 1,
    tasks: Array.isArray(payload?.tasks) ? payload.tasks : [],
    updated_at: payload?.updated_at,
  };
}

function entityCollection(kind) {
  return ENTITY_CONFIG[kind]?.collection || "features";
}

function cloneMap(value) {
  return value ? JSON.parse(JSON.stringify(value)) : null;
}

function draftStorageKeyForMap(id) {
  return `go2.coarse-map.draft.v1:${id}`;
}

function updateDraftControls() {
  const revision = Number(draftBaseRevision || serverMap?.revision || 1);
  if (draftStatus) {
    draftStatus.textContent = draftDirty
      ? `DRAFT / BASE REVISION ${revision} / UNSAVED`
      : `SERVER MAP / REVISION ${Number(serverMap?.revision || revision)}`;
    draftStatus.dataset.state = draftDirty ? "dirty" : "clean";
  }
  if (saveRevisionButton) saveRevisionButton.disabled = !draftDirty || workspaceMode !== "coarse";
  if (discardDraftButton) discardDraftButton.disabled = workspaceMode !== "coarse";
}

function saveDraftToStorage() {
  if (!draftStorageKey || !coarseMap || !draftDirty) return;
  try {
    localStorage.setItem(draftStorageKey, JSON.stringify({
      schema_version: 1,
      map_id: mapId,
      base_revision: draftBaseRevision,
      saved_at: new Date().toISOString(),
      draft: coarseMap,
      active_tool: editor.tool,
      draft_points: editor.draftPoints,
    }));
  } catch (_error) {
    setEditorStatus("无法保存浏览器草稿；当前编辑仍在页面中保留。", "warning");
  }
}

function scheduleDraftStorage() {
  if (draftStorageTimer) window.clearTimeout(draftStorageTimer);
  draftStorageTimer = window.setTimeout(() => {
    draftStorageTimer = null;
    saveDraftToStorage();
  }, 250);
}

function loadDraftFromStorage() {
  if (!draftStorageKey) return null;
  try {
    const raw = localStorage.getItem(draftStorageKey);
    if (!raw) return null;
    const stored = JSON.parse(raw);
    if (stored?.schema_version !== 1 || stored.map_id !== mapId || !stored.draft) {
      localStorage.removeItem(draftStorageKey);
      return null;
    }
    const draft = toV2(stored.draft);
    const baseRevision = Number(stored.base_revision);
    if (draft.map_id !== mapId || !Number.isInteger(baseRevision) || baseRevision <= 0) {
      localStorage.removeItem(draftStorageKey);
      return null;
    }
    return {
      draft,
      baseRevision,
      activeTool: ENTITY_CONFIG[stored.active_tool] ? stored.active_tool : "select",
      draftPoints: Array.isArray(stored.draft_points) ? stored.draft_points : [],
    };
  } catch (_error) {
    try { localStorage.removeItem(draftStorageKey); } catch (_ignored) { /* storage may be unavailable */ }
    return null;
  }
}

function blankDraftFromServer() {
  const base = cloneMap(serverMap) || toV2({ map_id: mapId });
  return toV2({
    ...base,
    rooms: [],
    corridors: [],
    walls: [],
    doors: [],
    features: [],
    keep_out_zones: [],
    anchors: [],
    carpets: [],
    thresholds: [],
    tasks: [],
    revision: Number(serverMap?.revision || 1),
    updated_at: null,
  });
}

function resetDraftHistory() {
  draftHistory = [];
  draftFuture = [];
}

function activateCoarseDraft() {
  if (!serverMap) return;
  mapId = String(serverMap.map_id || DEFAULT_MAP_ID);
  draftStorageKey = draftStorageKeyForMap(mapId);
  const stored = loadDraftFromStorage();
  if (stored) {
    coarseMap = stored.draft;
    draftMap = cloneMap(coarseMap);
    draftBaseRevision = stored.baseRevision;
    draftDirty = true;
    editor.tool = stored.activeTool;
    editor.draftPoints = stored.draftPoints;
    setEditorStatus("已恢复本地草稿；保存前不会覆盖服务器版本。", "warning");
  } else {
    coarseMap = blankDraftFromServer();
    draftMap = cloneMap(coarseMap);
    draftBaseRevision = Number(serverMap.revision || 1);
    draftDirty = false;
    editor.tool = "select";
    editor.draftPoints = [];
    setEditorStatus("空白草稿已就绪；选择房间工具开始绘制。", "idle");
  }
  draftInitialized = true;
  resetDraftHistory();
  updateDraftControls();
  renderMap();
  renderLists();
  showSelectedEntity();
}

function markDraftChanged(message = "草稿已修改，保存后生成新的 USER_PRIOR revision。") {
  draftDirty = true;
  draftMap = cloneMap(coarseMap);
  updateDraftControls();
  if (message) setEditorStatus(message, "editing");
  scheduleDraftStorage();
}

function captureDraftHistory() {
  if (!coarseMap || workspaceMode !== "coarse") return;
  draftHistory.push(cloneMap(coarseMap));
  if (draftHistory.length > 50) draftHistory.shift();
  draftFuture = [];
}

function undoDraft() {
  if (workspaceMode !== "coarse" || !draftHistory.length) return;
  draftFuture.push(cloneMap(coarseMap));
  coarseMap = draftHistory.pop();
  selectedEntity = null;
  markDraftChanged("已撤销；草稿尚未保存。");
  renderMap();
  renderLists();
  showSelectedEntity();
}

function redoDraft() {
  if (workspaceMode !== "coarse" || !draftFuture.length) return;
  draftHistory.push(cloneMap(coarseMap));
  coarseMap = draftFuture.pop();
  selectedEntity = null;
  markDraftChanged("已重做；草稿尚未保存。");
  renderMap();
  renderLists();
  showSelectedEntity();
}

function updateWorkspaceModeUrl(mode) {
  const url = new URL(window.location.href);
  if (mode === "coarse") url.searchParams.set("mode", "coarse");
  else url.searchParams.delete("mode");
  history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
}

function setWorkspaceMode(mode, { syncUrl = true } = {}) {
  workspaceMode = mode === "coarse" ? "coarse" : "observe";
  document.documentElement.dataset.workspaceMode = workspaceMode;
  document.documentElement.dataset.referenceOpen = "false";
  if (observationReference) observationReference.open = false;
  if (workspaceModeObserveButton) workspaceModeObserveButton.setAttribute("aria-pressed", String(workspaceMode === "observe"));
  if (workspaceModeCoarseButton) workspaceModeCoarseButton.setAttribute("aria-pressed", String(workspaceMode === "coarse"));
  if (workspaceModeState) workspaceModeState.textContent = workspaceMode.toUpperCase();
  if (syncUrl) updateWorkspaceModeUrl(workspaceMode);
  if (workspaceMode === "coarse") {
    document.querySelector(".coarse-task-panel")?.removeAttribute("open");
    document.getElementById("evidence-drawer")?.removeAttribute("open");
    if (serverMap && !draftInitialized) activateCoarseDraft();
    if (serverMap && draftInitialized && draftMap) coarseMap = cloneMap(draftMap);
  } else {
    coarseMap = cloneMap(serverMap) || coarseMap;
    selectedEntity = null;
    updateDraftControls();
    renderMap();
    renderLists();
    showSelectedEntity();
  }
}

function loadCurrentMapIntoDraft() {
  if (!serverMap || workspaceMode !== "coarse") return;
  coarseMap = cloneMap(serverMap);
  draftMap = cloneMap(coarseMap);
  draftBaseRevision = Number(serverMap.revision || 1);
  draftDirty = false;
  draftInitialized = true;
  resetDraftHistory();
  try { localStorage.removeItem(draftStorageKey); } catch (_error) { /* storage may be unavailable */ }
  setEditorStatus(`已加载服务器版 revision ${draftBaseRevision}；当前可继续编辑。`, "saved");
  updateDraftControls();
  renderMap();
  renderLists();
  showSelectedEntity();
}

function discardLocalDraft() {
  if (workspaceMode !== "coarse") return;
  try { localStorage.removeItem(draftStorageKey); } catch (_error) { /* storage may be unavailable */ }
  coarseMap = blankDraftFromServer();
  draftMap = cloneMap(coarseMap);
  draftBaseRevision = Number(serverMap?.revision || 1);
  draftDirty = false;
  draftInitialized = true;
  selectedEntity = null;
  resetDraftHistory();
  setEditorStatus("本地草稿已放弃；服务器 revision 未改变。", "saved");
  updateDraftControls();
  renderMap();
  renderLists();
  showSelectedEntity();
}

function entityGeometry(kind, entity) {
  const config = ENTITY_CONFIG[kind];
  if (!config) return [];
  if (config.geometry === "point") return Array.isArray(entity?.position) ? entity.position : [];
  return Array.isArray(entity?.polygon) ? entity.polygon : (Array.isArray(entity?.geometry) ? entity.geometry : []);
}

function entityLabel(kind, entity) {
  return String(entity?.name || entity?.label || entity?.id || ENTITY_CONFIG[kind]?.label || kind);
}

function allEntities(map) {
  if (!map) return [];
  const result = [];
  const visitedCollections = new Set();
  Object.entries(ENTITY_CONFIG).forEach(([kind, config]) => {
    if (visitedCollections.has(config.collection)) return;
    visitedCollections.add(config.collection);
    (map[config.collection] || []).forEach((entity) => {
      const entityKind = ENTITY_CONFIG[entity.kind]?.collection === config.collection ? entity.kind : kind;
      result.push({ kind: entityKind, entity });
    });
  });
  return result;
}

function findEntity(kind, id) {
  return (coarseMap?.[entityCollection(kind)] || []).find((entity) => entity.id === id) || null;
}

function coverageStatus(entity) {
  const diagnostics = entity?.diagnostics;
  if (entity?.coverage_state === "QUARANTINED" || entity?.status === "QUARANTINED" || diagnostics?.state === "QUARANTINED") return "QUARANTINED";
  if (entity?.coverage_state === "VERIFIED" || entity?.status === "VERIFIED" || entity?.provenance === "VERIFIED" || finiteNumber(entity?.coverage_percent) >= 80) return "VERIFIED";
  if (entity?.coverage_state === "OBSERVED" || entity?.status === "OBSERVED" || entity?.provenance === "OBSERVED" || finiteNumber(entity?.coverage_percent) > 0) return "OBSERVED";
  if (entity?.provenance === "USER_PRIOR") return "USER_PRIOR";
  return "UNSEEN";
}

function bounds(map) {
  const points = allEntities(map).flatMap(({ kind, entity }) => {
    const geometry = entityGeometry(kind, entity);
    return ENTITY_CONFIG[kind].geometry === "point" ? (geometry.length === 2 ? [geometry] : []) : geometry;
  });
  if (robotPosition?.length === 2) points.push(robotPosition);
  if (!points.length) return { minX: -2, minY: -1.5, maxX: 6, maxY: 4.5 };
  const xs = points.map((point) => finiteNumber(point[0]));
  const ys = points.map((point) => finiteNumber(point[1]));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padding = Math.max(0.35, Math.max(maxX - minX, maxY - minY) * 0.1);
  return { minX: minX - padding, minY: minY - padding, maxX: maxX + padding, maxY: maxY + padding };
}

function projection(box) {
  const width = Math.max(1, box.maxX - box.minX);
  const height = Math.max(1, box.maxY - box.minY);
  const scale = Math.min(450 / width, 250 / height);
  return { scale, offsetX: (VIEWBOX.width - width * scale) / 2, offsetY: (VIEWBOX.height - height * scale) / 2 };
}

function project(point, box) {
  const transform = projection(box);
  return [
    transform.offsetX + (finiteNumber(point[0]) - box.minX) * transform.scale,
    transform.offsetY + (box.maxY - finiteNumber(point[1])) * transform.scale,
  ];
}

function clientToMapPoint(event) {
  const rect = svg.getBoundingClientRect();
  const x = (event.clientX - rect.left) * VIEWBOX.width / Math.max(1, rect.width);
  const y = (event.clientY - rect.top) * VIEWBOX.height / Math.max(1, rect.height);
  const box = bounds(coarseMap);
  const transform = projection(box);
  const viewportX = (x - mapViewport.offsetX) / mapViewport.scale;
  const viewportY = (y - mapViewport.offsetY) / mapViewport.scale;
  const point = [box.minX + (viewportX - transform.offsetX) / transform.scale, box.maxY - (viewportY - transform.offsetY) / transform.scale];
  if (snapInput?.checked) {
    point[0] = Math.round(point[0] * 4) / 4;
    point[1] = Math.round(point[1] * 4) / 4;
  }
  return point;
}

function renderEntity(kind, entity, box) {
  const config = ENTITY_CONFIG[kind];
  const selected = selectedEntity?.kind === kind && selectedEntity.id === entity.id
    || selectedEntityIds.includes(entity.id);
  const geometry = entityGeometry(kind, entity);
  const state = coverageStatus(entity);
  const id = escapeText(entity.id);
  const label = escapeText(entityLabel(kind, entity));
  const data = `data-entity-kind="${escapeText(kind)}" data-entity-id="${id}"`;
  if (config.geometry === "point") {
    if (geometry.length !== 2) return "";
    const [x, y] = project(geometry, box);
    const pointClass = kind === "door" ? "door" : kind === "anchor" ? "anchor" : kind === "inspection_target" ? "inspection-target" : "equipment";
    return `<g class="coarse-feature ${pointClass} coverage-${state.toLowerCase()}${selected ? " selected" : ""}" data-coverage-state="${state}" ${data} transform="translate(${x} ${y})"><circle r="7"/><text x="11" y="4">${label}</text></g>`;
  }
  if (geometry.length < 2) return "";
  const points = geometry.map((point) => project(point, box).join(",")).join(" ");
  const isKeepOut = kind === "keep_out_zone";
  const isArea = ["room", "corridor", "stair", "landing", "elevator_hall", "keep_out_zone", "carpet"].includes(kind);
  const coverageClass = ` coverage-${state.toLowerCase()}`;
  const fill = kind === "carpet" ? "#e7d7f0" : isKeepOut || state === "QUARANTINED" ? "#4b2928" : selected ? "#16463f" : state === "VERIFIED" ? "#173d32" : state === "OBSERVED" ? "#123d39" : "#102f2d";
  const stroke = isKeepOut || state === "QUARANTINED" ? "#ff7867" : kind === "wall" ? "#51e5d0" : selected ? "#80eadb" : state === "VERIFIED" ? "#72d3a4" : state === "OBSERVED" ? "#51c9bd" : "#397a70";
  const shape = isArea
    ? `<polygon points="${points}" fill="${fill}" stroke="${stroke}" stroke-width="${selected ? 3 : 2}" ${isKeepOut ? "stroke-dasharray=\"7 5\"" : ""}/>`
    : `<polyline points="${points}" fill="none" stroke="${stroke}" stroke-width="${selected ? 4 : 2}" stroke-linecap="round" stroke-linejoin="round"/>`;
  const first = project(geometry[0], box);
  const labelMarkup = isArea
    ? `<text x="${first[0] + 8}" y="${first[1] - 8}" class="coarse-room-label">${label}</text><text x="${first[0] + 8}" y="${first[1] + 9}" class="coarse-room-coverage">${finiteNumber(entity.coverage_percent).toFixed(0)}% ${isKeepOut ? "restricted" : "verified"}</text>`
    : `<text x="${first[0] + 8}" y="${first[1] - 6}" class="coarse-wall-label">${label}</text>`;
  const handles = selected ? geometry.map((point, index) => {
    const [x, y] = project(point, box);
    return `<circle class="coarse-vertex-handle" data-entity-kind="${escapeText(kind)}" data-entity-id="${id}" data-vertex-index="${index}" cx="${x}" cy="${y}" r="5"/>`;
  }).join("") : "";
  return `<g class="coarse-entity coarse-${escapeText(kind)}${coverageClass}${selected ? " selected" : ""}" data-coverage-state="${state}" ${data}>${shape}${labelMarkup}${handles}</g>`;
}

function resetMapView() {
  mapViewport = { scale: 1, offsetX: 0, offsetY: 0, dragging: null };
  setEditorStatus("地图视图已复位。", "saved");
  renderMap();
}

function zoomAt(screenX, screenY, factor) {
  const rect = svg.getBoundingClientRect();
  const x = (screenX - rect.left) * VIEWBOX.width / Math.max(1, rect.width);
  const y = (screenY - rect.top) * VIEWBOX.height / Math.max(1, rect.height);
  const nextScale = Math.max(0.65, Math.min(4, mapViewport.scale * factor));
  const ratio = nextScale / mapViewport.scale;
  mapViewport.offsetX = x - (x - mapViewport.offsetX) * ratio;
  mapViewport.offsetY = y - (y - mapViewport.offsetY) * ratio;
  mapViewport.scale = nextScale;
  renderMap();
}

function zoomIn() {
  const rect = svg.getBoundingClientRect();
  zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.2);
}

function zoomOut() {
  const rect = svg.getBoundingClientRect();
  zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1 / 1.2);
}

function renderMap() {
  if (!coarseMap || !svg) return;

  window.dispatchEvent(new CustomEvent("go2:coarse-map", { detail: cloneMap(coarseMap) }));

  if (viewMode === "3d" || viewMode === "threejs") {
    render3DView();
    return;
  }

  const box = bounds(coarseMap);
  const grid = [];
  for (let x = 10; x < VIEWBOX.width; x += 25) grid.push(`<line x1="${x}" y1="0" x2="${x}" y2="${VIEWBOX.height}"/>`);
  for (let y = 10; y < VIEWBOX.height; y += 25) grid.push(`<line x1="0" y1="${y}" x2="${VIEWBOX.width}" y2="${y}"/>`);
  const entities = allEntities(coarseMap).map(({ kind, entity }) => renderEntity(kind, entity, box)).join("");
  const frontierMarkup = frontierCandidates.map((candidate, index) => {
    const point = candidate?.position || {};
    const [x, y] = project([finiteNumber(point.x), finiteNumber(point.y)], box);
    return `<g class="coarse-frontier-candidate" data-frontier-id="${escapeText(candidate.candidate_id)}"><circle cx="${x}" cy="${y}" r="8"/><text x="${x + 11}" y="${y + 4}">F${index + 1}</text></g>`;
  }).join("");
  const robotMarkup = robotPosition?.length === 2 ? (() => {
    const [x, y] = project(robotPosition, box);
    return `<g class="coarse-robot" transform="translate(${x} ${y})"><circle r="11"/><path d="M0-16 L6-4 L-6-4 Z"/><text x="14" y="4">Go2</text></g>`;
  })() : "";
  const rescanRouteMarkup = rescanPreviewTask ? (() => {
    const route = Array.isArray(rescanPreviewTask.waypoints) && rescanPreviewTask.waypoints.length >= 2
      ? rescanPreviewTask.waypoints
      : Array.isArray(rescanPreviewTask.room_polygon) ? rescanPreviewTask.room_polygon : [];
    if (route.length < 2) return "";
    const points = route.map((point) => project(point, box).join(",")).join(" ");
    const labelPoint = project(route[0], box);
    return `<g class="coarse-rescan-route"><polyline points="${points}"/><text x="${labelPoint[0] + 8}" y="${labelPoint[1] - 8}">RESCAN PREVIEW</text></g>`;
  })() : "";
  const draftMarkup = editor.draftPoints.length ? (() => {
    const points = editor.draftPoints.map((point) => project(point, box).join(",")).join(" ");
    const shape = editor.tool === "wall" ? `<polyline points="${points}" class="coarse-draft-line"/>` : `<polygon points="${points}" class="coarse-draft-polygon"/>`;
    const handles = editor.draftPoints.map((point) => {
      const [x, y] = project(point, box);
      return `<circle class="coarse-draft-handle" cx="${x}" cy="${y}" r="5"/>`;
    }).join("");
    return `<g class="coarse-draft">${shape}${handles}</g>`;
  })() : "";

  const hatchPatterns = `<defs>
    <pattern id="hatch-unseen" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="#83b8a7" stroke-width="0.8" opacity="0.25"/>
    </pattern>
    <pattern id="hatch-observed" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="#56a990" stroke-width="0.8" opacity="0.3"/>
    </pattern>
    <pattern id="hatch-verified" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(-45)">
      <line x1="0" y1="0" x2="0" y2="5" stroke="#6ea476" stroke-width="0.9" opacity="0.35"/>
    </pattern>
    <pattern id="hatch-quarantined" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="8" stroke="#cf625a" stroke-width="1.2" opacity="0.4"/>
    </pattern>
  </defs>`;

  svg.innerHTML = `<rect x="0" y="0" width="${VIEWBOX.width}" height="${VIEWBOX.height}" rx="12" class="coarse-map-background"/>${hatchPatterns}<g class="coarse-map-viewport" transform="translate(${mapViewport.offsetX} ${mapViewport.offsetY}) scale(${mapViewport.scale})"><g class="coarse-map-grid">${grid.join("")}</g>${entities}${frontierMarkup}${rescanRouteMarkup}${draftMarkup}${robotMarkup}</g>`;
}

function render3DView() {
  if (viewMode === "threejs") {
    import("./coarse-map-3d-threejs.js").then((module) => {
      const render = async () => {
        const floors = buildingIndex?.floors || [];
        if (floors.length > 1) {
          const maps = await Promise.all(floors.map(async (floor) => {
            if (String(floor.map_id) === String(mapId)) return coarseMap;
            const response = await fetch(`/api/maps/coarse/${encodeURIComponent(floor.map_id)}`, { cache: "no-store" });
            return response.ok ? toV2(await response.json()) : null;
          }));
          await module.renderBuildingMapsThreeJS(maps, threejsContainer, ({ kind, id }) => selectEntity(kind, id), mapId, transformEntityFromThree);
        } else {
          await module.renderMapThreeJS(coarseMap, threejsContainer, ({ kind, id }) => selectEntity(kind, id), { onTransformEntity: transformEntityFromThree });
        }
      };
      configureThreeTransformControls();
      render().catch((error) => {
        console.error("Failed to render Three.js view:", error);
        setEditorStatus("Three.js 渲染失败", "error");
      });
    }).catch((error) => {
      console.error("Failed to load Three.js renderer:", error);
      setEditorStatus("Three.js 模块加载失败", "error");
    });
  } else {
    // Isometric SVG fallback
    import("./coarse-map-3d.js").then((module) => {
      const geometry = module.renderMapIsometric(coarseMap);
      const svgContent = module.createIsometricSVG(geometry, mapViewport);
      svg3d.innerHTML = svgContent;
    }).catch((error) => {
      console.error("Failed to load 3D renderer:", error);
      setEditorStatus("3D 视图加载失败", "error");
    });
  }
}

function toggleViewMode() {
  // Cycle: 2d -> 3d (isometric SVG) -> threejs (Three.js WebGL) -> 2d
  const modes = ["2d", "3d", "threejs"];
  const currentIndex = modes.indexOf(viewMode);
  viewMode = modes[(currentIndex + 1) % modes.length];

  updateViewModeUI();
  renderMap();
}

function setViewMode(mode) {
  if (!["2d", "3d", "threejs"].includes(mode)) return;
  viewMode = mode;
  updateViewModeUI();
  renderMap();
}

function updateViewModeUI() {
  // Update radio button states
  [view2DButton, viewIsoButton, view3DButton].forEach(btn => {
    if (btn) btn.setAttribute("aria-checked", "false");
  });

  const show = (element, visible) => {
    if (!element) return;
    if (visible) element.removeAttribute("hidden");
    else element.setAttribute("hidden", "");
  };

  if (viewMode === "2d") {
    view2DButton?.setAttribute("aria-checked", "true");
    show(svg, true);
    show(svg3d, false);
    show(threejsContainer, false);
    show(threeTransformTools, false);
    document.querySelectorAll(".coarse-map-tools button").forEach((btn) => btn.disabled = false);
    setEditorStatus("2D 编辑模式：可绘制和编辑功能区。", "saved");
  } else if (viewMode === "3d") {
    viewIsoButton?.setAttribute("aria-checked", "true");
    show(svg, false);
    show(svg3d, true);
    show(threejsContainer, false);
    show(threeTransformTools, false);
    document.querySelectorAll(".coarse-map-tools button").forEach((btn) => btn.disabled = true);
    setEditorStatus("等轴视图：仅供预览，切换回 2D 可编辑。", "info");
  } else if (viewMode === "threejs") {
    view3DButton?.setAttribute("aria-checked", "true");
    show(svg, false);
    show(svg3d, false);
    show(threejsContainer, true);
    show(threeTransformTools, true);
    document.querySelectorAll(".coarse-map-tools button").forEach((btn) => btn.disabled = true);
    setEditorStatus("3D 透视：选择家具后可移动或旋转；拖动空白区域调整视角。", "info");
  }
}

// Handle window resize for Three.js
if (threejsContainer) {
  const resizeObserver = new ResizeObserver(() => {
    if (viewMode === "threejs" && !threejsContainer.hidden) {
      import("./coarse-map-3d-threejs.js").then((module) => {
        module.handleResize(threejsContainer);
      });
    }
  });
  resizeObserver.observe(threejsContainer);
}

function renderLists() {
  if (!coarseMap) return;
  const mapHeading = document.querySelector(".coarse-map-head h2");
  if (mapHeading) mapHeading.textContent = "大楼功能区";
  const spaces = [...(coarseMap.rooms || []), ...(coarseMap.corridors || [])];
  if (buildingIdLabel) buildingIdLabel.textContent = String(coarseMap.building_id || "building-1");
  if (floorIdLabel) floorIdLabel.textContent = String(coarseMap.floor_id || coarseMap.map_id || "当前楼层");
  if (spaceCountLabel) spaceCountLabel.textContent = `${spaces.length} 个功能区`;
  if (floorSelect && buildingIndex) {
    const floors = buildingIndex.floors || [];
    floorSelect.innerHTML = floors.map((floor) => `<option value="${escapeText(floor.map_id)}">${escapeText(floor.label || floor.floor_id)}</option>`).join("");
    floorSelect.value = mapId;
  }
  renderBuildingOverview();
  const tasks = (coarseMap.tasks || []).filter((task) => !selectedRoomId || task.room_id === selectedRoomId);
  roomList.innerHTML = spaces.map((room) => `<button type="button" class="coarse-room-row ${room.id === selectedRoomId ? "selected" : ""}" data-room-id="${escapeText(room.id)}"><span class="room-status-dot ${finiteNumber(room.coverage_percent) >= 80 ? "done" : "prior"}"></span><span><strong>${escapeText(room.name || room.id)}</strong><small>${escapeText(room.kind || "room")} / ${finiteNumber(room.coverage_percent).toFixed(0)}% verified</small></span><b>${finiteNumber(room.coverage_percent).toFixed(0)}%</b></button>`).join("") || `<p class="coarse-empty">从地图工具栏添加房间或走廊</p>`;
  taskList.innerHTML = tasks.map((task) => {
    const missionTask = currentMissionTaskFor(task.id);
    const routeAudit = missionTask ? routeAuditFor(missionTask.task_id) : null;
    const status = String(missionTask?.status || task.status || "PENDING");
    const mode = String(missionTask?.mode || task.mode || "ASSISTED");
    const leaseId = String(missionTask?.lease_id || task.lease_id || "");
    const revision = Number(missionTask?.revision || task.revision || 1);
    const lifecycle = status === "ACTIVE" ? "暂停" : status === "PAUSED" ? "恢复" : "开始";
    const taskCoverage = status === "QUARANTINED" ? "QUARANTINED" : status === "DONE" ? "VERIFIED" : status === "ACTIVE" || status === "PAUSED" ? "OBSERVED" : "UNSEEN";
    const missionNote = missionTask ? ` / mission ${escapeText(missionTask.status)} / ${escapeText(missionTask.mode)}` : "";
    const routeState = routeAuditLabel(routeAudit);
    const routeEligible = routeAudit?.navigation_eligible === true && routeAudit?.state === "READY_FOR_HUMAN_REVIEW";
    return `<div class="coarse-task-row ${status.toLowerCase()} coverage-${taskCoverage.toLowerCase()}" data-coverage-state="${taskCoverage}" data-task-id="${escapeText(task.id)}" data-task-status="${escapeText(status)}" data-task-mode="${escapeText(mode)}" data-task-lease-id="${escapeText(leaseId)}" data-task-revision="${revision}"><button type="button" class="task-main" data-task-action="lifecycle" ${["DONE", "QUARANTINED"].includes(status) ? "disabled" : ""}><span class="task-icon">${taskStatusIcon(status)}</span><span><strong>${escapeText(task.title)}</strong><small>${escapeText(task.kind)} / ${escapeText(mode)} / ${taskCoverage} / rev ${revision}${missionNote}</small></span><em>${escapeText(status)} · ${lifecycle}</em></button><div class="task-route-safety" data-route-state="${escapeText(routeState)}">ROUTE ${escapeText(routeState)}</div><div class="task-actions"><button type="button" class="task-go" data-task-action="go" ${routeEligible ? "" : "disabled"} title="${escapeText(routeEligible ? "Send this reviewed route to navigation" : "Route audit must pass before navigation")}">GO</button><button type="button" class="task-mode" data-task-action="mode">${escapeText(mode)}</button><button type="button" class="task-lease" data-task-action="lease">${escapeText(leaseId ? `LEASE ${leaseId}` : "LEASE NONE")}</button><button type="button" class="task-complete" data-task-action="complete" ${["DONE", "QUARANTINED"].includes(status) ? "disabled" : ""}>完成</button><button type="button" class="task-quarantine" data-task-action="quarantine" ${["DONE", "QUARANTINED"].includes(status) ? "disabled" : ""}>隔离</button></div></div>`;
  }).join("") || `<p class="coarse-empty">当前房间暂无验证任务</p>`;
  renderRouteAuditSummary(tasks[0]?.id || null);
  taskCount.textContent = `${(coarseMap.tasks || []).filter((task) => task.status !== "DONE").length} 待处理`;
  const selected = spaces.find((room) => room.id === selectedRoomId) || spaces[0];
  const active = (coarseMap.tasks || []).find((task) => (currentMissionTaskFor(task.id)?.status || task.status) === "ACTIVE");
  activeTask.textContent = active?.title || "选择一个验证任务";
  coverage.textContent = `${finiteNumber(selected?.coverage_percent).toFixed(0)}%`;
  authority.textContent = selected?.provenance === "VERIFIED" ? "VERIFIED" : "USER PRIOR";
  missionState.textContent = currentMission ? `${currentMission.mission_id} / ${currentMission.state}` : "UNCOMPILED";
  navigationState.textContent = currentNavigation ? `${currentNavigation.state} / ${currentNavigation.mode || "UNKNOWN"}` : "NAV_READY";
  roomList.querySelectorAll("[data-room-id]").forEach((button) => button.addEventListener("click", () => {
    selectedRoomId = button.dataset.roomId || null;
    const space = spaces.find((item) => item.id === selectedRoomId);
    selectEntity(space?.kind === "corridor" ? "corridor" : "room", selectedRoomId);
  }));
  taskList.querySelectorAll("[data-task-id]").forEach((row) => {
    const taskId = row.dataset.taskId;
    const status = row.dataset.taskStatus || "PENDING";
    const mode = row.dataset.taskMode || "ASSISTED";
    const leaseId = row.dataset.taskLeaseId || "";
    const revision = Number(row.dataset.taskRevision || "1");
    row.querySelector('[data-task-action="lifecycle"]')?.addEventListener("click", () => lifecycleTask(taskId, status));
    row.querySelector('[data-task-action="go"]')?.addEventListener("click", () => { void navigateTask(taskId); });
    row.querySelector('[data-task-action="mode"]')?.addEventListener("click", () => {
      const nextMode = TASK_MODE_SEQUENCE[(TASK_MODE_SEQUENCE.indexOf(mode) + 1 + TASK_MODE_SEQUENCE.length) % TASK_MODE_SEQUENCE.length];
      updateTaskMode(taskId, nextMode, revision, nextMode === "AUTONOMOUS" ? (leaseId || `lease-${taskId}`) : leaseId);
    });
    row.querySelector('[data-task-action="lease"]')?.addEventListener("click", () => updateTaskLease(taskId, leaseId ? "" : `lease-${taskId}`, revision));
    row.querySelector('[data-task-action="complete"]')?.addEventListener("click", () => completeTask(taskId, status));
    row.querySelector('[data-task-action="quarantine"]')?.addEventListener("click", () => quarantineTask(taskId));
  });
}

function renderBuildingOverview() {
  if (!buildingFloorCards || !buildingIndex) return;
  const floors = Array.isArray(buildingIndex.floors) ? buildingIndex.floors : [];
  const connections = Array.isArray(buildingIndex.connections) ? buildingIndex.connections : [];
  if (overviewSummary) overviewSummary.textContent = `${floors.length} 个楼层 · ${connections.length} 个连接`;
  buildingFloorCards.innerHTML = floors.map((floor, index) => {
    const active = String(floor.map_id) === String(mapId);
    const links = connections.filter((item) => item.from_floor_id === floor.floor_id || item.to_floor_id === floor.floor_id).length;
    return `<button type="button" class="coarse-floor-card${active ? " active" : ""}" data-overview-floor="${escapeText(floor.map_id)}"><span class="coarse-floor-level">${index + 1}</span><span><strong>${escapeText(floor.label || floor.floor_id)}</strong><small>${escapeText(floor.floor_id)} · REV ${Number(floor.revision || 1)} · ${links} 个连接</small></span><b>${active ? "当前" : "进入"}</b></button>`;
  }).join("") || `<span class="coarse-overview-empty">先新增一个楼层开始整栋楼粗建。</span>`;
  buildingConnections.innerHTML = connections.map((connection) => `<div class="coarse-connection-row"><span class="coarse-connection-mark">${connection.kind === "elevator" ? "↕" : "⇅"}</span><span><strong>${escapeText(connection.label || connection.id)}</strong><small>${escapeText(connection.from_floor_id)} → ${escapeText(connection.to_floor_id)}${connection.stair_profile ? ` · ${Number(connection.stair_profile.step_count)} 级` : ""}</small></span></div>`).join("") || `<span class="coarse-overview-empty">暂无跨层连接</span>`;
  buildingConnections.querySelectorAll(".coarse-connection-row").forEach((row, index) => {
    const connection = connections[index];
    if (!connection) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "coarse-connection-delete";
    button.title = "删除跨层连接";
    button.textContent = "删除";
    button.addEventListener("click", () => { void deleteBuildingConnection(connection.id); });
    row.appendChild(button);
  });
  buildingFloorCards.querySelectorAll("[data-overview-floor]").forEach((button) => button.addEventListener("click", () => { void switchFloor(button.dataset.overviewFloor); }));
}

async function deleteBuildingConnection(connectionId) {
  if (!connectionId || !buildingIndex || !window.confirm("删除这条跨层连接？楼层地图不会被修改。")) return;
  const buildingId = String(buildingIndex.building_id || serverMap?.building_id || "building-1");
  const next = { ...buildingIndex, connections: (buildingIndex.connections || []).filter((connection) => connection.id !== connectionId) };
  try {
    const response = await fetch(`/api/buildings/coarse/${encodeURIComponent(buildingId)}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(next) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    buildingIndex = payload;
    renderLists();
    setEditorStatus("跨层连接已删除，楼层地图保持不变。", "saved");
  } catch (error) { setEditorStatus(`删除跨层连接失败：${error.message}`, "error"); }
}

async function compileBuildingMission() {
  const buildingId = String(buildingIndex?.building_id || serverMap?.building_id || "building-1");
  if (buildingMissionButton) buildingMissionButton.disabled = true;
  try {
    const response = await fetch(`/api/buildings/coarse/${encodeURIComponent(buildingId)}/mission`, { method: "POST" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    currentMission = payload;
    setEditorStatus(`整栋楼任务已编译：${payload.task_count || 0} 项，未自动激活。`, "saved");
    renderLists();
    void refreshMissionRouteAudits();
  } catch (error) {
    setEditorStatus(`整栋楼任务编译失败：${error.message}`, "error");
  } finally {
    if (buildingMissionButton) buildingMissionButton.disabled = false;
  }
}

async function loadBuildingIndex() {
  try {
    const response = await fetch(`/api/buildings/coarse/${encodeURIComponent(serverMap?.building_id || "building-1")}`, { cache: "no-store" });
    if (!response.ok) return;
    buildingIndex = await response.json();
    renderLists();
    if (viewMode === "threejs") renderMap();
  } catch (_error) {
    buildingIndex = null;
  }
}

async function switchFloor(nextMapId) {
  if (!nextMapId || nextMapId === mapId) return;
  if (workspaceMode === "coarse" && draftDirty && !window.confirm("当前楼层有未保存草稿，切换楼层将保留草稿但离开当前编辑。继续吗？")) {
    floorSelect.value = mapId;
    return;
  }
  try {
    const response = await fetch(`/api/maps/coarse/${encodeURIComponent(nextMapId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error("楼层地图不可用");
    mapId = nextMapId;
    serverMap = toV2(await response.json());
    draftStorageKey = draftStorageKeyForMap(mapId);
    draftInitialized = false;
    draftDirty = false;
    selectedEntity = null;
    selectedRoomId = null;
    if (workspaceMode === "coarse") activateCoarseDraft();
    else {
      coarseMap = cloneMap(serverMap);
      renderMap();
      renderLists();
      showSelectedEntity();
    }
    setEditorStatus(`已切换到 ${serverMap.floor_id || mapId}`, "saved");
  } catch (error) {
    setEditorStatus(`切换楼层失败：${error.message}`, "error");
    floorSelect.value = mapId;
  }
}

async function createFloor() {
  const buildingId = String(serverMap?.building_id || buildingIndex?.building_id || "building-1");
  const floorId = floorFormId?.value.trim();
  const label = floorFormLabel?.value.trim() || floorId;
  if (!floorId || !floorForm?.reportValidity()) return;
  if (floorFormStatus) floorFormStatus.textContent = "正在创建楼层...";
  try {
    floorAddButton.disabled = true;
    const response = await fetch(`/api/buildings/coarse/${encodeURIComponent(buildingId)}/floors`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ floor_id: floorId, label }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    buildingIndex = payload.building;
    renderLists();
    await switchFloor(payload.map.map_id);
    setEditorStatus(`已创建 ${label}，当前为独立空白楼层草稿`, "saved");
    floorDialog?.close();
    floorForm?.reset();
  } catch (error) {
    setEditorStatus(`新建楼层失败：${error.message}`, "error");
    if (floorFormStatus) floorFormStatus.textContent = error.message;
  } finally {
    floorAddButton.disabled = false;
  }
}

function populateConnectionFloors() {
  const floors = buildingIndex?.floors || [];
  const options = floors.map((floor) => `<option value="${escapeText(floor.floor_id)}">${escapeText(floor.label || floor.floor_id)}</option>`).join("");
  if (connectionFrom) connectionFrom.innerHTML = options;
  if (connectionTo) connectionTo.innerHTML = options;
  if (connectionTo && floors.length > 1) connectionTo.value = floors[1].floor_id;
}

async function saveConnection() {
  const buildingId = String(buildingIndex?.building_id || serverMap?.building_id || "building-1");
  if (!connectionFrom?.value || !connectionTo?.value || connectionFrom.value === connectionTo.value) {
    if (connectionStatus) connectionStatus.textContent = "请选择两个不同楼层。";
    return;
  }
  const connection = { id: `vertical-${Date.now()}`, kind: connectionKind?.value || "stairs", from_floor_id: connectionFrom.value, to_floor_id: connectionTo.value, label: connectionLabel?.value.trim() || (connectionKind?.value === "elevator" ? "电梯" : "楼梯") };
  const next = { ...(buildingIndex || {}), building_id: buildingId, connections: [...(buildingIndex?.connections || []), connection] };
  try {
    const response = await fetch(`/api/buildings/coarse/${encodeURIComponent(buildingId)}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(next) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    buildingIndex = payload;
    connectionDialog?.close();
    setEditorStatus("跨层楼梯/电梯连接已保存到楼栋索引。", "saved");
  } catch (error) {
    if (connectionStatus) connectionStatus.textContent = error.message;
  }
}

function setEditorStatus(message, state = "") {
  if (!editorStatus) return;
  editorStatus.textContent = message;
  editorStatus.dataset.state = state;
}

function setTool(tool) {
  if (!ENTITY_CONFIG[tool] && tool !== "select") return;
  if (editor.tool !== tool && editor.draftPoints.length) {
    setEditorStatus("当前绘制尚未完成，请先按 Enter 完成或按 Esc 取消。", "warning");
    return;
  }
  editor.tool = tool;
  editor.draftPoints = [];
  editor.dragging = null;
  lastDrawClick = null;
  document.querySelectorAll("[data-map-tool]").forEach((button) => button.setAttribute("aria-pressed", button.dataset.mapTool === tool ? "true" : "false"));
  document.querySelectorAll(".coarse-tools-more").forEach((menu) => {
    menu.dataset.active = String(Boolean(menu.querySelector(`[data-map-tool="${tool}"]`)));
  });
  finishButton.disabled = tool === "select";
  cancelButton.disabled = tool === "select";
  setEditorStatus(tool === "select" ? "选择实体，或从工具栏开始绘制。" : `${ENTITY_CONFIG[tool].label}：在地图上点击添加点，完成后提交。`, tool === "select" ? "idle" : "editing");
  renderMap();
}

function selectedEntityRecord() {
  if (!selectedEntity) return null;
  const entity = findEntity(selectedEntity.kind, selectedEntity.id);
  return entity ? { ...selectedEntity, entity } : null;
}

function showSelectedEntity() {
  const selected = selectedEntityRecord();
  entityForm.hidden = !selected;
  if (!selected) return;
  entityNameInput.value = entityLabel(selected.kind, selected.entity);
  if (entityLockedInput) entityLockedInput.checked = selected.entity.locked === true;
  const kindEditor = ensureEntityKindEditor();
  if (kindEditor) {
    kindEditor.value = selected.entity.kind || selected.kind;
    kindEditor.disabled = !["room", "corridor", "equipment", "obstacle", "inspection_target"].includes(selected.kind);
  }
  const dimensions = selected.entity.dimensions_m;
  const isFurniture = selected.kind === "equipment" && Array.isArray(dimensions) && dimensions.length === 3;
  if (furnitureFields) furnitureFields.hidden = !isFurniture;
  const isStair = selected.entity.kind === "stair";
  if (stairFields) stairFields.hidden = !isStair;
  if (isStair) {
    const profile = selected.entity.stair_profile || {};
    stairStepCount.value = profile.step_count || 8;
    stairRiser.value = profile.riser_height_m || 0.18;
    stairTread.value = profile.tread_depth_m || 0.28;
    stairWidth.value = profile.width_m || 1.0;
    stairDirection.value = profile.direction || "bidirectional";
    stairEntryX.value = selected.entity.entry?.[0] ?? "";
    stairEntryY.value = selected.entity.entry?.[1] ?? "";
    stairExitX.value = selected.entity.exit?.[0] ?? "";
    stairExitY.value = selected.entity.exit?.[1] ?? "";
  }
  if (isFurniture) {
    furnitureWidth.value = dimensions[0];
    furnitureDepth.value = dimensions[1];
    furnitureHeight.value = dimensions[2];
    furnitureRotation.value = (Number(selected.entity.rotation_rad || 0) * 180 / Math.PI).toFixed(0);
  }
}

function selectEntity(kind, id) {
  const additive = arguments.length > 2 && arguments[2] === true;
  if (!id) {
    selectedEntity = null;
    selectedEntityIds = [];
  } else if (additive && ["room", "corridor", "stair", "landing", "elevator_hall"].includes(kind)) {
    selectedEntityIds = selectedEntityIds.includes(id)
      ? selectedEntityIds.filter((value) => value !== id)
      : [...selectedEntityIds, id];
    selectedEntity = { kind, id };
  } else {
    selectedEntity = { kind, id };
    selectedEntityIds = [id];
  }
  if (["room", "corridor", "stair", "landing", "elevator_hall"].includes(kind)) selectedRoomId = id;
  showSelectedEntity();
  setEditorStatus(selectedEntity ? `${ENTITY_CONFIG[kind]?.label || kind} 已选择，可拖动顶点或修改名称。` : "选择实体，或从工具栏开始绘制。", "selected");
  renderMap();
  renderLists();
}

function transformEntityFromThree({ kind, id, position, rotation_rad }) {
  if (kind !== "equipment" || !Array.isArray(position) || position.length !== 2) return;
  const entity = findEntity(kind, id);
  if (!entity || entity.locked === true) return;
  captureDraftHistory();
  entity.position = [Number(position[0]), Number(position[1])];
  if (Number.isFinite(Number(rotation_rad))) entity.rotation_rad = Number(rotation_rad);
  selectEntity(kind, id);
  markDraftChanged("3D 家具位置和朝向已更新，点击保存 revision 写入服务端。");
  renderMap();
}

let threeTransformMode = "translate";
let threeTransformSpaceMode = "world";

async function configureThreeTransformControls() {
  const module = await import("./coarse-map-3d-threejs.js");
  module.setThreeTransformMode(threeTransformMode);
  module.setThreeTransformSpace(threeTransformSpaceMode);
  module.setThreeTransformSnap(threeTransformSnap?.checked !== false);
}

function setThreeTransformMode(mode) {
  threeTransformMode = mode === "rotate" ? "rotate" : "translate";
  threeTransformTranslate?.setAttribute("aria-pressed", String(threeTransformMode === "translate"));
  threeTransformRotate?.setAttribute("aria-pressed", String(threeTransformMode === "rotate"));
  void configureThreeTransformControls();
}

function setThreeTransformSpace() {
  threeTransformSpaceMode = threeTransformSpaceMode === "world" ? "local" : "world";
  if (threeTransformSpace) {
    threeTransformSpace.textContent = threeTransformSpaceMode === "world" ? "世界" : "局部";
    threeTransformSpace.setAttribute("aria-pressed", String(threeTransformSpaceMode === "local"));
  }
  void configureThreeTransformControls();
}

function setThreeTransformSnap() { void configureThreeTransformControls(); }

function rectangleBounds(points) {
  if (!Array.isArray(points) || points.length !== 4) return null;
  const xs = [...new Set(points.map((point) => Number(point[0])))].sort((a, b) => a - b);
  const ys = [...new Set(points.map((point) => Number(point[1])))].sort((a, b) => a - b);
  if (xs.length !== 2 || ys.length !== 2) return null;
  const expected = new Set([[xs[0], ys[0]], [xs[1], ys[0]], [xs[1], ys[1]], [xs[0], ys[1]].map(String)]);
  const actual = new Set(points.map((point) => `${Number(point[0])},${Number(point[1])}`));
  if (actual.size !== 4 || ![`${xs[0]},${ys[0]}`, `${xs[1]},${ys[0]}`, `${xs[1]},${ys[1]}`, `${xs[0]},${ys[1]}`].every((value) => actual.has(value))) return null;
  return { minX: xs[0], minY: ys[0], maxX: xs[1], maxY: ys[1] };
}

function splitSelectedZone(direction = "vertical") {
  const selected = selectedEntityRecord();
  if (!selected || !["room", "corridor"].includes(selected.kind)) {
    setEditorStatus("请先选择一个功能区再分割。", "warning"); return;
  }
  const polygon = selected.entity.polygon;
  if (!Array.isArray(polygon) || polygon.length < 3) { setEditorStatus("当前功能区没有有效多边形。", "error"); return; }
  const axis = direction === "horizontal" ? 1 : 0;
  const values = polygon.map((point) => Number(point[axis]));
  const middle = (Math.min(...values) + Math.max(...values)) / 2;
  const clip = (keepLess) => {
    const result = [];
    polygon.forEach((current, index) => {
      const previous = polygon[(index + polygon.length - 1) % polygon.length];
      const currentValue = current[axis] - middle;
      const previousValue = previous[axis] - middle;
      const currentInside = keepLess ? currentValue <= 1e-9 : currentValue >= -1e-9;
      const previousInside = keepLess ? previousValue <= 1e-9 : previousValue >= -1e-9;
      if (currentInside !== previousInside) {
        const ratio = Math.abs(previousValue - currentValue) > 1e-12 ? previousValue / (previousValue - currentValue) : 0;
        result.push([previous[0] + ratio * (current[0] - previous[0]), previous[1] + ratio * (current[1] - previous[1])]);
      }
      if (currentInside) result.push([...current]);
    });
    return result.filter((point, index) => index === 0 || Math.hypot(point[0] - result[index - 1][0], point[1] - result[index - 1][1]) > 1e-9);
  };
  const polygons = [clip(true), clip(false)];
  if (polygons.some((item) => item.length < 3)) { setEditorStatus("当前多边形无法沿中心线分割。", "error"); return; }
  captureDraftHistory();
  const collection = entityCollection(selected.kind);
  coarseMap[collection] = (coarseMap[collection] || []).filter((item) => item.id !== selected.id).concat(polygons.map((polygon, index) => ({ ...selected.entity, id: `${selected.id}-${index ? "b" : "a"}`, name: `${entityLabel(selected.kind, selected.entity)} ${index ? "B" : "A"}`, polygon, coverage_percent: 0 })));
  selectedEntityIds = [`${selected.id}-a`, `${selected.id}-b`];
  selectedEntity = { kind: selected.kind, id: `${selected.id}-a` };
  markDraftChanged("功能区已分割，保存 revision 后写入服务器。"); renderMap(); renderLists(); showSelectedEntity();
}

function mergePolygonsOnSharedEdge(first, second) {
  if (!Array.isArray(first) || !Array.isArray(second) || first.length < 3 || second.length < 3) return null;
  const same = (a, b) => Math.hypot(Number(a[0]) - Number(b[0]), Number(a[1]) - Number(b[1])) <= 1e-8;
  let shared = null;
  first.forEach((start, i) => {
    if (shared) return;
    const end = first[(i + 1) % first.length];
    second.forEach((otherStart, j) => {
      const otherEnd = second[(j + 1) % second.length];
      if (!shared && same(start, otherEnd) && same(end, otherStart)) shared = [i, j];
    });
  });
  if (!shared) return null;
  const edges = [];
  [first, second].forEach((polygon, polygonIndex) => polygon.forEach((start, index) => {
    if (index === shared[polygonIndex]) return;
    edges.push([[...start], [...polygon[(index + 1) % polygon.length]]]);
  }));
  if (!edges.length) return null;
  const merged = [edges[0][0]];
  let current = edges[0][1];
  const remaining = edges.slice(1);
  while (remaining.length && !same(current, merged[0])) {
    merged.push(current);
    const match = remaining.findIndex((edge) => same(edge[0], current));
    if (match < 0) return null;
    current = remaining.splice(match, 1)[0][1];
  }
  if (!same(current, merged[0]) || merged.length < 3) return null;
  return merged.filter((point, index) => index === 0 || !same(point, merged[index - 1]));
}

function mergeSelectedZones() {
  const ids = selectedEntityIds.filter((id) => coarseMap?.rooms?.some((room) => room.id === id));
  if (ids.length !== 2) { setEditorStatus("请按住 Shift 选择两个功能区再合并。", "warning"); return; }
  const first = coarseMap.rooms.find((room) => room.id === ids[0]);
  const second = coarseMap.rooms.find((room) => room.id === ids[1]);
  const mergedFreeform = mergePolygonsOnSharedEdge(first?.polygon, second?.polygon);
  if (mergedFreeform) {
    captureDraftHistory();
    const merged = { ...first, id: `${first.id}-merged`, name: `${entityLabel("room", first)} + ${entityLabel("room", second)}`, polygon: mergedFreeform, coverage_percent: 0 };
    coarseMap.rooms = coarseMap.rooms.filter((room) => !ids.includes(room.id)).concat(merged);
    selectedEntityIds = [merged.id]; selectedEntity = { kind: "room", id: merged.id };
    markDraftChanged("功能区已合并，保存 revision 后写入服务端。"); renderMap(); renderLists(); showSelectedEntity();
    return;
  }
  const a = rectangleBounds(first?.polygon); const b = rectangleBounds(second?.polygon);
  if (!a || !b) { setEditorStatus("只有相邻正交矩形功能区可以自动合并。", "error"); return; }
  const horizontal = a.minY === b.minY && a.maxY === b.maxY && (a.maxX === b.minX || b.maxX === a.minX);
  const vertical = a.minX === b.minX && a.maxX === b.maxX && (a.maxY === b.minY || b.maxY === a.minY);
  if (!horizontal && !vertical) { setEditorStatus("两个功能区必须共享完整边才能合并。", "error"); return; }
  captureDraftHistory();
  const polygon = horizontal ? [[Math.min(a.minX, b.minX), a.minY], [Math.max(a.maxX, b.maxX), a.minY], [Math.max(a.maxX, b.maxX), a.maxY], [Math.min(a.minX, b.minX), a.maxY]] : [[a.minX, Math.min(a.minY, b.minY)], [a.maxX, Math.min(a.minY, b.minY)], [a.maxX, Math.max(a.maxY, b.maxY)], [a.minX, Math.max(a.maxY, b.maxY)]];
  const merged = { ...first, id: `${first.id}-merged`, name: `${entityLabel("room", first)} + ${entityLabel("room", second)}`, polygon, coverage_percent: 0 };
  coarseMap.rooms = coarseMap.rooms.filter((room) => !ids.includes(room.id)).concat(merged);
  selectedEntityIds = [merged.id]; selectedEntity = { kind: "room", id: merged.id };
  markDraftChanged("功能区已合并，保存 revision 后写入服务器。"); renderMap(); renderLists(); showSelectedEntity();
}

function nextEntityId(kind) {
  const ids = new Set((coarseMap?.[entityCollection(kind)] || []).map((entity) => String(entity.id)));
  const prefix = kind === "keep_out_zone" ? "keep-out" : kind;
  let index = 1;
  while (ids.has(`${prefix}-${index}`)) index += 1;
  return `${prefix}-${index}`;
}

async function saveMapRevision() {
  if (!coarseMap || editor.saving || workspaceMode !== "coarse" || !draftDirty) return false;
  if (!(coarseMap.rooms?.length || coarseMap.corridors?.length)) {
    setEditorStatus("至少需要一个房间或走廊后才能保存 revision。", "error");
    return false;
  }
  editor.saving = true;
  setEditorStatus(`保存地图 revision ${Number(coarseMap.revision || 1) + 1}...`, "saving");
  try {
    const response = await fetch(`/api/maps/coarse/${encodeURIComponent(mapId)}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...toV2(coarseMap), expected_revision: Number(draftBaseRevision || 1) }),
    });
    if (response.status === 409) {
      draftConflictPayload = await response.json().catch(() => ({}));
      draftConflict.hidden = false;
      draftConflict.dataset.state = "conflict";
      reloadServerButton.hidden = false;
      keepDraftButton.hidden = false;
      throw new Error("服务器地图已更新；本地草稿已保留。请选择加载服务器版或继续本地草稿。");
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }
    serverMap = toV2(await response.json());
    coarseMap = cloneMap(serverMap);
    draftMap = cloneMap(coarseMap);
    draftBaseRevision = Number(serverMap.revision || 1);
    draftDirty = false;
    draftConflictPayload = null;
    draftConflict.hidden = true;
    draftConflict.dataset.state = "hidden";
    reloadServerButton.hidden = true;
    keepDraftButton.hidden = true;
    try { localStorage.removeItem(draftStorageKey); } catch (_error) { /* storage may be unavailable */ }
    updateDraftControls();
    setEditorStatus(`已保存 USER_PRIOR / revision ${coarseMap.revision}`, "saved");
    renderMap();
    renderLists();
    await autoCompileMission();
    return true;
  } catch (error) {
    setEditorStatus(`保存失败：${error.message}`, "error");
    return false;
  } finally {
    editor.saving = false;
  }
}

const saveCanonicalMap = saveMapRevision;

async function commitDraft() {
  if (draftCommitInFlight) return;
  if (!coarseMap) {
    setEditorStatus("地图仍在加载，请稍后再提交。", "warning");
    return;
  }
  draftCommitInFlight = true;
  const kind = editor.tool;
  const config = ENTITY_CONFIG[kind];
  const minimum = MIN_DRAW_POINTS[kind] || 1;
  if (!config || editor.draftPoints.length < minimum) {
    setEditorStatus(`至少需要 ${minimum} 个点。`, "error");
    draftCommitInFlight = false;
    return;
  }
  const id = nextEntityId(kind);
  const label = `${config.label} ${id.split("-").pop()}`;
  const entity = { id, name: label, label, kind, coordinate_frame: coarseMap.coordinate_frame || "BUILDING_LOCAL", provenance: "USER_PRIOR", confidence: 0.35 };
  if (config.geometry === "point") entity.position = [...editor.draftPoints[0]];
  else if (config.geometry === "line") entity.geometry = editor.draftPoints.map((point) => [...point]);
  else entity.polygon = editor.draftPoints.map((point) => [...point]);
  if (["room", "corridor", "stair", "landing", "elevator_hall"].includes(kind)) {
    entity.coverage_percent = 0;
    selectedRoomId = id;
    if (kind === "stair") {
      entity.stair_profile = { step_count: 8, riser_height_m: 0.18, tread_depth_m: 0.28, width_m: 1.0, direction: "bidirectional" };
      entity.entry = [...entity.polygon[0]];
      entity.exit = [...entity.polygon[Math.floor(entity.polygon.length / 2)]];
    }
  } else if (["equipment", "inspection_target"].includes(kind)) {
    entity.room_id = selectedRoomId;
    if (kind === "equipment" && pendingFurnitureType && furnitureCatalog[pendingFurnitureType]) {
      const preset = furnitureCatalog[pendingFurnitureType];
      entity.kind = "equipment";
      entity.name = preset.label;
      entity.label = preset.label;
      entity.furniture_type = pendingFurnitureType;
      entity.asset_id = preset.asset_id;
      entity.asset_source = preset.asset_source || "PROCEDURAL_FALLBACK";
      if (preset.asset_url) entity.asset_url = preset.asset_url;
      entity.dimensions_m = [...preset.dimensions_m];
      entity.rotation_rad = 0;
      pendingFurnitureType = null;
    }
  }
  const collection = entityCollection(kind);
  captureDraftHistory();
  coarseMap[collection] = [...(coarseMap[collection] || []), entity];
  editor.draftPoints = [];
  selectedEntity = { kind, id };
  setTool("select");
  markDraftChanged("对象已加入草稿；点击保存 revision 写入服务器。");
  showSelectedEntity();
  draftCommitInFlight = false;
}

function cancelDraft() {
  editor.draftPoints = [];
  editor.dragging = null;
  setTool("select");
  setEditorStatus("当前绘制已取消；草稿中的已完成对象仍保留。", "idle");
}

function setSvgPointerCapture(pointerId) {
  try { svg.setPointerCapture?.(pointerId); } catch (_error) { /* synthetic events may have no active pointer */ }
}

function releaseSvgPointerCapture(pointerId) {
  try { svg.releasePointerCapture?.(pointerId); } catch (_error) { /* synthetic events may have no active pointer */ }
}

async function handleSvgPointerDown(event) {
  const handle = event.target.closest?.("[data-vertex-index]");
  const entityNode = event.target.closest?.("[data-entity-kind]");
  if (editor.tool === "select") {
    if (handle) {
      captureDraftHistory();
      editor.dragging = { kind: handle.dataset.entityKind, id: handle.dataset.entityId, vertexIndex: Number(handle.dataset.vertexIndex) };
      setSvgPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (entityNode) {
      selectEntity(entityNode.dataset.entityKind, entityNode.dataset.entityId, event.shiftKey);
      const entity = findEntity(entityNode.dataset.entityKind, entityNode.dataset.entityId);
      if (entityNode.dataset.entityKind === "equipment" && entity?.locked !== true) {
        captureDraftHistory();
        editor.dragging = { kind: "equipment", id: entityNode.dataset.entityId, point: true };
        setSvgPointerCapture(event.pointerId);
        event.preventDefault();
      }
      return;
    }
    mapViewport.dragging = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, offsetX: mapViewport.offsetX, offsetY: mapViewport.offsetY, moved: false };
    setSvgPointerCapture(event.pointerId);
    return;
  }
  const point = clientToMapPoint(event);
  const now = performance.now();
  // Detect the dblclick gesture here so SVG redraws cannot detach a second handler.
  const isDoubleClick = lastDrawClick
    && now - lastDrawClick.time <= 350
    && Math.hypot(point[0] - lastDrawClick.point[0], point[1] - lastDrawClick.point[1]) <= 0.01;
  editor.draftPoints.push(point);
  if (isDoubleClick) {
    editor.draftPoints.pop();
    lastDrawClick = null;
    void commitDraft();
    return;
  }
  lastDrawClick = { point, time: now };
  markDraftChanged(`当前绘制：${editor.draftPoints.length} 个点；双击或按 Enter 完成。`);
  if (ENTITY_CONFIG[editor.tool].geometry === "point") await commitDraft();
  else {
    setEditorStatus(`${ENTITY_CONFIG[editor.tool].label} 已添加 ${editor.draftPoints.length} 个点。`, "editing");
    renderMap();
  }
}

function updateDraggedVertex(point) {
  if (!editor.dragging) return;
  const { kind, id, vertexIndex } = editor.dragging;
  const entity = findEntity(kind, id);
  const config = ENTITY_CONFIG[kind];
  if (!entity || !config) return;
  if (editor.dragging.point) {
    entity.position = [...point];
    markDraftChanged("家具位置已更新，点击保存 revision 写入服务端。");
    renderMap();
    return;
  }
  if (config.geometry === "point") return;
  const geometry = entityGeometry(kind, entity);
  if (!geometry[vertexIndex]) return;
  geometry[vertexIndex][0] = point[0];
  geometry[vertexIndex][1] = point[1];
  if (config.geometry === "line") entity.geometry = geometry;
  else entity.polygon = geometry;
  markDraftChanged("顶点已修改；点击保存 revision 写入服务器。");
  setEditorStatus(`编辑 ${ENTITY_CONFIG[kind].label} 顶点 ${vertexIndex + 1}，松开后保存。`, "editing");
  renderMap();
}

function updateMapPan(event) {
  const drag = mapViewport.dragging;
  if (!drag || drag.pointerId !== event.pointerId) return;
  const rect = svg.getBoundingClientRect();
  const deltaX = (event.clientX - drag.clientX) * VIEWBOX.width / Math.max(1, rect.width);
  const deltaY = (event.clientY - drag.clientY) * VIEWBOX.height / Math.max(1, rect.height);
  drag.moved = drag.moved || Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2;
  mapViewport.offsetX = drag.offsetX + deltaX;
  mapViewport.offsetY = drag.offsetY + deltaY;
  renderMap();
}

async function handleSvgPointerUp(event) {
  if (mapViewport.dragging?.pointerId === event.pointerId) {
    const moved = mapViewport.dragging.moved;
    releaseSvgPointerCapture(event.pointerId);
    mapViewport.dragging = null;
    if (!moved) selectEntity("room", null);
    return;
  }
  if (!editor.dragging) return;
  releaseSvgPointerCapture(event.pointerId);
  editor.dragging = null;
}

async function updateEntity(event) {
  event.preventDefault();
  const selected = selectedEntityRecord();
  const value = entityNameInput.value.trim();
  if (!selected || !value) return;
  captureDraftHistory();
  if (["room", "corridor"].includes(selected.kind)) selected.entity.name = value;
  else selected.entity.label = value;
  const kindEditor = byId("coarse-entity-kind");
  if (kindEditor && selected.kind === "room" && ["room", "landing", "stair", "elevator_hall"].includes(kindEditor.value)) {
    selected.entity.kind = kindEditor.value;
  }
  if (kindEditor && selected.kind === "corridor") selected.entity.kind = "corridor";
  if (kindEditor && ["equipment", "obstacle", "inspection_target"].includes(selected.kind)) selected.entity.kind = kindEditor.value;
  if (selected.kind === "equipment" && Array.isArray(selected.entity.dimensions_m)) {
    const values = [furnitureWidth, furnitureDepth, furnitureHeight].map((input) => Number(input?.value));
    if (values.every((value) => Number.isFinite(value) && value > 0)) selected.entity.dimensions_m = values;
    const degrees = Number(furnitureRotation?.value);
    if (Number.isFinite(degrees)) selected.entity.rotation_rad = degrees * Math.PI / 180;
  }
  selected.entity.locked = entityLockedInput?.checked === true;
  if (selected.entity.kind === "stair") {
    const profile = {
      step_count: Number(stairStepCount?.value),
      riser_height_m: Number(stairRiser?.value),
      tread_depth_m: Number(stairTread?.value),
      width_m: Number(stairWidth?.value),
      direction: stairDirection?.value || "bidirectional",
    };
    if (!Number.isInteger(profile.step_count) || profile.step_count < 1 || profile.step_count > 500
      || ![profile.riser_height_m, profile.tread_depth_m, profile.width_m].every((value) => Number.isFinite(value) && value > 0)) {
      setEditorStatus("楼梯参数无效，请检查台阶数和尺寸。", "error");
      return;
    }
    selected.entity.stair_profile = profile;
    const entry = [Number(stairEntryX?.value), Number(stairEntryY?.value)];
    const exitPoint = [Number(stairExitX?.value), Number(stairExitY?.value)];
    if (entry.every(Number.isFinite)) selected.entity.entry = entry; else delete selected.entity.entry;
    if (exitPoint.every(Number.isFinite)) selected.entity.exit = exitPoint; else delete selected.entity.exit;
  }
  markDraftChanged("属性已修改；点击保存 revision 写入服务器。");
  showSelectedEntity();
}

async function deleteSelectedEntity() {
  const selected = selectedEntityRecord();
  if (!selected) return;
  captureDraftHistory();
  const collection = entityCollection(selected.kind);
  coarseMap[collection] = (coarseMap[collection] || []).filter((entity) => entity.id !== selected.id);
  if (selectedRoomId === selected.id) selectedRoomId = coarseMap.rooms?.[0]?.id || coarseMap.corridors?.[0]?.id || null;
  selectedEntity = null;
  entityForm.hidden = true;
  markDraftChanged("对象已删除；点击保存 revision 写入服务器。");
}

async function coarseTaskRequest(taskId, suffix = "", options = {}) {
  const canonicalUrl = `/api/maps/coarse/${encodeURIComponent(mapId)}/tasks/${encodeURIComponent(taskId)}${suffix}`;
  const legacyUrl = `/api/coarse-map/tasks/${encodeURIComponent(taskId)}${suffix}`;
  const response = await fetch(canonicalUrl, options);
  if (response.status === 404 || response.status === 405) return fetch(legacyUrl, options);
  return response;
}

async function loadMap() {
  try {
    const preferredCanonicalResponse = await fetch(`/api/maps/coarse/${encodeURIComponent(DEFAULT_MAP_ID)}`, { cache: "no-store" });
    if (preferredCanonicalResponse.ok) {
      const loadedMap = toV2(await preferredCanonicalResponse.json());
      mapId = String(loadedMap.map_id || DEFAULT_MAP_ID);
      serverMap = cloneMap(loadedMap);
      draftStorageKey = draftStorageKeyForMap(mapId);
      if (workspaceMode === "coarse") {
        if (!draftInitialized) activateCoarseDraft();
        return;
      }
      coarseMap = loadedMap;
      if (!selectedRoomId || !findEntity("room", selectedRoomId)) selectedRoomId = coarseMap.rooms?.[0]?.id || coarseMap.corridors?.[0]?.id || null;
      renderMap();
      renderLists();
      showSelectedEntity();
      return;
    }
    const seedResponse = await fetch("/api/coarse-map", { cache: "no-store" });
    if (!seedResponse.ok) throw new Error("粗地图服务不可用");
    const seed = await seedResponse.json();
    mapId = String(seed.map_id || DEFAULT_MAP_ID);
    const canonicalResponse = await fetch(`/api/maps/coarse/${encodeURIComponent(mapId)}`, { cache: "no-store" });
    const loadedMap = toV2(canonicalResponse.ok ? await canonicalResponse.json() : seed);
    serverMap = cloneMap(loadedMap);
    draftStorageKey = draftStorageKeyForMap(mapId);
    if (workspaceMode === "coarse") {
      if (!draftInitialized) activateCoarseDraft();
      return;
    }
    coarseMap = loadedMap;
    if (!selectedRoomId || !findEntity("room", selectedRoomId)) selectedRoomId = coarseMap.rooms?.[0]?.id || coarseMap.corridors?.[0]?.id || null;
    renderMap();
    renderLists();
    showSelectedEntity();
  } catch (_error) {
    activeTask.textContent = "粗地图不可用";
    authority.textContent = "UNAVAILABLE";
    setEditorStatus("无法读取粗地图服务。", "error");
  }
}

async function rebuildTasks() {
  const response = await coarseTaskRequest("rebuild", "", { method: "POST" });
  if (response.ok) {
    coarseMap = toV2(await response.json());
    document.querySelector(".coarse-task-panel")?.setAttribute("open", "");
    renderMap();
    renderLists();
    setEditorStatus("验证任务已按当前地图重算。", "saved");
  }
}

async function loadMission() {
  try {
    const response = await fetch("/api/missions", { cache: "no-store" });
    if (!response.ok) return;
    const summary = (await response.json()).missions?.[0] || null;
    if (!summary?.mission_id) {
      currentMission = null;
      routeAudits.clear();
      routeAuditRequestKey = "";
      renderLists();
      return;
    }
    const detailResponse = await fetch(`/api/missions/${encodeURIComponent(summary.mission_id)}`, { cache: "no-store" });
    currentMission = detailResponse.ok ? await detailResponse.json() : summary;
    renderLists();
    void refreshMissionRouteAudits();
  } catch (_error) {
    // Keep the previous mission summary until the API reconnects.
  }
}

async function compileMission() {
  if (!coarseMap) return;
  try {
    const response = await fetch("/api/missions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ source_coarse_map: coarseMap }) });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "任务编译失败");
    currentMission = await response.json();
    document.querySelector(".coarse-task-panel")?.setAttribute("open", "");
    renderLists();
    void refreshMissionRouteAudits();
    setEditorStatus(`任务已编译：${currentMission.task_count} 项 / ${currentMission.state}`, "saved");
  } catch (error) {
    setEditorStatus(error.message, "error");
  }
}

async function autoCompileMission() {
  if (!coarseMap || workspaceMode !== "coarse") return;
  try {
    const missionsResponse = await fetch("/api/missions", { cache: "no-store" });
    if (missionsResponse.ok) {
      const summary = (await missionsResponse.json()).missions?.[0] || null;
      if (summary && ["ACTIVE", "PAUSED"].includes(summary.state)) {
        setEditorStatus("地图已保存；当前 mission 正在执行，未自动替换任务。", "warning");
        return;
      }
    }
    const response = await fetch("/api/missions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ source_coarse_map: coarseMap }),
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`);
    currentMission = await response.json();
    document.querySelector(".coarse-task-panel")?.setAttribute("open", "");
    renderLists();
    void refreshMissionRouteAudits();
    setEditorStatus(`已保存并自动编译 mission / ${currentMission.task_count} 项。`, "saved");
  } catch (error) {
    setEditorStatus(`地图已保存，但 mission 自动编译失败：${error.message}`, "warning");
  }
}

async function exportNav2Map() {
  if (!coarseMap || workspaceMode !== "coarse") return;
  nav2ExportButton.disabled = true;
  setEditorStatus("正在生成 Nav2 occupancy map；不会启动机器人。", "saving");
  try {
    const response = await fetch(`/api/maps/coarse/${encodeURIComponent(mapId)}/nav2-export`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ resolution: 0.05, padding: 1.0, wall_thickness: 0.12 }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    setEditorStatus(`Nav2 map 已生成：${payload.width}x${payload.height} / ${payload.resolution}m / revision ${payload.source_revision}。`, "saved");
  } catch (error) {
    setEditorStatus(`Nav2 map 导出失败：${error.message}`, "error");
  } finally {
    nav2ExportButton.disabled = false;
  }
}

async function previewNav2Frontiers() {
  if (!coarseMap || workspaceMode !== "coarse") return;
  const x = Number(robotPosition?.[0] ?? 0);
  const y = Number(robotPosition?.[1] ?? 0);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    setEditorStatus("Frontier preview requires finite robot coordinates", "error");
    return;
  }
  nav2FrontierButton.disabled = true;
  setEditorStatus("Ranking frontier candidates; no navigation goal will be sent", "saving");
  try {
    const response = await fetch(`/api/maps/coarse/${encodeURIComponent(mapId)}/frontiers`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ robot_pose: { x, y }, resolution: 0.05, padding: 1.0, wall_thickness: 0.12 }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    frontierCandidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    renderMap();
    const top = payload.candidates?.[0]?.position;
    const target = top ? ` / top ${Number(top.x).toFixed(2)},${Number(top.y).toFixed(2)}` : "";
    setEditorStatus(`Frontier preview: ${payload.candidates?.length || 0} candidates${target} / read-only`, "saved");
  } catch (error) {
    setEditorStatus(`Frontier preview failed: ${error.message}`, "error");
  } finally {
    nav2FrontierButton.disabled = false;
  }
}

async function loadNavigation() {
  try {
    const response = await fetch("/api/robot/navigation/status", { cache: "no-store" });
    if (!response.ok) return;
    currentNavigation = await response.json();
    renderLists();
  } catch (_error) {
    // Keep the previous navigation summary while the API reconnects.
  }
}

function renderStairSurvey(report) {
  const payload = report || {};
  stairSurveyState.textContent = String(payload.state || "NO_SURVEY");
  stairSurveyProfile.textContent = payload.profile
    ? `${payload.profile.step_count} steps / ${Number(payload.profile.riser_height_m).toFixed(3)}m riser / ${Number(payload.profile.tread_depth_m).toFixed(3)}m tread`
    : "N/A";
  const source = payload.source_identity || {};
  stairSurveySource.textContent = source.capture_id ? `${source.capture_id} / ${source.frame_id || "N/A"}` : "N/A";
  stairSurveyBlockers.textContent = Array.isArray(payload.blocking_reasons) && payload.blocking_reasons.length
    ? payload.blocking_reasons.join(" / ")
    : "NONE";
  stairSurveyState.dataset.state = String(payload.state || "NO_SURVEY");
}

async function loadStairSurvey() {
  try {
    const response = await fetch("/api/stairs/survey/latest", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderStairSurvey(await response.json());
  } catch (error) {
    renderStairSurvey({ state: "UNAVAILABLE", blocking_reasons: [String(error.message)] });
  }
}

async function updateTaskMode(taskId, mode, expectedRevision, leaseId = "") {
  try {
    const response = await coarseTaskRequest(taskId, "/mode", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ mode, lease_id: leaseId, expected_revision: expectedRevision }) });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "模式更新失败");
    coarseMap = toV2(await response.json());
    renderMap();
    renderLists();
  } catch (error) {
    setEditorStatus(error.message, "error");
  }
}

async function updateTaskLease(taskId, leaseId, expectedRevision) {
  try {
    const response = await coarseTaskRequest(taskId, "/lease", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lease_id: leaseId, expected_revision: expectedRevision }) });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "租约更新失败");
    coarseMap = toV2(await response.json());
    renderMap();
    renderLists();
  } catch (error) {
    setEditorStatus(error.message, "error");
  }
}

async function lifecycleTask(taskId, status) {
  if (currentMission) {
    const missionTask = currentMissionTaskFor(taskId);
    if (!missionTask || ["DONE", "QUARANTINED"].includes(status)) return;
    const action = status === "ACTIVE" ? "pause" : status === "PAUSED" ? "resume" : "activate";
    try {
      const response = await fetch(`/api/missions/${encodeURIComponent(currentMission.mission_id)}/tasks/${encodeURIComponent(taskId)}/${action}`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "任务状态更新失败");
      currentMission = await response.json();
      renderLists();
    } catch (error) {
      setEditorStatus(error.message, "error");
    }
    return;
  }
  const response = await coarseTaskRequest(taskId, "", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ status: status === "ACTIVE" ? "DONE" : "ACTIVE" }) });
  if (response.ok) {
    coarseMap = toV2(await response.json());
    renderMap();
    renderLists();
  }
}

async function quarantineTask(taskId) {
  try {
    if (currentMission) {
      const response = await fetch(`/api/missions/${encodeURIComponent(currentMission.mission_id)}/tasks/${encodeURIComponent(taskId)}/quarantine`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "任务隔离失败");
      currentMission = await response.json();
    } else {
      const response = await coarseTaskRequest(taskId, "", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ status: "QUARANTINED" }) });
      if (!response.ok) throw new Error("任务隔离失败");
      coarseMap = toV2(await response.json());
    }
    setEditorStatus(`任务 ${taskId} 已隔离，禁止自动扩展。`, "warning");
    renderMap();
    renderLists();
  } catch (error) {
    setEditorStatus(error.message, "error");
  }
}

async function completeTask(taskId, status) {
  if (["DONE", "QUARANTINED"].includes(status)) return;
  try {
    if (currentMission) {
      const response = await fetch(`/api/missions/${encodeURIComponent(currentMission.mission_id)}/tasks/${encodeURIComponent(taskId)}/complete`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "任务完成失败");
      currentMission = await response.json();
    } else {
      const response = await coarseTaskRequest(taskId, "", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ status: "DONE", coverage_percent: 100 }) });
      if (!response.ok) throw new Error("任务完成失败");
      coarseMap = toV2(await response.json());
    }
    setEditorStatus(`任务 ${taskId} 已完成。`, "saved");
    renderMap();
    renderLists();
  } catch (error) {
    setEditorStatus(error.message, "error");
  }
}

async function navigateTask(taskId) {
  if (!currentMission) {
    setEditorStatus("请先编译任务。", "warning");
    return;
  }
  const task = currentMission.tasks?.find((item) => item.task_id === taskId);
  if (!task) {
    setEditorStatus("任务不在当前 mission 中。", "error");
    return;
  }
  const audit = routeAuditFor(taskId);
  if (!audit || audit.navigation_eligible !== true || audit.state !== "READY_FOR_HUMAN_REVIEW") {
    renderRouteAuditSummary(taskId);
    setEditorStatus(`Route audit blocked: ${routeAuditLabel(audit)}`, "warning");
    return;
  }
  try {
    const response = await fetch("/api/robot/navigation/goal", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ mission_id: currentMission.mission_id, task_id: task.task_id, lease_id: task.lease_id }) });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "导航目标被拒绝");
    currentNavigation = await response.json();
    renderLists();
  } catch (error) {
    setEditorStatus(error.message, "error");
  }
}

async function postNavigationAction(action, message) {
  try {
    const response = await fetch(`/api/robot/navigation/${action}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: action === "safety-stop" ? JSON.stringify({ reason: "DASHBOARD_OPERATOR_REQUEST" }) : "{}",
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || message);
    currentNavigation = await response.json();
    setEditorStatus(message, action === "safety-stop" ? "warning" : "saved");
    renderLists();
  } catch (error) {
    setEditorStatus(error.message, "error");
  }
}

async function acknowledgeNavigation() {
  try {
    const response = await fetch("/api/robot/navigation/acknowledge", { method: "POST" });
    if (!response.ok) throw new Error("安全状态确认失败");
    currentNavigation = await response.json();
    setEditorStatus("安全状态已确认，导航仍保持受控。", "saved");
    renderLists();
  } catch (error) {
    setEditorStatus(error.message, "error");
  }
}

async function loadRevisionHistory() {
  historyList.innerHTML = "<li>读取中...</li>";
  try {
    const response = await fetch(`/api/maps/coarse/${encodeURIComponent(mapId)}/revisions`, { cache: "no-store" });
    if (!response.ok) throw new Error("历史版本不可用");
    const revisions = [...((await response.json()).revisions || [])].reverse();
    historyList.innerHTML = revisions.map((revision) => `<li><strong>rev ${escapeText(revision.revision)}</strong><span>${(revision.rooms || []).length} 房间 / ${(revision.corridors || []).length} 走廊</span><em>USER_PRIOR</em></li>`).join("") || "<li>暂无历史版本</li>";
  } catch (error) {
    historyList.innerHTML = `<li class="history-error">${escapeText(error.message)}</li>`;
  }
}

async function loadRevisionIntoDraft(revisionNumber) {
  if (workspaceMode !== "coarse" || !Number.isInteger(revisionNumber) || revisionNumber <= 0) return false;
  if (draftDirty && !window.confirm("当前草稿尚未保存。载入历史版本会替换页面中的草稿，但不会修改服务器。继续吗？")) return false;
  setEditorStatus(`正在载入历史 revision ${revisionNumber}...`, "saving");
  try {
    const response = await fetch(`/api/maps/coarse/${encodeURIComponent(mapId)}/revisions/${encodeURIComponent(revisionNumber)}`, { cache: "no-store" });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "历史版本不可用");
    const payload = await response.json();
    const historical = toV2(payload.map);
    if (historical.map_id !== mapId) throw new Error("历史版本 map_id 与当前地图不一致");
    coarseMap = historical;
    draftMap = cloneMap(coarseMap);
    draftBaseRevision = Number(serverMap?.revision || 1);
    draftDirty = true;
    draftInitialized = true;
    selectedEntity = null;
    resetDraftHistory();
    saveDraftToStorage();
    historyPanel.hidden = true;
    updateDraftControls();
    setEditorStatus(`已载入历史 revision ${revisionNumber}；当前为本地草稿，保存后生成新 revision。`, "warning");
    renderMap();
    renderLists();
    showSelectedEntity();
    return true;
  } catch (error) {
    setEditorStatus(`载入历史版本失败：${error.message}`, "error");
    return false;
  }
}

async function loadRevisionHistoryInteractive() {
  historyList.innerHTML = "<li>读取中...</li>";
  try {
    const response = await fetch(`/api/maps/coarse/${encodeURIComponent(mapId)}/revisions`, { cache: "no-store" });
    if (!response.ok) throw new Error("历史版本不可用");
    const revisions = [...((await response.json()).revisions || [])].reverse();
    historyList.innerHTML = revisions.map((revision) => {
      const number = Number(revision.revision);
      const current = number === Number(serverMap?.revision);
      const savedAt = Number(revision.updated_at);
      const stamp = Number.isFinite(savedAt) && savedAt > 0
        ? new Date(savedAt * 1000).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false })
        : "";
      return `<li${current ? ' class="current"' : ""}><div class="coarse-history-meta"><strong>rev ${escapeText(number)}</strong><span>${(revision.rooms || []).length} 房间 / ${(revision.corridors || []).length} 走廊</span><em>${current ? "CURRENT" : "USER_PRIOR"}</em>${stamp ? `<time>${escapeText(stamp)}</time>` : ""}</div><button type="button" class="coarse-history-load" data-revision="${escapeText(number)}" title="Load this historical revision as a new local draft">载入为草稿</button></li>`;
    }).join("") || "<li>暂无历史版本</li>";
    historyList.querySelectorAll("[data-revision]").forEach((button) => button.addEventListener("click", () => loadRevisionIntoDraft(Number(button.dataset.revision))));
  } catch (error) {
    historyList.innerHTML = `<li class="history-error">${escapeText(error.message)}</li>`;
  }
}

function exportMap() {
  if (!coarseMap) return;
  const blob = new Blob([JSON.stringify(toV2(coarseMap), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${mapId}-revision-${coarseMap.revision || 1}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
  setEditorStatus(`已导出 ${mapId} / revision ${coarseMap.revision || 1}`, "saved");
}

function importMapFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.addEventListener("load", async () => {
    try {
      const candidate = toV2(JSON.parse(String(reader.result || "")));
      const response = await fetch("/api/maps/coarse/import", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(candidate) });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || "导入验证失败");
      if (candidate.map_id !== mapId) throw new Error(`导入地图 ID ${candidate.map_id} 与当前 ${mapId} 不一致`);
      const validated = await response.json();
      captureDraftHistory();
      coarseMap = { ...toV2(validated.map), revision: coarseMap.revision };
      markDraftChanged("导入地图已加入草稿；点击保存 revision 写入服务器。");
      setEditorStatus("导入地图已验证并保存为新的 revision。", "saved");
    } catch (error) {
      setEditorStatus(`导入失败：${error.message}`, "error");
    } finally {
      importInput.value = "";
    }
  });
  reader.readAsText(file);
}

function setRoomScanStatus(message, state = "idle") {
  if (!roomScanStatus) return;
  roomScanStatus.textContent = message;
  roomScanStatus.dataset.state = state;
}

function roomScanPhaseFor(payload = {}) {
  const state = String(payload.phase || payload.state || "").toUpperCase();
  if (state === "STARTING" || state === "PREFLIGHT") return "PREFLIGHT";
  if (state === "RUNNING" || state === "RECORDING") return "RECORDING";
  if (state === "STOPPING") return "STOPPING";
  if (state === "FINALIZING") return "FINALIZING";
  if (["PREVIEW_READY", "FINAL_SURFACE_AVAILABLE", "COMPLETED", "COMPLETE"].includes(state)) return "PREVIEW_READY";
  if (["FAILED", "ERROR", "QUARANTINED"].includes(state)) return "FAILED";
  if (state === "STOPPED") return "STOPPED";
  return "READY";
}

function renderRoomScanWorkflow(payload = {}, phase = roomScanPhaseFor(payload)) {
  const requestId = payload.request_id || roomScanRequestId;
  const sessionId = payload.session_id || payload.request?.session_id || "NONE";
  const blocker = payload.blocker || payload.error || payload.detail || "";
  if (roomScanPhase) {
    roomScanPhase.textContent = phase;
    roomScanPhase.dataset.state = phase === "FAILED" ? "error" : ["RECORDING", "FINALIZING", "STOPPING"].includes(phase) ? "working" : phase === "PREVIEW_READY" ? "ready" : "idle";
  }
  if (roomScanSession) roomScanSession.textContent = String(sessionId);
  if (roomScanHealth) roomScanHealth.textContent = blocker ? `BLOCKED / ${blocker}` : phase === "RECORDING" ? "LIVE" : phase === "FINALIZING" ? "VERIFYING" : phase === "PREVIEW_READY" ? "READY" : "WAITING";
  if (roomScanDetail) roomScanDetail.textContent = blocker || (phase === "READY" ? "等待开始采集" : `SESSION ${sessionId}`);
  if (roomScanStopButton) roomScanStopButton.hidden = !requestId || !["PREFLIGHT", "RECORDING", "STOPPING", "FINALIZING"].includes(phase);
}

function stopRoomScanPolling() {
  if (roomScanPollTimer !== null) window.clearInterval(roomScanPollTimer);
  roomScanPollTimer = null;
}

async function pollRoomScanStatus() {
  if (!roomScanRequestId || !roomScanOperatorToken) return;
  try {
    const response = await fetch(`/api/room-scan/status/${encodeURIComponent(roomScanRequestId)}`, {
      headers: { "X-Go2-Room-Scan-Operator-Token": roomScanOperatorToken },
      cache: "no-store",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    const phase = roomScanPhaseFor(payload);
    renderRoomScanWorkflow(payload, phase);
    setRoomScanStatus(`${phase} / ${payload.session_id || roomScanRequestId}`, phase === "FAILED" ? "error" : "working");
    if (["PREVIEW_READY", "STOPPED", "FAILED", "READY"].includes(phase)) stopRoomScanPolling();
  } catch (error) {
    renderRoomScanWorkflow({ detail: error.message }, "FAILED");
    setRoomScanStatus(`状态不可用：${error.message}`, "error");
    stopRoomScanPolling();
  }
}

function startRoomScanPolling() {
  stopRoomScanPolling();
  void pollRoomScanStatus();
  roomScanPollTimer = window.setInterval(() => { void pollRoomScanStatus(); }, 1500);
}

async function stopRoomScan() {
  if (!roomScanRequestId || !roomScanOperatorToken) return;
  roomScanStopButton.disabled = true;
  renderRoomScanWorkflow({ request_id: roomScanRequestId, session_id: roomScanSession?.textContent }, "STOPPING");
  try {
    const response = await fetch(`/api/room-scan/stop/${encodeURIComponent(roomScanRequestId)}`, {
      method: "POST",
      headers: { "X-Go2-Room-Scan-Operator-Token": roomScanOperatorToken },
      body: "{}",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    renderRoomScanWorkflow(payload, roomScanPhaseFor(payload));
    setRoomScanStatus("已请求停止，等待会话封存", "working");
    startRoomScanPolling();
  } catch (error) {
    setRoomScanStatus(`停止失败：${error.message}`, "error");
    roomScanStopButton.disabled = false;
  }
}

async function loadRoomScanCapabilities() {
  if (!roomScanStartButton) return;
  try {
    const response = await fetch("/api/room-scan/capabilities", { cache: "no-store" });
    const payload = response.ok ? await response.json() : {};
    roomScanStartButton.hidden = payload.enabled !== true;
  } catch (_error) {
    roomScanStartButton.hidden = true;
  }
}

async function startRoomScan(event) {
  event.preventDefault();
  const duration = Number(roomScanDuration?.value || 600);
  const token = String(roomScanToken?.value || "").trim();
  if (!Number.isInteger(duration) || duration < 60 || duration > 3600 || !token) {
    setRoomScanStatus("请输入有效时长和短期操作令牌。", "error");
    return;
  }
  const confirmButton = roomScanForm?.querySelector("[type=submit]");
  if (confirmButton) confirmButton.disabled = true;
  setRoomScanStatus("正在执行 readiness 和机器人 preflight...", "working");
  try {
    const response = await fetch("/api/room-scan/start", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ duration_seconds: duration, operator_token: token }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    roomScanRequestId = String(payload.request_id || "");
    roomScanOperatorToken = token;
    renderRoomScanWorkflow(payload, "PREFLIGHT");
    setRoomScanStatus(`已启动 ${payload.session_id || payload.request_id} / SESSION_ODOM_LOCAL。`, "started");
    setEditorStatus("单房间采集已启动；当前页面不会发送运动指令。", "warning");
    if (roomScanToken) roomScanToken.value = "";
    roomScanDialog?.close();
    startRoomScanPolling();
  } catch (error) {
    setRoomScanStatus(`未启动：${error.message}`, "error");
  } finally {
    if (confirmButton) confirmButton.disabled = false;
  }
}

document.documentElement.dataset.workspaceMode = workspaceMode;
document.documentElement.dataset.referenceOpen = "false";
workspaceModeObserveButton?.addEventListener("click", () => setWorkspaceMode("observe"));
workspaceModeCoarseButton?.addEventListener("click", () => setWorkspaceMode("coarse"));
observationReference?.addEventListener("toggle", () => {
  document.documentElement.dataset.referenceOpen = String(observationReference.open);
});
loadCurrentButton?.addEventListener("click", loadCurrentMapIntoDraft);
discardDraftButton?.addEventListener("click", discardLocalDraft);
saveRevisionButton?.addEventListener("click", () => { void saveMapRevision(); });
nav2ExportButton?.addEventListener("click", () => { void exportNav2Map(); });
nav2FrontierButton?.addEventListener("click", () => { void previewNav2Frontiers(); });
stairSurveyRefresh?.addEventListener("click", () => { void loadStairSurvey(); });
reloadServerButton?.addEventListener("click", loadCurrentMapIntoDraft);
keepDraftButton?.addEventListener("click", () => {
  draftConflict.hidden = true;
  draftConflict.dataset.state = "hidden";
  reloadServerButton.hidden = true;
  keepDraftButton.hidden = true;
  setEditorStatus("继续本地草稿；服务器版本未被覆盖。", "warning");
});
roomScanForm?.addEventListener("submit", startRoomScan);
roomScanStartButton?.addEventListener("click", () => {
  setRoomScanStatus("等待确认", "idle");
  roomScanDialog?.showModal();
  roomScanToken?.focus();
});
roomScanCancelButton?.addEventListener("click", () => roomScanDialog?.close());
roomScanStopButton?.addEventListener("click", () => { void stopRoomScan(); });
document.addEventListener("keydown", (event) => {
  if (workspaceMode !== "coarse") return;
  const target = event.target;
  const typing = target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
  if (event.key === "Escape") {
    document.querySelectorAll(".coarse-more[open]").forEach((menu) => menu.removeAttribute("open"));
  }
  if (event.key === "Escape" && !roomScanDialog?.open && editor.draftPoints.length) {
    event.preventDefault();
    cancelDraft();
    return;
  }
  if (typing && event.key !== "Escape") return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) redoDraft();
    else undoDraft();
    return;
  }
  if (event.key === "Enter" && editor.tool !== "select" && editor.draftPoints.length) {
    event.preventDefault();
    void commitDraft();
    return;
  }
  if (event.key.toLowerCase() === "r") resetMapView();
});

document.querySelectorAll("[data-map-tool]").forEach((button) => button.addEventListener("click", () => setTool(button.dataset.mapTool)));
document.querySelectorAll(".coarse-more-menu").forEach((menu) => {
  menu.addEventListener("click", (event) => {
    if (event.target instanceof HTMLElement && event.target.closest("button")) {
      menu.closest("details")?.removeAttribute("open");
    }
  });
});
document.addEventListener("click", (event) => {
  document.querySelectorAll(".coarse-more[open]").forEach((menu) => {
    if (!(event.target instanceof Node) || menu.contains(event.target)) return;
    menu.removeAttribute("open");
  });
});
svg?.addEventListener("pointerdown", handleSvgPointerDown);
svg?.addEventListener("pointermove", (event) => {
  if (editor.dragging) updateDraggedVertex(clientToMapPoint(event));
  else if (mapViewport.dragging) updateMapPan(event);
});
svg?.addEventListener("pointerup", handleSvgPointerUp);
svg?.addEventListener("pointercancel", handleSvgPointerUp);
svg?.addEventListener("click", (event) => {
  const target = event.target instanceof Element ? event.target.closest("[data-frontier-id]") : null;
  if (!target) return;
  const candidate = frontierCandidates.find((item) => item.candidate_id === target.dataset.frontierId);
  if (!candidate) return;
  const point = candidate.position || {};
  setEditorStatus(`Frontier ${candidate.candidate_id}: ${Number(point.x).toFixed(2)},${Number(point.y).toFixed(2)} / read-only`, "saved");
});
svg?.addEventListener("wheel", (event) => {
  event.preventDefault();
  zoomAt(event.clientX, event.clientY, event.deltaY < 0 ? 1.12 : 1 / 1.12);
}, { passive: false });
finishButton?.addEventListener("click", () => { void commitDraft(); });
cancelButton?.addEventListener("click", cancelDraft);
entityForm?.addEventListener("submit", updateEntity);
entityDeleteButton?.addEventListener("click", deleteSelectedEntity);
function rotateSelectedFurniture(deltaDegrees) {
  const selected = selectedEntityRecord();
  if (!selected || selected.kind !== "equipment" || selected.entity.locked === true) {
    setEditorStatus("请先选择未锁定家具。", "warning");
    return;
  }
  captureDraftHistory();
  selected.entity.rotation_rad = Number(selected.entity.rotation_rad || 0) + deltaDegrees * Math.PI / 180;
  markDraftChanged("家具旋转已更新，点击保存 revision 写入服务端。");
  showSelectedEntity();
  renderMap();
}
rotateFurnitureLeft?.addEventListener("click", () => rotateSelectedFurniture(-15));
rotateFurnitureRight?.addEventListener("click", () => rotateSelectedFurniture(15));
refreshButton?.addEventListener("click", loadMap);
rebuildButton?.addEventListener("click", rebuildTasks);
compileButton?.addEventListener("click", compileMission);
importButton?.addEventListener("click", () => importInput?.click());
exportButton?.addEventListener("click", exportMap);
importInput?.addEventListener("change", importMapFile);
historyToggle?.addEventListener("click", async () => { historyPanel.hidden = !historyPanel.hidden; if (!historyPanel.hidden) await loadRevisionHistoryInteractive(); });
historyClose?.addEventListener("click", () => { historyPanel.hidden = true; });
zoomInButton?.addEventListener("click", zoomIn);
zoomOutButton?.addEventListener("click", zoomOut);
resetViewButton?.addEventListener("click", resetMapView);
navPauseButton?.addEventListener("click", () => postNavigationAction("pause", "导航已暂停。"));
navResumeButton?.addEventListener("click", () => postNavigationAction("resume", "导航已恢复。"));
navCancelButton?.addEventListener("click", () => postNavigationAction("cancel", "导航目标已取消。"));
navSafetyButton?.addEventListener("click", () => postNavigationAction("safety-stop", "安全停机已触发。"));
navAcknowledgeButton?.addEventListener("click", acknowledgeNavigation);
view2DButton?.addEventListener("click", () => setViewMode("2d"));
viewIsoButton?.addEventListener("click", () => setViewMode("3d"));
view3DButton?.addEventListener("click", () => setViewMode("threejs"));
threeTransformTranslate?.addEventListener("click", () => setThreeTransformMode("translate"));
threeTransformRotate?.addEventListener("click", () => setThreeTransformMode("rotate"));
threeTransformSpace?.addEventListener("click", setThreeTransformSpace);
threeTransformSnap?.addEventListener("change", setThreeTransformSnap);
floorSelect?.addEventListener("change", () => { void switchFloor(floorSelect.value); });
floorAddButton?.addEventListener("click", () => {
  if (floorFormId) floorFormId.value = "";
  if (floorFormLabel) floorFormLabel.value = "";
  if (floorFormStatus) floorFormStatus.textContent = "";
  floorDialog?.showModal();
  floorFormId?.focus();
});
templateOpenButton?.addEventListener("click", () => { renderTemplateCatalog(); templateDialog?.showModal(); });
templateCancel?.addEventListener("click", () => templateDialog?.close());
connectionAddButton?.addEventListener("click", () => { populateConnectionFloors(); connectionLabel.value = ""; connectionStatus.textContent = ""; connectionDialog?.showModal(); connectionFrom?.focus(); });
async function saveConnectionWithDetails() {
  const buildingId = String(buildingIndex?.building_id || serverMap?.building_id || "building-1");
  if (!connectionFrom?.value || !connectionTo?.value || connectionFrom.value === connectionTo.value) {
    if (connectionStatus) connectionStatus.textContent = "请选择两个不同楼层。";
    return;
  }
  const kind = connectionKind?.value || "stairs";
  const connection = { id: `vertical-${Date.now()}`, kind, from_floor_id: connectionFrom.value, to_floor_id: connectionTo.value, label: connectionLabel?.value.trim() || kind };
  const fromAnchor = [Number(connectionFromX?.value), Number(connectionFromY?.value)];
  const toAnchor = [Number(connectionToX?.value), Number(connectionToY?.value)];
  const hasAnyAnchorValue = [connectionFromX, connectionFromY, connectionToX, connectionToY].some((input) => input?.value !== "");
  if (hasAnyAnchorValue && (!fromAnchor.every(Number.isFinite) || !toAnchor.every(Number.isFinite))) {
    if (connectionStatus) connectionStatus.textContent = "跨层坐标必须同时填写完整。";
    return;
  }
  if (hasAnyAnchorValue) { connection.from_anchor = fromAnchor; connection.to_anchor = toAnchor; }
  if (kind === "stairs") connection.stair_profile = { step_count: Number(connectionStepCount?.value), riser_height_m: Number(connectionRiser?.value), tread_depth_m: Number(connectionTread?.value), width_m: Number(connectionWidth?.value), direction: connectionDirection?.value || "bidirectional" };
  const next = { ...(buildingIndex || {}), building_id: buildingId, connections: [...(buildingIndex?.connections || []), connection] };
  try {
    const response = await fetch(`/api/buildings/coarse/${encodeURIComponent(buildingId)}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(next) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    buildingIndex = payload;
    connectionDialog?.close();
    setEditorStatus("跨层连接已保存。", "saved");
  } catch (error) { if (connectionStatus) connectionStatus.textContent = error.message; }
}
connectionForm?.addEventListener("submit", (event) => { event.preventDefault(); void saveConnectionWithDetails(); });
connectionKind?.addEventListener("change", () => { if (connectionStairFields) connectionStairFields.hidden = connectionKind.value !== "stairs"; });
buildingMissionButton?.addEventListener("click", () => { void compileBuildingMission(); });
connectionCancel?.addEventListener("click", () => connectionDialog?.close());
floorForm?.addEventListener("submit", (event) => { event.preventDefault(); void createFloor(); });
floorFormCancel?.addEventListener("click", () => floorDialog?.close());
document.querySelectorAll("[data-quick-action]").forEach((button) => button.addEventListener("click", () => {
  const action = button.dataset.quickAction;
  if (action === "rename") {
    if (selectedEntity && entityNameInput) { entityNameInput.focus(); entityNameInput.select(); }
    else setEditorStatus("请先选择一个功能区，再修改区域名称。", "warning");
    return;
  }
  if (action === "wall") { setTool("wall"); return; }
  if (action === "equipment") { furnitureDialog?.showModal(); return; }
  if (action === "split") { splitSelectedZone("vertical"); return; }
  if (action === "merge") { mergeSelectedZones(); return; }
  if (action === "carpet") { setTool("carpet"); return; }
  if (action === "threshold") { setTool("threshold"); return; }
  if (action === "material") {
    const selected = selectedEntityRecord();
    if (!selected) { setEditorStatus("请先选择要设置材质的实体。", "warning"); return; }
    materialDialog?.showModal();
    return;
  }
  if (false) {
    const selected = selectedEntityRecord();
    if (selected) {
      selected.entity.material = window.prompt("材质名称", selected.entity.material || "floor")?.trim() || selected.entity.material || "floor";
      markDraftChanged("材质已修改，保存 revision 后写入服务器。");
      showSelectedEntity(); renderMap();
    } else setEditorStatus("请先选择功能区或地毯。", "warning");
    return;
  }
  setEditorStatus(`${button.innerText.trim()}：当前版本保留入口，待补充专用参数后启用。`, "warning");
}));
document.querySelectorAll("[data-furniture-type]").forEach((button) => button.addEventListener("click", () => {
  pendingFurnitureType = button.dataset.furnitureType || null;
  furnitureDialog?.close();
  setTool("equipment");
  setEditorStatus(`${furnitureCatalog[pendingFurnitureType]?.label || "家具"}：点击地图放置。`, "editing");
}));
function filterFurnitureCatalog() {
  const query = String(furnitureSearch?.value || "").trim().toLowerCase();
  const category = furnitureCategory?.value || "all";
  document.querySelectorAll("[data-furniture-type]").forEach((button) => {
    const label = button.textContent.toLowerCase();
    button.hidden = (category !== "all" && button.dataset.furnitureCategory !== category) || (query && !label.includes(query));
  });
}
async function loadFurnitureCatalog() {
  try {
    const response = await fetch("/api/catalog/furniture", { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    if (!Array.isArray(payload.items)) return;
    payload.items.forEach((item) => {
      if (item.id && Array.isArray(item.dimensions_m)) furnitureCatalog[item.id] = item;
    });
  } catch (_error) { /* local preset catalog remains authoritative fallback */ }
}
async function loadCoarseMapTemplates() {
  try {
    const response = await fetch("/api/catalog/coarse-map-templates", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    coarseMapTemplates = Array.isArray(payload.templates) ? payload.templates : [];
    renderTemplateCatalog();
  } catch (_error) {
    coarseMapTemplates = [];
    if (templateGrid) templateGrid.innerHTML = '<p class="coarse-overview-empty">模板目录暂不可用</p>';
  }
}
async function loadPbrMaterialCatalog() {
  try {
    const response = await fetch("/api/catalog/pbr-materials", { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    pbrMaterialCatalog = Array.isArray(payload.materials) ? payload.materials : [];
    document.querySelectorAll("[data-material]").forEach((button) => {
      const material = pbrMaterialCatalog.find((item) => item.id === button.dataset.material);
      if (material) button.title = `${material.label} / ${material.asset_id} / ${payload.license}`;
    });
  } catch (_error) { pbrMaterialCatalog = []; }
}

function renderTemplateCatalog() {
  if (!templateGrid) return;
  templateGrid.innerHTML = coarseMapTemplates.map((template) => `<button type="button" data-template-id="${escapeText(template.id)}"><span><strong>${escapeText(template.label)}</strong><small>${escapeText(template.description)}</small></span><b>${(template.rooms || []).length} 区域</b></button>`).join("");
  templateGrid.querySelectorAll("[data-template-id]").forEach((button) => button.addEventListener("click", () => applyCoarseMapTemplate(button.dataset.templateId)));
}


function applyCoarseMapTemplate(templateId) {
  const template = coarseMapTemplates.find((item) => item.id === templateId);
  if (!template || !coarseMap || workspaceMode !== "coarse") return;
  if (draftDirty && !window.confirm("当前草稿已有修改。使用模板将替换当前草稿中的空间和对象，继续吗？")) return;
  captureDraftHistory();
  coarseMap = toV2({
    ...coarseMap,
    rooms: cloneMap(template.rooms || []),
    corridors: cloneMap(template.corridors || []),
    walls: cloneMap(template.walls || []),
    doors: cloneMap(template.doors || []),
    features: cloneMap(template.features || []),
    keep_out_zones: cloneMap(template.keep_out_zones || []),
    anchors: cloneMap(template.anchors || []),
    carpets: cloneMap(template.carpets || []),
    thresholds: cloneMap(template.thresholds || []),
    tasks: [],
  });
  selectedRoomId = coarseMap.rooms?.[0]?.id || coarseMap.corridors?.[0]?.id || null;
  selectedEntity = selectedRoomId ? { kind: coarseMap.rooms?.some((room) => room.id === selectedRoomId) ? "room" : "corridor", id: selectedRoomId } : null;
  resetMapView();
  markDraftChanged(`已套用“${template.label}”模板；保存 revision 前只保留在本地草稿。`);
  templateDialog?.close();
  renderMap();
  renderLists();
  showSelectedEntity();
}
furnitureSearch?.addEventListener("input", filterFurnitureCatalog);
furnitureCategory?.addEventListener("change", filterFurnitureCatalog);
materialCancel?.addEventListener("click", () => materialDialog?.close());
document.querySelectorAll("[data-material]").forEach((button) => button.addEventListener("click", () => {
  const selected = selectedEntityRecord();
  if (!selected) return;
  captureDraftHistory();
  selected.entity.material = button.dataset.material;
  markDraftChanged("材质属性已更新，保存 revision 后写入服务端。");
  materialDialog?.close();
  showSelectedEntity();
  renderMap();
}));
furnitureCancel?.addEventListener("click", () => furnitureDialog?.close());

setWorkspaceMode(workspaceMode, { syncUrl: false });
loadMap().then(() => loadBuildingIndex());
loadMission();
loadNavigation();
loadRoomScanCapabilities();
loadStairSurvey();
void loadFurnitureCatalog();
void loadCoarseMapTemplates();
void loadPbrMaterialCatalog();
setInterval(loadMap, 5000);
setInterval(loadMission, 5000);
setInterval(loadNavigation, 3000);
setInterval(loadStairSurvey, 5000);
