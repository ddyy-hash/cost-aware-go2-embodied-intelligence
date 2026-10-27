const workflowById = (id) => document.getElementById(id);
const segmentStateNode = workflowById("workflow-segment-state");
const qualityStateNode = workflowById("workflow-quality-state");
const coverageStateNode = workflowById("workflow-coverage-state");
const appearanceStateNode = workflowById("workflow-appearance-state");
const poseP95Node = workflowById("workflow-pose-p95");
const planeP95Node = workflowById("workflow-plane-p95");
const clockFitNode = workflowById("workflow-clock-fit");
const revisitCountNode = workflowById("workflow-revisit-count");
const exportStateNode = workflowById("workflow-export");
const replayStateNode = workflowById("workflow-replay");
const blockersNode = workflowById("workflow-blockers");
const replayButton = workflowById("workflow-replay-button");
const exportButton = workflowById("workflow-export-button");
const exportLink = workflowById("workflow-export-link");

let workflowMissionId = null;
let workflowRequestInFlight = false;
let workflowActionInFlight = false;

const workflowChinese = {
  "N/A": "暂无", NONE: "无", LOADING: "加载中", UNKNOWN: "未知", UNAVAILABLE: "不可用",
  "NO SEGMENT": "暂无重建片段", "NO MISSION": "暂无任务", "NO_RECONSTRUCTION_SEGMENT": "暂无重建片段",
  NO_MISSION: "暂无任务", ACCEPTED: "已通过", VALID: "有效", VERIFIED: "已验证",
  QUARANTINED: "已隔离", LOCKED: "已锁定", AVAILABLE: "可用", FAILED: "失败",
  STARTED: "已开始", CHECKING: "正在检查", "CHECKING GATE...": "正在检查门控...",
  "CHECKING...": "正在检查...", "423 LOCKED": "已锁定", TASKS: "项任务",
};

function workflowChineseText(value) {
  return String(value).split(" / ").map((part) => workflowChinese[part] || part).join(" / ");
}

function workflowText(value, fallback = "N/A") {
  if (value === null || value === undefined || value === "") return fallback;
  return workflowChineseText(value);
}

function workflowSet(node, value, state = "") {
  if (!node) return;
  node.textContent = workflowChineseText(value);
  node.dataset.state = state;
}

function workflowUnique(values) {
  return [...new Set(values.filter((value) => typeof value === "string" && value))];
}

async function workflowJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const payload = await response.json().catch(() => ({}));
  return { response, payload };
}

function workflowRenderBlockers(blockers) {
  const values = workflowUnique(blockers);
  blockersNode.textContent = values.length ? values.map(workflowChineseText).join(" / ") : "无";
  blockersNode.dataset.state = values.length ? "blocked" : "clear";
}

async function workflowLoadSegments(blockers) {
  try {
    const { response, payload } = await workflowJson("/api/reconstruction/segments");
    if (!response.ok) throw new Error(`SEGMENTS HTTP ${response.status}`);
    const segments = Array.isArray(payload.segments) ? payload.segments : [];
    const latest = segments[segments.length - 1];
    if (!latest) {
      workflowSet(segmentStateNode, "NO SEGMENT", "unknown");
      workflowSet(qualityStateNode, "UNKNOWN", "unknown");
      workflowSet(poseP95Node, "N/A", "unknown");
      workflowSet(planeP95Node, "N/A", "unknown");
      workflowSet(clockFitNode, "N/A", "unknown");
      blockers.push("NO_RECONSTRUCTION_SEGMENT");
      return;
    }
    workflowSet(segmentStateNode, `${workflowText(latest.segment_id)} / ${workflowText(latest.state)}`, latest.state === "QUARANTINED" ? "blocked" : "observed");
    try {
      const qualityResult = await workflowJson(`/api/reconstruction/segments/${encodeURIComponent(latest.segment_id)}/quality`);
      if (!qualityResult.response.ok) throw new Error(`QUALITY HTTP ${qualityResult.response.status}`);
      const quality = qualityResult.payload;
      const qualityBlockers = Array.isArray(quality.blockers) ? quality.blockers : [];
      blockers.push(...qualityBlockers);
      workflowSet(qualityStateNode, `${workflowText(quality.decision)} / ${workflowText(quality.quality_state)}`, quality.decision === "ACCEPTED" ? "clear" : "blocked");
      const poseP95 = quality.point_to_trajectory?.p95_m;
      const planeP95 = quality.wall_plane?.residual_p95_m;
      workflowSet(poseP95Node, Number.isFinite(Number(poseP95)) ? `${Number(poseP95).toFixed(3)} m` : "N/A", "metric");
      workflowSet(planeP95Node, Number.isFinite(Number(planeP95)) ? `${Number(planeP95).toFixed(3)} m` : "N/A", "metric");
      workflowSet(clockFitNode, workflowText(quality.clock_model?.status, "UNKNOWN"), quality.clock_model?.status === "VALID" ? "clear" : "blocked");
    } catch (error) {
      workflowSet(qualityStateNode, "UNKNOWN", "unknown");
      workflowSet(poseP95Node, "N/A", "unknown");
      workflowSet(planeP95Node, "N/A", "unknown");
      workflowSet(clockFitNode, "N/A", "unknown");
      blockers.push(error.message);
    }
  } catch (error) {
    workflowSet(segmentStateNode, "UNAVAILABLE", "blocked");
    workflowSet(qualityStateNode, "UNKNOWN", "unknown");
    blockers.push(error.message);
  }
}

async function workflowLoadCoverage(blockers) {
  try {
    const { response, payload } = await workflowJson("/api/missions");
    if (!response.ok) throw new Error(`MISSIONS HTTP ${response.status}`);
    const mission = payload.missions?.[0];
    if (!mission?.mission_id) {
      workflowMissionId = null;
      workflowSet(coverageStateNode, "NO MISSION", "unknown");
      workflowSet(revisitCountNode, "N/A", "unknown");
      blockers.push("NO_MISSION");
      return;
    }
    workflowMissionId = mission.mission_id;
    const coverageResult = await workflowJson(`/api/coverage/${encodeURIComponent(workflowMissionId)}`);
    if (!coverageResult.response.ok) throw new Error(`COVERAGE HTTP ${coverageResult.response.status}`);
    const coverage = coverageResult.payload;
    workflowSet(coverageStateNode, `${workflowText(coverage.coverage_percent, "0")}% / ${workflowText(coverage.state)}`, coverage.state === "VERIFIED" ? "clear" : coverage.state === "QUARANTINED" ? "blocked" : "observed");
    const revisitCount = (coverage.tasks || []).filter((task) => task.coverage_state === "UNSEEN").length;
      workflowSet(revisitCountNode, `${revisitCount} 项任务`, revisitCount ? "blocked" : "clear");
    blockers.push(...(coverage.rooms || []).flatMap((room) => room.blockers || []));
  } catch (error) {
    workflowSet(coverageStateNode, "UNAVAILABLE", "blocked");
    workflowSet(revisitCountNode, "N/A", "unknown");
    blockers.push(error.message);
  }
}

async function workflowLoadAppearance(blockers) {
  try {
    const { response, payload } = await workflowJson("/api/appearance/measured-surface/status");
    if (!response.ok) throw new Error(`APPEARANCE HTTP ${response.status}`);
    const state = workflowText(payload.state || payload.status, "UNKNOWN");
    const appearanceBlockers = Array.isArray(payload.blockers) ? payload.blockers : [];
    blockers.push(...appearanceBlockers);
    workflowSet(appearanceStateNode, state, payload.delivery_authorized === true ? "clear" : "blocked");
  } catch (error) {
    workflowSet(appearanceStateNode, "UNAVAILABLE", "blocked");
    blockers.push(error.message);
  }
}

async function loadWorkflowStatus() {
  if (workflowRequestInFlight || workflowActionInFlight || document.visibilityState === "hidden") return;
  workflowRequestInFlight = true;
  const blockers = [];
  try {
    await Promise.all([workflowLoadSegments(blockers), workflowLoadCoverage(blockers), workflowLoadAppearance(blockers)]);
    if (!workflowActionInFlight) workflowRenderBlockers(blockers);
  } finally {
    workflowRequestInFlight = false;
  }
}

async function exportFormalGlb() {
  if (workflowActionInFlight) return;
  workflowActionInFlight = true;
  exportButton.disabled = true;
  workflowSet(exportStateNode, "CHECKING GATE...", "checking");
  try {
    const evidenceResult = await workflowJson("/api/reconstruction/field-acceptance-evidence");
    const evidence = evidenceResult.payload;
    if (!evidenceResult.response.ok || evidence.available !== true || !evidence.exports?.building_request) {
      const blockers = evidence.blockers || [`HTTP ${evidenceResult.response.status}`];
      workflowSet(exportStateNode, "423 LOCKED", "blocked");
      workflowRenderBlockers(blockers);
      return;
    }
    const result = await workflowJson("/api/exports/glb", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind: "building", ...evidence.exports.building_request }),
    });
    if (result.response.status === 423) {
      workflowSet(exportStateNode, "423 LOCKED", "blocked");
      workflowRenderBlockers(result.payload.detail?.blockers || ["FORMAL_EXPORT_BLOCKED"]);
      return;
    }
    if (!result.response.ok) throw new Error(result.payload.detail || `EXPORT HTTP ${result.response.status}`);
    exportLink.href = result.payload.glb_url;
    exportLink.hidden = false;
    workflowSet(exportStateNode, "AVAILABLE", "clear");
    workflowRenderBlockers([]);
  } catch (error) {
    workflowSet(exportStateNode, "FAILED", "blocked");
    workflowRenderBlockers([error.message]);
  } finally {
    exportButton.disabled = false;
    workflowActionInFlight = false;
  }
}

async function workflowReplay() {
  if (workflowActionInFlight) return;
  workflowActionInFlight = true;
  replayButton.disabled = true;
  workflowSet(replayStateNode, "CHECKING...", "checking");
  try {
    const segmentResult = await workflowJson("/api/reconstruction/segments");
    const latest = segmentResult.payload.segments?.at(-1);
    if (!segmentResult.response.ok || !latest?.segment_id) {
      workflowSet(replayStateNode, "423 LOCKED", "blocked");
      workflowRenderBlockers(["NO_RECONSTRUCTION_SEGMENT"]);
      return;
    }
    const result = await workflowJson(`/api/reconstruction/segments/${encodeURIComponent(latest.segment_id)}/replay`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    if (result.response.status === 423) {
      workflowSet(replayStateNode, "423 LOCKED", "blocked");
      workflowRenderBlockers(result.payload.detail?.blockers || ["REPLAY_BLOCKED"]);
      return;
    }
    if (!result.response.ok) throw new Error(result.payload.detail || `REPLAY HTTP ${result.response.status}`);
    workflowSet(replayStateNode, workflowText(result.payload.state, "STARTED"), "clear");
    workflowRenderBlockers([]);
  } catch (error) {
    workflowSet(replayStateNode, "FAILED", "blocked");
    workflowRenderBlockers([error.message]);
  } finally {
    replayButton.disabled = false;
    workflowActionInFlight = false;
  }
}

exportButton?.addEventListener("click", exportFormalGlb);
replayButton?.addEventListener("click", workflowReplay);
loadWorkflowStatus();
setInterval(loadWorkflowStatus, 4000);
