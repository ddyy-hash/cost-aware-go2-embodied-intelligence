// Empty state connection checklist management

const checklistItems = {
  calibration: document.getElementById("check-calibration"),
  robot: document.getElementById("check-robot"),
  lidar: document.getElementById("check-lidar"),
  surface: document.getElementById("check-surface"),
};

const statusPanel = document.getElementById("status-panel");
const statusTitle = document.getElementById("status-title");
const statusMessage = document.getElementById("status-message");

let lastSnapshotState = {
  hasCalibration: false,
  hasRobotConnection: false,
  hasLidarKeyframe: false,
  hasSurface: false,
  lastStatusUpdate: 0,
};

export function updateEmptyStateChecklist(snapshot) {
  if (!snapshot || !checklistItems.calibration) return;

  const session = snapshot.session || {};
  const live = snapshot.live_map || {};
  const activeMap = snapshot.active_map || {};
  const buildingScan = snapshot.building_scan || {};

  // Check 1: Calibration loaded - must have calibration_id from quality gate
  const hasCalibration = Boolean(snapshot.camera?.calibration_id);
  if (hasCalibration !== lastSnapshotState.hasCalibration) {
    checklistItems.calibration.dataset.status = hasCalibration ? "success" : "checking";
    if (hasCalibration) {
      checklistItems.calibration.querySelector("span:last-child").textContent =
        `Calibration: ${snapshot.camera.calibration_id}`;
    }
    lastSnapshotState.hasCalibration = hasCalibration;
  }

  // Check 2: Robot connection
  const hasRobotConnection = Boolean(session.session_id || session.capture_active || live.last_sequence >= 0);
  if (hasRobotConnection !== lastSnapshotState.hasRobotConnection) {
    checklistItems.robot.dataset.status = hasRobotConnection ? "success" : "checking";
    if (hasRobotConnection && session.session_id) {
      checklistItems.robot.querySelector("span:last-child").textContent = `Connected: ${session.session_id}`;
    }
    lastSnapshotState.hasRobotConnection = hasRobotConnection;
  }

  // Check 3: LiDAR keyframe
  const hasLidarKeyframe = (live.last_sequence || -1) >= 0 && (activeMap.points || 0) > 0;
  if (hasLidarKeyframe !== lastSnapshotState.hasLidarKeyframe) {
    checklistItems.lidar.dataset.status = hasLidarKeyframe ? "success" :
      hasRobotConnection ? "checking" : "waiting";
    if (hasLidarKeyframe) {
      checklistItems.lidar.querySelector("span:last-child").textContent =
        `Keyframe ${live.last_sequence} received (${activeMap.points} points)`;
    }
    lastSnapshotState.hasLidarKeyframe = hasLidarKeyframe;
  }

  // Check 4: Surface built
  const hasSurface = Boolean(buildingScan.triangle_count > 0 || (Array.isArray(live.trajectory) && live.trajectory.length > 0));
  if (hasSurface !== lastSnapshotState.hasSurface) {
    checklistItems.surface.dataset.status = hasSurface ? "success" :
      hasLidarKeyframe ? "checking" : "waiting";
    if (hasSurface && buildingScan.triangle_count) {
      checklistItems.surface.querySelector("span:last-child").textContent =
        `Surface ready (${buildingScan.triangle_count} triangles)`;
    }
    lastSnapshotState.hasSurface = hasSurface;
  }

  // Update status panel for important state changes
  const now = Date.now();
  if (statusPanel && now - lastSnapshotState.lastStatusUpdate > 3000) {
    if (!hasRobotConnection && session.state === "STALE") {
      statusPanel.hidden = false;
      statusTitle.textContent = "Connection Lost";
      statusMessage.textContent = "Robot connection has become stale. Check network connectivity.";
      lastSnapshotState.lastStatusUpdate = now;
    } else if (hasRobotConnection && !hasLidarKeyframe && activeMap.quality === "AWAITING KEYFRAME") {
      statusPanel.hidden = false;
      statusTitle.textContent = "Waiting for LiDAR";
      statusMessage.textContent = "Connected to robot. Waiting for first LiDAR keyframe...";
      lastSnapshotState.lastStatusUpdate = now;
    } else if (hasSurface) {
      statusPanel.hidden = true;
    }
  }
}
