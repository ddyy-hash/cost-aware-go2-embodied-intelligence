// 3D isometric rendering for coarse map editor
// Converts 2D polygon map data into isometric 3D view matching Ecovacs aesthetic

const ISOMETRIC_CONFIG = {
  angle: Math.PI / 6, // 30° isometric angle
  wallHeight: 2.5, // meters
  floorThickness: 0.15,
  wallThickness: 0.12,
  roomColors: [
    { base: "#FFD89E", accent: "#FFC266" }, // warm yellow
    { base: "#A8D8EA", accent: "#7CBDD6" }, // cool blue
    { base: "#FFB6C1", accent: "#FF9AA2" }, // soft pink
    { base: "#C5E1A5", accent: "#AED581" }, // lime green
    { base: "#CE93D8", accent: "#BA68C8" }, // lavender
  ],
  floorFill: "#E8EAF6",
  wallStroke: "#37474F",
  shadowAlpha: 0.15,
};

function projectIsometric(x, y, z) {
  const cos30 = Math.cos(ISOMETRIC_CONFIG.angle);
  const sin30 = Math.sin(ISOMETRIC_CONFIG.angle);
  return {
    x: (x - y) * cos30,
    y: (x + y) * sin30 - z,
  };
}

function renderRoomIsometric(room, colorIndex) {
  const color = ISOMETRIC_CONFIG.roomColors[colorIndex % ISOMETRIC_CONFIG.roomColors.length];
  const points = (room.polygon || room.geometry || room.boundary?.exterior || [])
    .map((point) => Array.isArray(point) ? { x: Number(point[0]), y: Number(point[1]) } : point)
    .filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y));
  if (points.length < 3) return null;

  const paths = [];
  const h = ISOMETRIC_CONFIG.wallHeight;
  const t = ISOMETRIC_CONFIG.floorThickness;

  // Calculate centroid for depth sorting
  const centroidX = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const centroidY = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  const depth = centroidX + centroidY;

  // Floor top surface (with hatching pattern)
  const floorTop = points.map((p) => projectIsometric(p.x, p.y, 0));
  paths.push({
    type: "floor-top",
    points: floorTop,
    fill: color.base,
    stroke: ISOMETRIC_CONFIG.wallStroke,
    pattern: "hatch-diagonal",
    depth,
  });

  // Floor thickness side faces (visible edges)
  for (let i = 0; i < points.length; i++) {
    const p1 = points[i];
    const p2 = points[(i + 1) % points.length];
    const iso1_top = projectIsometric(p1.x, p1.y, 0);
    const iso2_top = projectIsometric(p2.x, p2.y, 0);
    const iso1_bot = projectIsometric(p1.x, p1.y, -t);
    const iso2_bot = projectIsometric(p2.x, p2.y, -t);

    // Only render if edge faces toward camera (simple depth check)
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const normal = { x: -dy, y: dx };
    if (normal.x + normal.y > 0) {
      paths.push({
        type: "floor-side",
        points: [iso1_top, iso2_top, iso2_bot, iso1_bot],
        fill: color.accent,
        stroke: ISOMETRIC_CONFIG.wallStroke,
        depth,
      });
    }
  }

  // Wall faces (vertical extrusion)
  for (let i = 0; i < points.length; i++) {
    const p1 = points[i];
    const p2 = points[(i + 1) % points.length];
    const iso1_bot = projectIsometric(p1.x, p1.y, 0);
    const iso2_bot = projectIsometric(p2.x, p2.y, 0);
    const iso1_top = projectIsometric(p1.x, p1.y, h);
    const iso2_top = projectIsometric(p2.x, p2.y, h);

    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const normal = { x: -dy, y: dx };
    if (normal.x + normal.y > 0) {
      paths.push({
        type: "wall",
        points: [iso1_bot, iso2_bot, iso2_top, iso1_top],
        fill: "#ECEFF1",
        stroke: ISOMETRIC_CONFIG.wallStroke,
        depth: depth + h,
      });
    }
  }

  return { room_id: room.room_id || room.id, name: room.name, paths };
}

function renderMapIsometric(coarseMap) {
  if (!coarseMap) return [];
  const areas = [
    ...(coarseMap.rooms || []),
    ...(coarseMap.corridors || []),
    ...(coarseMap.keep_out_zones || []),
    ...(coarseMap.carpets || []),
  ];
  const result = areas.map((room, idx) => renderRoomIsometric(room, idx)).filter(Boolean);
  (coarseMap.walls || []).forEach((wall) => {
    const points = wall.geometry || wall.line || [];
    for (let i = 0; i < points.length - 1; i += 1) {
      const p1 = { x: Number(points[i][0]), y: Number(points[i][1]) };
      const p2 = { x: Number(points[i + 1][0]), y: Number(points[i + 1][1]) };
      if (![p1.x, p1.y, p2.x, p2.y].every(Number.isFinite)) continue;
      result.push({ room_id: wall.id, name: wall.name || wall.id, paths: [{
        type: "wall", points: [projectIsometric(p1.x, p1.y, 0), projectIsometric(p2.x, p2.y, 0)],
        fill: "none", stroke: "#71978b", depth: p1.x + p1.y + p2.x + p2.y,
      }] });
    }
  });
  (coarseMap.thresholds || []).forEach((threshold) => {
    const points = threshold.geometry || threshold.line || [];
    for (let i = 0; i < points.length - 1; i += 1) {
      const p1 = { x: Number(points[i][0]), y: Number(points[i][1]) };
      const p2 = { x: Number(points[i + 1][0]), y: Number(points[i + 1][1]) };
      if (![p1.x, p1.y, p2.x, p2.y].every(Number.isFinite)) continue;
      result.push({ room_id: threshold.id, name: threshold.name || threshold.id, paths: [{
        type: "wall", points: [projectIsometric(p1.x, p1.y, 0), projectIsometric(p2.x, p2.y, 0)],
        fill: "none", stroke: "#d09235", depth: p1.x + p1.y + p2.x + p2.y,
      }] });
    }
  });
  return result;
}

function createIsometricSVG(geometry, viewport) {
  const fragments = [];

  // Define hatch pattern once
  fragments.push(`<defs>
    <pattern id="hatch-diagonal" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="8" stroke="${ISOMETRIC_CONFIG.wallStroke}" stroke-width="0.5" opacity="0.15"/>
    </pattern>
  </defs>`);

  // Render paths in painter's algorithm order (back to front by depth)
  const allPaths = geometry.flatMap((g) => g.paths);
  const sortedPaths = allPaths.sort((a, b) => {
    const depthDiff = (a.depth || 0) - (b.depth || 0);
    if (Math.abs(depthDiff) > 0.01) return depthDiff;
    const order = { "floor-side": 1, "floor-top": 2, wall: 3 };
    return (order[a.type] || 0) - (order[b.type] || 0);
  });

  sortedPaths.forEach((path) => {
    const pathData = `M ${path.points.map((p) => `${p.x},${p.y}`).join(" L ")}${path.type === "wall" && path.points.length === 2 ? "" : " Z"}`;
    const fill = path.pattern ? `url(#${path.pattern})` : path.fill;
    fragments.push(
      `<path d="${pathData}" fill="${fill}" stroke="${path.stroke}" stroke-width="${path.type === "wall" ? 3 : 1.5}" stroke-linejoin="miter"/>`
    );
  });

  return fragments.join("\n");
}

export { renderMapIsometric, createIsometricSVG, projectIsometric, ISOMETRIC_CONFIG };
