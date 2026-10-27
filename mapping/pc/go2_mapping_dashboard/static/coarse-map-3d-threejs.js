// Three.js 3D rendering for coarse map editor
// Inspired by blueprint3d-modern architecture: full perspective 3D with camera control

let scene, camera, renderer, controls, transformControls, transformControlsHelper, floorplanGroup, entitySelectionHandler, entityTransformHandler;
let isThreeJSInitialized = false;
let threeInitPromise = null;
let selectedTransformIdentity = null;
let threeTransformContainer = null;
let threeTransformCommits = 0;
let activeFloorMapId = null;
let transformDragStart = null;
let threeTransformMode = 'translate';
let threeTransformSpace = 'world';
let threeTransformSnap = true;
const furnitureAssetCache = new Map();
const pbrTextureCache = new Map();
let pbrTextureLoads = 0;
let pbrTextureCacheHits = 0;

const PBR_MATERIAL_SPECS = Object.freeze({
  wood: Object.freeze({ folder: 'WoodFloor040', realSizeM: Object.freeze([1.9, 1.9]) }),
  tile: Object.freeze({ folder: 'Tiles074', realSizeM: Object.freeze([2.1, 2.1]) }),
  concrete: Object.freeze({ folder: 'Concrete034', realSizeM: Object.freeze([1.1, 0.55]) }),
  carpet: Object.freeze({ folder: 'Carpet001', realSizeM: Object.freeze([1.0, 1.0]) }),
});

const THREEJS_CONFIG = {
  wallHeight: 2.6,
  wallThickness: 0.12,
  floorThickness: 0.2,
  cameraHeight: 8,
  cameraDistance: 12,
  roomColors: [
    0xFFD89E, // warm yellow
    0xA8D8EA, // cool blue
    0xFFB6C1, // soft pink
    0xC5E1A5, // lime green
    0xCE93D8, // lavender
  ],
  wallMaterial: { color: 0xECEFF1, roughness: 0.8, metalness: 0.1 },
  floorMaterial: { color: 0xF5F5F5, roughness: 0.9, metalness: 0.0 },
};

function materialColor(THREE, material, fallback) {
  const colors = { tile: 0xdfeae5, wood: 0xc99d6b, concrete: 0xaab5b0, carpet: 0xc8b1d6 };
  return colors[String(material || '').toLowerCase()] || fallback;
}

async function loadPbrMaterial(THREE, id) {
  const spec = PBR_MATERIAL_SPECS[String(id || '').toLowerCase()];
  if (!spec) return null;
  const { folder } = spec;
  const urls = { color: `/static/assets/textures/pbr/ambientcg/${folder}/Color.jpg`, normal: `/static/assets/textures/pbr/ambientcg/${folder}/NormalGL.jpg`, roughness: `/static/assets/textures/pbr/ambientcg/${folder}/Roughness.jpg` };
  let pending = pbrTextureCache.get(folder);
  if (pending) pbrTextureCacheHits += 1;
  else {
    const loader = new THREE.TextureLoader();
    pending = Promise.all(Object.entries(urls).map(([key, url]) => new Promise((resolve, reject) => loader.load(url, (texture) => resolve([key, texture]), undefined, reject)))).then((entries) => { pbrTextureLoads += 1; return Object.fromEntries(entries); });
    pbrTextureCache.set(folder, pending);
  }
  const cached = await pending;
  return {
    spec,
    textures: Object.fromEntries(Object.entries(cached).map(([key, texture]) => [key, texture.clone()])),
  };
}

function roomTextureRepeat(points, spec) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const roomBounds = {
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys),
  };
  return [
    Math.max(roomBounds.width / spec.realSizeM[0], 0.01),
    Math.max(roomBounds.height / spec.realSizeM[1], 0.01),
  ];
}

function furnitureGeometry(THREE, type, dimensions) {
  const [width, depth, height] = dimensions;
  switch (String(type || '').toLowerCase()) {
    case 'plant': return new THREE.CylinderGeometry(Math.min(width, depth) * 0.42, Math.min(width, depth) * 0.5, height, 12);
    case 'chair': return new THREE.CylinderGeometry(Math.min(width, depth) * 0.45, Math.min(width, depth) * 0.5, height, 12);
    case 'table': return new THREE.BoxGeometry(width, Math.max(0.06, height * 0.16), depth);
    case 'bed': return new THREE.BoxGeometry(width, height, depth);
    case 'cabinet': return new THREE.BoxGeometry(width, height, depth);
    case 'sofa': return new THREE.BoxGeometry(width, height, depth);
    default: return new THREE.BoxGeometry(width, height, depth);
  }
}

async function initThreeJS(container) {
  if (isThreeJSInitialized) return;
  if (threeInitPromise) return threeInitPromise;
  threeInitPromise = createThreeJS(container);
  try {
    await threeInitPromise;
  } finally {
    threeInitPromise = null;
  }
}

async function createThreeJS(container) {
  if (isThreeJSInitialized) return;

  const THREE = await import('./vendor/three.module.js');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xF7FBF9);
  scene.fog = new THREE.Fog(0xF7FBF9, 20, 50);

  // Camera setup (perspective, not orthographic like isometric)
  const aspect = container.clientWidth / container.clientHeight;
  camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
  camera.position.set(
    THREEJS_CONFIG.cameraDistance,
    THREEJS_CONFIG.cameraHeight,
    THREEJS_CONFIG.cameraDistance
  );
  camera.lookAt(0, 0, 0);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  threeTransformContainer = container;

  const { TransformControls } = await import('./three-examples/controls/TransformControls.js');
  transformControls = new TransformControls(camera, renderer.domElement);
  transformControlsHelper = transformControls.getHelper();
  scene.add(transformControlsHelper);
  // Register TransformControls before orbit so its picker hover runs first.
  controls = createOrbitControls(camera, renderer.domElement, THREE);
  transformControls.addEventListener('change', syncThreeTransformDiagnostics);
  transformControls.addEventListener('dragging-changed', (event) => {
    if (event.value) {
      const object = transformControls.object;
      transformDragStart = object ? {
        object,
        id: object.userData?.entityId,
        floorMapId: object.userData?.floorMapId,
        rotation: object.rotation.y,
      } : null;
    } else {
      transformDragStart = null;
    }
    controls.setEnabled(!event.value);
    syncThreeTransformDiagnostics();
  });
  transformControls.addEventListener('mouseUp', () => {
    const object = transformControls.object;
    const start = transformDragStart;
    if (!object || !start || object !== start.object || !entityTransformHandler) return;
    if (!activeFloorMapId || String(object.userData?.floorMapId || '') !== String(activeFloorMapId)) return;
    const rotation = Number(object.rotation.y);
    if (!Number.isFinite(rotation) || !Number.isFinite(start.rotation)) return;
    const wrapped = ((rotation + Math.PI) % (2 * Math.PI) + (2 * Math.PI)) % (2 * Math.PI) - Math.PI;
    if (Math.abs(rotation - start.rotation) < 1e-4) return;
    threeTransformCommits += 1;
    syncThreeTransformDiagnostics();
    entityTransformHandler({
      id: object.userData.entityId,
      kind: object.userData.kind,
      position: [object.position.x, object.position.z],
      rotation_rad: Object.is(wrapped, -0) ? 0 : wrapped,
    });
  });
  applyThreeTransformSettings();
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerDown = null;
  const pointerRay = (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
  };
  renderer.domElement.addEventListener('pointerdown', (event) => {
    if (transformControls?.axis || transformControls?.dragging) {
      pointerDown = null;
      return;
    }
    pointerDown = { x: event.clientX, y: event.clientY };
    pointerRay(event);
    const hit = raycaster.intersectObjects(floorplanGroup.children, true)
      .find((item) => item.object.userData?.entityId && item.object.userData?.kind === 'equipment');
    if (hit && hit.object.userData.locked !== true) {
      let entityObject = hit.object;
      while (entityObject.parent && entityObject.parent !== floorplanGroup && entityObject.parent.userData?.entityId === hit.object.userData.entityId) entityObject = entityObject.parent;
      if (String(entityObject.userData?.floorMapId || '') !== String(activeFloorMapId || '')) return;
      selectedTransformIdentity = { id: entityObject.userData.entityId, floorMapId: entityObject.userData.floorMapId };
      transformControls.attach(entityObject);
      syncThreeTransformDiagnostics();
      if (entitySelectionHandler) entitySelectionHandler({ id: entityObject.userData.entityId, kind: entityObject.userData.kind });
      pointerDown = null;
      return;
    }
  });
  renderer.domElement.addEventListener('pointermove', (event) => {
    if (transformControls?.dragging) pointerDown = null;
  });
  renderer.domElement.addEventListener('pointerup', (event) => {
    if (!pointerDown || Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 5) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(floorplanGroup.children, true)
      .find((item) => item.object.userData?.entityId);
    if (hit && entitySelectionHandler) {
      entitySelectionHandler({ id: hit.object.userData.entityId, kind: hit.object.userData.kind });
    }
  });

  // Lights (blueprint3d uses ambient + directional)
  const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.6);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xFFFFFF, 0.8);
  directionalLight.position.set(10, 15, 10);
  directionalLight.castShadow = true;
  directionalLight.shadow.camera.left = -20;
  directionalLight.shadow.camera.right = 20;
  directionalLight.shadow.camera.top = 20;
  directionalLight.shadow.camera.bottom = -20;
  directionalLight.shadow.mapSize.width = 2048;
  directionalLight.shadow.mapSize.height = 2048;
  scene.add(directionalLight);

  // Grid helper (floor reference)
  const gridHelper = new THREE.GridHelper(30, 30, 0xD5E2DC, 0xE6ECE9);
  gridHelper.position.y = -THREEJS_CONFIG.floorThickness;
  scene.add(gridHelper);

  floorplanGroup = new THREE.Group();
  scene.add(floorplanGroup);

  isThreeJSInitialized = true;
  animate();
}

function createOrbitControls(camera, element, THREE) {
  const state = { yaw: Math.PI / 4, pitch: 0.72, radius: 12, target: new THREE.Vector3(0, 0, 0), dragging: false, enabled: true, x: 0, y: 0 };
  const update = () => {
    const horizontal = Math.cos(state.pitch) * state.radius;
    camera.position.set(
      state.target.x + Math.cos(state.yaw) * horizontal,
      state.target.y + Math.sin(state.pitch) * state.radius,
      state.target.z + Math.sin(state.yaw) * horizontal,
    );
    camera.lookAt(state.target);
  };
  element.addEventListener('pointerdown', (event) => {
    if (!state.enabled || transformControls?.axis || transformControls?.dragging) return;
    state.dragging = true; state.x = event.clientX; state.y = event.clientY;
    element.setPointerCapture?.(event.pointerId);
  });
  element.addEventListener('pointermove', (event) => {
    if (!state.dragging || !state.enabled) return;
    state.yaw -= (event.clientX - state.x) * 0.01;
    state.pitch = Math.max(0.16, Math.min(1.35, state.pitch - (event.clientY - state.y) * 0.01));
    state.x = event.clientX; state.y = event.clientY; update();
  });
  const release = (event) => { state.dragging = false; element.releasePointerCapture?.(event.pointerId); };
  element.addEventListener('pointerup', release);
  element.addEventListener('pointercancel', release);
  element.addEventListener('wheel', (event) => {
    event.preventDefault();
    state.radius = Math.max(5, Math.min(30, state.radius * (event.deltaY < 0 ? 0.9 : 1.1)));
    update();
  }, { passive: false });
  update();
  return {
    target: state.target,
    update,
    setEnabled(enabled) { state.enabled = enabled; if (!enabled) state.dragging = false; },
    dispose() {},
  };
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

function applyThreeTransformSettings() {
  if (!transformControls) return;
  transformControls.setMode(threeTransformMode);
  transformControls.setSpace(threeTransformSpace);
  transformControls.setTranslationSnap(threeTransformSnap ? 0.1 : null);
  transformControls.setRotationSnap(threeTransformSnap ? Math.PI / 12 : null);
  transformControls.showX = threeTransformMode === 'translate';
  transformControls.showY = threeTransformMode === 'rotate';
  transformControls.showZ = threeTransformMode === 'translate';
  syncThreeTransformDiagnostics();
}

function syncThreeTransformDiagnostics() {
  if (!threeTransformContainer) return;
  threeTransformContainer.dataset.transformMode = threeTransformMode;
  threeTransformContainer.dataset.transformSpace = threeTransformSpace;
  threeTransformContainer.dataset.transformSnap = String(threeTransformSnap);
  threeTransformContainer.dataset.transformAttached = transformControls?.object?.userData?.entityId || '';
  threeTransformContainer.dataset.transformCommits = String(threeTransformCommits);
  threeTransformContainer.dataset.transformAxis = transformControls?.axis || '';
  threeTransformContainer.dataset.transformDragging = String(transformControls?.dragging === true);
  if (transformControls?.object && camera && renderer) {
    const screen = transformControls.object.getWorldPosition(new (camera.position.constructor)()).project(camera);
    const rect = renderer.domElement.getBoundingClientRect();
    threeTransformContainer.dataset.transformScreenX = String((screen.x * 0.5 + 0.5) * rect.width);
    threeTransformContainer.dataset.transformScreenY = String((-screen.y * 0.5 + 0.5) * rect.height);
  } else {
    threeTransformContainer.dataset.transformScreenX = '';
    threeTransformContainer.dataset.transformScreenY = '';
  }
}

function setThreeTransformMode(mode) {
  threeTransformMode = mode === 'rotate' ? 'rotate' : 'translate';
  applyThreeTransformSettings();
}

function setThreeTransformSpace(space) {
  threeTransformSpace = space === 'local' ? 'local' : 'world';
  applyThreeTransformSettings();
}

function setThreeTransformSnap(enabled) {
  threeTransformSnap = enabled !== false;
  applyThreeTransformSettings();
}

function clearFloorplan() {
  if (!floorplanGroup) return;
  transformControls?.detach();
  syncThreeTransformDiagnostics();
  while (floorplanGroup.children.length > 0) {
    const child = floorplanGroup.children[0];
    child.traverse((node) => {
      if (node.userData?.sharedFurnitureAsset) return;
      (node.userData?.ownedTextures || []).forEach((texture) => texture.dispose());
      if (node.geometry) node.geometry.dispose();
      if (Array.isArray(node.material)) node.material.forEach((material) => material.dispose());
      else node.material?.dispose?.();
    });
    floorplanGroup.remove(child);
  }
}

async function renderMapThreeJS(coarseMap, container, onSelectEntity, options = {}) {
  const THREE = await import('./vendor/three.module.js');

  if (!isThreeJSInitialized) {
    await initThreeJS(container);
  }

  const append = options.append === true;
  if (options.activeFloorMapId !== undefined) activeFloorMapId = String(options.activeFloorMapId || '');
  else if (!append) activeFloorMapId = String(coarseMap?.map_id || '');
  const floorOffset = Number(options.floorOffset || 0);
  let gltfAssetsLoaded = append ? Number(container.dataset.gltfAssetsLoaded || 0) : 0;
  let assetFallbacks = append ? Number(container.dataset.assetFallbacks || 0) : 0;
  let assetCacheHits = append ? Number(container.dataset.assetCacheHits || 0) : 0;
  const initialChildren = new Set(floorplanGroup?.children || []);
  if (!append) clearFloorplan();

  if (!coarseMap) return;
  entitySelectionHandler = typeof onSelectEntity === 'function' ? onSelectEntity : null;
  entityTransformHandler = typeof options.onTransformEntity === 'function' ? options.onTransformEntity : null;

  const areas = [
    ...(coarseMap.rooms || []),
    ...(coarseMap.corridors || []),
    ...(coarseMap.keep_out_zones || []),
    ...(coarseMap.carpets || []),
  ];
  await Promise.all(areas.map(async (room, idx) => {
    const points = (room.polygon || room.geometry || room.boundary?.exterior || [])
      .map((point) => Array.isArray(point) ? { x: Number(point[0]), y: Number(point[1]) } : point)
      .filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y));
    if (points.length < 3) return;

    const colorIndex = idx % THREEJS_CONFIG.roomColors.length;
    const roomColor = materialColor(THREE, room.material, THREEJS_CONFIG.roomColors[colorIndex]);

    // Floor mesh (extruded polygon)
    const shape = new THREE.Shape();
    shape.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      shape.lineTo(points[i].x, points[i].y);
    }
    shape.lineTo(points[0].x, points[0].y);

    const floorGeometry = new THREE.ExtrudeGeometry(shape, {
      depth: THREEJS_CONFIG.floorThickness,
      bevelEnabled: false,
    });
    const floorMaterial = new THREE.MeshStandardMaterial({
      color: roomColor,
      roughness: THREEJS_CONFIG.floorMaterial.roughness,
      metalness: THREEJS_CONFIG.floorMaterial.metalness,
    });
    if (room.material) {
      try {
        const loaded = await loadPbrMaterial(THREE, room.material);
        if (loaded) {
          const { spec, textures } = loaded;
          const repeat = roomTextureRepeat(points, spec);
          floorMaterial.map = textures.color;
          floorMaterial.normalMap = textures.normal;
          floorMaterial.roughnessMap = textures.roughness;
          [textures.color, textures.normal, textures.roughness].forEach((texture) => {
            texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
            texture.repeat.set(...repeat);
            texture.needsUpdate = true;
          });
          floorMaterial.needsUpdate = true;
        }
      } catch (error) { console.warn(`PBR material fallback for ${room.id}:`, error); }
    }
    const floorMesh = new THREE.Mesh(floorGeometry, floorMaterial);
    floorMesh.rotation.x = -Math.PI / 2; // Lay flat on XZ plane
    floorMesh.position.y = -THREEJS_CONFIG.floorThickness;
    floorMesh.receiveShadow = true;
    floorMesh.castShadow = true;
    floorMesh.userData = {
      entityId: room.id,
      kind: room.kind || 'room',
      ownedTextures: [floorMaterial.map, floorMaterial.normalMap, floorMaterial.roughnessMap].filter(Boolean),
    };
    floorplanGroup.add(floorMesh);

    if (room.kind === 'keep_out_zone') {
      floorMaterial.color.set(0xe9a7a2);
      floorMaterial.transparent = true;
      floorMaterial.opacity = 0.65;
    }
    if (room.kind === 'carpet') {
      floorMaterial.color.set(0xe7d7f0);
      floorMaterial.transparent = true;
      floorMaterial.opacity = 0.82;
    }

    // Walls (vertical extrusion from edges); carpets remain flat overlays.
    for (let i = 0; i < points.length; i++) {
      if (room.kind === 'carpet') break;
      const p1 = points[i];
      const p2 = points[(i + 1) % points.length];

      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const length = Math.sqrt(dx * dx + dy * dy);
      const angle = Math.atan2(dy, dx);

      const wallGeometry = new THREE.BoxGeometry(
        length,
        THREEJS_CONFIG.wallHeight,
        THREEJS_CONFIG.wallThickness
      );
      const wallMaterial = new THREE.MeshStandardMaterial(THREEJS_CONFIG.wallMaterial);
      const wallMesh = new THREE.Mesh(wallGeometry, wallMaterial);

      wallMesh.position.set(
        (p1.x + p2.x) / 2,
        THREEJS_CONFIG.wallHeight / 2,
        (p1.y + p2.y) / 2
      );
      wallMesh.rotation.y = -angle;
      wallMesh.castShadow = true;
      wallMesh.receiveShadow = true;
      wallMesh.userData = { entityId: room.id, kind: room.kind || 'room' };
      floorplanGroup.add(wallMesh);
    }

    if (room.kind === 'stair') {
      const minX = Math.min(...points.map((point) => point.x));
      const maxX = Math.max(...points.map((point) => point.x));
      const minY = Math.min(...points.map((point) => point.y));
      const maxY = Math.max(...points.map((point) => point.y));
      const profile = room.stair_profile || {};
      const steps = Math.max(1, Math.min(500, Number(profile.step_count) || 8));
      const alongX = (maxX - minX) >= (maxY - minY);
      const reverse = profile.direction === 'down';
      for (let step = 0; step < steps; step += 1) {
        const width = (alongX ? maxX - minX : maxY - minY) / steps;
        const sequence = reverse ? steps - step : step + 1;
        const rise = (Number(profile.riser_height_m) || 0.18) * sequence;
        const geometry = new THREE.BoxGeometry(alongX ? width : maxX - minX, rise, alongX ? maxY - minY : width);
        const material = new THREE.MeshStandardMaterial({ color: 0xd0a45e, roughness: 0.85, metalness: 0.02 });
        const mesh = new THREE.Mesh(geometry, material);
        if (alongX) {
          mesh.position.set(minX + width * (step + 0.5), rise / 2, (minY + maxY) / 2);
        } else {
          mesh.position.set((minX + maxX) / 2, rise / 2, minY + width * (step + 0.5));
        }
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { entityId: room.id, kind: 'stair' };
        floorplanGroup.add(mesh);
      }
    }

    // Room label (3D text - simplified as sprite for now)
    if (room.name) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = 256;
      canvas.height = 64;
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#263633';
      ctx.font = 'bold 28px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(room.name, 128, 32);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(spriteMaterial);

      const centroidX = points.reduce((sum, p) => sum + p.x, 0) / points.length;
      const centroidY = points.reduce((sum, p) => sum + p.y, 0) / points.length;
      sprite.position.set(centroidX, THREEJS_CONFIG.wallHeight + 0.5, centroidY);
      sprite.scale.set(2, 0.5, 1);
      floorplanGroup.add(sprite);
    }
  }));

  (coarseMap.walls || []).forEach((wall) => {
    const points = wall.geometry || wall.line || [];
    if (!Array.isArray(points) || points.length < 2) return;
    for (let i = 0; i < points.length - 1; i += 1) {
      const p1 = points[i];
      const p2 = points[i + 1];
      const x1 = Number(p1[0]); const z1 = Number(p1[1]);
      const x2 = Number(p2[0]); const z2 = Number(p2[1]);
      const length = Math.hypot(x2 - x1, z2 - z1);
      if (!Number.isFinite(length) || length <= 0) continue;
      const geometry = new THREE.BoxGeometry(length, THREEJS_CONFIG.wallHeight, THREEJS_CONFIG.wallThickness);
      const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x9bb8ae, roughness: 0.82 }));
      mesh.position.set((x1 + x2) / 2, THREEJS_CONFIG.wallHeight / 2, (z1 + z2) / 2);
      mesh.rotation.y = -Math.atan2(z2 - z1, x2 - x1);
      mesh.userData = { entityId: wall.id, kind: 'wall' };
      mesh.castShadow = true;
      floorplanGroup.add(mesh);
    }
  });

  (coarseMap.thresholds || []).forEach((threshold) => {
    const points = threshold.geometry || threshold.line || [];
    if (!Array.isArray(points) || points.length < 2) return;
    const p1 = points[0]; const p2 = points[1];
    const x1 = Number(p1[0]); const z1 = Number(p1[1]);
    const x2 = Number(p2[0]); const z2 = Number(p2[1]);
    const length = Math.hypot(x2 - x1, z2 - z1);
    if (!Number.isFinite(length) || length <= 0) return;
    const geometry = new THREE.BoxGeometry(length, 0.06, 0.08);
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0xd09235, roughness: 0.75 }));
    mesh.position.set((x1 + x2) / 2, 0.03, (z1 + z2) / 2);
    mesh.rotation.y = -Math.atan2(z2 - z1, x2 - x1);
    mesh.userData = { entityId: threshold.id, kind: 'threshold' };
    floorplanGroup.add(mesh);
  });

  const pointEntities = [
    ...(coarseMap.doors || []).map((entity) => ({ ...entity, kind: 'door' })),
    ...(coarseMap.features || []),
    ...(coarseMap.anchors || []).map((entity) => ({ ...entity, kind: 'anchor' })),
  ];
  const loadGltfFurniture = async (entity) => {
    if (entity.kind !== 'equipment' || entity.asset_source !== 'LOCAL_GLTF' || !entity.asset_url) return null;
    try {
      const [{ GLTFLoader }, { clone }] = await Promise.all([
        import('./three-examples/loaders/GLTFLoader.js'),
        import('./three-examples/utils/SkeletonUtils.js'),
      ]);
      let pending = furnitureAssetCache.get(entity.asset_url);
      if (pending) assetCacheHits += 1;
      else {
        const loader = new GLTFLoader();
        pending = new Promise((resolve, reject) => loader.load(entity.asset_url, (gltf) => {
          gltf.scene.traverse((node) => { node.userData = { ...(node.userData || {}), sharedFurnitureAsset: true }; });
          resolve(gltf.scene);
        }, undefined, reject));
        furnitureAssetCache.set(entity.asset_url, pending);
      }
      const root = clone(await pending);
      root.traverse((node) => { if (node.isMesh) node.userData = { entityId: entity.id, kind: entity.kind, locked: entity.locked === true, sharedFurnitureAsset: true }; });
      gltfAssetsLoaded += 1;
      return root;
    } catch (error) {
      furnitureAssetCache.delete(entity.asset_url);
      console.warn(`GLTF furniture asset fallback for ${entity.id}:`, error);
      assetFallbacks += 1;
      return null;
    }
  };
  await Promise.all(pointEntities.map(async (entity) => {
    const position = entity.position;
    if (!Array.isArray(position) || position.length !== 2) return;
    const color = entity.kind === 'door' ? 0x2b9b86 : entity.kind === 'anchor' ? 0x75a85b : 0xd09235;
    const dimensions = Array.isArray(entity.dimensions_m) && entity.dimensions_m.length === 3
      ? entity.dimensions_m.map(Number) : null;
    const gltfRoot = await loadGltfFurniture(entity);
    const geometry = dimensions && dimensions.every((value) => Number.isFinite(value) && value > 0)
      ? furnitureGeometry(THREE, entity.furniture_type, dimensions)
      : new THREE.CylinderGeometry(0.12, 0.12, 0.22, 16);
    const marker = gltfRoot || new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: materialColor(THREE, entity.material, color), roughness: 0.8 }));
    marker.position.set(Number(position[0]), gltfRoot ? 0 : dimensions ? dimensions[2] / 2 : 0.11, Number(position[1]));
    marker.rotation.y = Number(entity.rotation_rad || 0);
    marker.userData = { ...(marker.userData || {}), entityId: entity.id, kind: entity.kind, locked: entity.locked === true, floorMapId: coarseMap.map_id };
    if (gltfRoot && dimensions) {
      const box = new THREE.Box3().setFromObject(marker);
      const size = box.getSize(new THREE.Vector3());
      if (size.x > 0 && size.y > 0 && size.z > 0) marker.scale.set(dimensions[0] / size.x, dimensions[2] / size.y, dimensions[1] / size.z);
    }
    floorplanGroup.add(marker);
  }));
  container.dataset.gltfAssetsLoaded = String(gltfAssetsLoaded);
  container.dataset.assetFallbacks = String(assetFallbacks);
  container.dataset.assetCacheHits = String(assetCacheHits);
  container.dataset.pbrTextureLoads = String(pbrTextureLoads);
  container.dataset.pbrTextureCacheHits = String(pbrTextureCacheHits);

  floorplanGroup.children.forEach((child) => {
    if (initialChildren.has(child)) return;
    child.position.y += floorOffset;
    child.userData = { ...(child.userData || {}), floorMapId: coarseMap.map_id };
  });
  if (selectedTransformIdentity) {
    const selected = floorplanGroup.children.find((child) => (
      child.userData?.entityId === selectedTransformIdentity.id
      && (!selectedTransformIdentity.floorMapId || child.userData?.floorMapId === selectedTransformIdentity.floorMapId)
      && child.userData?.locked !== true
    ));
    if (selected) transformControls?.attach(selected);
    else transformControls?.detach();
    syncThreeTransformDiagnostics();
  }

  // Center camera on floorplan bounds
  const box = new THREE.Box3().setFromObject(floorplanGroup);
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  controls.update();
  syncThreeTransformDiagnostics();
}

async function renderBuildingMapsThreeJS(floorMaps, container, onSelectEntity, currentMapId, onTransformEntity) {
  const maps = Array.isArray(floorMaps) ? floorMaps.filter(Boolean) : [];
  if (!maps.length) return;
  for (let index = 0; index < maps.length; index += 1) {
    await renderMapThreeJS(maps[index], container, onSelectEntity, {
      append: index > 0,
      floorOffset: index * 3.0,
      activeFloorMapId: currentMapId,
      onTransformEntity,
    });
  }
  // Keep the active floor visually distinct without changing canonical data.
  if (floorplanGroup) {
    floorplanGroup.children.forEach((child) => {
      if (child.userData?.floorMapId) child.material?.emissive?.set(child.userData.floorMapId === currentMapId ? 0x24483b : 0x000000);
    });
  }
}

function handleResize(container) {
  if (!camera || !renderer) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function disposeThreeJS() {
  clearFloorplan();
  if (renderer) {
    renderer.dispose();
    if (renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
  }
  if (controls) controls.dispose();
  if (transformControls) transformControls.dispose();
  if (transformControlsHelper?.parent) transformControlsHelper.parent.remove(transformControlsHelper);
  scene = camera = renderer = controls = transformControls = transformControlsHelper = floorplanGroup = null;
  selectedTransformIdentity = null;
  threeTransformContainer = null;
  isThreeJSInitialized = false;
}

export {
  renderMapThreeJS,
  renderBuildingMapsThreeJS,
  setThreeTransformMode,
  setThreeTransformSpace,
  setThreeTransformSnap,
  handleResize,
  disposeThreeJS,
  isThreeJSInitialized,
};
