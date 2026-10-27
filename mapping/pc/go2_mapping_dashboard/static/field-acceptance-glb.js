import { GLTFLoader } from "/static/three-examples/loaders/GLTFLoader.js";

const artifact = new URLSearchParams(window.location.search).get("artifact");
const canonical = /^\/api\/reconstruction\/artifacts\/(?:tile-[0-9a-f]{64}-[0-9a-f]{64}-[0-9a-f]{64}|building-[0-9a-f]{64}-[0-9a-f]{64})\.glb$/;

function finish(state, detail = "") {
  document.documentElement.dataset.state = state;
  document.documentElement.dataset.detail = detail;
  document.title = state;
}

if (!artifact || !canonical.test(artifact)) {
  finish("THREEJS_REJECTED", "INVALID_ARTIFACT_URL");
} else {
  new GLTFLoader().load(
    artifact,
    (gltf) => {
      let meshes = 0;
      let vertices = 0;
      gltf.scene.traverse((node) => {
        if (!node.isMesh || !node.geometry) return;
        meshes += 1;
        vertices += node.geometry.getAttribute("position")?.count || 0;
      });
      finish(meshes > 0 && vertices > 0 ? "THREEJS_LOADED" : "THREEJS_REJECTED", String(meshes) + ":" + String(vertices));
    },
    undefined,
    (error) => finish("THREEJS_REJECTED", String(error)),
  );
}
