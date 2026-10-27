import { sha256Hex } from "./sha256.js";

const provenanceNames = { 1: "MEASURED_LIDAR", 2: "MEASURED_RGBD", 3: "NEURAL_MULTIVIEW" };

async function parseSurfaceTile(buffer, expectedSha256) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 20) throw new Error("surface tile header is truncated");
  if (await sha256Hex(buffer) !== expectedSha256) throw new Error("surface tile checksum mismatch");
  const view = new DataView(buffer);
  const binarySchemaVersion = view.getUint32(4, true);
  if (String.fromCharCode(...new Uint8Array(buffer, 0, 4)) !== "G2TM"
      || !(binarySchemaVersion === 1 || binarySchemaVersion === 2)) {
    throw new Error("surface tile header is invalid");
  }
  const vertexCount = view.getUint32(8, true);
  const triangleCount = view.getUint32(12, true);
  const scaleCount = view.getUint32(16, true);
  const positionsOffset = 20;
  const normalsOffset = positionsOffset + vertexCount * 12;
  const indicesOffset = normalsOffset + vertexCount * 12;
  const scalesOffset = indicesOffset + triangleCount * 12;
  const uncertaintyOffset = scalesOffset + scaleCount * 4;
  const provenanceOffset = uncertaintyOffset + (binarySchemaVersion === 2 ? triangleCount * 4 : 0);
  const expectedBytes = provenanceOffset + (binarySchemaVersion === 2 ? triangleCount : 0);
  if (scaleCount !== triangleCount || expectedBytes !== buffer.byteLength) {
    throw new Error("surface tile byte count is invalid");
  }
  const positions = new Float32Array(new Float32Array(buffer, positionsOffset, vertexCount * 3));
  const normals = new Float32Array(new Float32Array(buffer, normalsOffset, vertexCount * 3));
  const indices = new Uint32Array(new Uint32Array(buffer, indicesOffset, triangleCount * 3));
  const triangleScales = new Float32Array(new Float32Array(buffer, scalesOffset, triangleCount));
  const triangleUncertainty = binarySchemaVersion === 2
    ? new Float32Array(new Float32Array(buffer, uncertaintyOffset, triangleCount))
    : new Float32Array(triangleScales);
  const triangleProvenance = binarySchemaVersion === 2
    ? new Uint8Array(new Uint8Array(buffer, provenanceOffset, triangleCount))
    : new Uint8Array(triangleCount).fill(1);
  const provenanceCounts = {};
  let maximumTriangleUncertaintyM = 0;
  for (let index = 0; index < triangleCount; index += 1) {
    const provenance = provenanceNames[triangleProvenance[index]];
    const uncertaintyM = triangleUncertainty[index];
    if (!provenance || !Number.isFinite(uncertaintyM) || uncertaintyM < 0 || uncertaintyM > 1) {
      throw new Error("surface tile triangle evidence is invalid");
    }
    provenanceCounts[provenance] = (provenanceCounts[provenance] || 0) + 1;
    maximumTriangleUncertaintyM = Math.max(maximumTriangleUncertaintyM, uncertaintyM);
  }
  if (indices.some((value) => value >= vertexCount)) throw new Error("surface tile index is out of range");
  return {
    binarySchemaVersion,
    positions,
    normals,
    indices,
    triangleScales,
    triangleProvenance,
    triangleUncertainty,
    provenanceCounts,
    maximumTriangleUncertaintyM,
  };
}

self.onmessage = async (event) => {
  const { id, operation, buffer, expectedSha256 } = event.data || {};
  try {
    if (operation === "verify_bytes") {
      if (await sha256Hex(buffer) !== expectedSha256) throw new Error("artifact checksum mismatch");
      self.postMessage({ id, ok: true, result: { buffer } }, [buffer]);
      return;
    }
    if (operation !== "parse_surface_tile") throw new Error("surface worker operation is unsupported");
    const result = await parseSurfaceTile(buffer, expectedSha256);
    const transfer = [
      result.positions.buffer,
      result.normals.buffer,
      result.indices.buffer,
      result.triangleScales.buffer,
      result.triangleProvenance.buffer,
      result.triangleUncertainty.buffer,
    ];
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error?.message || error) });
  }
};
