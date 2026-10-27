import { sha256Hex } from "./sha256.js";

const VERTEX_PROPERTIES = [
  "property double x",
  "property double y",
  "property double z",
  "property double nx",
  "property double ny",
  "property double nz",
  "property uchar red",
  "property uchar green",
  "property uchar blue",
];
const FACE_PROPERTY = "property list uchar uint vertex_indices";
const VERTEX_BYTES = 51;
const FACE_BYTES = 13;

function headerEnd(bytes) {
  const marker = new TextEncoder().encode("end_header\n");
  const limit = Math.min(bytes.length, 65536);
  for (let offset = 0; offset <= limit - marker.length; offset += 1) {
    let matches = true;
    for (let index = 0; index < marker.length; index += 1) {
      if (bytes[offset + index] !== marker[index]) {
        matches = false;
        break;
      }
    }
    if (matches) return offset + marker.length;
  }
  throw new Error("legacy PLY header is missing or too large");
}

function elementCount(line, name) {
  const prefix = `element ${name} `;
  if (!line.startsWith(prefix)) throw new Error(`legacy PLY ${name} element is missing`);
  const value = Number(line.slice(prefix.length));
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`legacy PLY ${name} count is invalid`);
  return value;
}

function parseHeader(buffer) {
  const bytes = new Uint8Array(buffer);
  const dataOffset = headerEnd(bytes);
  const lines = new TextDecoder("ascii", { fatal: true })
    .decode(bytes.subarray(0, dataOffset))
    .replaceAll("\r\n", "\n")
    .split("\n")
    .filter((line) => line !== "");
  if (lines[0] !== "ply" || lines[1] !== "format binary_little_endian 1.0"
      || lines.at(-1) !== "end_header") {
    throw new Error("legacy PLY must use binary little-endian version 1.0");
  }
  const content = lines.slice(2, -1).filter((line) => !line.startsWith("comment "));
  const vertexCount = elementCount(content[0] || "", "vertex");
  const vertexEnd = 1 + VERTEX_PROPERTIES.length;
  if (JSON.stringify(content.slice(1, vertexEnd)) !== JSON.stringify(VERTEX_PROPERTIES)) {
    throw new Error("legacy PLY vertex layout is unsupported");
  }
  const triangleCount = elementCount(content[vertexEnd] || "", "face");
  if (content.length !== vertexEnd + 2 || content[vertexEnd + 1] !== FACE_PROPERTY) {
    throw new Error("legacy PLY face layout is unsupported");
  }
  const expectedBytes = dataOffset + vertexCount * VERTEX_BYTES + triangleCount * FACE_BYTES;
  if (buffer.byteLength !== expectedBytes) throw new Error("legacy PLY byte count does not match its header");
  return { dataOffset, vertexCount, triangleCount };
}

export async function parseLegacyPlySurface(buffer, options = {}) {
  if (!(buffer instanceof ArrayBuffer)) throw new Error("legacy PLY payload is not an ArrayBuffer");
  if (!options.skipSha256) {
    if (!/^[0-9a-f]{64}$/.test(options.expectedSha256 || "")) {
      throw new Error("legacy PLY checksum identity is invalid");
    }
    if (await sha256Hex(buffer) !== options.expectedSha256) {
      throw new Error("legacy PLY checksum mismatch");
    }
  }
  const layout = parseHeader(buffer);
  if (layout.vertexCount !== options.expectedVertexCount
      || layout.triangleCount !== options.expectedTriangleCount) {
    throw new Error("legacy PLY counts do not match status");
  }

  const view = new DataView(buffer);
  const positions = new Float32Array(layout.vertexCount * 3);
  const normals = new Float32Array(layout.vertexCount * 3);
  const colors = new Uint8Array(layout.vertexCount * 3);
  let offset = layout.dataOffset;
  for (let index = 0; index < layout.vertexCount; index += 1) {
    const x = view.getFloat64(offset, true); offset += 8;
    const y = view.getFloat64(offset, true); offset += 8;
    const z = view.getFloat64(offset, true); offset += 8;
    const nx = view.getFloat64(offset, true); offset += 8;
    const ny = view.getFloat64(offset, true); offset += 8;
    const nz = view.getFloat64(offset, true); offset += 8;
    if (![x, y, z, nx, ny, nz].every(Number.isFinite)) {
      throw new Error("legacy PLY contains a non-finite vector");
    }
    const target = index * 3;
    positions[target] = x;
    positions[target + 1] = z;
    positions[target + 2] = -y;
    normals[target] = nx;
    normals[target + 1] = nz;
    normals[target + 2] = -ny;
    colors[target] = view.getUint8(offset++);
    colors[target + 1] = view.getUint8(offset++);
    colors[target + 2] = view.getUint8(offset++);
  }

  const indices = new Uint32Array(layout.triangleCount * 3);
  for (let triangle = 0; triangle < layout.triangleCount; triangle += 1) {
    if (view.getUint8(offset++) !== 3) throw new Error("legacy PLY contains a non-triangular face");
    const target = triangle * 3;
    for (let corner = 0; corner < 3; corner += 1) {
      const vertexIndex = view.getUint32(offset, true);
      offset += 4;
      if (vertexIndex >= layout.vertexCount) throw new Error("legacy PLY triangle index is out of range");
      indices[target + corner] = vertexIndex;
    }
  }
  return {
    positions,
    normals,
    colors,
    indices,
    vertexCount: layout.vertexCount,
    triangleCount: layout.triangleCount,
  };
}

if (typeof self !== "undefined" && typeof self.addEventListener === "function") {
  self.addEventListener("message", async (event) => {
    const { id, buffer, expectedSha256, expectedVertexCount, expectedTriangleCount } = event.data || {};
    try {
      const result = await parseLegacyPlySurface(buffer, {
        expectedSha256,
        expectedVertexCount,
        expectedTriangleCount,
      });
      const transfer = [
        result.positions.buffer,
        result.normals.buffer,
        result.colors.buffer,
        result.indices.buffer,
      ];
      self.postMessage({ id, ok: true, result }, transfer);
    } catch (error) {
      self.postMessage({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  });
}
