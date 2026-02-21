"""Export visible scene meshes to GLB (binary glTF 2.0) for Blender import.

GLB format:
  12-byte header | JSON chunk | BIN chunk

Each visible mesh becomes a glTF mesh primitive with:
  - POSITION accessor (vec3 float32)
  - NORMAL accessor (vec3 float32)
  - indices accessor (scalar uint32)
  - PBR material with baseColorFactor from mesh material color

Scene hierarchy is flattened: each mesh gets its world transform baked
into vertex positions/normals so Blender receives them in the correct pose.
"""

import json
import struct
import logging
from pathlib import Path

import numpy as np

from faceforge.core.scene_graph import Scene
from faceforge.core.mesh import MeshInstance
from faceforge.core.math_utils import Mat4

logger = logging.getLogger(__name__)

# glTF constants
GLTF_FLOAT = 5126       # GL_FLOAT
GLTF_UNSIGNED_INT = 5125  # GL_UNSIGNED_INT
GLTF_ARRAY_BUFFER = 34962
GLTF_ELEMENT_ARRAY_BUFFER = 34963


def export_glb(scene: Scene, path: str | Path) -> int:
    """Export all visible meshes from the scene to a GLB file.

    Parameters
    ----------
    scene : Scene
        The scene graph to export.
    path : str or Path
        Output file path (should end with .glb).

    Returns
    -------
    int
        Number of meshes exported.
    """
    path = Path(path)
    scene.update()
    mesh_pairs = scene.collect_meshes()

    if not mesh_pairs:
        logger.warning("No visible meshes to export")
        return 0

    # Build glTF JSON + binary buffer
    gltf, bin_data = _build_gltf(mesh_pairs)
    json_str = json.dumps(gltf, separators=(",", ":"))

    # Pad JSON to 4-byte alignment
    json_bytes = json_str.encode("utf-8")
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad

    # Pad binary to 4-byte alignment
    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_data += b"\x00" * bin_pad

    # GLB header
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack("<III", 0x46546C67, 2, total_length)  # magic, version, length

    # JSON chunk
    json_chunk_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # length, JSON

    # BIN chunk
    bin_chunk_header = struct.pack("<II", len(bin_data), 0x004E4942)  # length, BIN

    with open(path, "wb") as f:
        f.write(header)
        f.write(json_chunk_header)
        f.write(json_bytes)
        f.write(bin_chunk_header)
        f.write(bin_data)

    count = len(mesh_pairs)
    logger.info("Exported %d meshes to %s (%.1f MB)", count, path,
                total_length / (1024 * 1024))
    return count


def _build_gltf(
    mesh_pairs: list[tuple[MeshInstance, Mat4]],
) -> tuple[dict, bytes]:
    """Build glTF JSON document and binary buffer from mesh list."""
    buffers_list: list[bytes] = []
    buffer_views = []
    accessors = []
    meshes = []
    nodes = []
    materials = []
    material_cache: dict[tuple, int] = {}

    byte_offset = 0

    for mesh, world_mat in mesh_pairs:
        geom = mesh.geometry
        if geom.vertex_count == 0:
            continue

        # Get or create material
        mat_key = (mesh.material.color, round(mesh.material.opacity, 3))
        if mat_key not in material_cache:
            mat_idx = len(materials)
            material_cache[mat_key] = mat_idx
            r, g, b = mesh.material.color
            a = mesh.material.opacity
            mat = {
                "name": mesh.name or f"material_{mat_idx}",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [r, g, b, a],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
            }
            if a < 1.0:
                mat["alphaMode"] = "BLEND"
            if mesh.material.double_sided:
                mat["doubleSided"] = True
            materials.append(mat)

        mat_index = material_cache[mat_key]

        # Bake world transform into positions and normals
        positions = geom.positions.reshape(-1, 3).copy()
        normals = geom.normals.reshape(-1, 3).copy()

        # Transform positions: P' = M * P (homogeneous)
        rot_scale = world_mat[:3, :3]
        translation = world_mat[:3, 3]
        positions = (rot_scale @ positions.T).T + translation

        # Transform normals: N' = (M^-T)[:3,:3] * N
        try:
            normal_mat = np.linalg.inv(rot_scale).T
        except np.linalg.LinAlgError:
            normal_mat = rot_scale  # fallback if singular
        normals = (normal_mat @ normals.T).T
        # Renormalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        normals = normals / lengths

        positions = positions.astype(np.float32)
        normals = normals.astype(np.float32)

        # Build indices (generate sequential if non-indexed)
        if geom.has_indices:
            indices = geom.indices.astype(np.uint32)
        else:
            indices = np.arange(geom.vertex_count, dtype=np.uint32)

        # Compute bounds for positions
        pos_min = positions.min(axis=0).tolist()
        pos_max = positions.max(axis=0).tolist()

        # --- Write position buffer view ---
        pos_bytes = positions.tobytes()
        pos_bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(pos_bytes),
            "target": GLTF_ARRAY_BUFFER,
        })
        buffers_list.append(pos_bytes)
        byte_offset += len(pos_bytes)

        pos_acc_idx = len(accessors)
        accessors.append({
            "bufferView": pos_bv_idx,
            "componentType": GLTF_FLOAT,
            "count": len(positions),
            "type": "VEC3",
            "min": pos_min,
            "max": pos_max,
        })

        # --- Write normal buffer view ---
        norm_bytes = normals.tobytes()
        norm_bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(norm_bytes),
            "target": GLTF_ARRAY_BUFFER,
        })
        buffers_list.append(norm_bytes)
        byte_offset += len(norm_bytes)

        norm_acc_idx = len(accessors)
        accessors.append({
            "bufferView": norm_bv_idx,
            "componentType": GLTF_FLOAT,
            "count": len(normals),
            "type": "VEC3",
        })

        # --- Write index buffer view ---
        idx_bytes = indices.tobytes()
        idx_bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(idx_bytes),
            "target": GLTF_ELEMENT_ARRAY_BUFFER,
        })
        buffers_list.append(idx_bytes)
        byte_offset += len(idx_bytes)

        idx_acc_idx = len(accessors)
        accessors.append({
            "bufferView": idx_bv_idx,
            "componentType": GLTF_UNSIGNED_INT,
            "count": len(indices),
            "type": "SCALAR",
            "min": [int(indices.min())],
            "max": [int(indices.max())],
        })

        # --- Mesh primitive ---
        mesh_idx = len(meshes)
        meshes.append({
            "name": mesh.name or f"mesh_{mesh_idx}",
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc_idx,
                    "NORMAL": norm_acc_idx,
                },
                "indices": idx_acc_idx,
                "material": mat_index,
            }],
        })

        # --- Node ---
        nodes.append({
            "name": mesh.name or f"node_{mesh_idx}",
            "mesh": mesh_idx,
        })

    # Assemble binary buffer
    bin_data = b"".join(buffers_list)

    # Build glTF document
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "FaceForge Anatomical Viewer",
        },
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_data)}],
    }

    return gltf, bin_data
