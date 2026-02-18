"""Load skull and face mesh data from extracted JSON files."""

from typing import Optional

import numpy as np

from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material, RenderMode
from faceforge.core.config_loader import load_meshdata


def load_skull_meshes() -> dict[str, MeshInstance]:
    """Load skull mesh groups from extracted JSON data.

    skull.json format:
      vertices: [[x,y,z], ...] — 40062 vertex positions (flat list of 3-arrays)
      groups: {group_name: [[[v0,v1,v2], ...], ...]} — face index lists per group

    Returns dict of named mesh instances: jaw, teeth, teeth_lower, cranium.
    """
    data = load_meshdata("skull.json")
    all_verts = np.array(data["vertices"], dtype=np.float32)  # (N, 3)

    # Map short names for convenience
    GROUP_NAMES = {
        "12140_Skull_v3_jaw": "jaw",
        "12140_Skull_v3_teeth": "upper_teeth",
        "12140_Skull_v3_teeth_lower": "lower_teeth",
        "12140_Skull_v3": "cranium",
    }
    GROUP_COLORS = {
        "jaw": 0xc89a6a,
        "upper_teeth": 0xe8e0d0,
        "lower_teeth": 0xe8e0d0,
        "cranium": 0xd4a574,
    }

    meshes = {}
    for orig_name, face_list in data["groups"].items():
        short_name = GROUP_NAMES.get(orig_name, orig_name)

        # face_list is a list of [v0, v1, v2] triangles
        faces = np.array(face_list, dtype=np.uint32).ravel()

        # Collect all unique vertex indices used by this group
        unique_indices = np.unique(faces)
        # Build a compact vertex array and remap face indices
        remap = np.zeros(len(all_verts), dtype=np.uint32)
        remap[unique_indices] = np.arange(len(unique_indices), dtype=np.uint32)

        positions = all_verts[unique_indices].ravel().astype(np.float32)
        remapped_faces = remap[faces]

        geom = BufferGeometry(
            positions=positions,
            normals=np.zeros_like(positions),
            indices=remapped_faces,
            vertex_count=len(unique_indices),
        )
        geom.compute_normals()

        color = GROUP_COLORS.get(short_name, 0xd4a574)
        mat = Material.from_hex(
            color, render_mode=RenderMode.WIREFRAME,
            opacity=0.7, transparent=True, double_sided=True,
        )

        mesh = MeshInstance(name=short_name, geometry=geom, material=mat)
        mesh.store_rest_pose()
        meshes[short_name] = mesh

    return meshes


def load_face_mesh() -> MeshInstance:
    """Load face mesh (468 MediaPipe landmarks).

    face.json format:
      vertices: [[x,y,z], ...] — 468 vertex positions
      faces: [[v0,v1,v2], ...] — 898 triangles

    Returns a MeshInstance with the face geometry.
    """
    data = load_meshdata("face.json")

    vertices = np.array(data["vertices"], dtype=np.float32)  # (468, 3)
    faces_raw = np.array(data["faces"], dtype=np.uint32)  # (898, 3)

    vert_count = len(vertices)
    positions = vertices.ravel().astype(np.float32)
    normals = np.zeros_like(positions)
    face_indices = faces_raw.ravel()

    geom = BufferGeometry(
        positions=positions,
        normals=normals,
        indices=face_indices,
        vertex_count=vert_count,
    )
    geom.compute_normals()

    mat = Material(
        color=(0xe0 / 255, 0xb8 / 255, 0x98 / 255),  # Skin tone 0xe0b898
        opacity=0.45,
        transparent=True,
        render_mode=RenderMode.SOLID,
    )

    mesh = MeshInstance(name="face", geometry=geom, material=mat)
    mesh.store_rest_pose()
    return mesh


def load_face_landmarks() -> Optional[np.ndarray]:
    """Load face landmark data if available."""
    try:
        data = load_meshdata("landmarks.json")
        if isinstance(data, list):
            return np.array(data, dtype=np.float32).reshape(-1, 3)
        return np.array(data.get("landmarks", data), dtype=np.float32).reshape(-1, 3)
    except (FileNotFoundError, KeyError, ValueError):
        return None
