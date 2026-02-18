"""Build face mesh with positioning for skull alignment.

Loads the 468-vertex MediaPipe face mesh and positions it relative to the
skull using the default face alignment constants (scale, offset, rotation).

The alignment transform lives on an intermediate ``faceAlignment`` node so
that the parent ``faceGroup`` node can be freely rotated by the head-rotation
system without losing the alignment.
"""

import numpy as np

from faceforge.core.math_utils import deg_to_rad, quat_from_euler
from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.constants import (
    DEFAULT_FACE_SCALE,
    DEFAULT_FACE_OFFSET_X,
    DEFAULT_FACE_OFFSET_Y,
    DEFAULT_FACE_OFFSET_Z,
    DEFAULT_FACE_ROT_X_DEG,
    FACE_VERT_COUNT,
)
from faceforge.loaders.asset_manager import AssetManager


def build_face(asset_manager: AssetManager) -> tuple[SceneNode, MeshInstance]:
    """Build the face scene-graph node with properly aligned mesh.

    Scene-graph structure::

        faceGroup  (no transform -- head rotation applied here)
          └─ faceAlignment  (alignment: scale, position, rotation)
               └─ face  (mesh)

    The alignment constants are:

    * Scale: 1.14 (uniform)
    * Position X offset: -0.2
    * Position Y offset: -10.6
    * Position Z offset: 9.5
    * Rotation about X: 88.5 degrees

    Args:
        asset_manager: Provides loaded face mesh via ``load_face()``.

    Returns:
        Tuple of (faceGroup SceneNode, face MeshInstance).
    """
    face_mesh = asset_manager.load_face()

    # Outer group -- head rotation system sets quaternion/position here
    face_group = SceneNode(name="faceGroup")

    # Intermediate alignment node -- preserves alignment independently
    align_node = SceneNode(name="faceAlignment")
    _apply_alignment_to_node(align_node)
    face_group.add(align_node)

    # Mesh node
    face_node = SceneNode(name="face")
    face_node.mesh = face_mesh
    align_node.add(face_node)

    return face_group, face_mesh


def _apply_alignment_to_node(
    node: SceneNode,
    scale: float = DEFAULT_FACE_SCALE,
    offset_x: float = DEFAULT_FACE_OFFSET_X,
    offset_y: float = DEFAULT_FACE_OFFSET_Y,
    offset_z: float = DEFAULT_FACE_OFFSET_Z,
    rot_x_deg: float = DEFAULT_FACE_ROT_X_DEG,
) -> None:
    """Set alignment transform on *node*."""
    node.set_scale(scale, scale, scale)
    node.set_position(offset_x, offset_y, offset_z)
    rot_x_rad = deg_to_rad(rot_x_deg)
    node.set_quaternion(quat_from_euler(rot_x_rad, 0.0, 0.0, "XYZ"))


def update_alignment(
    face_group: SceneNode,
    scale: float = DEFAULT_FACE_SCALE,
    offset_x: float = DEFAULT_FACE_OFFSET_X,
    offset_y: float = DEFAULT_FACE_OFFSET_Y,
    offset_z: float = DEFAULT_FACE_OFFSET_Z,
    rot_x_deg: float = DEFAULT_FACE_ROT_X_DEG,
) -> None:
    """Update face group alignment transform.

    Called when the user adjusts alignment sliders.  Finds the
    ``faceAlignment`` child node and updates its transform.

    Args:
        face_group: The faceGroup SceneNode (parent).
        scale: Uniform scale factor.
        offset_x: X translation offset.
        offset_y: Y translation offset.
        offset_z: Z translation offset.
        rot_x_deg: Rotation about X axis in degrees.
    """
    # Find the alignment child node
    align_node = None
    for child in face_group.children:
        if child.name == "faceAlignment":
            align_node = child
            break

    if align_node is None:
        return

    _apply_alignment_to_node(align_node, scale, offset_x, offset_y, offset_z, rot_x_deg)
    align_node.mark_dirty()
