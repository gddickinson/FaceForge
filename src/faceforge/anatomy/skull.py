"""Build skull mesh hierarchy from loaded mesh data.

Creates a SceneNode hierarchy containing cranium, jaw (with TMJ pivot),
upper teeth, and lower teeth meshes loaded from the AssetManager.

Supports two modes:
- ``"original"``: embedded skull data (cranium + jaw + teeth from skull.json)
- ``"bp3d"``: individual BodyParts3D skull bone STLs with dynamic TMJ pivot
"""

import logging
from typing import Optional

from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.constants import JAW_PIVOT, get_jaw_pivot
from faceforge.loaders.asset_manager import AssetManager

logger = logging.getLogger(__name__)


def build_skull(
    asset_manager: AssetManager,
    jaw_pivot: tuple[float, float, float] | None = None,
    mode: str = "original",
) -> tuple[SceneNode, dict[str, MeshInstance], tuple[float, float, float]]:
    """Build skull hierarchy in the given mode.

    Args:
        asset_manager: Asset loading manager.
        jaw_pivot: Optional explicit jaw pivot.
        mode: ``"original"`` for embedded skull, ``"bp3d"`` for individual bones.

    Returns:
        ``(skullGroup, meshes_dict, jaw_pivot_tuple)``
    """
    if mode == "bp3d":
        return build_bp3d_skull(asset_manager)
    return build_original_skull(asset_manager, jaw_pivot=jaw_pivot)


def build_original_skull(
    asset_manager: AssetManager,
    jaw_pivot: tuple[float, float, float] | None = None,
) -> tuple[SceneNode, dict[str, MeshInstance], tuple[float, float, float]]:
    """Build the skull scene-graph hierarchy.

    Structure::

        skullGroup
        ├── cranium_node  (cranium mesh)
        ├── jawPivot  (positioned at JAW_PIVOT)
        │   ├── jaw_node  (jaw/mandible mesh)
        │   └── lower_teeth_node  (lower teeth mesh)
        └── upper_teeth_node  (upper teeth mesh)

    The jawPivot node is positioned at the TMJ hinge point so that
    rotating it opens/closes the jaw naturally.

    Args:
        asset_manager: Provides loaded skull mesh data via ``load_skull()``.
        jaw_pivot: Optional explicit jaw pivot ``(x, y, z)``.
            Defaults to the global ``JAW_PIVOT`` constant.

    Returns:
        Tuple of ``(skullGroup, meshes_dict, jaw_pivot_tuple)``.
        The dict keys are: ``"cranium"``, ``"jaw"``, ``"upper_teeth"``, ``"lower_teeth"``.
    """
    skull_meshes = asset_manager.load_skull()
    jp = jaw_pivot if jaw_pivot is not None else JAW_PIVOT

    skull_group = SceneNode(name="skullGroup")

    # --- Cranium (fixed to skull group) ---
    cranium_node = SceneNode(name="cranium")
    cranium_mesh = skull_meshes.get("cranium")
    if cranium_mesh is not None:
        cranium_node.mesh = cranium_mesh
    skull_group.add(cranium_node)

    # --- Jaw pivot group at TMJ hinge point ---
    jaw_pivot_node = SceneNode(name="jawPivot")
    jaw_pivot_node.set_position(jp[0], jp[1], jp[2])
    skull_group.add(jaw_pivot_node)

    # Jaw (mandible) — attached to the jaw pivot so it rotates with it.
    jaw_node = SceneNode(name="jaw")
    jaw_node.set_position(-jp[0], -jp[1], -jp[2])
    jaw_mesh = skull_meshes.get("jaw")
    if jaw_mesh is not None:
        jaw_node.mesh = jaw_mesh
    jaw_pivot_node.add(jaw_node)

    # Lower teeth — also rotate with jaw, same compensating offset
    lower_teeth_node = SceneNode(name="lower_teeth")
    lower_teeth_node.set_position(-jp[0], -jp[1], -jp[2])
    lower_teeth_mesh = skull_meshes.get("lower_teeth")
    if lower_teeth_mesh is not None:
        lower_teeth_node.mesh = lower_teeth_mesh
    jaw_pivot_node.add(lower_teeth_node)

    # --- Upper teeth (fixed to skull group) ---
    upper_teeth_node = SceneNode(name="upper_teeth")
    upper_teeth_mesh = skull_meshes.get("upper_teeth")
    if upper_teeth_mesh is not None:
        upper_teeth_node.mesh = upper_teeth_mesh
    skull_group.add(upper_teeth_node)

    meshes = {
        "cranium": cranium_mesh,
        "jaw": jaw_mesh,
        "upper_teeth": upper_teeth_mesh,
        "lower_teeth": lower_teeth_mesh,
    }

    return skull_group, meshes, jp


def build_bp3d_skull(
    asset_manager: AssetManager,
) -> tuple[SceneNode, dict[str, MeshInstance], tuple[float, float, float]]:
    """Build skull from individual BodyParts3D bone STLs.

    Structure::

        skullGroup
        ├── cranium_bones_group  (20 fixed cranial/facial bones)
        ├── jawPivot  (at computed TMJ position)
        │   ├── mandible_node  (offset by -pivot)
        │   └── lower_teeth_group  (offset by -pivot)
        └── upper_teeth_group

    Returns:
        ``(skullGroup, meshes_dict, jaw_pivot_tuple)``
    """
    from faceforge.body.joint_pivots import compute_tmj_pivot
    from faceforge.loaders.stl_batch_loader import load_stl_batch
    from faceforge.core.config_loader import load_config

    skull_group = SceneNode(name="skullGroup")
    meshes: dict[str, MeshInstance] = {}

    # Load skull bone STLs
    bone_defs = load_config("skull_bones.json")
    bone_result = load_stl_batch(
        bone_defs,
        label="skull_bones",
        transform=asset_manager.transform,
        stl_dir=asset_manager.stl_dir,
        indexed=True,
    )

    # Load teeth STLs
    teeth_defs = load_config("teeth.json")
    teeth_result = load_stl_batch(
        teeth_defs,
        label="teeth",
        transform=asset_manager.transform,
        stl_dir=asset_manager.stl_dir,
        indexed=True,
    )

    if bone_result.failed:
        logger.warning("Failed skull bones: %s", bone_result.failed)
    if teeth_result.failed:
        logger.warning("Failed teeth: %s", teeth_result.failed)

    # Identify mandible and temporal bones for TMJ computation
    mandible_node = None
    temporal_nodes = []
    cranium_group = SceneNode(name="cranium_bones_group")

    for node, defn in zip(bone_result.nodes, bone_defs):
        if defn.get("jaw_attached"):
            mandible_node = node
        else:
            cranium_group.add(node)
            if defn.get("tmj_reference"):
                temporal_nodes.append(node)

    skull_group.add(cranium_group)

    # Store all bone meshes
    for mesh in bone_result.meshes:
        meshes[mesh.name] = mesh
    # Map "cranium" key to the group for backward compat
    meshes["cranium"] = bone_result.meshes[0] if bone_result.meshes else None

    # Compute TMJ pivot dynamically from mandible + temporal bones
    jaw_pivot_pos = compute_tmj_pivot(mandible_node, temporal_nodes)
    jp = (float(jaw_pivot_pos[0]), float(jaw_pivot_pos[1]), float(jaw_pivot_pos[2]))
    logger.info("BP3D TMJ pivot computed: (%.2f, %.2f, %.2f)", *jp)

    # Create jaw pivot node
    jaw_pivot_node = SceneNode(name="jawPivot")
    jaw_pivot_node.set_position(jp[0], jp[1], jp[2])
    skull_group.add(jaw_pivot_node)

    # Attach mandible under jaw pivot with compensating offset
    if mandible_node is not None:
        mandible_node.set_position(-jp[0], -jp[1], -jp[2])
        jaw_pivot_node.add(mandible_node)
        if mandible_node.mesh:
            meshes["jaw"] = mandible_node.mesh
    else:
        meshes["jaw"] = None

    # Classify teeth into upper/lower and attach
    upper_teeth_group = SceneNode(name="upper_teeth")
    lower_teeth_group = SceneNode(name="lower_teeth")
    lower_teeth_group.set_position(-jp[0], -jp[1], -jp[2])

    for node, defn in zip(teeth_result.nodes, teeth_defs):
        if defn.get("jaw") == "lower":
            lower_teeth_group.add(node)
        else:
            upper_teeth_group.add(node)

    jaw_pivot_node.add(lower_teeth_group)
    skull_group.add(upper_teeth_group)

    # Store teeth meshes
    for mesh in teeth_result.meshes:
        meshes[mesh.name] = mesh
    meshes["upper_teeth"] = teeth_result.meshes[0] if teeth_result.meshes else None
    meshes["lower_teeth"] = None  # Multiple individual tooth meshes instead

    return skull_group, meshes, jp


def get_jaw_pivot_node(skull_group: SceneNode) -> Optional[SceneNode]:
    """Find and return the jawPivot node from the skull hierarchy."""
    return skull_group.find("jawPivot")
