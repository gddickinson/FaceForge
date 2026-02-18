"""Batch STL loader with BodyParts3D coordinate transform."""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np

from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material
from faceforge.core.scene_graph import SceneNode
from faceforge.core.config_loader import load_config
from faceforge.loaders.stl_parser import load_stl_file
from faceforge.constants import STL_DIR


@dataclass
class CoordinateTransform:
    """BodyParts3D → skull/world coordinate transform.

    Transform: result = (bp3d_pos - center) * scale + skull_center
    Note: X is negated (BP3D left=+X, skull left=-X).
    """
    center_x: float = -0.67
    center_y: float = -139.24
    center_z: float = 1475.58
    skull_center_x: float = 0.00
    skull_center_y: float = -8.35
    skull_center_z: float = 5.54
    scale_x: float = -0.154  # Negated for mirror
    scale_y: float = 0.174
    scale_z: float = 0.137

    @classmethod
    def from_config(cls) -> "CoordinateTransform":
        """Load transform parameters from coordinate_transform.json."""
        cfg = load_config("coordinate_transform.json")
        c = cfg["bp3d_center"]
        s = cfg["skull_center"]
        sc = cfg["bp3d_scale"]
        return cls(
            center_x=c["x"], center_y=c["y"], center_z=c["z"],
            skull_center_x=s["x"], skull_center_y=s["y"], skull_center_z=s["z"],
            scale_x=sc["x"], scale_y=sc["y"], scale_z=sc["z"],
        )

    def transform_point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Transform a single BP3D point to skull coordinates."""
        return (
            (x - self.center_x) * self.scale_x + self.skull_center_x,
            (y - self.center_y) * self.scale_y + self.skull_center_y,
            (z - self.center_z) * self.scale_z + self.skull_center_z,
        )

    def transform_positions_in_place(self, positions: np.ndarray, vert_count: int) -> None:
        """Transform position array in-place from BP3D to skull coordinates."""
        p = positions.reshape(-1, 3)[:vert_count]
        p[:, 0] = (p[:, 0] - self.center_x) * self.scale_x + self.skull_center_x
        p[:, 1] = (p[:, 1] - self.center_y) * self.scale_y + self.skull_center_y
        p[:, 2] = (p[:, 2] - self.center_z) * self.scale_z + self.skull_center_z

    def transform_normals_in_place(self, normals: np.ndarray, vert_count: int) -> None:
        """Transform normals in-place (only flip X for mirrored axis)."""
        normals.reshape(-1, 3)[:vert_count, 0] *= -1


@dataclass
class STLBatchResult:
    """Result from loading a batch of STL files."""
    group: SceneNode
    meshes: list[MeshInstance]
    nodes: list[SceneNode]
    pivot_groups: dict[int, SceneNode]  # level → pivot node (if createPivots)
    failed: list[str]  # Names of STLs that failed to load


def load_stl_batch(
    defs: list[dict],
    *,
    label: str = "",
    transform: Optional[CoordinateTransform] = None,
    create_pivots: bool = False,
    pivot_key: str = "level",
    indexed: bool = True,
    stl_dir: Optional[Path] = None,
) -> STLBatchResult:
    """Load a batch of STL files from definition dicts.

    Each def dict should have at minimum: name, stl, color.
    Optional fields: tier, opacity, shininess, side, etc.

    Args:
        defs: List of definition dicts from config.
        label: Human-readable label for progress reporting.
        transform: Coordinate transform to apply (BP3D → skull).
        create_pivots: If True, create pivot groups keyed by pivot_key field.
        pivot_key: Field name in def dict used to group into pivot nodes.
        indexed: Whether to build indexed geometry.
        stl_dir: Override STL directory.
    """
    if stl_dir is None:
        stl_dir = STL_DIR
    if transform is None:
        transform = CoordinateTransform()

    group = SceneNode(name=label or "stl_batch")
    meshes = []
    nodes = []
    pivot_groups: dict[int, SceneNode] = {}
    failed = []

    for defn in defs:
        stl_name = defn["stl"]
        stl_path = stl_dir / f"{stl_name}.stl"
        name = defn.get("name", stl_name)

        try:
            geom = load_stl_file(stl_path, indexed=indexed)
        except (FileNotFoundError, ValueError) as e:
            failed.append(name)
            continue

        # Apply coordinate transform
        transform.transform_positions_in_place(geom.positions, geom.vertex_count)
        transform.transform_normals_in_place(geom.normals, geom.vertex_count)

        # Create material
        color_int = defn.get("color", 0xcccccc)
        mat = Material.from_hex(
            color_int,
            opacity=defn.get("opacity", 0.7),
            shininess=defn.get("shininess", 15.0),
            transparent=defn.get("opacity", 0.7) < 1.0,
        )

        mesh = MeshInstance(name=name, geometry=geom, material=mat)
        mesh.store_rest_pose()
        meshes.append(mesh)

        node = SceneNode(name=name)
        node.mesh = mesh
        nodes.append(node)

        # Pivot grouping
        if create_pivots and pivot_key in defn:
            level = defn[pivot_key]
            if level not in pivot_groups:
                pivot_node = SceneNode(name=f"{label}_pivot_{level}")
                pivot_groups[level] = pivot_node
                group.add(pivot_node)
            pivot_groups[level].add(node)
        else:
            group.add(node)

    # Set pivot positions to centroids of their children, and offset
    # mesh vertices by -centroid so they're in local pivot coordinates.
    # This matches the original HTML: pos[i]-=center[i] per vertex.
    if create_pivots:
        for level, pivot in pivot_groups.items():
            positions = []
            for child in pivot.children:
                if child.mesh is not None:
                    center = child.mesh.geometry.get_bounding_center()
                    positions.append(center)
            if positions:
                centroid = np.mean(positions, axis=0)
                pivot.set_position(float(centroid[0]), float(centroid[1]), float(centroid[2]))
                # Offset each child mesh's vertices by -centroid
                for child in pivot.children:
                    if child.mesh is not None:
                        geom = child.mesh.geometry
                        p = geom.positions.reshape(-1, 3)[:geom.vertex_count]
                        p -= centroid
                        # Update rest pose with new local positions
                        child.mesh.store_rest_pose()

    return STLBatchResult(
        group=group,
        meshes=meshes,
        nodes=nodes,
        pivot_groups=pivot_groups,
        failed=failed,
    )
