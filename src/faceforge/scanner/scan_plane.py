"""3D scan plane visualization in the viewport.

Adds a semi-transparent quad to the scene showing where the scan slab
is positioned. Updated each time the scanner parameters change.
"""

from __future__ import annotations

import numpy as np

from faceforge.core.scene_graph import SceneNode, Scene
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material, RenderMode


class ScanPlaneViz:
    """Semi-transparent quad showing the scan plane in the 3D viewport."""

    def __init__(self, scene: Scene):
        self.node = SceneNode(name="scan_plane")
        self.node.visible = False

        # Build a simple quad (2 triangles)
        positions = np.zeros(4 * 3, dtype=np.float32)
        normals = np.array([
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        geom = BufferGeometry(positions=positions, normals=normals, indices=indices)
        mat = Material(
            color=(0.2, 0.8, 0.2),
            opacity=0.3,
            transparent=True,
            double_sided=True,
            render_mode=RenderMode.SOLID,
            depth_write=False,
        )
        self.mesh = MeshInstance(name="scan_plane_quad", geometry=geom, material=mat)
        self.mesh.scene_affected = False
        self.node.mesh = self.mesh

        scene.add(self.node)

    def update(
        self,
        origin: np.ndarray,
        normal: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        width: float,
        height: float,
    ) -> None:
        """Reposition the quad vertices to match the scan plane."""
        hw = width * 0.5
        hh = height * 0.5
        corners = np.array([
            origin - right * hw - up * hh,
            origin + right * hw - up * hh,
            origin + right * hw + up * hh,
            origin - right * hw + up * hh,
        ], dtype=np.float32)

        self.mesh.geometry.positions = corners.ravel()
        # Normal for all 4 verts
        n = normal.astype(np.float32)
        self.mesh.geometry.normals = np.tile(n, 4)
        self.mesh.needs_update = True

    def set_visible(self, visible: bool) -> None:
        """Show or hide the scan plane."""
        self.node.visible = visible
        self.mesh.visible = visible
