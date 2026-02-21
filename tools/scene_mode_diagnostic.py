"""Headless scene mode diagnostic.

Loads the scene, applies supine wrapper transform, and renders to PNG
to verify that the body appears lying on the table (not standing upright).

Usage::

    python -m tools.scene_mode_diagnostic
"""

from __future__ import annotations

import logging
import math
import sys

import numpy as np

from faceforge.core.math_utils import (
    quat_from_axis_angle,
    quat_multiply,
    vec3,
)
from faceforge.core.scene_graph import SceneNode

from tools.headless_loader import load_headless_scene, load_layer
from tools.mesh_renderer import render_mesh

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
logger = logging.getLogger(__name__)

# Mirror the supine constants from scene_mode_controller.py
_Q_SUPINE = quat_multiply(
    quat_from_axis_angle(vec3(0, 0, 1), math.pi / 2),
    quat_multiply(
        quat_from_axis_angle(vec3(0, 1, 0), math.pi / 2),
        quat_from_axis_angle(vec3(1, 0, 0), -math.pi / 2),
    ),
)
_BODY_TABLE_Y = 90 + 15  # TABLE_HEIGHT(90) + 15
_BODY_CENTER_X = -85.0


def run_diagnostic() -> None:
    """Load scene headlessly, apply supine wrapper, render PNGs."""
    logger.info("Loading headless scene...")
    hs = load_headless_scene()
    scene = hs.scene
    named_nodes = hs.named_nodes

    logger.info("Loading skin layer...")
    skin_meshes = load_layer(hs, "skin")
    logger.info("Loaded %d skin meshes", len(skin_meshes))

    body_root = named_nodes.get("bodyRoot")
    if body_root is None:
        logger.error("bodyRoot not found in scene")
        sys.exit(1)

    # Create wrapper node and apply supine transform (mirrors scene_mode_controller)
    wrapper = SceneNode("scene_wrapper")
    wrapper.set_position(_BODY_CENTER_X, _BODY_TABLE_Y, 0)
    wrapper.set_quaternion(_Q_SUPINE.copy())

    # Reparent bodyRoot under wrapper, then add wrapper to scene
    wrapper.add(body_root)
    scene.add(wrapper)

    # Force world matrix rebuild
    scene.update()

    # Hierarchy audit: check which meshes are/aren't under the wrapper
    all_meshes = scene.collect_meshes()
    under_wrapper = 0
    not_under_wrapper = 0
    not_under_names = []

    for mesh, world in all_meshes:
        # Walk parent chain to see if this mesh's node is under wrapper
        node = None
        # Find the node for this mesh
        def _find_mesh_node(n: SceneNode):
            nonlocal node
            if n.mesh is mesh:
                node = n
        scene.traverse(_find_mesh_node)

        if node is None:
            continue

        # Walk up parent chain
        found_wrapper = False
        p = node
        while p is not None:
            if p.name == "scene_wrapper":
                found_wrapper = True
                break
            p = p.parent

        if found_wrapper:
            under_wrapper += 1
        else:
            not_under_wrapper += 1
            not_under_names.append(mesh.name)

    logger.info("=== Hierarchy Audit ===")
    logger.info("Total meshes: %d", len(all_meshes))
    logger.info("Under scene_wrapper: %d", under_wrapper)
    logger.info("NOT under scene_wrapper: %d", not_under_wrapper)
    if not_under_names:
        for name in not_under_names[:20]:
            logger.info("  orphan: %s", name)

    # Log world matrix diagonals for a few body meshes
    logger.info("=== World Matrix Samples ===")
    for mesh, world in all_meshes[:5]:
        diag = np.diag(world).round(3)
        pos = world[:3, 3].round(1)
        logger.info("  %s: diag=%s pos=%s", mesh.name, diag, pos)

    # Transform vertex positions by their world matrices and collect
    # all positions + triangles for rendering
    all_positions = []
    all_rest_positions = []
    all_triangles = []
    vert_offset = 0

    for mesh, world in all_meshes:
        verts = np.asarray(mesh.positions, dtype=np.float64).reshape(-1, 3)
        if verts.size == 0:
            continue

        # Transform to world space: world_pos = (world @ [x,y,z,1])[:3]
        ones = np.ones((len(verts), 1), dtype=np.float64)
        verts_h = np.hstack([verts, ones])  # (V, 4)
        world_verts = (world @ verts_h.T).T[:, :3]  # (V, 3)

        all_positions.append(world_verts)
        all_rest_positions.append(world_verts.copy())  # rest == posed (no deformation)

        geom = mesh.geometry
        if geom.has_indices:
            tris = np.asarray(geom.indices, dtype=np.int32).reshape(-1, 3) + vert_offset
            all_triangles.append(tris)
        else:
            # Non-indexed: generate sequential triangle indices
            n_tris = len(verts) // 3
            if n_tris > 0:
                seq = np.arange(len(verts), dtype=np.int32).reshape(-1, 3) + vert_offset
                all_triangles.append(seq)

        vert_offset += len(verts)

    if not all_positions:
        logger.error("No mesh positions found")
        sys.exit(1)

    positions = np.vstack(all_positions)
    rest_positions = np.vstack(all_rest_positions)
    triangles = np.vstack(all_triangles)

    logger.info("Combined mesh: %d verts, %d triangles", len(positions), len(triangles))

    # Render side view (body should appear lying flat)
    logger.info("Rendering side view...")
    render_mesh(
        positions, rest_positions, triangles,
        azimuth=0, elevation=5,
        output_path="results/scene_mode_diagnostic.png",
        title="Scene Mode - Side View (should be supine)",
        subsample=3,
    )
    logger.info("Saved: results/scene_mode_diagnostic.png")

    # Render overhead view
    logger.info("Rendering overhead view...")
    render_mesh(
        positions, rest_positions, triangles,
        azimuth=0, elevation=75,
        output_path="results/scene_mode_overhead.png",
        title="Scene Mode - Overhead (should see body from above)",
        subsample=3,
    )
    logger.info("Saved: results/scene_mode_overhead.png")

    # Render from the right side (along X axis)
    logger.info("Rendering right view...")
    render_mesh(
        positions, rest_positions, triangles,
        azimuth=90, elevation=5,
        output_path="results/scene_mode_right.png",
        title="Scene Mode - Right Side (along body length)",
        subsample=3,
    )
    logger.info("Saved: results/scene_mode_right.png")

    logger.info("Done. Inspect PNGs in results/ directory.")


if __name__ == "__main__":
    run_diagnostic()
