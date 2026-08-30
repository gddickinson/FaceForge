"""Keep newly loaded meshes in step with the render mode already on screen.

Structures load on demand, long after the user has chosen a render mode, so a
freshly loaded layer arrives carrying whatever mode its loader defaulted to.
Without this, ticking a layer in XRAY mode drops in an opaque group.

The current mode is read off the scene rather than tracked in a variable: the
mode is a per-material property, and the scene is the only place that cannot
drift out of step with what is actually being drawn.
"""

from __future__ import annotations

from typing import Any, Iterable

from faceforge.core.material import RenderMode


def current_render_mode(scene: Any) -> RenderMode:
    """The mode the scene is currently drawn in.

    Taken from the first mesh the scene traversal yields.  An empty scene has
    no mode to read, and ``WIREFRAME`` is what the application starts in.
    """
    existing = scene.collect_meshes()
    if existing:
        return existing[0][0].material.render_mode
    return RenderMode.WIREFRAME


def apply_current_render_mode(scene: Any, meshes: Iterable[Any]) -> RenderMode:
    """Set every mesh in *meshes* to the scene's current mode; return that mode."""
    mode = current_render_mode(scene)
    for mesh in meshes:
        mesh.material.render_mode = mode
    return mode
