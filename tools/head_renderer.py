"""Multi-mesh head renderer for close-up head rotation visualization.

Renders multiple mesh groups (skull, jaw muscles, expression muscles, face
features, neck muscles, vertebrae) in a single image with per-group coloring,
depth-interleaved painter's algorithm, and a group legend overlay.

Follows the same PIL-based orthographic pattern as ``mesh_renderer.py``.

Usage::

    from tools.head_renderer import render_head_multimesh, HEAD_CAMERA_PRESETS

    groups = [
        MeshGroup("skull", positions, triangles, color=(200, 190, 170)),
        MeshGroup("jaw_muscles", positions, triangles, color=(200, 80, 80)),
    ]
    img = render_head_multimesh(groups, azimuth=30, elevation=15)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from tools.mesh_renderer import orthographic_project


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class MeshGroup:
    """A named mesh for multi-group rendering."""
    name: str
    positions: NDArray[np.float64]   # (V, 3)
    triangles: NDArray[np.int32]     # (T, 3)
    color: tuple[int, int, int] = (200, 190, 170)
    opacity: float = 1.0


# ── Camera presets (head-level close-ups) ─────────────────────────────

HEAD_CAMERA_PRESETS = {
    "head_front":  {"azimuth": 0,   "elevation": 5},
    "head_right":  {"azimuth": 90,  "elevation": 5},
    "head_3q":     {"azimuth": 30,  "elevation": 15},
    "head_top":    {"azimuth": 0,   "elevation": 70},
    "head_below":  {"azimuth": 0,   "elevation": -30},
}

HEAD_QUICK_VIEWS = ["head_front", "head_right", "head_3q"]


# ── Group color palette ──────────────────────────────────────────────

GROUP_COLORS = {
    "skull":              (200, 190, 170),
    "jaw":                (180, 170, 150),
    "face":               (224, 184, 152),
    "jaw_muscles":        (200, 80, 80),
    "expression_muscles": (180, 100, 60),
    "face_features":      (100, 160, 200),
    "neck_muscles":       (200, 80, 80),
    "vertebrae":          (212, 184, 150),
    "brain":              (220, 180, 200),
}


# ── Lighting ─────────────────────────────────────────────────────────

def _apply_group_lighting(
    base_colors: NDArray[np.uint8],
    positions: NDArray[np.float64],
    triangles: NDArray[np.int32],
    light_dir: NDArray[np.float64] | None = None,
) -> NDArray[np.uint8]:
    """Apply diffuse lighting to per-triangle colors."""
    if light_dir is None:
        light_dir = np.array([0.3, 0.8, 0.4], dtype=np.float64)
    light_dir = light_dir / np.linalg.norm(light_dir)

    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0 = positions[i0]
    e1 = positions[i1] - v0
    e2 = positions[i2] - v0

    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    normals /= lengths

    dot = np.abs(np.einsum("ij,j->i", normals, light_dir))
    brightness = 0.35 + 0.65 * np.clip(dot, 0, 1)

    lit = base_colors.astype(np.float64) * brightness[:, np.newaxis]
    return np.clip(lit, 0, 255).astype(np.uint8)


# ── Core render function ─────────────────────────────────────────────

def render_head_multimesh(
    groups: list[MeshGroup],
    azimuth: float = 0,
    elevation: float = 5,
    width: int = 800,
    height: int = 800,
    output_path: str | Path | None = None,
    title: str = "",
    subsample: int = 1,
    bg_color: tuple[int, int, int] = (30, 30, 40),
    margin: int = 40,
    lighting: bool = True,
    z_range: tuple[float, float] | None = None,
) -> Image.Image:
    """Render multiple mesh groups with depth-interleaved painter's algorithm.

    Parameters
    ----------
    groups : list[MeshGroup]
        Mesh groups to render. Each gets its own base color.
    azimuth, elevation : camera angles in degrees
    width, height : image size in pixels
    output_path : save PNG here (None = don't save)
    title : text overlay at top-left
    subsample : render every Nth triangle per group
    bg_color : background RGB
    margin : pixel margin around mesh
    lighting : apply simple diffuse lighting
    z_range : optional (z_min, z_max) to clip meshes to a vertical range

    Returns
    -------
    PIL.Image.Image
    """
    if not groups:
        img = Image.new("RGB", (width, height), bg_color)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path))
        return img

    # Collect all vertices to compute unified bounding box
    all_sx, all_sy, all_depth = [], [], []

    # Per-group projected data
    group_data = []

    for g in groups:
        pos = np.asarray(g.positions, dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(g.triangles, dtype=np.int32).reshape(-1, 3)

        if len(pos) == 0 or len(tris) == 0:
            continue

        # Optional Z-range clipping
        if z_range is not None:
            z_lo, z_hi = z_range
            # Keep triangles where all 3 vertices are within range
            v_in = (pos[:, 2] >= z_lo) & (pos[:, 2] <= z_hi)
            tri_mask = v_in[tris[:, 0]] & v_in[tris[:, 1]] & v_in[tris[:, 2]]
            tris = tris[tri_mask]
            if len(tris) == 0:
                continue

        if subsample > 1:
            tris = tris[::subsample]

        sx, sy, depth = orthographic_project(pos, azimuth, elevation)
        all_sx.append(sx)
        all_sy.append(sy)
        all_depth.append(depth)

        group_data.append((g, pos, tris, sx, sy, depth))

    if not group_data:
        img = Image.new("RGB", (width, height), bg_color)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path))
        return img

    # Unified bounding box
    cat_sx = np.concatenate(all_sx)
    cat_sy = np.concatenate(all_sy)
    xmin, xmax = float(cat_sx.min()), float(cat_sx.max())
    ymin, ymax = float(cat_sy.min()), float(cat_sy.max())
    x_range = max(xmax - xmin, 1e-6)
    y_range = max(ymax - ymin, 1e-6)
    scale = min((width - 2 * margin) / x_range, (height - 2 * margin) / y_range)

    x_off = margin + (width - 2 * margin - x_range * scale) / 2
    y_off = margin + (height - 2 * margin - y_range * scale) / 2

    # Process each group: backface cull, compute colors
    # Collect all triangles into a single sorted list for painter's algorithm
    all_screen_tris = []  # list of (depth, screen_pts, color_rgb)

    for g, pos, tris, sx, sy, depth in group_data:
        px = (sx - xmin) * scale + x_off
        py = height - ((sy - ymin) * scale + y_off)

        # Backface culling
        i0, i1, i2 = tris[:, 0], tris[:, 1], tris[:, 2]
        e1x = px[i1] - px[i0]
        e1y = py[i1] - py[i0]
        e2x = px[i2] - px[i0]
        e2y = py[i2] - py[i0]
        cross_z = e1x * e2y - e1y * e2x
        front = cross_z > 0
        if np.sum(front) < np.sum(~front):
            front = ~front

        visible_idx = np.where(front)[0]
        if len(visible_idx) == 0:
            continue

        vis_tris = tris[visible_idx]

        # Per-triangle base color
        n_vis = len(vis_tris)
        color_rgb = g.color
        if isinstance(color_rgb[0], float):
            color_rgb = (int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255))
        base_colors = np.tile(np.array(color_rgb, dtype=np.uint8), (n_vis, 1))

        # Apply lighting
        if lighting:
            az = np.radians(azimuth)
            light_world = np.array([
                0.3 * np.cos(az) + 0.5 * np.sin(az),
                -0.3 * np.sin(az) + 0.5 * np.cos(az),
                0.4,
            ])
            base_colors = _apply_group_lighting(base_colors, pos, vis_tris, light_world)

        # Depth per triangle
        tri_depth = (depth[vis_tris[:, 0]] + depth[vis_tris[:, 1]] + depth[vis_tris[:, 2]]) / 3

        # Collect into global list
        for i in range(n_vis):
            v0, v1, v2 = vis_tris[i]
            pts = [
                (int(px[v0] + 0.5), int(py[v0] + 0.5)),
                (int(px[v1] + 0.5), int(py[v1] + 0.5)),
                (int(px[v2] + 0.5), int(py[v2] + 0.5)),
            ]
            c = (int(base_colors[i, 0]), int(base_colors[i, 1]), int(base_colors[i, 2]))
            all_screen_tris.append((float(tri_depth[i]), pts, c))

    # Sort by depth (back to front)
    all_screen_tris.sort(key=lambda x: x[0])

    # Draw
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    for _, pts, c in all_screen_tris:
        draw.polygon(pts, fill=c)

    # Title overlay
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))

    # Group legend
    _draw_group_legend(draw, width, groups)

    # Stats
    stats = f"Groups: {len(groups)} | Tris: {len(all_screen_tris)}"
    draw.text((10, height - 25), stats, fill=(180, 180, 180))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))

    return img


def _draw_group_legend(
    draw: ImageDraw.ImageDraw, width: int, groups: list[MeshGroup],
) -> None:
    """Draw a color legend for the rendered mesh groups."""
    # Deduplicate by name for the legend
    seen: dict[str, tuple] = {}
    for g in groups:
        if g.name not in seen:
            seen[g.name] = g.color

    lx = width - 180
    ly = 15
    for i, (name, color) in enumerate(seen.items()):
        y = ly + i * 22
        # Ensure color is (int, int, int) for PIL
        if isinstance(color[0], float):
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        draw.rectangle([lx, y, lx + 14, y + 14], fill=color, outline=(100, 100, 100))
        draw.text((lx + 20, y), name, fill=(200, 200, 200))


# ── Multi-view rendering ─────────────────────────────────────────────

def render_head_multiview(
    groups: list[MeshGroup],
    output_dir: str | Path,
    prefix: str = "",
    views: dict | None = None,
    subsample: int = 1,
    width: int = 800,
    height: int = 800,
    z_range: tuple[float, float] | None = None,
) -> dict[str, Path]:
    """Render head from multiple camera angles.

    Parameters
    ----------
    views : dict of {name: {azimuth, elevation}}, or None for HEAD_QUICK_VIEWS

    Returns
    -------
    dict mapping view name to output file path
    """
    if views is None:
        views = {k: HEAD_CAMERA_PRESETS[k] for k in HEAD_QUICK_VIEWS}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for vname, params in views.items():
        fname = f"{prefix}_{vname}.png" if prefix else f"{vname}.png"
        fpath = output_dir / fname
        render_head_multimesh(
            groups,
            azimuth=params["azimuth"],
            elevation=params.get("elevation", 5),
            width=width, height=height,
            output_path=fpath,
            title=f"{prefix} - {vname}" if prefix else vname,
            subsample=subsample,
            z_range=z_range,
        )
        paths[vname] = fpath

    return paths
