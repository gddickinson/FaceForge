"""Headless mesh renderer with edge-stretch coloring.

Renders 3D meshes to PNG images using PIL for fast triangle rasterization.
Provides orthographic projection, backface culling, painter's algorithm
depth sorting, and per-triangle coloring based on edge stretch ratio.

Usage::

    from tools.mesh_renderer import render_mesh, CAMERA_PRESETS

    img = render_mesh(
        positions, rest_positions, triangles,
        azimuth=30, elevation=15,
        output_path="results/test.png",
        title="Sitting - 3/4 view",
    )
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ── Camera presets ─────────────────────────────────────────────────────

CAMERA_PRESETS = {
    "front":        {"azimuth": 0,   "elevation": 5},
    "right":        {"azimuth": 90,  "elevation": 5},
    "back":         {"azimuth": 180, "elevation": 5},
    "left":         {"azimuth": 270, "elevation": 5},
    "3q_front_r":   {"azimuth": 30,  "elevation": 15},
    "3q_front_l":   {"azimuth": -30, "elevation": 15},
    "top":          {"azimuth": 0,   "elevation": 75},
}

# Key views for quick rendering
QUICK_VIEWS = ["front", "right", "3q_front_r"]


# ── Projection ─────────────────────────────────────────────────────────

def orthographic_project(
    positions: np.ndarray,
    azimuth: float = 0,
    elevation: float = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D positions to 2D screen coordinates.

    The body coordinate system has Z as vertical (head ≈ 0, feet ≈ -200),
    X as left-right, Y as front-back.

    Parameters
    ----------
    positions : (V, 3) float array
    azimuth : degrees, 0=front, 90=right, 180=back
    elevation : degrees, 0=level, 90=top-down

    Returns
    -------
    screen_x, screen_y, depth : (V,) float arrays
    """
    az = np.radians(azimuth)
    el = np.radians(elevation)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Rotate around Z (vertical) by azimuth
    x1 = x * ca - y * sa
    y1 = x * sa + y * ca

    # Rotate around screen-X by elevation (tilt camera up)
    screen_x = x1
    screen_y = y1 * se + z * ce
    depth = y1 * ce - z * se

    return screen_x, screen_y, depth


# ── Edge stretch computation ───────────────────────────────────────────

def compute_triangle_stretch(
    positions: np.ndarray,
    rest_positions: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Compute per-triangle max edge stretch ratio.

    Returns (T,) array: max(current_edge_len / rest_edge_len) per triangle.
    """
    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    # Current edge lengths
    e01 = np.linalg.norm(positions[i1] - positions[i0], axis=1)
    e12 = np.linalg.norm(positions[i2] - positions[i1], axis=1)
    e20 = np.linalg.norm(positions[i0] - positions[i2], axis=1)

    # Rest edge lengths
    r01 = np.linalg.norm(rest_positions[i1] - rest_positions[i0], axis=1)
    r12 = np.linalg.norm(rest_positions[i2] - rest_positions[i1], axis=1)
    r20 = np.linalg.norm(rest_positions[i0] - rest_positions[i2], axis=1)

    s01 = e01 / np.maximum(r01, 1e-6)
    s12 = e12 / np.maximum(r12, 1e-6)
    s20 = e20 / np.maximum(r20, 1e-6)

    return np.maximum(np.maximum(s01, s12), s20)


def stretch_to_rgb(stretch: np.ndarray) -> np.ndarray:
    """Map stretch ratios to RGB uint8 colors.

    Color scheme:
    - Compressed (<0.8): blue tint
    - Normal (0.8-1.2): neutral skin tone
    - Mild (1.2-1.5): yellow
    - Moderate (1.5-2.0): orange
    - High (2.0-3.0): red
    - Extreme (3.0+): magenta

    Returns (T, 3) uint8 array.
    """
    T = len(stretch)
    colors = np.full((T, 3), 218, dtype=np.uint8)  # default skin base

    # Skin base
    sr, sg, sb = 218, 190, 160

    normal = (stretch >= 0.8) & (stretch < 1.2)
    colors[normal] = [sr, sg, sb]

    # Compressed (<0.8): lerp skin → blue
    comp = stretch < 0.8
    if np.any(comp):
        t = np.clip((0.8 - stretch[comp]) / 0.4, 0, 1)
        colors[comp, 0] = (sr * (1 - t) + 80 * t).astype(np.uint8)
        colors[comp, 1] = (sg * (1 - t) + 120 * t).astype(np.uint8)
        colors[comp, 2] = (sb * (1 - t) + 220 * t).astype(np.uint8)

    # Mild (1.2-1.5): lerp skin → yellow
    mild = (stretch >= 1.2) & (stretch < 1.5)
    if np.any(mild):
        t = (stretch[mild] - 1.2) / 0.3
        colors[mild, 0] = np.clip(sr + (255 - sr) * t, 0, 255).astype(np.uint8)
        colors[mild, 1] = np.clip(sg + (255 - sg) * t * 0.9, 0, 255).astype(np.uint8)
        colors[mild, 2] = np.clip(sb * (1 - t * 0.85), 0, 255).astype(np.uint8)

    # Moderate (1.5-2.0): lerp yellow → orange-red
    mod = (stretch >= 1.5) & (stretch < 2.0)
    if np.any(mod):
        t = (stretch[mod] - 1.5) / 0.5
        colors[mod, 0] = 255
        colors[mod, 1] = np.clip(210 * (1 - t * 0.8), 0, 255).astype(np.uint8)
        colors[mod, 2] = 0

    # High (2.0-3.0): red
    high = (stretch >= 2.0) & (stretch < 3.0)
    colors[high] = [255, 30, 0]

    # Extreme (3.0+): magenta
    ext = stretch >= 3.0
    colors[ext] = [255, 0, 200]

    return colors


# ── Simple lighting ────────────────────────────────────────────────────

def _apply_lighting(
    colors: np.ndarray,
    positions: np.ndarray,
    triangles: np.ndarray,
    light_dir: np.ndarray | None = None,
) -> np.ndarray:
    """Modulate face colors by a simple diffuse lighting model.

    Makes the 3D shape more visible by darkening faces angled away
    from the light source.  Returns modified (T, 3) uint8 array.
    """
    if light_dir is None:
        light_dir = np.array([0.3, 0.8, 0.4], dtype=np.float64)
    light_dir = light_dir / np.linalg.norm(light_dir)

    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0 = positions[i0]
    e1 = positions[i1] - v0
    e2 = positions[i2] - v0

    # Face normals
    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    normals /= lengths

    # Diffuse: dot(normal, light), clamped to [0.3, 1.0] for ambient
    dot = np.einsum("ij,j->i", normals, light_dir)
    dot = np.abs(dot)  # two-sided lighting
    brightness = 0.35 + 0.65 * np.clip(dot, 0, 1)

    lit = colors.astype(np.float64) * brightness[:, np.newaxis]
    return np.clip(lit, 0, 255).astype(np.uint8)


# ── Core render function ───────────────────────────────────────────────

def render_mesh(
    positions: np.ndarray,
    rest_positions: np.ndarray,
    triangles: np.ndarray,
    azimuth: float = 0,
    elevation: float = 5,
    width: int = 800,
    height: int = 1000,
    output_path: str | Path | None = None,
    title: str = "",
    subsample: int = 1,
    bg_color: tuple[int, int, int] = (30, 30, 40),
    margin: int = 40,
    lighting: bool = True,
) -> Image.Image:
    """Render a mesh with edge-stretch coloring to a PIL Image.

    Parameters
    ----------
    positions : (V, 3) current vertex positions
    rest_positions : (V, 3) rest-pose vertex positions
    triangles : (T, 3) int triangle indices
    azimuth, elevation : camera angles in degrees
    width, height : image size in pixels
    output_path : save PNG here (None = don't save)
    title : text overlay at top-left
    subsample : render every Nth triangle (1=all, 3=fast preview)
    bg_color : background RGB
    margin : pixel margin around mesh
    lighting : apply simple diffuse lighting

    Returns
    -------
    PIL.Image.Image
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    rest_positions = np.asarray(rest_positions, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)

    # Subsample triangles for speed
    if subsample > 1:
        triangles = triangles[::subsample]

    # Project to 2D
    sx, sy, depth = orthographic_project(positions, azimuth, elevation)

    # Fit to image coordinates
    xmin, xmax = float(sx.min()), float(sx.max())
    ymin, ymax = float(sy.min()), float(sy.max())
    x_range = max(xmax - xmin, 1e-6)
    y_range = max(ymax - ymin, 1e-6)
    scale = min((width - 2 * margin) / x_range, (height - 2 * margin) / y_range)

    # Center in image
    x_off = margin + (width - 2 * margin - x_range * scale) / 2
    y_off = margin + (height - 2 * margin - y_range * scale) / 2
    px = (sx - xmin) * scale + x_off
    py = height - ((sy - ymin) * scale + y_off)  # flip Y for image coords

    # Backface culling (screen-space cross product)
    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1x = px[i1] - px[i0]
    e1y = py[i1] - py[i0]
    e2x = px[i2] - px[i0]
    e2y = py[i2] - py[i0]
    cross_z = e1x * e2y - e1y * e2x
    front = cross_z > 0
    if np.sum(front) < np.sum(~front):
        front = ~front  # flip winding convention if needed

    visible_idx = np.where(front)[0]
    if len(visible_idx) == 0:
        img = Image.new("RGB", (width, height), bg_color)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path))
        return img

    visible_tris = triangles[visible_idx]

    # Compute edge stretch coloring
    stretch = compute_triangle_stretch(positions, rest_positions, visible_tris)
    colors = stretch_to_rgb(stretch)

    # Apply lighting
    if lighting:
        # Light direction in pre-rotation world space
        az = np.radians(azimuth)
        el = np.radians(elevation)
        light_world = np.array([
            0.3 * np.cos(az) + 0.5 * np.sin(az),
            -0.3 * np.sin(az) + 0.5 * np.cos(az),
            0.4,
        ])
        colors = _apply_lighting(colors, positions, visible_tris, light_world)

    # Depth sort (painter's algorithm: back-to-front)
    tri_depth = (depth[visible_tris[:, 0]] + depth[visible_tris[:, 1]] + depth[visible_tris[:, 2]]) / 3
    sort_order = np.argsort(tri_depth)
    visible_tris = visible_tris[sort_order]
    colors = colors[sort_order]
    stretch = stretch[sort_order]

    # Draw triangles using PIL
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    n_tris = len(visible_tris)
    for i in range(n_tris):
        v0, v1, v2 = visible_tris[i]
        pts = [
            (int(px[v0] + 0.5), int(py[v0] + 0.5)),
            (int(px[v1] + 0.5), int(py[v1] + 0.5)),
            (int(px[v2] + 0.5), int(py[v2] + 0.5)),
        ]
        c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        draw.polygon(pts, fill=c)

    # Overlay: title and legend
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))

    _draw_legend(draw, width, height)

    # Stats overlay
    n_mild = int(np.sum((stretch >= 1.2) & (stretch < 1.5)))
    n_mod = int(np.sum((stretch >= 1.5) & (stretch < 2.0)))
    n_high = int(np.sum((stretch >= 2.0) & (stretch < 3.0)))
    n_ext = int(np.sum(stretch >= 3.0))
    stats_text = f"Tris: {n_tris} | >1.2x:{n_mild} >1.5x:{n_mod} >2x:{n_high} >3x:{n_ext}"
    draw.text((10, height - 25), stats_text, fill=(180, 180, 180))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))

    return img


def _draw_legend(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    """Draw a color legend for stretch values."""
    lx = width - 150
    ly = 15
    items = [
        ((218, 190, 160), "Normal (1.0x)"),
        ((255, 250, 50),  "Mild (1.2-1.5x)"),
        ((255, 160, 0),   "Moderate (1.5-2x)"),
        ((255, 30, 0),    "High (2-3x)"),
        ((255, 0, 200),   "Extreme (3x+)"),
        ((80, 120, 220),  "Compressed"),
    ]
    for i, (color, label) in enumerate(items):
        y = ly + i * 22
        draw.rectangle([lx, y, lx + 14, y + 14], fill=color, outline=(100, 100, 100))
        draw.text((lx + 20, y), label, fill=(200, 200, 200))


# ── Multi-view rendering ──────────────────────────────────────────────

def render_multiview(
    positions: np.ndarray,
    rest_positions: np.ndarray,
    triangles: np.ndarray,
    output_dir: str | Path,
    prefix: str = "",
    views: dict | None = None,
    subsample: int = 1,
    width: int = 800,
    height: int = 1000,
) -> dict[str, Path]:
    """Render mesh from multiple camera angles.

    Parameters
    ----------
    views : dict of {name: {azimuth, elevation}}, or None for QUICK_VIEWS

    Returns
    -------
    dict mapping view name to output file path
    """
    if views is None:
        views = {k: CAMERA_PRESETS[k] for k in QUICK_VIEWS}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for vname, params in views.items():
        fname = f"{prefix}_{vname}.png" if prefix else f"{vname}.png"
        fpath = output_dir / fname
        render_mesh(
            positions, rest_positions, triangles,
            azimuth=params["azimuth"],
            elevation=params.get("elevation", 5),
            width=width, height=height,
            output_path=fpath,
            title=f"{prefix} - {vname}" if prefix else vname,
            subsample=subsample,
        )
        paths[vname] = fpath

    return paths
