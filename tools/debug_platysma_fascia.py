#!/usr/bin/env python3
"""Debug visualization for Platysma fascia pinning.

Renders Platysma R at yaw=1.0 with and without fascia pinning,
saves comparison images, and prints diagnostic metrics to identify
why fascia attachment isn't visible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
from PIL import Image, ImageDraw

from faceforge.core.state import FaceState
from faceforge.core.math_utils import quat_identity
from tools.headless_loader import load_headless_scene, apply_head_rotation
from tools.mesh_renderer import render_mesh, orthographic_project, CAMERA_PRESETS

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("debug_platysma_fascia")

OUT_DIR = Path("results/platysma_fascia_debug")


# ── Custom coloring renderers ────────────────────────────────────────

def _color_by_displacement(
    positions: np.ndarray,
    rest: np.ndarray,
    max_disp: float = 5.0,
) -> np.ndarray:
    """Per-vertex color by displacement from rest (V,3) → (V,3) uint8."""
    disp = np.linalg.norm(positions - rest, axis=1)
    t = np.clip(disp / max_disp, 0, 1)
    # Blue(0) → Cyan(0.25) → Green(0.5) → Yellow(0.75) → Red(1)
    colors = np.zeros((len(t), 3), dtype=np.float64)
    for i in range(len(t)):
        v = t[i]
        if v < 0.25:
            s = v / 0.25
            colors[i] = [0, s, 1 - s * 0.5]
        elif v < 0.5:
            s = (v - 0.25) / 0.25
            colors[i] = [0, 1, 0.5 * (1 - s)]
        elif v < 0.75:
            s = (v - 0.5) / 0.25
            colors[i] = [s, 1, 0]
        else:
            s = (v - 0.75) / 0.25
            colors[i] = [1, 1 - s, 0]
    return (colors * 255).astype(np.uint8)


_REGION_PALETTE = {
    0: (50, 100, 230),    # pectoral — blue
    1: (230, 200, 50),    # investing — yellow
    2: (180, 50, 200),    # supraclavicular — purple
    3: (50, 200, 80),     # deltoid — green
    4: (230, 80, 50),     # trapezius — red-orange
    -1: (80, 80, 80),     # unassigned — grey
}


def render_custom_colors(
    positions: np.ndarray,
    triangles: np.ndarray,
    vert_colors: np.ndarray,
    azimuth: float = 0,
    elevation: float = 5,
    width: int = 800,
    height: int = 1000,
    output_path: str | Path | None = None,
    title: str = "",
    bg_color: tuple = (30, 30, 40),
    margin: int = 40,
) -> Image.Image:
    """Render mesh with custom per-vertex colors (no edge-stretch)."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    vert_colors = np.asarray(vert_colors, dtype=np.uint8).reshape(-1, 3)

    sx, sy, depth = orthographic_project(positions, azimuth, elevation)

    xmin, xmax = float(sx.min()), float(sx.max())
    ymin, ymax = float(sy.min()), float(sy.max())
    x_range = max(xmax - xmin, 1e-6)
    y_range = max(ymax - ymin, 1e-6)
    scale = min((width - 2 * margin) / x_range, (height - 2 * margin) / y_range)

    x_off = margin + (width - 2 * margin - x_range * scale) / 2
    y_off = margin + (height - 2 * margin - y_range * scale) / 2
    px = (sx - xmin) * scale + x_off
    py = height - ((sy - ymin) * scale + y_off)

    # Backface culling
    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
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
        img = Image.new("RGB", (width, height), bg_color)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path))
        return img

    visible_tris = triangles[visible_idx]

    # Compute per-triangle color as average of vertex colors
    c0 = vert_colors[visible_tris[:, 0]].astype(np.float64)
    c1 = vert_colors[visible_tris[:, 1]].astype(np.float64)
    c2 = vert_colors[visible_tris[:, 2]].astype(np.float64)
    tri_colors = ((c0 + c1 + c2) / 3).astype(np.uint8)

    # Simple two-sided diffuse lighting
    v0 = positions[visible_tris[:, 0]]
    edge1 = positions[visible_tris[:, 1]] - v0
    edge2 = positions[visible_tris[:, 2]] - v0
    normals = np.cross(edge1, edge2)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(nlen, 1e-10)
    az_r = np.radians(azimuth)
    light_dir = np.array([0.3 * np.cos(az_r) + 0.5 * np.sin(az_r),
                          -0.3 * np.sin(az_r) + 0.5 * np.cos(az_r),
                          0.4])
    light_dir /= np.linalg.norm(light_dir)
    dot = np.abs(np.einsum("ij,j->i", normals, light_dir))
    brightness = 0.35 + 0.65 * np.clip(dot, 0, 1)
    lit_colors = (tri_colors.astype(np.float64) * brightness[:, None])
    lit_colors = np.clip(lit_colors, 0, 255).astype(np.uint8)

    # Depth sort
    tri_depth = (depth[visible_tris[:, 0]] + depth[visible_tris[:, 1]] +
                 depth[visible_tris[:, 2]]) / 3
    sort_order = np.argsort(tri_depth)
    visible_tris = visible_tris[sort_order]
    lit_colors = lit_colors[sort_order]

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    for i in range(len(visible_tris)):
        v0i, v1i, v2i = visible_tris[i]
        pts = [(int(px[v0i] + .5), int(py[v0i] + .5)),
               (int(px[v1i] + .5), int(py[v1i] + .5)),
               (int(px[v2i] + .5), int(py[v2i] + .5))]
        c = tuple(int(x) for x in lit_colors[i])
        draw.polygon(pts, fill=c)

    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))

    return img


# ── Diagnostic helpers ───────────────────────────────────────────────

def get_triangles(geom) -> np.ndarray:
    """Get triangle indices from a BufferGeometry."""
    if geom.has_indices:
        return geom.indices.reshape(-1, 3)
    # Non-indexed: sequential triplets
    n = geom.vertex_count
    return np.arange(n, dtype=np.int32).reshape(-1, 3)


def print_platysma_diagnostics(pd, label: str = ""):
    """Print detailed diagnostics for a Platysma muscle."""
    name = pd.md.defn.get("name", "?")
    rest = pd.rest_positions
    posed = pd.md.mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    fracs = pd.spine_fracs
    disp = np.linalg.norm(posed - rest, axis=1)

    body_mask = fracs < 0.35
    mid_mask = (fracs >= 0.35) & (fracs < 0.65)
    skull_mask = fracs >= 0.65

    logger.info("=== %s %s ===", name, label)
    logger.info("  Total verts: %d", pd.vert_count)
    logger.info("  Body-end (frac<0.35): %d verts, disp: mean=%.4f max=%.4f",
                body_mask.sum(), disp[body_mask].mean(), disp[body_mask].max())
    logger.info("  Mid (0.35-0.65): %d verts, disp: mean=%.4f max=%.4f",
                mid_mask.sum(), disp[mid_mask].mean(), disp[mid_mask].max())
    logger.info("  Skull-end (frac>0.65): %d verts, disp: mean=%.4f max=%.4f",
                skull_mask.sum(), disp[skull_mask].mean(), disp[skull_mask].max())

    if pd.fascia_assignments is not None:
        assigns = pd.fascia_assignments
        for ri, rn in enumerate(pd.fascia_region_names or []):
            mask = assigns == ri
            if mask.any():
                region_disp = disp[mask]
                logger.info("  Region[%d] '%s': %d verts, disp mean=%.4f max=%.4f",
                            ri, rn, mask.sum(),
                            region_disp.mean(), region_disp.max())

    # Vertex position ranges
    logger.info("  Rest X: [%.1f, %.1f], Y: [%.1f, %.1f], Z: [%.1f, %.1f]",
                rest[:, 0].min(), rest[:, 0].max(),
                rest[:, 1].min(), rest[:, 1].max(),
                rest[:, 2].min(), rest[:, 2].max())
    logger.info("  Posed X: [%.1f, %.1f], Y: [%.1f, %.1f], Z: [%.1f, %.1f]",
                posed[:, 0].min(), posed[:, 0].max(),
                posed[:, 1].min(), posed[:, 1].max(),
                posed[:, 2].min(), posed[:, 2].max())


def print_fascia_diagnostics(fascia, bone_anchors):
    """Print fascia target positions and bone status."""
    logger.info("=== FASCIA SYSTEM ===")
    logger.info("  Regions: %s", fascia.region_names)
    for name in fascia.region_names:
        rest = fascia.get_target_rest(name)
        current = fascia.get_target_current(name)
        if rest is not None:
            delta = current - rest if current is not None else None
            delta_str = f"delta=[{delta[0]:.3f},{delta[1]:.3f},{delta[2]:.3f}]" if delta is not None else "current=None!"
            logger.info("  %s: rest=[%.1f,%.1f,%.1f] %s",
                        name, rest[0], rest[1], rest[2], delta_str)
        else:
            logger.info("  %s: rest=None!", name)


# ── Main visualization ───────────────────────────────────────────────

def run_debug():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading headless scene...")
    hs = load_headless_scene()
    pipeline = hs.pipeline

    if pipeline.platysma is None or not pipeline.platysma.registered:
        logger.error("Platysma not registered!")
        return

    logger.info("Platysma registered with %d muscles", len(pipeline.platysma._platysma))

    # Print fascia system status
    if pipeline.fascia is not None:
        print_fascia_diagnostics(pipeline.fascia, pipeline.bone_anchors)

    # ── Step 1: Get Platysma R data ──
    pd = pipeline.platysma._platysma[0]
    name = pd.md.defn.get("name", "?")
    geom = pd.md.mesh.geometry
    triangles = get_triangles(geom)
    rest = pd.rest_positions.copy()

    logger.info("Platysma R: %d verts, %d triangles, has_indices=%s",
                pd.vert_count, len(triangles), geom.has_indices)

    # ── Step 2: Render at REST ──
    logger.info("\n--- REST POSE ---")
    face_rest = FaceState()
    apply_head_rotation(hs, face_rest)
    print_platysma_diagnostics(pd, "REST")

    posed_rest = geom.positions.reshape(-1, 3).astype(np.float64).copy()
    for view in ["front", "3q_front_r"]:
        vp = CAMERA_PRESETS[view]
        render_mesh(posed_rest, rest, triangles,
                    azimuth=vp["azimuth"], elevation=vp.get("elevation", 5),
                    output_path=OUT_DIR / f"01_rest_{view}.png",
                    title=f"{name} REST {view}")

    # ── Step 3: Render at YAW=1.0 WITH fascia ──
    logger.info("\n--- YAW=1.0 WITH FASCIA ---")
    # Reset mesh to rest before applying
    geom.positions[:] = pd.md.rest_positions.copy()
    face_yaw = FaceState()
    face_yaw.head_yaw = 1.0
    apply_head_rotation(hs, face_yaw)
    print_platysma_diagnostics(pd, "YAW=1.0 WITH FASCIA")

    posed_with = geom.positions.reshape(-1, 3).astype(np.float64).copy()
    for view in ["front", "3q_front_r", "right"]:
        vp = CAMERA_PRESETS[view]
        render_mesh(posed_with, rest, triangles,
                    azimuth=vp["azimuth"], elevation=vp.get("elevation", 5),
                    output_path=OUT_DIR / f"02_yaw1_with_fascia_{view}.png",
                    title=f"{name} YAW=1.0 WITH fascia {view}")

    # Color by displacement
    disp_colors = _color_by_displacement(posed_with, rest, max_disp=8.0)
    for view in ["front", "3q_front_r"]:
        vp = CAMERA_PRESETS[view]
        render_custom_colors(posed_with, triangles, disp_colors,
                             azimuth=vp["azimuth"], elevation=vp.get("elevation", 5),
                             output_path=OUT_DIR / f"03_yaw1_displacement_{view}.png",
                             title=f"{name} YAW=1.0 displacement {view}")

    # Color by fascia region
    if pd.fascia_assignments is not None:
        region_colors = np.zeros((pd.vert_count, 3), dtype=np.uint8)
        for vi in range(pd.vert_count):
            region_colors[vi] = _REGION_PALETTE.get(int(pd.fascia_assignments[vi]),
                                                     (80, 80, 80))
        for view in ["front", "3q_front_r"]:
            vp = CAMERA_PRESETS[view]
            render_custom_colors(posed_with, triangles, region_colors,
                                 azimuth=vp["azimuth"], elevation=vp.get("elevation", 5),
                                 output_path=OUT_DIR / f"04_fascia_regions_{view}.png",
                                 title=f"{name} fascia regions {view}")

    # ── Step 4: Render at YAW=1.0 WITHOUT fascia ──
    logger.info("\n--- YAW=1.0 WITHOUT FASCIA ---")
    # Reset and disable fascia
    geom.positions[:] = pd.md.rest_positions.copy()
    apply_head_rotation(hs, face_rest)  # reset to identity first

    saved_fascia = pipeline.platysma._fascia
    pipeline.platysma._fascia = None

    geom.positions[:] = pd.md.rest_positions.copy()
    apply_head_rotation(hs, face_yaw)
    print_platysma_diagnostics(pd, "YAW=1.0 WITHOUT FASCIA")

    posed_without = geom.positions.reshape(-1, 3).astype(np.float64).copy()
    for view in ["front", "3q_front_r", "right"]:
        vp = CAMERA_PRESETS[view]
        render_mesh(posed_without, rest, triangles,
                    azimuth=vp["azimuth"], elevation=vp.get("elevation", 5),
                    output_path=OUT_DIR / f"05_yaw1_without_fascia_{view}.png",
                    title=f"{name} YAW=1.0 WITHOUT fascia {view}")

    # Restore fascia
    pipeline.platysma._fascia = saved_fascia

    # ── Step 5: Difference analysis ──
    logger.info("\n--- DIFFERENCE ANALYSIS ---")
    diff = posed_with - posed_without
    diff_mag = np.linalg.norm(diff, axis=1)
    fracs = pd.spine_fracs
    body_mask = fracs < 0.35

    logger.info("  Overall position diff: mean=%.6f max=%.6f",
                diff_mag.mean(), diff_mag.max())
    logger.info("  Body-end diff: mean=%.6f max=%.6f",
                diff_mag[body_mask].mean(), diff_mag[body_mask].max())
    logger.info("  Non-body diff: mean=%.6f max=%.6f",
                diff_mag[~body_mask].mean(), diff_mag[~body_mask].max())

    # Color by difference magnitude
    diff_colors = _color_by_displacement(posed_with, posed_without, max_disp=2.0)
    for view in ["front", "3q_front_r"]:
        vp = CAMERA_PRESETS[view]
        render_custom_colors(posed_with, triangles, diff_colors,
                             azimuth=vp["azimuth"], elevation=vp.get("elevation", 5),
                             output_path=OUT_DIR / f"06_fascia_diff_{view}.png",
                             title=f"{name} fascia effect (with-without) {view}")

    # ── Step 6: Examine _apply_fascia_pinning internals ──
    logger.info("\n--- FASCIA PINNING INTERNALS ---")
    if pd.fascia_assignments is not None and pd.fascia_region_names is not None:
        for ri, rn in enumerate(pd.fascia_region_names):
            mask = pd.fascia_assignments == ri
            if not mask.any():
                continue
            target_rest = pipeline.fascia.get_target_rest(rn)
            target_current = pipeline.fascia.get_target_current(rn)
            if target_rest is None:
                logger.warning("  Region '%s': target_rest is None!", rn)
                continue
            if target_current is None:
                logger.warning("  Region '%s': target_current is None!", rn)
                continue
            delta = target_current - target_rest
            logger.info("  Region '%s': %d verts, rest=[%.2f,%.2f,%.2f] "
                        "current=[%.2f,%.2f,%.2f] delta=[%.4f,%.4f,%.4f] |delta|=%.4f",
                        rn, mask.sum(),
                        target_rest[0], target_rest[1], target_rest[2],
                        target_current[0], target_current[1], target_current[2],
                        delta[0], delta[1], delta[2], np.linalg.norm(delta))

            # What pin_weight do these verts get?
            pin_fracs = fracs[mask].astype(np.float64)
            pin_weight = np.clip(1.0 - pin_fracs, 0.0, 1.0) ** 2
            logger.info("    pin_weight: mean=%.4f min=%.4f max=%.4f",
                        pin_weight.mean(), pin_weight.min(), pin_weight.max())

    # ── Step 7: Check if body-end verts actually move ──
    logger.info("\n--- BODY-END VERTEX ANALYSIS ---")
    body_rest = rest[body_mask]
    body_with = posed_with[body_mask]
    body_without = posed_without[body_mask]
    body_fracs = fracs[body_mask]

    logger.info("  Body-end verts: %d (frac range: [%.4f, %.4f])",
                body_mask.sum(), body_fracs.min(), body_fracs.max())

    # What rotation angles do body-end verts get?
    # Full yaw angle at yaw=1.0 is 45 degrees
    import math
    full_angle_deg = 45.0  # max yaw
    body_angles = body_fracs * full_angle_deg
    logger.info("  Body-end rotation angles: mean=%.2f° max=%.2f°",
                body_angles.mean(), body_angles.max())

    # How far do they move from rest WITHOUT fascia?
    body_disp_no_fascia = np.linalg.norm(body_without - body_rest, axis=1)
    logger.info("  Without fascia: body-end displacement mean=%.4f max=%.4f",
                body_disp_no_fascia.mean(), body_disp_no_fascia.max())

    # How far do they move from rest WITH fascia?
    body_disp_with_fascia = np.linalg.norm(body_with - body_rest, axis=1)
    logger.info("  With fascia: body-end displacement mean=%.4f max=%.4f",
                body_disp_with_fascia.mean(), body_disp_with_fascia.max())

    logger.info("\n=== Images saved to %s ===", OUT_DIR.resolve())
    logger.info("  01_rest_*.png — Platysma at rest")
    logger.info("  02_yaw1_with_fascia_*.png — Yaw=1.0 WITH fascia")
    logger.info("  03_yaw1_displacement_*.png — Displacement heatmap (WITH fascia)")
    logger.info("  04_fascia_regions_*.png — Vertex fascia region assignments")
    logger.info("  05_yaw1_without_fascia_*.png — Yaw=1.0 WITHOUT fascia")
    logger.info("  06_fascia_diff_*.png — Difference (with - without fascia)")


if __name__ == "__main__":
    run_debug()
