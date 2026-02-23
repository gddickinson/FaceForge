"""Visual test for gender bone scaling — comprehensive comparison suite.

Produces side-by-side, overlay, and heatmap images comparing male (gender=0)
and female (gender=1) skeleton meshes.  Includes scale bars and per-bone
proportion change annotations.

Usage::

    cd faceforge
    python -m tools.gender_scaling_test
    # Results saved to results/gender/
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from faceforge.loaders.asset_manager import AssetManager
from faceforge.coordination.scene_builder import SceneBuilder
from faceforge.coordination.visibility import VisibilityManager
from faceforge.coordination.loading_pipeline import LoadingPipeline
from faceforge.core.events import EventBus
from faceforge.body.bone_scaling import BoneScaler
from faceforge.body.gender_morph import GenderMorphSystem
from tools.mesh_renderer import orthographic_project

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "gender"

# ── Colors ──────────────────────────────────────────────────────────────
MALE_COLOR = np.array([100, 160, 220], dtype=np.float32)    # Blue
FEMALE_COLOR = np.array([220, 100, 120], dtype=np.float32)  # Red/Pink
BONE_COLOR = np.array([212, 184, 150], dtype=np.float32)    # Bone
BG_COLOR = (30, 30, 40)
TEXT_COLOR = (255, 255, 255)
SCALE_BAR_COLOR = (200, 200, 200)
GRID_COLOR = (60, 60, 70)


# ── Mesh collection ─────────────────────────────────────────────────────

def collect_all_bone_meshes(skeleton):
    """Recursively collect (name, mesh) from all skeleton groups."""
    bone_meshes = []

    def _walk(node):
        if node.mesh is not None and node.name:
            bone_meshes.append((node.name, node.mesh))
        for ch in node.children:
            _walk(ch)

    for group_node in skeleton.groups.values():
        _walk(group_node)
    return bone_meshes


def collect_group_meshes(skeleton, group_key):
    """Collect (name, mesh) from a specific skeleton group."""
    meshes = []
    group = skeleton.groups.get(group_key)
    if group is None:
        return meshes

    def _walk(node):
        if node.mesh is not None and node.name:
            meshes.append((node.name, node.mesh))
        for ch in node.children:
            _walk(ch)

    _walk(group)
    return meshes


# ── Geometry extraction ─────────────────────────────────────────────────

def extract_geometry(bone_meshes):
    """Combine bone meshes into a single (positions, triangles) array."""
    all_positions = []
    all_triangles = []
    bone_ids = []  # per-vertex bone index for coloring
    vert_offset = 0

    for bi, (name, mesh) in enumerate(bone_meshes):
        pos = mesh.geometry.positions.reshape(-1, 3)
        all_positions.append(pos)
        bone_ids.extend([bi] * len(pos))
        if mesh.geometry.indices is not None:
            tris = mesh.geometry.indices.reshape(-1, 3) + vert_offset
            all_triangles.append(tris)
        vert_offset += len(pos)

    if not all_positions:
        return None, None, None

    positions = np.vstack(all_positions)
    triangles = np.vstack(all_triangles) if all_triangles else np.zeros((0, 3), dtype=np.uint32)
    bone_ids = np.array(bone_ids, dtype=np.int32)
    return positions, triangles, bone_ids


# ── Projection helpers ──────────────────────────────────────────────────

def project_and_fit(positions, azimuth, elevation, width, height, margin=40,
                    fixed_center=None, fixed_scale=None):
    """Project positions and fit to viewport. Returns (screen_x, screen_y, depths, scale, center)."""
    screen_x, screen_y, depths = orthographic_project(positions, azimuth, elevation)

    x_min, x_max = screen_x.min(), screen_x.max()
    y_min, y_max = screen_y.min(), screen_y.max()
    w_range = max(x_max - x_min, 1.0)
    h_range = max(y_max - y_min, 1.0)

    if fixed_scale is not None:
        scale = fixed_scale
    else:
        scale = min((width - 2 * margin) / w_range, (height - 2 * margin) / h_range)

    if fixed_center is not None:
        cx, cy = fixed_center
    else:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

    sx = (screen_x - cx) * scale + width / 2
    sy = (screen_y - cy) * scale + height / 2

    return sx, sy, depths, scale, (cx, cy)


def compute_face_normal(positions, i0, i1, i2):
    """Compute face normal and lighting intensity."""
    v0, v1, v2 = positions[i0], positions[i1], positions[i2]
    e1 = v1 - v0
    e2 = v2 - v0
    fn = np.cross(e1, e2)
    fn_len = np.linalg.norm(fn)
    if fn_len < 1e-10:
        return 0.0
    fn /= fn_len
    light_dir = np.array([0.3, -0.8, 0.5])
    light_dir /= np.linalg.norm(light_dir)
    return max(0.15, np.dot(fn, light_dir))


# ── Drawing primitives ──────────────────────────────────────────────────

def draw_triangles(draw, positions, triangles, screen_x, screen_y, depths,
                   base_color, alpha=1.0, bone_ids=None, bone_colors=None):
    """Draw filled triangles with lighting and optional per-bone coloring."""
    if len(triangles) == 0:
        return

    face_depths = depths[triangles].mean(axis=1)
    order = np.argsort(face_depths)

    for fi in order:
        i0, i1, i2 = triangles[fi]
        p0x, p0y = screen_x[i0], screen_y[i0]
        p1x, p1y = screen_x[i1], screen_y[i1]
        p2x, p2y = screen_x[i2], screen_y[i2]

        # Backface culling
        cross = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
        if cross <= 0:
            continue

        intensity = compute_face_normal(positions, i0, i1, i2)
        if intensity == 0.0:
            continue

        if bone_colors is not None and bone_ids is not None:
            bc = bone_colors[bone_ids[i0]]
        else:
            bc = base_color

        color = (bc * intensity * alpha).clip(0, 255).astype(np.uint8)
        poly = [(p0x, p0y), (p1x, p1y), (p2x, p2y)]
        draw.polygon(poly, fill=tuple(color), outline=tuple(color))


def draw_wireframe(draw, triangles, screen_x, screen_y, depths,
                   color=(100, 100, 120), alpha=0.5):
    """Draw triangle wireframe edges."""
    if len(triangles) == 0:
        return

    face_depths = depths[triangles].mean(axis=1)
    order = np.argsort(face_depths)

    r, g, b = color
    edge_color = (int(r * alpha), int(g * alpha), int(b * alpha))

    for fi in order[-len(order) // 3:]:  # Only draw nearest third
        i0, i1, i2 = triangles[fi]
        p0x, p0y = screen_x[i0], screen_y[i0]
        p1x, p1y = screen_x[i1], screen_y[i1]
        p2x, p2y = screen_x[i2], screen_y[i2]
        draw.line([(p0x, p0y), (p1x, p1y)], fill=edge_color, width=1)
        draw.line([(p1x, p1y), (p2x, p2y)], fill=edge_color, width=1)
        draw.line([(p2x, p2y), (p0x, p0y)], fill=edge_color, width=1)


def draw_scale_bar(draw, x, y, length_px, length_mm, label=None, color=SCALE_BAR_COLOR):
    """Draw a horizontal scale bar with tick marks."""
    # Main bar
    draw.line([(x, y), (x + length_px, y)], fill=color, width=2)
    # End ticks
    draw.line([(x, y - 5), (x, y + 5)], fill=color, width=2)
    draw.line([(x + length_px, y - 5), (x + length_px, y + 5)], fill=color, width=2)
    # Label
    text = label or f"{length_mm:.0f} mm"
    draw.text((x + length_px // 2 - 15, y + 8), text, fill=color)


def draw_dimension_lines(draw, screen_x, screen_y, positions, label_prefix,
                         scale, width, height, color=SCALE_BAR_COLOR):
    """Draw bounding dimension annotations (width/height in mm)."""
    x_min_s, x_max_s = screen_x.min(), screen_x.max()
    y_min_s, y_max_s = screen_y.min(), screen_y.max()

    # Width dimension at bottom
    y_dim = min(y_max_s + 30, height - 30)
    draw.line([(x_min_s, y_dim), (x_max_s, y_dim)], fill=color, width=1)
    draw.line([(x_min_s, y_dim - 4), (x_min_s, y_dim + 4)], fill=color, width=1)
    draw.line([(x_max_s, y_dim - 4), (x_max_s, y_dim + 4)], fill=color, width=1)

    # Compute real-world width from positions
    real_width = positions[:, 0].max() - positions[:, 0].min()
    mid_x = (x_min_s + x_max_s) / 2
    draw.text((mid_x - 20, y_dim + 6), f"{label_prefix}W: {real_width:.1f}mm", fill=color)

    # Height dimension on right
    x_dim = min(x_max_s + 20, width - 60)
    draw.line([(x_dim, y_min_s), (x_dim, y_max_s)], fill=color, width=1)
    draw.line([(x_dim - 4, y_min_s), (x_dim + 4, y_min_s)], fill=color, width=1)
    draw.line([(x_dim - 4, y_max_s), (x_dim + 4, y_max_s)], fill=color, width=1)

    real_height = positions[:, 2].max() - positions[:, 2].min()
    mid_y = (y_min_s + y_max_s) / 2
    draw.text((x_dim + 6, mid_y - 6), f"H: {real_height:.1f}", fill=color)


def draw_legend(draw, x, y, items, box_size=12, spacing=20):
    """Draw a color legend. items = [(color_tuple, label), ...]"""
    for i, (color, label) in enumerate(items):
        yy = y + i * spacing
        draw.rectangle([(x, yy), (x + box_size, yy + box_size)], fill=color)
        draw.text((x + box_size + 6, yy - 1), label, fill=TEXT_COLOR)


# ── Proportion heatmap color ────────────────────────────────────────────

def proportion_to_color(ratio):
    """Map a scale ratio to a color.

    1.0 = neutral gray, <1.0 = blue (shrunk), >1.0 = red (expanded).
    """
    if ratio < 1.0:
        # Shrink: gray → blue
        t = min(1.0, (1.0 - ratio) / 0.15)  # Full blue at 15% shrink
        r = int(180 * (1 - t) + 60 * t)
        g = int(180 * (1 - t) + 100 * t)
        b = int(180 * (1 - t) + 240 * t)
    else:
        # Expand: gray → red
        t = min(1.0, (ratio - 1.0) / 0.15)  # Full red at 15% expand
        r = int(180 * (1 - t) + 240 * t)
        g = int(180 * (1 - t) + 80 * t)
        b = int(180 * (1 - t) + 60 * t)
    return np.array([r, g, b], dtype=np.float32)


# ── Main rendering functions ────────────────────────────────────────────

def render_single(bone_meshes, output_path, title="", azimuth=0, elevation=5,
                  width=800, height=600, base_color=BONE_COLOR, show_scale=True):
    """Render a bone group with scale bars and dimensions."""
    positions, triangles, bone_ids = extract_geometry(bone_meshes)
    if positions is None:
        img = Image.new("RGB", (width, height), BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "No meshes", fill=TEXT_COLOR)
        img.save(output_path)
        return img

    sx, sy, depths, scale, center = project_and_fit(
        positions, azimuth, elevation, width, height, margin=60
    )

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw_triangles(draw, positions, triangles, sx, sy, depths, base_color)

    if show_scale:
        # Scale bar: 50mm in screen pixels
        bar_px = 50.0 * scale
        if bar_px > 20:
            draw_scale_bar(draw, 20, height - 25, int(bar_px), 50.0)
        draw_dimension_lines(draw, sx, sy, positions, "", scale, width, height)

    if title:
        draw.text((10, 10), title, fill=TEXT_COLOR)

    img.save(output_path)
    return img


def render_side_by_side(male_meshes, female_meshes, output_path, title="",
                        azimuth=0, elevation=5, width=1400, height=700):
    """Render male (left) and female (right) side-by-side with shared scale."""
    half_w = width // 2
    pos_m, tri_m, bids_m = extract_geometry(male_meshes)
    pos_f, tri_f, bids_f = extract_geometry(female_meshes)

    if pos_m is None or pos_f is None:
        img = Image.new("RGB", (width, height), BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Missing meshes", fill=TEXT_COLOR)
        img.save(output_path)
        return img

    # Use shared viewport: compute from combined bounds
    all_pos = np.vstack([pos_m, pos_f])
    _, _, _, shared_scale, shared_center = project_and_fit(
        all_pos, azimuth, elevation, half_w, height, margin=60
    )

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Divider line
    draw.line([(half_w, 0), (half_w, height)], fill=(80, 80, 100), width=2)

    # Male (left panel)
    sx_m, sy_m, d_m = orthographic_project(pos_m, azimuth, elevation)
    cx, cy = shared_center
    sx_m = (sx_m - cx) * shared_scale + half_w / 2
    sy_m = (sy_m - cy) * shared_scale + height / 2
    draw_triangles(draw, pos_m, tri_m, sx_m, sy_m, d_m, MALE_COLOR * 0.85 + BONE_COLOR * 0.15)

    # Female (right panel)
    sx_f, sy_f, d_f = orthographic_project(pos_f, azimuth, elevation)
    sx_f = (sx_f - cx) * shared_scale + half_w + half_w / 2
    sy_f = (sy_f - cy) * shared_scale + height / 2
    draw_triangles(draw, pos_f, tri_f, sx_f, sy_f, d_f, FEMALE_COLOR * 0.7 + BONE_COLOR * 0.3)

    # Dimensions
    m_w = pos_m[:, 0].max() - pos_m[:, 0].min()
    m_h = pos_m[:, 2].max() - pos_m[:, 2].min()
    f_w = pos_f[:, 0].max() - pos_f[:, 0].min()
    f_h = pos_f[:, 2].max() - pos_f[:, 2].min()

    draw.text((10, height - 60), f"Width: {m_w:.1f}mm", fill=(140, 190, 240))
    draw.text((10, height - 40), f"Height: {m_h:.1f}mm", fill=(140, 190, 240))
    draw.text((half_w + 10, height - 60), f"Width: {f_w:.1f}mm", fill=(240, 140, 150))
    draw.text((half_w + 10, height - 40), f"Height: {f_h:.1f}mm", fill=(240, 140, 150))

    # Ratio annotations
    if m_w > 0 and m_h > 0:
        w_ratio = f_w / m_w
        h_ratio = f_h / m_h
        draw.text((half_w + 10, height - 20),
                  f"W ratio: {w_ratio:.3f}  H ratio: {h_ratio:.3f}",
                  fill=(200, 200, 200))

    # Scale bar
    bar_px = 50.0 * shared_scale
    if bar_px > 20:
        draw_scale_bar(draw, width // 2 - int(bar_px) // 2, height - 15,
                       int(bar_px), 50.0)

    # Title and labels
    if title:
        draw.text((width // 2 - len(title) * 3, 5), title, fill=TEXT_COLOR)
    draw.text((half_w // 2 - 20, 25), "MALE (g=0.0)", fill=(140, 190, 240))
    draw.text((half_w + half_w // 2 - 30, 25), "FEMALE (g=1.0)", fill=(240, 140, 150))

    # Legend
    draw_legend(draw, width - 160, 10, [
        (tuple(MALE_COLOR.astype(int)), "Male"),
        (tuple(FEMALE_COLOR.astype(int)), "Female"),
    ])

    img.save(output_path)
    return img


def render_overlay(male_meshes, female_meshes, output_path, title="",
                   azimuth=0, elevation=5, width=800, height=700):
    """Overlay male and female meshes with transparency to show differences."""
    pos_m, tri_m, _ = extract_geometry(male_meshes)
    pos_f, tri_f, _ = extract_geometry(female_meshes)

    if pos_m is None or pos_f is None:
        img = Image.new("RGB", (width, height), BG_COLOR)
        img.save(output_path)
        return img

    # Shared viewport
    all_pos = np.vstack([pos_m, pos_f])
    _, _, _, shared_scale, shared_center = project_and_fit(
        all_pos, azimuth, elevation, width, height, margin=60
    )
    cx, cy = shared_center

    # Render male first (behind), female on top with slight transparency effect
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Male: faded blue
    sx_m, sy_m, d_m = orthographic_project(pos_m, azimuth, elevation)
    sx_m = (sx_m - cx) * shared_scale + width / 2
    sy_m = (sy_m - cy) * shared_scale + height / 2
    draw_triangles(draw, pos_m, tri_m, sx_m, sy_m, d_m, MALE_COLOR, alpha=0.5)

    # Female: wireframe overlay in red
    sx_f, sy_f, d_f = orthographic_project(pos_f, azimuth, elevation)
    sx_f = (sx_f - cx) * shared_scale + width / 2
    sy_f = (sy_f - cy) * shared_scale + height / 2
    draw_wireframe(draw, tri_f, sx_f, sy_f, d_f, color=(240, 100, 120), alpha=0.8)

    # Dimensions comparison
    m_w = pos_m[:, 0].max() - pos_m[:, 0].min()
    m_h = pos_m[:, 2].max() - pos_m[:, 2].min()
    f_w = pos_f[:, 0].max() - pos_f[:, 0].min()
    f_h = pos_f[:, 2].max() - pos_f[:, 2].min()

    y_text = height - 80
    draw.text((10, y_text), "Male:   ", fill=(140, 190, 240))
    draw.text((60, y_text), f"W={m_w:.1f}mm  H={m_h:.1f}mm", fill=(140, 190, 240))
    draw.text((10, y_text + 18), "Female: ", fill=(240, 140, 150))
    draw.text((60, y_text + 18), f"W={f_w:.1f}mm  H={f_h:.1f}mm", fill=(240, 140, 150))
    if m_w > 0:
        draw.text((10, y_text + 36),
                  f"Change: W={((f_w/m_w)-1)*100:+.1f}%  H={((f_h/m_h)-1)*100:+.1f}%",
                  fill=(200, 200, 200))

    # Scale bar
    bar_px = 50.0 * shared_scale
    if bar_px > 20:
        draw_scale_bar(draw, 20, height - 20, int(bar_px), 50.0)

    if title:
        draw.text((10, 10), title, fill=TEXT_COLOR)

    draw_legend(draw, width - 180, 10, [
        ((100, 160, 220), "Male (solid)"),
        ((240, 100, 120), "Female (wireframe)"),
    ])

    img.save(output_path)
    return img


def render_proportion_heatmap(male_meshes, female_meshes, bone_scaler,
                              output_path, title="", azimuth=0, elevation=5,
                              width=800, height=700):
    """Render female skeleton colored by per-bone scale ratio."""
    # We need per-bone scale ratios
    bone_ratios = {}
    for name, mesh in female_meshes:
        scale = bone_scaler.compute_scale(name, 1.0)
        if scale:
            # Average of the three axis scales
            avg = (scale[0] + scale[1] + scale[2]) / 3.0
            bone_ratios[name] = avg

    pos_f, tri_f, bids_f = extract_geometry(female_meshes)
    if pos_f is None:
        img = Image.new("RGB", (width, height), BG_COLOR)
        img.save(output_path)
        return img

    # Build per-bone color array
    bone_names = [name for name, _ in female_meshes]
    bone_colors = np.zeros((len(bone_names), 3), dtype=np.float32)
    for i, name in enumerate(bone_names):
        ratio = bone_ratios.get(name, 1.0)
        bone_colors[i] = proportion_to_color(ratio)

    sx, sy, depths, scale, center = project_and_fit(
        pos_f, azimuth, elevation, width, height, margin=80
    )

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    draw_triangles(draw, pos_f, tri_f, sx, sy, depths,
                   base_color=BONE_COLOR, bone_ids=bids_f, bone_colors=bone_colors)

    # Color bar legend
    bar_x = width - 50
    bar_h = height - 120
    bar_top = 60
    n_steps = 20
    step_h = bar_h // n_steps
    for i in range(n_steps):
        ratio = 0.85 + (1.15 - 0.85) * i / n_steps
        c = proportion_to_color(ratio)
        yy = bar_top + (n_steps - 1 - i) * step_h
        draw.rectangle([(bar_x, yy), (bar_x + 25, yy + step_h)],
                       fill=tuple(c.astype(int)))

    draw.text((bar_x - 25, bar_top - 15), "Expand", fill=(240, 120, 80))
    draw.text((bar_x - 25, bar_top + bar_h + 5), "Shrink", fill=(80, 120, 240))
    draw.text((bar_x - 5, bar_top + bar_h // 2 - 6), "1.0", fill=TEXT_COLOR)

    # Scale ticks
    for ratio, label in [(0.85, "0.85"), (0.90, "0.90"), (0.95, "0.95"),
                          (1.00, "1.00"), (1.05, "1.05"), (1.10, "1.10")]:
        t = (ratio - 0.85) / (1.15 - 0.85)
        yy = bar_top + int((1 - t) * bar_h)
        draw.text((bar_x + 28, yy - 6), label, fill=(180, 180, 180))

    # Scale bar
    bar_px = 50.0 * scale
    if bar_px > 20:
        draw_scale_bar(draw, 20, height - 20, int(bar_px), 50.0)

    if title:
        draw.text((10, 10), title, fill=TEXT_COLOR)

    img.save(output_path)
    return img


def render_bone_table(male_extents, female_extents, output_path, width=900, height=None):
    """Render a table image showing per-bone dimension changes."""
    # Collect bones that changed
    rows = []
    for name in sorted(male_extents.keys()):
        if name not in female_extents:
            continue
        m = male_extents[name]
        f = female_extents[name]
        ratio_x = f["extent"][0] / max(m["extent"][0], 1e-6)
        ratio_y = f["extent"][1] / max(m["extent"][1], 1e-6)
        ratio_z = f["extent"][2] / max(m["extent"][2], 1e-6)
        if any(abs(r - 1.0) > 0.005 for r in [ratio_x, ratio_y, ratio_z]):
            rows.append((name, ratio_x, ratio_y, ratio_z,
                         m["extent"][0], m["extent"][1], m["extent"][2]))

    if not rows:
        return

    row_h = 18
    if height is None:
        height = 60 + len(rows) * row_h + 20

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((10, 10), "Per-Bone Dimension Changes: Male → Female", fill=TEXT_COLOR)
    y = 35
    header = f"{'Bone Name':<32s} {'X ratio':>8s} {'Y ratio':>8s} {'Z ratio':>8s}  {'Mx(mm)':>7s} {'My(mm)':>7s} {'Mz(mm)':>7s}"
    draw.text((10, y), header, fill=(180, 180, 200))
    y += row_h + 2
    draw.line([(10, y), (width - 10, y)], fill=(80, 80, 100), width=1)
    y += 4

    for name, rx, ry, rz, mx, my, mz in rows:
        # Color code ratios
        cx = proportion_to_color(rx)
        cy_c = proportion_to_color(ry)
        cz = proportion_to_color(rz)

        draw.text((10, y), f"{name[:30]:<32s}", fill=(200, 200, 200))
        draw.text((270, y), f"{rx:>7.3f}", fill=tuple(cx.astype(int)))
        draw.text((340, y), f"{ry:>7.3f}", fill=tuple(cy_c.astype(int)))
        draw.text((410, y), f"{rz:>7.3f}", fill=tuple(cz.astype(int)))
        draw.text((490, y), f"{mx:>7.1f}", fill=(160, 160, 160))
        draw.text((560, y), f"{my:>7.1f}", fill=(160, 160, 160))
        draw.text((630, y), f"{mz:>7.1f}", fill=(160, 160, 160))
        y += row_h

    img.save(output_path)
    return img


# ── Measurement ─────────────────────────────────────────────────────────

def measure_bone_extents(bone_meshes):
    """Measure bounding box extents per bone."""
    results = {}
    for name, mesh in bone_meshes:
        pos = mesh.geometry.positions.reshape(-1, 3)
        results[name] = {
            "centroid": pos.mean(axis=0),
            "min": pos.min(axis=0),
            "max": pos.max(axis=0),
            "extent": pos.max(axis=0) - pos.min(axis=0),
            "n_verts": len(pos),
        }
    return results


def snapshot_positions(bone_meshes):
    """Take a snapshot of current vertex positions for each mesh."""
    return {name: mesh.geometry.positions.copy() for name, mesh in bone_meshes}


def restore_positions(bone_meshes, snapshot):
    """Restore vertex positions from a snapshot (used for switching between states)."""
    for name, mesh in bone_meshes:
        if name in snapshot:
            mesh.geometry.positions = snapshot[name].copy()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load scene
    logger.info("Loading assets...")
    assets = AssetManager()
    assets.init_transform()
    visibility = VisibilityManager()
    builder = SceneBuilder(assets, visibility)
    scene, named_nodes = builder.build()

    event_bus = EventBus()
    pipeline = LoadingPipeline(assets, event_bus, named_nodes)
    pipeline.load_head(skull_mode="bp3d")
    pipeline.load_body_skeleton()

    skeleton = pipeline.skeleton
    if skeleton is None:
        logger.error("No skeleton loaded")
        return

    # Also collect skull/head meshes
    skull_group = named_nodes.get("skullGroup")
    skull_meshes = []
    if skull_group:
        def _walk_skull(node):
            if node.mesh is not None and node.name:
                skull_meshes.append((node.name, node.mesh))
            for ch in node.children:
                _walk_skull(ch)
        _walk_skull(skull_group)
    logger.info("Skull meshes: %d", len(skull_meshes))

    # Collect all bone meshes (body skeleton)
    all_bones = collect_all_bone_meshes(skeleton)
    logger.info("Body bone meshes: %d", len(all_bones))

    # Combined: body + skull
    combined = all_bones + skull_meshes

    # Groups to compare
    groups_to_render = {
        "pelvis": "Pelvis",
        "ribs": "Rib Cage",
        "thoracic": "Thoracic Spine",
        "upper_limb": "Upper Limb",
        "lower_limb": "Lower Limb",
    }

    # Create bone scaler
    gm = GenderMorphSystem()
    bone_scaler = gm.bone_scaler

    # ─── Step 1: Snapshot male positions ─────────────────────────
    logger.info("=== Snapshotting male (gender=0.0) ===")
    bone_scaler.apply_to_all(combined, 0.0)
    male_snapshot = snapshot_positions(combined)
    male_extents = measure_bone_extents(combined)

    # Log unmatched bones
    unmatched = []
    for name, _ in combined:
        scale = bone_scaler.compute_scale(name, 1.0)
        if scale is None:
            unmatched.append(name)
    if unmatched:
        logger.warning("Unmatched bones (%d): %s", len(unmatched),
                       ", ".join(unmatched[:20]))
    else:
        logger.info("All %d bones matched scale patterns", len(combined))

    # ─── Step 2: Apply female scaling ────────────────────────────
    logger.info("=== Applying female (gender=1.0) ===")
    n_scaled = bone_scaler.apply_to_all(combined, 1.0)
    logger.info("Scaled %d/%d meshes", n_scaled, len(combined))
    female_snapshot = snapshot_positions(combined)
    female_extents = measure_bone_extents(combined)

    # ─── Step 3: Revert to male and verify ──────────────────────
    bone_scaler.apply_to_all(combined, 0.0)
    revert_extents = measure_bone_extents(combined)
    all_match = True
    for name in male_extents:
        if name in revert_extents:
            diff = np.linalg.norm(
                revert_extents[name]["centroid"] - male_extents[name]["centroid"]
            )
            if diff > 0.01:
                logger.warning("  MISMATCH: %s centroid diff=%.4f", name, diff)
                all_match = False
    logger.info("Revert check: %s", "ALL PASS" if all_match else "FAILED")

    # ─── Step 4: Per-bone dimension table ────────────────────────
    logger.info("=== Generating comparison images ===")
    render_bone_table(male_extents, female_extents,
                      RESULTS_DIR / "bone_table.png")
    logger.info("  bone_table.png")

    # ─── Step 5: Full skeleton comparisons ──────────────────────
    for view_name, az, el in [("front", 0, 5), ("3q", 30, 15), ("side", 90, 5)]:
        # Side-by-side
        restore_positions(combined, male_snapshot)
        male_copy = [(n, m) for n, m in combined]
        restore_positions(combined, female_snapshot)
        female_copy = [(n, m) for n, m in combined]

        # For side-by-side we need separate position arrays, so we build
        # temporary geometry from snapshots
        render_side_by_side_from_snapshots(
            combined, male_snapshot, female_snapshot,
            RESULTS_DIR / f"sidebyside_{view_name}.png",
            title=f"Male vs Female Skeleton — {view_name}",
            azimuth=az, elevation=el,
            width=1400, height=1200 if view_name != "side" else 800,
        )
        logger.info("  sidebyside_%s.png", view_name)

        # Overlay
        render_overlay_from_snapshots(
            combined, male_snapshot, female_snapshot,
            RESULTS_DIR / f"overlay_{view_name}.png",
            title=f"Overlay — {view_name}",
            azimuth=az, elevation=el,
            width=800, height=1200 if view_name != "side" else 800,
        )
        logger.info("  overlay_%s.png", view_name)

        # Proportion heatmap
        restore_positions(combined, female_snapshot)
        render_proportion_heatmap(
            combined, combined, bone_scaler,
            RESULTS_DIR / f"heatmap_{view_name}.png",
            title=f"Proportion Heatmap (Female) — {view_name}",
            azimuth=az, elevation=el,
            width=900, height=1200 if view_name != "side" else 800,
        )
        logger.info("  heatmap_%s.png", view_name)

    # ─── Step 6: Skull close-up comparisons ─────────────────────
    if skull_meshes:
        skull_male_snap = {n: male_snapshot[n] for n, _ in skull_meshes if n in male_snapshot}
        skull_female_snap = {n: female_snapshot[n] for n, _ in skull_meshes if n in female_snapshot}

        for view_name, az, el in [("front", 0, 5), ("3q", 30, 10), ("side", 90, 5)]:
            render_side_by_side_from_snapshots(
                skull_meshes, skull_male_snap, skull_female_snap,
                RESULTS_DIR / f"skull_sidebyside_{view_name}.png",
                title=f"Skull — Male vs Female ({view_name})",
                azimuth=az, elevation=el,
                width=1400, height=800,
            )
            logger.info("  skull_sidebyside_%s.png", view_name)

            render_overlay_from_snapshots(
                skull_meshes, skull_male_snap, skull_female_snap,
                RESULTS_DIR / f"skull_overlay_{view_name}.png",
                title=f"Skull Overlay — {view_name}",
                azimuth=az, elevation=el,
                width=800, height=700,
            )
            logger.info("  skull_overlay_%s.png", view_name)

        # Skull heatmap
        restore_positions(skull_meshes, skull_female_snap)
        render_proportion_heatmap(
            skull_meshes, skull_meshes, bone_scaler,
            RESULTS_DIR / "skull_heatmap.png",
            title="Skull — Proportion Heatmap (Female)",
            azimuth=30, elevation=10,
            width=900, height=700,
        )
        logger.info("  skull_heatmap.png")

    # ─── Step 7: Key body group comparisons ──────────────────────
    for group_key, group_label in [("pelvis", "Pelvis"), ("ribs", "Rib Cage")]:
        group_bones = collect_group_meshes(skeleton, group_key)
        if not group_bones:
            continue

        restore_positions(group_bones, male_snapshot)
        male_snap_group = snapshot_positions(group_bones)
        restore_positions(group_bones, female_snapshot)
        female_snap_group = snapshot_positions(group_bones)

        render_overlay_from_snapshots(
            group_bones, male_snap_group, female_snap_group,
            RESULTS_DIR / f"overlay_{group_key}.png",
            title=f"{group_label} — Overlay",
            azimuth=30, elevation=15,
            width=700, height=600,
        )
        logger.info("  overlay_%s.png", group_key)

    # Restore to male for clean state
    restore_positions(combined, male_snapshot)

    logger.info("=== All results saved to %s ===", RESULTS_DIR)
    logger.info("Generated: bone_table, 3 views × 3 types, skull 3 views × 2 types + heatmap, "
                "2 group overlays")


# ── Snapshot-based renderers ────────────────────────────────────────────
# These work with position snapshots so we don't need to mutate meshes.

def _geom_from_snapshot(bone_meshes, snapshot):
    """Build (positions, triangles, bone_ids) using a position snapshot."""
    all_positions = []
    all_triangles = []
    bone_ids = []
    vert_offset = 0

    for bi, (name, mesh) in enumerate(bone_meshes):
        if name in snapshot:
            pos = snapshot[name].reshape(-1, 3)
        else:
            pos = mesh.geometry.positions.reshape(-1, 3)
        all_positions.append(pos)
        bone_ids.extend([bi] * len(pos))
        if mesh.geometry.indices is not None:
            tris = mesh.geometry.indices.reshape(-1, 3) + vert_offset
            all_triangles.append(tris)
        vert_offset += len(pos)

    if not all_positions:
        return None, None, None
    return (np.vstack(all_positions),
            np.vstack(all_triangles) if all_triangles else np.zeros((0, 3), dtype=np.uint32),
            np.array(bone_ids, dtype=np.int32))


def render_side_by_side_from_snapshots(bone_meshes, male_snap, female_snap,
                                       output_path, title="",
                                       azimuth=0, elevation=5,
                                       width=1400, height=700):
    """Side-by-side from position snapshots."""
    half_w = width // 2
    pos_m, tri_m, _ = _geom_from_snapshot(bone_meshes, male_snap)
    pos_f, tri_f, _ = _geom_from_snapshot(bone_meshes, female_snap)

    if pos_m is None or pos_f is None:
        img = Image.new("RGB", (width, height), BG_COLOR)
        img.save(output_path)
        return img

    # Shared scale from combined bounds
    all_pos = np.vstack([pos_m, pos_f])
    _, _, _, shared_scale, shared_center = project_and_fit(
        all_pos, azimuth, elevation, half_w, height, margin=60
    )
    cx, cy = shared_center

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Divider
    draw.line([(half_w, 0), (half_w, height)], fill=(80, 80, 100), width=2)

    # Male panel
    sx_m, sy_m, d_m = orthographic_project(pos_m, azimuth, elevation)
    sx_m = (sx_m - cx) * shared_scale + half_w / 2
    sy_m = (sy_m - cy) * shared_scale + height / 2
    male_render_color = MALE_COLOR * 0.4 + BONE_COLOR * 0.6
    draw_triangles(draw, pos_m, tri_m, sx_m, sy_m, d_m, male_render_color)

    # Female panel
    sx_f, sy_f, d_f = orthographic_project(pos_f, azimuth, elevation)
    sx_f = (sx_f - cx) * shared_scale + half_w + half_w / 2
    sy_f = (sy_f - cy) * shared_scale + height / 2
    female_render_color = FEMALE_COLOR * 0.4 + BONE_COLOR * 0.6
    draw_triangles(draw, pos_f, tri_f, sx_f, sy_f, d_f, female_render_color)

    # Measurement annotations
    m_w = pos_m[:, 0].max() - pos_m[:, 0].min()
    m_h = pos_m[:, 2].max() - pos_m[:, 2].min()
    m_d = pos_m[:, 1].max() - pos_m[:, 1].min()
    f_w = pos_f[:, 0].max() - pos_f[:, 0].min()
    f_h = pos_f[:, 2].max() - pos_f[:, 2].min()
    f_d = pos_f[:, 1].max() - pos_f[:, 1].min()

    y_info = height - 80
    draw.text((15, y_info), f"W: {m_w:.1f}mm", fill=(140, 190, 240))
    draw.text((15, y_info + 16), f"H: {m_h:.1f}mm", fill=(140, 190, 240))
    draw.text((15, y_info + 32), f"D: {m_d:.1f}mm", fill=(140, 190, 240))

    draw.text((half_w + 15, y_info), f"W: {f_w:.1f}mm", fill=(240, 140, 150))
    draw.text((half_w + 15, y_info + 16), f"H: {f_h:.1f}mm", fill=(240, 140, 150))
    draw.text((half_w + 15, y_info + 32), f"D: {f_d:.1f}mm", fill=(240, 140, 150))

    # Change percentages
    if m_w > 0:
        draw.text((half_w + 120, y_info),
                  f"({((f_w/m_w)-1)*100:+.1f}%)", fill=(200, 200, 200))
        draw.text((half_w + 120, y_info + 16),
                  f"({((f_h/m_h)-1)*100:+.1f}%)", fill=(200, 200, 200))
        draw.text((half_w + 120, y_info + 32),
                  f"({((f_d/m_d)-1)*100:+.1f}%)", fill=(200, 200, 200))

    # Scale bar (centered at bottom)
    bar_px = 50.0 * shared_scale
    if bar_px > 20:
        draw_scale_bar(draw, width // 2 - int(bar_px) // 2, height - 15,
                       int(bar_px), 50.0)

    # Labels
    draw.text((half_w // 2 - 30, 30), "MALE (g=0.0)", fill=(140, 190, 240))
    draw.text((half_w + half_w // 2 - 40, 30), "FEMALE (g=1.0)", fill=(240, 140, 150))
    if title:
        draw.text((width // 2 - len(title) * 3, 8), title, fill=TEXT_COLOR)

    img.save(output_path)
    return img


def render_overlay_from_snapshots(bone_meshes, male_snap, female_snap,
                                  output_path, title="",
                                  azimuth=0, elevation=5,
                                  width=800, height=700):
    """Overlay from position snapshots: male solid blue, female wireframe red."""
    pos_m, tri_m, _ = _geom_from_snapshot(bone_meshes, male_snap)
    pos_f, tri_f, _ = _geom_from_snapshot(bone_meshes, female_snap)

    if pos_m is None or pos_f is None:
        img = Image.new("RGB", (width, height), BG_COLOR)
        img.save(output_path)
        return img

    all_pos = np.vstack([pos_m, pos_f])
    _, _, _, shared_scale, shared_center = project_and_fit(
        all_pos, azimuth, elevation, width, height, margin=60
    )
    cx, cy = shared_center

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Male: faded solid
    sx_m, sy_m, d_m = orthographic_project(pos_m, azimuth, elevation)
    sx_m = (sx_m - cx) * shared_scale + width / 2
    sy_m = (sy_m - cy) * shared_scale + height / 2
    draw_triangles(draw, pos_m, tri_m, sx_m, sy_m, d_m, MALE_COLOR, alpha=0.45)

    # Female: wireframe
    sx_f, sy_f, d_f = orthographic_project(pos_f, azimuth, elevation)
    sx_f = (sx_f - cx) * shared_scale + width / 2
    sy_f = (sy_f - cy) * shared_scale + height / 2
    draw_wireframe(draw, tri_f, sx_f, sy_f, d_f, color=(240, 100, 120), alpha=0.9)

    # Stats
    m_w = pos_m[:, 0].max() - pos_m[:, 0].min()
    m_h = pos_m[:, 2].max() - pos_m[:, 2].min()
    f_w = pos_f[:, 0].max() - pos_f[:, 0].min()
    f_h = pos_f[:, 2].max() - pos_f[:, 2].min()

    y_info = height - 70
    draw.text((10, y_info), f"Male:   W={m_w:.1f}  H={m_h:.1f}mm", fill=(140, 190, 240))
    draw.text((10, y_info + 16), f"Female: W={f_w:.1f}  H={f_h:.1f}mm", fill=(240, 140, 150))
    if m_w > 0:
        draw.text((10, y_info + 32),
                  f"Change: W={((f_w/m_w)-1)*100:+.1f}%  H={((f_h/m_h)-1)*100:+.1f}%",
                  fill=(200, 200, 200))

    bar_px = 50.0 * shared_scale
    if bar_px > 20:
        draw_scale_bar(draw, 20, height - 15, int(bar_px), 50.0)

    if title:
        draw.text((10, 10), title, fill=TEXT_COLOR)

    draw_legend(draw, width - 200, 10, [
        ((100, 160, 220), "Male (solid)"),
        ((240, 100, 120), "Female (wireframe)"),
    ])

    img.save(output_path)
    return img


if __name__ == "__main__":
    main()
