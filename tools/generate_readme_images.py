"""Generate showcase images for the README.

Renders anatomical views, facial expressions, body poses, and layered
anatomy using the headless scene loader and PIL-based renderers.

Camera convention (orthographic_project in mesh_renderer.py):
  azimuth=0   → back view (spine visible)
  azimuth=180 → front view (face visible)
  azimuth=90  → left side
  azimuth=270 → right side
  azimuth=210 → 3/4 front-right

Usage::

    python -m tools.generate_readme_images
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from PIL import Image, ImageDraw

from tools.headless_loader import (
    load_headless_scene,
    load_layer,
    register_layer,
    apply_pose,
    apply_head_rotation,
    HeadlessScene,
)
from tools.head_renderer import (
    MeshGroup,
    render_head_multimesh,
)
from tools.mesh_renderer import orthographic_project, _apply_lighting
from faceforge.core.state import BodyState, FaceState

OUTPUT_DIR = Path("docs/images")

# Background color used across all images
BG = (24, 26, 32)

# Camera presets (corrected for this coordinate system)
FRONT = 180
FRONT_3Q = 210  # 3/4 front-right
RIGHT_SIDE = 270
BACK = 0
BACK_3Q = 330

# ── Color palette ────────────────────────────────────────────────────

COLORS = {
    "skull":              (225, 215, 195),
    "face":               (238, 200, 168),
    "jaw_muscles":        (195, 72, 72),
    "expression_muscles": (195, 115, 72),
    "face_features":      (115, 175, 215),
    "neck_muscles":       (195, 72, 72),
    "vertebrae":          (225, 200, 165),
    "skeleton":           (235, 225, 205),
    "body_muscles":       (185, 78, 68),
    "organs":             (175, 105, 115),
    "skin":               (222, 194, 164),
}


# ── Scene mesh extraction ────────────────────────────────────────────

def _collect_node_meshes(
    node, all_pos: list, all_tris: list, offset: list[int],
) -> None:
    """Traverse a scene node collecting mesh data with world transforms."""
    def _collect(n):
        if n.mesh is not None:
            pos = n.mesh.geometry.positions
            idx = n.mesh.geometry.indices
            if pos is not None and len(pos) > 0 and idx is not None and len(idx) > 0:
                pos3 = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
                wm = n.world_matrix
                transformed = (wm[:3, :3] @ pos3.T).T + wm[:3, 3]
                all_pos.append(transformed)
                all_tris.append(
                    np.asarray(idx, dtype=np.int32).reshape(-1, 3) + offset[0]
                )
                offset[0] += len(pos3)
    node.traverse(_collect)


def _meshgroup_from_nodes(
    hs: HeadlessScene,
    named_nodes: list[str],
    skel_groups: list[str] | None,
    name: str,
    color: tuple[int, int, int],
) -> MeshGroup | None:
    """Build a MeshGroup by collecting from named nodes + skeleton groups."""
    pos, tris = [], []
    off = [0]
    for gname in named_nodes:
        node = hs.named_nodes.get(gname)
        if node:
            _collect_node_meshes(node, pos, tris, off)
    if skel_groups and hs.skeleton:
        for gname in skel_groups:
            node = hs.skeleton.groups.get(gname)
            if node:
                _collect_node_meshes(node, pos, tris, off)
    if not pos:
        return None
    return MeshGroup(name, np.concatenate(pos), np.concatenate(tris), color)


def _load_batch_as_meshgroup(
    hs: HeadlessScene, config: str, name: str, color: tuple[int, int, int],
) -> MeshGroup | None:
    """Load an STL batch directly and return as MeshGroup."""
    try:
        result = hs.assets.load_skeleton_batch(config, label=name)
    except Exception:
        return None
    pos, tris = [], []
    off = 0
    for m in result.meshes:
        p = m.geometry.positions
        i = m.geometry.indices
        if p is not None and len(p) > 0 and i is not None:
            p3 = np.asarray(p, dtype=np.float64).reshape(-1, 3)
            pos.append(p3)
            tris.append(np.asarray(i, dtype=np.int32).reshape(-1, 3) + off)
            off += len(p3)
    if not pos:
        return None
    return MeshGroup(name, np.concatenate(pos), np.concatenate(tris), color)


def _load_layer_as_meshgroup(
    hs: HeadlessScene, layer_name: str, name: str, color: tuple[int, int, int],
) -> MeshGroup | None:
    """Load a tissue layer and return as MeshGroup (no skinning registration)."""
    try:
        meshes = load_layer(hs, layer_name)
    except Exception as e:
        print(f"    Could not load {layer_name}: {e}")
        return None
    pos, tris = [], []
    off = 0
    for m in meshes:
        p = m.geometry.positions
        i = m.geometry.indices
        if p is not None and len(p) > 0 and i is not None:
            p3 = np.asarray(p, dtype=np.float64).reshape(-1, 3)
            pos.append(p3)
            tris.append(np.asarray(i, dtype=np.int32).reshape(-1, 3) + off)
            off += len(p3)
    if not pos:
        return None
    print(f"    {layer_name}: {len(meshes)} meshes")
    return MeshGroup(name, np.concatenate(pos), np.concatenate(tris), color)


def _extract_head_groups(hs: HeadlessScene) -> list[MeshGroup]:
    """Extract head mesh groups with per-group coloring."""
    groups = []
    specs = [
        (["skullGroup"], None, "skull"),
        (["faceGroup"], None, "face"),
        (["stlMuscleGroup"], None, "jaw_muscles"),
        (["exprMuscleGroup"], None, "expression_muscles"),
        (["faceFeatureGroup"], None, "face_features"),
        (["neckMuscleGroup"], None, "neck_muscles"),
        (["vertebraeGroup"], None, "vertebrae"),
    ]
    for nodes, skel, name in specs:
        mg = _meshgroup_from_nodes(hs, nodes, skel, name, COLORS.get(name, (180, 180, 180)))
        if mg:
            groups.append(mg)
    return groups


# ── Layout helpers ───────────────────────────────────────────────────

def _stitch_horizontal(images: list[Image.Image], gap: int = 6) -> Image.Image:
    """Stitch images horizontally."""
    if not images:
        return Image.new("RGB", (1, 1))
    h = max(img.height for img in images)
    w = sum(img.width for img in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), BG)
    x = 0
    for img in images:
        out.paste(img, (x, (h - img.height) // 2))
        x += img.width + gap
    return out


def _stitch_grid(
    images: list[Image.Image],
    cols: int,
    gap: int = 6,
    labels: list[str] | None = None,
) -> Image.Image:
    """Stitch images in a grid with optional labels."""
    if not images:
        return Image.new("RGB", (1, 1))
    cw = max(img.width for img in images)
    ch = max(img.height for img in images)
    label_h = 30 if labels else 0
    rows = (len(images) + cols - 1) // cols
    w = cols * cw + (cols - 1) * gap
    h = rows * (ch + label_h) + (rows - 1) * gap
    out = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(out)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = c * (cw + gap) + (cw - img.width) // 2
        y = r * (ch + label_h + gap)
        if labels and i < len(labels):
            tw = draw.textlength(labels[i])
            draw.text((c * (cw + gap) + (cw - tw) // 2, y + 4),
                       labels[i], fill=(200, 200, 210))
            y += label_h
        out.paste(img, (x, y))
    return out


# ── Clean body renderer (no diagnostic overlays) ────────────────────

def _render_body_clean(
    positions: np.ndarray,
    triangles: np.ndarray,
    azimuth: float = FRONT_3Q,
    elevation: float = 5,
    width: int = 400,
    height: int = 560,
    base_color: tuple[int, int, int] = COLORS["skin"],
    margin: int = 30,
) -> Image.Image:
    """Render body mesh with uniform skin-tone color and lighting."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)

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
    cross_z = (px[i1] - px[i0]) * (py[i2] - py[i0]) - (py[i1] - py[i0]) * (px[i2] - px[i0])
    front = cross_z > 0
    if np.sum(front) < np.sum(~front):
        front = ~front

    vis_idx = np.where(front)[0]
    if len(vis_idx) == 0:
        return Image.new("RGB", (width, height), BG)

    vis = triangles[vis_idx]
    n = len(vis)
    colors = np.tile(np.array(base_color, dtype=np.uint8), (n, 1))

    az_rad = np.radians(azimuth)
    light = np.array([
        0.3 * np.cos(az_rad) + 0.5 * np.sin(az_rad),
        -0.3 * np.sin(az_rad) + 0.5 * np.cos(az_rad),
        0.4,
    ])
    colors = _apply_lighting(colors, positions, vis, light)

    tri_d = (depth[vis[:, 0]] + depth[vis[:, 1]] + depth[vis[:, 2]]) / 3
    order = np.argsort(tri_d)
    vis = vis[order]
    colors = colors[order]

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    for i in range(n):
        v0, v1, v2 = vis[i]
        pts = [(int(px[v0]+.5), int(py[v0]+.5)),
               (int(px[v1]+.5), int(py[v1]+.5)),
               (int(px[v2]+.5), int(py[v2]+.5))]
        draw.polygon(pts, fill=(int(colors[i,0]), int(colors[i,1]), int(colors[i,2])))
    return img


# ── Image generators ────────────────────────────────────────────────

def generate_skull_views(hs: HeadlessScene) -> None:
    """High-res skull from 4 angles."""
    print("  [1/7] Skull views...")
    mg = _meshgroup_from_nodes(hs, ["skullGroup"], None, "skull", COLORS["skull"])
    if not mg:
        print("    No skull data")
        return

    views = [
        (FRONT, 5),      # front
        (FRONT_3Q, 15),   # 3/4 front
        (RIGHT_SIDE, 5),  # side
        (FRONT, 75),      # top-down
    ]
    imgs = [render_head_multimesh(
        [mg], azimuth=az, elevation=el, width=400, height=400, bg_color=BG,
    ) for az, el in views]

    _stitch_horizontal(imgs).save(str(OUTPUT_DIR / "skull_views.png"))
    print("    OK")


def generate_head_anatomy(hs: HeadlessScene) -> None:
    """Head with all tissue layers from 3 angles."""
    print("  [2/7] Head anatomy...")
    groups = _extract_head_groups(hs)

    views = [
        (FRONT, 5),
        (FRONT_3Q, 12),
        (RIGHT_SIDE, 5),
    ]
    imgs = [render_head_multimesh(
        groups, azimuth=az, elevation=el, width=500, height=500, bg_color=BG,
    ) for az, el in views]

    _stitch_horizontal(imgs).save(str(OUTPUT_DIR / "head_anatomy.png"))
    print("    OK")


def generate_head_rotation(hs: HeadlessScene) -> None:
    """Head rotation showcase: neutral, yaw, pitch, combined."""
    print("  [3/7] Head rotation...")
    poses = [
        ("Neutral",  {"head_yaw": 0.0, "head_pitch": 0.0,  "head_roll": 0.0}),
        ("Yaw Left", {"head_yaw": 0.7, "head_pitch": 0.0,  "head_roll": 0.0}),
        ("Pitch Up", {"head_yaw": 0.0, "head_pitch": 0.6,  "head_roll": 0.0}),
        ("Combined", {"head_yaw": 0.4, "head_pitch": 0.3,  "head_roll": 0.2}),
    ]

    imgs, labels = [], []
    for label, vals in poses:
        fs = FaceState()
        for k, v in vals.items():
            setattr(fs, k, v)
        apply_head_rotation(hs, fs)
        groups = _extract_head_groups(hs)
        imgs.append(render_head_multimesh(
            groups, azimuth=FRONT_3Q, elevation=10,
            width=400, height=400, bg_color=BG,
        ))
        labels.append(label)

    apply_head_rotation(hs, FaceState())  # reset
    _stitch_grid(imgs, cols=4, labels=labels).save(
        str(OUTPUT_DIR / "head_rotation.png"))
    print("    OK")


def generate_body_skeleton(hs: HeadlessScene) -> None:
    """Full-body skeleton from 3 angles."""
    print("  [4/7] Body skeleton...")

    # Collect all skeleton mesh groups into lists of (positions, triangles).
    # NOTE: The scene graph versions of some groups (upper_limb, lower_limb,
    # hand, foot) are incomplete -- they drop bones like the humerus, radius,
    # ulna, femur, etc. So we direct-load ALL skeleton batches from STL files
    # to get the complete skeleton.
    parts_pos: list[np.ndarray] = []
    parts_tris: list[np.ndarray] = []
    off = 0

    # Head groups from scene graph (these are complete)
    skel_head = _meshgroup_from_nodes(
        hs, ["skullGroup", "vertebraeGroup"], None, "skel", COLORS["skeleton"],
    )
    if skel_head:
        parts_pos.append(skel_head.positions)
        parts_tris.append(skel_head.triangles + off)
        off += len(skel_head.positions)

    # Spine/ribs/pelvis from scene graph (complete)
    for gname in ["thoracic", "lumbar", "ribs", "pelvis"]:
        mg = _meshgroup_from_nodes(hs, [], [gname], gname, COLORS["skeleton"])
        if mg and len(mg.positions) > 0:
            parts_pos.append(mg.positions)
            parts_tris.append(mg.triangles + off)
            off += len(mg.positions)

    # Directly load ALL limb parts from STL files (scene graph drops bones)
    for cfg in ["upper_limb.json", "lower_limb.json", "hand.json", "foot.json"]:
        mg = _load_batch_as_meshgroup(hs, cfg, cfg.split(".")[0], COLORS["skeleton"])
        if mg:
            parts_pos.append(mg.positions)
            parts_tris.append(mg.triangles + off)
            off += len(mg.positions)

    if not parts_pos:
        print("    No skeleton data")
        return

    combined = MeshGroup(
        "skeleton",
        np.concatenate(parts_pos),
        np.concatenate(parts_tris),
        COLORS["skeleton"],
    )

    views = [
        (FRONT, 5),
        (FRONT_3Q, 10),
        (RIGHT_SIDE, 5),
    ]
    imgs = [render_head_multimesh(
        [combined], azimuth=az, elevation=el, width=500, height=750, bg_color=BG,
    ) for az, el in views]

    _stitch_horizontal(imgs).save(str(OUTPUT_DIR / "body_skeleton.png"))
    print("    OK")


def generate_body_layers(hs: HeadlessScene) -> None:
    """Layered anatomy: skeleton → + muscles → + organs (waist-up to avoid genitals)."""
    print("  [5/7] Body layers...")

    # Skeleton (head + torso + arms, no lower body for waist-up crop)
    skel_pos: list[np.ndarray] = []
    skel_tris: list[np.ndarray] = []
    off = 0
    head_skel = _meshgroup_from_nodes(
        hs, ["skullGroup", "vertebraeGroup"], None, "skeleton", COLORS["skeleton"],
    )
    if head_skel:
        skel_pos.append(head_skel.positions)
        skel_tris.append(head_skel.triangles + off)
        off += len(head_skel.positions)
    for gname in ["thoracic", "lumbar", "ribs", "pelvis"]:
        mg = _meshgroup_from_nodes(hs, [], [gname], gname, COLORS["skeleton"])
        if mg and len(mg.positions) > 0:
            skel_pos.append(mg.positions)
            skel_tris.append(mg.triangles + off)
            off += len(mg.positions)
    # Direct-load upper_limb to get ALL arm bones (scene graph is incomplete)
    ul_mg = _load_batch_as_meshgroup(hs, "upper_limb.json", "upper_limb", COLORS["skeleton"])
    if ul_mg:
        skel_pos.append(ul_mg.positions)
        skel_tris.append(ul_mg.triangles + off)
        off += len(ul_mg.positions)
    skel_mg = MeshGroup("skeleton", np.concatenate(skel_pos), np.concatenate(skel_tris),
                         COLORS["skeleton"]) if skel_pos else None

    # Head muscles
    head_muscles = _meshgroup_from_nodes(
        hs, ["stlMuscleGroup", "exprMuscleGroup", "neckMuscleGroup"],
        None, "head muscles", COLORS["jaw_muscles"],
    )

    # Body muscles
    body_muscle_layers = []
    for ln in ["back_muscles", "shoulder_muscles", "arm_muscles", "torso_muscles"]:
        mg = _load_layer_as_meshgroup(hs, ln, ln, COLORS["body_muscles"])
        if mg:
            body_muscle_layers.append(mg)

    all_muscles_mg = None
    if body_muscle_layers:
        mp, mt = [], []
        off = 0
        for mg in body_muscle_layers:
            mp.append(mg.positions)
            mt.append(mg.triangles + off)
            off += len(mg.positions)
        all_muscles_mg = MeshGroup("body muscles", np.concatenate(mp),
                                    np.concatenate(mt), COLORS["body_muscles"])

    # Organs
    organs_mg = _load_layer_as_meshgroup(hs, "organs", "organs", COLORS["organs"])

    # Z-range: crop below hip to exclude genitals (Z: head ~0 to feet ~-200)
    z_crop = (-100.0, 30.0)
    az, el = FRONT_3Q, 8
    w, h = 480, 620

    imgs, labels = [], []

    # Panel 1: Skeleton
    if skel_mg:
        imgs.append(render_head_multimesh(
            [skel_mg], azimuth=az, elevation=el, width=w, height=h,
            bg_color=BG, z_range=z_crop))
        labels.append("Skeleton")

    # Panel 2: + muscles
    grps = [g for g in [skel_mg, head_muscles, all_muscles_mg] if g]
    if grps:
        imgs.append(render_head_multimesh(
            grps, azimuth=az, elevation=el, width=w, height=h,
            bg_color=BG, z_range=z_crop))
        labels.append("+ Muscles")

    # Panel 3: + organs
    grps = [g for g in [skel_mg, organs_mg] if g]
    if grps:
        imgs.append(render_head_multimesh(
            grps, azimuth=az, elevation=el, width=w, height=h,
            bg_color=BG, z_range=z_crop))
        labels.append("+ Organs")

    # Panel 4: skeleton + muscles + organs
    grps = [g for g in [skel_mg, head_muscles, all_muscles_mg, organs_mg] if g]
    if grps:
        imgs.append(render_head_multimesh(
            grps, azimuth=az, elevation=el, width=w, height=h,
            bg_color=BG, z_range=z_crop))
        labels.append("Combined")

    if imgs:
        _stitch_grid(imgs, cols=len(imgs), labels=labels).save(
            str(OUTPUT_DIR / "body_layers.png"))
    print("    OK")


def generate_body_poses(hs: HeadlessScene) -> None:
    """Body poses with skin mesh."""
    print("  [6/7] Body poses...")

    try:
        meshes = load_layer(hs, "skin")
        register_layer(hs, meshes, "skin")
    except Exception as e:
        print(f"    Skin load failed: {e}")
        return

    binding = hs.skinning.bindings[0]
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    triangles = mesh.geometry.indices.reshape(-1, 3)

    poses_file = Path("assets/config/body_poses.json")
    all_poses = json.loads(poses_file.read_text()) if poses_file.exists() else {}

    pose_list = [
        ("Anatomical", {}),
        ("Relaxed", all_poses.get("relaxed", {})),
        ("Walking", all_poses.get("walking", {})),
        ("Sitting", all_poses.get("sitting", {})),
    ]

    imgs, labels = [], []
    for label, vals in pose_list:
        state = BodyState()
        state.set_from_js_dict(vals)
        apply_pose(hs, state)
        positions = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
        # Use a slightly more rotated angle to be discrete about lower body
        imgs.append(_render_body_clean(
            positions, triangles,
            azimuth=FRONT_3Q + 10, elevation=10,
            width=450, height=650,
        ))
        labels.append(label)

    apply_pose(hs, BodyState())  # reset
    _stitch_grid(imgs, cols=4, labels=labels).save(
        str(OUTPUT_DIR / "body_poses.png"))
    print("    OK")


def generate_anatomy_layers_head(hs: HeadlessScene) -> None:
    """Side-by-side head: skull only vs. skull + muscles."""
    print("  [7/7] Head anatomy layers...")

    skull = _meshgroup_from_nodes(hs, ["skullGroup"], None, "skull", COLORS["skull"])
    muscles = _meshgroup_from_nodes(
        hs, ["stlMuscleGroup", "exprMuscleGroup", "neckMuscleGroup"],
        None, "muscles", COLORS["jaw_muscles"],
    )

    az, el = FRONT_3Q, 10
    w, h = 500, 500

    img1 = render_head_multimesh(
        [g for g in [skull] if g], azimuth=az, elevation=el,
        width=w, height=h, bg_color=BG)
    img2 = render_head_multimesh(
        [g for g in [skull, muscles] if g], azimuth=az, elevation=el,
        width=w, height=h, bg_color=BG)

    _stitch_grid([img1, img2], cols=2,
                 labels=["Skull", "Skull + Muscles"]).save(
        str(OUTPUT_DIR / "anatomy_layers.png"))
    print("    OK")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading headless scene...")
    t0 = time.time()
    hs = load_headless_scene()
    print(f"  Scene loaded in {time.time() - t0:.1f}s\n")

    generate_skull_views(hs)
    generate_head_anatomy(hs)
    generate_head_rotation(hs)
    generate_body_skeleton(hs)
    generate_body_layers(hs)
    generate_body_poses(hs)
    generate_anatomy_layers_head(hs)

    print(f"\nAll images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
