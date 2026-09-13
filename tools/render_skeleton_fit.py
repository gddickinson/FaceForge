"""Draw the skeleton inside the body-surface mesh, with and without the fit.

Arithmetic says how far a bone vertex is outside the surface
(``tools/skeleton_containment.py``); this says what that looks like.  The
surface is drawn as a wireframe shell so the bones inside it are visible, and
the frames come from the same GL renderer and blank-frame guard the headless
CLI uses.

    PYTHONPATH=src python -m tools.render_skeleton_fit --tag before
    PYTHONPATH=src python -m tools.render_skeleton_fit --tag after --fit

Frames go to ``results/skeleton_fit/<tag>_<view>.png``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

OUT_DIR = Path("results/skeleton_fit")

#: name -> (azimuth about the long axis, elevation, zoom, centre bias in Z)
VIEWS = {
    "front": (0.0, 0.10, 2.4, 0.0),
    "side": (90.0, 0.10, 2.4, 0.0),
    "three-quarter": (40.0, 0.10, 2.4, 0.0),
    "head": (25.0, 0.05, 0.55, 0.44),
    "shoulders": (15.0, 0.05, 0.85, 0.30),
    "feet": (20.0, 0.05, 0.60, -0.45),
}

#: Groups that are neither skeleton nor the body surface: hidden, so the
#: picture answers one question.
HIDE = ("faceGroup", "faceFeatureGroup", "stlMuscleGroup", "exprMuscleGroup",
        "platysmaGroup", "neckMuscleGroup", "brainGroup", "fasciaGroup")


def _show_only_skeleton_and_surface(ctx):
    """Skeleton solid, surface a wireframe shell.  Returns the surface bounds."""
    from faceforge.core.material import RenderMode

    scene = ctx.scene
    root = getattr(scene, "root", scene)
    for name in HIDE:
        node = root.find(name)
        if node is not None:
            node.visible = False
    surface_group = root.find("bodyMeshGroup")
    if surface_group is None:
        return None
    node = surface_group
    while node is not None:
        node.visible = True
        node = node.parent

    pts = []
    for mesh, matrix in scene.collect_meshes():
        if mesh.name != "body_surface":
            continue
        mesh.material.render_mode = RenderMode.WIREFRAME
        mesh.material.opacity = 1.0
        mesh.material.color = (0.35, 0.75, 0.95)
        mesh.material.wireframe_color = (0.35, 0.75, 0.95)
        g = mesh.geometry
        v = np.asarray(g.positions, dtype=np.float64).reshape(-1, 3)[:g.vertex_count]
        m = np.asarray(matrix, dtype=np.float64)
        pts.append(v[::7] @ m[:3, :3].T + m[:3, 3])
    if not pts:
        return None
    p = np.vstack(pts)
    return p.min(axis=0), p.max(axis=0)


def _colour_by_protrusion(ctx, limit: float = 8.0) -> tuple[float, float]:
    """Paint every bone vertex by how far outside the surface it is.

    Blue inside, through white at the surface, to red at ``limit`` units out.
    The surface itself is then hidden, so the picture is the skeleton alone
    and the eye is not asked to judge depth through a cage of wireframe.
    """
    from faceforge.body.fit_regions import SKIP_SUBTREES
    from tools.skeleton_containment import SurfaceDepth, surface_of

    morph = ctx.pipeline.gender_morph
    pos, tris = surface_of(morph)
    depth = SurfaceDepth(pos, tris, np.array([0.0, 0.0, -100.0]))
    scene = ctx.scene
    root = getattr(scene, "root", scene)
    surface_group = root.find("bodyMeshGroup")
    if surface_group is not None:
        surface_group.visible = False

    worst, painted = 0.0, []
    stack = [(root, np.zeros(3))]
    while stack:
        node, offset = stack.pop()
        for child in node.children:
            name = getattr(child, "name", "") or ""
            if name in SKIP_SUBTREES:
                continue
            here = offset + np.asarray(child.position, dtype=np.float64)
            mesh = getattr(child, "mesh", None)
            if mesh is not None and name:
                g = mesh.geometry
                v = np.asarray(g.positions, dtype=np.float64).reshape(-1, 3)
                v = v[:g.vertex_count]
                d = depth(v + here)
                t = np.clip(d / limit, -1.0, 1.0)[:, None]
                cold = np.array([[0.25, 0.45, 0.95]])
                warm = np.array([[0.95, 0.12, 0.10]])
                white = np.array([[0.96, 0.96, 0.92]])
                cols = np.where(t >= 0, white + t * (warm - white),
                                white + (-t) * (cold - white))
                g.vertex_colors = cols.astype(np.float32)
                g.colors_dirty = True
                mesh.material.vertex_colors_active = True
                mesh.needs_update = True
                worst = max(worst, float(d.max()))
                painted.append(d)
            stack.append((child, here))
    alld = np.concatenate(painted) if painted else np.zeros(1)
    return float(100.0 * (alld > 0).mean()), worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="now")
    ap.add_argument("--fit", action="store_true", help="apply the skeleton fit")
    ap.add_argument("--gender", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--protrusion", action="store_true",
                    help="colour the bones by how far they are outside")
    ap.add_argument("--prefer", default="hardware",
                    choices=("auto", "hardware", "software"))
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.controllers import build_controllers
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence
    from faceforge.core.events import EventType
    from faceforge.session import Session

    ctx = build_app_context(argv=[])
    build_controllers(ctx)
    AssetLoadSequence(ctx).run()
    if args.gender:
        ctx.event_bus.publish(EventType.GENDER_RELEASED, gender=args.gender)
    if args.fit:
        ctx.event_bus.publish(EventType.SKELETON_FIT_TOGGLED, enabled=True)
    ctx.scene.update()

    bounds = _show_only_skeleton_and_surface(ctx)
    if bounds is None:
        print("no body-surface mesh in the scene -- nothing to render")
        return 1
    if args.protrusion:
        pct, worst = _colour_by_protrusion(ctx)
        print(f"  {pct:.1f}% of bone vertices outside, worst {worst:.2f} units")
    lo, hi = bounds
    centre = 0.5 * (lo + hi)
    span = hi - lo
    radius = float(np.linalg.norm(span)) * 0.5

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = Session.create(width=args.width, height=args.height,
                             prefer=args.prefer)
    try:
        session._scene = ctx.scene
        for name in args.views.split(","):
            azimuth, elevation, zoom, bias = VIEWS[name.strip()]
            a = math.radians(azimuth)
            target = centre + np.array([0.0, 0.0, bias * span[2]])
            eye = target + np.array(
                [math.sin(a), -math.cos(a), elevation]) * radius * zoom
            session.camera.look_at(np.asarray(eye, dtype=np.float64),
                                   np.asarray(target, dtype=np.float64))
            path = OUT_DIR / f"{args.tag}_{name.strip()}.png"
            session.save_png(path)
            print(f"  {path}  content "
                  f"{session.last_content_fraction * 100:.2f}% of pixels")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
