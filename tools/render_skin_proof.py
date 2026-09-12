"""Render the body skin through the real renderer, before and after a change.

Every other skin measurement in this project is arithmetic on vertex
positions.  This one draws pixels: it puts the app's own scene in front of
``faceforge.session.Session`` -- the same GL renderer, framebuffer and
blank-frame guard the headless CLI uses -- and saves the frames.

    PYTHONPATH=src python -m tools.render_skin_proof --tag after
    PYTHONPATH=src python -m tools.render_skin_proof --tag before \\
        --min-spatial 12 --seed-margin 1.0 --bridge 0 --muscle-weight 0

The engine overrides exist so an earlier state can be rendered from the same
working tree.  Frames go to ``results/skin_render/<tag>_<view>.png``; the
session refuses a uniform frame, so a blank render fails rather than being
saved.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

OUT_DIR = Path("results/skin_render")

#: Azimuth in degrees about the body's long axis, and a name.
VIEWS = (("front", 0.0), ("three-quarter", 40.0), ("side", 90.0), ("back", 180.0))

POSES = {
    "arms to shoulder height": {"shoulder_r_abduct": 1.0, "shoulder_l_abduct": 1.0},
    "hip hinge": {"hip_r_flex": 0.85, "hip_l_flex": 0.85,
                  "knee_r_flex": 0.35, "knee_l_flex": 0.35},
    "neutral": {},
}

SETTLE_FRAMES = 90


def _apply_pose(ctx, pose: dict) -> None:
    from faceforge.core.state import BodyState

    defaults = BodyState()
    for state in (ctx.state.body, ctx.state.target_body):
        for name, value in vars(defaults).items():
            if hasattr(state, name):
                setattr(state, name, value)
        for name, value in pose.items():
            setattr(state, name, value)
    for _ in range(SETTLE_FRAMES):
        ctx.simulation.step(1 / 60)


def _skin_only(ctx):
    """Hide everything but the skin, and return the skin's world bounds."""
    scene = ctx.scene
    root = getattr(scene, "root", scene)
    skin_group = root.find("bodyMeshGroup")
    pts = []
    for mesh, matrix in scene.collect_meshes():
        if mesh.name != "Skin":
            mesh.visible = False
            continue
        g = mesh.geometry
        v = np.asarray(g.positions, dtype=np.float64).reshape(-1, 3)[:g.vertex_count]
        m = np.asarray(matrix, dtype=np.float64)
        pts.append(v[::37] @ m[:3, :3].T + m[:3, 3])
    if not pts:
        return None
    p = np.vstack(pts)
    return p.min(axis=0), p.max(axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="now")
    ap.add_argument("--pose", default="arms to shoulder height", choices=sorted(POSES))
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--min-spatial", type=float, default=None)
    ap.add_argument("--seed-margin", type=float, default=None)
    ap.add_argument("--bridge", type=float, default=None)
    ap.add_argument("--muscle-weight", type=float, default=None)
    ap.add_argument("--muscle-bias", type=float, default=None)
    ap.add_argument("--flesh-radius", type=float, default=None)
    ap.add_argument("--inward-seeds", action="store_true")
    ap.add_argument("--hybrid-limit", action="store_true")
    ap.add_argument("--hybrid-scale", type=float, default=None)
    ap.add_argument("--cutoff-ratio", type=float, default=None)
    ap.add_argument("--own-flesh", action="store_true")
    ap.add_argument("--cross-cost", type=float, default=None)
    ap.add_argument("--cut-below", type=float, default=None)
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.body.soft_tissue import SoftTissueSkinning
    from faceforge.controllers import build_controllers
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence
    from faceforge.session import Session

    if args.seed_margin is not None:
        SoftTissueSkinning.SEED_CONFIDENCE_MARGIN = args.seed_margin
    if args.bridge is not None:
        SoftTissueSkinning.GEODESIC_BRIDGE = args.bridge
    if args.muscle_weight is not None:
        SoftTissueSkinning.MUSCLE_FIELD_WEIGHT = args.muscle_weight
    if args.muscle_bias is not None:
        SoftTissueSkinning.MUSCLE_WEIGHT_BIAS = args.muscle_bias
    if args.flesh_radius is not None:
        SoftTissueSkinning.MUSCLE_ELIGIBILITY_RADIUS = args.flesh_radius
    if args.inward_seeds:
        SoftTissueSkinning.SEED_INWARD_ONLY = True
    if args.hybrid_limit:
        SoftTissueSkinning.SPATIAL_LIMIT_ON_HYBRID = True
    if args.hybrid_scale is not None:
        SoftTissueSkinning.HYBRID_LIMIT_SCALE = args.hybrid_scale
    if args.cutoff_ratio is not None:
        SoftTissueSkinning.INFLUENCE_CUTOFF_RATIO = args.cutoff_ratio
    if args.own_flesh:
        SoftTissueSkinning.SEED_ON_OWN_FLESH = True
    if args.cross_cost is not None:
        SoftTissueSkinning.CROSS_PART_EDGE_COST = args.cross_cost
    if args.cut_below is not None:
        SoftTissueSkinning.CROSS_PART_CUT_BELOW = args.cut_below

    ctx = build_app_context(argv=[])
    controllers = build_controllers(ctx)
    AssetLoadSequence(ctx).run()
    if args.min_spatial is not None:
        ctx.simulation.soft_tissue.min_spatial = args.min_spatial
    controllers.loaders.load_skin()
    _apply_pose(ctx, POSES[args.pose])
    bounds = _skin_only(ctx)
    if bounds is None:
        print("no skin mesh in the scene -- nothing to render")
        return 1
    lo, hi = bounds
    centre = 0.5 * (lo + hi)
    radius = float(np.linalg.norm(hi - lo)) * 0.5

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = Session.create(width=args.width, height=args.height)
    try:
        session._scene = ctx.scene          # the app's scene, the app's skinning
        for name, azimuth in VIEWS:
            a = math.radians(azimuth)
            eye = centre + np.array([math.sin(a), -math.cos(a), 0.18]) * radius * 2.6
            session.camera.look_at(
                np.asarray(eye, dtype=np.float64),
                np.asarray(centre, dtype=np.float64))
            path = OUT_DIR / f"{args.tag}_{name}.png"
            session.save_png(path)
            print(f"  {path}  content {session.last_content_fraction * 100:.2f}% "
                  f"of pixels")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
