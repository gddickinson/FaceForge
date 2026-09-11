"""Measure how the body SKIN deforms: tearing, collapse and containment.

``tools/deformation_quality.py`` measures muscle layers.  The skin is a
different problem: one 791,729-vertex sheet bound to every chain, where the
failure is a torn edge rather than a muscle off its bone.  Reported as
"individual pixels either left behind or attached to movements of the
different body parts", and visible in `results/skin_tearing.png` as straight
horizontal seams at the hips, elbows and knees plus speckle over the hands
and feet.

    PYTHONPATH=src python -m tools.skin_deformation_quality
    PYTHONPATH=src python -m tools.skin_deformation_quality --diffuse
    PYTHONPATH=src python -m tools.skin_deformation_quality --diffuse --locality 0.3

Every number is measured at run time.  ``--diffuse`` turns on
``SoftTissueSkinning.DIFFUSE_WEIGHTS`` before the binding is solved; the
binding cache keys on it, so the two configurations do not share a solve.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

#: Poses to measure, in normalised BodyState DOFs.  The first is the control.
POSES: dict[str, dict] = {
    "neutral": {},
    "hip hinge (deadlift setup)": {
        "hip_r_flex": 0.85, "hip_l_flex": 0.85,
        "knee_r_flex": 0.35, "knee_l_flex": 0.35,
        "shoulder_r_flex": 0.25, "shoulder_l_flex": 0.25,
    },
    "deep squat": {
        "hip_r_flex": 1.0, "hip_l_flex": 1.0,
        "knee_r_flex": 0.9, "knee_l_flex": 0.9,
        "ankle_r_flex": 0.6, "ankle_l_flex": 0.6,
    },
    "arms overhead": {
        "shoulder_r_abduct": 1.0, "shoulder_l_abduct": 1.0,
        "elbow_r_flex": 0.2, "elbow_l_flex": 0.2,
    },
    "trunk flexed and rotated": {
        "spine_flex": 0.8, "spine_rotation": 0.6, "spine_lat_bend": 0.4,
    },
}

#: An edge stretched past this is counted as torn.
TORN = 2.0

SETTLE_FRAMES = 90


def _skin_binding(soft):
    for b in soft.bindings:
        if not b.is_muscle and b.edge_pairs is not None:
            return b
    return None


def _edges(soft, b):
    ref = soft._resolved_reference(b)
    n = len(ref)
    e = np.asarray(b.edge_pairs).reshape(-1, 2)
    e = e[(e[:, 0] < n) & (e[:, 1] < n)]
    rest_len = np.linalg.norm(ref[e[:, 0]] - ref[e[:, 1]], axis=1)
    keep = rest_len > 1e-6
    return ref, n, e[keep], rest_len[keep]


def _moved_joints(soft, before):
    after = np.array([np.asarray(j.node.world_matrix, dtype=np.float64)
                      for j in soft.joints])
    return np.linalg.norm((after - before).reshape(len(before), -1), axis=1) > 1e-6


def _static_mask(b, moved, n):
    """True where no joint driving this vertex has moved."""
    ji = np.asarray(b.joint_indices)[:n]
    si = np.asarray(b.secondary_indices)[:n]
    driven = moved[ji] | moved[si]
    inf = getattr(b, "influences", None)
    if inf is not None and b.influence_weights is not None:
        w = np.asarray(b.influence_weights)[:n]
        driven = driven | (moved[np.asarray(inf)[:n]] & (w > 0.0)).any(axis=1)
    return ~driven


def measure(pose: dict, ctx, soft, b, cache) -> dict:
    from faceforge.core.state import BodyState

    ref, n, e, rest_len = cache
    defaults = BodyState()
    for state in (ctx.state.body, ctx.state.target_body):
        for name, value in vars(defaults).items():
            if hasattr(state, name):
                setattr(state, name, value)
        for name, value in pose.items():
            if not hasattr(state, name):
                raise SystemExit(f"unknown BodyState field {name!r}")
            setattr(state, name, value)

    before = np.array([np.asarray(j.node.world_matrix, dtype=np.float64)
                       for j in soft.joints])
    for _ in range(SETTLE_FRAMES):
        ctx.simulation.step(1 / 60)
    moved = _moved_joints(soft, before)

    pos = np.asarray(b.mesh.geometry.positions,
                     dtype=np.float64).reshape(-1, 3)[:n]
    cur = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)
    s = cur / rest_len

    ji = np.asarray(b.joint_indices)[:n]
    seam = ji[e[:, 0]] != ji[e[:, 1]]
    static = _static_mask(b, moved, n)
    drift = (float(np.linalg.norm(pos[static] - ref[static], axis=1).max())
             if static.any() else 0.0)

    return {
        "p99": float(np.percentile(s, 99)),
        "p999": float(np.percentile(s, 99.9)),
        "max": float(s.max()),
        "min": float(s.min()),
        "torn": int((s > TORN).sum()),
        "seam_p99": float(np.percentile(s[seam], 99)) if seam.any() else 0.0,
        "bulk_p99": float(np.percentile(s[~seam], 99)) if (~seam).any() else 0.0,
        "containment": drift,
        "static": int(static.sum()),
        "moved_joints": int(moved.sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diffuse", action="store_true",
                    help="turn on DIFFUSE_WEIGHTS before the binding is solved")
    ap.add_argument("--locality", type=float, default=None,
                    help="DIFFUSION_LOCALITY (smaller spreads influence further)")
    ap.add_argument("--radius-seeds", action="store_true",
                    help="seed every chain from any vertex within SEED_RADIUS "
                         "(the behaviour before ownership seeding)")
    ap.add_argument("--spatial-limit", type=float, default=None,
                    help="SKIN_SPATIAL_LIMIT: Euclidean guard on chain eligibility")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="INFLUENCE_CUTOFF_BAND: compact support as a band in model "
                         "units past the nearest segment (0 = rank-based)")
    ap.add_argument("--influences", type=int, default=None,
                    help="SKIN_INFLUENCES: bones per skin vertex (2 restores the "
                         "two-joint blend)")
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.body.soft_tissue import SoftTissueSkinning
    from faceforge.controllers import build_controllers
    from faceforge.coordination import demand_loaders
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence

    if args.diffuse:
        SoftTissueSkinning.DIFFUSE_WEIGHTS = True
    if args.locality is not None:
        SoftTissueSkinning.DIFFUSION_LOCALITY = args.locality
    if args.influences is not None:
        SoftTissueSkinning.SKIN_INFLUENCES = args.influences
    if args.cutoff is not None:
        SoftTissueSkinning.INFLUENCE_CUTOFF_BAND = args.cutoff
    if args.spatial_limit is not None:
        demand_loaders.SKIN_SPATIAL_LIMIT = args.spatial_limit
    if args.radius_seeds:
        SoftTissueSkinning.SEED_FROM_OWNED_SKIN = False

    ctx = build_app_context(argv=[])
    controllers = build_controllers(ctx)
    AssetLoadSequence(ctx).run()
    soft = ctx.simulation.soft_tissue

    t0 = time.perf_counter()
    controllers.loaders.load_skin()
    solve_s = time.perf_counter() - t0

    b = _skin_binding(soft)
    if b is None:
        print("no skin binding with edges -- nothing to measure")
        return 1
    cache = _edges(soft, b)
    print(f"skin: {cache[1]} vertices, {len(cache[2])} edges; "
          f"load+solve {solve_s:.1f} s; "
          f"SKIN_INFLUENCES={SoftTissueSkinning.SKIN_INFLUENCES} "
          f"CUTOFF={SoftTissueSkinning.INFLUENCE_CUTOFF_BAND} "
          f"SPATIAL_LIMIT={demand_loaders.SKIN_SPATIAL_LIMIT} "
          f"OWNED_SKIN_SEEDS={SoftTissueSkinning.SEED_FROM_OWNED_SKIN} "
          f"DIFFUSE_WEIGHTS={SoftTissueSkinning.DIFFUSE_WEIGHTS} "
          f"LOCALITY={SoftTissueSkinning.DIFFUSION_LOCALITY}")
    print(f"{'pose':30s} {'p99':>7s} {'p99.9':>8s} {'max':>9s} {'min':>7s} "
          f"{'torn':>8s} {'seam p99':>9s} {'bulk p99':>9s} {'contain':>8s}")
    for title, pose in POSES.items():
        m = measure(pose, ctx, soft, b, cache)
        print(f"{title:30s} {m['p99']:7.3f} {m['p999']:8.3f} {m['max']:9.2f} "
              f"{m['min']:7.4f} {m['torn']:8d} {m['seam_p99']:9.3f} "
              f"{m['bulk_p99']:9.3f} {m['containment']:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
