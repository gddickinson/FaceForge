"""How far the neck muscles deform, and how much they stretch doing it.

The neck muscles do not go through the soft-tissue skinning: they have their
own head-follow + body-follow path in :mod:`faceforge.anatomy.neck_muscles`.
This measures that path directly -- edge stretch against the rest mesh, and
displacement from rest -- for a handful of poses, plus the scene-wrapper
control that exposed the frame bug (the gym wrapper stands the body up, and
any reading of the skeleton that forgets to cancel it lands ~186 units away).

    PYTHONPATH=src python -m tools.neck_deformation_quality
    PYTHONPATH=src python -m tools.neck_deformation_quality --top 12

Every number is measured at run time; nothing here is asserted.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

#: Poses to measure, as BodyState/FaceState field overrides.
POSES: dict[str, dict] = {
    "neutral": {},
    "spine flex (full)": {"spine_flex": 1.0},
    "spine lateral bend": {"spine_lat_bend": 1.0},
    "spine rotation": {"spine_rotation": 1.0},
    "arms abducted": {"shoulder_r_abduct": 1.0, "shoulder_l_abduct": 1.0},
    "arms flexed forward": {"shoulder_r_flex": 1.0, "shoulder_l_flex": 1.0},
}

#: Head rotations to measure, in normalised head-state units.
HEAD_POSES: dict[str, dict] = {
    "head yaw (full)": {"head_yaw": 1.0},
    "head pitch (full)": {"head_pitch": 1.0},
}

SETTLE_FRAMES = 120


def _edge_cache(muscle_data):
    cache = {}
    for md in muscle_data:
        tri = np.asarray(md.mesh.geometry.indices).reshape(-1, 3)
        e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        rest = md.rest_positions.reshape(-1, 3).astype(np.float64)
        lengths = np.linalg.norm(rest[e[:, 0]] - rest[e[:, 1]], axis=1)
        keep = lengths > 1e-6
        cache[md.defn.get("name", "?")] = (e[keep], lengths[keep], rest)
    return cache


def _measure(muscle_data, cache):
    rows = []
    for md in muscle_data:
        name = md.defn.get("name", "?")
        e, lengths, rest = cache[name]
        cur = md.mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
        ratio = np.linalg.norm(cur[e[:, 0]] - cur[e[:, 1]], axis=1) / lengths
        rows.append({
            "name": name,
            "p99": float(np.percentile(ratio, 99)),
            "max": float(ratio.max()),
            "moved": float(np.linalg.norm(cur - rest, axis=1).max()),
        })
    return rows


def _apply(ctx, body_fields, head_fields):
    """Reset every joint and head angle, then set the ones this pose names."""
    from dataclasses import fields
    from faceforge.core.state import BodyState

    defaults = BodyState()
    for state in (ctx.state.body, ctx.state.target_body):
        for f in fields(BodyState):
            setattr(state, f.name, getattr(defaults, f.name))
        for name, value in body_fields.items():
            if not hasattr(state, name):
                raise SystemExit(f"unknown BodyState field {name!r}")
            setattr(state, name, value)

    for state in (ctx.state.face, ctx.state.target_head):
        for name in ("head_yaw", "head_pitch", "head_roll"):
            if hasattr(state, name):
                setattr(state, name, 0.0)
        for name, value in head_fields.items():
            if hasattr(state, name):
                setattr(state, name, value)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=6,
                    help="how many muscles to name per pose (default 6)")
    ap.add_argument("--wrapper", action="store_true",
                    help="also measure with the body inside the gym scene")
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.controllers import build_controllers
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence
    from faceforge.core.events import EventType

    ctx = build_app_context(argv=[])
    build_controllers(ctx)
    AssetLoadSequence(ctx).run()
    sim = ctx.simulation
    if sim.neck_muscles is None:
        print("no neck muscles loaded", file=sys.stderr)
        return 1
    if sim.neck_muscle_group is not None:
        sim.neck_muscle_group.visible = True

    cache = _edge_cache(sim.neck_muscles.muscle_data)

    cases = [(n, f, {}) for n, f in POSES.items()]
    cases += [(n, {}, f) for n, f in HEAD_POSES.items()]
    if args.wrapper:
        ctx.event_bus.publish(EventType.SCENE_MODE_TOGGLED, enabled=True,
                              scene_type="gym")
        cases = [(f"{n} (in the gym)", b, h) for n, b, h in cases]

    print(f"{'pose':26s} {'p99':>7s} {'max':>7s} {'moved':>8s}  worst muscles")
    for title, body_fields, head_fields in cases:
        _apply(ctx, body_fields, head_fields)
        for _ in range(SETTLE_FRAMES):
            sim.step(1 / 60)
        rows = sorted(_measure(sim.neck_muscles.muscle_data, cache),
                      key=lambda r: -r["p99"])
        worst = rows[0]
        names = ", ".join(r["name"] for r in rows[:args.top] if r["p99"] > 1.01)
        print(f"{title:26s} {worst['p99']:7.3f} {worst['max']:7.3f} "
              f"{worst['moved']:8.3f}  {names or 'none above 1.01'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
