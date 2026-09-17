"""Does the exercise fit inside the room it is performed in?

Every other audit asks about the floor.  Nothing asks about the ceiling or the
walls, and the gym has both: a muscle-up's support position puts the shoulders
66 units above a bar that is already at 275, and the cable exercises run their
cable to an anchor that is simply outside the building.  The ceiling is drawn
double-sided, so a body through it is visible rather than silently clipped.

The gym is the *studio* room -- ``STUDIO_HEIGHT`` (300), ``STUDIO_WIDTH`` (500)
and ``STUDIO_DEPTH`` (400).  ``ROOM_HEIGHT`` (250) belongs to the clinical
scene; measuring the gym against it overstates every breach by 50 and makes
every overhead lockout look like one.

    python -m tools.audit_room_fit
    python -m tools.audit_room_fit --exercise muscle_up --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.scene.scene_environment import STUDIO_DEPTH, STUDIO_HEIGHT, STUDIO_WIDTH

logger = logging.getLogger("audit_room_fit")

#: How far past a surface counts.  The walls carry a thickness and the body's
#: own surface sits a little outside its pivots, so a unit or two is noise.
TOLERANCE = 2.0


def mesh_named(scene, name):
    """The node and mesh for a named mesh anywhere in the scene."""
    stack = [getattr(scene, "root", scene)]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(getattr(node, "children", ()))
        mesh = getattr(node, "mesh", None)
        if mesh is not None and mesh.name == name:
            return node, mesh
    return None, None


def world_extent(node, mesh, axis: int) -> tuple[float, float]:
    v = np.asarray(mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(node.world_matrix, dtype=np.float64)
    w = v @ m[axis, :3] + m[axis, 3]
    return float(w.min()), float(w.max())


def item_points(items) -> np.ndarray:
    """Every equipment vertex in world space."""
    out = []
    for item in items:
        stack = list(getattr(item.node, "children", ())) or [item.node]
        while stack:
            child = stack.pop()
            stack.extend(getattr(child, "children", ()))
            mesh = getattr(child, "mesh", None)
            if mesh is None or getattr(mesh, "geometry", None) is None:
                continue
            v = np.asarray(mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)
            m = np.asarray(child.world_matrix, dtype=np.float64)
            out.append(v @ m[:3, :3].T + m[:3, 3])
    return np.vstack(out) if out else np.zeros((0, 3))


def measure(demo, defn) -> list[dict]:
    demo.start(defn, reps=1, tempo=1.0)
    piv = demo.hs.pipeline.joint_setup.pivots
    rows = []
    for span in demo.runtime.built.spans[:len(defn.phases)]:
        demo.evaluate(span.t1 - 1e-3)
        body = np.array([p.get_world_position() for p in piv.values()], dtype=np.float64)
        skull_node, skull_mesh = mesh_named(demo.scene, "cranium")
        top = float(body[:, 1].max())
        if skull_mesh is not None:
            top = max(top, world_extent(skull_node, skull_mesh, 1)[1])
        equip = item_points(demo.runtime.rig.items)
        rows.append({
            "phase": span.name,
            "top": top,
            "equip_top": float(equip[:, 1].max()) if len(equip) else None,
            "x": (float(body[:, 0].min()), float(body[:, 0].max())),
            "z": (float(body[:, 2].min()), float(body[:, 2].max())),
        })
    demo.runtime.stop()
    return rows


def flags_for(row) -> list[str]:
    out = []
    ceiling = STUDIO_HEIGHT
    if row["top"] - ceiling > TOLERANCE:
        out.append(f"body {row['top'] - ceiling:.1f} through the ceiling")
    if row["equip_top"] is not None and row["equip_top"] - ceiling > TOLERANCE:
        out.append(f"equipment {row['equip_top'] - ceiling:.1f} through the ceiling")
    half_w, half_d = STUDIO_WIDTH / 2, STUDIO_DEPTH / 2
    if row["x"][0] < -half_w + TOLERANCE or row["x"][1] > half_w - TOLERANCE:
        out.append(f"body reaches x [{row['x'][0]:.0f}, {row['x'][1]:.0f}] "
                   f"against walls at +-{half_w:.0f}")
    if row["z"][0] < -half_d + TOLERANCE or row["z"][1] > half_d - TOLERANCE:
        out.append(f"body reaches z [{row['z'][0]:.0f}, {row['z'][1]:.0f}] "
                   f"against walls at +-{half_d:.0f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None, help="comma list; default every exercise")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every phase")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    from tools.render_exercise_demo import DemoScene, _NullCamera, _NullLights

    catalog = get_exercise_catalog()
    ids = ([x.strip() for x in args.exercise.split(",")] if args.exercise else list(catalog))
    demo = DemoScene([], with_skin=False)
    demo.activate(_NullCamera(), _NullLights(), "gym")
    print(f"gym: {STUDIO_WIDTH:.0f} x {STUDIO_DEPTH:.0f} x {STUDIO_HEIGHT:.0f} "
          f"(walls at x/z +-{STUDIO_WIDTH / 2:.0f}/+-{STUDIO_DEPTH / 2:.0f}, "
          f"ceiling at y {STUDIO_HEIGHT:.0f})\n")

    flagged = 0
    for n, eid in enumerate(ids, 1):
        defn = catalog[eid]
        try:
            rows = measure(demo, defn)
        except Exception as exc:                       # keep going: report at the end
            print(f"\n== {eid} ==\n  <-- FAILED: {exc}")
            continue
        worst = [(r, flags_for(r)) for r in rows]
        hits = [(r, f) for r, f in worst if f]
        if hits or args.verbose:
            print(f"\n== {eid} ==")
            for r, f in (worst if args.verbose else hits):
                tail = "".join(f"\n      <-- {x}" for x in f)
                print(f"  {r['phase'][:26]:28s} top={r['top']:7.1f} "
                      f"equip_top={r['equip_top'] if r['equip_top'] is None else round(r['equip_top'], 1)}"
                      f"{tail}")
        if hits:
            flagged += 1
        print(f"  [{n}/{len(ids)}]", flush=True)
    print(f"\n{flagged} exercises do not fit the room, of {len(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
