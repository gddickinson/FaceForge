"""Is the body anywhere near the equipment it is supposed to be using?

The rig puts hand-held items in the hands by construction, so a barbell is
always where the hands are.  Static equipment is not: a bench, a box, a bike
or a rower is placed at a fixed point in the room and the body is placed
separately, so the two can miss each other entirely -- the rowing machine sat
154 units from its athlete's feet and nothing said so.

For each static item this measures its world bounding box and reports the
nearest body pivot to it, and how far any pivot is *inside* it.  Nothing near
means the body is not using it; deep inside means the body is standing in it.
Neither needs a table of what should touch what, which is the point: a table
would encode the same assumptions that produced the fault.

    python -m tools.audit_equipment_contact
    python -m tools.audit_equipment_contact --exercise stationary_bike --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.exercise.catalog import get_exercise_catalog

#: Further than this from the nearest body pivot and nothing is using it.
UNUSED = 26.0
#: Deeper inside than this and the body is standing in the furniture.
BURIED = 8.0


def part_bounds(node) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """World AABB of each *part* under ``node``, not one box round the lot.

    A pull-up bar, a dip station, a bike and a treadmill are frames: the box
    around the whole item is mostly air, and a body hanging inside it reads as
    30 units "inside the equipment" when it is touching nothing at all.  The
    first version of this tool did exactly that and flagged every pull-up.
    Per part, a pivot inside a box is inside a tube or a pad.
    """
    parts = []
    stack = [node]
    while stack:
        current = stack.pop()
        stack.extend(getattr(current, "children", ()))
        mesh = getattr(current, "mesh", None)
        geometry = getattr(mesh, "geometry", None)
        if geometry is None:
            continue
        pts = np.asarray(geometry.positions, dtype=np.float64).reshape(-1, 3)
        # Through the node's own world matrix, not its world position plus a
        # local offset: the rowing machine is turned 180 degrees about Y, and
        # measuring it by offsets put its footplate 120 units from where it is.
        m = np.asarray(current.world_matrix, dtype=np.float64)
        world = pts @ m[:3, :3].T + m[:3, 3]
        parts.append((getattr(current, "name", "?"), world.min(axis=0), world.max(axis=0)))
    return parts


def oriented_parts(node) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Each part as ``(name, centre, axes, half)`` -- its OWN box, not an AABB.

    An axis-aligned box round a rotated part is mostly air.  A bench pad is
    150 x 6 and lies along X; tilted 30 degrees its AABB is 80 units tall, so
    a lifter resting on the surface measures as deep inside the *box* while
    being nowhere near the pad.  That is what made "the clavicle is 15.3
    inside the bench pad" impossible to act on: the number was the AABB's.

    The axes come from the node's own world matrix, normalised, so this is the
    part's real box wherever the rig has turned it -- no PCA guess needed.
    """
    parts = []
    stack = [node]
    while stack:
        current = stack.pop()
        stack.extend(getattr(current, "children", ()))
        geometry = getattr(getattr(current, "mesh", None), "geometry", None)
        if geometry is None:
            continue
        pts = np.asarray(geometry.positions, dtype=np.float64).reshape(-1, 3)
        m = np.asarray(current.world_matrix, dtype=np.float64)
        axes = m[:3, :3].T.copy()
        norms = np.linalg.norm(axes, axis=1)
        if np.any(norms < 1e-9):
            continue
        axes /= norms[:, None]                      # scale belongs in the extent
        world = pts @ m[:3, :3].T + m[:3, 3]
        local = world @ axes.T
        lo, hi = local.min(axis=0), local.max(axis=0)
        centre = (lo + hi) / 2.0 @ axes             # back out of the part's frame
        parts.append((getattr(current, "name", "?"), centre, axes, (hi - lo) / 2.0))
    return parts


def depth_in_oriented(centre, axes, half, point: np.ndarray) -> float:
    """How far *point* is inside the oriented box; 0.0 if it is outside."""
    local = np.abs((point - centre) @ axes.T)
    if np.any(local > half):
        return 0.0
    return float(np.min(half - local))


def gap_to_box(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """``(distance outside, depth inside)`` for a point against an AABB."""
    outside = float(np.linalg.norm(np.maximum(np.maximum(lo - point, point - hi), 0.0)))
    inside = 0.0 if outside > 0.0 else float(np.min(np.minimum(point - lo, hi - point)))
    return outside, inside


def audit(ids, verbose: bool) -> int:
    from tools.render_exercise_demo import DemoScene, _NullCamera, _NullLights

    catalog = get_exercise_catalog()
    demo = DemoScene(["leg_muscles"], with_skin=False)
    demo.activate(_NullCamera(), _NullLights(), "gym")
    pivots = demo.hs.pipeline.joint_setup.pivots
    names = sorted(pivots)

    flagged = 0
    for exercise_id in ids:
        defn = catalog[exercise_id]
        statics = [e for e in defn.equipment if e.attach == "static" and e.kind != "mat"]
        if not statics:
            continue
        demo.start(defn, reps=1, tempo=1.0)
        items = [i for i in demo.runtime.rig.items
                 if i.spec.attach == "static" and i.spec.kind != "mat"]
        rows = []
        for span in demo.runtime.built.spans[:len(defn.phases)]:
            demo.evaluate(span.t1 - 1e-3)
            pts = np.array([pivots[n].get_world_position() for n in names], dtype=np.float64)
            for item in items:
                parts = part_bounds(item.node)
                if not parts:
                    continue
                best_out, best_near = float("inf"), ""
                worst_in, worst_name, worst_part = 0.0, "", ""
                for part_name, lo, hi in parts:
                    gaps = [gap_to_box(p, lo, hi) for p in pts]
                    i_out = int(np.argmin([g[0] for g in gaps]))
                    if gaps[i_out][0] < best_out:
                        best_out, best_near = gaps[i_out][0], names[i_out]
                    i_in = int(np.argmax([g[1] for g in gaps]))
                    if gaps[i_in][1] > worst_in:
                        worst_in, worst_name, worst_part = gaps[i_in][1], names[i_in], part_name
                note = ""
                if best_out > UNUSED:
                    note = (f"<-- nothing is on it (nearest {best_near} is "
                            f"{best_out:.1f} away)")
                elif worst_in > BURIED:
                    note = f"<-- {worst_name} is {worst_in:.1f} inside its {worst_part}"
                rows.append((span.name, item.spec.kind, best_out, best_near, worst_in,
                             f"{worst_name}/{worst_part}" if worst_name else "-", note))
        demo.runtime.stop()
        bad = [r for r in rows if r[6]]
        flagged += len(bad)
        if bad or verbose:
            print(f"\n== {exercise_id} ==", flush=True)
            for name, kind, out, near, deep, deep_name, note in (rows if verbose else bad):
                print(f"  {name:26s} {kind:12s} nearest={out:6.1f} ({near:16s}) "
                      f"deepest_inside={deep:5.1f} ({deep_name:28s}) {note}", flush=True)
    print(f"\n{flagged} flagged (phase, item) pairs over {len(ids)} exercises", flush=True)
    return flagged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None, help="comma list; default every exercise")
    ap.add_argument("--verbose", action="store_true", help="print every row, not just flags")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    catalog = get_exercise_catalog()
    ids = ([x.strip() for x in args.exercise.split(",") if x.strip()]
           if args.exercise else list(catalog))
    unknown = [i for i in ids if i not in catalog]
    if unknown:
        print(f"unknown exercise(s): {', '.join(unknown)}")
        return 2
    audit(ids, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
