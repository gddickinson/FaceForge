"""Are the feet doing anything, or are they rigid wedges?

The ground lock puts the body's lowest extremity on the floor.  If the toes
never bend, a foot whose heel is raised touches down on the *point* of its
longest toe and the model balances there -- which is both wrong to look at and
a lie about where the load goes.  This reports, per phase and per side:

* which foot pivot is lowest and how far the heel sits above it;
* whether any foot pivot has gone through the floor;
* whether the toes are bent at all (``toe_curl``).

A phase is flagged ``tiptoe`` when the lowest pivot is a toe *tip* with the
heel well above it and no toe angle authored -- a rigid wedge standing on its
point -- and ``buried`` when a pivot is below the floor.

    python -m tools.audit_foot_contact
    python -m tools.audit_foot_contact --exercise high_lunge --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.body.dof_ranges import dof_to_degrees
from faceforge.exercise.catalog import get_exercise_catalog
from tools.audit_equipment_contact import part_bounds

#: Foot pivots, heel (the ankle) first then out along the third ray.  The
#: pivots sit at each bone's proximal end, so ``prox`` -- the base of the
#: proximal phalanx -- is the BALL of the foot and ``dist`` is the toe tip.
_RAY = ("mt", "prox", "mid", "dist")
#: How far the heel must be above the contact before "the heel is up".
HEEL_UP = 6.0
#: How far the tip may drop below the ball before the foot is a rigid spike
#: standing on its points rather than a foot with its pads down.
TIP_DROP = 3.0
#: How far below the floor a pivot may sit (the pivot is inside the flesh).
BURIED = 1.0
#: How close the foot must be to whatever is under it to count as standing on
#: it.  A dip's feet hang in the air and a leg curl's point at the ceiling;
#: neither is a foot that should be flat, so neither is a fault.
CONTACT = 6.0


def surface_under(point: np.ndarray, items) -> float:
    """Height of the highest thing directly under ``point`` -- the floor at 0,
    or the top of an equipment box whose footprint contains it."""
    best = 0.0
    for item in items:
        for _name, lo, hi in part_bounds(item.node):
            if lo[0] <= point[0] <= hi[0] and lo[2] <= point[2] <= hi[2] \
                    and hi[1] <= point[1] + CONTACT:
                best = max(best, float(hi[1]))
    return best


def foot_pivots(pivots: dict, side: str) -> list[tuple[str, np.ndarray]]:
    names = [f"ankle_{side}"] + [f"toe_{side}_{d}_{seg}"
                                 for d in range(1, 6) for seg in _RAY]
    return [(n, np.asarray(pivots[n].get_world_position(), dtype=np.float64))
            for n in names if n in pivots]


def audit(ids, verbose: bool) -> int:
    from tools.render_exercise_demo import DemoScene, _NullCamera, _NullLights

    catalog = get_exercise_catalog()
    demo = DemoScene(["leg_muscles"], with_skin=False)
    demo.activate(_NullCamera(), _NullLights(), "gym")
    pivots = demo.hs.pipeline.joint_setup.pivots

    flagged = 0
    for exercise_id in ids:
        defn = catalog[exercise_id]
        demo.start(defn, reps=1, tempo=1.0)
        rows = []
        for phase, span in zip(defn.phases, demo.runtime.built.spans):
            demo.evaluate(span.t1 - 1e-3)
            for side in "RL":
                pts = foot_pivots(pivots, side)
                if not pts:
                    continue
                by_name = dict(pts)
                name, low = min(pts, key=lambda kv: kv[1][1])
                heel = by_name[f"ankle_{side}"][1]
                ball = by_name.get(f"toe_{side}_3_prox")
                tip = by_name.get(f"toe_{side}_3_dist")
                toe = dof_to_degrees(f"toe_curl_{side.lower()}",
                                     float(phase.pose.get(f"toe_curl_{side.lower()}", 0.0)))
                ground = surface_under(low, demo.runtime.rig.items)
                standing = low[1] - ground <= CONTACT
                note = []
                if (standing and ball is not None and tip is not None
                        and heel - ball[1] > HEEL_UP and ball[1] - tip[1] > TIP_DROP):
                    note.append(f"tip {ball[1] - tip[1]:.1f} below the ball")
                if low[1] < ground - BURIED:
                    through = "the floor" if ground == 0.0 else f"a surface at {ground:.0f}"
                    note.append(f"{name.replace(f'_{side}_', '_')} "
                                f"{ground - low[1]:.1f} through {through}")
                if note:
                    rows.append((phase.name, side, name, low[1], heel, toe, "; ".join(note)))
        demo.runtime.stop()
        if rows:
            flagged += len(rows)
            print(f"\n== {exercise_id} ==", flush=True)
            for phase_name, side, name, y, heel, toe, note in rows:
                print(f"  {phase_name:24s} {side}  lowest {name:16s} y={y:6.1f} "
                      f"heel y={heel:6.1f} toe={toe:5.1f}  <-- {note}", flush=True)
        elif verbose:
            print(f"\n== {exercise_id} ==  feet fine", flush=True)
    print(f"\n{flagged} flagged (phase, foot) pairs over {len(ids)} exercises", flush=True)
    return flagged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None, help="comma list; default every exercise")
    ap.add_argument("--verbose", action="store_true")
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
