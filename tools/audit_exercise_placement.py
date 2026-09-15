"""Every exercise, every phase: is the body on the floor, through it, or above it?

``render_exercise_demo --probe`` answers this for the feet and the hands, and
only against the exercise's own anchor.  That misses the class of fault the
Turkish get-up had, where a planted foot floated 55 units in the air while the
probe -- looking at the minimum over both feet -- reported the other one fine,
and it misses an exercise that has no anchor at all and simply hovers.

This walks **every joint pivot** instead and reports, per phase, the lowest
one.  A body standing, lying or kneeling on the floor has its lowest pivot
within a few units of it; a body whose lowest pivot is 25 units up is either
hanging from something, in the air on purpose, or wrong.

    python -m tools.audit_exercise_placement            # everything
    python -m tools.audit_exercise_placement --exercise turkish_get_up
    python -m tools.audit_exercise_placement --quiet    # only the flagged
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.exercise.catalog import get_exercise_catalog

#: Below this, a pivot is inside the floor.
THROUGH = -3.0
#: Above this, nothing is touching down.  A standing ankle pivot is ~5, a
#: kneeling one ~13, and a foot resting on a bench ~50, so the threshold is
#: generous: it is looking for a body in mid-air, not a tight fit.
FLOATING = 26.0

#: Exercises that are supposed to be off the floor, and why.
AIRBORNE_ANCHORS = {"hands"}          # hanging from a bar, on parallel bars
AIRBORNE_TAGS = {"plyometric"}        # a jump has a flight phase


def audit(ids, quiet: bool) -> int:
    from tools.render_exercise_demo import DemoScene, _NullCamera, _NullLights

    catalog = get_exercise_catalog()
    demo = DemoScene(["leg_muscles"], with_skin=False)
    demo.activate(_NullCamera(), _NullLights(), "gym")
    pivots = demo.hs.pipeline.joint_setup.pivots
    names = sorted(pivots)

    flagged = 0
    for n, exercise_id in enumerate(ids, 1):
        defn = catalog[exercise_id]
        demo.start(defn, reps=1, tempo=1.0)
        rows = []
        for span, phase in zip(demo.runtime.built.spans[:len(defn.phases)], defn.phases):
            demo.evaluate(span.t1 - 1e-3)
            ys = np.array([pivots[p].get_world_position()[1] for p in names])
            low = int(np.argmin(ys))
            note = ""
            if ys[low] < THROUGH:
                note = f"<-- {names[low]} is {abs(ys[low]):.1f} through the floor"
            elif ys[low] > FLOATING and defn.anchor not in AIRBORNE_ANCHORS \
                    and not (set(defn.tags) & AIRBORNE_TAGS) and not phase.lift:
                note = f"<-- nothing is touching down (lowest {names[low]} at {ys[low]:.1f})"
            rows.append((span.name, float(ys[low]), names[low], note))
        demo.runtime.stop()
        bad = [r for r in rows if r[3]]
        flagged += len(bad)
        if bad or not quiet:
            print(f"\n== {exercise_id} ({defn.orientation}, anchor={defn.anchor}) ==",
                  flush=True)
            for name, y, pivot, note in (bad if quiet else rows):
                print(f"  {name:28s} lowest={y:7.1f}  {pivot:16s} {note}", flush=True)
        if not quiet:
            print(f"  [{n}/{len(ids)}]", flush=True)
    print(f"\n{flagged} flagged phases over {len(ids)} exercises", flush=True)
    return flagged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None, help="comma list; default every exercise")
    ap.add_argument("--quiet", action="store_true", help="print only the flagged phases")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    catalog = get_exercise_catalog()
    ids = ([x.strip() for x in args.exercise.split(",") if x.strip()]
           if args.exercise else list(catalog))
    unknown = [i for i in ids if i not in catalog]
    if unknown:
        print(f"unknown exercise(s): {', '.join(unknown)}")
        return 2
    audit(ids, args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
