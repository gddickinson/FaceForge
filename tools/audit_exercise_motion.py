"""Does the exercise actually move the body, and does the load reach the floor?

The placement audit asks where the *lowest* pivot is, which the ground lock
pins by construction: a standing calf raise whose heels never leave the floor
reports the same 3.1 in every phase as one that works.  This asks the
complementary questions, which nothing else does:

* **Excursion.**  Per exercise, how far does each joint pivot travel between
  phases, and how far does the whole body rise?  An exercise whose largest
  pivot excursion is a few units is a still photograph with phase names.
* **Heel lift.**  For a calf raise or a jump, the heel must leave the floor.
* **The loaded bar.**  A barbell's plates have a radius; when the exercise
  puts the bar on the floor the lowest plate point should be near y = 0, and
  when it does not, the bar should be clear of it.

    python -m tools.audit_exercise_motion
    python -m tools.audit_exercise_motion --exercise standing_calf_raise -v
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.exercise.catalog import get_exercise_catalog

logger = logging.getLogger("audit_exercise_motion")

STILL = 6.0        # an exercise whose biggest pivot moves less than this
#: Exercises whose name promises the heels leave the floor.  "snatch" and
#: "clean" alone were too broad: a KETTLEBELL clean swings the bell through
#: a hike pass with both feet planted and is right to.  The barbell lifts
#: keep the check because triple extension does come onto the toes.
HEEL_EXERCISES = ("calf_raise", "jump", "hop", "skip", "jerk",
                  "power_clean", "power_snatch", "squat_snatch", "clean_and_jerk",
                  "clean_pull", "snatch_pull")


def lowest_point(node) -> float:
    """The lowest world y any of this item's own geometry reaches.

    The parts carry their vertices in their own local frame and may be rotated
    (the bar is a cylinder turned onto X), so the mesh vertices are taken
    through each part's world matrix rather than approximated by a radius.
    """
    lo = float("inf")
    stack = [node]
    while stack:
        n = stack.pop()
        stack.extend(getattr(n, "children", ()))
        mesh = getattr(n, "mesh", None)
        geo = getattr(mesh, "geometry", None)
        pos = getattr(geo, "positions", None)
        if pos is None:
            continue
        v = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        m = np.asarray(n.world_matrix, dtype=np.float64)
        y = v @ m[1, :3] + m[1, 3]
        lo = min(lo, float(y.min()))
    return lo if lo < float("inf") else float(node.get_world_position()[1])


def measure(demo, defn) -> list[dict]:
    demo.start(defn, reps=1, tempo=1.0)
    piv = demo.hs.pipeline.joint_setup.pivots
    rows = []
    for span in demo.runtime.built.spans[:len(defn.phases)]:
        demo.evaluate(span.t1 - 1e-3)
        pos = {n: np.asarray(p.get_world_position(), dtype=np.float64)
               for n, p in piv.items()}
        equip = {}
        for item in demo.runtime.rig.items:
            base = np.asarray(item.node.get_world_position(), dtype=np.float64)
            equip[item.spec.kind] = (float(base[1]), lowest_point(item.node))
        rows.append({"phase": span.name, "pos": pos, "equip": equip})
    demo.runtime.stop()
    return rows


def report(defn, rows, verbose: bool) -> list[str]:
    names = sorted(set.intersection(*(set(r["pos"]) for r in rows)))
    span = {n: float(np.linalg.norm(
        np.array([r["pos"][n] for r in rows]).max(axis=0)
        - np.array([r["pos"][n] for r in rows]).min(axis=0))) for n in names}
    worst = max(span.values()) if span else 0.0
    mover = max(span, key=span.get) if span else "-"
    flags = []
    if worst < STILL and len(rows) > 1:
        flags.append(f"STILL: largest pivot excursion {worst:.1f} ({mover})")

    if any(k in defn.id for k in HEEL_EXERCISES):
        heels = [min(r["pos"][f"ankle_{s}"][1] for s in "RL") for r in rows]
        lift = max(heels) - min(heels)
        if lift < 5.0:
            flags.append(f"HEEL: ankle height varies only {lift:.1f} over the rep")

    for kind in set().union(*(set(r["equip"]) for r in rows)) if rows else ():
        lows = [r["equip"][kind][1] for r in rows if kind in r["equip"]]
        if not lows:
            continue
        if min(lows) < -3.0:
            flags.append(f"{kind}: reaches {min(lows):.1f}, through the floor")
        # Only exercises that SAY they go to the floor.  Matching on the id
        # ("clean", "snatch") called the kettlebell clean's hike pass a fault:
        # that phase swings the bell back between the legs at hip height and
        # is not meant to reach anything.  The phase names are the honest
        # source -- "Lower to the floor", "Floor", "Reset", "Pins".
        floor_phases = [r for r in rows
                        if "floor" in r["phase"].lower() and kind in r["equip"]]
        if floor_phases:
            deepest = min(r["equip"][kind][1] for r in floor_phases)
            if deepest > 12.0:
                flags.append(f"{kind}: the '{min(floor_phases, key=lambda r: r['equip'][kind][1])['phase']}'"
                             f" phase leaves it {deepest:.1f} above the floor")

    out = []
    if flags or verbose:
        out.append(f"\n== {defn.id} ==")
        for f in flags:
            out.append(f"  <-- {f}")
        if verbose:
            for r in rows:
                p = r["pos"]
                out.append(
                    f"  {r['phase'][:26]:28s} ankle={min(p['ankle_R'][1], p['ankle_L'][1]):6.1f} "
                    f"knee={p['knee_R'][1]:6.1f} hip={p['hip_R'][1]:6.1f} "
                    f"shoulder={p['shoulder_R'][1]:6.1f} wrist={p['wrist_R'][1]:6.1f}  "
                    + "  ".join(f"{k}(base {b:.0f}, low {l:.0f})"
                                for k, (b, l) in r["equip"].items()))
            out.append(f"  largest pivot excursion {worst:.1f} ({mover})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    from tools.render_exercise_demo import DemoScene, _NullCamera, _NullLights

    catalog = get_exercise_catalog()
    ids = ([x.strip() for x in args.exercise.split(",")] if args.exercise
           else list(catalog))
    demo = DemoScene([], with_skin=False)
    demo.activate(_NullCamera(), _NullLights(), "gym")
    flagged = 0
    for n, eid in enumerate(ids, 1):
        defn = catalog[eid]
        try:
            lines = report(defn, measure(demo, defn), args.verbose)
        except Exception as exc:
            lines = [f"\n== {eid} ==", f"  <-- FAILED: {exc}"]
        if lines:
            flagged += 1
            print("\n".join(lines), flush=True)
        print(f"  [{n}/{len(ids)}]", flush=True)
    print(f"\n{flagged} exercises with something to say, of {len(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
