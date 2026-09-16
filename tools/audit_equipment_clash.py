"""Does the equipment pass through the body?

``audit_equipment_contact`` asks whether the body is near its equipment; this
asks the opposite question -- whether any of it is *inside* the body -- and it
has to work on the body's limbs and skull rather than on joint pivots.  A bar
through the neck touches no pivot at all, which is why the muscle-up's bar
could pass through the model unreported.

The body is approximated as capsules: one per bone segment between connected
pivots, plus a sphere for the skull, each with a radius taken from the segment
it stands for.  Every equipment part is a box.  A part that reaches inside a
capsule by more than the tolerance is drawn through the body.

Phases are sampled *through*, not just at their keyframes: a muscle-up's bar
is above the chest at one keyframe and below it at the next, and the sweep in
between is the whole fault.

    python -m tools.audit_equipment_clash
    python -m tools.audit_equipment_clash --exercise muscle_up --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.exercise.catalog import get_exercise_catalog
from tools.audit_equipment_contact import part_bounds

#: Bone segments, as (proximal pivot, distal pivot, radius).  Radii are the
#: flesh, not the bone: a thigh is thicker than a forearm and a bar grazing
#: the skin is not the same fault as one through the femur.
SEGMENTS: tuple[tuple[str, str, float], ...] = tuple(
    [(f"shoulder_{s}", f"elbow_{s}", 9.0) for s in "RL"]
    + [(f"elbow_{s}", f"wrist_{s}", 7.0) for s in "RL"]
    # 12 was too fat to tell resting from penetrating: at 12 every deadlift
    # lockout -- where the bar legitimately RESTS on the thigh -- read as 7.4
    # units inside it, i.e. the bar's surface sitting 4.6 from the femur axis
    # was called a clash.  10 keeps those quiet without hiding anything the
    # renders show.  (Calibrated from that behaviour, not from a measured
    # thigh: the radii here are a coarse stand-in for flesh.)
    + [(f"hip_{s}", f"knee_{s}", 10.0) for s in "RL"]
    + [(f"knee_{s}", f"ankle_{s}", 8.5) for s in "RL"]
    + [("shoulder_R", "shoulder_L", 13.0), ("hip_R", "hip_L", 13.0),
       ("shoulder_R", "hip_R", 13.0), ("shoulder_L", "hip_L", 13.0)]
)
#: The skull, as a sphere above the shoulder midpoint: there is no head pivot,
#: and the head is the part a bar most obviously must not pass through.
SKULL_RISE, SKULL_RADIUS = 34.0, 11.0
#: How far a part may reach inside a capsule before it counts.  Equipment
#: rests ON the body constantly -- a bar on the back, a bell at the chest --
#: so the tolerance is the depth at which resting becomes passing through.
TOLERANCE = 7.0
#: Parts a body is *meant* to be held by: it sits on them, stands on them or
#: is encircled by them, so a capsule always reports them deep inside it.
#: They are counted, but apart, so a bar through an arm is not lost in them.
SUPPORT_PARTS = frozenset({
    "pad", "box", "seat", "bench", "footplate", "saddle", "band", "rail",
    "step", "platform", "deck", "belt", "backrest", "cushion", "block",
})
#: Samples per phase.  The keyframes alone miss everything that happens on
#: the way between them.
STEPS = 8


def segment_points(pivots: dict) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    out = []
    for a, b, radius in SEGMENTS:
        na, nb = pivots.get(a), pivots.get(b)
        if na is None or nb is None:
            continue
        out.append((f"{a}->{b}",
                    np.asarray(na.get_world_position(), dtype=np.float64),
                    np.asarray(nb.get_world_position(), dtype=np.float64), radius))
    sr, sl = pivots.get("shoulder_R"), pivots.get("shoulder_L")
    if sr is not None and sl is not None:
        mid = 0.5 * (np.asarray(sr.get_world_position(), dtype=np.float64)
                     + np.asarray(sl.get_world_position(), dtype=np.float64))
        # Up the trunk's own axis, so a bent-over lifter's head is in front of
        # the shoulders rather than above them.
        hip = pivots.get("hip_R")
        if hip is not None:
            axis = mid - np.asarray(hip.get_world_position(), dtype=np.float64)
            n = float(np.linalg.norm(axis))
            axis = axis / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
        else:
            axis = np.array([0.0, 1.0, 0.0])
        head = mid + axis * SKULL_RISE
        out.append(("skull", head, head, SKULL_RADIUS))
    return out


def deepest_inside(lo: np.ndarray, hi: np.ndarray, a: np.ndarray, b: np.ndarray,
                   radius: float, samples: int = 24) -> float:
    """How far the box reaches inside the capsule, as a depth in units."""
    best = 0.0
    for t in np.linspace(0.0, 1.0, samples):
        p = a + (b - a) * t
        nearest = np.clip(p, lo, hi)
        gap = float(np.linalg.norm(p - nearest))
        if gap < radius:
            best = max(best, radius - gap)
    return best


def audit(ids, verbose: bool, steps: int = STEPS) -> int:
    from tools.render_exercise_demo import DemoScene, _NullCamera, _NullLights

    catalog = get_exercise_catalog()
    demo = DemoScene(["leg_muscles"], with_skin=False)
    demo.activate(_NullCamera(), _NullLights(), "gym")
    pivots = demo.hs.pipeline.joint_setup.pivots

    flagged = 0
    for exercise_id in ids:
        defn = catalog[exercise_id]
        if not defn.equipment:
            continue
        demo.start(defn, reps=1, tempo=1.0)
        rows = []
        for span in demo.runtime.built.spans[:len(defn.phases)]:
            for step in range(steps):
                t = span.t0 + (span.t1 - span.t0) * (step + 1) / steps
                demo.evaluate(min(t, span.t1 - 1e-3))
                segs = segment_points(pivots)
                for item in demo.runtime.rig.items:
                    if item.spec.kind == "mat":
                        continue
                    for part_name, lo, hi in part_bounds(item.node):
                        for seg_name, a, b, radius in segs:
                            depth = deepest_inside(lo, hi, a, b, radius)
                            if depth > TOLERANCE:
                                rows.append((span.name, item.spec.kind, part_name,
                                             seg_name, depth))
        demo.runtime.stop()
        if rows:
            worst: dict = {}
            for span_name, kind, part, seg, depth in rows:
                key = (kind, part, seg)
                if depth > worst.get(key, (0.0, ""))[0]:
                    worst[key] = (depth, span_name)
            hard = {k: v for k, v in worst.items() if k[1] not in SUPPORT_PARTS}
            soft = len(worst) - len(hard)
            flagged += len(hard)
            print(f"\n== {exercise_id} ==" + (f"   (+{soft} on support surfaces)" if soft else ""),
                  flush=True)
            for (kind, part, seg), (depth, span_name) in sorted(
                    hard.items(), key=lambda kv: -kv[1][0]):
                print(f"  {kind}/{part:16s} is {depth:5.1f} inside {seg:22s} "
                      f"(worst at {span_name})", flush=True)
        elif verbose:
            print(f"\n== {exercise_id} ==  clear", flush=True)
    print(f"\n{flagged} (part, body segment) clashes over {len(ids)} exercises "
          f"(support surfaces excluded)", flush=True)
    return flagged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None, help="comma list; default every exercise")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS, help="samples per phase")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    catalog = get_exercise_catalog()
    ids = ([x.strip() for x in args.exercise.split(",") if x.strip()]
           if args.exercise else list(catalog))
    unknown = [i for i in ids if i not in catalog]
    if unknown:
        print(f"unknown exercise(s): {', '.join(unknown)}")
        return 2
    audit(ids, args.verbose, args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
