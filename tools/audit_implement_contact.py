"""Does a racked implement rest ON the body, or float above it / sink into it?

`audit_equipment_clash` asks whether equipment is inside the body, and answers
with capsules: one cylinder per bone segment plus a sphere for the skull.  That
is coarse enough to miss both halves of the question a *racked* implement asks.
A back squat's bar measured "clear" on capsules while sitting 9 units above the
trapezius it is supposed to rest on and wedged behind the neck -- because the
neck capsule is a straight tube and the real trapezius is a sloping shelf.

This measures against the loaded meshes instead.  For every held item, in the
vertical column the item occupies (its x/z footprint, grown by `MARGIN`):

* **penetration** -- body surface above the item's underside and below its top,
  i.e. the item is inside the flesh;
* **gap** -- how far the item's underside floats above the highest body
  surface beneath it.

Neither number is a fault on its own: a deadlift's bar hangs at arm's length
and *should* show a large gap.  A big gap on an item the catalogue says is
racked, carried or rested, and any penetration at all, are faults.

    python -m tools.audit_implement_contact
    python -m tools.audit_implement_contact --exercise barbell_back_squat -v
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.body.hand_points import finger_ring_centre
from faceforge.exercise.catalog import get_exercise_catalog

logger = logging.getLogger("audit_implement_contact")

#: How far outside the item's x/z footprint still counts as "beneath" it.
MARGIN = 4.0
#: Only the part of an implement within this of the midline is measured.  A
#: barbell is 280 long and its ends are held in the hands, so scoring the whole
#: of it reported the bar "5.4 inside the body" when what it was inside was the
#: fingers gripping it.  The question a racked implement asks is about the
#: trunk, and the trunk is about 28 units half-width.
TRUNK_HALF = 28.0
#: Words in an exercise's own description or setup that say the implement is
#: held against the body rather than hanging from the hands.
RESTING_WORDS = ("rests on", "resting on", "racked", "front rack", "rack position",
                 "on the upper trap", "on the traps", "front deltoid", "against the chest",
                 "against the sternum", "on the shoulders", "on the back", "by the horns")
#: Body surface within this of a grip point is the hand that HOLDS the item,
#: and is excluded: a kettlebell is gripped at its handle, so the fingers sit
#: inside its bounding box by construction and every bell in the catalogue
#: reported "22 units inside the body" -- the 22 being its own handle.
GRIP_EXCLUDE = 22.0
#: A racked implement further than this from the body is not racked.
GAP_LIMIT = 4.0
#: An implement in contact with the body compresses it.  Measured across the
#: catalogue once the shape test was right: a bar resting on the trapezius 0.6,
#: a front-racked bar 1.9, a racked bell 0.7-1.0, a bench-press bar on the
#: chest 2.4, a kettlebell against the shin 2.6, a slam ball against the calf
#: 2.4.  So 2.0 called ordinary contact a fault; 4 is the point past which an
#: implement is in the flesh rather than on it.
PENETRATION_LIMIT = 4.0
#: A racked one should: a loaded bar compresses the trapezius it sits on, and
#: measured, a bar resting properly on the upper back sinks 4.6-4.9 units
#: (3-4 cm) into it.  Past this it is buried rather than resting.
RACKED_PENETRATION_LIMIT = 9.0


def rig_meshes(items) -> set[int]:
    """Every MeshInstance belonging to a rigged item, by identity."""
    out: set[int] = set()
    for item in items:
        stack = [item.node]
        while stack:
            n = stack.pop()
            stack.extend(getattr(n, "children", ()))
            mesh = getattr(n, "mesh", None)
            if mesh is not None:
                out.add(id(mesh))
    return out


def body_surface(scene, exclude_meshes: set[int]) -> np.ndarray:
    """Every vertex of every DRAWN body mesh, in world space.

    `scene.collect_meshes()` and not a walk of the tree: it is what the
    renderer and the exporter use, so it is the definition of "on screen".
    Walking the tree and testing `mesh.visible` myself included
    `body_surface` -- the sex morph's surface, which is loaded and bound but
    not drawn -- and a kettlebell swing's bell was then reported 3.3 units
    inside a body nobody can see.
    """
    out = []
    for mesh, world in scene.collect_meshes():
        if id(mesh) in exclude_meshes:
            continue
        geo = getattr(mesh, "geometry", None)
        pos = getattr(geo, "positions", None)
        if pos is None:
            continue
        v = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        m = np.asarray(world, dtype=np.float64)
        out.append(v @ m[:3, :3].T + m[:3, 3])
    return np.vstack(out) if out else np.zeros((0, 3))


def item_parts(item):
    """(name, world vertices) for every drawn part of a rigged item."""
    out = []
    stack = list(getattr(item.node, "children", ())) or [item.node]
    while stack:
        c = stack.pop()
        stack.extend(getattr(c, "children", ()))
        mesh = getattr(c, "mesh", None)
        geo = getattr(mesh, "geometry", None)
        if geo is None or getattr(geo, "positions", None) is None:
            continue
        v = np.asarray(geo.positions, dtype=np.float64).reshape(-1, 3)
        m = np.asarray(c.world_matrix, dtype=np.float64)
        out.append((c.name, v @ m[:3, :3].T + m[:3, 3]))
    return out


def drop_grips(surface: np.ndarray, grips) -> np.ndarray:
    """Body surface with the gripping hands removed."""
    keep = np.ones(len(surface), dtype=bool)
    for g in grips:
        if g is None:
            continue
        keep &= np.linalg.norm(surface - np.asarray(g, dtype=np.float64), axis=1) > GRIP_EXCLUDE
    return surface[keep]


def inside_depth(part: np.ndarray, surface: np.ndarray) -> float:
    """How far body surface reaches inside the part's own ORIENTED box.

    Five shapes had to be got right and four tries got them wrong, so the test
    is now one thing rather than a classifier:

    * an axis-aligned box called a kettlebell hanging between the shins "22
      units inside the body" -- the 22 being the bell's own height, with the
      shins either side of it;
    * distance-to-axis needed a radius, and taking it from the bounding-box
      cross-section turned a 3.6-unit bar tilted 10 degrees in a front rack
      into a 28.8-unit one, reported 14.4 inside when the nearest body point
      was 0.03 from its axis, i.e. resting on it;
    * distance-to-centroid treated a flat PLATE as a ball: a curl's 24-wide
      disc lying beside the thigh, 2.1 from the skin, read as 9.8 inside it,
      and a bike pedal with a foot standing on it read as 6.6 inside the foot.

    The principal axes of the part's own vertices give an oriented box that
    fits a bar, a plate, a bell and a ball alike.  Depth is the smallest
    distance to a face, which is what "how far in" means.
    """
    if len(surface) == 0 or len(part) < 2:
        return 0.0
    centre = part.mean(axis=0)
    centred = part - centre
    axes = (np.linalg.svd(centred, full_matrices=False)[2] if len(part) >= 3
            else np.eye(3))
    half = np.abs(centred @ axes.T).max(axis=0)
    local = np.abs((surface - centre) @ axes.T)
    inside = local[(local <= half).all(axis=1)]
    if len(inside) == 0:
        return 0.0
    return float((half - inside).min(axis=1).max())


#: Samples per phase.  Keyframes alone miss everything that happens between
#: them, which is where a bar passes a knee.
PHASE_STEPS = 6
# WHAT THIS TOOL CANNOT SEE, and which instrument can
# ---------------------------------------------------
# `inside_depth` asks how far BODY points sit inside the IMPLEMENT's oriented
# box.  For a bell, a plate, a ball or a pad that is the right question.  For
# a thin one it is not even the right shape of question: a 3.6-unit-wide
# barbell can never report more than 1.8, its own radius, so **a bar can pass
# clean through a thigh and score 4.0-clear by construction**.  That is not
# hypothetical -- the snatch's first pull took the bar 9.5 units into the
# right thigh, `audit_equipment_clash` flagged it from its capsules, this tool
# said clear, and the capsule audit was disbelieved on the strength of it.
#
# An enclosure test was written and measured and is not kept: asking whether
# body surface lies all round the bar's axis cannot tell a bar buried in a
# thigh from one nestled against the body with the forearms over it, and it
# called the deadlift's lockout 5.6 inside (the bar at the hips, thighs behind,
# forearms above) and a hinged lifter's own start position 15.7 inside -- the
# pocket a folded body makes, with the nearest flesh 15.7 away.
#
# So: **for "is the implement through a limb", use `audit_equipment_clash`**,
# whose capsules are crude about depth but right about the question.  This
# tool answers "is it resting where it should, and is the body inside it",
# which is the one the clash audit's saturation cannot.

def measure_part(part: np.ndarray, surface: np.ndarray) -> tuple[float, float]:
    """(penetration, gap) for one part against the body surface, at the midline.

    The part's box is CLIPPED to the trunk's width rather than filtering its
    vertices: a barbell is a 12-segment cylinder whose vertices sit only at
    its two ends, so filtering on |x| left nothing in the middle -- where the
    body is.  ``gap`` still uses the column, which is what "resting on" means.
    """
    lo, hi = part.min(axis=0), part.max(axis=0)
    if hi[0] < -TRUNK_HALF or lo[0] > TRUNK_HALF:
        return 0.0, float("nan")          # entirely outboard of the trunk
    lo = lo.copy(); hi = hi.copy()
    lo[0], hi[0] = max(lo[0], -TRUNK_HALF), min(hi[0], TRUNK_HALF)
    column = surface[(surface[:, 0] >= lo[0] - MARGIN) & (surface[:, 0] <= hi[0] + MARGIN)
                     & (surface[:, 2] >= lo[2] - MARGIN) & (surface[:, 2] <= hi[2] + MARGIN)]
    penetration = inside_depth(part, surface)
    if len(column) == 0:
        return penetration, float("inf")
    below = column[column[:, 1] <= lo[1]]
    gap = float(lo[1] - below[:, 1].max()) if len(below) else float("inf")
    return penetration, gap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None, help="comma list; default every exercise")
    ap.add_argument("-v", "--verbose", action="store_true", help="every phase and part")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    from faceforge.exercise.muscle_groups import ALL_MUSCLE_REGIONS
    from tools.render_exercise_demo import DemoScene, _NullCamera, _NullLights

    catalog = get_exercise_catalog()
    ids = ([x.strip() for x in args.exercise.split(",")] if args.exercise else list(catalog))
    demo = DemoScene(list(ALL_MUSCLE_REGIONS), with_skin=True)
    demo.activate(_NullCamera(), _NullLights(), "gym")

    flagged = 0
    for n, eid in enumerate(ids, 1):
        defn = catalog[eid]
        held = [e for e in defn.equipment if e.attach not in ("static", "floor")]
        if not held:
            print(f"  [{n}/{len(ids)}]", flush=True)
            continue
        text = " ".join([defn.description, *defn.setup]).lower()
        racked = any(w in text for w in RESTING_WORDS)
        try:
            demo.start(defn, reps=1, tempo=1.0)
        except Exception as exc:
            print(f"\n== {eid} ==\n  <-- FAILED: {exc}")
            continue
        rows = []
        for span in demo.runtime.built.spans[:len(defn.phases)]:
          # Sampled THROUGH each phase, not at its keyframe.  The snatch takes
          # the bar 9.5 units into the thigh a third of the way through its
          # first pull and is clear at both ends of it; measuring the
          # keyframes alone reported the whole family clear.
          for _step in range(PHASE_STEPS):
            demo.evaluate(min(span.t0 + (span.t1 - span.t0) * (_step + 1) / PHASE_STEPS,
                              span.t1 - 1e-3))
            surface = body_surface(
                demo.scene, rig_meshes(demo.runtime.rig.items))
            pivots = demo.hs.pipeline.joint_setup.pivots
            surface = drop_grips(surface,
                                 [finger_ring_centre(pivots, s) for s in "RL"])
            for item in demo.runtime.rig.items:
                if item.spec.attach in ("static", "floor"):
                    continue
                for name, part in item_parts(item):
                    pen, gap = measure_part(part, surface)
                    rows.append((span.name, f"{item.spec.kind}/{name}", pen, gap))
        demo.runtime.stop()
        limit = RACKED_PENETRATION_LIMIT if racked else PENETRATION_LIMIT
        hits = [r for r in rows if r[2] > limit]
        finite = [r[3] for r in rows if np.isfinite(r[3])]
        best_gap = min(finite) if finite else float("inf")
        note = []
        if hits:
            worst = max(hits, key=lambda r: r[2])
            note.append(f"{worst[1]} is {worst[2]:.1f} inside the body at '{worst[0]}'"
                        f" (limit {limit:.0f})")
        if racked and best_gap > GAP_LIMIT:
            note.append(f"says it rests on the body but the closest it gets is {best_gap:.1f}")
        if note or args.verbose:
            print(f"\n== {eid} ==" + ("  (racked)" if racked else ""))
            for t in note:
                print(f"  <-- {t}")
            if args.verbose:
                for ph, part, pen, gap in rows:
                    g = "-" if not np.isfinite(gap) else f"{gap:7.1f}"
                    print(f"  {ph[:22]:24s} {part[:24]:26s} inside {pen:6.1f}  gap {g}")
        if note:
            flagged += 1
        print(f"  [{n}/{len(ids)}]", flush=True)
    print(f"\n{flagged} exercises with an implement that is inside the body "
          f"or floating clear of one it should rest on, of {len(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
