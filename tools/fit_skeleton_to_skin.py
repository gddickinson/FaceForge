"""Solve the transform that puts the skeleton inside the body-surface mesh.

Run once, offline; the answer is shipped in ``assets/config/skeleton_fit.json``
and applied at runtime by :class:`faceforge.body.skeleton_fit.SkeletonFit`.

    python -m tools.fit_skeleton_to_skin --measure          # what it is now
    python -m tools.fit_skeleton_to_skin --solve --write    # solve and ship it
    python -m tools.fit_skeleton_to_skin --measure --fit    # after the fit

The objective is the thing the user asked for and nothing else: how far the
skeleton's vertices stick out of the surface.  Each region is given a 3x3
matrix -- a rotation times a per-axis scale, about the joint it hangs from --
and the matrices are chosen by coordinate descent on the squared protrusion,
with a penalty that stops a bone shrinking away from a surface it cannot
reach.  Regions are solved parents first, and each region's objective counts
every point it carries, so the trunk is placed knowing where the limbs will
end up.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from faceforge.body.fit_regions import (
    OFFSET_REGIONS, REGIONS, REGION_NAMES, RegionTransforms, anchors,
)
from faceforge.body.skeleton_fit import SkeletonFit, node_offset
from faceforge.constants import CONFIG_DIR
from tools.skeleton_containment import (
    SurfaceDepth, bone_points, report, surface_of,
)

logger = logging.getLogger(__name__)

#: Clearance a bone is asked to keep inside the surface.  Zero would call a
#: vertex lying exactly in the skin "contained"; the skin has thickness.
MARGIN = 1.0

#: A per-region margin was tried for the skull, whose scalp is thick, and it
#: made the head worse rather than better: unable to satisfy 2.5 units
#: anywhere, the search simply rebalanced and the occiput came out 5.65 units
#: proud instead of 4.90.  The skull is reshaped by an authored posture
#: instead (``fit_regions.SHAPE_POSTURE``), measured from the two bounding
#: boxes, which is the one place where stating the answer beats searching for
#: it.

#: Containment alone is not enough, and the failure is spectacular rather than
#: subtle: left to minimise protrusion, the search swung both forearms across
#: the body until the hands lay inside the thighs, where nothing sticks out of
#: anything.  Measured, the fitted right wrist sat at x = -6.1 -- across the
#: midline -- and the right fingertips between the knees.  "Inside the
#: surface" is not "inside the matching part of the surface", and no
#: protrusion measure can tell the difference.
#:
#: Two ways of saying "the matching part" were tried and both are worse.
#: Tying each limb's distal end to the body mesh's own landmark for it fails
#: because those landmarks are biased by how they are found: the "ankle" is
#: the mean of a band taken from the *lateral* half of the leg, so it sits 13
#: units out from the leg's axis, and pulling the ankle onto it dragged both
#: feet clean out of the mesh -- 100% of the foot outside, a median of 8
#: units.  Calibrating how deep a bone may sit from the skin that came with
#: the skeleton fails because that mesh is not a hollow surface: 24,757 of its
#: vertices lie inside a 12-unit column through the chest, so every depth
#: measured against it comes out near zero.
#:
#: What is left is the simplest true statement about the job: this is a fit,
#: not a reposing.  The whole misfit is 27.7 units at its very worst and 2.4
#: at the median, so no bone has any business travelling much further than
#: that.  A bone may move freely up to ``MOVE_FREE`` and pays beyond it.  The
#: shoulder's real correction is 24 units and costs almost nothing; the
#: degenerate wrist's 51 costs 48, which is more than the whole rest of the
#: objective.
MOVE_FREE = 20.0
MOVE_PENALTY = 0.05

#: A mean tolerates one deep patch, and a deep patch is exactly what a viewer
#: sees.  Measured on the head: the skull sat 27.8 units deep inside a 26.0
#: head with its occiput 4.4 units out the back, and the mean-squared
#: protrusion barely noticed -- 11% of the region outside at a p95 of 0.44.
#: Shrinking and centring it takes the worst case from 5.3 to under 2 and
#: costs almost nothing anywhere else, so the worst case is in the objective.
#: A mean tolerates one deep patch, and a deep patch is what a viewer sees.
#: Pressed harder than this, though, the search starts buying a better worst
#: case by burying a region somewhere roomy: at 3.0 the hands left their
#: sleeves altogether.  The skull, which is the case that wanted a harder
#: press, is reshaped by an authored posture instead.
WORST_PERCENTILE = 98.0
WORST_WEIGHT = 1.0

#: Protrusion rewards depth without limit, so the harder the worst case is
#: pressed the more the search wants to bury a region somewhere roomy: at
#: WORST_WEIGHT 3 and nothing else, the right hand ended 9.4 units inside the
#: surface, drawn up its own sleeve.  The travel limit bounds how far a bone
#: may go; this bounds how much deeper it may end up than it started.  The
#: unfitted skeleton is a real one in roughly the right place, so its own
#: depth is the reference, and no anatomy has to be named to use it.
#: Loose on purpose: it is a guard against a region being hidden somewhere
#: roomy, not a rule about how deep a bone sits.  At 5 units it was pushing
#: the toes and hands back out against the skin -- 84% of the toes outside --
#: because a bone that sits comfortably inside is not a fault.
BURY_FREE = 12.0
BURY_PENALTY = 1.0

#: Bounds on what a region may do to itself.  A skeleton that may shrink
#: without limit fits any surface by vanishing.  The rotation bound is per
#: region and *relative* to its parent, so a chain of five reaches much
#: further than one link: the hand used to end against this bound because it
#: had to express the whole arm's turn by itself.
#: 0.70 was too tight, and the grid drawings said so before the numbers did:
#: the ribcage's anteroposterior scale sat exactly on the bound while the
#: sternum and costal cartilages still stood 4 to 5 units through the chest.
#: The cadaver's chest is simply deeper than the MakeHuman figure's.
SCALE_RANGE = (0.60, 1.20)
ROT_RANGE = 20.0                       # degrees, relative to the parent

#: How hard a region is held to its own shape.  Chosen so that a uniform 10%
#: shrink costs about as much as leaving one unit of mean protrusion.  The
#: rotation penalty is small: turning a limb costs the skeleton nothing --
#: a real one does it -- where scaling a bone changes what it is.
SCALE_PENALTY = 30.0
ROT_PENALTY = 0.0015                   # per squared degree

#: Points sampled per region for the solve.  The final report uses the whole
#: cloud; the search does not need it, and the cost is linear.
SOLVE_POINTS = 300

#: How much a region answers for the regions it carries, against its own bones.
#:
#: A region must answer for its descendants -- a thorax placed without regard
#: to where it puts the arms is no use -- but not mostly for them.  Pooling
#: the whole subtree into one mean, which is what this did first, made the
#: thorax's own bones a thirteenth of its objective, and flattening the chest
#: then cost more in the scale penalty than it saved: the sternum and costal
#: cartilages were left standing 7 to 9 units out of the mesh's chest.
#:
#: Weighting each region's sample back up to its true vertex count fixed that
#: and broke something worse, because vertex count is tessellation, not
#: anatomy: the hand and fingers carry 10,000 vertices across their many small
#: bones against the humerus's 400, so the arm's objective became the hand's,
#: and the humerus and scapula were left 10 to 17 units out.
#:
#: So: a region's own bones are one half of its objective and everything it
#: carries is the other half, each descendant region counting once.
DESCENDANT_WEIGHT = 1.0

SWEEPS = 4
PASSES = 3

#: Under a mirror in X, a scale is unchanged, a rotation vector is an axial
#: vector and flips its Y and Z components, and a translation flips its X.
MIRROR_ROTATION = np.array([1.0, -1.0, -1.0])
MIRROR_OFFSET = np.array([-1.0, 1.0, 1.0])


def mirror_of(region: str) -> str | None:
    """The region on the other side, or None for a midline one."""
    if region.endswith("_R"):
        return region[:-2] + "_L"
    if region.endswith("_L"):
        return region[:-2] + "_R"
    return None


def subtree_map() -> dict[str, list[str]]:
    """Every region reachable from each region, itself included."""
    kids: dict[str, list[str]] = {n: [] for n in REGION_NAMES}
    for rd in REGIONS:
        if rd.parent is not None:
            kids[rd.parent].append(rd.name)
    out: dict[str, list[str]] = {}
    for name in REGION_NAMES:
        stack, seen = [name], []
        while stack:
            n = stack.pop()
            seen.append(n)
            stack.extend(kids[n])
        out[name] = seen
    return out


class Solver:
    """Coordinate descent on the protrusion of the skeleton out of a surface."""

    def __init__(self, pts: NDArray, regions: NDArray, anchor_points: dict,
                 depth: SurfaceDepth) -> None:
        self.depth = depth
        self.anchors = anchor_points
        self.rest_depth: dict[str, NDArray] = {}
        self.subtree = subtree_map()
        rng = np.random.default_rng(0)
        stray = set(regions.tolist()) - set(REGION_NAMES)
        if stray:
            raise SystemExit(
                f"points landed in regions the solver does not know: {sorted(stray)}. "
                "Every point must be fitted by something, or that part of the "
                "skeleton is silently left where it is.")
        self.points: dict[str, NDArray] = {}
        for name in REGION_NAMES:
            p = pts[regions == name]
            if len(p) > SOLVE_POINTS:
                p = p[rng.choice(len(p), SOLVE_POINTS, replace=False)]
            self.points[name] = p
            self.rest_depth[name] = (depth(p) if len(p)
                                     else np.zeros(0, dtype=np.float64))
        self.params: dict[str, NDArray] = {
            n: np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) for n in REGION_NAMES}
        self.offsets: dict[str, NDArray] = {
            n: np.zeros(3) for n in OFFSET_REGIONS}

    # -- objective -----------------------------------------------------------

    def table(self) -> dict[str, dict]:
        out = {n: {"scale": self.params[n][0:3].tolist(),
                   "rotation": self.params[n][3:6].tolist()}
               for n in REGION_NAMES}
        for n in OFFSET_REGIONS:
            out[n]["offset"] = self.offsets[n].tolist()
        return out

    def cost(self, region: str) -> float:
        """Squared protrusion of everything ``region`` carries, plus its penalty."""
        t = RegionTransforms(self.table(), self.anchors, 1.0)
        own, carried = 0.0, []
        for name in self.subtree[region]:
            p = self.points[name]
            if not len(p):
                continue
            moved = t.apply(name, p)
            d = self.depth(moved)
            out = np.maximum(d + MARGIN, 0.0)
            travel = np.maximum(
                np.linalg.norm(moved - p, axis=1) - MOVE_FREE, 0.0)
            buried = np.maximum(
                self.rest_depth[name] - d - BURY_FREE, 0.0)
            value = float(np.mean(out ** 2
                                  + MOVE_PENALTY * travel ** 2
                                  + BURY_PENALTY * buried ** 2)
                          + WORST_WEIGHT
                          * np.percentile(out, WORST_PERCENTILE) ** 2)
            if name == region:
                own = value
            else:
                carried.append(value)
        q = self.params[region]
        penalty = (SCALE_PENALTY * float(np.sum((q[0:3] - 1.0) ** 2))
                   + ROT_PENALTY * float(np.sum(q[3:6] ** 2)))
        if not carried:
            return own + penalty
        return own + DESCENDANT_WEIGHT * float(np.mean(carried)) + penalty

    # -- search --------------------------------------------------------------

    def solve_region(self, region: str, verbose: bool = True) -> float:
        """Search one region's six numbers, mirroring them to the other side.

        The body-surface mesh is symmetric and so is the skeleton, so solving
        the two halves independently only lets them find different local
        optima -- which they did: the right hand ended 6.8 units inside the
        mesh's hand while the left was 44% outside it.  A midline region is
        held on the midline for the same reason: it may lengthen, widen and
        nod, but it may not twist or lean.
        """
        midline = mirror_of(region) is None
        axes = (0, 1, 2, 3) if midline else (0, 1, 2, 3, 4, 5)
        base = self.cost(region)
        start = base
        steps = [0.06, 0.03, 0.015, 0.0075]
        rot_steps = [6.0, 3.0, 1.5, 0.75]
        for sweep in range(SWEEPS):
            for axis in axes:
                step = steps[sweep] if axis < 3 else rot_steps[sweep]
                lo, hi = (SCALE_RANGE if axis < 3
                          else (-ROT_RANGE, ROT_RANGE))
                current = self.params[region][axis]
                best, best_cost = current, base
                for delta in (-step, step, -2 * step, 2 * step):
                    trial = float(np.clip(current + delta, lo, hi))
                    if trial == current:
                        continue
                    self.params[region][axis] = trial
                    self._mirror(region)
                    c = self.cost(region)
                    if c < best_cost:
                        best, best_cost = trial, c
                self.params[region][axis] = best
                self._mirror(region)
                base = best_cost
            if region in OFFSET_REGIONS:
                base = self._sweep_offset(region, base, steps[sweep] * 30.0)
        if verbose:
            q = self.params[region]
            print(f"  {region:<12} cost {start:8.3f} -> {base:8.3f}   "
                  f"scale {q[0]:.3f},{q[1]:.3f},{q[2]:.3f}  "
                  f"rot {q[3]:+6.1f},{q[4]:+6.1f},{q[5]:+6.1f}")
        return base

    def _mirror(self, region: str) -> None:
        """Copy a side region's parameters to its mirror image."""
        other = mirror_of(region)
        if other is None:
            return
        q = self.params[region]
        self.params[other][0:3] = q[0:3]
        self.params[other][3:6] = q[3:6] * MIRROR_ROTATION
        if region in OFFSET_REGIONS and other in self.offsets:
            self.offsets[other] = self.offsets[region] * MIRROR_OFFSET

    def _sweep_offset(self, region: str, base: float, step: float) -> float:
        # A midline region's offset stays on the midline.
        offset = self.offsets[region]
        axes = (1, 2) if mirror_of(region) is None else (0, 1, 2)
        for axis in axes:
            current = offset[axis]
            best, best_cost = current, base
            for delta in (-step, step, -2 * step, 2 * step):
                offset[axis] = current + delta
                c = self.cost(region)
                if c < best_cost:
                    best, best_cost = offset[axis], c
            offset[axis] = best
            base = best_cost
        return base

    def solve(self) -> dict[str, dict]:
        """Parents before children, right side only: the left is its mirror."""
        for p in range(PASSES):
            print(f"pass {p + 1}/{PASSES}")
            for rd in REGIONS:
                if rd.name.endswith("_L"):
                    continue
                self.solve_region(rd.name)
        return self.table()


# -- driver ------------------------------------------------------------------


def load_scene():
    from tools.headless_loader import load_headless_scene

    hs = load_headless_scene()
    morph = getattr(hs.pipeline, "gender_morph", None)
    if morph is None or not morph.loaded:
        raise SystemExit("The body-surface mesh did not load; nothing to fit to.")
    return hs, morph


def measure(hs, morph, label: str) -> dict:
    root = hs.named_nodes["bodyRoot"]
    pos, tris = surface_of(morph)
    jp = getattr(hs.pipeline.joint_setup, "joint_positions", {}) or {}
    hips = [np.asarray(jp[k], dtype=np.float64) for k in ("hip_R", "hip_L") if k in jp]
    probe = np.mean(hips, axis=0) if hips else pos.mean(axis=0)
    depth = SurfaceDepth(pos, tris, probe)
    pts, regions, names = bone_points(root)
    d = depth(pts)
    return report(label, d, regions, top=len(REGION_NAMES))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure", action="store_true",
                    help="report how far the skeleton is outside the surface")
    ap.add_argument("--solve", action="store_true", help="solve the fit")
    ap.add_argument("--fit", action="store_true",
                    help="apply the shipped fit before measuring")
    ap.add_argument("--gender", type=float, default=0.0)
    ap.add_argument("--write", action="store_true",
                    help="write assets/config/skeleton_fit.json")
    ap.add_argument("--out", type=Path, default=CONFIG_DIR / "skeleton_fit.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    hs, morph = load_scene()
    root = hs.named_nodes["bodyRoot"]
    jp = getattr(hs.pipeline.joint_setup, "joint_positions", {}) or {}
    if args.gender:
        morph.set_gender(args.gender)
        morph.scale_skeleton(root, jp)

    if args.fit:
        SkeletonFit().apply(root, 1.0, args.gender, jp)

    if args.measure or not args.solve:
        measure(hs, morph, "fitted" if args.fit else "as loaded")

    if not args.solve:
        return 0

    before = measure(hs, morph, "before")
    pos, tris = surface_of(morph)
    hips = [np.asarray(jp[k], dtype=np.float64) for k in ("hip_R", "hip_L") if k in jp]
    probe = np.mean(hips, axis=0) if hips else pos.mean(axis=0)
    depth = SurfaceDepth(pos, tris, probe)
    pts, regions, _ = bone_points(root)
    solver = Solver(pts, regions, anchors(root, jp, node_offset), depth)
    t0 = time.time()
    table = solver.solve()
    print(f"solved in {time.time() - t0:.0f}s")

    sex = "female" if args.gender >= 0.5 else "male"
    payload = {"version": 2,
               "_comment": "Per-region rotation (degrees, relative to the "
                           "parent region), per-axis scale and, for the "
                           "pelvis and head, an offset: the transform that "
                           "puts the skeleton inside the body-surface mesh. "
                           "Solved by tools/fit_skeleton_to_skin.py.",
               "male": {}, "female": {}}
    if args.out.exists():
        try:
            payload.update(json.loads(args.out.read_text()))
        except ValueError:
            pass
    payload[sex] = table
    if args.write:
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out}")
    else:
        print("(not written; pass --write)")

    SkeletonFit({"male": payload.get("male", {}),
                 "female": payload.get("female", {})}
                ).apply(root, 1.0, args.gender, jp)
    after = measure(hs, morph, "after")
    print(f"\noutside {before['outside_pct']:.1f}% -> {after['outside_pct']:.1f}%   "
          f"median {before['median']:+.2f} -> {after['median']:+.2f}   "
          f"p95 {before['p95']:.2f} -> {after['p95']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
