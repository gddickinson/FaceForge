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
    REGIONS, REGION_NAMES, RegionTransforms, anchors,
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

#: Bounds on what a region may do to itself.  A skeleton that may shrink
#: without limit fits any surface by vanishing.
SCALE_RANGE = (0.70, 1.15)
ROT_RANGE = 0.35                       # radians, ~20 degrees

#: How hard a region is held to its own shape.  Chosen so that a uniform 10%
#: shrink costs about as much as leaving one unit of mean protrusion.
SCALE_PENALTY = 30.0
ROT_PENALTY = 4.0

#: Points sampled per region for the solve.  The final report uses the whole
#: cloud; the search does not need it, and the cost is linear.
SOLVE_POINTS = 300

SWEEPS = 4
PASSES = 3

#: Regions allowed to move bodily as well as deform.  The trunk hangs from
#: nothing; the skull is a group of its own rather than a bone on a cervical
#: pivot, so moving it opens no articulation.
OFFSET_REGIONS = ("trunk", "head")


def rodrigues(v: NDArray) -> NDArray:
    """Rotation matrix from a rotation vector."""
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(v, dtype=np.float64) / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def matrix_of(params: NDArray) -> NDArray:
    """``(sx, sy, sz, rx, ry, rz)`` -> the region's 3x3."""
    return rodrigues(params[3:6]) @ np.diag(params[0:3])


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
        self.subtree = subtree_map()
        rng = np.random.default_rng(0)
        self.points: dict[str, NDArray] = {}
        for name in REGION_NAMES:
            p = pts[regions == name]
            if len(p) > SOLVE_POINTS:
                p = p[rng.choice(len(p), SOLVE_POINTS, replace=False)]
            self.points[name] = p
        self.params: dict[str, NDArray] = {
            n: np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) for n in REGION_NAMES}
        self.offsets: dict[str, NDArray] = {
            n: np.zeros(3) for n in OFFSET_REGIONS}

    # -- objective -----------------------------------------------------------

    def table(self) -> dict[str, dict]:
        out = {n: {"matrix": matrix_of(self.params[n]).tolist()}
               for n in REGION_NAMES}
        for n in OFFSET_REGIONS:
            out[n]["offset"] = self.offsets[n].tolist()
        return out

    def cost(self, region: str) -> float:
        """Squared protrusion of everything ``region`` carries, plus its penalty."""
        t = RegionTransforms(self.table(), self.anchors, 1.0)
        chunks = []
        for name in self.subtree[region]:
            p = self.points[name]
            if len(p):
                chunks.append(t.apply(name, p))
        if not chunks:
            return 0.0
        d = self.depth(np.vstack(chunks))
        out = np.maximum(d + MARGIN, 0.0)
        q = self.params[region]
        penalty = (SCALE_PENALTY * float(np.sum((q[0:3] - 1.0) ** 2))
                   + ROT_PENALTY * float(np.sum(q[3:6] ** 2)))
        return float(np.mean(out ** 2)) + penalty

    # -- search --------------------------------------------------------------

    def solve_region(self, region: str, verbose: bool = True) -> float:
        base = self.cost(region)
        start = base
        steps = [0.06, 0.03, 0.015, 0.0075]
        rot_steps = [0.10, 0.05, 0.025, 0.012]
        for sweep in range(SWEEPS):
            for axis in range(6):
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
                    c = self.cost(region)
                    if c < best_cost:
                        best, best_cost = trial, c
                self.params[region][axis] = best
                base = best_cost
            if region in OFFSET_REGIONS:
                base = self._sweep_offset(region, base, steps[sweep] * 30.0)
        if verbose:
            q = self.params[region]
            print(f"  {region:<12} cost {start:8.3f} -> {base:8.3f}   "
                  f"scale {q[0]:.3f},{q[1]:.3f},{q[2]:.3f}  "
                  f"rot {np.degrees(q[3]):+5.1f},{np.degrees(q[4]):+5.1f},"
                  f"{np.degrees(q[5]):+5.1f}")
        return base

    def _sweep_offset(self, region: str, base: float, step: float) -> float:
        offset = self.offsets[region]
        for axis in range(3):
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
        for p in range(PASSES):
            print(f"pass {p + 1}/{PASSES}")
            for rd in REGIONS:
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
    payload = {"version": 1,
               "_comment": "Per-region 3x3 matrices putting the skeleton "
                           "inside the body-surface mesh; solved by "
                           "tools/fit_skeleton_to_skin.py.",
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
