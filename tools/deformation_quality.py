"""Measure deformation quality: containment, seam tearing, and bulk distortion.

Why this exists
---------------
The suite had 1813 passing tests and none of them measured deformation quality.
That is how a muscle distorting to 501% of its own rest span coexisted with a
green suite, and how cross-region motion -- raising an arm dragging trunk and leg
geometry -- went undetected until a human noticed it on screen.

The three metrics are deliberately independent, so a change that improves one by
wrecking another cannot hide:

CONTAINMENT (hard invariant, must be exactly zero)
    If none of a vertex's own driving joints moved, that vertex must not move.
    Needs no per-region configuration -- each vertex is checked against the
    joints that actually drive it -- so it cannot misclassify which mesh belongs
    to which body part. Any non-zero value is a bug, not a quality shortfall.

SEAM DISTORTION
    Stretch of edges whose endpoints have different primary joints. 94.5% of
    extreme-distortion edges are in this population, and the hull bound cannot
    reach them: each endpoint is legitimately inside its own hull and the two
    bones genuinely separate. This is the weights metric.

BULK DISTORTION
    Stretch of all edges, against the CAPTURED rest reference rather than
    mesh.rest_positions. The raw BodyParts3D surfaces self-intersect, so the
    engine's resolved neutral differs from the asset by up to 2.62 units;
    measuring against the asset corrupted every comparison until that was found.

Controls, because six measurements in this project were aimed at something
static and returned meaningless numbers:

  * the reference control asserts the rest pose reproduces exactly;
  * the variance control asserts the pose actually moved some joints;
  * both are hard failures, since a metric measured on a static mesh is not a
    weak result but a meaningless one.

Usage
-----
    python -m tools.deformation_quality [--pose FIELD=VALUE ...] [--json PATH]
                                        [--layers L,L,...]

Exit status is 0 when every threshold in ``THRESHOLDS`` holds, 1 otherwise, so
this is usable as a gate as well as a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: Gate thresholds: a RATCHET on the shipped configuration, not a statement
#: that the shipped configuration is good.
#:
#: The first version of this table was inconsistent -- seam thresholds taken
#: from a multi-influence run and the bulk threshold from a two-influence run,
#: so no single configuration could satisfy all four. Thresholds have to come
#: from ONE measurement of the tree they gate.
#:
#: Measured on the shipped configuration (hull bound on, multi-influence
#: muscles off, diffusion off), back and arm muscles, shoulder_r_flex=1.0:
#:
#:   containment 0.000   bulk p99 0.1028   seam p99 44.719   seam max 664.99
#:
#: containment is a hard invariant with no tolerance. The other three carry
#: headroom for pose and load-order variation.
#:
#: The seam numbers are BAD and are the open defect, not an accepted state:
#: multi-influence muscles cuts seam p99 to 5.113 and seam max to 184.19, but
#: costs 43x on bulk p99 (0.1028 -> 4.4375), so it is disabled pending a visual
#: comparison. Tighten these two the moment that trade is resolved.
THRESHOLDS = {
    "containment": 0.0,
    "seam_p99": 50.0,
    "seam_max": 700.0,
    "bulk_p99": 0.15,
}

DEFAULT_LAYERS = ("back_muscles", "arm_muscles")
DEFAULT_POSE = {"shoulder_r_flex": 1.0}


def _joint_transforms(soft) -> np.ndarray:
    for j in soft.joints:
        j.node.update_world_matrix(force=True)
    return np.array([np.asarray(j.node.world_matrix, dtype=np.float64)
                     for j in soft.joints])


def _driving_mask(soft, binding, moved, n) -> np.ndarray:
    """True where at least one joint driving this vertex has moved."""
    ji = np.asarray(binding.joint_indices)[:n]
    si = np.asarray(binding.secondary_indices)[:n]
    driven = moved[ji] | moved[si]
    inf = getattr(binding, "influences", None)
    if inf is not None and binding.influence_weights is not None:
        w = np.asarray(binding.influence_weights)[:n]
        # Only slots with non-zero weight actually drive the vertex; a
        # zero-weight slot holds a valid joint index that is multiplied by 0.
        driven = driven | (moved[np.asarray(inf)[:n]] & (w > 0.0)).any(axis=1)
    return driven


def measure(layers=DEFAULT_LAYERS, pose=None) -> dict:
    """Load the body, pose it, and return the three metrics plus controls."""
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence
    from faceforge.coordination.demand_loaders import DemandLoaders
    from faceforge.core.state import BodyState

    pose = dict(pose or DEFAULT_POSE)
    ctx = build_app_context(argv=[])
    AssetLoadSequence(ctx).run()
    soft = ctx.simulation.soft_tissue
    anim = (getattr(ctx.simulation, "body_animation", None)
            or ctx.simulation.body_anim)
    loaders = DemandLoaders(ctx)
    for layer in layers:
        if layer == "skin":
            loaders.load_skin()
        else:
            loaders.load_body_muscle_region(layer, f"{layer}.json")

    # Neutral frame first: it both captures the rest reference and gives the
    # reference control something to check.
    anim.apply(BodyState(), 1 / 60)
    soft._last_signature = None
    soft.update(BodyState())

    bindings = [b for b in soft.bindings
                if b.edge_pairs is not None and b.mesh.rest_positions is not None]
    if not bindings:
        raise SystemExit("no bindings with edges loaded -- nothing to measure")

    rest_dev = 0.0
    for b in bindings:
        ref = soft._resolved_reference(b)
        n = len(ref)
        pos = np.asarray(b.mesh.geometry.positions,
                         dtype=np.float64).reshape(-1, 3)[:n]
        rest_dev = max(rest_dev, float(np.abs(pos - ref).max()))

    before = _joint_transforms(soft)
    state = BodyState(**pose)
    anim.apply(state, 1 / 60)
    soft._last_signature = None
    soft.update(state)
    after = _joint_transforms(soft)
    moved = np.linalg.norm((after - before).reshape(len(before), -1), axis=1) > 1e-6

    seam, bulk = [], []
    containment = 0.0
    worst_binding = ""
    for b in bindings:
        ref = soft._resolved_reference(b)
        n = len(ref)
        e = np.asarray(b.edge_pairs).reshape(-1, 2)
        e = e[(e[:, 0] < n) & (e[:, 1] < n)]
        if len(e) == 0:
            continue
        pos = np.asarray(b.mesh.geometry.positions,
                         dtype=np.float64).reshape(-1, 3)[:n]
        rl = np.linalg.norm(ref[e[:, 0]] - ref[e[:, 1]], axis=1)
        cl = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)
        d = np.abs(cl / np.maximum(rl, 1e-9) - 1.0)
        bulk.append(d)
        ji = np.asarray(b.joint_indices)[:n]
        is_seam = ji[e[:, 0]] != ji[e[:, 1]]
        if is_seam.any():
            seam.append(d[is_seam])
        static = ~_driving_mask(soft, b, moved, n)
        if static.any():
            drift = float(np.linalg.norm(pos[static] - ref[static], axis=1).max())
            if drift > containment:
                containment = drift
                worst_binding = b.muscle_name or b.mesh.name

    bulk_all = np.concatenate(bulk)
    seam_all = np.concatenate(seam) if seam else np.zeros(1)
    return {
        "pose": pose,
        "layers": list(layers),
        "bindings": len(bindings),
        "moved_joints": int(moved.sum()),
        "rest_deviation": rest_dev,
        "containment": containment,
        "containment_worst": worst_binding,
        "seam_p99": float(np.percentile(seam_all, 99)),
        "seam_max": float(seam_all.max()),
        "seam_edges": int(sum(len(s) for s in seam)),
        "bulk_p99": float(np.percentile(bulk_all, 99)),
        "bulk_median": float(np.median(bulk_all)),
        "edges": int(len(bulk_all)),
    }


def check(m: dict) -> list[str]:
    """Return a list of failure descriptions; empty means every gate held."""
    bad = []
    # Controls first: a metric measured on a mesh that did not move, or against
    # a reference the engine does not reproduce, is meaningless rather than bad.
    if m["moved_joints"] == 0:
        bad.append(f"CONTROL: pose {m['pose']} moved no joints -- "
                   "nothing was measured")
    if m["rest_deviation"] > 1e-6:
        bad.append(f"CONTROL: rest pose not reproduced "
                   f"(max |pos - ref| = {m['rest_deviation']:.6g})")
    if m["containment"] > THRESHOLDS["containment"]:
        bad.append(f"containment {m['containment']:.3f} > "
                   f"{THRESHOLDS['containment']} on {m['containment_worst']} -- "
                   "a vertex moved although none of its own joints did")
    for key in ("seam_p99", "seam_max", "bulk_p99"):
        if m[key] > THRESHOLDS[key]:
            bad.append(f"{key} {m[key]:.3f} > {THRESHOLDS[key]}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pose", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="body state field to set, repeatable "
                         "(default shoulder_r_flex=1.0)")
    ap.add_argument("--layers", default=",".join(DEFAULT_LAYERS),
                    help="comma-separated demand layers, or 'skin'")
    ap.add_argument("--json", type=Path, help="write the measurement here")
    args = ap.parse_args(argv)

    pose = {}
    for item in args.pose:
        field, _, value = item.partition("=")
        pose[field] = float(value)

    m = measure(tuple(x for x in args.layers.split(",") if x), pose or None)
    print(f"bindings {m['bindings']}  edges {m['edges']:,}  "
          f"seam edges {m['seam_edges']:,}  moved joints {m['moved_joints']}")
    print(f"  rest deviation  {m['rest_deviation']:.6g}   (control, must be 0)")
    print(f"  containment     {m['containment']:.3f}"
          + (f"   worst: {m['containment_worst']}" if m["containment"] else ""))
    print(f"  seam p99        {m['seam_p99']:.3f}   max {m['seam_max']:.2f}")
    print(f"  bulk p99        {m['bulk_p99']:.4f}   median {m['bulk_median']:.5f}")

    if args.json:
        args.json.write_text(json.dumps(m, indent=1))

    failures = check(m)
    for f in failures:
        print(f"FAIL: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
