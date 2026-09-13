"""Does the soft tissue still fit the body once the skeleton has been fitted?

The skeleton fit moves and reshapes bone.  Everything else -- muscle, organs,
vasculature, ligaments, the skin -- is carried by a displacement field, and
three things can go wrong with that: a mesh can tear, a mesh can come away
from the bone it belongs to, and a mesh can end up outside the body.  This
measures all three, per layer and per mesh, at either sex.

    PYTHONPATH=src python -m tools.fit_tissue_check
    PYTHONPATH=src python -m tools.fit_tissue_check --gender 1.0

For each mesh: the 99th-percentile edge stretch against its pre-fit rest
pose (tearing), how far its centroid moved *relative to the nearest bone's*
(coming away from the skeleton), and how far it protrudes through the body
surface.  Bones move; a muscle that moves with its bone has a small relative
displacement, and one that does not has a large one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

#: Every layer the demand loaders can produce, in the order a user would
#: switch them on.
LAYERS = (
    "back_muscles", "shoulder_muscles", "arm_muscles", "torso_muscles",
    "hip_muscles", "leg_muscles", "hand_muscles", "foot_muscles",
    "organs", "vasculature", "ligaments", "pelvic_floor", "brain",
    "cardiac_additional", "intestinal", "cns_additional", "oral",
)

#: Above this the mesh is torn rather than stretched.
STRETCH_LIMIT = 2.5
#: Above this the mesh has come away from the bone it belongs to.
DRIFT_LIMIT = 6.0
#: Above this the mesh stands out through the skin.
PROTRUSION_LIMIT = 4.0


def edges_of(mesh) -> NDArray | None:
    idx = mesh.geometry.indices
    if idx is None:
        return None
    tris = np.asarray(idx).reshape(-1, 3)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return e[:: max(1, len(e) // 150_000)]


def rest_of(mesh) -> NDArray:
    src = mesh.rest_positions if mesh.rest_positions is not None else \
        mesh.geometry.positions
    return np.asarray(src, dtype=np.float64).reshape(-1, 3)


def bone_centroids(root, soft: set[int]) -> tuple[list[str], NDArray]:
    """Every *bone's* centroid in body coordinates, for the drift reference.

    ``soft`` is the set of ``id(mesh)`` the skinning owns.  Without it the
    nearest "bone" to a muscle is the muscle itself, and every drift comes out
    as whatever the soft tissue happened to do -- which is the question, not
    the reference.
    """
    from faceforge.body.fit_regions import SKIP_SUBTREES

    names, pts = [], []
    stack = [(root, np.zeros(3))]
    while stack:
        node, offset = stack.pop()
        for child in node.children:
            name = getattr(child, "name", "") or ""
            if name in SKIP_SUBTREES:
                continue
            here = offset + np.asarray(child.position, dtype=np.float64)
            mesh = getattr(child, "mesh", None)
            if mesh is not None and name and id(mesh) not in soft:
                geo = mesh.geometry
                v = np.asarray(geo.positions, dtype=np.float64).reshape(-1, 3)
                names.append(name)
                pts.append(v[:geo.vertex_count].mean(axis=0) + here)
            stack.append((child, here))
    return names, (np.asarray(pts) if pts else np.zeros((0, 3)))


def layer_of(name: str, layers: dict[str, set[str]]) -> str:
    for layer, members in layers.items():
        if name in members:
            return layer
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gender", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.controllers import build_controllers
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence
    from faceforge.core.events import EventType
    from tools.skeleton_containment import SurfaceDepth, surface_of

    ctx = build_app_context(argv=[])
    controllers = build_controllers(ctx)
    AssetLoadSequence(ctx).run()

    skinning = ctx.simulation.soft_tissue
    layers: dict[str, set[str]] = {}
    for layer in LAYERS:
        before = {b.mesh.name for b in skinning.bindings}
        ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer=layer, visible=True)
        layers[layer] = {b.mesh.name for b in skinning.bindings} - before
    before = {b.mesh.name for b in skinning.bindings}
    controllers.loaders.load_skin()
    layers["skin"] = {b.mesh.name for b in skinning.bindings} - before
    print(f"loaded {len(skinning.bindings)} bound meshes across "
          f"{sum(1 for v in layers.values() if v)} layers")

    if args.gender:
        ctx.event_bus.publish(EventType.GENDER_RELEASED, gender=args.gender)
    root = ctx.node("bodyRoot")

    soft = {id(b.mesh) for b in skinning.bindings}
    rest = {id(b.mesh): rest_of(b.mesh).copy() for b in skinning.bindings}
    edges = {id(b.mesh): edges_of(b.mesh) for b in skinning.bindings}
    bones_before = bone_centroids(root, soft)

    pos, tris = surface_of(ctx.pipeline.gender_morph)
    depth = SurfaceDepth(pos, tris, np.array([0.0, 0.0, -100.0]))
    # The control: how far each mesh already stood out of the surface before
    # the fit touched anything.  Without it there is no telling what the fit
    # caused and what the two bodies disagreed about all along.
    control = {}
    for binding in skinning.bindings:
        a = rest[id(binding.mesh)]
        step = max(1, len(a) // 4000)
        control[id(binding.mesh)] = float(depth(a[::step]).max())

    ctx.event_bus.publish(EventType.SKELETON_FIT_TOGGLED, enabled=True)
    bone_names, bones_after = bone_centroids(root, soft)
    assert bone_names == bones_before[0], "the bone list changed under the fit"
    bone_shift = bones_after - bones_before[1]
    print(f"bone reference: {len(bone_names)} bones, median shift "
          f"{np.median(np.linalg.norm(bone_shift, axis=1)):.2f}")

    from scipy.spatial import cKDTree
    tree = cKDTree(bones_before[1])

    rows = []
    for binding in skinning.bindings:
        mesh = binding.mesh
        a = rest[id(mesh)]
        b = rest_of(mesh)
        if len(a) != len(b):
            print(f"  !! {mesh.name}: rest length changed {len(a)} -> {len(b)}")
            continue
        e = edges[id(mesh)]
        if e is None:
            continue
        la = np.linalg.norm(a[e[:, 0]] - a[e[:, 1]], axis=1)
        lb = np.linalg.norm(b[e[:, 0]] - b[e[:, 1]], axis=1)
        ok = la > 1e-6
        stretch = float(np.percentile(lb[ok] / la[ok], 99)) if ok.any() else 1.0
        _, near = tree.query(a.mean(axis=0))
        drift = float(np.linalg.norm(
            (b.mean(axis=0) - a.mean(axis=0)) - bone_shift[near]))
        step = max(1, len(b) // 4000)
        out = float(depth(b[::step]).max())
        rows.append((mesh.name, layer_of(mesh.name, layers), stretch, drift,
                     out, bone_names[near], control[id(mesh)]))

    print(f"\n{len(rows)} meshes measured at gender {args.gender:.1f}\n")
    print(f"{'layer':<20}{'n':>5}{'stretch p99':>13}{'drift':>9}"
          f"{'out before':>12}{'out after':>11}")
    per: dict[str, list] = defaultdict(list)
    for name, layer, st, d, o, _, c in rows:
        per[layer].append((st, d, o, c))
    for layer in sorted(per):
        v = np.asarray(per[layer])
        print(f"{layer:<20}{len(v):>5}{v[:, 0].max():>13.2f}"
              f"{v[:, 1].max():>9.2f}{v[:, 3].max():>12.2f}{v[:, 2].max():>11.2f}")

    print(f"\nworst meshes\n{'mesh':<28}{'layer':<18}{'stretch':>9}"
          f"{'drift':>8}{'before':>8}{'after':>8}  nearest bone")
    for name, layer, st, d, o, bone, c in sorted(
            rows, key=lambda r: -(r[2] + r[3] / 3 + max(r[4] - r[6], 0)))[:args.top]:
        print(f"{name:<28}{layer:<18}{st:>9.2f}{d:>8.2f}{c:>8.2f}{o:>8.2f}  {bone}")

    bad = [r for r in rows if r[2] > STRETCH_LIMIT or r[3] > DRIFT_LIMIT
           or r[4] - r[6] > PROTRUSION_LIMIT]
    print(f"\n{len(bad)} of {len(rows)} meshes past a limit "
          f"(stretch {STRETCH_LIMIT}x, drift {DRIFT_LIMIT}, or "
          f"{PROTRUSION_LIMIT} units further out than before)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
