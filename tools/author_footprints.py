"""Author attachment footprints for ``assets/config/muscle_footprints.json``.

Two ways to make a footprint, both written so the result can be MEASURED
(``tools/deformation_quality.py``, or the per-muscle stretch report in
``docs/exercise_animation.md``) before it is kept:

``mirror``
    Reflect an authored right-side footprint onto the left-side muscle.  The
    two sides are different STLs, so indices do not correspond; each right
    footprint vertex is mirrored (x -> -x) and matched to the nearest left
    vertex.  Only matches within ``--tolerance`` units are kept, and the
    mapping is reported, so a poorly mirrored asset fails loudly.

``seed``
    Seed a footprint from BONE PROXIMITY, for muscles whose attachments are
    unambiguous: every muscle vertex within ``--radius`` of any vertex of the
    named origin bones becomes origin, likewise for the insertion bones.
    The repo's own record is that proximity misfires for muscles that WRAP
    the humerus (contact is not attachment), so this is for muscles such as
    latissimus dorsi whose origin is a long aponeurosis on static bone and
    whose insertion is a small tendon.  The footprint's mid-belly grading is
    the same geodesic interpolation the app already applies.

Usage::

    python -m tools.author_footprints mirror "Deltoid Acr. R" "Deltoid Clav. R" "Pect. Major Clav. R"
    python -m tools.author_footprints seed "Supraspinatus R" "Teres Major R" --dry-run
    python -m tools.author_footprints seed "Latissimus Dorsi R" --origin-radius 6 --insertion-radius 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from scipy.spatial import cKDTree

from faceforge.constants import CONFIG_DIR

FOOTPRINTS = CONFIG_DIR / "muscle_footprints.json"
logger = logging.getLogger("author_footprints")


def _load():
    with open(FOOTPRINTS) as fh:
        return json.load(fh)


def _save(data: dict) -> None:
    FOOTPRINTS.write_text(json.dumps(data, indent=1) + "\n")


def _scene(layers):
    from tools.headless_loader import load_headless_scene, load_layer

    hs = load_headless_scene()
    meshes = {}
    for layer in layers:
        for mesh in load_layer(hs, layer):
            meshes[mesh.name] = mesh
    return hs, meshes


def _rest(mesh) -> np.ndarray:
    return np.asarray(mesh.rest_positions, dtype=np.float64).reshape(-1, 3)


def _left_name(right: str) -> str:
    if right.endswith(" R"):
        return right[:-2] + " L"
    if right.startswith("R "):
        return "L " + right[2:]
    raise ValueError(f"{right!r} is not a right-side muscle name")


def mirror(names, layers, tolerance: float, dry_run: bool) -> int:
    data = _load()
    hs, meshes = _scene(layers)
    for right in names:
        fp = data.get(right)
        if fp is None:
            print(f"no authored footprint for {right!r}")
            return 1
        left = _left_name(right)
        if right not in meshes or left not in meshes:
            print(f"load the layers containing {right!r} and {left!r} (--layers)")
            return 1
        r_rest, l_rest = _rest(meshes[right]), _rest(meshes[left])
        tree = cKDTree(l_rest)
        out = {}
        for key in ("origin_indices", "insertion_indices"):
            idx = np.asarray(fp[key], dtype=np.int64)
            idx = idx[idx < len(r_rest)]
            mirrored = r_rest[idx] * np.array([-1.0, 1.0, 1.0])
            dist, match = tree.query(mirrored)
            keep = dist <= tolerance
            out[key] = sorted(set(int(m) for m in match[keep]))
            print(f"{right} -> {left} {key}: {int(keep.sum())}/{len(idx)} matched within "
                  f"{tolerance} (median {np.median(dist):.2f}, max {dist.max():.2f})")
        if not dry_run:
            data[left] = out
    if not dry_run:
        _save(data)
        print(f"wrote {FOOTPRINTS}")
    return 0


def _config_bones(name: str) -> tuple[list[str], list[str]]:
    """originBones / insertionBones for a muscle, from the muscle configs."""
    from faceforge.coordination.demand_loaders import MUSCLE_REGIONS
    from faceforge.core.config_loader import load_muscle_config

    for config_name in MUSCLE_REGIONS.values():
        for entry in load_muscle_config(config_name):
            if entry.get("name") == name:
                return list(entry.get("originBones", [])), list(entry.get("insertionBones", []))
    raise KeyError(f"{name!r} is in no muscle config")


def seed(names, layers, origin_radius: float, insertion_radius: float, dry_run: bool,
         origin_bones=None, insertion_bones=None) -> int:
    """Seed footprints for ``names`` from bone proximity; bones from the config unless given."""
    data = _load()
    hs, meshes = _scene(layers)
    registry = hs.pipeline.bone_anchors
    bone_nodes = getattr(registry, "_bone_nodes", {})

    def bone_points(bones) -> np.ndarray:
        pts = []
        for b in bones:
            node = bone_nodes.get(b)
            if node is None or node.mesh is None:
                print(f"  bone {b!r} not in the registry")
                continue
            geom = node.mesh.geometry
            local = np.asarray(node.mesh.rest_positions if node.mesh.rest_positions is not None
                               else geom.positions, dtype=np.float64).reshape(-1, 3)
            node.update_world_matrix(force=True)
            w = (node.world_matrix @ np.c_[local, np.ones(len(local))].T).T[:, :3]
            pts.append(w)
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    for name in names:
        if name not in meshes:
            print(f"{name!r} not loaded; pass its layer with --layers")
            return 1
        o_bones, i_bones = (origin_bones, insertion_bones) if origin_bones else _config_bones(name)
        rest = _rest(meshes[name])
        out = {}
        for key, bones, radius in (("origin_indices", o_bones, origin_radius),
                                   ("insertion_indices", i_bones, insertion_radius)):
            pts = bone_points(bones)
            if not len(pts):
                print(f"{name}: no bone points for {key}")
                return 1
            dist, _ = cKDTree(pts).query(rest)
            hit = np.where(dist <= radius)[0]
            out[key] = [int(i) for i in hit]
            print(f"{name} {key}: {len(hit)}/{len(rest)} vertices within {radius} of {bones}")
        both = set(out["origin_indices"]) & set(out["insertion_indices"])
        if both:
            # A vertex near both bones is graded by the geodesic interpolation
            # anyway; keep it in neither seed so the ends stay distinct.
            out = {k: [i for i in v if i not in both] for k, v in out.items()}
            print(f"  {len(both)} vertices near both ends dropped from the seeds")
        if not out["origin_indices"] or not out["insertion_indices"]:
            print(f"{name}: a footprint needs both ends; widen the radii or check the bones")
            return 1
        if not dry_run:
            data[name] = out
    if not dry_run:
        _save(data)
        print(f"wrote {FOOTPRINTS}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mirror")
    m.add_argument("names", nargs="+")
    m.add_argument("--layers", default="shoulder_muscles,torso_muscles,back_muscles")
    m.add_argument("--tolerance", type=float, default=3.0)
    m.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("seed")
    s.add_argument("names", nargs="+")
    s.add_argument("--origin", nargs="*", default=None, help="override the config's origin bones")
    s.add_argument("--insertion", nargs="*", default=None)
    s.add_argument("--layers", default="shoulder_muscles,torso_muscles,back_muscles")
    s.add_argument("--origin-radius", type=float, default=4.0)
    s.add_argument("--insertion-radius", type=float, default=4.0)
    s.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    if args.cmd == "mirror":
        return mirror(args.names, args.layers.split(","), args.tolerance, args.dry_run)
    return seed(args.names, args.layers.split(","), args.origin_radius, args.insertion_radius,
                args.dry_run, origin_bones=args.origin or None,
                insertion_bones=args.insertion or None)


if __name__ == "__main__":
    sys.exit(main())
