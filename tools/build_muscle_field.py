"""Build the muscle distance field the skin binding reads.

Loads every muscle layer once, samples their vertices, tags each sample with
the body part its muscle belongs to, and writes the result to
``assets/config/muscle_field.npz``.  See
:mod:`faceforge.body.muscle_field` for why the skin binding wants it.

    PYTHONPATH=src python -m tools.build_muscle_field
    PYTHONPATH=src python -m tools.build_muscle_field --stride 5 --out other.npz

The muscle layers are loaded on demand in a session, and the skin binding does
not wait for them, so this is run once and the result is read from disk.

The default stride is coarse on purpose.  The field answers "which body
part's flesh is nearest", and muscles are large, so it does not need a dense
sample: measured over the four gate poses, stride 7 (9.6 MB, 1.13 M points)
and stride 60 (1.2 MB, 158 k points) give the same answer to within noise --
56,279 torn edges against 55,971.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Muscle layer -> body part, before the side suffix is applied.
REGION_GROUP = {
    "back_muscles": "trunk",
    "torso_muscles": "trunk",
    "shoulder_muscles": "trunk",
    "hip_muscles": "trunk",
    "arm_muscles": "arm",
    "leg_muscles": "leg",
    "hand_muscles": "hand",
    "foot_muscles": "foot",
}

#: Muscles whose body part is not their layer's.  The shoulder layer holds
#: both kinds: the glenohumeral muscles wrap the humerus and the head of the
#: arm, and the skin over them follows the arm, while serratus anterior and
#: subclavius lie on the ribs and do not.  Labelling the whole layer trunk
#: stopped the arm chain claiming the skin over its own deltoid; measured,
#: 3,131 of the 3,184 vertices in the triangles that blow up at shoulder
#: height sit on flesh this file called trunk, half-driven by the arm.
#: EMPTY: measured and rejected.  Moving the glenohumeral muscles to the arm
#: group improves the spike count at shoulder height (110 to 98) and costs
#: everything else -- that pose's torn edges 4,958 to 7,901 and its worst edge
#: 59.69 to 184.79 -- while the rendering does not move at all, 135 pixels of
#: 95,038.  Skin over the deltoid does follow the arm, but the chain machinery
#: already arranges that; what this changed was which skin the arm may SEED,
#: and it let the arm back onto the shoulder's trunk side.
MUSCLE_GROUP_OVERRIDES: dict[str, str] = {}


def _side(name: str) -> str | None:
    if name.endswith(" R") or name.startswith("R "):
        return "R"
    if name.endswith(" L") or name.startswith("L "):
        return "L"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output .npz")
    ap.add_argument("--stride", type=int, default=60,
                    help="keep every Nth muscle vertex (default 60)")
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.body.muscle_field import GROUPS, MuscleChainField
    from faceforge.constants import CONFIG_DIR
    from faceforge.controllers import build_controllers
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence

    out = Path(args.out) if args.out else Path(CONFIG_DIR) / "muscle_field.npz"

    ctx = build_app_context(argv=[])
    controllers = build_controllers(ctx)
    AssetLoadSequence(ctx).run()
    soft = ctx.simulation.soft_tissue

    owner: dict[int, str] = {}
    for layer, group in REGION_GROUP.items():
        before = {id(b) for b in soft.bindings}
        if layer in ("hand_muscles", "foot_muscles"):
            loader = getattr(controllers.loaders, f"load_{layer}", None)
            if loader is None:
                continue
            loader()
        else:
            controllers.loaders.load_body_muscle_region(layer, f"{layer}.json")
        for b in soft.bindings:
            if id(b) not in before:
                owner[id(b)] = group

    points: dict[str, list[np.ndarray]] = {g: [] for g in GROUPS}
    counted = 0
    for b in soft.bindings:
        group = owner.get(id(b))
        if group is None or b.mesh.rest_positions is None:
            continue
        name = b.muscle_name or b.mesh.name
        side = _side(name)
        stem = name[:-2] if name.endswith((" R", " L")) else name
        group = MUSCLE_GROUP_OVERRIDES.get(stem, group)
        key = group if group == "trunk" else f"{group}_{side or 'R'}"
        if key not in points:
            continue
        v = np.asarray(b.mesh.rest_positions,
                       dtype=np.float64).reshape(-1, 3)[::max(1, args.stride)]
        points[key].append(v)
        counted += 1

    packed = {g: (np.vstack(v) if v else np.zeros((0, 3))) for g, v in points.items()}
    field = MuscleChainField({g: v for g, v in packed.items() if len(v)})
    out.parent.mkdir(parents=True, exist_ok=True)
    field.save(out)
    print(f"{counted} muscles sampled (stride {args.stride}); wrote {out}")
    for g in field.groups:
        print(f"  {g:8s} {len(field.points[g]):8d} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
