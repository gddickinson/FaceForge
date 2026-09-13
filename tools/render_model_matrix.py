"""Render the model across every layer, both sexes, with the fit off and on.

The point is not any single frame but the matrix: a change that looks right on
the skeleton and wrong on the organs, or right at gender 0 and wrong at
gender 1, is the kind this project keeps finding by accident.  This renders
them all through the application's own path -- the same event bus, the same
controllers, the same GL renderer -- so what comes out is what a user sees.

    PYTHONPATH=src python -m tools.render_model_matrix
    PYTHONPATH=src python -m tools.render_model_matrix --views front,side

Frames go to ``results/model_matrix/<layer>_<sex>_<fit>_<view>.png``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

OUT_DIR = Path("results/model_matrix")

#: layer name -> the groups to show, and which demand layers to switch on.
LAYERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "skeleton": (("bodyMeshGroup",), ()),
    "muscles": ((), ("back_muscles", "shoulder_muscles", "arm_muscles",
                     "torso_muscles", "hip_muscles", "leg_muscles")),
    "organs": ((), ("organs",)),
    "vasculature": ((), ("vasculature",)),
    "skin": ((), ("skin",)),
}

#: Groups never wanted in these pictures.
ALWAYS_HIDE = ("faceGroup", "faceFeatureGroup", "fasciaGroup", "brainGroup")

VIEWS = {"front": 0.0, "side": 90.0, "three-quarter": 40.0}

SEXES = (("male", 0.0), ("female", 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", default="front")
    ap.add_argument("--layers", default=",".join(LAYERS))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from faceforge.appcontext import build_app_context
    from faceforge.controllers import build_controllers
    from faceforge.coordination.asset_load_sequence import AssetLoadSequence
    from faceforge.core.events import EventType
    from faceforge.core.material import RenderMode
    from faceforge.session import Session

    ctx = build_app_context(argv=[])
    build_controllers(ctx)
    AssetLoadSequence(ctx).run()
    root = getattr(ctx.scene, "root", ctx.scene)

    wanted = [n.strip() for n in args.layers.split(",") if n.strip() in LAYERS]
    for name in wanted:
        for layer in LAYERS[name][1]:
            ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer=layer,
                                  visible=True)
    print(f"loaded; {len(ctx.simulation.soft_tissue.bindings)} bound meshes")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = Session.create(width=args.width, height=args.height,
                             prefer="hardware")
    session._scene = ctx.scene
    centre = np.array([0.0, 0.0, -100.0])
    radius = 115.0
    try:
        for sex, gender in SEXES:
            ctx.event_bus.publish(EventType.GENDER_RELEASED, gender=gender)
            for fit in (False, True):
                ctx.event_bus.publish(EventType.SKELETON_FIT_TOGGLED,
                                      enabled=fit)
                for _ in range(3):
                    ctx.simulation.step(1 / 60)
                for name in wanted:
                    show, own = LAYERS[name]
                    for group in ALWAYS_HIDE:
                        node = root.find(group)
                        if node is not None:
                            node.visible = False
                    # One layer at a time: a demand-loaded layer stays in the
                    # scene once it is loaded, so the others are switched off
                    # rather than merely not switched on.
                    for other in wanted:
                        for layer in LAYERS[other][1]:
                            ctx.event_bus.publish(
                                EventType.LAYER_TOGGLED, layer=layer,
                                visible=layer in own)
                    for group in show:
                        node = root.find(group)
                        while node is not None:
                            node.visible = True
                            node = node.parent
                    ctx.scene.update()
                    if "bodyMeshGroup" in show:
                        for mesh, _m in ctx.scene.collect_meshes():
                            if mesh.name == "body_surface":
                                mesh.material.render_mode = RenderMode.WIREFRAME
                                mesh.material.wireframe_color = (0.35, .75, .95)
                    for view in args.views.split(","):
                        az = math.radians(VIEWS[view.strip()])
                        eye = centre + np.array(
                            [math.sin(az), -math.cos(az), 0.12]) * radius * 2.3
                        session.camera.look_at(np.asarray(eye),
                                               np.asarray(centre))
                        tag = f"{name}_{sex}_{'fit' if fit else 'nofit'}"
                        path = OUT_DIR / f"{tag}_{view.strip()}.png"
                        session.save_png(path)
                        print(f"  {path}  "
                              f"{session.last_content_fraction * 100:5.2f}%")
                    for group in show:
                        node = root.find(group)
                        if node is not None:
                            node.visible = False
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
