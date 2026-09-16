"""Every exercise in the catalogue, two frames each, as contact sheets.

One scene, one GL session, 100+ exercises: the point is to *look* at the whole
catalogue after editing it, which is impractical one render at a time.  Each
exercise contributes two frames -- the end of its first phase and the end of
the phase that reaches furthest from it -- drawn with the skin on and the
exercise's own camera preset, and the sheets are written to
``results/exercise_grid/``.

    python -m tools.render_exercise_grid                    # everything
    python -m tools.render_exercise_grid --category "Yoga, stretches and mobility"
    python -m tools.render_exercise_grid --exercise muscle_up,l_sit --per-sheet 4

Blank frames are refused by ``Session.render``, so a sheet that appears is
evidence the figure was actually drawn.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.exercise.muscle_groups import ALL_MUSCLE_REGIONS

logger = logging.getLogger("render_exercise_grid")

DEFAULT_OUT = Path("results/exercise_grid")
TILE = (300, 380)


def phase_times(built, defn) -> list[float]:
    """Two sample times: the first phase's end, and the most distant phase's end."""
    spans = list(built.spans[:len(defn.phases)])
    if not spans:
        return [0.0]
    first = spans[0]
    rest = spans[1:] or spans
    # "Most distant" = the phase whose pose differs most from the first one.
    def distance(span):
        a = defn.phases[0].pose
        b = defn.phases[spans.index(span)].pose
        return sum(abs(b.get(k, 0.0) - v) for k, v in a.items()) + abs(
            defn.phases[spans.index(span)].pitch - defn.phases[0].pitch) / 30.0
    far = max(rest, key=distance)
    return [first.t1 - 1e-3, far.t1 - 1e-3]


def frame_body(camera, pivots, size, margin: float = 1.32, extra=()) -> None:
    """Aim the preset camera at the body and widen its field of view to fit it.

    The scene presets are framed for a standing figure; a muscle-up is three
    metres up a bar and a supine twist is flat on the floor, so every sheet
    would otherwise be a crop of a thigh.  The body's own joint pivots give the
    box to fit -- with a margin, because the pivots stop at the skull base and
    the head is 20 units taller.  ``extra`` adds points that must also be in
    shot: the equipment, without which a bench press is a man lying in the
    air and you cannot see whether the bar is where his hands are.  The camera is NOT moved back to fit it: the
    gym has walls, and a camera pushed through one renders flat grey.
    """
    import math

    import numpy as np

    pts = np.array([p.get_world_position() for p in pivots.values()]
                   + [np.asarray(e, dtype=np.float64) for e in extra], dtype=np.float64)
    if len(pts) < 2:
        return
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    distance = float(np.linalg.norm(np.asarray(camera.position, dtype=np.float64) - centre))
    if distance < 1e-6 or radius < 1e-6:
        return
    camera.set_target(*centre)
    camera.fov = min(90.0, max(28.0, math.degrees(2.0 * math.atan(radius * margin / distance))))
    # Camera caches its projection and only `set_aspect` invalidates it, so a
    # bare `camera.fov = ...` renders at the old field of view.  (It did: every
    # figure in the first sweep was framed by the target alone.)
    camera.set_aspect(*size)


def equipment_points(demo) -> list:
    """Corners of every equipment item's own bounds, in world space."""
    import numpy as np

    pts = []
    for item in getattr(demo.runtime, "rig", None).items if demo.runtime else ():
        base = np.asarray(item.node.get_world_position(), dtype=np.float64)
        local = [np.asarray(child.position, dtype=np.float64)
                 for child in getattr(item.node, "children", ())]
        pts.append(base)
        pts.extend(base + p for p in local)
    return pts


def render_sheets(ids, out: Path, per_sheet: int, size, all_muscles: bool) -> list[Path]:
    from PIL import Image

    from tools.render_exercise_demo import DemoScene
    from faceforge.session import Session

    catalog = get_exercise_catalog()
    layers = list(ALL_MUSCLE_REGIONS) if all_muscles else ["leg_muscles"]
    demo = DemoScene(layers, with_skin=True)
    out.mkdir(parents=True, exist_ok=True)

    tiles: list[tuple[str, Image.Image]] = []
    written: list[Path] = []
    with Session.create(width=size[0], height=size[1], prefer="hardware") as session:
        session.adopt_scene(demo.scene)
        demo.activate(session.camera, session.lights, "gym")
        session.camera.set_aspect(size[0], size[1])
        for n, exercise_id in enumerate(ids, 1):
            defn = catalog[exercise_id]
            try:
                demo.start(defn, reps=1, tempo=1.0)
                demo.smc.set_camera_preset(session.camera, defn.camera,
                                           target=defn.camera_target)
                pivots = demo.hs.pipeline.joint_setup.pivots
                for i, t in enumerate(phase_times(demo.runtime.built, defn)):
                    demo.evaluate(t)
                    demo.smc.set_camera_preset(session.camera, defn.camera,
                                               target=defn.camera_target)
                    frame_body(session.camera, pivots, size, extra=equipment_points(demo))
                    image = session.render()
                    tiles.append((f"{exercise_id} [{i + 1}]",
                                  Image.fromarray(image[:, :, :3]).resize(TILE)))
                demo.runtime.stop()
            except Exception as exc:                       # keep going: report at the end
                logger.error("%s failed: %s", exercise_id, exc)
                tiles.append((f"{exercise_id} FAILED", Image.new("RGB", TILE, (90, 20, 20))))
            print(f"  {n:3d}/{len(ids)}  {exercise_id}", flush=True)
            while len(tiles) >= per_sheet:
                written.append(_write_sheet(tiles[:per_sheet], out, len(written)))
                del tiles[:per_sheet]
        if tiles:
            written.append(_write_sheet(tiles, out, len(written)))
    return written


def _write_sheet(tiles, out: Path, index: int) -> Path:
    from PIL import Image, ImageDraw

    cols = min(6, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tw, th = TILE
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (20, 22, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * (th + 22)
        sheet.paste(img, (x, y + 22))
        draw.text((x + 6, y + 5), label, fill=(232, 232, 232))
    path = out / f"grid_{index:02d}.png"
    sheet.save(path)
    print(f"wrote {path}", flush=True)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None, help="comma list; default every exercise")
    ap.add_argument("--category", default=None, help="only this Category value")
    ap.add_argument("--tag", default=None, help="only exercises carrying this tag")
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--size", default="440x560")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--all-muscles", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(name)s: %(message)s")

    catalog = get_exercise_catalog()
    if args.exercise:
        ids = [x.strip() for x in args.exercise.split(",") if x.strip()]
    else:
        ids = [k for k, d in catalog.items()
               if (args.category is None or d.category.value == args.category)
               and (args.tag is None or args.tag in d.tags)]
    unknown = [i for i in ids if i not in catalog]
    if unknown:
        print(f"unknown exercise(s): {', '.join(unknown)}")
        return 2
    if not ids:
        print("no exercises matched")
        return 2
    w, h = (int(v) for v in args.size.lower().split("x"))
    render_sheets(ids, Path(args.out), args.per_sheet, (w, h), args.all_muscles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
