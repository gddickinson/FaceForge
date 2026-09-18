"""Every exercise from several viewpoints, on a world-space measuring grid.

``render_exercise_grid`` draws each exercise once, from its own camera, which
is the view the app shows and therefore the view that hides whatever the
camera is not pointing at: a bar that misses the hands by 20 units in Z is
invisible from the front, and a knee that caves is invisible from the side.

This renders the same pose from **front, side and three-quarter** and draws a
grid derived from the camera's own view-projection on top of the pixels, so a
distance in the picture is a distance in the room:

* the floor plane (y = 0) as a 50-unit lattice, which shows contact;
* a height rule through the body's own measuring plane, ticked every 25 units;
* the midline (x = 0) and, on the side view, the frontal plane (z = 0);
* optional joint-pivot dots, projected the same way.

    python -m tools.render_exercise_views --exercise bodyweight_squat
    python -m tools.render_exercise_views --category "Lower body" --views front,side
    python -m tools.render_exercise_views --all --per-sheet 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.exercise.muscle_groups import ALL_MUSCLE_REGIONS

logger = logging.getLogger("render_exercise_views")

DEFAULT_OUT = Path("results/exercise_views")
VIEWS = ("front", "side", "three_quarter")
GRID_STEP = 50.0          # floor lattice, world units
GRID_HALF = 200.0         # lattice reaches +-200 in x and z
RULE_STEP = 25.0          # height ticks


# ── projection ───────────────────────────────────────────────────────

def projector(camera, size):
    """World point -> pixel, plus a visibility flag (in front of the camera)."""
    vp = np.asarray(camera.get_projection_matrix_for_size(*size), dtype=np.float64) @ \
        np.asarray(camera.get_view_matrix(), dtype=np.float64)
    w, h = size

    def project(p):
        clip = vp @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        return ((ndc[0] * 0.5 + 0.5) * w, (0.5 - ndc[1] * 0.5) * h)

    return project


def _seg(draw, project, a, b, fill, width=1):
    """Draw a world-space segment, subdivided so perspective stays honest."""
    n = 8
    pts = [project(a + (b - a) * (i / n)) for i in range(n + 1)]
    for p, q in zip(pts, pts[1:]):
        if p is None or q is None:
            continue
        if max(abs(p[0]), abs(p[1]), abs(q[0]), abs(q[1])) > 8000:
            continue
        draw.line([p, q], fill=fill, width=width)


def draw_grid(image, camera, size, view: str, centre, marks=()):
    """Overlay the floor lattice, the height rule and the reference planes."""
    from PIL import Image, ImageDraw

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    project = projector(camera, size)
    faint, mid, axis = (110, 130, 160, 110), (140, 170, 210, 170), (90, 200, 255, 220)

    # Floor lattice at y = 0.
    g = np.arange(-GRID_HALF, GRID_HALF + 1, GRID_STEP)
    for v in g:
        _seg(draw, project, np.array([v, 0, -GRID_HALF]), np.array([v, 0, GRID_HALF]),
             axis if abs(v) < 1e-6 else faint)
        _seg(draw, project, np.array([-GRID_HALF, 0, v]), np.array([GRID_HALF, 0, v]),
             axis if abs(v) < 1e-6 else faint)

    # The measuring plane: the vertical plane through the body that faces the
    # camera.  Front and three-quarter read heights against the body's own z;
    # the side view reads them against its own x.
    cx, cz = float(centre[0]), float(centre[2])
    if view == "side":
        base, along = np.array([cx, 0.0, -GRID_HALF]), np.array([0.0, 0.0, 1.0])
        span = 2 * GRID_HALF
    else:
        base, along = np.array([-GRID_HALF, 0.0, cz]), np.array([1.0, 0.0, 0.0])
        span = 2 * GRID_HALF

    for y in np.arange(0.0, 251.0, RULE_STEP):
        a = base + np.array([0.0, y, 0.0])
        b = a + along * span
        _seg(draw, project, a, b, mid if y % 50 == 0 else faint)
        label_at = a + along * (span * 0.5 + 120.0)
        p = project(label_at)
        if p is not None and -40 < p[0] < size[0] + 40 and -20 < p[1] < size[1] + 20:
            draw.text((min(max(p[0], 2), size[0] - 26), p[1] - 6), f"{int(y)}",
                      fill=(150, 190, 230, 200))

    for name, pos, colour in marks:
        p = project(np.asarray(pos, dtype=np.float64))
        if p is None:
            continue
        draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=colour)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


# ── framing ──────────────────────────────────────────────────────────

#: The studio, as half-extents in x/y/z (`scene_environment.STUDIO_*`), and
#: how far inside it the camera must stay.  The preset's offset is applied to
#: the BODY's centre, so a body whose pivots run a long way along one axis
#: carries the camera out with it: a prone push-up's reach to x 186 put the
#: side camera at x 316, outside the 500-wide room, rendering the back of a
#: wall.  Every prone exercise -- plank, push-up, mountain climber, bird dog,
#: the dogs -- had a flat grey side view, which is the one view that shows
#: whether the body is in a straight line.
ROOM_MARGIN = 14.0
#: (min, max) the camera must stay within, per axis.  The studio has three
#: walls -- back, left and right -- and **no front wall**, which is why a
#: camera far out on +Z sees straight in and one on +X sees the back of
#: `wall_right`.  So +Z is unbounded and the rest are the studio's own
#: `STUDIO_WIDTH` / `HEIGHT` / `DEPTH`, inset by the margin.
ROOM_BOUNDS = ((-236.0, 236.0), (14.0, 386.0), (-186.0, float("inf")))


def _inside_room(centre, offset):
    """Shorten *offset* until ``centre + offset`` is inside the studio.

    Only where there is a wall to be behind: shortening it everywhere cost the
    prone family its one useful view, pulling the open-side camera in from
    z 340 to 186 and widening the lens from 55 to 88 degrees to compensate.
    """
    t = 1.0
    for axis, (lo, hi) in enumerate(ROOM_BOUNDS):
        if abs(offset[axis]) < 1e-9:
            continue
        for bound in (lo, hi):
            if not np.isfinite(bound):
                continue
            s = (bound - centre[axis]) / offset[axis]
            if s > 0:
                t = min(t, s)
    return offset * max(t, 0.05)


def frame(camera, pts, size, margin=1.35):
    import math
    pts = np.asarray(pts, dtype=np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    eye = np.asarray(camera.position, dtype=np.float64)
    offset = eye - np.asarray(camera.target, dtype=np.float64)
    offset = _inside_room(centre, offset)
    camera.set_position(*(centre + offset))
    camera.set_target(*centre)
    distance = float(np.linalg.norm(offset))
    if distance > 1e-6 and radius > 1e-6:
        camera.fov = min(95.0, max(25.0, math.degrees(2.0 * math.atan(radius * margin / distance))))
    camera.set_aspect(*size)
    return centre


PIVOT_MARKS = {
    "shoulder_R": (255, 120, 120, 230), "shoulder_L": (255, 120, 120, 230),
    "elbow_R": (255, 190, 90, 230), "elbow_L": (255, 190, 90, 230),
    "wrist_R": (120, 255, 160, 230), "wrist_L": (120, 255, 160, 230),
    "hip_R": (200, 140, 255, 230), "hip_L": (200, 140, 255, 230),
    "knee_R": (255, 240, 120, 230), "knee_L": (255, 240, 120, 230),
    "ankle_R": (120, 200, 255, 230), "ankle_L": (120, 200, 255, 230),
}


def body_points(demo):
    pivots = demo.hs.pipeline.joint_setup.pivots
    pts = [p.get_world_position() for p in pivots.values()]
    for item in (demo.runtime.rig.items if demo.runtime else ()):
        base = np.asarray(item.node.get_world_position(), dtype=np.float64)
        pts.append(base)
        for child in getattr(item.node, "children", ()):
            pts.append(base + np.asarray(child.position, dtype=np.float64))
    return pts


def marks_for(demo):
    pivots = demo.hs.pipeline.joint_setup.pivots
    return [(n, pivots[n].get_world_position(), c)
            for n, c in PIVOT_MARKS.items() if n in pivots]


# ── the sweep ────────────────────────────────────────────────────────

def phase_times(built, defn):
    spans = list(built.spans[:len(defn.phases)])
    return [(s.name, s.t1 - 1e-3) for s in spans]


def pick_phases(times, defn, limit):
    """The first phase, the one furthest from it, and the last.

    Sampling at even *indices* skips index 1, which on a four-phase exercise is
    where the movement actually happens: a lunge's phases are Step / Lower /
    Drive / Return, and evenly-spaced indices give Step, Drive, Return -- three
    pictures of a man standing in a split stance, from which the lunge looks to
    have no depth at all.  Its "Lower" drops the hip 52 units.
    """
    if not limit or len(times) <= limit:
        return times

    def distance(i):
        a, b = defn.phases[0], defn.phases[i]
        pose_gap = sum(abs(b.pose.get(k, 0.0) - v) for k, v in a.pose.items())
        pose_gap += sum(abs(v) for k, v in b.pose.items() if k not in a.pose)
        return pose_gap + abs(b.pitch - a.pitch) / 30.0

    far = max(range(1, len(times)), key=distance)
    keep = sorted({0, far, len(times) - 1})
    # Fill any remaining slots with the phases furthest from those already kept.
    while len(keep) < limit:
        rest = [i for i in range(len(times)) if i not in keep]
        if not rest:
            break
        keep = sorted(keep + [max(rest, key=distance)])
    return [times[i] for i in keep[:limit]]


def run(ids, out: Path, size, views, per_sheet, all_muscles, dots, max_phases):
    from PIL import Image, ImageDraw

    from faceforge.session import Session
    from tools.render_exercise_demo import DemoScene

    catalog = get_exercise_catalog()
    layers = list(ALL_MUSCLE_REGIONS) if all_muscles else ["leg_muscles", "arm_muscles"]
    demo = DemoScene(layers, with_skin=True)
    out.mkdir(parents=True, exist_ok=True)

    with Session.create(width=size[0], height=size[1], prefer="hardware") as session:
        session.adopt_scene(demo.scene)
        demo.activate(session.camera, session.lights, "gym")
        for n, exercise_id in enumerate(ids, 1):
            defn = catalog[exercise_id]
            tiles = []
            try:
                demo.start(defn, reps=1, tempo=1.0)
                times = pick_phases(phase_times(demo.runtime.built, defn), defn, max_phases)
                # One framing for the whole exercise, from every phase at once.
                # Re-framing per phase makes the pictures incomparable: a calf
                # raise that lifts the body 12 units looks identical in two
                # frames the camera has silently zoomed to match.
                sweep = []
                for _, t in times:
                    demo.evaluate(t)
                    sweep.extend(body_points(demo))
                for phase_name, t in times:
                    demo.evaluate(t)
                    marks = marks_for(demo) if dots else ()
                    for view in views:
                        demo.smc.set_camera_preset(session.camera, view)
                        centre = frame(session.camera, sweep, size)
                        img = Image.fromarray(session.render()[:, :, :3])
                        img = draw_grid(img, session.camera, size, view, centre, marks)
                        tiles.append((f"{phase_name} / {view}", img))
                demo.runtime.stop()
            except Exception as exc:
                logger.error("%s failed: %s", exercise_id, exc)
                tiles.append((f"{exercise_id} FAILED: {exc}", Image.new("RGB", size, (90, 20, 20))))
            _write_sheet(exercise_id, defn, tiles, out, len(views))
            print(f"  {n:3d}/{len(ids)}  {exercise_id}", flush=True)


def _write_sheet(exercise_id, defn, tiles, out: Path, cols):
    from PIL import Image, ImageDraw

    if not tiles:
        return
    tw, th = tiles[0][1].size
    rows = (len(tiles) + cols - 1) // cols
    head = 26
    sheet = Image.new("RGB", (cols * tw, head + rows * (th + 20)), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), f"{exercise_id}  |  {defn.category.value}  |  "
                      f"orientation={defn.orientation} anchor={defn.anchor} "
                      f"cam={defn.camera}", fill=(240, 240, 240))
    for i, (label, img) in enumerate(tiles):
        x, y = (i % cols) * tw, head + (i // cols) * (th + 20)
        sheet.paste(img, (x, y + 20))
        draw.text((x + 6, y + 4), label, fill=(210, 220, 235))
    path = out / f"{exercise_id}.png"
    sheet.save(path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exercise", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--size", default="420x520")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--all-muscles", action="store_true")
    ap.add_argument("--no-dots", action="store_true")
    ap.add_argument("--max-phases", type=int, default=3)
    ap.add_argument("--per-sheet", type=int, default=0)
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
    run(ids, Path(args.out), (w, h), [v.strip() for v in args.views.split(",")],
        args.per_sheet, args.all_muscles, not args.no_dots, args.max_phases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
