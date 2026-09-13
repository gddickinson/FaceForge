"""Look at the skeleton inside the body surface, on a grid, and measure by eye.

A 3D render answers "does this look right"; it does not answer "by how much,
and where".  These are orthographic drawings on a labelled grid in body units,
so a bone standing out of the skin can be read off in the same numbers the
solver uses.

    PYTHONPATH=src python -m tools.inspect_skeleton_fit --tag before
    PYTHONPATH=src python -m tools.inspect_skeleton_fit --tag after --fit
    PYTHONPATH=src python -m tools.inspect_skeleton_fit --fit --slices

Two kinds of drawing, both to ``results/skeleton_fit/inspect/``:

* **silhouettes** -- front, side and closeups.  The surface is its own
  outline in pale blue; bones are drawn dark where they are inside it and red
  where they are outside, the red deepening with depth.
* **slices** -- transverse sections every few units up the body.  The
  surface's outline at that height is a closed pale curve and the bones are
  points inside it, which is the one view where "inside" is unambiguous.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = Path("results/skeleton_fit/inspect")

#: name -> (axes to plot, window (x0, x1, y0, y1) in body units, title)
#: Axes: 0 = X (right), 1 = Y (anterior is negative), 2 = Z (superior).
VIEWS: dict[str, tuple[tuple[int, int], tuple[float, float, float, float]]] = {
    "front": ((0, 2), (-60, 60, -205, 20)),
    "side": ((1, 2), (-45, 45, -205, 20)),
    "head": ((0, 2), (-25, 25, -35, 20)),
    "head-side": ((1, 2), (-30, 25, -35, 20)),
    "shoulder": ((0, 2), (0, 55, -60, 5)),
    "chest-side": ((1, 2), (-35, 20, -70, -5)),
    "hand": ((0, 2), (25, 60, -120, -75)),
    "foot-side": ((1, 2), (-35, 20, -205, -165)),
    "pelvis": ((0, 2), (-35, 35, -115, -60)),
}

#: Transverse sections, in body Z.
SLICE_LEVELS = (5, -10, -25, -40, -55, -70, -85, -100, -120, -145, -170, -190)

GRID = 10.0
PIXELS_PER_UNIT = 9.0
MARGIN = 46
BG = (250, 250, 248)
GRID_RGB = (214, 214, 208)
AXIS_RGB = (150, 150, 142)
SURFACE_RGB = (90, 155, 200)


class Canvas:
    """An orthographic drawing surface in body units, with a labelled grid."""

    def __init__(self, window, axes, scale=PIXELS_PER_UNIT):
        self.x0, self.x1, self.y0, self.y1 = window
        self.axes = axes
        self.scale = scale
        self.w = int((self.x1 - self.x0) * scale) + 2 * MARGIN
        self.h = int((self.y1 - self.y0) * scale) + 2 * MARGIN
        self.img = Image.new("RGB", (self.w, self.h), BG)
        self.draw = ImageDraw.Draw(self.img)

    def to_px(self, pts: np.ndarray) -> np.ndarray:
        a, b = self.axes
        u = (pts[:, a] - self.x0) * self.scale + MARGIN
        v = self.h - MARGIN - (pts[:, b] - self.y0) * self.scale
        return np.stack([u, v], axis=1)

    def grid(self, label: str) -> None:
        d = self.draw
        start = np.ceil(self.x0 / GRID) * GRID
        for x in np.arange(start, self.x1 + 1e-6, GRID):
            u = (x - self.x0) * self.scale + MARGIN
            d.line([(u, MARGIN), (u, self.h - MARGIN)],
                   fill=AXIS_RGB if abs(x) < 1e-6 else GRID_RGB)
            d.text((u + 2, self.h - MARGIN + 4), f"{x:.0f}", fill=AXIS_RGB)
        start = np.ceil(self.y0 / GRID) * GRID
        for y in np.arange(start, self.y1 + 1e-6, GRID):
            v = self.h - MARGIN - (y - self.y0) * self.scale
            d.line([(MARGIN, v), (self.w - MARGIN, v)],
                   fill=AXIS_RGB if abs(y) < 1e-6 else GRID_RGB)
            d.text((6, v - 6), f"{y:.0f}", fill=AXIS_RGB)
        d.rectangle([MARGIN, MARGIN, self.w - MARGIN, self.h - MARGIN],
                    outline=AXIS_RGB)
        d.text((MARGIN, 12), label, fill=(40, 40, 40))

    def points(self, pts: np.ndarray, colours, size: int = 1) -> None:
        if not len(pts):
            return
        px = self.to_px(pts)
        inside = ((px[:, 0] >= MARGIN) & (px[:, 0] <= self.w - MARGIN)
                  & (px[:, 1] >= MARGIN) & (px[:, 1] <= self.h - MARGIN))
        px = px[inside]
        cols = colours[inside] if isinstance(colours, np.ndarray) else None
        pix = self.img.load()
        for i, (u, v) in enumerate(px.astype(int)):
            rgb = tuple(int(c) for c in cols[i]) if cols is not None else colours
            for du in range(-size, size + 1):
                for dv in range(-size, size + 1):
                    uu, vv = u + du, v + dv
                    if 0 <= uu < self.w and 0 <= vv < self.h:
                        pix[uu, vv] = rgb

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.img.save(path)
        print(f"  {path}")


def depth_colours(depth: np.ndarray, limit: float = 6.0) -> np.ndarray:
    """Dark grey inside, through orange, to red at ``limit`` units outside."""
    t = np.clip(depth / limit, 0.0, 1.0)[:, None]
    inside = np.array([[70, 70, 78]])
    out = np.array([[220, 30, 20]])
    mid = np.array([[235, 150, 40]])
    near = inside + np.clip(depth, -limit, 0.0)[:, None] / -limit * 0.0
    warm = np.where(t < 0.5, mid + (1 - 2 * t) * (inside - mid),
                    out + (2 - 2 * t) * (mid - out))
    return np.where(depth[:, None] > 0, warm, near).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="now")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--gender", type=float, default=0.0)
    ap.add_argument("--views", default="front,side,head,head-side,shoulder,"
                                       "chest-side,hand,foot-side,pelvis")
    ap.add_argument("--slices", action="store_true",
                    help="transverse sections instead of silhouettes")
    args = ap.parse_args(argv)

    from faceforge.body.skeleton_fit import SkeletonFit
    from tools.headless_loader import load_headless_scene
    from tools.skeleton_containment import (
        SurfaceDepth, bone_points, surface_of,
    )

    hs = load_headless_scene()
    morph = hs.pipeline.gender_morph
    root = hs.named_nodes["bodyRoot"]
    joints = getattr(hs.pipeline.joint_setup, "joint_positions", {}) or {}
    if args.gender:
        morph.set_gender(args.gender)
        morph.scale_skeleton(root, joints)
    if args.fit:
        SkeletonFit().apply(root, 1.0, args.gender, joints)

    surface, tris = surface_of(morph)
    depth_of = SurfaceDepth(surface, tris, np.array([0.0, 0.0, -100.0]))
    pts, _regions, _names = bone_points(root, per_bone=900)
    depth = depth_of(pts)
    colours = depth_colours(depth)
    print(f"{len(pts)} bone points, {100 * (depth > 0).mean():.1f}% outside, "
          f"worst {depth.max():.2f}")

    if args.slices:
        for level in SLICE_LEVELS:
            band = np.abs(surface[:, 2] - level) < 1.2
            bone = np.abs(pts[:, 2] - level) < 1.2
            if band.sum() < 12:
                continue
            c = Canvas((-60, 60, -45, 45), (0, 1), scale=7.0)
            c.grid(f"transverse section z = {level}   "
                   f"(x right, y posterior)   {args.tag}")
            c.points(np.stack([surface[band][:, 0], surface[band][:, 1],
                               surface[band][:, 1]], axis=1),
                     SURFACE_RGB, size=1)
            c.points(np.stack([pts[bone][:, 0], pts[bone][:, 1],
                               pts[bone][:, 1]], axis=1),
                     colours[bone], size=0)
            c.save(OUT_DIR / f"{args.tag}_slice_z{level}.png")
        return 0

    for name in args.views.split(","):
        name = name.strip()
        axes, window = VIEWS[name]
        c = Canvas(window, axes)
        c.grid(f"{name}   {args.tag}"
               + (f"   gender {args.gender:.1f}" if args.gender else ""))
        c.points(surface, SURFACE_RGB, size=0)
        c.points(pts, colours, size=0)
        c.save(OUT_DIR / f"{args.tag}_{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
