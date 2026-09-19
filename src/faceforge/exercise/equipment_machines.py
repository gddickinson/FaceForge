"""The selectorised leg machines, and the pad that rides the shins.

`equipment.py` had grown past this project's 500-line limit, so the machines
that are frames rather than implements live here.  It imports the shared
primitives from `equipment` and registers itself back into
`EQUIPMENT_BUILDERS` at the bottom of that module, which is why the import
there is the last thing it does.

Why these exist at all: `leg_extension` and `lying_leg_curl` are both named
"(machine)" in the catalogue and both shipped a plain `bench` as their only
equipment, so the render was a man sitting on a box kicking the air.  No
geometric audit could see it -- they all measure what IS in the scene --
which is what `tools/audit_equipment_promised.py` was written to catch.

Sizes are measured against the body the catalogue poses, not chosen:

* seated, the hip sits at y 68 and the knee at (64.4, z 60), so the seat top
  is 58 and its front edge is at z 60 -- the knee hangs just off it;
* the extended ankle reaches (59.0, z 106.5) and the lowered one (17.6,
  z 58.7), so the pad has to travel that arc.  It is an ``attach="ankles"``
  item for exactly that reason: drawn on the frame it would sit at one point
  of the arc while the shin swung away from it, which is the same mistake the
  bike's pedals made before they were attached to the feet;
* prone on the curl bench the hip is at y 73.2 and the ankle swings from
  82.2 to 116.6, so that bench top is 68.
"""

from __future__ import annotations

import math

from faceforge.core.scene_graph import SceneNode
from faceforge.core.math_utils import quat_from_axis_angle
from faceforge.exercise.equipment import (
    FRAME, IRON, PAD, RUBBER, STEEL, WOOD, _axis_to, _TO_X, _TO_Z, _X, _Z, _part,
)
from faceforge.scene.procedural_geometry import (
    make_box, make_cylinder, make_plane, make_torus,
)

#: Seat top for the extension, and bench top for the curl.
EXT_SEAT_Y = 58.0
CURL_BENCH_Y = 68.0
#: Front edge of the seat, under the knee.
EXT_SEAT_FRONT_Z = 60.0


def _stack(root: SceneNode, x: float, y: float, z: float, plates: int = 6) -> None:
    """A selectorised weight stack: what makes a machine read as one."""
    for i in range(plates):
        root.add(_part(f"plate_{i}", make_box(26.0, 6.0, 20.0), IRON,
                       x=x, y=y + i * 7.5, z=z))
    root.add(_part("stack_post", make_cylinder(1.4, plates * 7.5 + 24.0, 8), STEEL,
                   x=x, y=y + plates * 3.75, z=z))


def make_leg_extension_machine() -> SceneNode:
    """Seat, backrest, side frames and a stack.  The pad is a separate item."""
    root = SceneNode("equip_leg_extension_machine")
    # The seat stops BEHIND the knee.  At full depth (front edge z 60, under
    # the knee) the lowered shank -- which hangs almost vertically at z 59 --
    # ran straight through it, 8.5 units into the shank capsule.
    root.add(_part("seat", make_box(46.0, 6.0, 50.0), PAD,
                   y=EXT_SEAT_Y - 3.0, z=EXT_SEAT_FRONT_Z - 33.0))
    # Upright behind the hip, not raked: a machine's back pad is what the
    # lifter braces against while the quadriceps pull him forward.
    root.add(_part("backrest", make_box(46.0, 52.0, 6.0), PAD,
                   y=EXT_SEAT_Y + 26.0, z=EXT_SEAT_FRONT_Z - 61.0))
    for x in (-25.0, 25.0):
        root.add(_part("side_frame", make_box(5.0, EXT_SEAT_Y, 96.0), FRAME,
                       x=x, y=EXT_SEAT_Y / 2.0, z=EXT_SEAT_FRONT_Z - 30.0))
        # No cross-lever: a 52-long cylinder at each side spans the midline,
        # which is where the shins are -- it read 7.4 units inside the left
        # shank at full extension.  The pad is the moving part and it is its
        # own item.
    # Well behind the backrest: at z -4 the post stood under the seat, where
    # the pelvis is, and read 10.6 units inside it.
    _stack(root, 0.0, 8.0, EXT_SEAT_FRONT_Z - 96.0)
    return root


#: The prone body lies along X, not Z: measured, the shoulder is at x -69.9,
#: the hip at -4.0 and the knee at 56.2, all within z 9..21.  A bench built
#: 150 long in Z was 46 long under him and the torso hung off it in mid-air.
CURL_BENCH_MID_X = -8.0
CURL_BENCH_LEN = 190.0


def make_leg_curl_machine() -> SceneNode:
    """A prone bench with side frames and a stack; the pad rides the ankles."""
    root = SceneNode("equip_leg_curl_machine")
    root.add(_part("bench", make_box(CURL_BENCH_LEN, 6.0, 46.0), PAD,
                   x=CURL_BENCH_MID_X, y=CURL_BENCH_Y - 3.0, z=12.0))
    # Legs UNDER the pad, not rails beside it: full-height rails at z -9 and
    # 33 caught the arms, which hang off the sides of a prone bench (9.0 units
    # inside the right upper arm).
    for x in (CURL_BENCH_MID_X - CURL_BENCH_LEN / 2.0 + 18.0,
              CURL_BENCH_MID_X + CURL_BENCH_LEN / 2.0 - 18.0):
        root.add(_part("leg", make_box(8.0, CURL_BENCH_Y - 6.0, 34.0), FRAME,
                       x=x, y=(CURL_BENCH_Y - 6.0) / 2.0, z=12.0))
    # Beyond the head, along the bench's own axis.
    _stack(root, CURL_BENCH_MID_X - CURL_BENCH_LEN / 2.0 - 22.0, 8.0, 12.0)
    return root


def make_shin_pad(width: float = 34.0, radius: float = 6.0) -> SceneNode:
    """The padded roller a leg machine loads the shins with."""
    # Offset ABOVE the joint it is attached to, not centred on it.  The rig
    # puts an attached item's origin at the joint, and a roller centred on the
    # ankle is inside the leg -- 8.5 units into the shank capsule, saturated.
    # Above is where the pad belongs on both machines: a leg extension drives
    # the shins UP into it and a lying curl pulls the heels up against it, so
    # in both the shin is underneath.
    #
    # LIMIT, measured: the offset is a fixed one in the item's own frame, and
    # `attach="ankles"` gives the rig the ankle AXIS but not which way the
    # shin points.  So it is right wherever the shin is horizontal -- both
    # machines' loaded positions, where the pad clears the shank by 9.6 -- and
    # wrong where the shin hangs vertically, because "above the ankle" is then
    # "along the bone".  The extension's bottom is that case, and the clash
    # audit reports the pad inside the shank on the transit through it.
    # Fixing it properly means the rig offsetting perpendicular to the shin,
    # which is a change to the attachment and not to this geometry.
    lift = radius + 11.0
    root = SceneNode("equip_shin_pad")
    root.add(_part("roller", make_cylinder(radius, width, 12), PAD,
                   y=lift, quat=_TO_X))
    for x in (-width / 2.0 - 2.0, width / 2.0 + 2.0):
        root.add(_part("cap", make_cylinder(radius * 0.5, 4.0, 8), STEEL,
                       x=x, y=lift, quat=_TO_X, shininess=80))
    return root



#: The bottom bracket, raised from 30.  At 30 the rider's toes still reached
#: 7.1 below the FLOOR at the bottom of the stroke, because a foot hangs about
#: 12 below the ball that sits on the pedal.
CRANK_Y = 36.0
#: How far the bike's saddle, bars and the rider on them all move up, so that
#: the BALL of the foot rides the crank circle rather than the ankle: the ankle
#: sits 9.3 above the ball (measured on a flat foot), so it must orbit
#: ``CRANK_Y + 9.3`` = 45.3 and it orbited 23.  Both `make_bike` and the
#: exercise's `base_position` use it.
SADDLE_RISE = 22.3


#: Where the seat post meets the frame.
_POST_FOOT = 42.5


def make_bike() -> SceneNode:
    """An upright stationary bike facing +Z: saddle, bars, crank and wheel."""
    root = SceneNode("equip_bike")
    root.add(_part("frame_down", make_box(6.0, 70.0, 6.0), FRAME, y=40.0, z=10.0,
                   quat=quat_from_axis_angle(_X, math.radians(20.0))))
    # The saddle was 16.3 too low for the crank it is bolted to.  The leg fit
    # in `conditioning._pedal_ik` is right (knee ~30 deg at the bottom), so the
    # bottom bracket has to sit 79.1 below the hip; with the saddle at 98 it sat
    # at 23 while the crank axle mesh is at 30, and the rider's toes went 24
    # units through the FLOOR at the bottom of the stroke.  Raising the saddle
    # -- which is this exercise's own first listed error -- puts the ball of
    # the foot on the pedal circle instead.
    #
    # The saddle sits so the rider's hip is 10 above its top, the seated
    # convention the bench family uses (`_helpers.SEATED_ON_BENCH`).  At 98 it
    # was level with him: measured 2026-09-17 the hip pivots sat at y 124.2
    # against a saddle top of 122.3, so the saddle was buried in his pelvis and
    # the post below it reached to 4.4 of the line between his hip joints --
    # 8.6 inside the trunk capsule, the catalogue's only hard equipment clash.
    # The post now stops at the saddle's underside rather than 1.5 inside it.
    saddle_y = 90.0 + SADDLE_RISE
    post_top = saddle_y - 2.0
    root.add(_part("seat_post", make_cylinder(2.5, post_top - _POST_FOOT, 8), FRAME,
                   y=(_POST_FOOT + post_top) / 2, z=-12.0))
    root.add(_part("saddle", make_box(14.0, 4.0, 26.0), RUBBER, y=saddle_y, z=-12.0))
    root.add(_part("head_tube", make_cylinder(2.5, 60.0 + SADDLE_RISE, 8), FRAME,
                   y=90.0 + SADDLE_RISE / 2, z=42.0))
    # Measured against the rider: the wrists sit at x +-50, y 129, z 62, so a
    # 46-wide bar at z 42 was 27 units narrower than the hands and 20 behind
    # them.  The rider held nothing.
    root.add(_part("handlebar", make_cylinder(1.8, 112.0, 10), STEEL,
                   y=128.0 + SADDLE_RISE, z=60.0, quat=_TO_X))
    root.add(_part("bar_stem", make_cylinder(2.0, 24.0, 8), FRAME,
                   y=116.0 + SADDLE_RISE, z=52.0,
                   quat=quat_from_axis_angle(_X, math.radians(-55.0))))
    root.add(_part("wheel", make_torus(24.0, 3.0, 32, 10), IRON, y=CRANK_Y, z=22.0,
                   quat=quat_from_axis_angle(_Z, math.pi / 2)))
    root.add(_part("crank_axle", make_cylinder(2.0, 24.0, 8), STEEL, y=CRANK_Y, z=0.0,
                   quat=_TO_X))
    # The pedals are NOT here: a pedal drawn on the frame stays at one point of
    # the crank circle while the foot rides round it.  They are separate items
    # attached to the feet (`make_pedal`, attach="foot_r"/"foot_l").
    root.add(_part("base", make_box(50.0, 3.0, 110.0), FRAME, y=1.5))
    return root


def make_rower() -> SceneNode:
    """A rowing ergometer along X: rail, sliding seat, footplate and handle."""
    root = SceneNode("equip_rower")
    # Measured against the athlete: the hips ride at y 52 (finish) to 74
    # (catch) while the seat top was 27, so the rower sat in the air above his
    # own seat.  The rail and seat come up to meet him; the footplate does not
    # move, because the feet measured 36 and it was already right.
    root.add(_part("rail", make_box(200.0, 6.0, 14.0), FRAME, x=-20.0, y=38.0))
    root.add(_part("seat", make_box(30.0, 4.0, 30.0), RUBBER, x=-40.0, y=43.0))
    for x in (-110.0, 100.0):
        root.add(_part("leg", make_box(8.0, 32.0, 14.0), FRAME, x=x, y=20.0))
    root.add(_part("footplate", make_box(10.0, 34.0, 44.0), IRON, x=60.0, y=28.0,
                   quat=quat_from_axis_angle(_Z, math.radians(-25.0))))
    root.add(_part("flywheel", make_cylinder(22.0, 12.0, 24), IRON, x=95.0, y=42.0, quat=_TO_X))
    # No handle here: it is held, so it travels with the hands (a cable_handle
    # attached to them).  Drawn on the frame it stayed at x = 76 while the
    # wrists went from 80 to 163.
    for x in (-110.0, 100.0):
        root.add(_part("foot", make_box(8.0, 17.0, 50.0), FRAME, x=x, y=8.5))
    return root


def make_pedal(width: float = 9.0, length: float = 12.0) -> SceneNode:
    """One pedal, drawn under the foot it belongs to."""
    root = SceneNode("equip_pedal")
    root.add(_part("plate", make_box(width, 2.0, length), RUBBER, y=-6.0))
    root.add(_part("spindle", make_cylinder(1.4, 10.0, 8), STEEL, y=-6.0, quat=_TO_X))
    return root


def make_treadmill(length: float = 170.0, width: float = 72.0, deck: float = 2.0,
                   console: float = 108.0) -> SceneNode:
    """A treadmill facing +Z, the direction the gym body faces.

    The deck top sits at ``deck`` so the ground lock's soles (y ~ 5) land on
    the belt rather than through it.
    """
    root = SceneNode("equip_treadmill")
    root.add(_part("belt", make_box(width, deck, length), RUBBER, y=deck / 2))
    for sx in (-1.0, 1.0):
        root.add(_part(f"rail_{'r' if sx > 0 else 'l'}", make_box(9.0, 5.0, length), FRAME,
                       x=sx * (width / 2 + 4.0), y=deck + 2.5))
        root.add(_part(f"post_{'r' if sx > 0 else 'l'}",
                       make_cylinder(2.4, console, 8), FRAME,
                       x=sx * (width / 2 - 4.0), y=console / 2, z=length / 2 - 8.0))
        root.add(_part(f"handrail_{'r' if sx > 0 else 'l'}",
                       make_cylinder(2.2, length * 0.42, 8), STEEL,
                       x=sx * (width / 2 - 2.0), y=console * 0.82,
                       z=length / 2 - 8.0 - length * 0.21, quat=_TO_Z))
    root.add(_part("console", make_box(width * 0.9, 22.0, 5.0), FRAME,
                   y=console, z=length / 2 - 5.0))
    return root



BUILDERS = {
    "leg_extension_machine": make_leg_extension_machine,
    "leg_curl_machine": make_leg_curl_machine,
    "shin_pad": make_shin_pad,
    "bike": make_bike, "rower": make_rower, "pedal": make_pedal,
    "treadmill": make_treadmill,
}
