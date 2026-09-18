"""Procedural gym equipment, built from the scene's own primitives.

Units are the model's: the figure is ~226 units tall (head +28 to sole -197
in body coordinates), so one unit is about 0.78 cm for a 175 cm person.  A
20 kg barbell is therefore 280 units long with 58-unit plates.

Every builder returns a :class:`SceneNode` whose meshes are marked
``scene_affected=False`` (they live in the room, like the table), with the
node's local +X axis along the handle for anything held in the hands, so the
rig can align it to the line between the wrists.
"""

from __future__ import annotations

import math
from typing import Callable

from faceforge.core.material import Material, RenderMode
from faceforge.core.math_utils import quat_from_axis_angle, quat_multiply, vec3
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.scene.procedural_geometry import (
    make_box, make_cylinder, make_disc, make_sphere, make_torus,
)

STEEL = 0x9AA0A8
IRON = 0x3A3D44
RUBBER = 0x2E3136
WOOD = 0x8B6914
PAD = 0x2F3A5A
FRAME = 0x3C3F46

_X = vec3(1.0, 0.0, 0.0)
_Z = vec3(0.0, 0.0, 1.0)
_TO_X = quat_from_axis_angle(_Z, -math.pi / 2)   # cylinder axis Y -> +X
_TO_Z = quat_from_axis_angle(_X, math.pi / 2)    # cylinder axis Y -> +Z

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


def _part(name: str, geometry: BufferGeometry, color: int, x: float = 0.0, y: float = 0.0,
          z: float = 0.0, quat=None, shininess: float | None = None) -> SceneNode:
    mat = Material.from_hex(color)
    mat.render_mode = RenderMode.SOLID
    if shininess is not None:
        mat.shininess = shininess
    mesh = MeshInstance(name=name, geometry=geometry, material=mat)
    mesh.scene_affected = False
    node = SceneNode(name)
    node.mesh = mesh
    node.set_position(x, y, z)
    if quat is not None:
        node.set_quaternion(quat)
    return node


def make_barbell(length: float = 280.0, bar_radius: float = 1.8, plates: int = 1,
                 plate_radius: float = 29.0, plate_thick: float = 4.0,
                 cable_to: tuple[float, float, float] | None = None) -> SceneNode:
    """An Olympic bar along +X with ``plates`` per side.

    ``cable_to`` draws a cable from the bar's centre to a far point in the
    item's own frame, for the machine bars that are not loaded with plates: a
    lat pulldown is a barbell in the hands here, and without its cable it
    rendered as a man holding a bar over his head for no reason.  The other
    cable exercises got theirs from `make_cable_handle`; this one was missed
    because it is not a handle.
    """
    root = SceneNode("equip_barbell")
    root.add(_part("bar", make_cylinder(bar_radius, length, 12), STEEL, quat=_TO_X, shininess=90))
    if cable_to is not None:
        root.add(_cable(cable_to))
    sleeve_start = length / 2 - 42.0
    for side in (1.0, -1.0):
        for i in range(plates):
            x = side * (sleeve_start + 6.0 + i * (plate_thick + 1.0))
            root.add(_part(f"plate_{i}_{'r' if side > 0 else 'l'}",
                           make_cylinder(plate_radius, plate_thick, 24), RUBBER,
                           x=x, quat=_TO_X))
        root.add(_part(f"collar_{'r' if side > 0 else 'l'}", make_cylinder(3.2, 3.0, 12), STEEL,
                       x=side * (sleeve_start + 2.0), quat=_TO_X))
    return root


def make_dumbbell(handle_length: float = 16.0, head_radius: float = 7.0,
                  head_length: float = 9.0) -> SceneNode:
    """A hex-style dumbbell along +X."""
    root = SceneNode("equip_dumbbell")
    root.add(_part("handle", make_cylinder(1.8, handle_length, 10), STEEL, quat=_TO_X))
    for side in (1.0, -1.0):
        root.add(_part("head", make_cylinder(head_radius, head_length, 6), IRON,
                       x=side * (handle_length / 2 + head_length / 2), quat=_TO_X))
    return root


def make_kettlebell(radius: float = 11.0, flip: bool = False) -> SceneNode:
    """A kettlebell: the handle (+X) is the grip; the bell hangs below it.

    ``flip`` puts the bell ABOVE the handle -- the bell held base-up by the
    horns, which is how a halo is held and what that exercise's own
    description already said.  Hanging, the bell sat on the shoulders as the
    hands passed the head: measured, its box contained the shoulder midpoint.
    """
    root = SceneNode("equip_kettlebell")
    sign = 1.0 if flip else -1.0
    handle = _part("handle", make_torus(9.0, 1.8, 24, 8), IRON,
                   quat=quat_from_axis_angle(_X, math.pi / 2))
    handle.set_position(0.0, sign * 4.0, 0.0)
    root.add(handle)
    root.add(_part("bell", make_sphere(radius, 20, 14), IRON, y=sign * (radius + 8.0)))
    return root


def make_bench(length: float = 150.0, width: float = 34.0, height: float = 58.0,
               incline_deg: float = 0.0, incline_pivot_x: float = 0.0) -> SceneNode:
    """A flat (or inclined) bench along X, top surface at ``height``.

    ``incline_pivot_x`` is where along the pad the tilt hinges, in bench-local
    x.  It matters because the *lifter* is tilted too, by a wrapper pitch about
    the hips, and two rigid bodies turned by the same angle about different
    centres come apart by a pure translation.  Hinging the pad at its own
    centre while the body hinged 41 units away did exactly that: measured
    2026-09-17 against the flat bench (shoulders +10.6, hips +13.7 above the
    pad face, which is what lying on it reads as), the 30 deg incline floated
    the lifter 18.1 clear of the pad and the 20 deg decline sank him 15.1
    through it -- 41 x sin(theta) in both cases.  Hinging under the hip is what
    keeps the contact the flat bench has.
    """
    root = SceneNode("equip_bench")
    if incline_deg:
        # A node turns about its own origin, so the hinge is a node placed at
        # the pivot with the pad hung off it at the opposite offset.
        # At the pad's top FACE, not at its mid-plane: the face is the contact
        # surface, and hinging 3 below it leaves a lever that tilts the face
        # away by 3 x sin(theta) -- the 2.4 that was left on the 30 deg incline
        # once the hinge moved under the hip.
        hinge = SceneNode("pad_hinge")
        hinge.set_position(incline_pivot_x, height, 0.0)
        hinge.set_quaternion(quat_from_axis_angle(_Z, math.radians(incline_deg)))
        hinge.add(_part("pad", make_box(length, 6.0, width), PAD,
                        x=-incline_pivot_x, y=-3.0))
        root.add(hinge)
    else:
        root.add(_part("pad", make_box(length, 6.0, width), PAD, y=height - 3.0))
    for x in (-length / 2 + 14.0, length / 2 - 14.0):
        root.add(_part("leg", make_box(6.0, height - 6.0, width - 6.0), FRAME,
                       x=x, y=(height - 6.0) / 2))
    return root


def make_pullup_bar(width: float = 170.0, height: float = 275.0) -> SceneNode:
    """A free-standing bar along X at ``height``.

    ``width`` is the span between the uprights, not the usable grip.  At 120
    the uprights stood at x = +-60 and a pull-up's flared elbow measures 60.4
    at the top of the pull (60.3 chinning, 59.6 wide-grip), so the post ran
    straight down the middle of the upper arm.  The span is now outside every
    grip the catalogue authors, which is what a real rack looks like anyway.
    """
    root = SceneNode("equip_pullup_bar")
    root.add(_part("bar", make_cylinder(1.8, width, 12), STEEL, y=height, quat=_TO_X,
                   shininess=90))
    for x in (-width / 2, width / 2):
        root.add(_part("upright", make_cylinder(2.5, height, 10), FRAME, x=x, y=height / 2))
        root.add(_part("foot", make_box(10.0, 3.0, 60.0), FRAME, x=x, y=1.5))
    return root


def make_plyo_box(height: float = 60.0, width: float = 60.0, depth: float = 50.0) -> SceneNode:
    root = SceneNode("equip_plyo_box")
    root.add(_part("box", make_box(width, height, depth), WOOD, y=height / 2))
    return root


def make_wall(width: float = 160.0, height: float = 220.0, thickness: float = 8.0) -> SceneNode:
    """A short wall panel in the XY plane, its face at z = 0, standing on the floor.

    The gym has walls, but they are 200 units away and the body is placed at
    the origin, so `wall_sit` -- whose every cue is about the wall -- and the
    standing calf stretch were performed against nothing at all.  This is the
    piece of wall the exercise actually touches, placed behind or in front of
    the athlete by the spec's ``position``.
    """
    root = SceneNode("equip_wall")
    root.add(_part("wall", make_box(width, height, thickness), FRAME,
                   y=height / 2.0, z=-thickness / 2.0))
    return root


def make_mat(length: float = 230.0, width: float = 120.0) -> SceneNode:
    """A floor mat along X.

    80 is a yoga mat (63 cm) and too narrow for the floor work in this
    catalogue: measured 2026-09-16, the main joints of a push-up reach z +-57,
    a downward dog and a mountain climber +-51, a cat-cow +-49 -- so half the
    body hung over the edge even once the mat was laid down under it.  120
    (95 cm) is a large exercise mat and covers them.

    Its top face is the floor plane, not 1.5 above it.  The ground lock
    anchors the body to y = 0 and every static height in the gym is measured
    from there, so a slab laid ON the floor puts the athlete inside their own
    mat: measured 2026-09-17, a glute bridge's little toe sat 2.5 below the
    mat surface and an upward dog's 1.7, while both were at or above the
    actual floor.  Raising the body onto the mat instead would desynchronise
    every bench, bar and box height in the catalogue for 1.5 units that no
    camera can see -- less than a real mat compresses under a heel.
    """
    root = SceneNode("equip_mat")
    # Top face 0.3 ABOVE the floor plane, not exactly on it.  Flush was worse
    # than either: the floor is a plane at y = 0 and a mat whose top face is
    # also at y = 0 z-fights with it, which rendered as blue streaks across
    # every mat exercise.  0.3 is clear of the fight and still inside a real
    # mat's compression under a heel.
    root.add(_part("mat", make_box(length, 1.5, width), 0x35566B, y=-0.45))
    return root


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


def make_jump_rope(span: float = 120.0, drop: float = 170.0) -> SceneNode:
    """Two handles (+X apart) with a rope arc hanging ``drop`` below them.

    ``drop`` is the arc's radius and a skipping rope's arc has to reach the
    floor -- that is the whole exercise.  At 100 it bottomed out at y 70 with
    the handles at 178, so the rope swept past the shins and the skipper
    jumped over nothing.  The handles measure 178-196 through the rep, and at
    a 170 radius the bottom of the arc arrives at y 1 on the landing -- 178
    put it 8 units under.
    """
    root = SceneNode("equip_jump_rope")
    for x in (-span / 2, span / 2):
        root.add(_part("handle", make_cylinder(1.6, 16.0, 8), RUBBER, x=x))
    segments = 14
    prev = (-span / 2, -8.0, 0.0)
    for i in range(1, segments + 1):
        u = i / segments
        x = -span / 2 + u * span
        y = -8.0 - drop * math.sin(math.pi * u)
        mid = ((prev[0] + x) / 2, (prev[1] + y) / 2, 0.0)
        dx, dy = x - prev[0], y - prev[1]
        length = math.hypot(dx, dy)
        angle = math.atan2(dx, dy)
        seg = _part(f"rope_{i}", make_cylinder(0.6, length, 6), STEEL, x=mid[0], y=mid[1],
                    quat=quat_from_axis_angle(_Z, -angle))
        root.add(seg)
        prev = (x, y, 0.0)
    return root


def make_dip_station(width: float = 81.0, height: float = 125.0,
                     length: float = 70.0) -> SceneNode:
    """Two parallel bars along Z, ``width`` apart, at ``height``.

    81 is measured, not chosen: the dip and L-sit poses put the closed-finger
    grip at x +-40.4, and at 54 the athlete was supported on air 16 to 26
    units outside each bar.
    """
    root = SceneNode("equip_dip_station")
    for x in (-width / 2, width / 2):
        root.add(_part("bar", make_cylinder(1.8, length, 12), STEEL, x=x, y=height,
                       quat=_TO_Z, shininess=90))
        for z in (-length / 2 + 6.0, length / 2 - 6.0):
            root.add(_part("upright", make_cylinder(2.2, height, 8), FRAME, x=x, y=height / 2,
                           z=z))
    root.add(_part("base", make_box(width + 16.0, 3.0, length), FRAME, y=1.5))
    return root


def make_medicine_ball(radius: float = 12.0) -> SceneNode:
    root = SceneNode("equip_medicine_ball")
    root.add(_part("ball", make_sphere(radius, 20, 14), 0x6B3A2A, shininess=20))
    return root


def _cable(cable_to, segments: int = 8) -> SceneNode | None:
    """A run of cable from the item's origin to ``cable_to``, as one node."""
    far = vec3(*cable_to)
    span = float((far[0] ** 2 + far[1] ** 2 + far[2] ** 2) ** 0.5)
    if span < 1e-6:
        return None
    node = SceneNode("cable")
    step = far / segments
    seg_len = span / segments
    for i in range(segments):
        centre = step * (i + 0.5)
        node.add(_part(f"cable_{i}", make_cylinder(0.9, seg_len * 1.1, 6), STEEL,
                       x=float(centre[0]), y=float(centre[1]), z=float(centre[2]),
                       quat=_axis_to(tuple(far))))
    return node


def make_cable_handle(cable_to=(0.0, 130.0, 0.0), radius: float = 1.6,
                      segments: int = 8, length: float = 14.0) -> SceneNode:
    """A handle along +X with the cable that makes it a cable exercise.

    Without the cable a pushdown, a row, a face pull and a Pallof press all
    render as a figure miming: the handle alone is a 14-unit dark cylinder
    inside the hands and invisible at body scale.  ``cable_to`` is the far end
    in the item's own frame, which after ``align_x_to`` has +X along the line
    between the hands, +Y up and +Z forward for the usual case of two hands
    side by side -- so the default runs the cable to a high pulley.

    ``length`` is the handle itself.  14 is a D-handle, one per hand; a
    two-handed straight bar has to span the grip or the hands hold nothing:
    measured 2026-09-16, a pushdown's grip is 75 units wide and a seated row's
    83-94, against a 14-unit handle whose ends were 30 and 40 units short.
    """
    root = SceneNode("equip_cable_handle")
    root.add(_part("handle", make_cylinder(radius, length, 8), RUBBER, quat=_TO_X))
    cable = _cable(cable_to, segments)
    if cable is None:
        return root
    root.add(cable)
    root.add(_part("anchor", make_box(16.0, 8.0, 8.0), FRAME,
                   x=float(cable_to[0]), y=float(cable_to[1]), z=float(cable_to[2])))
    return root


def make_band(length: float = 60.0) -> SceneNode:
    root = SceneNode("equip_band")
    root.add(_part("band", make_cylinder(1.2, length, 8), 0xC03030, quat=_TO_X))
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


def _axis_to(direction) -> object:
    """A quaternion taking a cylinder's +Y axis onto ``direction``."""
    d = vec3(*direction)
    n = float((d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5)
    if n < 1e-9:
        return None
    d = d / n
    dot = float(max(-1.0, min(1.0, d[1])))
    if dot > 1.0 - 1e-9:
        return None
    if dot < -1.0 + 1e-9:
        return quat_from_axis_angle(_X, math.pi)
    axis = vec3(d[2], 0.0, -d[0])          # cross((0,1,0), d)
    return quat_from_axis_angle(axis, math.acos(dot))


def make_battle_rope(length: float = 200.0, radius: float = 3.4, waves: float = 1.4,
                     amplitude: float = 16.0, drop: float = 120.0,
                     segments: int = 18) -> SceneNode:
    """One heavy rope trailing from a hand, waving, and sloping to the floor.

    Built along local +Z, which is where a hand-attached item's untouched axis
    points once ``align_x_to`` has put its +X on the line between the hands --
    so the rope runs out in front of the athlete.  Each segment is turned onto
    the curve's own tangent, because cylinders merely offset from one another
    read as a staircase, which is what the first attempt rendered.

    ``drop`` is how far the far end falls over ``length``.  At 55 both ropes
    simply stopped in mid-air 57 to 118 units up, with nothing at the end of
    them; a battle rope is anchored at the floor.  120 lands the low hand's
    rope on the floor (measured hand 126) and leaves the high hand's 66 up,
    which is the wave still travelling down it.
    """
    root = SceneNode("equip_battle_rope")

    def curve(u: float) -> tuple[float, float]:
        return (amplitude * (1.0 - u) * math.sin(2.0 * math.pi * waves * u), -drop * u * u)

    for i in range(segments):
        u0, u1 = i / segments, (i + 1) / segments
        y0, s0 = curve(u0)
        y1, s1 = curve(u1)
        p0 = (y0 + s0, u0 * length)
        p1 = (y1 + s1, u1 * length)
        dy, dz = p1[0] - p0[0], p1[1] - p0[1]
        span = math.hypot(dy, dz)
        root.add(_part(f"rope_{i}", make_cylinder(radius, span * 1.25, 6), RUBBER,
                       y=(p0[0] + p1[0]) / 2.0, z=(p0[1] + p1[1]) / 2.0,
                       quat=_axis_to((0.0, dy, dz))))
    return root


EQUIPMENT_BUILDERS: dict[str, Callable[..., SceneNode]] = {
    "barbell": make_barbell, "dumbbell": make_dumbbell, "kettlebell": make_kettlebell,
    "bench": make_bench, "pullup_bar": make_pullup_bar, "plyo_box": make_plyo_box,
    "mat": make_mat, "bike": make_bike, "rower": make_rower, "jump_rope": make_jump_rope,
    "cable_handle": make_cable_handle, "band": make_band, "dip_station": make_dip_station,
    "wall": make_wall,
    "medicine_ball": make_medicine_ball, "treadmill": make_treadmill,
    "pedal": make_pedal,
    "battle_rope": make_battle_rope,
}


def known_equipment() -> set[str]:
    return set(EQUIPMENT_BUILDERS)


def build_equipment(kind: str, **params) -> SceneNode:
    """Build one piece of equipment by name."""
    try:
        builder = EQUIPMENT_BUILDERS[kind]
    except KeyError as exc:
        raise KeyError(f"unknown equipment {kind!r}; have {sorted(EQUIPMENT_BUILDERS)}") from exc
    return builder(**params)
