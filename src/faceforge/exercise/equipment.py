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
                 plate_radius: float = 29.0, plate_thick: float = 4.0) -> SceneNode:
    """An Olympic bar along +X with ``plates`` per side."""
    root = SceneNode("equip_barbell")
    root.add(_part("bar", make_cylinder(bar_radius, length, 12), STEEL, quat=_TO_X, shininess=90))
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


def make_kettlebell(radius: float = 11.0) -> SceneNode:
    """A kettlebell: the handle (+X) is the grip; the bell hangs below it."""
    root = SceneNode("equip_kettlebell")
    handle = _part("handle", make_torus(9.0, 1.8, 24, 8), IRON,
                   quat=quat_from_axis_angle(_X, math.pi / 2))
    handle.set_position(0.0, -4.0, 0.0)
    root.add(handle)
    root.add(_part("bell", make_sphere(radius, 20, 14), IRON, y=-radius - 8.0))
    return root


def make_bench(length: float = 150.0, width: float = 34.0, height: float = 58.0,
               incline_deg: float = 0.0) -> SceneNode:
    """A flat (or inclined) bench along X, top surface at ``height``."""
    root = SceneNode("equip_bench")
    pad = _part("pad", make_box(length, 6.0, width), PAD, y=height - 3.0)
    if incline_deg:
        pad.set_quaternion(quat_from_axis_angle(_Z, math.radians(incline_deg)))
    root.add(pad)
    for x in (-length / 2 + 14.0, length / 2 - 14.0):
        root.add(_part("leg", make_box(6.0, height - 6.0, width - 6.0), FRAME,
                       x=x, y=(height - 6.0) / 2))
    return root


def make_pullup_bar(width: float = 120.0, height: float = 275.0) -> SceneNode:
    """A free-standing bar along X at ``height``."""
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


def make_mat(length: float = 230.0, width: float = 80.0) -> SceneNode:
    root = SceneNode("equip_mat")
    root.add(_part("mat", make_box(length, 1.5, width), 0x35566B, y=0.75))
    return root


def make_bike() -> SceneNode:
    """An upright stationary bike facing +Z: saddle, bars, crank and wheel."""
    root = SceneNode("equip_bike")
    root.add(_part("frame_down", make_box(6.0, 70.0, 6.0), FRAME, y=40.0, z=10.0,
                   quat=quat_from_axis_angle(_X, math.radians(20.0))))
    root.add(_part("seat_post", make_cylinder(2.5, 55.0, 8), FRAME, y=70.0, z=-12.0))
    root.add(_part("saddle", make_box(14.0, 4.0, 26.0), RUBBER, y=98.0, z=-12.0))
    root.add(_part("head_tube", make_cylinder(2.5, 60.0, 8), FRAME, y=90.0, z=42.0))
    root.add(_part("handlebar", make_cylinder(1.8, 46.0, 10), STEEL, y=120.0, z=42.0, quat=_TO_X))
    root.add(_part("wheel", make_torus(24.0, 3.0, 32, 10), IRON, y=30.0, z=22.0,
                   quat=quat_from_axis_angle(_Z, math.pi / 2)))
    root.add(_part("crank_axle", make_cylinder(2.0, 24.0, 8), STEEL, y=30.0, z=0.0, quat=_TO_X))
    for x, name in ((13.0, "pedal_r"), (-13.0, "pedal_l")):
        root.add(_part(name, make_box(8.0, 2.0, 10.0), RUBBER, x=x, y=30.0, z=0.0))
    root.add(_part("base", make_box(50.0, 3.0, 110.0), FRAME, y=1.5))
    return root


def make_rower() -> SceneNode:
    """A rowing ergometer along X: rail, sliding seat, footplate and handle."""
    root = SceneNode("equip_rower")
    root.add(_part("rail", make_box(200.0, 6.0, 14.0), FRAME, x=-20.0, y=20.0))
    root.add(_part("seat", make_box(30.0, 4.0, 30.0), RUBBER, x=-40.0, y=25.0))
    root.add(_part("footplate", make_box(10.0, 34.0, 44.0), IRON, x=60.0, y=28.0,
                   quat=quat_from_axis_angle(_Z, math.radians(-25.0))))
    root.add(_part("flywheel", make_cylinder(22.0, 12.0, 24), IRON, x=95.0, y=42.0, quat=_TO_X))
    root.add(_part("handle", make_cylinder(1.8, 40.0, 10), WOOD, x=40.0, y=60.0, quat=_TO_Z))
    for x in (-110.0, 100.0):
        root.add(_part("foot", make_box(8.0, 17.0, 50.0), FRAME, x=x, y=8.5))
    return root


def make_jump_rope(span: float = 120.0, drop: float = 100.0) -> SceneNode:
    """Two handles (+X apart) with a rope arc hanging ``drop`` below them."""
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


def make_dip_station(width: float = 54.0, height: float = 125.0,
                     length: float = 70.0) -> SceneNode:
    """Two parallel bars along Z, ``width`` apart, at ``height``."""
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


def make_cable_handle() -> SceneNode:
    root = SceneNode("equip_cable_handle")
    root.add(_part("handle", make_cylinder(1.6, 14.0, 8), RUBBER, quat=_TO_X))
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
                     amplitude: float = 16.0, drop: float = 55.0,
                     segments: int = 18) -> SceneNode:
    """One heavy rope trailing from a hand, waving, and sloping to the floor.

    Built along local +Z, which is where a hand-attached item's untouched axis
    points once ``align_x_to`` has put its +X on the line between the hands --
    so the rope runs out in front of the athlete.  Each segment is turned onto
    the curve's own tangent, because cylinders merely offset from one another
    read as a staircase, which is what the first attempt rendered.
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
    "medicine_ball": make_medicine_ball, "treadmill": make_treadmill,
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
