"""Assembles the room environment from procedural primitives.

Supports two scene types:

**Examination room** (default)::

    SceneNode "scene_env_root"
    +-- SceneNode "room"
    |   +-- floor, ceiling, wall_back, wall_left, wall_right
    +-- SceneNode "table"
    |   +-- tabletop, leg_FL, leg_FR, leg_BL, leg_BR
    +-- SceneNode "lamp"
        +-- lamp_arm, lamp_shade, lamp_disc

**Dance studio** (scene_type="dance_studio")::

    SceneNode "scene_env_root"
    +-- SceneNode "room"
    |   +-- floor (dark polished), ceiling (dark), walls (dark)
    +-- SceneNode "spotlight"
        +-- spot_arm, spot_shade, spot_disc
"""

import numpy as np

from faceforge.core.material import Material, RenderMode
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode

from faceforge.scene.procedural_geometry import (
    make_box, make_plane, make_cylinder, make_disc,
)

# Room dimensions (examination room)
ROOM_WIDTH = 300.0   # X
ROOM_HEIGHT = 250.0  # Y
ROOM_DEPTH = 200.0   # Z

# Dance studio dimensions (larger)
STUDIO_WIDTH = 500.0   # X
STUDIO_HEIGHT = 300.0  # Y
STUDIO_DEPTH = 400.0   # Z

# Table
TABLE_LENGTH = 200.0   # X
TABLE_WIDTH = 80.0     # Z
TABLE_HEIGHT = 90.0    # Y (surface height)
TABLE_TOP_THICK = 4.0
TABLE_LEG_RADIUS = 2.0
TABLE_LEG_HEIGHT = TABLE_HEIGHT - TABLE_TOP_THICK

# Lamp
LAMP_ARM_RADIUS = 1.5
LAMP_SHADE_RADIUS_TOP = 12.0
LAMP_SHADE_RADIUS_BOT = 18.0
LAMP_SHADE_HEIGHT = 10.0

# Spotlight (dance studio)
SPOT_ARM_RADIUS = 2.0
SPOT_SHADE_RADIUS = 15.0
SPOT_SHADE_HEIGHT = 12.0

# Wall thickness
WALL_THICK = 2.0


def _make_node(name: str, geometry: BufferGeometry, color_hex: int,
               x: float = 0, y: float = 0, z: float = 0,
               double_sided: bool = False) -> SceneNode:
    """Helper: wrap geometry + material in a SceneNode at position (x, y, z)."""
    mat = Material.from_hex(color_hex, double_sided=double_sided)
    mesh = MeshInstance(name=name, geometry=geometry, material=mat)
    mesh.scene_affected = False  # environment stays fixed in world space
    node = SceneNode(name)
    node.mesh = mesh
    node.set_position(x, y, z)
    return node


class SceneEnvironment:
    """Builds a procedural room environment.

    Parameters
    ----------
    scene_type : str
        ``"examination"`` (default) — room with table and lamp.
        ``"dance_studio"`` — larger dark room with spotlight, no table.
    """

    def __init__(self, scene_type: str = "examination") -> None:
        self.root: SceneNode | None = None
        self.scene_type = scene_type
        self._light_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def build(self) -> SceneNode:
        """Construct the full environment and return the root node."""
        if self.scene_type == "dance_studio":
            return self._build_dance_studio()
        return self._build_examination()

    # ── Examination room ──────────────────────────────────────────────

    def _build_examination(self) -> SceneNode:
        """Build the default examination room with table and lamp."""
        root = SceneNode("scene_env_root")

        # ── Room ──────────────────────────────────────────────
        room = SceneNode("room")
        root.add(room)

        # Floor at Y=0
        floor = _make_node(
            "floor", make_plane(ROOM_WIDTH, ROOM_DEPTH, 4, 4),
            0x666666, y=0, double_sided=True,
        )
        room.add(floor)

        # Ceiling at Y=ROOM_HEIGHT
        ceiling = _make_node(
            "ceiling", make_plane(ROOM_WIDTH, ROOM_DEPTH, 4, 4),
            0xCCCCCC, y=ROOM_HEIGHT, double_sided=True,
        )
        room.add(ceiling)

        # Wall back (at Z = -ROOM_DEPTH/2)
        wall_back = _make_node(
            "wall_back", make_box(ROOM_WIDTH, ROOM_HEIGHT, WALL_THICK),
            0x999999, y=ROOM_HEIGHT / 2, z=-ROOM_DEPTH / 2,
        )
        room.add(wall_back)

        # Wall left (at X = -ROOM_WIDTH/2)
        wall_left = _make_node(
            "wall_left", make_box(WALL_THICK, ROOM_HEIGHT, ROOM_DEPTH),
            0x999999, x=-ROOM_WIDTH / 2, y=ROOM_HEIGHT / 2,
        )
        room.add(wall_left)

        # Wall right (at X = +ROOM_WIDTH/2)
        wall_right = _make_node(
            "wall_right", make_box(WALL_THICK, ROOM_HEIGHT, ROOM_DEPTH),
            0x999999, x=ROOM_WIDTH / 2, y=ROOM_HEIGHT / 2,
        )
        room.add(wall_right)

        # ── Table ─────────────────────────────────────────────
        table = SceneNode("table")
        root.add(table)

        # Tabletop: centered at (0, TABLE_HEIGHT - TABLE_TOP_THICK/2, 0)
        tabletop = _make_node(
            "tabletop", make_box(TABLE_LENGTH, TABLE_TOP_THICK, TABLE_WIDTH),
            0x8B6914, y=TABLE_HEIGHT - TABLE_TOP_THICK / 2,
        )
        table.add(tabletop)

        # Four legs
        leg_h = TABLE_LEG_HEIGHT
        leg_y = leg_h / 2  # center of leg
        inset_x = TABLE_LENGTH / 2 - 6
        inset_z = TABLE_WIDTH / 2 - 6
        for label, lx, lz in [
            ("leg_FL", -inset_x, inset_z),
            ("leg_FR", inset_x, inset_z),
            ("leg_BL", -inset_x, -inset_z),
            ("leg_BR", inset_x, -inset_z),
        ]:
            leg = _make_node(
                label, make_cylinder(TABLE_LEG_RADIUS, leg_h, 8),
                0x6B4F14, x=lx, y=leg_y, z=lz,
            )
            table.add(leg)

        # ── Lamp ──────────────────────────────────────────────
        lamp = SceneNode("lamp")
        root.add(lamp)

        # Lamp arm: short vertical cylinder hanging from ceiling
        arm_height = 30.0  # only extends slightly from ceiling
        arm_y = ROOM_HEIGHT - arm_height / 2
        lamp_arm = _make_node(
            "lamp_arm", make_cylinder(LAMP_ARM_RADIUS, arm_height, 8),
            0x444444, y=arm_y,
        )
        lamp.add(lamp_arm)

        # Lamp shade: truncated cone approximated by a cylinder at the bottom
        shade_y = ROOM_HEIGHT - arm_height - LAMP_SHADE_HEIGHT / 2
        lamp_shade = _make_node(
            "lamp_shade",
            make_cylinder(LAMP_SHADE_RADIUS_TOP, LAMP_SHADE_HEIGHT, 16),
            0x555555, y=shade_y,
        )
        lamp.add(lamp_shade)

        # Disc at bottom of shade (emissive to suggest light source)
        disc_y = shade_y - LAMP_SHADE_HEIGHT / 2
        lamp_disc = _make_node(
            "lamp_disc", make_disc(LAMP_SHADE_RADIUS_TOP, 16),
            0xFFFAE0, y=disc_y, double_sided=True,
        )
        # Make the disc slightly emissive
        lamp_disc.mesh.material.emissive = (0.4, 0.38, 0.3)
        lamp.add(lamp_disc)

        # Point light position = center of lamp shade bottom
        self._light_pos = np.array([0.0, disc_y, 0.0], dtype=np.float64)

        self.root = root
        return root

    # ── Dance studio ──────────────────────────────────────────────────

    def _build_dance_studio(self) -> SceneNode:
        """Build a large dark dance studio with a single overhead spotlight."""
        root = SceneNode("scene_env_root")

        room = SceneNode("room")
        root.add(room)

        # Dark polished floor
        floor = _make_node(
            "floor", make_plane(STUDIO_WIDTH, STUDIO_DEPTH, 6, 6),
            0x1a1a1a, y=0, double_sided=True,
        )
        floor.mesh.material.shininess = 80.0
        room.add(floor)

        # Dark ceiling
        ceiling = _make_node(
            "ceiling", make_plane(STUDIO_WIDTH, STUDIO_DEPTH, 4, 4),
            0x0a0a0a, y=STUDIO_HEIGHT, double_sided=True,
        )
        room.add(ceiling)

        # Dark walls
        wall_back = _make_node(
            "wall_back", make_box(STUDIO_WIDTH, STUDIO_HEIGHT, WALL_THICK),
            0x151515, y=STUDIO_HEIGHT / 2, z=-STUDIO_DEPTH / 2,
        )
        room.add(wall_back)

        wall_left = _make_node(
            "wall_left", make_box(WALL_THICK, STUDIO_HEIGHT, STUDIO_DEPTH),
            0x151515, x=-STUDIO_WIDTH / 2, y=STUDIO_HEIGHT / 2,
        )
        room.add(wall_left)

        wall_right = _make_node(
            "wall_right", make_box(WALL_THICK, STUDIO_HEIGHT, STUDIO_DEPTH),
            0x151515, x=STUDIO_WIDTH / 2, y=STUDIO_HEIGHT / 2,
        )
        room.add(wall_right)

        # ── Spotlight ─────────────────────────────────────────
        spotlight = SceneNode("spotlight")
        root.add(spotlight)

        # Arm hanging from ceiling
        arm_height = 20.0
        arm_y = STUDIO_HEIGHT - arm_height / 2
        spot_arm = _make_node(
            "spot_arm", make_cylinder(SPOT_ARM_RADIUS, arm_height, 8),
            0x222222, y=arm_y,
        )
        spotlight.add(spot_arm)

        # Shade
        shade_y = STUDIO_HEIGHT - arm_height - SPOT_SHADE_HEIGHT / 2
        spot_shade = _make_node(
            "spot_shade", make_cylinder(SPOT_SHADE_RADIUS, SPOT_SHADE_HEIGHT, 16),
            0x333333, y=shade_y,
        )
        spotlight.add(spot_shade)

        # Emissive disc at bottom
        disc_y = shade_y - SPOT_SHADE_HEIGHT / 2
        spot_disc = _make_node(
            "spot_disc", make_disc(SPOT_SHADE_RADIUS, 16),
            0xFFF8E0, y=disc_y, double_sided=True,
        )
        spot_disc.mesh.material.emissive = (0.6, 0.55, 0.4)
        spotlight.add(spot_disc)

        self._light_pos = np.array([0.0, disc_y, 0.0], dtype=np.float64)

        self.root = root
        return root

    # ── Public API ────────────────────────────────────────────────────

    def get_light_position(self) -> np.ndarray:
        """Return the world-space position for the point light."""
        return self._light_pos.copy()

    def set_render_mode(self, mode: RenderMode) -> None:
        """Apply a render mode to all environment meshes."""
        if self.root is None:
            return

        def _apply(node: SceneNode):
            if node.mesh is not None:
                node.mesh.material.render_mode = mode

        self.root.traverse(_apply)
