"""Equipment geometry is well formed; the rig puts it in the hands."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.core.math_utils import quat_rotate_vec3, vec3
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.exercise.equipment import EQUIPMENT_BUILDERS, build_equipment, known_equipment
from faceforge.exercise.equipment_rig import EquipmentRig, align_x_to
from faceforge.exercise.model import EquipmentSpec


@pytest.mark.parametrize("kind", sorted(EQUIPMENT_BUILDERS))
def test_every_builder_returns_valid_geometry(kind):
    node = build_equipment(kind)
    meshes = node.subtree_meshes()
    assert meshes, kind
    for mesh in meshes:
        geom = mesh.geometry
        pos = geom.positions.reshape(-1, 3)
        assert np.isfinite(pos).all()
        assert geom.has_indices and int(geom.indices.max()) < geom.vertex_count
        nrm = np.linalg.norm(geom.normals.reshape(-1, 3), axis=1)
        assert np.allclose(nrm, 1.0, atol=1e-3)
        assert mesh.scene_affected is False, "equipment lives in the room, not the body"


def test_unknown_equipment_is_an_error():
    with pytest.raises(KeyError):
        build_equipment("treadmill")
    assert "barbell" in known_equipment()


def test_align_x_to_handles_degenerate_directions():
    q = align_x_to(vec3(0.0, 1.0, 0.0))
    assert quat_rotate_vec3(q, vec3(1, 0, 0)) == pytest.approx([0, 1, 0], abs=1e-9)
    assert align_x_to(vec3(1, 0, 0)) == pytest.approx([0, 0, 0, 1])
    q = align_x_to(vec3(-1, 0, 0))
    assert quat_rotate_vec3(q, vec3(1, 0, 0)) == pytest.approx([-1, 0, 0], abs=1e-9)
    assert align_x_to(vec3(0, 0, 0)) == pytest.approx([0, 0, 0, 1])


class _Pivot(SceneNode):
    def __init__(self, name, world):
        super().__init__(name)
        self._world = np.asarray(world, dtype=np.float64)

    def get_world_position(self):
        return self._world.copy()


def _hands():
    return {
        "wrist_R": _Pivot("wrist_R", (30.0, 100.0, 20.0)),
        "wrist_L": _Pivot("wrist_L", (-30.0, 100.0, 20.0)),
        "elbow_R": _Pivot("elbow_R", (30.0, 130.0, 20.0)),
        "elbow_L": _Pivot("elbow_L", (-30.0, 130.0, 20.0)),
    }


def test_two_handed_item_sits_between_the_palms_along_the_hand_line():
    rig = EquipmentRig()
    bar = build_equipment("barbell")
    rig.add(bar, EquipmentSpec("barbell", attach="hands"), grip_offset=7.0)
    rig.update(_hands())
    # Palms are 7 units past the wrists along the forearm (straight down here).
    assert bar.position == pytest.approx([0.0, 93.0, 20.0])
    axis = quat_rotate_vec3(bar.quaternion, vec3(1, 0, 0))
    assert axis == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_one_handed_items_follow_their_own_wrist_and_hang():
    rig = EquipmentRig()
    kb = build_equipment("kettlebell")
    rig.add(kb, EquipmentSpec("kettlebell", attach="hand_r"), grip_offset=0.0, hang=10.0)
    rig.update(_hands())
    assert kb.position == pytest.approx([30.0, 90.0, 20.0])


def test_static_items_are_placed_once_and_never_moved():
    rig = EquipmentRig()
    bench = build_equipment("bench")
    rig.add(bench, EquipmentSpec("bench", attach="static", position=(5.0, 0.0, -9.0)))
    rig.update(_hands())
    assert bench.position == pytest.approx([5.0, 0.0, -9.0])


def test_spin_turns_the_item_about_the_hand_axis_with_time():
    rig = EquipmentRig()
    rope = build_equipment("jump_rope")
    rig.add(rope, EquipmentSpec("jump_rope", attach="hands"), spin=1.0)
    rig.update(_hands(), time=0.0)
    q0 = rope.quaternion.copy()
    rig.update(_hands(), time=0.25)
    q1 = rope.quaternion.copy()
    assert not np.allclose(q0, q1)
    down0 = quat_rotate_vec3(q0, vec3(0, -1, 0))
    down1 = quat_rotate_vec3(q1, vec3(0, -1, 0))
    assert abs(float(np.dot(down0, down1))) < 1e-6, "a quarter turn"


def test_clear_returns_the_nodes_for_removal():
    rig = EquipmentRig()
    scene = Scene()
    node = build_equipment("dumbbell")
    scene.add(node)
    rig.add(node, EquipmentSpec("dumbbell", attach="hand_l"))
    assert rig.clear() == [node]
    assert rig.items == []


def _hands_with_fingers():
    """Wrists plus curled fingers whose joint ring is centred 10 units below each wrist."""
    pivots = _hands()
    for side, x in (("R", 30.0), ("L", -30.0)):
        for digit in (2, 3, 4, 5):
            # MCP, PIP, DIP and the extrapolated tip go round a bar centred at (x, 90, 20).
            pivots[f"finger_{side}_{digit}_prox"] = _Pivot(f"finger_{side}_{digit}_prox", (x, 92.0, 18.0))
            pivots[f"finger_{side}_{digit}_mid"] = _Pivot(f"finger_{side}_{digit}_mid", (x, 90.0, 24.0))
            pivots[f"finger_{side}_{digit}_dist"] = _Pivot(f"finger_{side}_{digit}_dist", (x, 86.0, 21.0))
    return pivots


def test_a_held_bar_passes_through_the_closed_fingers_not_the_palm():
    rig = EquipmentRig()
    bar = build_equipment("barbell")
    rig.add(bar, EquipmentSpec("barbell", attach="hands"), grip_offset=7.0)
    pivots = _hands_with_fingers()
    rig.update(pivots)
    # tip = dist + 0.8 * (dist - mid) = (x, 82.8, 18.6); ring centroid = (x, 87.7, 20.4)
    ring = np.mean([(30.0, 92.0, 18.0), (30.0, 90.0, 24.0), (30.0, 86.0, 21.0), (30.0, 82.8, 18.6)], axis=0)
    assert bar.position == pytest.approx([0.0, ring[1], ring[2]], abs=1e-6)
    assert bar.position[1] != pytest.approx(93.0), "not the wrist-plus-offset palm point"
    assert EquipmentRig.grip_point(_hands(), "R", 7.0) == pytest.approx([30.0, 93.0, 20.0]), \
        "a rig without finger pivots keeps the palm fallback"
