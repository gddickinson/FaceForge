"""Tests for SceneState serialisation: exactness, versioning, provenance.

What these tests are for
------------------------
A SceneState file is a reproducibility claim.  Three things can make that claim
false without anything appearing to go wrong:

1. a float that does not survive the round trip (a camera position written with
   ``str()`` moves the shot by a fraction of a degree),
2. a field that is silently dropped or defaulted on load,
3. a state that renders differently because the configs changed underneath it.

Every test below exists to make one of those detectable.  The negative-control
tests matter as much as the positive ones: an equality assertion that cannot
fail proves nothing, so ``test_single_field_difference_is_detected`` mutates one
field at a time and requires each mutation to be caught.
"""

import json
import math
import struct
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from faceforge.core import scene_state as ss
from faceforge.core.material import Material, RenderMode
from faceforge.core.math_utils import vec3
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.scene_state import codec, confighash
from faceforge.core.state import BodyState, StateManager
from faceforge.loaders.stl_batch_loader import load_fma_labels, load_stl_batch
from faceforge.rendering.camera import Camera
from faceforge.rendering.lights import LightSetup, PointLight

# ---------------------------------------------------------------------------
# Awkward float values.
#
# Chosen to break specific plausible implementations:
#   0.1, 0.2, 0.1+0.2  -- decimal-inexact; str() vs repr() differ historically
#   1e-8, -1e-8        -- below any tolerance a sloppy comparison would use
#   -0.0               -- equal to 0.0 under ==, so only a byte comparison sees
#                         the lost sign; the sign is real (it flips a normal)
#   5e-324             -- smallest subnormal; lost entirely by a float32 round
#   1.797...e308       -- largest finite double; overflows to inf if scaled
#   float(np.float32(0.1)) -- what a float32 vertex buffer actually holds
# ---------------------------------------------------------------------------
AWKWARD_FLOATS = (
    0.1,
    0.2,
    0.1 + 0.2,
    1e-8,
    -1e-8,
    -0.0,
    5e-324,
    1e-300,
    1.7976931348623157e308,
    math.pi,
    1.0 / 3.0,
    float(np.float32(0.1)),
)


def bits(x: float) -> bytes:
    """The IEEE-754 bytes of *x*.  ``-0.0 == 0.0`` but their bits differ."""
    return struct.pack("<d", x)


def assert_float_identical(got: float, want: float, what: str) -> None:
    assert bits(got) == bits(want), (
        f"{what}: {got!r} is not bit-identical to {want!r} "
        f"({bits(got).hex()} vs {bits(want).hex()})"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: A small, real, ordered slice of the skull plus one vertebra.  The vertebra is
#: here on purpose: its display name ("T9") differs from its FMA preferred
#: label ("Ninth thoracic vertebra"), which is the case that proves the
#: provenance block carries information the display name does not.
REAL_DEFS = [
    {"name": "Mandible", "stl": "FMA52748", "color": 0xD4C8B0},
    {"name": "Frontal Bone", "stl": "FMA52734", "color": 0xD4C8B0, "opacity": 0.5},
    {"name": "Vomer", "stl": "FMA9710", "color": 0xCCCCCC},
    {"name": "L Parietal Bone", "stl": "FMA52789", "color": 0xD4C8B0},
    {"name": "T9", "stl": "FMA10014", "color": 0xC0C0C0},
]


@pytest.fixture(scope="module")
def real_scene():
    """A scene built by the real STL loader, so provenance is really populated.

    Module-scoped: the load is ~55 ms warm (the .npz mesh cache is what makes
    that affordable), and every test that needs it only reads geometry.  The
    per-test fixtures below deep-copy nothing -- tests that mutate materials get
    a fresh ``Scene`` wrapper around the same meshes and restore nothing, which
    is safe because every capture reads the *current* material state and every
    assertion is made against a capture taken in the same test.
    """
    result = load_stl_batch(REAL_DEFS, label="skullGroup")
    assert not result.failed, f"STL load failed for {result.failed}"
    assert len(result.meshes) == len(REAL_DEFS)
    scene = Scene()
    scene.add(result.group)
    return scene


def tiny_geometry() -> BufferGeometry:
    return BufferGeometry(
        positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
        normals=np.array([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32),
    )


def synthetic_scene() -> Scene:
    """A scene with nesting, duplicate names and procedural (unprovenanced) geometry."""
    scene = Scene()
    group = SceneNode(name="skullGroup")
    scene.add(group)
    for name, sid in (("Mandible", "FMA52748"), ("Temporal bone", "FMA52738"),
                      ("Temporal bone", "FMA52739")):
        node = SceneNode(name=name)
        node.mesh = MeshInstance(
            name=name, geometry=tiny_geometry(), material=Material(),
            source_id=sid, ontology_id=f"FMA:{sid[3:]}", preferred_label=f"{name} (FMA)",
        )
        group.add(node)
    # Procedural geometry: no anatomical referent, so no provenance.
    plane = SceneNode(name="scanPlane")
    plane.mesh = MeshInstance(name="scanPlane", geometry=tiny_geometry())
    scene.add(plane)
    return scene


def perturbed_camera(scene: Scene) -> Camera:
    """A camera placed the way ``tools/capture_golden.py`` places one.

    The eye position is a centroid-plus-radius computation over real mesh
    vertices, so its components are full-precision doubles with no round decimal
    form -- exactly the values that a naive ``f"{v:.6f}"`` serialiser would
    quietly truncate.
    """
    meshes = [n.mesh for _, n in ss.mesh_paths(scene)]
    pos = np.concatenate([m.positions.reshape(-1, 3) for m in meshes])
    centroid = pos.mean(axis=0).astype(np.float64)
    radius = float(np.linalg.norm(pos - centroid, axis=1).max())
    direction = np.array([-0.62, -0.68, 0.39])
    direction = direction / np.linalg.norm(direction)
    eye = centroid + direction * radius * 2.9

    cam = Camera(fov=37.123456789012345, near=0.10000000000000001, far=1234.5678901234567)
    cam.look_at(vec3(*eye), vec3(*centroid), vec3(0.0, 0.0, 1.0))
    cam.aspect = 1920.0 / 1080.0
    return cam


def perturbed_lights() -> LightSetup:
    lights = LightSetup()
    lights.ambient_color = vec3(*AWKWARD_FLOATS[:3])
    lights.light_dir = vec3(0.5773502691896258, -0.5773502691896258, 0.5773502691896258)
    lights.light_color = vec3(0.8, 0.75, 1.0 / 3.0)
    lights.point_light = PointLight(
        position=np.array([1e-8, -0.0, 123.456789], dtype=np.float64),
        color=(0.1, 0.2, 0.1 + 0.2),
        intensity=1.7976931348623157e308,
        range=5e-324,
        enabled=True,
    )
    return lights


def perturbed_state_manager() -> StateManager:
    sm = StateManager()
    for i, au in enumerate(("AU1", "AU2", "AU4", "AU5", "AU6", "AU9")):
        sm.face.set_au(au, AWKWARD_FLOATS[i] if 0.0 <= AWKWARD_FLOATS[i] <= 1.0 else 0.5)
    sm.face.head_yaw = -1e-8
    sm.face.head_pitch = 0.1 + 0.2
    sm.face.eye_color = "hazel"
    sm.face.current_expression = "surprise"
    sm.face.auto_blink = True
    sm.face.pupil_dilation = 1.0 / 3.0
    for i, name in enumerate(BodyState.POSE_FIELDS[:12]):
        setattr(sm.body, name, AWKWARD_FLOATS[i % len(AWKWARD_FLOATS)])
    sm.body.gender = 0.1
    sm.body.heart_rate = 72.0
    sm.body.auto_heartbeat = True
    # An int written onto a float field, as a Qt slider or a JSON preset does.
    sm.body.knee_r_flex = 1
    sm.target_au.set("AU12", 0.30000000000000004)
    sm.target_head.head_roll = -0.0
    sm.target_body.spine_flex = math.pi / 4.0
    return sm


def perturb_materials(scene: Scene) -> None:
    """Give every mesh a distinct, awkward material and a mix of render modes."""
    modes = [RenderMode.XRAY, RenderMode.SOLID, RenderMode.SOLID,
             RenderMode.HOLOGRAM, RenderMode.SOLID]
    for i, (_, node) in enumerate(sorted(ss.mesh_paths(scene))):
        mesh = node.mesh
        mesh.material.render_mode = modes[i % len(modes)]
        mesh.material.color = (0.1, 1.0 / 3.0, AWKWARD_FLOATS[i % len(AWKWARD_FLOATS)]
                               if 0.0 <= AWKWARD_FLOATS[i % len(AWKWARD_FLOATS)] <= 1.0
                               else 0.25)
        mesh.material.opacity = 1.0 - i * 1e-8
        mesh.material.shininess = 15.0 + i * 0.1
        mesh.material.emissive = (0.0, -0.0, 1e-8)
        mesh.material.double_sided = bool(i % 2)
        mesh.material.transparent = mesh.material.opacity < 1.0
        mesh.material.depth_write = not bool(i % 3)
        mesh.material.wireframe_color = None if i % 2 else (0.2, 0.3, 0.4)
        mesh.visible = i != 1
        node.visible = i != 2


class FakeRenderer:
    """Stands in for GLRenderer's serialisable surface, with no GL context.

    Only the five attributes the binding layer reads or writes.  A real
    GLRenderer needs a live context to construct usefully; the pixel-level
    proof that a captured state reproduces a render is a separate GL script
    (see the track report), and what is tested here is the serialisation.
    """

    CLEAR_COLOR = (0.12, 0.12, 0.15, 1.0)

    def __init__(self, width=1920, height=1080):
        self._width = width
        self._height = height
        self._bg_color_dirty = False
        self.clip_plane_enabled = False
        self.clip_plane = (1.0, 0.0, 0.0, 0.0)
        self.scene_transform = None

    def set_clip_plane(self, normal, offset):
        self.clip_plane_enabled = True
        self.clip_plane = (float(normal[0]), float(normal[1]), float(normal[2]),
                           float(offset))

    def clear_clip_plane(self):
        self.clip_plane_enabled = False


class FakeVisibility:
    """The VisibilityManager surface the binding layer uses."""

    def __init__(self, toggles):
        self._toggles = dict(toggles)

    def get_toggle_names(self):
        return list(self._toggles)

    def is_visible(self, name):
        return self._toggles[name]

    def set_visible(self, name, visible):
        self._toggles[name] = bool(visible)


def perturbed_renderer() -> FakeRenderer:
    r = FakeRenderer()
    r.CLEAR_COLOR = (0.1, 0.2, 0.1 + 0.2, 1.0)
    r.set_clip_plane((-1.0, 0.0, 0.0), -12.345678901234567)
    r.scene_transform = np.arange(16, dtype=np.float64).reshape(4, 4) / 7.0
    return r


def full_state(scene: Scene) -> ss.SceneState:
    """A realistic, fully-populated state over *scene*."""
    perturb_materials(scene)
    return ss.capture_scene_state(
        scene=scene,
        camera=perturbed_camera(scene),
        lights=perturbed_lights(),
        renderer=perturbed_renderer(),
        state=perturbed_state_manager(),
        visibility=FakeVisibility({"skull": True, "muscles": False, "organs": True}),
        tier=3,
        skull_mode="separated",
        stl_dir=Path("assets/stl"),
        gender_applied=0.1,
        alignment={"scale": 1.14, "offset_x": -0.2, "offset_y": -10.6,
                   "offset_z": 9.5, "rot_x": 88.5},
    )


# ---------------------------------------------------------------------------
# Round-trip exactness
# ---------------------------------------------------------------------------


def test_round_trip_is_byte_identical_real_scene(real_scene):
    state = full_state(real_scene)
    text1 = ss.dumps(state)
    reloaded = ss.loads(text1)
    text2 = ss.dumps(reloaded)
    assert text2 == text1, "save -> load -> save is not byte-identical"
    assert reloaded == state, "reloaded state does not compare equal to the original"
    assert ss.payload_digest(reloaded) == ss.payload_digest(state)


def test_round_trip_is_byte_identical_synthetic_scene():
    state = full_state(synthetic_scene())
    text1 = ss.dumps(state)
    text2 = ss.dumps(ss.loads(text1))
    assert text2 == text1


def test_round_trip_is_idempotent_over_repeated_cycles(real_scene):
    """Three cycles, not one: a one-cycle test passes even if the codec is
    quietly normalising values, as long as it normalises to a fixed point after
    the first write."""
    text = ss.dumps(full_state(real_scene))
    for _ in range(3):
        text_next = ss.dumps(ss.loads(text))
        assert text_next == text
        text = text_next


@pytest.mark.parametrize("value", AWKWARD_FLOATS, ids=lambda v: repr(v))
def test_awkward_float_survives_bit_exactly(value):
    """Every awkward value, in every float-typed position that can hold it."""
    scene = synthetic_scene()
    cam = Camera()
    cam.look_at(vec3(value, -value, value), vec3(value, value, -value),
                vec3(0.0, 0.0, 1.0))
    cam.fov = value
    cam.near = value
    cam.far = value
    cam.aspect = value
    lights = LightSetup()
    lights.ambient_color = vec3(value, value, value)
    sm = StateManager()
    sm.body.spine_flex = value
    sm.face.head_yaw = value

    state = ss.capture_scene_state(scene=scene, camera=cam, lights=lights, state=sm,
                                   viewport=(64, 64))
    reloaded = ss.loads(ss.dumps(state))

    assert_float_identical(reloaded.camera.position[0], value, "camera.position[0]")
    assert_float_identical(reloaded.camera.position[1], -value, "camera.position[1]")
    assert_float_identical(reloaded.camera.fov_deg, value, "camera.fov_deg")
    assert_float_identical(reloaded.camera.near, value, "camera.near")
    assert_float_identical(reloaded.camera.far, value, "camera.far")
    assert_float_identical(reloaded.camera.aspect, value, "camera.aspect")
    assert_float_identical(reloaded.lighting.ambient_color[2], value, "ambient[2]")
    assert_float_identical(reloaded.body["spine_flex"], value, "body.spine_flex")
    assert_float_identical(reloaded.face["head_yaw"], value, "face.head_yaw")


def test_negative_zero_keeps_its_sign_through_the_file():
    """``-0.0 == 0.0``, so only a bit-level check can see this. The sign is not
    cosmetic: a -0.0 in a light direction or a normal flips a hemisphere."""
    scene = synthetic_scene()
    cam = Camera()
    cam.look_at(vec3(-0.0, 0.0, -0.0), vec3(0.0, -0.0, 0.0), vec3(0.0, 0.0, 1.0))
    state = ss.capture_scene_state(scene=scene, camera=cam, lights=LightSetup(),
                                   viewport=(8, 8))
    text = ss.dumps(state)
    assert '"position": [\n      -0.0,' in text, "the minus sign is not in the file"
    reloaded = ss.loads(text)
    assert_float_identical(reloaded.camera.position[0], -0.0, "position[0]")
    assert_float_identical(reloaded.camera.position[2], -0.0, "position[2]")
    assert_float_identical(reloaded.camera.target[1], -0.0, "target[1]")


def test_int_on_a_float_field_normalises_once_not_twice():
    """A Qt slider writing ``1`` onto a float field must not make the first
    save differ from the second."""
    sm = StateManager()
    sm.body.knee_r_flex = 1
    state = ss.capture_scene_state(scene=synthetic_scene(), camera=Camera(),
                                   lights=LightSetup(), state=sm, viewport=(8, 8))
    text = ss.dumps(state)
    assert json.loads(text)["body"]["knee_r_flex"] == 1.0
    assert isinstance(json.loads(text)["body"]["knee_r_flex"], float)
    assert ss.dumps(ss.loads(text)) == text


def test_non_finite_values_are_refused_at_capture():
    cam = Camera()
    cam.look_at(vec3(float("nan"), 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0))
    with pytest.raises(ss.SceneStateFormatError, match="not finite"):
        ss.capture_scene_state(scene=synthetic_scene(), camera=cam,
                               lights=LightSetup(), viewport=(8, 8))
    cam2 = Camera()
    cam2.fov = float("inf")
    with pytest.raises(ss.SceneStateFormatError, match="not finite"):
        ss.capture_scene_state(scene=synthetic_scene(), camera=cam2,
                               lights=LightSetup(), viewport=(8, 8))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_structure_order_does_not_affect_the_file():
    """Two scenes with the same structures added in a different order must
    serialise identically -- otherwise a state file's diff is dominated by
    load-order noise."""
    scene_a = Scene()
    scene_b = Scene()
    group_a = SceneNode(name="g")
    group_b = SceneNode(name="g")
    scene_a.add(group_a)
    scene_b.add(group_b)
    names = ["Mandible", "Vomer", "Frontal bone"]
    for name in names:
        n = SceneNode(name=name)
        n.mesh = MeshInstance(name=name, geometry=tiny_geometry(), source_id="FMA1")
        group_a.add(n)
    for name in reversed(names):
        n = SceneNode(name=name)
        n.mesh = MeshInstance(name=name, geometry=tiny_geometry(), source_id="FMA1")
        group_b.add(n)

    kw = dict(camera=Camera(), lights=LightSetup(), viewport=(8, 8))
    a = ss.capture_scene_state(scene=scene_a, **kw)
    b = ss.capture_scene_state(scene=scene_b, **kw)
    assert ss.canonical_json(a) == ss.canonical_json(b)
    assert a == b


def test_metadata_is_excluded_from_equality_and_from_the_canonical_payload():
    state = full_state(synthetic_scene())
    other = replace(state, metadata={"generated_utc": "1999-01-01T00:00:00Z",
                                     "generator": "something else",
                                     "comparable": False})
    assert other == state, "metadata must not participate in equality"
    assert ss.canonical_json(other) == ss.canonical_json(state)
    assert ss.payload_digest(other) == ss.payload_digest(state)
    # ...but it is still in the file, and still round-trips verbatim.
    assert "generated_utc" in json.loads(ss.dumps(state))["metadata"]
    assert ss.dumps(ss.loads(ss.dumps(other))) == ss.dumps(other)


def test_file_is_sorted_and_human_readable():
    text = ss.dumps(full_state(synthetic_scene()))
    top = list(json.loads(text).keys())
    assert top == sorted(top), "top-level keys are not sorted"
    lines = text.splitlines()
    assert lines[0] == "{" and text.endswith("}\n")
    assert any(line.startswith('  "schema_version"') for line in lines)
    # Structures in scene-graph-path order, which is what makes a diff readable.
    paths = [s["path"] for s in json.loads(text)["structures"]]
    assert paths == sorted(paths)


# ---------------------------------------------------------------------------
# Negative controls: a difference in ONE field must be detected
# ---------------------------------------------------------------------------

def _mutations():
    """(label, mutate) pairs, each changing exactly one field of a state."""
    return [
        ("camera.position",
         lambda s: replace(s, camera=replace(s.camera, position=(1.0, 2.0, 3.0)))),
        ("camera.position by 1 ulp",
         lambda s: replace(s, camera=replace(
             s.camera, position=(math.nextafter(s.camera.position[0], math.inf),)
                                + tuple(s.camera.position[1:])))),
        ("camera.fov_deg",
         lambda s: replace(s, camera=replace(s.camera, fov_deg=s.camera.fov_deg + 1e-9))),
        ("camera.aspect",
         lambda s: replace(s, camera=replace(s.camera, aspect=1.0))),
        ("viewport.width",
         lambda s: replace(s, viewport=replace(s.viewport, width=s.viewport.width + 1))),
        ("lighting.light_dir",
         lambda s: replace(s, lighting=replace(s.lighting, light_dir=(0.0, 0.0, 1.0)))),
        ("point_light.enabled",
         lambda s: replace(s, lighting=replace(
             s.lighting, point_light=replace(s.lighting.point_light, enabled=False)))),
        ("point_light dropped",
         lambda s: replace(s, lighting=replace(s.lighting, point_light=None))),
        ("render.global_mode",
         lambda s: replace(s, render=replace(s.render, global_mode="SEPIA"))),
        ("render.clear_color",
         lambda s: replace(s, render=replace(s.render, clear_color=(0.0, 0.0, 0.0, 1.0)))),
        ("clip_plane.offset",
         lambda s: replace(s, render=replace(
             s.render, clip_plane=replace(s.render.clip_plane,
                                          offset=s.render.clip_plane.offset + 1e-9)))),
        ("clip_plane.enabled",
         lambda s: replace(s, render=replace(
             s.render, clip_plane=replace(s.render.clip_plane, enabled=False)))),
        ("scene_transform dropped",
         lambda s: replace(s, render=replace(s.render, scene_transform=None))),
        ("structure visibility",
         lambda s: replace(s, structures=(replace(s.structures[0], visible=not s.structures[0].visible),)
                           + s.structures[1:])),
        ("structure node visibility",
         lambda s: replace(s, structures=(replace(s.structures[0],
                                                  node_visible=not s.structures[0].node_visible),)
                           + s.structures[1:])),
        ("structure render mode override",
         lambda s: replace(s, structures=(replace(s.structures[0], render_mode="THERMAL"),)
                           + s.structures[1:])),
        ("structure opacity by 1e-8",
         lambda s: replace(s, structures=(replace(
             s.structures[0],
             material=replace(s.structures[0].material,
                              opacity=s.structures[0].material.opacity - 1e-8)),)
             + s.structures[1:])),
        ("structure colour",
         lambda s: replace(s, structures=(replace(
             s.structures[0],
             material=replace(s.structures[0].material, color=(0.9, 0.9, 0.9))),)
             + s.structures[1:])),
        ("structure provenance",
         lambda s: replace(s, structures=(replace(
             s.structures[0],
             provenance=replace(s.structures[0].provenance, ontology_id="FMA:1")),)
             + s.structures[1:])),
        ("structure dropped",
         lambda s: replace(s, structures=s.structures[1:])),
        ("face AU",
         lambda s: replace(s, face={**s.face, "AU12": s.face["AU12"] + 1e-8})),
        ("face expression name",
         lambda s: replace(s, face={**s.face, "current_expression": "neutral"})),
        ("face auto toggle",
         lambda s: replace(s, face={**s.face, "auto_blink": not s.face["auto_blink"]})),
        ("body pose DOF",
         lambda s: replace(s, body={**s.body,
                                    "spine_flex": s.body["spine_flex"] + 1e-8})),
        ("body gender",
         lambda s: replace(s, body={**s.body, "gender": 0.9})),
        ("target AU",
         lambda s: replace(s, target_au={**s.target_au,
                                         "AU12": s.target_au["AU12"] + 1e-9})),
        ("target body",
         lambda s: replace(s, target_body={**s.target_body,
                                           "spine_flex": s.target_body["spine_flex"] * 2})),
        ("morph.gender_applied",
         lambda s: replace(s, morph=replace(s.morph, gender_applied=0.9))),
        ("morph.alignment",
         lambda s: replace(s, morph=replace(s.morph, alignment=(("scale", 1.15),)))),
        ("assets.tier",
         lambda s: replace(s, assets=replace(s.assets, tier=1))),
        ("assets.skull_mode",
         lambda s: replace(s, assets=replace(s.assets, skull_mode="original"))),
        ("assets.layer_visibility",
         lambda s: replace(s, assets=replace(s.assets,
                                             layer_visibility=(("skull", False),)))),
        ("config.digest",
         lambda s: replace(s, config=replace(s.config, digest="0" * 64))),
    ]


@pytest.mark.parametrize("label,mutate", _mutations(), ids=[m[0] for m in _mutations()])
def test_single_field_difference_is_detected(label, mutate):
    """The negative control for every equality assertion in this module.

    Without this, ``reloaded == original`` could pass on a state object whose
    ``__eq__`` ignored half its fields, and the round-trip tests would prove
    nothing at all.
    """
    base = full_state(synthetic_scene())
    changed = mutate(base)
    assert changed != base, f"{label}: mutated state still compares equal"
    assert ss.canonical_json(changed) != ss.canonical_json(base), (
        f"{label}: mutation is invisible in the canonical payload"
    )
    assert ss.payload_digest(changed) != ss.payload_digest(base), (
        f"{label}: mutation does not change the payload digest"
    )


def test_identical_states_are_equal_so_the_control_is_meaningful():
    """The positive half of the control: the comparison used above is capable of
    returning equal, so its failures above are informative."""
    a = full_state(synthetic_scene())
    b = ss.loads(ss.dumps(a))
    assert a == b
    assert ss.payload_digest(a) == ss.payload_digest(b)


# ---------------------------------------------------------------------------
# Version policy
# ---------------------------------------------------------------------------


def test_current_version_is_written():
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    assert payload["schema_version"] == ss.SCHEMA_VERSION == 1


def test_newer_version_is_refused_with_a_clear_message():
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["schema_version"] = ss.SCHEMA_VERSION + 1
    with pytest.raises(ss.SceneStateVersionError) as exc:
        ss.loads(json.dumps(payload))
    msg = str(exc.value)
    assert "newer than this build" in msg
    assert str(ss.SCHEMA_VERSION + 1) in msg
    assert "Refusing to load" in msg


def test_older_version_without_a_migration_is_refused():
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["schema_version"] = 0
    with pytest.raises(ss.SceneStateVersionError) as exc:
        ss.loads(json.dumps(payload))
    msg = str(exc.value)
    assert "no migration is registered from version 0 to 1" in msg
    assert "rather than guess" in msg


def test_older_version_with_a_migration_loads(monkeypatch):
    """The migration chain, exercised end to end.

    Version 0 here is a stand-in for a future older format: the payload has its
    camera under a legacy key and no morph block, and the registered migration
    is what makes it loadable.  There are no released schema versions below 1,
    so this is the mechanism being tested, not a historical format.
    """
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    legacy = dict(payload)
    legacy["schema_version"] = 0
    legacy["camera_v0"] = legacy.pop("camera")
    legacy.pop("morph")

    def upgrade_0_to_1(p):
        p = dict(p)
        p["camera"] = p.pop("camera_v0")
        p["morph"] = {"gender_applied": None, "alignment": None}
        return p

    monkeypatch.setitem(codec.MIGRATIONS, 0, upgrade_0_to_1)
    state = ss.loads(json.dumps(legacy))
    assert state.camera == ss.loads(json.dumps(payload)).camera
    assert state.morph.gender_applied is None
    # And the migrated state is a first-class version-1 state.
    assert json.loads(ss.dumps(state))["schema_version"] == 1


def test_missing_schema_version_is_refused():
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    del payload["schema_version"]
    with pytest.raises(ss.SceneStateFormatError, match="no schema_version"):
        ss.loads(json.dumps(payload))


@pytest.mark.parametrize("bad", ["1", 1.0, True, None, [1]])
def test_non_integer_schema_version_is_refused(bad):
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["schema_version"] = bad
    with pytest.raises(ss.SceneStateFormatError, match="schema_version"):
        ss.loads(json.dumps(payload))


def test_malformed_and_truncated_files_are_refused():
    with pytest.raises(ss.SceneStateFormatError, match="not valid JSON"):
        ss.loads("{not json")
    with pytest.raises(ss.SceneStateFormatError, match="JSON object at the top level"):
        ss.loads("[1, 2, 3]")
    text = ss.dumps(full_state(synthetic_scene()))
    with pytest.raises(ss.SceneStateFormatError):
        ss.loads(text[: len(text) // 2])


def test_unknown_and_missing_fields_are_refused_not_defaulted():
    """A dropped field must fail loudly.  Silently defaulting it is how a state
    file starts describing a render it does not produce."""
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["camera"].pop("far")
    with pytest.raises(ss.SceneStateFormatError, match="missing key"):
        ss.loads(json.dumps(payload))

    payload2 = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload2["camera"]["zoom"] = 2.0
    with pytest.raises(ss.SceneStateFormatError, match="unexpected key"):
        ss.loads(json.dumps(payload2))

    payload3 = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload3["body"].pop("gender")
    with pytest.raises(ss.SceneStateFormatError, match="missing field"):
        ss.loads(json.dumps(payload3))


def test_unknown_render_mode_is_refused_and_lists_the_valid_ones():
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["render"]["global_mode"] = "GOURAUD"
    with pytest.raises(ss.SceneStateFormatError) as exc:
        ss.loads(json.dumps(payload))
    assert "not a RenderMode" in str(exc.value)
    assert "SOLID" in str(exc.value)


def test_every_render_mode_round_trips():
    """All 16 modes, so a mode added to the enum without a shader is still
    representable and a mode removed from it is caught here."""
    names = ss.render_mode_names()
    assert len(names) == 16
    scene = synthetic_scene()
    entries = sorted(ss.mesh_paths(scene))
    for name in names:
        for _, node in entries:
            node.mesh.material.render_mode = RenderMode[name]
        state = ss.capture_scene_state(scene=scene, camera=Camera(),
                                       lights=LightSetup(), viewport=(8, 8))
        assert state.render.global_mode == name
        reloaded = ss.loads(ss.dumps(state))
        assert reloaded == state
        assert set(reloaded.structure_modes().values()) == {name}


# ---------------------------------------------------------------------------
# Config fingerprint
# ---------------------------------------------------------------------------


def test_capture_records_a_real_config_fingerprint(real_scene):
    state = full_state(real_scene)
    fp = state.config
    assert fp.algorithm == "sha256"
    assert len(fp.digest) == 64
    assert fp.file_count > 20, f"only {fp.file_count} config files fingerprinted"
    assert len(fp.fma_labels_digest) == 64, "fma_labels.json was not fingerprinted"
    assert fp.root == "config"
    # Stable across calls, and it is a digest of the real files.
    assert ss.fingerprint_configs() == fp


def test_matching_configs_load_without_a_warning():
    text = ss.dumps(full_state(synthetic_scene()))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ss.ConfigFingerprintMismatch)
        ss.loads(text)


def test_config_mismatch_warns_loudly_but_still_loads(caplog):
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["config"]["digest"] = "d" * 64
    text = json.dumps(payload)

    with caplog.at_level("WARNING"):
        with pytest.warns(ss.ConfigFingerprintMismatch) as record:
            state = ss.loads(text)

    # It loaded: re-rendering against updated configs is legitimate.
    assert state.config.digest == "d" * 64
    message = str(record[0].message)
    assert "anatomy configs have changed" in message
    assert "config digest dddddddddddd" in message
    assert "may not reproduce that figure" in message
    # And the same thing reached the log, which is what a user actually sees.
    assert any("anatomy configs have changed" in r.message for r in caplog.records)


def test_config_check_can_be_turned_off():
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["config"]["digest"] = "e" * 64
    with warnings.catch_warnings():
        warnings.simplefilter("error", ss.ConfigFingerprintMismatch)
        ss.loads(json.dumps(payload), check_config=False)


def test_an_empty_fingerprint_is_reported_as_uncheckable():
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["config"]["digest"] = ""
    payload["config"]["fma_labels_digest"] = ""
    payload["config"]["file_count"] = 0
    with pytest.warns(ss.ConfigFingerprintMismatch, match="carries no config fingerprint"):
        ss.loads(json.dumps(payload))


def test_editing_a_config_changes_the_digest(tmp_path):
    """The fingerprint must actually track file contents, including in
    subdirectories, and must not be fooled by the stat-signature cache."""
    root = tmp_path / "config"
    (root / "muscles").mkdir(parents=True)
    (root / "a.json").write_text('{"x": 1}')
    (root / "muscles" / "b.json").write_text('{"y": 2}')
    first = ss.fingerprint_configs(root)
    assert first.file_count == 2

    confighash.clear_cache()
    assert ss.fingerprint_configs(root).digest == first.digest, "digest is not stable"

    (root / "muscles" / "b.json").write_text('{"y": 3}')
    confighash.clear_cache()
    second = ss.fingerprint_configs(root)
    assert second.digest != first.digest, "a config edit did not change the digest"
    assert second.file_count == 2

    (root / "c.json").write_text("{}")
    confighash.clear_cache()
    third = ss.fingerprint_configs(root)
    assert third.file_count == 3
    assert third.digest != second.digest

    detail = ss.describe_mismatch(first, third)
    assert detail is not None
    assert "file count 2 -> 3" in detail


def test_fingerprint_ignores_absolute_location(tmp_path):
    """Two checkouts of the same commit must produce the same digest, so the
    digest cannot depend on where the repo lives."""
    a = tmp_path / "one" / "config"
    b = tmp_path / "two" / "config"
    for root in (a, b):
        root.mkdir(parents=True)
        (root / "x.json").write_text('{"k": [1, 2, 3]}')
    confighash.clear_cache()
    fa = ss.fingerprint_configs(a)
    confighash.clear_cache()
    fb = ss.fingerprint_configs(b)
    assert fa == fb


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_every_visible_structure_carries_matching_fma_provenance(real_scene):
    """The citability claim, checked against the crosswalk itself."""
    labels = load_fma_labels()
    state = full_state(real_scene)
    visible = state.visible_structures
    assert visible, "the fixture scene has no visible structures to check"

    checked = 0
    for s in visible:
        sid = s.provenance.source_id
        assert sid, f"{s.path}: a loaded anatomical structure has no source_id"
        assert sid in labels, f"{s.path}: source_id {sid} is not in fma_labels.json"
        entry = labels[sid]
        assert s.provenance.ontology_id == f"FMA:{entry['fma_id']}", (
            f"{s.path}: ontology_id {s.provenance.ontology_id!r} does not match "
            f"fma_labels.json ({entry['fma_id']})"
        )
        assert s.provenance.preferred_label == entry["preferred_label"], (
            f"{s.path}: preferred_label {s.provenance.preferred_label!r} != "
            f"{entry['preferred_label']!r}"
        )
        checked += 1
    assert checked == len(visible)
    assert not ss.structures_missing_provenance(state)


def test_provenance_survives_the_round_trip(real_scene):
    state = full_state(real_scene)
    reloaded = ss.loads(ss.dumps(state))
    before = {s.path: s.provenance for s in state.structures}
    after = {s.path: s.provenance for s in reloaded.structures}
    assert after == before


def test_preferred_label_differs_from_the_display_name_somewhere(real_scene):
    """If display name and preferred label were always the same, the provenance
    block would be decoration.  T9 -> 'Ninth thoracic vertebra' is the case."""
    state = full_state(real_scene)
    divergent = [s for s in state.structures
                 if s.provenance.preferred_label and s.provenance.preferred_label != s.name]
    assert divergent, "no structure has a preferred label distinct from its display name"
    t9 = [s for s in state.structures if s.provenance.source_id == "FMA10014"]
    assert t9 and t9[0].name == "T9"
    assert t9[0].provenance.preferred_label == "Ninth thoracic vertebra"


def test_procedural_geometry_is_not_given_a_fake_id():
    state = full_state(synthetic_scene())
    plane = state.structure("/scanPlane")
    assert plane is not None
    assert plane.provenance.source_id == ""
    assert plane.provenance.ontology_id == ""
    assert not plane.provenance.is_anatomical
    # ...and it is not reported as missing provenance, because it has none to miss.
    assert not ss.structures_missing_provenance(state, visible_only=False)


def test_a_structure_with_an_id_but_no_ontology_term_is_reported():
    """The negative control for the provenance check: an unresolved structure
    must be detectable, or the test above could pass on an empty scene."""
    state = full_state(synthetic_scene())
    # Pick a structure that actually has a source_id -- clearing the ontology
    # term on the procedural scan plane would prove nothing, since it has no id.
    target = next(i for i, s in enumerate(state.structures) if s.provenance.source_id)
    broken = replace(
        state,
        structures=state.structures[:target]
        + (replace(state.structures[target],
                   provenance=replace(state.structures[target].provenance,
                                      ontology_id="")),)
        + state.structures[target + 1:],
    )
    missing = ss.structures_missing_provenance(broken, visible_only=False)
    assert len(missing) == 1
    assert missing[0].provenance.source_id
    assert missing[0].path == state.structures[target].path


# ---------------------------------------------------------------------------
# Paths and structure identity
# ---------------------------------------------------------------------------


def test_same_named_siblings_get_distinct_stable_paths():
    state = full_state(synthetic_scene())
    paths = [s.path for s in state.structures]
    assert paths == sorted(paths)
    assert "/skullGroup/Temporal bone[0]" in paths
    assert "/skullGroup/Temporal bone[1]" in paths
    # A unique name gets no index suffix.
    assert "/skullGroup/Mandible" in paths
    assert len(set(paths)) == len(paths)


def test_paths_escape_separators_in_names():
    scene = Scene()
    node = SceneNode(name="a/b%c")
    node.mesh = MeshInstance(name="a/b%c", geometry=tiny_geometry())
    scene.add(node)
    state = ss.capture_scene_state(scene=scene, camera=Camera(), lights=LightSetup(),
                                   viewport=(8, 8))
    assert [s.path for s in state.structures] == ["/a%2Fb%25c"]
    assert ss.dumps(ss.loads(ss.dumps(state))) == ss.dumps(state)


def test_duplicate_paths_are_refused():
    payload = json.loads(ss.dumps(full_state(synthetic_scene())))
    payload["structures"].append(dict(payload["structures"][0]))
    with pytest.raises(ss.SceneStateFormatError, match="duplicate path"):
        ss.loads(json.dumps(payload))


# ---------------------------------------------------------------------------
# Apply: a loaded state puts the scene back
# ---------------------------------------------------------------------------


def test_apply_then_recapture_reproduces_the_state(real_scene):
    """State-level reproducibility: capture, wreck the live objects, apply the
    file, re-capture -- and the canonical payload must be identical."""
    original = full_state(real_scene)
    text = ss.dumps(original)

    # Wreck everything the state is supposed to restore.
    camera = Camera()
    camera.look_at(vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))
    camera.fov, camera.near, camera.far, camera.aspect = 90.0, 5.0, 50.0, 0.5
    lights = LightSetup()
    renderer = FakeRenderer(width=100, height=100)
    sm = StateManager()
    visibility = FakeVisibility({"skull": False, "muscles": True, "organs": False})
    for _, node in ss.mesh_paths(real_scene):
        node.mesh.material = Material(color=(0.0, 0.0, 0.0), opacity=0.25,
                                      render_mode=RenderMode.PEN_INK)
        node.mesh.visible = True
        node.visible = True

    loaded = ss.loads(text)
    report = ss.apply_scene_state(
        loaded, scene=real_scene, camera=camera, lights=lights, renderer=renderer,
        state_manager=sm, visibility=visibility, strict=True,
    )
    assert report.ok, report.summary()
    assert report.applied == len(original.structures)
    assert report.camera_applied and report.lighting_applied
    assert report.render_applied and report.animation_applied

    recaptured = ss.capture_scene_state(
        scene=real_scene, camera=camera, lights=lights, renderer=renderer, state=sm,
        visibility=visibility, tier=original.assets.tier,
        skull_mode=original.assets.skull_mode, stl_dir=Path("assets/stl"),
        gender_applied=original.morph.gender_applied,
        alignment=dict(original.morph.alignment),
        viewport=(original.viewport.width, original.viewport.height),
    )
    assert ss.canonical_json(recaptured) == ss.canonical_json(original)
    assert recaptured == original


def test_apply_restores_camera_projection_not_just_the_view():
    """A restored fov/near/far/aspect must reach the projection matrix.  Camera
    caches it behind a dirty flag, so this is the test that would catch a
    restore that only moved the eye."""
    cam_src = Camera(fov=37.5, near=0.25, far=555.0)
    cam_src.aspect = 1.6
    cam_src.look_at(vec3(10.0, -20.0, 30.0), vec3(0.0, 0.0, -50.0), vec3(0.0, 0.0, 1.0))
    want_proj = cam_src.get_projection_matrix().copy()
    want_view = cam_src.get_view_matrix().copy()

    state = ss.capture_scene_state(scene=synthetic_scene(), camera=cam_src,
                                   lights=LightSetup(), viewport=(16, 9))
    dst = Camera()
    dst.get_projection_matrix()  # prime the cache, so a missing dirty flag shows
    dst.get_view_matrix()
    ss.apply_scene_state(ss.loads(ss.dumps(state)), camera=dst)

    assert np.array_equal(dst.get_projection_matrix(), want_proj)
    assert np.array_equal(dst.get_view_matrix(), want_view)


def test_apply_restores_the_renderer_globals():
    state = full_state(synthetic_scene())
    dst = FakeRenderer()
    ss.apply_scene_state(ss.loads(ss.dumps(state)), renderer=dst)
    assert tuple(dst.CLEAR_COLOR) == state.render.clear_color
    assert dst._bg_color_dirty is True
    assert dst.clip_plane_enabled is state.render.clip_plane.enabled
    assert dst.clip_plane[:3] == state.render.clip_plane.normal
    assert dst.clip_plane[3] == state.render.clip_plane.offset
    assert np.array_equal(dst.scene_transform,
                          np.array([list(r) for r in state.render.scene_transform]))

    # A state with no clip plane must clear one that is set.
    off = replace(state, render=replace(state.render,
                                        clip_plane=replace(state.render.clip_plane,
                                                           enabled=False),
                                        scene_transform=None))
    ss.apply_scene_state(off, renderer=dst)
    assert dst.clip_plane_enabled is False
    assert dst.scene_transform is None


def test_apply_restores_point_light_presence_and_absence():
    state = full_state(synthetic_scene())
    lights = LightSetup()
    ss.apply_scene_state(state, lights=lights)
    assert lights.point_light is not None
    assert lights.point_light.enabled is True
    assert_float_identical(float(lights.point_light.position[0]), 1e-8, "pl.position[0]")

    none_state = replace(state, lighting=replace(state.lighting, point_light=None))
    ss.apply_scene_state(none_state, lights=lights)
    assert lights.point_light is None


def test_apply_reports_missing_and_extra_structures_without_raising():
    state = full_state(synthetic_scene())
    smaller = synthetic_scene()
    # Remove one node so the scene no longer matches the state.
    group = smaller.children[0]
    group.remove(group.children[0])

    report = ss.apply_scene_state(state, scene=smaller)
    assert not report.ok
    assert len(report.missing_paths) == 1
    assert report.missing_paths[0].startswith("/skullGroup/")
    assert report.applied == len(state.structures) - 1
    assert "absent from the scene" in report.summary()

    with pytest.raises(ss.SceneStateError, match="did not reproduce its structure set"):
        ss.apply_scene_state(state, scene=smaller, strict=True)


def test_apply_reports_structures_the_state_does_not_know_about():
    state = full_state(synthetic_scene())
    bigger = synthetic_scene()
    extra = SceneNode(name="newLayer")
    extra.mesh = MeshInstance(name="newLayer", geometry=tiny_geometry())
    bigger.add(extra)
    report = ss.apply_scene_state(state, scene=bigger)
    assert report.extra_paths == ("/newLayer",)
    assert not report.ok
    assert "absent from the state file" in report.summary()


def test_apply_reports_unregistered_layer_toggles():
    state = full_state(synthetic_scene())
    vis = FakeVisibility({"skull": False})
    report = ss.apply_scene_state(state, visibility=vis)
    assert vis.is_visible("skull") is True
    assert any("not registered in this scene" in n for n in report.notes)


def test_apply_restores_animation_state():
    state = full_state(synthetic_scene())
    sm = StateManager()
    ss.apply_scene_state(ss.loads(ss.dumps(state)), state_manager=sm)
    assert sm.face.current_expression == "surprise"
    assert sm.face.eye_color == "hazel"
    assert sm.face.auto_blink is True
    assert isinstance(sm.face.auto_blink, bool)
    assert_float_identical(sm.face.head_yaw, -1e-8, "face.head_yaw")
    assert_float_identical(sm.body.gender, 0.1, "body.gender")
    assert sm.body.auto_heartbeat is True
    assert_float_identical(sm.target_head.head_roll, -0.0, "target_head.head_roll")
    assert_float_identical(sm.target_body.spine_flex, math.pi / 4.0, "target spine_flex")


def test_apply_with_no_targets_is_a_no_op_that_says_so():
    report = ss.apply_scene_state(full_state(synthetic_scene()))
    assert report.applied == 0
    assert report.ok
    assert "camera not applied" in report.summary()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def test_save_and_load_a_file(tmp_path, real_scene):
    state = full_state(real_scene)
    path = ss.save(state, tmp_path / "figure_3a.state.json")
    assert path.is_file()
    assert not (tmp_path / "figure_3a.state.json.tmp").exists(), "temp file left behind"
    assert path.read_text(encoding="utf-8") == ss.dumps(state)

    reloaded = ss.load(path)
    assert reloaded == state
    # Re-saving the loaded state must not change the file on disk.
    again = tmp_path / "again.json"
    ss.save(reloaded, again)
    assert again.read_bytes() == path.read_bytes()


def test_loading_a_missing_file_says_which_one(tmp_path):
    with pytest.raises(ss.SceneStateFormatError, match="no such state file"):
        ss.load(tmp_path / "nope.json")


def test_capture_works_with_only_the_required_objects():
    """A script-built scene has no renderer, no StateManager and no visibility
    manager; the state must still be complete and honest about what is absent."""
    state = ss.capture_scene_state(scene=synthetic_scene(), camera=Camera(),
                                   lights=LightSetup(), viewport=(320, 240))
    assert state.assets.tier is None
    assert state.assets.skull_mode is None
    assert state.morph.gender_applied is None
    assert state.morph.alignment is None
    assert state.assets.layer_visibility == ()
    assert state.viewport == ss.ViewportState(320, 240)
    assert state.face["AU1"] == 0.0
    assert ss.dumps(ss.loads(ss.dumps(state))) == ss.dumps(state)
