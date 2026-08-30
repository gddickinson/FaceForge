"""A saved SceneState must reproduce the *pixels*, not merely compare equal.

Why this file exists separately from ``test_scene_state.py``
------------------------------------------------------------
``test_scene_state.py`` proves the format is exact: byte-identical
save/load/save, float-exact round-trip, and 33 single-field negative
controls.  All of that is about the *file*.  None of it proves that
applying a state actually puts the renderer back into the same visual
state -- a binding could silently drop a field and every one of those
tests would still pass.

The proof was originally a one-off script (``tools/verify_state_pixels.py``)
run by hand.  A reproducibility guarantee that is not in the suite is not a
guarantee: the first change that breaks state application would go
unnoticed.  This module runs the same comparison as a test.

It is marked ``slow``: it needs a GL context and renders a 230,490-triangle
scene twice through a CPU rasteriser.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def gl():
    """Acquire a headless GL context, or skip.

    Skipping rather than failing is deliberate: this must not break a
    contributor on a machine where no context is obtainable.  CI asserts
    separately that the expected number of tests actually ran, so a silent
    skip cannot masquerade as a pass there.
    """
    glcontext = pytest.importorskip("tools.glcontext")
    try:
        return glcontext.acquire_offscreen_gl()
    except Exception as exc:                     # noqa: BLE001 - report and skip
        pytest.skip(f"no headless GL context available: {exc}")


@pytest.fixture(scope="module")
def frames(gl):
    """Render A (perturbed scene) and B (fresh scene + reapplied state)."""
    from faceforge.constants import STL_DIR
    from faceforge.core.scene_state import binding, codec
    from tools import verify_state_pixels as v

    size, count = 256, 8          # smaller than the 512/16 one-off: same claim, faster
    v.make_fbo(size, size)        # one framebuffer for every render in this module

    scene_a, camera_a, lights_a, renderer_a = _setup(v, count, STL_DIR, size)
    v.perturb(scene_a, camera_a, lights_a, renderer_a, size, size)
    frame_a, _ms = v.render_once(scene_a, camera_a, lights_a, renderer_a, size, size)

    state = binding.capture_scene_state(
        scene=scene_a, camera=camera_a, lights=lights_a, renderer=renderer_a,
        viewport=(size, size),
    )
    blob = codec.dumps(state)

    # Deliberately a *fresh* scene, camera, lights and renderer: if apply()
    # relied on leftover state from A this would silently pass.
    scene_b, camera_b, lights_b, renderer_b = _setup(v, count, STL_DIR, size)
    binding.apply_scene_state(codec.loads(blob), scene=scene_b, camera=camera_b,
                              lights=lights_b, renderer=renderer_b)
    frame_b, _ms = v.render_once(scene_b, camera_b, lights_b, renderer_b, size, size)

    return frame_a, frame_b, blob, size


def _setup(v, count, stl_dir, size):
    """Build a scene + camera + lights + renderer, as the proof script does.

    ``build_scene`` returns only (scene, meshes); the render collaborators are
    constructed by the script's ``main``.  Assembling them here rather than
    reusing ``main`` keeps the test in control of the comparison.
    """
    from faceforge.rendering.camera import Camera
    from faceforge.rendering.lights import LightSetup
    from faceforge.rendering.renderer import GLRenderer

    built = v.build_scene(count, stl_dir)
    assert len(built) == 2, f"build_scene returned {len(built)} values, expected 2"
    scene, _meshes = built

    camera, lights = Camera(), LightSetup()
    renderer = GLRenderer()
    renderer.init_gl()
    renderer.resize(size, size)
    return scene, camera, lights, renderer


def test_reapplied_state_reproduces_the_frame_exactly(frames):
    frame_a, frame_b, _blob, size = frames
    diff = np.abs(frame_a.astype(np.int32) - frame_b.astype(np.int32))
    changed = int((diff.sum(axis=2) > 0).sum())
    assert changed == 0, (
        f"{changed} of {size * size} pixels differ after save->load->apply "
        f"(max abs channel diff {int(diff.max())}). A SceneState that does not "
        f"reproduce the render is not reproducible."
    )


def test_the_frame_is_not_blank(frames):
    """Guards the test above from passing vacuously.

    Two identical *blank* frames also differ in zero pixels.  Without this the
    exactness assertion would survive a renderer that draws nothing at all.
    """
    frame_a, _frame_b, _blob, _size = frames
    unique = len(np.unique(frame_a.reshape(-1, frame_a.shape[-1]), axis=0))
    assert unique > 16, f"frame has only {unique} distinct colours; it is blank or near-blank"


def _nudge_fov(state, delta=0.05):
    """Return *state* with the camera fov shifted.

    The state dataclasses are frozen -- deliberately, so a loaded state cannot
    be mutated behind the caller's back -- so this rebuilds rather than assigns.
    """
    import dataclasses
    return dataclasses.replace(
        state, camera=dataclasses.replace(
            state.camera, fov_deg=state.camera.fov_deg + delta))


@pytest.mark.parametrize("field,mutate,least_fraction", [
    ("camera fov +0.05 deg", _nudge_fov, 0.01),
])
def test_a_mutated_state_changes_the_pixels(frames, field, mutate, least_fraction):
    """Negative control: the comparison must be able to see a difference.

    If mutating the state left the render identical, the exactness test above
    would prove nothing -- it would just mean apply() is a no-op.
    """
    from faceforge.constants import STL_DIR
    from faceforge.core.scene_state import binding, codec
    from tools import verify_state_pixels as v

    frame_a, _frame_b, blob, size = frames
    state = mutate(codec.loads(blob))

    scene, camera, lights, renderer = _setup(v, 8, STL_DIR, size)
    binding.apply_scene_state(state, scene=scene, camera=camera,
                              lights=lights, renderer=renderer)
    frame_c, _ms = v.render_once(scene, camera, lights, renderer, size, size)

    diff = np.abs(frame_a.astype(np.int32) - frame_c.astype(np.int32))
    fraction = float((diff.sum(axis=2) > 0).sum()) / (size * size)
    assert fraction >= least_fraction, (
        f"mutating {field} changed only {fraction:.4%} of pixels; the pixel "
        f"comparison cannot detect a real difference and the exactness test "
        f"above is therefore vacuous"
    )
