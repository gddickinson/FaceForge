"""What a Session does with a real OpenGL context and real meshes.

Marked ``slow`` in full: every test here needs the BodyParts3D dataset and a
headless context, and renders through Apple's software rasteriser.  Frame times
measured here are CPU numbers and are never renderer performance.

The claims made here, in order of what they are worth:

1. A saved state file reproduces its render bit-for-bit through a Session.
2. A Session render of ``tools/capture_golden.py``'s scene is pixel-identical
   to what ``capture_golden`` itself writes -- so the Session is not "a second
   renderer that looks about right", it is the same renderer.
3. Two sessions in one process cannot corrupt each other, including when they
   share ``MeshInstance`` objects, which is the specific failure that broke an
   earlier harness in this repo.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from faceforge import session as fs  # noqa: E402 - after the module-wide mark


SIZE = 128
MESH_COUNT = 6


@pytest.fixture
def fixed_ids():
    """The first few of capture_golden's fixed mesh list, in its order."""
    cg = pytest.importorskip("tools.capture_golden")
    ids = [fma_id for fma_id, _label in cg.FIXED_MESHES[:MESH_COUNT]]
    missing = [i for i in ids if not (cg.stl_dir() / f"{i}.stl").is_file()]
    if missing:
        pytest.skip(f"dataset incomplete: {missing}")
    return ids


def _frame_subject(session):
    """Point the camera at the loaded subject, deterministically."""
    from faceforge.core.math_utils import vec3
    from faceforge.core.scene_state import mesh_paths

    entries = sorted(mesh_paths(session.scene))
    positions = np.concatenate([n.mesh.positions.reshape(-1, 3) for _, n in entries])
    centroid = positions.mean(axis=0).astype(np.float64)
    radius = float(np.linalg.norm(positions - centroid, axis=1).max())
    direction = np.array([-0.62, -0.68, 0.39])
    eye = centroid + direction / np.linalg.norm(direction) * radius * 2.9
    session.camera.look_at(vec3(*eye), vec3(*centroid), vec3(0.0, 0.0, 1.0))
    return centroid, radius


def _changed_pixels(a, b):
    delta = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return int((delta.sum(axis=2) > 0).sum())


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_session_renders_and_reports_what_it_rendered_with(gl_or_skip, fixed_ids):
    with fs.Session.create(width=SIZE, height=SIZE) as session:
        report = session.load_anatomy(structures=fixed_ids)
        assert report.ok and report.structures == MESH_COUNT, report.summary()
        _frame_subject(session)
        image = session.render()
        assert image.shape == (SIZE, SIZE, 4) and image.dtype == np.uint8
        assert session.last_content_fraction > 0.05, (
            f"only {session.last_content_fraction:.2%} of the frame has content"
        )
        info = session.info()
        assert info.gl_renderer and info.width == SIZE
        assert "gl_renderer" in info.as_dict()


def test_a_closed_session_refuses_to_be_used(gl_or_skip, fixed_ids):
    session = fs.Session.create(width=64, height=64)
    session.load_anatomy(structures=fixed_ids[:1])
    session.close()
    session.close()                              # idempotent
    assert session.closed and repr(session) == "<Session closed>"
    for call in (lambda: session.render(), lambda: session.scene, lambda: session.size):
        with pytest.raises(fs.SessionError, match="closed"):
            call()


def test_an_empty_scene_raises_rather_than_returning_a_blank_frame(gl_or_skip):
    """The blank-image discipline, enforced at the one place images escape."""
    with fs.Session.create(width=32, height=32) as session:
        with pytest.raises(fs.BlankFrameError, match="produced nothing"):
            session.render()
        allowed = session.render(allow_blank=True)
        assert len(np.unique(allowed.reshape(-1, 4), axis=0)) == 1


def test_save_png_writes_a_readable_image(gl_or_skip, fixed_ids, tmp_path):
    Image = pytest.importorskip("PIL.Image")

    with fs.Session.create(width=SIZE, height=SIZE) as session:
        session.load_anatomy(structures=fixed_ids)
        _frame_subject(session)
        out = session.save_png(tmp_path / "frame.png")
    with Image.open(out) as im:
        assert im.size == (SIZE, SIZE)


# ---------------------------------------------------------------------------
# Two sessions.  This is the bug class the single-instance rule exists for.
# ---------------------------------------------------------------------------


def test_a_second_concurrent_session_is_refused_and_the_first_still_works(
    gl_or_skip, fixed_ids,
):
    with fs.Session.create(width=SIZE, height=SIZE) as session:
        session.load_anatomy(structures=fixed_ids)
        _frame_subject(session)
        before = session.render()

        with pytest.raises(fs.SessionInUseError, match="already live"):
            fs.Session.create(width=64, height=64)

        after = session.render()
        assert _changed_pixels(before, after) == 0, (
            "the refused create disturbed the live session's GL state"
        )


def test_sequential_sessions_render_identically(gl_or_skip, fixed_ids):
    """Close, create again, render the same thing: the pixels must not move.

    A second context acquisition, a re-created framebuffer or a renderer that
    kept state across the boundary would all show up here.
    """
    frames = []
    for _ in range(2):
        with fs.Session.create(width=SIZE, height=SIZE) as session:
            session.load_anatomy(structures=fixed_ids)
            _frame_subject(session)
            frames.append(session.render())
    assert _changed_pixels(frames[0], frames[1]) == 0


def test_meshes_can_be_reused_by_a_later_session(gl_or_skip, fixed_ids):
    """The stale-VAO case, directly.

    ``MeshInstance.gl_handle`` caches a GL upload on the *mesh*, which outlives
    the renderer that made it.  Hand the same mesh objects to a second session
    and, if close() did not release those handles, the second render either
    draws nothing or draws from another context's buffers.  The assertion is
    that it draws exactly what a fresh load draws.
    """
    session = fs.Session.create(width=SIZE, height=SIZE)
    session.load_anatomy(structures=fixed_ids)
    _frame_subject(session)
    reference = session.render()
    shared_scene = session.scene
    shared_meshes = shared_scene.subtree_meshes()
    assert shared_meshes, "no meshes to share"
    session.close()

    assert all(m.gl_handle is None for m in shared_meshes), (
        "close() left GL handles on the meshes; a later session would reuse "
        "buffers from a destroyed renderer"
    )

    with fs.Session.create(width=SIZE, height=SIZE) as second:
        second.adopt_scene(shared_scene)
        _frame_subject(second)
        again = second.render()
    assert _changed_pixels(reference, again) == 0, (
        "re-rendering the same meshes in a second session changed the picture"
    )


# ---------------------------------------------------------------------------
# State round trip
# ---------------------------------------------------------------------------


@pytest.fixture
def state_round_trip(gl_or_skip, fixed_ids, tmp_path):
    """Render A, save its state, rebuild from the file alone, render B."""
    from faceforge.core.scene_state import codec

    with fs.Session.create(width=SIZE, height=SIZE) as session:
        session.load_anatomy(structures=fixed_ids)
        _frame_subject(session)
        # Move things off their defaults so the comparison has something to say.
        from faceforge.core.material import RenderMode

        for index, mesh in enumerate(session.scene.subtree_meshes()):
            mesh.material.color = (0.5 + 0.04 * index, 0.42, 0.31)
            mesh.material.render_mode = RenderMode.XRAY if index == 1 else RenderMode.SOLID
            mesh.visible = index != 2
        frame_a = session.render()
        path = session.save_state(tmp_path / "fig.state.json")

    # A deliberately different starting size: a state must restore its own
    # viewport, or the reproduction is only accidentally the same shot.
    with fs.Session.create(width=64, height=64) as session:
        state = codec.load(path)
        report = session.load_state_scene(state)
        assert report.structures == MESH_COUNT, report.summary()
        apply_report = session.apply_state(state, strict=True)
        assert apply_report.ok, apply_report.summary()
        assert session.size == (SIZE, SIZE)
        frame_b = session.render()
    return frame_a, frame_b, path


def test_a_state_file_reproduces_its_render_exactly(state_round_trip):
    frame_a, frame_b, _path = state_round_trip
    changed = _changed_pixels(frame_a, frame_b)
    assert changed == 0, (
        f"{changed} of {SIZE * SIZE} pixels differ after save -> load -> rebuild "
        "-> apply -> render.  A state file that does not reproduce its render is "
        "not reproducible."
    )


def test_the_round_trip_frame_is_not_blank(state_round_trip):
    """Guards the test above from passing vacuously on two blank frames."""
    frame_a, _b, _path = state_round_trip
    assert len(np.unique(frame_a.reshape(-1, 4), axis=0)) > 16


def test_a_one_field_change_moves_pixels(state_round_trip, gl_or_skip):
    """Negative control: the comparison can see a difference."""
    import dataclasses

    from faceforge.core.scene_state import codec

    frame_a, _b, path = state_round_trip
    state = codec.load(path)
    nudged = dataclasses.replace(
        state, camera=dataclasses.replace(state.camera,
                                          fov_deg=state.camera.fov_deg + 0.05),
    )
    with fs.Session.create(width=SIZE, height=SIZE) as session:
        session.load_state_scene(nudged)
        session.apply_state(nudged, strict=True)
        frame_c = session.render()
    fraction = _changed_pixels(frame_a, frame_c) / (SIZE * SIZE)
    assert fraction > 0.01, (
        f"a 0.05 deg fov change moved only {fraction:.4%} of pixels; the "
        "exactness test above would be vacuous"
    )


def test_capture_state_refuses_to_be_lied_to(gl_or_skip, fixed_ids):
    """A caller must not be able to record a camera other than the rendered one."""
    with fs.Session.create(width=64, height=64) as session:
        session.load_anatomy(structures=fixed_ids[:1])
        with pytest.raises(TypeError, match="from the session"):
            session.capture_state(camera=object())


# ---------------------------------------------------------------------------
# Equivalence with the golden-image capture tool
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_capture(tmp_path_factory):
    """Run ``tools/capture_golden.py`` in a subprocess and return its directory.

    A subprocess, not an in-process call: ``capture_golden`` builds its own
    framebuffer and renderer, which would be a second, unguarded GL user inside
    a test process that also creates Sessions.
    """
    out = tmp_path_factory.mktemp("golden") / "ref"
    result = subprocess.run(
        [sys.executable, "-m", "tools.capture_golden",
         "--out", str(out), "--modes", "SOLID,XRAY", "--size", "128x128",
         "--prefer", "software", "--quiet"],
        capture_output=True, text=True,
        env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"capture_golden could not run here: {result.stderr[-400:]}")
    return out


def _golden_scene():
    """capture_golden's fixed scene, assembled exactly as the tool assembles it."""
    from faceforge.core.material import Material
    from faceforge.core.mesh import MeshInstance
    from faceforge.core.scene_graph import Scene, SceneNode
    from faceforge.loaders.stl_parser import load_stl_file
    from tools import capture_golden as cg

    scene = Scene()
    meshes = []
    for fma_id, label, path in cg.mesh_paths(cg.MAX_MESHES):
        mesh = MeshInstance(
            name=label, geometry=load_stl_file(path),
            material=Material(color=(0.82, 0.76, 0.68), opacity=1.0),
            source_id=fma_id,
        )
        node = SceneNode(name=fma_id)
        node.mesh = mesh
        scene.add(node)
        meshes.append(mesh)
    return scene, meshes


def _place_golden_camera(session, meshes, size):
    from faceforge.core.math_utils import vec3
    from tools import capture_golden as cg

    positions = np.concatenate([m.positions.reshape(-1, 3) for m in meshes])
    centroid = positions.mean(axis=0).astype(np.float64)
    radius = float(np.linalg.norm(positions - centroid, axis=1).max())
    eye, target, up = cg.camera_placement(
        centroid, radius, cg.CAMERA_PRESETS[cg.DEFAULT_CAMERA],
    )
    session.camera.set_aspect(size, size)
    session.camera.look_at(vec3(*eye), vec3(*target), vec3(*up))


@pytest.mark.parametrize("mode", ["SOLID", "XRAY"])
def test_session_render_equals_capture_golden(golden_capture, gl_or_skip, mode):
    """The Session is the same renderer, not a lookalike.

    Diffed with ``tools/compare_golden.py``'s own primitive at threshold 0; the
    measured noise floor between two identical captures in this repo is exactly
    zero, so any non-zero result here is a real difference.
    """
    cmp = pytest.importorskip("tools.compare_golden")
    from faceforge.core.material import RenderMode

    size = 128
    scene, meshes = _golden_scene()
    with fs.Session.create(width=size, height=size, prefer="software") as session:
        session.adopt_scene(scene)
        _place_golden_camera(session, meshes, size)
        for mesh in meshes:
            mesh.material.render_mode = RenderMode[mode]
        image = session.render()

    reference = cmp.load_rgb(golden_capture / f"{mode}.png")
    diff = cmp.diff_images(reference, image[:, :, :3], pixel_threshold=0, mode=mode)
    assert diff.pixels_above == 0, (
        f"{mode}: {diff.pixels_above} of {diff.total_pixels} pixels differ from "
        f"capture_golden (max abs channel diff {diff.max_abs}, bbox {diff.bbox})"
    )
