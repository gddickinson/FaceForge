"""``faceforge-cli`` against real assets and a real framebuffer.

Marked ``slow``: the dataset and a GL context are both required.

The claim this module exists to establish is the one a paper depends on:

    faceforge-cli render --state S --out fig.png

renders byte-identical PNGs on repeated runs, and the pixels it produces are
the same pixels ``tools/capture_golden.py`` produces for the same scene -- so a
committed state file *is* the figure, not an approximation of it.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from faceforge import cli  # noqa: E402 - after the module-wide mark
from faceforge import session as fs  # noqa: E402


GOLDEN_SIZE = 128
GOLDEN_MODES = ("SOLID", "XRAY")


# ---------------------------------------------------------------------------
# A state file made from capture_golden's own fixed scene
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden_capture(tmp_path_factory):
    """``tools/capture_golden.py``'s output, produced in a subprocess."""
    out = tmp_path_factory.mktemp("cap") / "ref"
    result = subprocess.run(
        [sys.executable, "-m", "tools.capture_golden",
         "--out", str(out), "--modes", ",".join(GOLDEN_MODES),
         "--size", f"{GOLDEN_SIZE}x{GOLDEN_SIZE}", "--prefer", "software", "--quiet"],
        capture_output=True, text=True, check=False,
        env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin:/bin"},
    )
    if result.returncode != 0:
        pytest.skip(f"capture_golden could not run here: {result.stderr[-400:]}")
    return out


@pytest.fixture(scope="module")
def golden_states(tmp_path_factory, golden_capture):
    """One state file per mode, captured from capture_golden's fixed scene.

    Written in a subprocess so that the module's own tests, which call
    ``cli.main`` in-process, never share a process with a second GL user.
    """
    out = tmp_path_factory.mktemp("states")
    script = f"""
import numpy as np
from faceforge.core.material import Material, RenderMode
from faceforge.core.math_utils import vec3
from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.loaders.stl_parser import load_stl_file
from faceforge.session import Session
from tools import capture_golden as cg

scene, meshes = Scene(), []
for fma_id, label, path in cg.mesh_paths(cg.MAX_MESHES):
    mesh = MeshInstance(name=label, geometry=load_stl_file(path),
                        material=Material(color=(0.82, 0.76, 0.68), opacity=1.0),
                        source_id=fma_id)
    node = SceneNode(name=fma_id); node.mesh = mesh
    scene.add(node); meshes.append(mesh)

with Session.create(width={GOLDEN_SIZE}, height={GOLDEN_SIZE},
                    prefer="software") as s:
    s.adopt_scene(scene)
    pos = np.concatenate([m.positions.reshape(-1, 3) for m in meshes])
    centroid = pos.mean(axis=0).astype(np.float64)
    radius = float(np.linalg.norm(pos - centroid, axis=1).max())
    eye, target, up = cg.camera_placement(
        centroid, radius, cg.CAMERA_PRESETS[cg.DEFAULT_CAMERA])
    s.camera.set_aspect({GOLDEN_SIZE}, {GOLDEN_SIZE})
    s.camera.look_at(vec3(*eye), vec3(*target), vec3(*up))
    for mode in {GOLDEN_MODES!r}:
        for m in meshes:
            m.material.render_mode = RenderMode[mode]
        s.save_state(r"{out}" + "/" + mode + ".state.json")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False,
        env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin:/bin"},
    )
    if result.returncode != 0:
        pytest.skip(f"could not capture golden states: {result.stderr[-600:]}")
    return out


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_twice_produces_byte_identical_files(golden_states, tmp_path):
    """The reproducibility claim, at the level a user can check with sha256sum."""
    state = golden_states / "SOLID.state.json"
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    for out in (first, second):
        assert cli.main(["-q", "render", "--state", str(state),
                         "--out", str(out), "--prefer", "software"]) == cli.EXIT_OK
    assert first.read_bytes() == second.read_bytes(), (
        "two renders of the same state file produced different bytes"
    )
    assert first.stat().st_size > 1000


@pytest.mark.parametrize("mode", GOLDEN_MODES)
def test_render_from_a_state_file_equals_capture_golden(
    golden_states, golden_capture, tmp_path, mode,
):
    """The strongest claim available: the CLI reproduces the golden capture.

    Nothing but the state file is given to the CLI -- the geometry is rebuilt
    from the BodyParts3D ids the state records -- and the result is diffed
    against ``capture_golden``'s PNG with ``tools/compare_golden.py``'s own
    primitive at threshold 0.  The measured noise floor between two identical
    captures in this repo is exactly zero, so any non-zero count is real.
    """
    cmp = pytest.importorskip("tools.compare_golden")

    out = tmp_path / f"{mode}.png"
    assert cli.main(["-q", "render", "--state",
                     str(golden_states / f"{mode}.state.json"),
                     "--out", str(out), "--prefer", "software"]) == cli.EXIT_OK

    reference = cmp.load_rgb(golden_capture / f"{mode}.png")
    current = cmp.load_rgb(out)
    diff = cmp.diff_images(reference, current, pixel_threshold=0, mode=mode)
    assert diff.pixels_above == 0, (
        f"{mode}: {diff.pixels_above} of {diff.total_pixels} pixels differ from "
        f"capture_golden (max abs channel diff {diff.max_abs}, bbox {diff.bbox})"
    )


def test_render_json_manifest_describes_what_was_written(
    golden_states, tmp_path, capsys,
):
    import hashlib

    out = tmp_path / "fig.png"
    assert cli.main(["-q", "render", "--state", str(golden_states / "SOLID.state.json"),
                     "--out", str(out), "--json", "--prefer", "software"]) == cli.EXIT_OK
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["size"] == [GOLDEN_SIZE, GOLDEN_SIZE]
    assert manifest["structures"] == 16
    assert manifest["apply"]["ok"] is True
    assert manifest["content_fraction"] > 0.05
    assert manifest["gl"]["gl_renderer"]


def test_size_override_reframes_at_the_requested_resolution(golden_states, tmp_path):
    Image = pytest.importorskip("PIL.Image")

    out = tmp_path / "big.png"
    assert cli.main(["-q", "render", "--state", str(golden_states / "SOLID.state.json"),
                     "--out", str(out), "--size", "96x64",
                     "--prefer", "software"]) == cli.EXIT_OK
    with Image.open(out) as im:
        assert im.size == (96, 64)


def test_mode_override_changes_the_picture(golden_states, tmp_path):
    """--mode must actually reach the renderer, not merely be accepted."""
    cmp = pytest.importorskip("tools.compare_golden")

    state = str(golden_states / "SOLID.state.json")
    plain, wire = tmp_path / "plain.png", tmp_path / "wire.png"
    assert cli.main(["-q", "render", "--state", state, "--out", str(plain),
                     "--prefer", "software"]) == cli.EXIT_OK
    assert cli.main(["-q", "render", "--state", state, "--out", str(wire),
                     "--mode", "WIREFRAME", "--prefer", "software"]) == cli.EXIT_OK
    diff = cmp.diff_images(cmp.load_rgb(plain), cmp.load_rgb(wire),
                           pixel_threshold=0, mode="WIREFRAME")
    assert diff.frac_above > 0.01, (
        f"--mode WIREFRAME changed only {diff.frac_above:.4%} of pixels"
    )


def test_a_drifted_config_warns_but_still_renders_the_same_picture(
    golden_states, tmp_path, capsys,
):
    """A state committed today must still render after the configs change.

    The fingerprint is tampered rather than the configs, so the *correct*
    outcome is a loud warning and an identical picture: the mechanism must warn
    without refusing, and without altering the render.
    """
    import dataclasses

    from faceforge.core.scene_state import codec

    original = golden_states / "SOLID.state.json"
    drifted = tmp_path / "drifted.state.json"
    state = codec.load(original, check_config=False)
    codec.save(dataclasses.replace(
        state, config=dataclasses.replace(state.config, digest="0" * 64)), drifted)

    clean, warned = tmp_path / "clean.png", tmp_path / "warned.png"
    assert cli.main(["-q", "render", "--state", str(original), "--out", str(clean),
                     "--prefer", "software"]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["-q", "render", "--state", str(drifted), "--out", str(warned),
                     "--prefer", "software"]) == cli.EXIT_OK
    assert "configs have changed" in capsys.readouterr().err
    assert clean.read_bytes() == warned.read_bytes()

    # ...and with --require-config-match it refuses instead, writing nothing.
    refused = tmp_path / "refused.png"
    assert cli.main(["-q", "render", "--state", str(drifted), "--out", str(refused),
                     "--require-config-match",
                     "--prefer", "software"]) == cli.EXIT_CONFIG_DRIFT
    assert not refused.exists()


def test_strict_rejects_a_scene_that_does_not_match_the_state(
    golden_states, tmp_path, capsys,
):
    """--strict is what a reproducibility check wants: the render, not a render."""
    assert cli.main(["-q", "render", "--state", str(golden_states / "SOLID.state.json"),
                     "--out", str(tmp_path / "x.png"), "--structures", "FMA52748",
                     "--strict", "--prefer", "software"]) == cli.EXIT_FAILED
    assert not (tmp_path / "x.png").exists()
    assert "did not reproduce its structure set" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def test_batch_renders_a_directory_and_writes_a_manifest(golden_states, tmp_path):
    states = tmp_path / "states"
    states.mkdir()
    for mode in GOLDEN_MODES:
        (states / f"{mode}.state.json").write_bytes(
            (golden_states / f"{mode}.state.json").read_bytes()
        )
    out = tmp_path / "out"
    assert cli.main(["-q", "batch", "--states", str(states), "--out", str(out),
                     "--prefer", "software"]) == cli.EXIT_OK

    for mode in GOLDEN_MODES:
        assert (out / f"{mode}.png").is_file()
    manifest = json.loads((out / "batch_manifest.json").read_text())
    assert manifest["rendered"] == len(GOLDEN_MODES) and manifest["failed"] == []
    assert {i["sha256"] for i in manifest["images"]}, "no digests recorded"
    for entry in manifest["images"]:
        assert entry["structures"] == 16 and entry["apply_ok"] is True


def test_batch_matches_single_render_byte_for_byte(golden_states, tmp_path):
    """One session for many states must not change any of their pixels."""
    states = tmp_path / "states"
    states.mkdir()
    for mode in GOLDEN_MODES:
        (states / f"{mode}.state.json").write_bytes(
            (golden_states / f"{mode}.state.json").read_bytes()
        )
    out = tmp_path / "batched"
    assert cli.main(["-q", "batch", "--states", str(states), "--out", str(out),
                     "--prefer", "software"]) == cli.EXIT_OK

    for mode in GOLDEN_MODES:
        single = tmp_path / f"single_{mode}.png"
        assert cli.main(["-q", "render", "--state", str(states / f"{mode}.state.json"),
                         "--out", str(single), "--prefer", "software"]) == cli.EXIT_OK
        assert single.read_bytes() == (out / f"{mode}.png").read_bytes(), (
            f"{mode}: batching changed the image"
        )


def test_batch_continue_on_error_reports_every_failure(golden_states, tmp_path, capsys):
    states = tmp_path / "states"
    states.mkdir()
    (states / "good.state.json").write_bytes(
        (golden_states / "SOLID.state.json").read_bytes()
    )
    (states / "broken.state.json").write_text("{not json")
    out = tmp_path / "out"

    code = cli.main(["-q", "batch", "--states", str(states), "--out", str(out),
                     "--continue-on-error", "--prefer", "software"])
    assert code == cli.EXIT_FAILED
    assert (out / "good.png").is_file(), "a later failure lost an earlier success"
    manifest = json.loads((out / "batch_manifest.json").read_text())
    assert manifest["rendered"] == 1 and len(manifest["failed"]) == 1
    assert "broken.state.json" in manifest["failed"][0]


def test_batch_aborts_on_the_first_failure_by_default(golden_states, tmp_path, capsys):
    states = tmp_path / "states"
    states.mkdir()
    (states / "broken.state.json").write_text("{not json")
    assert cli.main(["-q", "batch", "--states", str(states),
                     "--out", str(tmp_path / "o"),
                     "--prefer", "software"]) == cli.EXIT_FAILED
    assert "--continue-on-error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# scan and export -- neither needs a GL context
# ---------------------------------------------------------------------------


def _subject_centre(state_path):
    from faceforge.core.scene_state import codec

    return [str(v) for v in codec.load(state_path, check_config=False).camera.target]


def test_scan_writes_an_image_and_the_raw_array(golden_states, tmp_path, capsys):
    state = golden_states / "SOLID.state.json"
    out, raw = tmp_path / "ct.png", tmp_path / "ct.npy"
    argv = ["-q", "scan", "--state", str(state), "--out", str(out),
            "--mode", "ct", "--orientation", "coronal", "--resolution", "64",
            "--width", "200", "--height", "200", "--depth", "8",
            "--npy", str(raw), "--json", "--position", *_subject_centre(state)]
    assert cli.main(argv) == cli.EXIT_OK

    manifest = json.loads(capsys.readouterr().out)
    assert manifest["resolution"] == 64 and manifest["mode"] == "ct"
    assert manifest["hit_fraction"] > 0.01, (
        "the scan plane hit nothing; the test would prove nothing"
    )
    array = np.load(raw)
    assert array.shape == (64, 64) and array.dtype == np.float32
    assert float(array.max()) > 0.0

    Image = pytest.importorskip("PIL.Image")
    with Image.open(out) as im:
        assert im.size == (64, 64)


def test_scan_needs_no_gl_context(golden_states, tmp_path):
    """A cluster node with no display must still be able to produce a section."""
    state = golden_states / "SOLID.state.json"
    script = (
        "import faceforge.cli as c, sys; "
        f"sys.exit(c.main(['-q','scan','--state',{str(state)!r},"
        f"'--out',{str(tmp_path / 'nogl.png')!r},'--resolution','32',"
        "'--orientation','axial','--width','200','--height','200','--depth','200',"
        f"'--position',{_subject_centre(state)[0]!r},{_subject_centre(state)[1]!r},"
        f"{_subject_centre(state)[2]!r}]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False,
        env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin:/bin",
             "FACEFORGE_NO_GL": "1"},
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert (tmp_path / "nogl.png").is_file()
    assert "OpenGL" not in result.stderr


def test_export_writes_a_valid_glb(golden_states, tmp_path, capsys):
    out = tmp_path / "skull.glb"
    assert cli.main(["-q", "export", "--state", str(golden_states / "SOLID.state.json"),
                     "--out", str(out), "--json"]) == cli.EXIT_OK
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["meshes"] == 16

    header = out.read_bytes()[:12]
    magic, version, length = (
        header[:4], int.from_bytes(header[4:8], "little"),
        int.from_bytes(header[8:12], "little"),
    )
    assert magic == b"glTF" and version == 2
    assert length == out.stat().st_size, "the GLB header length does not match the file"


def test_verify_assets_reaches_the_real_tool(capsys):
    """Not mocked here: the delegation must work against the actual dataset.

    The exit code is whatever ``tools/fetch_assets.py`` decides -- this repo's
    expression_muscles.json names two meshes BodyParts3D does not ship, so a
    complete install still reports a shortfall unless --allow-missing is given.
    That is the tool's policy, and the CLI must pass it through rather than
    reinterpret it.
    """
    code = cli.main(["-q", "verify-assets", "--", "verify", "--allow-missing", "2"])
    out = capsys.readouterr().out
    assert "STL" in out or "mesh" in out.lower(), out[:200]
    assert code in (0, 1), f"unexpected exit code {code}"
