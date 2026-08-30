"""The parts of :mod:`faceforge.session` that need neither GL nor the dataset.

Everything here runs in the fast tier, against a checkout with no BodyParts3D
data: the PNG encoder, the path grammar, the argument validation, the scan
colour maps and the failure modes.  The GL half lives in ``test_session_gl.py``
and is marked ``slow``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from faceforge import session as fs


# ---------------------------------------------------------------------------
# Import purity
# ---------------------------------------------------------------------------


def test_importing_the_module_does_not_require_qt_or_gl():
    """A cluster node with no Qt and no display must be able to import this.

    Asserted on ``sys.modules`` rather than on the source text so that an
    indirect import -- a helper that pulls in ``faceforge.ui`` three levels
    down -- is caught too.
    """
    import subprocess

    code = (
        "import sys; import faceforge.session; "
        "print('PySide6' in sys.modules, 'OpenGL' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
        env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin:/bin"},
    )
    assert result.stdout.strip() == "False False", (
        f"importing faceforge.session pulled in Qt or PyOpenGL: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# PNG encoding
# ---------------------------------------------------------------------------


def _checkerboard(height=7, width=5, channels=4):
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    arr[..., :3] = 40
    if channels == 4:
        arr[..., 3] = 255
    arr[::2, ::2, 0] = 200
    arr[1::2, 1::2, 1] = 170
    return arr


def test_png_round_trips_through_a_real_decoder():
    """The encoder is hand-written; a decoder that is not ours must accept it."""
    Image = pytest.importorskip("PIL.Image")
    import io

    for channels in (3, 4):
        arr = _checkerboard(channels=channels)
        decoded = np.array(Image.open(io.BytesIO(fs.png_bytes(arr))))
        assert decoded.shape == arr.shape, f"{channels}ch: {decoded.shape} != {arr.shape}"
        assert np.array_equal(decoded, arr), f"{channels}ch: pixels changed in the file"


def test_png_encoding_is_deterministic():
    """The reproducibility claim in the CLI rests on this."""
    arr = _checkerboard()
    assert fs.png_bytes(arr) == fs.png_bytes(arr.copy())


def test_png_rejects_the_wrong_dtype_and_shape():
    for bad, why in (
        (np.zeros((4, 4, 4), np.float32), "float array"),
        (np.zeros((4, 4), np.uint8), "2-D array"),
        (np.zeros((4, 4, 2), np.uint8), "2-channel array"),
        (np.zeros((0, 4, 4), np.uint8), "empty image"),
    ):
        with pytest.raises(ValueError):
            fs.png_bytes(bad)
        assert why  # keeps the loop readable in a failure report


def test_write_png_is_atomic_and_leaves_no_temp_file(tmp_path):
    out = tmp_path / "nested" / "frame.png"
    written = fs.write_png(out, _checkerboard())
    assert written == out and out.is_file()
    assert list(out.parent.iterdir()) == [out], "a .tmp file was left behind"


# ---------------------------------------------------------------------------
# Frame content: must agree with the golden-capture tool, not merely resemble it
# ---------------------------------------------------------------------------


def test_content_fraction_agrees_with_capture_golden():
    """Two implementations of "did anything draw" must not drift apart.

    ``tools/capture_golden.py`` gates its writes on its own copy; a session
    that used a different tolerance would call a frame blank that the capture
    tool calls fine, or worse, the reverse.
    """
    cg = pytest.importorskip("tools.capture_golden")

    rng = np.random.default_rng(20260830)
    clear = (31, 31, 38)
    for _ in range(20):
        img = np.zeros((16, 16, 4), np.uint8)
        img[..., :3] = clear
        img[..., 3] = 255
        mask = rng.random((16, 16)) < 0.3
        img[mask, :3] = rng.integers(0, 256, size=(int(mask.sum()), 3), dtype=np.uint8)
        assert fs.content_fraction(img, clear) == cg.frame_content_fraction(img, clear)


def test_content_thresholds_match_capture_golden():
    cg = pytest.importorskip("tools.capture_golden")

    assert fs.MIN_CONTENT_FRACTION == cg.MIN_CONTENT_FRACTION
    assert cg.EXPECTED_CLEAR_RGB8 == (31, 31, 38)


# ---------------------------------------------------------------------------
# Scene-graph path grammar.  SceneNode needs no assets, so this is fast.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("names", [
    ["skullGroup", "Frontal bone"],
    ["a/b", "c%d"],                     # both escape characters
    ["group", "Mandible", "Mandible"],  # duplicate siblings -> [0]/[1] suffixes
    ["%2F", "plain"],                   # a name that looks like an escape
])
def test_split_path_inverts_the_scene_state_path_grammar(names):
    """``_split_path`` must recover exactly the names ``scene_paths`` encoded.

    This is what makes "rebuild the scene the state describes" correct: get the
    inverse wrong and a structure lands at the wrong scene-graph path, where
    ``apply_scene_state`` then reports it missing.
    """
    from faceforge.core.scene_graph import Scene, SceneNode
    from faceforge.core.scene_state import scene_paths

    scene = Scene()
    parent = scene
    made = []
    for name in names:
        node = SceneNode(name=name)
        parent.add(node)
        made.append(node)
        parent = node if name != names[-1] else parent

    got = {p: n for p, n in scene_paths(scene)}
    for path, node in got.items():
        recovered = fs._split_path(path)
        assert recovered[-1][0] == node.name, (
            f"path {path!r} decodes to {recovered[-1][0]!r}, not {node.name!r}"
        )


def test_split_path_reads_duplicate_indices():
    assert fs._split_path("/g/Mandible[1]") == [("g", 0), ("Mandible", 1)]
    assert fs._split_path("/g/odd[name]") == [("g", 0), ("odd[name]", 0)]


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_session_cannot_be_constructed_directly():
    with pytest.raises(TypeError, match="Session.create"):
        fs.Session()


def test_no_gl_context_is_reported_and_releases_the_guard(monkeypatch):
    """A failed create must not leave the one-session guard set.

    Otherwise a machine that fails the first acquisition reports "a Session is
    already live" forever after, which sends the reader hunting for a leak that
    does not exist.
    """
    class _Stub:
        class GLContextError(RuntimeError):
            pass

        @staticmethod
        def acquire_offscreen_gl(prefer="auto"):
            raise _Stub.GLContextError("no window server (test stub)")

    monkeypatch.setattr(fs, "glcontext_module", lambda: _Stub)
    with pytest.raises(fs.NoGLContextError, match="test stub"):
        fs.Session.create()
    assert fs.Session.active() is None, "the guard survived a failed create"
    with pytest.raises(fs.NoGLContextError):
        fs.Session.create()


def test_create_rejects_an_unrenderable_size(monkeypatch):
    monkeypatch.setattr(fs, "glcontext_module", lambda: pytest.fail("size checked too late"))
    with pytest.raises(ValueError, match="not a renderable size"):
        fs.Session.create(width=0, height=10)


# ---------------------------------------------------------------------------
# Scene building: the failure modes, without touching the dataset
# ---------------------------------------------------------------------------


def _state(structures=(), **kwargs):
    from faceforge.core.scene_state import SceneState

    return SceneState(structures=tuple(structures), **kwargs)


def _structure(path, source_id=""):
    from faceforge.core.scene_state import ProvenanceState, StructureState

    return StructureState(path=path, name=path.rsplit("/", 1)[-1],
                          provenance=ProvenanceState(source_id=source_id))


def test_build_scene_from_state_refuses_an_empty_state():
    with pytest.raises(fs.AnatomyError, match="no structures"):
        fs.build_scene_from_state(_state())


def test_build_scene_from_state_refuses_structures_without_provenance():
    """Without a source_id there is no file to load; guessing would be worse."""
    state = _state([_structure("/g/a", "FMA1"), _structure("/g/b", "")])
    with pytest.raises(fs.AnatomyError, match="no source_id"):
        fs.build_scene_from_state(state)


def test_build_scene_from_state_reports_missing_files(tmp_path):
    state = _state([_structure("/g/a", "NOSUCHMESH")])
    with pytest.raises(fs.AnatomyError, match="could not be rebuilt"):
        fs.build_scene_from_state(state, stl_dir=tmp_path)


def test_load_structures_refuses_an_empty_request():
    with pytest.raises(fs.AnatomyError, match="empty structure list"):
        fs.load_structures([])


def test_load_structures_refuses_a_partial_set(tmp_path):
    with pytest.raises(fs.AnatomyError, match="missing"):
        fs.load_structures(["NOSUCH1", "NOSUCH2"], stl_dir=tmp_path)


def test_build_scene_needs_exactly_one_source():
    with pytest.raises(fs.AnatomyError, match="exactly one"):
        fs.build_scene()
    with pytest.raises(fs.AnatomyError, match="exactly one"):
        fs.build_scene(tier=0, structures=["FMA1"])


def test_unloadable_tier_names_the_layer_route():
    """Tiers 3-5 exist in constants but are layers, not load phases."""
    from faceforge import constants

    assert constants.TIER_MUSCLES == 3
    with pytest.raises(fs.AnatomyError) as excinfo:
        fs.load_tier_scene(3)
    message = str(excinfo.value)
    assert "layers=" in message and "organs" in message


def test_unknown_layer_is_rejected_before_anything_loads():
    with pytest.raises(fs.AnatomyError, match="unknown layer"):
        fs.add_layers(object(), ["not_a_layer"])


def test_every_declared_layer_maps_to_a_real_asset_manager_method():
    """The layer table is a promise about another class; check it."""
    from faceforge.loaders.asset_manager import AssetManager

    for name, (method, _args) in fs.LAYER_LOADERS.items():
        assert callable(getattr(AssetManager, method, None)), (
            f"layer {name!r} maps to AssetManager.{method}, which does not exist"
        )


# ---------------------------------------------------------------------------
# Scanner helpers
# ---------------------------------------------------------------------------


def test_scan_modes_come_from_the_tissue_map():
    from faceforge.scanner.tissue_map import MODES

    assert fs.scan_modes() == tuple(MODES)


def test_scan_orientations_are_orthonormal_frames():
    for name in sorted(fs.SCAN_ORIENTATIONS):
        normal, right, up = fs.plane_frame(name)
        for label, vec in (("normal", normal), ("right", right), ("up", up)):
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-12, f"{name}.{label} is not unit"
        assert abs(float(np.dot(normal, right))) < 1e-12, f"{name}: normal . right != 0"
        assert abs(float(np.dot(normal, up))) < 1e-12, f"{name}: normal . up != 0"
        assert abs(float(np.dot(right, up))) < 1e-12, f"{name}: right . up != 0"


def test_plane_frame_rejects_an_unknown_orientation():
    with pytest.raises(ValueError, match="unknown orientation"):
        fs.plane_frame("oblique")


def test_vectorised_colormap_matches_the_scalar_one():
    """``scan_to_rgb`` is a vectorised copy of tissue_map's scalar colour maps.

    A copy that drifts would silently recolour every scan the CLI writes, so
    the two are compared on every 8-bit input rather than trusted.
    """
    from faceforge.scanner.tissue_map import TissueMapper

    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    for mode in fs.scan_modes():
        colormap = TissueMapper.get_colormap(mode)
        expected = np.array([colormap(float(v)) for v in values], dtype=np.uint8)
        got = fs.scan_to_rgb(values.reshape(1, -1), mode)[0]
        assert np.array_equal(got, expected), f"{mode}: vectorised colour map differs"


def test_scan_to_rgb_passes_anatomical_rgb_through():
    image = np.zeros((2, 2, 3), np.float32)
    image[0, 0] = (1.0, 0.5, 0.0)
    got = fs.scan_to_rgb(image, "anatomical")
    assert got.dtype == np.uint8 and tuple(got[0, 0]) == (255, 127, 0)


def test_scan_rejects_an_unknown_mode_or_reduction():
    from faceforge.core.scene_graph import Scene

    for kwargs, match in (
        ({"mode": "pet"}, "unknown scan mode"),
        ({"reduction": "median"}, "unknown reduction"),
        ({"resolution": 4}, "too small"),
    ):
        with pytest.raises(ValueError, match=match):
            fs.scan_scene(Scene(), origin=(0, 0, 0), **kwargs)


def test_scan_refuses_an_empty_scene():
    from faceforge.core.scene_graph import Scene

    with pytest.raises(fs.AnatomyError, match="no visible meshes"):
        fs.scan_scene(Scene(), origin=(0, 0, 0))


def test_reductions_are_the_ones_the_engine_implements():
    """SCAN_REDUCTIONS is a claim about ScannerEngine.scan; read its source."""
    import inspect

    from faceforge.scanner.engine import ScannerEngine

    source = inspect.getsource(ScannerEngine.scan)
    for reduction in fs.SCAN_REDUCTIONS:
        assert f'reduction == "{reduction}"' in source, (
            f"{reduction!r} is offered but ScannerEngine.scan does not branch on it"
        )


def test_repo_tool_module_finds_the_capture_tool():
    assert fs.repo_tool_module("capture_golden").MAX_MESHES == 16


def test_repo_tool_module_reports_a_missing_tool():
    with pytest.raises(fs.SessionError, match="cannot locate"):
        fs.repo_tool_module("no_such_tool_module")


def test_gl_available_returns_a_bool():
    assert isinstance(fs.gl_available(), bool)
