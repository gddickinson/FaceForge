"""The standalone OBJ viewer: reading, turning Z-up, framing."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.loaders.obj_parser import parse_obj
from faceforge.ui.obj_viewer import (
    _Y_UP_TO_Z_UP, _turn_z_up, load_geometry, read_obj_fast, scene_from_obj,
)

# A unit tetrahedron, written the way `export.mesh_export` writes one.
_OBJ = "\n".join([
    "# comment",
    "v 0 0 0", "v 1 0 0", "v 0 1 0", "v 0 0 1",
    "vn 0 0 -1", "vn 0 -1 0", "vn -1 0 0", "vn 1 1 1",
    "g something",
    "f 1//1 3//3 2//2",
    "f 1//1 2//2 4//4",
    "f 1//1 4//4 3//3",
    "f 2//2 3//3 4//4",
]) + "\n"


def _write(tmp_path, text, name="m.obj"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_the_fast_reader_agrees_with_the_real_parser(tmp_path):
    fast = read_obj_fast(_write(tmp_path, _OBJ))
    slow = parse_obj(_OBJ)
    assert fast.vertex_count == slow.vertex_count
    np.testing.assert_allclose(np.asarray(fast.positions), np.asarray(slow.positions))
    np.testing.assert_array_equal(np.asarray(fast.indices), np.asarray(slow.indices))


def test_the_fast_reader_handles_the_other_face_spellings(tmp_path):
    bare = _OBJ.replace("//1", "").replace("//2", "").replace("//3", "").replace("//4", "")
    slashed = _OBJ.replace("//", "/7/")
    for text in (bare, slashed):
        geom = read_obj_fast(_write(tmp_path, text))
        assert geom.vertex_count == 4
        np.testing.assert_array_equal(np.asarray(geom.indices),
                                      np.asarray(parse_obj(_OBJ).indices))


def test_a_file_with_no_faces_is_refused(tmp_path):
    with pytest.raises(ValueError):
        read_obj_fast(_write(tmp_path, "v 0 0 0\nv 1 0 0\n"))


def test_small_files_go_through_the_real_parser(tmp_path):
    """Below the size threshold the vectorised path is not worth its risk."""
    geom = load_geometry(_write(tmp_path, _OBJ))
    assert geom.vertex_count == 4


def test_turning_z_up_stands_a_y_up_model_upright():
    """The gym scene is Y-up; OrbitControls measures its polar angle from +Z."""
    np.testing.assert_allclose(np.array([0.0, 1.0, 0.0]) @ _Y_UP_TO_Z_UP.T,
                               [0.0, 0.0, 1.0], atol=1e-12)

    class G:
        positions = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        normals = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    g = G()
    _turn_z_up(g)
    np.testing.assert_allclose(g.positions, [0.0, 0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(g.normals, [0.0, 0.0, 1.0], atol=1e-6)


def test_scene_from_obj_reports_the_models_own_bounds(tmp_path):
    scene, lo, hi = scene_from_obj(_write(tmp_path, _OBJ))
    assert len(scene.collect_meshes()) == 1
    # y-up (0,1,0) has become z-up (0,0,1), so the tallest point is on z.
    np.testing.assert_allclose(hi - lo, [1.0, 1.0, 1.0], atol=1e-6)


def test_an_empty_file_is_refused_rather_than_shown_as_nothing(tmp_path):
    with pytest.raises(Exception):
        scene_from_obj(_write(tmp_path, "# nothing here\n"))
