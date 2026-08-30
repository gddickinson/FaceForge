"""Tests for :mod:`faceforge.core.mesh` -- geometry, cloning and rest poses.

``BufferGeometry`` and ``MeshInstance`` sit under every deformation system
(skinning, gender morph, FACS), the renderer and the GLB exporter, but had no
direct test coverage.
"""

import numpy as np
import pytest

from faceforge.core.material import Material
from faceforge.core.mesh import BufferGeometry, MeshInstance

TRI_POS = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
TRI_NRM = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32)


def _geom(**kw) -> BufferGeometry:
    kw.setdefault("positions", TRI_POS.copy())
    kw.setdefault("normals", TRI_NRM.copy())
    return BufferGeometry(**kw)


def _mesh() -> MeshInstance:
    return MeshInstance(name="m", geometry=_geom(), material=Material())


# ----------------------------------------------------------------------
# BufferGeometry basics
# ----------------------------------------------------------------------

def test_vertex_count_is_inferred_from_positions():
    assert _geom().vertex_count == 3


def test_explicit_vertex_count_is_respected():
    assert _geom(vertex_count=3).vertex_count == 3


def test_triangle_count_non_indexed():
    assert _geom().triangle_count == 1


def test_triangle_count_indexed():
    g = _geom(indices=np.array([0, 1, 2, 0, 2, 1], dtype=np.uint32))
    assert g.has_indices is True
    assert g.triangle_count == 2


def test_empty_index_array_is_not_treated_as_indexed():
    g = _geom(indices=np.array([], dtype=np.uint32))
    assert g.has_indices is False


def test_get_bounding_center_is_the_centroid():
    np.testing.assert_allclose(_geom().get_bounding_center(), [1 / 3, 1 / 3, 0.0])


# ----------------------------------------------------------------------
# compute_normals
# ----------------------------------------------------------------------

def test_compute_normals_non_indexed_gives_unit_face_normal():
    g = _geom()
    g.normals = np.zeros(9, dtype=np.float32)
    g.compute_normals()
    n = g.normals.reshape(-1, 3)
    # CCW triangle in the XY plane -> +Z
    np.testing.assert_allclose(n, np.tile([0, 0, 1], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)


def test_compute_normals_indexed_averages_shared_vertices():
    # Two triangles folded about the shared edge (0,1): one +Z, one -Z-ish.
    pos = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, -1, 0,
    ], dtype=np.float32)
    g = BufferGeometry(
        positions=pos,
        normals=np.zeros(12, dtype=np.float32),
        indices=np.array([0, 1, 2, 0, 3, 1], dtype=np.uint32),
    )
    g.compute_normals()
    n = g.normals.reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    # Vertex 2 belongs only to the first face -> pure +Z
    np.testing.assert_allclose(n[2], [0, 0, 1], atol=1e-6)


def test_compute_normals_output_is_float32_for_gl():
    g = _geom()
    g.compute_normals()
    assert g.normals.dtype == np.float32


def test_compute_normals_survives_a_degenerate_triangle():
    """Zero-area faces must not produce NaN (they divide by a clamped length)."""
    g = _geom(positions=np.zeros(9, dtype=np.float32))
    g.compute_normals()
    assert np.isfinite(g.normals).all()


# ----------------------------------------------------------------------
# clone
# ----------------------------------------------------------------------

def test_clone_deep_copies_positions_and_normals():
    g = _geom()
    c = g.clone()
    c.positions[0] = 99.0
    assert g.positions[0] == 0.0
    assert c.vertex_count == g.vertex_count


def test_clone_deep_copies_indices():
    g = _geom(indices=np.array([0, 1, 2], dtype=np.uint32))
    c = g.clone()
    assert c.indices is not g.indices
    np.testing.assert_array_equal(c.indices, g.indices)


def test_clone_preserves_vertex_colors():
    """FIXED (was DEFECT clone-drops-vertex-colors).

    ``clone()`` rebuilt the dataclass with only
    positions/normals/indices/vertex_count, so vertex_colors and colors_dirty
    were silently lost and any clone of a COLOR_ATLAS / pathology-tinted mesh
    rendered untinted.
    """
    colors = np.tile([1.0, 0.0, 0.0], 3).astype(np.float32)
    g = _geom(vertex_colors=colors, colors_dirty=True)
    c = g.clone()

    assert c.vertex_colors is not None, "vertex_colors dropped by clone()"
    np.testing.assert_array_equal(c.vertex_colors, colors)
    assert c.vertex_colors is not g.vertex_colors, "vertex_colors aliased, not copied"


def test_clone_marks_copied_colors_dirty_for_upload():
    """The clone has no GL buffer, so its colours have never been uploaded.

    Replaces test_clone_drops_vertex_colors_measured, which pinned the old
    (broken) behaviour.  ``colors_dirty`` is forced True on the clone regardless
    of the source flag -- otherwise a clone of an already-uploaded geometry
    would never stream its colours to its own new VBO.
    """
    g = _geom(vertex_colors=np.zeros(9, dtype=np.float32), colors_dirty=False)
    c = g.clone()
    assert c.vertex_colors is not None
    assert c.colors_dirty is True


def test_clone_without_colors_leaves_them_none():
    c = _geom().clone()
    assert c.vertex_colors is None
    assert c.colors_dirty is False


# ----------------------------------------------------------------------
# MeshInstance
# ----------------------------------------------------------------------

def test_positions_setter_marks_needs_update():
    m = _mesh()
    m.needs_update = False
    m.positions = TRI_POS.copy() * 2
    assert m.needs_update is True


def test_normals_setter_marks_needs_update():
    m = _mesh()
    m.needs_update = False
    m.normals = TRI_NRM.copy()
    assert m.needs_update is True


def test_store_rest_pose_snapshots_and_does_not_alias():
    m = _mesh()
    m.store_rest_pose()
    np.testing.assert_array_equal(m.rest_positions, TRI_POS)

    m.geometry.positions[0] = 42.0
    assert m.rest_positions[0] == 0.0, "rest pose aliases the live positions"


def test_store_rest_pose_can_be_retaken():
    m = _mesh()
    m.store_rest_pose()
    m.positions = TRI_POS.copy() + 5.0
    m.store_rest_pose()
    np.testing.assert_allclose(m.rest_positions, TRI_POS + 5.0)


def test_default_flags():
    m = _mesh()
    assert m.visible is True
    assert m.scene_affected is True
    assert m.needs_update is True
    assert m.gl_handle is None
    assert m.rest_positions is None
