"""The harmonic fibre field: a footprinted muscle stretches between its ends, never bows.

Blending the RIGID images of two joints bowed the latissimus into a loop away
from the trunk at a pull-up's dead hang (a belly vertex 40 units from the
shoulder has a 40-unit arc as its humerus image).  Harmonic interpolation of
the ATTACHMENT displacements puts the belly on the straight fibre between
wherever the two ends are now.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy.fibre_field import build_fibre_field, cached_fibre_field
from faceforge.body import skinning_cache


def _strip(columns: int = 41, rows: int = 3):
    """A flat strip along +X: rows*columns vertices, grid edges."""
    xs = np.arange(columns, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, rows)
    rest = np.array([[x, y, 0.0] for y in ys for x in xs])
    idx = np.arange(rows * columns).reshape(rows, columns)
    edges = []
    edges += [(idx[r, c], idx[r, c + 1]) for r in range(rows) for c in range(columns - 1)]
    edges += [(idx[r, c], idx[r + 1, c]) for r in range(rows - 1) for c in range(columns)]
    origin = idx[:, :2].ravel()
    insertion = idx[:, -2:].ravel()
    return rest, np.array(edges), origin, insertion, idx


def _rotation_about(centre, degrees):
    a = np.radians(degrees)
    r = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0, 0, 1.0]])
    d = np.eye(4)
    d[:3, :3] = r
    d[:3, 3] = np.asarray(centre) - r @ np.asarray(centre)
    return d


def test_nothing_moves_when_neither_joint_moved():
    rest, edges, o, i, _ = _strip()
    field = build_fibre_field(rest, edges, o, i)
    assert field is not None and len(field.solved) == len(rest)
    pos = rest.astype(np.float32).copy()
    field.apply(pos, np.eye(4), np.eye(4))
    assert np.allclose(pos, rest, atol=1e-6)


def test_footprints_land_on_their_rigid_images_and_the_belly_interpolates():
    rest, edges, o, i, idx = _strip()
    field = build_fibre_field(rest, edges, o, i)
    delta_i = _rotation_about((60.0, 0.0, 0.0), 90.0)     # a far pivot: the humerus
    pos = rest.astype(np.float32).copy()
    field.apply(pos, np.eye(4), delta_i)

    assert np.allclose(pos[o], rest[o], atol=1e-4), "origin held by its (static) joint"
    image = rest[i] @ delta_i[:3, :3].T + delta_i[:3, 3]
    assert np.allclose(pos[i], image, atol=1e-4), "insertion on its rotated image"

    h = pos.astype(np.float64) - rest
    h_boundary = np.concatenate([h[o], h[i]])
    lo, hi = h_boundary.min(axis=0) - 1e-6, h_boundary.max(axis=0) + 1e-6
    assert (h >= lo).all() and (h <= hi).all(), \
        "harmonic maximum principle: the belly never overshoots its ends (no bowing)"

    # Along the strip the insertion weight rises monotonically from 0 to 1 and
    # the middle carries half the insertion displacement: a straight fibre.
    mid_row = idx[1]
    c_ins = field.c[mid_row, 1]
    assert np.all(np.diff(c_ins) >= -1e-9)
    assert c_ins[0] == pytest.approx(0.0, abs=1e-9) and c_ins[-1] == pytest.approx(1.0, abs=1e-9)
    assert c_ins[20] == pytest.approx(0.5, abs=0.05)
    mean_ins = h[i].mean(axis=0)
    assert np.allclose(h[idx[1, 20]], 0.5 * mean_ins, atol=0.15 * np.linalg.norm(mean_ins) + 1e-6)


def test_rigid_image_blending_would_have_bowed_the_belly():
    """The control: the defect this replaces.  Half-weight blend of the
    rigid images swings the middle far off the straight fibre."""
    rest, edges, o, i, idx = _strip()
    field = build_fibre_field(rest, edges, o, i)
    delta_i = _rotation_about((60.0, 0.0, 0.0), 90.0)
    pos = rest.astype(np.float32).copy()
    field.apply(pos, np.eye(4), delta_i)
    mid = idx[1, 20]
    fibre = pos[mid].astype(np.float64)
    rigid_mid = rest[mid] @ delta_i[:3, :3].T + delta_i[:3, 3]
    blended = 0.5 * rest[mid] + 0.5 * rigid_mid
    straight = 0.5 * rest[mid] + 0.5 * (rest[i] @ delta_i[:3, :3].T + delta_i[:3, 3]).mean(axis=0) \
        - 0.5 * rest[i].mean(axis=0) + 0.5 * rest[mid]
    assert np.linalg.norm(blended - straight) > 5.0 * np.linalg.norm(fibre - straight)


def test_a_component_without_footprints_is_left_to_the_skinning():
    rest, edges, o, i, _ = _strip()
    n = len(rest)
    island = np.array([[100.0, 100.0, 0.0], [101.0, 100.0, 0.0], [100.0, 101.0, 0.0]])
    rest2 = np.vstack([rest, island])
    edges2 = np.vstack([edges, [(n, n + 1), (n + 1, n + 2), (n, n + 2)]])
    field = build_fibre_field(rest2, edges2, o, i)
    assert field is not None
    assert not np.isin([n, n + 1, n + 2], field.solved).any()
    pos = rest2.astype(np.float32).copy()
    pos[n:] += 3.0                                           # "skinned" elsewhere
    field.apply(pos, np.eye(4), _rotation_about((60.0, 0.0, 0.0), 30.0))
    assert np.allclose(pos[n:], rest2[n:] + 3.0)


def test_degenerate_footprints_give_no_field():
    rest, edges, o, i, _ = _strip()
    assert build_fibre_field(rest, edges, np.array([], int), i) is None
    assert build_fibre_field(rest, edges, o, o) is None, "identical sets cancel out"


def test_cached_field_round_trips_through_the_skinning_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("FACEFORGE_SKIN_CACHE_OFF", raising=False)
    skinning_cache.set_cache_dir(tmp_path)
    try:
        rest, edges, o, i, _ = _strip()
        first = cached_fibre_field(rest, edges, o, i)
        files = list(tmp_path.glob("fibre.*.npz"))
        assert len(files) == 1
        second = cached_fibre_field(rest, edges, o, i)
        assert np.array_equal(first.solved, second.solved)
        assert np.allclose(first.c, second.c) and np.allclose(first.m, second.m)
    finally:
        skinning_cache.set_cache_dir(None)


def test_touching_footprints_are_trimmed_to_a_geodesic_gap():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    from faceforge.anatomy.fibre_field import trim_footprints

    rest, edges, _o, _i, idx = _strip()
    n = len(rest)
    w = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    g = csr_matrix((np.concatenate([w, w]), (np.concatenate([edges[:, 0], edges[:, 1]]),
                                             np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n))
    origin = idx[:, :21].ravel()            # columns 0..20
    insertion = idx[:, 20:].ravel()         # columns 20..40: they touch at column 20
    g_o = dijkstra(g, indices=origin, min_only=True)
    g_i = dijkstra(g, indices=insertion, min_only=True)
    o2, i2 = trim_footprints(g_o, g_i, origin, insertion, fraction=0.25, drop=0.0)
    # The farthest vertex from either set is 20 away, so each side gives up
    # the 5 columns nearest the seam.
    assert set(rest[o2, 0]) == set(range(0, 16))
    assert set(rest[i2, 0]) == set(range(25, 41))
    # The default additionally drops the near 30% of each set: the far ends stay.
    o3, i3 = trim_footprints(g_o, g_i, origin, insertion)
    assert len(o3) < len(o2) and len(i3) < len(i2)
    assert 0.0 in rest[o3, 0] and 40.0 in rest[i3, 0]
    assert rest[o3, 0].max() < rest[o2, 0].max() and rest[i3, 0].min() > rest[i2, 0].min()
    delta = _rotation_about((60.0, 0.0, 0.0), 20.0)

    def worst_edge(o_set, i_set):
        field = build_fibre_field(rest, edges, o_set, i_set)
        pos = rest.astype(np.float32).copy()
        field.apply(pos, np.eye(4), delta)
        return float(np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1).max())

    # Touching sets: the seam edge at column 20 carries the whole insertion
    # displacement (about 13 units); with the gap, ten columns of belly share it.
    assert worst_edge(origin, insertion) > 5.0, "the control: adjacent footprints tear"
    assert worst_edge(o2, i2) < 3.0, "no edge is torn across the seam once a gap exists"


def test_an_isolated_footprint_vertex_is_not_a_point_load():
    rest, edges, o, i, idx = _strip()
    speck = idx[1, 30]                                   # alone, mid-belly
    i_with_speck = np.append(i, speck)
    field = build_fibre_field(rest, edges, o, i_with_speck)
    shift = np.eye(4)
    shift[:3, 3] = [0.0, 0.0, 8.0]
    pos = rest.astype(np.float32).copy()
    field.apply(pos, np.eye(4), shift)
    assert pos[speck, 2] == pytest.approx(0.75 * 8.0, abs=0.6), \
        "the speck interpolates with its neighbours instead of being pinned"
    assert np.allclose(pos[i, 2], 8.0, atol=1e-4)
