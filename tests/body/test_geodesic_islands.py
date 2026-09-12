"""Two rules that stop Euclidean proximity deciding what the skin follows.

**Seed confidence.**  In the rest pose the arms hang against the trunk, so a
straight-line distance treats the gap between forearm and waist as tissue.
Measured at full abduction, the worst edge in the mesh joined two vertices
0.109 units apart on the flank: one took the lumbar spine, the other took
elbow, shoulder and wrist, and flew 60 units with the raised arm.  A vertex
whose nearest chain is not clearly nearest now seeds nothing; the geodesic
fields reach it from unambiguous territory instead.

**Island bridging.**  The body skin is not one surface: 791,729 vertices fall
into 528 connected components, 28,201 of them off the main one.  Dijkstra
never reaches an island, so its geodesic distance is infinite and the solve
falls back to the Euclidean measurement the geodesic pass exists to overrule.
Islands are joined into the Dijkstra graph at their few closest contacts --
not vertex by vertex, which glued a patch on the lateral chest across the
armpit to the arm.  The bridge is never added to ``edge_pairs``, which the
stretch metrics and edge relaxation read as real topology.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.soft_tissue import SoftTissueSkinning


def _two_patches():
    """A main sheet of 5x5, plus a detached 2x2 island 1 unit above its edge."""
    g = np.linspace(0.0, 8.0, 5)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    main = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    island = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0],
                       [10.0, 1.0, 0.0], [11.0, 1.0, 0.0]])
    positions = np.vstack([main, island])

    idx = np.arange(25).reshape(5, 5)
    edges = np.concatenate([
        np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1),
        np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1),
    ])
    island_edges = np.array([[25, 26], [25, 27], [26, 28], [27, 28]])
    edges = np.vstack([edges, island_edges])
    lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    return positions, edges, lengths


def test_an_island_is_joined_to_the_main_surface():
    positions, edges, lengths = _two_patches()
    sk = SoftTissueSkinning()
    sk.GEODESIC_BRIDGE = 5.0
    out_e, out_l = sk._bridge_mesh_islands(positions, edges, lengths)
    assert len(out_e) > len(edges)
    assert sk.last_bridge["components"] == 2
    assert sk.last_bridge["island_vertices"] == 4
    added = out_e[len(edges):]
    # Every new edge joins the island to the main sheet.
    assert ((added >= 25).sum(axis=1) == 1).all(), added


def test_a_patch_is_joined_at_a_few_contacts_not_at_every_vertex():
    positions, edges, lengths = _two_patches()
    sk = SoftTissueSkinning()
    sk.GEODESIC_BRIDGE = 5.0
    sk.BRIDGE_CONTACTS = 2
    out_e, _ = sk._bridge_mesh_islands(positions, edges, lengths)
    assert len(out_e) - len(edges) == 2
    assert sk.last_bridge["bridged"] == 2


def test_bridging_off_leaves_the_graph_alone():
    positions, edges, lengths = _two_patches()
    sk = SoftTissueSkinning()
    sk.GEODESIC_BRIDGE = 0.0
    out_e, out_l = sk._bridge_mesh_islands(positions, edges, lengths)
    assert out_e is edges and out_l is lengths


def test_a_gap_wider_than_the_limit_is_not_bridged():
    positions, edges, lengths = _two_patches()
    positions[25:, 0] += 40.0          # move the island far away
    sk = SoftTissueSkinning()
    sk.GEODESIC_BRIDGE = 5.0
    out_e, out_l = sk._bridge_mesh_islands(positions, edges, lengths)
    assert out_e is edges and out_l is lengths


def test_a_connected_mesh_is_returned_unchanged():
    positions, edges, lengths = _two_patches()
    edges = np.vstack([edges, np.array([[24, 25]])])
    lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    sk = SoftTissueSkinning()
    sk.GEODESIC_BRIDGE = 5.0
    out_e, out_l = sk._bridge_mesh_islands(positions, edges, lengths)
    assert out_e is edges and out_l is lengths


def test_an_ambiguous_vertex_seeds_nothing():
    """Equidistant from two chains, a vertex must not claim either."""
    g = np.linspace(-10.0, 10.0, 21)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    positions = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    idx = np.arange(21 * 21).reshape(21, 21)
    edges = np.concatenate([
        np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1),
        np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1),
    ])
    lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    # Two parallel bones the same distance below the sheet, left and right.
    seg_starts = np.array([[-8.0, -10.0, -2.0], [8.0, -10.0, -2.0]])
    seg_ends = np.array([[-8.0, 10.0, -2.0], [8.0, 10.0, -2.0]])
    seg_chains = np.array([0, 1], dtype=np.int32)

    sk = SoftTissueSkinning()
    sk.SEED_FROM_OWNED_SKIN = True
    sk.SEED_CONFIDENCE_MARGIN = 1.5
    geo = sk._geodesic_chain_dists(
        positions, edges, lengths, seg_starts, seg_ends, seg_chains)
    # The midline is equidistant, so neither field may read as ~0 there.
    midline = np.where(np.abs(positions[:, 0]) < 1e-9)[0]
    assert geo[midline].min() > 2.0, geo[midline].min()
    # Directly over a bone, that chain is still nearly at zero.
    over = np.argmin(np.linalg.norm(positions - np.array([-8.0, 0.0, 0.0]), axis=1))
    assert geo[over, 0] < geo[over, 1]


def test_the_margin_and_bridge_are_in_the_binding_cache_key():
    from faceforge.body import skinning_cache

    sk = SoftTissueSkinning()
    a = dict(skinning_cache.scalar_tunables(sk))
    for name in ("SEED_CONFIDENCE_MARGIN", "GEODESIC_BRIDGE", "BRIDGE_CONTACTS"):
        assert name in a, name
    sk.SEED_CONFIDENCE_MARGIN = 9.0
    b = dict(skinning_cache.scalar_tunables(sk))
    assert a["SEED_CONFIDENCE_MARGIN"] != b["SEED_CONFIDENCE_MARGIN"]
