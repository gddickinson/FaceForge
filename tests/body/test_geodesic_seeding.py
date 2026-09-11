"""A chain's geodesic field must reach the skin that lies over its bones.

The skin's chain ranking is geodesic: each chain seeds a distance field on the
mesh and a vertex takes the chains whose fields reach it first.  Seeding used
to require the skin to come within ``SEED_RADIUS`` of the bone, which makes it
a contest between *superficial* bones rather than the right ones.  The
clavicle and scapula are subcutaneous and the vertebral bodies are not, so
skin over the upper thoracic spine -- 8.78 units from its own vertebra against
12.13 to the clavicle -- was never a spine seed and was handed a short path to
the shoulder girdle instead.  It came out with a quarter of its motion on the
collar bone, and abducting an arm moved midline back skin by up to 10.4 units.

Seeding each chain from the skin it owns, as well as from the radius, removes
that.  These tests pin the rule on a synthetic two-chain rig: a sheet lying
close to a superficial bone and far from a deep one, where the deep bone is
still the nearer of the two.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.soft_tissue import SoftTissueSkinning


def _sheet(n: int = 40, span: float = 20.0) -> np.ndarray:
    """A flat sheet of vertices in the z = 0 plane."""
    g = np.linspace(-span, span, n)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)


def _grid_edges(n: int = 40):
    idx = np.arange(n * n).reshape(n, n)
    e = np.concatenate([
        np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1),
        np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1),
    ])
    return e


def _fields(skinning, positions, edges):
    lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    # Chain 0: a DEEP bone under the middle of the sheet, 8 units below.
    # Chain 1: a SUPERFICIAL bone off to one side, 1 unit below the sheet.
    seg_starts = np.array([[-20.0, 0.0, -8.0], [16.0, -20.0, -1.0]])
    seg_ends = np.array([[20.0, 0.0, -8.0], [16.0, 20.0, -1.0]])
    seg_chains = np.array([0, 1], dtype=np.int32)
    return skinning._geodesic_chain_dists(
        positions, edges, lengths, seg_starts, seg_ends, seg_chains)


def test_a_deep_bone_reaches_the_skin_directly_over_it():
    positions = _sheet()
    edges = _grid_edges()
    sk = SoftTissueSkinning()
    sk.SEED_FROM_OWNED_SKIN = True
    geo = _fields(sk, positions, edges)

    # A vertex on the midline: 8 units from the deep bone, 16 from the
    # superficial one, so the deep bone is nearer and must win geodesically.
    centre = np.argmin(np.linalg.norm(positions - np.array([0.0, 0.0, 0.0]), axis=1))
    assert geo[centre, 0] < geo[centre, 1], geo[centre]


def test_the_radius_rule_alone_hands_that_skin_to_the_superficial_bone():
    """The control: this is the defect, reproduced."""
    positions = _sheet()
    edges = _grid_edges()
    sk = SoftTissueSkinning()
    sk.SEED_FROM_OWNED_SKIN = False
    geo = _fields(sk, positions, edges)

    centre = np.argmin(np.linalg.norm(positions - np.array([0.0, 0.0, 0.0]), axis=1))
    assert geo[centre, 1] < geo[centre, 0], geo[centre]


def test_the_superficial_bone_still_owns_the_skin_beside_it():
    positions = _sheet()
    edges = _grid_edges()
    sk = SoftTissueSkinning()
    sk.SEED_FROM_OWNED_SKIN = True
    geo = _fields(sk, positions, edges)

    near = np.argmin(np.linalg.norm(positions - np.array([20.0, 0.0, 0.0]), axis=1))
    assert geo[near, 1] < geo[near, 0], geo[near]


def test_every_vertex_is_reached_by_both_fields():
    positions = _sheet()
    edges = _grid_edges()
    sk = SoftTissueSkinning()
    sk.SEED_FROM_OWNED_SKIN = True
    geo = _fields(sk, positions, edges)
    assert np.isfinite(geo).all()
    assert (geo >= 0.0).all()


def test_the_seeding_rule_is_carried_into_the_binding_cache_key():
    from faceforge.body import skinning_cache

    sk = SoftTissueSkinning()
    sk.SEED_FROM_OWNED_SKIN = True
    a = dict(skinning_cache.scalar_tunables(sk))
    sk.SEED_FROM_OWNED_SKIN = False
    b = dict(skinning_cache.scalar_tunables(sk))
    assert a["SEED_FROM_OWNED_SKIN"] != b["SEED_FROM_OWNED_SKIN"]
    # The rule itself changed, not just this flag, so the cache version moved.
    assert skinning_cache.CACHE_VERSION >= 5
