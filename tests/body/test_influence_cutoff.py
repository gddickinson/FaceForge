"""A skin vertex's influences must be local, and must fade rather than drop.

The influence solve gives each skin vertex its K nearest bone segments with
inverse-distance weights, cut off at a distance ``d_cut``.  ``d_cut`` used to
be the distance to the (K+1)-th nearest segment, which is smooth but not
local: for a limb vertex the third and fourth segments are a whole joint
further along the chain, so a thigh vertex carried real weight on the ankle.

Measured on the 791,729-vertex skin at a deadlift-style hip hinge, torn edges
(stretched past 2x their rest length) were 18,850 with two influences, 58,841
with three and 55,568 with four -- admitting a third influence at all is what
tore it.  A band measured from the *nearest* segment keeps the set local and
still lets a departing segment's weight reach zero smoothly.

The band is additive, not a multiple of the nearest distance: a multiple
collapses to nothing where the skin lies on the bone, and every vertex with
d1 ~ 0 then fell back to rigid binding beside neighbours that blended.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.soft_tissue import SoftTissueSkinning


def _solver(band: float, k: int = 4) -> SoftTissueSkinning:
    sk = SoftTissueSkinning()
    sk.SKIN_INFLUENCES = k
    sk.INFLUENCE_CUTOFF_BAND = band
    return sk


#: Six candidate segments, indices 0..5.
SEG = np.arange(6, dtype=np.int32)


def _weights_for(sk, distances):
    dists = np.asarray([distances], dtype=np.float64)
    inf, w = sk._solve_multi_influence(dists, SEG)
    out = np.zeros(len(SEG))
    for slot in range(inf.shape[1]):
        out[int(inf[0, slot])] += float(w[0, slot])
    return out


def test_a_segment_beyond_the_band_gets_no_weight():
    sk = _solver(band=3.0)
    # Nearest at 2.0, so the band reaches 5.0.  The segment at 9.0 is outside.
    w = _weights_for(sk, [2.0, 3.0, 4.0, 9.0, 20.0, 50.0])
    assert w[3] == pytest.approx(0.0)
    assert w[4] == pytest.approx(0.0)
    assert w[0] > w[1] > w[2] > 0.0
    assert w.sum() == pytest.approx(1.0)


def test_the_rank_based_cutoff_does_reach_that_far():
    """The control: without a band the fourth segment keeps real weight."""
    sk = _solver(band=0.0)
    w = _weights_for(sk, [2.0, 3.0, 4.0, 9.0, 20.0, 50.0])
    assert w[3] > 0.01, w


def test_weight_fades_to_zero_as_a_segment_leaves_the_band():
    sk = _solver(band=3.0)
    tail = []
    for d in (4.6, 4.8, 4.95, 4.99, 5.0, 5.2):
        w = _weights_for(sk, [2.0, 2.5, d, 30.0, 40.0, 50.0])
        tail.append(w[2])
    assert tail == sorted(tail, reverse=True), tail
    assert tail[-1] == pytest.approx(0.0)
    assert tail[-2] == pytest.approx(0.0, abs=2e-3)
    assert tail[0] > 0.0


def test_the_support_does_not_collapse_on_the_bone():
    """A vertex sitting on its segment still blends; a ratio cutoff did not."""
    sk = _solver(band=3.0)
    w = _weights_for(sk, [0.0, 1.0, 2.0, 30.0, 40.0, 50.0])
    assert w[0] > 0.9, w            # the segment it sits on dominates
    assert w.sum() == pytest.approx(1.0)

    nearby = _weights_for(sk, [0.05, 1.0, 2.0, 30.0, 40.0, 50.0])
    assert abs(nearby[1] - w[1]) < 0.05, (w, nearby)


def test_rows_sum_to_one_and_indices_stay_valid():
    sk = _solver(band=3.0)
    rng = np.random.default_rng(0)
    dists = rng.uniform(0.1, 40.0, size=(200, 6))
    inf, w = sk._solve_multi_influence(dists, SEG)
    assert inf.shape == (200, 4) and w.shape == (200, 4)
    assert w.sum(axis=1) == pytest.approx(np.ones(200))
    assert inf.min() >= 0 and inf.max() < len(SEG)


def test_an_ineligible_segment_never_contributes():
    """Excluded chains arrive as inf and must stay at zero weight."""
    sk = _solver(band=3.0)
    w = _weights_for(sk, [2.0, np.inf, 3.0, np.inf, 4.0, np.inf])
    assert w[1] == 0.0 and w[3] == 0.0 and w[5] == 0.0
    assert w.sum() == pytest.approx(1.0)


def test_the_band_is_carried_into_the_binding_cache_key():
    """Two bands must not share a cached solve."""
    from faceforge.body import skinning_cache

    sk = _solver(band=3.0)
    a = dict(skinning_cache.scalar_tunables(sk))
    sk.INFLUENCE_CUTOFF_BAND = 1.5
    b = dict(skinning_cache.scalar_tunables(sk))
    assert a["INFLUENCE_CUTOFF_BAND"] != b["INFLUENCE_CUTOFF_BAND"]
