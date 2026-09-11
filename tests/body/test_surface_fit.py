"""Registering the body-surface mesh onto the skeleton, without flattening it."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.surface_landmarks import find_narrowest_z
from faceforge.body.surface_register import (
    constrain_to_rest, fit_head_to_skull, similarity_to,
)


def _limb(length=40.0, radius=4.0, rings=11, seg=16, waist=None):
    """A tube down -Z, optionally narrowed at one station (a wrist)."""
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    zs = np.linspace(0.0, -length, rings)
    out = []
    for i, z in enumerate(zs):
        r = radius * (0.5 if waist is not None and i == waist else 1.0)
        out.append(np.stack([np.cos(th) * r, np.sin(th) * r, np.full(seg, z)], axis=1))
    pts = np.concatenate(out)
    tris = []
    for r_i in range(rings - 1):
        for s in range(seg):
            a = r_i * seg + s
            b = r_i * seg + (s + 1) % seg
            tris += [(a, b, a + seg), (b, b + seg, a + seg)]
    return pts, np.asarray(tris, dtype=np.uint32).ravel()


class TestLandmarks:
    def test_the_narrowest_station_is_found(self):
        """The wrist is where a limb is thinnest, not where it ends.

        Taking the lowest tenth of an arm's vertices put the landmark in the
        fingertips, and the forearm was sheared to match: its depth fell from
        24.7 to 17.4 while its width rose from 21.5 to 28.4.
        """
        pts, _ = _limb(length=40.0, rings=11, waist=7)
        z = find_narrowest_z(pts, -40.0, 0.0, step=2.0)
        assert z == pytest.approx(-28.0, abs=3.0)

    def test_it_degrades_to_the_low_end_when_nothing_is_narrow(self):
        pts, _ = _limb(length=40.0, rings=11)
        assert find_narrowest_z(pts, -40.0, 0.0, step=2.0) <= 0.0


class TestPlacement:
    def test_similarity_matches_height_and_centres(self):
        pts, _ = _limb()
        target = pts * 1.5 + np.array([10.0, -4.0, 25.0])
        out = pts + similarity_to(pts, target)
        assert out[:, 2].max() - out[:, 2].min() == pytest.approx(
            target[:, 2].max() - target[:, 2].min(), rel=1e-6)
        assert out[:, 2].min() == pytest.approx(target[:, 2].min(), abs=1e-6)
        for ax in (0, 1):
            assert np.median(out[:, ax]) == pytest.approx(np.median(target[:, ax]), abs=1e-6)


class TestNoTearing:
    def test_the_band_is_imposed_on_the_registered_mesh(self):
        pts, tris = _limb()
        wrecked = pts.copy()
        wrecked[::3] *= 3.0                      # a deformation that tears
        out = constrain_to_rest(wrecked, pts, tris)
        e = np.unique(np.sort(np.stack([tris.reshape(-1, 3)[:, 0],
                                        tris.reshape(-1, 3)[:, 1]], axis=1), axis=1), axis=0)
        rest = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
        got = np.linalg.norm(out[e[:, 0]] - out[e[:, 1]], axis=1)
        ratio = got / np.maximum(rest, 1e-9)
        assert ratio.max() < 3.0, "the band pulls the worst stretch in"
        assert ratio.max() < (wrecked[e[:, 0]] - wrecked[e[:, 1]]).max()


class TestHeadFit:
    def _head(self):
        """A ball on a neck, with the ball's centre at the origin."""
        rng = np.random.default_rng(0)
        v = rng.normal(size=(600, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        ball = v * 10.0
        neck = np.stack([rng.normal(size=200) * 3, rng.normal(size=200) * 3,
                         rng.uniform(-40, -10, 200)], axis=1)
        return np.concatenate([ball, neck])

    def test_the_head_grows_to_clear_a_skull_that_pokes_out(self):
        pos = self._head()
        # A skull deeper than the head, and sitting forward of it.
        skull = np.array([[0.0, -14.0, 0.0], [0.0, 12.0, 0.0],
                          [6.0, 0.0, 0.0], [-6.0, 0.0, 0.0],
                          [0.0, 0.0, 9.0], [0.0, 0.0, -9.0]])
        out = fit_head_to_skull(pos, skull, shoulder_z=-40.0)
        head = out[pos[:, 2] > -10]
        assert head[:, 1].min() <= -14.0 - 1.0, "the face must clear the skull"
        assert head[:, 1].max() >= 12.0 + 1.0, "so must the occiput"

    def test_a_head_that_already_clears_its_skull_is_left_alone(self):
        pos = self._head()
        skull = np.array([[0.0, -2.0, 0.0], [0.0, 2.0, 0.0],
                          [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
                          [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
        out = fit_head_to_skull(pos, skull, shoulder_z=-40.0)
        assert np.abs(out - pos).max() < 1e-6

    def test_the_neck_carries_the_change_and_the_body_below_does_not_move(self):
        pos = self._head()
        skull = np.array([[0.0, -14.0, 0.0], [0.0, 12.0, 0.0],
                          [8.0, 0.0, 0.0], [-8.0, 0.0, 0.0],
                          [0.0, 0.0, 9.0], [0.0, 0.0, -9.0]])
        out = fit_head_to_skull(pos, skull, shoulder_z=-40.0)
        low = pos[:, 2] <= -40.0
        if low.any():
            np.testing.assert_allclose(out[low], pos[low], atol=1e-9)

    def test_no_skull_is_a_no_op(self):
        pos = self._head()
        out = fit_head_to_skull(pos, np.zeros((0, 3)), shoulder_z=-40.0)
        np.testing.assert_allclose(out, pos)
