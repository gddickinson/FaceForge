"""`inside_depth`: the shapes that made four earlier versions of it lie.

Each test here is a shape that a previous implementation got wrong, kept so
the next simplification has to survive them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.audit_implement_contact import inside_depth


def _bar(length=280.0, radius=1.8, tilt_deg=0.0, centre=(0.0, 0.0, 0.0), n=12):
    """A cylinder along +X, optionally tilted about +Y, as world vertices."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.zeros_like(t), radius * np.cos(t), radius * np.sin(t)], axis=1)
    ends = np.array([-length / 2, length / 2])
    pts = np.concatenate([ring + np.array([e, 0, 0]) for e in ends])
    a = np.radians(tilt_deg)
    rot = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    return pts @ rot.T + np.asarray(centre)


def _disc(radius=12.0, thick=4.0, centre=(0.0, 0.0, 0.0), n=24):
    """A plate: thin in x, wide in y/z."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.zeros_like(t), radius * np.cos(t), radius * np.sin(t)], axis=1)
    return np.concatenate([ring + np.array([s * thick / 2, 0, 0])
                           for s in (-1, 1)]) + np.asarray(centre)


def _blob(centre, spread=1.0, n=200, seed=0):
    """A little cloud of 'body surface'."""
    rng = np.random.default_rng(seed)
    return np.asarray(centre) + rng.normal(scale=spread, size=(n, 3))


def test_a_bar_resting_against_a_surface_is_not_inside_it():
    bar = _bar()
    just_below = _blob((0.0, -2.2, 0.0), spread=0.05)
    assert inside_depth(bar, just_below) == pytest.approx(0.0, abs=0.2)


def test_a_tilted_bar_is_measured_by_its_own_radius_not_its_bounding_box():
    """A front-racked bar sits ~10 deg off level; its AABB cross-section is the tilt."""
    bar = _bar(tilt_deg=10.0)
    touching = _blob((0.0, -1.9, 0.0), spread=0.02)
    assert inside_depth(bar, touching) < 1.0, "the tilt must not inflate the radius"


def test_a_plate_beside_a_limb_is_not_inside_it():
    """A curl's plate lies flat beside the thigh: 2 units clear in x, not 10 in y."""
    plate = _disc(centre=(24.0, 0.0, 0.0))
    thigh = _blob((20.0, 0.0, 0.0), spread=0.05)   # 2 units clear of the -x face
    assert inside_depth(plate, thigh) == pytest.approx(0.0, abs=0.3)


def _ball(radius=11.0, centre=(0.0, 0.0, 0.0), n=16):
    """A sphere's surface, the shape of a kettlebell's bell."""
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    pts = np.stack([np.sin(uu) * np.cos(vv), np.sin(uu) * np.sin(vv),
                    np.cos(uu)], axis=-1).reshape(-1, 3) * radius
    return pts + np.asarray(centre)


def test_a_bell_hanging_between_two_shins_is_not_inside_them():
    """An 11-radius bell swung between shins 13 units either side of it."""
    bell = _ball(radius=11.0, centre=(0.0, 20.0, 0.0))
    shins = np.concatenate([_blob((-13.0, 20.0, 0.0), 0.05, seed=2),
                            _blob((13.0, 20.0, 0.0), 0.05, seed=3)])
    assert inside_depth(bell, shins) == pytest.approx(0.0, abs=0.5)


def test_something_genuinely_buried_is_reported():
    plate = _disc(radius=12.0, thick=4.0)
    through_the_middle = _blob((0.0, 0.0, 0.0), spread=0.05)
    assert inside_depth(plate, through_the_middle) > 1.5


def test_an_empty_surface_or_a_degenerate_part_is_zero_not_a_crash():
    assert inside_depth(_bar(), np.zeros((0, 3))) == 0.0
    assert inside_depth(np.zeros((1, 3)), _blob((0, 0, 0))) == 0.0
