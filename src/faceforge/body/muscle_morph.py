"""The sex difference in muscle bulk, as a change of belly cross-section.

Muscle *length* is set by the bones, so a skeleton that has been scaled
already gives the muscles the right length through the skinning.  What the
skeleton cannot give them is girth, and girth is where the sex difference
actually lives: whole-body skeletal muscle mass is about two thirds of the
male value in females, and the published cross-sectional-area ratios are
strongly regional -- roughly 0.6 in the upper limb and shoulder girdle,
0.75 in the lower limb, with the trunk in between.  Upper-body dimorphism
being the larger is one of the most repeatable findings in the literature
(women's arm muscle CSA is about 60 % of men's, their thigh about 75 %).

So each muscle is thinned *perpendicular to its own long axis*, leaving its
length alone.  The thinning is tapered to zero at the two axial extremes,
which is where the tendons and the bony footprints are: a belly thins, an
attachment stays attached.  Without the taper a broad muscle's footprint
pulls off its bone.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Chain-name prefix -> female/male belly cross-section ratio.  A muscle is
#: assigned the region of the kinematic chain most of its vertices bind to.
REGION_BULK: dict[str, float] = {
    "arm": 0.62,        # upper limb: the most dimorphic region
    "hand": 0.78,       # intrinsics scale closer to hand size than to bulk
    "leg": 0.76,        # lower limb
    "foot": 0.82,
    "spine": 0.70,      # trunk
    "ribs": 0.72,
}

#: Used when a muscle's chain cannot be identified.
DEFAULT_BULK = 0.70

#: Below this the taper leaves the ends alone; see ``_taper``.
TAPER_POWER = 2.0


def _principal_axis(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """The muscle's long axis: the first principal component of its vertices."""
    centred = pts - pts.mean(axis=0)
    # A full SVD of a 60k x 3 matrix is wasteful; the 3x3 covariance is exact
    # for what is wanted and costs one pass.
    cov = centred.T @ centred
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmax(w))]
    n = np.linalg.norm(axis)
    return axis / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _taper(t_norm: NDArray[np.float64]) -> NDArray[np.float64]:
    """1 at mid-belly, 0 at both ends, so attachments do not move."""
    return np.clip(1.0 - np.abs(t_norm) ** TAPER_POWER, 0.0, 1.0)


class MuscleMorph:
    """Thins muscle bellies toward the female cross-section and back again.

    Every value is computed from a captured copy of the original rest
    positions, so the slider can be moved in any order and 0 restores the
    muscles exactly.
    """

    def __init__(self, region_bulk: dict[str, float] | None = None) -> None:
        self._bulk = dict(REGION_BULK if region_bulk is None else region_bulk)
        self._axis: dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float]] = {}
        self._region: dict[int, str] = {}

    # -- classification ------------------------------------------------------

    def region_of(self, binding: Any, chain_names: dict[int, str] | None) -> str:
        """Which region a muscle belongs to, from the chain it mostly binds to."""
        key = id(binding)
        cached = self._region.get(key)
        if cached is not None:
            return cached
        name = "?"
        chain_of = getattr(binding, "_chain_of_joint", None)
        if chain_names and chain_of is not None:
            ji = np.asarray(binding.joint_indices)
            if len(ji):
                chains = chain_of[ji]
                counts = np.bincount(chains[chains >= 0]) if (chains >= 0).any() else None
                if counts is not None and len(counts):
                    name = chain_names.get(int(np.argmax(counts)), "?")
        self._region[key] = name
        return name

    def bulk_factor(self, region: str, gender: float) -> float:
        """Radial factor at ``gender`` for a chain named ``region``."""
        target = DEFAULT_BULK
        for prefix, value in self._bulk.items():
            if region.startswith(prefix):
                target = value
                break
        g = float(max(0.0, min(1.0, gender)))
        return 1.0 + (target - 1.0) * g

    # -- application ---------------------------------------------------------

    def thin_belly(self, mesh: Any, points: NDArray, factor: float) -> NDArray:
        """Thin ``points`` perpendicular to the muscle's own long axis.

        The axis and the half-length are taken once from the muscle's original
        rest pose and cached, so the belly is thinned about a fixed anatomical
        direction rather than about whatever axis the current shape happens to
        have -- which would drift as the slider moved.
        """
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 4 or abs(factor - 1.0) < 1e-6:
            return pts
        key = id(mesh)
        cached = self._axis.get(key)
        if cached is None:
            centre = pts.mean(axis=0)
            axis = _principal_axis(pts)
            t0 = (pts - centre) @ axis
            half = float(np.abs(t0).max())
            cached = self._axis[key] = (centre, axis, half if half > 1e-9 else 1.0)
        _, axis, half = cached
        centre = pts.mean(axis=0)
        t = (pts - centre) @ axis
        radial = (pts - centre) - t[:, None] * axis[None, :]
        w = _taper(t / half)
        scale = 1.0 - (1.0 - factor) * w
        return centre + t[:, None] * axis[None, :] + radial * scale[:, None]
