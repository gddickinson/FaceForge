"""Points on the hand derived from the digit pivots: where a held bar goes.

When the fingers are curled, the metacarpophalangeal, proximal and distal
interphalangeal pivots of the four fingers and an extrapolated fingertip form
a ring round the held object; the object's axis passes through the ring's
centroid.  Two consumers share this: the equipment rig places bars and
handles there, and the ground lock anchors a hanging body by it, so that the
hands close ON a pull-up bar rather than the wrists sitting at bar height
with the fingers waving above it.

Rigs without finger pivots (test fakes, the head-only scene) get None and
fall back to the wrist.
"""

from __future__ import annotations

import numpy as np

RING_DIGITS = (2, 3, 4, 5)
#: The distal phalanx is about four fifths of the middle one; the tip is
#: extrapolated from the two distal pivots.
TIP_FRACTION = 0.8


def _world(node) -> np.ndarray:
    return np.asarray(node.get_world_position(), dtype=np.float64)


def finger_ring_centre(pivots: dict, side: str,
                       digits: tuple[int, ...] = RING_DIGITS) -> np.ndarray | None:
    """World centroid of the closed-finger ring of ``side`` ("R"/"L"), or None."""
    ring = []
    for digit in digits:
        prox = pivots.get(f"finger_{side}_{digit}_prox")
        mid = pivots.get(f"finger_{side}_{digit}_mid")
        dist = pivots.get(f"finger_{side}_{digit}_dist")
        if prox is None or mid is None or dist is None:
            continue
        p, m, d = _world(prox), _world(mid), _world(dist)
        tip = d + (d - m) * TIP_FRACTION
        ring.append((p + m + d + tip) / 4.0)
    if not ring:
        return None
    return np.mean(ring, axis=0)
