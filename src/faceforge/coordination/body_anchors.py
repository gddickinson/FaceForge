"""The body anchors the neck muscles track, read in the body frame.

The neck muscle group hangs off ``bodyRoot``.  In scene mode a
``scene_wrapper`` node is inserted above ``bodyRoot`` and carries the whole
body's placement in the room (the gym wrapper is Rx(-90 deg) at Y = 203), so
a pivot's *world* position there is the room position, not the body one.
The rest anchors are snapshotted at load time, when no wrapper exists, so
every current reading has to be brought back into that frame before anyone
subtracts the two.

Measured before this module existed, one frame into a bodyweight squat:
the thoracic anchor read (0.225, 192.088, -5.661) against a rest of
(0.225, 5.661, -10.912), a 186-unit delta that the neck muscles then blended
into their lower vertices -- Sternohyoid L moved 183 units off its rest
position and its 99th-percentile edge stretched 55x.

This is the same frame mismatch that :meth:`SoftTissueSkinning._wrapper_cancel`
exists to prevent, and that :class:`BoneAnchorRegistry` cancels for muscle
pinning; the neck path was the one reader still comparing frames.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.math_utils import mat4_inverse
from faceforge.core.scene_graph import SceneNode

#: The anchor names the neck muscles look up by ``lowerAttach``.
ANCHOR_NAMES = ("shoulder", "ribcage", "thoracic")

#: How many of the upper ribs the ribcage anchor averages.
RIB_SAMPLE = 4


def wrapper_cancel(wrapper: Optional[SceneNode]) -> Optional[NDArray[np.float64]]:
    """Inverse of the scene wrapper's world matrix, or ``None`` outside scene mode.

    A wrapper that is not in the scene graph carries no placement, so it is
    treated as absent -- the same test :meth:`_wrapper_cancel` uses.

    The refresh is *not* forced.  Forcing the wrapper rebuilds every one of
    the ~900 nodes hanging beneath it, and the skinning already pays that
    once a frame; call this after a scene update, which is where the callers
    in :mod:`faceforge.coordination.simulation` sit.
    """
    if wrapper is None or wrapper.parent is None:
        return None
    wrapper.update_world_matrix()
    return np.asarray(mat4_inverse(wrapper.world_matrix), dtype=np.float64)


def to_body_frame(
    point: NDArray[np.float64],
    cancel: Optional[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """A world-space point brought back into the body frame."""
    p = np.asarray(point, dtype=np.float64)
    if cancel is None:
        return p
    return (cancel @ np.append(p, 1.0))[:3]


def body_anchor_positions(
    body_animation: Any,
    cancel: Optional[NDArray[np.float64]] = None,
) -> dict[str, NDArray[np.float64]]:
    """Shoulder, ribcage and thoracic anchor positions in the body frame.

    Parameters
    ----------
    body_animation:
        The live :class:`BodyAnimation`; its pivots are read, never written.
    cancel:
        The scene wrapper's inverse from :func:`wrapper_cancel`, or ``None``
        when the body is not inside a scene.

    Returns
    -------
    dict
        Only the anchors whose pivots exist.  An empty dict means the body
        skeleton is not wired yet and the caller should leave the previous
        anchors alone.
    """
    anchors: dict[str, NDArray[np.float64]] = {}
    if body_animation is None:
        return anchors

    def _at(node: SceneNode) -> NDArray[np.float64]:
        return to_body_frame(node.get_world_position(), cancel)

    # Thoracic: the highest thoracic pivot (T1, the one nearest the neck).
    thoracic_pivots = getattr(body_animation, "thoracic_pivots", None) or []
    if thoracic_pivots:
        group = thoracic_pivots[0].get("group")
        if group is not None:
            anchors["thoracic"] = _at(group)

    # Shoulder: the mean of the two glenohumeral pivots.
    joints = getattr(body_animation, "joints", None)
    if joints is not None:
        pivots = getattr(joints, "pivots", {})
        shoulders = [pivots[f"shoulder_{side}"] for side in ("R", "L")
                     if pivots.get(f"shoulder_{side}") is not None]
        if shoulders:
            anchors["shoulder"] = np.mean([_at(n) for n in shoulders], axis=0)

    # Ribcage: the mean of the upper ribs.
    rib_pivots = getattr(body_animation, "_rib_pivots", None) or []
    ribs = [p for p in rib_pivots[:RIB_SAMPLE] if p is not None]
    if ribs:
        anchors["ribcage"] = np.mean([_at(n) for n in ribs], axis=0)

    return anchors
