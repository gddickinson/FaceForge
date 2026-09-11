"""Keep an articulation shut when the two bones either side of it scale apart.

The hierarchy keeps a joint together whenever one bone hangs off the other's
pivot.  One articulation in the rig has neither bone hanging off the other:
the clavicle pivots at the sternoclavicular joint and the scapula about its
own centroid, and the two carry different scale factors, so the
acromioclavicular joint drifted 1.25 units open at gender 1.

The fix is measured rather than assumed.  The contact patch -- the distal
bone's vertices that lie within :data:`CONTACT_EPS` of the proximal bone on
the unscaled skeleton -- is remembered, and after scaling the distal pivot is
translated by the *change* in that patch's mean offset to the proximal
surface.  Taking the raw offset instead would bias the joint at gender 0,
because the nearest point on a patch of finite thickness is never the point
itself, and the skeleton would not return to where it started.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: How near two bones' vertices must be to count as the same articulation.
CONTACT_EPS = 3.0

#: ``(proximal bone, distal bone, distal pivot)``.
ARTICULATIONS: tuple[tuple[str, str, str], ...] = (
    ("Right Clavicle", "Right Scapula", "scapula_R_pivot"),
    ("Left Clavicle", "Left Scapula", "scapula_L_pivot"),
)

#: How near two bones' vertices must be to count as the same articulation.
CONTACT_EPS = 3.0

#: Width of the kernel that turns the joint displacements into a continuous
#: field, in body units.  About the length of a forearm: wide enough that the
#: field varies on the scale of a limb rather than of a mesh edge, narrow
#: enough that the pelvis does not smear into the shoulder.
FIELD_SIGMA = 22.0



def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _body_points(root: Any, bone: str, rest: Optional[dict],
                 node_offset: Callable[[Any], NDArray]) -> Optional[NDArray]:
    """A bone's vertices in body coordinates: captured ones, or live ones."""
    for node in _walk(root):
        if node.name == bone and getattr(node, "mesh", None) is not None:
            if rest is not None:
                src = rest.get(id(node.mesh))
                if src is None:
                    return None
            else:
                src = node.mesh.geometry.positions
            return (np.asarray(src, dtype=np.float64).reshape(-1, 3)
                    + node_offset(node))
    return None


def _patch_offset(pm: Optional[NDArray], dm: Optional[NDArray],
                  patch: NDArray, tree_cls: Any) -> Optional[NDArray]:
    """Mean vector from the contact patch to the proximal bone's surface."""
    if pm is None or dm is None or len(patch) == 0:
        return None
    pts = dm[patch]
    _, near = tree_cls(pm).query(pts, k=1)
    return np.asarray(pm[near] - pts, dtype=np.float64).mean(axis=0)


def close_articulations(root: Any, bone_rest: dict[int, NDArray],
                        node_offset: Callable[[Any], NDArray],
                        cache: dict) -> int:
    """Translate each distal pivot so its articulation shuts again."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:                                       # pragma: no cover
        return 0
    nodes = {n.name: n for n in _walk(root) if n.name}
    closed = 0
    for proximal, distal, pivot_name in ARTICULATIONS:
        pivot = nodes.get(pivot_name)
        if pivot is None:
            continue
        rest_p = _body_points(root, proximal, bone_rest, node_offset)
        rest_d = _body_points(root, distal, bone_rest, node_offset)
        if rest_p is None or rest_d is None:
            continue
        key = (proximal, distal)
        patch = cache.get(key)
        if patch is None:
            d, _ = cKDTree(rest_p).query(rest_d, k=1)
            idx = np.flatnonzero(d <= CONTACT_EPS)
            patch = cache[key] = idx if len(idx) else np.zeros(0, dtype=np.int64)
        if len(patch) == 0:
            continue
        before = _patch_offset(rest_p, rest_d, patch, cKDTree)
        now = _patch_offset(_body_points(root, proximal, None, node_offset),
                            _body_points(root, distal, None, node_offset),
                            patch, cKDTree)
        if before is None or now is None:
            continue
        shift = now - before
        if not np.all(np.isfinite(shift)):
            continue
        pos = np.asarray(pivot.position, dtype=np.float64) + shift
        pivot.set_position(float(pos[0]), float(pos[1]), float(pos[2]))
        closed += 1
    return closed
