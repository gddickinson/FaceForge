"""Two joint angles that differ between the sexes, and did not in the model.

Most of sexual dimorphism in the skeleton is proportion, and the morph handles
it by scaling bones.  Two of the features a person actually recognises are not
proportions but *angles*, and no amount of scaling produces them:

* **The carrying angle.**  With the arm hanging extended and supinated the
  forearm deviates laterally from the humerus, about 11 degrees in men and 13
  in women.  It is why a woman carrying a bucket holds it further from her leg.
* **Genu valgum.**  The knee, about 6 degrees of valgus in men and 8 in women,
  because a wider pelvis puts the hip further from the midline while the foot
  stays under the body.  It is the skeletal half of the larger female Q-angle.

Measured on the model before this, the carrying angle was 9.1 degrees on the
right and 7.2 on the left, and the knee 1.3 and 4.2 -- the donor's own
anatomy, asymmetric as a real body is -- and neither moved by a hundredth of a
degree between gender 0 and gender 1.

Only the *difference* is applied.  The absolute angles are what this cadaver
had, and correcting those would be inventing a different donor; what the sex
morph is entitled to change is how the two sexes differ from each other.

The rotation is applied to the bones below the joint, not to the joint's own
quaternion, for two reasons: the animation would overwrite a quaternion on the
next frame, and the soft-tissue warp is built by comparing captured bone rest
positions with where the bones end up, so a change made anywhere else is
invisible to it and the muscles would stay behind.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Joint, and how many more degrees of frontal-plane deviation a female
#: skeleton has than a male one.
SEX_DELTA: tuple[tuple[str, float], ...] = (
    ("elbow", 2.0),
    ("knee", 2.0),
)

#: Which way round a positive deviation is, per side.  A rotation about the
#: body's anteroposterior axis by a negative angle carries a downward-pointing
#: segment laterally on the right; the left is its mirror.
SIDE_SIGN: dict[str, float] = {"R": -1.0, "L": 1.0}


def _is_pivot(node: Any) -> bool:
    return "pivot" in (getattr(node, "name", "") or "").lower()


def rotation_about_y(degrees: float) -> NDArray:
    """Rotation in the frontal plane, about the anteroposterior axis."""
    a = np.radians(float(degrees))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _subtree(node: Any) -> Iterable[Any]:
    stack = list(node.children)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def rotate_below(pivot: Any, matrix: NDArray, exclude: set[int]) -> int:
    """Turn everything below a joint about an axis through it.

    A descendant's position in the body is the sum of the local positions down
    the chain from the joint, so rotating every one of them rotates the sum;
    and every mesh below it is expressed in a frame that does not itself
    rotate, so each takes the same matrix.
    """
    moved = 0
    for node in _subtree(pivot):
        position = np.asarray(node.position, dtype=np.float64)
        if np.any(position):
            turned = matrix @ position
            node.set_position(float(turned[0]), float(turned[1]),
                              float(turned[2]))
        mesh = getattr(node, "mesh", None)
        if mesh is None or id(mesh) in exclude:
            continue
        pts = np.asarray(mesh.geometry.positions,
                         dtype=np.float64).reshape(-1, 3) @ matrix.T
        flat = pts.reshape(-1).astype(np.float32)
        mesh.geometry.positions = flat
        mesh.rest_positions = flat.copy()
        mesh.needs_update = True
        moved += 1
    return moved


def apply(root: Any, gender: float, exclude: set[int] | None = None) -> int:
    """Open the elbow and the knee by the female difference.  Returns meshes moved."""
    g = float(max(0.0, min(1.0, gender)))
    if g <= 0.0:
        return 0
    skip = exclude or set()
    pivots = {n.name: n for n in _subtree(root) if _is_pivot(n) and n.name}
    moved = 0
    for joint, delta in SEX_DELTA:
        for side, sign in SIDE_SIGN.items():
            node = pivots.get(f"{joint}_{side}_pivot")
            if node is None:
                continue
            moved += rotate_below(node, rotation_about_y(sign * delta * g), skip)
    if moved:
        logger.info("Carrying angle and knee valgus opened by %.1f degrees at "
                    "gender %.2f: %d meshes", SEX_DELTA[0][1] * g, g, moved)
    return moved
