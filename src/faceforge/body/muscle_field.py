"""How far the flesh of each body part is, sampled anywhere in the body.

Nearest-bone binding fails wherever the bone is not under the skin.  With the
arms hanging in the rest pose the forearm is about 3 units from the flank and
the lumbar spine about 19, so the flank binds to the arm and is drawn out
along it when the arm lifts.  Muscle does not have that problem: it fills the
soft tissue, so the flesh nearest a patch of skin is the flesh that skin sits
on.  Measured on the spikes that survived every binding fix, the nearest
muscle is a trunk muscle for 78% of them, while their bones say arm.

This holds a distance field per body part, sampled from the muscle meshes
themselves.  The skinning adds it to the bone distance, so a chain whose
flesh is far cannot win on a bone that happens to be near.

The muscle layers are loaded on demand and the skin binding does not wait for
them, so the field is built once by ``tools/build_muscle_field.py`` and read
back from disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Body parts the field distinguishes.  A joint or a muscle is assigned to one
#: of these by :func:`group_of_joint` / the builder's region mapping.
GROUPS = ("trunk", "arm_R", "arm_L", "hand_R", "hand_L",
          "leg_R", "leg_L", "foot_R", "foot_L")

_ARM = ("shoulder", "elbow", "wrist", "clavicle", "scapula")
_LEG = ("hip", "knee", "ankle")


def group_of_joint(name: str) -> str:
    """The body part a skinning joint belongs to, from its name."""
    n = name.lower()
    side = "R" if n.endswith("_r") or "_r_" in n else ("L" if n.endswith("_l")
                                                       or "_l_" in n else None)
    if n.startswith("finger"):
        return f"hand_{side or 'R'}"
    if n.startswith("toe"):
        return f"foot_{side or 'R'}"
    if any(n.startswith(k) for k in _ARM):
        return f"arm_{side or 'R'}"
    if any(n.startswith(k) for k in _LEG):
        return f"leg_{side or 'R'}"
    return "trunk"


class MuscleChainField:
    """Distance to the nearest muscle of each body part, by KD-tree.

    Built from a sample of muscle vertices rather than a voxel grid: the
    sample is what the distance is actually measured from, so there is no
    resolution to choose and no interpolation to smooth.
    """

    def __init__(self, points: dict[str, np.ndarray]) -> None:
        from scipy.spatial import cKDTree

        self.points = {k: np.asarray(v, dtype=np.float64)
                       for k, v in points.items() if len(v)}
        self._trees = {k: cKDTree(v) for k, v in self.points.items()}

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(self._trees))

    @property
    def digest(self) -> str:
        """Short hash of the field's contents.

        The binding cache keys on every public scalar the skinning carries, so
        exposing this as one is what stops a rebuilt field serving a binding
        solved against the old one.  Counts and bounds rather than every
        point: the field is a sample, and two samples that agree on those
        agree on what the binding does with them.
        """
        import hashlib

        h = hashlib.blake2b(digest_size=8)
        for group in sorted(self.points):
            pts = self.points[group]
            h.update(group.encode())
            h.update(f"{len(pts)}".encode())
            if len(pts):
                h.update(np.round(pts.min(axis=0), 3).tobytes())
                h.update(np.round(pts.max(axis=0), 3).tobytes())
                h.update(np.round(pts.mean(axis=0), 3).tobytes())
        return h.hexdigest()

    def distance(self, query: np.ndarray, group: str) -> np.ndarray:
        """Distance from every query point to the nearest muscle of *group*.

        A group with no muscles returns ``inf``, which adds nothing the
        caller can rank by -- the caller must treat it as "no information".
        """
        tree = self._trees.get(group)
        if tree is None:
            return np.full(len(query), np.inf)
        return tree.query(np.asarray(query, dtype=np.float64), k=1)[0]

    def nearest_point(self, query: np.ndarray, group: str):
        """The nearest muscle point of *group* to each query point, or None.

        Used to get an INWARD direction without trusting the mesh's triangle
        winding, which on this asset is inconsistent: the flesh a patch of
        skin sits on is under it, so the direction to the nearest muscle
        point is the direction into the body.
        """
        tree = self._trees.get(group)
        if tree is None:
            return None
        idx = tree.query(np.asarray(query, dtype=np.float64), k=1)[1]
        return self.points[group][idx]

    def save(self, path: Path | str) -> None:
        np.savez_compressed(
            str(path), groups=np.array(sorted(self.points)),
            **{f"p_{k}": v.astype(np.float32) for k, v in self.points.items()})

    @classmethod
    def load(cls, path: Path | str) -> "MuscleChainField | None":
        """Read a saved field, or ``None`` if it is absent or unreadable."""
        try:
            with np.load(str(path)) as z:
                groups = [str(g) for g in z["groups"]]
                return cls({g: z[f"p_{g}"] for g in groups})
        except (OSError, ValueError, KeyError):
            return None
