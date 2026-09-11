"""Scale the skeleton as an articulated hierarchy, not as loose bones.

The previous approach scaled every bone about its own centroid.  Two things
follow from that, and both were visible:

* **Joints come apart.**  A femur scaled 0.92 about its own centroid pulls its
  distal end 4 % of the bone's length toward the middle; the tibia does the
  same at its proximal end.  The knee therefore opens by 4 % of *both* bones.
  Measured on the shipped configuration at gender = 1: knee 0.07 -> 3.55,
  elbow 0.59 -> 3.89, humero-ulnar 0.05 -> 0.84, patello-femoral 0.36 -> 1.51.
* **The skeleton does not change proportion.**  Scaling about the centroid
  leaves the centroid where it was, so no bone ever moves.  The median bone
  centroid displacement across the whole male -> female change was 0.00.
  A shorter clavicle did not bring the shoulder in; a wider pelvis did not
  carry the hip joints out.

Both are fixed by treating the skeleton as what it is: a tree of segments
joined at joints.  Bones are already reparented under joint pivots by
:mod:`faceforge.body.joint_pivots`, which leaves each bone's vertices
expressed *relative to the joint it hangs from*.  So:

1. A bone's geometry is scaled about its own proximal joint (the local
   origin), which keeps the proximal articulation exactly where it was.
2. A child pivot's local offset -- the vector from one joint to the next, i.e.
   the segment -- is scaled by the same factor, so the distal joint lands on
   the end of the scaled bone and the next bone follows it.

Joints whose position is absolute rather than an offset (the glenohumeral
joint, the hip joints, the rib pivots, the cervical pivots) belong to a
*region* and are transformed about that region's anatomical anchor: the
shoulder rides on the clavicle from the sternoclavicular joint, the hip
joints ride on the pelvis, the ribs ride on the thoracic spine.

Nothing here rebinds the skinning.  Moving the joints is exactly the kind of
change the delta-matrix skinning already expresses, so muscles and skin
follow the new skeleton the same way they follow a pose.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.body.skeleton_field import (
    control_points, displacement_warp, sampled_warp,
)
from faceforge.body.skeleton_joints import close_articulations

logger = logging.getLogger(__name__)

Vec3 = NDArray[np.float64]


def _is_pivot(node: Any) -> bool:
    """A node that represents a joint rather than a bone or a grouping node."""
    name = getattr(node, "name", "") or ""
    return "pivot" in name.lower()


@dataclass
class _Backup:
    """The male skeleton: every pivot position and every bone's rest geometry.

    Kept so that any gender value is computed from the unscaled original
    rather than from the last result, which would compound rounding and make
    the slider path-dependent.
    """

    pivot_positions: dict[int, Vec3] = field(default_factory=dict)
    bone_positions: dict[int, NDArray[np.float32]] = field(default_factory=dict)
    joint_positions: dict[str, Vec3] = field(default_factory=dict)


#: Regions whose joints are positioned in body coordinates rather than as an
#: offset from a parent joint.  ``anchor`` names the anatomical point the
#: region pivots about; ``scale_key`` is the bone whose change carries it.
#: Ordered: an anchor may depend on a region resolved before it.
_REGIONS: tuple[tuple[str, str, str], ...] = (
    # The sternoclavicular joint sits on the manubrium, so it follows the sternum.
    (r"clavicle_([RL])_pivot", "sternal_notch", "sternum"),
    # The glenohumeral joint and the scapula ride on the lateral end of the
    # clavicle: a shorter clavicle brings the whole shoulder medially, which
    # is what makes biacromial breadth the strongest dimorphic measure there is.
    (r"shoulder_([RL])_pivot", "sternoclavicular", "clavicle"),
    (r"scapula_([RL])_pivot", "sternoclavicular", "clavicle"),
    # The acetabulum is part of the hip bone, so the hip joint moves with the pelvis.
    (r"hip_([RL])_pivot", "pelvic_centre", "pelvis"),
    # Ribs articulate with the thoracic vertebrae: they scale about the spinal axis.
    (r".*_breath_pivot", "thoracic_axis", "rib"),
    # The cervical column stacks on T1.
    (r"vertebrae_pivot_\d+", "t1", "vertebra"),
    # The sacrum is part of the pelvis; the lumbar column stacks on it.
    (r"lumbar_spine_pivot_\d+", "pelvic_centre", "sacrum"),
    (r"thoracic_spine_pivot_\d+", "lumbosacral", "vertebra"),
    # The jaw hangs from the temporal bone, the eyes sit in the orbits.
    (r"jawPivot", "cranial_centre", "temporal_bone"),
    (r"eyeball_pivot_\w+", "cranial_centre", "cranium_group"),
)

#: Bones that are not under a joint pivot: the anchor their region scales about.
_GROUP_ANCHORS: tuple[tuple[str, str], ...] = (
    (r"^(Right|Left) Hip Bone$", "pelvic_centre"),
    (r"Sternum|Manubrium|Xiphoid", "sternal_notch"),
    (r"^(cranium|upper_teeth|face)$", "cranial_centre"),
)


class SkeletonMorph:
    """Applies a sex morph to the skeleton as an articulated hierarchy.

    Parameters
    ----------
    scaler :
        A :class:`~faceforge.body.bone_scaling.BoneScaler`; only its name to
        scale-factor lookup is used.
    """

    def __init__(self, scaler: Any) -> None:
        self._scaler = scaler
        self._backup = _Backup()
        self._patch_cache: dict = {}
        self._exclude: set[int] = set()
        self._captured = False
        self._anchors: dict[str, Vec3] = {}
        self._segment_cache: dict[int, Optional[tuple[float, float, float]]] = {}

    # -- capture -------------------------------------------------------------

    def capture(self, root: Any, joint_positions: dict[str, Any] | None = None) -> None:
        """Snapshot the unscaled skeleton.  Idempotent; the first call wins."""
        if self._captured:
            return
        b = self._backup
        for node in _walk(root):
            if _is_pivot(node):
                b.pivot_positions[id(node)] = np.asarray(node.position, dtype=np.float64).copy()
            mesh = getattr(node, "mesh", None)
            if mesh is not None:
                src = mesh.rest_positions if mesh.rest_positions is not None else mesh.geometry.positions
                b.bone_positions[id(mesh)] = np.asarray(src, dtype=np.float32).copy()
        if joint_positions:
            b.joint_positions = {k: np.asarray(v, dtype=np.float64).copy()
                                 for k, v in joint_positions.items()}
        self._captured = True
        logger.info("Skeleton morph captured: %d pivots, %d meshes",
                    len(b.pivot_positions), len(b.bone_positions))

    @property
    def captured(self) -> bool:
        return self._captured

    # -- anchors -------------------------------------------------------------

    def _compute_anchors(self, root: Any) -> dict[str, Vec3]:
        """Anatomical points the regions scale about, from the unscaled skeleton."""
        meshes = {}
        pivots = {}
        for node in _walk(root):
            if node.name:
                if getattr(node, "mesh", None) is not None:
                    meshes[node.name] = node
                if _is_pivot(node):
                    pivots[node.name] = node

        def centroid(*names: str) -> Optional[Vec3]:
            pts = []
            for n in names:
                node = meshes.get(n)
                if node is None:
                    continue
                rest = self._backup.bone_positions.get(id(node.mesh))
                if rest is None:
                    continue
                pts.append(np.asarray(rest, dtype=np.float64).reshape(-1, 3).mean(axis=0)
                           + self._node_offset(node))
            return None if not pts else np.mean(pts, axis=0)

        a: dict[str, Vec3] = {}
        pelvic = centroid("Right Hip Bone", "Left Hip Bone")
        if pelvic is None:
            pelvic = np.zeros(3)
        # The pelvis flares about its own midline, so the anchor sits on it.
        a["pelvic_centre"] = np.array([0.0, pelvic[1], pelvic[2]])

        sternum = centroid("Manubrium of Sternum") or centroid("Body of Sternum")
        if sternum is None:
            sternum = np.array([0.0, -10.0, -20.0])
        a["sternal_notch"] = np.array([0.0, float(sternum[1]), float(sternum[2])])

        thorax = centroid(*[f"{s} {n} Rib" for s in ("Right", "Left")
                            for n in ("1st", "6th", "12th")])
        if thorax is None:
            thorax = a["sternal_notch"]
        a["thoracic_axis"] = np.array([0.0, float(thorax[1]), float(thorax[2])])

        cranial = centroid("cranium")
        a["cranial_centre"] = cranial if cranial is not None else np.zeros(3)

        for side in ("R", "L"):
            node = pivots.get(f"clavicle_{side}_pivot")
            if node is not None:
                a[f"sternoclavicular_{side}"] = (
                    self._backup.pivot_positions[id(node)] + self._node_offset(node))

        t1 = meshes.get("T1")
        if t1 is not None:
            rest = self._backup.bone_positions.get(id(t1.mesh))
            if rest is not None:
                a["t1"] = (np.asarray(rest, dtype=np.float64).reshape(-1, 3).mean(axis=0)
                           + self._node_offset(t1))
        a.setdefault("t1", np.array([0.0, 0.0, -10.0]))

        sacrum = centroid("Sacrum")
        a["lumbosacral"] = sacrum if sacrum is not None else a["pelvic_centre"]
        return a

    @staticmethod
    def _node_offset(node: Any) -> Vec3:
        """A node's rest position in body coordinates, from its ancestors' positions.

        Pivot rotations are identity at rest, so summing the local positions is
        the world position; this avoids depending on the scene having been
        updated, and on any pose that happens to be applied.
        """
        out = np.zeros(3)
        n = node
        while n is not None and getattr(n, "name", "") != "bodyRoot":
            out = out + np.asarray(n.position, dtype=np.float64)
            n = n.parent
        return out

    # -- scales --------------------------------------------------------------

    def _scale_of(self, name: str, gender: float) -> Optional[tuple[float, float, float]]:
        return self._scaler.compute_scale(name, gender)

    def _segment_scale(self, pivot: Any, gender: float) -> Optional[tuple[float, float, float]]:  # noqa: D401
        """The scale of the segment hanging off ``pivot``: its largest keyed bone.

        Every bone under a joint takes the same factor, so a joint cannot come
        apart from the bones that meet in it: the patella follows the femur,
        the carpals follow each other, the two forearm bones stay level.
        """
        best = None
        best_verts = -1
        for child in pivot.children:
            mesh = getattr(child, "mesh", None)
            if mesh is None or not child.name or id(mesh) in self._exclude:
                continue
            s = self._scale_of(child.name, gender)
            if s is None:
                continue
            n = int(getattr(mesh.geometry, "vertex_count", 0) or 0)
            if n > best_verts:
                best, best_verts = s, n
        return best

    # -- apply ---------------------------------------------------------------

    def apply(self, root: Any, gender: float,
              joint_positions: dict[str, Any] | None = None,
              exclude: set[int] | None = None) -> dict[str, int]:
        """Scale the whole skeleton to ``gender`` (0 = male, 1 = female).

        ``exclude`` is a set of ``id(mesh)`` this must never touch -- the
        caller passes everything the skinning owns, because soft tissue is
        deformed by the morph rather than scaled by it.  Name matching alone
        is not enough of a guard: "Tibialis", "Fibularis", "Subscapularis" and
        "Iliocostalis" all read as bones to a substring test.

        Always recomputed from the captured original, so any value can be set
        at any time and setting 0 restores the skeleton exactly.
        """
        self._exclude = exclude or set()
        self.capture(root, joint_positions)
        gender = float(max(0.0, min(1.0, gender)))
        self._anchors = self._compute_anchors(root)
        stats = {"bones": 0, "pivots": 0}

        self._walk_apply(root, gender, parent_segment=None, stats=stats)
        stats["closed"] = close_articulations(
            root, self._backup.bone_positions, self._node_offset, self._patch_cache)

        if joint_positions is not None:
            self._update_joint_positions(root, joint_positions)
        logger.info("Skeleton morph at gender %.2f: %d bones, %d joints moved",
                    gender, stats["bones"], stats["pivots"])
        return stats

    def _walk_apply(self, node: Any, gender: float,
                    parent_segment: Optional[tuple[float, float, float]],
                    stats: dict[str, int]) -> None:
        segment = self._segment_scale(node, gender) if _is_pivot(node) else None
        if segment is None and _is_pivot(node):
            segment = parent_segment

        for child in node.children:
            if _is_pivot(child):
                rest = self._backup.pivot_positions.get(id(child))
                if rest is not None:
                    moved = self._new_pivot_position(child, rest, gender, parent_segment=segment)
                    if moved is not None:
                        child.set_position(float(moved[0]), float(moved[1]), float(moved[2]))
                        stats["pivots"] += 1
            mesh = getattr(child, "mesh", None)
            if mesh is not None and child.name:
                if self._scale_bone(child, mesh, gender, segment):
                    stats["bones"] += 1
            self._walk_apply(child, gender, segment, stats)

    def _new_pivot_position(self, pivot: Any, rest: Vec3, gender: float,
                            parent_segment: Optional[tuple[float, float, float]]) -> Optional[Vec3]:
        """Where a joint goes: along its own segment, or about its region's anchor.

        Which of the two applies is decided by the parent, not by the name: a
        joint whose parent is another joint holds an *offset* -- the segment
        between them -- and must scale with that segment, or the chain comes
        apart.  Treating the chained thoracic pivots as region roots (they
        share a name pattern with the root) opened T12/L1 by 7.3 units.
        """
        name = pivot.name or ""
        chained = pivot.parent is not None and _is_pivot(pivot.parent)
        if chained:
            if parent_segment is not None:
                return rest * np.asarray(parent_segment, dtype=np.float64)
            return None
        for pattern, anchor_key, scale_key in _REGIONS:
            m = re.fullmatch(pattern, name)
            if m is None:
                continue
            side = m.group(1) if m.groups() else None
            anchor = self._anchors.get(f"{anchor_key}_{side}" if side else anchor_key)
            if anchor is None:
                anchor = self._anchors.get(anchor_key)
            scale = self._scale_of_key(scale_key, gender)
            if anchor is None or scale is None:
                return None
            # The anchor is in body coordinates; the pivot's position is
            # relative to its parent, so bring the anchor into that frame.
            parent_offset = self._node_offset(pivot.parent) if pivot.parent is not None else np.zeros(3)
            local_anchor = np.asarray(anchor, dtype=np.float64) - parent_offset
            return local_anchor + (rest - local_anchor) * np.asarray(scale, dtype=np.float64)
        return None

    def _scale_of_key(self, key: str, gender: float) -> Optional[tuple[float, float, float]]:
        if key == "cranium_group":
            key = "parietal_bone"
        table = getattr(self._scaler, "_bone_scales", {})
        female = table.get(key)
        if female is None:
            return None
        g = float(max(0.0, min(1.0, gender)))
        return tuple(1.0 + (float(f) - 1.0) * g for f in female)

    def _scale_bone(self, node: Any, mesh: Any, gender: float,
                    segment: Optional[tuple[float, float, float]]) -> bool:
        """Scale a bone about the joint it hangs from, or about its region's anchor."""
        if id(mesh) in self._exclude:
            return False
        rest = self._backup.bone_positions.get(id(mesh))
        if rest is None:
            return False
        own = self._scale_of(node.name, gender)
        parent_is_pivot = node.parent is not None and _is_pivot(node.parent)

        if parent_is_pivot:
            # Local origin IS the joint: every bone in the segment takes the
            # segment's factor so the joint cannot open.
            scale = segment if segment is not None else own
            anchor = np.zeros(3)
        else:
            scale = own
            anchor = self._group_anchor(node)
            if anchor is not None:
                anchor = anchor - self._node_offset(node)
        if scale is None:
            return False
        if anchor is None:
            anchor = np.asarray(rest, dtype=np.float64).reshape(-1, 3).mean(axis=0)

        pts = np.asarray(rest, dtype=np.float64).reshape(-1, 3)
        out = anchor + (pts - anchor) * np.asarray(scale, dtype=np.float64)
        flat = out.reshape(-1).astype(np.float32)
        mesh.geometry.positions = flat
        mesh.rest_positions = flat.copy()
        mesh.needs_update = True
        return True

    def _group_anchor(self, node: Any) -> Optional[Vec3]:
        for pattern, key in _GROUP_ANCHORS:
            if re.search(pattern, node.name or ""):
                return self._anchors.get(key)
        return None

    # -- joint positions -----------------------------------------------------

    def _update_joint_positions(self, root: Any, joint_positions: dict[str, Any]) -> None:
        """Refresh ``JointSetup.joint_positions`` so rest-pose consumers agree.

        The ground lock and the equipment rig measure against these; leaving
        them at the male skeleton's values would put a morphed body's feet
        through the floor.
        """
        by_name = {n.name: n for n in _walk(root) if _is_pivot(n) and n.name}
        for key in list(joint_positions):
            node = by_name.get(f"{key}_pivot")
            if node is None:
                continue
            joint_positions[key] = self._node_offset(node)

    # -- the change as a smooth field ----------------------------------------

    def control_points(self, root: Any) -> tuple[Vec3, Vec3]:
        """Where the skeleton was and where it went; see :mod:`skeleton_field`."""
        return control_points(root, self._backup.pivot_positions,
                              self._backup.bone_positions,
                              lambda name: self._scale_of(name, 1.0),
                              self._node_offset, _is_pivot)

    def displacement_field(self, root: Any, sigma: float | None = None,
                           sampled: bool = True):
        """A smooth warp taking the male body's soft tissue to the morphed one.

        Sampled on a lattice by default: the field is smooth, and evaluating
        the spline per vertex over six million of them is the single most
        expensive part of a morph.
        """
        pts, disp = self.control_points(root)
        warp = displacement_warp(pts, disp, sigma)
        if not sampled or len(pts) == 0:
            return warp
        return sampled_warp(warp, pts)

    # -- reset ---------------------------------------------------------------

    def reset(self, root: Any, joint_positions: dict[str, Any] | None = None) -> None:
        """Put the skeleton back exactly as it was captured."""
        if not self._captured:
            return
        self.apply(root, 0.0, joint_positions)


def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)
