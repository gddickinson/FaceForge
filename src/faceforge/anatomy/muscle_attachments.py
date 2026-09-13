"""Bone attachment constraints for body muscles.

Body muscles have anatomical origin and insertion bones.  With authored
footprints (``assets/config/muscle_footprints.json``) a muscle is placed from
the rigid images of its two attachments by a harmonic fibre field
(:mod:`faceforge.anatomy.fibre_field`); without them the skinning solver's
own assignment stands.

Also provides per-muscle stretch measurement (Layer 3) and fascia region
constraints (Layer 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from numpy.typing import NDArray

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.anatomy.fibre_field import FibreField, cached_fibre_field, trim_footprints
from faceforge.body.soft_tissue import SkinBinding

logger = logging.getLogger(__name__)


# Layer 3: maximum physiological stretch ratio before clamping
MAX_STRETCH = 1.35


@dataclass
class MuscleAttachmentData:
    """Per-muscle attachment data computed at registration time."""
    muscle_name: str
    origin_bones: list[str]
    insertion_bones: list[str]
    # Per-vertex attachment fraction: 0 = insertion end, 1 = origin end
    attachment_frac: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    # Masks for origin/insertion zones
    origin_mask: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=bool))
    insertion_mask: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=bool))
    # Rest-pose length (centroid-to-centroid of top/bottom 15%)
    rest_length: float = 1.0
    # Per-frame stretch ratio (updated each frame)
    current_stretch: float = 1.0
    # Fascia regions for body-end pinning
    fascia_regions: list[str] = field(default_factory=list)

    # Thresholds for origin/insertion zones
    origin_frac_threshold: float = 0.8
    insertion_frac_threshold: float = 0.2

    #: True once authored footprints replaced the Y-extent masks.  The rest
    #: and current lengths are then measured between the footprint centroids
    #: rather than along the mesh's Y extent -- which in this Z-up body frame
    #: is the anterior-posterior axis, meaningless as a muscle's length for
    #: anything but a prevertebral strap.
    footprint_masks: bool = False
    #: Skin-joint indices the footprints resolved to (origin, insertion).
    origin_joint: int | None = None
    insertion_joint: int | None = None
    #: Harmonic interpolation between the two footprints; None until they
    #: resolve.  Placed every frame by :meth:`apply_bone_pinning`.
    fibre_field: "FibreField | None" = None

    # Per-muscle overrides for the module globals; None means "use the global".
    #
    # A single global cannot be right for every muscle: physiological
    # excursion depends on optimal fibre length relative to moment arm and on
    # how many joints the muscle spans, and it varies several-fold across the
    # body.  Measured evidence from this model, too -- raising PIN_STRENGTH
    # from 0.6 to 1.0 moved deltoid acromial +5% but pectoralis major sternal
    # +67%, and lowering MAX_STRETCH improved deltoid acromial while degrading
    # deltoid clavicular, i.e. opposite directions from one change.
    #
    # Left as None unless a value has a source. Measured excursion across the
    # six authored poses (tools-side, muscle_excursion.json) shows only 1 of
    # 112 muscles reaching the 1.35 global, so populating these from guesses
    # would start clamping muscles that are currently unclamped.
    max_stretch: float | None = None
    pin_strength: float | None = None


# Pinning strength (matching neck muscle pattern)
PIN_STRENGTH = 0.6


class MuscleAttachmentSystem:
    """Attachment placement and stretch measurement for body muscles.

    For each registered muscle:
    1. Computes attachment fraction from mesh Y-extent (like neck spine_fracs)
    2. Identifies origin-end and insertion-end vertex zones, replaced by the
       authored footprints when :meth:`reassign_by_footprints` runs
    3. Per-frame: places a footprinted muscle from its attachments' rigid
       images through its fibre field
    4. Measures the stretch ratio between the attachments
    """

    def __init__(self, bone_registry: BoneAnchorRegistry) -> None:
        self._bones = bone_registry
        self._attachments: dict[int, MuscleAttachmentData] = {}  # binding id → data

    def register_muscle(
        self,
        binding: SkinBinding,
        origin_bones: list[str],
        insertion_bones: list[str],
        fascia_regions: list[str] | None = None,
        max_stretch: float | None = None,
        pin_strength: float | None = None,
    ) -> None:
        """Register a muscle binding for bone-pinning constraints.

        Parameters
        ----------
        binding : SkinBinding
            The soft-tissue binding for this muscle.
        origin_bones : list[str]
            Bone names where the muscle originates (proximal end).
        insertion_bones : list[str]
            Bone names where the muscle inserts (distal end).
        fascia_regions : list[str], optional
            Fascia region names for additional body-end constraint.
        """
        mesh = binding.mesh
        if mesh.rest_positions is None:
            return

        rest_pos = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        V = len(rest_pos)

        # Compute attachment fraction from Y-extent (top=origin, bottom=insertion)
        y_vals = rest_pos[:, 1]
        y_min = y_vals.min()
        y_max = y_vals.max()
        y_range = y_max - y_min
        if y_range < 1e-6:
            frac = np.full(V, 0.5)
        else:
            frac = (y_vals - y_min) / y_range  # 0 at bottom, 1 at top

        data = MuscleAttachmentData(
            muscle_name=binding.muscle_name or "unknown",
            origin_bones=origin_bones,
            insertion_bones=insertion_bones,
            attachment_frac=frac,
            origin_mask=frac > 0.8,
            insertion_mask=frac < 0.2,
            fascia_regions=fascia_regions or [],
            max_stretch=max_stretch,
            pin_strength=pin_strength,
        )

        # Compute rest-pose length (centroid of top 15% to centroid of bottom 15%)
        n15 = max(1, V // 7)  # ~15%
        top_idx = np.argpartition(y_vals, -n15)[-n15:]
        bot_idx = np.argpartition(y_vals, n15)[:n15]
        top_centroid = rest_pos[top_idx].mean(axis=0)
        bot_centroid = rest_pos[bot_idx].mean(axis=0)
        data.rest_length = max(1e-3, float(np.linalg.norm(top_centroid - bot_centroid)))

        self._attachments[id(binding)] = data
        logger.debug(
            "Registered muscle attachment: %s (V=%d, origin=%s, insertion=%s, "
            "rest_len=%.1f, origin_verts=%d, insertion_verts=%d)",
            data.muscle_name, V, origin_bones, insertion_bones,
            data.rest_length, data.origin_mask.sum(), data.insertion_mask.sum(),
        )

    def origin_zone_mask(self, binding: SkinBinding) -> "NDArray[np.bool_] | None":
        """Boolean mask of the muscle's origin-end vertices, or None.

        Exposed for the physics pass, which holds the origin zone fixed while
        relaxing edge lengths: bone pinning has already moved those vertices
        toward their bone, and the relaxation must not undo it.
        """
        data = self._attachments.get(id(binding))
        if data is None or data.origin_mask.size == 0:
            return None
        return data.origin_mask

    def reassign_by_footprints(self, binding: SkinBinding, joints: list,
                               footprints: dict) -> int:
        """Assign primary joints from AUTHORED attachment footprints.

        Footprints have to be authored; they cannot be inferred. Four measured
        attempts to infer them all failed the same way -- an along-muscle axis
        from attachment_frac (a mesh Y-extent) sent serratus anterior 0.04 ->
        7.10, and nearest-joint-to-centroid, centroid-distance and
        bone-surface-distance rules each sent both deltoid divisions to 100%
        humerus against the 71.8%/48.8% they were meant to reduce. The reason
        is structural: these muscles WRAP the humerus, so it is the nearest
        bone to most of their mass by every distance measure. Contact is not
        attachment.

        Registering published attachment points was tried as well: the
        licence-compatible source ships no bone geometry to register against,
        and a three-landmark similarity fit (RMS 2.70) placed 18 of 19 points
        4-36 units off the muscle surface.

        What survives from that work is the INTERPOLATION -- geodesic distance
        between the two footprints along the muscle's own edges, which is
        immune to wrapping. This consumes authored footprints and interpolates
        exactly that way.

        There is deliberately NO fallback: a muscle without footprints keeps
        the solver's own assignment. Substituting a proxy is what produced
        every regression above.
        """
        data = self._attachments.get(id(binding))
        name = (data.muscle_name if data else None) or ""
        fp = footprints.get(name)
        if data is None or fp is None or not joints:
            return 0
        rest = binding.mesh.rest_positions
        edges = binding.edge_pairs
        if rest is None or edges is None or len(edges) == 0:
            return 0
        rest = np.asarray(rest, dtype=np.float64).reshape(-1, 3)
        n = min(len(rest), len(binding.joint_indices))
        o_idx = np.asarray(fp.get("origin_indices", []), dtype=np.int64)
        i_idx = np.asarray(fp.get("insertion_indices", []), dtype=np.int64)
        o_idx = o_idx[o_idx < n]
        i_idx = i_idx[i_idx < n]
        shared = np.intersect1d(o_idx, i_idx)   # near both bones: neither attachment
        if len(shared):
            o_idx, i_idx = np.setdiff1d(o_idx, shared), np.setdiff1d(i_idx, shared)
        if not len(o_idx) or not len(i_idx):
            return 0

        joint_of_node = {id(j.node): k for k, j in enumerate(joints)}
        bone_nodes = getattr(self._bones, "_bone_nodes", {})

        def resolve_end(bones):
            for bn in bones:
                node = bone_nodes.get(bn)
                hops = 0
                while node is not None and hops < 12:
                    if id(node) in joint_of_node:
                        return joint_of_node[id(node)]
                    node = getattr(node, "parent", None)
                    hops += 1
            return None

        j_o = resolve_end(data.origin_bones)
        j_i = resolve_end(data.insertion_bones)
        if j_o is None or j_i is None or j_o == j_i:
            return 0

        e = edges[(edges[:, 0] < n) & (edges[:, 1] < n)]
        if len(e) == 0:
            return 0
        w = np.linalg.norm(rest[e[:, 0]] - rest[e[:, 1]], axis=1)
        g = csr_matrix((np.concatenate([w, w]),
                        (np.concatenate([e[:, 0], e[:, 1]]),
                         np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n))
        g_o = dijkstra(g, indices=o_idx, min_only=True)
        g_i = dijkstra(g, indices=i_idx, min_only=True)
        # Footprints seeded by bone proximity touch wherever a muscle wraps
        # its own joint; the fibre field tears across such a seam, so a
        # geodesic gap is kept between them (fibre_field.trim_footprints).
        o_idx, i_idx = trim_footprints(g_o, g_i, o_idx, i_idx)
        if not len(o_idx) or not len(i_idx):
            logger.info("Footprints for %s touch everywhere; solver assignment kept", name)
            return 0
        g_o = dijkstra(g, indices=o_idx, min_only=True)
        g_i = dijkstra(g, indices=i_idx, min_only=True)
        both = np.isfinite(g_o) & np.isfinite(g_i)
        if not both.any():
            return 0
        zone = np.full(n, 0.5)
        zone[both] = g_o[both] / np.maximum(g_o[both] + g_i[both], 1e-9)

        ji = np.asarray(binding.joint_indices)
        before = ji[:n].copy()
        to_ins = zone > 0.5
        ji[:n] = np.where(to_ins, j_i, j_o)
        binding.secondary_indices[:n] = np.where(to_ins, j_o, j_i)
        # Primary weight: 1.0 at a vertex's OWN footprint, falling to 0.5 at
        # the muscle's midline -- NOT to 0.0.
        #
        # The first version wrote 2*(grade-0.5), which is 0.0 at the midline,
        # and zero primary weight means the vertex is driven ENTIRELY by its
        # secondary, i.e. by the opposite attachment. Measured on pectoralis
        # major sternal in the reaching pose, that put 3,305 vertices nominally
        # assigned to rib_40 -- a STATIC joint -- 17.9 units higher than rest
        # with a residual of 18.71 against their own joint's rigid image: none
        # of their motion came from the joint they were assigned to. The
        # user saw it as geometry spiking up past the clavicle.
        #
        # `grade` is already max(zone, 1-zone) in [0.5, 1], so it IS the
        # correct primary weight and needs no rescaling.
        grade = np.where(to_ins, zone, 1.0 - zone)
        binding.weights[:n] = np.clip(
            grade, 0.5, 1.0).astype(binding.weights.dtype)
        # Record the authored sets as this muscle's attachment masks, so
        # every consumer (pinning zones, the balloon solve's anchors) uses the
        # real footprints rather than a Y-extent threshold.
        om = np.zeros(len(data.attachment_frac), dtype=bool)
        im = np.zeros(len(data.attachment_frac), dtype=bool)
        om[o_idx[o_idx < len(om)]] = True
        im[i_idx[i_idx < len(im)]] = True
        data.origin_mask = om
        data.insertion_mask = im
        # The muscle's length is now origin footprint -> insertion footprint.
        rest_full = np.asarray(binding.mesh.rest_positions, dtype=np.float64).reshape(-1, 3)
        no, ni = min(len(om), len(rest_full)), min(len(im), len(rest_full))
        if om[:no].any() and im[:ni].any():
            data.rest_length = max(1e-3, float(np.linalg.norm(
                rest_full[:no][om[:no]].mean(axis=0) - rest_full[:ni][im[:ni]].mean(axis=0))))
            data.footprint_masks = True

        # The belly interpolates harmonically between the two footprints; the
        # eight bind-time solves are cached on disk beside the binding solve.
        try:
            data.fibre_field = cached_fibre_field(rest[:n], e, o_idx, i_idx)
        except Exception as exc:  # noqa: BLE001 -- a failed solve must not abort loading
            logger.warning("Fibre field for %s failed: %s", name, exc)
            data.fibre_field = None

        binding.footprint_graded = True
        data.origin_joint, data.insertion_joint = int(j_o), int(j_i)
        changed = int((before != ji[:n]).sum())
        logger.info("Footprint reassignment for %s: %d/%d vertices "
                    "(origin joint %d, insertion joint %d, %d unreached)",
                    name, changed, n, j_o, j_i, int((~both).sum()))
        return changed

    def anchor_mask(self, binding: SkinBinding) -> "NDArray[np.bool_] | None":
        """Vertices that must be held during a constrained solve, or None.

        Both attachments, not just the origin: a balloon is positioned by its
        ends, and holding only one lets the other wander -- which is the
        "pulled away from the attachment" failure. Authored footprints are used
        when present because they are the real attachment sets; otherwise the
        threshold zones are the best available.
        """
        data = self._attachments.get(id(binding))
        if data is None:
            return None
        n = len(data.attachment_frac)
        if n == 0:
            return None
        # origin_mask / insertion_mask are the authored footprints when
        # reassign_by_footprints has run, and the threshold zones otherwise --
        # it overwrites them, so this needs no separate lookup.
        om = np.asarray(data.origin_mask, dtype=bool)
        im = np.asarray(data.insertion_mask, dtype=bool)
        if len(om) >= n and len(im) >= n:
            return om[:n] | im[:n]
        return ((data.attachment_frac > data.origin_frac_threshold)
                | (data.attachment_frac < data.insertion_frac_threshold))

    def origin_bone_top(self, binding: SkinBinding) -> float | None:
        """Superior extent of this muscle's origin bones in the CURRENT frame.

        Current rather than rest, so the envelope travels with the body: a
        whole-body translation must not clamp anything.
        """
        data = self._attachments.get(id(binding))
        if data is None or not data.origin_bones:
            return None
        tops = []
        store = getattr(self._bones, "_bone_nodes", {})
        for bn in data.origin_bones:
            node = store.get(bn)
            geo = getattr(getattr(node, "mesh", None), "geometry", None)
            if geo is None or geo.positions is None:
                continue
            v = np.asarray(geo.positions, dtype=np.float64).reshape(-1, 3)
            v = v[:getattr(geo, "vertex_count", len(v))]
            node.update_world_matrix()
            m = np.asarray(node.world_matrix, dtype=np.float64)
            tops.append(float(((v @ m[:3, :3].T) + m[:3, 3])[:, 2].max()))
        return max(tops) if tops else None

    def set_frame_cancel(self, cancel) -> None:
        """Forward the scene wrapper's inverse to the bone registry (per frame)."""
        self._bones.set_frame_cancel(cancel)

    def apply_bone_pinning(self, binding: SkinBinding, joint_delta=None) -> None:
        """Place a footprinted muscle from its attachments.

        Call after delta-matrix transform + neighbor clamping.

        With authored footprints and ``joint_delta`` (the skinning's
        current-times-inverse-rest transform for a joint index) the muscle's
        harmonic fibre field places every vertex from the RIGID images of its
        two footprints (:mod:`faceforge.anatomy.fibre_field`); without a
        field, the footprint vertices alone are blended toward their images.

        A muscle without footprints is left to the skinning solver.  The
        older path pinned the top and bottom 20% of the mesh's
        anterior-posterior extent toward the translation of the origin or
        insertion bone's centroid, which on a humerus swung through 90
        degrees pinned part of the belly to a place the bone no longer was:
        at the back-squat rack pose it took the biceps' stretch p99 from
        1.45x to 3.08x and the triceps medial head's from 1.30x to 5.90x.
        """
        data = self._attachments.get(id(binding))
        if (data is None or not data.footprint_masks or joint_delta is None
                or data.origin_joint is None or data.insertion_joint is None):
            return

        mesh = binding.mesh
        positions = mesh.geometry.positions.reshape(-1, 3)
        rest_pos = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        delta_o = np.asarray(joint_delta(data.origin_joint), dtype=np.float64)
        delta_i = np.asarray(joint_delta(data.insertion_joint), dtype=np.float64)

        if data.fibre_field is not None:
            data.fibre_field.apply(positions, delta_o, delta_i)
            return

        strength = (data.pin_strength if data.pin_strength is not None else PIN_STRENGTH)
        for mask, delta in ((data.origin_mask, delta_o), (data.insertion_mask, delta_i)):
            n = min(len(mask), len(positions), len(rest_pos))
            idx = np.where(mask[:n])[0]
            if not len(idx):
                continue
            target = rest_pos[idx] @ delta[:3, :3].T + delta[:3, 3]
            current = positions[idx].astype(np.float64)
            positions[idx] = (current + strength * (target - current)).astype(np.float32)

    def refresh_rest_poses(self, bindings) -> int:
        """Re-read every registered muscle's rest pose after the skeleton moved.

        The fibre field and the stretch clamp both hold numbers measured on
        the rest pose, and a sex morph or a skeleton fit rewrites that pose
        under them.  Left stale, the field writes the muscle's *old* geometry
        back over the new one every frame.
        """
        refreshed = 0
        for binding in bindings:
            data = self._attachments.get(id(binding))
            mesh = getattr(binding, "mesh", None)
            if data is None or mesh is None or mesh.rest_positions is None:
                continue
            rest = np.asarray(mesh.rest_positions, dtype=np.float64).reshape(-1, 3)
            if data.fibre_field is not None and data.fibre_field.refresh_rest(rest):
                refreshed += 1
            if not data.footprint_masks:
                continue
            om, im = data.origin_mask, data.insertion_mask
            no, ni = min(len(om), len(rest)), min(len(im), len(rest))
            if om[:no].any() and im[:ni].any():
                data.rest_length = max(1e-3, float(np.linalg.norm(
                    rest[:no][om[:no]].mean(axis=0)
                    - rest[:ni][im[:ni]].mean(axis=0))))
        if refreshed:
            logger.info("Fibre fields moved onto the new rest pose: %d muscles",
                        refreshed)
        return refreshed

    def has_fibre_field(self, binding: SkinBinding) -> bool:
        """True when this muscle is placed by its harmonic fibre field."""
        data = self._attachments.get(id(binding))
        return data is not None and data.fibre_field is not None

    def apply_stretch_clamp(self, binding: SkinBinding) -> float:
        """Measure muscle stretch.  Returns the excess above MAX_STRETCH.

        Measurement only.  This used to blend the WHOLE mesh halfway back
        toward its rest position in space when the ratio exceeded the limit,
        and for any muscle on a moving limb that is a place the limb has
        left: at the back-squat rack pose it held the biceps a median 7.2
        units off the humerus (0.4 with the pull-back off) -- the muscles
        "sagging off the bone" that a user reported.  Excursion is the fibre
        field's business now; the ratio feeds the tension readout.
        """
        data = self._attachments.get(id(binding))
        if data is None:
            return 0.0

        mesh = binding.mesh
        positions = mesh.geometry.positions.reshape(-1, 3)
        V = len(positions)

        # Compute current length (same method as rest)
        if data.footprint_masks:
            om, im = data.origin_mask, data.insertion_mask
            no, ni = min(len(om), V), min(len(im), V)
            top_centroid = positions[:no][om[:no]].astype(np.float64).mean(axis=0)
            bot_centroid = positions[:ni][im[:ni]].astype(np.float64).mean(axis=0)
        else:
            y_vals = positions[:, 1].astype(np.float64)
            n15 = max(1, V // 7)
            top_idx = np.argpartition(y_vals, -n15)[-n15:]
            bot_idx = np.argpartition(y_vals, n15)[:n15]
            top_centroid = positions[top_idx].astype(np.float64).mean(axis=0)
            bot_centroid = positions[bot_idx].astype(np.float64).mean(axis=0)
        current_length = float(np.linalg.norm(top_centroid - bot_centroid))

        ratio = current_length / data.rest_length
        data.current_stretch = ratio

        limit = data.max_stretch if data.max_stretch is not None else MAX_STRETCH
        return max(0.0, ratio - limit)

    def get_total_tension_excess(self) -> float:
        """Sum of all muscles' stretch excess above MAX_STRETCH."""
        return sum(
            max(0.0, d.current_stretch
                - (d.max_stretch if d.max_stretch is not None else MAX_STRETCH))
            for d in self._attachments.values()
        )

    @property
    def attachment_count(self) -> int:
        return len(self._attachments)
