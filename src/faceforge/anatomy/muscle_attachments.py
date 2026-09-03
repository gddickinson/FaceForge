"""Bone attachment constraints for body muscles.

Body muscles have anatomical origin and insertion bones.  This system pins
the endpoints of each muscle toward their respective bones, preventing
muscles from floating away from the skeleton during deformation.

Also provides per-muscle stretch monitoring (Layer 3) and fascia region
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
    """Bone-pinning and stretch monitoring for body muscles.

    For each registered muscle:
    1. Computes attachment fraction from mesh Y-extent (like neck spine_fracs)
    2. Identifies origin-end and insertion-end vertex zones
    3. Per-frame: queries current bone positions and pins muscle endpoints
    4. Monitors stretch ratio and clamps if exceeded
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
        changed = int((before != ji[:n]).sum())
        logger.info("Footprint reassignment for %s: %d/%d vertices "
                    "(origin joint %d, insertion joint %d, %d unreached)",
                    name, changed, n, j_o, j_i, int((~both).sum()))
        return changed

    def apply_bone_pinning(self, binding: SkinBinding) -> None:
        """Pin muscle endpoints toward their attachment bones.

        Call after delta-matrix transform + neighbor clamping.
        """
        data = self._attachments.get(id(binding))
        if data is None:
            return

        mesh = binding.mesh
        positions = mesh.geometry.positions.reshape(-1, 3)
        rest_pos = mesh.rest_positions.reshape(-1, 3).astype(np.float64)

        # Get current and rest bone positions for origin
        origin_cur = self._get_bone_centroid_current(data.origin_bones)
        origin_rest = self._get_bone_centroid_rest(data.origin_bones)

        if origin_cur is not None and origin_rest is not None:
            bone_delta = origin_cur - origin_rest  # (3,)
            self._pin_zone(
                positions, rest_pos, data.origin_mask, data.attachment_frac,
                bone_delta, data.origin_frac_threshold, towards_high=True,
                pin_strength=(data.pin_strength
                              if data.pin_strength is not None
                              else PIN_STRENGTH),
            )

        # Get current and rest bone positions for insertion
        insert_cur = self._get_bone_centroid_current(data.insertion_bones)
        insert_rest = self._get_bone_centroid_rest(data.insertion_bones)

        if insert_cur is not None and insert_rest is not None:
            bone_delta = insert_cur - insert_rest
            self._pin_zone(
                positions, rest_pos, data.insertion_mask, data.attachment_frac,
                bone_delta, data.insertion_frac_threshold, towards_high=False,
                pin_strength=(data.pin_strength
                              if data.pin_strength is not None
                              else PIN_STRENGTH),
            )

    def apply_stretch_clamp(self, binding: SkinBinding) -> float:
        """Monitor and clamp muscle stretch.  Returns excess above MAX_STRETCH.

        Call after bone pinning.
        """
        data = self._attachments.get(id(binding))
        if data is None:
            return 0.0

        mesh = binding.mesh
        positions = mesh.geometry.positions.reshape(-1, 3)
        rest_pos = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        V = len(positions)

        # Compute current length (same method as rest)
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
        excess = max(0.0, ratio - limit)
        if excess <= 0.0:
            return 0.0

        # Clamp: blend positions back toward rest + limited stretch
        # The amount to pull back is proportional to how much we exceed
        scale = limit / ratio  # < 1.0 when over-stretched
        pull_back = 1.0 - scale
        # Blend deformed positions back toward rest positions
        current_f64 = positions.astype(np.float64)
        clamped = current_f64 * (1.0 - pull_back * 0.5) + rest_pos * (pull_back * 0.5)
        positions[:] = clamped.astype(np.float32)

        return excess

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

    def _pin_zone(
        self,
        positions: np.ndarray,
        rest_pos: np.ndarray,
        mask: np.ndarray,
        frac: np.ndarray,
        bone_delta: np.ndarray,
        threshold: float,
        towards_high: bool,
        pin_strength: float = PIN_STRENGTH,
    ) -> None:
        """Pin vertices in a zone toward bone displacement.

        Parameters
        ----------
        positions : (V, 3) float32 — modified in place
        rest_pos : (V, 3) float64
        mask : (V,) bool — vertices in the zone
        frac : (V,) float64 — attachment fraction
        bone_delta : (3,) float64 — bone displacement (current - rest)
        threshold : float — fraction threshold (0.2 or 0.8)
        towards_high : bool — True for origin (high frac), False for insertion (low frac)
        """
        if not mask.any():
            return

        idx = np.where(mask)[0]
        current = positions[idx].astype(np.float64)

        # Quadratic falloff from the boundary of the zone
        if towards_high:
            # Origin: frac > threshold → strength increases toward 1.0
            zone_t = (frac[idx] - threshold) / (1.0 - threshold + 1e-6)
        else:
            # Insertion: frac < threshold → strength increases toward 0.0
            zone_t = (threshold - frac[idx]) / (threshold + 1e-6)

        zone_t = np.clip(zone_t, 0.0, 1.0)
        # Per-muscle override, resolved by the caller (this helper has no
        # access to the attachment record).
        strength = pin_strength * zone_t * zone_t  # quadratic falloff

        # Target: rest position + bone displacement
        target = rest_pos[idx] + bone_delta[np.newaxis, :]

        # Blend toward target
        pinned = current + strength[:, np.newaxis] * (target - current)
        positions[idx] = pinned.astype(np.float32)

    def _get_bone_centroid_current(self, bone_names: list[str]) -> NDArray[np.float64] | None:
        """Get averaged current position of named bones."""
        positions = []
        for name in bone_names:
            pos = self._bones.get_muscle_anchor_current(name, [name])
            if pos is not None:
                positions.append(pos)
        if not positions:
            return None
        return np.mean(positions, axis=0).astype(np.float64)

    def _get_bone_centroid_rest(self, bone_names: list[str]) -> NDArray[np.float64] | None:
        """Get averaged rest position of named bones."""
        positions = []
        for name in bone_names:
            pos = self._bones.get_muscle_anchor(name, [name])
            if pos is not None:
                positions.append(pos)
        if not positions:
            return None
        return np.mean(positions, axis=0).astype(np.float64)
