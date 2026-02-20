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
            )

        # Get current and rest bone positions for insertion
        insert_cur = self._get_bone_centroid_current(data.insertion_bones)
        insert_rest = self._get_bone_centroid_rest(data.insertion_bones)

        if insert_cur is not None and insert_rest is not None:
            bone_delta = insert_cur - insert_rest
            self._pin_zone(
                positions, rest_pos, data.insertion_mask, data.attachment_frac,
                bone_delta, data.insertion_frac_threshold, towards_high=False,
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

        excess = max(0.0, ratio - MAX_STRETCH)
        if excess <= 0.0:
            return 0.0

        # Clamp: blend positions back toward rest + limited stretch
        # The amount to pull back is proportional to how much we exceed
        scale = MAX_STRETCH / ratio  # < 1.0 when over-stretched
        pull_back = 1.0 - scale
        # Blend deformed positions back toward rest positions
        current_f64 = positions.astype(np.float64)
        clamped = current_f64 * (1.0 - pull_back * 0.5) + rest_pos * (pull_back * 0.5)
        positions[:] = clamped.astype(np.float32)

        return excess

    def get_total_tension_excess(self) -> float:
        """Sum of all muscles' stretch excess above MAX_STRETCH."""
        return sum(
            max(0.0, d.current_stretch - MAX_STRETCH)
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
        strength = PIN_STRENGTH * zone_t * zone_t  # quadratic falloff

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
