"""Per-bone affine scaling for gender dimorphism."""

import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.mesh import MeshInstance
from faceforge.core.config_loader import load_config

logger = logging.getLogger(__name__)


class BoneScaler:
    """Scales skeleton bone meshes for gender dimorphism.

    Supports two scaling modes:

    1. **Individual scaling** — each bone scales around its own centroid.
       Good for large bones (pelvis, femur, clavicle) where the shape
       change is important but position stays fixed.

    2. **Group scaling** — bones in a cluster (hands, feet, rib cage)
       scale around the group's shared centroid.  This prevents gaps
       between small adjacent bones that would appear with individual scaling.

    Groups are defined in ``gender_dimorphism.json`` under ``scale_groups``.
    """

    _VERTEBRA_RE = re.compile(r'^[TLC]\d{1,2}$')

    def __init__(self):
        self._config = load_config("gender_dimorphism.json")
        self._bone_scales: dict[str, list[float]] = self._config.get("bone_scales", {})
        self._patterns: list[dict] = (
            self._config.get("bone_name_patterns", {}).get("patterns", [])
        )
        # Build reverse lookup: scale_key → group_name (or None for individual)
        self._key_to_group: dict[str, Optional[str]] = {}
        groups_cfg = self._config.get("scale_groups", {})
        for group_name, scale_keys in groups_cfg.items():
            if group_name.startswith("_"):
                continue
            for key in scale_keys:
                self._key_to_group[key] = group_name

        # Backup of original rest positions: mesh id → positions array
        self._rest_backup: dict[int, NDArray[np.float32]] = {}
        # Cache: bone_name → scale key
        self._name_cache: dict[str, Optional[str]] = {}

    def _match_bone(self, bone_name: str) -> Optional[str]:
        """Find the scale key for a bone name using case-insensitive pattern matching."""
        if bone_name in self._name_cache:
            return self._name_cache[bone_name]

        name_lower = bone_name.lower()

        # Check JSON patterns first
        for pat in self._patterns:
            if pat["match"].lower() in name_lower:
                key = pat["key"]
                self._name_cache[bone_name] = key
                return key

        # Short vertebra names like T2, L1, C3
        if self._VERTEBRA_RE.match(bone_name):
            self._name_cache[bone_name] = "vertebra"
            return "vertebra"

        self._name_cache[bone_name] = None
        return None

    def _get_group(self, bone_name: str) -> Optional[str]:
        """Get the scale group for a bone, or None for individual scaling."""
        key = self._match_bone(bone_name)
        if key is None:
            return None
        return self._key_to_group.get(key)

    def compute_scale(self, bone_name: str, gender: float) -> Optional[tuple[float, float, float]]:
        """Compute the (sx, sy, sz) scale for a bone at a given gender value.

        Returns None if no scale is defined for this bone.
        """
        key = self._match_bone(bone_name)
        if key is None:
            return None

        female_scale = self._bone_scales.get(key)
        if female_scale is None:
            return None

        # Lerp from (1,1,1) at gender=0 to female_scale at gender=1
        g = max(0.0, min(1.0, gender))
        sx = 1.0 + (female_scale[0] - 1.0) * g
        sy = 1.0 + (female_scale[1] - 1.0) * g
        sz = 1.0 + (female_scale[2] - 1.0) * g
        return (sx, sy, sz)

    def _ensure_backup(self, mesh: MeshInstance) -> NDArray[np.float32]:
        """Back up a mesh's rest positions on first call; return the backup."""
        mesh_id = id(mesh)
        if mesh_id not in self._rest_backup:
            if mesh.rest_positions is not None:
                self._rest_backup[mesh_id] = mesh.rest_positions.copy()
            else:
                self._rest_backup[mesh_id] = mesh.positions.copy()
        return self._rest_backup[mesh_id]

    def _scale_around_centroid(
        self,
        mesh: MeshInstance,
        scale: tuple[float, float, float],
        centroid: NDArray[np.float32],
    ) -> None:
        """Scale a mesh's positions around a given centroid point."""
        rest = self._ensure_backup(mesh)
        n_verts = len(rest) // 3
        pos = rest.reshape(n_verts, 3)

        sx, sy, sz = scale
        scaled = pos.copy()
        scaled[:, 0] = centroid[0] + (pos[:, 0] - centroid[0]) * sx
        scaled[:, 1] = centroid[1] + (pos[:, 1] - centroid[1]) * sy
        scaled[:, 2] = centroid[2] + (pos[:, 2] - centroid[2]) * sz

        mesh.geometry.positions = scaled.reshape(-1).astype(np.float32)
        mesh.needs_update = True

    def apply_to_mesh(
        self,
        mesh: MeshInstance,
        bone_name: str,
        gender: float,
    ) -> bool:
        """Scale a single mesh around its own centroid.

        For group-scaled bones, prefer ``apply_to_all`` which handles
        group centroid computation automatically.

        Returns True if scaling was applied.
        """
        scale = self.compute_scale(bone_name, gender)
        if scale is None:
            return False

        rest = self._ensure_backup(mesh)
        centroid = rest.reshape(-1, 3).mean(axis=0)
        self._scale_around_centroid(mesh, scale, centroid)
        return True

    def reset_mesh(self, mesh: MeshInstance) -> None:
        """Restore a mesh to its original (backed up) positions."""
        mesh_id = id(mesh)
        backup = self._rest_backup.get(mesh_id)
        if backup is not None:
            mesh.geometry.positions = backup.copy()
            mesh.needs_update = True

    def apply_to_all(
        self,
        bone_meshes: list[tuple[str, MeshInstance]],
        gender: float,
    ) -> int:
        """Apply gender scaling to a list of (bone_name, mesh) pairs.

        Bones belonging to a scale group are scaled around the group's
        shared centroid.  All other bones scale around their own centroid.

        Returns the number of meshes that were scaled.
        """
        # Partition into individual and group bones
        individual: list[tuple[str, MeshInstance]] = []
        groups: dict[str, list[tuple[str, MeshInstance]]] = defaultdict(list)

        for bone_name, mesh in bone_meshes:
            scale = self.compute_scale(bone_name, gender)
            if scale is None:
                continue
            group = self._get_group(bone_name)
            if group is not None:
                groups[group].append((bone_name, mesh))
            else:
                individual.append((bone_name, mesh))

        count = 0

        # Individual bones: scale around own centroid
        for bone_name, mesh in individual:
            scale = self.compute_scale(bone_name, gender)
            rest = self._ensure_backup(mesh)
            centroid = rest.reshape(-1, 3).mean(axis=0)
            self._scale_around_centroid(mesh, scale, centroid)
            count += 1

        # Group bones: compute shared centroid, then scale each bone around it
        for group_name, members in groups.items():
            # Compute group centroid from all member bones' rest positions
            all_centroids = []
            all_weights = []
            for bone_name, mesh in members:
                rest = self._ensure_backup(mesh)
                pos = rest.reshape(-1, 3)
                all_centroids.append(pos.mean(axis=0))
                all_weights.append(len(pos))

            # Weighted centroid (by vertex count)
            centroids = np.array(all_centroids)
            weights = np.array(all_weights, dtype=np.float64)
            group_centroid = (centroids * weights[:, None]).sum(axis=0) / weights.sum()

            for bone_name, mesh in members:
                scale = self.compute_scale(bone_name, gender)
                self._scale_around_centroid(mesh, scale, group_centroid.astype(np.float32))
                count += 1

            logger.debug(
                "Group '%s': %d bones scaled around centroid (%.1f, %.1f, %.1f)",
                group_name, len(members),
                group_centroid[0], group_centroid[1], group_centroid[2],
            )

        return count

    def reset_all(self, meshes: list[MeshInstance]) -> None:
        """Restore all meshes to their original positions."""
        for mesh in meshes:
            self.reset_mesh(mesh)
