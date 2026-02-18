"""Cervical vertebrae (15 STLs) with articulation pivots.

Loads 8 vertebrae (C1-C7 + T1) and 7 intervertebral discs from STL files.
Each vertebra level gets a pivot group at its centroid for articulation.
The pivot groups are driven by ``HeadRotationSystem`` via the
``VERTEBRA_FRACTIONS`` table.

This module has ZERO GL imports; all data is stored as NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.loaders.stl_batch_loader import (
    CoordinateTransform, load_stl_batch, STLBatchResult,
)


@dataclass
class VertebraLevel:
    """Per-level vertebra data including its articulation pivot."""
    level: int
    group: SceneNode          # Pivot group node
    meshes: list[MeshInstance]  # Vertebra and/or disc meshes at this level
    fractions: dict[str, float]  # {yaw, pitch, roll} fraction from vertebra_fractions


class VertebraeSystem:
    """Manages cervical vertebrae with articulation pivots.

    Parameters
    ----------
    vertebra_defs:
        List of definition dicts loaded from ``cervical_vertebrae.json``.
        Each dict has: ``name``, ``stl``, ``type`` (vertebra/disc),
        ``level``, ``color``.
    vertebra_fractions:
        List of dicts ``{yaw, pitch, roll}`` for 8 levels (C1=index 0,
        T1=index 7), loaded from ``vertebra_fractions.json``.
    transform:
        BP3D-to-skull coordinate transform.
    """

    def __init__(
        self,
        vertebra_defs: list[dict],
        vertebra_fractions: list[dict[str, float]],
        transform: Optional[CoordinateTransform] = None,
    ) -> None:
        self._defs = vertebra_defs
        self._fractions = vertebra_fractions
        self._transform = transform or CoordinateTransform()
        self._levels: list[VertebraLevel] = []
        self._group: Optional[SceneNode] = None
        self._all_meshes: list[MeshInstance] = []

    @property
    def group(self) -> Optional[SceneNode]:
        return self._group

    @property
    def levels(self) -> list[VertebraLevel]:
        return self._levels

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, stl_dir=None) -> tuple[SceneNode, list[dict]]:
        """Load all cervical vertebrae and disc STL meshes.

        Uses ``create_pivots=True`` with ``pivot_key="level"`` so the batch
        loader automatically groups meshes into pivot nodes per level and
        positions each pivot at the centroid of its children.

        Returns
        -------
        tuple[SceneNode, list[dict]]
            The vertebrae group node, and a list of pivot info dicts
            compatible with ``HeadRotationSystem.apply(vertebrae_pivots=...)``.
            Each dict has: ``group`` (SceneNode), ``level`` (int),
            ``fractions`` (dict with yaw/pitch/roll).
        """
        from faceforge.constants import STL_DIR
        kwargs = {}
        if stl_dir is not None:
            kwargs["stl_dir"] = stl_dir

        result: STLBatchResult = load_stl_batch(
            self._defs,
            label="vertebrae",
            transform=self._transform,
            create_pivots=True,
            pivot_key="level",
            indexed=True,
            **kwargs,
        )

        self._group = result.group
        self._all_meshes = result.meshes

        # Build VertebraLevel entries from pivot groups
        level_meshes: dict[int, list[MeshInstance]] = {}
        for mesh, defn in zip(result.meshes, self._defs):
            level = defn.get("level", 0)
            if level not in level_meshes:
                level_meshes[level] = []
            level_meshes[level].append(mesh)

        for level, pivot_node in sorted(result.pivot_groups.items()):
            fracs = self._get_fractions(level)
            meshes = level_meshes.get(level, [])

            self._levels.append(VertebraLevel(
                level=level,
                group=pivot_node,
                meshes=meshes,
                fractions=fracs,
            ))

        # Build the pivot info list for HeadRotationSystem
        pivot_info = self._build_pivot_info()

        return self._group, pivot_info

    def _get_fractions(self, level: int) -> dict[str, float]:
        """Get rotation fractions for a given vertebra level."""
        if level < len(self._fractions):
            return self._fractions[level].copy()
        # Default: no rotation (T1 and below)
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    def _build_pivot_info(self) -> list[dict]:
        """Build the pivot info list expected by HeadRotationSystem."""
        return [
            {
                "group": vl.group,
                "level": vl.level,
                "fractions": vl.fractions,
            }
            for vl in self._levels
        ]

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        """Toggle visibility of all vertebrae and discs."""
        if self._group is not None:
            self._group.visible = visible
        for mesh in self._all_meshes:
            mesh.visible = visible

    @property
    def pivot_info(self) -> list[dict]:
        """Return pivot info list for use by HeadRotationSystem."""
        return self._build_pivot_info()
