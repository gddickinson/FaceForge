"""Virtual fascia constraint surfaces for muscle attachment.

Fascia are connective tissue sheets that cover muscles.  The Platysma's
inferior border attaches to the fascia over the Pectoralis Major and
Deltoid muscles.  Since body muscles are demand-loaded (Tier 3-5) and
may not be present at startup, this module computes constraint target
positions from skeleton bones (clavicle, sternum, ribs, scapula) which
ARE loaded during Phase 3.

Each ``FasciaRegion`` defines a named attachment zone as a weighted
average of skeleton bone positions.  ``FasciaSystem`` caches rest-pose
targets and provides current-frame targets for per-vertex pinning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material, RenderMode
from faceforge.core.scene_graph import SceneNode

logger = logging.getLogger(__name__)


@dataclass
class FasciaRegion:
    """A named attachment zone defined by weighted bone positions.

    Attributes
    ----------
    name : str
        Unique region identifier (e.g. ``"pectoral_R"``).
    bone_names : list[str]
        Skeleton bone names that define this region.
    bone_weights : list[float]
        Per-bone weights (will be normalised internally).
    side : str
        Laterality: ``"R"``, ``"L"``, or ``"M"`` (midline).
    """

    name: str
    bone_names: list[str] = field(default_factory=list)
    bone_weights: list[float] = field(default_factory=list)
    side: str = "M"

    def __post_init__(self) -> None:
        if len(self.bone_names) != len(self.bone_weights):
            raise ValueError(
                f"FasciaRegion '{self.name}': bone_names ({len(self.bone_names)}) "
                f"and bone_weights ({len(self.bone_weights)}) must have equal length"
            )


class FasciaSystem:
    """Computes weighted bone centroids as fascia constraint targets.

    Parameters
    ----------
    regions : list[FasciaRegion]
        Pre-defined anatomical fascia regions.
    bone_registry : BoneAnchorRegistry
        Live bone node registry (must have rest positions snapshotted).
    """

    def __init__(
        self,
        regions: list[FasciaRegion],
        bone_registry: BoneAnchorRegistry,
    ) -> None:
        self._regions: dict[str, FasciaRegion] = {r.name: r for r in regions}
        self._bones = bone_registry
        self._rest_targets: dict[str, NDArray[np.float64]] = {}

    @property
    def region_names(self) -> list[str]:
        return list(self._regions.keys())

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def snapshot_rest(self) -> None:
        """Cache rest-pose targets for all regions.

        Call once after ``BoneAnchorRegistry.snapshot_rest_positions()``.
        """
        self._rest_targets.clear()
        for name, region in self._regions.items():
            target = self._weighted_centroid(region, use_rest=True)
            if target is not None:
                self._rest_targets[name] = target
            else:
                logger.debug("Fascia region '%s': no bones found in registry", name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_target_rest(self, region_name: str) -> Optional[NDArray[np.float64]]:
        """Return cached rest-pose target position, or ``None``."""
        return self._rest_targets.get(region_name)

    def get_target_current(self, region_name: str) -> Optional[NDArray[np.float64]]:
        """Return current-frame target from live bone nodes, or ``None``."""
        region = self._regions.get(region_name)
        if region is None:
            return None
        return self._weighted_centroid(region, use_rest=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _weighted_centroid(
        self,
        region: FasciaRegion,
        use_rest: bool,
    ) -> Optional[NDArray[np.float64]]:
        """Compute weighted average position of region bones.

        Parameters
        ----------
        region : FasciaRegion
            The region definition.
        use_rest : bool
            If True, use snapshotted rest positions from the registry.
            If False, query live world positions.

        Returns
        -------
        NDArray or None
            (3,) weighted centroid, or None if no bones found.
        """
        positions = []
        weights = []

        for bone_name, weight in zip(region.bone_names, region.bone_weights):
            if use_rest:
                pos = self._bones.get_muscle_anchor(bone_name, [bone_name])
            else:
                pos = self._bones.get_muscle_anchor_current(bone_name, [bone_name])

            if pos is not None:
                positions.append(pos)
                weights.append(weight)

        if not positions:
            return None

        pos_arr = np.array(positions, dtype=np.float64)
        w_arr = np.array(weights, dtype=np.float64)
        w_sum = w_arr.sum()
        if w_sum < 1e-12:
            return np.mean(pos_arr, axis=0)

        w_arr /= w_sum
        return (pos_arr * w_arr[:, None]).sum(axis=0)


def build_anatomical_fascia_regions() -> list[FasciaRegion]:
    """Factory returning pre-defined anatomical fascia regions.

    Returns ten regions covering the pectoral, deltoid, supraclavicular,
    and trapezius fasciae (bilateral) plus investing fascia.

    Regions per side (medial → posterior):
      1. pectoral — anterior chest/sternum
      2. investing — anteromedial (clavicle + sternum)
      3. supraclavicular — anterolateral (clavicle + rib + scapula)
      4. deltoid — lateral (clavicle + scapula)
      5. trapezius — posterolateral (scapula + thoracic spine)
    """
    return [
        # Pectoral fascia — covers Pectoralis Major
        FasciaRegion(
            name="pectoral_R",
            bone_names=[
                "Right Clavicle",
                "Manubrium",
                "Body of Sternum",
                "Right 1st Rib",
                "Right 2nd Rib",
                "Right 3rd Rib",
                "Right 4th Rib",
                "Right 5th Rib",
                "Right 6th Rib",
            ],
            bone_weights=[2.0, 1.5, 1.0, 0.5, 0.5, 0.5, 0.4, 0.3, 0.3],
            side="R",
        ),
        FasciaRegion(
            name="pectoral_L",
            bone_names=[
                "Left Clavicle",
                "Manubrium",
                "Body of Sternum",
                "Left 1st Rib",
                "Left 2nd Rib",
                "Left 3rd Rib",
                "Left 4th Rib",
                "Left 5th Rib",
                "Left 6th Rib",
            ],
            bone_weights=[2.0, 1.5, 1.0, 0.5, 0.5, 0.5, 0.4, 0.3, 0.3],
            side="L",
        ),
        # Deltoid fascia — covers Deltoid
        FasciaRegion(
            name="deltoid_R",
            bone_names=["Right Clavicle", "Right Scapula"],
            bone_weights=[0.4, 0.6],
            side="R",
        ),
        FasciaRegion(
            name="deltoid_L",
            bone_names=["Left Clavicle", "Left Scapula"],
            bone_weights=[0.4, 0.6],
            side="L",
        ),
        # Investing fascia — anteromedial (clavicle + sternum)
        FasciaRegion(
            name="investing_R",
            bone_names=["Right Clavicle", "Manubrium"],
            bone_weights=[1.0, 1.0],
            side="R",
        ),
        FasciaRegion(
            name="investing_L",
            bone_names=["Left Clavicle", "Manubrium"],
            bone_weights=[1.0, 1.0],
            side="L",
        ),
        # Supraclavicular fascia — anterolateral (clavicle + rib + scapula)
        FasciaRegion(
            name="supraclavicular_R",
            bone_names=["Right Clavicle", "Right 1st Rib", "Right Scapula"],
            bone_weights=[2.0, 0.5, 0.3],
            side="R",
        ),
        FasciaRegion(
            name="supraclavicular_L",
            bone_names=["Left Clavicle", "Left 1st Rib", "Left Scapula"],
            bone_weights=[2.0, 0.5, 0.3],
            side="L",
        ),
        # Trapezius fascia — posterolateral (scapula + thoracic spine)
        # Note: thoracic vertebrae are registered as pivot nodes, not by
        # their config names.  T2 = level 0, T3 = level 1.
        FasciaRegion(
            name="trapezius_R",
            bone_names=["Right Scapula", "thoracic_spine_pivot_0", "thoracic_spine_pivot_1"],
            bone_weights=[2.0, 1.0, 0.5],
            side="R",
        ),
        FasciaRegion(
            name="trapezius_L",
            bone_names=["Left Scapula", "thoracic_spine_pivot_0", "thoracic_spine_pivot_1"],
            bone_weights=[2.0, 1.0, 0.5],
            side="L",
        ),
        # SCM sternal tendon fascia — manubrium attachment
        FasciaRegion(
            name="scm_sternal_R",
            bone_names=["Manubrium"],
            bone_weights=[1.0],
            side="R",
        ),
        FasciaRegion(
            name="scm_sternal_L",
            bone_names=["Manubrium"],
            bone_weights=[1.0],
            side="L",
        ),
        # SCM clavicular tendon fascia — clavicle attachment
        FasciaRegion(
            name="scm_clavicular_R",
            bone_names=["Right Clavicle"],
            bone_weights=[1.0],
            side="R",
        ),
        FasciaRegion(
            name="scm_clavicular_L",
            bone_names=["Left Clavicle"],
            bone_weights=[1.0],
            side="L",
        ),
        # Levator scapulae tendon fascia — superior scapula angle
        FasciaRegion(
            name="lev_scap_R",
            bone_names=["Right Scapula"],
            bone_weights=[1.0],
            side="R",
        ),
        FasciaRegion(
            name="lev_scap_L",
            bone_names=["Left Scapula"],
            bone_weights=[1.0],
            side="L",
        ),
    ]


# ------------------------------------------------------------------
# Fascia Visualization
# ------------------------------------------------------------------

# Color map for fascia region types
_FASCIA_COLORS: dict[str, tuple[float, float, float]] = {
    "pectoral": (0.2, 0.4, 0.9),        # blue
    "deltoid": (0.2, 0.8, 0.3),         # green
    "investing": (0.9, 0.8, 0.2),       # yellow
    "trapezius": (0.9, 0.3, 0.2),       # red-orange
    "supraclavicular": (0.7, 0.2, 0.8), # purple
    "scm_sternal": (1.0, 0.5, 0.0),     # orange
    "scm_clavicular": (0.0, 0.7, 0.7),  # teal
    "lev_scap": (0.8, 0.4, 0.6),        # mauve
}


def _make_octahedron(
    center: NDArray[np.float64],
    size: float = 1.5,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Create an octahedron mesh at *center* with given *size*.

    Returns (positions, normals) as flat float32 arrays (24 vertices,
    8 triangles, non-indexed with per-face normals).
    """
    s = size
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    # 6 vertices of a regular octahedron
    verts = np.array([
        [cx, cy + s, cz],      # 0: top
        [cx, cy - s, cz],      # 1: bottom
        [cx + s, cy, cz],      # 2: right
        [cx - s, cy, cz],      # 3: left
        [cx, cy, cz + s],      # 4: front
        [cx, cy, cz - s],      # 5: back
    ], dtype=np.float64)

    # 8 triangular faces (CCW winding)
    faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),  # top 4
        (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),  # bottom 4
    ]

    positions = []
    normals = []
    for i0, i1, i2 in faces:
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        n = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(n)
        if n_len > 1e-10:
            n /= n_len
        for v in (v0, v1, v2):
            positions.extend(v)
            normals.extend(n)

    return (
        np.array(positions, dtype=np.float32),
        np.array(normals, dtype=np.float32),
    )


def create_fascia_markers(fascia: FasciaSystem) -> SceneNode:
    """Create visible octahedron markers at each fascia region's rest position.

    Returns a SceneNode group containing all markers.  Caller should add
    this group as a child of the fasciaGroup node.
    """
    group = SceneNode(name="fascia_markers")

    for region_name in fascia.region_names:
        target = fascia.get_target_rest(region_name)
        if target is None:
            logger.debug("Skipping fascia marker for '%s' (no target)", region_name)
            continue

        # Determine color from region type
        color = (0.7, 0.7, 0.7)  # default grey
        for prefix, c in _FASCIA_COLORS.items():
            if region_name.startswith(prefix):
                color = c
                break

        positions, normals = _make_octahedron(target, size=2.0)
        geom = BufferGeometry(
            positions=positions,
            normals=normals,
            vertex_count=len(positions) // 3,
        )
        mat = Material(
            color=color,
            opacity=0.8,
            transparent=True,
            render_mode=RenderMode.SOLID,
        )
        mesh = MeshInstance(
            name=f"fascia_{region_name}",
            geometry=geom,
            material=mat,
        )
        node = SceneNode(name=f"fascia_{region_name}")
        node.mesh = mesh
        group.add(node)

        logger.info("Fascia marker '%s' at [%.1f, %.1f, %.1f] color=%s",
                     region_name, target[0], target[1], target[2], color)

    return group
