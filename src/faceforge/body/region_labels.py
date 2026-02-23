"""Anatomical region labeling for body mesh segmentation.

Segments MakeHuman and BP3D meshes into 15 matching anatomical regions
to constrain surface projection and prevent cross-region errors.
"""

from __future__ import annotations

import json
import logging
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default override file location
_ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "config"
DEFAULT_REGION_OVERRIDES_PATH = _ASSETS_DIR / "body_mesh_regions.json"


class BodyRegion(IntEnum):
    """15 anatomical regions for body mesh segmentation."""
    HEAD = 0
    NECK = 1
    TORSO = 2
    UPPER_ARM_R = 3
    UPPER_ARM_L = 4
    FOREARM_R = 5
    FOREARM_L = 6
    HAND_R = 7
    HAND_L = 8
    THIGH_R = 9
    THIGH_L = 10
    CALF_R = 11
    CALF_L = 12
    FOOT_R = 13
    FOOT_L = 14


# Distinct colors for each region (float32, 0-1 range)
REGION_COLORS = np.array([
    [0.90, 0.30, 0.30],  # HEAD — red
    [0.90, 0.55, 0.25],  # NECK — orange
    [0.85, 0.85, 0.30],  # TORSO — yellow
    [0.30, 0.70, 0.30],  # UPPER_ARM_R — green
    [0.30, 0.90, 0.50],  # UPPER_ARM_L — light green
    [0.25, 0.60, 0.85],  # FOREARM_R — blue
    [0.40, 0.75, 0.95],  # FOREARM_L — light blue
    [0.55, 0.30, 0.85],  # HAND_R — purple
    [0.75, 0.45, 0.95],  # HAND_L — light purple
    [0.85, 0.45, 0.60],  # THIGH_R — pink
    [0.95, 0.55, 0.70],  # THIGH_L — light pink
    [0.50, 0.80, 0.80],  # CALF_R — teal
    [0.60, 0.90, 0.85],  # CALF_L — light teal
    [0.80, 0.65, 0.35],  # FOOT_R — gold
    [0.90, 0.75, 0.45],  # FOOT_L — light gold
], dtype=np.float32)


# ── Auto-segmentation ────────────────────────────────────────────


def _extract_z_thresholds(
    skel_lm: dict[str, NDArray],
) -> dict[str, float]:
    """Extract Z thresholds from skeleton landmarks.

    Returns dict with keys: head_z, neck_z, shoulder_z, elbow_z_r/l,
    wrist_z_r/l, hip_z, knee_z_r/l, ankle_z_r/l.
    """
    def _avg(key: str, axis: int = 2) -> Optional[float]:
        r = skel_lm.get(f"{key}_R")
        l = skel_lm.get(f"{key}_L")
        if r is not None and l is not None:
            return (float(r[axis]) + float(l[axis])) / 2
        if r is not None:
            return float(r[axis])
        return float(l[axis]) if l is not None else None

    def _side(key: str, side: str, axis: int = 2) -> Optional[float]:
        v = skel_lm.get(f"{key}_{side}")
        return float(v[axis]) if v is not None else None

    result: dict[str, float] = {}
    sh_z = _avg("shoulder")
    if sh_z is not None:
        result["shoulder_z"] = sh_z
        # Head starts ~10 units above shoulders, neck is in between
        result["neck_z"] = sh_z + 5.0
        result["head_z"] = sh_z + 10.0

    hip_z = _avg("hip")
    if hip_z is not None:
        result["hip_z"] = hip_z

    for side in ("R", "L"):
        s = side.lower()
        el_z = _side("elbow", side)
        if el_z is not None:
            result[f"elbow_z_{s}"] = el_z
        wr_z = _side("wrist", side)
        if wr_z is not None:
            result[f"wrist_z_{s}"] = wr_z
        kn_z = _side("knee", side)
        if kn_z is not None:
            result[f"knee_z_{s}"] = kn_z
        an_z = _side("ankle", side)
        if an_z is not None:
            result[f"ankle_z_{s}"] = an_z

    return result


def segment_mh_mesh(
    positions: NDArray,
    skel_lm: dict[str, NDArray],
) -> NDArray:
    """Segment MakeHuman body mesh vertices into anatomical regions.

    Parameters
    ----------
    positions : (V, 3) float
        Vertex positions (in BP3D/world coords after alignment).
    skel_lm : dict
        Skeleton landmarks with shoulder/elbow/wrist/hip/knee/ankle positions.

    Returns
    -------
    (V,) int32 — region label per vertex (BodyRegion values).
    """
    V = len(positions)
    labels = np.full(V, BodyRegion.TORSO, dtype=np.int32)

    z = positions[:, 2]
    x = positions[:, 0]

    th = _extract_z_thresholds(skel_lm)
    if not th:
        logger.warning("No skeleton landmarks — returning all TORSO")
        return labels

    sh_z = th.get("shoulder_z", 0.0)
    neck_z = th.get("neck_z", sh_z + 5.0)
    head_z = th.get("head_z", sh_z + 10.0)
    hip_z = th.get("hip_z", sh_z - 40.0)

    # Head (above neck transition)
    labels[z > head_z] = BodyRegion.HEAD

    # Neck (between shoulder and head)
    neck_mask = (z > neck_z) & (z <= head_z) & (np.abs(x) < 14)
    labels[neck_mask] = BodyRegion.NECK

    # Arms: lateral vertices (|X| > 14) above hip
    for side, region_ua, region_fa, region_hand, x_sign in (
        ("r", BodyRegion.UPPER_ARM_R, BodyRegion.FOREARM_R, BodyRegion.HAND_R, 1),
        ("l", BodyRegion.UPPER_ARM_L, BodyRegion.FOREARM_L, BodyRegion.HAND_L, -1),
    ):
        lateral = x * x_sign
        arm_mask = (lateral > 14) & (z <= sh_z + 8) & (z > hip_z)

        el_z = th.get(f"elbow_z_{side}", sh_z - 15)
        wr_z = th.get(f"wrist_z_{side}", el_z - 15)

        ua_mask = arm_mask & (z > el_z)
        fa_mask = arm_mask & (z <= el_z) & (z > wr_z)
        hand_mask = arm_mask & (z <= wr_z)

        labels[ua_mask] = region_ua
        labels[fa_mask] = region_fa
        labels[hand_mask] = region_hand

    # Legs: below hip
    for side, region_th, region_ca, region_ft, x_sign in (
        ("r", BodyRegion.THIGH_R, BodyRegion.CALF_R, BodyRegion.FOOT_R, 1),
        ("l", BodyRegion.THIGH_L, BodyRegion.CALF_L, BodyRegion.FOOT_L, -1),
    ):
        kn_z = th.get(f"knee_z_{side}", hip_z - 25)
        an_z = th.get(f"ankle_z_{side}", kn_z - 25)

        # For legs, use X sign to separate sides (X>0 = right, X<0 = left)
        side_mask = (x * x_sign >= 0) if x_sign == 1 else (x * x_sign > 0)

        thigh_mask = side_mask & (z <= hip_z) & (z > kn_z)
        calf_mask = side_mask & (z <= kn_z) & (z > an_z)
        foot_mask = side_mask & (z <= an_z)

        labels[thigh_mask] = region_th
        labels[calf_mask] = region_ca
        labels[foot_mask] = region_ft

    return labels


def segment_bp3d_skin(
    positions: NDArray,
    tri_indices: NDArray,
    skel_lm: dict[str, NDArray],
) -> NDArray:
    """Segment BP3D skin mesh triangles into anatomical regions.

    Parameters
    ----------
    positions : (V, 3) float
        Mesh vertex positions.
    tri_indices : (F, 3) int
        Triangle index array.
    skel_lm : dict
        Skeleton landmarks.

    Returns
    -------
    (F,) int32 — region label per triangle (BodyRegion values).
    """
    tri_verts = positions[tri_indices]  # (F, 3, 3)
    centroids = tri_verts.mean(axis=1)  # (F, 3)

    # Reuse the vertex segmentation on centroids
    return segment_mh_mesh(centroids, skel_lm)


# ── Per-region KDTree builder ─────────────────────────────────────


def build_region_kdtrees(
    mesh_pos: NDArray,
    mesh_tris: NDArray,
    tri_regions: NDArray,
) -> dict[int, tuple]:
    """Build per-region KDTrees from triangle centroids.

    Parameters
    ----------
    mesh_pos : (V, 3) float64
        BP3D skin vertex positions.
    mesh_tris : (F, 3) int
        Triangle index array.
    tri_regions : (F,) int32
        Per-triangle region labels.

    Returns
    -------
    dict[int, (cKDTree, NDArray)]
        region_id → (tree built from centroids, global_tri_indices array).
    """
    from scipy.spatial import cKDTree

    tri_verts = mesh_pos[mesh_tris]  # (F, 3, 3)
    centroids = tri_verts.mean(axis=1)  # (F, 3)

    result: dict[int, tuple] = {}
    for region_id in np.unique(tri_regions):
        mask = tri_regions == region_id
        region_tri_idx = np.nonzero(mask)[0]
        region_centroids = centroids[mask]
        if len(region_centroids) == 0:
            continue
        tree = cKDTree(region_centroids)
        result[int(region_id)] = (tree, region_tri_idx)

    logger.info(
        "Built %d region KDTrees (total %d triangles)",
        len(result), len(mesh_tris),
    )
    return result


# ── Visualization ─────────────────────────────────────────────────


def compute_region_colors(labels: NDArray) -> NDArray:
    """Map region labels to vertex colors.

    Parameters
    ----------
    labels : (V,) int32
        Region label per vertex.

    Returns
    -------
    (V*3,) float32 — flat color array for GL upload.
    """
    color_idx = np.clip(labels, 0, len(REGION_COLORS) - 1)
    colors = REGION_COLORS[color_idx]  # (V, 3)
    return colors.ravel().astype(np.float32)


# ── Override persistence ──────────────────────────────────────────


def save_region_overrides(
    overrides: dict,
    path: Path | str | None = None,
) -> Path:
    """Save region label overrides to JSON.

    Parameters
    ----------
    overrides : dict
        ``{"mh_body": {vertex_idx_str: region_id}, "bp3d_skin": {tri_idx_str: region_id}}``
    path : Path or str, optional
        Output file. Defaults to ``assets/config/body_mesh_regions.json``.

    Returns
    -------
    Path — the file that was written.
    """
    if path is None:
        path = DEFAULT_REGION_OVERRIDES_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"version": 1}
    for key in ("mh_body", "bp3d_skin"):
        section = overrides.get(key, {})
        data[key] = {str(k): int(v) for k, v in section.items()}

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    total = sum(len(overrides.get(k, {})) for k in ("mh_body", "bp3d_skin"))
    logger.info("Saved %d region overrides to %s", total, path)
    return path


def load_region_overrides(
    path: Path | str | None = None,
) -> dict:
    """Load region label overrides from JSON.

    Returns
    -------
    dict
        ``{"mh_body": {int_idx: int_region}, "bp3d_skin": {int_idx: int_region}}``
        Empty sub-dicts if file doesn't exist.
    """
    if path is None:
        path = DEFAULT_REGION_OVERRIDES_PATH
    path = Path(path)

    if not path.exists():
        return {"mh_body": {}, "bp3d_skin": {}}

    with open(path) as f:
        data = json.load(f)

    if data.get("version") != 1:
        logger.warning("Unknown region overrides version: %s", data.get("version"))
        return {"mh_body": {}, "bp3d_skin": {}}

    result: dict[str, dict[int, int]] = {}
    for key in ("mh_body", "bp3d_skin"):
        section = data.get(key, {})
        result[key] = {int(k): int(v) for k, v in section.items()}

    total = sum(len(v) for v in result.values())
    logger.info("Loaded %d region overrides from %s", total, path)
    return result


def apply_region_overrides(
    labels: NDArray,
    overrides: dict[int, int],
) -> NDArray:
    """Apply manual overrides to region labels (in place).

    Parameters
    ----------
    labels : (N,) int32
        Region labels to modify.
    overrides : dict
        ``{index: region_id}`` pairs.

    Returns
    -------
    labels — the same array, modified in place.
    """
    for idx, region_id in overrides.items():
        if 0 <= idx < len(labels):
            labels[idx] = region_id
    return labels
