"""Landmarks: the joints of a skeleton, and the same joints on a body mesh.

The surface fit needs the two in the same terms, so the skeleton's come from
its own bone meshes -- the ends of a humerus, a radius, a femur, a tibia, the
tip of a middle finger -- and the mesh's are found by shape: the wrist and the
ankle are where a limb is narrowest between a forearm and a hand, or a shank
and a foot.  Taking the lowest tenth of an arm's vertices instead put the
"wrist" in the fingertips, and the forearm was sheared to match.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_bone_endpoints(
    assets: Any, fma_id: str, frac: float = 0.15,
) -> tuple[NDArray, NDArray] | None:
    """Load a bone STL and return (proximal, distal) centroids in skull coords.

    Proximal = top ``frac`` by Z, distal = bottom ``frac`` by Z.
    """
    try:
        geom = assets.get_stl(fma_id)
    except Exception:
        return None
    pos = geom.positions.reshape(-1, 3)[:geom.vertex_count].copy()
    t = assets.transform
    pos[:, 0] = (pos[:, 0] - t.center_x) * t.scale_x + t.skull_center_x
    pos[:, 1] = (pos[:, 1] - t.center_y) * t.scale_y + t.skull_center_y
    pos[:, 2] = (pos[:, 2] - t.center_z) * t.scale_z + t.skull_center_z

    n = max(1, int(len(pos) * frac))
    z_sorted = np.argsort(pos[:, 2])
    proximal = pos[z_sorted[-n:]].mean(axis=0)
    distal = pos[z_sorted[:n]].mean(axis=0)
    return proximal, distal

def extract_skeleton_landmarks(assets: Any) -> dict[str, NDArray] | None:
    """Extract joint landmarks from skeleton bone STLs."""
    result: dict[str, NDArray] = {}

    for side_char, fma_offset in (("R", 0), ("L", 1)):
        hum = load_bone_endpoints(assets, f"FMA{23130 + fma_offset}")
        rad = load_bone_endpoints(assets, f"FMA{23464 + fma_offset}")
        if hum is None:
            logger.warning("Cannot load humerus — skipping warp")
            return None
        result[f"shoulder_{side_char}"] = hum[0]
        result[f"elbow_{side_char}"] = hum[1]
        if rad is not None:
            result[f"wrist_{side_char}"] = rad[1]

        # The middle finger's distal phalanx gives the hand a landmark.  Without
        # one the hand is only carried along by the forearm's transform, and
        # the skeleton's fingertips stood 15.6 units outside the surface.
        tip_id = "FMA24461" if side_char == "R" else "FMA23955"
        tip = load_bone_endpoints(assets, tip_id)
        if tip is not None:
            result[f"hand_{side_char}"] = tip[1]

        fem = load_bone_endpoints(assets, f"FMA{24474 + fma_offset}")
        tib = load_bone_endpoints(assets, f"FMA{24477 + fma_offset}")
        if fem is None:
            logger.warning("Cannot load femur — skipping warp")
            return None
        result[f"hip_{side_char}"] = fem[0]
        result[f"knee_{side_char}"] = fem[1]
        if tib is not None:
            result[f"ankle_{side_char}"] = tib[1]

    logger.info("Skeleton landmarks: %d entries", len(result))
    return result

# ── BP3D skin mesh loading ──────────────────────────────────

def load_bp3d_skin_mesh(
    assets: Any, cache: dict,
) -> Optional[tuple[NDArray, NDArray, NDArray]]:
    """Load BP3D skin STL (FMA7163) and return full mesh data.

    Returns (positions, normals, tri_indices) in skull coords, or None.
    Caches the result for reuse.
    """
    if "skin" in cache:
        return cache["skin"]

    try:
        geom = assets.get_stl("FMA7163", indexed=True)
    except Exception:
        logger.warning("BP3D skin (FMA7163) not available — skipping surface refinement")
        return None

    pos = geom.positions.reshape(-1, 3)[:geom.vertex_count].copy().astype(np.float64)
    nrm = geom.normals.reshape(-1, 3)[:geom.vertex_count].copy().astype(np.float64)

    if geom.indices is None:
        logger.warning("BP3D skin has no indices — skipping surface refinement")
        return None
    tri_indices = geom.indices.reshape(-1, 3)

    # Transform positions to skull coords
    t = assets.transform
    pos[:, 0] = (pos[:, 0] - t.center_x) * t.scale_x + t.skull_center_x
    pos[:, 1] = (pos[:, 1] - t.center_y) * t.scale_y + t.skull_center_y
    pos[:, 2] = (pos[:, 2] - t.center_z) * t.scale_z + t.skull_center_z

    # Transform normals: flip X, renormalize
    nrm[:, 0] = -nrm[:, 0]
    lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm /= np.maximum(lengths, 1e-12)

    cache["skin"] = (pos, nrm, tri_indices)
    logger.info(
        "Loaded BP3D skin mesh: %d vertices, %d triangles",
        len(pos), len(tri_indices),
    )
    return cache["skin"]

def extract_mesh_landmarks(pos: NDArray) -> dict[str, NDArray]:
    """Extract approximate joint landmarks from the body mesh vertices."""
    result: dict[str, NDArray] = {}
    z = pos[:, 2]
    x = pos[:, 0]

    for side_char, x_sign in (("R", 1), ("L", -1)):
        # ── Arms: lateral vertices above hip level ──
        arm_mask = (x * x_sign > 14) & (z > -95)
        if arm_mask.sum() < 10:
            continue
        arm_pts = pos[arm_mask]
        arm_z = arm_pts[:, 2]

        n_top = max(1, int(arm_mask.sum() * 0.10))
        result[f"shoulder_{side_char}"] = arm_pts[
            np.argsort(arm_z)[-n_top:]
        ].mean(axis=0)

        # The wrist is where the arm is narrowest, not where it ends: taking
        # the lowest tenth of the arm's vertices put this landmark in the
        # fingertips, so the forearm rotation mapped fingertip-to-elbow onto
        # radius-to-elbow and sheared the forearm -- measured, its depth fell
        # from 24.7 to 17.4 while its width rose from 21.5 to 28.4.
        wrist_z = find_narrowest_z(arm_pts, float(arm_z.min()),
                                   (float(arm_z.min()) + float(arm_z.max())) / 2)
        band = np.abs(arm_z - wrist_z) < 2.5
        if band.sum() >= 4:
            result[f"wrist_{side_char}"] = arm_pts[band].mean(axis=0)
        else:
            n_bot = max(1, int(arm_mask.sum() * 0.10))
            result[f"wrist_{side_char}"] = arm_pts[
                np.argsort(arm_z)[:n_bot]
            ].mean(axis=0)

        mid_z = (arm_z.max() + arm_z.min()) / 2
        elbow_band = (arm_z > mid_z - 5) & (arm_z < mid_z + 5)
        if elbow_band.sum() > 0:
            result[f"elbow_{side_char}"] = arm_pts[elbow_band].mean(axis=0)
        else:
            result[f"elbow_{side_char}"] = (
                result[f"shoulder_{side_char}"]
                + result[f"wrist_{side_char}"]
            ) / 2

        # The fingertip: the point furthest from the elbow.  The arm mask
        # stops at hip level so that it does not swallow the thigh, which
        # leaves the fingers outside it; they are added back by their
        # distance from the midline, where no thigh reaches.  Without this
        # the "fingertip" landed 8 units from the wrist against the
        # skeleton's 26, and the hand segment was scaled 3.1x.
        el = result.get(f"elbow_{side_char}")
        hand_mask = arm_mask | ((x * x_sign > 26) & (z <= -95) & (z > -130))
        hand_pts = pos[hand_mask]
        if el is not None and len(hand_pts):
            far = np.linalg.norm(hand_pts - el, axis=1)
            result[f"hand_{side_char}"] = hand_pts[int(np.argmax(far))]

        # ── Legs: below hip, not too wide ──
        leg_mask = (z < -90) & (x * x_sign > 3) & (np.abs(x) < 28)
        if leg_mask.sum() < 10:
            continue
        leg_pts = pos[leg_mask]
        leg_z = leg_pts[:, 2]

        n_top = max(1, int(leg_mask.sum() * 0.10))
        result[f"hip_{side_char}"] = leg_pts[
            np.argsort(leg_z)[-n_top:]
        ].mean(axis=0)

        # Ankle: detect leg→foot transition via Y-extent widening
        ankle_z = find_ankle_z(pos, x_sign)
        ankle_band = leg_mask & (z > ankle_z - 3) & (z < ankle_z + 3)
        if ankle_band.sum() > 0:
            result[f"ankle_{side_char}"] = pos[ankle_band].mean(axis=0)
        else:
            n_bot = max(1, int(leg_mask.sum() * 0.10))
            result[f"ankle_{side_char}"] = leg_pts[
                np.argsort(leg_z)[:n_bot]
            ].mean(axis=0)

        hi_z = float(result[f"hip_{side_char}"][2])
        an_z = float(result[f"ankle_{side_char}"][2])
        knee_z_val = (hi_z + an_z) / 2
        knee_band = leg_mask & (z > knee_z_val - 5) & (z < knee_z_val + 5)
        if knee_band.sum() > 0:
            result[f"knee_{side_char}"] = pos[knee_band].mean(axis=0)
        else:
            result[f"knee_{side_char}"] = (
                result[f"hip_{side_char}"]
                + result[f"ankle_{side_char}"]
            ) / 2

        # Foot centroid (below ankle)
        foot_mask = (z < ankle_z) & (x * x_sign > 2)
        if foot_mask.sum() > 5:
            result[f"foot_{side_char}"] = pos[foot_mask].mean(axis=0)

    return result

def find_narrowest_z(pts: NDArray, z_lo: float, z_hi: float,
                     step: float = 1.5) -> float:
    """The Z between ``z_lo`` and ``z_hi`` where the limb's cross-section is smallest.

    Measured as the mean distance from each band's own centroid, so it does
    not care where the limb is, only how thick it is.  This is the wrist and
    the ankle: the narrow point between a forearm and a hand, or a shank and
    a foot.
    """
    levels = np.arange(z_lo + step, z_hi, step)
    best_z, best_r = None, np.inf
    for zl in levels:
        band = np.abs(pts[:, 2] - zl) < step
        if band.sum() < 6:
            continue
        q = pts[band][:, :2]
        r = float(np.linalg.norm(q - q.mean(axis=0), axis=1).mean())
        if r < best_r:
            best_r, best_z = r, float(zl)
    return best_z if best_z is not None else float(z_lo)


def find_ankle_z(pos: NDArray, x_sign: int) -> float:
    """Find ankle Z where leg transitions to foot (Y-extent increase).

    Scans from leg (top) downward; the ankle is the first Z where the
    cross-section Y-extent widens beyond the leg baseline.
    """
    z = pos[:, 2]
    x = pos[:, 0]
    y = pos[:, 1]
    z_min = float(z.min())

    # Scan Z-bands from bottom to z_min + 50 (covers foot + lower leg)
    z_levels = np.arange(z_min + 2, z_min + 50, 1.0)
    y_extents = []
    for zl in z_levels:
        band = (z > zl - 1.5) & (z < zl + 1.5) & (x * x_sign > 5)
        if band.sum() > 3:
            y_extents.append(float(y[band].max() - y[band].min()))
        else:
            y_extents.append(0.0)

    if not y_extents:
        return z_min + 10  # fallback

    n = len(y_extents)
    # Leg baseline from the upper 50% of z_levels (clearly leg)
    leg_extents = [ye for ye in y_extents[n // 2:] if ye > 0]
    if not leg_extents:
        return z_min + 10
    leg_median = float(np.median(leg_extents))
    threshold = leg_median * 1.3

    # Scan from TOP (leg) downward — ankle is first Z where
    # Y-extent exceeds the leg baseline threshold
    for i in range(n - 1, -1, -1):
        if y_extents[i] > threshold:
            return float(z_levels[i])

    return float(z_levels[0])
