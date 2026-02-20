#!/usr/bin/env python3
"""Minimal isolated test of Platysma fascia anchoring effect.

Compares body-end vertex displacement WITH vs WITHOUT fascia at yaw=1.0,
with proper constraint state reset between runs to avoid contamination.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
from faceforge.core.state import FaceState, ConstraintState
from faceforge.core.math_utils import quat_identity
from tools.headless_loader import load_headless_scene, apply_head_rotation
from tools.mesh_renderer import render_mesh, CAMERA_PRESETS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("debug_minimal")
logger.setLevel(logging.INFO)

OUT = Path("results/platysma_fascia_debug")
OUT.mkdir(parents=True, exist_ok=True)


def _reset_state(hs, pd):
    """Reset positions and constraint state to clean baseline."""
    pd.md.mesh.geometry.positions[:] = pd.md.rest_positions.copy()
    # Reset constraint state to avoid cross-test contamination
    hs.constraint_state = ConstraintState()


def _capture_yaw(hs, pd, face_yaw, face_rest):
    """Apply rest then yaw, return (head_q, final_positions)."""
    _reset_state(hs, pd)
    apply_head_rotation(hs, face_rest)
    _reset_state(hs, pd)
    q = apply_head_rotation(hs, face_yaw)
    pos = pd.md.mesh.geometry.positions.reshape(-1, 3).astype(np.float64).copy()
    return q, pos


def main():
    logger.info("Loading scene...")
    hs = load_headless_scene()
    pipeline = hs.pipeline
    plat = pipeline.platysma
    if plat is None or not plat.registered:
        logger.error("Platysma not registered")
        return

    pd = plat._platysma[0]
    name = pd.md.defn.get("name", "?")
    geom = pd.md.mesh.geometry
    rest = pd.rest_positions.copy()
    tris = geom.indices.reshape(-1, 3) if geom.has_indices else np.arange(pd.vert_count).reshape(-1, 3)

    face_yaw = FaceState()
    face_yaw.head_yaw = 1.0
    face_rest = FaceState()

    fracs = pd.spine_fracs
    body_mask = fracs < 0.35
    skull_mask = fracs >= 0.65

    # ── Test A: yaw=1.0 WITH fascia (effective_fracs active) ──
    logger.info("\n=== TEST A: yaw=1.0 WITH fascia (effective_fracs) ===")
    q_a, pos_a = _capture_yaw(hs, pd, face_yaw, face_rest)
    logger.info("  head_q = [%.6f, %.6f, %.6f, %.6f]", *q_a)

    # ── Test B: yaw=1.0 WITHOUT fascia ──
    logger.info("\n=== TEST B: yaw=1.0 WITHOUT fascia ===")
    saved_fascia = plat._fascia
    plat._fascia = None
    saved_assigns = []
    for p in plat._platysma:
        saved_assigns.append((p.fascia_assignments, p.fascia_region_names))
        p.fascia_assignments = None
        p.fascia_region_names = None

    q_b, pos_b = _capture_yaw(hs, pd, face_yaw, face_rest)
    logger.info("  head_q = [%.6f, %.6f, %.6f, %.6f]", *q_b)

    # Restore
    plat._fascia = saved_fascia
    for i, p in enumerate(plat._platysma):
        p.fascia_assignments, p.fascia_region_names = saved_assigns[i]

    # ── Compare ──
    logger.info("\n=== COMPARISON ===")
    head_q_diff = np.linalg.norm(np.array(q_a) - np.array(q_b))
    logger.info("  head_q diff: %.10f", head_q_diff)
    if head_q_diff > 1e-4:
        logger.warning("  HEAD QUATERNIONS DIFFER! Using same head_q for fair comparison.")
        # Re-run test A with test B's head_q for fair comparison
        logger.info("  Re-running test A with q_b for fair comparison...")
        _reset_state(hs, pd)
        apply_head_rotation(hs, face_rest)
        _reset_state(hs, pd)
        # Directly call platysma update with known head_q
        plat.update(q_b, hs.pipeline.head_rotation._head_pivot)
        pos_a = geom.positions.reshape(-1, 3).astype(np.float64).copy()
        q_a = q_b
        logger.info("  Now using same head_q for both tests")

    disp_a = np.linalg.norm(pos_a - rest, axis=1)
    disp_b = np.linalg.norm(pos_b - rest, axis=1)
    diff_ab = np.linalg.norm(pos_a - pos_b, axis=1)

    logger.info("  BODY-END (frac<0.35, %d verts):", body_mask.sum())
    logger.info("    With fascia: disp mean=%.4f max=%.4f", disp_a[body_mask].mean(), disp_a[body_mask].max())
    logger.info("    No fascia:   disp mean=%.4f max=%.4f", disp_b[body_mask].mean(), disp_b[body_mask].max())
    logger.info("    Reduction:   mean=%.1f%% max=%.1f%%",
                (1 - disp_a[body_mask].mean() / max(disp_b[body_mask].mean(), 1e-10)) * 100,
                (1 - disp_a[body_mask].max() / max(disp_b[body_mask].max(), 1e-10)) * 100)

    logger.info("  SKULL-END (frac>0.65, %d verts):", skull_mask.sum())
    logger.info("    With fascia: disp mean=%.4f max=%.4f", disp_a[skull_mask].mean(), disp_a[skull_mask].max())
    logger.info("    No fascia:   disp mean=%.4f max=%.4f", disp_b[skull_mask].mean(), disp_b[skull_mask].max())
    skull_diff = diff_ab[skull_mask]
    logger.info("    Diff (A-B):  mean=%.6f max=%.6f (should be ~0)", skull_diff.mean(), skull_diff.max())

    logger.info("  ALL VERTS:")
    logger.info("    With fascia: disp mean=%.4f max=%.4f", disp_a.mean(), disp_a.max())
    logger.info("    No fascia:   disp mean=%.4f max=%.4f", disp_b.mean(), disp_b.max())

    # Show frac distribution for body-end verts
    logger.info("\n=== FRAC DISTRIBUTION (body-end) ===")
    for lo, hi in [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.35)]:
        m = (fracs >= lo) & (fracs < hi)
        if m.any():
            logger.info("  frac [%.2f, %.2f): %d verts, disp_a mean=%.4f, disp_b mean=%.4f, reduction=%.1f%%",
                        lo, hi, m.sum(),
                        disp_a[m].mean(), disp_b[m].mean(),
                        (1 - disp_a[m].mean() / max(disp_b[m].mean(), 1e-10)) * 100)

    # Render comparison images
    logger.info("\n=== RENDERING ===")
    for label, pos in [("A_with_fascia", pos_a), ("B_without_fascia", pos_b)]:
        for view in ["front", "3q_front_r", "right"]:
            vp = CAMERA_PRESETS[view]
            render_mesh(pos, rest, tris,
                        azimuth=vp["azimuth"], elevation=vp.get("elevation", 5),
                        output_path=OUT / f"minimal_{label}_{view}.png",
                        title=f"{name} {label} {view}")

    logger.info("\nDone. Images in %s", OUT.resolve())


if __name__ == "__main__":
    main()
