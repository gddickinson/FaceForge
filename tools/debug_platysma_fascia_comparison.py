#!/usr/bin/env python3
"""Comprehensive comparison of Platysma fascia anchoring.

Generates images at multiple yaw levels showing WITH vs WITHOUT
fascia anchoring effect.
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
logger = logging.getLogger("fascia_comparison")
logger.setLevel(logging.INFO)

OUT = Path("results/platysma_fascia_comparison")
OUT.mkdir(parents=True, exist_ok=True)


def _reset(hs, pd):
    pd.md.mesh.geometry.positions[:] = pd.md.rest_positions.copy()
    hs.constraint_state = ConstraintState()


def main():
    logger.info("Loading scene...")
    hs = load_headless_scene()
    plat = hs.pipeline.platysma
    if plat is None or not plat.registered:
        logger.error("Platysma not registered")
        return

    pd = plat._platysma[0]
    name = pd.md.defn.get("name", "?")
    geom = pd.md.mesh.geometry
    rest = pd.rest_positions.copy()
    tris = (geom.indices.reshape(-1, 3) if geom.has_indices
            else np.arange(pd.vert_count).reshape(-1, 3))
    body_mask = pd.spine_fracs < 0.35

    yaw_levels = [0.0, 0.5, 1.0]
    views = ["front", "3q_front_r"]

    for yaw in yaw_levels:
        face = FaceState()
        face.head_yaw = yaw
        face_rest = FaceState()

        for mode in ["with_fascia", "without_fascia"]:
            # Disable/enable fascia
            if mode == "without_fascia":
                saved_fascia = plat._fascia
                plat._fascia = None
                saved_assigns = []
                for p in plat._platysma:
                    saved_assigns.append((p.fascia_assignments, p.fascia_region_names))
                    p.fascia_assignments = None
                    p.fascia_region_names = None

            _reset(hs, pd)
            apply_head_rotation(hs, face_rest)
            _reset(hs, pd)
            q = apply_head_rotation(hs, face)
            pos = geom.positions.reshape(-1, 3).astype(np.float64).copy()

            disp = np.linalg.norm(pos[body_mask] - rest[body_mask], axis=1)
            logger.info("yaw=%.1f %s: body-end disp mean=%.3f max=%.3f",
                        yaw, mode, disp.mean(), disp.max())

            for view in views:
                vp = CAMERA_PRESETS[view]
                render_mesh(
                    pos, rest, tris,
                    azimuth=vp["azimuth"],
                    elevation=vp.get("elevation", 5),
                    output_path=OUT / f"yaw{yaw:.1f}_{mode}_{view}.png",
                    title=f"{name} yaw={yaw:.1f} {mode} {view}",
                )

            # Restore fascia
            if mode == "without_fascia":
                plat._fascia = saved_fascia
                for i, p in enumerate(plat._platysma):
                    p.fascia_assignments, p.fascia_region_names = saved_assigns[i]

    logger.info("Done. Images in %s", OUT.resolve())


if __name__ == "__main__":
    main()
