"""Visual head rotation test runner.

Loads headless scene, applies head poses, extracts mesh groups, and renders
multi-mesh views to PNG images.

Usage::

    python -m tools.head_rotation_test [--quick] [--poses yaw_left combined]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from faceforge.core.state import FaceState
from tools.head_renderer import (
    MeshGroup, GROUP_COLORS,
    HEAD_CAMERA_PRESETS, HEAD_QUICK_VIEWS,
    render_head_multimesh, render_head_multiview,
)
from tools.head_rotation_diagnostic import HEAD_POSES

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/head_rotation")


def _extract_mesh_groups(hs) -> list[MeshGroup]:
    """Extract mesh groups from the headless scene for rendering."""
    groups = []

    group_specs = [
        ("skullGroup", "skull"),
        ("faceGroup", "face"),
        ("stlMuscleGroup", "jaw_muscles"),
        ("exprMuscleGroup", "expression_muscles"),
        ("faceFeatureGroup", "face_features"),
        ("neckMuscleGroup", "neck_muscles"),
        ("vertebraeGroup", "vertebrae"),
    ]

    for node_name, display_name in group_specs:
        node = hs.named_nodes.get(node_name)
        if node is None:
            continue

        all_positions = []
        all_triangles = []
        vertex_offset = 0

        def _collect(n):
            nonlocal vertex_offset
            if n.mesh is not None:
                pos = n.mesh.geometry.positions
                idx = n.mesh.geometry.indices
                if pos is not None and len(pos) > 0 and idx is not None and len(idx) > 0:
                    pos3 = np.asarray(pos, dtype=np.float64).reshape(-1, 3)

                    # Transform by the node's world matrix
                    wm = n.world_matrix
                    transformed = (wm[:3, :3] @ pos3.T).T + wm[:3, 3]
                    all_positions.append(transformed)

                    tris = np.asarray(idx, dtype=np.int32).reshape(-1, 3) + vertex_offset
                    all_triangles.append(tris)
                    vertex_offset += len(pos3)

        node.traverse(_collect)

        if all_positions and all_triangles:
            combined_pos = np.concatenate(all_positions, axis=0)
            combined_tris = np.concatenate(all_triangles, axis=0)
            color = GROUP_COLORS.get(display_name, (180, 180, 180))
            groups.append(MeshGroup(
                name=display_name,
                positions=combined_pos,
                triangles=combined_tris,
                color=color,
            ))

    return groups


def run_visual_test(
    pose_names: list[str] | None = None,
    quick: bool = False,
    output_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """Run visual head rotation tests.

    Parameters
    ----------
    pose_names : list of pose names from HEAD_POSES, or None for all
    quick : if True, use fewer poses and views
    output_dir : output directory (default: results/head_rotation)

    Returns
    -------
    dict mapping pose_name → {view_name → Path}
    """
    from tools.headless_loader import load_headless_scene, apply_head_rotation

    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if pose_names is None:
        if quick:
            pose_names = ["neutral", "yaw_left", "combined"]
        else:
            pose_names = list(HEAD_POSES.keys())

    views = {k: HEAD_CAMERA_PRESETS[k] for k in (HEAD_QUICK_VIEWS if quick else HEAD_CAMERA_PRESETS)}
    subsample = 2 if quick else 1

    print("Loading headless scene...")
    t0 = time.time()
    hs = load_headless_scene()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Update scene matrices
    hs.scene.update()

    all_results: dict[str, dict[str, Path]] = {}

    for pose_name in pose_names:
        pose_values = HEAD_POSES.get(pose_name)
        if pose_values is None:
            print(f"  Unknown pose: {pose_name}, skipping")
            continue

        print(f"\nPose: {pose_name}")

        # Create face state
        face_state = FaceState()
        for key, val in pose_values.items():
            setattr(face_state, key, val)

        # Apply head rotation
        t1 = time.time()
        head_q = apply_head_rotation(hs, face_state)
        hs.scene.update()
        print(f"  Applied in {time.time() - t1:.3f}s")

        # Extract mesh groups (with world transforms)
        groups = _extract_mesh_groups(hs)
        print(f"  Extracted {len(groups)} groups, "
              f"{sum(len(g.triangles) for g in groups)} total triangles")

        # Render multi-view
        t2 = time.time()
        paths = render_head_multiview(
            groups,
            output_dir=output_dir,
            prefix=pose_name,
            views=views,
            subsample=subsample,
        )
        print(f"  Rendered {len(paths)} views in {time.time() - t2:.1f}s")
        for vname, fpath in paths.items():
            print(f"    {vname}: {fpath}")

        all_results[pose_name] = paths

        # Reset for next pose
        if hs.pipeline.head_rotation is not None:
            hs.pipeline.head_rotation.reset()
        if hs.pipeline.neck_muscles is not None:
            hs.pipeline.neck_muscles.reset()
        for gname in ["skullGroup", "faceGroup", "stlMuscleGroup",
                       "exprMuscleGroup", "faceFeatureGroup"]:
            node = hs.named_nodes.get(gname)
            if node is not None:
                from faceforge.core.math_utils import quat_identity
                node.set_quaternion(quat_identity())
                node.set_position(0, 0, 0)
                node.mark_dirty()
        hs.scene.update()

    print(f"\nAll results saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Visual head rotation test")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer poses/views)")
    parser.add_argument("--poses", nargs="*", help="Specific poses to test")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output) if args.output else None
    run_visual_test(
        pose_names=args.poses,
        quick=args.quick,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
