"""Visual skinning test runner.

Loads the body mesh headlessly, applies various poses, renders from
multiple camera angles with edge-stretch coloring, and saves images
to the results/ folder.  Provides both visual output (PNG images) and
quantitative metrics for automated analysis.

Usage::

    # Full test: all poses, key views
    python -m tools.visual_skinning_test

    # Quick test: 2 poses, 2 views, subsampled
    python -m tools.visual_skinning_test --quick

    # Specific poses
    python -m tools.visual_skinning_test --poses sitting arm_raise

    # All views
    python -m tools.visual_skinning_test --all-views
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from tools.headless_loader import (
    load_headless_scene,
    load_layer,
    register_layer,
    apply_pose,
)
from tools.skinning_scorer import get_all_poses, _make_body_state
from tools.mesh_renderer import (
    render_mesh,
    render_multiview,
    compute_triangle_stretch,
    CAMERA_PRESETS,
    QUICK_VIEWS,
)
from faceforge.core.state import BodyState

RESULTS_DIR = Path("results")

# Standard test poses
STANDARD_POSES = {
    "anatomical": {},
    "sitting": None,       # loaded from body_poses.json
    "walking": None,
    "reaching": None,
    "crouching": None,
    "arm_raise": {
        "shoulder_r_abduct": 1.0,
        "shoulder_l_abduct": 0.3,
    },
    "arm_raise_full": {
        "shoulder_r_abduct": 1.0,
        "shoulder_l_abduct": 1.0,
        "elbow_r_flex": 0.3,
        "elbow_l_flex": 0.3,
    },
}

QUICK_POSES = ["sitting", "arm_raise"]


def _load_pose_values() -> dict[str, dict]:
    """Load pose values, filling in presets from body_poses.json."""
    file_poses = get_all_poses(["sitting", "walking", "reaching", "crouching"])
    result = {}
    for name, vals in STANDARD_POSES.items():
        if vals is None:
            result[name] = file_poses.get(name, {})
        else:
            result[name] = vals
    return result


def run_visual_test(
    pose_names: list[str] | None = None,
    view_names: list[str] | None = None,
    output_dir: Path | None = None,
    subsample: int = 1,
    width: int = 800,
    height: int = 1000,
    verbose: bool = True,
) -> dict:
    """Run the visual skinning test.

    Returns a dict of per-pose, per-view metrics.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_poses = _load_pose_values()
    if pose_names is None:
        pose_names = list(all_poses.keys())
    else:
        pose_names = [p for p in pose_names if p in all_poses]

    if view_names is None:
        view_names = QUICK_VIEWS
    views = {k: CAMERA_PRESETS[k] for k in view_names if k in CAMERA_PRESETS}

    if verbose:
        print("Loading scene...")
    t0 = time.time()
    hs = load_headless_scene()
    if verbose:
        print(f"  Scene loaded in {time.time() - t0:.1f}s")

    if verbose:
        print("Loading skin...")
    t1 = time.time()
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")
    if verbose:
        print(f"  Skin loaded in {time.time() - t1:.1f}s")

    binding = hs.skinning.bindings[0]
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    triangles = mesh.geometry.indices.reshape(-1, 3)

    if verbose:
        print(f"  Mesh: {len(rest)} vertices, {len(triangles)} triangles")
        print(f"  Rendering {len(pose_names)} poses x {len(views)} views "
              f"(subsample={subsample})")

    results = {}

    for pose_name in pose_names:
        pose_vals = all_poses[pose_name]
        state = _make_body_state(pose_vals)

        if verbose:
            print(f"\n  Pose: {pose_name}...")
        apply_pose(hs, state)

        positions = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)

        # Compute stretch metrics
        stretch = compute_triangle_stretch(positions, rest, triangles)
        metrics = {
            "mean_stretch": float(stretch.mean()),
            "p95_stretch": float(np.percentile(stretch, 95)),
            "max_stretch": float(stretch.max()),
            "tris_over_1_5": int(np.sum(stretch > 1.5)),
            "tris_over_2": int(np.sum(stretch > 2.0)),
            "tris_over_3": int(np.sum(stretch > 3.0)),
            "pct_over_1_5": float(np.sum(stretch > 1.5) / len(stretch) * 100),
        }
        results[pose_name] = {"metrics": metrics, "views": {}}

        if verbose:
            print(f"    stretch: mean={metrics['mean_stretch']:.3f} "
                  f"p95={metrics['p95_stretch']:.3f} "
                  f"max={metrics['max_stretch']:.1f} "
                  f"| >1.5x:{metrics['tris_over_1_5']} "
                  f">2x:{metrics['tris_over_2']} "
                  f">3x:{metrics['tris_over_3']}")

        # Render views
        for vname, vparams in views.items():
            fname = f"{pose_name}_{vname}.png"
            fpath = output_dir / fname
            t_render = time.time()
            render_mesh(
                positions, rest, triangles,
                azimuth=vparams["azimuth"],
                elevation=vparams.get("elevation", 5),
                width=width, height=height,
                output_path=fpath,
                title=f"{pose_name} - {vname}",
                subsample=subsample,
            )
            dt = time.time() - t_render
            results[pose_name]["views"][vname] = str(fpath)
            if verbose:
                print(f"    {vname}: {fpath.name} ({dt:.1f}s)")

    # Reset to anatomical
    apply_pose(hs, BodyState())

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Pose':<20s} {'mean':>6s} {'p95':>6s} {'max':>6s} "
              f"{'> 1.5x':>8s} {'> 2x':>8s} {'> 3x':>8s}")
        print("-" * 60)
        for pname in pose_names:
            m = results[pname]["metrics"]
            print(f"{pname:<20s} {m['mean_stretch']:6.3f} {m['p95_stretch']:6.3f} "
                  f"{m['max_stretch']:6.1f} {m['tris_over_1_5']:8d} "
                  f"{m['tris_over_2']:8d} {m['tris_over_3']:8d}")

        print(f"\nImages saved to: {output_dir.resolve()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visual skinning test runner",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 2 poses, 2 views, subsampled 3x",
    )
    parser.add_argument(
        "--poses", nargs="+", default=None,
        help="Specific poses to render (default: all standard)",
    )
    parser.add_argument(
        "--views", nargs="+", default=None,
        help="Specific camera views (default: front, right, 3q_front_r)",
    )
    parser.add_argument(
        "--all-views", action="store_true",
        help="Render from all camera presets",
    )
    parser.add_argument(
        "--subsample", type=int, default=1,
        help="Render every Nth triangle (default: 1 = all)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--width", type=int, default=800,
        help="Image width (default: 800)",
    )
    parser.add_argument(
        "--height", type=int, default=1000,
        help="Image height (default: 1000)",
    )

    args = parser.parse_args()

    pose_names = args.poses
    view_names = args.views
    subsample = args.subsample
    output_dir = Path(args.output) if args.output else RESULTS_DIR

    if args.quick:
        pose_names = pose_names or QUICK_POSES
        view_names = view_names or ["front", "3q_front_r"]
        subsample = max(subsample, 3)

    if args.all_views:
        view_names = list(CAMERA_PRESETS.keys())

    run_visual_test(
        pose_names=pose_names,
        view_names=view_names,
        output_dir=output_dir,
        subsample=subsample,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
