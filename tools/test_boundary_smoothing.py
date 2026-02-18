"""Test the effect of boundary displacement smoothing on cross-chain seams.

Compares cross-chain edge displacement discontinuity with smoothing ON vs OFF
across multiple poses. This measures whether the smoothing is actually reducing
the visible seams at chain boundaries.
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer, apply_pose
from tools.skinning_scorer import get_all_poses, _make_body_state
from faceforge.core.state import BodyState


def measure_cross_chain_discontinuity(hs, binding):
    """Measure displacement discontinuity at cross-chain edges."""
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    edges = binding.edge_pairs
    if edges is None:
        return {}

    disp = current - rest

    # Classify edges as same-chain or cross-chain
    ji = binding.joint_indices
    vert_chains = np.array(
        [hs.skinning.joints[idx].chain_id for idx in ji], dtype=np.int32
    )
    e0_chain = vert_chains[edges[:, 0]]
    e1_chain = vert_chains[edges[:, 1]]
    cross = e0_chain != e1_chain
    same = ~cross

    # Edge displacement difference: how different are the displacements of
    # connected vertices?
    edge_disp_diff = np.linalg.norm(disp[edges[:, 0]] - disp[edges[:, 1]], axis=1)

    result = {}
    if np.any(same):
        sd = edge_disp_diff[same]
        result["same_chain"] = {
            "count": int(np.sum(same)),
            "mean": float(sd.mean()),
            "p95": float(np.percentile(sd, 95)),
            "max": float(sd.max()),
        }
    if np.any(cross):
        cd = edge_disp_diff[cross]
        result["cross_chain"] = {
            "count": int(np.sum(cross)),
            "mean": float(cd.mean()),
            "p95": float(np.percentile(cd, 95)),
            "max": float(cd.max()),
        }

    # Neighbor stretch stats
    V = len(rest)
    counts = binding.neighbor_counts
    has_neighbors = counts > 0
    neighbor_sum = np.zeros((V, 3), dtype=np.float64)
    np.add.at(neighbor_sum, edges[:, 0], current[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], current[edges[:, 0]])
    neighbor_avg = np.zeros_like(neighbor_sum)
    neighbor_avg[has_neighbors] = neighbor_sum[has_neighbors] / counts[has_neighbors, np.newaxis]
    cur_dist = np.linalg.norm(current - neighbor_avg, axis=1)
    rest_dist = binding.rest_neighbor_dist
    stretch = cur_dist / (rest_dist + 0.01)
    meaningful = (rest_dist > 0.05) & has_neighbors
    result["stretch"] = {
        "mean": float(stretch[meaningful].mean()),
        "p95": float(np.percentile(stretch[meaningful], 95)),
        "max": float(stretch[meaningful].max()),
        "over_2x": int(np.sum(meaningful & (stretch > 2.0))),
        "over_3x": int(np.sum(meaningful & (stretch > 3.0))),
    }

    return result


def main():
    print("Loading scene...")
    hs = load_headless_scene()
    print("Loading skin...")
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")

    binding = hs.skinning.bindings[0]

    # Report smooth_zone stats
    sz = binding.smooth_zone
    if sz is not None:
        active = sz > 0.01
        print(f"\nSmooth zone stats:")
        print(f"  Total vertices: {len(sz)}")
        print(f"  Active (zone > 0.01): {int(np.sum(active))} ({100*np.sum(active)/len(sz):.1f}%)")
        print(f"  Boundary blend stats: pure interior={int(np.sum(binding.boundary_blend > 0.99))}, "
              f"mixed={int(np.sum((binding.boundary_blend > 0.01) & (binding.boundary_blend <= 0.99)))}, "
              f"pure boundary={int(np.sum(binding.boundary_blend <= 0.01))}")
        print(f"  Zone range: [{sz.min():.3f}, {sz.max():.3f}]")
        print(f"  Zone mean (active only): {sz[active].mean():.3f}")

    poses_dict = get_all_poses(["sitting", "reaching", "crouching", "walking"])
    # Add arm raise
    poses_dict["arm_raise"] = {
        "shoulder_r_abduct": 1.0,
        "shoulder_l_abduct": 0.3,
    }

    for pose_name, pose_vals in poses_dict.items():
        state = _make_body_state(pose_vals)

        # Test with smoothing OFF
        saved_passes = hs.skinning.BOUNDARY_SMOOTH_PASSES
        hs.skinning.BOUNDARY_SMOOTH_PASSES = 0  # disable
        apply_pose(hs, state)
        no_smooth = measure_cross_chain_discontinuity(hs, binding)

        # Test with smoothing ON
        hs.skinning.BOUNDARY_SMOOTH_PASSES = saved_passes
        apply_pose(hs, state)
        with_smooth = measure_cross_chain_discontinuity(hs, binding)

        print(f"\n{'='*70}")
        print(f"Pose: {pose_name}")
        print(f"{'='*70}")

        # Cross-chain comparison
        if "cross_chain" in no_smooth and "cross_chain" in with_smooth:
            ns = no_smooth["cross_chain"]
            ws = with_smooth["cross_chain"]
            print(f"  Cross-chain edge disp diff:")
            print(f"    {'':15s} {'No smooth':>12s} {'Smoothed':>12s} {'Change':>12s}")
            for key in ["mean", "p95", "max"]:
                nv = ns[key]
                sv = ws[key]
                pct = (sv - nv) / max(nv, 0.001) * 100
                print(f"    {key:15s} {nv:12.2f} {sv:12.2f} {pct:+11.1f}%")

        # Same-chain comparison
        if "same_chain" in no_smooth and "same_chain" in with_smooth:
            ns = no_smooth["same_chain"]
            ws = with_smooth["same_chain"]
            print(f"  Same-chain edge disp diff:")
            print(f"    {'':15s} {'No smooth':>12s} {'Smoothed':>12s} {'Change':>12s}")
            for key in ["mean", "p95", "max"]:
                nv = ns[key]
                sv = ws[key]
                pct = (sv - nv) / max(nv, 0.001) * 100
                print(f"    {key:15s} {nv:12.2f} {sv:12.2f} {pct:+11.1f}%")

        # Stretch comparison
        if "stretch" in no_smooth and "stretch" in with_smooth:
            ns = no_smooth["stretch"]
            ws = with_smooth["stretch"]
            print(f"  Neighbor stretch:")
            print(f"    {'':15s} {'No smooth':>12s} {'Smoothed':>12s}")
            for key in ["mean", "p95", "max", "over_2x", "over_3x"]:
                nv = ns[key]
                sv = ws[key]
                if isinstance(nv, int):
                    print(f"    {key:15s} {nv:12d} {sv:12d}")
                else:
                    print(f"    {key:15s} {nv:12.2f} {sv:12.2f}")

    # Reset
    apply_pose(hs, BodyState())
    print("\nDone.")


if __name__ == "__main__":
    main()
