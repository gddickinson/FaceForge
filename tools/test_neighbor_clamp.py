"""Test neighbor-stretch clamping and diagnostic.

Shows the effect of the clamping system on arm raise pose, and runs
the new check_neighbor_stretch() diagnostic.
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer, apply_pose
from tools.skinning_scorer import get_all_poses, _make_body_state
from faceforge.body.diagnostics import SkinningDiagnostic
from faceforge.core.state import BodyState


BODY_REGIONS = {
    "head":       (-15.0, float("inf")),
    "neck":       (-25.0, -15.0),
    "upper_torso":(-45.0, -25.0),
    "lower_torso":(-80.0, -45.0),
    "pelvis":     (-95.0, -80.0),
    "upper_leg":  (-140.0, -95.0),
    "lower_leg":  (-190.0, -140.0),
    "foot":       (float("-inf"), -190.0),
}


def main():
    print("Loading scene...")
    hs = load_headless_scene()
    print("Loading skin...")
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")

    # Verify neighbor data was built
    binding = hs.skinning.bindings[0]
    print(f"\nNeighbor data built:")
    print(f"  edge_pairs: {binding.edge_pairs.shape if binding.edge_pairs is not None else 'None'}")
    print(f"  neighbor_counts: min={binding.neighbor_counts.min()}, "
          f"max={binding.neighbor_counts.max()}, "
          f"mean={binding.neighbor_counts.mean():.1f}")
    print(f"  rest_neighbor_dist: min={binding.rest_neighbor_dist.min():.3f}, "
          f"max={binding.rest_neighbor_dist.max():.3f}, "
          f"mean={binding.rest_neighbor_dist.mean():.3f}")

    # Test with extreme_arm_raise
    poses = get_all_poses(["extreme_arm_raise", "sitting"])

    for pose_name in ["extreme_arm_raise", "sitting"]:
        print(f"\n{'='*60}")
        print(f"POSE: {pose_name}")
        print(f"{'='*60}")

        state = _make_body_state(poses[pose_name])

        # First apply WITHOUT clamping to see raw deformation
        hs.skinning.MAX_NEIGHBOR_STRETCH = float('inf')  # disable clamping
        apply_pose(hs, state)

        mesh = binding.mesh
        rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        current_unclamped = mesh.geometry.positions.reshape(-1, 3).astype(np.float64).copy()
        disps_unclamped = np.linalg.norm(current_unclamped - rest, axis=1)

        # Compute stretch ratios (unclamped)
        edges = binding.edge_pairs
        counts = binding.neighbor_counts
        rest_dist = binding.rest_neighbor_dist
        V = len(rest)

        neighbor_sum = np.zeros((V, 3), dtype=np.float64)
        np.add.at(neighbor_sum, edges[:, 0], current_unclamped[edges[:, 1]])
        np.add.at(neighbor_sum, edges[:, 1], current_unclamped[edges[:, 0]])
        has_neighbors = counts > 0
        neighbor_avg = np.zeros_like(neighbor_sum)
        neighbor_avg[has_neighbors] = neighbor_sum[has_neighbors] / counts[has_neighbors, np.newaxis]
        current_dist = np.linalg.norm(current_unclamped - neighbor_avg, axis=1)
        stretch_unclamped = current_dist / (rest_dist + 0.01)

        # Now apply WITH clamping
        hs.skinning.MAX_NEIGHBOR_STRETCH = 3.0  # enable clamping
        apply_pose(hs, state)

        current_clamped = mesh.geometry.positions.reshape(-1, 3).astype(np.float64).copy()
        disps_clamped = np.linalg.norm(current_clamped - rest, axis=1)

        # Recompute stretch (clamped)
        neighbor_sum_c = np.zeros((V, 3), dtype=np.float64)
        np.add.at(neighbor_sum_c, edges[:, 0], current_clamped[edges[:, 1]])
        np.add.at(neighbor_sum_c, edges[:, 1], current_clamped[edges[:, 0]])
        neighbor_avg_c = np.zeros_like(neighbor_sum_c)
        neighbor_avg_c[has_neighbors] = neighbor_sum_c[has_neighbors] / counts[has_neighbors, np.newaxis]
        current_dist_c = np.linalg.norm(current_clamped - neighbor_avg_c, axis=1)
        stretch_clamped = current_dist_c / (rest_dist + 0.01)

        # Count clamped vertices
        clamped_mask = np.linalg.norm(current_unclamped - current_clamped, axis=1) > 0.01
        n_clamped = int(np.sum(clamped_mask))
        print(f"\nVertices clamped: {n_clamped}")

        # Region breakdown of stretch outliers (before clamping)
        over_3x = has_neighbors & (stretch_unclamped > 3.0)
        n_over = int(np.sum(over_3x))
        print(f"Vertices with stretch > 3.0x (before clamp): {n_over}")
        if n_over > 0:
            over_rest = rest[over_3x]
            over_stretch = stretch_unclamped[over_3x]
            print(f"  max stretch: {over_stretch.max():.1f}x")
            for rname, (zlo, zhi) in BODY_REGIONS.items():
                rmask = (over_rest[:, 2] >= zlo) & (over_rest[:, 2] < zhi)
                n = int(np.sum(rmask))
                if n > 0:
                    print(f"  {rname:15s}: {n} vertices (max stretch {over_stretch[rmask].max():.1f}x)")

        # Compare displacements in hip region (|X|<20)
        print(f"\nHip region displacement comparison (|X|<20):")
        for rname, (zlo, zhi) in [("lower_torso", (-80, -45)), ("pelvis", (-95, -80)),
                                    ("upper_leg", (-140, -95))]:
            mask = (rest[:, 2] >= zlo) & (rest[:, 2] < zhi) & (np.abs(rest[:, 0]) < 20)
            if not np.any(mask):
                continue
            n = int(np.sum(mask))
            d_unc = disps_unclamped[mask]
            d_cla = disps_clamped[mask]
            moved_unc = d_unc > 2.0
            moved_cla = d_cla > 2.0
            print(f"  {rname:15s}: V={n:6d}  "
                  f"moved>2 unclamped={int(np.sum(moved_unc)):5d}  "
                  f"moved>2 clamped={int(np.sum(moved_cla)):5d}  "
                  f"max_disp_unc={d_unc.max():.1f}  max_disp_cla={d_cla.max():.1f}")

        # Run neighbor stretch diagnostic
        diag = SkinningDiagnostic(hs.skinning)
        stretch_anomalies = diag.check_neighbor_stretch(max_stretch=3.0)
        if stretch_anomalies:
            for sa in stretch_anomalies:
                print(f"\nNeighbor Stretch Diagnostic:")
                print(f"  {sa.mesh_name}: {sa.stretched_count}/{sa.vertex_count} "
                      f"({100*sa.stretched_count/sa.vertex_count:.2f}%) "
                      f"max_ratio={sa.max_stretch_ratio:.1f}x")
                if sa.region_breakdown:
                    for rn, cnt in sorted(sa.region_breakdown.items(), key=lambda x: -x[1]):
                        print(f"    {rn}: {cnt}")
                for jn, cnt in sorted(sa.joint_breakdown.items(), key=lambda x: -x[1])[:10]:
                    print(f"    joint={jn}: {cnt}")
        else:
            print(f"\nNeighbor Stretch Diagnostic: No anomalies (all within 3.0x)")

    # Reset
    hs.skinning.MAX_NEIGHBOR_STRETCH = 3.0
    apply_pose(hs, BodyState())


if __name__ == "__main__":
    main()
