"""Comprehensive stuck-vertex detection with multiple strategies.

Runs 4 different detection approaches to find vertices that don't move
when they should, regardless of which joint they're bound to:

1. ASSIGNED-JOINT: vertex vs its assigned joint (existing approach)
2. NEAREST-JOINT: vertex vs the nearest joint that actually moved
3. REGION-BASED: groups vertices by body region, flags regions with low movement
4. NEIGHBOR-CONTRAST: compares vertex displacement to its mesh neighbors

Usage::

    python -m tools.detect_stuck --poses sitting extreme_crouch
    python -m tools.detect_stuck --strategy all --threshold 0.5
"""

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer, apply_pose
from tools.skinning_scorer import get_all_poses, _make_body_state
from faceforge.core.state import BodyState


# ── Body region definitions (rest-pose world Z coordinates) ──────────
# Z = vertical axis in the Python coordinate system (BP3D-derived)
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


def _get_region(z: float) -> str:
    """Map a Z coordinate to a body region name."""
    for name, (lo, hi) in BODY_REGIONS.items():
        if lo <= z < hi:
            return name
    return "unknown"


def strategy_assigned_joint(hs, binding, joint_disps, joints):
    """Strategy 1: Check vertex vs its ASSIGNED joint displacement.

    This is the existing approach — if the assigned joint moved significantly
    but the vertex barely moved, it's stuck.
    """
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    V = len(rest)

    vert_disps = np.linalg.norm(current - rest, axis=1)
    ji = binding.joint_indices
    vert_joint_disps = joint_disps[ji]

    THRESH = 5.0
    RATIO = 0.1
    static_mask = (vert_joint_disps > THRESH) & (vert_disps < RATIO * vert_joint_disps)
    return static_mask, vert_disps, rest


def strategy_nearest_joint(hs, binding, joint_disps, joints):
    """Strategy 2: Check vertex vs the NEAREST joint that moved.

    For each vertex, find the closest joint (by rest position) that displaced
    significantly.  If that joint moved > 5 units but the vertex moved < 10%
    of it, the vertex is stuck — even if it's assigned to a different joint.
    """
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    V = len(rest)

    vert_disps = np.linalg.norm(current - rest, axis=1)

    # Get joint rest positions
    joint_rest_pos = np.array([j.rest_world[:3, 3] for j in joints], dtype=np.float64)

    # For each vertex, find the nearest joint that moved > 5 units
    moved_mask = joint_disps > 5.0
    if not np.any(moved_mask):
        return np.zeros(V, dtype=bool), vert_disps, rest

    moved_indices = np.where(moved_mask)[0]
    moved_positions = joint_rest_pos[moved_indices]
    moved_disps = joint_disps[moved_indices]

    # Compute distance from each vertex to each moved joint
    # (V, J_moved) matrix
    diff = rest[:, np.newaxis, :] - moved_positions[np.newaxis, :, :]  # (V, J, 3)
    dist_to_joints = np.sqrt(np.sum(diff * diff, axis=2))  # (V, J)

    # For each vertex, find nearest moved joint
    nearest_idx = np.argmin(dist_to_joints, axis=1)  # (V,)
    nearest_joint_disp = moved_disps[nearest_idx]  # (V,)
    nearest_dist = dist_to_joints[np.arange(V), nearest_idx]  # (V,)

    # Only consider vertices within 30 units of a moved joint
    DIST_LIMIT = 30.0
    RATIO = 0.1
    static_mask = (
        (nearest_dist < DIST_LIMIT) &
        (nearest_joint_disp > 5.0) &
        (vert_disps < RATIO * nearest_joint_disp)
    )
    return static_mask, vert_disps, rest


def strategy_region_based(hs, binding, joint_disps, joints):
    """Strategy 3: Group vertices by body region, flag underperforming regions.

    Computes the median displacement per body region.  If a region's median
    displacement is < 20% of its expected displacement (from nearest moving
    joint), flag all low-displacement vertices in that region.
    """
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    V = len(rest)

    vert_disps = np.linalg.norm(current - rest, axis=1)

    # Assign each vertex to a body region by rest Z coordinate
    vert_z = rest[:, 2]
    region_ids = np.zeros(V, dtype=np.int32)
    region_names = list(BODY_REGIONS.keys())
    for idx, (name, (lo, hi)) in enumerate(BODY_REGIONS.items()):
        mask = (vert_z >= lo) & (vert_z < hi)
        region_ids[mask] = idx

    # Get joint displacements per region
    joint_rest_pos = np.array([j.rest_world[:3, 3] for j in joints], dtype=np.float64)

    static_mask = np.zeros(V, dtype=bool)

    for idx, name in enumerate(region_names):
        rmask = region_ids == idx
        if not np.any(rmask):
            continue
        region_verts = rest[rmask]
        region_disps = vert_disps[rmask]

        # Expected movement: max joint displacement of joints in this Z range
        lo, hi = list(BODY_REGIONS.values())[idx]
        joint_z = joint_rest_pos[:, 2]
        # Joints in or near this region
        margin = 20.0
        j_in_region = (joint_z >= lo - margin) & (joint_z < hi + margin)
        if not np.any(j_in_region):
            continue
        expected_disp = float(joint_disps[j_in_region].max())
        if expected_disp < 2.0:
            continue

        median_disp = float(np.median(region_disps))

        # Flag if median displacement is < 20% of expected
        if median_disp < 0.2 * expected_disp:
            # Flag individual verts that are below 10% of expected
            low_verts = region_disps < 0.1 * expected_disp
            # Map back to full mesh indices
            full_indices = np.where(rmask)[0]
            static_mask[full_indices[low_verts]] = True

    return static_mask, vert_disps, rest


def strategy_neighbor_contrast(hs, binding, joint_disps, joints):
    """Strategy 4: Compare vertex displacement to its mesh neighbors.

    Uses the triangle connectivity to find each vertex's neighbors.
    If a vertex moved much less than its neighbors' average, it's stuck.
    """
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    V = len(rest)

    vert_disps = np.linalg.norm(current - rest, axis=1)

    # Build adjacency from index buffer
    indices = mesh.geometry.indices
    if indices is None or len(indices) < 3:
        return np.zeros(V, dtype=bool), vert_disps, rest

    # Build neighbor list
    neighbor_sum = np.zeros(V, dtype=np.float64)
    neighbor_count = np.zeros(V, dtype=np.int32)

    tri_indices = indices.reshape(-1, 3)
    for col in range(3):
        a = tri_indices[:, col]
        b = tri_indices[:, (col + 1) % 3]
        # a → b and b → a
        np.add.at(neighbor_sum, a, vert_disps[b])
        np.add.at(neighbor_count, a, 1)
        np.add.at(neighbor_sum, b, vert_disps[a])
        np.add.at(neighbor_count, b, 1)

    has_neighbors = neighbor_count > 0
    avg_neighbor_disp = np.where(
        has_neighbors,
        neighbor_sum / np.maximum(neighbor_count, 1),
        0.0
    )

    # Flag: vertex displaced < 10% of its neighbors' average, AND
    # neighbors moved significantly (> 2 units)
    RATIO = 0.1
    MIN_NEIGHBOR_DISP = 2.0
    static_mask = (
        has_neighbors &
        (avg_neighbor_disp > MIN_NEIGHBOR_DISP) &
        (vert_disps < RATIO * avg_neighbor_disp)
    )
    return static_mask, vert_disps, rest


STRATEGIES = {
    "assigned_joint": strategy_assigned_joint,
    "nearest_joint": strategy_nearest_joint,
    "region_based": strategy_region_based,
    "neighbor_contrast": strategy_neighbor_contrast,
}


def run_detection(layers=None, poses=None, strategies=None):
    """Run all detection strategies and print results."""
    if layers is None:
        layers = ["skin", "back_muscles", "leg_muscles", "hip_muscles"]
    if poses is None:
        poses = ["sitting", "extreme_crouch", "extreme_arm_raise"]
    if strategies is None:
        strategies = list(STRATEGIES.keys())

    print("Loading scene...")
    t0 = time.time()
    hs = load_headless_scene()
    print(f"  Scene loaded in {time.time() - t0:.1f}s")

    for name in layers:
        print(f"  Loading {name}...")
        meshes = load_layer(hs, name)
        register_layer(hs, meshes, name)

    all_poses = get_all_poses(poses)
    joints = hs.skinning.joints

    for pose_name, pose_dict in all_poses.items():
        state = _make_body_state(pose_dict)
        apply_pose(hs, state)

        # Pre-compute joint displacements
        joint_disps = np.zeros(len(joints), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            joint.node.update_world_matrix()
            cur = joint.node.world_matrix[:3, 3]
            rst = joint.rest_world[:3, 3]
            joint_disps[j_idx] = float(np.linalg.norm(cur - rst))

        print(f"\n{'='*70}")
        print(f"POSE: {pose_name}")
        print(f"{'='*70}")

        # Show joint displacement summary
        moved = joint_disps > 2.0
        print(f"  Joints that moved > 2 units: {int(np.sum(moved))}/{len(joints)}")
        if np.any(moved):
            top_indices = np.argsort(-joint_disps)[:10]
            for ji in top_indices:
                if joint_disps[ji] > 1.0:
                    print(f"    {joints[ji].name:30s}  disp={joint_disps[ji]:.1f}")

        for strat_name in strategies:
            strat_fn = STRATEGIES[strat_name]
            print(f"\n  --- Strategy: {strat_name} ---")

            total_flagged = 0
            total_verts = 0

            for binding in hs.skinning.bindings:
                mesh = binding.mesh
                if mesh.rest_positions is None:
                    continue

                static_mask, vert_disps, rest = strat_fn(
                    hs, binding, joint_disps, joints
                )
                V = len(rest)
                total_verts += V
                n_flagged = int(np.sum(static_mask))
                total_flagged += n_flagged

                if n_flagged > 0:
                    flagged_rest = rest[static_mask]
                    flagged_disps = vert_disps[static_mask]

                    # Region breakdown
                    region_counts = {}
                    for fv in range(n_flagged):
                        z = float(flagged_rest[fv, 2])
                        region = _get_region(z)
                        region_counts[region] = region_counts.get(region, 0) + 1

                    print(f"    {mesh.name:30s}  "
                          f"stuck={n_flagged:6d}/{V}  "
                          f"max_disp={float(flagged_disps.max()):.3f}  "
                          f"mean_disp={float(flagged_disps.mean()):.3f}")
                    for region, count in sorted(
                        region_counts.items(), key=lambda x: -x[1]
                    ):
                        print(f"      {region:20s}: {count}")

            if total_flagged == 0:
                print(f"    No stuck vertices detected ({total_verts} total)")
            else:
                print(f"    TOTAL: {total_flagged} stuck / {total_verts} total")

        # Displacement histogram per region (always shown)
        print(f"\n  --- Region Displacement Summary ---")
        all_rest = []
        all_disps = []
        for binding in hs.skinning.bindings:
            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue
            rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
            current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
            disps = np.linalg.norm(current - rest, axis=1)
            all_rest.append(rest)
            all_disps.append(disps)

        if all_rest:
            all_rest = np.concatenate(all_rest, axis=0)
            all_disps = np.concatenate(all_disps, axis=0)

            for name, (lo, hi) in BODY_REGIONS.items():
                mask = (all_rest[:, 2] >= lo) & (all_rest[:, 2] < hi)
                if not np.any(mask):
                    continue
                region_disps = all_disps[mask]
                n = int(np.sum(mask))
                zero_pct = 100.0 * np.sum(region_disps < 0.01) / n
                low_pct = 100.0 * np.sum(region_disps < 0.5) / n
                print(f"    {name:15s}: V={n:7d}  "
                      f"mean={float(region_disps.mean()):7.2f}  "
                      f"median={float(np.median(region_disps)):7.2f}  "
                      f"max={float(region_disps.max()):7.2f}  "
                      f"zero(<0.01)={zero_pct:5.1f}%  "
                      f"low(<0.5)={low_pct:5.1f}%")

    # Reset
    apply_pose(hs, BodyState())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-strategy stuck vertex detection")
    parser.add_argument("--layers", nargs="*",
                        default=["skin", "back_muscles", "leg_muscles", "hip_muscles"])
    parser.add_argument("--poses", nargs="*",
                        default=["sitting", "extreme_crouch"])
    parser.add_argument("--strategy", nargs="*", default=None,
                        help="Strategies to run (default: all)")
    args = parser.parse_args()

    strategies = args.strategy if args.strategy else list(STRATEGIES.keys())
    run_detection(layers=args.layers, poses=args.poses, strategies=strategies)


if __name__ == "__main__":
    main()
