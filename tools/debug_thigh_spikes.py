"""Diagnose the spiky skin distortions visible on outer thighs in sitting pose.

Focuses on vertices in the thigh region (Z=-95 to -140) that have high
neighbor stretch ratios, to identify exactly which chain/joint assignments
are causing the visible spikes.
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer, apply_pose
from tools.skinning_scorer import get_all_poses, _make_body_state
from faceforge.core.state import BodyState


def main():
    print("Loading scene...")
    hs = load_headless_scene()
    print("Loading skin...")
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")

    joints = hs.skinning.joints
    chain_id_to_name = {v: k for k, v in hs.chain_ids.items()}

    # Apply sitting pose
    poses = get_all_poses(["sitting"])
    state = _make_body_state(poses["sitting"])

    # Disable clamp to see raw distortions
    hs.skinning.MAX_NEIGHBOR_STRETCH = float('inf')
    apply_pose(hs, state)

    binding = hs.skinning.bindings[0]
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    V = len(rest)
    ji = binding.joint_indices
    edges = binding.edge_pairs
    counts = binding.neighbor_counts
    rest_ndist = binding.rest_neighbor_dist

    # Compute stretch ratios
    neighbor_sum = np.zeros((V, 3), dtype=np.float64)
    np.add.at(neighbor_sum, edges[:, 0], current[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], current[edges[:, 0]])
    has_neighbors = counts > 0
    neighbor_avg = np.zeros_like(neighbor_sum)
    neighbor_avg[has_neighbors] = neighbor_sum[has_neighbors] / counts[has_neighbors, np.newaxis]
    current_dist = np.linalg.norm(current - neighbor_avg, axis=1)
    stretch = current_dist / (rest_ndist + 0.01)

    # Also compute per-vertex displacement
    disps = np.linalg.norm(current - rest, axis=1)

    # Focus on thigh/pelvis region with high stretch
    for region_name, zlo, zhi in [
        ("pelvis", -95, -80),
        ("upper_leg", -140, -95),
        ("lower_leg", -190, -140),
    ]:
        mask = (rest[:, 2] >= zlo) & (rest[:, 2] < zhi)
        if not np.any(mask):
            continue

        region_stretch = stretch[mask]
        meaningful = rest_ndist[mask] > 0.05
        high_stretch = meaningful & (region_stretch > 2.0)
        n_high = int(np.sum(high_stretch))

        print(f"\n{'='*60}")
        print(f"Region: {region_name} (Z=[{zlo},{zhi}])")
        print(f"  Total verts: {int(np.sum(mask))}")
        print(f"  High stretch (>2.0x): {n_high}")

        if n_high == 0:
            continue

        # Get the high-stretch vertices
        region_idx = np.where(mask)[0]
        high_idx = region_idx[high_stretch]
        high_rest = rest[high_idx]
        high_current = current[high_idx]
        high_stretch_vals = stretch[high_idx]
        high_disps = disps[high_idx]
        high_ji = ji[high_idx]

        # Breakdown by stretch range
        for lo_s, hi_s in [(2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 50.0), (50.0, float('inf'))]:
            s_mask = (high_stretch_vals >= lo_s) & (high_stretch_vals < hi_s)
            n = int(np.sum(s_mask))
            if n > 0:
                print(f"  stretch [{lo_s:.0f}x, {hi_s:.0f}x): {n}")

        # Joint/chain breakdown for stretch > 3x
        very_high = high_stretch_vals > 3.0
        if np.any(very_high):
            vh_ji = high_ji[very_high]
            vh_rest = high_rest[very_high]
            vh_stretch = high_stretch_vals[very_high]
            vh_disp = high_disps[very_high]

            print(f"\n  Stretch > 3.0x: {int(np.sum(very_high))} verts")
            jcounts = {}
            for j_idx in vh_ji:
                jname = joints[j_idx].name
                cname = chain_id_to_name.get(joints[j_idx].chain_id, "?")
                key = f"{jname} ({cname})"
                jcounts[key] = jcounts.get(key, 0) + 1
            for key, cnt in sorted(jcounts.items(), key=lambda x: -x[1])[:15]:
                print(f"    {key:40s}: {cnt}")

            # X distribution
            print(f"\n  X distribution of stretch>3x verts:")
            for xlo, xhi in [(-50,-30),(-30,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,30),(30,50)]:
                n = int(np.sum((vh_rest[:, 0] >= xlo) & (vh_rest[:, 0] < xhi)))
                if n > 0:
                    avg_s = float(vh_stretch[(vh_rest[:, 0] >= xlo) & (vh_rest[:, 0] < xhi)].mean())
                    print(f"    X=[{xlo:4d},{xhi:4d}): {n:5d} (avg stretch {avg_s:.1f}x)")

            # Sample worst vertices
            worst_order = np.argsort(-vh_stretch)[:20]
            print(f"\n  Worst 20 vertices:")
            for i in worst_order:
                v = vh_rest[i]
                c = current[np.where(mask)[0][high_stretch][i]]  # current pos
                s = vh_stretch[i]
                d = vh_disp[i]
                jname = joints[vh_ji[i]].name
                cname = chain_id_to_name.get(joints[vh_ji[i]].chain_id, "?")
                # Direction of distortion
                delta = c - v
                print(f"    rest=({v[0]:6.1f},{v[1]:6.1f},{v[2]:6.1f}) "
                      f"delta=({delta[0]:6.1f},{delta[1]:6.1f},{delta[2]:6.1f}) "
                      f"stretch={s:6.1f}x disp={d:5.1f} {jname}({cname})")

    # Now test different clamp thresholds
    print(f"\n{'='*60}")
    print("CLAMP THRESHOLD COMPARISON")
    print(f"{'='*60}")

    for threshold in [3.0, 2.5, 2.0, 1.5]:
        hs.skinning.MAX_NEIGHBOR_STRETCH = threshold
        apply_pose(hs, state)

        current_t = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
        disps_t = np.linalg.norm(current_t - rest, axis=1)

        # Recompute stretch after clamping
        ns = np.zeros((V, 3), dtype=np.float64)
        np.add.at(ns, edges[:, 0], current_t[edges[:, 1]])
        np.add.at(ns, edges[:, 1], current_t[edges[:, 0]])
        na = np.zeros_like(ns)
        na[has_neighbors] = ns[has_neighbors] / counts[has_neighbors, np.newaxis]
        cd = np.linalg.norm(current_t - na, axis=1)
        st = cd / (rest_ndist + 0.01)

        meaningful = rest_ndist > 0.05
        over_3 = has_neighbors & meaningful & (st > 3.0)
        over_2 = has_neighbors & meaningful & (st > 2.0)

        # Check thigh region max displacement delta (unclamped vs clamped)
        thigh_mask = (rest[:, 2] >= -140) & (rest[:, 2] < -95) & (np.abs(rest[:, 0]) < 30)
        thigh_max_disp = disps_t[thigh_mask].max() if np.any(thigh_mask) else 0

        print(f"\n  Threshold={threshold:.1f}x:")
        print(f"    Remaining >3x: {int(np.sum(over_3))}, >2x: {int(np.sum(over_2))}")
        print(f"    Upper leg (|X|<30) max disp: {thigh_max_disp:.1f}")

    # Reset
    hs.skinning.MAX_NEIGHBOR_STRETCH = 3.0
    apply_pose(hs, BodyState())


if __name__ == "__main__":
    main()
