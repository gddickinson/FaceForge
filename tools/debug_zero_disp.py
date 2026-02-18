"""Deep investigation of zero-displacement vertices in the skin mesh.

For vertices with displacement < 0.01 in sitting pose, examines:
- Which joint they're assigned to
- What chain that joint is on
- What their weight is
- How far they are from their assigned joint
- The delta matrix magnitude for their joint
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer, apply_pose
from tools.skinning_scorer import get_all_poses, _make_body_state


def main():
    print("Loading scene...")
    hs = load_headless_scene()
    print("Loading skin...")
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")

    # Apply sitting pose
    poses = get_all_poses(["sitting"])
    state = _make_body_state(poses["sitting"])
    apply_pose(hs, state)

    joints = hs.skinning.joints
    chain_id_to_name = {v: k for k, v in hs.chain_ids.items()}

    # Pre-compute joint displacements and delta norms
    joint_disps = np.zeros(len(joints), dtype=np.float64)
    delta_norms = np.zeros(len(joints), dtype=np.float64)
    for j_idx, joint in enumerate(joints):
        joint.node.update_world_matrix()
        cur = joint.node.world_matrix[:3, 3]
        rst = joint.rest_world[:3, 3]
        joint_disps[j_idx] = float(np.linalg.norm(cur - rst))

    for binding in hs.skinning.bindings:
        mesh = binding.mesh
        if mesh.rest_positions is None:
            continue
        rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
        V = len(rest)

        vert_disps = np.linalg.norm(current - rest, axis=1)

        # Find zero-displacement vertices
        zero_mask = vert_disps < 0.01
        n_zero = int(np.sum(zero_mask))

        print(f"\n{'='*70}")
        print(f"Mesh: {mesh.name} ({V} verts)")
        print(f"Zero-displacement vertices: {n_zero} ({100*n_zero/V:.1f}%)")

        if n_zero == 0:
            continue

        # Get binding details for zero-disp verts
        ji = binding.joint_indices[zero_mask]
        si = binding.secondary_indices[zero_mask]
        weights = binding.weights[zero_mask]
        zero_rest = rest[zero_mask]
        zero_current = current[zero_mask]

        # Group by body region (Z coordinate)
        regions = {}
        for i in range(n_zero):
            z = float(zero_rest[i, 2])
            if z > -15: region = "head"
            elif z > -25: region = "neck"
            elif z > -45: region = "upper_torso"
            elif z > -80: region = "lower_torso"
            elif z > -95: region = "pelvis"
            elif z > -140: region = "upper_leg"
            elif z > -190: region = "lower_leg"
            else: region = "foot"
            regions.setdefault(region, []).append(i)

        print(f"\nRegion breakdown:")
        for region, indices in sorted(regions.items(), key=lambda x: -len(x[1])):
            print(f"  {region:15s}: {len(indices)} zero-disp verts")

        # For the most affected regions, show joint assignment details
        for region in ["foot", "lower_leg", "upper_leg", "pelvis"]:
            if region not in regions:
                continue
            indices = regions[region]
            r_ji = ji[indices]
            r_si = si[indices]
            r_w = weights[indices]

            # Group by assigned joint
            joint_groups = {}
            for i, idx in enumerate(indices):
                j_name = joints[r_ji[i]].name
                chain = chain_id_to_name.get(joints[r_ji[i]].chain_id, "?")
                key = f"{j_name} ({chain})"
                if key not in joint_groups:
                    joint_groups[key] = {
                        "count": 0,
                        "joint_disp": joint_disps[r_ji[i]],
                        "weights": [],
                        "sec_joints": [],
                        "distances": [],
                    }
                joint_groups[key]["count"] += 1
                joint_groups[key]["weights"].append(float(r_w[i]))
                joint_groups[key]["sec_joints"].append(joints[r_si[i]].name)

                # Distance from vertex to joint rest position
                j_rest = joints[r_ji[i]].rest_world[:3, 3]
                dist = float(np.linalg.norm(zero_rest[idx] - j_rest))
                joint_groups[key]["distances"].append(dist)

            print(f"\n  Region: {region} ({len(indices)} verts)")
            for key, info in sorted(joint_groups.items(), key=lambda x: -x[1]["count"]):
                wts = np.array(info["weights"])
                dists = np.array(info["distances"])
                print(f"    {key:40s}  count={info['count']:5d}  "
                      f"joint_disp={info['joint_disp']:6.1f}  "
                      f"w_mean={float(wts.mean()):.3f}  "
                      f"w_min={float(wts.min()):.3f}  "
                      f"dist_mean={float(dists.mean()):.1f}  "
                      f"dist_max={float(dists.max()):.1f}")
                # Show secondary joint distribution
                sec_counts = {}
                for sj in info["sec_joints"]:
                    sec_counts[sj] = sec_counts.get(sj, 0) + 1
                top_sec = sorted(sec_counts.items(), key=lambda x: -x[1])[:3]
                for sj_name, sc in top_sec:
                    print(f"      secondary: {sj_name} ({sc})")

        # Show a few sample zero-disp vertices with full details
        print(f"\n  Sample zero-disp vertices (first 10):")
        sample_indices = list(range(min(10, n_zero)))
        for si_idx in sample_indices:
            vz = float(zero_rest[si_idx, 2])
            j_idx = int(ji[si_idx])
            s_idx = int(si[si_idx])
            w = float(weights[si_idx])
            j_name = joints[j_idx].name
            s_name = joints[s_idx].name
            j_disp = joint_disps[j_idx]
            s_disp = joint_disps[s_idx]

            # Check: are primary and secondary the same joint?
            same = "SAME" if j_idx == s_idx else "DIFF"
            print(f"    Z={vz:7.1f}  j0={j_name:25s}(d={j_disp:5.1f})  "
                  f"j1={s_name:25s}(d={s_disp:5.1f})  "
                  f"w={w:.3f}  {same}")


if __name__ == "__main__":
    main()
