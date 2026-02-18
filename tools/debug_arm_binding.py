"""Debug arm chain binding: show which vertices are assigned to arm chains
and whether any are in the hip/leg region.

Outputs:
- Chain geometry: Z-range, X-centroid, spatial limit, Z-margin for each chain
- Arm-bound vertices in lower body regions
- Cross-chain blending details for arm↔leg boundary
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer, apply_pose
from tools.skinning_scorer import get_all_poses, _make_body_state


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


def _get_region(z):
    for name, (lo, hi) in BODY_REGIONS.items():
        if lo <= z < hi:
            return name
    return "unknown"


def main():
    print("Loading scene...")
    hs = load_headless_scene()
    print("Loading skin...")
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")

    joints = hs.skinning.joints
    chain_id_to_name = {v: k for k, v in hs.chain_ids.items()}

    # ── Chain geometry summary ──
    print("\n" + "=" * 80)
    print("CHAIN GEOMETRY")
    print("=" * 80)

    chain_joints: dict[int, list] = {}
    for j in joints:
        chain_joints.setdefault(j.chain_id, []).append(j)

    for cid in sorted(chain_joints.keys()):
        cname = chain_id_to_name.get(cid, f"id={cid}")
        cj = chain_joints[cid]
        zs = [float(j.rest_world[2, 3]) for j in cj]
        xs = [float(j.rest_world[0, 3]) for j in cj]
        ys = [float(j.rest_world[1, 3]) for j in cj]
        z_lo, z_hi = min(zs), max(zs)
        z_extent = z_hi - z_lo
        x_cent = sum(xs) / len(xs)
        y_cent = sum(ys) / len(ys)

        # Compute the actual spatial limit and Z-margin that would be used
        spatial_limit = 20.0
        min_spatial = 12.0
        spatial_factor = 0.35
        chain_spatial = min(spatial_limit, max(min_spatial, z_extent * spatial_factor))

        chain_z_margin = 15.0
        min_z_pad = 8.0
        actual_z_margin = min(chain_z_margin, max(min_z_pad, z_extent * 0.25))

        print(f"\n  Chain {cid} ({cname}): {len(cj)} joints")
        print(f"    Z range: [{z_lo:.1f}, {z_hi:.1f}]  extent={z_extent:.1f}")
        print(f"    X centroid: {x_cent:.1f}  Y centroid: {y_cent:.1f}")
        print(f"    Spatial limit: {chain_spatial:.1f}  (from extent={z_extent:.1f})")
        print(f"    Z margin: {actual_z_margin:.1f}")
        print(f"    Effective Z filter: [{z_lo - actual_z_margin:.1f}, {z_hi + actual_z_margin:.1f}]")

        # List joint names and positions
        for j in cj:
            pos = j.rest_world[:3, 3]
            print(f"      {j.name:30s}  pos=({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f})")

    # ── Binding analysis ──
    print("\n" + "=" * 80)
    print("VERTEX-TO-CHAIN BINDING ANALYSIS")
    print("=" * 80)

    for binding in hs.skinning.bindings:
        mesh = binding.mesh
        if mesh.rest_positions is None:
            continue
        rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        V = len(rest)
        ji = binding.joint_indices
        si = binding.secondary_indices
        weights = binding.weights

        # Get chain for each vertex
        vert_chains = np.array([joints[idx].chain_id for idx in ji], dtype=np.int32)

        print(f"\nMesh: {mesh.name} ({V} verts)")

        # Count per chain
        for cid in sorted(np.unique(vert_chains)):
            cname = chain_id_to_name.get(cid, f"id={cid}")
            mask = vert_chains == cid
            n = int(np.sum(mask))
            chain_rest = rest[mask]
            z_min = float(chain_rest[:, 2].min())
            z_max = float(chain_rest[:, 2].max())
            x_min = float(chain_rest[:, 0].min())
            x_max = float(chain_rest[:, 0].max())
            print(f"  {cname:15s}: {n:7d} verts  "
                  f"Z=[{z_min:.1f}, {z_max:.1f}]  "
                  f"X=[{x_min:.1f}, {x_max:.1f}]")

        # ── Focus: arm-bound vertices in lower body ──
        arm_chain_ids = set()
        for cid, cname in chain_id_to_name.items():
            if "arm" in cname.lower():
                arm_chain_ids.add(cid)

        if arm_chain_ids:
            arm_mask = np.isin(vert_chains, list(arm_chain_ids))
            arm_rest = rest[arm_mask]
            arm_ji = ji[arm_mask]
            arm_si = si[arm_mask]
            arm_w = weights[arm_mask]

            # Find arm-bound verts below the upper_torso (Z < -45)
            low_mask = arm_rest[:, 2] < -45
            n_low = int(np.sum(low_mask))

            if n_low > 0:
                print(f"\n  *** ARM-BOUND VERTICES IN LOWER BODY: {n_low} ***")
                low_rest = arm_rest[low_mask]
                low_ji = arm_ji[low_mask]
                low_si = arm_si[low_mask]
                low_w = arm_w[low_mask]

                # Region breakdown
                region_counts = {}
                for i in range(n_low):
                    z = float(low_rest[i, 2])
                    region = _get_region(z)
                    region_counts[region] = region_counts.get(region, 0) + 1

                for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
                    print(f"    {region:20s}: {count}")

                # Sample vertices
                print(f"\n    Sample arm-bound lower-body vertices (first 20):")
                for i in range(min(20, n_low)):
                    v = low_rest[i]
                    j_name = joints[low_ji[i]].name
                    s_name = joints[low_si[i]].name
                    w = low_w[i]
                    print(f"      pos=({v[0]:7.1f}, {v[1]:7.1f}, {v[2]:7.1f})  "
                          f"joint={j_name:25s}  sec={s_name:25s}  w={w:.3f}")
            else:
                print(f"\n  No arm-bound vertices in lower body (good!)")

    # ── Apply extreme_arm_raise and check displacement ──
    print("\n" + "=" * 80)
    print("ARM RAISE DISPLACEMENT CHECK")
    print("=" * 80)

    poses = get_all_poses(["extreme_arm_raise"])
    state = _make_body_state(poses["extreme_arm_raise"])
    apply_pose(hs, state)

    for binding in hs.skinning.bindings:
        mesh = binding.mesh
        if mesh.rest_positions is None:
            continue
        rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
        V = len(rest)
        disps = np.linalg.norm(current - rest, axis=1)

        print(f"\nMesh: {mesh.name}")

        # Region summary
        for name, (lo, hi) in BODY_REGIONS.items():
            mask = (rest[:, 2] >= lo) & (rest[:, 2] < hi)
            if not np.any(mask):
                continue
            region_disps = disps[mask]
            n = int(np.sum(mask))
            moved = region_disps > 0.5
            n_moved = int(np.sum(moved))
            print(f"  {name:15s}: V={n:7d}  "
                  f"mean_disp={float(region_disps.mean()):7.2f}  "
                  f"max_disp={float(region_disps.max()):7.2f}  "
                  f"moved(>0.5)={n_moved:6d} ({100*n_moved/n:.1f}%)")

        # Find vertices that shouldn't move in arm raise but did
        # Hip/leg regions should be mostly stationary in arm raise
        for region_name in ["pelvis", "upper_leg", "lower_leg", "foot"]:
            lo, hi = BODY_REGIONS[region_name]
            mask = (rest[:, 2] >= lo) & (rest[:, 2] < hi)
            if not np.any(mask):
                continue
            region_disps = disps[mask]
            moved_thresh = 2.0
            moved_mask = region_disps > moved_thresh
            n_moved = int(np.sum(moved_mask))
            if n_moved > 0:
                moved_rest = rest[mask][moved_mask]
                moved_disps_vals = region_disps[moved_mask]
                ji_region = binding.joint_indices[np.where(mask)[0][moved_mask]]

                print(f"\n  *** {region_name}: {n_moved} verts moved > {moved_thresh} in arm raise ***")
                # Joint breakdown
                joint_counts = {}
                for idx in ji_region:
                    jname = joints[idx].name
                    cname = chain_id_to_name.get(joints[idx].chain_id, "?")
                    key = f"{jname} ({cname})"
                    joint_counts[key] = joint_counts.get(key, 0) + 1
                for key, count in sorted(joint_counts.items(), key=lambda x: -x[1])[:10]:
                    print(f"    {key:40s}: {count}")

                # Sample
                print(f"    Sample (first 10):")
                for i in range(min(10, n_moved)):
                    v = moved_rest[i]
                    d = moved_disps_vals[i]
                    j_name = joints[ji_region[i]].name
                    print(f"      pos=({v[0]:7.1f}, {v[1]:7.1f}, {v[2]:7.1f})  "
                          f"disp={d:.2f}  joint={j_name}")


if __name__ == "__main__":
    main()
