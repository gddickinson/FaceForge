"""Focused investigation: which vertices in the hip X range (X=0..20)
are bound to arm chains and move incorrectly during arm raise?"""

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

    binding = hs.skinning.bindings[0]
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    V = len(rest)
    ji = binding.joint_indices
    si = binding.secondary_indices
    weights = binding.weights

    # Identify arm chain IDs (arm_R, arm_L + all hand chains)
    arm_like_ids = set()
    hand_ids = set()
    for cid, cname in chain_id_to_name.items():
        if "arm" in cname.lower():
            arm_like_ids.add(cid)
        if "hand" in cname.lower():
            arm_like_ids.add(cid)
            hand_ids.add(cid)

    vert_chains = np.array([joints[idx].chain_id for idx in ji], dtype=np.int32)
    sec_chains = np.array([joints[idx].chain_id for idx in si], dtype=np.int32)

    # Find vertices in hip/pelvis X range (X between -20 and 20)
    # that are bound to arm/hand chains
    hip_x_mask = (np.abs(rest[:, 0]) < 20)
    arm_bound_mask = np.isin(vert_chains, list(arm_like_ids))
    cross_sec_mask = np.isin(sec_chains, list(arm_like_ids)) & ~arm_bound_mask

    problem_primary = hip_x_mask & arm_bound_mask
    problem_secondary = hip_x_mask & cross_sec_mask

    n_primary = int(np.sum(problem_primary))
    n_secondary = int(np.sum(problem_secondary))

    print(f"\nVertices with |X| < 20 bound to arm/hand chains (PRIMARY): {n_primary}")
    if n_primary > 0:
        prest = rest[problem_primary]
        pji = ji[problem_primary]
        # Group by body region
        for region, (zlo, zhi) in [
            ("head/neck", (-25, float("inf"))),
            ("upper_torso", (-45, -25)),
            ("lower_torso", (-80, -45)),
            ("pelvis", (-95, -80)),
            ("upper_leg", (-140, -95)),
            ("lower_leg", (-200, -140)),
        ]:
            mask = (prest[:, 2] >= zlo) & (prest[:, 2] < zhi)
            n = int(np.sum(mask))
            if n > 0:
                print(f"  {region:15s}: {n}")

    print(f"\nVertices with |X| < 20 with arm/hand SECONDARY (cross-chain): {n_secondary}")
    if n_secondary > 0:
        srest = rest[problem_secondary]
        sji = ji[problem_secondary]
        ssi = si[problem_secondary]
        sw = weights[problem_secondary]
        # Group by body region
        for region, (zlo, zhi) in [
            ("head/neck", (-25, float("inf"))),
            ("upper_torso", (-45, -25)),
            ("lower_torso", (-80, -45)),
            ("pelvis", (-95, -80)),
            ("upper_leg", (-140, -95)),
            ("lower_leg", (-200, -140)),
        ]:
            mask = (srest[:, 2] >= zlo) & (srest[:, 2] < zhi)
            n = int(np.sum(mask))
            if n > 0:
                # Show primary chain, secondary chain, weight
                sec_w = sw[mask]
                print(f"  {region:15s}: {n} verts, sec_weight_mean={1-float(sec_w.mean()):.3f}")
                # Show sample
                idx_list = np.where(mask)[0][:5]
                for idx in idx_list:
                    v = srest[idx]
                    pj = joints[sji[idx]].name
                    pc = chain_id_to_name.get(joints[sji[idx]].chain_id, "?")
                    sj = joints[ssi[idx]].name
                    sc = chain_id_to_name.get(joints[ssi[idx]].chain_id, "?")
                    w = sw[idx]
                    print(f"    pos=({v[0]:7.1f},{v[1]:7.1f},{v[2]:7.1f})  "
                          f"primary={pj}({pc})  sec={sj}({sc})  w={w:.3f}")

    # Now apply arm raise and measure displacement in hip region
    print("\n" + "=" * 60)
    print("ARM RAISE: Hip region displacement analysis")
    print("=" * 60)

    poses = get_all_poses(["extreme_arm_raise"])
    state = _make_body_state(poses["extreme_arm_raise"])
    apply_pose(hs, state)

    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    disps = np.linalg.norm(current - rest, axis=1)

    # Focus on hip/pelvis/upper_leg region, |X| < 20
    for region, (zlo, zhi) in [
        ("lower_torso", (-80, -45)),
        ("pelvis", (-95, -80)),
        ("upper_leg", (-140, -95)),
    ]:
        mask = (rest[:, 2] >= zlo) & (rest[:, 2] < zhi) & (np.abs(rest[:, 0]) < 20)
        if not np.any(mask):
            continue
        n = int(np.sum(mask))
        region_disps = disps[mask]
        moved = region_disps > 2.0
        n_moved = int(np.sum(moved))
        print(f"\n{region} (|X|<20): {n} verts, {n_moved} moved >2.0 ({100*n_moved/n:.1f}%)")
        if n_moved > 0:
            moved_rest = rest[mask][moved]
            moved_disps = region_disps[moved]
            moved_ji = ji[mask][moved]
            print(f"  mean_disp={float(moved_disps.mean()):.2f}, max={float(moved_disps.max()):.2f}")
            # Joint breakdown
            jcounts = {}
            for j_idx in moved_ji:
                jname = joints[j_idx].name
                cname = chain_id_to_name.get(joints[j_idx].chain_id, "?")
                key = f"{jname} ({cname})"
                jcounts[key] = jcounts.get(key, 0) + 1
            for key, cnt in sorted(jcounts.items(), key=lambda x: -x[1])[:10]:
                print(f"    {key:40s}: {cnt}")
            # Sample
            print(f"  Samples:")
            for i in range(min(10, n_moved)):
                v = moved_rest[i]
                d = moved_disps[i]
                jn = joints[moved_ji[i]].name
                cn = chain_id_to_name.get(joints[moved_ji[i]].chain_id, "?")
                print(f"    ({v[0]:7.1f},{v[1]:7.1f},{v[2]:7.1f}) d={d:6.2f} {jn} ({cn})")

    # Also show X histogram for arm-bound pelvis verts
    print("\n" + "=" * 60)
    print("X distribution of ALL arm-bound pelvis verts that moved in arm raise")
    print("=" * 60)
    pelvis_mask = (rest[:, 2] >= -95) & (rest[:, 2] < -80)
    moved_mask = disps > 2.0
    arm_in_pelvis = pelvis_mask & moved_mask & arm_bound_mask
    if np.any(arm_in_pelvis):
        x_vals = rest[arm_in_pelvis, 0]
        for xlo, xhi in [(-50,-40),(-40,-30),(-30,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,30),(30,40),(40,50),(50,60)]:
            n = int(np.sum((x_vals >= xlo) & (x_vals < xhi)))
            if n > 0:
                print(f"  X=[{xlo:4d},{xhi:4d}): {n}")

    apply_pose(hs, BodyState())


if __name__ == "__main__":
    main()
