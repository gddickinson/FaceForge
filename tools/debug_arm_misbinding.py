"""Diagnose arm-chain mis-binding of torso/midline skin vertices.

Identifies vertices assigned to arm chains that are near the body midline,
which should be on spine/ribs instead.
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

    binding = hs.skinning.bindings[0]
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    ji = binding.joint_indices
    joints = hs.skinning.joints
    chain_id_to_name = {v: k for k, v in hs.chain_ids.items()}

    # Get chain info per vertex
    vert_chains = np.array([joints[idx].chain_id for idx in ji], dtype=np.int32)

    # Find arm chain IDs
    arm_r_cid = hs.chain_ids.get("arm_R")
    arm_l_cid = hs.chain_ids.get("arm_L")
    print(f"\nArm chain IDs: arm_R={arm_r_cid}, arm_L={arm_l_cid}")

    # Arm chain joint positions
    for cid, name in [(arm_r_cid, "arm_R"), (arm_l_cid, "arm_L")]:
        if cid is None:
            continue
        chain_joints = [j for j in joints if j.chain_id == cid]
        print(f"\n{name} joints:")
        for j in chain_joints:
            pos = j.rest_world[:3, 3]
            print(f"  {j.name:20s} X={pos[0]:7.2f} Y={pos[1]:7.2f} Z={pos[2]:7.2f}")

    # Analyze arm-bound vertices by X position
    for cid, name in [(arm_r_cid, "arm_R"), (arm_l_cid, "arm_L")]:
        if cid is None:
            continue
        arm_mask = vert_chains == cid
        arm_rest = rest[arm_mask]
        n_arm = int(np.sum(arm_mask))

        print(f"\n{'='*60}")
        print(f"{name}: {n_arm} vertices total")

        # X distribution
        print(f"\n  X distribution (rest position):")
        for xlo, xhi in [(-5, 0), (0, 5), (5, 8), (8, 10), (10, 12), (12, 15),
                         (15, 20), (20, 25), (25, 30), (30, 40)]:
            if name == "arm_L":
                mask_x = (arm_rest[:, 0] >= -xhi) & (arm_rest[:, 0] < -xlo)
                label = f"X=[{-xhi},{-xlo})"
            else:
                mask_x = (arm_rest[:, 0] >= xlo) & (arm_rest[:, 0] < xhi)
                label = f"X=[{xlo},{xhi})"
            n = int(np.sum(mask_x))
            if n > 0:
                avg_z = float(arm_rest[mask_x, 2].mean())
                print(f"    {label:15s}: {n:6d} verts (avg Z={avg_z:.1f})")

        # Z distribution
        print(f"\n  Z distribution (rest position):")
        for zlo, zhi in [(-90, -70), (-70, -50), (-50, -30), (-30, -10), (-10, 10)]:
            mask_z = (arm_rest[:, 2] >= zlo) & (arm_rest[:, 2] < zhi)
            n = int(np.sum(mask_z))
            if n > 0:
                avg_x = float(arm_rest[mask_z, 0].mean())
                print(f"    Z=[{zlo},{zhi}): {n:6d} verts (avg X={avg_x:.1f})")

        # Medial vertices (|X| < 10 for arm, close to midline)
        if name == "arm_R":
            medial = arm_rest[:, 0] < 10
        else:
            medial = arm_rest[:, 0] > -10
        n_medial = int(np.sum(medial))
        print(f"\n  MEDIAL vertices (|X| < 10): {n_medial} ({100*n_medial/n_arm:.1f}%)")

        if n_medial > 0:
            medial_rest = arm_rest[medial]
            print(f"    X range: [{medial_rest[:, 0].min():.1f}, {medial_rest[:, 0].max():.1f}]")
            print(f"    Z range: [{medial_rest[:, 2].min():.1f}, {medial_rest[:, 2].max():.1f}]")

    # Now apply arm raise and measure displacement
    poses_dict = {
        "arm_raise": {"shoulder_r_abduct": 1.0, "shoulder_l_abduct": 0.3},
    }
    for pose_name, pose_vals in poses_dict.items():
        state = _make_body_state(pose_vals)
        apply_pose(hs, state)

        current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
        disp = np.linalg.norm(current - rest, axis=1)

        print(f"\n{'='*60}")
        print(f"Pose: {pose_name}")

        for cid, name in [(arm_r_cid, "arm_R"), (arm_l_cid, "arm_L")]:
            if cid is None:
                continue
            arm_mask = vert_chains == cid
            arm_disp = disp[arm_mask]
            arm_rest_pos = rest[arm_mask]

            # Displacement by X position
            print(f"\n  {name} displacement by X position:")
            if name == "arm_R":
                x_vals = arm_rest_pos[:, 0]
            else:
                x_vals = -arm_rest_pos[:, 0]  # flip so positive = lateral

            for xlo, xhi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]:
                mask_x = (x_vals >= xlo) & (x_vals < xhi)
                n = int(np.sum(mask_x))
                if n > 0:
                    d = arm_disp[mask_x]
                    print(f"    |X|=[{xlo},{xhi}): {n:6d} verts, "
                          f"disp mean={d.mean():.1f} max={d.max():.1f}")

        # Find the worst 30 vertices across all chains
        worst_idx = np.argsort(-disp)[:30]
        print(f"\n  Worst 30 displaced vertices:")
        for i in worst_idx:
            v = rest[i]
            c = current[i]
            d = disp[i]
            jname = joints[ji[i]].name
            cname = chain_id_to_name.get(joints[ji[i]].chain_id, "?")
            delta = c - v
            print(f"    rest=({v[0]:6.1f},{v[1]:6.1f},{v[2]:6.1f}) "
                  f"delta=({delta[0]:6.1f},{delta[1]:6.1f},{delta[2]:6.1f}) "
                  f"disp={d:5.1f} {jname}({cname})")

    apply_pose(hs, BodyState())
    print("\nDone.")


if __name__ == "__main__":
    main()
