"""Diagnose hip/leg-chain mis-binding of pelvis/groin skin vertices.

Identifies vertices assigned to leg chains that are near the pelvis/midline,
which should be on spine instead.  Similar to debug_arm_misbinding but for
the hip/groin region.
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

    # Find leg chain IDs
    leg_r_cid = hs.chain_ids.get("leg_R")
    leg_l_cid = hs.chain_ids.get("leg_L")
    spine_cid = hs.chain_ids.get("spine", 0)
    print(f"\nChain IDs: leg_R={leg_r_cid}, leg_L={leg_l_cid}, spine={spine_cid}")

    # Print all chain centroids
    print("\nChain centroids:")
    chain_joints_map: dict[int, list] = {}
    for j in joints:
        chain_joints_map.setdefault(j.chain_id, []).append(j)
    for cid, cjoints in sorted(chain_joints_map.items()):
        positions = np.array([j.rest_world[:3, 3] for j in cjoints])
        cx, cy, cz = positions.mean(axis=0)
        name = chain_id_to_name.get(cid, f"chain_{cid}")
        print(f"  {name:15s} (id={cid}): centroid=({cx:7.2f}, {cy:7.2f}, {cz:7.2f}) "
              f"joints={len(cjoints)}")

    # Leg chain joint positions
    for cid, name in [(leg_r_cid, "leg_R"), (leg_l_cid, "leg_L")]:
        if cid is None:
            continue
        chain_joints = [j for j in joints if j.chain_id == cid]
        print(f"\n{name} joints:")
        for j in chain_joints:
            pos = j.rest_world[:3, 3]
            print(f"  {j.name:25s} X={pos[0]:7.2f} Y={pos[1]:7.2f} Z={pos[2]:7.2f}")

    # Spine bottom
    spine_joints = [j for j in joints if j.chain_id == spine_cid]
    spine_z_vals = [j.rest_world[2, 3] for j in spine_joints]
    spine_z_min = min(spine_z_vals)
    print(f"\nSpine Z range: [{min(spine_z_vals):.2f}, {max(spine_z_vals):.2f}]")
    print(f"Spine bottom joint: {spine_z_min:.2f}")

    # Analyze leg-bound vertices by position
    for cid, name in [(leg_r_cid, "leg_R"), (leg_l_cid, "leg_L")]:
        if cid is None:
            continue
        leg_mask = vert_chains == cid
        leg_rest = rest[leg_mask]
        n_leg = int(np.sum(leg_mask))

        print(f"\n{'='*60}")
        print(f"{name}: {n_leg} vertices total")

        # X distribution
        print(f"\n  X distribution (rest position):")
        if name == "leg_R":
            x_vals = leg_rest[:, 0]
        else:
            x_vals = -leg_rest[:, 0]  # flip so positive = lateral

        for xlo, xhi in [(-5, 0), (0, 3), (3, 5), (5, 8), (8, 10), (10, 15),
                         (15, 20), (20, 25)]:
            mask_x = (x_vals >= xlo) & (x_vals < xhi)
            n = int(np.sum(mask_x))
            if n > 0:
                avg_z = float(leg_rest[mask_x, 2].mean())
                print(f"    |X|=[{xlo},{xhi}): {n:6d} verts (avg Z={avg_z:.1f})")

        # Z distribution - focus on pelvis region
        print(f"\n  Z distribution (rest position):")
        for zlo, zhi in [(-60, -70), (-70, -75), (-75, -80), (-80, -85),
                         (-85, -90), (-90, -100), (-100, -110), (-110, -130)]:
            mask_z = (leg_rest[:, 2] >= zlo) & (leg_rest[:, 2] < zhi)
            n = int(np.sum(mask_z))
            if n > 0:
                avg_x = float(x_vals[mask_z[leg_mask[leg_mask].searchsorted(True):] if False else mask_z].mean()) if False else 0
                # Simpler approach
                sub_x = x_vals[mask_z]
                avg_x = float(sub_x.mean())
                print(f"    Z=[{zlo},{zhi}): {n:6d} verts (avg |X|={avg_x:.1f})")

        # Proximal vertices (above spine bottom, in pelvis region)
        proximal = leg_rest[:, 2] > spine_z_min
        n_prox = int(np.sum(proximal))
        print(f"\n  PROXIMAL vertices (Z > spine bottom {spine_z_min:.1f}): "
              f"{n_prox} ({100*n_prox/max(n_leg,1):.1f}%)")
        if n_prox > 0:
            prox_rest = leg_rest[proximal]
            prox_x = x_vals[proximal]
            print(f"    X range: [{prox_rest[:, 0].min():.1f}, {prox_rest[:, 0].max():.1f}]")
            print(f"    Z range: [{prox_rest[:, 2].min():.1f}, {prox_rest[:, 2].max():.1f}]")
            # How many are near midline?
            near_mid = np.abs(prox_rest[:, 0]) < 8
            print(f"    Near midline (|X| < 8): {int(np.sum(near_mid))}")

        # Medial + proximal (groin area)
        medial_mask = np.abs(leg_rest[:, 0]) < 10
        groin = medial_mask & (leg_rest[:, 2] > spine_z_min - 10)
        n_groin = int(np.sum(groin))
        print(f"\n  GROIN vertices (|X| < 10 AND Z > {spine_z_min-10:.1f}): "
              f"{n_groin} ({100*n_groin/max(n_leg,1):.1f}%)")
        if n_groin > 0:
            groin_rest = leg_rest[groin]
            print(f"    X range: [{groin_rest[:, 0].min():.1f}, {groin_rest[:, 0].max():.1f}]")
            print(f"    Z range: [{groin_rest[:, 2].min():.1f}, {groin_rest[:, 2].max():.1f}]")

    # Apply sitting pose and measure displacement
    sitting_vals = get_all_poses(["sitting"]).get("sitting", {})
    if not sitting_vals:
        sitting_vals = {"hip_r_flex": 1.0, "hip_l_flex": 1.0, "knee_r_flex": 1.0, "knee_l_flex": 1.0}

    state = _make_body_state(sitting_vals)
    apply_pose(hs, state)

    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    disp = np.linalg.norm(current - rest, axis=1)

    print(f"\n{'='*60}")
    print(f"Pose: sitting")

    for cid, name in [(leg_r_cid, "leg_R"), (leg_l_cid, "leg_L"),
                      (spine_cid, "spine")]:
        if cid is None:
            continue
        chain_mask = vert_chains == cid
        chain_disp = disp[chain_mask]
        chain_rest = rest[chain_mask]

        print(f"\n  {name} displacement stats:")
        print(f"    Total verts: {int(np.sum(chain_mask))}")
        print(f"    Displaced >5: {int(np.sum(chain_disp > 5))}")
        print(f"    Displaced >10: {int(np.sum(chain_disp > 10))}")
        print(f"    Displaced >20: {int(np.sum(chain_disp > 20))}")
        print(f"    Max displacement: {chain_disp.max():.1f}")

        # Displacement by Z position (pelvis region)
        print(f"    Displacement by Z position:")
        for zlo, zhi in [(-70, -75), (-75, -80), (-80, -85), (-85, -90),
                         (-90, -100), (-100, -120)]:
            mask_z = (chain_rest[:, 2] >= zlo) & (chain_rest[:, 2] < zhi)
            n = int(np.sum(mask_z))
            if n > 0:
                d = chain_disp[mask_z]
                print(f"      Z=[{zlo},{zhi}): {n:5d} verts, "
                      f"disp mean={d.mean():.1f} max={d.max():.1f}")

    # Top 30 displaced vertices
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
