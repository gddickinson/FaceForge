"""Debug why pelvis skin vertices bind to lumbar_5 instead of hip joints."""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer, apply_pose
from tools.skinning_scorer import get_all_poses, _make_body_state


def main():
    hs = load_headless_scene()
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")

    joints = hs.skinning.joints
    chain_id_to_name = {v: k for k, v in hs.chain_ids.items()}

    print("JOINT SEGMENTS:")
    print(f"{'Joint':30s} {'Chain':15s} {'SegStart':>30s} {'SegEnd':>30s} {'Length':>8s}")
    for j in joints:
        s = j.segment_start
        e = j.segment_end
        cname = chain_id_to_name.get(j.chain_id, f"id={j.chain_id}")
        if s is not None and e is not None:
            seg_len = float(np.linalg.norm(e - s))
            degenerate = " (DEGEN)" if seg_len < 0.01 else ""
            print(f"  {j.name:30s} {cname:15s} "
                  f"({s[0]:7.1f},{s[1]:7.1f},{s[2]:7.1f}) "
                  f"({e[0]:7.1f},{e[1]:7.1f},{e[2]:7.1f})  "
                  f"{seg_len:7.1f}{degenerate}")
        else:
            print(f"  {j.name:30s} {cname:15s}  NO SEGMENT")

    # Now check a specific pelvis vertex
    binding = hs.skinning.bindings[0]
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)

    # Find pelvis vertices assigned to lumbar_5
    ji = binding.joint_indices
    lumbar5_idx = None
    for idx, j in enumerate(joints):
        if j.name == "lumbar_5":
            lumbar5_idx = idx
            break

    if lumbar5_idx is not None:
        # Find pelvis verts (Z = -80 to -95) assigned to lumbar_5
        pelvis_mask = (rest[:, 2] >= -95) & (rest[:, 2] < -80)
        lumbar5_mask = ji == lumbar5_idx
        stuck = pelvis_mask & lumbar5_mask

        n = int(np.sum(stuck))
        print(f"\nPelvis vertices assigned to lumbar_5: {n}")

        if n > 0:
            stuck_rest = rest[stuck]
            # Compute distances to various joints
            for jname in ["lumbar_5", "lumbar_4", "hip_R", "hip_L", "knee_R"]:
                j = next((j for j in joints if j.name == jname), None)
                if j is None:
                    continue
                # Distance to joint rest position
                j_pos = j.rest_world[:3, 3]
                dists = np.linalg.norm(stuck_rest - j_pos, axis=1)
                # Distance to segment
                if j.segment_start is not None and j.segment_end is not None:
                    seg_len = float(np.linalg.norm(j.segment_end - j.segment_start))
                    if seg_len > 0.01:
                        ab = j.segment_end - j.segment_start
                        ap = stuck_rest - j.segment_start
                        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
                        closest = j.segment_start + t[:, np.newaxis] * ab
                        seg_dists = np.linalg.norm(stuck_rest - closest, axis=1)
                        print(f"  {jname:20s} chain={chain_id_to_name.get(j.chain_id, '?'):10s} "
                              f"joint_dist: mean={dists.mean():.1f} min={dists.min():.1f} max={dists.max():.1f}  "
                              f"seg_dist: mean={seg_dists.mean():.1f} min={seg_dists.min():.1f} max={seg_dists.max():.1f}")
                    else:
                        print(f"  {jname:20s} chain={chain_id_to_name.get(j.chain_id, '?'):10s} "
                              f"joint_dist: mean={dists.mean():.1f} min={dists.min():.1f} max={dists.max():.1f}  "
                              f"seg=DEGENERATE")
                else:
                    print(f"  {jname:20s} chain={chain_id_to_name.get(j.chain_id, '?'):10s} "
                          f"joint_dist: mean={dists.mean():.1f}  NO SEGMENT")

            # Show sample stuck pelvis verts
            print(f"\n  Sample stuck pelvis verts (first 5):")
            for i in range(min(5, n)):
                v = stuck_rest[i]
                print(f"    pos=({v[0]:7.1f}, {v[1]:7.1f}, {v[2]:7.1f})")


if __name__ == "__main__":
    main()
