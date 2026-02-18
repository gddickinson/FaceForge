"""Quick investigation: find vertices that don't move when skeleton moves."""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer, apply_pose
from tools.skinning_scorer import get_all_poses, _make_body_state


def main():
    print("Loading scene...")
    hs = load_headless_scene()

    layers = ["skin", "back_muscles", "torso_muscles", "shoulder_muscles",
              "arm_muscles", "hip_muscles", "leg_muscles"]
    for name in layers:
        print(f"  Loading {name}...")
        meshes = load_layer(hs, name)
        register_layer(hs, meshes, name)

    # Apply sitting pose
    poses = get_all_poses(["sitting"])
    state = _make_body_state(poses["sitting"])
    apply_pose(hs, state)

    print(f"\n{'='*70}")
    print(f"STATIC VERTEX CHECK — sitting pose")
    print(f"{'='*70}\n")

    # For each registered mesh, find vertices that barely moved
    total_static = 0
    total_verts = 0

    for binding in hs.skinning.bindings:
        mesh = binding.mesh
        if mesh.rest_positions is None:
            continue

        rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
        V = len(rest)

        # Per-vertex displacement
        displacements = np.linalg.norm(current - rest, axis=1)

        # Per-joint displacement: how much did each joint move?
        ji = binding.joint_indices
        joint_disps = np.zeros(V, dtype=np.float64)
        for j_idx in range(len(hs.skinning.joints)):
            joint = hs.skinning.joints[j_idx]
            joint.node.update_world_matrix()
            current_pos = joint.node.world_matrix[:3, 3]
            rest_pos = joint.rest_world[:3, 3]
            j_disp = float(np.linalg.norm(current_pos - rest_pos))
            mask = ji == j_idx
            joint_disps[mask] = j_disp

        # Static vertices: joint moved > 5 units but vertex moved < 0.5 units
        static_mask = (joint_disps > 5.0) & (displacements < 0.5)
        # Also check: vertex moved < 10% of its joint's movement
        ratio_static = (joint_disps > 2.0) & (displacements < joint_disps * 0.1)

        n_static = int(np.sum(static_mask))
        n_ratio = int(np.sum(ratio_static))

        total_static += n_static
        total_verts += V

        if n_static > 0 or n_ratio > 50:
            print(f"  {mesh.name:30s}  V={V:6d}  "
                  f"static(abs)={n_static:5d}  static(ratio)={n_ratio:5d}  "
                  f"max_joint_disp={joint_disps.max():.1f}  "
                  f"min_vert_disp={displacements.min():.3f}")

            # Show breakdown by joint for static verts
            if n_static > 0:
                static_joints = ji[static_mask]
                unique_joints = np.unique(static_joints)
                for uj in unique_joints[:5]:  # top 5
                    j = hs.skinning.joints[uj]
                    count = int(np.sum(static_joints == uj))
                    j_d = float(np.linalg.norm(
                        j.node.world_matrix[:3, 3] - j.rest_world[:3, 3]
                    ))
                    chain_name = {v: k for k, v in hs.chain_ids.items()}.get(j.chain_id, "?")
                    print(f"    joint={j.name:20s} chain={chain_name:10s} "
                          f"joint_moved={j_d:.1f}  stuck_verts={count}")

    print(f"\nTotal: {total_static} static vertices out of {total_verts}")

    # Also check: are there any unregistered meshes?
    registered_names = {b.mesh.name for b in hs.skinning.bindings}
    print(f"\nRegistered meshes: {len(registered_names)}")


if __name__ == "__main__":
    main()
