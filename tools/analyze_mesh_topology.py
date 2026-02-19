"""Analyze skin mesh topology to find connected components and bridges.

Determines whether the skin mesh is one connected piece or has separate
arm/leg/torso components that can be segmented topologically.
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
from tools.headless_loader import load_headless_scene, load_layer, register_layer


def find_connected_components(indices: np.ndarray, vert_count: int) -> list[np.ndarray]:
    """Find connected components using triangle adjacency (union-find)."""
    parent = np.arange(vert_count, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Union vertices in each triangle
    tris = indices.reshape(-1, 3)
    for a, b, c in tris:
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[rb] = ra
        rc2 = find(c)
        ra2 = find(a)
        if ra2 != rc2:
            parent[rc2] = ra2

    # Collect components
    roots = np.array([find(i) for i in range(vert_count)], dtype=np.int32)
    unique_roots = np.unique(roots)
    components = []
    for r in unique_roots:
        members = np.where(roots == r)[0]
        components.append(members)

    # Sort by size descending
    components.sort(key=len, reverse=True)
    return components


def main():
    print("Loading scene...")
    hs = load_headless_scene()
    print("Loading skin...")
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")

    binding = hs.skinning.bindings[0]
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    indices = mesh.geometry.indices

    V = len(rest)
    T = len(indices) // 3
    print(f"\nMesh: {V} vertices, {T} triangles")

    # Find connected components
    print("\nFinding connected components...")
    components = find_connected_components(indices, V)
    print(f"Found {len(components)} connected components")

    # Analyze each component
    print(f"\n{'Comp':>5s} {'Verts':>8s} {'%':>6s} "
          f"{'X range':>18s} {'Y range':>18s} {'Z range':>18s} {'Region':>15s}")
    print("-" * 90)

    for i, comp in enumerate(components[:30]):  # show top 30
        verts = rest[comp]
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
        pct = len(comp) / V * 100

        # Guess region from position
        x_center = (x_min + x_max) / 2
        z_center = (z_min + z_max) / 2
        region = "?"
        if z_max > -20:
            region = "head"
        elif z_center > -80 and abs(x_center) < 15:
            region = "torso"
        elif z_center > -80 and x_center > 15:
            region = "arm_R"
        elif z_center > -80 and x_center < -15:
            region = "arm_L"
        elif z_center < -80 and x_center > 3:
            region = "leg_R"
        elif z_center < -80 and x_center < -3:
            region = "leg_L"
        elif abs(x_center) <= 3 and z_center < -80:
            region = "pelvis/groin"

        # Check if crosses midline
        crosses_midline = x_min < 0 and x_max > 0
        spans_body = z_max - z_min > 50

        print(f"{i:5d} {len(comp):8d} {pct:5.1f}% "
              f"[{x_min:7.1f},{x_max:7.1f}] "
              f"[{y_min:7.1f},{y_max:7.1f}] "
              f"[{z_min:7.1f},{z_max:7.1f}] "
              f"{region:>15s}")

    # Check if the largest component spans the whole body
    if components:
        largest = components[0]
        lv = rest[largest]
        print(f"\nLargest component analysis:")
        print(f"  Vertices: {len(largest)} ({len(largest)/V*100:.1f}%)")
        print(f"  X: [{lv[:, 0].min():.1f}, {lv[:, 0].max():.1f}]")
        print(f"  Z: [{lv[:, 2].min():.1f}, {lv[:, 2].max():.1f}]")

        # Check if arms are connected to torso in the largest component
        arm_r_verts = (lv[:, 0] > 20) & (lv[:, 2] > -80)
        arm_l_verts = (lv[:, 0] < -20) & (lv[:, 2] > -80)
        torso_verts = (np.abs(lv[:, 0]) < 15) & (lv[:, 2] > -80)
        leg_r_verts = (lv[:, 0] > 3) & (lv[:, 2] < -90)
        leg_l_verts = (lv[:, 0] < -3) & (lv[:, 2] < -90)

        print(f"\n  Region vertex counts in largest component:")
        print(f"    arm_R (X>20, Z>-80):  {int(np.sum(arm_r_verts))}")
        print(f"    arm_L (X<-20, Z>-80): {int(np.sum(arm_l_verts))}")
        print(f"    torso (|X|<15, Z>-80): {int(np.sum(torso_verts))}")
        print(f"    leg_R (X>3, Z<-90):   {int(np.sum(leg_r_verts))}")
        print(f"    leg_L (X<-3, Z<-90):  {int(np.sum(leg_l_verts))}")

        # If separate components exist per limb, detect them
        if len(components) > 1:
            print(f"\n  Component-region mapping:")
            for i, comp in enumerate(components[:10]):
                cv = rest[comp]
                cx = cv[:, 0].mean()
                cz = cv[:, 2].mean()
                print(f"    Comp {i}: {len(comp)} verts, "
                      f"center=({cx:.1f}, {cv[:, 1].mean():.1f}, {cz:.1f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
