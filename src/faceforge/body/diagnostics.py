"""Skinning diagnostics: detect mesh vertices displaced beyond expected limits.

Provides tools to:
1. Analyze binding assignments at registration time (per-vertex chain/joint info)
2. After update(), detect vertices displaced beyond a threshold relative to
   their assigned skeletal elements
3. Check topological distortion: edge stretch, triangle inversion, area collapse
4. Generate per-mesh reports identifying misbehaving structures

Usage:
    diag = SkinningDiagnostic(soft_tissue_skinning)
    report = diag.analyze_bindings()        # registration-time analysis
    anomalies = diag.check_displacements()  # post-update displacement check
    distortion = diag.check_mesh_distortion()  # topology checks
    print(diag.format_report(report, anomalies, distortion))
"""

from dataclasses import dataclass, field

import numpy as np

from faceforge.body.soft_tissue import SoftTissueSkinning, SkinBinding, SkinJoint


@dataclass
class ChainBindingStats:
    """Per-chain vertex binding statistics for a single mesh."""
    chain_id: int
    chain_name: str
    vertex_count: int
    mean_distance: float  # mean rest-pose distance from vertex to assigned segment
    max_distance: float   # max rest-pose distance from vertex to assigned segment
    vertex_indices: np.ndarray = field(repr=False, default_factory=lambda: np.array([], dtype=np.int32))


@dataclass
class BindingReport:
    """Binding analysis for a single registered mesh."""
    mesh_name: str
    total_vertices: int
    chain_stats: list[ChainBindingStats]
    cross_chain_blend_count: int  # how many vertices blend between chains
    mean_primary_weight: float


@dataclass
class DisplacementAnomaly:
    """A set of vertices in a mesh that moved beyond the expected threshold."""
    mesh_name: str
    vertex_count: int          # total vertices in mesh
    anomaly_count: int         # vertices exceeding threshold
    max_displacement: float    # largest displacement observed
    mean_displacement: float   # mean displacement of anomalous vertices
    chain_breakdown: dict[str, int]  # chain_name → count of anomalous vertices
    worst_vertex_idx: int      # index of the worst offending vertex
    worst_vertex_rest: np.ndarray    # rest position of worst vertex
    worst_vertex_current: np.ndarray # current position of worst vertex
    worst_joint_name: str      # joint assigned to worst vertex


@dataclass
class StaticVertexAnomaly:
    """Vertices that remain near rest position when their joint has moved."""
    mesh_name: str
    vertex_count: int          # total vertices in mesh
    static_count: int          # vertices that didn't follow their joint
    worst_ratio: float         # worst vertex_disp / joint_disp ratio (lower = more stuck)
    joint_breakdown: dict[str, int]  # joint_name → count of stuck vertices


@dataclass
class NeighborStretchAnomaly:
    """Vertices stretched anomalously far from their mesh neighbors."""
    mesh_name: str
    vertex_count: int           # total vertices in mesh
    stretched_count: int        # vertices exceeding stretch threshold
    max_stretch_ratio: float    # worst stretch ratio
    joint_breakdown: dict[str, int]  # joint_name → count of stretched vertices
    region_breakdown: dict[str, int]  # body_region → count of stretched vertices


@dataclass
class MeshDistortionReport:
    """Per-mesh topological quality analysis: edge stretch + triangle quality."""
    mesh_name: str
    total_edges: int
    total_triangles: int
    # Edge stretch
    stretched_edge_count: int = 0   # edges with ratio > stretch_threshold
    compressed_edge_count: int = 0  # edges with ratio < compress_threshold
    max_stretch_ratio: float = 1.0  # worst stretched edge
    min_stretch_ratio: float = 1.0  # worst compressed edge
    mean_stretch_ratio: float = 1.0
    p95_stretch_ratio: float = 1.0  # 95th percentile
    p5_stretch_ratio: float = 1.0   # 5th percentile
    # Triangle quality
    inverted_tri_count: int = 0     # face normals flipped vs rest
    degenerate_tri_count: int = 0   # area collapsed to near-zero
    max_area_ratio: float = 1.0
    min_area_ratio: float = 1.0
    # Chain boundary analysis
    cross_chain_edge_count: int = 0   # edges where vertices are on different chains
    cross_chain_stretched: int = 0    # of those, how many exceed stretch threshold
    cross_chain_details: dict[str, int] = field(default_factory=dict)
    # Worst offenders
    worst_edge_verts: tuple[int, int] = (0, 0)
    worst_edge_rest_len: float = 0.0
    worst_edge_posed_len: float = 0.0


class SkinningDiagnostic:
    """Diagnostic tool for the soft tissue skinning system.

    Attach to a SoftTissueSkinning instance after registration to inspect
    binding assignments and detect abnormal displacements at runtime.
    """

    def __init__(self, skinning: SoftTissueSkinning):
        self.skinning = skinning
        # Build chain name lookup from joints
        self._chain_names: dict[int, str] = {}
        self._build_chain_names()

    def _build_chain_names(self) -> None:
        """Build chain_id → descriptive name mapping from joint names."""
        chain_joints: dict[int, list[str]] = {}
        for j in self.skinning.joints:
            chain_joints.setdefault(j.chain_id, []).append(j.name)
        for cid, names in chain_joints.items():
            # Infer chain name from first joint name
            first = names[0]
            if "thoracic" in first or "lumbar" in first:
                self._chain_names[cid] = "spine"
            elif "shoulder" in first or "elbow" in first or "wrist" in first:
                side = first.split("_")[-1] if "_" in first else "?"
                self._chain_names[cid] = f"arm_{side}"
            elif "hip" in first or "knee" in first or "ankle" in first:
                side = first.split("_")[-1] if "_" in first else "?"
                self._chain_names[cid] = f"leg_{side}"
            elif "finger" in first:
                parts = first.split("_")
                self._chain_names[cid] = f"hand_{parts[1]}_{parts[2]}" if len(parts) >= 3 else "hand"
            elif "toe" in first:
                parts = first.split("_")
                self._chain_names[cid] = f"foot_{parts[1]}_{parts[2]}" if len(parts) >= 3 else "foot"
            elif "rib" in first:
                self._chain_names[cid] = "ribs"
            else:
                self._chain_names[cid] = f"chain_{cid}"

    def chain_name(self, chain_id: int) -> str:
        return self._chain_names.get(chain_id, f"chain_{chain_id}")

    def analyze_bindings(self) -> list[BindingReport]:
        """Analyze all registered mesh bindings for potential issues.

        Returns a BindingReport per registered mesh showing which chains
        each vertex is bound to, distances, and blending statistics.
        """
        reports = []
        joints = self.skinning.joints

        for binding in self.skinning.bindings:
            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue

            rest_pos = mesh.rest_positions.reshape(-1, 3)
            V = len(rest_pos)
            ji = binding.joint_indices
            si = binding.secondary_indices
            w = binding.weights

            # Compute per-vertex distance to assigned joint position
            joint_positions = np.array([
                joints[idx].rest_world[:3, 3] for idx in ji
            ])
            dists = np.linalg.norm(rest_pos - joint_positions, axis=1)

            # Group by chain
            chain_ids_per_vert = np.array([joints[idx].chain_id for idx in ji])
            unique_chains = np.unique(chain_ids_per_vert)

            chain_stats = []
            for cid in unique_chains:
                mask = chain_ids_per_vert == cid
                chain_stats.append(ChainBindingStats(
                    chain_id=int(cid),
                    chain_name=self.chain_name(int(cid)),
                    vertex_count=int(mask.sum()),
                    mean_distance=float(dists[mask].mean()),
                    max_distance=float(dists[mask].max()),
                    vertex_indices=np.where(mask)[0],
                ))

            # Cross-chain blending: vertices where primary and secondary are different chains
            sec_chain_ids = np.array([joints[idx].chain_id for idx in si])
            cross_chain = (chain_ids_per_vert != sec_chain_ids) & (w < 0.999)

            reports.append(BindingReport(
                mesh_name=mesh.name,
                total_vertices=V,
                chain_stats=sorted(chain_stats, key=lambda s: -s.vertex_count),
                cross_chain_blend_count=int(cross_chain.sum()),
                mean_primary_weight=float(w.mean()),
            ))

        return reports

    def check_displacements(
        self,
        max_displacement: float = 5.0,
        relative: bool = True,
    ) -> list[DisplacementAnomaly]:
        """Check all registered meshes for abnormal vertex displacements.

        Parameters
        ----------
        max_displacement : float
            Threshold for flagging. If ``relative=True``, this is a multiplier
            on the rest-pose distance from vertex to its assigned joint
            (default 5x). If ``relative=False``, this is an absolute distance
            in world units.
        relative : bool
            If True, uses joint-relative threshold: the displacement expected
            from the joint's own movement plus rotational amplification scaled
            by the vertex's rest distance from the joint. This correctly
            handles extremity vertices (hands/feet) that undergo large but
            correct displacements when their parent limb moves.
            If False, threshold is an absolute world-unit distance.

        Returns
        -------
        list[DisplacementAnomaly]
            One entry per mesh that has any anomalous vertices.
        """
        anomalies = []
        joints = self.skinning.joints

        # Precompute per-joint displacement from rest (how far each joint moved)
        joint_disps = np.zeros(len(joints), dtype=np.float64)
        for i, j in enumerate(joints):
            j.node.update_world_matrix()
            joint_disps[i] = np.linalg.norm(
                j.node.world_matrix[:3, 3] - j.rest_world[:3, 3],
            )

        for binding in self.skinning.bindings:
            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue

            rest_pos = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
            curr_pos = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
            V = len(rest_pos)

            if len(curr_pos) != V:
                continue

            # Displacement from rest
            disp = np.linalg.norm(curr_pos - rest_pos, axis=1)

            if relative:
                # Compare displacement to rest-pose distance from assigned joint
                joint_positions = np.array([
                    joints[idx].rest_world[:3, 3] for idx in binding.joint_indices
                ])
                rest_dists = np.linalg.norm(rest_pos - joint_positions, axis=1)

                # Per-vertex joint displacement (how far the joint itself moved)
                per_vert_jdisp = joint_disps[binding.joint_indices]

                # Joint-relative threshold:
                # A vertex should move roughly as far as its joint (translation)
                # plus rotational amplification proportional to distance.
                # For a 90° rotation at distance d, displacement ≈ d * sqrt(2).
                # Use max_displacement as a generous factor for the distance
                # component, and add the joint's own translation.
                thresholds = (
                    per_vert_jdisp
                    + np.maximum(rest_dists * max_displacement, 2.0)
                )
                bad = disp > thresholds
            else:
                bad = disp > max_displacement

            if not np.any(bad):
                continue

            bad_count = int(bad.sum())
            bad_disps = disp[bad]
            worst_idx = int(np.argmax(disp))

            # Chain breakdown for anomalous vertices
            chain_ids_per_vert = np.array([
                joints[idx].chain_id for idx in binding.joint_indices
            ])
            bad_chains = chain_ids_per_vert[bad]
            unique_bad, counts = np.unique(bad_chains, return_counts=True)
            chain_breakdown = {
                self.chain_name(int(c)): int(n)
                for c, n in zip(unique_bad, counts)
            }

            anomalies.append(DisplacementAnomaly(
                mesh_name=mesh.name,
                vertex_count=V,
                anomaly_count=bad_count,
                max_displacement=float(disp[worst_idx]),
                mean_displacement=float(bad_disps.mean()),
                chain_breakdown=chain_breakdown,
                worst_vertex_idx=worst_idx,
                worst_vertex_rest=rest_pos[worst_idx].copy(),
                worst_vertex_current=curr_pos[worst_idx].copy(),
                worst_joint_name=joints[binding.joint_indices[worst_idx]].name,
            ))

        return anomalies

    def check_cross_body_bindings(
        self,
        lateral_threshold: float = 5.0,
        midline_tolerance: float = 5.0,
    ) -> dict[str, int]:
        """Count vertices bound to chains on the opposite body side.

        A cross-body binding is a vertex at X > midline_tolerance bound to
        a chain whose joint centroid is at X < -lateral_threshold (or vice
        versa).  These are the most problematic binding errors because they
        cause skin on one side to deform when the opposite limb moves.

        Returns dict of chain_name → count of cross-body bound vertices.
        """
        joints = self.skinning.joints

        # Compute chain X centroids
        chain_x: dict[int, list[float]] = {}
        for j in joints:
            chain_x.setdefault(j.chain_id, []).append(float(j.rest_world[0, 3]))
        chain_centroids = {cid: np.mean(xs) for cid, xs in chain_x.items()}

        cross_body: dict[str, int] = {}
        for binding in self.skinning.bindings:
            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue
            rest_pos = mesh.rest_positions.reshape(-1, 3)
            vert_x = rest_pos[:, 0]

            for vi in range(len(rest_pos)):
                ji = binding.joint_indices[vi]
                cid = joints[ji].chain_id
                cx = chain_centroids.get(cid, 0.0)
                vx = vert_x[vi]

                is_cross = False
                if cx < -lateral_threshold and vx > midline_tolerance:
                    is_cross = True  # right chain grabbing left vertex
                elif cx > lateral_threshold and vx < -midline_tolerance:
                    is_cross = True  # left chain grabbing right vertex

                if is_cross:
                    name = self.chain_name(cid)
                    cross_body[name] = cross_body.get(name, 0) + 1

        return cross_body

    def check_static_vertices(
        self,
        joint_move_threshold: float = 2.0,
        vertex_ratio_threshold: float = 0.1,
        strategy: str = "nearest",
        proximity_limit: float = 30.0,
    ) -> list[StaticVertexAnomaly]:
        """Detect vertices that remain near rest position when they should move.

        Two strategies are available:

        - ``"assigned"`` (legacy): compares each vertex to its assigned joint.
          Misses vertices incorrectly assigned to a non-moving joint.
        - ``"nearest"`` (default): compares each vertex to the nearest joint
          that actually moved, regardless of assignment.  Catches vertices
          bound to the wrong joint (e.g. pelvis skin on lumbar_5).

        Parameters
        ----------
        joint_move_threshold : float
            Minimum joint displacement (world units) to consider a joint as
            "moving".  Joints that moved less than this are ignored.
        vertex_ratio_threshold : float
            Maximum vertex_disp / reference_joint_disp ratio to flag as stuck.
            0.1 means the vertex moved less than 10% of the reference joint.
        strategy : str
            ``"assigned"`` or ``"nearest"`` (default).
        proximity_limit : float
            For ``"nearest"`` strategy only: maximum rest-pose distance from
            vertex to a moving joint to consider (world units).
        """
        results: list[StaticVertexAnomaly] = []
        joints = self.skinning.joints

        # Pre-compute per-joint displacement
        joint_disps = np.zeros(len(joints), dtype=np.float64)
        for j_idx, joint in enumerate(joints):
            joint.node.update_world_matrix()
            current_pos = joint.node.world_matrix[:3, 3]
            rest_pos = joint.rest_world[:3, 3]
            joint_disps[j_idx] = float(np.linalg.norm(current_pos - rest_pos))

        # For "nearest" strategy, pre-compute moved joint positions
        if strategy == "nearest":
            moved_mask = joint_disps > joint_move_threshold
            if not np.any(moved_mask):
                return results
            moved_indices = np.where(moved_mask)[0]
            moved_positions = np.array(
                [joints[i].rest_world[:3, 3] for i in moved_indices],
                dtype=np.float64,
            )
            moved_disps = joint_disps[moved_indices]

        for binding in self.skinning.bindings:
            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue

            rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
            current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
            V = len(rest)

            # Per-vertex displacement
            vert_disps = np.linalg.norm(current - rest, axis=1)

            if strategy == "nearest":
                # Find nearest moved joint for each vertex
                diff = rest[:, np.newaxis, :] - moved_positions[np.newaxis, :, :]
                dist_to_moved = np.sqrt(np.sum(diff * diff, axis=2))  # (V, J_moved)
                nearest_idx = np.argmin(dist_to_moved, axis=1)  # (V,)
                nearest_joint_disp = moved_disps[nearest_idx]  # (V,)
                nearest_dist = dist_to_moved[np.arange(V), nearest_idx]  # (V,)

                static_mask = (
                    (nearest_dist < proximity_limit)
                    & (nearest_joint_disp > joint_move_threshold)
                    & (vert_disps < vertex_ratio_threshold * nearest_joint_disp)
                )
                # For breakdown, map to actual joint indices
                ref_ji = moved_indices[nearest_idx]
            else:
                # Legacy "assigned" strategy
                ji = binding.joint_indices
                vert_joint_disps = joint_disps[ji]
                static_mask = (
                    (vert_joint_disps > joint_move_threshold)
                    & (vert_disps < vertex_ratio_threshold * vert_joint_disps)
                )
                ref_ji = ji

            n_static = int(np.sum(static_mask))

            if n_static > 0:
                # Compute worst ratio
                if strategy == "nearest":
                    ref_disps = nearest_joint_disp
                else:
                    ref_disps = joint_disps[binding.joint_indices]
                ratios = np.where(
                    ref_disps > 1e-6,
                    vert_disps / ref_disps,
                    1.0,
                )
                worst_ratio = float(ratios[static_mask].min())

                # Breakdown by nearest moving joint (or assigned joint)
                static_ref = ref_ji[static_mask]
                joint_breakdown: dict[str, int] = {}
                for j_idx in np.unique(static_ref):
                    j = joints[j_idx]
                    count = int(np.sum(static_ref == j_idx))
                    joint_breakdown[j.name] = count

                results.append(StaticVertexAnomaly(
                    mesh_name=mesh.name,
                    vertex_count=V,
                    static_count=n_static,
                    worst_ratio=worst_ratio,
                    joint_breakdown=joint_breakdown,
                ))

        return results

    # Body region definitions for breakdown reporting (Z coordinates)
    _BODY_REGIONS = {
        "head":       (-15.0, float("inf")),
        "neck":       (-25.0, -15.0),
        "upper_torso":(-45.0, -25.0),
        "lower_torso":(-80.0, -45.0),
        "pelvis":     (-95.0, -80.0),
        "upper_leg":  (-140.0, -95.0),
        "lower_leg":  (-190.0, -140.0),
        "foot":       (float("-inf"), -190.0),
    }

    def check_neighbor_stretch(
        self,
        max_stretch: float = 3.0,
    ) -> list[NeighborStretchAnomaly]:
        """Detect vertices stretched too far from their mesh neighbors.

        For each vertex, computes its distance to its mesh-neighbor average
        in the current pose and compares to rest-pose baseline.  Vertices
        with a stretch ratio exceeding *max_stretch* are flagged as anomalous
        — they are likely mis-bound to the wrong kinematic chain (e.g. arm
        chain pulling hip skin).

        Unlike ``check_static_vertices`` (which detects vertices that don't
        move *enough*), this detects vertices that move *too much* relative
        to their neighbors — the complement problem.

        Parameters
        ----------
        max_stretch : float
            Maximum allowed ratio of current / rest neighbor-average distance.
            3.0 means a vertex can be at most 3× its rest-pose neighbor
            distance before being flagged.
        """
        results: list[NeighborStretchAnomaly] = []
        joints = self.skinning.joints

        for binding in self.skinning.bindings:
            if binding.edge_pairs is None:
                continue

            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue

            rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
            current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
            V = len(rest)

            edges = binding.edge_pairs
            counts = binding.neighbor_counts

            # Current neighbor averages
            neighbor_sum = np.zeros((V, 3), dtype=np.float64)
            np.add.at(neighbor_sum, edges[:, 0], current[edges[:, 1]])
            np.add.at(neighbor_sum, edges[:, 1], current[edges[:, 0]])

            has_neighbors = counts > 0
            neighbor_avg = np.zeros_like(neighbor_sum)
            neighbor_avg[has_neighbors] = (
                neighbor_sum[has_neighbors]
                / counts[has_neighbors, np.newaxis]
            )

            current_dist = np.linalg.norm(current - neighbor_avg, axis=1)
            rest_dist = binding.rest_neighbor_dist

            stretch = current_dist / (rest_dist + 0.01)
            # Skip vertices with near-zero rest_neighbor_dist (inflated ratios)
            meaningful = rest_dist > 0.05
            stretched_mask = has_neighbors & meaningful & (stretch > max_stretch)
            n_stretched = int(np.sum(stretched_mask))

            if n_stretched > 0:
                max_ratio = float(stretch[stretched_mask].max())

                # Joint breakdown
                ji = binding.joint_indices[stretched_mask]
                joint_breakdown: dict[str, int] = {}
                for j_idx in np.unique(ji):
                    j = joints[j_idx]
                    count = int(np.sum(ji == j_idx))
                    joint_breakdown[j.name] = count

                # Region breakdown
                stretched_z = rest[stretched_mask, 2]
                region_breakdown: dict[str, int] = {}
                for rname, (zlo, zhi) in self._BODY_REGIONS.items():
                    n = int(np.sum((stretched_z >= zlo) & (stretched_z < zhi)))
                    if n > 0:
                        region_breakdown[rname] = n

                results.append(NeighborStretchAnomaly(
                    mesh_name=mesh.name,
                    vertex_count=V,
                    stretched_count=n_stretched,
                    max_stretch_ratio=max_ratio,
                    joint_breakdown=joint_breakdown,
                    region_breakdown=region_breakdown,
                ))

        return results

    def check_mesh_distortion(
        self,
        stretch_threshold: float = 3.0,
        compress_threshold: float = 0.3,
        degenerate_area_ratio: float = 0.01,
        min_edge_length: float = 0.5,
        min_tri_area: float = 0.1,
    ) -> list[MeshDistortionReport]:
        """Check all registered meshes for topological distortion.

        Analyzes edge stretch ratios and triangle quality to detect:
        - Edge tearing (stretch ratio > threshold)
        - Edge collapse (stretch ratio < compress threshold)
        - Triangle inversion (face normal flip)
        - Triangle degeneration (area collapse to near-zero)

        These catch visual artifacts that per-vertex displacement checks miss:
        two adjacent vertices can each be at individually reasonable positions
        while the mesh between them is visually torn or collapsed.

        Size-aware filtering: very short edges and tiny triangles are excluded
        since their distortion is visually insignificant.  Most meshes have
        many near-degenerate slivers from STL tessellation that flip trivially.

        Parameters
        ----------
        stretch_threshold : float
            Flag edges with posed_length / rest_length above this (default 3.0).
        compress_threshold : float
            Flag edges with posed_length / rest_length below this (default 0.3).
        degenerate_area_ratio : float
            Flag triangles where posed_area / rest_area < this (default 0.01).
        min_edge_length : float
            Only check edges with rest-pose length >= this value (default 0.5).
            Filters out near-degenerate edges that produce extreme ratios
            but negligible visual impact.
        min_tri_area : float
            Only check triangles with rest-pose area >= this value (default 0.1).
            Filters out slivers that flip trivially during any deformation.

        Returns
        -------
        list[MeshDistortionReport]
            One entry per mesh that has any distortion. Clean meshes are omitted.
        """
        reports: list[MeshDistortionReport] = []
        joints = self.skinning.joints

        for binding in self.skinning.bindings:
            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue

            rest_pos = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
            curr_pos = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
            V = len(rest_pos)

            if len(curr_pos) != V:
                continue

            # ── Extract triangles and edges ──
            geom = mesh.geometry
            if geom.has_indices:
                tris = geom.indices.reshape(-1, 3)
            else:
                tris = np.arange(V).reshape(-1, 3)

            T = len(tris)
            if T == 0:
                continue

            # Build unique edge list from triangles
            raw_edges = np.concatenate([
                tris[:, [0, 1]],
                tris[:, [1, 2]],
                tris[:, [2, 0]],
            ], axis=0)  # (3T, 2)
            raw_edges = np.sort(raw_edges, axis=1)
            unique_edges = np.unique(raw_edges, axis=0)  # (E, 2)
            E = len(unique_edges)

            # ── Edge stretch analysis ──
            e0 = unique_edges[:, 0]
            e1 = unique_edges[:, 1]
            rest_edge_vec = rest_pos[e1] - rest_pos[e0]
            curr_edge_vec = curr_pos[e1] - curr_pos[e0]
            rest_edge_len = np.linalg.norm(rest_edge_vec, axis=1)
            curr_edge_len = np.linalg.norm(curr_edge_vec, axis=1)

            # Avoid division by zero for degenerate rest edges
            safe_rest = np.maximum(rest_edge_len, 1e-10)
            stretch_ratios = curr_edge_len / safe_rest

            # Size-aware filtering: only flag edges with meaningful rest length
            valid_edges = rest_edge_len >= min_edge_length
            stretched = valid_edges & (stretch_ratios > stretch_threshold)
            compressed = valid_edges & (stretch_ratios < compress_threshold)
            stretched_count = int(stretched.sum())
            compressed_count = int(compressed.sum())

            # ── Triangle normal analysis ──
            v0_rest = rest_pos[tris[:, 0]]
            v1_rest = rest_pos[tris[:, 1]]
            v2_rest = rest_pos[tris[:, 2]]
            rest_cross = np.cross(v1_rest - v0_rest, v2_rest - v0_rest)
            rest_area = np.linalg.norm(rest_cross, axis=1) * 0.5

            v0_curr = curr_pos[tris[:, 0]]
            v1_curr = curr_pos[tris[:, 1]]
            v2_curr = curr_pos[tris[:, 2]]
            curr_cross = np.cross(v1_curr - v0_curr, v2_curr - v0_curr)
            curr_area = np.linalg.norm(curr_cross, axis=1) * 0.5

            # Normal flip: dot product of rest and posed cross products
            dots = np.sum(rest_cross * curr_cross, axis=1)
            # Size-aware: only check triangles above minimum area threshold
            meaningful = rest_area >= max(min_tri_area, 1e-8)
            inverted = meaningful & (dots < 0)
            inverted_count = int(inverted.sum())

            # Area ratio
            safe_rest_area = np.maximum(rest_area, 1e-10)
            area_ratios = curr_area / safe_rest_area
            degenerate = meaningful & (area_ratios < degenerate_area_ratio)
            degenerate_count = int(degenerate.sum())

            # ── Chain boundary edge analysis ──
            chain_per_vert = np.array(
                [joints[idx].chain_id for idx in binding.joint_indices],
                dtype=np.int32,
            )
            chain_e0 = chain_per_vert[e0]
            chain_e1 = chain_per_vert[e1]
            cross_chain_edges = chain_e0 != chain_e1
            cross_chain_count = int(cross_chain_edges.sum())

            # Cross-chain edges that are also distorted (using size-filtered flags)
            cross_chain_bad = cross_chain_edges & (stretched | compressed)
            cross_chain_stretched_count = int(cross_chain_bad.sum())

            # Breakdown by chain pair
            cross_chain_details: dict[str, int] = {}
            if cross_chain_stretched_count > 0:
                bad_idx = np.where(cross_chain_bad)[0]
                for idx in bad_idx:
                    c0 = self.chain_name(int(chain_e0[idx]))
                    c1 = self.chain_name(int(chain_e1[idx]))
                    pair = f"{min(c0, c1)}<->{max(c0, c1)}"
                    cross_chain_details[pair] = cross_chain_details.get(pair, 0) + 1

            # ── Skip clean meshes ──
            has_problems = (
                stretched_count > 0
                or compressed_count > 0
                or inverted_count > 0
                or degenerate_count > 0
            )
            if not has_problems:
                continue

            # ── Worst edge ──
            worst_idx = int(np.argmax(stretch_ratios))
            worst_verts = (int(e0[worst_idx]), int(e1[worst_idx]))

            reports.append(MeshDistortionReport(
                mesh_name=mesh.name,
                total_edges=E,
                total_triangles=T,
                stretched_edge_count=stretched_count,
                compressed_edge_count=compressed_count,
                max_stretch_ratio=float(stretch_ratios.max()),
                min_stretch_ratio=float(stretch_ratios.min()),
                mean_stretch_ratio=float(stretch_ratios.mean()),
                p95_stretch_ratio=float(np.percentile(stretch_ratios, 95)),
                p5_stretch_ratio=float(np.percentile(stretch_ratios, 5)),
                inverted_tri_count=inverted_count,
                degenerate_tri_count=degenerate_count,
                max_area_ratio=float(area_ratios[meaningful].max()) if meaningful.any() else 1.0,
                min_area_ratio=float(area_ratios[meaningful].min()) if meaningful.any() else 1.0,
                cross_chain_edge_count=cross_chain_count,
                cross_chain_stretched=cross_chain_stretched_count,
                cross_chain_details=cross_chain_details,
                worst_edge_verts=worst_verts,
                worst_edge_rest_len=float(rest_edge_len[worst_idx]),
                worst_edge_posed_len=float(curr_edge_len[worst_idx]),
            ))

        return reports

    def get_chain_vertical_ranges(self) -> dict[str, tuple[float, float]]:
        """Return the Z-axis (vertical) extent of each chain's joints.

        Useful for understanding spatial overlap between chains.
        Returns dict of chain_name → (z_min, z_max).
        """
        chain_z: dict[int, list[float]] = {}
        for j in self.skinning.joints:
            chain_z.setdefault(j.chain_id, []).append(float(j.rest_world[2, 3]))
        return {
            self.chain_name(cid): (min(zvals), max(zvals))
            for cid, zvals in chain_z.items()
        }

    def get_binding_summary(self, binding_idx: int = -1) -> dict:
        """Return a detailed summary dict for a specific binding (default: last).

        Includes per-chain vertex counts, Z-ranges, and cross-chain stats.
        Useful for debugging specific meshes (e.g., skin).
        """
        if not self.skinning.bindings:
            return {}
        binding = self.skinning.bindings[binding_idx]
        mesh = binding.mesh
        if mesh.rest_positions is None:
            return {"mesh": mesh.name, "error": "no rest positions"}

        rest_pos = mesh.rest_positions.reshape(-1, 3)
        ji = binding.joint_indices
        joints = self.skinning.joints

        chain_ids = np.array([joints[idx].chain_id for idx in ji])
        unique_chains = np.unique(chain_ids)

        per_chain = {}
        for cid in unique_chains:
            mask = chain_ids == cid
            verts = rest_pos[mask]
            per_chain[self.chain_name(int(cid))] = {
                "vertex_count": int(mask.sum()),
                "z_range": (float(verts[:, 2].min()), float(verts[:, 2].max())),
                "x_range": (float(verts[:, 0].min()), float(verts[:, 0].max())),
                "y_range": (float(verts[:, 1].min()), float(verts[:, 1].max())),
            }

        return {
            "mesh": mesh.name,
            "total_vertices": len(rest_pos),
            "is_muscle": binding.is_muscle,
            "per_chain": per_chain,
        }

    def format_report(
        self,
        binding_reports: list[BindingReport] | None = None,
        anomalies: list[DisplacementAnomaly] | None = None,
        distortion: list[MeshDistortionReport] | None = None,
        static_verts: list[StaticVertexAnomaly] | None = None,
        neighbor_stretch: list[NeighborStretchAnomaly] | None = None,
    ) -> str:
        """Generate a human-readable diagnostic report string."""
        lines = ["=" * 60, "SKINNING DIAGNOSTIC REPORT", "=" * 60, ""]

        # Cross-body binding check
        cross_body = self.check_cross_body_bindings()
        if cross_body:
            total_cb = sum(cross_body.values())
            lines.append(f"CROSS-BODY BINDINGS: {total_cb} vertices (ERRORS)")
            lines.append("-" * 40)
            for name, cnt in sorted(cross_body.items(), key=lambda x: -x[1]):
                lines.append(f"  {name:20s}: {cnt}")
            lines.append("")
        else:
            lines.append("CROSS-BODY BINDINGS: 0 (clean)")
            lines.append("")

        # Chain vertical ranges
        ranges = self.get_chain_vertical_ranges()
        if ranges:
            lines.append("Chain Vertical Ranges (Z-axis):")
            for name, (zmin, zmax) in sorted(ranges.items()):
                lines.append(f"  {name:20s}: Z=[{zmin:7.1f}, {zmax:7.1f}]")
            lines.append("")

        # Binding reports
        if binding_reports:
            lines.append("BINDING ANALYSIS:")
            lines.append("-" * 40)
            for br in binding_reports:
                lines.append(f"\nMesh: {br.mesh_name} ({br.total_vertices} verts)")
                lines.append(f"  Cross-chain blended: {br.cross_chain_blend_count}")
                lines.append(f"  Mean primary weight: {br.mean_primary_weight:.3f}")
                lines.append("  Chain assignments:")
                for cs in br.chain_stats:
                    lines.append(
                        f"    {cs.chain_name:20s}: {cs.vertex_count:6d} verts, "
                        f"dist mean={cs.mean_distance:.1f} max={cs.max_distance:.1f}"
                    )
            lines.append("")

        # Displacement anomalies
        if anomalies:
            lines.append("DISPLACEMENT ANOMALIES:")
            lines.append("-" * 40)
            for a in anomalies:
                pct = 100 * a.anomaly_count / max(1, a.vertex_count)
                lines.append(f"\nMesh: {a.mesh_name}")
                lines.append(f"  Anomalous: {a.anomaly_count}/{a.vertex_count} ({pct:.1f}%)")
                lines.append(f"  Max displacement: {a.max_displacement:.2f}")
                lines.append(f"  Mean displacement: {a.mean_displacement:.2f}")
                lines.append(f"  Worst vertex #{a.worst_vertex_idx}: "
                             f"joint={a.worst_joint_name}")
                lines.append(f"    rest={a.worst_vertex_rest}")
                lines.append(f"    curr={a.worst_vertex_current}")
                lines.append("  By chain:")
                for cn, cnt in sorted(a.chain_breakdown.items(), key=lambda x: -x[1]):
                    lines.append(f"    {cn:20s}: {cnt} anomalous vertices")
        elif anomalies is not None:
            lines.append("No displacement anomalies detected.")

        # Mesh distortion
        if distortion:
            lines.append("")
            lines.append("MESH DISTORTION:")
            lines.append("-" * 40)
            total_stretched = sum(d.stretched_edge_count for d in distortion)
            total_compressed = sum(d.compressed_edge_count for d in distortion)
            total_inverted = sum(d.inverted_tri_count for d in distortion)
            total_degenerate = sum(d.degenerate_tri_count for d in distortion)
            lines.append(f"  Total stretched edges:    {total_stretched}")
            lines.append(f"  Total compressed edges:   {total_compressed}")
            lines.append(f"  Total inverted triangles: {total_inverted}")
            lines.append(f"  Total degenerate tris:    {total_degenerate}")
            lines.append("")
            for d in sorted(distortion, key=lambda x: -(x.stretched_edge_count + x.compressed_edge_count + x.inverted_tri_count)):
                lines.append(f"  Mesh: {d.mesh_name}")
                lines.append(f"    Edges: {d.total_edges} total, "
                             f"{d.stretched_edge_count} stretched (>{d.max_stretch_ratio:.1f}x max), "
                             f"{d.compressed_edge_count} compressed (<{d.min_stretch_ratio:.2f}x min)")
                lines.append(f"    Stretch stats: mean={d.mean_stretch_ratio:.3f}, "
                             f"p5={d.p5_stretch_ratio:.3f}, p95={d.p95_stretch_ratio:.3f}")
                lines.append(f"    Triangles: {d.inverted_tri_count} inverted, "
                             f"{d.degenerate_tri_count} degenerate "
                             f"(area ratio [{d.min_area_ratio:.4f}, {d.max_area_ratio:.2f}])")
                if d.cross_chain_stretched > 0:
                    lines.append(f"    Cross-chain boundary: {d.cross_chain_stretched}/"
                                 f"{d.cross_chain_edge_count} boundary edges distorted")
                    for pair, cnt in sorted(d.cross_chain_details.items(), key=lambda x: -x[1]):
                        lines.append(f"      {pair}: {cnt} bad edges")
                lines.append(f"    Worst edge: verts ({d.worst_edge_verts[0]}, {d.worst_edge_verts[1]}), "
                             f"rest={d.worst_edge_rest_len:.2f}, posed={d.worst_edge_posed_len:.2f}")
        elif distortion is not None:
            lines.append("")
            lines.append("No mesh distortion detected.")

        # Static vertex anomalies
        if static_verts:
            total_static = sum(s.static_count for s in static_verts)
            lines.append("")
            lines.append(f"STATIC VERTEX ANOMALIES: {total_static} vertices stuck at rest position")
            lines.append("-" * 40)
            for s in sorted(static_verts, key=lambda x: -x.static_count):
                pct = 100.0 * s.static_count / s.vertex_count if s.vertex_count > 0 else 0
                lines.append(f"  {s.mesh_name}: {s.static_count}/{s.vertex_count} "
                             f"({pct:.1f}%) stuck, worst ratio={s.worst_ratio:.4f}")
                for jname, cnt in sorted(s.joint_breakdown.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"    joint={jname}: {cnt} stuck verts")
        elif static_verts is not None:
            lines.append("")
            lines.append("No static vertex anomalies detected.")

        # Neighbor stretch anomalies
        if neighbor_stretch:
            total_stretched = sum(s.stretched_count for s in neighbor_stretch)
            lines.append("")
            lines.append(f"NEIGHBOR STRETCH ANOMALIES: {total_stretched} vertices over-stretched")
            lines.append("-" * 40)
            for s in sorted(neighbor_stretch, key=lambda x: -x.stretched_count):
                pct = 100.0 * s.stretched_count / s.vertex_count if s.vertex_count > 0 else 0
                lines.append(f"  {s.mesh_name}: {s.stretched_count}/{s.vertex_count} "
                             f"({pct:.1f}%) stretched, max_ratio={s.max_stretch_ratio:.1f}x")
                if s.region_breakdown:
                    for rname, cnt in sorted(s.region_breakdown.items(), key=lambda x: -x[1]):
                        lines.append(f"    region={rname}: {cnt}")
                for jname, cnt in sorted(s.joint_breakdown.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"    joint={jname}: {cnt} stretched verts")
        elif neighbor_stretch is not None:
            lines.append("")
            lines.append("No neighbor stretch anomalies detected.")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
