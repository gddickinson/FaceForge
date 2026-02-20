"""Delta-matrix soft tissue skinning for body muscles/organs/vasculature."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from faceforge.core.math_utils import (
    Mat4,
    mat4_identity,
    mat4_inverse,
    batch_mat4_to_dual_quat,
    batch_quat_multiply,
    batch_quat_rotate,
)
from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import BodyState


@dataclass
class SkinJoint:
    """A skeletal joint used for skinning."""
    name: str
    node: SceneNode
    rest_world: Mat4 = field(default_factory=mat4_identity)
    rest_world_inv: Mat4 = field(default_factory=mat4_identity)
    # Bone segment for nearest-bone assignment
    segment_start: Optional[np.ndarray] = None
    segment_end: Optional[np.ndarray] = None
    chain_id: int = 0  # which kinematic chain this joint belongs to


@dataclass
class SkinBinding:
    """Per-mesh skinning data: vertex → joint assignments + weights."""
    mesh: MeshInstance
    joint_indices: np.ndarray  # per-vertex primary joint index
    weights: np.ndarray  # per-vertex blend weight (0..1) for two-bone blending
    secondary_indices: np.ndarray  # per-vertex secondary joint index
    is_muscle: bool = False
    base_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    rest_y_span: float = 1.0  # rest-pose Y extent for activation ratio
    # Neighbor clamping data (built at registration time)
    edge_pairs: Optional[np.ndarray] = None       # (E, 2) unique edge vertex pairs
    neighbor_counts: Optional[np.ndarray] = None   # (V,) neighbors per vertex
    rest_neighbor_dist: Optional[np.ndarray] = None  # (V,) rest distance to neighbor avg
    boundary_blend: Optional[np.ndarray] = None    # (V,) fraction of same-chain neighbors (0=all different, 1=all same)
    smooth_zone: Optional[np.ndarray] = None       # (V,) propagated boundary weight for displacement smoothing
    # Per-muscle head-follow configuration (Layer 1)
    head_follow_config: Optional[dict] = None  # {"upperFrac": float, "lowerFrac": float}
    muscle_name: Optional[str] = None  # for debug/lookup
    # Set True for muscles spanning into digit chains — these skip neighbor
    # clamping so finger curl tendons can move freely.
    skip_neighbor_clamp: bool = False


class SoftTissueSkinning:
    """Delta-matrix skinning system for body soft tissue.

    Mirrors buildSkinJoints/registerSkinMesh/updateBodySoftTissue from JS.

    Algorithm:
    1. buildSkinJoints(): collect spine+limb joints, snapshot rest matrices, build bone segments
    2. registerSkinMesh(): assign vertices to nearest bone, compute blend weights at 15% endpoints
    3. updateBodySoftTissue(): per-frame delta transform (restWorldInv × currentWorld)
    """

    BLEND_ZONE = 0.15  # 15% of bone segment length for within-chain blending (non-muscle)
    CROSS_CHAIN_RADIUS = 20.0  # distance threshold for cross-chain blending (gap-based)
    MAX_CROSS_WEIGHT_MUSCLE = 0.5  # max cross-chain blend for muscles (need to span joints)
    MAX_CROSS_WEIGHT_OTHER = 0.45  # max cross-chain blend for skin/organs

    SPINE_CHAIN = 0  # Convention: chain 0 is always spine

    # Cross-chain divergence clamping: when primary and secondary transforms
    # place a vertex more than DIVERGENCE_MIN apart, reduce secondary influence.
    # With DQS (Dual Quaternion Skinning), large rotations are handled well,
    # so these thresholds can be generous.  The main purpose is preventing
    # extreme divergence from producing distorted geometry in edge cases.
    DIVERGENCE_MIN = 30.0   # start reducing cross-chain blend
    DIVERGENCE_MAX = 100.0  # fully primary (no cross-chain blend)

    # Neighbor-stretch clamping: vertices stretched more than this multiple
    # of their rest-pose distance to neighbor average get snapped back.
    # Interior vertices (all neighbors on same chain) use MAX_NEIGHBOR_STRETCH.
    # Chain-boundary vertices get a relaxed threshold to allow natural
    # transition stretch (arm raise pulling away from torso, etc).
    MAX_NEIGHBOR_STRETCH = 2.0    # tight: catches interior mis-binding spikes
    BOUNDARY_RELAX_FACTOR = 1.5   # boundary verts allow up to MAX * (1 + factor) = 5× stretch
    CLAMP_PASSES = 10  # iterations per frame for cascade convergence

    # Boundary displacement smoothing: iterative Laplacian smoothing of
    # displacements (not positions) at chain-boundary vertices to reduce
    # the visible seams where different chains pull in different directions.
    BOUNDARY_SMOOTH_PASSES = 5    # smoothing iterations per frame
    BOUNDARY_SMOOTH_STRENGTH = 0.5  # blend factor toward neighbor avg displacement
    BOUNDARY_SMOOTH_RINGS = 3     # edge rings to propagate boundary zone outward

    # Medial distance penalty: prevents lateral chains (arm) from binding
    # torso vertices that are closer to the midline than the chain's inner
    # joint boundary.  Without this, arm bone segments passing through the
    # torso cause armpit/lateral-chest vertices to be assigned to the arm
    # chain, creating dramatic stretching when the arm raises.
    #
    # The penalty adds (offset × factor) to the chain's segment distances,
    # where offset = max(0, chain_inner_x - vertex_x) for right-side chains.
    # Only applied to chains with |centroid_x| > MEDIAL_CENTROID_THRESHOLD.
    MEDIAL_PENALTY_FACTOR = 2.0   # penalty units per unit of medial offset
    MEDIAL_CENTROID_THRESHOLD = 15.0  # only penalize chains this far from midline

    # Proximal chain penalty: prevents limb chains (especially legs) from
    # binding vertices that sit above the chain's topmost joint.  The pelvis
    # region between the spine bottom (Z≈-76) and hip joints (Z≈-83) should
    # follow the spine, not the legs.  Without this, the spine overshoot
    # penalty pushes those vertices to leg chains, causing extreme stretching
    # in sitting/crouching poses when the hip flexes.
    #
    # Analogous to medial penalty (which protects X-axis midline from arms)
    # but protects the Z-axis transition between spine and limb chains.
    PROXIMAL_PENALTY_FACTOR = 3.0   # penalty per unit of vertical overshoot
    PROXIMAL_HARD_CUTOFF = 12.0     # above this: exclude chain entirely

    # Geodesic mesh segmentation: uses mesh-edge distances to separate limbs
    # from torso at narrow bridges (armpit, groin) instead of relying on
    # Euclidean distance which is deceptively short when bone segments pass
    # through the torso interior.
    GEODESIC_BLEND = 0.1   # Euclidean weight in hybrid distance (10% Euclidean)
    SEED_RADIUS = 5.0      # max Euclidean dist from bone segment to seed vertex

    def __init__(self):
        self.joints: list[SkinJoint] = []
        self.bindings: list[SkinBinding] = []
        self.chain_count: int = 0
        self._dirty: bool = True
        self._last_signature: tuple = ()
        # Muscle attachment system for bone pinning + stretch limits (Layers 2-3)
        self.attachment_system = None  # MuscleAttachmentSystem or None
        # Bone collision system for penetration resolution (Layer 4)
        self.collision_system = None  # BoneCollisionSystem or None

        # Registration-time filter constants (tunable for optimization).
        # Defaults match the original hard-coded values.
        self.min_spatial: float = 12.0      # min spatial limit for very small chains (foot/hand ~4-12 extent)
        self.spatial_factor: float = 0.25   # fraction of chain Z extent for proportional limit
        self.min_z_pad: float = 8.0         # minimum Z margin for very small chains (foot/hand)
        self.lateral_threshold: float = 5.0 # |centroid_x| above this triggers X filter
        self.midline_tolerance: float = 5.0 # vertices within ± this of midline aren't filtered

    def build_skin_joints(
        self,
        joint_chains: list[list[tuple[str, SceneNode]]],
    ) -> None:
        """Collect joints from multiple chains, build bone segments per-chain.

        Parameters
        ----------
        joint_chains : list of chains
            Each chain is a list of (name, SceneNode) tuples representing a
            connected kinematic chain (e.g. spine, right arm, left leg).
            Bone segments are only built between consecutive joints *within*
            a chain — never across chain boundaries.
        """
        self.joints.clear()

        # Track which global joint indices belong to each chain
        chain_ranges: list[tuple[int, int]] = []  # (start_idx, end_idx) per chain

        for chain_id, chain in enumerate(joint_chains):
            chain_start = len(self.joints)
            for name, node in chain:
                node.update_world_matrix(force=True)
                rest = node.world_matrix.copy()
                joint = SkinJoint(
                    name=name,
                    node=node,
                    rest_world=rest,
                    rest_world_inv=mat4_inverse(rest),
                    chain_id=chain_id,
                )
                self.joints.append(joint)
            chain_end = len(self.joints)
            if chain_end > chain_start:
                chain_ranges.append((chain_start, chain_end))

        self.chain_count = len(chain_ranges)

        # Cache per-joint chain IDs for fast lookup in update()
        self._joint_chain_ids = np.array(
            [j.chain_id for j in self.joints], dtype=np.int32,
        )

        # Build bone segments only within each chain
        for start, end in chain_ranges:
            chain_len = end - start
            # Segments between consecutive joints in this chain
            for i in range(start, end - 1):
                self.joints[i].segment_start = self.joints[i].rest_world[:3, 3].copy()
                self.joints[i].segment_end = self.joints[i + 1].rest_world[:3, 3].copy()
            # Last joint in chain: extend from previous
            if chain_len >= 2:
                last = self.joints[end - 1]
                prev = self.joints[end - 2]
                last.segment_start = last.rest_world[:3, 3].copy()
                direction = last.segment_start - prev.rest_world[:3, 3]
                last.segment_end = last.segment_start + direction
            elif chain_len == 1:
                # Single-joint chain: no segment (will be skipped in registration)
                pass

    def register_skin_mesh(
        self,
        mesh: MeshInstance,
        is_muscle: bool = False,
        allowed_chains: set[int] | None = None,
        spatial_limit: float | None = None,
        chain_z_margin: float | None = None,
        use_geodesic: bool = True,
        head_follow_config: dict | None = None,
        muscle_name: str | None = None,
    ) -> None:
        """Assign each vertex to nearest bone segment with blend weights.

        Parameters
        ----------
        allowed_chains : set of chain IDs, or None for all chains.
            Restricts which kinematic chains this mesh can bind to.
            Use ``{SoftTissueSkinning.SPINE_CHAIN}`` for torso structures
            that should only follow the spine, not limb joints.
        spatial_limit : float or None
            When set, a vertex can only bind to chains that have at least
            one joint within this distance (in world units).  This prevents
            vertices in one body region (e.g. legs) from binding to distant
            chains (e.g. arms) even when ``allowed_chains`` includes both.
            Computed once at registration time (rest pose).
        chain_z_margin : float or None
            When set, uses axis-aligned bounding box filtering on the Z axis
            (vertical): a vertex can only bind to a chain if the vertex's Z
            coordinate falls within the chain's joint Z range expanded by
            this margin.  This robustly prevents arm↔leg cross-binding
            because arm and leg chains occupy different vertical ranges even
            though they're spatially close in anatomical rest position.
            Preferred over spatial_limit for full-body skin meshes.
            The spine chain (chain 0) is never Z-filtered because it spans
            the full vertical range.
        use_geodesic : bool
            When True, uses geodesic (mesh-edge) distances for chain
            assignment instead of pure Euclidean.  This naturally separates
            limbs from the torso at narrow bridges (armpit, groin).
            Defaults to True.  Disable for muscles or debugging.
        """
        if not self.joints or mesh.rest_positions is None:
            return

        positions = mesh.rest_positions.reshape(-1, 3)
        vert_count = len(positions)

        joint_indices = np.zeros(vert_count, dtype=np.int32)
        secondary_indices = np.zeros(vert_count, dtype=np.int32)
        weights = np.ones(vert_count, dtype=np.float32)

        # Build arrays of valid bone segments, filtered by allowed_chains
        seg_starts = []
        seg_ends = []
        seg_indices = []
        seg_chain_ids = []
        for ji, joint in enumerate(self.joints):
            if allowed_chains is not None and joint.chain_id not in allowed_chains:
                continue
            if joint.segment_start is not None and joint.segment_end is not None:
                ab = joint.segment_end - joint.segment_start
                if np.dot(ab, ab) >= 1e-10:
                    seg_starts.append(joint.segment_start)
                    seg_ends.append(joint.segment_end)
                    seg_indices.append(ji)
                    seg_chain_ids.append(joint.chain_id)

        if not seg_starts:
            return

        seg_starts_arr = np.array(seg_starts, dtype=np.float64)  # (S, 3)
        seg_ends_arr = np.array(seg_ends, dtype=np.float64)      # (S, 3)
        seg_idx_arr = np.array(seg_indices, dtype=np.int32)       # (S,)
        seg_chain_arr = np.array(seg_chain_ids, dtype=np.int32)   # (S,)
        ab = seg_ends_arr - seg_starts_arr                        # (S, 3)
        ab_len_sq = np.sum(ab * ab, axis=1)                       # (S,)

        # Vectorized: compute distance from each vertex to each segment
        p_exp = positions[:, np.newaxis, :].astype(np.float64)    # (V, 1, 3)
        ap = p_exp - seg_starts_arr[np.newaxis, :, :]             # (V, S, 3)
        t_vals = np.sum(ap * ab[np.newaxis, :, :], axis=2) / ab_len_sq[np.newaxis, :]  # (V, S)
        t_vals = np.clip(t_vals, 0.0, 1.0)

        closest = seg_starts_arr[np.newaxis, :, :] + t_vals[:, :, np.newaxis] * ab[np.newaxis, :, :]  # (V, S, 3)
        diff = p_exp - closest                                    # (V, S, 3)
        dists = np.sqrt(np.sum(diff * diff, axis=2))              # (V, S)

        # ── Geodesic mesh segmentation ──
        # Replace pure Euclidean dists with hybrid geodesic+Euclidean to
        # correctly separate limbs from torso at narrow mesh bridges.
        _precomputed_edges: np.ndarray | None = None
        # Keep Euclidean dists for spatial-limit proximity checks
        # (geodesic should affect chain RANKING, not chain ELIGIBILITY).
        dists_euclidean = dists
        if use_geodesic and not is_muscle and mesh.geometry.indices is not None:
            edge_result = self._extract_mesh_edges(mesh)
            if edge_result is not None:
                mesh_edges, mesh_edge_lengths = edge_result
                _precomputed_edges = mesh_edges
                dists_euclidean = dists.copy()
                geo_chain = self._geodesic_chain_dists(
                    positions.astype(np.float64),
                    mesh_edges, mesh_edge_lengths,
                    seg_starts_arr, seg_ends_arr, seg_chain_arr,
                )
                # Build hybrid: geodesic for chain separation + Euclidean for
                # segment refinement within a chain.
                # Vertices in disconnected mesh components have inf geodesic
                # distance (Dijkstra can't reach them from any seed) — fall
                # back to pure Euclidean for those vertices.
                unique_chains_geo = np.unique(seg_chain_arr)
                chain_to_idx = {int(c): i for i, c in enumerate(unique_chains_geo)}
                for si in range(len(seg_chain_arr)):
                    ci = chain_to_idx[int(seg_chain_arr[si])]
                    geo_col = geo_chain[:, ci]
                    reachable = np.isfinite(geo_col)
                    dists[reachable, si] = (
                        geo_col[reachable]
                        + self.GEODESIC_BLEND * dists_euclidean[reachable, si]
                    )
                    # Unreachable vertices keep their Euclidean distances

        # ── Spatial limit: mask out chains whose bone segments are too far ──
        # Uses the already-computed segment distances (not joint distances)
        # because long bone segments (e.g. hip→knee) may have a closest
        # point much nearer than either joint endpoint.
        #
        # The limit is PROPORTIONAL to chain size: small chains (hands, feet)
        # get tighter limits so they only bind nearby skin, preventing
        # hand chains from grabbing thigh vertices even on the same side.
        if spatial_limit is not None:
            _MIN_SPATIAL = self.min_spatial
            _SPATIAL_FACTOR = self.spatial_factor

            # Compute chain Z extents for proportional spatial limit
            _chain_z_ext: dict[int, tuple[float, float]] = {}
            for joint in self.joints:
                cid = joint.chain_id
                if allowed_chains is not None and cid not in allowed_chains:
                    continue
                z = float(joint.rest_world[2, 3])
                if cid not in _chain_z_ext:
                    _chain_z_ext[cid] = (z, z)
                else:
                    lo, hi = _chain_z_ext[cid]
                    _chain_z_ext[cid] = (min(lo, z), max(hi, z))

            unique_chains = np.unique(seg_chain_arr)
            for cid in unique_chains:
                if cid == self.SPINE_CHAIN:
                    continue  # Spine is never spatially filtered
                chain_seg_mask = seg_chain_arr == cid  # (S,)
                if not np.any(chain_seg_mask):
                    continue
                # Proportional limit: small chains get tight limits
                if cid in _chain_z_ext:
                    z_lo, z_hi = _chain_z_ext[cid]
                    extent = z_hi - z_lo
                    chain_limit = min(spatial_limit, max(_MIN_SPATIAL, extent * _SPATIAL_FACTOR))
                else:
                    chain_limit = spatial_limit
                # Min Euclidean distance from each vertex to any segment in
                # this chain.  Use Euclidean (not geodesic) so that physical
                # proximity determines chain eligibility — geodesic only
                # affects the ranking among eligible chains.
                min_seg_dist = dists_euclidean[:, chain_seg_mask].min(axis=1)  # (V,)
                too_far = min_seg_dist > chain_limit
                if np.any(too_far):
                    dists[np.ix_(too_far, chain_seg_mask)] = np.inf

        # ── Chain Z-axis AABB filter ──
        # Uses the vertical extent of each chain's joints to restrict binding.
        # In anatomical rest position, arm and leg chains occupy different Z
        # ranges even though they overlap in XY.  Spine chain is never filtered
        # because it spans the full torso.
        #
        # Margin is PROPORTIONAL to chain extent: small chains (hands, feet)
        # get small margins so they don't grab distant vertices.
        # Formula: actual_margin = min(chain_z_margin, max(MIN_Z_PAD, extent * 0.25))
        #
        # X-axis cross-body filtering prevents opposite-side binding: chains
        # whose joints are laterally offset (|centroid_x| > 5) cannot grab
        # vertices on the opposite side of the body midline.  This stops
        # right-hand chains from binding left-thigh vertices and vice versa.
        if chain_z_margin is not None:
            _MIN_Z_PAD = self.min_z_pad
            _LATERAL_THRESHOLD = self.lateral_threshold
            _MIDLINE_TOLERANCE = self.midline_tolerance

            chain_z_ranges: dict[int, tuple[float, float]] = {}
            chain_x_ranges: dict[int, tuple[float, float]] = {}
            chain_x_centroids: dict[int, tuple[float, int]] = {}  # (sum_x, count)
            for joint in self.joints:
                cid = joint.chain_id
                if allowed_chains is not None and cid not in allowed_chains:
                    continue
                z = float(joint.rest_world[2, 3])
                x = float(joint.rest_world[0, 3])
                if cid not in chain_z_ranges:
                    chain_z_ranges[cid] = (z, z)
                    chain_x_ranges[cid] = (x, x)
                    chain_x_centroids[cid] = (x, 1)
                else:
                    lo, hi = chain_z_ranges[cid]
                    chain_z_ranges[cid] = (min(lo, z), max(hi, z))
                    xlo, xhi = chain_x_ranges[cid]
                    chain_x_ranges[cid] = (min(xlo, x), max(xhi, x))
                    sx, cnt = chain_x_centroids[cid]
                    chain_x_centroids[cid] = (sx + x, cnt + 1)

            vert_z = positions[:, 2].astype(np.float64)  # (V,)
            vert_x = positions[:, 0].astype(np.float64)  # (V,)
            for cid, (z_lo, z_hi) in chain_z_ranges.items():
                # Never filter the spine chain — it spans the full body
                if cid == self.SPINE_CHAIN:
                    continue
                # Z-axis filter
                z_extent = z_hi - z_lo
                z_margin = min(chain_z_margin, max(_MIN_Z_PAD, z_extent * 0.25))
                z_outside = (vert_z < z_lo - z_margin) | (vert_z > z_hi + z_margin)

                # Cross-body filter: right-side chains can't grab left-side
                # vertices and vice versa.  Only applied when the chain's
                # centroid is clearly lateral (|centroid_x| > threshold).
                sx, cnt = chain_x_centroids[cid]
                x_centroid = sx / cnt
                x_outside = np.zeros(vert_count, dtype=bool)
                if x_centroid < -_LATERAL_THRESHOLD:
                    # Right-side chain: exclude left-side vertices
                    x_outside = vert_x > _MIDLINE_TOLERANCE
                elif x_centroid > _LATERAL_THRESHOLD:
                    # Left-side chain: exclude right-side vertices
                    x_outside = vert_x < -_MIDLINE_TOLERANCE

                # X-range AABB filter: prevents laterally-offset chains
                # from grabbing medial hip/thigh vertices.  Applied to:
                # 1. Chains with large X extent (> 10, e.g. arm chains)
                # 2. Chains whose joints are all far from midline
                #    (min |X| > 20, e.g. hand/finger chains at X=34-48)
                # This prevents both arm and hand chains from binding
                # torso/pelvis skin.
                x_range_outside = np.zeros(vert_count, dtype=bool)
                if cid in chain_x_ranges:
                    x_lo, x_hi = chain_x_ranges[cid]
                    x_extent = x_hi - x_lo
                    # Chain is laterally far from midline
                    inner_x = min(abs(x_lo), abs(x_hi))
                    if x_extent > 10.0 or inner_x > 20.0:
                        x_margin = max(3.0, x_extent * 0.4)
                        x_range_outside = (
                            (vert_x < x_lo - x_margin)
                            | (vert_x > x_hi + x_margin)
                        )

                outside = z_outside | x_outside | x_range_outside
                if np.any(outside):
                    chain_seg_mask = seg_chain_arr == cid
                    if np.any(chain_seg_mask):
                        dists[np.ix_(outside, chain_seg_mask)] = np.inf

        # ── Spine overshoot penalty ──
        # Vertices below the lowest spine joint (pelvis/buttock region) get a
        # distance penalty on spine segments, making limb chains more competitive.
        # This prevents pelvis skin from binding to the last lumbar vertebra
        # when hip joints are the anatomically correct controllers.
        #
        # Two tiers:
        #   - Soft penalty (overshoot 1-15 units): 2× overshoot added to spine dists
        #   - Hard cutoff (overshoot > 15 units): spine excluded entirely (inf)
        # The 15-unit threshold extends well below the hip joints (Z≈-83)
        # to allow the spine to compete for upper thigh/groin vertices.
        # Combined with the proximal chain penalty, this creates a smooth
        # transition zone at the pelvis where spine influence fades out
        # gradually rather than cutting off abruptly.
        if not is_muscle:
            spine_seg_mask = seg_chain_arr == self.SPINE_CHAIN
            if np.any(spine_seg_mask):
                spine_z_lo = min(
                    float(j.rest_world[2, 3])
                    for j in self.joints
                    if j.chain_id == self.SPINE_CHAIN
                )
                vert_z = positions[:, 2].astype(np.float64)  # (V,)
                overshoot = np.maximum(0.0, spine_z_lo - vert_z)  # positive below spine
                # Hard cutoff: deeply below spine → exclude spine entirely
                deep_overshoot = overshoot > 15.0
                if np.any(deep_overshoot):
                    dists[np.ix_(deep_overshoot, spine_seg_mask)] = np.inf
                # Soft penalty: moderate overshoot → penalize spine segments
                has_overshoot = (overshoot > 1.0) & ~deep_overshoot
                if np.any(has_overshoot):
                    penalty = overshoot[has_overshoot] * 2.0  # (N,)
                    dists[np.ix_(has_overshoot, spine_seg_mask)] += penalty[:, np.newaxis]

        # ── Medial distance penalty for lateral chains ──
        # Arm bone segments pass through the torso, making their segment
        # distances deceptively short for torso vertices.  This adds a
        # distance penalty for vertices that are more medial (closer to
        # the body midline X=0) than the chain's innermost joint.
        #
        # Applied AFTER spatial/Z filters (which use raw distances for
        # eligibility) but BEFORE argmin (which determines assignment).
        # This affects ranking without changing which chains are available.
        #
        # Only applied to chains whose X centroid is far from the midline
        # (|centroid| > threshold), which targets arm chains (|centroid|≈24)
        # but skips legs (|centroid|≈6), spine, and ribs.
        if not is_muscle:
            vert_x_m = positions[:, 0].astype(np.float64)
            _chain_x_info: dict[int, tuple[float, float, float, int]] = {}
            for j in self.joints:
                cid = j.chain_id
                if allowed_chains is not None and cid not in allowed_chains:
                    continue
                x = float(j.rest_world[0, 3])
                if cid not in _chain_x_info:
                    _chain_x_info[cid] = (x, x, x, 1)
                else:
                    mn, mx, sx, cnt = _chain_x_info[cid]
                    _chain_x_info[cid] = (min(mn, x), max(mx, x), sx + x, cnt + 1)

            for cid, (mn, mx, sx, cnt) in _chain_x_info.items():
                if cid == self.SPINE_CHAIN:
                    continue
                centroid_x = sx / cnt
                if abs(centroid_x) < self.MEDIAL_CENTROID_THRESHOLD:
                    continue  # Skip trunk/midline/leg chains

                chain_seg_mask = seg_chain_arr == cid
                if not np.any(chain_seg_mask):
                    continue

                if centroid_x > 0:
                    # Right-side chain: inner boundary is at min X
                    inner_x = mn
                    x_offset = np.maximum(0.0, inner_x - vert_x_m)
                else:
                    # Left-side chain: inner boundary is at max X (least negative)
                    inner_x = mx
                    x_offset = np.maximum(0.0, vert_x_m - inner_x)

                penalty = x_offset * self.MEDIAL_PENALTY_FACTOR
                has_penalty = penalty > 0.0
                if np.any(has_penalty):
                    dists[np.ix_(has_penalty, chain_seg_mask)] += penalty[has_penalty, np.newaxis]

        # ── Proximal chain penalty for non-spine chains ──
        # Penalizes limb chains for vertices above (higher Z than) the chain's
        # topmost joint.  The pelvis/buttock region between spine bottom
        # (Z≈-76) and hip joints (Z≈-83) gets caught in a dead zone: spine
        # overshoot penalty pushes them away from spine, but they should NOT
        # follow the hip joint rotation.  This penalty makes leg chains less
        # competitive for those vertices, restoring spine as the winner.
        #
        # Works for all non-spine chains (arms benefit too: prevents shoulder
        # vertices from binding to arm chains when they're above the shoulder).
        if not is_muscle:
            vert_z_prox = positions[:, 2].astype(np.float64)
            _chain_z_top: dict[int, float] = {}
            for j in self.joints:
                cid = j.chain_id
                if cid == self.SPINE_CHAIN:
                    continue
                if allowed_chains is not None and cid not in allowed_chains:
                    continue
                z = float(j.rest_world[2, 3])
                if cid not in _chain_z_top:
                    _chain_z_top[cid] = z
                else:
                    _chain_z_top[cid] = max(_chain_z_top[cid], z)

            for cid, top_z in _chain_z_top.items():
                chain_seg_mask = seg_chain_arr == cid
                if not np.any(chain_seg_mask):
                    continue

                # Overshoot = how far above the chain's topmost joint
                z_overshoot = vert_z_prox - top_z  # positive = above chain
                # Hard cutoff: far above chain top → exclude chain entirely
                hard_exclude = z_overshoot > self.PROXIMAL_HARD_CUTOFF
                if np.any(hard_exclude):
                    dists[np.ix_(hard_exclude, chain_seg_mask)] = np.inf
                # Soft penalty: vertices above chain top get penalized
                has_overshoot = (z_overshoot > 0.0) & ~hard_exclude
                if np.any(has_overshoot):
                    penalty = z_overshoot[has_overshoot] * self.PROXIMAL_PENALTY_FACTOR
                    dists[np.ix_(has_overshoot, chain_seg_mask)] += penalty[:, np.newaxis]

        # Find nearest segment per vertex
        best_seg = np.argmin(dists, axis=1)                       # (V,)
        best_ji = seg_idx_arr[best_seg]                           # (V,)
        best_t = t_vals[np.arange(vert_count), best_seg]          # (V,)
        best_chain = seg_chain_arr[best_seg]                      # (V,)

        joint_indices[:] = best_ji
        secondary_indices[:] = best_ji  # default: same as primary

        best_dists = dists[np.arange(vert_count), best_seg]  # (V,)

        # Build chain_id array for all joints (for vectorized chain checks)
        all_chain_ids = np.array([j.chain_id for j in self.joints], dtype=np.int32)

        # ── Within-chain blending ──
        prev_ji = best_ji - 1
        next_ji = best_ji + 1
        # Safe clamp for indexing
        prev_ji_safe = np.clip(prev_ji, 0, len(self.joints) - 1)
        next_ji_safe = np.clip(next_ji, 0, len(self.joints) - 1)

        prev_same_chain = (prev_ji >= 0) & (all_chain_ids[prev_ji_safe] == best_chain)
        next_same_chain = (next_ji < len(self.joints)) & (all_chain_ids[next_ji_safe] == best_chain)

        if is_muscle:
            # ── Muscle: full-range smooth blending along entire bone segment ──
            # weight = 1 - t: at segment start (t=0) → 100% primary (start joint);
            # at segment end (t=1) → 100% secondary (end joint).
            # This makes muscles stretch smoothly between joints instead of
            # swinging as rigid blocks.
            has_next = next_same_chain
            weights[has_next] = 1.0 - best_t[has_next]
            secondary_indices[has_next] = next_ji[has_next]
            # Vertices at the last segment in a chain (no next joint) stay rigid,
            # which is correct — they're at the end of the kinematic chain.
        else:
            # ── Non-muscle: small blend zones at segment endpoints only ──
            near_start = (best_t < self.BLEND_ZONE) & prev_same_chain
            weights[near_start] = best_t[near_start] / self.BLEND_ZONE
            secondary_indices[near_start] = prev_ji[near_start]

            near_end = (best_t > (1.0 - self.BLEND_ZONE)) & next_same_chain & ~near_start
            weights[near_end] = (1.0 - best_t[near_end]) / self.BLEND_ZONE
            secondary_indices[near_end] = next_ji[near_end]

        # ── Cross-chain blending for multi-chain meshes ──
        # If allowed_chains has >1 chain, find the nearest segment on a
        # DIFFERENT chain and blend based on the distance GAP between
        # primary and secondary chains.  Full cross-chain influence when
        # equidistant (gap=0), fading to zero when the secondary is
        # CROSS_CHAIN_RADIUS farther than the primary.
        #
        # This gap-based formula gives strong blending at actual chain
        # boundaries (where vertices are equidistant to both chains) and
        # minimal blending for vertices deep inside one chain, regardless
        # of the absolute distance to the secondary chain.
        if allowed_chains is not None and len(allowed_chains) > 1:
            # For each vertex, find the nearest segment on a different chain
            # Mask out same-chain segments: set their distances to infinity
            chain_mask = (seg_chain_arr[np.newaxis, :] == best_chain[:, np.newaxis])  # (V, S)
            dists_other = dists.copy()
            dists_other[chain_mask] = np.inf

            has_other = np.any(~chain_mask, axis=1)
            if np.any(has_other):
                other_seg = np.argmin(dists_other, axis=1)  # (V,)
                other_dist = dists_other[np.arange(vert_count), other_seg]
                other_ji = seg_idx_arr[other_seg]

                # Distance gap: how much farther the secondary chain is
                finite_other = np.isfinite(other_dist)
                safe_other = np.where(finite_other, other_dist, 0.0)
                dist_gap = np.where(finite_other, safe_other - best_dists, np.inf)

                # Blend when the gap is less than CROSS_CHAIN_RADIUS
                blend_mask = has_other & finite_other & (dist_gap < self.CROSS_CHAIN_RADIUS)
                if np.any(blend_mask):
                    # Cross-chain weight: full at gap=0 (equidistant), zero at gap=radius
                    cross_w = 1.0 - dist_gap[blend_mask] / self.CROSS_CHAIN_RADIUS

                    # Cap: muscles need more cross-chain (they span joints),
                    # skin/organs need less (prevents torso-pulling-with-arm)
                    max_w = self.MAX_CROSS_WEIGHT_MUSCLE if is_muscle else self.MAX_CROSS_WEIGHT_OTHER
                    # Primary weight = portion that stays with original chain
                    weights[blend_mask] = 1.0 - cross_w * max_w
                    secondary_indices[blend_mask] = other_ji[blend_mask]

        # ── Post-registration weight smoothing at joint boundaries ──
        # For non-muscle meshes (skin), smooth blend weights at edges where
        # adjacent vertices are bound to different primary joints.  This
        # reduces the weight discontinuity that causes tearing/inversion
        # artifacts at joint bends during large rotations (an inherent
        # limitation of linear blend skinning).
        if not is_muscle and mesh.geometry.has_indices:
            self._smooth_boundary_weights(
                joint_indices, secondary_indices, weights, mesh,
            )

        # Compute rest Y-span for activation coloring (top/bottom 25%)
        rest_y_span = 1.0
        if is_muscle:
            y_vals = positions[:, 1]
            n25 = max(1, vert_count // 4)
            top_idx = np.argpartition(y_vals, -n25)[-n25:]
            bot_idx = np.argpartition(y_vals, n25)[:n25]
            rest_y_span = max(1e-6, float(y_vals[top_idx].mean() - y_vals[bot_idx].mean()))

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=joint_indices,
            weights=weights,
            secondary_indices=secondary_indices,
            is_muscle=is_muscle,
            base_color=mesh.material.color,
            rest_y_span=rest_y_span,
            head_follow_config=head_follow_config,
            muscle_name=muscle_name,
        )
        self._build_neighbor_data(binding, precomputed_edges=_precomputed_edges)
        self.bindings.append(binding)

    def snap_hierarchy_blends(self, child_chain_ids: set[int]) -> None:
        """Remove cross-chain blending between child chains and parent chains.

        Digit chain joints inherit all parent transforms (wrist, elbow,
        shoulder) via the scene graph hierarchy.  Cross-chain blending
        between a digit chain and a limb chain double-counts the parent
        contribution, making the vertex only partially follow finger curl.

        This snaps any muscle vertex whose primary and secondary chains
        straddle the child/parent boundary to 100% primary weight.

        Parameters
        ----------
        child_chain_ids : set of int
            Chain IDs of child chains (hand digit chains, foot digit
            chains) that inherit parent transforms via scene graph.
        """
        if not child_chain_ids:
            return

        all_chain_ids = np.array(
            [j.chain_id for j in self.joints], dtype=np.int32,
        )
        max_cid = int(all_chain_ids.max()) + 1
        is_child = np.zeros(max_cid, dtype=bool)
        for cid in child_chain_ids:
            if cid < max_cid:
                is_child[cid] = True

        for binding in self.bindings:
            if not binding.is_muscle:
                continue
            blended = binding.weights < 1.0
            if not np.any(blended):
                continue

            b_idx = np.where(blended)[0]
            pri_cid = all_chain_ids[binding.joint_indices[b_idx]]
            sec_cid = all_chain_ids[binding.secondary_indices[b_idx]]
            pri_child = is_child[pri_cid]
            sec_child = is_child[sec_cid]

            # Snap where one is child and the other isn't
            mismatch = pri_child != sec_child
            if np.any(mismatch):
                snap_idx = b_idx[mismatch]
                binding.weights[snap_idx] = 1.0
                binding.secondary_indices[snap_idx] = binding.joint_indices[snap_idx]

            # If this binding has ANY child-chain vertices, mark it to skip
            # neighbor-stretch clamping.  Muscles spanning into digit chains
            # (e.g. FDP, FDS with tendons) need free movement; clamping would
            # pull digit-bound vertices back toward the forearm.
            all_cids = all_chain_ids[binding.joint_indices]
            if np.any(is_child[all_cids]):
                binding.skip_neighbor_clamp = True

    def reassign_orphan_vertices(self, child_chain_ids: set[int]) -> int:
        """Reassign non-digit muscle vertices that should follow digit chains.

        After ``snap_hierarchy_blends``, some muscle vertices at the
        digit/limb transition may remain on the limb chain while their
        mesh neighbors are on digit chains.  These orphan vertices follow
        wrist movement but not finger curl, causing visible spikes.

        Two phases:

        Phase 1 — topology voting (iterative):
            Non-child vertices whose majority of mesh neighbors are on
            child chains get reassigned to the nearest child-chain
            neighbor's joint.  Iterates to cascade.

        Phase 2 — spatial proximity:
            Remaining non-child vertices that are closer to a digit-chain
            joint than to their currently assigned joint get reassigned.
            This catches clusters of orphan vertices (e.g. the middle
            finger tendon) that can't reach majority through topology
            alone because they form a connected non-child group.

        Returns the total number of vertices reassigned.
        """
        if not child_chain_ids:
            return 0

        all_chain_ids = np.array(
            [j.chain_id for j in self.joints], dtype=np.int32,
        )
        max_cid = int(all_chain_ids.max()) + 1
        is_child_lut = np.zeros(max_cid, dtype=bool)
        for cid in child_chain_ids:
            if cid < max_cid:
                is_child_lut[cid] = True

        # Collect digit-chain joint indices and positions for Phase 2
        child_ji_list = [
            j_idx for j_idx in range(len(self.joints))
            if is_child_lut[all_chain_ids[j_idx]]
        ]
        child_j_pos = (
            np.array([self.joints[j].rest_world[:3, 3] for j in child_ji_list])
            if child_ji_list else np.empty((0, 3))
        )

        total = 0
        for binding in self.bindings:
            if not binding.is_muscle or binding.edge_pairs is None:
                continue

            ji = binding.joint_indices
            si = binding.secondary_indices
            w = binding.weights
            V = len(ji)
            edges = binding.edge_pairs  # (E, 2) unique undirected edges
            positions = binding.mesh.rest_positions.reshape(-1, 3).astype(
                np.float64,
            )

            # ── Phase 1: topology-based majority voting ──
            for _pass in range(5):
                is_child = is_child_lut[all_chain_ids[ji]]
                if np.all(is_child) or not np.any(is_child):
                    break

                child_count = np.zeros(V, dtype=np.int32)
                total_count = np.zeros(V, dtype=np.int32)
                e0, e1 = edges[:, 0], edges[:, 1]
                np.add.at(child_count, e0, is_child[e1].astype(np.int32))
                np.add.at(child_count, e1, is_child[e0].astype(np.int32))
                np.add.at(total_count, e0, 1)
                np.add.at(total_count, e1, 1)

                orphan = (
                    ~is_child
                    & (total_count > 0)
                    & (child_count * 2 > total_count)
                )
                if not np.any(orphan):
                    break

                orphan_idx = np.where(orphan)[0]
                for vi in orphan_idx:
                    mask_a = e0 == vi
                    mask_b = e1 == vi
                    neighbors = np.concatenate([e1[mask_a], e0[mask_b]])
                    child_nbrs = neighbors[is_child[neighbors]]
                    if len(child_nbrs) == 0:
                        continue
                    dists = np.linalg.norm(
                        positions[child_nbrs] - positions[vi], axis=1,
                    )
                    nearest = child_nbrs[np.argmin(dists)]
                    ji[vi] = ji[nearest]
                    w[vi] = 1.0
                    si[vi] = ji[vi]

                total += len(orphan_idx)

            # ── Phase 2: spatial proximity to digit joints ──
            # Catches clusters of non-child vertices (e.g. middle finger
            # tendon) where no single vertex reaches majority through
            # topology.  If a non-child vertex is closer to any digit
            # joint than to any non-digit joint, reassign it.
            #
            # Important: compare against the nearest NON-CHILD joint, not
            # the assigned joint.  A vertex at t=0.7 on the elbow→wrist
            # segment is assigned to the elbow (far away) but physically
            # near the wrist.  Using the assigned joint would incorrectly
            # reassign mid-forearm vertices whose elbow is farther than
            # the nearest mc joint.
            if len(child_ji_list) == 0:
                continue

            is_child = is_child_lut[all_chain_ids[ji]]
            if np.all(is_child) or not np.any(is_child):
                continue

            remaining = np.where(~is_child)[0]
            rem_pos = positions[remaining]  # (R, 3)

            # Distance to nearest digit joint
            d_to_digit = np.linalg.norm(
                rem_pos[:, np.newaxis, :] - child_j_pos[np.newaxis, :, :],
                axis=2,
            )  # (R, D)
            nearest_d_idx = np.argmin(d_to_digit, axis=1)  # (R,)
            d_nearest_digit = d_to_digit[
                np.arange(len(remaining)), nearest_d_idx,
            ]  # (R,)

            # Distance to nearest non-child joint (e.g. wrist, elbow)
            non_child_ji_list = [
                j_idx for j_idx in range(len(self.joints))
                if not is_child_lut[all_chain_ids[j_idx]]
            ]
            if not non_child_ji_list:
                continue
            non_child_j_pos = np.array([
                self.joints[j].rest_world[:3, 3] for j in non_child_ji_list
            ])  # (P, 3)
            d_to_parent = np.linalg.norm(
                rem_pos[:, np.newaxis, :] - non_child_j_pos[np.newaxis, :, :],
                axis=2,
            )  # (R, P)
            d_nearest_parent = d_to_parent.min(axis=1)  # (R,)

            # Reassign if digit joint is closer than any non-digit joint
            closer = d_nearest_digit < d_nearest_parent
            if np.any(closer):
                closer_idx = remaining[closer]
                new_ji = np.array(child_ji_list)[nearest_d_idx[closer]]
                ji[closer_idx] = new_ji
                w[closer_idx] = 1.0
                si[closer_idx] = new_ji
                total += int(np.sum(closer))

        return total

    # Number of Laplacian smoothing iterations for boundary weights.
    SMOOTH_ITERATIONS = 5
    # Blend factor per iteration: 0 = no smoothing, 1 = full neighbor average.
    SMOOTH_STRENGTH = 0.5

    def _smooth_boundary_weights(
        self,
        joint_indices: np.ndarray,
        secondary_indices: np.ndarray,
        weights: np.ndarray,
        mesh: MeshInstance,
    ) -> None:
        """Laplacian smooth of blend weights at joint/chain boundaries.

        At joint boundaries, adjacent vertices on different primary joints
        can have very different blend weights — e.g. vertex A at 90% joint_1
        and adjacent vertex B at 90% joint_2.  When the joint bends, this
        discontinuity causes edge stretching and triangle inversion.

        This pass identifies boundary edges (endpoints on different primary
        joints) and iteratively smooths the weights of boundary vertices
        toward their neighbor average, creating a gradual transition zone.

        Works in terms of "effective transform influence" rather than raw
        weight values, since weights mean different things depending on
        which joints are primary vs secondary.
        """
        V = len(weights)
        if V < 3:
            return

        indices = mesh.geometry.indices
        if indices is None:
            return
        tris = indices.reshape(-1, 3)

        # Build edge list (both directions for neighbor lookup)
        raw_edges = np.concatenate([
            tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]],
        ], axis=0)  # (3T, 2)
        # Add reverse edges
        all_edges = np.concatenate([raw_edges, raw_edges[:, ::-1]], axis=0)

        # Identify boundary edges: endpoints on different primary joints
        ji = joint_indices
        edge_src = all_edges[:, 0]
        edge_dst = all_edges[:, 1]
        is_boundary_edge = ji[edge_src] != ji[edge_dst]

        # Boundary vertices: any vertex involved in a boundary edge
        boundary = np.zeros(V, dtype=bool)
        boundary_src = edge_src[is_boundary_edge]
        boundary[boundary_src] = True

        if not boundary.any():
            return

        # Expand boundary zone by 1 ring
        zone = boundary.copy()
        zone_expand = np.zeros(V, dtype=bool)
        zone_expand[edge_dst[np.isin(edge_src, np.where(boundary)[0])]] = True
        zone |= zone_expand
        zone_verts = np.where(zone)[0]
        if len(zone_verts) == 0:
            return

        # Build CSR adjacency for zone vertices only (vectorized)
        # Filter edges to those involving zone vertices
        zone_set = zone
        zone_edges_mask = zone_set[edge_src] | zone_set[edge_dst]
        zone_edge_src = edge_src[zone_edges_mask]
        zone_edge_dst = edge_dst[zone_edges_mask]

        # Sort by source for CSR construction
        sort_idx = np.argsort(zone_edge_src)
        sorted_src = zone_edge_src[sort_idx]
        sorted_dst = zone_edge_dst[sort_idx]

        # Build offset array using bincount
        counts = np.bincount(sorted_src, minlength=V)
        offsets = np.zeros(V + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        # Precompute per-edge weight contribution type:
        # same_primary: neighbor has same primary joint → use their weight directly
        # swapped: neighbor's primary is our secondary and vice versa → invert weight
        # other: different joint pair → use our weight (no smoothing)
        ji_src = ji[sorted_src]
        ji_dst = ji[sorted_dst]
        si_src = secondary_indices[sorted_src]
        si_dst = secondary_indices[sorted_dst]

        same_primary = ji_src == ji_dst
        swapped = (~same_primary) & (ji_dst == si_src) & (si_dst == ji_src)
        # 'other' is ~same_primary & ~swapped

        # Iterative smoothing (vectorized per zone vertex)
        strength = self.SMOOTH_STRENGTH
        w = weights.astype(np.float64)

        for _it in range(self.SMOOTH_ITERATIONS):
            # Compute effective neighbor weight based on edge type
            nbr_w = np.where(same_primary, w[sorted_dst],
                    np.where(swapped, 1.0 - w[sorted_dst],
                             w[sorted_src]))  # 'other': keep own weight

            # Sum neighbor weights and counts per vertex
            w_sum = np.bincount(sorted_src, weights=nbr_w, minlength=V).astype(np.float64)
            w_cnt = counts.astype(np.float64)

            # Compute average where count > 0, blend with current
            has_nbrs = w_cnt > 0
            avg = np.zeros(V, dtype=np.float64)
            avg[has_nbrs] = w_sum[has_nbrs] / w_cnt[has_nbrs]

            # Apply smoothing only to zone vertices
            new_w = w.copy()
            mask = zone & has_nbrs
            new_w[mask] = (1.0 - strength) * w[mask] + strength * avg[mask]
            w = np.clip(new_w, 0.001, 0.999)

        weights[:] = w.astype(np.float32)

    @staticmethod
    def _extract_mesh_edges(
        mesh: MeshInstance,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract unique undirected edges and their lengths from a mesh.

        Returns ``(unique_edges, edge_lengths)`` where *unique_edges* is
        ``(E, 2)`` vertex-index pairs and *edge_lengths* is ``(E,)``, or
        ``None`` if the mesh has no index buffer.
        """
        indices = mesh.geometry.indices
        if indices is None or len(indices) < 3:
            return None
        if mesh.rest_positions is None:
            return None

        rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)

        tri = indices.reshape(-1, 3)
        e01 = np.column_stack([tri[:, 0], tri[:, 1]])
        e12 = np.column_stack([tri[:, 1], tri[:, 2]])
        e20 = np.column_stack([tri[:, 2], tri[:, 0]])
        all_edges = np.concatenate([e01, e12, e20], axis=0)  # (3T, 2)

        sorted_edges = np.sort(all_edges, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)  # (E, 2)

        diff = rest[unique_edges[:, 1]] - rest[unique_edges[:, 0]]
        edge_lengths = np.sqrt(np.sum(diff * diff, axis=1))  # (E,)

        return unique_edges, edge_lengths

    def _geodesic_chain_dists(
        self,
        positions: np.ndarray,
        edges: np.ndarray,
        edge_lengths: np.ndarray,
        seg_starts: np.ndarray,
        seg_ends: np.ndarray,
        seg_chain_arr: np.ndarray,
    ) -> np.ndarray:
        """Compute geodesic (mesh-edge) distance from each vertex to each chain.

        For each chain, seed vertices are those within ``SEED_RADIUS`` of any
        bone segment in that chain.  The seed distance is the Euclidean
        distance to the nearest point on the segment.  A virtual super-source
        is connected to all seeds with edge weights equal to their seed
        distances, and a single-source Dijkstra gives the shortest surface
        path from any bone in the chain to every mesh vertex.

        Parameters
        ----------
        positions : (V, 3) array
        edges : (E, 2) array of unique undirected edge vertex pairs
        edge_lengths : (E,) array of edge lengths
        seg_starts, seg_ends : (S, 3) arrays of bone segment endpoints
        seg_chain_arr : (S,) chain ID per segment

        Returns
        -------
        (V, C) array of geodesic distances, where C = number of unique chains.
        """
        V = len(positions)
        unique_chains = np.unique(seg_chain_arr)
        C = len(unique_chains)
        chain_to_idx = {int(c): i for i, c in enumerate(unique_chains)}

        # Try scipy for fast C-implemented Dijkstra
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra as sp_dijkstra
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        # Base mesh edges (both directions) — used for both scipy and heapq
        base_row = np.concatenate([edges[:, 0], edges[:, 1]])
        base_col = np.concatenate([edges[:, 1], edges[:, 0]])
        base_data = np.concatenate([edge_lengths, edge_lengths])

        # Build adjacency list for heapq fallback (only if needed)
        adj: list[list[tuple[int, float]]] | None = None
        if not _has_scipy:
            adj = [[] for _ in range(V)]
            for i in range(len(edges)):
                u, v = int(edges[i, 0]), int(edges[i, 1])
                w = float(edge_lengths[i])
                adj[u].append((v, w))
                adj[v].append((u, w))

        result = np.full((V, C), np.inf, dtype=np.float64)

        for chain_id in unique_chains:
            ci = chain_to_idx[int(chain_id)]
            chain_mask = seg_chain_arr == chain_id
            chain_seg_starts = seg_starts[chain_mask]  # (Sc, 3)
            chain_seg_ends = seg_ends[chain_mask]      # (Sc, 3)

            # Find seed vertices: those within SEED_RADIUS of any segment
            ab = chain_seg_ends - chain_seg_starts  # (Sc, 3)
            ab_len_sq = np.sum(ab * ab, axis=1)     # (Sc,)

            p_exp = positions[:, np.newaxis, :]  # (V, 1, 3)
            ap = p_exp - chain_seg_starts[np.newaxis, :, :]
            t = np.sum(ap * ab[np.newaxis, :, :], axis=2) / np.maximum(ab_len_sq[np.newaxis, :], 1e-10)
            t = np.clip(t, 0.0, 1.0)
            closest = chain_seg_starts[np.newaxis, :, :] + t[:, :, np.newaxis] * ab[np.newaxis, :, :]
            diff = p_exp - closest
            seg_dists = np.sqrt(np.sum(diff * diff, axis=2))  # (V, Sc)
            min_seg_dist = seg_dists.min(axis=1)  # (V,)

            seed_mask = min_seg_dist <= self.SEED_RADIUS
            seed_indices = np.where(seed_mask)[0]
            seed_dists = min_seg_dist[seed_mask]

            if len(seed_indices) == 0:
                # Fallback: closest vertex to each segment endpoint
                fallback_list: list[int] = []
                for seg_start, seg_end in zip(chain_seg_starts, chain_seg_ends):
                    for pt in [seg_start, seg_end]:
                        d = np.linalg.norm(positions - pt[np.newaxis, :], axis=1)
                        fallback_list.append(int(np.argmin(d)))
                seed_indices = np.unique(fallback_list).astype(np.intp)
                seed_dists = min_seg_dist[seed_indices]

            if _has_scipy:
                # Virtual super-source approach: add node V connected to all
                # seeds with edge weight = seed Euclidean distance.  One
                # Dijkstra from the super-source gives multi-source distances.
                super_src = V  # virtual node index
                seed_row = np.full(len(seed_indices), super_src, dtype=np.intp)
                seed_col = seed_indices.astype(np.intp)

                # Bidirectional edges to super-source
                row = np.concatenate([base_row, seed_row, seed_col])
                col = np.concatenate([base_col, seed_col, seed_row])
                data = np.concatenate([base_data, seed_dists, seed_dists])

                graph = csr_matrix(
                    (data, (row, col)), shape=(V + 1, V + 1),
                )
                dists_from_super = sp_dijkstra(
                    graph, directed=False, indices=super_src,
                )  # (V+1,)
                result[:, ci] = dists_from_super[:V]
            else:
                # Python heapq multi-source Dijkstra
                import heapq
                assert adj is not None
                dist = np.full(V, np.inf, dtype=np.float64)
                heap: list[tuple[float, int]] = []
                for si_idx in range(len(seed_indices)):
                    sv = int(seed_indices[si_idx])
                    sd = float(seed_dists[si_idx])
                    if sd < dist[sv]:
                        dist[sv] = sd
                        heapq.heappush(heap, (sd, sv))

                while heap:
                    d_u, u = heapq.heappop(heap)
                    if d_u > dist[u]:
                        continue
                    for v_nb, w_nb in adj[u]:
                        d_new = d_u + w_nb
                        if d_new < dist[v_nb]:
                            dist[v_nb] = d_new
                            heapq.heappush(heap, (d_new, v_nb))

                result[:, ci] = dist

        return result

    def _build_neighbor_data(
        self,
        binding: SkinBinding,
        precomputed_edges: np.ndarray | None = None,
    ) -> None:
        """Build mesh adjacency and rest-pose neighbor distances for clamping.

        Extracts unique edges from the triangle index buffer, then computes
        each vertex's average distance to its mesh neighbors in rest pose.
        This baseline is used per-frame to detect vertices that have been
        stretched anomalously far from their neighbors (e.g. by arm chains
        pulling hip vertices) and clamp them back.

        Parameters
        ----------
        precomputed_edges : (E, 2) ndarray or None
            If provided, skip edge extraction and use these edges directly.
        """
        mesh = binding.mesh
        if mesh.rest_positions is None:
            return

        rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        V = len(rest)

        if precomputed_edges is not None:
            unique_edges = precomputed_edges
        else:
            result = self._extract_mesh_edges(mesh)
            if result is None:
                return
            unique_edges = result[0]

        # Compute neighbor counts
        neighbor_counts = np.zeros(V, dtype=np.int32)
        np.add.at(neighbor_counts, unique_edges[:, 0], 1)
        np.add.at(neighbor_counts, unique_edges[:, 1], 1)

        # Compute rest-pose neighbor average positions
        neighbor_sum = np.zeros((V, 3), dtype=np.float64)
        np.add.at(neighbor_sum, unique_edges[:, 0], rest[unique_edges[:, 1]])
        np.add.at(neighbor_sum, unique_edges[:, 1], rest[unique_edges[:, 0]])

        has_neighbors = neighbor_counts > 0
        neighbor_avg = np.zeros_like(neighbor_sum)
        neighbor_avg[has_neighbors] = (
            neighbor_sum[has_neighbors]
            / neighbor_counts[has_neighbors, np.newaxis]
        )

        # Rest distance to neighbor average (baseline for stretch ratio)
        rest_dist = np.linalg.norm(rest - neighbor_avg, axis=1)

        # Compute boundary blend: fraction of neighbors on the SAME chain.
        # Interior vertices (blend ≈ 1.0) get tight clamping to catch spikes.
        # Chain-boundary vertices (blend < 1.0) get relaxed clamping to allow
        # natural stretch at arm↔torso, leg↔pelvis transitions.
        ji = binding.joint_indices
        if ji is not None and len(ji) == V and self.joints:
            vert_chains = np.array(
                [self.joints[idx].chain_id for idx in ji], dtype=np.int32
            )
            edge_same = (
                vert_chains[unique_edges[:, 0]]
                == vert_chains[unique_edges[:, 1]]
            ).astype(np.float64)
            same_count = np.zeros(V, dtype=np.float64)
            total_count = np.zeros(V, dtype=np.float64)
            np.add.at(same_count, unique_edges[:, 0], edge_same)
            np.add.at(same_count, unique_edges[:, 1], edge_same)
            np.add.at(total_count, unique_edges[:, 0], 1.0)
            np.add.at(total_count, unique_edges[:, 1], 1.0)
            boundary_blend = np.ones(V, dtype=np.float64)
            has_any = total_count > 0
            boundary_blend[has_any] = same_count[has_any] / total_count[has_any]
        else:
            boundary_blend = np.ones(V, dtype=np.float64)

        # Propagated boundary zone: expand boundary influence outward by
        # several edge rings so the displacement smoothing acts over a wider
        # transition band, not just the immediate boundary vertices.
        # smooth_zone = 1.0 at boundary, decaying outward over BOUNDARY_SMOOTH_RINGS.
        smooth_zone = 1.0 - boundary_blend  # 0 for interior, 1 for boundary
        for _ring in range(self.BOUNDARY_SMOOTH_RINGS):
            zone_sum = np.zeros(V, dtype=np.float64)
            np.add.at(zone_sum, unique_edges[:, 0], smooth_zone[unique_edges[:, 1]])
            np.add.at(zone_sum, unique_edges[:, 1], smooth_zone[unique_edges[:, 0]])
            zone_avg = np.zeros(V, dtype=np.float64)
            zone_avg[has_neighbors] = zone_sum[has_neighbors] / neighbor_counts[has_neighbors]
            # Expand: keep existing zone, propagate outward with decay
            smooth_zone = np.maximum(smooth_zone, zone_avg * 0.7)

        binding.edge_pairs = unique_edges
        binding.neighbor_counts = neighbor_counts
        binding.rest_neighbor_dist = rest_dist
        binding.boundary_blend = boundary_blend
        binding.smooth_zone = smooth_zone

    def _apply_neighbor_clamp(self, binding: SkinBinding) -> int:
        """Clamp vertices stretched too far from their mesh neighbors.

        After delta-matrix skinning, some vertices may be pulled far from
        their mesh neighbors due to incorrect chain assignment (e.g. arm
        chain pulling hip skin).  This detects those outliers by comparing
        each vertex's distance to its neighbor average against the rest-pose
        baseline, and snaps them back to the maximum allowed distance.

        Returns the number of vertices clamped.
        """
        if binding.edge_pairs is None:
            return 0

        mesh = binding.mesh
        positions = mesh.geometry.positions.reshape(-1, 3)
        current = positions.astype(np.float64)
        V = len(current)

        edges = binding.edge_pairs
        counts = binding.neighbor_counts
        rest_dist = binding.rest_neighbor_dist

        # Compute current neighbor average positions (reuse buffer)
        if not hasattr(binding, '_neighbor_sum_buf') or binding._neighbor_sum_buf is None or len(binding._neighbor_sum_buf) != V:
            binding._neighbor_sum_buf = np.zeros((V, 3), dtype=np.float64)
        neighbor_sum = binding._neighbor_sum_buf
        neighbor_sum[:] = 0.0
        np.add.at(neighbor_sum, edges[:, 0], current[edges[:, 1]])
        np.add.at(neighbor_sum, edges[:, 1], current[edges[:, 0]])

        has_neighbors = counts > 0
        neighbor_avg = np.zeros_like(neighbor_sum)
        neighbor_avg[has_neighbors] = (
            neighbor_sum[has_neighbors] / counts[has_neighbors, np.newaxis]
        )

        # Distance from each vertex to its neighbor average
        diff = current - neighbor_avg
        current_dist = np.linalg.norm(diff, axis=1)

        # Stretch ratio vs rest-pose baseline
        stretch = current_dist / (rest_dist + 0.01)

        # Per-vertex threshold: interior vertices get tight clamping to
        # catch mis-binding spikes; chain-boundary vertices get a relaxed
        # threshold to allow natural stretch at arm↔torso transitions etc.
        # Cached since boundary_blend only changes on reassignment, not per frame.
        if not hasattr(binding, '_per_vert_max') or binding._per_vert_max is None or len(binding._per_vert_max) != V:
            bb = binding.boundary_blend
            if bb is not None:
                binding._per_vert_max = self.MAX_NEIGHBOR_STRETCH * (
                    1.0 + (1.0 - bb) * self.BOUNDARY_RELAX_FACTOR
                )
            else:
                binding._per_vert_max = np.full(V, self.MAX_NEIGHBOR_STRETCH)
        per_vert_max = binding._per_vert_max

        # Find outliers.  Skip vertices with very small rest_neighbor_dist
        # (< 0.05) — these sit at the centroid of their neighborhood and any
        # movement gives inflated ratios.
        meaningful_rest = rest_dist > 0.05
        outlier = has_neighbors & meaningful_rest & (stretch > per_vert_max)
        n_clamped = int(np.sum(outlier))
        if n_clamped == 0:
            return 0

        # Snap outlier vertices back: keep direction from neighbor avg,
        # but cap distance at per-vertex threshold × rest distance
        allowed = rest_dist[outlier] * per_vert_max[outlier]
        out_diff = diff[outlier]
        out_dist = current_dist[outlier]
        safe_dist = np.maximum(out_dist, 1e-6)
        direction = out_diff / safe_dist[:, np.newaxis]
        clamped_pos = neighbor_avg[outlier] + direction * allowed[:, np.newaxis]

        # Write back
        positions[outlier] = clamped_pos.astype(positions.dtype)
        return n_clamped

    def _smooth_boundary_displacements(self, binding: SkinBinding) -> None:
        """Smooth displacement discontinuities at chain-boundary vertices.

        After delta-matrix skinning and neighbor clamping, adjacent vertices
        on different chains may have very different displacements, creating
        visible seams.  This iteratively blends each boundary vertex's
        displacement toward its neighbors' average displacement, producing a
        smooth gradient across chain transitions.

        Only affects vertices within the propagated smooth_zone (boundary +
        a few rings outward).  Interior vertices are untouched.
        """
        sz = binding.smooth_zone
        if sz is None:
            return
        edges = binding.edge_pairs
        if edges is None:
            return
        counts = binding.neighbor_counts
        if counts is None:
            return

        mesh = binding.mesh
        positions = mesh.geometry.positions.reshape(-1, 3)
        V = len(positions)

        # Cache rest_pos float64 and smooth_alpha (never change)
        if not hasattr(binding, '_rest_f64') or binding._rest_f64 is None:
            binding._rest_f64 = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
        rest = binding._rest_f64

        # Only smooth vertices with smooth_zone > small threshold
        if not hasattr(binding, '_smooth_active') or binding._smooth_active is None:
            binding._smooth_active = sz > 0.01
            binding._smooth_mask = binding._smooth_active & (counts > 0)
            binding._smooth_alpha = (sz * self.BOUNDARY_SMOOTH_STRENGTH)[binding._smooth_mask, np.newaxis]
        active = binding._smooth_active
        if not np.any(active):
            return

        has_neighbors = counts > 0
        mask = binding._smooth_mask
        a = binding._smooth_alpha

        # Work on displacements, not absolute positions
        disp = positions.astype(np.float64) - rest

        # Reuse pre-allocated buffer for disp_sum across passes
        if not hasattr(binding, '_disp_sum_buf') or binding._disp_sum_buf is None or len(binding._disp_sum_buf) != V:
            binding._disp_sum_buf = np.zeros((V, 3), dtype=np.float64)
        disp_sum = binding._disp_sum_buf

        for _pass in range(self.BOUNDARY_SMOOTH_PASSES):
            disp_sum[:] = 0.0
            np.add.at(disp_sum, edges[:, 0], disp[edges[:, 1]])
            np.add.at(disp_sum, edges[:, 1], disp[edges[:, 0]])
            # In-place divide for neighbor average (only where has_neighbors)
            disp_sum[has_neighbors] /= counts[has_neighbors, np.newaxis]

            # Blend displacement toward neighbor average
            disp[mask] = (1.0 - a) * disp[mask] + a * disp_sum[mask]

        # Write smoothed positions back
        disp += rest
        positions[:] = disp.astype(positions.dtype)

    # ── Bone-follow correction for muscle blended vertices ──
    #
    # DQS blending of two joint transforms produces an arc between joints.
    # For muscles along long bones (biceps on humerus, quads on femur),
    # vertices should stay close to the bone axis, not arc away from it.
    #
    # The primary-only transform (delta[primary] @ vertex) IS the bone-
    # following position — it keeps the vertex at its rest offset from
    # the bone, rotated with the bone.  DQS blends this with the secondary
    # transform, creating the arc.
    #
    # This correction blends between the primary-only result (bone-following)
    # and the DQS result.  Mid-bone vertices mostly use the primary transform
    # (staying on the bone), while vertices near the secondary joint use DQS
    # for a smooth transition at the joint boundary.

    def _apply_bone_follow(
        self,
        binding: SkinBinding,
        blend_idx: np.ndarray,
        result_pri: np.ndarray,
        pri_only: np.ndarray,
        dqs_result: np.ndarray,
    ) -> None:
        """Blend DQS toward primary-only transform for muscle blended vertices.

        Parameters
        ----------
        binding : SkinBinding
        blend_idx : indices of blended vertices
        result_pri : output positions array (modified in-place)
        pri_only : primary-only transform results for blended vertices (B, 3)
        dqs_result : DQS blend results for blended vertices (B, 3)
        """
        if not binding.is_muscle or len(blend_idx) == 0:
            return

        w = binding.weights[blend_idx]  # 1.0 = fully primary, 0.0 = fully secondary

        # Bone-follow strength: ramps from 0 near secondary joint (w ≈ 0)
        # to 1.0 for mid-bone and primary-adjacent vertices (w > 0.5).
        # This keeps muscles hugging the bone while allowing smooth transition
        # at joint boundaries.
        #   w=0.0 → strength=0.0 (pure DQS for joint transition)
        #   w=0.3 → strength=0.6 (mostly bone-following)
        #   w=0.5+ → strength=1.0 (fully bone-following)
        strength = np.clip((w - 0.0) / 0.5, 0.0, 1.0)  # (B,)
        s = strength[:, np.newaxis]  # (B, 1)

        result_pri[blend_idx] = s * pri_only + (1.0 - s) * dqs_result

    def update(self, body_state: BodyState) -> None:
        """Per-frame delta-matrix transform for all registered meshes.

        Uses Dual Quaternion Skinning (DQS) for vertices that blend between
        two joints, preventing volume collapse and candy-wrapper artifacts
        inherent to linear blend skinning.  Single-joint vertices still use
        standard matrix transforms (equivalent result, faster).

        For each vertex:
          delta = restWorldInv[joint] @ currentWorld[joint]
          new_pos = DQ_blend(delta_pri, delta_sec, weight) @ rest_pos
        """
        if not self.joints or not self.bindings:
            return

        # Compute signature for early-exit
        sig = self._compute_signature(body_state)
        if sig == self._last_signature:
            return
        self._last_signature = sig

        # Compute delta matrices for each joint
        deltas = []
        for joint in self.joints:
            joint.node.update_world_matrix()
            current = joint.node.world_matrix
            delta = current @ joint.rest_world_inv
            deltas.append(delta)

        # Stack delta matrices for vectorized lookup
        delta_stack = np.array(deltas, dtype=np.float64)  # (J, 4, 4)

        # Pre-compute dual quaternions for DQS blending
        dq_stack = batch_mat4_to_dual_quat(delta_stack)  # (J, 8)

        # Apply to each registered mesh
        for binding in self.bindings:
            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue

            # Cache rest positions as float64 (never changes, avoids per-frame copy)
            if not hasattr(binding, '_rest_f64') or binding._rest_f64 is None:
                binding._rest_f64 = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
            rest_pos = binding._rest_f64
            V = len(rest_pos)
            ji = binding.joint_indices   # (V,)
            si = binding.secondary_indices  # (V,)
            w = binding.weights          # (V,)

            # Homogeneous positions: (V, 4) — cached to avoid per-frame allocation
            if not hasattr(binding, '_pos_h') or binding._pos_h is None:
                ones = np.ones((V, 1), dtype=np.float64)
                binding._pos_h = np.concatenate([rest_pos, ones], axis=1)
            pos_h = binding._pos_h

            # Primary transform: delta[ji] @ pos for each vertex
            d_pri = delta_stack[ji]
            result_pri = np.einsum('vij,vj->vi', d_pri, pos_h)[:, :3]

            # Check which vertices need blending
            needs_blend = (w < 0.999) & (ji != si)

            # Track DQS blend results for normal computation
            blend_idx = None
            q_r_blend = None

            if np.any(needs_blend):
                # Compute LBS secondary positions for divergence check
                d_sec = delta_stack[si]
                result_sec = np.einsum('vij,vj->vi', d_sec, pos_h)[:, :3]

                # Divergence-based cross-chain blend clamping:
                # When primary and secondary joints are on DIFFERENT chains
                # and their transforms place the vertex far apart, reduce
                # secondary influence to prevent extreme distortion.
                w_eff = w.astype(np.float64)
                if hasattr(self, '_joint_chain_ids') and len(self._joint_chain_ids) > 0:
                    pri_chain = self._joint_chain_ids[ji]
                    sec_chain = self._joint_chain_ids[si]
                    cross_chain = (pri_chain != sec_chain) & needs_blend
                    if np.any(cross_chain):
                        divergence = np.linalg.norm(
                            result_pri - result_sec, axis=1,
                        )
                        drange = self.DIVERGENCE_MAX - self.DIVERGENCE_MIN
                        if drange > 0:
                            scale = np.clip(
                                1.0 - (divergence - self.DIVERGENCE_MIN) / drange,
                                0.0, 1.0,
                            )
                        else:
                            scale = np.where(
                                divergence <= self.DIVERGENCE_MIN, 1.0, 0.0,
                            )
                        w_eff = w_eff.copy()
                        w_eff[cross_chain] = (
                            1.0 - (1.0 - w[cross_chain]) * scale[cross_chain]
                        )

                # ── Dual Quaternion Skinning for blended vertices ──
                blend_idx = np.where(needs_blend)[0]
                dq_pri = dq_stack[ji[blend_idx]].copy()   # (B, 8)
                dq_sec = dq_stack[si[blend_idx]].copy()   # (B, 8)

                # Shortest-path: flip secondary if dot(real parts) < 0
                dot = np.sum(dq_pri[:, :4] * dq_sec[:, :4], axis=1)
                flip = dot < 0
                dq_sec[flip] *= -1.0

                # Weighted blend of dual quaternions
                w_b = w_eff[blend_idx, np.newaxis]  # (B, 1)
                dq_blend = w_b * dq_pri + (1.0 - w_b) * dq_sec  # (B, 8)

                # Normalize by real quaternion magnitude
                norm_r = np.linalg.norm(dq_blend[:, :4], axis=1, keepdims=True)
                norm_r = np.maximum(norm_r, 1e-10)
                dq_blend /= norm_r

                # Extract rotation quaternion and translation
                q_r_blend = dq_blend[:, :4]   # (B, 4) [x, y, z, w]
                q_d = dq_blend[:, 4:8]        # (B, 4)

                # Translation: t = 2 * q_d * conjugate(q_r)
                q_r_conj = q_r_blend.copy()
                q_r_conj[:, :3] *= -1
                t_quat = 2.0 * batch_quat_multiply(q_d, q_r_conj)
                t_vec = t_quat[:, :3]  # (B, 3)

                # Transform: rotate rest position then translate
                new_blend_pos = batch_quat_rotate(q_r_blend, rest_pos[blend_idx]) + t_vec

                # Save primary-only results for bone-follow correction
                # (before DQS overwrites them).
                pri_only_saved = result_pri[blend_idx].copy() if binding.is_muscle else None

                # Assemble: write DQS results into primary (already a fresh array)
                result_pri[blend_idx] = new_blend_pos

                # Bone-follow correction: blend DQS arc toward primary-only
                # (bone-hugging) positions for muscle blended vertices.
                if binding.is_muscle and pri_only_saved is not None:
                    self._apply_bone_follow(
                        binding, blend_idx, result_pri,
                        pri_only_saved, new_blend_pos,
                    )

            mesh.geometry.positions = result_pri.ravel().astype(np.float32)

            # Neighbor-stretch clamping: snap back vertices that are
            # anomalously far from their mesh neighbors (mis-binding safety net).
            # Multiple passes catch cascading shifts from clamped neighbors.
            # Bindings flagged skip_neighbor_clamp (muscles with digit-chain
            # vertices) skip this — they span into child chains and clamping
            # would pull digit-bound vertices back toward the parent limb.
            # All other muscles still get clamped to prevent sagging.
            if not binding.skip_neighbor_clamp:
                for _clamp_pass in range(self.CLAMP_PASSES):
                    if self._apply_neighbor_clamp(binding) == 0:
                        break

            # Bone attachment pinning for muscles (Layer 2)
            if binding.is_muscle and self.attachment_system is not None:
                self.attachment_system.apply_bone_pinning(binding)

            # Per-muscle stretch clamping (Layer 3)
            if binding.is_muscle and self.attachment_system is not None:
                self.attachment_system.apply_stretch_clamp(binding)

            # Bone collision resolution (Layer 4)
            if binding.is_muscle and self.collision_system is not None:
                self.collision_system.resolve_penetrations(
                    binding.mesh.geometry.positions,
                    binding.mesh.rest_positions,
                )

            # Boundary displacement smoothing: reduce seams at chain transitions.
            # Runs after clamping, then a final clamp pass catches any new
            # stretch introduced by the smoothing.
            if not binding.skip_neighbor_clamp:
                self._smooth_boundary_displacements(binding)
                for _post_pass in range(3):
                    if self._apply_neighbor_clamp(binding) == 0:
                        break

            # Normals — cache rest normals as float64
            if not hasattr(binding, '_rest_nrm_f64'):
                binding._rest_nrm_f64 = (
                    mesh.rest_normals.reshape(-1, 3).astype(np.float64)
                    if mesh.rest_normals is not None else None
                )
            rest_nrm = binding._rest_nrm_f64
            if rest_nrm is not None:
                # Primary normals via rotation matrix
                rot_pri = delta_stack[ji, :3, :3]  # (V, 3, 3)
                nrm_pri = np.einsum('vij,vj->vi', rot_pri, rest_nrm)

                if blend_idx is not None and q_r_blend is not None:
                    # Blended normals: write into nrm_pri (already a fresh array)
                    nrm_blend = batch_quat_rotate(q_r_blend, rest_nrm[blend_idx])
                    nrm_pri[blend_idx] = nrm_blend

                lengths = np.linalg.norm(nrm_pri, axis=1, keepdims=True)
                lengths = np.maximum(lengths, 1e-10)
                nrm_pri /= lengths
                mesh.geometry.normals = nrm_pri.ravel().astype(np.float32)

            mesh.needs_update = True

            # Muscle activation coloring
            if binding.is_muscle:
                self._apply_activation_coloring(binding)

    # ── Activation coloring ─────────────────────────────────────────
    _RED = np.array([1.0, 0.27, 0.27])

    def _apply_activation_coloring(self, binding: SkinBinding) -> None:
        """Color muscles based on contraction/stretch ratio."""
        positions = binding.mesh.geometry.positions.reshape(-1, 3)
        y_vals = positions[:, 1]
        vert_count = len(y_vals)
        n25 = max(1, vert_count // 4)

        top_idx = np.argpartition(y_vals, -n25)[-n25:]
        bot_idx = np.argpartition(y_vals, n25)[:n25]
        current_span = float(y_vals[top_idx].mean() - y_vals[bot_idx].mean())
        ratio = current_span / binding.rest_y_span

        base = np.array(binding.base_color)
        if ratio < 0.98:
            # Contracting: lerp toward red
            t = min(1.0, (1.0 - ratio) * 0.8)
            color = base * (1.0 - t) + self._RED * t
        elif ratio > 1.02:
            # Stretching: lerp toward darker
            t = min(1.0, (ratio - 1.0) * 0.6)
            color = base * (1.0 - t * 0.5)
        else:
            color = base

        binding.mesh.material.color = (float(color[0]), float(color[1]), float(color[2]))

    def _compute_signature(self, state: BodyState) -> tuple:
        """Compute a state signature for early-exit optimization.

        Returns a tuple of rounded floats — fast equality check, no string
        formatting overhead.
        """
        return (
            round(state.spine_flex, 4), round(state.spine_lat_bend, 4),
            round(state.spine_rotation, 4),
            round(state.shoulder_r_abduct, 4), round(state.shoulder_r_flex, 4),
            round(state.shoulder_r_rotate, 4),
            round(state.shoulder_l_abduct, 4), round(state.shoulder_l_flex, 4),
            round(state.shoulder_l_rotate, 4),
            round(state.elbow_r_flex, 4), round(state.elbow_l_flex, 4),
            round(state.forearm_r_rotate, 4), round(state.forearm_l_rotate, 4),
            round(state.wrist_r_flex, 4), round(state.wrist_r_deviate, 4),
            round(state.wrist_l_flex, 4), round(state.wrist_l_deviate, 4),
            round(state.hip_r_flex, 4), round(state.hip_r_abduct, 4),
            round(state.hip_r_rotate, 4),
            round(state.hip_l_flex, 4), round(state.hip_l_abduct, 4),
            round(state.hip_l_rotate, 4),
            round(state.knee_r_flex, 4), round(state.knee_l_flex, 4),
            round(state.ankle_r_flex, 4), round(state.ankle_r_invert, 4),
            round(state.ankle_l_flex, 4), round(state.ankle_l_invert, 4),
            round(state.breath_phase_body, 4), round(state.breath_depth, 4),
            round(state.finger_curl_r, 4), round(state.finger_spread_r, 4),
            round(state.thumb_op_r, 4),
            round(state.finger_curl_l, 4), round(state.finger_spread_l, 4),
            round(state.thumb_op_l, 4),
            round(state.toe_curl_r, 4), round(state.toe_spread_r, 4),
            round(state.toe_curl_l, 4), round(state.toe_spread_l, 4),
        )
