"""Reassign selected vertices to a different kinematic chain."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from faceforge.body.soft_tissue import SkinBinding, SoftTissueSkinning


@dataclass
class UndoRecord:
    """Stores state for undoing a single reassignment operation."""
    binding_idx: int
    vertex_indices: np.ndarray
    old_joint_indices: np.ndarray
    old_secondary_indices: np.ndarray
    old_weights: np.ndarray


class ChainReassigner:
    """Reassigns vertices to a different kinematic chain with single-level undo."""

    def __init__(self, skinning: SoftTissueSkinning) -> None:
        self.skinning = skinning
        self._undo_stack: list[UndoRecord] = []

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def reassign(
        self,
        binding_idx: int,
        vertex_indices: list[int] | np.ndarray,
        target_chain_id: int,
    ) -> int:
        """Reassign vertices in a binding to a target chain.

        Parameters
        ----------
        binding_idx : int
            Index into skinning.bindings.
        vertex_indices : array-like
            Vertex indices within the binding to reassign.
        target_chain_id : int
            The chain ID to assign vertices to.

        Returns
        -------
        int
            Number of vertices actually reassigned.
        """
        if binding_idx >= len(self.skinning.bindings):
            return 0

        binding = self.skinning.bindings[binding_idx]
        vi = np.asarray(vertex_indices, dtype=np.int32)
        if len(vi) == 0:
            return 0

        # Filter valid indices
        V = binding.mesh.geometry.vertex_count
        vi = vi[(vi >= 0) & (vi < V)]
        if len(vi) == 0:
            return 0

        # Save undo state
        self._undo_stack.append(UndoRecord(
            binding_idx=binding_idx,
            vertex_indices=vi.copy(),
            old_joint_indices=binding.joint_indices[vi].copy(),
            old_secondary_indices=binding.secondary_indices[vi].copy(),
            old_weights=binding.weights[vi].copy(),
        ))

        # Find all segments in the target chain
        seg_starts = []
        seg_ends = []
        seg_indices = []
        for ji, joint in enumerate(self.skinning.joints):
            if joint.chain_id != target_chain_id:
                continue
            if joint.segment_start is not None and joint.segment_end is not None:
                ab = joint.segment_end - joint.segment_start
                if np.dot(ab, ab) >= 1e-10:
                    seg_starts.append(joint.segment_start)
                    seg_ends.append(joint.segment_end)
                    seg_indices.append(ji)

        if not seg_starts:
            self._undo_stack.pop()
            return 0

        seg_starts_arr = np.array(seg_starts, dtype=np.float64)
        seg_ends_arr = np.array(seg_ends, dtype=np.float64)
        seg_idx_arr = np.array(seg_indices, dtype=np.int32)
        ab = seg_ends_arr - seg_starts_arr
        ab_len_sq = np.sum(ab * ab, axis=1)

        # Get rest positions for selected vertices
        positions = binding.mesh.rest_positions.reshape(-1, 3)[vi].astype(np.float64)

        # Distance from each vertex to each segment
        p_exp = positions[:, np.newaxis, :]
        ap = p_exp - seg_starts_arr[np.newaxis, :, :]
        t_vals = np.sum(ap * ab[np.newaxis, :, :], axis=2) / ab_len_sq[np.newaxis, :]
        t_vals = np.clip(t_vals, 0.0, 1.0)

        closest = seg_starts_arr[np.newaxis, :, :] + t_vals[:, :, np.newaxis] * ab[np.newaxis, :, :]
        diff = p_exp - closest
        dists = np.sqrt(np.sum(diff * diff, axis=2))

        # Assign to nearest segment
        best_seg = np.argmin(dists, axis=1)
        best_joint = seg_idx_arr[best_seg]

        binding.joint_indices[vi] = best_joint

        # Compute blend weights vectorized
        blend_zone = self.skinning.BLEND_ZONE
        t_best = t_vals[np.arange(len(vi)), best_seg]  # (K,) t along best segment

        # Build lookup: for each joint in target chain, find prev/next in same chain
        chain_joints = [ji for ji, j in enumerate(self.skinning.joints)
                        if j.chain_id == target_chain_id]
        chain_joint_set = set(chain_joints)
        prev_in_chain = {}
        next_in_chain = {}
        for idx, ji in enumerate(chain_joints):
            prev_in_chain[ji] = chain_joints[idx - 1] if idx > 0 else ji
            next_in_chain[ji] = chain_joints[idx + 1] if idx < len(chain_joints) - 1 else ji

        # Vectorized: default to primary joint, weight 1.0
        sec = best_joint.copy()
        w = np.ones(len(vi), dtype=np.float32)

        # Near start of segment: blend with previous joint
        near_start = t_best < blend_zone
        if np.any(near_start):
            idx_ns = np.where(near_start)[0]
            for i in idx_ns:
                sec[i] = prev_in_chain.get(int(best_joint[i]), int(best_joint[i]))
            w[near_start] = t_best[near_start] / blend_zone

        # Near end of segment: blend with next joint
        near_end = t_best > (1.0 - blend_zone)
        if np.any(near_end):
            idx_ne = np.where(near_end)[0]
            for i in idx_ne:
                sec[i] = next_in_chain.get(int(best_joint[i]), int(best_joint[i]))
            w[near_end] = (1.0 - t_best[near_end]) / blend_zone

        binding.weights[vi] = w
        binding.secondary_indices[vi] = sec

        # Recompute boundary_blend for affected vertices and neighbors
        self._update_boundary_blend(binding, vi)

        return len(vi)

    def undo(self) -> bool:
        """Undo the last reassignment. Returns True if undo was performed."""
        if not self._undo_stack:
            return False

        record = self._undo_stack.pop()
        if record.binding_idx >= len(self.skinning.bindings):
            return False

        binding = self.skinning.bindings[record.binding_idx]
        vi = record.vertex_indices
        binding.joint_indices[vi] = record.old_joint_indices
        binding.secondary_indices[vi] = record.old_secondary_indices
        binding.weights[vi] = record.old_weights

        self._update_boundary_blend(binding, vi)
        return True

    def _update_boundary_blend(self, binding: SkinBinding, affected: np.ndarray) -> None:
        """Recompute boundary_blend for affected vertices and their neighbors.

        Fully vectorized — O(E) regardless of selection size.
        """
        if binding.edge_pairs is None or binding.boundary_blend is None:
            return

        edges = binding.edge_pairs  # (E, 2)
        e0 = edges[:, 0]
        e1 = edges[:, 1]

        # Build affected mask and expand to neighbors via edge lookup
        V = binding.mesh.geometry.vertex_count
        affected_mask = np.zeros(V, dtype=bool)
        affected_mask[affected] = True

        # Expand: any vertex adjacent to an affected vertex
        expand_mask = affected_mask.copy()
        expand_mask[e1[affected_mask[e0]]] = True
        expand_mask[e0[affected_mask[e1]]] = True

        # Chain IDs for all vertices
        all_chain_ids = np.array(
            [self.skinning.joints[ji].chain_id for ji in binding.joint_indices],
            dtype=np.int32,
        )

        # For expanded vertices, compute boundary_blend from edges
        # same_chain: edge connects two vertices of the same chain
        same_chain = (all_chain_ids[e0] == all_chain_ids[e1])

        # Count total neighbors and same-chain neighbors per vertex
        # Only for vertices in expand_mask
        expand_idx = np.where(expand_mask)[0]
        if len(expand_idx) == 0:
            return

        # Use np.add.at for O(E) accumulation
        total_count = np.zeros(V, dtype=np.int32)
        same_count = np.zeros(V, dtype=np.int32)
        np.add.at(total_count, e0, 1)
        np.add.at(total_count, e1, 1)
        np.add.at(same_count, e0, same_chain.astype(np.int32))
        np.add.at(same_count, e1, same_chain.astype(np.int32))

        # Update only expanded vertices
        valid = expand_mask & (total_count > 0)
        binding.boundary_blend[valid] = same_count[valid] / total_count[valid]

        # Invalidate cached per-vertex clamp threshold (depends on boundary_blend)
        if hasattr(binding, '_per_vert_max'):
            binding._per_vert_max = None
