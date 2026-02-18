"""Scoring system for skinning quality across poses.

Evaluates soft tissue skinning by applying multiple body poses and
running ``SkinningDiagnostic`` checks, producing a single composite
score suitable for optimization.

Usage::

    from tools.headless_loader import load_headless_scene, load_layer, register_layer
    from tools.skinning_scorer import SkinningScorer

    hs = load_headless_scene()
    meshes = load_layer(hs, "skin")
    register_layer(hs, meshes, "skin")
    scorer = SkinningScorer(hs)
    result = scorer.evaluate()
    print(f"Composite score: {result.composite}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from faceforge.body.diagnostics import SkinningDiagnostic
from faceforge.core.config_loader import load_config
from faceforge.core.state import BodyState

from tools.headless_loader import HeadlessScene, apply_pose

logger = logging.getLogger(__name__)


# ── Test Poses ────────────────────────────────────────────────────────

def _load_preset_poses() -> dict[str, dict[str, float]]:
    """Load the 6 preset poses from body_poses.json."""
    try:
        return load_config("body_poses.json")
    except FileNotFoundError:
        logger.warning("body_poses.json not found, using empty preset set")
        return {}


# 5 extreme poses for stress-testing
_EXTREME_POSES: dict[str, dict[str, float]] = {
    "extreme_arm_raise": {
        "shoulderRAbduct": 1.0,
        "shoulderLAbduct": 1.0,
        "elbowRFlex": 0.5,
        "elbowLFlex": 0.5,
    },
    "extreme_crouch": {
        "hipRFlex": 1.0,
        "hipLFlex": 1.0,
        "kneeRFlex": 1.0,
        "kneeLFlex": 1.0,
        "ankleRFlex": 0.5,
        "ankleLFlex": 0.5,
    },
    "extreme_twist": {
        "spineRotation": 1.0,
        "spineLatBend": 0.8,
    },
    "extreme_asymmetric": {
        "shoulderRAbduct": 1.0,
        "hipLFlex": 1.0,
    },
    "extreme_hands": {
        "fingerCurlR": 1.0,
        "fingerCurlL": 1.0,
        "wristRFlex": 1.0,
        "wristLFlex": 1.0,
        "thumbOpR": 1.0,
        "thumbOpL": 1.0,
    },
}


def get_all_poses(
    pose_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Return pose dict, optionally filtered by name.

    Combines presets from ``body_poses.json`` with 5 built-in extreme poses.
    """
    presets = _load_preset_poses()
    all_poses = {**presets, **_EXTREME_POSES}
    if pose_names is not None:
        filtered = {}
        for name in pose_names:
            if name in all_poses:
                filtered[name] = all_poses[name]
            else:
                logger.warning("Unknown pose %r, skipping", name)
        return filtered
    return all_poses


def _make_body_state(pose_dict: dict[str, float]) -> BodyState:
    """Create a BodyState from a JS-style camelCase pose dict."""
    state = BodyState()
    state.set_from_js_dict(pose_dict)
    return state


# ── Per-pose scoring ─────────────────────────────────────────────────

@dataclass
class PoseScore:
    """Diagnostic scores for a single pose."""
    pose_name: str
    cross_body_count: int = 0
    anomaly_count: int = 0
    anomaly_pct: float = 0.0
    max_displacement: float = 0.0
    mean_displacement: float = 0.0
    total_vertices: int = 0
    # Distortion metrics
    stretched_edges: int = 0
    compressed_edges: int = 0
    inverted_tris: int = 0
    degenerate_tris: int = 0
    max_stretch_ratio: float = 1.0
    cross_chain_stretched: int = 0
    # Static vertex metrics
    static_vertex_count: int = 0

    def to_dict(self) -> dict:
        return {
            "pose_name": self.pose_name,
            "cross_body_count": self.cross_body_count,
            "anomaly_count": self.anomaly_count,
            "anomaly_pct": round(self.anomaly_pct, 4),
            "max_displacement": round(self.max_displacement, 4),
            "mean_displacement": round(self.mean_displacement, 4),
            "total_vertices": self.total_vertices,
            "stretched_edges": self.stretched_edges,
            "compressed_edges": self.compressed_edges,
            "inverted_tris": self.inverted_tris,
            "degenerate_tris": self.degenerate_tris,
            "max_stretch_ratio": round(self.max_stretch_ratio, 4),
            "cross_chain_stretched": self.cross_chain_stretched,
            "static_vertex_count": self.static_vertex_count,
        }


# ── Layer weights for multi-layer optimization ───────────────────────

LAYER_WEIGHTS: dict[str, float] = {
    "skin": 3.0,
    "back_muscles": 1.0,
    "shoulder_muscles": 1.0,
    "arm_muscles": 1.0,
    "torso_muscles": 1.0,
    "hip_muscles": 1.0,
    "leg_muscles": 1.0,
    "organs": 0.5,
    "vasculature": 0.5,
}


# ── Evaluation result ────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Full scoring result across all poses and layers."""
    composite: float = 0.0
    per_pose: list[PoseScore] = field(default_factory=list)
    total_cross_body: int = 0
    worst_anomaly_pct: float = 0.0
    worst_max_displacement: float = 0.0
    mean_mean_displacement: float = 0.0
    layers_evaluated: list[str] = field(default_factory=list)
    # Distortion totals
    worst_stretched_edges: int = 0
    worst_inverted_tris: int = 0
    worst_max_stretch_ratio: float = 1.0
    total_cross_chain_stretched: int = 0
    worst_static_vertices: int = 0

    def to_dict(self) -> dict:
        return {
            "composite": round(self.composite, 6),
            "total_cross_body": self.total_cross_body,
            "worst_anomaly_pct": round(self.worst_anomaly_pct, 4),
            "worst_max_displacement": round(self.worst_max_displacement, 4),
            "mean_mean_displacement": round(self.mean_mean_displacement, 4),
            "worst_stretched_edges": self.worst_stretched_edges,
            "worst_inverted_tris": self.worst_inverted_tris,
            "worst_max_stretch_ratio": round(self.worst_max_stretch_ratio, 4),
            "total_cross_chain_stretched": self.total_cross_chain_stretched,
            "worst_static_vertices": self.worst_static_vertices,
            "layers_evaluated": self.layers_evaluated,
            "per_pose": [ps.to_dict() for ps in self.per_pose],
        }


# ── Scorer ───────────────────────────────────────────────────────────

class SkinningScorer:
    """Evaluates skinning quality across multiple poses.

    Parameters
    ----------
    hs : HeadlessScene
        Scene with skinning already built and layers registered.
    pose_names : list[str], optional
        Subset of poses to test. Default: all 11 (6 preset + 5 extreme).
    """

    def __init__(
        self,
        hs: HeadlessScene,
        pose_names: list[str] | None = None,
    ):
        self.hs = hs
        self.poses = get_all_poses(pose_names)

    def evaluate(self) -> EvaluationResult:
        """Run all poses and compute composite score.

        Returns
        -------
        EvaluationResult
            Contains per-pose scores and the scalar composite.
        """
        result = EvaluationResult()
        diag = SkinningDiagnostic(self.hs.skinning)

        all_mean_disps: list[float] = []

        for pose_name, pose_dict in self.poses.items():
            state = _make_body_state(pose_dict)
            apply_pose(self.hs, state)

            score = self._score_pose(diag, pose_name)
            result.per_pose.append(score)

            result.total_cross_body += score.cross_body_count
            result.worst_anomaly_pct = max(result.worst_anomaly_pct, score.anomaly_pct)
            result.worst_max_displacement = max(
                result.worst_max_displacement, score.max_displacement
            )
            if score.mean_displacement > 0:
                all_mean_disps.append(score.mean_displacement)

            # Distortion aggregation
            result.worst_stretched_edges = max(
                result.worst_stretched_edges, score.stretched_edges
            )
            result.worst_inverted_tris = max(
                result.worst_inverted_tris, score.inverted_tris
            )
            result.worst_max_stretch_ratio = max(
                result.worst_max_stretch_ratio, score.max_stretch_ratio
            )
            result.total_cross_chain_stretched += score.cross_chain_stretched
            result.worst_static_vertices = max(
                result.worst_static_vertices, score.static_vertex_count
            )

        if all_mean_disps:
            result.mean_mean_displacement = sum(all_mean_disps) / len(all_mean_disps)

        # Composite score (lower is better)
        result.composite = (
            10.0 * result.total_cross_body
            + 2.0 * result.worst_anomaly_pct
            + 1.0 * result.worst_max_displacement
            + 0.5 * result.mean_mean_displacement
            + 5.0 * result.worst_stretched_edges
            + 8.0 * result.worst_inverted_tris
            + 3.0 * result.total_cross_chain_stretched
            + 10.0 * result.worst_static_vertices
        )

        # Reset to anatomical pose
        apply_pose(self.hs, BodyState())

        return result

    def _score_pose(
        self, diag: SkinningDiagnostic, pose_name: str,
    ) -> PoseScore:
        """Score a single pose after it's been applied."""
        score = PoseScore(pose_name=pose_name)

        # Cross-body bindings (registration-time metric, doesn't change per pose
        # but useful as a baseline)
        cross_body = diag.check_cross_body_bindings()
        score.cross_body_count = sum(cross_body.values())

        # Displacement anomalies (the per-pose metric)
        anomalies = diag.check_displacements(max_displacement=5.0, relative=True)

        total_verts = 0
        total_anomalies = 0
        max_disp = 0.0
        mean_disps: list[float] = []

        for a in anomalies:
            total_verts += a.vertex_count
            total_anomalies += a.anomaly_count
            max_disp = max(max_disp, a.max_displacement)
            mean_disps.append(a.mean_displacement)

        # Also count total verts from non-anomalous bindings
        for binding in self.hs.skinning.bindings:
            if binding.mesh.rest_positions is not None:
                total_verts_actual = len(binding.mesh.rest_positions.reshape(-1, 3))
                # total_verts is already per-anomaly, recalculate from all bindings
        total_verts = sum(
            len(b.mesh.rest_positions.reshape(-1, 3))
            for b in self.hs.skinning.bindings
            if b.mesh.rest_positions is not None
        )

        score.total_vertices = total_verts
        score.anomaly_count = total_anomalies
        score.anomaly_pct = (
            100.0 * total_anomalies / total_verts if total_verts > 0 else 0.0
        )
        score.max_displacement = max_disp
        score.mean_displacement = (
            sum(mean_disps) / len(mean_disps) if mean_disps else 0.0
        )

        # Mesh distortion checks
        distortion = diag.check_mesh_distortion()
        for d in distortion:
            score.stretched_edges += d.stretched_edge_count
            score.compressed_edges += d.compressed_edge_count
            score.inverted_tris += d.inverted_tri_count
            score.degenerate_tris += d.degenerate_tri_count
            score.max_stretch_ratio = max(score.max_stretch_ratio, d.max_stretch_ratio)
            score.cross_chain_stretched += d.cross_chain_stretched

        # Static vertex check (vertices that don't move when their joint does)
        static_verts = diag.check_static_vertices()
        for sv in static_verts:
            score.static_vertex_count += sv.static_count

        return score
