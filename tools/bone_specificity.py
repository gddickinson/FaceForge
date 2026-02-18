"""Bone-specificity diagnostic: verify vertices only respond to their associated bones.

For each degree-of-freedom (DOF), isolate the movement and verify:
- Muscle vertices only move when their registered bone chains are activated
- Skin vertices in each body region only move when that region's chain is activated

This detects binding contamination — vertices incorrectly bound to distant chains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from faceforge.core.state import BodyState

from tools.headless_loader import HeadlessScene, apply_pose

logger = logging.getLogger(__name__)

# ── DOF → expected chain names ──────────────────────────────────────────
# Maps each BodyState DOF to the chain names that DIRECTLY move.
# Spine DOFs are excluded because they affect the entire body.

DOF_CHAIN_MAP: dict[str, set[str]] = {
    # Right arm
    "shoulder_r_abduct": {"arm_R"},
    "shoulder_r_flex":   {"arm_R"},
    "shoulder_r_rotate": {"arm_R"},
    "elbow_r_flex":      {"arm_R"},
    "forearm_r_rotate":  {"arm_R"},
    "wrist_r_flex":      {"arm_R"},
    "wrist_r_deviate":   {"arm_R"},
    # Left arm
    "shoulder_l_abduct": {"arm_L"},
    "shoulder_l_flex":   {"arm_L"},
    "shoulder_l_rotate": {"arm_L"},
    "elbow_l_flex":      {"arm_L"},
    "forearm_l_rotate":  {"arm_L"},
    "wrist_l_flex":      {"arm_L"},
    "wrist_l_deviate":   {"arm_L"},
    # Right leg
    "hip_r_flex":        {"leg_R"},
    "hip_r_abduct":      {"leg_R"},
    "hip_r_rotate":      {"leg_R"},
    "knee_r_flex":       {"leg_R"},
    "ankle_r_flex":      {"leg_R"},
    "ankle_r_invert":    {"leg_R"},
    # Left leg
    "hip_l_flex":        {"leg_L"},
    "hip_l_abduct":      {"leg_L"},
    "hip_l_rotate":      {"leg_L"},
    "knee_l_flex":       {"leg_L"},
    "ankle_l_flex":      {"leg_L"},
    "ankle_l_invert":    {"leg_L"},
    # Right hand
    "finger_curl_r":     {"arm_R"},
    "finger_spread_r":   {"arm_R"},
    "thumb_op_r":        {"arm_R"},
    # Left hand
    "finger_curl_l":     {"arm_L"},
    "finger_spread_l":   {"arm_L"},
    "thumb_op_l":        {"arm_L"},
    # Right foot
    "toe_curl_r":        {"leg_R"},
    "toe_spread_r":      {"leg_R"},
    # Left foot
    "toe_curl_l":        {"leg_L"},
    "toe_spread_l":      {"leg_L"},
}

# Downstream chains: when a parent chain moves, children move too.
DOWNSTREAM_CHAINS: dict[str, set[str]] = {
    "arm_R": {f"hand_R_{d}" for d in range(1, 6)},
    "arm_L": {f"hand_L_{d}" for d in range(1, 6)},
    "leg_R": {f"foot_R_{d}" for d in range(1, 6)},
    "leg_L": {f"foot_L_{d}" for d in range(1, 6)},
}

# Reduced set of DOFs for faster testing (one per major joint)
QUICK_DOFS = [
    "shoulder_r_abduct",
    "shoulder_l_abduct",
    "elbow_r_flex",
    "hip_r_flex",
    "hip_l_flex",
    "knee_r_flex",
]


def get_affected_chains(dof_name: str) -> set[str]:
    """Return all chain names that should move when this DOF fires."""
    direct = DOF_CHAIN_MAP.get(dof_name, set())
    result = set(direct)
    for chain in direct:
        result |= DOWNSTREAM_CHAINS.get(chain, set())
    return result


# ── Skin body region definitions ────────────────────────────────────────
# Skin vertices are partitioned by their primary chain binding.
# For spine-bound vertices, we further segment by Z height.
# This creates testable "regions" — e.g., upper_torso spine vertices
# should not move when knee bends.

# DOF → set of chain names that should NOT be affected.
# We only test clear lateral (R vs L) and regional (arm vs leg) independence.
# Spine/ribs are expected to partially move with everything, so we don't flag them.
_ARM_CHAINS_R = {"arm_R"} | {f"hand_R_{d}" for d in range(1, 6)}
_ARM_CHAINS_L = {"arm_L"} | {f"hand_L_{d}" for d in range(1, 6)}
_LEG_CHAINS_R = {"leg_R"} | {f"foot_R_{d}" for d in range(1, 6)}
_LEG_CHAINS_L = {"leg_L"} | {f"foot_L_{d}" for d in range(1, 6)}


@dataclass
class Violation:
    """A single bone-specificity violation."""

    mesh_name: str
    dof_name: str
    dof_affected_chains: set[str]
    mesh_chains: set[str]
    displaced_count: int
    total_verts: int
    max_displacement: float
    mean_displacement: float


@dataclass
class SkinRegionViolation:
    """Skin vertices in a body region responding to an unrelated DOF."""

    region_name: str  # e.g. "arm_L", "leg_R"
    dof_name: str
    displaced_count: int
    region_total: int
    max_displacement: float
    mean_displacement: float


@dataclass
class BoneSpecificityReport:
    """Results of bone-specificity testing."""

    muscle_violations: list[Violation] = field(default_factory=list)
    skin_violations: list[SkinRegionViolation] = field(default_factory=list)
    skin_displacements: list[SkinRegionDisplacement] = field(default_factory=list)
    tests_run: int = 0
    muscle_tests_passed: int = 0
    skin_tests_passed: int = 0

    @property
    def total_violations(self) -> int:
        return len(self.muscle_violations) + len(self.skin_violations)

    @property
    def is_clean(self) -> bool:
        return self.total_violations == 0


def _build_chain_name_map(hs: HeadlessScene) -> dict[int, str]:
    """Build chain_id → chain_name mapping."""
    return {v: k for k, v in hs.chain_ids.items()}


def _get_binding_chains(hs: HeadlessScene, binding) -> set[str]:
    """Get the chain names a binding's vertices are assigned to."""
    chain_id_to_name = _build_chain_name_map(hs)
    unique_jis = np.unique(np.concatenate([
        binding.joint_indices,
        binding.secondary_indices,
    ]))
    chains = set()
    for ji in unique_jis:
        cid = hs.skinning.joints[ji].chain_id
        name = chain_id_to_name.get(cid)
        if name:
            chains.add(name)
    return chains


def _apply_rest_pose(hs: HeadlessScene) -> None:
    """Reset to anatomical rest position."""
    rest = BodyState()
    apply_pose(hs, rest)


def _apply_single_dof(hs: HeadlessScene, dof_name: str, value: float = 0.5) -> None:
    """Apply a pose with only one DOF activated."""
    state = BodyState()
    setattr(state, dof_name, value)
    apply_pose(hs, state)


def check_muscle_specificity(
    hs: HeadlessScene,
    threshold: float = 0.1,
    dof_names: list[str] | None = None,
) -> list[Violation]:
    """Test that each muscle only responds to its associated bone chains.

    For each DOF, activates only that joint and checks if muscles bound to
    unrelated chains show displacement. Any non-trivial displacement of
    vertices on unrelated chains indicates a binding problem.

    Parameters
    ----------
    threshold : float
        Minimum vertex displacement (world units) to count as "moved".
    dof_names : list of DOF names, or None for QUICK_DOFS.
    """
    if dof_names is None:
        dof_names = QUICK_DOFS

    violations: list[Violation] = []

    # Snapshot rest positions
    rest_snapshots: dict[str, np.ndarray] = {}
    for binding in hs.skinning.bindings:
        if binding.mesh.rest_positions is not None:
            rest_snapshots[binding.mesh.name] = (
                binding.mesh.rest_positions.reshape(-1, 3).copy()
            )

    # Get each binding's chain names
    binding_chains: dict[str, set[str]] = {}
    for binding in hs.skinning.bindings:
        binding_chains[binding.mesh.name] = _get_binding_chains(hs, binding)

    for dof_name in dof_names:
        affected = get_affected_chains(dof_name)
        if not affected:
            continue

        # Apply single-DOF pose
        _apply_single_dof(hs, dof_name)

        for binding in hs.skinning.bindings:
            mesh_name = binding.mesh.name
            mesh_chains = binding_chains.get(mesh_name, set())

            # Skip if mesh's chains overlap with affected chains
            if mesh_chains & affected:
                continue
            # Skip spine/ribs-only meshes — they're expected to be stable
            # but small numerical noise from scene update can cause tiny displacements
            if mesh_chains <= {"spine", "ribs"}:
                continue

            rest = rest_snapshots.get(mesh_name)
            if rest is None:
                continue

            current = binding.mesh.geometry.positions.reshape(-1, 3)
            displacements = np.linalg.norm(
                current.astype(np.float64) - rest.astype(np.float64), axis=1,
            )
            displaced_mask = displacements > threshold
            count = int(np.sum(displaced_mask))

            if count > 0:
                violations.append(Violation(
                    mesh_name=mesh_name,
                    dof_name=dof_name,
                    dof_affected_chains=affected,
                    mesh_chains=mesh_chains,
                    displaced_count=count,
                    total_verts=len(displacements),
                    max_displacement=float(np.max(displacements)),
                    mean_displacement=float(np.mean(displacements[displaced_mask])),
                ))

    # Reset to rest
    _apply_rest_pose(hs)
    return violations


@dataclass
class SkinRegionDisplacement:
    """Per-region displacement summary for one DOF activation."""

    region_name: str
    dof_name: str
    total_verts: int
    displaced_count: int  # vertices above threshold
    max_displacement: float
    mean_displacement: float  # mean of all verts (including non-displaced)
    p95_displacement: float  # 95th percentile


# Anatomical skin regions: combines chain binding + Z-height for spine/ribs.
# Z ranges from the chain vertical ranges (from memory):
#   head/neck:  Z > -15
#   upper torso: -15 to -45
#   lower torso: -45 to -80
#   upper arm: arm chain, above elbow (Z > -50 approx)
#   lower arm: arm chain, below elbow (Z < -50 approx)
SPINE_SUB_REGIONS = {
    "neck":        (-15.0, float("inf")),    # Z > -15
    "upper_back":  (-45.0, -15.0),           # -45 to -15
    "lower_back":  (-80.0, -45.0),           # -80 to -45
}

ARM_SUB_REGIONS = {
    "shoulder":   (-35.0, float("inf")),  # Z > -35 (near shoulder joint)
    "upper_arm":  (-55.0, -35.0),         # -55 to -35
    "forearm":    (-90.0, -55.0),         # -90 to -55
    "hand":       (float("-inf"), -90.0), # Z < -90
}

LEG_SUB_REGIONS = {
    "hip":        (-90.0, float("inf")),   # Z > -90 (hip area)
    "thigh":      (-140.0, -90.0),         # -140 to -90
    "calf":       (-190.0, -140.0),        # -190 to -140
    "foot":       (float("-inf"), -190.0), # Z < -190
}


def _build_skin_regions(
    hs: HeadlessScene,
    skin_binding,
) -> dict[str, np.ndarray]:
    """Build detailed anatomical skin regions.

    Partitions skin vertices into fine-grained regions:
    - spine → neck, upper_back, lower_back (by Z)
    - ribs → ribs (single region, central)
    - arm_R → shoulder_R, upper_arm_R, forearm_R, hand_R
    - arm_L → shoulder_L, upper_arm_L, forearm_L, hand_L
    - leg_R → hip_R, thigh_R, calf_R, foot_R
    - leg_L → hip_L, thigh_L, calf_L, foot_L
    - hand_R_* → fingers_R (all finger chains combined)
    - hand_L_* → fingers_L
    - foot_R_* → toes_R
    - foot_L_* → toes_L
    """
    chain_id_to_name = _build_chain_name_map(hs)
    rest = skin_binding.mesh.rest_positions.reshape(-1, 3)
    V = len(rest)

    regions: dict[str, list[int]] = {}

    for vi in range(V):
        ji = skin_binding.joint_indices[vi]
        cid = hs.skinning.joints[ji].chain_id
        chain_name = chain_id_to_name.get(cid, f"chain_{cid}")
        z = float(rest[vi, 2])

        if chain_name == "spine":
            assigned = False
            for sub_name, (z_lo, z_hi) in SPINE_SUB_REGIONS.items():
                if z_lo <= z < z_hi:
                    regions.setdefault(sub_name, []).append(vi)
                    assigned = True
                    break
            if not assigned:
                regions.setdefault("spine_other", []).append(vi)
        elif chain_name == "ribs":
            regions.setdefault("ribs", []).append(vi)
        elif chain_name.startswith("arm_"):
            side = chain_name[-1]  # R or L
            assigned = False
            for sub_name, (z_lo, z_hi) in ARM_SUB_REGIONS.items():
                if z_lo <= z < z_hi:
                    regions.setdefault(f"{sub_name}_{side}", []).append(vi)
                    assigned = True
                    break
            if not assigned:
                regions.setdefault(f"arm_other_{side}", []).append(vi)
        elif chain_name.startswith("leg_"):
            side = chain_name[-1]
            assigned = False
            for sub_name, (z_lo, z_hi) in LEG_SUB_REGIONS.items():
                if z_lo <= z < z_hi:
                    regions.setdefault(f"{sub_name}_{side}", []).append(vi)
                    assigned = True
                    break
            if not assigned:
                regions.setdefault(f"leg_other_{side}", []).append(vi)
        elif chain_name.startswith("hand_"):
            side = chain_name.split("_")[1]  # R or L
            regions.setdefault(f"fingers_{side}", []).append(vi)
        elif chain_name.startswith("foot_"):
            side = chain_name.split("_")[1]
            regions.setdefault(f"toes_{side}", []).append(vi)
        else:
            regions.setdefault(chain_name, []).append(vi)

    return {k: np.array(v, dtype=np.int64) for k, v in regions.items()}


# Which skin regions should NOT move for each DOF group.
# Regions not listed here are expected to move or are ambiguous.
_SKIN_REGION_INDEPENDENCE: dict[str, set[str]] = {
    # Right shoulder/arm DOFs → left arm, all legs should not move
    "shoulder_r_abduct": {"shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
                          "fingers_L", "hip_L", "thigh_L", "calf_L", "foot_L",
                          "toes_L", "hip_R", "thigh_R", "calf_R", "foot_R", "toes_R"},
    "shoulder_r_flex":   {"shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
                          "fingers_L", "hip_L", "thigh_L", "calf_L", "foot_L",
                          "toes_L", "hip_R", "thigh_R", "calf_R", "foot_R", "toes_R"},
    "elbow_r_flex":      {"shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
                          "fingers_L", "hip_L", "thigh_L", "calf_L", "foot_L",
                          "toes_L", "hip_R", "thigh_R", "calf_R", "foot_R", "toes_R"},
    # Left shoulder/arm DOFs → right arm, all legs should not move
    "shoulder_l_abduct": {"shoulder_R", "upper_arm_R", "forearm_R", "hand_R",
                          "fingers_R", "hip_L", "thigh_L", "calf_L", "foot_L",
                          "toes_L", "hip_R", "thigh_R", "calf_R", "foot_R", "toes_R"},
    "shoulder_l_flex":   {"shoulder_R", "upper_arm_R", "forearm_R", "hand_R",
                          "fingers_R", "hip_L", "thigh_L", "calf_L", "foot_L",
                          "toes_L", "hip_R", "thigh_R", "calf_R", "foot_R", "toes_R"},
    "elbow_l_flex":      {"shoulder_R", "upper_arm_R", "forearm_R", "hand_R",
                          "fingers_R", "hip_L", "thigh_L", "calf_L", "foot_L",
                          "toes_L", "hip_R", "thigh_R", "calf_R", "foot_R", "toes_R"},
    # Right hip/leg DOFs → both arms, left leg should not move
    "hip_r_flex":        {"shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
                          "fingers_L", "shoulder_R", "upper_arm_R", "forearm_R",
                          "hand_R", "fingers_R",
                          "hip_L", "thigh_L", "calf_L", "foot_L", "toes_L"},
    "knee_r_flex":       {"shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
                          "fingers_L", "shoulder_R", "upper_arm_R", "forearm_R",
                          "hand_R", "fingers_R",
                          "hip_L", "thigh_L", "calf_L", "foot_L", "toes_L"},
    # Left hip/leg DOFs → both arms, right leg should not move
    "hip_l_flex":        {"shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
                          "fingers_L", "shoulder_R", "upper_arm_R", "forearm_R",
                          "hand_R", "fingers_R",
                          "hip_R", "thigh_R", "calf_R", "foot_R", "toes_R"},
    "knee_l_flex":       {"shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
                          "fingers_L", "shoulder_R", "upper_arm_R", "forearm_R",
                          "hand_R", "fingers_R",
                          "hip_R", "thigh_R", "calf_R", "foot_R", "toes_R"},
}


def check_skin_specificity(
    hs: HeadlessScene,
    threshold: float = 0.5,
    dof_names: list[str] | None = None,
) -> tuple[list[SkinRegionViolation], list[SkinRegionDisplacement]]:
    """Test that skin regions only respond to their associated bones.

    Returns both violations (above threshold) and a full displacement
    summary per region per DOF for diagnostic visibility.

    Parameters
    ----------
    threshold : float
        Minimum vertex displacement (world units) to flag as violation.
    dof_names : list of DOF names, or None for QUICK_DOFS.
    """
    if dof_names is None:
        dof_names = QUICK_DOFS

    violations: list[SkinRegionViolation] = []
    all_disps: list[SkinRegionDisplacement] = []

    # Find skin binding(s)
    skin_bindings = [
        b for b in hs.skinning.bindings if not b.is_muscle and len(b.weights) > 100_000
    ]
    if not skin_bindings:
        return violations, all_disps

    for skin_binding in skin_bindings:
        rest = skin_binding.mesh.rest_positions.reshape(-1, 3).copy()
        regions = _build_skin_regions(hs, skin_binding)

        print(f"  Skin regions: {', '.join(f'{k}({len(v)})' for k, v in sorted(regions.items()))}")

        for dof_name in dof_names:
            independence_set = _SKIN_REGION_INDEPENDENCE.get(dof_name)
            if independence_set is None:
                continue

            _apply_single_dof(hs, dof_name, value=0.7)  # Moderate-strong activation
            current = skin_binding.mesh.geometry.positions.reshape(-1, 3)

            for region_name, vert_indices in sorted(regions.items()):
                region_rest = rest[vert_indices]
                region_curr = current[vert_indices]
                displacements = np.linalg.norm(
                    region_curr.astype(np.float64) - region_rest.astype(np.float64),
                    axis=1,
                )

                # Always record displacement for visibility
                disp_entry = SkinRegionDisplacement(
                    region_name=region_name,
                    dof_name=dof_name,
                    total_verts=len(vert_indices),
                    displaced_count=int(np.sum(displacements > threshold)),
                    max_displacement=float(np.max(displacements)) if len(displacements) > 0 else 0.0,
                    mean_displacement=float(np.mean(displacements)) if len(displacements) > 0 else 0.0,
                    p95_displacement=float(np.percentile(displacements, 95)) if len(displacements) > 0 else 0.0,
                )
                all_disps.append(disp_entry)

                # Check violation: region should be independent of this DOF
                if region_name in independence_set:
                    displaced_mask = displacements > threshold
                    count = int(np.sum(displaced_mask))
                    if count > 0:
                        violations.append(SkinRegionViolation(
                            region_name=region_name,
                            dof_name=dof_name,
                            displaced_count=count,
                            region_total=len(vert_indices),
                            max_displacement=float(np.max(displacements)),
                            mean_displacement=float(np.mean(displacements[displaced_mask])),
                        ))

    # Reset to rest
    _apply_rest_pose(hs)
    return violations, all_disps


def run_bone_specificity(
    hs: HeadlessScene,
    muscle_threshold: float = 0.1,
    skin_threshold: float = 0.5,
    dof_names: list[str] | None = None,
    full: bool = False,
) -> BoneSpecificityReport:
    """Run the complete bone-specificity diagnostic.

    Parameters
    ----------
    muscle_threshold : float
        Displacement threshold for flagging muscle vertices.
    skin_threshold : float
        Displacement threshold for flagging skin vertices.
        Higher because cross-chain blending causes expected spillover.
    dof_names : list or None
        Specific DOFs to test. None uses QUICK_DOFS (6 DOFs).
    full : bool
        If True, test ALL DOFs (slower but comprehensive).
    """
    if full:
        dof_names = list(DOF_CHAIN_MAP.keys())

    report = BoneSpecificityReport()

    if dof_names is None:
        dof_names = QUICK_DOFS

    print(f"Testing bone specificity across {len(dof_names)} DOFs...")
    print(f"  Muscle threshold: {muscle_threshold} units")
    print(f"  Skin threshold:   {skin_threshold} units")
    print()

    # Muscle test
    print("Checking muscle specificity...")
    report.muscle_violations = check_muscle_specificity(
        hs, threshold=muscle_threshold, dof_names=dof_names,
    )
    n_muscle_bindings = sum(1 for b in hs.skinning.bindings if b.is_muscle)
    report.tests_run = len(dof_names) * n_muscle_bindings
    report.muscle_tests_passed = report.tests_run - len(report.muscle_violations)

    # Skin test
    print("Checking skin region specificity...")
    report.skin_violations, report.skin_displacements = check_skin_specificity(
        hs, threshold=skin_threshold, dof_names=dof_names,
    )
    report.skin_tests_passed = 0

    return report


def format_report(report: BoneSpecificityReport) -> str:
    """Format bone-specificity report as human-readable text."""
    lines = [
        "=" * 60,
        "BONE SPECIFICITY REPORT",
        "=" * 60,
        "",
    ]

    # Muscle violations
    if report.muscle_violations:
        lines.append(f"MUSCLE VIOLATIONS: {len(report.muscle_violations)}")
        lines.append("-" * 50)

        by_dof: dict[str, list[Violation]] = {}
        for v in report.muscle_violations:
            by_dof.setdefault(v.dof_name, []).append(v)

        for dof, vlist in sorted(by_dof.items()):
            lines.append(f"\n  DOF: {dof} (affects {', '.join(sorted(vlist[0].dof_affected_chains))})")
            for v in sorted(vlist, key=lambda x: -x.max_displacement):
                chains_str = ", ".join(sorted(v.mesh_chains))
                lines.append(
                    f"    {v.mesh_name:30s}  chains=[{chains_str}]  "
                    f"displaced={v.displaced_count}/{v.total_verts}  "
                    f"max={v.max_displacement:.3f}  mean={v.mean_displacement:.3f}"
                )
    else:
        lines.append("MUSCLE SPECIFICITY: CLEAN (no cross-contamination)")

    lines.append("")

    # Skin violations
    if report.skin_violations:
        lines.append(f"SKIN REGION VIOLATIONS: {len(report.skin_violations)}")
        lines.append("-" * 50)

        by_dof: dict[str, list[SkinRegionViolation]] = {}
        for v in report.skin_violations:
            by_dof.setdefault(v.dof_name, []).append(v)

        for dof, vlist in sorted(by_dof.items()):
            lines.append(f"\n  DOF: {dof}")
            for v in sorted(vlist, key=lambda x: -x.max_displacement):
                pct = 100.0 * v.displaced_count / v.region_total if v.region_total > 0 else 0
                lines.append(
                    f"    region={v.region_name:20s}  "
                    f"displaced={v.displaced_count:6d}/{v.region_total:6d} ({pct:5.1f}%)  "
                    f"max={v.max_displacement:.3f}  mean={v.mean_displacement:.3f}"
                )
    else:
        lines.append("SKIN REGION SPECIFICITY: CLEAN (no cross-contamination)")

    # Displacement heatmap: region × DOF grid
    if report.skin_displacements:
        lines.append("")
        lines.append("SKIN DISPLACEMENT HEATMAP (max displacement per region per DOF):")
        lines.append("-" * 60)

        # Group by DOF
        dof_names_ordered = []
        by_dof_disp: dict[str, dict[str, SkinRegionDisplacement]] = {}
        for d in report.skin_displacements:
            if d.dof_name not in by_dof_disp:
                dof_names_ordered.append(d.dof_name)
                by_dof_disp[d.dof_name] = {}
            by_dof_disp[d.dof_name][d.region_name] = d

        # Get all regions (sorted)
        all_regions = sorted({d.region_name for d in report.skin_displacements})

        # Print header
        dof_short = [d.replace("shoulder_", "sh_").replace("elbow_", "el_")
                      .replace("hip_", "h_").replace("knee_", "k_")
                      .replace("_flex", "F").replace("_abduct", "A")
                      for d in dof_names_ordered]
        header = f"{'Region':>18s} |"
        for ds in dof_short:
            header += f" {ds:>8s}"
        lines.append(header)
        lines.append("-" * len(header))

        for region in all_regions:
            row = f"{region:>18s} |"
            for dof_name in dof_names_ordered:
                d = by_dof_disp.get(dof_name, {}).get(region)
                if d is None:
                    row += f" {'---':>8s}"
                elif d.max_displacement < 0.001:
                    row += f" {'.':>8s}"
                elif d.max_displacement < 0.1:
                    row += f" {d.max_displacement:>8.3f}"
                else:
                    row += f" {d.max_displacement:>8.2f}"
            lines.append(row)

        lines.append("")
        lines.append("  '.' = max < 0.001 (negligible)")

    lines.append("")
    lines.append(f"Total violations: {report.total_violations}")
    lines.append("=" * 60)
    return "\n".join(lines)
