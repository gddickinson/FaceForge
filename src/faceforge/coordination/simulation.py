"""Per-frame simulation orchestrator — mirrors the JS animate() function."""

import logging
from typing import Optional

import numpy as np

from faceforge.core.state import StateManager
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.math_utils import Quat, quat_identity, quat_from_euler
from faceforge.animation.interpolation import StateInterpolator
from faceforge.animation.auto_blink import AutoBlink
from faceforge.animation.auto_breathing import AutoBreathing
from faceforge.animation.micro_expressions import MicroExpressionGen
from faceforge.animation.eye_tracking import EyeTracking

logger = logging.getLogger(__name__)


class Simulation:
    """Per-frame orchestrator that drives all animation systems.

    Call order:
      1. State interpolation (smooth lerp toward targets)
      2. Auto-animations (blink, breathing, micro-expressions, eye tracking)
      3. FACS displacement + jaw pivot rotation
      4. Jaw muscles
      5. Expression muscles
      6. Face features (eyes, eyebrows, nose, ears)
      7. Head rotation + vertebra distribution (skull, face, brain,
         stlMuscleGroup, exprMuscleGroup, faceFeatureGroup)
      8. Body joint limits (clamp)
      9. Body animation (spine, limbs, breathing)
      10. Neck muscles (uses head quat + body state)
      11. Neck constraints
      12. Soft tissue skinning
      13. Scene graph matrix update

    Systems whose parent group is invisible are skipped for performance.
    """

    def __init__(self, state: StateManager, scene: Scene):
        self.state = state
        self.scene = scene

        # Animation subsystems
        self.interpolator = StateInterpolator()
        self.auto_blink = AutoBlink()
        self.auto_breathing = AutoBreathing()
        self.micro_expressions = MicroExpressionGen()
        self.eye_tracking = EyeTracking()

        # Anatomy systems (set after loading)
        self.facs_engine = None
        self.jaw_muscles = None
        self.expression_muscles = None
        self.face_features = None
        self.head_rotation = None
        self.neck_muscles = None
        self.neck_constraints = None
        self.vertebrae_pivots: list[dict] = []

        # Scene node references (set after loading)
        self.skull_group: Optional[SceneNode] = None
        self.face_group: Optional[SceneNode] = None
        self.jaw_pivot_node: Optional[SceneNode] = None

        # Visibility group references for gating expensive updates
        self.jaw_muscle_group: Optional[SceneNode] = None
        self.expr_muscle_group: Optional[SceneNode] = None
        self.neck_muscle_group: Optional[SceneNode] = None
        self.face_feature_group: Optional[SceneNode] = None

        # Brain group (independent of skull, needs explicit head rotation)
        self.brain_group: Optional[SceneNode] = None

        # Body systems (set after loading)
        self.body_animation = None
        self.body_constraints = None
        self.soft_tissue = None

        # Bone anchor registry for per-muscle attachment pinning
        self.bone_anchors = None  # BoneAnchorRegistry or None

        # Platysma body-spanning handler
        self.platysma = None  # PlatysmaHandler or None
        self.platysma_group: Optional[SceneNode] = None  # identity-transform group

        # Fascia constraint system (for Platysma attachment)
        self.fascia = None  # FasciaSystem or None

        # Back-of-neck muscle body-end pinning handler
        self.back_neck_muscles = None  # BackNeckMuscleHandler or None

        # Physiological simulations (set by app.py)
        self.physiology = None  # PhysiologySystem or None

        # Muscle activation heatmap system (set by app.py)
        self.muscle_activation = None  # MuscleActivationSystem or None

        # Pathology visualization system (set by app.py)
        self.pathology = None  # PathologySystem or None

        # Scene animation player (set by app.py when scene mode wired)
        self.anim_player = None  # AnimationPlayer or None

        # Cached head quaternion for neck muscles
        self._head_quat: Quat = quat_identity()

    @staticmethod
    def _is_visible(node: Optional[SceneNode]) -> bool:
        """Check if a node and all its ancestors are visible."""
        while node is not None:
            if not node.visible:
                return False
            node = node.parent
        return True

    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds."""
        face = self.state.face
        body = self.state.body

        # 0. Scene animation (drives wrapper transform + body/face targets)
        if self.anim_player is not None and self.anim_player.is_playing:
            self.anim_player.tick(dt)

        # 1. Interpolate state toward targets
        self.interpolator.interpolate(
            face, self.state.target_au, self.state.target_head,
            self.state.target_ear_wiggle,
            body, self.state.target_body, dt,
        )

        # 2. Auto-animations
        self.auto_blink.update(face, dt)
        self.auto_breathing.update(face, dt)
        self.micro_expressions.update(face, self.state.target_au, dt)
        self.eye_tracking.update(face, dt)

        # Check visibility flags once per frame to skip expensive systems
        face_visible = self._is_visible(self.face_group)
        jaw_muscles_visible = self._is_visible(self.jaw_muscle_group)
        expr_muscles_visible = self._is_visible(self.expr_muscle_group)
        neck_muscles_visible = self._is_visible(self.neck_muscle_group)
        face_features_visible = self._is_visible(self.face_feature_group)

        # 3. FACS + jaw pivot rotation
        # Always compute jaw angle from AU values (jaw is a skull bone, not face mesh).
        # Only run full face mesh deformation when the face layer is visible.
        au26 = face.get_au("AU26")
        au25 = face.get_au("AU25")
        jaw_angle = au26 * 0.28 + au25 * 0.06
        if self.facs_engine is not None and face_visible:
            jaw_angle = self.facs_engine.apply(face)

        # Rotate the jawPivot node (like JS: jawPivot.rotation.x = jawAngle)
        if self.jaw_pivot_node is not None:
            q = quat_from_euler(jaw_angle, 0.0, 0.0, "XYZ")
            self.jaw_pivot_node.set_quaternion(q)

        # 4. Jaw muscles (skip if group invisible)
        if self.jaw_muscles is not None and jaw_muscles_visible:
            self.jaw_muscles.update(jaw_angle)

        # 5. Expression muscles (skip if group invisible)
        if self.expression_muscles is not None and expr_muscles_visible:
            self.expression_muscles.update(face)

        # 6. Face features (skip if group invisible)
        if self.face_features is not None and face_features_visible:
            self.face_features.update(face)

        # 7. Head rotation (always needed — skull/face groups rotate)
        if self.head_rotation is not None and self.skull_group is not None:
            self._head_quat = self.head_rotation.apply(
                face,
                self.skull_group,
                self.face_group,
                self.vertebrae_pivots or None,
                self.state.constraints,
                brain_group=self.brain_group,
                stl_muscle_group=self.jaw_muscle_group,
                expr_muscle_group=self.expr_muscle_group,
                face_feature_group=self.face_feature_group,
            )

        # 8. Body joint limits (clamp before animation)
        if self.body_constraints is not None:
            self.body_constraints.clamp(body)

        # 9. Body animation (before neck muscles so body state is current)
        if self.body_animation is not None:
            self.body_animation.apply(body, dt)

        # 9.5 Extract body anchor positions for neck muscle body-tracking
        if self.neck_muscles is not None and neck_muscles_visible and self.body_animation is not None:
            self._update_neck_body_anchors()

        # 9.6 Update bone anchor current positions (for per-muscle pinning)
        if self.bone_anchors is not None and self.body_animation is not None:
            self.scene.update()  # ensure world matrices are fresh
            # Note: snapshot_rest_positions is only called once during init

        # 10. Neck muscles (skip if group invisible; now has body anim results)
        if self.neck_muscles is not None and neck_muscles_visible:
            self.neck_muscles.update(self._head_quat, face_state=face, body_state=body)

        # 10.5 Platysma correction (after head rotation + body animation)
        # Note: NOT gated on expr_muscles_visible.  Platysma is a body-spanning
        # structural correction that must always run — even if the expression
        # muscle layer is toggled off for rendering, the underlying mesh still
        # exists and gets rotated by head_rotation.apply().
        if self.platysma is not None and self.platysma.registered:
            plat_bone_cur = None
            plat_bone_rest = None
            if self.bone_anchors is not None:
                plat_bone_cur = self.bone_anchors.get_muscle_anchor_current(
                    "Platysma", ["Right Clavicle", "Left Clavicle"],
                )
                plat_bone_rest = self.bone_anchors.get_muscle_anchor(
                    "Platysma", ["Right Clavicle", "Left Clavicle"],
                )

            self.platysma.update(
                self._head_quat,
                bone_anchor_current=plat_bone_cur,
                bone_anchor_rest=plat_bone_rest,
            )

        # 11. Neck constraints (skip if neck muscles invisible)
        if self.neck_constraints is not None and neck_muscles_visible:
            self.neck_constraints.solve(
                self.state.constraints,
                self.neck_muscles.muscle_data if self.neck_muscles else [],
                self._head_quat,
            )

        # 11.5 Platysma stretch → constraint feedback
        # Platysma tension adds to the total excess so the head rotation
        # soft-clamp accounts for platysma overstretching.
        if (self.platysma is not None and self.platysma.registered
                and self.platysma.tension_excess > 0.0):
            plat_excess = self.platysma.tension_excess
            cs = self.state.constraints
            cs.total_excess += plat_excess
            # Re-smooth with the added platysma contribution
            DAMPING = 0.2
            cs.smoothed_total_excess += plat_excess * DAMPING

        # 12. Soft tissue skinning
        if self.soft_tissue is not None:
            self.soft_tissue.update(body)

        # 12.3 Muscle stretch tension → constraint feedback (Layer 3)
        if (self.soft_tissue is not None
                and self.soft_tissue.attachment_system is not None):
            muscle_excess = self.soft_tissue.attachment_system.get_total_tension_excess()
            if muscle_excess > 0.0:
                cs = self.state.constraints
                cs.total_excess += muscle_excess
                DAMPING = 0.2
                cs.smoothed_total_excess += muscle_excess * DAMPING

        # 12.5 Apply head rotation to skin meshes (after body skinning)
        if self.head_rotation is not None and self.soft_tissue is not None:
            self._apply_head_to_skin()

        # 12.6 Back-of-neck muscle body-end pinning (after head-follow)
        if self.back_neck_muscles is not None and self.back_neck_muscles.registered:
            self.back_neck_muscles.update(self._head_quat)

        # 12.7 Muscle activation heatmap (after soft tissue, before scene update)
        if self.muscle_activation is not None:
            self.muscle_activation.update(body)

        # 13. Update scene graph matrices
        self.scene.update()

        # 14. Physiological simulations (after scene graph, works on current positions)
        if self.physiology is not None:
            positions_modified = self.physiology.step(body, dt)
            # When physiology modified mesh positions, invalidate the soft tissue
            # skinning cache so it recalculates from scratch next frame.  Without
            # this, the signature-based early-exit would leave stale positions
            # (including our delta) in place, and the next physiology step would
            # stack another delta on top — causing unbounded growth.
            if positions_modified and self.soft_tissue is not None:
                self.soft_tissue._last_signature = ()

        # 14.5 Pathology visualization (after physiology)
        if self.pathology is not None:
            self.pathology.update()

        # Frame counter
        self.state.frame_count += 1

    def init_neck_body_anchors(self) -> None:
        """Snapshot rest-pose body anchor positions for neck muscle tracking.

        Call once after body animation is wired and scene is in rest pose.
        """
        if self.neck_muscles is None or self.body_animation is None:
            return
        # Force a scene update to get current world matrices
        self.scene.update()
        self._update_neck_body_anchors()
        # Copy current anchors as rest anchors
        if self.neck_muscles._body_anchor_current:
            self.neck_muscles.set_body_anchors_rest(
                dict(self.neck_muscles._body_anchor_current)
            )

    def _update_neck_body_anchors(self) -> None:
        """Extract body anchor world positions from body animation pivots.

        Provides shoulder, ribcage, and thoracic anchor positions to neck
        muscles for body-delta tracking of lower-end vertices.
        """
        ba = self.body_animation
        if ba is None or self.neck_muscles is None:
            return

        anchors: dict[str, np.ndarray] = {}
        pivots = getattr(ba, '_joint_setup', None)
        if pivots is None:
            pivots = getattr(ba, 'joint_setup', None)

        # Thoracic: use highest thoracic pivot (T1, closest to neck)
        thoracic_pivots = getattr(ba, 'thoracic_pivots', [])
        if thoracic_pivots:
            top_pivot = thoracic_pivots[0]
            group = top_pivot.get("group")
            if group is not None:
                anchors["thoracic"] = group.get_world_position()

        # Shoulder: average of R/L shoulder pivots
        jp = getattr(ba, 'joints', None)
        if jp is not None:
            jp_pivots = getattr(jp, 'pivots', {})
            shoulder_positions = []
            for side in ("R", "L"):
                node = jp_pivots.get(f"shoulder_{side}")
                if node is not None:
                    shoulder_positions.append(node.get_world_position())
            if shoulder_positions:
                anchors["shoulder"] = np.mean(shoulder_positions, axis=0)

        # Ribcage: use rib pivot average if available
        rib_pivots = getattr(ba, '_rib_pivots', [])
        if rib_pivots:
            rib_positions = [p.get_world_position() for p in rib_pivots[:4]]
            if rib_positions:
                anchors["ribcage"] = np.mean(rib_positions, axis=0)

        if anchors:
            self.neck_muscles.set_body_anchors_current(anchors)

    def _apply_head_to_skin(self) -> None:
        """Apply head rotation to skin mesh vertices in the head/neck zone.

        Runs after soft tissue skinning so body deformation is already applied.

        Muscle-aware blending (Layer 1):
        - Muscles with head_follow_config: use per-muscle Y-based fractions
        - Muscles without head_follow_config: skip (purely joint-skinned)
        - Non-muscles (skin/organs): use existing Z-based global blend
        """
        head_rot = self.head_rotation
        if head_rot is None or self.soft_tissue is None:
            return

        head_q = self._head_quat
        from faceforge.core.math_utils import quat_identity as _qi
        if np.allclose(head_q, _qi(), atol=1e-6):
            return

        for binding in self.soft_tissue.bindings:
            mesh = binding.mesh
            if mesh.rest_positions is None:
                continue

            if binding.is_muscle:
                # Muscles: only apply head rotation if they have headFollow config
                if binding.head_follow_config is not None:
                    upper_frac = binding.head_follow_config.get("upperFrac", 0.0)
                    lower_frac = binding.head_follow_config.get("lowerFrac", 0.0)
                    head_rot.apply_to_skin_muscle(
                        mesh.geometry.positions,
                        mesh.rest_positions,
                        upper_frac,
                        lower_frac,
                        head_q,
                    )
                    mesh.needs_update = True
                # else: skip — muscle is purely joint-skinned, no head follow
            else:
                # Non-muscles: existing Z-based global blend
                head_rot.apply_to_skin(
                    mesh.geometry.positions,
                    mesh.rest_positions,
                    head_q,
                )
                mesh.needs_update = True

