"""Per-frame simulation orchestrator — mirrors the JS animate() function."""

from typing import Optional

from faceforge.core.state import StateManager
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.math_utils import Quat, quat_identity, quat_from_euler
from faceforge.animation.interpolation import StateInterpolator
from faceforge.animation.auto_blink import AutoBlink
from faceforge.animation.auto_breathing import AutoBreathing
from faceforge.animation.micro_expressions import MicroExpressionGen
from faceforge.animation.eye_tracking import EyeTracking


class Simulation:
    """Per-frame orchestrator that drives all animation systems.

    Call order mirrors the JS animate():
      1. State interpolation (smooth lerp toward targets)
      2. Auto-animations (blink, breathing, micro-expressions, eye tracking)
      3. FACS displacement + jaw pivot rotation
      4. Jaw muscles
      5. Expression muscles
      6. Face features (eyes, eyebrows, nose, ears)
      7. Head rotation + vertebra distribution
      8. Neck muscles
      9. Neck constraints
      10. Body animation (spine, limbs, breathing)
      11. Soft tissue skinning
      12. Scene graph matrix update

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
            )

        # 8. Neck muscles (skip if group invisible)
        if self.neck_muscles is not None and neck_muscles_visible:
            self.neck_muscles.update(self._head_quat, face_state=face, body_state=body)

        # 9. Neck constraints (skip if neck muscles invisible)
        if self.neck_constraints is not None and neck_muscles_visible:
            self.neck_constraints.solve(
                self.state.constraints,
                self.neck_muscles.muscle_data if self.neck_muscles else [],
                self._head_quat,
            )

        # 9.5. Body joint limits (clamp before animation)
        if self.body_constraints is not None:
            self.body_constraints.clamp(body)

        # 10. Body animation
        if self.body_animation is not None:
            self.body_animation.apply(body, dt)

        # 11. Soft tissue skinning
        if self.soft_tissue is not None:
            self.soft_tissue.update(body)

        # 12. Update scene graph matrices
        self.scene.update()

        # Frame counter
        self.state.frame_count += 1
